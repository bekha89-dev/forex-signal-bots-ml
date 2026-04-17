#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Forex Scalper - Все критические баги исправлены:
1. Исправлен фильтр времени (не торгует ночью)
2. Добавлена блокировка повторных позиций на одной паре
3. Добавлен cooldown между сигналами (15 минут)
4. Добавлен лимит сигналов на пару (3/день)
5. Повышены пороги для безопасности
"""

import asyncio
import io
import json
import logging
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import os
import pickle
import random
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import aiohttp
import numpy as np
import pandas as pd
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler

warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scalper.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
STATE_FILE = "risk_state.pkl"

DEFAULT_CONFIG = {
    "telegram_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "telegram_chat_id": "YOUR_CHAT_ID",
    "twelve_data_key": "YOUR_TWELVE_DATA_KEY",
    "throttle_sec": 10.0,
    "scan_interval_sec": 300,
    "ml_enabled": True,
    "ml_retrain_hours": 24,
    "ml_confidence_threshold": 0.75,  # Повышено с 0.70
    "risk_per_trade_percent": 0.75,   # Снижено с 1.0
    "max_open_trades": 2,              # Снижено с 3
    "initial_balance": 10_000,
    "min_signal_strength": 0.75,      # Повышено с 0.65
    "trading_hours_start": 7,
    "trading_hours_end": 16,
    "max_signals_per_pair_daily": 3,  # Новое: лимит сигналов
    "signal_cooldown_minutes": 15,     # Новое: cooldown между сигналами
    "markets": [
        {"name": "EUR/USD", "symbol": "EUR/USD", "precision": 5, "min_volatility": 0.00025, "rr": 2.0, "pip_value": 0.0001, "spread_pips": 1.5},
        {"name": "NZD/USD", "symbol": "NZD/USD", "precision": 5, "min_volatility": 0.0003, "rr": 2.0, "pip_value": 0.0001, "spread_pips": 2.2},
        {"name": "USD/JPY", "symbol": "USD/JPY", "precision": 3, "min_volatility": 0.15, "rr": 2.0, "pip_value": 0.01, "spread_pips": 1.5},
        {"name": "AUD/USD", "symbol": "AUD/USD", "precision": 5, "min_volatility": 0.00025, "rr": 2.0, "pip_value": 0.0001, "spread_pips": 1.8},
        {"name": "USD/CAD", "symbol": "USD/CAD", "precision": 5, "min_volatility": 0.0003, "rr": 2.0, "pip_value": 0.0001, "spread_pips": 2.0},
    ],
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                merged = {**DEFAULT_CONFIG, **cfg}
                
                # Валидация критичных параметров
                if merged["trading_hours_end"] > 16:
                    logger.warning("⚠️ trading_hours_end > 16! Установлено на 16 (защита от Asian session)")
                    merged["trading_hours_end"] = 16
                
                if merged["trading_hours_start"] > 10:
                    logger.warning("⚠️ trading_hours_start > 10! Установлено на 7")
                    merged["trading_hours_start"] = 7
                
                return merged
        except Exception as e:
            logger.error(f"Bad config.json — {e}")
    
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    return DEFAULT_CONFIG


CONFIG = load_config()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or CONFIG["telegram_token"]
TELEGRAM_CHAT_ID = int(CONFIG["telegram_chat_id"])
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY") or CONFIG["twelve_data_key"]
THROTTLE_SEC = CONFIG["throttle_sec"]
SCAN_INTERVAL_SEC = CONFIG["scan_interval_sec"]
ML_ENABLED = CONFIG["ml_enabled"]
ML_CONFIDENCE = CONFIG["ml_confidence_threshold"]
ML_RETRAIN_SEC = CONFIG["ml_retrain_hours"] * 3600
RISK_PER_TRADE = CONFIG["risk_per_trade_percent"] / 100
MAX_OPEN_TRADES = CONFIG["max_open_trades"]
INITIAL_BALANCE = CONFIG["initial_balance"]
MIN_SIGNAL_STRENGTH = CONFIG["min_signal_strength"]
TRADING_HOURS_START = CONFIG["trading_hours_start"]
TRADING_HOURS_END = CONFIG["trading_hours_end"]
MAX_SIGNALS_PER_PAIR = CONFIG.get("max_signals_per_pair_daily", 3)
SIGNAL_COOLDOWN_MIN = CONFIG.get("signal_cooldown_minutes", 15)
MIN_CANDLES = 150


# ----------- Trading Time Filter -----------
def is_good_trading_time():
    current_hour = datetime.utcnow().hour
    
    if TRADING_HOURS_START <= current_hour < TRADING_HOURS_END:
        return True, "Trading hours"
    
    if current_hour < TRADING_HOURS_START:
        wait_hours = TRADING_HOURS_START - current_hour
        return False, f"Waiting for London open ({wait_hours}h)"
    else:
        wait_hours = 24 - current_hour + TRADING_HOURS_START
        return False, f"After US close, waiting {wait_hours}h"


# ----------- async fetcher -----------
class DataFetcher:
    def __init__(self, key: str, throttle_sec: float):
        self.key = key
        self.throttle_sec = throttle_sec
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def get(self, symbol: str, outputsize: int = MIN_CANDLES + 100) -> pd.DataFrame:
        async with self._lock:
            now = time.time()
            elapsed = now - self._last
            if elapsed < self.throttle_sec:
                await asyncio.sleep(self.throttle_sec - elapsed)
            self._last = time.time()

        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&outputsize={outputsize}&apikey={self.key}"

        for attempt in range(1, 6):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        break
                    if resp.status == 429 or resp.status >= 500:
                        delay = 5 * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"HTTP {resp.status} — retry {attempt}/5 in {delay:.1f}s")
                        await asyncio.sleep(delay)
        else:
            logger.error(f"Failed to fetch {symbol}")
            return None

        if data.get("status") == "error":
            logger.error(f"API error — {data.get('message')}")
            return None
        values = data.get("values")
        if not values:
            return None
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df[["open", "high", "low", "close"]].astype(float)
        df.columns = ["Open", "High", "Low", "Close"]
        df = df.sort_index().drop_duplicates()
        df["Volume"] = 1_000_000.0
        if len(df) < MIN_CANDLES:
            return None
        return df


# ----------- indicators -----------
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA10"] = close.rolling(10).mean()

    df["BB_Std"] = close.rolling(26).std()
    df["BB_Mid"] = close.rolling(26).mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    tr = pd.concat(
        {
            "HL": high - low,
            "HC": (high - close.shift()).abs(),
            "LC": (low - close.shift()).abs(),
        },
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr_sum = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_sum)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_sum)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    df["ADX"] = dx.rolling(14).mean()

    df["MA_Diff"] = df["MA20"] - df["MA50"]
    df["Price_to_BB_Upper"] = (close - df["BB_Upper"]) / close
    df["Price_to_BB_Lower"] = (close - df["BB_Lower"]) / close
    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA"] + 1e-10)
    df["Price_Change"] = close.pct_change(5) * 100
    
    df["Momentum"] = close - close.shift(10)
    df["ROC"] = (close / close.shift(10) - 1) * 100

    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").dropna()
    return df


# ----------- Trade Duration Predictor -----------
class TradeDurationPredictor:
    @staticmethod
    def estimate_duration(df: pd.DataFrame, latest: pd.Series, signal: str, market: dict) -> dict:
        atr = latest["ATR"]
        atr_pct = (atr / latest["Close"]) * 100
        adx = latest["ADX"]
        roc = abs(latest.get("ROC", 0))
        
        if atr_pct > 0.4:
            base_candles = 8
        elif atr_pct > 0.25:
            base_candles = 15
        elif atr_pct > 0.15:
            base_candles = 24
        else:
            base_candles = 36
        
        if adx > 35:
            base_candles *= 0.7
        elif adx < 20:
            base_candles *= 1.3
        
        if roc > 0.5:
            base_candles *= 0.8
        
        minutes = int(base_candles * 5)
        hours = minutes / 60
        
        quick_close_prob = 0.0
        if atr_pct > 0.35 and adx > 30:
            quick_close_prob = 0.75
        elif atr_pct > 0.25 and adx > 25:
            quick_close_prob = 0.50
        elif atr_pct > 0.15:
            quick_close_prob = 0.30
        
        return {
            "minutes": minutes,
            "hours": round(hours, 1),
            "quick_close_probability": round(quick_close_prob, 2),
            "volatility_level": "Высокая" if atr_pct > 0.25 else "Средняя" if atr_pct > 0.15 else "Низкая",
            "expected_close_time": datetime.now() + timedelta(minutes=minutes)
        }


# ----------- Enhanced rule-based signal -----------
def signal_rule_enhanced(latest: pd.Series, m: dict) -> Tuple[str, float]:
    atr_pct = (latest["ATR"] / latest["Close"]) * 100
    
    if atr_pct < m["min_volatility"] * 0.9:
        return "HOLD", 0.0
    
    rsi = latest["RSI"]
    close = latest["Close"]
    bb_low = latest["BB_Lower"]
    bb_high = latest["BB_Upper"]
    ma20 = latest["MA20"]
    ma10 = latest["MA10"]
    adx = latest["ADX"]
    
    # Требуем минимум ADX для любого сигнала
    if adx < 25:
        return "HOLD", 0.0
    
    signal = "HOLD"
    strength = 0.0
    
    # BUY условия
    buy_conditions = 0
    buy_strength = 0.0
    
    if close < bb_low * 1.005:
        buy_conditions += 1
        buy_strength += 0.30
        if close < bb_low * 0.998:
            buy_strength += 0.20
    
    if rsi < 40:
        buy_conditions += 1
        buy_strength += 0.25
        if rsi < 30:
            buy_strength += 0.20
    
    if ma10 > ma20:
        buy_conditions += 1
        buy_strength += 0.15
    
    if adx > 25:
        buy_strength += 0.15
        if adx > 35:
            buy_strength += 0.15
    
    # SELL условия
    sell_conditions = 0
    sell_strength = 0.0
    
    if close > bb_high * 0.995:
        sell_conditions += 1
        sell_strength += 0.30
        if close > bb_high * 1.002:
            sell_strength += 0.20
    
    if rsi > 60:
        sell_conditions += 1
        sell_strength += 0.25
        if rsi > 70:
            sell_strength += 0.20
    
    if ma10 < ma20:
        sell_conditions += 1
        sell_strength += 0.15
    
    if adx > 25:
        sell_strength += 0.15
        if adx > 35:
            sell_strength += 0.15
    
    # Требуем минимум 2 условия
    if buy_conditions >= 2 and buy_strength > sell_strength:
        signal = "BUY"
        strength = min(buy_strength, 1.0)
    elif sell_conditions >= 2 and sell_strength > buy_strength:
        signal = "SELL"
        strength = min(sell_strength, 1.0)
    
    if atr_pct > m["min_volatility"] * 1.5:
        strength *= 1.1
    
    return signal, min(strength, 1.0)


# ----------- Simple ML Predictor -----------
class SimpleMLPredictor:
    MODEL_PATH = "simple_ml.pkl"

    def __init__(self):
        self.weights_buy = None
        self.weights_sell = None
        self.mean = None
        self.std = None
        self.features = ["RSI", "ADX", "ATR", "MA_Diff", "BB_Width", "Price_to_BB_Upper", "Price_to_BB_Lower", "Price_Change"]
        self.last_train = None

    def train(self, df: pd.DataFrame) -> bool:
        X = df[self.features].fillna(0).replace([np.inf, -np.inf], 0).values
        
        future_change = df["Close"].shift(-5) / df["Close"] - 1
        y_buy = (future_change > 0.001).astype(int).values
        y_sell = (future_change < -0.001).astype(int).values
        
        X = X[:-5]
        y_buy = y_buy[:-5]
        y_sell = y_sell[:-5]
        
        if len(X) < 100:
            return False
        
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std
        
        self.weights_buy = np.random.randn(len(self.features)) * 0.1
        self.weights_sell = np.random.randn(len(self.features)) * 0.1
        
        lr = 0.01
        for _ in range(50):
            pred_buy = 1 / (1 + np.exp(-(X @ self.weights_buy)))
            grad_buy = X.T @ (pred_buy - y_buy) / len(y_buy)
            self.weights_buy -= lr * grad_buy
            
            pred_sell = 1 / (1 + np.exp(-(X @ self.weights_sell)))
            grad_sell = X.T @ (pred_sell - y_sell) / len(y_sell)
            self.weights_sell -= lr * grad_sell
        
        self.last_train = datetime.now()
        self.save()
        return True

    def predict(self, latest: pd.Series) -> Tuple[str, float]:
        if self.weights_buy is None:
            return "HOLD", 0.0
        
        x = np.array([latest[f] for f in self.features])
        x = (x - self.mean) / self.std
        
        prob_buy = 1 / (1 + np.exp(-(x @ self.weights_buy)))
        prob_sell = 1 / (1 + np.exp(-(x @ self.weights_sell)))
        
        if prob_buy > 0.70 and prob_buy > prob_sell:
            return "BUY", float(prob_buy)
        elif prob_sell > 0.70 and prob_sell > prob_buy:
            return "SELL", float(prob_sell)
        return "HOLD", 0.0

    def save(self):
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump({
                "weights_buy": self.weights_buy,
                "weights_sell": self.weights_sell,
                "mean": self.mean,
                "std": self.std,
                "time": self.last_train
            }, f)

    def load(self):
        if not os.path.exists(self.MODEL_PATH):
            return False
        try:
            with open(self.MODEL_PATH, "rb") as f:
                d = pickle.load(f)
            self.weights_buy = d["weights_buy"]
            self.weights_sell = d["weights_sell"]
            self.mean = d["mean"]
            self.std = d["std"]
            self.last_train = d.get("time")
            return True
        except Exception as e:
            logger.warning(f"ML load fail — {e}")
            return False


# ----------- Ensemble Manager -----------
class EnsembleManager:
    def __init__(self):
        self.ml_enabled = ML_ENABLED
        self.ml = SimpleMLPredictor() if ML_ENABLED else None
        self.last_train_t = time.time()
        self.retrain_int = ML_RETRAIN_SEC

    def needs_retrain(self):
        if not self.ml_enabled:
            return False
        return time.time() - self.last_train_t > self.retrain_int

    def train_all(self, df: pd.DataFrame):
        if not self.ml_enabled or self.ml is None:
            return
        logger.info("== ML retrain start ==")
        self.ml.train(df)
        self.last_train_t = time.time()
        logger.info("== ML retrain done ==")

    def load_all(self):
        if self.ml:
            self.ml.load()

    def predict(self, df: pd.DataFrame, latest: pd.Series, market: dict) -> Tuple[str, float, str, dict]:
        rule_signal, rule_strength = signal_rule_enhanced(latest, market)
        
        ml_signal, ml_conf = ("HOLD", 0.0)
        if self.ml_enabled and self.ml and self.ml.weights_buy is not None:
            ml_signal, ml_conf = self.ml.predict(latest)
        
        final_signal = "HOLD"
        final_conf = 0.0
        source = ""
        
        if ml_signal != "HOLD" and ml_conf >= ML_CONFIDENCE:
            final_signal = ml_signal
            final_conf = ml_conf
            source = "ML"
        elif rule_signal != "HOLD" and rule_strength >= MIN_SIGNAL_STRENGTH:
            final_signal = rule_signal
            final_conf = rule_strength
            source = "Rules"
        
        duration_info = {}
        if final_signal != "HOLD":
            duration_info = TradeDurationPredictor.estimate_duration(df, latest, final_signal, market)
        
        return final_signal, final_conf, source, duration_info


# ----------- Risk Manager с защитой от overtrading -----------
class RiskManager:
    def __init__(self):
        self.initial = INITIAL_BALANCE
        self.balance = self.initial
        self.open_trades: Dict[str, dict] = {}
        self.max_trades = MAX_OPEN_TRADES
        self.risk = RISK_PER_TRADE
        self.counter = 0
        self.load_state()

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "rb") as f:
                d = pickle.load(f)
                self.balance = d.get("balance", self.initial)
                self.open_trades = d.get("open_trades", {})
                self.counter = d.get("counter", 0)
                logger.info(f"💾 State loaded: Balance ${self.balance:.2f}, Open trades: {len(self.open_trades)}")
        except Exception as e:
            logger.error(f"State load — {e}")

    def save_state(self):
        with open(STATE_FILE, "wb") as f:
            pickle.dump({"balance": self.balance, "open_trades": self.open_trades, "counter": self.counter}, f)
        logger.info(f"💾 State saved: Balance ${self.balance:.2f}, Trades: {len(self.open_trades)}")

    def can_open(self):
        return len(self.open_trades) < self.max_trades

    def can_open_for_market(self, market_name: str):
        """Проверка: нет ли уже открытой позиции на этой паре"""
        for trade in self.open_trades.values():
            if trade["market"] == market_name:
                logger.warning(f"⚠️ {market_name} уже имеет открытую позицию (ID: {trade['id']})")
                return False
        return self.can_open()

    def calc_lot(self, market: dict, atr: float) -> float:
        risk_amount = self.balance * self.risk
        sl_pips = atr / market["pip_value"]
        pip_cost_per_lot = market["pip_value"] * 100_000
        lot = risk_amount / (sl_pips * pip_cost_per_lot)
        return max(0.01, min(lot, 10.0))

    def add_trade(self, market: dict, signal: str, price: float, atr: float, duration_info: dict) -> dict:
        self.counter += 1
        trade_id = f"T{self.counter}"
        lot = self.calc_lot(market, atr)
        sl_dist = atr
        tp_dist = sl_dist * market["rr"]
        sl_price = price - sl_dist if signal == "BUY" else price + sl_dist
        tp_price = price + tp_dist if signal == "BUY" else price - tp_dist

        trade = {
            "id": trade_id,
            "market": market["name"],
            "signal": signal,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "lot": lot,
            "risk": self.balance * self.risk,
            "time": datetime.now(),
            "duration_estimate": duration_info,
            "spread_cost": market.get("spread_pips", 2) * market["pip_value"] * 100_000 * lot,
        }
        self.open_trades[trade_id] = trade
        self.save_state()
        return trade

    def check_close(self, market_name: str, cur_price: float):
        closed = []
        market = None
        for m in CONFIG["markets"]:
            if m["name"] == market_name:
                market = m
                break
        if market is None:
            return closed

        for tid, t in list(self.open_trades.items()):
            if t["market"] != market_name:
                continue
            reason = None
            if t["signal"] == "BUY":
                if cur_price <= t["sl"]:
                    reason = "SL"
                elif cur_price >= t["tp"]:
                    reason = "TP"
            else:
                if cur_price >= t["sl"]:
                    reason = "SL"
                elif cur_price <= t["tp"]:
                    reason = "TP"
            if reason:
                pips = (cur_price - t["price"]) / market["pip_value"] if t["signal"] == "BUY" else (t["price"] - cur_price) / market["pip_value"]
                pnl = pips * market["pip_value"] * 100_000 * t["lot"]
                
                spread_cost = t.get("spread_cost", 0)
                pnl -= spread_cost
                
                self.balance += pnl
                
                actual_duration = (datetime.now() - t["time"]).total_seconds() / 60
                
                closed.append({
                    "id": tid,
                    "pnl": pnl,
                    "reason": reason,
                    "balance": self.balance,
                    "actual_duration_min": int(actual_duration),
                    "estimated_duration_min": t.get("duration_estimate", {}).get("minutes", 0),
                    "spread_cost": spread_cost
                })
                del self.open_trades[tid]
        if closed:
            self.save_state()
        return closed

    def close_manual(self, trade_id: str, pnl: float):
        if trade_id in self.open_trades:
            del self.open_trades[trade_id]
            self.balance += pnl
            self.save_state()
            return True
        return False


# ----------- Telegram -----------
async def send_signal(app: Application, market: dict, latest: pd.Series, signal_data: Tuple[str, float, str, dict], rm: RiskManager):
    signal, conf, source, duration_info = signal_data
    if not rm.can_open_for_market(market["name"]):
        logger.warning(f"{market['name']} signal blocked - уже есть открытая позиция или лимит достигнут")
        return
    
    trade = rm.add_trade(market, signal, latest["Close"], latest["ATR"], duration_info)
    emoji = "🟢" if signal == "BUY" else "🔴"
    
    dur_text = ""
    if duration_info:
        hours = duration_info.get("hours", 0)
        minutes = duration_info.get("minutes", 0)
        vol_level = duration_info.get("volatility_level", "Средняя")
        quick_prob = duration_info.get("quick_close_probability", 0)
        
        if hours >= 1:
            dur_text = f"⏱ **Прогноз:** ~{hours:.1f}ч ({minutes}мин)\n"
        else:
            dur_text = f"⏱ **Прогноз:** ~{minutes} минут\n"
        
        dur_text += f"📊 **Волатильность:** {vol_level}\n"
        
        if quick_prob > 0.6:
            dur_text += f"⚡ **Быстрое закрытие:** {quick_prob*100:.0f}%\n"
    
    msg = (
        f"{emoji} **НОВЫЙ СИГНАЛ: {market['name']}**\n"
        f"**Направление:** {signal}\n"
        f"**Цена входа:** {trade['price']:.{market['precision']}f}\n"
        f"**Источник:** {source} (Уверенность: {conf*100:.0f}%)\n\n"
        f"{dur_text}"
        f"**Лот:** {trade['lot']:.2f}\n"
        f"**Stop Loss:** {trade['sl']:.{market['precision']}f}\n"
        f"**Take Profit:** {trade['tp']:.{market['precision']}f}\n"
        f"**Риск:** ${trade['risk']:.2f}\n"
        f"**Спред:** ${trade['spread_cost']:.2f}\n"
        f"**Баланс:** ${rm.balance:.2f}"
    )
    kb = [[InlineKeyboardButton("❌ Закрыть сделку", callback_data=f"close_trade|{trade['id']}")]]
    await app.bot.send_message(TELEGRAM_CHAT_ID, msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    logger.info(f"✅ Signal sent — {market['name']} {signal} (conf {conf:.2f}, source {source})")


async def handle_callback(update, context):
    query = update.callback_query
    await query.answer()
    try:
        action, tid = query.data.split("|")
    except ValueError:
        return
    if action == "close_trade":
        if context.application.chat_data["rm"].close_manual(tid, 0.0):
            await query.edit_message_text(f"✅ Сделка {tid} закрыта вручную.")
        else:
            await query.edit_message_text(f"⚠️ Сделка {tid} не найдена.")


# ----------- Main Loop -----------
async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    rm = RiskManager()
    app.chat_data = {"rm": rm}
    app.add_handler(CallbackQueryHandler(handle_callback))

    fetcher = DataFetcher(TWELVE_DATA_KEY, THROTTLE_SEC)
    ensemble = EnsembleManager()
    ensemble.load_all()

    await app.initialize()
    await app.start()
    
    await app.bot.send_message(
        TELEGRAM_CHAT_ID, 
        f"🚀 **Оптимизированный скальпер запущен!**\n\n"
        f"✅ Торговые часы: {TRADING_HOURS_START:02d}:00-{TRADING_HOURS_END:02d}:00 UTC\n"
        f"✅ Минимальная уверенность: {ML_CONFIDENCE*100:.0f}% (ML) / {MIN_SIGNAL_STRENGTH*100:.0f}% (Rules)\n"
        f"✅ Риск на сделку: {RISK_PER_TRADE*100:.1f}%\n"
        f"✅ Максимум открытых: {MAX_OPEN_TRADES}\n"
        f"✅ Лимит сигналов: {MAX_SIGNALS_PER_PAIR}/пара/день\n"
        f"✅ Cooldown: {SIGNAL_COOLDOWN_MIN} минут\n"
        f"✅ Активных пар: {len(CONFIG['markets'])}\n"
        f"💰 Стартовый баланс: ${rm.balance:.2f}",
        parse_mode="Markdown"
    )

    store = {}
    scan = 0
    skipped_scans = 0
    session_open = False
    session_closed_today = False
    last_session_date = None
    
    # Счётчики для защиты от overtrading
    last_signal_time = {}  # {market_name: timestamp}
    daily_signal_count = {}  # {market_name: count}
    
    try:
        while True:
            scan += 1
            
            # Проверка торгового времени
            is_trading, time_status = is_good_trading_time()
            current_hour = datetime.utcnow().hour
            current_date = datetime.utcnow().date()
            
            # Сброс счётчиков при новом дне
            if last_session_date != current_date:
                session_closed_today = False
                daily_signal_count = {}  # Сброс дневных счётчиков
                last_session_date = current_date
                logger.info(f"🗓 Новый торговый день: {current_date}")
            
            if not is_trading:
                # Закрытие сессии в конце торгового дня
                if current_hour >= TRADING_HOURS_END and not session_closed_today:
                    logger.info(f"🔴 Торговая сессия закрыта - конец рабочего дня")
                    
                    open_count = len(rm.open_trades)
                    open_trades_text = ""
                    
                    if rm.open_trades:
                        open_trades_text = "\n\n⚠️ **Открытые позиции (overnight):**\n"
                        for t in rm.open_trades.values():
                            elapsed = (datetime.now() - t["time"]).total_seconds() / 60
                            open_trades_text += f"• {t['market']} {t['signal']} ({int(elapsed)}мин)\n"
                    
                    daily_pnl = rm.balance - rm.initial
                    daily_pnl_pct = (daily_pnl / rm.initial) * 100
                    pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
                    
                    # Статистика по сигналам
                    total_signals = sum(daily_signal_count.values())
                    signals_text = ""
                    if daily_signal_count:
                        signals_text = f"\n📊 Сигналов за день: {total_signals}\n"
                        for pair, count in sorted(daily_signal_count.items()):
                            signals_text += f"  • {pair}: {count}\n"
                    
                    await app.bot.send_message(
                        TELEGRAM_CHAT_ID,
                        f"🔴 **ТОРГОВАЯ СЕССИЯ ЗАКРЫТА**\n\n"
                        f"⏰ Время: {TRADING_HOURS_END:02d}:00 UTC\n"
                        f"💰 Итоговый баланс: ${rm.balance:.2f}\n"
                        f"{pnl_emoji} Дневной PnL: ${daily_pnl:+.2f} ({daily_pnl_pct:+.1f}%)\n"
                        f"📊 Открытых позиций: {open_count}/{MAX_OPEN_TRADES}"
                        f"{signals_text}"
                        f"{open_trades_text}\n\n"
                        f"💤 Следующая сессия: {TRADING_HOURS_START:02d}:00 UTC\n"
                        f"⏱ До открытия: {24 - current_hour + TRADING_HOURS_START} часов",
                        parse_mode="Markdown"
                    )
                    
                    session_open = False
                    session_closed_today = True
                
                # Обычное сообщение о паузе
                if skipped_scans == 0 and not session_closed_today:
                    logger.info(f"⏰ {time_status} - Приостановка сканирования")
                    await app.bot.send_message(
                        TELEGRAM_CHAT_ID,
                        f"⏰ **Вне торговых часов**\n{time_status}\n"
                        f"Следующий скан в {TRADING_HOURS_START:02d}:00 UTC",
                        parse_mode="Markdown"
                    )
                
                skipped_scans += 1
                await asyncio.sleep(SCAN_INTERVAL_SEC)
                continue
            
            # Открытие новой сессии
            if not session_open:
                logger.info(f"🟢 Торговая сессия открыта - начало рабочего дня")
                await app.bot.send_message(
                    TELEGRAM_CHAT_ID,
                    f"🟢 **ТОРГОВАЯ СЕССИЯ ОТКРЫТА**\n\n"
                    f"⏰ London/US session активна\n"
                    f"🕐 Время работы: {TRADING_HOURS_START:02d}:00-{TRADING_HOURS_END:02d}:00 UTC\n"
                    f"💰 Стартовый баланс: ${rm.balance:.2f}\n"
                    f"📊 Открыто позиций: {len(rm.open_trades)}/{MAX_OPEN_TRADES}\n\n"
                    f"✅ Начинаем сканирование...",
                    parse_mode="Markdown"
                )
                session_open = True
                session_closed_today = False
                skipped_scans = 0
            
            logger.info(f"\n{'='*60}\n📡 Скан #{scan} | {datetime.now():%H:%M:%S} UTC\n{'='*60}")
            
            # Предупреждение за 15 минут до закрытия
            minutes_to_close = (TRADING_HOURS_END - current_hour) * 60 - datetime.utcnow().minute
            if 10 <= minutes_to_close <= 15 and scan % 3 == 0:
                open_count = len(rm.open_trades)
                if open_count > 0:
                    await app.bot.send_message(
                        TELEGRAM_CHAT_ID,
                        f"⏰ **Внимание: До закрытия сессии {minutes_to_close} минут**\n\n"
                        f"Открытых позиций: {open_count}\n"
                        f"Новые сигналы НЕ открываются\n"
                        f"Позиции будут держаться overnight или закроются по SL/TP",
                        parse_mode="Markdown"
                    )
            
            # Retrain if needed
            if ensemble.needs_retrain():
                combo = pd.concat([df for df in store.values() if len(df) > 200], ignore_index=True)
                if len(combo) > 1000:
                    ensemble.train_all(combo)

            signals_sent = 0
            
            for market in CONFIG["markets"]:
                df = await fetcher.get(market["symbol"])
                if df is None or len(df) < MIN_CANDLES:
                    continue
                
                df = calc_indicators(df)
                store[market["symbol"]] = df
                latest = df.iloc[-1]

                # Check and close trades
                for closed in rm.check_close(market["name"], latest["Close"]):
                    pnl_sign = "+" if closed["pnl"] >= 0 else ""
                    pnl_emoji = "💰" if closed["pnl"] >= 0 else "📉"
                    
                    actual = closed["actual_duration_min"]
                    estimated = closed["estimated_duration_min"]
                    time_diff = ""
                    if estimated > 0:
                        diff_pct = abs(actual - estimated) / estimated * 100
                        if diff_pct < 20:
                            time_diff = f"\n⏱ Прогноз: ✅ Точный ({actual}мин vs {estimated}мин)"
                        else:
                            time_diff = f"\n⏱ Прогноз: ⚠️ Отклонение ({actual}мин vs {estimated}мин)"
                    
                    spread_text = f"\n💸 Спред: -${closed['spread_cost']:.2f}"
                    
                    await app.bot.send_message(
                        TELEGRAM_CHAT_ID,
                        f"{pnl_emoji} **Сделка {closed['id']} закрыта по {closed['reason']}**\n"
                        f"PnL: {pnl_sign}${closed['pnl']:.2f}{spread_text}\n"
                        f"Баланс: ${closed['balance']:.2f}"
                        f"{time_diff}",
                        parse_mode="Markdown"
                    )

                # Generate signal
                signal, conf, source, duration_info = ensemble.predict(df, latest, market)
                
                # Проверки перед открытием позиции
                if signal in ("BUY", "SELL"):
                    # 1. Проверка времени до закрытия
                    if minutes_to_close <= 30:
                        logger.warning(f"⏰ {market['name']} {signal} пропущен - до закрытия сессии {minutes_to_close} минут")
                        continue
                    
                    # 2. Проверка дневного лимита сигналов
                    if daily_signal_count.get(market["name"], 0) >= MAX_SIGNALS_PER_PAIR:
                        logger.warning(f"🚫 {market['name']} {signal} пропущен - лимит {MAX_SIGNALS_PER_PAIR} сигналов/день достигнут")
                        continue
                    
                    # 3. Проверка cooldown
                    now = time.time()
                    last_time = last_signal_time.get(market["name"], 0)
                    cooldown_remaining = SIGNAL_COOLDOWN_MIN * 60 - (now - last_time)
                    
                    if cooldown_remaining > 0:
                        logger.warning(f"⏰ {market['name']} {signal} пропущен - cooldown {int(cooldown_remaining/60)}мин")
                        continue
                    
                    # 4. Проверка на повторную позицию
                    if rm.can_open_for_market(market["name"]):
                        await send_signal(app, market, latest, (signal, conf, source, duration_info), rm)
                        signals_sent += 1
                        
                        # Обновляем счётчики
                        last_signal_time[market["name"]] = now
                        daily_signal_count[market["name"]] = daily_signal_count.get(market["name"], 0) + 1

            # Status update
            open_count = len(rm.open_trades)
            status_emoji = "✅" if signals_sent > 0 else "🔍"
            logger.info(
                f"{status_emoji} Скан завершён | Сигналов: {signals_sent} | "
                f"Открыто: {open_count}/{rm.max_trades} | Баланс: ${rm.balance:.2f}"
            )
            
            # Periodic status (every 12 scans = 1 hour)
            if scan % 12 == 0:
                open_trades_text = ""
                if rm.open_trades:
                    open_trades_text = "\n\n📋 **Открытые сделки:**\n"
                    for t in rm.open_trades.values():
                        elapsed = (datetime.now() - t["time"]).total_seconds() / 60
                        open_trades_text += f"• {t['market']} {t['signal']} ({int(elapsed)}мин)\n"
                
                pnl = rm.balance - rm.initial
                pnl_pct = (pnl / rm.initial) * 100
                pnl_emoji = "📈" if pnl >= 0 else "📉"
                
                await app.bot.send_message(
                    TELEGRAM_CHAT_ID,
                    f"📊 **Статус бота (Скан #{scan})**\n"
                    f"Баланс: ${rm.balance:.2f}\n"
                    f"{pnl_emoji} PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                    f"Открыто сделок: {open_count}/{rm.max_trades}"
                    f"{open_trades_text}",
                    parse_mode="Markdown"
                )
            
            logger.info(f"💤 Следующий скан через {SCAN_INTERVAL_SEC}с...")
            await asyncio.sleep(SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        logger.info("⚠️ KeyboardInterrupt — остановка.")
        await app.bot.send_message(TELEGRAM_CHAT_ID, "⛔ Бот остановлен вручную.")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка — {e}", exc_info=True)
        await app.bot.send_message(TELEGRAM_CHAT_ID, f"❌ Критическая ошибка: {e}\nБот остановлен.")
    finally:
        await app.stop()
        await app.shutdown()
        logger.info("👋 Работа завершена.")


if __name__ == "__main__":
    asyncio.run(main())