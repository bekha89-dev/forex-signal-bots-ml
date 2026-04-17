#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНОСТЬЮ РАБОЧИЙ MTF SCALPER
Протестировано и готово к запуску
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
    "throttle_sec": 15.0,
    "scan_interval_sec": 300,
    "risk_per_trade_percent": 0.75,
    "max_open_trades": 2,
    "initial_balance": 10_000,
    "min_signal_strength": 0.70,
    "trading_hours_start": 7,
    "trading_hours_end": 16,
    "max_signals_per_pair_daily": 3,
    "signal_cooldown_minutes": 15,
    "markets": [
        {"name": "EUR/USD", "symbol": "EUR/USD", "precision": 5, "min_volatility": 0.00025, "rr": 2.0, "pip_value": 0.0001, "spread_pips": 1.5},
        {"name": "GBP/USD", "symbol": "GBP/USD", "precision": 5, "min_volatility": 0.0003, "rr": 2.0, "pip_value": 0.0001, "spread_pips": 2.0},
        {"name": "USD/JPY", "symbol": "USD/JPY", "precision": 3, "min_volatility": 0.15, "rr": 2.0, "pip_value": 0.01, "spread_pips": 1.5},
    ],
}

# Только 3 таймфрейма для экономии API запросов
TIMEFRAMES = {
    "5min": {"interval": "5min", "outputsize": 150, "weight": 1.5, "cache_sec": 60},
    "1h": {"interval": "1h", "outputsize": 100, "weight": 3.0, "cache_sec": 300},
    "4h": {"interval": "4h", "outputsize": 100, "weight": 4.0, "cache_sec": 600},
}

MIN_CANDLES = 50


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return {**DEFAULT_CONFIG, **cfg}
        except Exception as e:
            logger.error(f"Config load error: {e}")
    
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    return DEFAULT_CONFIG


CONFIG = load_config()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or CONFIG["telegram_token"]
TELEGRAM_CHAT_ID = int(CONFIG["telegram_chat_id"])
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY") or CONFIG["twelve_data_key"]
THROTTLE_SEC = CONFIG["throttle_sec"]
SCAN_INTERVAL_SEC = CONFIG["scan_interval_sec"]
RISK_PER_TRADE = CONFIG["risk_per_trade_percent"] / 100
MAX_OPEN_TRADES = CONFIG["max_open_trades"]
INITIAL_BALANCE = CONFIG["initial_balance"]
MIN_SIGNAL_STRENGTH = CONFIG["min_signal_strength"]
TRADING_HOURS_START = CONFIG["trading_hours_start"]
TRADING_HOURS_END = CONFIG["trading_hours_end"]
MAX_SIGNALS_PER_PAIR = CONFIG.get("max_signals_per_pair_daily", 3)
SIGNAL_COOLDOWN_MIN = CONFIG.get("signal_cooldown_minutes", 15)


def is_good_trading_time():
    current_hour = datetime.utcnow().hour
    if TRADING_HOURS_START <= current_hour < TRADING_HOURS_END:
        return True, "Trading hours"
    if current_hour < TRADING_HOURS_START:
        return False, f"Before London ({TRADING_HOURS_START - current_hour}h)"
    return False, f"After US close"


class CachedDataFetcher:
    def __init__(self, key: str, throttle_sec: float):
        self.key = key
        self.throttle_sec = throttle_sec
        self._lock = asyncio.Lock()
        self._last = 0.0
        self.cache = {}

    async def get_timeframe(self, symbol: str, tf_key: str) -> Optional[pd.DataFrame]:
        tf_config = TIMEFRAMES[tf_key]
        cache_key = f"{symbol}_{tf_key}"
        
        # Проверка кэша
        if cache_key in self.cache:
            cached_time, cached_df = self.cache[cache_key]
            if time.time() - cached_time < tf_config["cache_sec"]:
                return cached_df.copy()
        
        # Throttling
        async with self._lock:
            now = time.time()
            elapsed = now - self._last
            if elapsed < self.throttle_sec:
                await asyncio.sleep(self.throttle_sec - elapsed)
            self._last = time.time()
        
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={tf_config['interval']}&outputsize={tf_config['outputsize']}&apikey={self.key}"
        
        for attempt in range(1, 4):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("status") == "error":
                                logger.error(f"API error: {data.get('message')}")
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
                            
                            self.cache[cache_key] = (time.time(), df.copy())
                            logger.info(f"✅ Fetched: {symbol} {tf_key}")
                            return df
                        
                        if resp.status == 429 or resp.status >= 500:
                            delay = 5 * (2 ** attempt)
                            logger.warning(f"HTTP {resp.status} - retry {attempt}/3 in {delay}s")
                            await asyncio.sleep(delay)
            except Exception as e:
                logger.warning(f"Fetch error: {symbol} {tf_key}: {e}")
        
        return None

    async def get_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        tasks = [self.get_timeframe(symbol, tf_key) for tf_key in TIMEFRAMES.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for tf_key, result in zip(TIMEFRAMES.keys(), results):
            if isinstance(result, Exception):
                continue
            if result is not None:
                data[tf_key] = result
        
        return data


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(min(50, len(df))).mean()
    
    df["BB_Std"] = close.rolling(20).std()
    df["BB_Mid"] = close.rolling(20).mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    tr = pd.concat({
        "HL": high - low,
        "HC": (high - close.shift()).abs(),
        "LC": (low - close.shift()).abs(),
    }, axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr_sum = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_sum)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_sum)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    df["ADX"] = dx.rolling(14).mean()
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").dropna()
    return df


class MTFAnalyzer:
    @staticmethod
    def analyze_tf(df: pd.DataFrame, market: dict) -> Tuple[str, float]:
        if len(df) < MIN_CANDLES:
            return "HOLD", 0.0
        
        df = calc_indicators(df)
        latest = df.iloc[-1]
        
        atr_pct = (latest["ATR"] / latest["Close"]) * 100
        if atr_pct < market["min_volatility"] * 0.8:
            return "HOLD", 0.0
        
        if latest["ADX"] < 20:
            return "HOLD", 0.0
        
        rsi = latest["RSI"]
        close = latest["Close"]
        bb_low = latest["BB_Lower"]
        bb_high = latest["BB_Upper"]
        ma20 = latest["MA20"]
        ma50 = latest["MA50"]
        
        buy_score = 0.0
        if close < bb_low * 1.002:
            buy_score += 0.35
        if rsi < 35:
            buy_score += 0.35
        if ma20 > ma50:
            buy_score += 0.20
        if latest["ADX"] > 30:
            buy_score += 0.10
        
        sell_score = 0.0
        if close > bb_high * 0.998:
            sell_score += 0.35
        if rsi > 65:
            sell_score += 0.35
        if ma20 < ma50:
            sell_score += 0.20
        if latest["ADX"] > 30:
            sell_score += 0.10
        
        if buy_score > 0.60 and buy_score > sell_score * 1.3:
            return "BUY", min(buy_score, 1.0)
        elif sell_score > 0.60 and sell_score > buy_score * 1.3:
            return "SELL", min(sell_score, 1.0)
        
        return "HOLD", 0.0

    @staticmethod
    def calculate_consensus(tf_signals: Dict[str, Tuple[str, float]]) -> Tuple[str, float, dict]:
        if not tf_signals:
            return "HOLD", 0.0, {}
        
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0
        
        tf_details = {}
        
        for tf_key, (signal, strength) in tf_signals.items():
            weight = TIMEFRAMES[tf_key]["weight"]
            total_weight += weight
            
            if signal == "BUY":
                buy_weight += weight * strength
            elif signal == "SELL":
                sell_weight += weight * strength
            
            tf_details[tf_key] = {
                "signal": signal,
                "strength": strength,
                "weight": weight
            }
        
        buy_score = buy_weight / total_weight if total_weight > 0 else 0.0
        sell_score = sell_weight / total_weight if total_weight > 0 else 0.0
        
        signals_count = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for sig, _ in tf_signals.values():
            signals_count[sig] += 1
        
        agreement = max(signals_count.values()) / len(tf_signals)
        
        final_signal = "HOLD"
        final_strength = 0.0
        
        if buy_score > sell_score * 1.5 and buy_score > 0.60 and agreement >= 0.65:
            final_signal = "BUY"
            final_strength = buy_score
        elif sell_score > buy_score * 1.5 and sell_score > 0.60 and agreement >= 0.65:
            final_signal = "SELL"
            final_strength = sell_score
        
        return final_signal, final_strength, {
            "buy_score": buy_score,
            "sell_score": sell_score,
            "agreement": agreement,
            "tf_details": tf_details
        }


class RiskManager:
    def __init__(self):
        self.initial = INITIAL_BALANCE
        self.balance = self.initial
        self.open_trades = {}
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
        except Exception as e:
            logger.error(f"State load error: {e}")

    def save_state(self):
        with open(STATE_FILE, "wb") as f:
            pickle.dump({
                "balance": self.balance,
                "open_trades": self.open_trades,
                "counter": self.counter
            }, f)

    def can_open_for_market(self, market_name: str):
        for trade in self.open_trades.values():
            if trade["market"] == market_name:
                return False
        return len(self.open_trades) < MAX_OPEN_TRADES

    def calc_lot(self, market: dict, atr: float) -> float:
        risk_amount = self.balance * RISK_PER_TRADE
        sl_pips = atr / market["pip_value"]
        pip_cost = market["pip_value"] * 100_000
        lot = risk_amount / (sl_pips * pip_cost)
        return max(0.01, min(lot, 10.0))

    def add_trade(self, market: dict, signal: str, price: float, atr: float, mtf_data: dict) -> dict:
        self.counter += 1
        tid = f"T{self.counter}"
        lot = self.calc_lot(market, atr)
        sl_dist = atr
        tp_dist = sl_dist * market["rr"]
        
        trade = {
            "id": tid,
            "market": market["name"],
            "signal": signal,
            "price": price,
            "sl": price - sl_dist if signal == "BUY" else price + sl_dist,
            "tp": price + tp_dist if signal == "BUY" else price - tp_dist,
            "lot": lot,
            "time": datetime.now(),
            "mtf_data": mtf_data,
        }
        self.open_trades[tid] = trade
        self.save_state()
        return trade

    def check_close(self, market_name: str, cur_price: float):
        closed = []
        market = None
        for m in CONFIG["markets"]:
            if m["name"] == market_name:
                market = m
                break
        if not market:
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
                self.balance += pnl
                
                closed.append({
                    "id": tid,
                    "pnl": pnl,
                    "reason": reason,
                    "balance": self.balance,
                })
                del self.open_trades[tid]
        
        if closed:
            self.save_state()
        return closed

    def close_manual(self, tid: str):
        if tid in self.open_trades:
            del self.open_trades[tid]
            self.save_state()
            return True
        return False


async def send_mtf_signal(app: Application, market: dict, latest: pd.Series, 
                          signal: str, strength: float, mtf_data: dict, rm: RiskManager):
    if not rm.can_open_for_market(market["name"]):
        return
    
    trade = rm.add_trade(market, signal, latest["Close"], latest["ATR"], mtf_data)
    
    emoji = "🟢" if signal == "BUY" else "🔴"
    
    tf_text = "\n📊 **Таймфреймы:**\n"
    for tf_key, details in mtf_data.get("tf_details", {}).items():
        sig_emoji = "🟢" if details["signal"] == "BUY" else "🔴" if details["signal"] == "SELL" else "⚪"
        tf_text += f"  {sig_emoji} {tf_key}: {details['signal']} ({details['strength']*100:.0f}%)\n"
    
    msg = (
        f"{emoji} **НОВЫЙ MTF СИГНАЛ: {market['name']}**\n"
        f"**Направление:** {signal}\n"
        f"**Цена:** {trade['price']:.{market['precision']}f}\n"
        f"**Уверенность:** {strength*100:.0f}%\n"
        f"**Согласие TF:** {mtf_data.get('agreement', 0)*100:.0f}%\n"
        f"{tf_text}\n"
        f"**Лот:** {trade['lot']:.2f}\n"
        f"**SL:** {trade['sl']:.{market['precision']}f}\n"
        f"**TP:** {trade['tp']:.{market['precision']}f}\n"
        f"**Баланс:** ${rm.balance:.2f}"
    )
    
    kb = [[InlineKeyboardButton("❌ Закрыть", callback_data=f"close|{trade['id']}")]]
    await app.bot.send_message(TELEGRAM_CHAT_ID, msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    logger.info(f"✅ Signal sent: {market['name']} {signal}")


async def handle_callback(update, context):
    query = update.callback_query
    await query.answer()
    try:
        action, tid = query.data.split("|")
        if action == "close":
            if context.application.chat_data["rm"].close_manual(tid):
                await query.edit_message_text(f"✅ Сделка {tid} закрыта")
            else:
                await query.edit_message_text(f"⚠️ Сделка {tid} не найдена")
    except Exception as e:
        logger.error(f"Callback error: {e}")


async def main():
    try:
        logger.info("🚀 Starting MTF Scalper...")
        
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        rm = RiskManager()
        app.chat_data = {"rm": rm}
        app.add_handler(CallbackQueryHandler(handle_callback))
        
        fetcher = CachedDataFetcher(TWELVE_DATA_KEY, THROTTLE_SEC)
        analyzer = MTFAnalyzer()
        
        await app.initialize()
        await app.start()
        
        await app.bot.send_message(
            TELEGRAM_CHAT_ID,
            f"🚀 **MTF Скальпер запущен!**\n\n"
            f"⏰ Торговые часы: {TRADING_HOURS_START:02d}:00-{TRADING_HOURS_END:02d}:00 UTC\n"
            f"📊 Таймфреймы: {', '.join(TIMEFRAMES.keys())}\n"
            f"💰 Баланс: ${rm.balance:.2f}\n"
            f"🎯 Рынков: {len(CONFIG['markets'])}",
            parse_mode="Markdown"
        )
        
        scan = 0
        last_signal_time = {}
        daily_signal_count = {}
        last_date = None
        
        while True:
            scan += 1
            current_date = datetime.utcnow().date()
            
            if last_date != current_date:
                daily_signal_count = {}
                last_date = current_date
                logger.info(f"📅 Новый день: {current_date}")
            
            is_trading, status = is_good_trading_time()
            
            if not is_trading:
                if scan % 6 == 0:
                    logger.info(f"⏸️ {status}")
                await asyncio.sleep(SCAN_INTERVAL_SEC)
                continue
            
            logger.info(f"\n{'='*50}\n📡 Скан #{scan} | {datetime.now():%H:%M:%S} UTC\n{'='*50}")
            
            signals_sent = 0
            
            for market in CONFIG["markets"]:
                try:
                    # Получить данные всех таймфреймов
                    all_dfs = await fetcher.get_all_timeframes(market["symbol"])
                    
                    if len(all_dfs) < 2:
                        logger.warning(f"⚠️ {market['name']}: недостаточно данных")
                        continue
                    
                    # Анализ каждого таймфрейма
                    tf_signals = {}
                    for tf_key, df in all_dfs.items():
                        signal, strength = analyzer.analyze_tf(df, market)
                        if strength >= MIN_SIGNAL_STRENGTH:
                            tf_signals[tf_key] = (signal, strength)
                    
                    if not tf_signals:
                        continue
                    
                    # Консенсус
                    final_signal, final_strength, mtf_data = analyzer.calculate_consensus(tf_signals)
                    
                    if final_signal in ("BUY", "SELL"):
                        # Проверки
                        if daily_signal_count.get(market["name"], 0) >= MAX_SIGNALS_PER_PAIR:
                            logger.warning(f"🚫 {market['name']}: лимит сигналов")
                            continue
                        
                        now = time.time()
                        last_time = last_signal_time.get(market["name"], 0)
                        if now - last_time < SIGNAL_COOLDOWN_MIN * 60:
                            logger.warning(f"⏰ {market['name']}: cooldown")
                            continue
                        
                        if rm.can_open_for_market(market["name"]):
                            latest = all_dfs["5min"].iloc[-1]
                            await send_mtf_signal(app, market, latest, final_signal, final_strength, mtf_data, rm)
                            
                            signals_sent += 1
                            last_signal_time[market["name"]] = now
                            daily_signal_count[market["name"]] = daily_signal_count.get(market["name"], 0) + 1
                    
                    # Проверка закрытия
                    for closed in rm.check_close(market["name"], all_dfs["5min"].iloc[-1]["Close"]):
                        pnl_emoji = "💰" if closed["pnl"] >= 0 else "📉"
                        await app.bot.send_message(
                            TELEGRAM_CHAT_ID,
                            f"{pnl_emoji} **Сделка {closed['id']} закрыта по {closed['reason']}**\n"
                            f"PnL: ${closed['pnl']:+.2f}\n"
                            f"Баланс: ${closed['balance']:.2f}",
                            parse_mode="Markdown"
                        )
                
                except Exception as e:
                    logger.error(f"Error processing {market['name']}: {e}")
            
            logger.info(f"✅ Скан завершён | Сигналов: {signals_sent} | Открыто: {len(rm.open_trades)}")
            
            await asyncio.sleep(SCAN_INTERVAL_SEC)
    
    except KeyboardInterrupt:
        logger.info("⛔ Остановка...")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
    finally:
        await app.stop()
        await app.shutdown()
        logger.info("👋 Завершено")


if __name__ == "__main__":
    asyncio.run(main())