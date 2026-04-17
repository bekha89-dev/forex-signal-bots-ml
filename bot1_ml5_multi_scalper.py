import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
import warnings
import threading
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler
import asyncio
import pickle
import os
import json
import logging
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler('scalper.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {"telegram_token": "YOUR_TELEGRAM_BOT_TOKEN", "telegram_chat_id": "YOUR_CHAT_ID", "throttle_sec": 8.5, "max_workers": 1, "min_candles": 100, "scan_interval_sec": 300, "ml_enabled": True, "ml_retrain_hours": 24, "ml_confidence_threshold": 0.70, "risk_per_trade_percent": 1.5, "max_open_trades": 3, "initial_balance": 10000}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logger.info(f"Конфигурация загружена")
            return config
        except Exception as e:
            logger.warning(f"Ошибка чтения config: {e}")
    with open(CONFIG_FILE, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    logger.info(f"Создан новый config")
    return DEFAULT_CONFIG

CONFIG = load_config()
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY") or CONFIG.get("twelve_data_key", "")
THROTTLE_SEC = CONFIG.get("throttle_sec", 8.5)
MAX_WORKERS = CONFIG.get("max_workers", 1)
MIN_CANDLES = CONFIG["min_candles"]
TELEGRAM_TOKEN = CONFIG["telegram_token"]
TELEGRAM_CHAT_ID = CONFIG["telegram_chat_id"]
ML_ENABLED = CONFIG["ml_enabled"]
ML_RETRAIN_HOURS = CONFIG["ml_retrain_hours"]
ML_CONFIDENCE = CONFIG["ml_confidence_threshold"]
RISK_PERCENT = CONFIG["risk_per_trade_percent"] / 100
MAX_OPEN_TRADES = CONFIG["max_open_trades"]
ML_MODEL_PATH = "scalper_nn_model.pkl"
LSTM_MODEL_PATH = "scalper_lstm_model.pkl"
CONFIDENCE_MODEL_PATH = "scalper_confidence_model.pkl"
INITIAL_BALANCE = CONFIG["initial_balance"]

GREEN = "\u001B[92m"
RED = "\u001B[91m"
YELLOW = "\u001B[93m"
RESET = "\u001B[0m"

MARKETS = {
    'AUDJPY': {'symbol': 'AUDJPY=X', 'name': 'AUD/JPY', 'precision': 3, 'min_vol': 0.15, 'rr': 2.0, 'pip_value': 0.01, 'min_lot': 0.01, 'max_lot': 100.0},
    'EURCAD': {'symbol': 'EURCAD=X', 'name': 'EUR/CAD', 'precision': 5, 'min_vol': 0.0003, 'rr': 2.0, 'pip_value': 0.0001, 'min_lot': 0.01, 'max_lot': 100.0},
    'AUDUSD': {'symbol': 'AUDUSD=X', 'name': 'AUD/USD', 'precision': 5, 'min_vol': 0.00025, 'rr': 2.0, 'pip_value': 0.0001, 'min_lot': 0.01, 'max_lot': 100.0},
    'CADCHF': {'symbol': 'CADCHF=X', 'name': 'CAD/CHF', 'precision': 5, 'min_vol': 0.0002, 'rr': 2.0, 'pip_value': 0.0001, 'min_lot': 0.01, 'max_lot': 100.0},
    'CHFJPY': {'symbol': 'CHFJPY=X', 'name': 'CHF/JPY', 'precision': 3, 'min_vol': 0.18, 'rr': 2.0, 'pip_value': 0.01, 'min_lot': 0.01, 'max_lot': 100.0},
    'AUDCAD': {'symbol': 'AUDCAD=X', 'name': 'AUD/CAD', 'precision': 5, 'min_vol': 0.0003, 'rr': 2.0, 'pip_value': 0.0001, 'min_lot': 0.01, 'max_lot': 100.0},
    'EURUSD': {'symbol': 'EURUSD=X', 'name': 'EUR/USD', 'precision': 5, 'min_vol': 0.00025, 'rr': 2.0, 'pip_value': 0.0001, 'min_lot': 0.01, 'max_lot': 100.0},
    'CADJPY': {'symbol': 'CADJPY=X', 'name': 'CAD/JPY', 'precision': 3, 'min_vol': 0.16, 'rr': 2.0, 'pip_value': 0.01, 'min_lot': 0.01, 'max_lot': 100.0},
    'USDJPY': {'symbol': 'USDJPY=X', 'name': 'USD/JPY', 'precision': 3, 'min_vol': 0.17, 'rr': 2.0, 'pip_value': 0.01, 'min_lot': 0.01, 'max_lot': 100.0}
}

_last_request_time = 0.0
_request_lock = threading.Lock()

def throttle():
    global _last_request_time
    with _request_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < THROTTLE_SEC:
            time.sleep(THROTTLE_SEC - elapsed)
        _last_request_time = time.time()

def get_market_data(symbol, days=90):
    try:
        throttle()
        yahoo_to_td = {'AUDJPY=X': 'AUD/JPY', 'EURCAD=X': 'EUR/CAD', 'AUDUSD=X': 'AUD/USD', 'CADCHF=X': 'CAD/CHF', 'CHFJPY=X': 'CHF/JPY', 'AUDCAD=X': 'AUD/CAD', 'EURUSD=X': 'EUR/USD', 'CADJPY=X': 'CAD/JPY', 'USDJPY=X': 'USD/JPY'}
        td_symbol = yahoo_to_td.get(symbol)
        if not td_symbol:
            return None
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": td_symbol, "interval": "5min", "outputsize": 2000, "apikey": TWELVE_DATA_KEY, "dp": 5, "timezone": "UTC"}
        headers = {'User-Agent': 'MultiScalper/v11'}
        r = requests.get(url, params=params, headers=headers, timeout=25)
        if r.status_code == 429:
            logger.error("Twelve Data: лимит")
            time.sleep(65)
            return None
        if r.status_code != 200:
            logger.error(f"HTTP {r.status_code}")
            return None
        data = r.json()
        if data.get("status") == "error":
            logger.error(f"Ошибка: {data.get('message')}")
            return None
        values = data.get("values")
        if not values:
            return None
        df = pd.DataFrame(values)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df[['open', 'high', 'low', 'close']]
        df.columns = ['Open', 'High', 'Low', 'Close']
        df = df.astype(float)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df['Volume'] = 1000000.0
        if len(df) < MIN_CANDLES:
            return None
        logger.info(f"→ {td_symbol}: {len(df)} свечей")
        return df
    except Exception as e:
        logger.error(f"Ошибка {symbol}: {e}")
        return None

def calculate_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    df['MA20'] = close.rolling(20).mean()
    df['MA50'] = close.rolling(50).mean()
    df['BB_Mid'] = close.rolling(26).mean()
    df['BB_Std'] = close.rolling(26).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).abs().rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    tr = pd.DataFrame({'HL': high - low, 'HC': abs(high - close.shift()), 'LC': abs(low - close.shift())}).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr_sum = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_sum)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_sum)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    df['ADX'] = dx.rolling(14).mean()
    df['MA_Diff'] = df['MA20'] - df['MA50']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Mid'] + 1e-10)
    df['Price_to_BB_Upper'] = (close - df['BB_Upper']) / close
    df['Price_to_BB_Lower'] = (close - df['BB_Lower']) / close
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
    df['Price_Change'] = close.pct_change(5) * 100
    return df

def is_volatile_enough(latest, m):
    if pd.isna(latest['ATR']) or latest['Close'] <= 0:
        return False
    return (latest['ATR'] / latest['Close']) * 100 >= m['min_vol']

def get_trend(latest):
    if pd.isna(latest['MA20']) or pd.isna(latest['MA50']):
        return "NEUTRAL"
    if latest['Close'] > latest['MA50'] and latest['MA20'] > latest['MA50']:
        return "UP"
    if latest['Close'] < latest['MA50'] and latest['MA20'] < latest['MA50']:
        return "DOWN"
    return "SIDEWAYS"

def generate_signal_rule_based(latest, m):
    if not is_volatile_enough(latest, m):
        return "HOLD"
    if pd.isna(latest['RSI']) or pd.isna(latest['ADX']):
        return "HOLD"
    if latest['ADX'] < 25:
        return "HOLD"
    
    atr_in_pips = latest['ATR'] / m['pip_value']
    if atr_in_pips < 8.0:
        return "HOLD"
    
    atr_pct = (latest['ATR'] / latest['Close']) * 100
    rsi_oversold = 30 if atr_pct > 0.6 else 35
    rsi_overbought = 70 if atr_pct > 0.6 else 65
    trend = get_trend(latest)
    
    if latest['Close'] <= latest['BB_Lower'] * 1.002 and latest['RSI'] <= rsi_oversold and trend in ["UP", "SIDEWAYS"]:
        return "BUY"
    if latest['Close'] >= latest['BB_Upper'] * 0.998 and latest['RSI'] >= rsi_overbought and trend in ["DOWN", "SIDEWAYS"]:
        return "SELL"
    return "HOLD"

class SimpleNeuralNetwork:
    def __init__(self, input_size=8, hidden_sizes=[32, 16], output_size=3, dropout_rate=0.2):
        self.dropout_rate = dropout_rate
        self.layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            W = np.random.randn(prev_size, hidden_size) * np.sqrt(2.0 / prev_size)
            b = np.zeros((1, hidden_size))
            self.layers.append({'W': W, 'b': b})
            prev_size = hidden_size
        W_out = np.random.randn(prev_size, output_size) * np.sqrt(2.0 / prev_size)
        b_out = np.zeros((1, output_size))
        self.layers.append({'W': W_out, 'b': b_out})
   
    def relu(self, x):
        return np.maximum(0, x)
   
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
   
    def forward(self, X, training=False):
        a = X
        for i in range(len(self.layers) - 1):
            z = a @ self.layers[i]['W'] + self.layers[i]['b']
            a = self.relu(z)
        z_out = a @ self.layers[-1]['W'] + self.layers[-1]['b']
        return self.softmax(z_out)
   
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1), np.max(probs, axis=1)
   
    def train(self, X, y, epochs=50, lr=0.01, batch_size=256):
        m = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_sh = X[indices]
            y_sh = y[indices]
            for i in range(0, m, batch_size):
                Xb = X_sh[i:i+batch_size]
                yb = y_sh[i:i+batch_size]
                a = Xb
                acts = [a]
                for layer in self.layers[:-1]:
                    z = a @ layer['W'] + layer['b']
                    a = self.relu(z)
                    acts.append(a)
                logits = a @ self.layers[-1]['W'] + self.layers[-1]['b']
                probs = self.softmax(logits)
                acts.append(probs)
                y_onehot = np.zeros_like(probs)
                y_onehot[np.arange(len(yb)), yb] = 1
                dz = (probs - y_onehot) / len(yb)
                for j in range(len(self.layers)-1, -1, -1):
                    dW = acts[j].T @ dz
                    db = np.sum(dz, axis=0, keepdims=True)
                    self.layers[j]['W'] -= lr * np.clip(dW, -5, 5)
                    self.layers[j]['b'] -= lr * db
                    if j > 0:
                        da = dz @ self.layers[j]['W'].T
                        dz = da * (acts[j] > 0)
            if epoch % 10 == 0 or epoch == epochs-1:
                acc = np.mean(np.argmax(self.forward(X[:1000]), axis=1) == y[:1000])
                logger.info(f"Эпоха {epoch} | Acc: {acc*100:.1f}%")

class MLPredictor:
    def __init__(self):
        self.model = None
        self.mean = None
        self.std = None
        self.feature_cols = ['RSI', 'ADX', 'ATR', 'MA_Diff', 'BB_Width', 'Price_to_BB_Upper', 'Price_to_BB_Lower', 'Price_Change']
        self.last_train_time = None
   
    def prepare_features(self, df):
        return df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
   
    def normalize(self, X, fit=False):
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1e-8
        if self.mean is None or self.std is None:
             raise ValueError("Scaler not fitted. Call normalize(X, fit=True) first.")
        return (X - self.mean) / self.std
   
    def create_labels(self, df, lookahead=12):
        labels = []
        closes = df['Close'].values
        for i in range(len(df) - lookahead):
            change = (closes[i + lookahead] - closes[i]) / closes[i] * 100
            if change > 0.1:
                labels.append(1)
            elif change < -0.1:
                labels.append(2)
            else:
                labels.append(0)
        labels.extend([0] * lookahead)
        return np.array(labels)
   
    def train(self, df):
        print("🧠 Обучение базовой нейросети...")
        X = self.prepare_features(df)
        y = self.create_labels(df, lookahead=12)
        X, y = X[:-12], y[:-12]
        if len(X) < 100:
            logger.warning("Недостаточно данных для обучения ML Predictor.")
            return False
        if len(X) > 5000:
            X, y = X[-5000:], y[-5000:]
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        X_train = self.normalize(X_train, fit=True)
        X_val = self.normalize(X_val)
        self.model = SimpleNeuralNetwork()
        self.model.train(X_train, y_train, epochs=50)
        self.last_train_time = datetime.now()
        self.save_model()
        return True
   
    def predict(self, row):
        if not self.model or self.mean is None:
            return "HOLD", 0.0
        x = np.array([row[c] for c in self.feature_cols]).reshape(1, -1)
        x = np.nan_to_num(x)
        x = self.normalize(x)
        pred, conf = self.model.predict(x)
        pred, conf = int(pred[0]), float(conf[0])
        if pred == 1 and conf >= ML_CONFIDENCE:
            return "BUY", conf
        if pred == 2 and conf >= ML_CONFIDENCE:
            return "SELL", conf
        return "HOLD", conf
   
    def save_model(self):
        try:
            with open(ML_MODEL_PATH, 'wb') as f:
                pickle.dump({'layers': self.model.layers, 'mean': self.mean, 'std': self.std, 'time': self.last_train_time}, f)
            logger.info("Базовая модель сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
   
    def load_model(self):
        try:
            if os.path.exists(ML_MODEL_PATH):
                with open(ML_MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                self.model = SimpleNeuralNetwork()
                self.model.layers = data['layers']
                self.mean, self.std = data['mean'], data['std']
                self.last_train_time = data.get('time')
                logger.info("Базовая модель загружена")
                return True
        except Exception as e:
            logger.warning(f"Ошибка загрузки: {e}")
        return False
   
    def needs_retraining(self):
        if not self.last_train_time:
            return True
        return (datetime.now() - self.last_train_time).total_seconds() / 3600 >= ML_RETRAIN_HOURS

class LSTMVolatilityPredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.last_train_time = None
   
    def create_sequences(self, df):
        features = ['Close', 'High', 'Low', 'Volume_Ratio', 'ATR']
        data = df[features].fillna(method='ffill').fillna(0).values
        X, y = [], []
        for i in range(len(data) - self.sequence_length - 5):
            X.append(data[i:i+self.sequence_length])
            future_atr = df['ATR'].iloc[i + self.sequence_length + 5]
            if pd.notna(future_atr):
                y.append(future_atr)
            else:
                continue
        if len(y) == 0:
            return np.array([]), np.array([])
        return np.array(X[:len(y)]), np.array(y)
   
    def train(self, df):
        logger.info("🔮 Обучение LSTM...")
        X, y = self.create_sequences(df)
        
        if len(X) < 100:
            logger.warning("Недостаточно данных для обучения LSTM.")
            return False
        
        if np.isnan(X).any() or np.isnan(y).any():
            logger.error("❌ NaN в исходных данных LSTM!")
            return False
        
        X_flat = X.reshape(-1, X.shape[-1])
        self.scaler_mean = np.mean(X_flat, axis=0)
        self.scaler_std = np.std(X_flat, axis=0)
        self.scaler_std = np.where(self.scaler_std < 1e-6, 1.0, self.scaler_std)
        
        X = (X - self.scaler_mean) / self.scaler_std
        
        if np.isnan(X).any():
            logger.error("❌ NaN после нормализации!")
            return False
        
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        self.model = self._build_simple_rnn(X.shape[1], X.shape[2])
        self._train_model(X_train, y_train, X_val, y_val)
        
        self.last_train_time = datetime.now()
        self.save_model()
        return True
   
    def _build_simple_rnn(self, seq_len, n_features):
        layers = []
        input_size = seq_len * n_features
        for hidden_size in [32, 16]:
            W = np.random.randn(input_size, hidden_size) * 0.001
            b = np.zeros((1, hidden_size))
            layers.append({'W': W, 'b': b})
            input_size = hidden_size
        W_out = np.random.randn(input_size, 1) * 0.001
        b_out = np.zeros((1, 1))
        layers.append({'W': W_out, 'b': b_out})
        return layers
   
    def _train_model(self, X_train, y_train, X_val, y_val, epochs=30):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        lr = 0.0001
        
        for epoch in range(epochs):
            a = X_train_flat
            activations = [a]
            
            if np.isnan(a).any():
                logger.error(f"❌ NaN в данных на эпохе {epoch}!")
                break
            
            for i, layer in enumerate(self.model[:-1]):
                z = a @ layer['W'] + layer['b']
                a = np.maximum(0, z)
                if np.isnan(a).any():
                    logger.error(f"❌ NaN после слоя {i}!")
                    break
                activations.append(a)
            
            z_out = a @ self.model[-1]['W'] + self.model[-1]['b']
            pred = z_out
            activations.append(pred)
            
            loss = np.mean((pred.flatten() - y_train) ** 2)
            
            if np.isnan(loss):
                logger.error(f"❌ NaN в Loss на эпохе {epoch}! Прерываю обучение LSTM.")
                break
            
            d_pred = 2 * (pred.flatten() - y_train).reshape(-1, 1) / len(y_train)
            
            for i in range(len(self.model) - 1, -1, -1):
                dW = activations[i].T @ d_pred
                db = np.sum(d_pred, axis=0, keepdims=True)
                self.model[i]['W'] -= lr * np.clip(dW, -0.5, 0.5)
                self.model[i]['b'] -= lr * db
                if i > 0:
                    d_pred = (d_pred @ self.model[i]['W'].T) * (activations[i] > 0)
            
            if epoch % 10 == 0:
                val_pred = self.predict_batch(X_val)
                val_loss = np.mean((val_pred - y_val) ** 2)
                logger.info(f"LSTM Epoch {epoch} | Loss: {loss:.6f} | Val: {val_loss:.6f}")
   
    def predict_batch(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        a = X_flat
        for i, layer in enumerate(self.model[:-1]):
            z = a @ layer['W'] + layer['b']
            a = np.maximum(0, z)
        z_out = a @ self.model[-1]['W'] + self.model[-1]['b']
        return z_out.flatten()
   
    def predict_future_volatility(self, df):
        if not self.model or len(df) < self.sequence_length or self.scaler_mean is None:
            return None
        features = ['Close', 'High', 'Low', 'Volume_Ratio', 'ATR']
        recent_data = df[features].tail(self.sequence_length).fillna(method='ffill').fillna(0).values
        if recent_data.shape[1] != len(self.scaler_mean):
            logger.error("Ошибка размерности данных в LSTM Predictor.")
            return None
        X = (recent_data - self.scaler_mean) / self.scaler_std
        X = X.reshape(1, -1)
        a = X
        for i, layer in enumerate(self.model[:-1]):
            z = a @ layer['W'] + layer['b']
            a = np.maximum(0, z)
        z_out = a @ self.model[-1]['W'] + self.model[-1]['b']
        predicted_atr = float(z_out[0, 0])
        current_atr = df['ATR'].iloc[-1]
        volatility_trend = "INCREASING" if predicted_atr > current_atr * 1.1 else "DECREASING" if predicted_atr < current_atr * 0.9 else "STABLE"
        return {'predicted_atr': predicted_atr, 'current_atr': current_atr, 'trend': volatility_trend, 'confidence': min(abs(predicted_atr - current_atr) / (current_atr + 1e-10) * 10, 1.0)}
   
    def save_model(self):
        try:
            with open(LSTM_MODEL_PATH, 'wb') as f:
                pickle.dump({'layers': self.model, 'mean': self.scaler_mean, 'std': self.scaler_std, 'time': self.last_train_time}, f)
            logger.info("LSTM сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения LSTM: {e}")
   
    def load_model(self):
        try:
            if os.path.exists(LSTM_MODEL_PATH):
                with open(LSTM_MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['layers']
                self.scaler_mean = data['mean']
                self.scaler_std = data['std']
                self.last_train_time = data.get('time')
                logger.info("LSTM загружена")
                return True
        except Exception as e:
            logger.warning(f"Ошибка загрузки LSTM: {e}")
        return False

class ConfidenceFilter:
    def __init__(self, input_size=8):
        self.input_size = input_size
        self.encoder = None
        self.decoder = None
        self.threshold = None
        self.last_train_time = None

    def build_model(self):
        encoder_layers = []
        sizes = [self.input_size, 4, 2]
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i+1]) * 0.1
            b = np.zeros((1, sizes[i+1]))
            encoder_layers.append({'W': W, 'b': b})
        decoder_layers = []
        sizes = [2, 4, self.input_size]
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i+1]) * 0.1
            b = np.zeros((1, sizes[i+1]))
            decoder_layers.append({'W': W, 'b': b})
        self.encoder = encoder_layers
        self.decoder = decoder_layers

    def train(self, X_normal, epochs=50):
        logger.info("🔍 Обучение Confidence Filter...")
        if self.encoder is None:
            self.build_model()
        lr = 0.01
        for epoch in range(epochs):
            a = X_normal
            enc_acts = [a]
            for layer in self.encoder:
                z = a @ layer['W'] + layer['b']
                a = np.maximum(0, z)
                enc_acts.append(a)
            latent = a
            a = latent
            dec_acts = [a]
            for layer in self.decoder:
                z = a @ layer['W'] + layer['b']
                a = np.maximum(0, z)
                dec_acts.append(a)
            reconstructed = a
            loss = np.mean((reconstructed - X_normal) ** 2)
            d_out = 2 * (reconstructed - X_normal) / len(X_normal)
            for i in range(len(self.decoder) - 1, -1, -1):
                dW = dec_acts[i].T @ d_out
                db = np.sum(d_out, axis=0, keepdims=True)
                self.decoder[i]['W'] -= lr * np.clip(dW, -1, 1)
                self.decoder[i]['b'] -= lr * db
                if i > 0:
                    d_out = (d_out @ self.decoder[i]['W'].T) * (dec_acts[i] > 0)
            d_latent = d_out
            for i in range(len(self.encoder) - 1, -1, -1):
                dW = enc_acts[i].T @ d_latent
                db = np.sum(d_latent, axis=0, keepdims=True)
                self.encoder[i]['W'] -= lr * np.clip(dW, -1, 1)
                self.encoder[i]['b'] -= lr * db
                if i > 0:
                    d_latent = (d_latent @ self.encoder[i]['W'].T) * (enc_acts[i] > 0)
       
            if epoch % 10 == 0:
                logger.info(f"AE Epoch {epoch} | Loss: {loss:.6f}")

        errors = self._compute_errors(X_normal)
        self.threshold = np.percentile(errors, 95) * 1.5
        self.last_train_time = datetime.now()
        self.save_model()
        logger.info(f"Confidence Filter обучен. Порог: {self.threshold:.6f}")

    def _compute_errors(self, X):
        a = X
        for layer in self.encoder:
            z = a @ layer['W'] + layer['b']
            a = np.maximum(0, z)
        for layer in self.decoder:
            z = a @ layer['W'] + layer['b']
            a = np.maximum(0, z)
        errors = np.mean((a - X) ** 2, axis=1)
        return errors

    def is_market_normal(self, x):
        if self.encoder is None or self.threshold is None:
            return {'is_normal': True, 'confidence': 1.0, 'reconstruction_error': 0.0}
        error = self._compute_errors(x.reshape(1, -1))[0]
        is_normal = error < self.threshold
        confidence = 1.0 - min(error / (self.threshold + 1e-10), 1.0)
        return {'is_normal': is_normal, 'confidence': confidence, 'reconstruction_error': error}

    def save_model(self):
        try:
            with open(CONFIDENCE_MODEL_PATH, 'wb') as f:
                pickle.dump({'encoder': self.encoder, 'decoder': self.decoder, 'threshold': self.threshold, 'time': self.last_train_time}, f)
            logger.info("Confidence Filter сохранен")
        except Exception as e:
            logger.error(f"Ошибка сохранения Confidence: {e}")

    def load_model(self):
        try:
            if os.path.exists(CONFIDENCE_MODEL_PATH):
                with open(CONFIDENCE_MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                self.encoder = data['encoder']
                self.decoder = data['decoder']
                self.threshold = data['threshold']
                self.last_train_time = data.get('time')
                logger.info("Confidence Filter загружен")
                return True
        except Exception as e:
            logger.warning(f"Ошибка загрузки Confidence: {e}")
        return False

class EnsembleManager:
    def __init__(self):
        self.base_ml = MLPredictor()
        self.volatility_lstm = LSTMVolatilityPredictor()
        self.confidence_filter = ConfidenceFilter()
        self.last_train_time = None

    def train_all(self, df):
        logger.info("🚀 Обучение ансамбля...")
        self.base_ml.train(df)
        if len(df) > 200:
            self.volatility_lstm.train(df)
        stable_periods = df[df['ADX'].fillna(0) < 25]
        if len(stable_periods) > 500:
            X_stable = self.base_ml.prepare_features(stable_periods)
            if self.base_ml.mean is not None:
                X_stable = self.base_ml.normalize(X_stable)
                self.confidence_filter.train(X_stable)
            else:
                logger.warning("Базовая ML не обучена, пропуск Confidence Filter.")
        else:
            logger.warning("Мало стабильных периодов для Confidence Filter")

    def predict_ensemble(self, df, latest):
        base_signal, base_conf = self.base_ml.predict(latest)
        if base_signal == "HOLD":
            return base_signal, base_conf
   
        final_conf = base_conf

        if self.volatility_lstm.model:
            vol_pred = self.volatility_lstm.predict_future_volatility(df)
            if vol_pred and vol_pred['trend'] == 'DECREASING':
                logger.warning(f"⚠️ Волатильность падает - снижаем уверенность (V-trend: {vol_pred['trend']})")
                final_conf *= 0.7
   
        if self.confidence_filter.encoder and self.base_ml.mean is not None:
            x_latest = np.array([latest[c] for c in self.base_ml.feature_cols]).reshape(1, -1)
            x_latest = np.nan_to_num(x_latest)
            x_latest = self.base_ml.normalize(x_latest)
            market_check = self.confidence_filter.is_market_normal(x_latest)
            if not market_check['is_normal']:
                logger.warning(f"⚠️ Аномальные условия (Error: {market_check['reconstruction_error']:.6f}, Conf: {market_check['confidence']:.2f})")
                final_conf *= market_check['confidence']
   
        final_signal = base_signal if final_conf >= ML_CONFIDENCE else "HOLD"
   
        if final_signal != "HOLD":
            logger.info(f"📊 Ансамбль: {final_signal} (Базовая: {base_conf:.2f} → Финальная: {final_conf:.2f})")
       
        return final_signal, final_conf

    def load_all_models(self):
        self.base_ml.load_model()
        self.volatility_lstm.load_model()
        self.confidence_filter.load_model()

    def needs_retraining(self):
        return self.base_ml.needs_retraining()

def generate_signal_hybrid(latest, m, ensemble, df):
    if ML_ENABLED:
        ml_signal, ml_conf = ensemble.predict_ensemble(df, latest)
    else:
        ml_signal, ml_conf = "HOLD", 0.0

    if ml_signal != "HOLD":
        return ml_signal, ml_conf

    rule_signal = generate_signal_rule_based(latest, m)
    if rule_signal != "HOLD":
        return rule_signal, 0.5

    return "HOLD", 0.0

class RiskManager:
    def __init__(self, max_open_trades=3, risk_percent=0.015, initial_balance=10000):
        self.max_open_trades = max_open_trades
        self.risk_percent = risk_percent
        self.balance = initial_balance
        self.open_trades = []
        self.trade_counter = 0

    def can_open_trade(self):
        return len(self.open_trades) < self.max_open_trades

    def add_trade(self, market, signal, price, current_atr):
        self.trade_counter += 1
        
        stop_loss_distance = current_atr
        take_profit_distance = current_atr * market['rr']
        
        stop_loss_pips = stop_loss_distance / market['pip_value']
        take_profit_pips = take_profit_distance / market['pip_value']
        
        risk_amount = self.balance * self.risk_percent
        
        try:
            contract_size = 100000
            pip_cost_per_lot = market['pip_value'] * contract_size
            lot_size = risk_amount / (stop_loss_pips * pip_cost_per_lot)
            lot_size = max(market['min_lot'], min(lot_size, market['max_lot']))
        except (ZeroDivisionError, Exception) as e:
            logger.error(f"Ошибка расчета лотов: {e}")
            lot_size = market['min_lot']
        
        trade = {
            'id': self.trade_counter,
            'market': market['name'],
            'signal': signal,
            'price': price,
            'time': datetime.now(),
            'sl_distance': stop_loss_distance,
            'tp_distance': take_profit_distance,
            'sl_pips': stop_loss_pips,
            'tp_pips': take_profit_pips,
            'lot_size': lot_size,
            'sl_price': price - stop_loss_distance if signal == 'BUY' else price + stop_loss_distance,
            'tp_price': price + take_profit_distance if signal == 'BUY' else price - take_profit_distance,
        }
        
        self.open_trades.append(trade)
        logger.info(f"💰 Расчет: Риск=${risk_amount:.2f}, SL={stop_loss_pips:.1f}pips, Лоты={lot_size:.2f}")
        return trade

    def check_and_close_trades(self, market_symbol, current_price):
        pass

async def handle_callback_query(update, context):
    query = update.callback_query
    await query.answer()
    try:
        action, trade_id_str = query.data.split('|')
        trade_id = int(trade_id_str)
    except ValueError:
        return

    trade_to_close = next((t for t in rm.open_trades if t['id'] == trade_id), None)

    if action == 'close_trade' and trade_to_close:
        rm.open_trades.remove(trade_to_close)
        await context.bot.edit_message_text(
            chat_id=query.message.chat_id,
            message_id=query.message.message_id,
            text=f"✅ Сделка #{trade_id} ({trade_to_close['market']} {trade_to_close['signal']}) закрыта вручную."
        )

async def send_signal(app, market, latest, signal_data, rm_manager):
    signal, conf = signal_data
    if not rm_manager.can_open_trade():
        logger.warning(f"❌ {market['name']}: Сигнал ({signal}), но превышен лимит открытых сделок ({rm_manager.max_open_trades}).")
        return

    current_atr = latest['ATR']
    price = latest['Close']

    trade = rm_manager.add_trade(market, signal, price, current_atr)

    color_emoji = "🟢" if signal == "BUY" else "🔴"

    message = (
        f"{color_emoji} **НОВЫЙ СИГНАЛ: {market['name']}**\n"
        f"**Тип:** {signal}\n"
        f"**Цена входа:** {price:.{market['precision']}f}\n"
        f"**Уверенность (Ensemble):** {conf*100:.1f}%\n"
        f"**Объем:** {trade['lot_size']:.2f} лота\n"
        f"**SL (Pips):** {trade['sl_pips']:.1f}\n"
        f"**TP (Pips):** {trade['tp_pips']:.1f}\n"
        f"**Баланс/Риск:** ${rm_manager.balance:.2f} / {rm_manager.risk_percent*100:.1f}%"
    )

    keyboard = [[InlineKeyboardButton("Закрыть сделку", callback_data=f"close_trade|{trade['id']}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await app.bot.send_message(
        TELEGRAM_CHAT_ID,
        message,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    logger.info(f"→ Отправлен сигнал: {market['name']} {signal} (Conf: {conf:.2f})")

async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    rm_manager = RiskManager(max_open_trades=MAX_OPEN_TRADES, risk_percent=RISK_PERCENT, initial_balance=INITIAL_BALANCE)
    ensemble = EnsembleManager()
    store = {}

    async with app:
        await app.start()
        await app.bot.send_message(TELEGRAM_CHAT_ID, "🚀 Скальпер запущен.")
       
        logger.info("Первичная загрузка данных...")
        for m in MARKETS.values():
            df = get_market_data(m['symbol'])
            if df is not None and len(df) >= MIN_CANDLES:
                df = calculate_indicators(df)
                store[m['symbol']] = df
                logger.info(f" {m['name']}: {len(df)} свечей")

        ensemble.load_all_models()
       
        if ensemble.needs_retraining():
            combined = pd.concat([df for df in store.values() if len(df) > 200], ignore_index=True)
            if len(combined) > 1000:
                ensemble.train_all(combined)

        scan = 0
        try:
            while True:
                scan += 1
                logger.info(f"\n{'='*60}\n СКАН #{scan} | {datetime.now().strftime('%H:%M:%S')}\n{'='*60}")
               
                for m in MARKETS.values():
                    df = get_market_data(m['symbol'])
                   
                    if df is None or len(df) < MIN_CANDLES:
                        continue
                       
                    df = calculate_indicators(df)
                    store[m['symbol']] = df
                    latest = df.iloc[-1]

                    rm_manager.check_and_close_trades(m['name'], latest['Close'])
                   
                    signal_data = generate_signal_hybrid(latest, m, ensemble, df)
                    signal = signal_data[0]
                   
                    if ("BUY" in signal or "SELL" in signal) and rm_manager.can_open_trade():
                        await send_signal(app, m, latest, signal_data, rm_manager)
               
                if ensemble.needs_retraining():
                    combined = pd.concat([df for df in store.values() if len(df) > 200], ignore_index=True)
                    if len(combined) > 1000:
                        ensemble.train_all(combined)

                logger.info(f"Следующий скан через {CONFIG['scan_interval_sec']} сек...")
                await asyncio.sleep(CONFIG['scan_interval_sec'])

        except KeyboardInterrupt:
            logger.info("Получен KeyboardInterrupt. Остановка.")
            await app.bot.send_message(TELEGRAM_CHAT_ID, "Скальпер остановлен.")
        finally:
            await app.stop()
            await app.shutdown()
            logger.info("Работа скальпера завершена.")

if __name__ == "__main__":
    asyncio.run(main())
