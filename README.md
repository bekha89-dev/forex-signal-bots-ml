# 🤖 Forex Signal Bots — ML + Telegram + Session Filter

3 production-ready Python bots for automated Forex signal detection.

## 📦 What's inside
- **Bot 1** — 9 pairs, ensemble ML (RandomForest + Neural Net), auto lot-size
- **Bot 2** — Multi-timeframe consensus M5+H1+H4
- **Bot 3** — Session filter, ML confidence 0.75, conservative

## ✅ Features
- Telegram alerts with BUY/SELL/CLOSE buttons
- ML signal filtering (retrained every 24h)
- Session filter — London/US only (07:00–16:00 UTC)
- Cooldown + daily signal cap per pair
- Risk manager — auto lot size based on ATR
- TwelveData free API tier compatible

## 🚀 Quick start
pip install -r requirements.txt
cp .env.example .env
python bot3_conf75_session_scalper.py

## 📥 Full pack with config templates
Download on Gumroad → https://bekha.gumroad.com/l/joqouu

## ⚙️ Requirements
- Python 3.10+
- TwelveData API key (free tier)  
- Telegram Bot token
