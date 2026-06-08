import telebot
import yfinance as yf
import pandas as pd
import threading
import time
import os

TOKEN = os.getenv("BOT_TOKEN") # Токен будем брать из настроек хостинга
bot = telebot.TeleBot(TOKEN)
active_chats = set()

def get_pivot_points(ticker):
    try:
        data = yf.download(tickers=ticker, period="2d", interval="1d", progress=False)
        if len(data) < 2: return None
        prev = data.iloc[-2]
        pivot = (prev['High'] + prev['Low'] + prev['Close']) / 3
        return {"s1": (pivot * 2) - prev['High'], "r1": (pivot * 2) - prev['Low']}
    except: return None

def get_signal(ticker):
    try:
        data = yf.download(tickers=ticker, period="5d", interval="1m", progress=False)
        if data.empty or len(data) < 200: return None
        curr_price = data['Close'].iloc[-1]
        sma200 = data['Close'].rolling(window=200).mean().iloc[-1]
        pivots = get_pivot_points(ticker)
        if not pivots: return None
        if abs(curr_price - pivots['s1']) < (curr_price * 0.0005) and curr_price > sma200:
            return f"🟢 *СИГНАЛ CALL*\nПара: {ticker.replace('=X', '')}\nЦена: {curr_price:.5f}"
        if abs(curr_price - pivots['r1']) < (curr_price * 0.0005) and curr_price < sma200:
            return f"🔴 *СИГНАЛ PUT*\nПара: {ticker.replace('=X', '')}\nЦена: {curr_price:.5f}"
    except: return None
    return None

@bot.message_handler(commands=['start'])
def start(message):
    active_chats.add(message.chat.id)
    bot.send_message(message.chat.id, "🎯 *Терминал 2.2 онлайн!*")

def scanner():
    pairs = ["EURUSD=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
    while True:
        for p in pairs:
            sig = get_signal(p)
            if sig:
                for chat_id in list(active_chats):
                    try: bot.send_message(chat_id, sig, parse_mode="Markdown")
                    except: pass
            time.sleep(10)
        time.sleep(60)

threading.Thread(target=scanner, daemon=True).start()
bot.infinity_polling(drop_pending_updates=True)
