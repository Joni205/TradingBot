import telebot
from telebot import types
import yfinance as yf
import pandas as pd
import os

TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

# Функция расчёта уровней (из 2.2)
def get_signal(ticker):
    try:
        data = yf.download(tickers=ticker, period="5d", interval="1m", progress=False)
        if data.empty or len(data) < 200: return "Ошибка: данных мало."
        
        curr_price = data['Close'].iloc[-1]
        sma200 = data['Close'].rolling(window=200).mean().iloc[-1]
        
        # Расчет уровней Pivot
        prev = data.iloc[-2]
        pivot = (prev['High'] + prev['Low'] + prev['Close']) / 3
        s1 = (pivot * 2) - prev['High']
        r1 = (pivot * 2) - prev['Low']
        
        # Формирование сигнала
        if abs(curr_price - s1) < (curr_price * 0.0005) and curr_price > sma200:
            return f"🟢 *CALL* | {ticker.replace('=X', '')}\nЦена: {curr_price:.5f}\nУровень S1: {s1:.5f}"
        if abs(curr_price - r1) < (curr_price * 0.0005) and curr_price < sma200:
            return f"🔴 *PUT* | {ticker.replace('=X', '')}\nЦена: {curr_price:.5f}\nУровень R1: {r1:.5f}"
        
        return f"💤 *Нет сигнала*\n{ticker.replace('=X', '')}: {curr_price:.5f}"
    except Exception as e:
        return f"Ошибка запроса: {str(e)}"

# Главное меню с кнопками
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    pairs = ["EURUSD=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
    buttons = [types.InlineKeyboardButton(p.replace('=X', ''), callback_data=p) for p in pairs]
    markup.add(*buttons)
    bot.send_message(message.chat.id, "🎯 *Терминал 2.3*\nВыбери пару для анализа:", reply_markup=markup, parse_mode="Markdown")

# Обработка нажатий на кнопки
@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    bot.answer_callback_query(call.id, "Анализирую...")
    result = get_signal(call.data)
    # Исправляем здесь: обращаемся через call.message
    bot.send_message(call.message.chat.id, result, parse_mode="Markdown")
# Запуск
try:
    bot.delete_webhook(drop_pending_updates=True)
except:
    pass
bot.infinity_polling()
