import telebot
from telebot import types
import yfinance as yf
import ccxt
import json
import os
import pandas as pd
import threading
import time
import sqlite3
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. БАЗОВЫЕ НАСТРОЙКИ И ПЕРЕМЕННЫЕ
# ==========================================
BOT_TOKEN = '7637396903:AAF3Y-VSbweV1FlCys91ePlxK9ouhLnspLc'  # <-- Вставь сюда свой токен!
bot = telebot.TeleBot(BOT_TOKEN)
binance = ccxt.binance()

PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "BTC/USDT": "BTC/USDT", "ETH/USDT": "ETH/USDT", "SOL/USDT": "SOL/USDT"
}

active_scans = {}  # Текущие дашборды {chat_id: {"pair": ..., "msg_id": ..., "last_signal": ...}}
user_settings = {}  # Настройки фильтров {chat_id: [список активных фильтров]}
AI_MODEL = None  # Модель ИИ

ALL_FILTERS = ["Pivot (Уровни)", "RSI (Перепроданность)", "MACD (Импульс)", "Volume (Объемы)"]


# ==========================================
# 2. БАЗА ДАННЫХ И НЕЙРОСЕТЬ (ИИ)
# ==========================================
def init_db():
    with sqlite3.connect('trades.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT, sig_type TEXT, entry_price REAL,
                rsi REAL, macd REAL, vol_ratio REAL,
                expiration_min INTEGER, check_time INTEGER, result INTEGER
            )
        ''')
        conn.commit()


def train_ai():
    global AI_MODEL
    try:
        # 1. Загружаем реальные сделки из базы данных
        with sqlite3.connect('trades.db') as conn:
            df_db = pd.read_sql_query(
                "SELECT sig_type, rsi, macd, vol_ratio, result FROM history_trades WHERE result IS NOT NULL", conn)

        # 2. Загружаем нашу "Базу знаний" (учебник для ИИ)
        df_kb = pd.DataFrame()
        if os.path.exists('knowledge_base.json'):
            with open('knowledge_base.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                df_kb = pd.DataFrame(data)

        # 3. Объединяем реальный опыт бота и "учебник"
        if not df_kb.empty and not df_db.empty:
            df = pd.concat([df_kb, df_db], ignore_index=True)
        elif not df_kb.empty:
            df = df_kb
        else:
            df = df_db

        # 4. Проверяем, хватает ли данных для обучения
        if len(df) < 20 or len(np.unique(df['result'])) < 2:
            print("⏳ [ИИ] Недостаточно данных для обучения.")
            return False

        # 5. Обучение
        df['sig_type_enc'] = df['sig_type'].apply(lambda x: 1 if x == 'CALL' else 0)
        X = df[['sig_type_enc', 'rsi', 'macd', 'vol_ratio']]
        y = df['result']

        # Уменьшаем "жесткость" леса (min_samples_leaf=2) чтобы он был более гибким
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42)
        model.fit(X, y)
        AI_MODEL = model
        print(f"🤖 [ИИ] Нейросеть АКТИВНА! Обучена на {len(df)} паттернах (База Знаний + Реальные сделки).")
        return True
    except Exception as e:
        print(f"Ошибка обучения ИИ: {e}")
        return False


# ==========================================
# 3. МАТЕМАТИЧЕСКИЙ АНАЛИЗ РЫНКА
# ==========================================
def get_signal(ticker, source, active_filters):
    try:
        # 1. Получение Дневных данных для Pivot
        if source == "binance":
            daily_ohlcv = binance.fetch_ohlcv(ticker, timeframe='1d', limit=3)
            if len(daily_ohlcv) < 2: return None
            prev_high, prev_low, prev_close = daily_ohlcv[-2][2], daily_ohlcv[-2][3], daily_ohlcv[-2][4]
        else:
            df_daily = yf.download(ticker, period="3d", interval="1d", progress=False)
            if df_daily.empty or len(df_daily) < 2: return None
            if isinstance(df_daily.columns, pd.MultiIndex): df_daily.columns = df_daily.columns.get_level_values(0)
            df_daily.columns = [str(c).lower() for c in df_daily.columns]
            prev_high, prev_low, prev_close = df_daily['high'].iloc[-2], df_daily['low'].iloc[-2], \
            df_daily['close'].iloc[-2]

        pivot = (prev_high + prev_low + prev_close) / 3
        s1, r1 = (pivot * 2) - prev_high, (pivot * 2) - prev_low

        # 2. Получение Минутных данных
        if source == "binance":
            m1_ohlcv = binance.fetch_ohlcv(ticker, timeframe='1m', limit=300)
            df = pd.DataFrame(m1_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        else:
            df = yf.download(ticker, period="2d", interval="1m", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]

        if df.empty or len(df) < 150: return None
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # 3. Расчет технических индикаторов
        df['sma200'] = df['close'].rolling(window=200).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))

        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['vol_sma'] = df['volume'].rolling(window=20).mean()

        # Текущие метрики последней свечи
        cp = float(df['close'].iloc[-1])
        sma = float(df['sma200'].iloc[-1])
        rsi = float(df['rsi'].iloc[-1])
        macd = float(df['macd'].iloc[-1])
        macd_sig = float(df['macd_signal'].iloc[-1])
        vol_ratio = float(df['volume'].iloc[-1]) / float(df['vol_sma'].iloc[-1]) if float(
            df['vol_sma'].iloc[-1]) > 0 else 1.0

        # 4. Проверка условий Конструктора Стратегий
        threshold = 0.0005
        is_uptrend = cp > sma
        is_downtrend = cp < sma

        pivot_call_ok = (abs(cp - s1) / cp <= threshold) if "Pivot (Уровни)" in active_filters else True
        pivot_put_ok = (abs(cp - r1) / cp <= threshold) if "Pivot (Уровни)" in active_filters else True

        rsi_call_ok = (rsi <= 35) if "RSI (Перепроданность)" in active_filters else True
        rsi_put_ok = (rsi >= 65) if "RSI (Перепроданность)" in active_filters else True

        macd_call_ok = (macd > macd_sig) if "MACD (Импульс)" in active_filters else True
        macd_put_ok = (macd < macd_sig) if "MACD (Импульс)" in active_filters else True

        vol_ok = (vol_ratio > 1.05) if "Volume (Объемы)" in active_filters else True

        # Вычисление времени экспирации
        diff_percent = abs(cp - s1) / cp
        exp_min = 3 if diff_percent <= 0.0001 else (5 if diff_percent <= 0.00025 else 10)

        # Проверка триггеров на вход
        if is_uptrend and pivot_call_ok and rsi_call_ok and macd_call_ok and vol_ok:
            return "CALL", exp_min, s1, cp, rsi, macd, vol_ratio
        elif is_downtrend and pivot_put_ok and rsi_put_ok and macd_put_ok and vol_ok:
            return "PUT", exp_min, r1, cp, rsi, macd, vol_ratio

        # Если сигнала нет — возвращаем маркер "НЕТ СИГНАЛА" и текущие метрики для Дашборда
        return "NO_SIGNAL", None, None, cp, rsi, macd, vol_ratio
    except Exception as e:
        print(f"Ошибка ТА: {e}")
        return None


# ==========================================
# 4. ФОНОВЫЕ ПОТОКИ (СКАНЕР И ПРОВЕРКА)
# ==========================================
def check_outcomes_loop():
    while True:
        try:
            with sqlite3.connect('trades.db') as conn:
                cursor = conn.cursor()
                now = int(time.time())
                cursor.execute(
                    "SELECT id, pair, sig_type, entry_price FROM history_trades WHERE result IS NULL AND check_time <= ?",
                    (now,))
                pending = cursor.fetchall()

                for trade in pending:
                    db_id, pair, sig_type, entry_price = trade
                    ticker, source = PAIRS[pair], "binance" if "USDT" in pair else "yahoo"

                    curr_price = None
                    if source == "binance":
                        curr_price = float(binance.fetch_ticker(ticker)['close'])
                    else:
                        df = yf.download(ticker, period="1d", interval="1m", progress=False)
                        if not df.empty:
                            curr_price = float(df['close'].iloc[-1]) if 'close' in df.columns else float(
                                df['Close'].iloc[-1])

                    if curr_price is not None:
                        result = 1 if (sig_type == "CALL" and curr_price > entry_price) or (
                                    sig_type == "PUT" and curr_price < entry_price) else 0
                        cursor.execute("UPDATE history_trades SET result = ? WHERE id = ?", (result, db_id))
                        conn.commit()
            train_ai()
        except Exception as e:
            print(f"Ошибка чекера результатов: {e}")
        time.sleep(15)


def background_scanner(chat_id, pair_name):
    ticker = PAIRS[pair_name]
    source = "binance" if "USDT" in pair_name else "yahoo"

    while True:
        # Проверка флага остановки
        if chat_id not in active_scans or active_scans[chat_id].get("pair") != pair_name:
            break

        filters = user_settings.get(chat_id, ALL_FILTERS)
        analysis = get_signal(ticker, source, filters)
        now_str = datetime.now().strftime("%H:%M:%S")

        if analysis and analysis[0] in ["CALL", "PUT"]:
            sig_type, exp_min, level, price, rsi, macd, vol_ratio = analysis

            # Защита от пустых уровней перед форматированием string f""
            if level is None or price is None:
                time.sleep(5)
                continue

            # Фильтрация через ИИ
            ai_allowed, ai_conf = True, 100.0
            if AI_MODEL is not None:
                proba = AI_MODEL.predict_proba([[1 if sig_type == "CALL" else 0, rsi, macd, vol_ratio]])[0][1]
                ai_conf = proba * 100
                if proba < 0.60: ai_allowed = False

            if not ai_allowed:
                status_text = f"🔄 *Сканирую {pair_name}...* [`{now_str}`]\n\n⚠️ *Фильтр ИИ:* Обнаружен сигнал {sig_type}, но Нейросеть заблокировала его (Уверенность всего {ai_conf:.1f}%). Ждем идеальных условий."
            else:
                status_text = f"🔄 *Сканирую {pair_name}...*\n⏱ Последний сигнал отправлен в `{now_str}`!"
                sig_key = f"{sig_type}_{level:.5f}"

                if active_scans[chat_id].get("last_signal") != sig_key:
                    active_scans[chat_id]["last_signal"] = sig_key

                    try:
                        with sqlite3.connect('trades.db') as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "INSERT INTO history_trades (pair, sig_type, entry_price, rsi, macd, vol_ratio, expiration_min, check_time, result) VALUES (?,?,?,?,?,?,?,?, NULL)",
                                (pair_name, sig_type, price, rsi, macd, vol_ratio, exp_min,
                                 int(time.time()) + (exp_min * 60)))
                            conn.commit()
                    except Exception as e:
                        print(f"Ошибка записи сделки в БД: {e}")

                    emoji = "🟢" if sig_type == "CALL" else "🔴"
                    ai_status_str = f"{ai_conf:.1f}%" if AI_MODEL is not None else "Сбор статистики..."

                    msg = f"{emoji} *СИГНАЛ: {sig_type}*\n📋 *Актив:* {pair_name}\n🎯 *ВХОД:* `{level:.5f}`\n⏱ *ЭКСПИРАЦИЯ:* {exp_min} мин\n\n📊 *Метрики:*\n• RSI: {rsi:.1f}\n🤖 *Доверие ИИ:* `{ai_status_str}`"
                    bot.send_message(chat_id, msg, parse_mode="Markdown")
        else:
            # Режим ожидания (NO_SIGNAL или ошибка данных)
            price = analysis[3] if (analysis and analysis[3] is not None) else 0.0
            rsi = analysis[4] if (analysis and analysis[4] is not None) else 0.0
            status_text = f"🔄 *Мониторинг {pair_name}*\n⏱ Текущее время: `{now_str}`\n\n💰 Цена: `{price:.5f}` | RSI: `{rsi:.1f}`\n\n_Индикаторы нейтральны. Ищу точку входа..._"

        # Обновление Дашборда на лету
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=active_scans[chat_id]["msg_id"], text=status_text,
                                  parse_mode="Markdown")
        except:
            pass

        time.sleep(15)


# ==========================================
# 5. ИНТЕРФЕЙС И ОБРАБОТКА КОМАНД
# ==========================================
def main_menu():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add('▶️ Старт сканирования', '⚙️ Настройки условий', '📊 Статистика ИИ', '⏹ Стоп')
    return markup


@bot.message_handler(commands=['start', 'menu'])
def start_cmd(message):
    if message.chat.id not in user_settings:
        user_settings[message.chat.id] = ALL_FILTERS.copy()
    bot.send_message(message.chat.id,
                     "🤖 *Мульти-терминал 7.1 запущен!*\n\nИспользуй клавиатуру ниже для гибкой настройки конструктора стратегий и запуска фонового сканирования.",
                     parse_mode="Markdown", reply_markup=main_menu())


@bot.message_handler(func=lambda m: m.text == '⚙️ Настройки условий')
def settings_menu(message):
    chat_id = message.chat.id
    if chat_id not in user_settings:
        user_settings[chat_id] = ALL_FILTERS.copy()

    markup = types.InlineKeyboardMarkup()
    for f in ALL_FILTERS:
        state = "✅" if f in user_settings[chat_id] else "❌"
        markup.add(types.InlineKeyboardButton(f"{state} {f}", callback_data=f"toggle_{f}"))
    bot.send_message(chat_id,
                     "🛠 *Конструктор Стратегии*\n\nНажимай на кнопки, чтобы включать или отключать фильтры. Изменения применятся мгновенно при следующем цикле сканирования:",
                     parse_mode="Markdown", reply_markup=markup)


@bot.callback_query_handler(func=lambda call: call.data.startswith('toggle_'))
def toggle_filter(call):
    f_name = call.data.replace('toggle_', '')
    chat_id = call.message.chat.id

    if chat_id not in user_settings:
        user_settings[chat_id] = ALL_FILTERS.copy()

    if f_name in user_settings[chat_id]:
        user_settings[chat_id].remove(f_name)
    else:
        user_settings[chat_id].append(f_name)

    markup = types.InlineKeyboardMarkup()
    for f in ALL_FILTERS:
        state = "✅" if f in user_settings[chat_id] else "❌"
        markup.add(types.InlineKeyboardButton(f"{state} {f}", callback_data=f"toggle_{f}"))
    try:
        bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=markup)
    except:
        pass
    bot.answer_callback_query(call.id, f"Фильтр изменен")


@bot.message_handler(func=lambda m: m.text == '▶️ Старт сканирования')
def start_scan_menu(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(*[types.InlineKeyboardButton(text=n, callback_data=f"scan_{n}") for n in PAIRS.keys()])
    bot.send_message(message.chat.id, "👇 Выбери валютную пару для непрерывного поиска точек входа:",
                     reply_markup=markup)


@bot.callback_query_handler(func=lambda call: call.data.startswith('scan_'))
def launch_scan(call):
    pair_name = call.data.replace('scan_', '')
    chat_id = call.message.chat.id

    # Если запущен старый дашборд — удаляем его из чата, чтобы навести порядок
    if chat_id in active_scans:
        try:
            bot.delete_message(chat_id, active_scans[chat_id]["msg_id"])
        except:
            pass

    msg = bot.send_message(chat_id, f"⏳ Инициализация фонового потока и загрузка данных по *{pair_name}*...",
                           parse_mode="Markdown")
    active_scans[chat_id] = {"pair": pair_name, "msg_id": msg.message_id, "last_signal": ""}

    # Запуск изолированного потока для сканирования выбранной пары
    threading.Thread(target=background_scanner, args=(chat_id, pair_name), daemon=True).start()
    bot.answer_callback_query(call.id, f"Запущено сканирование {pair_name}")


@bot.message_handler(func=lambda m: m.text == '⏹ Стоп')
def stop_scan(message):
    chat_id = message.chat.id
    if chat_id in active_scans:
        try:
            bot.delete_message(chat_id, active_scans[chat_id]["msg_id"])
        except:
            pass
        del active_scans[chat_id]
        bot.send_message(chat_id, "⏹ Поток сканирования успешно завершен. Бот переведен в режим сна.")
    else:
        bot.send_message(chat_id, "У вас нет запущенных процессов сканирования.")


@bot.message_handler(func=lambda m: m.text == '📊 Статистика ИИ')
def show_stats(message):
    try:
        with sqlite3.connect('trades.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(result) FROM history_trades WHERE result IS NOT NULL")
            total, wins = cursor.fetchone()
    except:
        total, wins = 0, 0

    total = total or 0
    wins = wins or 0
    winrate = (wins / total * 100) if total > 0 else 0.0
    ai_status = "✅ Модель обучена и фильтрует сделки" if AI_MODEL is not None else f"⏳ Накопление датасета ({total}/20 сделок)"

    msg = f"🤖 *Статус Модуля Машинного Обучения:*\n\n" \
          f"• Состояние ИИ: `{ai_status}`\n" \
          f"• Всего сделок в базе: `{total}`\n" \
          f"• Закрыто в ПЛЮС: `{wins}`\n\n" \
          f"🏆 *Текущий Winrate бота:* `{winrate:.1f}%`"
    bot.send_message(message.chat.id, msg, parse_mode="Markdown")


# ==========================================
# 6. ЗАПУСК ПРИЛОЖЕНИЯ
# ==========================================
if __name__ == '__main__':
    init_db()  # Создаем БД при старте
    train_ai()  # Загружаем ИИ, если база уже была наполнена ранее

    # Запускаем независимый фоновый поток проверки экспираций
    threading.Thread(target=check_outcomes_loop, daemon=True).start()

    print(">>> Бот-Терминал успешно запущен и готов к работе!")
    bot.infinity_polling()