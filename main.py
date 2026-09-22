"""
Bot Telegram per segnali di trading su XAUUSD.

Strategia:
  - MACD: EMA veloce 12, EMA lenta 26, signal = SMA a 9 periodi
  - Trend filter: EMA20 vs EMA50
  - LONG  -> MACD incrocia al rialzo la signal line  E  EMA20 > EMA50
  - SHORT -> MACD incrocia al ribasso la signal line  E  EMA20 < EMA50
  - TP/SL calcolati automaticamente in base all'ATR(14)

Pensato per girare periodicamente (es. ogni 15 minuti) via GitHub Actions,
senza bisogno di un server sempre acceso.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ----------------------------------------------------------------------
# CONFIGURAZIONE (modifica pure questi valori)
# ----------------------------------------------------------------------

SYMBOL = "XAU/USD"
TIMEFRAME = os.environ.get("TIMEFRAME", "15min")   # es: 5min, 15min, 1h
OUTPUT_SIZE = 200                                  # candele storiche da scaricare

EMA_FAST = 12
EMA_SLOW = 26
MACD_SIGNAL_PERIOD = 9   # SMA, come richiesto (non EMA)

TREND_EMA_FAST = 20
TREND_EMA_SLOW = 50

ATR_PERIOD = 14
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0        # risk/reward 1:2

# Finestra di anticipo per gli alert sulle notizie (minuti prima dell'evento)
NEWS_LOOKAHEAD_MIN = 45
NEWS_IMPACT_LEVELS = {"High"}          # eventi da notificare
NEWS_CURRENCIES = {"USD"}              # valute rilevanti per l'oro

STATE_FILE = "state.json"

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]

# Feed calendario economico gratuito (mirror comunitario stile ForexFactory)
NEWS_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


# ----------------------------------------------------------------------
# UTILITY
# ----------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_signal_bar": None, "notified_news": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


# ----------------------------------------------------------------------
# DATI DI MERCATO E INDICATORI
# ----------------------------------------------------------------------

def fetch_candles() -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": TIMEFRAME,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if "values" not in data:
        raise RuntimeError(f"Errore API Twelve Data: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)

    # Scarto l'ultima candela: potrebbe essere ancora in formazione
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast_macd"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow_macd"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["macd"] = df["ema_fast_macd"] - df["ema_slow_macd"]
    df["macd_signal"] = df["macd"].rolling(MACD_SIGNAL_PERIOD).mean()  # SMA
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["ema20"] = df["close"].ewm(span=TREND_EMA_FAST, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=TREND_EMA_SLOW, adjust=False).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    return df


def detect_signal(df: pd.DataFrame):
    """Guarda le ultime due candele chiuse per un incrocio MACD/signal."""
    if len(df) < max(EMA_SLOW, TREND_EMA_SLOW, ATR_PERIOD) + 2:
        return None

    prev, curr = df.iloc[-2], df.iloc[-1]

    crossed_up = prev["macd"] <= prev["macd_signal"] and curr["macd"] > curr["macd_signal"]
    crossed_down = prev["macd"] >= prev["macd_signal"] and curr["macd"] < curr["macd_signal"]

    if crossed_up and curr["ema20"] > curr["ema50"]:
        direction = "LONG"
    elif crossed_down and curr["ema20"] < curr["ema50"]:
        direction = "SHORT"
    else:
        return None

    entry = curr["close"]
    atr = curr["atr"]
    if pd.isna(atr):
        return None

    if direction == "LONG":
        sl = entry - SL_ATR_MULT * atr
        tp = entry + TP_ATR_MULT * atr
    else:
        sl = entry + SL_ATR_MULT * atr
        tp = entry - TP_ATR_MULT * atr

    return {
        "direction": direction,
        "bar_time": curr["datetime"].isoformat(),
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "atr": round(atr, 2),
    }


# ----------------------------------------------------------------------
# NOTIZIE
# ----------------------------------------------------------------------

def fetch_upcoming_news():
    resp = requests.get(NEWS_FEED_URL, timeout=20)
    resp.raise_for_status()
    events = resp.json()

    now = datetime.now(timezone.utc)
    upcoming = []

    for ev in events:
        if ev.get("impact") not in NEWS_IMPACT_LEVELS:
            continue
        if ev.get("country") not in NEWS_CURRENCIES:
            continue
        try:
            ev_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue

        minutes_to_event = (ev_time - now).total_seconds() / 60
        if 0 <= minutes_to_event <= NEWS_LOOKAHEAD_MIN:
            upcoming.append({
                "id": f"{ev.get('title')}_{ev['date']}",
                "title": ev.get("title"),
                "time": ev_time.isoformat(),
                "minutes_to_event": round(minutes_to_event),
            })

    return upcoming


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    state = load_state()

    # --- Segnale di trading ---
    try:
        df = fetch_candles()
        df = compute_indicators(df)
        signal = detect_signal(df)

        if signal and signal["bar_time"] != state.get("last_signal_bar"):
            emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
            msg = (
                f"{emoji} <b>Segnale {signal['direction']} XAUUSD</b>\n"
                f"Timeframe: {TIMEFRAME}\n"
                f"Entry: {signal['entry']}\n"
                f"SL: {signal['sl']}\n"
                f"TP: {signal['tp']}\n"
                f"ATR: {signal['atr']}\n"
                f"Candela: {signal['bar_time']}"
            )
            send_telegram(msg)
            state["last_signal_bar"] = signal["bar_time"]
    except Exception as e:
        send_telegram(f"⚠️ Errore nel modulo segnali: {e}")

    # --- Notizie ---
    try:
        upcoming = fetch_upcoming_news()
        already = set(state.get("notified_news", []))
        for ev in upcoming:
            if ev["id"] in already:
                continue
            msg = (
                f"📰 <b>Notizia in arrivo ({ev['minutes_to_event']} min)</b>\n"
                f"{ev['title']}\n"
                f"Orario: {ev['time']}"
            )
            send_telegram(msg)
            already.add(ev["id"])
        # tieni solo le ultime 200 per non far crescere il file all'infinito
        state["notified_news"] = list(already)[-200:]
    except Exception as e:
        send_telegram(f"⚠️ Errore nel modulo notizie: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
