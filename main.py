"""
Bot Telegram per segnali di trading su XAUUSD.

Strategia:
  - MACD: EMA veloce 12, EMA lenta 26, signal = SMA a 9 periodi (timeframe M15)
  - Trend filter M15: EMA20 vs EMA50
  - Conferma multi-timeframe: il trend su H1 (EMA20 vs EMA50) deve essere
    coerente con la direzione del segnale
  - Filtro di volatilità minima: l'ATR corrente deve essere almeno
    MIN_ATR_RATIO volte la sua media mobile, altrimenti il mercato è
    considerato troppo piatto e il segnale viene scartato
  - LONG  -> MACD incrocia al rialzo la signal line, EMA20 > EMA50 su M15,
             trend H1 rialzista, volatilità sufficiente
  - SHORT -> MACD incrocia al ribasso la signal line, EMA20 < EMA50 su M15,
             trend H1 ribassista, volatilità sufficiente
  - TP/SL calcolati automaticamente in base all'ATR(14)

Pensato per girare periodicamente (es. ogni 15 minuti) via GitHub Actions,
senza bisogno di un server sempre acceso.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

# ----------------------------------------------------------------------
# CONFIGURAZIONE (modifica pure questi valori)
# ----------------------------------------------------------------------

SYMBOL = "XAU/USD"
TIMEFRAME = os.environ.get("TIMEFRAME", "15min")   # timeframe del segnale
TREND_TIMEFRAME = os.environ.get("TREND_TIMEFRAME", "1h")  # timeframe di conferma
OUTPUT_SIZE = 200          # candele storiche da scaricare (timeframe segnale)
TREND_OUTPUT_SIZE = 100    # candele storiche da scaricare (timeframe di conferma)

EMA_FAST = 12
EMA_SLOW = 26
MACD_SIGNAL_PERIOD = 9   # SMA, come richiesto (non EMA)

TREND_EMA_FAST = 20
TREND_EMA_SLOW = 50

ATR_PERIOD = 14
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0        # risk/reward 1:2

# Filtro di volatilità minima: l'ATR attuale deve valere almeno questa
# frazione della sua media mobile su ATR_MA_PERIOD candele, altrimenti
# il mercato è considerato troppo piatto e il segnale viene scartato.
ATR_MA_PERIOD = 50
MIN_ATR_RATIO = 0.8

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

# Pattern noti per stimare impatto/durata attesa sul mercato (USD/Gold).
# La direzione reale dipende dal dato effettivo vs le attese: qui indichiamo
# lo scenario "sopra attese" e "sotto attese", non una previsione certa.
NEWS_IMPACT_PATTERNS = [
    (["non-farm payroll", "nonfarm payroll", "nfp"],
     "Dato occupazionale chiave. Sopra attese → USD si rafforza → oro tende a scendere. "
     "Sotto attese → USD si indebolisce → oro tende a salire.",
     "Picco di volatilità nei primi 15-30 minuti, effetti sul trend che possono proseguire per l'intera giornata."),
    (["cpi", "consumer price index", "inflation rate"],
     "Dato sull'inflazione. Sopra attese → si rafforzano le aspettative di tassi alti → USD forte → oro tende a scendere. "
     "Sotto attese → effetto opposto, oro tende a salire.",
     "Forte volatilità nei primi 15-30 minuti, possibili riprese di movimento nelle ore successive."),
    (["pce price index", "core pce"],
     "Indicatore di inflazione preferito dalla Fed. Stesso schema del CPI: sopra attese penalizza l'oro, sotto attese lo favorisce.",
     "Volatilità concentrata nei primi 15-30 minuti."),
    (["fomc", "federal funds rate", "interest rate decision", "fed interest rate"],
     "Decisione sui tassi Fed. Tono/rialzo hawkish → USD forte → oro tende a scendere. "
     "Tono/taglio dovish → USD debole → oro tende a salire.",
     "Reazione immediata forte alla decisione, spesso seguita da una seconda ondata di volatilità durante la conferenza stampa (fino a 1-2 ore dopo)."),
    (["fomc press conference", "powell speak", "fed chair"],
     "Conferenza stampa/discorso del presidente Fed. Alta imprevedibilità, il mercato reagisce parola per parola.",
     "Volatilità elevata e irregolare per tutta la durata dell'intervento, 30-90 minuti."),
    (["unemployment claims", "jobless claims"],
     "Dato settimanale sul mercato del lavoro USA, impatto minore ma può muovere il dollaro a breve termine.",
     "Volatilità contenuta, di solito esaurita entro 15-20 minuti."),
    (["gdp"],
     "Dato sulla crescita economica. Sopra attese tende a sostenere USD (oro giù); sotto attese l'opposto.",
     "Volatilità moderata nei primi 20-30 minuti."),
    (["retail sales"],
     "Consumi USA. Dato forte sostiene USD (oro giù), dato debole lo indebolisce (oro su).",
     "Volatilità moderata, 15-30 minuti."),
    (["ism manufacturing", "ism services", "pmi"],
     "Indice di attività economica. Sopra attese sostiene USD (oro giù); sotto attese l'opposto.",
     "Volatilità moderata, 15-30 minuti."),
]


def classify_news_impact(title: str):
    t = (title or "").lower()
    for keywords, impact_text, duration_text in NEWS_IMPACT_PATTERNS:
        if any(k in t for k in keywords):
            return impact_text, duration_text
    return (
        "Evento ad alto impatto per USD: la direzione dipende dal dato effettivo vs le attese "
        "(migliore del previsto tende a rafforzare il USD e a far scendere l'oro, e viceversa).",
        "Possibile volatilità nei primi 15-30 minuti dalla pubblicazione.",
    )


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
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


def fmt_time(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str)
    return dt.strftime("%d/%m %H:%M UTC")


# ----------------------------------------------------------------------
# DATI DI MERCATO E INDICATORI
# ----------------------------------------------------------------------

def fetch_candles(interval: str, output_size: int) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": output_size,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if "values" not in data:
        raise RuntimeError(f"Errore API Twelve Data ({interval}): {data}")

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
    df["atr_ma"] = df["atr"].rolling(ATR_MA_PERIOD).mean()

    return df


def get_trend_confirmation():
    """Trend sul timeframe superiore (es. H1): 'up', 'down' oppure None."""
    df = fetch_candles(TREND_TIMEFRAME, TREND_OUTPUT_SIZE)
    if len(df) < TREND_EMA_SLOW + 2:
        return None

    df["ema20"] = df["close"].ewm(span=TREND_EMA_FAST, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=TREND_EMA_SLOW, adjust=False).mean()
    curr = df.iloc[-1]

    if curr["ema20"] > curr["ema50"]:
        return "up"
    if curr["ema20"] < curr["ema50"]:
        return "down"
    return None


def detect_signal(df: pd.DataFrame, trend_confirmation):
    """Guarda le ultime due candele chiuse per un incrocio MACD/signal,
    applicando conferma multi-timeframe e filtro di volatilità minima."""
    min_bars = max(EMA_SLOW, TREND_EMA_SLOW, ATR_PERIOD + ATR_MA_PERIOD) + 2
    if len(df) < min_bars:
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

    # --- Filtro di volatilità minima ---
    atr = curr["atr"]
    atr_ma = curr["atr_ma"]
    if pd.isna(atr) or pd.isna(atr_ma):
        return None
    if atr < MIN_ATR_RATIO * atr_ma:
        return None  # mercato troppo piatto, segnale scartato

    # --- Conferma multi-timeframe ---
    needed_trend = "up" if direction == "LONG" else "down"
    if trend_confirmation != needed_trend:
        return None  # trend sul timeframe superiore non conferma

    entry = curr["close"]
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
# MESSAGGI TELEGRAM
# ----------------------------------------------------------------------

def build_signal_message(signal) -> str:
    is_long = signal["direction"] == "LONG"
    emoji = "🟢" if is_long else "🔴"
    arrow = "📈" if is_long else "📉"
    rr = TP_ATR_MULT / SL_ATR_MULT

    return (
        f"{emoji} <b>SEGNALE {signal['direction']}  —  XAUUSD</b> {arrow}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⏱ <b>Timeframe:</b> {TIMEFRAME}  (trend {TREND_TIMEFRAME} confermato ✅)\n\n"
        f"💰 <b>Entry:</b> <code>{signal['entry']}</code>\n"
        f"🛑 <b>Stop Loss:</b> <code>{signal['sl']}</code>  (−{SL_ATR_MULT}×ATR)\n"
        f"🎯 <b>Take Profit:</b> <code>{signal['tp']}</code>  (+{TP_ATR_MULT}×ATR)\n"
        f"⚖️ <b>Risk/Reward:</b> 1:{rr:.1f}\n\n"
        f"📊 <b>ATR attuale:</b> {signal['atr']}\n"
        f"🕓 <b>Candela:</b> {fmt_time(signal['bar_time'])}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )


def build_news_message(ev) -> str:
    impact_text, duration_text = classify_news_impact(ev["title"])
    forecast = ev.get("forecast") or "n/d"
    previous = ev.get("previous") or "n/d"

    return (
        f"📰 <b>NOTIZIA IN ARRIVO</b>  ⏳ tra {ev['minutes_to_event']} min\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🗞 <b>{ev['title']}</b>\n"
        f"🕒 {fmt_time(ev['time'])}\n"
        f"📌 Previsione: <b>{forecast}</b>  |  Precedente: <b>{previous}</b>\n\n"
        f"📈 <b>Possibile impatto:</b>\n{impact_text}\n\n"
        f"⏱ <b>Durata attesa:</b>\n{duration_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )


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
                "forecast": ev.get("forecast"),
                "previous": ev.get("previous"),
            })

    return upcoming


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    state = load_state()

    # --- Segnale di trading ---
    try:
        trend_confirmation = get_trend_confirmation()

        df = fetch_candles(TIMEFRAME, OUTPUT_SIZE)
        df = compute_indicators(df)
        signal = detect_signal(df, trend_confirmation)

        if signal and signal["bar_time"] != state.get("last_signal_bar"):
            send_telegram(build_signal_message(signal))
            state["last_signal_bar"] = signal["bar_time"]
    except Exception as e:
        send_telegram(f"⚠️ <b>Errore nel modulo segnali</b>\n{e}")

    # --- Notizie ---
    try:
        upcoming = fetch_upcoming_news()
        already = set(state.get("notified_news", []))
        for ev in upcoming:
            if ev["id"] in already:
                continue
            send_telegram(build_news_message(ev))
            already.add(ev["id"])
        # tieni solo le ultime 200 per non far crescere il file all'infinito
        state["notified_news"] = list(already)[-200:]
    except Exception as e:
        send_telegram(f"⚠️ <b>Errore nel modulo notizie</b>\n{e}")

    save_state(state)


if __name__ == "__main__":
    main()
