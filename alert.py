#!/usr/bin/env python3
"""
crypto alert bot - full feature set:
- Tracks BTC, ETH, SOL, DOGE
- 1h & 2h % change alerts (default ±4%)
- /price, /status, /alert, /gnfi, /help commands
- GNFI (/gnfi) + auto alerts every 4 hours (thresholds <=25, >=75)
- Daily summary (at 21:00 Asia/Kolkata)
- Cooldown per-coin to avoid spam
- Persistent settings (alert_settings.json)
- Local log file crypto_log.txt
- Works in Termux and in Choreo (use env vars in Choreo)
"""

import os
import json
import time
import threading
import requests
import statistics
import json
import pytz
from flask import Flask
from wsgiref.simple_server import make_server
from datetime import datetime, timezone, timedelta
from math import isfinite
from telegram import Bot
from telegram.ext import Updater, CommandHandler
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
load_dotenv()  # optional local .env

BOT_TOKEN = os.getenv("BOT_TOKEN")            # required (set in Choreo env)
CHAT_ID = os.getenv("CHAT_ID")                # required
COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]
PERCENT_THRESHOLD = float(os.getenv("PERCENT_THRESHOLD", "4"))  # default ±4%
TIMEFRAMES_HOURS = [1, 2]                     # check 1h & 2h
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "150"))  # seconds between checks
GNFI_CHECK_INTERVAL = int(os.getenv("GNFI_CHECK_INTERVAL", str(4 * 3600)))
DAILY_SUMMARY_HOUR_IST = int(os.getenv("DAILY_SUMMARY_HOUR_IST", "21"))  # 21:00 IST
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", str(3600)))  # per-coin cooldown in seconds
# alert cooldown tracking
last_alert_time = {c: 0 for c in COINS}

# volume alert cooldown tracking (seconds)
VOL_ALERT_COOLDOWN = int(os.getenv("VOL_ALERT_COOLDOWN", str(3600)))  # default 3600s = 1 hour
last_volume_alert_time = {c: 0 for c in COINS}

SETTINGS_FILE = "alert_settings.json"
LOG_FILE = "crypto_log.txt"

# ----- INSERT B: add flags and tiny Flask webserver -----
# Flags to control noisy messages (set via environment variables)
STARTUP_MSG_ENABLED = os.getenv("STARTUP_MSG_ENABLED", "false").lower() in ("1", "true", "yes")
GNFI_SCHEDULED_ENABLED = os.getenv("GNFI_SCHEDULED_ENABLED", "false").lower() in ("1", "true", "yes")
VOL_ALERT_ENABLED = os.getenv("VOL_ALERT_ENABLED", "false").lower() in ("1", "true", "yes")

# simple Flask app for uptime / health checks
app = Flask(__name__)

@app.route("/ping")
def ping():
    return "OK", 200

def run_webserver():
    port = int(os.environ.get("PORT", "8000"))  # use Choreo PORT or fallback to 8000
    host = "0.0.0.0"
    try:
        server = make_server(host, port, app)
        print(f"Webserver listening on {host}:{port} (for /ping)")
        server.serve_forever()
    except Exception as e:
        print("Webserver failed to start:", e)
# ----------------------------------------------------------------------

# ----------------- SAFETY / CHECKS ----------------
if not BOT_TOKEN or not CHAT_ID:
    # In cloud, these must be provided via env vars. We raise early so logs show reason.
    raise SystemExit("ERROR: BOT_TOKEN and CHAT_ID environment variables must be set.")

# allow numeric chat id passed as string in env
try:
    # try to convert to int; if escapes, keep as string
    CHAT_ID_PARSED = int(CHAT_ID)
except Exception:
    CHAT_ID_PARSED = CHAT_ID

# ----------------- Globals -----------------
bot = Bot(token=BOT_TOKEN)
updater = Updater(token=BOT_TOKEN, use_context=True)
dispatcher = updater.dispatcher

# price history: coin -> list of (epoch_seconds, price)
price_history = {c: [] for c in COINS}

# GNFI state tracking
last_gnfi_state = None
# timestamp (epoch seconds) when GNFI auto-update/alert was last sent by scheduled job (05:45)
last_gnfi_auto_sent = 0
# cooldown (seconds) to suppress duplicate GNFI alerts if scheduled just ran (1 hour)
GNFI_AUTO_ALERT_COOLDOWN = int(os.getenv("GNFI_AUTO_ALERT_COOLDOWN", str(3600)))
last_gnfi_check = 0

# settings persistence
def load_settings():
    global PERCENT_THRESHOLD
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
            if "percent_threshold" in data:
                PERCENT_THRESHOLD = float(data["percent_threshold"])
    except FileNotFoundError:
        save_settings()

def save_settings():
    data = {"percent_threshold": PERCENT_THRESHOLD}
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f)

def write_log(line):
    ts = datetime.now(timezone.utc).astimezone(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {line}\n")

# ----------------- API helpers -----------------
def fetch_prices_bulk(coins):
    """Fetch multiple coin prices (USD) from CoinGecko in one call. Returns dict coin->price or None."""
    ids = ",".join(coins)
    url = "https://api.coingecko.com/api/v3/simple/price"
    try:
        r = requests.get(url, params={"ids": ids, "vs_currencies": "usd"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {c: data.get(c, {}).get("usd", None) for c in coins}
    except Exception as e:
        print("Error fetch_prices_bulk:", e)
        write_log(f"Error fetch_prices_bulk: {e}")
        return {c: None for c in coins}

def fetch_coin_market_chart_volumes(coin, days=1):
    """Return list of [ts_ms, volume] from CoinGecko market_chart for last 'days' days.
       May return empty list on failure."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        r = requests.get(url, params={"vs_currency": "usd", "days": days}, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("total_volumes", [])  # list of [ts_ms, volume]
    except Exception as e:
        print(f"Error fetch_coin_market_chart_volumes {coin}:", e)
        write_log(f"Error fetch_coin_market_chart_volumes {coin}: {e}")
        return []

def compute_recent_volume_change_percent(coin, recent_hours=2, window_days=1, min_samples=6):
    """
    Robustly compute percent change of volume in the most recent `recent_hours`
    versus the baseline earlier in the last `window_days` days.

    Returns percent (positive = increase, negative = decrease) or None if not enough data.
    """
    try:
        vols = fetch_coin_market_chart_volumes(coin, days=window_days)
        if not vols or len(vols) < min_samples:
            return None

        now_ms = int(time.time() * 1000)
        recent_start = now_ms - recent_hours * 3600 * 1000

        # Separate recent points and baseline points (only include baseline points before recent_start)
        recent_points = [(ts, v) for ts, v in vols if ts >= recent_start]
        baseline_points = [(ts, v) for ts, v in vols if ts < recent_start]

        # Need enough samples on both sides
        if len(recent_points) < 2 or len(baseline_points) < 2:
            return None

        # compute sums
        recent_sum = sum(float(v) for ts, v in recent_points)
        baseline_sum = sum(float(v) for ts, v in baseline_points)

        # compute average per-point on baseline
        baseline_avg_per_point = baseline_sum / len(baseline_points)

        # expected baseline for the recent window scaled to the number of recent samples
        expected_baseline = baseline_avg_per_point * len(recent_points)

        # guard against division by zero
        if expected_baseline == 0:
            return None

        percent = ((recent_sum - expected_baseline) / expected_baseline) * 100.0

        # LOG helpful debug info for investigation
        write_log(f"VOLUME DEBUG: {coin} recent_sum={recent_sum:.2f}, expected_baseline={expected_baseline:.2f}, "
                  f"recent_count={len(recent_points)}, baseline_count={len(baseline_points)}, pct={percent:.2f}")

        return percent
    except Exception as e:
        write_log(f"compute_recent_volume_change_percent ERROR for {coin}: {e}")
        return None

# -------------------- Range Detector (Binance 1H) --------------------
# Settings (matches your TradingView screenshot)
RANGE_MIN_LEN = 25     # Minimum Range Length
RANGE_WIDTH = 1.1      # Range Width multiplier (ATR * RANGE_WIDTH)
ATR_LENGTH = 500       # ATR length
VOL_CONFIRM_FACTOR = 0.9  # breakout candle volume must be >= avg_vol * this
RANGE_SLACK = 5        # extra candles to fetch for safety

# Which symbol(s) to check. Use Binance symbols.
RANGE_SYMBOLS = ["BTCUSDT"]   # add others if you want

RANGE_STATE_FILE = "range_state.json"

def load_range_state():
    try:
        with open(RANGE_STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        # default structure per symbol
        return {s: {"formed": False, "hh": None, "ll": None, "formed_at": 0, "breakout_at": 0} for s in RANGE_SYMBOLS}

def save_range_state(state):
    try:
        with open(RANGE_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        write_log(f"range_state save error: {e}")

def get_binance_ohlc(symbol="BTCUSDT", interval="1h", limit=600):
    """
    Returns list of candles dicts (earliest->latest):
    {time, open, high, low, close, volume}
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    candles = []
    for c in data:
        candles.append({
            "time": int(c[0]),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        })
    return candles

def compute_atr_from_candles(candles, atr_len):
    """
    candles: list of candle dicts in chronological order (earliest->latest)
    atr_len: number of TR values to average (needs atr_len+1 candles)
    returns ATR (float) or None if not enough data
    """
    if len(candles) < atr_len + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < atr_len:
        return None
    # use last atr_len TRs
    recent_trs = trs[-atr_len:]
    return statistics.mean(recent_trs)

def detect_range_and_breakout_for_symbol(symbol):
    """
    Core logic:
      - Fetch candles (ATR_LENGTH + RANGE_MIN_LEN + slack)
      - Compute ATR
      - Compute HH/LL over last RANGE_MIN_LEN closed candles (we use latest closed candles)
      - If width <= ATR * RANGE_WIDTH => formation
      - If formation exists, monitor closed candle close for breakout with volume confirmation
    Returns a tuple (formation_msg_or_None, breakout_msg_or_None)
    """
    try:
        limit = ATR_LENGTH + RANGE_MIN_LEN + RANGE_SLACK
        candles = get_binance_ohlc(symbol=symbol, interval="1h", limit=limit)
    except Exception as e:
        write_log(f"Range detector: Binance fetch failed for {symbol}: {e}")
        return None, None

    # require enough candles
    if len(candles) < ATR_LENGTH + RANGE_MIN_LEN:
        write_log(f"Range detector: not enough candles for {symbol} (have {len(candles)})")
        return None, None

    # compute ATR using all returned candles
    atr = compute_atr_from_candles(candles, ATR_LENGTH)
    if atr is None or atr <= 0:
        write_log(f"Range detector: ATR invalid for {symbol}")
        return None, None

    # consider the last RANGE_MIN_LEN closed candles (latest included)
    recent_slice = candles[-RANGE_MIN_LEN:]
    highs = [c["high"] for c in recent_slice]
    lows = [c["low"] for c in recent_slice]
    HH = max(highs)
    LL = min(lows)
    width = HH - LL

    # average volume for recent window (used for breakout confirmation)
    vol_samples = candles[-max(50, RANGE_MIN_LEN*2):]  # use a larger window for avg vol
    avg_vol = statistics.mean([c["volume"] for c in vol_samples]) if vol_samples else 0.0

    # last closed candle (the most recent closed 1h candle)
    last_closed = candles[-1]
    last_close = last_closed["close"]
    last_vol = last_closed["volume"]

    formation_msg = None
    breakout_msg = None

    # load persisted state
    state = load_range_state()
    s = state.get(symbol, {"formed": False, "hh": None, "ll": None, "formed_at": 0, "breakout_at": 0})

    # check formation condition
    if width <= atr * RANGE_WIDTH:
        # formation detected
        # if previously not formed or HH/LL changed meaningfully, set new formation
        if (not s.get("formed")) or (abs(s.get("hh", 0) - HH) > 1e-8 or abs(s.get("ll", 0) - LL) > 1e-8):
            s["formed"] = True
            s["hh"] = HH
            s["ll"] = LL
            s["formed_at"] = int(time.time())
            s["breakout_at"] = 0
            formation_msg = (f"📦 <b>New Range Formed</b>\nSymbol: {symbol}\nHH: {HH:.2f}\nLL: {LL:.2f}\nWidth: {width:.2f}\nATR: {atr:.2f}")
            write_log(f"Range formed for {symbol} HH={HH} LL={LL} width={width:.2f} atr={atr:.2f}")
    else:
        # if previously formed but now range condition broke, keep state as not formed so new formation can be detected later
        if s.get("formed"):
            s["formed"] = False
            s["hh"] = None
            s["ll"] = None
            s["formed_at"] = 0
            s["breakout_at"] = 0
            write_log(f"Range cleared for {symbol} (width {width:.2f} > atr*{RANGE_WIDTH})")

    # check breakout only if we have an active formation
    if s.get("formed") and s.get("hh") is not None and s.get("ll") is not None:
        HH_saved = s["hh"]
        LL_saved = s["ll"]
        # breakout up: last_close > HH_saved
        if last_close > HH_saved and s.get("breakout_at", 0) == 0:
            # volume confirmation
            if avg_vol == 0 or last_vol >= avg_vol * VOL_CONFIRM_FACTOR:
                breakout_msg = f"🚀 <b>Breakout UP</b>\nSymbol: {symbol}\nPrice: {last_close:.2f}\nRange HH: {HH_saved:.2f}"
                s["breakout_at"] = int(time.time())
                write_log(f"Breakout UP detected for {symbol} price={last_close} hh={HH_saved} vol={last_vol} avg={avg_vol:.2f}")
            else:
                write_log(f"Breakout UP suppressed for {symbol} due low vol {last_vol:.2f} avg {avg_vol:.2f}")
        # breakout down: last_close < LL_saved
        elif last_close < LL_saved and s.get("breakout_at", 0) == 0:
            if avg_vol == 0 or last_vol >= avg_vol * VOL_CONFIRM_FACTOR:
                breakout_msg = f"⚠️ <b>Breakout DOWN</b>\nSymbol: {symbol}\nPrice: {last_close:.2f}\nRange LL: {LL_saved:.2f}"
                s["breakout_at"] = int(time.time())
                write_log(f"Breakout DOWN detected for {symbol} price={last_close} ll={LL_saved} vol={last_vol} avg={avg_vol:.2f}")
            else:
                write_log(f"Breakout DOWN suppressed for {symbol} due low vol {last_vol:.2f} avg {avg_vol:.2f}")

    # save state back
    state[symbol] = s
    save_range_state(state)

    return formation_msg, breakout_msg

def range_detector_job():
    """
    Scheduled job to check for range formation and breakout for each symbol.
    Run this at minute=3 of each hour.
    """
    try:
        for symbol in RANGE_SYMBOLS:
            form_msg, break_msg = detect_range_and_breakout_for_symbol(symbol)
            if form_msg:
                # one formation alert per formation
                send_message(form_msg)
            if break_msg:
                # one breakout alert per breakout
                send_message(break_msg)
    except Exception as e:
        write_log(f"range_detector_job error: {e}")
# --------------------------------------------------------------------

def fetch_gnfi():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=15)
        r.raise_for_status()
        data = r.json()
        if "data" in data and data["data"]:
            item = data["data"][0]
            return {
                "value": int(item.get("value", 0)),
                "classification": item.get("value_classification", "Unknown"),
                "timestamp": int(item.get("timestamp", time.time()))
            }
    except Exception as e:
        print("Error fetch_gnfi:", e)
        write_log(f"Error fetch_gnfi: {e}")
    return None

def pct_change(old, new):
    try:
        return ((new - old) / old) * 100.0
    except Exception:
        return None

# ----------------- Telegram utils -----------------
def send_message(text):
    try:
        bot.send_message(chat_id=CHAT_ID_PARSED, text=text, parse_mode="HTML")
    except Exception as e:
        print("Telegram send error:", e)
        write_log(f"Telegram send error: {e}")

# ----------------- Command handlers -----------------
def help_command(update, context):
    msg = (
        "🤖 <b>Crypto Alert Bot</b>\n\n"
        "/price [coin] - Current price + 1h & 2h % changes\n"
        "/status - Prices of all tracked coins\n"
        "/alert [percent] - Set global alert threshold (e.g. /alert 7)\n"
        "/gnfi - Show Greed & Fear Index\n"
        "/help - Show this message\n\n"
        f"Tracked coins: {', '.join(COINS)}\nThreshold: ±{PERCENT_THRESHOLD}%\nTimeframes: {', '.join([str(h)+'h' for h in TIMEFRAMES_HOURS])}"
    )
    update.message.reply_text(msg, parse_mode="HTML")

def price_command(update, context):
    coin = (context.args[0].lower() if context.args else "bitcoin")
    if coin not in COINS:
        update.message.reply_text("❌ Unknown coin. Use: " + ", ".join(COINS))
        return
    prices = fetch_prices_bulk([coin])
    price = prices.get(coin)
    if price is None:
        update.message.reply_text("⚠️ Price data unavailable right now. Try again soon.")
        return
    now = time.time()
    lines = [f"💰 <b>{coin.upper()}</b>\nPrice: ${price:,.2f}"]
    hist = price_history.get(coin, [])
    for hours in TIMEFRAMES_HOURS:
        target = now - hours * 3600
        # find closest historical price within +/- 3 minutes
        closest = None
        best_diff = 10**9
        for t, p in hist:
            diff = abs(t - target)
            if diff < best_diff and diff <= 180:
                best_diff = diff
                closest = p
        if closest:
            ch = pct_change(closest, price)
            if ch is not None:
                emoji = "⬆️" if ch > 0 else "⬇️"
                lines.append(f"{hours}h: {emoji} {abs(ch):.2f}%")
            else:
                lines.append(f"{hours}h: —")
        else:
            lines.append(f"{hours}h: —")
    update.message.reply_text("\n".join(lines), parse_mode="HTML")

def status_command(update, context):
    prices = fetch_prices_bulk(COINS)
    lines = ["📊 <b>Market Status</b>"]
    for coin in COINS:
        p = prices.get(coin)
        if p is None:
            lines.append(f"{coin.upper()}: unavailable")
        else:
            lines.append(f"{coin.upper()}: ${p:,.2f}")
    update.message.reply_text("\n".join(lines), parse_mode="HTML")

def alert_command(update, context):
    global PERCENT_THRESHOLD
    if not context.args:
        update.message.reply_text(f"Current threshold: ±{PERCENT_THRESHOLD}% (use /alert 4 to set)")
        return
    try:
        val = float(context.args[0])
        PERCENT_THRESHOLD = val
        save_settings()
        update.message.reply_text(f"✅ Alert threshold set to ±{PERCENT_THRESHOLD}%")
        write_log(f"Threshold updated to ±{PERCENT_THRESHOLD}% via /alert by user {update.effective_user.id}")
    except Exception:
        update.message.reply_text("Usage: /alert 7")

def gnfi_command(update, context):
    """
    Reply with current GNFI. If value is extreme and the morning job didn't
    already send an alert recently, also send an immediate alert message.
    This prevents duplicates: primary alert time is the 05:45 IST morning job.
    """
    global last_gnfi_auto_sent, last_gnfi_state

    obj = fetch_gnfi()
    if not obj:
        update.message.reply_text("⚠️ GNFI data unavailable right now.")
        return

    # Format and reply the GNFI info to the user who called /gnfi
    dt = datetime.fromtimestamp(obj["timestamp"], timezone.utc).astimezone(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")
    value = obj["value"]
    classification = obj["classification"]

    reply_msg = (
        f"📊 <b>Greed & Fear Index</b>\n\n"
        f"Value: {value} ({classification})\n"
        f"Last updated: {dt}\n"
        f"Source: alternative.me"
    )
    update.message.reply_text(reply_msg, parse_mode="HTML")

    # If GNFI is extreme, decide whether to send a bot-wide alert.
    # We only send if the morning auto-job hasn't already sent an alert in the cooldown window.
    try:
        now_ts = int(time.time())

        if value <= 25 or value >= 75:
            # If morning job already sent an alert recently, suppress duplicate alert
            if last_gnfi_auto_sent and (now_ts - last_gnfi_auto_sent) < GNFI_AUTO_ALERT_COOLDOWN:
                write_log(f"/gnfi called but morning GNFI alert recently sent; suppressing duplicate alert. value={value}")
                # Do not send a second bot-wide alert — user already got morning alert.
                return

            # Otherwise send an alert because user explicitly requested /gnfi
            if value <= 25:
                send_message(f"⚠️ ALERT: Market is in <b>Extreme Fear</b> ({value}) 😰 — manual /gnfi")
                write_log(f"Manual /gnfi ALERT: Extreme Fear ({value})")
                last_gnfi_state = "fear"
                last_gnfi_auto_sent = now_ts
            else:  # value >= 75
                send_message(f"🚨 ALERT: Market is in <b>Extreme Greed</b> ({value}) 🤪 — manual /gnfi")
                write_log(f"Manual /gnfi ALERT: Extreme Greed ({value})")
                last_gnfi_state = "greed"
                last_gnfi_auto_sent = now_ts

    except Exception as e:
        write_log(f"gnfi_command alert logic error: {e}")

# Register command handlers
dispatcher.add_handler(CommandHandler("help", help_command))
dispatcher.add_handler(CommandHandler("price", price_command))
dispatcher.add_handler(CommandHandler("status", status_command))
dispatcher.add_handler(CommandHandler("alert", alert_command))
dispatcher.add_handler(CommandHandler("gnfi", gnfi_command))

# ----------------- Background worker (price checks, alerts) -----------------
def background_worker():
    global last_gnfi_check, last_gnfi_state
    load_settings()
    print("Background worker started.")
    write_log("Background worker started.")
    while True:
        now = time.time()
        timestamp = now

        # 1) Bulk fetch current prices
        prices = fetch_prices_bulk(COINS)

        # 2) Append to history and prune older than 2h + buffer
        for coin, price in prices.items():
            if price is None:
                continue
            price_history[coin].append((timestamp, price))
            price_history[coin] = [(t, p) for (t, p) in price_history[coin] if timestamp - t <= 2 * 3600 + 300]

        # 3) Check thresholds (1h and 2h) per coin
        for coin in COINS:
            current_price = prices.get(coin)
            if current_price is None:
                continue
            hist = price_history.get(coin, [])
            for hours in TIMEFRAMES_HOURS:
                target = timestamp - hours * 3600
                closest = None
                best_diff = 10**9
                for t, p in hist:
                    diff = abs(t - target)
                    if diff < best_diff and diff <= 180:
                        best_diff = diff
                        closest = p
                if closest is None:
                    continue
                change = pct_change(closest, current_price)
                if change is None:
                    continue
                if abs(change) >= PERCENT_THRESHOLD:
                    # cooldown check
                    last_time = last_alert_time.get(coin, 0)
                    if timestamp - last_time >= ALERT_COOLDOWN:
                        arrow = "⬆️" if change > 0 else "⬇️"
                        msg = (
                            f"🚨 <b>{coin.upper()}</b> {arrow} {abs(change):.2f}% in last {hours}h\n"
                            f"Price: ${current_price:,.2f}\n"
                            f"Time (IST): {datetime.now(timezone.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z')}"
                        )
                        send_message(msg)
                        write_log(f"ALERT: {coin} {change:.2f}% over {hours}h | price ${current_price:,.2f}")
                        last_alert_time[coin] = timestamp
                        # prune small history for this coin to avoid duplicate slightly different alerts
                        price_history[coin] = price_history[coin][-1:]
                        break  # skip checking the other timeframe this cycle

        # 4) Volume alert: compare recent 2h volume to baseline (using market_chart days=1)
        for coin in COINS:
            try:
                vol_pct = compute_recent_volume_change_percent(coin, recent_hours=2, window_days=1)
                if vol_pct is None:
                    continue

                # only alert if absolute change crosses the percentage threshold (20% default)
                if abs(vol_pct) >= 20.0:
                    # cooldown logic & global enable flag
                    if VOL_ALERT_ENABLED:
                        now_ts = timestamp  # current loop timestamp (already defined earlier)
                        last_vol_ts = last_volume_alert_time.get(coin, 0)
                        if (now_ts - last_vol_ts) >= VOL_ALERT_COOLDOWN:
                            send_message(f"⚡ <b>{coin.upper()}</b> volume changed {vol_pct:.2f}% in last 2h (approx).")
                            write_log(f"VOLUME ALERT: {coin} vol change {vol_pct:.2f}%")
                            last_volume_alert_time[coin] = now_ts
                        else:
                            # cooldown active — suppressed
                            write_log(f"VOLUME ALERT suppressed for {coin} (cooldown). pct={vol_pct:.2f}")
                    else:
                        # volume alerts globally disabled
                        write_log(f"Volume alert detected for {coin}, but VOL_ALERT_ENABLED=false. pct={vol_pct:.2f}")
            except Exception as e:
                print("Volume check error:", e)
                write_log(f"Volume check error for {coin}: {e}")

        # Sleep then loop
        time.sleep(CHECK_INTERVAL)

# ----------------- Scheduler jobs (daily summary) -----------------
def daily_summary_job():
    prices = fetch_prices_bulk(COINS)
    lines = ["📅 <b>Daily Summary (IST)</b>"]
    for coin, p in prices.items():
        if p is None:
            lines.append(f"{coin.upper()}: unavailable")
        else:
            lines.append(f"{coin.upper()}: ${p:,.2f}")
    gnfi = fetch_gnfi()
    if gnfi:
        lines.append(f"\n🧠 GNFI: {gnfi['value']} ({gnfi['classification']})")
    send_message("\n".join(lines))
    write_log("Daily summary sent.")

def morning_gnfi_job():
    """
    Runs once per day at 05:45 IST.
    Sends GNFI update and, if extreme, sends an alert.
    """
    global last_gnfi_auto_sent, last_gnfi_state
    obj = fetch_gnfi()
    if not obj:
        write_log("Morning GNFI job: GNFI data unavailable.")
        return

    value = obj["value"]
    classification = obj["classification"]
    dt = datetime.fromtimestamp(obj["timestamp"], timezone.utc).astimezone(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")

    # Informational morning update
    msg = (
        f"🌅 <b>Morning GNFI Update (IST)</b>\n\n"
        f"Value: {value} ({classification})\n"
        f"Last updated: {dt}\n"
        f"Source: alternative.me"
    )
    send_message(msg)
    write_log(f"Morning GNFI update sent: {value} ({classification})")

    # Send alert only if extreme; record when we did so to suppress duplicates after manual /gnfi
    now_ts = int(time.time())
    if value <= 25:
        send_message(f"⚠️ ALERT: Market is in <b>Extreme Fear</b> ({value}) 😰 — Morning check")
        write_log(f"Morning GNFI ALERT: Extreme Fear ({value})")
        last_gnfi_auto_sent = now_ts
        last_gnfi_state = "fear"
    elif value >= 75:
        send_message(f"🚨 ALERT: Market is in <b>Extreme Greed</b> ({value}) 🤪 — Morning check")
        write_log(f"Morning GNFI ALERT: Extreme Greed ({value})")
        last_gnfi_auto_sent = now_ts
        last_gnfi_state = "greed"
    else:
        last_gnfi_state = "normal"

# ----------------- Start & run -----------------
def main():
    # quick start message and setup
    try:
        if STARTUP_MSG_ENABLED:
            send_message("✅ Bot started and monitoring crypto prices...")
        else:
            write_log("Startup message suppressed (STARTUP_MSG_ENABLED=false).")
    except Exception as e:
        print("Start send failed:", e)
        write_log(f"Start send failed: {e}")
    load_settings()

    # ----- INSERT 3: place this just before updater.start_polling() inside main() -----
    # start the small webserver thread so uptime monitors can ping /ping
    web_thread = threading.Thread(target=run_webserver, daemon=True)
    web_thread.start()
    # ------------------------------------------------------------------------------------
    # start updater (handles commands)
    updater.start_polling()

    # start background price worker thread
    worker = threading.Thread(target=background_worker, daemon=True)
    worker.start()

    # APScheduler for daily summary and optionally other timed jobs
    scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Kolkata"))
    # daily summary at DAILY_SUMMARY_HOUR_IST IST
    scheduler.add_job(daily_summary_job, "cron", hour=DAILY_SUMMARY_HOUR_IST, minute=0, timezone=pytz.timezone("Asia/Kolkata"))
    # Morning GNFI job at 05:45 IST
    scheduler.add_job(morning_gnfi_job, "cron", hour=5, minute=45, timezone=pytz.timezone("Asia/Kolkata"))
    # run detector at :03 each hour so the H:00 candle is closed
    scheduler.add_job(range_detector_job, "cron", minute=3, timezone=pytz.timezone("Asia/Kolkata"))
    # GNFI regular check - we rely on background_worker for GNFI alerts, but we also keep a scheduled GNFI post if wanted:
    # only add the scheduled GNFI manual message if enabled
    if GNFI_SCHEDULED_ENABLED:
        scheduler.add_job(
            lambda: send_message("📊 GNFI check (manual scheduled)"),
            "interval",
            hours=GNFI_CHECK_INTERVAL/3600,
            timezone=pytz.timezone("Asia/Kolkata")
        )
    else:
        print("GNFI scheduled messages are disabled (GNFI_SCHEDULED_ENABLED=false)")
    scheduler.start()

    print("Bot running. Press Ctrl+C to stop.")
    write_log("Bot started.")
    updater.idle()
# start the small webserver thread so uptime monitors can hit /ping
if __name__ == "__main__":
    main()
