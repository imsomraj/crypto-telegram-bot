#!/usr/bin/env python3
"""
crypto alert bot - full feature set:
- Tracks BTC, ETH, SOL, DOGE
- 1h & 2h % change alerts (default ±7%)
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
PERCENT_THRESHOLD = float(os.getenv("PERCENT_THRESHOLD", "7"))  # default ±7%
TIMEFRAMES_HOURS = [1, 2]                     # check 1h & 2h
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))  # seconds between checks
GNFI_CHECK_INTERVAL = int(os.getenv("GNFI_CHECK_INTERVAL", str(4 * 3600)))
DAILY_SUMMARY_HOUR_IST = int(os.getenv("DAILY_SUMMARY_HOUR_IST", "21"))  # 21:00 IST
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", str(3600)))  # per-coin cooldown in seconds

SETTINGS_FILE = "alert_settings.json"
LOG_FILE = "crypto_log.txt"

# ----- INSERT 2: paste this once after your global constants (e.g. after LOG_FILE) -----
# simple Flask app for uptime pings (keeps Choreo alive when pinged)
app = Flask(__name__)

# public health route
@app.route("/ping")
def ping():
    return "OK", 200

# optional secret ping (recommended)
# SECRET_PING = os.environ.get("SECRET_PING", "")
# if SECRET_PING:
#     @app.route(f"/ping/{SECRET_PING}")
#     def secret_ping():
#         return "OK", 200

def run_webserver():
    """
    Start a simple WSGI server on the port provided by the environment (Choreo sets PORT).
    Runs in a daemon thread so it doesn't block the bot.
    """
    port = int(os.environ.get("PORT", "8080"))
    try:
        server = make_server("0.0.0.0", port, app)
        print(f"Webserver listening on 0.0.0.0:{port} (for /ping)")
        server.serve_forever()
    except Exception as e:
        print("Webserver failed to start:", e)
# -----------------------------------------------------------------------------------------

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

# alert cooldown tracking
last_alert_time = {c: 0 for c in COINS}

# GNFI state tracking
last_gnfi_state = None
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

def compute_recent_volume_change_percent(coin, recent_hours=2, window_days=1):
    """Compute percent change: volume in last recent_hours vs avg hourly volume over previous window_days (rough).
       Returns percent or None if not possible."""
    vols = fetch_coin_market_chart_volumes(coin, days=window_days)
    if not vols:
        return None
    now_ms = int(time.time() * 1000)
    recent_start = now_ms - recent_hours * 3600 * 1000
    recent_sum = 0.0
    prev_sum = 0.0
    for ts_ms, vol in vols:
        if ts_ms >= recent_start:
            recent_sum += float(vol)
        else:
            prev_sum += float(vol)
    # avoid dividing by zero; compute percent relative to prev_avg scaled to recent_hours
    if prev_sum <= 0:
        return None
    # scale prev_sum proportionally: prev_sum covers approx (window_days*24 - recent_hours) hours -> convert to rate
    # Simpler: percent = (recent_sum - (prev_sum * recent_hours / (len(vols)/24 * 24))) / baseline
    # We'll compute baseline_avg_per_ms = prev_sum / (total_prev_ms)
    # But we don't have exact spacing; approximate by ratio of counts:
    prev_count = sum(1 for ts_ms, v in vols if ts_ms < recent_start)
    if prev_count == 0:
        return None
    # compute average per-point
    prev_avg = prev_sum / prev_count
    # expected baseline for recent_hours points ~ prev_avg * (recent_count)
    recent_count = sum(1 for ts_ms, v in vols if ts_ms >= recent_start)
    if recent_count == 0:
        return None
    baseline = prev_avg * recent_count
    if baseline <= 0:
        return None
    percent = ((recent_sum - baseline) / baseline) * 100.0
    return percent

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
        update.message.reply_text(f"Current threshold: ±{PERCENT_THRESHOLD}% (use /alert 7 to set)")
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
    obj = fetch_gnfi()
    if not obj:
        update.message.reply_text("⚠️ GNFI data unavailable right now.")
        return
    dt = datetime.fromtimestamp(obj["timestamp"], timezone.utc).astimezone(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")
    msg = (
        f"📊 <b>Greed & Fear Index</b>\n\n"
        f"Value: {obj['value']} ({obj['classification']})\n"
        f"Last updated: {dt}\n"
        f"Source: alternative.me"
    )
    update.message.reply_text(msg, parse_mode="HTML")

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
                if vol_pct is not None and abs(vol_pct) >= 20.0:  # threshold 20% by default
                    send_message(f"⚡ <b>{coin.upper()}</b> volume changed {vol_pct:.2f}% in last 2h (approx).")
                    write_log(f"VOLUME ALERT: {coin} vol change {vol_pct:.2f}%")
            except Exception as e:
                print("Volume check error:", e)

        # 5) GNFI periodic check and auto-alert (every GNFI_CHECK_INTERVAL)
        if now - last_gnfi_check >= GNFI_CHECK_INTERVAL:
            obj = fetch_gnfi()
            last_gnfi_check = now
            if obj:
                val = obj["value"]
                # determine state
                state = None
                if val <= 25:
                    state = "fear"
                elif val >= 75:
                    state = "greed"
                else:
                    state = "normal"
                # alert only on change to fear/greed
                if state != "normal" and state != last_gnfi_state:
                    if state == "fear":
                        send_message(f"⚠️ ALERT: Market is in <b>Extreme Fear</b> ({val}) 😰\nClassification: {obj['classification']}")
                        write_log(f"GNFI ALERT: Extreme Fear ({val})")
                    elif state == "greed":
                        send_message(f"🚨 ALERT: Market is in <b>Extreme Greed</b> ({val}) 🤪\nClassification: {obj['classification']}")
                        write_log(f"GNFI ALERT: Extreme Greed ({val})")
                last_gnfi_state = state

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

# ----------------- Start & run -----------------
def main():
    # quick start message and setup
    try:
        send_message("✅ Bot started and monitoring crypto prices...")
    except Exception as e:
        print("Start send failed (may be OK if CHAT_ID invalid):", e)
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
    # GNFI regular check - we rely on background_worker for GNFI alerts, but we also keep a scheduled GNFI post if wanted:
    scheduler.add_job(lambda: send_message("📊 GNFI check (manual scheduled)"), "interval", hours=GNFI_CHECK_INTERVAL/3600, timezone=pytz.timezone("Asia/Kolkata"))
    scheduler.start()

    print("Bot running. Press Ctrl+C to stop.")
    write_log("Bot started.")
    updater.idle()

if __name__ == "__main__":
    main()
