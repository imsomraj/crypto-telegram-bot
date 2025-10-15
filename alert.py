import os
import requests
import datetime
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

COINS = ["bitcoin", "ethereum", "dogecoin", "solana"]
THRESHOLD = 7  # % change
TIMEFRAMES = ["1h", "2h"]

def get_price(coin):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
    response = requests.get(url).json()
    return response[coin]["usd"]

def get_gnfi():
    url = "https://api.alternative.me/fng/?limit=1&format=json"
    data = requests.get(url).json()
    return data["data"][0]["value"], data["data"][0]["value_classification"]

def price_command(update: Update, context: CallbackContext):
    message = "💰 *Crypto Prices (USD)*\n"
    for coin in COINS:
        price = get_price(coin)
        message += f"{coin.capitalize()}: ${price}\n"
    update.message.reply_text(message, parse_mode="Markdown")

def gnfi_command(update: Update, context: CallbackContext):
    value, level = get_gnfi()
    update.message.reply_text(f"😈 *Fear & Greed Index:* {value} ({level})", parse_mode="Markdown")

def alert_prices(context: CallbackContext):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"🚨 Auto Crypto Alert ({now})\n"
    for coin in COINS:
        price = get_price(coin)
        message += f"{coin.capitalize()}: ${price}\n"
    context.bot.send_message(chat_id=CHAT_ID, text=message)

def alert_gnfi(context: CallbackContext):
    value, level = get_gnfi()
    message = f"📊 Auto GNFI Alert:\nFear & Greed Index = {value} ({level})"
    context.bot.send_message(chat_id=CHAT_ID, text=message)

def main():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("price", price_command))
    dp.add_handler(CommandHandler("gnfi", gnfi_command))

    scheduler = BackgroundScheduler()
    scheduler.add_job(alert_prices, "interval", hours=2, args=[updater.job_queue])
    scheduler.add_job(alert_gnfi, "interval", hours=6, args=[updater.job_queue])
    scheduler.start()

    print("✅ Bot running on Choreo...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
