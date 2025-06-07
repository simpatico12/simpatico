
import schedule
import time
from core.strategy import analyze_coin
from core.trade_engine import execute_trading_decision
from utils import send_telegram

COINS = ["BTC", "ETH", "XRP"]

def run():
    for coin in COINS:
        try:
            signal = analyze_coin(coin)
            execute_trading_decision(coin, signal)
            time.sleep(1)
        except Exception as e:
            send_telegram(f"❌ [{coin}] 시스템 오류 발생: {e}")

if __name__ == "__main__":
    schedule.every().day.at("08:30").do(run)
    schedule.every().day.at("09:00").do(run)
    schedule.every().day.at("14:30").do(run)
    schedule.every().day.at("15:00").do(run)
    send_telegram("✅ AI 자동매매 스케줄러 시작 (08:30 / 09:00 / 14:30 / 15:00)")
    while True:
        schedule.run_pending()
        time.sleep(10)
