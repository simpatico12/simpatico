
import sqlite3
import os
import time
import requests
import pyupbit
import schedule
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
IS_LIVE = False

upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
DB_PATH = "trading.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS trading_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    coin TEXT,
    decision TEXT,
    percentage INTEGER,
    confidence_score INTEGER,
    reason TEXT,
    reaction TEXT,
    coin_balance REAL,
    krw_balance REAL,
    avg_buy_price REAL,
    coin_price REAL
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS trading_reflection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trading_id INTEGER NOT NULL,
    reflection_date DATETIME NOT NULL,
    market_condition TEXT NOT NULL,
    decision_analysis TEXT NOT NULL,
    improvement_points TEXT NOT NULL,
    success_rate REAL NOT NULL,
    learning_points TEXT NOT NULL,
    FOREIGN KEY (trading_id) REFERENCES trading_history(id)
)
""")
conn.commit()

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("텔레그램 오류:", e)

def record_trade(coin, decision, percentage, confidence_score, reason, reaction, coin_balance, krw_balance, avg_price, coin_price):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO trading_history (timestamp, coin, decision, percentage, confidence_score, reason, reaction, coin_balance, krw_balance, avg_buy_price, coin_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, coin, decision, percentage, confidence_score, reason, reaction, coin_balance, krw_balance, avg_price, coin_price)
    )
    conn.commit()

def analyze_market(coin):
    return {
        "decision": "buy",
        "reason": f"{coin} 기술적 분석 기반 상승 예상",
        "confidence_score": 84,
        "percentage": 30
    }

COINS = ["BTC", "ETH", "XRP"]
def run_auto_trade():
    for coin in COINS:
        try:
            krw = pyupbit.get_balance("KRW")
            coin_balance = pyupbit.get_balance(f"KRW-{coin}")
            price = pyupbit.get_current_price(f"KRW-{coin}") or 1
            avg_price = next((float(b['avg_buy_price']) for b in pyupbit.get_balances() if b['currency'] == coin), 0)

            signal = analyze_market(coin)
            ratio = signal["percentage"] / 100
            parts = 3
            reaction = ""

            if signal["decision"] == "buy" and krw * ratio > 5000:
                unit = (krw * ratio) / parts
                for i in range(parts):
                    if IS_LIVE:
                        upbit.buy_market_order(f"KRW-{coin}", unit)
                    send_telegram(f"💸 [{coin}] {i+1}차 매수 - {unit:,.0f}원")
                    time.sleep(1)
                reaction = f"{parts}회 분할매수 실행"

            elif signal["decision"] == "sell" and coin_balance > 0:
                qty = coin_balance * ratio
                unit_qty = qty / 2
                for i in range(2):
                    if IS_LIVE:
                        upbit.sell_market_order(f"KRW-{coin}", unit_qty)
                    send_telegram(f"📉 [{coin}] {i+1}차 매도 - {unit_qty:.6f} {coin}")
                    time.sleep(1)
                reaction = "2회 분할매도 실행"
            else:
                reaction = "보류됨"

            record_trade(coin, signal["decision"], signal["percentage"], signal["confidence_score"], signal["reason"], reaction, coin_balance, krw, avg_price, price)
            send_telegram(f"✅ [{coin}] 거래 기록 완료: {signal['decision'].upper()} | 신뢰도 {signal['confidence_score']}%\n사유: {signal['reason']}")
            time.sleep(2)

        except Exception as e:
            send_telegram(f"❌ [{coin}] 오류 발생: {e}")

def run_scheduler():
    schedule.every().day.at("09:00").do(run_auto_trade)
    schedule.every().day.at("15:00").do(run_auto_trade)
    send_telegram("✅ 자동매매 스케줄 시작됨 (09시 / 15시)")
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    run_scheduler()
