import sqlite3
import os
import time
import requests
import pyupbit
import schedule
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1) .env 파일 로드
load_dotenv()
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
IS_LIVE = True  # 실전 매매하려면 True, 테스트 모드로 둘 땐 False

# ─────────────────────────────────────────────────────────────────────────────
# 2) Upbit 인스턴스 생성 (IP 화이트리스트에 EC2 공인 IP 등록 필요)
upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# 3) SQLite DB 설정 (거래 기록, 회고용 테이블 생성)
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

# ─────────────────────────────────────────────────────────────────────────────
# 4) Telegram 메시지 전송 함수
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("텔레그램 오류:", e)

# ─────────────────────────────────────────────────────────────────────────────
# 5) 거래 기록 저장 함수
def record_trade(
    coin, decision, percentage, confidence_score,
    reason, reaction, coin_balance, krw_balance,
    avg_price, coin_price
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO trading_history "
        "(timestamp, coin, decision, percentage, confidence_score, "
        " reason, reaction, coin_balance, krw_balance, avg_buy_price, coin_price) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            timestamp, coin, decision, percentage,
            confidence_score, reason, reaction,
            coin_balance, krw_balance, avg_price, coin_price
        )
    )
    conn.commit()

# ─────────────────────────────────────────────────────────────────────────────
# 6) 예시용 시장 분석 함수 (항상 "buy" 반환)
def analyze_market(coin):
    return {
        "decision": "buy",
        "reason": f"{coin} 기술적 분석 기반 상승 예상",
        "confidence_score": 84,
        "percentage": 30
    }

# ─────────────────────────────────────────────────────────────────────────────
COINS = ["BTC", "ETH", "XRP"]

def run_auto_trade():
    """
    1) upbit.get_balance()로 원화/코인 잔고 조회
    2) upbit.get_balances()로 평균 매수가 조회
    3) analyze_market() 호출 → 시그널
    4) 시그널에 따라 분할 매수(3회) 또는 분할 매도(2회)
    5) record_trade()로 DB 저장 + Telegram 알림
    """
    for coin in COINS:
        try:
            # ─── (1) 원화 잔고, 코인 잔고 조회 ───
            krw = upbit.get_balance("KRW")
            coin_balance = upbit.get_balance(coin)  # "BTC", "ETH", "XRP" 심볼만

            # ─── (2) 현재가, 평균 매수가 조회 ───
            price = pyupbit.get_current_price(f"KRW-{coin}") or 1

            avg_price = 0.0
            balances = upbit.get_balances()
            for b in balances:
                if b.get("currency") == coin and b.get("avg_buy_price"):
                    avg_price = float(b["avg_buy_price"])
                    break

            # ─── (3) 시장 분석 → 시그널 생성 ───
            signal = analyze_market(coin)
            ratio = signal["percentage"] / 100.0
            parts = 3
            reaction = ""

            # ─── (4) 분할 매수 로직 ───
            if signal["decision"] == "buy" and krw is not None and krw * ratio > 5000:
                unit = (krw * ratio) / parts
                for i in range(parts):
                    if IS_LIVE:
                        upbit.buy_market_order(f"KRW-{coin}", unit)
                    send_telegram(f"💸 [{coin}] {i+1}차 매수 - {unit:,.0f}원")
                    time.sleep(1)
                reaction = f"{parts}회 분할매수 실행"

            # ─── (5) 분할 매도 로직 ───
            elif signal["decision"] == "sell" and coin_balance is not None and coin_balance > 0:
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

            # ─── (6) 거래 기록 저장 및 완료 메시지 전송 ───
            record_trade(
                coin=coin,
                decision=signal["decision"],
                percentage=signal["percentage"],
                confidence_score=signal["confidence_score"],
                reason=signal["reason"],
                reaction=reaction,
                coin_balance=coin_balance if coin_balance is not None else 0,
                krw_balance=krw if krw is not None else 0,
                avg_price=avg_price,
                coin_price=price
            )
            send_telegram(
                f"✅ [{coin}] 거래 기록 완료: {signal['decision'].upper()} | "
                f"신뢰도 {signal['confidence_score']}%\n사유: {signal['reason']}"
            )
            time.sleep(2)

        except Exception as e:
            send_telegram(f"❌ [{coin}] 오류 발생: {e}")

def run_scheduler():
    """
    매일 09:00, 15:00에 run_auto_trade() 실행
    """
    schedule.every().day.at("09:00").do(run_auto_trade)
    schedule.every().day.at("15:00").do(run_auto_trade)
    send_telegram("✅ 자동매매 스케줄 시작됨 (09시 / 15시)")
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    run_scheduler()



