# main.py

import schedule
import time
from core.strategy import analyze_coin
from core.trade_engine import execute_trading_decision
from utils import send_telegram

COINS = ["XRP", "ADA",]  # 현재 설정된 2개 코인

def analyze_only():
    for coin in COINS:
        try:
            signal = analyze_coin(coin)
            send_telegram(f"🔎 [{coin}] 판단 결과: {signal['decision']} | 신뢰도: {signal['confidence_score']}%\n사유: {signal['reason']}")
        except Exception as e:
            send_telegram(f"❌ [{coin}] 판단 오류: {e}")

def run():
    for coin in COINS:
        try:
            signal = analyze_coin(coin)
            execute_trading_decision(coin, signal)
            time.sleep(1)
        except Exception as e:
            send_telegram(f"❌ [{coin}] 매매 오류: {e}")

if __name__ == "__main__":
    # 판단만 먼저 수행
    schedule.every().day.at("08:30").do(analyze_only)
    schedule.every().day.at("14:30").do(analyze_only)
    schedule.every().day.at("21:00").do(analyze_only)
    schedule.every().day.at("00:30").do(analyze_only)

    # 30분 후 실제 매매 실행
    schedule.every().day.at("09:00").do(run)
    schedule.every().day.at("15:00").do(run)
    schedule.every().day.at("21:30").do(run)
    schedule.every().day.at("01:00").do(run)

    send_telegram("✅ AI 자동매매 스케줄러 시작됨 (전략: 4회, 매매: 4회)")
    while True:
        schedule.run_pending()
        time.sleep(10)
