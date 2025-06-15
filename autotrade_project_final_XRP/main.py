# main.py

import schedule
import time
from core.strategy import analyze_coin, analyze_japan, analyze_us
from core.trade_engine import execute_trading_decision
from utils import send_telegram

COINS = ["XRP", "ADA"]
JPN_STOCKS = ["7203.T", "6758.T"]  # 예: 도요타, 소니
US_STOCKS = ["AAPL", "MSFT"]       # 예: 애플, 마이크로소프트

# ----- 판단 함수 -----
def analyze_only_coin():
    for coin in COINS:
        try:
            signal = analyze_coin(coin)
            send_telegram(f"🔎 [코인 {coin}] 판단 결과: {signal['decision']} | 신뢰도: {signal['confidence_score']}%\n사유: {signal['reason']}")
        except Exception as e:
            send_telegram(f"❌ [코인 {coin}] 판단 오류: {e}")

def analyze_only_japan():
    for stock in JPN_STOCKS:
        try:
            signal = analyze_japan(stock)
            send_telegram(f"🔎 [일본 {stock}] 판단 결과: {signal['decision']} | 신뢰도: {signal['confidence_score']}%\n사유: {signal['reason']}")
        except Exception as e:
            send_telegram(f"❌ [일본 {stock}] 판단 오류: {e}")

def analyze_only_us():
    for stock in US_STOCKS:
        try:
            signal = analyze_us(stock)
            send_telegram(f"🔎 [미국 {stock}] 판단 결과: {signal['decision']} | 신뢰도: {signal['confidence_score']}%\n사유: {signal['reason']}")
        except Exception as e:
            send_telegram(f"❌ [미국 {stock}] 판단 오류: {e}")

# ----- 매매 함수 -----
def run_coin():
    for coin in COINS:
        try:
            signal = analyze_coin(coin)
            execute_trading_decision(coin, signal)
            time.sleep(1)
        except Exception as e:
            send_telegram(f"❌ [코인 {coin}] 매매 오류: {e}")

def run_japan():
    for stock in JPN_STOCKS:
        try:
            signal = analyze_japan(stock)
            execute_trading_decision(stock, signal)
            time.sleep(1)
        except Exception as e:
            send_telegram(f"❌ [일본 {stock}] 매매 오류: {e}")

def run_us():
    for stock in US_STOCKS:
        try:
            signal = analyze_us(stock)
            execute_trading_decision(stock, signal)
            time.sleep(1)
        except Exception as e:
            send_telegram(f"❌ [미국 {stock}] 매매 오류: {e}")

# ----- 스케줄러 -----
if __name__ == "__main__":
    # 코인
    schedule.every().day.at("08:30").do(analyze_only_coin)
    schedule.every().day.at("14:30").do(analyze_only_coin)
    schedule.every().day.at("21:00").do(analyze_only_coin)
    schedule.every().day.at("00:30").do(analyze_only_coin)

    schedule.every().day.at("09:00").do(run_coin)
    schedule.every().day.at("15:00").do(run_coin)
    schedule.every().day.at("21:30").do(run_coin)
    schedule.every().day.at("01:00").do(run_coin)

    # 일본
    schedule.every().day.at("09:30").do(analyze_only_japan)
    schedule.every().day.at("13:30").do(analyze_only_japan)
    schedule.every().day.at("10:00").do(run_japan)
    schedule.every().day.at("14:00").do(run_japan)

    # 미국
    schedule.every().day.at("22:30").do(analyze_only_us)
    schedule.every().day.at("02:30").do(analyze_only_us)
    schedule.every().day.at("23:00").do(run_us)
    schedule.every().day.at("03:00").do(run_us)

    send_telegram("✅ AI 자동매매 스케줄러 시작됨 (코인/일본/미국 통합)")
    
    while True:
        schedule.run_pending()
        time.sleep(10)
