#!/bin/bash

# 디렉토리 생성
mkdir -p ~/autotrade/core
cd ~/autotrade || exit

# main.py 생성
cat <<EOF > main.py
import schedule
import time
from core.strategy import analyze_coin
from core.trade_engine import execute_trading_decision
from utils import send_telegram

COINS = ["BTC", "ETH", "XRP", "SOL", "ADA", "DOGE", "AVAX", "TRX", "DOT", "MATIC"]

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
    send_telegram("✅ AI 자동매번 스케줄러 시작 (08:30 / 09:00 / 14:30 / 15:00)")
    while True:
        schedule.run_pending()
        time.sleep(10)
EOF

# core/strategy.py 생성
cat <<EOF > core/strategy.py
import os
import openai
import requests
from utils import get_fear_greed_index, fetch_news, evaluate_news
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def get_news_sentiment(coin):
    articles = fetch_news(coin)
    summary = evaluate_news(articles)
    return summary

def strategy_buffett():
    return "hold"

def strategy_jesse():
    return "buy"

def strategy_wonyo():
    return "buy"

def strategy_jim_rogers():
    return "buy"

def analyze_coin(coin):
    sentiment = get_news_sentiment(coin)
    fg_index = get_fear_greed_index()

    votes = [strategy_buffett(), strategy_jesse(), strategy_wonyo(), strategy_jim_rogers()]
    vote_result = max(set(votes), key=votes.count)

    decision = vote_result if fg_index <= 60 else "보률"
    confidence = 85 if vote_result == "buy" else 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 30,
        "reason": f"FG지수:{fg_index} | 뉴스요약:{sentiment}"
    }
EOF

# core/trade_engine.py 생성
cat <<EOF > core/trade_engine.py
import pyupbit
import time
from utils import send_telegram, log_trade
import os
from dotenv import load_dotenv

load_dotenv()
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)

IS_LIVE = True
MAX_COIN_RATIO = 0.3
ALLOWED_RATIO = 0.7

def execute_trading_decision(coin, signal):
    ticker = f"KRW-{coin}"
    balance_krw = upbit.get_balance("KRW")
    balances = upbit.get_balances()
    coin_data = next((b for b in balances if b['currency'] == coin), {})
    coin_balance = float(coin_data.get("balance", 0))
    avg_price = float(coin_data.get("avg_buy_price", 0))
    now_price = pyupbit.get_current_price(ticker) or 1

    total_asset = balance_krw + coin_balance * now_price
    coin_value_ratio = (coin_balance * now_price) / total_asset if total_asset > 0 else 0

    ratio = signal["percentage"] / 100

    if signal["confidence_score"] < 70:
        send_telegram(f"⛔ 신리도 낮음({signal['confidence_score']}%), {coin} 매수 보률")
        return

    if signal["decision"] == "buy" and balance_krw * ratio > 5000:
        if coin_value_ratio > MAX_COIN_RATIO:
            send_telegram(f"🚫 {coin} 보유 비중 초가로 매수 보률")
            return
        if (balance_krw / total_asset) > ALLOWED_RATIO:
            send_telegram(f"💡 차삭 70% 초가 매수 보률")
            return
        unit = (balance_krw * ratio) / 3
        for i in range(3):
            if IS_LIVE:
                upbit.buy_market_order(ticker, unit)
            send_telegram(f"💸 [{coin}] {i+1}차 분할매수 - {unit:,.0f}원")
            time.sleep(1)

    elif signal["decision"] == "sell" and coin_balance > 0:
        profit_rate = (now_price - avg_price) / avg_price
        if profit_rate >= 0.05:
            sell_qty = coin_balance * 0.5
            for i in range(2):
                if IS_LIVE:
                    upbit.sell_market_order(ticker, sell_qty / 2)
                send_telegram(f"📈 익절 [{coin}] {i+1}차 매동 - {sell_qty/2:.6f}개")
                time.sleep(1)
        elif profit_rate <= -0.03:
            if IS_LIVE:
                upbit.sell_market_order(ticker, coin_balance)
            send_telegram(f"🛑 손절 [{coin}] 전체 매동 - {coin_balance:.6f}개")
        else:
            send_telegram(f"⏸️ {coin} 매동 보률 (익절/손절 조건 불충당)")

    log_trade(coin, signal, coin_balance, balance_krw, avg_price, now_price)
EOF

# utils.py 생성
cat <<EOF > utils.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram error:", e)


def get_fear_greed_index():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1")
        data = res.json()
        return int(data['data'][0]['value'])
    except:
        return 50


def fetch_news(coin):
    try:
        url = f"https://serpapi.com/search.json?q={coin}+crypto+news&hl=ko&gl=kr&api_key={os.getenv('SERPAPI_API_KEY')}"
        res = requests.get(url)
        results = res.json().get("news_results", [])
        return [r['title'] for r in results[:3]]
    except:
        return []


def evaluate_news(articles):
    prompt = f"\ub2e4\uc74c \ub274\uc2a4 \uc81c목\ub4e4\uc744 \ubc14\ud0d5\uc73c\ub85c \uc2dc\ud669\uc744 \uc694약\ud558고 \ub9e4수/\ub9e4도/\ubcf4\ub960 \uc911 \ud558나\ub85c \ud310단\ud574줘:\n{articles}"
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return response.json()['choices'][0]['message']['content']
    except:
        return "뉴스 평가 실패"


def log_trade(coin, signal, coin_balance, krw_balance, avg_price, now_price):
    try:
        with open("trade_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{coin}] {signal['decision']} | 신리도:{signal['confidence_score']}% | 코인:{coin_balance:.4f}, 원화:{krw_balance:,.0f}, 평균가:{avg_price:.0f}, 현재가:{now_price:.0f}\n")
    except Exception as e:
        print("로그 저장 실패:", e)
EOF

# .env 템플릿 생성
cat <<EOF > .env
UPBIT_ACCESS_KEY=your_access_key
UPBIT_SECRET_KEY=your_secret_key
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
OPENAI_API_KEY=your_openai_key
SERPAPI_API_KEY=your_serpapi_key
EOF

# requirements.txt 생성
cat <<EOF > requirements.txt
pyupbit
requests
schedule
openai
python-dotenv
EOF

# 필요한 패키지 설치
pip install -r requirements.txt

# 완료 메시지
echo "✅ Autotrade AI 설치 완료! .env 파일을 첫 초부 설정해주세요."

