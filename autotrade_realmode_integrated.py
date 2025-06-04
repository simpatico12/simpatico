# main.py

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
    send_telegram("✅ AI 자동매매 스케줄러 시작 (08:30 / 09:00 / 14:30 / 15:00)")
    while True:
        schedule.run_pending()
        time.sleep(10)

# core/strategy.py

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

    decision = vote_result if fg_index <= 60 else "보류"
    confidence = 85 if vote_result == "buy" else 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 30,
        "reason": f"FG지수:{fg_index} | 뉴스요약:{sentiment}"
    }

# core/trade_engine.py

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
        send_telegram(f"⛔ 신뢰도 낮음({signal['confidence_score']}%), {coin} 매수 보류")
        return

    if signal["decision"] == "buy" and balance_krw * ratio > 5000:
        if coin_value_ratio > MAX_COIN_RATIO:
            send_telegram(f"🚫 {coin} 보유 비중 초과로 매수 보류")
            return
        if (balance_krw / total_asset) > ALLOWED_RATIO:
            send_telegram(f"💡 총 자산 중 70% 초과 사용 방지로 매수 보류")
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
                send_telegram(f"📈 익절 [{coin}] {i+1}차 매도 - {sell_qty/2:.6f}개")
                time.sleep(1)
        elif profit_rate <= -0.03:
            if IS_LIVE:
                upbit.sell_market_order(ticker, coin_balance)
            send_telegram(f"🛑 손절 [{coin}] 전체 매도 - {coin_balance:.6f}개")
        else:
            send_telegram(f"⏸️ {coin} 매도 보류 (익절/손절 조건 불충분)")

    log_trade(coin, signal, coin_balance, balance_krw, avg_price, now_price)

# utils.py

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
        print("텔레그램 오류:", e)

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
    prompt = f"다음 뉴스 제목들을 바탕으로 시황을 요약하고 매수/매도/보류 중 하나로 판단해줘:\n{articles}"
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
            f.write(f"[{coin}] {signal['decision']} | 신뢰도:{signal['confidence_score']}% | 코인:{coin_balance:.4f}, 원화:{krw_balance:,.0f}, 평균가:{avg_price:.0f}, 현재가:{now_price:.0f}\n")
    except Exception as e:
        print("로그 저장 실패:", e)
