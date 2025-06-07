
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
        url = f"https://openapi.naver.com/v1/search/news.json?query={coin}&display=3&sort=sim"
        headers = {
            "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
            "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
        }
        res = requests.get(url, headers=headers)
        items = res.json().get("items", [])
        return [item["title"] for item in items]
    except:
        return []

def evaluate_news(articles):
    prompt = f"다음 뉴스 제목들을 바탕으로 시황을 요약하고 매수/매도/보류 중 하나로 판단해줘:
{articles}"
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
