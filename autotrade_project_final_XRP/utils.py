import os
import requests
from bs4 import BeautifulSoup
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

def fetch_all_news(coin):
    news_titles = []
    headers = {"User-Agent": "Mozilla/5.0"}
    keywords = {
        "BTC": ["btc", "bitcoin", "비트코인"],
        "ETH": ["eth", "ethereum", "이더리움"],
    }.get(coin.upper(), [coin])

    for kw in keywords:
        try:
            url = f"https://search.naver.com/search.naver?where=news&query={kw}+암호화폐"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            news_titles += [a.text for a in soup.select(".news_tit")[:2]]
        except:
            continue
    return news_titles if news_titles else ["뉴스 없음 (평가 불가)"]

def evaluate_news(articles):
    if not articles:
        return "뉴스 없음"
    prompt = f"다음 뉴스 제목으로 시황을 요약하고 매수/매도/보류 중 하나를 판단해줘:\n{articles}"
    try:
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return res.json()['choices'][0]['message']['content']
    except:
        return "뉴스 평가 실패"

def log_trade(coin, signal, coin_balance, krw_balance, avg_price, now_price):
    try:
        with open("trade_log.txt", "a", encoding="utf-8") as f:
            f.write(
                f"[{coin}] {signal['decision']} | 신뢰도:{signal['confidence_score']}% | "
                f"코인:{coin_balance:.4f}, 원화:{krw_balance:,.0f}, 평균가:{avg_price:,.0f}, 현재가:{now_price:,.0f}\n"
            )
    except Exception as e:
        print("📛 거래 로그 저장 오류:", e)

        
