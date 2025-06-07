import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 텔레그램 전송
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("텔레그램 오류:", e)

# 공포탐욕지수
def get_fear_greed_index():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1")
        data = res.json()
        return int(data['data'][0]['value'])
    except:
        return 50

# 네이버 뉴스 크롤링
def fetch_news(coin):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://search.naver.com/search.naver?where=news&query={coin}+코인"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        news_titles = soup.select("a.news_tit")
        return [title.text for title in news_titles[:3]]
    except:
        return []

# GPT 뉴스 평가
def evaluate_news(articles):
    if not articles:
        return "뉴스 없음 (평가 불가)"

    prompt = f"다음 뉴스 제목들을 보고 시장 상황을 요약하고, '매수', '매도', '보류' 중 하나로 판단해줘:\n{articles}"

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return response.json()['choices'][0]['message']['content']
    except:
        return "뉴스 평가 실패"

# 로그 저장
def log_trade(coin, signal, coin_balance, krw_balance, avg_price, now_price):
    try:
        with open("trade_log.txt", "a", encoding="utf-8") as f:
            f.write(
                f"[{coin}] {signal['decision']} | 신뢰도:{signal['confidence_score']}% | "
                f"코인:{coin_balance:.4f}, 원화:{krw_balance:,.0f}, 평균가:{avg_price:.0f}, 현재가:{now_price:.0f}\n"
            )
    except Exception as e:
        print("로그 저장 실패:", e)

