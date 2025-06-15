import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import datetime
import holidays
import pyupbit

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
kr_holidays = holidays.KR()

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print(f"❌ 텔레그램 오류: {e}")

def get_fear_greed_index():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1")
        return int(res.json()['data'][0]['value'])
    except:
        send_telegram("❌ FG 지수 가져오기 실패")
        return 50

def fetch_all_news(asset):
    headers = {"User-Agent": "Mozilla/5.0"}
    keywords_map = {
        "XRP": ["XRP", "엑스알피", "리플"],
        "ADA": ["ADA", "카르다노", "에이다"],
        "7203.T": ["도요타", "トヨタ", "トヨタ自動車"],
        "6758.T": ["소니", "ソニー"],
        "AAPL": ["애플", "Apple"],
        "MSFT": ["마이크로소프트", "Microsoft"]
    }
    keywords = keywords_map.get(asset.upper(), [asset])
    news = []
    for kw in keywords:
        try:
            url = f"https://search.naver.com/search.naver?where=news&query={kw}"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            for a in soup.select(".news_tit")[:2]:
                title = a.text
                link = a["href"]
                content = "본문 실패"
                try:
                    art = requests.get(link, headers=headers, timeout=5)
                    art_soup = BeautifulSoup(art.text, "html.parser")
                    paragraphs = [p.text for p in art_soup.select("p")]
                    content = " ".join(paragraphs)[:500]
                except:
                    pass
                news.append({"title": title, "content": content})
        except Exception as e:
            send_telegram(f"❌ 뉴스 크롤링 실패: {e}")
    return news

def evaluate_news(news):
    if not news:
        return "뉴스 없음"
    prompt = "\n".join([f"{n['title']} 본문: {n['content']}" for n in news])
    prompt = f"다음 뉴스 내용을 요약하고 긍정/부정/중립 평가:\n{prompt}"
    try:
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
        )
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        send_telegram(f"❌ 뉴스 평가 실패: {e}")
        return "평가 실패"

def is_holiday_or_weekend():
    today = datetime.date.today()
    return today.weekday() >= 5 or today in kr_holidays

def get_price(asset, asset_type):
    if asset_type == "coin":
        return pyupbit.get_current_price(f"KRW-{asset}") or 100000
    else:
        return 100000  # IBKR 연동 필요

        
