import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 공포탐욕지수 API
def get_fear_greed_index():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1")
        data = res.json()
        return int(data['data'][0]['value'])
    except:
        return 50

# ✅ 텔레그램 전송 함수
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("텔레그램 오류:", e)

# ✅ 뉴스 수집 (다국어 6개 소스 + 링크 포함)
def fetch_all_news(coin):
    news_entries = []
    headers = {"User-Agent": "Mozilla/5.0"}

    coin_keywords = {
        "BTC": ["btc", "bitcoin", "비트코인"],
        "ETH": ["eth", "ethereum", "이더리움"],
        "XRP": ["xrp", "ripple", "리플"]
    }
    keywords = coin_keywords.get(coin.upper(), [coin])

    for keyword in keywords:
        # 1. 네이버 뉴스
        try:
            url = f"https://search.naver.com/search.naver?where=news&query={keyword}+암호화폐"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            links = soup.select(".news_tit")
            for link in links[:2]:
                news_entries.append(f"{link.text.strip()} - {link['href']}")
        except Exception as e:
            print("네이버 오류:", e)

        # 2. Google News
        try:
            url = f"https://news.google.com/search?q={keyword}+crypto&hl=en-US&gl=US&ceid=US:en"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select("article h3 a")
            for a in articles[:2]:
                title = a.text.strip()
                href = "https://news.google.com" + (a['href'][1:] if a['href'].startswith('.') else a['href'])
                news_entries.append(f"{title} - {href}")
        except Exception as e:
            print("Google 오류:", e)

        # 3. CoinDesk
        try:
            res = requests.get("https://www.coindesk.com/", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select("a.card-title")
            for a in articles:
                if keyword.lower() in a.text.lower():
                    news_entries.append(f"{a.text.strip()} - https://www.coindesk.com{a['href']}")
            news_entries = news_entries[:2]
        except Exception as e:
            print("CoinDesk 오류:", e)

        # 4. Cointelegraph
        try:
            res = requests.get("https://cointelegraph.com/", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select("a.post-card-inline__title-link")
            for a in articles:
                if keyword.lower() in a.text.lower():
                    news_entries.append(f"{a.text.strip()} - https://cointelegraph.com{a['href']}")
            news_entries = news_entries[:2]
        except Exception as e:
            print("Cointelegraph 오류:", e)

        # 5. Yahoo Finance
        try:
            url = f"https://finance.yahoo.com/quote/{keyword.upper()}-USD/news"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select("h3 a")
            for a in articles[:2]:
                news_entries.append(f"{a.text.strip()} - https://finance.yahoo.com{a['href']}")
        except Exception as e:
            print("Yahoo 오류:", e)

        # 6. Binance Blog
        try:
            res = requests.get("https://www.binance.com/en/blog", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select("a.css-1ej4hfo")
            for a in articles:
                if keyword.lower() in a.text.lower():
                    news_entries.append(f"{a.text.strip()} - https://www.binance.com{a['href']}")
            news_entries = news_entries[:2]
        except Exception as e:
            print("Binance 오류:", e)

    return news_entries if news_entries else ["뉴스 없음 (평가 불가)"]

# ✅ GPT 뉴스 평가
def evaluate_news(articles):
    if not articles or "뉴스 없음" in articles[0]:
        return "뉴스 없음 (평가 불가)"
    prompt = "다음 뉴스 제목들을 참고해 암호화폐 시장의 심리를 요약하고, '매수', '매도', '보류' 중 하나로 판단해줘:\n"
    prompt += "\n".join(articles)
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
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("뉴스 평가 오류:", e)
        return "뉴스 평가 실패"

# ✅ 거래 로그 저장 함수
def log_trade(coin, signal, coin_balance, krw_balance, avg_price, now_price):
    try:
        with open("trade_log.txt", "a", encoding="utf-8") as f:
            f.write(
                f"[{coin}] {signal['decision']} | 신뢰도:{signal['confidence_score']}% | "
                f"보유:{coin_balance:.4f}개, 원화:{krw_balance:,.0f}, 평균가:{avg_price:,.0f}, 현재가:{now_price:,.0f}\n"
            )
    except Exception as e:
        print("거래 로그 저장 오류:", e)
