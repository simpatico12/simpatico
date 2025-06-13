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
        print(f"텔레그램 전송 오류: {e}")

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

    coin_keywords = {
        "BTC": ["btc", "bitcoin", "비트코인"],
        "ETH": ["eth", "ethereum", "이더리움"],
        "XRP": ["xrp", "ripple", "리플"]
    }
    keywords = coin_keywords.get(coin.upper(), [coin])

    for keyword in keywords:
        try:
            url = f"https://search.naver.com/search.naver?where=news&query={keyword}+암호화폐"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            links = soup.select(".news_tit")
            news_titles += [link.text for link in links[:2]]
        except:
            pass

        try:
            url = f"https://news.google.com/search?q={keyword}+crypto&hl=en-US&gl=US&ceid=US:en"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select("article h3")
            news_titles += [a.text.strip() for a in articles[:2]]
        except:
            pass

        try:
            res = requests.get("https://www.coindesk.com/", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select("h4.heading")
            coin_related = [a for a in articles if keyword.lower() in a.text.lower()]
            news_titles += [a.text.strip() for a in coin_related[:2]]
        except:
            pass

        try:
            res = requests.get("https://cointelegraph.com/", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select("span.post-card-inline__title")
            coin_related = [a for a in articles if keyword.lower() in a.text.lower()]
            news_titles += [a.text.strip() for a in coin_related[:2]]
        except:
            pass

        try:
            url = f"https://finance.yahoo.com/quote/{keyword.upper()}-USD/news"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select("h3")
            news_titles += [a.text.strip() for a in articles[:2]]
        except:
            pass

        try:
            res = requests.get("https://www.binance.com/en/blog", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select(".css-1ej4hfo h6")
            coin_related = [a for a in articles if keyword.lower() in a.text.lower()]
            news_titles += [a.text.strip() for a in coin_related[:2]]
        except:
            pass

    return news_titles if news_titles else ["뉴스 없음 (평가 불가)"]

def evaluate_news(articles):
    if not articles:
        return "뉴스 없음"
    prompt = f"다음 뉴스 제목을 바탕으로 시황을 요약하고, 매수/매도/보류 중 하나로 판단해줘:\n{articles}"
    try:
        response = requests.post(
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
        return response.json()['choices'][0]['message']['content']
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

        
