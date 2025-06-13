import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

def get_fear_greed_index():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1")
        data = res.json()
        return int(data['data'][0]['value'])
    except:
        return 50

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
        try:
            url = f"https://search.naver.com/search.naver?where=news&query={keyword}+암호화폐"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.select(".news_tit")
            for item in items[:2]:
                title = item.text
                link = item.get("href", "")
                news_entries.append(f"{title} - {link}")
        except Exception as e:
            print("네이버 오류:", e)

        try:
            url = f"https://news.google.com/search?q={keyword}+crypto&hl=en-US&gl=US&ceid=US:en"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.select("article h3 a")
            for item in items[:2]:
                title = item.text.strip()
                link = "https://news.google.com" + item.get("href", "")
                news_entries.append(f"{title} - {link}")
        except Exception as e:
            print("Google 오류:", e)

        try:
            res = requests.get("https://www.coindesk.com/", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.select("h4.heading")
            for item in items:
                if keyword.lower() in item.text.lower():
                    title = item.text.strip()
                    parent = item.find_parent("a")
                    link = parent["href"] if parent else ""
                    news_entries.append(f"{title} - {link}")
        except Exception as e:
            print("CoinDesk 오류:", e)

        try:
            res = requests.get("https://cointelegraph.com/", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.select("span.post-card-inline__title")
            for item in items:
                if keyword.lower() in item.text.lower():
                    title = item.text.strip()
                    parent = item.find_parent("a")
                    link = "https://cointelegraph.com" + parent["href"] if parent else ""
                    news_entries.append(f"{title} - {link}")
        except Exception as e:
            print("Cointelegraph 오류:", e)

        try:
            url = f"https://finance.yahoo.com/quote/{keyword.upper()}-USD/news"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.select("h3")
            for item in items[:2]:
                title = item.text.strip()
                link_tag = item.find("a")
                link = "https://finance.yahoo.com" + link_tag.get("href", "") if link_tag else ""
                news_entries.append(f"{title} - {link}")
        except Exception as e:
            print("Yahoo 오류:", e)

        try:
            res = requests.get("https://www.binance.com/en/blog", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.select(".css-1ej4hfo h6")
            for item in items:
                if keyword.lower() in item.text.lower():
                    title = item.text.strip()
                    parent = item.find_parent("a")
                    link = "https://www.binance.com" + parent["href"] if parent else ""
                    news_entries.append(f"{title} - {link}")
        except Exception as e:
            print("Binance 오류:", e)

    return news_entries if news_entries else ["뉴스 없음 (평가 불가)"]

def evaluate_news(articles):
    if not articles:
        return "뉴스 없음"
    prompt = f"다음은 암호화폐 관련 뉴스입니다. 각 제목과 링크를 참고하여 시장 심리를 분석하고 요약한 뒤, '매수/매도/보류' 중 하나로 판단해줘:\\n\\n" + "\\n".join(articles)
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
