import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# 다국어 뉴스 소스 통합: 네이버(한글), Google News(영문), CoinDesk, Cointelegraph 등 포함

def fetch_all_news(coin):
    news_titles = []

    # 1. 네이버 뉴스 (한글)
    try:
        naver_url = f"https://search.naver.com/search.naver?where=news&query={coin}+%EC%95%94%ED%98%B8%ED%99%94%ED%8F%90"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(naver_url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.select(".news_tit")
        news_titles += [link.text for link in links[:3]]
    except Exception as e:
        print("네이버 뉴스 오류:", e)

    # 2. Google News (영문)
    try:
        google_url = f"https://news.google.com/search?q={coin}+crypto&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(google_url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.select("article h3")
        news_titles += [a.text.strip() for a in articles[:3]]
    except Exception as e:
        print("Google News 오류:", e)

    # 3. CoinDesk
    try:
        cd_url = "https://www.coindesk.com/"
        res = requests.get(cd_url)
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.select("h4.heading")
        coin_related = [a for a in articles if coin.lower() in a.text.lower()]
        news_titles += [a.text.strip() for a in coin_related[:3]]
    except Exception as e:
        print("CoinDesk 오류:", e)

    # 4. Cointelegraph
    try:
        ct_url = "https://cointelegraph.com/"
        res = requests.get(ct_url)
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.select("span.post-card-inline__title")
        coin_related = [a for a in articles if coin.lower() in a.text.lower()]
        news_titles += [a.text.strip() for a in coin_related[:3]]
    except Exception as e:
        print("Cointelegraph 오류:", e)

    return news_titles if news_titles else ["뉴스 없음 (평가 불가)"]
