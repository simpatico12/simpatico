import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

    # 1. 네이버 뉴스
    try:
        naver_url = f"https://search.naver.com/search.naver?where=news&query={coin}+코인"
        res = requests.get(naver_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.select("a.news_tit")
        news_titles += [link.text for link in links[:3]]
    except:
        pass

    # 2. 코인마켓캡 뉴스
    try:
        cmc_url = "https://coinmarketcap.com/headlines/news/"
        res = requests.get(cmc_url)
        soup = BeautifulSoup(res.text, "html.parser")
        items = soup.select("a[class^='svowul-5']")
        news_titles += [item.text.strip() for item in items[:3]]
    except:
        pass

    # 3. 바이낸스 공지
    try:
        binance_url = "https://www.binance.com/en/support/announcement"
        res = requests.get(binance_url)
        soup = BeautifulSoup(res.text, "html.parser")
        items = soup.select("a.css-1ej4hfo")  # class는 수시로 바뀔 수 있음
        news_titles += [item.text.strip() for item in items[:3]]
    except:
        pass

    # 4. 코인데스크
    try:
        cd_url = "https://www.coindesk.com/"
        res = requests.get(cd_url)
        soup = BeautifulSoup(res.text, "html.parser")
        items = soup.select("h4.heading")
        news_titles += [item.text.strip() for item in items[:3]]
    except:
        pass

    # 5. 코인텔레그래프
    try:
        ct_url = "https://cointelegraph.com/"
        res = requests.get(ct_url)
        soup = BeautifulSoup(res.text, "html.parser")
        items = soup.select("span.post-card-inline__title")
        news_titles += [item.text.strip() for item in items[:3]]
    except:
        pass

    # 6. 업비트 공지
    try:
        upbit_url = "https://upbit.com/service_center/notice"
        res = requests.get(upbit_url)
        soup = BeautifulSoup(res.text, "html.parser")
        items = soup.select("div.css-1x9bshx")  # 구조가 바뀌면 class 확인
        news_titles += [item.text.strip() for item in items[:3]]
    except:
        pass

    return news_titles if news_titles else ["뉴스 없음 (평가 불가)"]

def evaluate_news(articles):
    prompt = f"다음 뉴스 제목들을 보고 요약하고 매수/매도/보류 중 판단:\n{articles}"
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
        print("GPT 뉴스 요약 오류:", e)
        return "뉴스 평가 실패"

def log_trade(coin, signal, coin_balance, krw_balance, avg_price, now_price):
    try:
        with open("trade_log.txt", "a", encoding="utf-8") as f:
            f.write(
                f"[{coin}] {signal['decision']} | 신뢰도:{signal['confidence_score']}% | "
                f"코인:{coin_balance:.4f}, 원화:{krw_balance:,.0f}, 평균가:{avg_price:,.0f}, 현재가:{now_price:,.0f}\n"
            )
    except Exception as e:
        print("거래 로그 저장 오류:", e)
