import requests
import datetime
import holidays
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
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
        print(f"❌ 텔레그램 전송 오류: {e}")

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
        "7203.T": ["도요타", "トヨタ"],
        "6758.T": ["소니", "ソニー"],
        "AAPL": ["Apple", "애플"],
        "MSFT": ["Microsoft", "마이크로소프트"]
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
        except:
            send_telegram(f"❌ 뉴스 크롤링 실패: {kw}")
    return news

def evaluate_news(news):
    prompt = "\n".join([f"{n['title']} 본문: {n['content']}" for n in news])
    prompt = f"다음 뉴스 내용을 요약하고 긍정/부정/중립 평가:\n{prompt}"
    try:
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
        )
        return res.json()['choices'][0]['message']['content']
    except:
        return "평가 실패"

def is_holiday_or_weekend():
    today = datetime.date.today()
    return today.weekday() >= 5 or today in kr_holidays

def get_price(asset, asset_type):
    if asset_type == "coin":
        return pyupbit.get_current_price(f"KRW-{asset}") or 100000
    else:
        return 100000  # IBKR API 연동 시 실가격 적용

def get_total_asset_value(upbit):
    krw = upbit.get_balance("KRW")
    balances = upbit.get_balances()
    total = krw
    for b in balances:
        if b['currency'] != "KRW":
            price = pyupbit.get_current_price(f"KRW-{b['currency']}") or 0
            total += float(b['balance']) * price
    return total

def get_cash_balance(upbit):
    return upbit.get_balance("KRW")

def log_trade(asset, signal, balance_info, now_price):
    try:
        with open("trade_log.txt", "a", encoding="utf-8") as f:
            f.write(
                f"[{asset}] {signal['decision']} | 신뢰도:{signal['confidence_score']}% | "
                f"자산잔고:{balance_info.get('asset_balance', 0):,.4f}, "
                f"현금:{balance_info.get('cash_balance', 0):,.0f}, "
                f"평균가:{balance_info.get('avg_price', 0):,.0f}, "
                f"현재가:{now_price:,.0f}, "
                f"총자산:{balance_info.get('total_asset', 0):,.0f}\n"
            )
    except Exception as e:
        send_telegram(f"📛 거래 로그 저장 오류: {e}")



        
