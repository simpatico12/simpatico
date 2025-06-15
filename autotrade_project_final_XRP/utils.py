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

def fetch_all_news(asset):
    news_titles = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 자산 종류별 키워드 매핑
    keywords_map = {
        "XRP": ["XRP", "엑스알피", "리플"],
        "ADA": ["ADA", "카르다노", "에이다"],
        "7203.T": ["도요타", "トヨタ", "トヨタ自動車"],
        "6758.T": ["소니", "ソニー"],
        "AAPL": ["애플", "Apple"],
        "MSFT": ["마이크로소프트", "Microsoft"]
    }
    
    keywords = keywords_map.get(asset.upper(), [asset])

    for kw in keywords:
        try:
            url = f"https://search.naver.com/search.naver?where=news&query={kw}"
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

def log_trade(asset, signal, balance_info, now_price):
    try:
        with open("trade_log.txt", "a", encoding="utf-8") as f:
            f.write(
                f"[{asset}] {signal['decision']} | 신뢰도:{signal['confidence_score']}% | "
                f"자산잔고:{balance_info.get('asset_balance', 0):,.4f}, "
                f"현금:{balance_info.get('cash_balance', 0):,.0f}, "
                f"평균가:{balance_info.get('avg_price', 0):,.0f}, "
                f"현재가:{now_price:,.0f}\n"
            )
    except Exception as e:
        print("📛 거래 로그 저장 오류:", e)


        
