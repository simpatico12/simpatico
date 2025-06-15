import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❗ TELEGRAM_TOKEN 또는 TELEGRAM_CHAT_ID 값이 .env에 없습니다.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

        if response.status_code != 200:
            print(f"❗ 텔레그램 전송 실패 - 코드: {response.status_code}, 응답: {response.text}")
        else:
            print(f"✅ 텔레그램 전송 성공: {msg[:30]}...")
    except Exception as e:
        print(f"❌ 텔레그램 전송 중 예외 발생: {e}")

def get_fear_greed_index():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1")
        data = res.json()
        return int(data['data'][0]['value'])
    except Exception as e:
        print(f"❌ FG 지수 가져오기 실패: {e}")
        send_telegram(f"❌ FG 지수 가져오기 실패: {e}")
        return 50

def fetch_all_news(asset):
    news_titles = []
    headers = {"User-Agent": "Mozilla/5.0"}

    keywords_map = {
        # 코인
        "XRP": ["XRP", "엑스알피", "리플"],
        "ADA": ["ADA", "카르다노", "에이다"],
        # 일본
        "7203.T": ["도요타", "トヨタ", "トヨタ自動車"],
        "6758.T": ["소니", "ソニー"],
        # 미국
        "AAPL": ["애플", "Apple"],
        "MSFT": ["마이크로소프트", "Microsoft"]
    }

    keywords = keywords_map.get(asset.upper(), [asset])

    for kw in keywords:
        try:
            # 코인은 암호화폐 키워드 추가
            query_kw = f"{kw} 암호화폐" if asset.upper() in ["XRP", "ADA"] else kw
            url = f"https://search.naver.com/search.naver?where=news&query={query_kw}"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            news_titles += [a.text for a in soup.select(".news_tit")[:2]]
        except Exception as e:
            print(f"❌ 뉴스 크롤링 실패 ({kw}): {e}")
            send_telegram(f"❌ 뉴스 크롤링 실패 ({kw})")
            continue

    return news_titles if news_titles else ["뉴스 없음 (평가 불가)"]

def evaluate_news(articles):
    if not articles:
        return "뉴스 없음"
    prompt = f"다음 뉴스 제목으로 시황을 요약하고 매수/매도/보류 중 하나를 판단해줘:\n{articles}"

    if not OPENAI_API_KEY:
        print("❗ OPENAI_API_KEY가 .env에 없습니다.")
        send_telegram("❗ OpenAI API 키 없음 - 뉴스 평가 실패")
        return "뉴스 평가 실패 (API 키 없음)"

    try:
        res = requests.post(
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
        if res.status_code != 200:
            print(f"❗ OpenAI 응답 실패 - 코드: {res.status_code}, 응답: {res.text}")
            send_telegram(f"❌ 뉴스 평가 실패 - OpenAI 응답 코드: {res.status_code}")
            return "뉴스 평가 실패"
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"❌ 뉴스 평가 중 예외 발생: {e}")
        send_telegram(f"❌ 뉴스 평가 예외: {e}")
        return "뉴스 평가 실패"

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
        print(f"📛 거래 로그 저장 오류: {e}")
        send_telegram(f"📛 거래 로그 저장 오류: {e}")

        
