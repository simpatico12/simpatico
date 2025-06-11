import os
from dotenv import load_dotenv
from utils import get_fear_greed_index, fetch_all_news, evaluate_news

load_dotenv()

# 전략 1: 워렌 버핏 - 보수적 보유
def strategy_buffett():
    return "hold"

# 전략 2: 제시 리버모어 - 추세 매매
def strategy_jesse():
    return "buy"

# 전략 3: 워뇨띠 - 눌림목 매수
def strategy_wonyo():
    return "buy"

# 전략 4: 짐 로저스 - 장기 상승 기대
def strategy_jim_rogers():
    return "buy"

def analyze_coin(coin):
    # 6개 뉴스 통합 크롤링
    articles = fetch_all_news(coin)
    sentiment = evaluate_news(articles)

    # 공포탐욕지수 수집
    fg_index = get_fear_greed_index()

    # 전략 투표
    votes = [strategy_buffett(), strategy_jesse(), strategy_wonyo(), strategy_jim_rogers()]
    result = max(set(votes), key=votes.count)

    # 판단 로직
    decision = result if fg_index <= 60 else "보류"
    confidence = 85 if decision == "buy" else 60

    # 결과 리턴
    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 30,
        "reason": f"FG지수:{fg_index} | 뉴스:{sentiment}"
    }

