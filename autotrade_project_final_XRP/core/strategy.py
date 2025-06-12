import os
from dotenv import load_dotenv
from utils import get_fear_greed_index, fetch_all_news, evaluate_news

load_dotenv()

# 투자자 전략 정의
def strategy_buffett():
    return "hold"

def strategy_jesse():
    return "buy"

def strategy_wonyo():
    return "buy"

def strategy_jim_rogers():
    return "buy"

# 전체 전략 분석 함수
def analyze_coin(coin):
    # 뉴스 수집 및 평가
    articles = fetch_all_news(coin)
    sentiment = evaluate_news(articles)

    # 공포탐욕지수 확인
    fg_index = get_fear_greed_index()

    # 전략 투표
    votes = [
        strategy_buffett(),
        strategy_jesse(),
        strategy_wonyo(),
        strategy_jim_rogers()
    ]

    # 다수결 판단
    result = max(set(votes), key=votes.count)

    # FG 지수가 70 이하일 때만 매수 고려
    decision = result if fg_index <= 70 else "보류"

    # 신뢰도 설정
    confidence = 85 if decision == "buy" else 60

    # 결과 반환
    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 30,
        "reason": f"FG지수: {fg_index} | 뉴스 평가: {sentiment}"
    }

