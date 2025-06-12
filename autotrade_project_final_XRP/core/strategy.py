import os
from dotenv import load_dotenv
from utils import get_fear_greed_index, fetch_all_news, evaluate_news

load_dotenv()

# ✅ 전략 4인방
def strategy_buffett(): return "hold"
def strategy_jesse(): return "buy"
def strategy_wonyo(): return "buy"
def strategy_jim_rogers(): return "buy"

# ✅ 분석 메인 함수
def analyze_coin(coin):
    # 뉴스 가져오기 및 평가
    articles = fetch_all_news(coin)
    sentiment_summary = evaluate_news(articles)

    # 공포탐욕지수
    fg_index = get_fear_greed_index()

    # 전략별 판단
    votes = [
        strategy_buffett(),
        strategy_jesse(),
        strategy_wonyo(),
        strategy_jim_rogers()
    ]
    majority_decision = max(set(votes), key=votes.count)

    # FG 기준 적용 (기본 70 이하일 때만 매매)
    decision = majority_decision if fg_index <= 70 else "보류"

    # 신뢰도 점수
    confidence = 85 if decision == "buy" else 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 30,
        "reason": f"FG지수: {fg_index} | 뉴스요약: {sentiment_summary}"
    }

