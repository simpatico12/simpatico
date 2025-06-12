import os
from dotenv import load_dotenv
from utils import get_fear_greed_index, fetch_all_news, evaluate_news

load_dotenv()

# 전략 함수 정의
def strategy_buffett(): return "hold"
def strategy_jesse(): return "buy"
def strategy_wonyo(): return "buy"
def strategy_jim_rogers(): return "buy"

# 분석 함수
def analyze_coin(coin):
    articles = fetch_all_news(coin)  # 6개 뉴스 통합 (네이버, 구글, 코인데스크, 코인텔레그래프, 야후파이낸스, 바이낸스 블로그)
    sentiment = evaluate_news(articles)
    fg_index = get_fear_greed_index()

    # 전략별 판단 결과
    votes = [
        strategy_buffett(),
        strategy_jesse(),
        strategy_wonyo(),
        strategy_jim_rogers()
    ]
    result = max(set(votes), key=votes.count)

    # FG 지수 기반 보류 판단
    decision = result if fg_index <= 60 else "보류"
    confidence = 85 if decision == "buy" else 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 30,
        "reason": f"FG지수:{fg_index} | 뉴스:{sentiment}"
    }


