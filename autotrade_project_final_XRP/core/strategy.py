import os
from dotenv import load_dotenv
from utils import get_fear_greed_index, fetch_all_news, evaluate_news

load_dotenv()

# 각 전략가의 판단 함수 (예시로 고정값 사용)
def strategy_buffett(): return "hold"
def strategy_jesse(): return "buy"
def strategy_wonyo(): return "buy"
def strategy_jim_rogers(): return "buy"

def analyze_coin(coin):
    articles = fetch_all_news(coin)
    summary = evaluate_news(articles)
    fg_index = get_fear_greed_index()

    # 4인의 전략 판단 합산
    votes = [strategy_buffett(), strategy_jesse(), strategy_wonyo(), strategy_jim_rogers()]
    result = max(set(votes), key=votes.count)

    # FG 지수에 따라 최종 판단 (70 이하일 때만 매수 가능)
    decision = result if fg_index <= 70 else "보류"
    confidence = 85 if decision == "buy" else 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 30,
        "reason": f"[FG:{fg_index}] | 요약: {summary}"
    }

