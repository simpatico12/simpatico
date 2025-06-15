import os
from dotenv import load_dotenv
from utils import get_fear_greed_index, fetch_all_news, evaluate_news

load_dotenv()

def strategy_buffett():
    return "hold"

def strategy_jesse():
    return "buy"

def strategy_wonyo():
    return "buy"

def strategy_jim_rogers():
    return "buy"

def analyze_coin(coin, use_dummy=False):
    if use_dummy:
        # 더미 모드: 고정된 값 리턴
        return {
            "decision": "buy",
            "confidence_score": 85,
            "percentage": 15,
            "reason": "코인 더미 판단"
        }

    # 실제 판단 로직
    articles = fetch_all_news(coin)
    sentiment = evaluate_news(articles)
    fg_index = get_fear_greed_index()
    votes = [strategy_buffett(), strategy_jesse(), strategy_wonyo(), strategy_jim_rogers()]
    result = max(set(votes), key=votes.count)
    decision = result if fg_index <= 70 else "보류"
    confidence = 85 if decision == "buy" else 60
    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 15,  # 분할매수 2회 + 비중 제한 고려
        "reason": f"FG지수:{fg_index} | 뉴스:{sentiment}"
    }

