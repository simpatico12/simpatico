import os
from dotenv import load_dotenv
from utils import get_fear_greed_index, fetch_news, evaluate_news

load_dotenv()

# 버핏: 보수적으로 보류
def strategy_buffett():
    return "hold"

# 제시 리버모어: 추세 따라 매수
def strategy_jesse():
    return "buy"

# 워뇨띠: 눌림목이면 매수
def strategy_wonyo():
    return "buy"

# 짐 로저스: 장기 상승 기대
def strategy_jim_rogers():
    return "buy"

def get_news_sentiment(coin):
    articles = fetch_news(coin)
    return evaluate_news(articles)

def analyze_coin(coin):
    sentiment = get_news_sentiment(coin)
    fg_index = get_fear_greed_index()

    # 전략별 투표
    votes = [strategy_buffett(), strategy_jesse(), strategy_wonyo(), strategy_jim_rogers()]
    result = max(set(votes), key=votes.count)

    # FG지수가 낮으면 매매 가능, 높으면 보류
    decision = result if fg_index <= 60 else "보류"
    confidence = 85 if decision == "buy" else 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 30,
        "reason": f"FG지수:{fg_index} | 뉴스:{sentiment}"
    }
