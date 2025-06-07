
import os
from utils import get_fear_greed_index, fetch_news, evaluate_news
from dotenv import load_dotenv

load_dotenv()

def get_news_sentiment(coin):
    articles = fetch_news(coin)
    return evaluate_news(articles)

def strategy_buffett(): return "hold"
def strategy_jesse(): return "buy"
def strategy_wonyo(): return "buy"
def strategy_jim_rogers(): return "buy"

def analyze_coin(coin):
    sentiment = get_news_sentiment(coin)
    fg_index = get_fear_greed_index()
    votes = [strategy_buffett(), strategy_jesse(), strategy_wonyo(), strategy_jim_rogers()]
    result = max(set(votes), key=votes.count)
    decision = result if fg_index <= 60 else "보류"
    confidence = 85 if result == "buy" else 60
    return {"decision": decision, "confidence_score": confidence, "percentage": 30, "reason": f"FG지수:{fg_index} | 뉴스:{sentiment}"}
