from utils import fetch_all_news, evaluate_news

def strategy_buffett(): return "hold"
def strategy_lynch(): return "buy"
def strategy_dalio(): return "buy"
def strategy_jesse(): return "buy"

def analyze_us(stock):
    news = fetch_all_news(stock)
    sentiment = evaluate_news(news)

    votes = [strategy_buffett(), strategy_lynch(), strategy_dalio(), strategy_jesse()]
    result = max(set(votes), key=votes.count)

    if "부정" not in sentiment:
        decision = result
    else:
        decision = "sell"

    confidence = 85 if decision == "buy" else 80 if decision == "sell" else 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "reason": f"뉴스: {sentiment}, 전략투표: {votes}"
    }
