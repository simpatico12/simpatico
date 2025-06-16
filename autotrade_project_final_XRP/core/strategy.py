from utils import get_fear_greed_index, fetch_all_news, evaluate_news

def strategy_buffett(): return "hold"
def strategy_jesse(): return "buy"
def strategy_wonyo(): return "buy"
def strategy_jim_rogers(): return "buy"

def analyze_coin(coin):
    fg = get_fear_greed_index()
    news = fetch_all_news(coin)
    sentiment = evaluate_news(news)

    votes = [strategy_buffett(), strategy_jesse(), strategy_wonyo(), strategy_jim_rogers()]
    result = max(set(votes), key=votes.count)

    if fg <= 70 and "부정" not in sentiment:
        decision = result
    else:
        decision = "hold"

    confidence = 85 if decision == "buy" else 80 if decision == "sell" else 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "reason": f"FG: {fg}, 뉴스: {sentiment}, 전략투표: {votes}"
    }
