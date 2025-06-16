from utils import fetch_all_news, evaluate_news

def strategy_honma(): return "buy"
def strategy_ichimoku(): return "buy"
def strategy_bnf(): return "buy"
def strategy_cis(): return "hold"

def analyze_japan(stock):
    news = fetch_all_news(stock)
    sentiment = evaluate_news(news)

    votes = [strategy_honma(), strategy_ichimoku(), strategy_bnf(), strategy_cis()]
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
