def decision_coin(fg, sentiment, rsi, momentum, price_change, volatility):
    votes = []
    
    # 워렌 버핏
    if fg > 60 and rsi > 60 and "긍정" in sentiment:
        votes.append("buy")
    
    # 제시 리버모어
    if momentum == "strong":
        votes.append("buy")
    
    # 워뇨띠
    if price_change < -0.05:
        votes.append("buy")
    
    # 짐 로저스
    if fg < 30 and price_change < -0.10:
        votes.append("buy")
    
    # 안정성 필터
    if volatility <= 0.05:
        votes.append("buy")

    decision = "buy" if votes.count("buy") >= 3 else "hold"
    confidence = 40 + votes.count(decision) * 12
    return decision, confidence, votes


