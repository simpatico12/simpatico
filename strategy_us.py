def decision_us(sector_trend, earnings_near, volatility, momentum):
    votes = []

    # 성장주 전략
    if sector_trend == "positive" and not earnings_near:
        votes.append("buy")

    # 변동성 안정성
    if volatility <= 0.05:
        votes.append("buy")

    # 모멘텀 전략
    if momentum == "strong":
        votes.append("buy")

    decision = "buy" if votes.count("buy") >= 2 else "hold"
    confidence = 40 + votes.count(decision) * 15
    return decision, confidence, votes
