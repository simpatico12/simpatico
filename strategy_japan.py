def decision_japan(ichimoku, candlestick, volume_spike, turnover, pattern):
    votes = []

    # 혼마 무네히사
    if ichimoku == "buy" and "강한양봉" in candlestick:
        votes.append("buy")

    # cis
    if volume_spike:
        votes.append("buy")

    # BNF
    if turnover > 2.0:
        votes.append("buy")

    # 하라 요시유키
    if pattern == "상승형":
        votes.append("buy")

    decision = "buy" if votes.count("buy") >= 2 else "hold"
    confidence = 40 + votes.count(decision) * 15
    return decision, confidence, votes
