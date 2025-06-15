from utils import send_telegram
import random

def analyze_us(stock):
    # 더미 뉴스/차트 점수 (실제 구현 시 뉴스 API, yfinance, TA 지표 분석 포함)
    news_sentiment_score = random.uniform(0.0, 1.0)
    momentum_signal = random.choice(["buy", "sell", "hold"])
    earnings_nearby = random.choice([True, False])

    reason = f"뉴스 감성: {news_sentiment_score:.2f}, 모멘텀: {momentum_signal}, 실적임박: {earnings_nearby}"
    
    # 조건 종합
    if momentum_signal == "buy" and news_sentiment_score > 0.6 and not earnings_nearby:
        decision = "buy"
        confidence = 90
    elif momentum_signal == "sell":
        decision = "sell"
        confidence = 85
    else:
        decision = "hold"
        confidence = 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 40,  # 자본의 40% 분할매수
        "reason": reason
    }
