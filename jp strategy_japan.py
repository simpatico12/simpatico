from utils import send_telegram
import random

def analyze_japan(stock):
    # 더미 뉴스/차트 점수 (실제 구현 시 뉴스 API, 차트 분석 포함)
    news_sentiment_score = random.uniform(0.0, 1.0)
    ichimoku_signal = random.choice(["buy", "sell", "hold"])
    volume_spike = random.choice([True, False])

    reason = f"뉴스 감성: {news_sentiment_score:.2f}, 이치모쿠: {ichimoku_signal}, 거래량 급등: {volume_spike}"
    
    # 조건 종합
    if ichimoku_signal == "buy" and news_sentiment_score > 0.6 and volume_spike:
        decision = "buy"
        confidence = 85
    elif ichimoku_signal == "sell":
        decision = "sell"
        confidence = 80
    else:
        decision = "hold"
        confidence = 60

    return {
        "decision": decision,
        "confidence_score": confidence,
        "percentage": 40,  # 자본의 40% 분할매수
        "reason": reason
    }
