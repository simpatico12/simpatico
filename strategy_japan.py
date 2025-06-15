from utils import send_telegram
import random

def analyze_japan(stock, use_dummy=False):
    if use_dummy:
        # 더미 모드: 고정된 값 리턴
        result = {
            "decision": "hold",
            "confidence_score": 60,
            "percentage": 40,
            "reason": "일본 더미 판단"
        }
    else:
        # 실제 판단 로직
        news_sentiment_score = random.uniform(0.0, 1.0)
        ichimoku_signal = random.choice(["buy", "sell", "hold"])
        volume_spike = random.choice([True, False])

        reason = f"뉴스 감성: {news_sentiment_score:.2f}, 이치모쿠: {ichimoku_signal}, 거래량 급등: {volume_spike}"

        if ichimoku_signal == "buy" and news_sentiment_score > 0.6 and volume_spike:
            decision = "buy"
            confidence = 85
        elif ichimoku_signal == "sell":
            decision = "sell"
            confidence = 80
        else:
            decision = "hold"
            confidence = 60

        result = {
            "decision": decision,
            "confidence_score": confidence,
            "percentage": 40,
            "reason": reason
        }

    # 판단 결과 텔레그램 전송
    send_telegram(f"🔎 [일본 {stock}] 판단: {result['decision']} | 신뢰도: {result['confidence_score']}%\n이유: {result['reason']}")
    return result
