from utils import send_telegram
import random

def analyze_us(stock, use_dummy=False):
    if use_dummy:
        # 더미 모드: 고정된 값 리턴
        result = {
            "decision": "sell",
            "confidence_score": 80,
            "percentage": 40,
            "reason": "미국 더미 판단"
        }
    else:
        # 실제 판단 로직
        news_sentiment_score = random.uniform(0.0, 1.0)
        momentum_signal = random.choice(["buy", "sell", "hold"])
        earnings_nearby = random.choice([True, False])

        reason = f"뉴스 감성: {news_sentiment_score:.2f}, 모멘텀: {momentum_signal}, 실적임박: {earnings_nearby}"

        if momentum_signal == "buy" and news_sentiment_score > 0.6 and not earnings_nearby:
            decision = "buy"
            confidence = 90
        elif momentum_signal == "sell":
            decision = "sell"
            confidence = 85
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
    send_telegram(f"🔎 [미국 {stock}] 판단: {result['decision']} | 신뢰도: {result['confidence_score']}%\n이유: {result['reason']}")
    return result
