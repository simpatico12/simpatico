# analyzer.py
"""
고급 AI 자산 분석 시스템 - 퀸트프로젝트 수준
멀티모델 앙상블과 설명 가능한 의사결정
"""

import json
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from utils import (
    get_fear_greed_index, 
    fetch_all_news, 
    is_holiday_or_weekend,
    get_price,
    get_volume
)
from core.api_wrapper import openai_chat
from logger import logger
from config import get_config
from exceptions import handle_errors
import pyupbit


@dataclass
class AnalysisContext:
    """분석 컨텍스트"""
    asset: str
    asset_type: str
    fg_index: int
    news_list: List[Dict]
    price_data: Dict
    technical_indicators: Dict
    market_sentiment: str
    timestamp: datetime


class AdvancedAnalyzer:
    """고급 AI 분석기"""
    
    def __init__(self):
        self.cfg = get_config()
        self.confidence_weights = {
            'ai_analysis': 0.4,
            'technical': 0.3,
            'sentiment': 0.2,
            'market': 0.1
        }
        
    async def analyze_asset(self, asset: str, asset_type: str) -> Dict:
        """
        종합적인 자산 분석
        AI + 기술적 분석 + 시장 심리 + 뉴스 감성
        """
        # 1) 시장 휴장일 체크
        if asset_type != "coin" and is_holiday_or_weekend():
            return self._create_response("hold", "시장 휴장일", 50)
        
        try:
            # 2) 병렬로 데이터 수집
            context = await self._gather_analysis_context(asset, asset_type)
            
            # 3) 멀티 분석 실행
            analyses = await asyncio.gather(
                self._ai_analysis(context),
                self._technical_analysis(context),
                self._sentiment_analysis(context),
                self._market_analysis(context)
            )
            
            # 4) 앙상블 의사결정
            final_decision = self._ensemble_decision(analyses)
            
            # 5) 설명 생성
            explanation = self._generate_explanation(analyses, final_decision)
            
            return final_decision | {'explanation': explanation}
            
        except Exception as e:
            logger.error(f"분석 오류 {asset}: {e}")
            return self._create_response("hold", "분석 실패", 30)
    
    async def _gather_analysis_context(self, asset: str, asset_type: str) -> AnalysisContext:
        """분석에 필요한 모든 데이터 수집"""
        # 병렬 데이터 수집
        fg_task = asyncio.create_task(asyncio.to_thread(get_fear_greed_index))
        news_task = asyncio.create_task(asyncio.to_thread(fetch_all_news, asset))
        price_task = asyncio.create_task(self._get_price_data(asset, asset_type))
        
        fg_index = await fg_task
        news_list = await news_task
        price_data = await price_task
        
        # 기술적 지표 계산
        technical_indicators = self._calculate_technical_indicators(price_data)
        
        # 시장 심리 판단
        market_sentiment = self._judge_market_sentiment(fg_index)
        
        return AnalysisContext(
            asset=asset,
            asset_type=asset_type,
            fg_index=fg_index,
            news_list=news_list,
            price_data=price_data,
            technical_indicators=technical_indicators,
            market_sentiment=market_sentiment,
            timestamp=datetime.now()
        )
    
    async def _get_price_data(self, asset: str, asset_type: str) -> Dict:
        """가격 데이터 조회"""
        try:
            if asset_type == "coin":
                ticker = f"KRW-{asset}"
                ohlcv = pyupbit.get_ohlcv(ticker, interval="day", count=30)
                
                return {
                    'current': float(ohlcv['close'].iloc[-1]),
                    'open': float(ohlcv['open'].iloc[-1]),
                    'high': float(ohlcv['high'].iloc[-1]),
                    'low': float(ohlcv['low'].iloc[-1]),
                    'volume': float(ohlcv['volume'].iloc[-1]),
                    'history': ohlcv['close'].tolist(),
                    'change_rate': (ohlcv['close'].iloc[-1] - ohlcv['close'].iloc[-2]) / ohlcv['close'].iloc[-2]
                }
            else:
                # 주식 데이터 (구현 필요)
                return {
                    'current': get_price(asset, asset_type),
                    'volume': get_volume(asset, asset_type),
                    'change_rate': 0.0
                }
        except Exception as e:
            logger.error(f"가격 데이터 조회 실패: {e}")
            return {}
    
    def _calculate_technical_indicators(self, price_data: Dict) -> Dict:
        """기술적 지표 계산"""
        if not price_data.get('history'):
            return {}
        
        prices = np.array(price_data['history'])
        
        # RSI 계산
        rsi = self._calculate_rsi(prices)
        
        # 이동평균
        ma5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        ma20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        
        # MACD
        macd, signal = self._calculate_macd(prices)
        
        # 볼린저 밴드
        upper, middle, lower = self._calculate_bollinger_bands(prices)
        
        current_price = prices[-1]
        
        return {
            'rsi': rsi,
            'ma5': ma5,
            'ma20': ma20,
            'ma_signal': 'golden_cross' if ma5 > ma20 else 'death_cross',
            'macd': macd,
            'macd_signal': signal,
            'bb_position': self._get_bb_position(current_price, upper, lower),
            'trend': 'up' if current_price > ma20 else 'down'
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """MACD 계산"""
        if len(prices) < 26:
            return 0.0, 0.0
        
        exp1 = np.array(prices[-12:]).mean()  # 간단한 12일 평균
        exp2 = np.array(prices[-26:]).mean()  # 간단한 26일 평균
        
        macd = exp1 - exp2
        signal = macd * 0.9  # 간단한 시그널
        
        return float(macd), float(signal)
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
        """볼린저 밴드 계산"""
        if len(prices) < period:
            current = prices[-1]
            return current * 1.02, current, current * 0.98
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        return float(upper), float(sma), float(lower)
    
    def _get_bb_position(self, price: float, upper: float, lower: float) -> str:
        """볼린저 밴드 포지션"""
        if price > upper:
            return "overbought"
        elif price < lower:
            return "oversold"
        else:
            return "neutral"
    
    def _judge_market_sentiment(self, fg_index: int) -> str:
        """시장 심리 판단"""
        if fg_index < 20:
            return "extreme_fear"
        elif fg_index < 40:
            return "fear"
        elif fg_index < 60:
            return "neutral"
        elif fg_index < 80:
            return "greed"
        else:
            return "extreme_greed"
    
    async def _ai_analysis(self, context: AnalysisContext) -> Dict:
        """AI 기반 분석"""
        prompt = self._create_ai_prompt(context)
        
        try:
            response = await asyncio.to_thread(openai_chat, prompt)
            return self._parse_ai_response(response)
        except Exception as e:
            logger.error(f"AI 분석 실패: {e}")
            return self._create_response("hold", "AI 분석 실패", 0)
    
    def _create_ai_prompt(self, context: AnalysisContext) -> str:
        """AI 프롬프트 생성"""
        news_summary = self._summarize_news(context.news_list)
        
        return f"""
전문 퀀트 트레이더로서 다음 자산을 분석해주세요:

자산: {context.asset} ({context.asset_type})
현재가: {context.price_data.get('current', 'N/A'):,.0f}
변동률: {context.price_data.get('change_rate', 0)*100:.2f}%

시장 지표:
- 공포·탐욕 지수: {context.fg_index} ({context.market_sentiment})
- RSI: {context.technical_indicators.get('rsi', 50):.1f}
- 추세: {context.technical_indicators.get('trend', 'unknown')}
- 볼린저밴드: {context.technical_indicators.get('bb_position', 'neutral')}

최근 뉴스:
{news_summary}

위 정보를 종합하여:
1. 투자 결정: buy/sell/hold 중 하나
2. 신뢰도: 0-100 숫자
3. 핵심 근거: 30자 이내

형식: "결정:신뢰도:근거"
예시: "buy:85:과매도 구간에서 반등 신호"
"""
    
    def _summarize_news(self, news_list: List[Dict]) -> str:
        """뉴스 요약"""
        if not news_list:
            return "관련 뉴스 없음"
        
        summaries = []
        for i, news in enumerate(news_list[:5], 1):
            summaries.append(f"{i}. {news.get('title', 'N/A')}")
        
        return "\n".join(summaries)
    
    def _parse_ai_response(self, response: str) -> Dict:
        """AI 응답 파싱"""
        try:
            parts = response.strip().split(":")
            if len(parts) >= 3:
                decision = parts[0].lower().strip()
                confidence = int(parts[1].strip())
                reason = ":".join(parts[2:]).strip()
                
                if decision in ['buy', 'sell', 'hold'] and 0 <= confidence <= 100:
                    return self._create_response(decision, reason, confidence)
        except:
            pass
        
        return self._create_response("hold", "AI 응답 파싱 실패", 30)
    
    async def _technical_analysis(self, context: AnalysisContext) -> Dict:
        """기술적 분석"""
        indicators = context.technical_indicators
        
        signals = []
        
        # RSI 신호
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            signals.append(('buy', 80, "RSI 과매도"))
        elif rsi > 70:
            signals.append(('sell', 80, "RSI 과매수"))
        
        # 이동평균 신호
        if indicators.get('ma_signal') == 'golden_cross':
            signals.append(('buy', 70, "골든크로스"))
        elif indicators.get('ma_signal') == 'death_cross':
            signals.append(('sell', 70, "데드크로스"))
        
        # 볼린저밴드 신호
        bb_pos = indicators.get('bb_position')
        if bb_pos == 'oversold':
            signals.append(('buy', 75, "볼린저밴드 하단"))
        elif bb_pos == 'overbought':
            signals.append(('sell', 75, "볼린저밴드 상단"))
        
        # 종합 판단
        if signals:
            # 가장 강한 신호 선택
            signals.sort(key=lambda x: x[1], reverse=True)
            return self._create_response(*signals[0])
        
        return self._create_response("hold", "기술적 중립", 60)
    
    async def _sentiment_analysis(self, context: AnalysisContext) -> Dict:
        """감성 분석"""
        if not context.news_list:
            return self._create_response("hold", "뉴스 없음", 50)
        
        # 간단한 감성 점수 계산
        positive_keywords = ['상승', '호재', '성장', '긍정', '최고', '신고가']
        negative_keywords = ['하락', '악재', '우려', '부정', '최저', '폭락']
        
        positive_count = 0
        negative_count = 0
        
        for news in context.news_list:
            text = news.get('title', '') + news.get('summary', '')
            for keyword in positive_keywords:
                if keyword in text:
                    positive_count += 1
            for keyword in negative_keywords:
                if keyword in text:
                    negative_count += 1
        
        total = positive_count + negative_count
        if total == 0:
            return self._create_response("hold", "중립적 뉴스", 60)
        
        positive_ratio = positive_count / total
        
        if positive_ratio > 0.7:
            return self._create_response("buy", "긍정적 뉴스", 75)
        elif positive_ratio < 0.3:
            return self._create_response("sell", "부정적 뉴스", 75)
        else:
            return self._create_response("hold", "혼재된 뉴스", 60)
    
    async def _market_analysis(self, context: AnalysisContext) -> Dict:
        """시장 분석"""
        fg = context.fg_index
        sentiment = context.market_sentiment
        
        if sentiment == "extreme_fear" and fg < 20:
            return self._create_response("buy", "극단적 공포 = 기회", 70)
        elif sentiment == "extreme_greed" and fg > 80:
            return self._create_response("sell", "극단적 탐욕 = 위험", 70)
        else:
            return self._create_response("hold", f"시장 심리 {sentiment}", 55)
    
    def _ensemble_decision(self, analyses: List[Dict]) -> Dict:
        """앙상블 의사결정"""
        decisions = {'buy': 0, 'sell': 0, 'hold': 0}
        weighted_confidence = 0
        reasons = []
        
        weights = [
            self.confidence_weights['ai_analysis'],
            self.confidence_weights['technical'],
            self.confidence_weights['sentiment'],
            self.confidence_weights['market']
        ]
        
        for analysis, weight in zip(analyses, weights):
            decision = analysis['decision']
            confidence = analysis['confidence_score']
            
            decisions[decision] += confidence * weight
            weighted_confidence += confidence * weight
            reasons.append(analysis['reason'])
        
        # 최종 결정
        final_decision = max(decisions, key=decisions.get)
        
        # 신뢰도 조정
        decision_strength = decisions[final_decision] / sum(weights)
        
        return {
            'decision': final_decision,
            'confidence_score': int(decision_strength),
            'reason': self._summarize_reasons(reasons),
            'detailed_scores': decisions,
            'analysis_results': analyses
        }
    
    def _summarize_reasons(self, reasons: List[str]) -> str:
        """이유 요약"""
        # 가장 중요한 2개 이유 선택
        unique_reasons = list(set(reasons))
        if len(unique_reasons) > 2:
            return ", ".join(unique_reasons[:2])
        return ", ".join(unique_reasons)
    
    def _generate_explanation(self, analyses: List[Dict], final_decision: Dict) -> str:
        """상세 설명 생성"""
        explanation = f"📊 종합 분석 결과: {final_decision['decision'].upper()}\n\n"
        
        analysis_names = ["AI 분석", "기술적 분석", "감성 분석", "시장 분석"]
        
        for name, analysis in zip(analysis_names, analyses):
            emoji = "✅" if analysis['decision'] == final_decision['decision'] else "❌"
            explanation += f"{emoji} {name}: {analysis['decision']} "
            explanation += f"(신뢰도 {analysis['confidence_score']}%)\n"
            explanation += f"   └─ {analysis['reason']}\n\n"
        
        explanation += f"💡 최종 신뢰도: {final_decision['confidence_score']}%"
        
        return explanation
    
    def _create_response(self, decision: str, reason: str, confidence: int) -> Dict:
        """응답 생성"""
        return {
            'decision': decision,
            'reason': reason,
            'confidence_score': confidence
        }


# 전역 인스턴스
analyzer = AdvancedAnalyzer()

# 기존 인터페이스 호환
@handle_errors(default_return={'decision': 'hold', 'reason': '분석 실패', 'confidence_score': 30})
def analyze_asset(asset: str, asset_type: str) -> Dict:
    """동기 인터페이스 (기존 호환성)"""
    return asyncio.run(analyzer.analyze_asset(asset, asset_type))


if __name__ == "__main__":
    # 테스트
    import time
    
    async def test_analysis():
        """분석 테스트"""
        test_assets = [
            ("BTC", "coin"),
            ("ETH", "coin"),
            ("AAPL", "us"),
            ("7203.T", "japan")
        ]
        
        for asset, asset_type in test_assets:
            print(f"\n{'='*50}")
            print(f"분석 중: {asset} ({asset_type})")
            print(f"{'='*50}")
            
            start = time.time()
            result = await analyzer.analyze_asset(asset, asset_type)
            elapsed = time.time() - start
            
            print(f"결정: {result['decision']}")
            print(f"신뢰도: {result['confidence_score']}%")
            print(f"이유: {result['reason']}")
            print(f"소요시간: {elapsed:.2f}초")
            
            if 'explanation' in result:
                print(f"\n{result['explanation']}")
    
    # 실행
    asyncio.run(test_analysis())