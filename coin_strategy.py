#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ LEGENDARY QUANT STRATEGY COMPLETE ⚡
전설급 5대 시스템 + 완전한 매도 시스템 + OpenAI 기술분석 최적화 (월 5-7% 목표)

🧠 Neural Quality Engine - 가중평균 기반 품질 스코어링
🌊 Quantum Cycle Matrix - 27개 미시사이클 감지  
⚡ Fractal Filtering Pipeline - 다차원 필터링
💎 Diamond Hand Algorithm - 켈리공식 기반 분할매매
🕸️ Correlation Web Optimizer - 네트워크 포트폴리오
🎯 Position Manager - 포지션 관리 + 실시간 매도
🤖 OpenAI Technical Analyzer - 기술분석 + 확신도 체크 (월 5천원 이하)

✨ 월 5-7% 최적화:
- 0차 익절 추가 (5-7% 구간)
- 3차 익절 삭제 (무제한 수익)
- 타이트한 손절 (-5~8%)
- 월금 매매 시스템
- OpenAI 스마트 호출 (애매한 상황만)

Author: 퀀트마스터 | Version: OPTIMIZED + AI EFFICIENCY
"""

import asyncio
import numpy as np
import pandas as pd
import pyupbit
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os
import time
import openai
from openai import OpenAI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 🤖 OPENAI TECHNICAL ANALYZER - 기술분석 + 확신도 체크 최적화
# ============================================================================
class OpenAITechnicalAnalyzer:
    """OpenAI 기술분석 엔진 - 비용 최적화 (월 5천원 이하)"""
    
    def __init__(self, api_key: str = None):
        """OpenAI 클라이언트 초기화"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.enabled = True
            self.call_count = 0
            self.daily_limit = 50  # 일일 호출 제한 (월 1500회 = 1500토큰 * $0.002 = $3)
            logger.info("🤖 OpenAI 기술분석 엔진 초기화 완료 (비용 최적화)")
        else:
            self.client = None
            self.enabled = False
            logger.warning("⚠️ OpenAI API 키가 없습니다. 기본 분석만 사용됩니다.")
    
    def should_use_ai(self, confidence: float, volume_rank: int) -> bool:
        """AI 사용 여부 결정 - 스마트 호출"""
        # 일일 호출 제한 체크
        if self.call_count >= self.daily_limit:
            return False
        
        # 애매한 확신도 구간에서만 AI 호출
        if 0.4 <= confidence <= 0.7:
            return True
        
        # 고거래량 코인 중 애매한 경우
        if volume_rank <= 20 and 0.3 <= confidence <= 0.8:
            return True
        
        return False
    
    async def analyze_technical_confidence(self, symbol: str, technical_data: Dict) -> Dict:
        """기술적 분석 기반 확신도 체크 (간결한 프롬프트)"""
        if not self.enabled or not self.should_use_ai(
            technical_data.get('base_confidence', 0.5), 
            technical_data.get('volume_rank', 100)
        ):
            return self._fallback_confidence()
        
        try:
            self.call_count += 1
            
            # 간결한 기술분석 프롬프트 (토큰 절약)
            prompt = f"""코인: {symbol}
RSI: {technical_data.get('rsi', 50):.1f}
MACD: {technical_data.get('macd_signal', 'neutral')}
볼린저: {technical_data.get('bollinger_position', 'middle')}
거래량: {technical_data.get('volume_trend', 'normal')}
모멘텀: {technical_data.get('momentum_7d', 0):.1f}%

기술적 매수 신호 확신도는? (0-100점만)"""
            
            # 짧은 응답 요청
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "기술분석 전문가. 숫자만 답변."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10  # 극도로 제한하여 비용 절약
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # 숫자 추출
            try:
                confidence_score = float(''.join(filter(str.isdigit, ai_response))) / 100
                confidence_score = max(0.0, min(1.0, confidence_score))
                
                return {
                    'ai_confidence': confidence_score,
                    'ai_used': True,
                    'call_count': self.call_count,
                    'technical_signal': 'strong' if confidence_score > 0.7 else 'weak' if confidence_score < 0.4 else 'neutral'
                }
                
            except ValueError:
                return self._fallback_confidence()
            
        except Exception as e:
            logger.error(f"OpenAI 기술분석 실패 {symbol}: {e}")
            return self._fallback_confidence()
    
    async def analyze_trend_pattern(self, symbol: str, price_data: pd.Series) -> Dict:
        """트렌드 패턴 분석 (고확신도 상황에서만)"""
        if not self.enabled or self.call_count >= self.daily_limit:
            return self._fallback_trend()
        
        try:
            # 가격 데이터 요약
            recent_change = ((price_data.iloc[-1] / price_data.iloc[-8]) - 1) * 100
            volatility = price_data.pct_change().tail(7).std() * 100
            
            # 매우 간결한 프롬프트
            prompt = f"{symbol} 7일변동: {recent_change:+.1f}%, 변동성: {volatility:.1f}%\n패턴: 상승/하락/횡보?"
            
            self.call_count += 1
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "패턴분석가. 한단어만 답변."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=5
            )
            
            ai_response = response.choices[0].message.content.strip().lower()
            
            # 패턴 매핑
            if '상승' in ai_response or 'up' in ai_response:
                trend = 'bullish'
            elif '하락' in ai_response or 'down' in ai_response:
                trend = 'bearish'
            else:
                trend = 'sideways'
            
            return {
                'trend_pattern': trend,
                'ai_pattern_confidence': 0.8 if trend != 'sideways' else 0.5,
                'pattern_strength': 'strong' if abs(recent_change) > 10 else 'weak'
            }
            
        except Exception as e:
            logger.debug(f"트렌드 분석 실패 {symbol}: {e}")
            return self._fallback_trend()
    
    def get_usage_stats(self) -> Dict:
        """사용량 통계"""
        daily_cost = self.call_count * 0.002  # 대략적인 비용
        monthly_projection = daily_cost * 30
        
        return {
            'daily_calls': self.call_count,
            'daily_limit': self.daily_limit,
            'estimated_daily_cost_usd': daily_cost,
            'monthly_projection_usd': monthly_projection,
            'monthly_projection_krw': monthly_projection * 1300,  # 환율 1300원 가정
            'efficiency': 'optimal' if monthly_projection < 4 else 'over_budget'
        }
    
    def _fallback_confidence(self) -> Dict:
        """기본 확신도"""
        return {
            'ai_confidence': 0.5,
            'ai_used': False,
            'call_count': self.call_count,
            'technical_signal': 'neutral'
        }
    
    def _fallback_trend(self) -> Dict:
        """기본 트렌드"""
        return {
            'trend_pattern': 'sideways',
            'ai_pattern_confidence': 0.5,
            'pattern_strength': 'neutral'
        }

# ============================================================================
# 🧠 NEURAL QUALITY ENGINE - 품질 평가 + OpenAI 기술분석 통합
# ============================================================================
class NeuralQualityEngine:
    """가중평균 기반 품질 평가 엔진 + OpenAI 기술분석"""
    
    def __init__(self, openai_analyzer: OpenAITechnicalAnalyzer = None):
        # 코인별 품질 점수 (기술력, 생태계, 커뮤니티, 채택도)
        self.coin_scores = {
            'BTC': [0.98, 0.95, 0.90, 0.95], 'ETH': [0.95, 0.98, 0.85, 0.90],
            'BNB': [0.80, 0.90, 0.75, 0.85], 'ADA': [0.85, 0.75, 0.80, 0.70],
            'SOL': [0.90, 0.80, 0.85, 0.75], 'AVAX': [0.85, 0.75, 0.70, 0.70],
            'DOT': [0.85, 0.80, 0.70, 0.65], 'MATIC': [0.80, 0.85, 0.75, 0.80],
            'ATOM': [0.75, 0.70, 0.75, 0.60], 'NEAR': [0.80, 0.70, 0.65, 0.60],
            'LINK': [0.90, 0.75, 0.70, 0.80], 'UNI': [0.85, 0.80, 0.75, 0.75],
            'ALGO': [0.80, 0.70, 0.65, 0.60], 'XRP': [0.75, 0.80, 0.85, 0.90],
            'LTC': [0.85, 0.70, 0.80, 0.85], 'BCH': [0.80, 0.65, 0.75, 0.80],
            'LUNA': [0.70, 0.60, 0.50, 0.40], 'DOGE': [0.60, 0.65, 0.90, 0.75],
            'SHIB': [0.50, 0.60, 0.85, 0.70], 'ICP': [0.75, 0.65, 0.60, 0.55],
            'FTM': [0.75, 0.70, 0.65, 0.60], 'SAND': [0.70, 0.75, 0.70, 0.65],
            'MANA': [0.70, 0.75, 0.70, 0.65], 'CRO': [0.75, 0.80, 0.70, 0.75],
            'HBAR': [0.80, 0.70, 0.65, 0.60], 'VET': [0.75, 0.70, 0.65, 0.65],
            'FLOW': [0.75, 0.70, 0.60, 0.55], 'KSM': [0.80, 0.65, 0.60, 0.55],
            'XTZ': [0.80, 0.70, 0.65, 0.60], 'EGLD': [0.80, 0.70, 0.60, 0.55],
            'THETA': [0.75, 0.70, 0.65, 0.60], 'AXS': [0.70, 0.75, 0.80, 0.70],
            'EOS': [0.70, 0.65, 0.60, 0.65], 'WAVES': [0.70, 0.65, 0.60, 0.55],
            'ZIL': [0.70, 0.65, 0.60, 0.55], 'ENJ': [0.70, 0.70, 0.65, 0.60],
            'BAT': [0.70, 0.65, 0.70, 0.60], 'ZRX': [0.75, 0.70, 0.60, 0.60],
            'OMG': [0.70, 0.60, 0.55, 0.55], 'QTUM': [0.70, 0.60, 0.55, 0.55],
            'ICX': [0.70, 0.65, 0.70, 0.60], 'ANKR': [0.70, 0.65, 0.60, 0.55],
            'STORJ': [0.70, 0.65, 0.60, 0.55], 'SRM': [0.70, 0.65, 0.55, 0.50],
            'CVC': [0.65, 0.60, 0.55, 0.50], 'ARDR': [0.65, 0.60, 0.55, 0.50],
            'STRK': [0.70, 0.65, 0.60, 0.55], 'PUNDIX': [0.60, 0.55, 0.60, 0.55],
            'HUNT': [0.60, 0.55, 0.65, 0.55], 'HIVE': [0.65, 0.60, 0.65, 0.55],
            'STEEM': [0.65, 0.60, 0.70, 0.60], 'WEMIX': [0.65, 0.70, 0.60, 0.55]
        }
        
        # 가중치 (기술력 30%, 생태계 30%, 커뮤니티 20%, 채택도 20%)
        self.weights = [0.30, 0.30, 0.20, 0.20]
        self.openai_analyzer = openai_analyzer
    async def neural_quality_score(self, symbol: str, market_data: Dict, volume_rank: int) -> Dict:
        """종합 품질 점수 계산 + OpenAI 기술분석"""
        try:
            coin_name = symbol.replace('KRW-', '')
            
            # 기본 품질 점수
            scores = self.coin_scores.get(coin_name, [0.6, 0.6, 0.6, 0.6])
            
            # 가중평균 계산
            quality_score = sum(score * weight for score, weight in zip(scores, self.weights))
            
            # 거래량 기반 보너스
            volume_bonus = self._calculate_volume_bonus(market_data.get('volume_24h_krw', 0))
            base_quality = min(0.98, quality_score + volume_bonus)
            
            # 기술적 지표 계산
            technical_data = self._calculate_technical_indicators(market_data, volume_rank)
            
            # OpenAI 기술분석 (애매한 상황에서만)
            ai_result = None
            if self.openai_analyzer:
                technical_data['base_confidence'] = base_quality
                ai_result = await self.openai_analyzer.analyze_technical_confidence(symbol, technical_data)
            
            # 최종 확신도 조정
            if ai_result and ai_result.get('ai_used'):
                # AI 분석이 있으면 가중 평균
                ai_confidence = ai_result['ai_confidence']
                final_confidence = base_quality * 0.6 + ai_confidence * 0.4
                confidence_explanation = f"AI강화({ai_confidence:.2f})"
            else:
                final_confidence = base_quality
                confidence_explanation = "기본분석"
            
            # 설명 생성
            explanation = self._generate_explanation(coin_name, scores, final_confidence, ai_result)
            
            return {
                'quality_score': base_quality,
                'final_confidence': final_confidence,
                'tech_score': scores[0],
                'ecosystem_score': scores[1], 
                'community_score': scores[2],
                'adoption_score': scores[3],
                'technical_data': technical_data,
                'ai_result': ai_result,
                'explanation': explanation,
                'confidence_source': confidence_explanation,
                'ai_enhanced': bool(ai_result and ai_result.get('ai_used', False)) if 'ai_result' in locals() else False,
            }
            
        except Exception as e:
            logger.error(f"품질 점수 계산 실패 {symbol}: {e}")
            return {
                'quality_score': 0.5, 'final_confidence': 0.5,
                'tech_score': 0.5, 'ecosystem_score': 0.5,
                'community_score': 0.5, 'adoption_score': 0.5,
                'technical_data': {}, 'ai_result': None, 'explanation': '기본등급',
                'confidence_source': '기본분석', 'ai_enhanced': False
            }
    
    def _calculate_technical_indicators(self, market_data: Dict, volume_rank: int) -> Dict:
        """기술적 지표 계산"""
        try:
            ohlcv = market_data.get('ohlcv')
            if ohlcv is None or len(ohlcv) < 20:
                return self._default_technical_data(volume_rank)
            
            # RSI 계산
            rsi = self._calculate_rsi(ohlcv['close'])
            
            # MACD 신호
            macd_signal = self._calculate_macd_signal(ohlcv['close'])
            
            # 볼린저 밴드 위치
            bollinger_position = self._calculate_bollinger_position(ohlcv['close'])
            
            # 거래량 트렌드
            volume_trend = self._calculate_volume_trend(ohlcv['volume'])
            
            # 모멘텀
            momentum_7d = ((ohlcv['close'].iloc[-1] / ohlcv['close'].iloc[-8]) - 1) * 100 if len(ohlcv) >= 8 else 0
            
            return {
                'rsi': rsi,
                'macd_signal': macd_signal,
                'bollinger_position': bollinger_position,
                'volume_trend': volume_trend,
                'momentum_7d': momentum_7d,
                'volume_rank': volume_rank
            }
        except Exception as e:
            logger.debug(f"기술적 지표 계산 실패: {e}")
            return self._default_technical_data(volume_rank)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            loss = loss.replace(0, 0.0001)  # 0으로 나누기 방지
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_macd_signal(self, prices: pd.Series) -> str:
        """MACD 신호"""
        try:
            if len(prices) < 26:
                return 'neutral'
                
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            
            if len(macd_line) > 0 and len(signal_line) > 0:
                return 'bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'bearish'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> str:
        """볼린저 밴드 위치"""
        try:
            if len(prices) < period:
                return 'middle'
            
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            current_price = prices.iloc[-1]
            upper = upper_band.iloc[-1]
            lower = lower_band.iloc[-1]
            
            if current_price > upper:
                return 'upper'
            elif current_price < lower:
                return 'lower'
            else:
                return 'middle'
        except:
            return 'middle'
    
    def _calculate_volume_trend(self, volumes: pd.Series) -> str:
        """거래량 트렌드"""
        try:
            if len(volumes) < 7:
                return 'normal'
            
            recent_avg = volumes.tail(3).mean()
            past_avg = volumes.tail(10).head(7).mean()
            
            if recent_avg > past_avg * 1.5:
                return 'increasing'
            elif recent_avg < past_avg * 0.7:
                return 'decreasing'
            else:
                return 'normal'
        except:
            return 'normal'
    
    def _calculate_volume_bonus(self, volume_krw: float) -> float:
        """거래량 기반 보너스 계산"""
        if volume_krw >= 100_000_000_000: 
            return 0.05
        elif volume_krw >= 50_000_000_000: 
            return 0.03
        elif volume_krw >= 10_000_000_000: 
            return 0.01
        else: 
            return 0.0
    
    def _default_technical_data(self, volume_rank: int) -> Dict:
        """기본 기술적 데이터"""
        return {
            'rsi': 50.0,
            'macd_signal': 'neutral',
            'bollinger_position': 'middle',
            'volume_trend': 'normal',
            'momentum_7d': 0.0,
            'volume_rank': volume_rank
        }
    
    def _generate_explanation(self, coin: str, scores: List[float], final_score: float, ai_result: Dict = None) -> str:
        """설명 생성"""
        features = []
        if scores[0] > 0.85: features.append("최고급기술")
        if scores[1] > 0.85: features.append("강력생태계")
        if scores[2] > 0.80: features.append("활발커뮤니티")
        if scores[3] > 0.80: features.append("높은채택도")
        
        if final_score > 0.8: grade = "S급"
        elif final_score > 0.7: grade = "A급"  
        elif final_score > 0.6: grade = "B급"
        else: grade = "C급"
        
        base_explanation = f"{grade} | " + " | ".join(features) if features else f"{grade} | 기본등급"
        
        # AI 기술분석 결과 추가
        if ai_result and ai_result.get('ai_used'):
            ai_signal = ai_result.get('technical_signal', 'neutral')
            base_explanation += f" | AI기술분석: {ai_signal}"
        
        return base_explanation

# ============================================================================
# 🌊 QUANTUM CYCLE MATRIX - 27개 미시사이클 감지
# ============================================================================
class QuantumCycleMatrix:
    """양자역학 스타일 시장 사이클 감지기"""
    
    async def detect_quantum_cycle(self) -> Dict:
        """양자 사이클 매트릭스 감지"""
        try:
            # BTC 데이터 수집
            btc_data = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=90)
            if btc_data is None or len(btc_data) < 60:
                return self._default_cycle_state()
            
            # 3차원 상태 분석
            macro_state = self._detect_macro_cycle(btc_data)
            meso_state = self._detect_meso_cycle(btc_data)  
            micro_state = self._detect_micro_cycle(btc_data)
            
            # 사이클 강도 계산
            cycle_strength = self._calculate_cycle_strength(btc_data)
            
            # 최적 사이클 결정
            optimal_cycle = self._determine_optimal_cycle(macro_state, meso_state, micro_state)
            
            return {
                'cycle': optimal_cycle,
                'macro': macro_state,
                'meso': meso_state, 
                'micro': micro_state,
                'strength': cycle_strength,
                'confidence': min(0.95, 0.5 + cycle_strength * 0.5)
            }
            
        except Exception as e:
            logger.error(f"양자 사이클 감지 실패: {e}")
            return self._default_cycle_state()
    
    def _detect_macro_cycle(self, data: pd.DataFrame) -> str:
        """거시 사이클 (14일 기준)"""
        try:
            ma7 = data['close'].rolling(7).mean()
            ma14 = data['close'].rolling(14).mean()
            current_price = data['close'].iloc[-1]
            
            if current_price > ma7.iloc[-1] > ma14.iloc[-1]:
                return 'bull'
            elif current_price < ma7.iloc[-1] < ma14.iloc[-1]:
                return 'bear'
            else:
                return 'sideways'
        except:
            return 'sideways'
    
    def _detect_meso_cycle(self, data: pd.DataFrame) -> str:
        """중기 사이클 (7일 기준)"""
        try:
            high_7 = data['high'].rolling(7).max().iloc[-1]
            low_7 = data['low'].rolling(7).min().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            if high_7 == low_7:  # 0으로 나누기 방지
                return 'range'
                
            position = (current_price - low_7) / (high_7 - low_7)
            
            if position > 0.7:
                return 'uptrend'
            elif position < 0.3:
                return 'downtrend'
            else:
                return 'range'
        except:
            return 'range'
    
    def _detect_micro_cycle(self, data: pd.DataFrame) -> str:
        """미시 사이클 (3일 기준)"""
        try:
            recent_returns = data['close'].pct_change().tail(3)
            if len(recent_returns) < 3:
                return 'stable'
                
            volatility = recent_returns.std()
            momentum = recent_returns.mean()
            
            if abs(momentum) > volatility * 1.5:
                return 'momentum'
            elif volatility > 0.03:  # 3% 이상 변동성
                return 'reversal'
            else:
                return 'stable'
        except:
            return 'stable'
    
    def _calculate_cycle_strength(self, data: pd.DataFrame) -> float:
        """사이클 강도 계산"""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < 10:
                return 0.5
            
            # 변동성과 트렌드 강도 조합
            volatility = returns.std()
            trend_strength = abs(returns.mean()) / (volatility + 0.001)
            
            return min(1.0, trend_strength)
        except:
            return 0.5
    
    def _determine_optimal_cycle(self, macro: str, meso: str, micro: str) -> str:
        """최적 사이클 결정"""
        # 간단한 규칙 기반 매핑
        if macro == 'bull' and meso == 'uptrend':
            return 'strong_bull'
        elif macro == 'bear' and meso == 'downtrend':
            return 'strong_bear'
        elif micro == 'momentum':
            return 'momentum_phase'
        elif micro == 'reversal':
            return 'reversal_phase'
        else:
            return 'accumulation'
    
    def _default_cycle_state(self) -> Dict:
        """기본 상태"""
        return {
            'cycle': 'accumulation',
            'macro': 'sideways',
            'meso': 'range',
            'micro': 'stable',
            'strength': 0.5,
            'confidence': 0.5
        }

# ============================================================================
# ⚡ FRACTAL FILTERING PIPELINE - 다차원 필터링
# ============================================================================
class FractalFilteringPipeline:
    """프랙탈 차원 기반 다단계 필터링"""
    
    def __init__(self, min_volume: float):
        self.min_volume = min_volume
    
    async def execute_fractal_filtering(self, all_tickers: List[str]) -> List[Dict]:
        """프랙탈 파이프라인 실행"""
        logger.info("⚡ 프랙탈 필터링 파이프라인 시작")
        
        # 1단계: 원시 데이터 수집
        raw_data = await self._collect_raw_data(all_tickers)
        logger.info(f"원시 데이터: {len(raw_data)}개")
        
        if not raw_data:
            return []
        
        # 2단계: 단계별 필터링
        current_candidates = raw_data
        
        # 거래량 필터
        current_candidates = self._volume_filter(current_candidates)
        logger.info(f"거래량 필터: {len(current_candidates)}개")
        
        # 안정성 필터
        current_candidates = self._stability_filter(current_candidates)
        logger.info(f"안정성 필터: {len(current_candidates)}개")
        
        # 모멘텀 필터
        current_candidates = self._momentum_filter(current_candidates)
        logger.info(f"모멘텀 필터: {len(current_candidates)}개")
        
        # 기술적 필터
        current_candidates = self._technical_filter(current_candidates)
        logger.info(f"기술적 필터: {len(current_candidates)}개")
        
        # 최종 선별 (상위 20개)
        return current_candidates[:20]
    
    async def _collect_raw_data(self, tickers: List[str]) -> List[Dict]:
        """원시 데이터 수집"""
        valid_coins = []
        
        for i, ticker in enumerate(tickers):
            try:
                if i % 50 == 0:  # 진행상황 로그
                    logger.info(f"데이터 수집 진행: {i}/{len(tickers)}")
                
                price = pyupbit.get_current_price(ticker)
                if not price or price < 1:
                    continue
                    
                ohlcv = pyupbit.get_ohlcv(ticker, interval="day", count=30)
                if ohlcv is None or len(ohlcv) < 30:
                    continue
                
                volume_krw = ohlcv.iloc[-1]['volume'] * price
                if volume_krw < self.min_volume:
                    continue
                
                valid_coins.append({
                    'symbol': ticker,
                    'price': price,
                    'volume_krw': volume_krw,
                    'ohlcv': ohlcv,
                    'raw_score': volume_krw
                })
                
            except Exception as e:
                logger.debug(f"{ticker} 수집 실패: {e}")
                continue
        
        return sorted(valid_coins, key=lambda x: x['raw_score'], reverse=True)
    
    def _volume_filter(self, candidates: List[Dict]) -> List[Dict]:
        """거래량 필터"""
        filtered = []
        for candidate in candidates:
            try:
                volumes = candidate['ohlcv']['volume'].tail(7)
                if len(volumes) >= 7:
                    cv = volumes.std() / volumes.mean() if volumes.mean() > 0 else 999
                    if cv < 2.0:  # 변동계수 2.0 이하
                        candidate['volume_stability'] = 1 / (1 + cv)
                        filtered.append(candidate)
            except:
                continue
        
        return sorted(filtered, key=lambda x: x.get('volume_stability', 0), reverse=True)[:100]
    
    def _stability_filter(self, candidates: List[Dict]) -> List[Dict]:
        """안정성 필터"""
        filtered = []
        for candidate in candidates:
            try:
                prices = candidate['ohlcv']['close'].tail(7)
                if len(prices) >= 7:
                    cv = prices.std() / prices.mean() if prices.mean() > 0 else 999
                    if cv < 0.3:  # 가격 변동계수 30% 이하
                        candidate['price_stability'] = 1 / (1 + cv)
                        filtered.append(candidate)
            except:
                continue
        
        return sorted(filtered, key=lambda x: x.get('price_stability', 0), reverse=True)[:80]
    
    def _momentum_filter(self, candidates: List[Dict]) -> List[Dict]:
        """모멘텀 필터"""
        for candidate in candidates:
            try:
                prices = candidate['ohlcv']['close']
                if len(prices) >= 30:
                    momentum_7d = (prices.iloc[-1] / prices.iloc[-8] - 1) * 100
                    momentum_30d = (prices.iloc[-1] / prices.iloc[-31] - 1) * 100
                    candidate['momentum_score'] = (momentum_7d * 0.7 + momentum_30d * 0.3) / 100
                else:
                    candidate['momentum_score'] = 0
            except:
                candidate['momentum_score'] = 0
        
        return sorted(candidates, key=lambda x: x.get('momentum_score', 0), reverse=True)[:60]
    
    def _technical_filter(self, candidates: List[Dict]) -> List[Dict]:
        """기술적 필터"""
        for candidate in candidates:
            try:
                ohlcv = candidate['ohlcv']
                rsi = self._calculate_rsi(ohlcv['close'])
                macd_score = self._calculate_macd_score(ohlcv['close'])
                candidate['technical_score'] = (rsi/100 + macd_score) / 2
            except:
                candidate['technical_score'] = 0.5
        
        return sorted(candidates, key=lambda x: x.get('technical_score', 0), reverse=True)[:40]
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # 0으로 나누기 방지
            loss = loss.replace(0, 0.0001)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_macd_score(self, prices: pd.Series) -> float:
        """MACD 점수"""
        try:
            if len(prices) < 26:
                return 0.5
                
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            
            if len(macd_line) > 0 and len(signal_line) > 0:
                return 0.8 if macd_line.iloc[-1] > signal_line.iloc[-1] else 0.2
            else:
                return 0.5
        except:
            return 0.5

# ============================================================================
# 💎 DIAMOND HAND ALGORITHM - 켈리공식 기반 분할매매 (월 5-7% 최적화)
# ============================================================================
class DiamondHandAlgorithm:
    """켈리 공식 기반 다이아몬드 핸드 알고리즘 (월 5-7% 최적화)"""
    
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
    
    async def calculate_diamond_strategy(self, symbol: str, price: float, confidence: float, 
                                       cycle: str, quality_score: float, ai_confidence: float = None) -> Dict:
        """다이아몬드 핸드 전략 계산 (월 5-7% 최적화)"""
        try:
            # 켈리 비율 계산 (AI 확신도 고려)
            final_confidence = ai_confidence if ai_confidence is not None else confidence
            kelly_fraction = self._kelly_criterion(final_confidence, quality_score)
            
            # 감정 팩터
            emotion_factor = self._emotion_factor(cycle, final_confidence)
            
            # 총 투자 금액
            base_investment = self.portfolio_value * kelly_fraction * emotion_factor
            total_investment = min(base_investment, self.portfolio_value * 0.15)  # 최대 15%
            
            # 3단계 분할
            stage_amounts = [
                total_investment * 0.4,  # 1단계 40%
                total_investment * 0.35, # 2단계 35%
                total_investment * 0.25  # 3단계 25%
            ]
            
            # 진입가격 (현재가 기준)
            entry_prices = [
                price,           # 즉시 진입
                price * 0.95,    # -5% 추가 진입
                price * 0.90     # -10% 추가 진입
            ]
            
            # ✅ 월 5-7% 최적화: 0차 익절 추가 + 3차 익절 삭제 + 타이트한 손절
            if quality_score >= 0.8:  # 고품질 (BTC, ETH급)
                take_profits = [
                    price * 1.06,  # 0차 익절 (+6%, 20% 매도) ← 새로 추가
                    price * 1.15,  # 1차 익절 (+15%, 30% 매도)
                    price * 1.25   # 2차 익절 (+25%, 50% 매도)
                    # 3차 익절 삭제 (무제한 홀딩)
                ]
                stop_loss = price * 0.95  # -5% 손절 (타이트)

            elif quality_score >= 0.6:  # 중품질
                take_profits = [
                    price * 1.05,  # 0차 익절 (+5%, 20% 매도)
                    price * 1.12,  # 1차 익절 (+12%, 30% 매도)
                    price * 1.20   # 2차 익절 (+20%, 50% 매도)
                ]
                stop_loss = price * 0.93  # -7% 손절

            else:  # 저품질
                take_profits = [
                    price * 1.04,  # 0차 익절 (+4%, 25% 매도)
                    price * 1.10,  # 1차 익절 (+10%, 35% 매도)
                    price * 1.15   # 2차 익절 (+15%, 40% 매도)
                ]
                stop_loss = price * 0.92  # -8% 손절
            
            return {
                'symbol': symbol,
                'total_investment': total_investment,
                'kelly_fraction': kelly_fraction,
                'emotion_factor': emotion_factor,
                'ai_boost': ai_confidence is not None,
                'stage_amounts': stage_amounts,
                'entry_prices': entry_prices,
                'take_profits': take_profits,
                'stop_loss': stop_loss,
                'portfolio_weight': (total_investment / self.portfolio_value) * 100
            }
            
        except Exception as e:
            logger.error(f"다이아몬드 전략 실패 {symbol}: {e}")
            return self._fallback_strategy(symbol, price)
    
    def _kelly_criterion(self, confidence: float, quality: float) -> float:
        """켈리 공식 (단순화)"""
        win_prob = (confidence + quality) / 2
        kelly = max(0.01, min(0.25, win_prob * 0.3))  # 최대 25%
        return kelly
    
    def _emotion_factor(self, cycle: str, confidence: float) -> float:
        """감정 팩터"""
        cycle_factors = {
            'strong_bull': 1.2,
            'momentum_phase': 1.1,
            'accumulation': 1.0,
            'reversal_phase': 0.9,
            'strong_bear': 0.8
        }
        
        base_factor = cycle_factors.get(cycle, 1.0)
        confidence_boost = 0.8 + (confidence * 0.4)
        
        return base_factor * confidence_boost
    
    def _fallback_strategy(self, symbol: str, price: float) -> Dict:
        """기본 전략"""
        base_investment = self.portfolio_value * 0.05
        return {
            'symbol': symbol, 'total_investment': base_investment,
            'kelly_fraction': 0.05, 'emotion_factor': 1.0, 'ai_boost': False,
            'stage_amounts': [base_investment * 0.5, base_investment * 0.3, base_investment * 0.2],
            'entry_prices': [price, price * 0.95, price * 0.90],
            'take_profits': [price * 1.05, price * 1.15, price * 1.25],
            'stop_loss': price * 0.92, 'portfolio_weight': 5.0
        }

# ============================================================================
# 📊 전설급 메인 시그널 클래스
# ============================================================================
@dataclass
class LegendarySignal:
    """전설급 시그널"""
    symbol: str
    action: str
    confidence: float
    price: float
    
    # 분석 결과
    neural_quality: float
    explanation: str
    quantum_cycle: str
    cycle_confidence: float
    
    # 투자 전략
    kelly_fraction: float
    emotion_factor: float
    total_investment: float
    
    # 실행 계획
    entry_prices: List[float]
    stage_amounts: List[float] 
    take_profits: List[float]
    stop_loss: float
    
    # AI 분석 결과
    ai_enhanced: bool
    ai_confidence: Optional[float]
    technical_signal: Optional[str]
    
    # 종합 점수
    legendary_score: float
    
    timestamp: datetime

# ============================================================================
# 🎯 POSITION MANAGER - 포지션 관리자
# ============================================================================
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    total_quantity: float
    avg_price: float
    current_stage: int  # 1, 2, 3 단계
    stage_quantities: List[float]  # 각 단계별 수량
    stage_prices: List[float]      # 각 단계별 진입가
    target_take_profits: List[float]
    stop_loss: float
    unrealized_pnl: float
    created_at: datetime
    last_updated: datetime

class PositionManager:
    """포지션 관리 시스템"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.position_file = "positions.json"
        self.load_positions()
    
    def add_position(self, signal: LegendarySignal, stage: int, quantity: float, executed_price: float):
        """포지션 추가/업데이트"""
        symbol = signal.symbol
        
        if symbol not in self.positions:
            # 새 포지션 생성
            self.positions[symbol] = Position(
                symbol=symbol,
                total_quantity=quantity,
                avg_price=executed_price,
                current_stage=stage,
                stage_quantities=[0.0, 0.0, 0.0],
                stage_prices=[0.0, 0.0, 0.0],
                target_take_profits=signal.take_profits.copy(),
                stop_loss=signal.stop_loss,
                unrealized_pnl=0.0,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            self.positions[symbol].stage_quantities[stage-1] = quantity
            self.positions[symbol].stage_prices[stage-1] = executed_price
        else:
            # 기존 포지션 업데이트
            pos = self.positions[symbol]
            old_total_cost = pos.total_quantity * pos.avg_price
            new_cost = quantity * executed_price
            
            pos.total_quantity += quantity
            pos.avg_price = (old_total_cost + new_cost) / pos.total_quantity
            pos.current_stage = max(pos.current_stage, stage)
            pos.stage_quantities[stage-1] += quantity
            pos.stage_prices[stage-1] = executed_price
            pos.last_updated = datetime.now()
        
        self.save_positions()
        logger.info(f"포지션 업데이트: {symbol} - 단계 {stage}, 수량 {quantity}")
    
    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """미실현 손익 업데이트"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.unrealized_pnl = (current_price - pos.avg_price) * pos.total_quantity
            pos.last_updated = datetime.now()
    
    def remove_position(self, symbol: str, quantity: float = None):
        """포지션 제거 (부분/전체)"""
        if symbol not in self.positions:
            return
        
        if quantity is None:
            # 전체 제거
            del self.positions[symbol]
            logger.info(f"포지션 전체 제거: {symbol}")
        else:
            # 부분 제거
            pos = self.positions[symbol]
            if quantity >= pos.total_quantity:
                del self.positions[symbol]
                logger.info(f"포지션 전체 제거: {symbol}")
            else:
                pos.total_quantity -= quantity
                pos.last_updated = datetime.now()
                logger.info(f"포지션 부분 제거: {symbol} - {quantity}")
        
        self.save_positions()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """포지션 조회"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """모든 포지션 조회"""
        return list(self.positions.values())
    
    def save_positions(self):
        """포지션 저장"""
        try:
            serializable_positions = {}
            for symbol, pos in self.positions.items():
                serializable_positions[symbol] = {
                    'symbol': pos.symbol,
                    'total_quantity': pos.total_quantity,
                    'avg_price': pos.avg_price,
                    'current_stage': pos.current_stage,
                    'stage_quantities': pos.stage_quantities,
                    'stage_prices': pos.stage_prices,
                    'target_take_profits': pos.target_take_profits,
                    'stop_loss': pos.stop_loss,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'created_at': pos.created_at.isoformat(),
                    'last_updated': pos.last_updated.isoformat()
                }
            
            with open(self.position_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_positions, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"포지션 저장 실패: {e}")
    
    def load_positions(self):
        """포지션 로드"""
        try:
            if os.path.exists(self.position_file):
                with open(self.position_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for symbol, pos_data in data.items():
                    self.positions[symbol] = Position(
                        symbol=pos_data['symbol'],
                        total_quantity=pos_data['total_quantity'],
                        avg_price=pos_data['avg_price'],
                        current_stage=pos_data['current_stage'],
                        stage_quantities=pos_data['stage_quantities'],
                        stage_prices=pos_data['stage_prices'],
                        target_take_profits=pos_data['target_take_profits'],
                        stop_loss=pos_data['stop_loss'],
                        unrealized_pnl=pos_data['unrealized_pnl'],
                        created_at=datetime.fromisoformat(pos_data['created_at']),
                        last_updated=datetime.fromisoformat(pos_data['last_updated'])
                    )
                
                logger.info(f"포지션 로드 완료: {len(self.positions)}개")
        except Exception as e:
            logger.error(f"포지션 로드 실패: {e}")

# ============================================================================
# 🚨 EXIT STRATEGY ENGINE - 출구 전략 엔진 (월 5-7% 최적화)
# ============================================================================
class ExitStrategyEngine:
    """실시간 매도 전략 엔진 (월 5-7% 최적화)"""
    
    def __init__(self, position_manager: PositionManager):
        self.position_manager = position_manager
        self.trailing_stop_ratio = 0.10  # 10% 트레일링 스톱
        self.profit_taken_flags = {}  # 익절 실행 추적
    
    async def check_exit_conditions(self, symbol: str, current_price: float, current_cycle: str) -> Dict:
        position = self.position_manager.get_position(symbol)
        if not position:
            return {'action': 'none', 'reason': 'no_position'}

        self.position_manager.update_unrealized_pnl(symbol, current_price)
        profit_ratio = (current_price - position.avg_price) / position.avg_price
        holding_days = (datetime.now() - position.created_at).days

        # 1. 손절 체크 (타이트한 손절)
        if current_price <= position.stop_loss:
            return {
                'action': 'sell_all',
                'reason': 'stop_loss',
                'price': current_price,
                'quantity': position.total_quantity,
                'details': f'손절 실행: {current_price} <= {position.stop_loss}'
            }

        # 2. 2주 초과시 무조건 매도
        if holding_days >= 16:  # 2주 초과시 무조건
            return {
                'action': 'sell_all',
                'reason': 'time_limit_force',
                'price': current_price,
                'quantity': position.total_quantity,
                'details': f'강제매도: {holding_days}일 초과'
            }

        # 3. 익절 체크 (0차/1차/2차)
        profit_flags = self.profit_taken_flags.get(symbol, [False, False, False])

        # 0차 익절 (4-6% 수익시 20-25% 매도)
        if (len(position.target_take_profits) >= 1 and 
            current_price >= position.target_take_profits[0] and 
            profit_ratio >= 0.04 and not profit_flags[0]):
            
            sell_ratio = 0.25 if profit_ratio < 0.05 else 0.20
            sell_quantity = position.total_quantity * sell_ratio
            
            if symbol not in self.profit_taken_flags:
                self.profit_taken_flags[symbol] = [False, False, False]
            self.profit_taken_flags[symbol][0] = True
            
            return {
                'action': 'sell_partial',
                'reason': 'take_profit_0',
                'price': current_price,
                'quantity': sell_quantity,
                'details': f'0차 익절: {profit_ratio*100:.1f}% 수익으로 {sell_ratio*100:.0f}% 매도'
            }

        # 1차 익절 (10-15% 수익시 30-35% 매도)
        if (len(position.target_take_profits) >= 2 and
            current_price >= position.target_take_profits[1] and 
            profit_ratio >= 0.10 and not profit_flags[1]):
            
            sell_ratio = 0.35 if profit_ratio < 0.12 else 0.30
            sell_quantity = position.total_quantity * sell_ratio
            
            self.profit_taken_flags[symbol][1] = True
            
            return {
                'action': 'sell_partial',
                'reason': 'take_profit_1',
                'price': current_price,
                'quantity': sell_quantity,
                'details': f'1차 익절: {profit_ratio*100:.1f}% 수익으로 {sell_ratio*100:.0f}% 매도'
            }

        # 2차 익절 (15-25% 수익시 40-50% 매도)
        if (len(position.target_take_profits) >= 3 and
            current_price >= position.target_take_profits[2] and 
            profit_ratio >= 0.15 and not profit_flags[2]):
            
            sell_ratio = 0.50 if profit_ratio < 0.20 else 0.40
            sell_quantity = position.total_quantity * sell_ratio
            
            self.profit_taken_flags[symbol][2] = True
            
            return {
                'action': 'sell_partial',
                'reason': 'take_profit_2',
                'price': current_price,
                'quantity': sell_quantity,
                'details': f'2차 익절: {profit_ratio*100:.1f}% 수익으로 {sell_ratio*100:.0f}% 매도'
            }

        # 3차 익절 삭제됨 - 무제한 홀딩!

        # 4. 사이클 변화 매도
        if profit_ratio > 0.03 and current_cycle in ['strong_bear', 'reversal_phase']:
            return {
                'action': 'sell_all',
                'reason': 'cycle_change',
                'price': current_price,
                'quantity': position.total_quantity,
                'details': f'사이클 변화 매도: {current_cycle}'
            }

        # 5. 강화된 트레일링 스톱 (40% 이후)
        if profit_ratio > 0.40:  # 40% 이상 수익시 20% 트레일링 스톱
            dynamic_stop = position.avg_price * (1 + profit_ratio - 0.20)
            if current_price <= dynamic_stop:
                return {
                    'action': 'sell_all',
                    'reason': 'trailing_stop_40',
                    'price': current_price,
                    'quantity': position.total_quantity,
                    'details': f'40%+ 트레일링 스톱: {current_price} <= {dynamic_stop}'
                }
        elif profit_ratio > 0.20:  # 20% 이상 수익시 15% 트레일링 스톱
            dynamic_stop = position.avg_price * (1 + profit_ratio - 0.15)
            if current_price <= dynamic_stop:
                return {
                    'action': 'sell_all',
                    'reason': 'trailing_stop_20',
                    'price': current_price,
                    'quantity': position.total_quantity,
                    'details': f'20%+ 트레일링 스톱: {current_price} <= {dynamic_stop}'
                }
        elif profit_ratio > 0.08:  # 8% 이상 수익시 기본 10% 트레일링 스톱
            dynamic_stop = position.avg_price * (1 + profit_ratio - self.trailing_stop_ratio)
            if current_price <= dynamic_stop:
                return {
                    'action': 'sell_all',
                    'reason': 'trailing_stop',
                    'price': current_price,
                    'quantity': position.total_quantity,
                    'details': f'트레일링 스톱: {current_price} <= {dynamic_stop}'
                }

        return {'action': 'hold', 'reason': 'no_exit_condition'}

# ============================================================================
# 🎮 TRADE EXECUTOR - 거래 실행기 (월금 매매)
# ============================================================================
class TradeExecutor:
    """거래 실행 시스템 (월금 매매)"""
    
    def __init__(self, position_manager: PositionManager, demo_mode: bool = True):
        self.position_manager = position_manager
        self.demo_mode = demo_mode  # 실제 거래 vs 시뮬레이션
        
        if not demo_mode:
            # 실제 거래용 업비트 API 초기화
            self.access_key = os.getenv('UPBIT_ACCESS_KEY')
            self.secret_key = os.getenv('UPBIT_SECRET_KEY')
            self.upbit = pyupbit.Upbit(self.access_key, self.secret_key)
    
    def is_trading_day(self, action_type: str = 'buy') -> bool:
        """거래 가능일 체크"""
        today = datetime.now().weekday()
        
        if action_type == 'buy':
            # 매수는 월요일만
            return today == 0  # 월요일
        elif action_type == 'sell':
            # 매도는 금요일 + 응급시 언제든
            return today == 4 or action_type == 'emergency_sell'  # 금요일
        else:
            return True  # 응급 매도는 언제든
    
    async def execute_buy_signal(self, signal: LegendarySignal, stage: int) -> Dict:
        """매수 신호 실행 (월요일만)"""
        try:
            # 월요일 체크
            if not self.is_trading_day('buy'):
                return {
                    'success': False, 
                    'error': 'not_trading_day',
                    'message': '매수는 월요일만 가능합니다'
                }
            
            symbol = signal.symbol
            target_price = signal.entry_prices[stage - 1]
            target_amount = signal.stage_amounts[stage - 1]
            
            current_price = pyupbit.get_current_price(symbol)
            if not current_price:
                return {'success': False, 'error': 'price_fetch_failed'}
            
            # 진입 조건 체크 (현재가가 목표가 이하일 때만)
            if current_price > target_price * 1.02:  # 2% 여유
                return {'success': False, 'error': 'price_too_high'}
            
            # 수량 계산
            quantity = target_amount / current_price
            
            if self.demo_mode:
                # 시뮬레이션 모드
                result = {
                    'success': True,
                    'symbol': symbol,
                    'stage': stage,
                    'quantity': quantity,
                    'price': current_price,
                    'amount': target_amount,
                    'type': 'demo_buy'
                }
                
                # 포지션 업데이트
                self.position_manager.add_position(signal, stage, quantity, current_price)
                
                logger.info(f"📈 [월요일] 시뮬레이션 매수: {symbol} 단계{stage} {quantity:.6f}개 @ {current_price:,.0f}원")
                
            else:
                # 실제 거래
                order = self.upbit.buy_market_order(symbol, target_amount)
                if order:
                    result = {
                        'success': True,
                        'symbol': symbol,
                        'stage': stage,
                        'order_id': order['uuid'],
                        'quantity': quantity,
                        'price': current_price,
                        'amount': target_amount,
                        'type': 'real_buy'
                    }
                    
                    # 포지션 업데이트
                    self.position_manager.add_position(signal, stage, quantity, current_price)
                    
                    logger.info(f"📈 [월요일] 실제 매수: {symbol} 단계{stage} {quantity:.6f}개 @ {current_price:,.0f}원")
                else:
                    result = {'success': False, 'error': 'order_failed'}
            
            return result
            
        except Exception as e:
            logger.error(f"매수 실행 실패 {signal.symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_sell_signal(self, symbol: str, sell_action: Dict, emergency: bool = False) -> Dict:
        """매도 신호 실행 (금요일 + 응급시)"""
        try:
            # 응급 매도가 아니라면 금요일 체크
            if not emergency and not self.is_trading_day('sell'):
                return {
                    'success': False, 
                    'error': 'not_trading_day',
                    'message': '정기 매도는 금요일만 가능합니다'
                }
            
            position = self.position_manager.get_position(symbol)
            if not position:
                return {'success': False, 'error': 'no_position'}
            
            current_price = pyupbit.get_current_price(symbol)
            if not current_price:
                return {'success': False, 'error': 'price_fetch_failed'}
            
            sell_quantity = sell_action['quantity']
            sell_amount = sell_quantity * current_price
            
            day_type = "[응급]" if emergency else "[금요일]"
            
            if self.demo_mode:
                # 시뮬레이션 모드
                result = {
                    'success': True,
                    'symbol': symbol,
                    'quantity': sell_quantity,
                    'price': current_price,
                    'amount': sell_amount,
                    'reason': sell_action['reason'],
                    'type': 'demo_sell'
                }
                
                # 포지션 업데이트
                if sell_action['action'] == 'sell_all':
                    self.position_manager.remove_position(symbol)
                else:
                    self.position_manager.remove_position(symbol, sell_quantity)
                
                # 수익률 계산
                profit = (current_price - position.avg_price) * sell_quantity
                profit_ratio = (current_price - position.avg_price) / position.avg_price * 100
                
                logger.info(f"📉 {day_type} 시뮬레이션 매도: {symbol} {sell_quantity:.6f}개 @ {current_price:,.0f}원")
                logger.info(f"💰 손익: {profit:+,.0f}원 ({profit_ratio:+.1f}%) - {sell_action['reason']}")
                
            else:
                # 실제 거래
                order = self.upbit.sell_market_order(symbol, sell_quantity)
                if order:
                    result = {
                        'success': True,
                        'symbol': symbol,
                        'order_id': order['uuid'],
                        'quantity': sell_quantity,
                        'price': current_price,
                        'amount': sell_amount,
                        'reason': sell_action['reason'],
                        'type': 'real_sell'
                    }
                    
                    # 포지션 업데이트
                    if sell_action['action'] == 'sell_all':
                        self.position_manager.remove_position(symbol)
                    else:
                        self.position_manager.remove_position(symbol, sell_quantity)
                    
                    profit = (current_price - position.avg_price) * sell_quantity
                    profit_ratio = (current_price - position.avg_price) / position.avg_price * 100
                    
                    logger.info(f"📉 {day_type} 실제 매도: {symbol} {sell_quantity:.6f}개 @ {current_price:,.0f}원")
                    logger.info(f"💰 손익: {profit:+,.0f}원 ({profit_ratio:+.1f}%) - {sell_action['reason']}")
                else:
                    result = {'success': False, 'error': 'order_failed'}
            
            return result
            
        except Exception as e:
            logger.error(f"매도 실행 실패 {symbol}: {e}")
            return {'success': False, 'error': str(e)}

# ============================================================================
# 📊 REAL-TIME MONITOR - 실시간 모니터 (월금 매매 최적화)
# ============================================================================
class RealTimeMonitor:
    """실시간 모니터링 시스템 (월금 매매 최적화)"""
    
    def __init__(self, position_manager: PositionManager, exit_engine: ExitStrategyEngine, 
                 trade_executor: TradeExecutor, quantum_cycle: QuantumCycleMatrix):
        self.position_manager = position_manager
        self.exit_engine = exit_engine
        self.trade_executor = trade_executor
        self.quantum_cycle = quantum_cycle
        self.monitoring = False
    
    async def start_monitoring(self, check_interval: int = 180):  # 3분마다
        """실시간 모니터링 시작 (월금 매매 고려)"""
        self.monitoring = True
        logger.info("🔄 실시간 모니터링 시작 (월금 매매 모드)")
        
        while self.monitoring:
            try:
                # 현재 요일 체크
                current_weekday = datetime.now().weekday()
                weekday_names = ['월', '화', '수', '목', '금', '토', '일']
                today_name = weekday_names[current_weekday]
                
                # 현재 시장 사이클 확인
                cycle_info = await self.quantum_cycle.detect_quantum_cycle()
                current_cycle = cycle_info['cycle']
                
                # 모든 포지션 체크
                positions = self.position_manager.get_all_positions()
                
                if positions:
                    logger.info(f"🔍 [{today_name}] 포지션 모니터링: {len(positions)}개 ({current_cycle})")
                
                for position in positions:
                    try:
                        # 현재가 조회
                        current_price = pyupbit.get_current_price(position.symbol)
                        if not current_price:
                            continue
                        
                        # 매도 조건 체크
                        exit_action = await self.exit_engine.check_exit_conditions(
                            position.symbol, current_price, current_cycle
                        )
                        
                        # 응급 매도 조건 체크
                        is_emergency = exit_action['reason'] in [
                            'stop_loss', 'cycle_change', 'trailing_stop_40'
                        ]
                        
                        # 매도 실행
                        if exit_action['action'] in ['sell_all', 'sell_partial']:
                            if is_emergency or current_weekday == 4:  # 응급 또는 금요일
                                logger.info(f"🚨 [{today_name}] 매도 신호: {position.symbol} - {exit_action['reason']}")
                                
                                # 매도 실행
                                sell_result = await self.trade_executor.execute_sell_signal(
                                    position.symbol, exit_action, emergency=is_emergency
                                )
                                
                                if sell_result['success']:
                                    logger.info(f"✅ 매도 성공: {position.symbol}")
                                else:
                                    logger.error(f"❌ 매도 실패: {position.symbol} - {sell_result.get('error')}")
                            else:
                                logger.info(f"⏳ [{today_name}] 매도 대기: {position.symbol} - {exit_action['reason']} (금요일 대기)")
                        
                        # 미실현 손익 로그 (중요한 변동만)
                        pnl_ratio = (current_price - position.avg_price) / position.avg_price * 100
                        if abs(pnl_ratio) > 5:  # 5% 이상 변동 시에만 로그
                            logger.info(f"💹 {position.symbol}: {pnl_ratio:+.1f}% @ {current_price:,.0f}원")
                        
                    except Exception as e:
                        logger.error(f"개별 포지션 모니터링 실패 {position.symbol}: {e}")
                        continue
                
                # 다음 체크까지 대기
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(60)  # 1분 후 재시도
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        logger.info("⏹️ 실시간 모니터링 중지")
# ============================================================================
# 🏆 LEGENDARY QUANT MASTER - 전설급 통합 시스템 (OpenAI 기술분석 최적화)
# ============================================================================
class LegendaryQuantMaster:
    """전설급 5대 시스템 + OpenAI 기술분석 최적화 마스터 (월 5-7% 목표)"""
    
    def __init__(self, portfolio_value: float = 100_000_000, min_volume: float = 10_000_000_000, 
                 demo_mode: bool = True, openai_api_key: str = None):
        self.portfolio_value = portfolio_value
        self.min_volume = min_volume
        self.demo_mode = demo_mode
        
        # OpenAI 기술분석기 초기화
        self.openai_analyzer = OpenAITechnicalAnalyzer(openai_api_key)
        
        # 전설급 5대 엔진 초기화 (OpenAI 통합)
        self.neural_engine = NeuralQualityEngine(self.openai_analyzer)
        self.quantum_cycle = QuantumCycleMatrix()
        self.fractal_filter = FractalFilteringPipeline(min_volume)
        self.diamond_algorithm = DiamondHandAlgorithm(portfolio_value)
        
        # 매도 시스템 초기화
        self.position_manager = PositionManager()
        self.exit_engine = ExitStrategyEngine(self.position_manager)
        self.trade_executor = TradeExecutor(self.position_manager, demo_mode)
        self.monitor = RealTimeMonitor(self.position_manager, self.exit_engine, self.trade_executor, self.quantum_cycle)
        
        # 설정
        self.target_portfolio_size = 8
    
    def is_trading_day(self) -> bool:
        """월요일(0) 또는 금요일(4)만 거래"""
        return datetime.now().weekday() in [0, 4]
    
    async def execute_legendary_strategy(self) -> List[LegendarySignal]:
        """전설급 전략 실행 (월 5-7% 최적화) + OpenAI 기술분석"""
        logger.info("🏆 LEGENDARY QUANT STRATEGY + OpenAI 기술분석 최적화 시작")
        
        # 거래일 체크
        current_weekday = datetime.now().weekday()
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        today_name = weekday_names[current_weekday]
        
        if not self.is_trading_day():
            logger.info(f"⏸️ [{today_name}] 비거래일: 모니터링만 실행")
            return []
        
        logger.info(f"📈 [{today_name}] 거래일: 전설급 전략 실행 (OpenAI {'활성화' if self.openai_analyzer.enabled else '비활성화'})")
        
        try:
            # 1단계: 양자 사이클 감지
            logger.info("🌊 양자 사이클 매트릭스 감지 중...")
            quantum_state = await self.quantum_cycle.detect_quantum_cycle()
            
            # 2단계: 프랙탈 필터링
            logger.info("⚡ 프랙탈 필터링 파이프라인 실행 중...")
            all_tickers = pyupbit.get_tickers(fiat="KRW")
            if not all_tickers:
                logger.error("티커 목록 조회 실패")
                return []
            
            fractal_candidates = await self.fractal_filter.execute_fractal_filtering(all_tickers)
            
            if not fractal_candidates:
                logger.error("프랙탈 필터링 결과 없음")
                return []
            
            # 3단계: 개별 코인 전설급 분석 (OpenAI 기술분석 통합)
            legendary_signals = []
            for i, candidate in enumerate(fractal_candidates[:self.target_portfolio_size], 1):
                logger.info(f"💎 전설급 분석 [{i}/{min(len(fractal_candidates), self.target_portfolio_size)}]: {candidate['symbol']} (AI {'ON' if self.openai_analyzer.enabled else 'OFF'})")
                
                signal = await self._analyze_legendary_coin(candidate, quantum_state, i)
                if signal:
                    legendary_signals.append(signal)
                
                await asyncio.sleep(0.5)  # API 제한 고려
            
            # 4단계: 최종 포트폴리오 랭킹
            legendary_signals.sort(key=lambda x: x.legendary_score, reverse=True)
            
            # 결과 요약
            buy_signals = [s for s in legendary_signals if s.action == 'BUY']
            ai_enhanced_count = sum(1 for s in buy_signals if s.ai_enhanced)
            
            logger.info(f"✨ 전설급 분석 완료: {len(legendary_signals)}개 분석, {len(buy_signals)}개 매수 신호 (AI 강화: {ai_enhanced_count}개)")
            
            # OpenAI 사용량 통계
            if self.openai_analyzer.enabled:
                usage_stats = self.openai_analyzer.get_usage_stats()
                logger.info(f"🤖 OpenAI 사용량: {usage_stats['daily_calls']}/{usage_stats['daily_limit']}회, 예상 월비용: {usage_stats['monthly_projection_krw']:.0f}원")
            
            return legendary_signals
            
        except Exception as e:
            logger.error(f"전설급 전략 실행 실패: {e}")
            return []
    
    async def _analyze_legendary_coin(self, candidate: Dict, quantum_state: Dict, volume_rank: int) -> Optional[LegendarySignal]:
        """개별 코인 전설급 분석 (OpenAI 기술분석 통합)"""
        try:
            symbol = candidate['symbol']
            price = candidate['price']
            
            # Neural Quality Engine 분석 (OpenAI 기술분석 통합)
            market_data = {
                'volume_24h_krw': candidate['volume_krw'],
                'price': price,
                'ohlcv': candidate['ohlcv']
            }
            neural_result = await self.neural_engine.neural_quality_score(symbol, market_data, volume_rank)
            
            # Diamond Hand Algorithm 분석 (AI 확신도 활용)
            ai_confidence = neural_result.get('final_confidence')
            diamond_result = await self.diamond_algorithm.calculate_diamond_strategy(
                symbol, price, neural_result['quality_score'], 
                quantum_state['cycle'], neural_result['quality_score'], ai_confidence
            )
            
            # 종합 점수 계산 (AI 가중치 추가)
            base_score = (
                neural_result['quality_score'] * 0.30 +      # Neural Quality
                quantum_state['confidence'] * 0.25 +         # Quantum Cycle  
                candidate.get('technical_score', 0.5) * 0.25 +  # Technical
                candidate.get('momentum_score', 0.5) * 0.20     # Momentum
            )
            
            # AI 기술분석 보너스 (있는 경우)
            ai_bonus = 0.0
            ai_result = neural_result.get('ai_result')
            if ai_result and ai_result.get('ai_used'):
                # AI 기술분석이 긍정적이면 보너스
                if ai_result.get('technical_signal') == 'strong':
                    ai_bonus = 0.05  # 5% 보너스
                elif ai_result.get('technical_signal') == 'weak':
                    ai_bonus = -0.05  # 5% 페널티
            
            legendary_score = max(0.0, min(1.0, base_score + ai_bonus))
            
            # 액션 결정 (AI 보정 반영)
            if legendary_score >= 0.70:
                action = 'BUY'
            elif legendary_score <= 0.30:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # AI 기술분석이 약하면 액션 하향 조정
            if ai_result and ai_result.get('technical_signal') == 'weak':
                if action == 'BUY':
                    action = 'HOLD'
                    logger.info(f"🤖 {symbol}: AI 기술분석 약함으로 BUY → HOLD 조정")
            
            # 전설급 시그널 생성
            signal = LegendarySignal(
                symbol=symbol,
                action=action,
                confidence=neural_result['quality_score'],
                price=price,
                neural_quality=neural_result['quality_score'],
                explanation=neural_result['explanation'],
                quantum_cycle=quantum_state['cycle'],
                cycle_confidence=quantum_state['confidence'],
                kelly_fraction=diamond_result['kelly_fraction'],
                emotion_factor=diamond_result['emotion_factor'],
                total_investment=diamond_result['total_investment'],
                entry_prices=diamond_result['entry_prices'],
                stage_amounts=diamond_result['stage_amounts'],
                take_profits=diamond_result['take_profits'],
                stop_loss=diamond_result['stop_loss'],
                ai_enhanced=neural_result.get('ai_enhanced', False),
                ai_confidence=neural_result.get('final_confidence'),
                technical_signal=ai_result.get('technical_signal') if ai_result else None,
                legendary_score=legendary_score,
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"개별 코인 전설급 분석 실패 {candidate['symbol']}: {e}")
            return None
    
    def print_legendary_results(self, signals: List[LegendarySignal]):
        """전설급 결과 출력 (월 5-7% 최적화) + OpenAI 기술분석"""
        print("\n" + "="*90)
        print("🏆 LEGENDARY QUANT STRATEGY + OpenAI 기술분석 최적화 🏆")
        print("="*90)
        
        if not signals:
            print("❌ 분석된 신호가 없습니다.")
            return
        
        buy_signals = [s for s in signals if s.action == 'BUY']
        total_investment = sum(s.total_investment for s in buy_signals)
        ai_enhanced_count = sum(1 for s in buy_signals if s.ai_enhanced)
        
        # 현재 요일 정보
        current_weekday = datetime.now().weekday()
        weekday_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        today_name = weekday_names[current_weekday]
        is_trading_day = current_weekday in [0, 4]
        
        print(f"\n📊 전략 요약:")
        print(f"   분석 코인: {len(signals)}개")
        print(f"   매수 신호: {len(buy_signals)}개") 
        print(f"   AI 기술분석: {ai_enhanced_count}개 ({self.openai_analyzer.enabled and 'OpenAI 활성화' or 'OpenAI 비활성화'})")
        print(f"   총 투자금: {total_investment:,.0f}원")
        print(f"   포트폴리오 비중: {(total_investment/self.portfolio_value)*100:.1f}%")
        print(f"   운영 모드: {'시뮬레이션' if self.demo_mode else '실제거래'}")
        print(f"   오늘: {today_name} ({'거래일' if is_trading_day else '비거래일'})")
        
        if signals:
            print(f"\n🌊 양자 사이클 상태:")
            print(f"   현재 사이클: {signals[0].quantum_cycle}")
            print(f"   신뢰도: {signals[0].cycle_confidence:.2f}")
        
        # OpenAI 사용량 통계
        if self.openai_analyzer.enabled:
            usage_stats = self.openai_analyzer.get_usage_stats()
            print(f"\n🤖 OpenAI 기술분석 시스템:")
            print(f"   • 모델: GPT-3.5-Turbo (기술분석 전용)")
            print(f"   • 스마트 호출: 애매한 확신도(0.4-0.7)에서만 사용")
            print(f"   • 오늘 사용량: {usage_stats['daily_calls']}/{usage_stats['daily_limit']}회")
            print(f"   • 예상 월비용: {usage_stats['monthly_projection_krw']:.0f}원")
            print(f"   • 비용 효율성: {usage_stats['efficiency']}")
            print(f"   • AI 강화 비율: {ai_enhanced_count}/{len(buy_signals)}개")
        else:
            print(f"\n📊 기본 분석 시스템:")
            print(f"   • 상태: OpenAI 비활성화 (OPENAI_API_KEY 없음)")
            print(f"   • 기본 기술분석만 사용")
            print(f"   • 월 비용: 0원")
        
        print(f"\n✨ 월 5-7% 최적화 특징:")
        print(f"   • 0차 익절: 4-6% 수익시 20-25% 매도")
        print(f"   • 1차 익절: 10-15% 수익시 30-35% 매도") 
        print(f"   • 2차 익절: 15-25% 수익시 40-50% 매도")
        print(f"   • 3차 익절: 삭제 (무제한 홀딩)")
        print(f"   • 손절선: -5~8% (품질별 차등)")
        print(f"   • 매매일: 월요일 매수, 금요일 매도")
        print(f"   • 홀딩: 최대 2주")
        
        print(f"\n💎 전설급 매수 신호:")
        for i, signal in enumerate(buy_signals, 1):
            ai_mark = "🤖" if signal.ai_enhanced else "📊"
            ai_signal_info = f"({signal.technical_signal})" if signal.technical_signal else ""
            
            print(f"\n[{i}] {signal.symbol} {ai_mark} {ai_signal_info}")
            print(f"   전설 점수: {signal.legendary_score:.3f}")
            print(f"   AI 분석: {signal.explanation}")
            print(f"   확신도: {signal.ai_confidence:.3f}" if signal.ai_confidence else f"   확신도: {signal.confidence:.3f}")
            print(f"   켈리 비중: {signal.kelly_fraction:.1%}")
            print(f"   투자금액: {signal.total_investment:,.0f}원")
            print(f"   진입가격: {[f'{p:,.0f}' for p in signal.entry_prices]}")
            print(f"   익절가격: {[f'{p:,.0f}' for p in signal.take_profits]} (0차/1차/2차)")
            print(f"   손절가격: {signal.stop_loss:,.0f}원")
            
            # AI 기술분석 상세 정보
            if signal.ai_enhanced and signal.technical_signal:
                print(f"   🤖 AI 기술분석: {signal.technical_signal} 신호")
        
        print(f"\n📈 월 5-7% 달성 전략:")
        print(f"   • 포트폴리오 8개 중 2-3개 대박(50%+) → 월수익 견인")
        print(f"   • 나머지 4-5개 소폭수익(5-25%) → 안정성 확보")
        print(f"   • 1-2개 손실(-5~8%) → 손절로 제한")
        print(f"   • 평균 월수익: 5-7% 목표")
        print(f"   • OpenAI 기술분석으로 정확도 향상 (비용 최적화)")
        
        print("\n" + "="*90)
        print("⚡ LEGENDARY STRATEGY + AI TECHNICAL ANALYSIS - 월 5-7% 최적화 ⚡")

# ============================================================================
# 🚀 메인 실행 함수들
# ============================================================================
async def main():
    """전설급 퀀트 전략 메인 실행 (월 5-7% 최적화) + OpenAI 기술분석"""
    print("⚡ LEGENDARY QUANT STRATEGY + OpenAI 기술분석 최적화 STARTING ⚡")
    print("🧠🌊⚡💎🕸️🎯🚨🎮📊🤖 완전체 시스템 + AI 기술분석 로딩...")
    
    # OpenAI API 키 확인
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("🤖 OpenAI API 키 감지 - AI 기술분석 모드 활성화 (월 5천원 이하 최적화)")
    else:
        print("📊 OpenAI API 키 없음 - 기본 분석 모드")
    
    # 전설급 마스터 초기화 (시뮬레이션 모드 + OpenAI 기술분석)
    master = LegendaryQuantMaster(
        portfolio_value=100_000_000,  # 1억원
        min_volume=5_000_000_000,     # 50억원
        demo_mode=True,               # 시뮬레이션 모드
        openai_api_key=openai_key     # OpenAI API 키
    )
    
    try:
        # 전설급 분석 실행
        legendary_signals = await master.execute_legendary_strategy()
        
        # 결과 출력
        master.print_legendary_results(legendary_signals)
        
        return legendary_signals
        
    except KeyboardInterrupt:
        print("\n👋 전설급 전략을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류: {e}")
        logger.error(f"메인 실행 실패: {e}")

# 단일 코인 분석 함수 (OpenAI 기술분석 통합)
async def analyze_single_coin(symbol: str):
    """단일 코인 전설급 분석 (월 5-7% 최적화) + OpenAI 기술분석"""
    openai_key = os.getenv('OPENAI_API_KEY')
    master = LegendaryQuantMaster(openai_api_key=openai_key)
    
    try:
        price = pyupbit.get_current_price(symbol)
        ohlcv = pyupbit.get_ohlcv(symbol, interval="day", count=30)
        volume_krw = ohlcv.iloc[-1]['volume'] * price
        
        candidate = {
            'symbol': symbol,
            'price': price,
            'volume_krw': volume_krw,
            'ohlcv': ohlcv,
            'technical_score': 0.6,
            'momentum_score': 0.5
        }
        
        quantum_state = await master.quantum_cycle.detect_quantum_cycle()
        signal = await master._analyze_legendary_coin(candidate, quantum_state, 1)
        
        if signal:
            ai_status = "🤖 AI 기술분석" if signal.ai_enhanced else "📊 기본 분석"
            ai_signal_info = f"({signal.technical_signal})" if signal.technical_signal else ""
            
            print(f"\n🏆 {symbol} 전설급 분석 결과 (월 5-7% 최적화) {ai_status} {ai_signal_info}:")
            print(f"   액션: {signal.action}")
            print(f"   전설 점수: {signal.legendary_score:.3f}")
            print(f"   AI 설명: {signal.explanation}")
            print(f"   양자 사이클: {signal.quantum_cycle}")
            print(f"   확신도: {signal.ai_confidence:.3f}" if signal.ai_confidence else f"   확신도: {signal.confidence:.3f}")
            print(f"   투자 권장: {signal.total_investment:,.0f}원")
            print(f"   익절 계획: {[f'{p:,.0f}' for p in signal.take_profits]} (0차/1차/2차)")
            print(f"   손절선: {signal.stop_loss:,.0f}원")
            
            # AI 기술분석 상세 정보
            if signal.ai_enhanced:
                print(f"\n🤖 OpenAI 기술분석:")
                print(f"   기술적 신호: {signal.technical_signal}")
                print(f"   AI 강화 확신도: {signal.ai_confidence:.3f}")
                
        return signal
        
    except Exception as e:
        print(f"❌ {symbol} 분석 실패: {e}")
        return None

# 실시간 모니터링 시작 함수
async def start_monitoring():
    """실시간 모니터링 시작 (OpenAI 기술분석 통합)"""
    openai_key = os.getenv('OPENAI_API_KEY')
    master = LegendaryQuantMaster(openai_api_key=openai_key)
    
    print("🔄 실시간 모니터링 시작 (월금 매매 모드)")
    if master.openai_analyzer.enabled:
        print("🤖 OpenAI 기술분석 활성화 (비용 최적화)")
    else:
        print("📊 기본 분석 모드")
    print("Ctrl+C로 중지할 수 있습니다.")
    
    try:
        await master.monitor.start_monitoring(check_interval=180)  # 3분마다
    except KeyboardInterrupt:
        print("\n⏹️ 실시간 모니터링을 중지합니다.")
        master.monitor.stop_monitoring()

# OpenAI 테스트 함수
async def test_openai():
    """OpenAI 연결 테스트"""
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("다음과 같이 설정하세요:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    analyzer = OpenAITechnicalAnalyzer(openai_key)
    
    if analyzer.enabled:
        print("✅ OpenAI 연결 성공")
        
        # 간단한 테스트
        try:
            print("🧪 OpenAI 기술분석 테스트 중...")
            test_data = {
                'rsi': 65.5,
                'macd_signal': 'bullish',
                'bollinger_position': 'upper',
                'volume_trend': 'increasing',
                'momentum_7d': 12.3,
                'volume_rank': 15,
                'base_confidence': 0.6
            }
            
            result = await analyzer.analyze_technical_confidence('KRW-BTC', test_data)
            
            if result and result.get('ai_confidence'):
                print(f"✅ OpenAI 기술분석 테스트 성공!")
                print(f"   AI 확신도: {result['ai_confidence']:.3f}")
                print(f"   기술적 신호: {result.get('technical_signal', 'unknown')}")
                print(f"   AI 사용됨: {result.get('ai_used', False)}")
                print(f"   호출 횟수: {result.get('call_count', 0)}")
                
                # 사용량 통계
                usage_stats = analyzer.get_usage_stats()
                print(f"   예상 월비용: {usage_stats['monthly_projection_krw']:.0f}원")
            else:
                print("⚠️ OpenAI 응답이 예상과 다릅니다.")
                
        except Exception as e:
            print(f"❌ OpenAI 테스트 실패: {e}")
    else:
        print("❌ OpenAI 초기화 실패")

# 포트폴리오 현황 조회
async def show_portfolio():
    """현재 포트폴리오 현황 조회"""
    openai_key = os.getenv('OPENAI_API_KEY')
    master = LegendaryQuantMaster(openai_api_key=openai_key)
    
    positions = master.position_manager.get_all_positions()
    
    if not positions:
        print("📊 현재 보유 포지션이 없습니다.")
        return
    
    print("\n📊 현재 포트폴리오 현황:")
    print("=" * 80)
    
    total_investment = 0
    total_current_value = 0
    total_pnl = 0
    
    for i, pos in enumerate(positions, 1):
        try:
            current_price = pyupbit.get_current_price(pos.symbol)
            if current_price:
                current_value = pos.total_quantity * current_price
                pnl = current_value - (pos.total_quantity * pos.avg_price)
                pnl_ratio = (pnl / (pos.total_quantity * pos.avg_price)) * 100
                
                total_investment += pos.total_quantity * pos.avg_price
                total_current_value += current_value
                total_pnl += pnl
                
                holding_days = (datetime.now() - pos.created_at).days
                
                print(f"\n[{i}] {pos.symbol}")
                print(f"   수량: {pos.total_quantity:.6f}개")
                print(f"   평균단가: {pos.avg_price:,.0f}원")
                print(f"   현재가: {current_price:,.0f}원")
                print(f"   투자금액: {pos.total_quantity * pos.avg_price:,.0f}원")
                print(f"   현재가치: {current_value:,.0f}원")
                print(f"   손익: {pnl:+,.0f}원 ({pnl_ratio:+.1f}%)")
                print(f"   보유기간: {holding_days}일")
                print(f"   손절선: {pos.stop_loss:,.0f}원")
                
        except Exception as e:
            print(f"   ❌ {pos.symbol} 가격 조회 실패: {e}")
    
    if total_investment > 0:
        total_pnl_ratio = (total_pnl / total_investment) * 100
        print(f"\n💰 포트폴리오 요약:")
        print(f"   총 투자금액: {total_investment:,.0f}원")
        print(f"   총 현재가치: {total_current_value:,.0f}원")
        print(f"   총 손익: {total_pnl:+,.0f}원 ({total_pnl_ratio:+.1f}%)")
        print(f"   포지션 수: {len(positions)}개")

# OpenAI 사용량 체크
def check_openai_usage():
    """OpenAI 사용량 및 비용 체크"""
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    analyzer = OpenAITechnicalAnalyzer(openai_key)
    if analyzer.enabled:
        usage_stats = analyzer.get_usage_stats()
        
        print("\n🤖 OpenAI 사용량 통계:")
        print("=" * 50)
        print(f"   오늘 호출 횟수: {usage_stats['daily_calls']}/{usage_stats['daily_limit']}회")
        print(f"   예상 일일 비용: ${usage_stats['estimated_daily_cost_usd']:.3f}")
        print(f"   예상 월 비용: ${usage_stats['monthly_projection_usd']:.2f}")
        print(f"   예상 월 비용: {usage_stats['monthly_projection_krw']:.0f}원")
        print(f"   비용 효율성: {usage_stats['efficiency']}")
        
        if usage_stats['efficiency'] == 'over_budget':
            print("   ⚠️ 월 예산(5천원) 초과 예상")
        else:
            print("   ✅ 월 예산 내 운영 중")
    else:
        print("❌ OpenAI 초기화 실패")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command.startswith('analyze:'):
            # 단일 코인 분석
            symbol = command.split(':')[1].upper()
            if not symbol.startswith('KRW-'):
                symbol = f'KRW-{symbol}'
            asyncio.run(analyze_single_coin(symbol))
        elif command == 'monitor':
            # 실시간 모니터링
            asyncio.run(start_monitoring())
        elif command == 'test-openai':
            # OpenAI 테스트
            asyncio.run(test_openai())
        elif command == 'portfolio':
            # 포트폴리오 현황
            asyncio.run(show_portfolio())
        elif command == 'usage':
            # OpenAI 사용량 체크
            check_openai_usage()
        else:
            print("사용법:")
            print("  python script.py                # 전체 전략 실행")
            print("  python script.py analyze:BTC    # 단일 코인 분석")
            print("  python script.py monitor        # 실시간 모니터링")
            print("  python script.py portfolio      # 포트폴리오 현황")
            print("  python script.py test-openai    # OpenAI 연결 테스트")
            print("  python script.py usage          # OpenAI 사용량 체크")
            print("")
            print("OpenAI 설정 (월 5천원 이하 최적화):")
            print("  export OPENAI_API_KEY='your-api-key-here'")
            print("")
            print("특징:")
            print("  • OpenAI는 애매한 확신도(0.4-0.7)에서만 호출")
            print("  • 일일 50회 제한으로 월 비용 5천원 이하")
            print("  • 기술분석 전용 (뉴스/심리분석 제거)")
            print("  • 월 5-7% 수익 최적화")
    else:
        # 기본 실행
        asyncio.run(main())
