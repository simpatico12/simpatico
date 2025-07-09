#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ LEGENDARY QUANT STRATEGY COMPLETE ⚡
전설급 5대 시스템 + 완전한 매도 시스템 + OpenAI 분석 (월 5-7% 최적화)

🧠 Neural Quality Engine - 가중평균 기반 품질 스코어링
🌊 Quantum Cycle Matrix - 27개 미시사이클 감지  
⚡ Fractal Filtering Pipeline - 다차원 필터링
💎 Diamond Hand Algorithm - 켈리공식 기반 분할매매
🕸️ Correlation Web Optimizer - 네트워크 포트폴리오
🎯 Position Manager - 포지션 관리 + 실시간 매도
🤖 OpenAI Integration - AI 기반 시장 분석

✨ 월 5-7% 최적화:
- 0차 익절 추가 (5-7% 구간)
- 3차 익절 삭제 (무제한 수익)
- 타이트한 손절 (-5~8%)
- 월금 매매 시스템
- OpenAI 시장 분석

Author: 퀀트마스터 | Version: MONTHLY 5-7% OPTIMIZED + OpenAI
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
# 🤖 OPENAI MARKET ANALYZER - AI 기반 시장 분석 엔진
# ============================================================================
class OpenAIMarketAnalyzer:
    """OpenAI 기반 시장 분석 엔진"""
    
    def __init__(self, api_key: str = None):
        """OpenAI 클라이언트 초기화"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.enabled = True
            logger.info("🤖 OpenAI 엔진 초기화 완료")
        else:
            self.client = None
            self.enabled = False
            logger.warning("⚠️ OpenAI API 키가 없습니다. AI 분석 기능이 비활성화됩니다.")
    
    async def analyze_market_sentiment(self, symbol: str, market_data: Dict, cycle_info: Dict) -> Dict:
        """시장 심리 및 전망 분석"""
        if not self.enabled:
            return self._fallback_sentiment()
        
        try:
            # 시장 데이터 정리
            price = market_data.get('price', 0)
            volume_krw = market_data.get('volume_krw', 0)
            ohlcv = market_data.get('ohlcv')
            
            # 가격 변동 계산
            if ohlcv is not None and len(ohlcv) >= 7:
                price_change_7d = ((ohlcv['close'].iloc[-1] / ohlcv['close'].iloc[-8]) - 1) * 100
                volume_avg_7d = ohlcv['volume'].tail(7).mean() * price
            else:
                price_change_7d = 0
                volume_avg_7d = volume_krw
            
            # AI 분석 요청 프롬프트
            prompt = f"""
다음 암호화폐의 시장 데이터를 분석하고 투자 관점에서 평가해주세요:

코인: {symbol}
현재가격: {price:,.0f}원
24시간 거래량: {volume_krw:,.0f}원
7일 가격변동: {price_change_7d:+.1f}%
7일 평균거래량: {volume_avg_7d:,.0f}원
현재 시장사이클: {cycle_info.get('cycle', 'unknown')}
사이클 신뢰도: {cycle_info.get('confidence', 0):.2f}

다음 관점에서 분석해주세요:
1. 현재 시장 심리 (0-100점)
2. 단기 전망 (1-2주)
3. 투자 매력도 (0-100점)
4. 주요 리스크 요인
5. 종합 추천도 (0-100점)

JSON 형식으로 답변해주세요:
{{
    "market_sentiment": 점수(0-100),
    "short_term_outlook": "긍정적/중립적/부정적",
    "investment_attractiveness": 점수(0-100),
    "risk_factors": ["리스크1", "리스크2"],
    "overall_recommendation": 점수(0-100),
    "reasoning": "분석 근거 요약"
}}
"""
            
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 전문 암호화폐 분석가입니다. 객관적이고 데이터 기반의 분석을 제공하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # 응답 파싱
            ai_response = response.choices[0].message.content
            
            # JSON 파싱 시도
            try:
                ai_analysis = json.loads(ai_response)
                ai_analysis['ai_confidence'] = 0.8
                ai_analysis['analysis_type'] = 'openai_gpt'
                return ai_analysis
            except json.JSONDecodeError:
                # JSON 파싱 실패시 텍스트 분석
                return self._parse_text_response(ai_response)
            
        except Exception as e:
            logger.error(f"OpenAI 분석 실패 {symbol}: {e}")
            return self._fallback_sentiment()
    
    async def analyze_coin_fundamentals(self, symbol: str, quality_data: Dict) -> Dict:
        """코인 펀더멘털 분석"""
        if not self.enabled:
            return self._fallback_fundamentals()
        
        try:
            coin_name = symbol.replace('KRW-', '')
            
            prompt = f"""
{coin_name} 코인의 펀더멘털을 분석해주세요:

현재 품질 점수:
- 기술력: {quality_data.get('tech_score', 0):.2f}
- 생태계: {quality_data.get('ecosystem_score', 0):.2f}  
- 커뮤니티: {quality_data.get('community_score', 0):.2f}
- 채택도: {quality_data.get('adoption_score', 0):.2f}

다음을 분석해주세요:
1. 기술적 혁신도 (0-100)
2. 실용성 및 활용도 (0-100)
3. 경쟁력 (0-100)
4. 성장 잠재력 (0-100)
5. 종합 펀더멘털 점수 (0-100)

JSON 형식으로 답변:
{{
    "tech_innovation": 점수,
    "utility_score": 점수,
    "competitive_advantage": 점수,
    "growth_potential": 점수,
    "fundamental_score": 점수,
    "key_strengths": ["강점1", "강점2"],
    "concerns": ["우려1", "우려2"]
}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 블록체인 기술 전문가입니다. 객관적인 펀더멘털 분석을 제공하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            ai_response = response.choices[0].message.content
            
            try:
                fundamental_analysis = json.loads(ai_response)
                fundamental_analysis['analysis_source'] = 'openai_fundamental'
                return fundamental_analysis
            except json.JSONDecodeError:
                return self._fallback_fundamentals()
                
        except Exception as e:
            logger.error(f"펀더멘털 분석 실패 {symbol}: {e}")
            return self._fallback_fundamentals()
    
    async def generate_trading_strategy(self, symbol: str, analysis_data: Dict) -> Dict:
        """AI 기반 거래 전략 생성"""
        if not self.enabled:
            return self._fallback_strategy()
        
        try:
            prompt = f"""
{symbol} 코인의 종합 분석 데이터를 바탕으로 거래 전략을 제안해주세요:

분석 데이터:
- 품질 점수: {analysis_data.get('quality_score', 0):.2f}
- 시장 심리: {analysis_data.get('market_sentiment', 50)}/100
- 투자 매력도: {analysis_data.get('investment_attractiveness', 50)}/100
- 사이클: {analysis_data.get('cycle', 'unknown')}
- 펀더멘털: {analysis_data.get('fundamental_score', 50)}/100

월 5-7% 수익 목표로 다음을 제안해주세요:
1. 진입 타이밍 (즉시/대기/분할)
2. 포지션 크기 (1-15% of portfolio)
3. 익절 전략 수정사항
4. 리스크 관리 방안
5. 홀딩 기간 조정

JSON 형식:
{{
    "entry_timing": "immediate/wait/gradual",
    "position_size_pct": 숫자(1-15),
    "profit_strategy": "conservative/standard/aggressive",
    "risk_adjustment": "tight/normal/loose",
    "holding_period": "short/medium/long",
    "confidence": 숫자(0-100),
    "strategy_reasoning": "전략 근거"
}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 전문 퀀트 트레이더입니다. 월 5-7% 수익을 목표로 하는 보수적이고 안정적인 전략을 제안하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content
            
            try:
                strategy = json.loads(ai_response)
                strategy['generated_by'] = 'openai_strategy'
                return strategy
            except json.JSONDecodeError:
                return self._fallback_strategy()
                
        except Exception as e:
            logger.error(f"전략 생성 실패 {symbol}: {e}")
            return self._fallback_strategy()
    
    def _parse_text_response(self, text: str) -> Dict:
        """텍스트 응답 파싱"""
        try:
            # 기본값으로 중간 점수 설정
            result = {
                'market_sentiment': 50,
                'short_term_outlook': '중립적',
                'investment_attractiveness': 50,
                'risk_factors': ['분석 불가'],
                'overall_recommendation': 50,
                'reasoning': text[:200],
                'ai_confidence': 0.5,
                'analysis_type': 'openai_text'
            }
            
            # 간단한 키워드 기반 점수 조정
            text_lower = text.lower()
            if '긍정' in text_lower or 'positive' in text_lower:
                result['market_sentiment'] += 20
                result['overall_recommendation'] += 15
            elif '부정' in text_lower or 'negative' in text_lower:
                result['market_sentiment'] -= 20
                result['overall_recommendation'] -= 15
            
            return result
        except:
            return self._fallback_sentiment()
    
    def _fallback_sentiment(self) -> Dict:
        """기본 심리 분석"""
        return {
            'market_sentiment': 50,
            'short_term_outlook': '중립적',
            'investment_attractiveness': 50,
            'risk_factors': ['AI 분석 불가'],
            'overall_recommendation': 50,
            'reasoning': 'OpenAI 분석 실패로 기본값 사용',
            'ai_confidence': 0.3,
            'analysis_type': 'fallback'
        }
    
    def _fallback_fundamentals(self) -> Dict:
        """기본 펀더멘털 분석"""
        return {
            'tech_innovation': 50,
            'utility_score': 50,
            'competitive_advantage': 50,
            'growth_potential': 50,
            'fundamental_score': 50,
            'key_strengths': ['기본 분석'],
            'concerns': ['AI 분석 불가'],
            'analysis_source': 'fallback'
        }
    
    def _fallback_strategy(self) -> Dict:
        """기본 전략"""
        return {
            'entry_timing': 'gradual',
            'position_size_pct': 5,
            'profit_strategy': 'standard',
            'risk_adjustment': 'normal',
            'holding_period': 'medium',
            'confidence': 50,
            'strategy_reasoning': 'AI 분석 실패로 기본 전략 사용',
            'generated_by': 'fallback'
        }

# ============================================================================
# 🧠 NEURAL QUALITY ENGINE - 가중평균 기반 품질 스코어링 (OpenAI 통합)
# ============================================================================
class NeuralQualityEngine:
    """가중평균 기반 품질 평가 엔진 (안정성 최우선) + OpenAI 통합"""
    
    def __init__(self, openai_analyzer: OpenAIMarketAnalyzer = None):
        # 코인별 품질 점수 (기술력, 생태계, 커뮤니티, 채택도)
        self.coin_scores = {
            'BTC': [0.98, 0.95, 0.90, 0.95], 'ETH': [0.95, 0.98, 0.85, 0.90],
            'BNB': [0.80, 0.90, 0.75, 0.85], 'ADA': [0.85, 0.75, 0.80, 0.70],
            'SOL': [0.90, 0.80, 0.85, 0.75], 'AVAX': [0.85, 0.75, 0.70, 0.70],
            'DOT': [0.85, 0.80, 0.70, 0.65], 'MATIC': [0.80, 0.85, 0.75, 0.80],
            'ATOM': [0.75, 0.70, 0.75, 0.60], 'NEAR': [0.80, 0.70, 0.65, 0.60],
            'LINK': [0.90, 0.75, 0.70, 0.80], 'UNI': [0.85, 0.80, 0.75, 0.75],
        }
        
        # 가중치 (기술력 30%, 생태계 30%, 커뮤니티 20%, 채택도 20%)
        self.weights = [0.30, 0.30, 0.20, 0.20]
        self.openai_analyzer = openai_analyzer
    
    async def neural_quality_score(self, symbol: str, market_data: Dict) -> Dict:
        """안전한 가중평균 품질 점수 계산 + OpenAI 분석"""
        try:
            coin_name = symbol.replace('KRW-', '')
            
            # 기본 점수 또는 등록된 점수 사용
            scores = self.coin_scores.get(coin_name, [0.6, 0.6, 0.6, 0.6])
            
            # 가중평균 계산 (안전한 방식)
            quality_score = sum(score * weight for score, weight in zip(scores, self.weights))
            
            # 거래량 기반 보너스
            volume_bonus = self._calculate_volume_bonus(market_data.get('volume_24h_krw', 0))
            base_quality = min(0.98, quality_score + volume_bonus)
            
            # OpenAI 펀더멘털 분석 (비동기)
            ai_fundamental = None
            if self.openai_analyzer and self.openai_analyzer.enabled:
                try:
                    quality_data = {
                        'tech_score': scores[0],
                        'ecosystem_score': scores[1],
                        'community_score': scores[2],
                        'adoption_score': scores[3]
                    }
                    ai_fundamental = await self.openai_analyzer.analyze_coin_fundamentals(symbol, quality_data)
                except Exception as e:
                    logger.debug(f"OpenAI 펀더멘털 분석 실패 {symbol}: {e}")
            
            # AI 분석이 있으면 가중 평균으로 조정
            if ai_fundamental and ai_fundamental.get('fundamental_score'):
                ai_score = ai_fundamental['fundamental_score'] / 100  # 0-1 스케일로 변환
                # 70% 기본 분석 + 30% AI 분석
                final_quality = base_quality * 0.7 + ai_score * 0.3
                ai_explanation = self._generate_ai_explanation(coin_name, scores, final_quality, ai_fundamental)
                confidence_boost = 0.1  # AI 분석이 있으면 신뢰도 증가
            else:
                final_quality = base_quality
                ai_explanation = self._generate_explanation(coin_name, scores, final_quality)
                confidence_boost = 0.0
            
            return {
                'quality_score': final_quality,
                'tech_score': scores[0],
                'ecosystem_score': scores[1], 
                'community_score': scores[2],
                'adoption_score': scores[3],
                'ai_explanation': ai_explanation,
                'confidence': min(0.95, final_quality + 0.05 + confidence_boost),
                'ai_fundamental': ai_fundamental,
                'ai_enhanced': ai_fundamental is not None
            }
            
        except Exception as e:
            logger.error(f"품질 점수 계산 실패 {symbol}: {e}")
            return {
                'quality_score': 0.5, 'tech_score': 0.5, 'ecosystem_score': 0.5,
                'community_score': 0.5, 'adoption_score': 0.5,
                'ai_explanation': '기본등급', 'confidence': 0.5,
                'ai_fundamental': None, 'ai_enhanced': False
            }
    
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
    
    def _generate_explanation(self, coin: str, scores: List[float], final_score: float) -> str:
        """기본 설명 생성"""
        features = []
        if scores[0] > 0.85: features.append("최고급기술")
        if scores[1] > 0.85: features.append("강력생태계")
        if scores[2] > 0.80: features.append("활발커뮤니티")
        if scores[3] > 0.80: features.append("높은채택도")
        
        if final_score > 0.8: grade = "S급"
        elif final_score > 0.7: grade = "A급"  
        elif final_score > 0.6: grade = "B급"
        else: grade = "C급"
        
        return f"{grade} | " + " | ".join(features) if features else f"{grade} | 기본등급"
    
    def _generate_ai_explanation(self, coin: str, scores: List[float], final_score: float, ai_data: Dict) -> str:
        """AI 강화 설명 생성"""
        base_explanation = self._generate_explanation(coin, scores, final_score)
        
        if ai_data and ai_data.get('key_strengths'):
            ai_strengths = ai_data['key_strengths'][:2]  # 최대 2개만
            ai_part = " | AI분석: " + ", ".join(ai_strengths)
            return base_explanation + ai_part
        
        return base_explanation + " | AI분석완료"

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
# 💎 DIAMOND HAND ALGORITHM - 켈리공식 기반 분할매매 (월 5-7% 최적화) + OpenAI 통합
# ============================================================================
class DiamondHandAlgorithm:
    """켈리 공식 기반 다이아몬드 핸드 알고리즘 (월 5-7% 최적화) + OpenAI 통합"""
    
    def __init__(self, portfolio_value: float, openai_analyzer: OpenAIMarketAnalyzer = None):
        self.portfolio_value = portfolio_value
        self.openai_analyzer = openai_analyzer
    
    async def calculate_diamond_strategy(self, symbol: str, price: float, confidence: float, 
                                       cycle: str, quality_score: float, ai_data: Dict = None) -> Dict:
        """다이아몬드 핸드 전략 계산 (월 5-7% 최적화) + AI 조정"""
        try:
            # 켈리 비율 계산 (단순화)
            kelly_fraction = self._kelly_criterion(confidence, quality_score)
            
            # 감정 팩터
            emotion_factor = self._emotion_factor(cycle, confidence)
            
            # AI 전략 조정 (있는 경우)
            ai_adjustment = 1.0
            if self.openai_analyzer and self.openai_analyzer.enabled and ai_data:
                try:
                    analysis_data = {
                        'quality_score': quality_score,
                        'cycle': cycle,
                        'market_sentiment': ai_data.get('market_sentiment', 50),
                        'investment_attractiveness': ai_data.get('investment_attractiveness', 50),
                        'fundamental_score': ai_data.get('ai_fundamental', {}).get('fundamental_score', 50) if ai_data.get('ai_fundamental') else 50
                    }
                    
                    ai_strategy = await self.openai_analyzer.generate_trading_strategy(symbol, analysis_data)
                    if ai_strategy:
                        # AI 포지션 크기 조정
                        ai_position_pct = ai_strategy.get('position_size_pct', 5) / 100
                        current_position_pct = kelly_fraction * emotion_factor
                        # 70% 기존 로직 + 30% AI 제안
                        ai_adjustment = (current_position_pct * 0.7 + ai_position_pct * 0.3) / current_position_pct
                        ai_adjustment = max(0.5, min(1.5, ai_adjustment))  # 50%-150% 범위로 제한
                        
                except Exception as e:
                    logger.debug(f"AI 전략 조정 실패 {symbol}: {e}")
            
            # 총 투자 금액 (AI 조정 적용)
            base_investment = self.portfolio_value * kelly_fraction * emotion_factor * ai_adjustment
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
                'ai_adjustment': ai_adjustment,
                'stage_amounts': stage_amounts,
                'entry_prices': entry_prices,
                'take_profits': take_profits,
                'stop_loss': stop_loss,
                'portfolio_weight': (total_investment / self.portfolio_value) * 100,
                'ai_enhanced': ai_adjustment != 1.0
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
            'kelly_fraction': 0.05, 'emotion_factor': 1.0, 'ai_adjustment': 1.0,
            'stage_amounts': [base_investment * 0.5, base_investment * 0.3, base_investment * 0.2],
            'entry_prices': [price, price * 0.95, price * 0.90],
            'take_profits': [price * 1.05, price * 1.15, price * 1.25],
            'stop_loss': price * 0.92, 'portfolio_weight': 5.0, 'ai_enhanced': False
        }

# ============================================================================
# 📊 전설급 메인 시그널 클래스 (OpenAI 통합)
# ============================================================================
@dataclass
class LegendarySignal:
    """전설급 시그널 (OpenAI 통합)"""
    symbol: str
    action: str
    confidence: float
    price: float
    
    # 분석 결과
    neural_quality: float
    ai_explanation: str
    quantum_cycle: str
    cycle_confidence: float
    
    # 투자 전략
    kelly_fraction: float
    emotion_factor: float
    total_investment: float
    ai_adjustment: float
    
    # 실행 계획
    entry_prices: List[float]
    stage_amounts: List[float] 
    take_profits: List[float]
    stop_loss: float
    
    # AI 분석 결과
    ai_market_sentiment: Optional[Dict]
    ai_fundamental: Optional[Dict]
    ai_enhanced: bool
    
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
        """매도 조건 체크 (월 5-7% 최적화)"""
        position = self.position_manager.get_position(symbol)
        if not position:
            return {'action': 'none', 'reason': 'no_position'}
        
        # 미실현 손익 업데이트
        self.position_manager.update_unrealized_pnl(symbol, current_price)
        
        # 수익률 계산
        profit_ratio = (current_price - position.avg_price) / position.avg_price
        
        # 1. 손절 체크 (타이트한 손절)
        if current_price <= position.stop_loss:
            return {
                'action': 'sell_all',
                'reason': 'stop_loss',
                'price': current_price,
                'quantity': position.total_quantity,
                'details': f'손절 실행: {current_price} <= {position.stop_loss}'
            }
        
        # 2. 시간 기반 매도 (2주 = 14일)
        holding_days = (datetime.now() - position.created_at).days
        if holding_days >= 14:
            if profit_ratio > 0.03:  # 3% 이상 수익시
                return {
                    'action': 'sell_all',
                    'reason': 'time_limit_profit',
                    'price': current_price,
                    'quantity': position.total_quantity,
                    'details': f'2주 완료: {holding_days}일, {profit_ratio*100:.1f}% 수익으로 매도'
                }
        elif holding_days >= 16:  # 2주 초과시 무조건
            return {
                'action': 'sell_all',
                'reason': 'time_limit_force',
                'price': current_price,
                'quantity': position.total_quantity,
                'details': f'강제매도: {holding_days}일 초과'
            }
        
        # 3. 익절 체크 (0차 익절 추가)
        profit_flags = self.profit_taken_flags.get(symbol, [False, False, False])
        
        # 0차 익절 (4-6% 수익시 20-25% 매도)
        if (len(position.target_take_profits) >= 1 and 
            current_price >= position.target_take_profits[0] and 
            profit_ratio >= 0.04 and not profit_flags[0]):
            
            sell_ratio = 0.25 if profit_ratio < 0.05 else 0.20
            sell_quantity = position.total_quantity * sell_ratio
            
            # 익절 플래그 설정
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
# 🏆 LEGENDARY QUANT MASTER - 전설급 통합 시스템 (월 5-7% 완전체) + OpenAI
# ============================================================================
class LegendaryQuantMaster:
    """전설급 5대 시스템 + 완전한 매도 시스템 + OpenAI 통합 마스터 (월 5-7% 최적화)"""
    
    def __init__(self, portfolio_value: float = 100_000_000, min_volume: float = 10_000_000_000, 
                 demo_mode: bool = True, openai_api_key: str = None):
        self.portfolio_value = portfolio_value
        self.min_volume = min_volume
        self.demo_mode = demo_mode
        
        # OpenAI 분석기 초기화
        self.openai_analyzer = OpenAIMarketAnalyzer(openai_api_key)
        
        # 전설급 5대 엔진 초기화 (OpenAI 통합)
        self.neural_engine = NeuralQualityEngine(self.openai_analyzer)
        self.quantum_cycle = QuantumCycleMatrix()
        self.fractal_filter = FractalFilteringPipeline(min_volume)
        self.diamond_algorithm = DiamondHandAlgorithm(portfolio_value, self.openai_analyzer)
        
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
        """전설급 전략 실행 (월 5-7% 최적화) + OpenAI"""
        logger.info("🏆 LEGENDARY QUANT STRATEGY COMPLETE + OpenAI (월 5-7% 최적화) 시작")
        
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
            
            # 3단계: 개별 코인 전설급 분석 (OpenAI 통합)
            legendary_signals = []
            for i, candidate in enumerate(fractal_candidates[:self.target_portfolio_size], 1):
                logger.info(f"💎 전설급 분석 [{i}/{min(len(fractal_candidates), self.target_portfolio_size)}]: {candidate['symbol']} (AI {'ON' if self.openai_analyzer.enabled else 'OFF'})")
                
                signal = await self._analyze_legendary_coin(candidate, quantum_state)
                if signal:
                    legendary_signals.append(signal)
                
                await asyncio.sleep(0.5)  # API 제한 + OpenAI 호출 고려
            
            # 4단계: 최종 포트폴리오 랭킹
            legendary_signals.sort(key=lambda x: x.legendary_score, reverse=True)
            
            # 결과 요약
            buy_signals = [s for s in legendary_signals if s.action == 'BUY']
            ai_enhanced_count = sum(1 for s in buy_signals if s.ai_enhanced)
            
            logger.info(f"✨ 전설급 분석 완료: {len(legendary_signals)}개 분석, {len(buy_signals)}개 매수 신호 (AI 강화: {ai_enhanced_count}개)")
            
            return legendary_signals
            
        except Exception as e:
            logger.error(f"전설급 전략 실행 실패: {e}")
            return []
    
    async def _analyze_legendary_coin(self, candidate: Dict, quantum_state: Dict) -> Optional[LegendarySignal]:
        """개별 코인 전설급 분석 (OpenAI 통합)"""
        try:
            symbol = candidate['symbol']
            price = candidate['price']
            
            # Neural Quality Engine 분석 (OpenAI 통합)
            market_data = {
                'volume_24h_krw': candidate['volume_krw'],
                'price': price,
                'ohlcv': candidate['ohlcv']
            }
            neural_result = await self.neural_engine.neural_quality_score(symbol, market_data)
            
            # OpenAI 시장 심리 분석 (있는 경우)
            ai_sentiment = None
            if self.openai_analyzer.enabled:
                try:
                    ai_sentiment = await self.openai_analyzer.analyze_market_sentiment(symbol, market_data, quantum_state)
                except Exception as e:
                    logger.debug(f"OpenAI 심리 분석 실패 {symbol}: {e}")
            
            # Diamond Hand Algorithm 분석 (OpenAI 통합)
            diamond_result = await self.diamond_algorithm.calculate_diamond_strategy(
                symbol, price, neural_result['confidence'], 
                quantum_state['cycle'], neural_result['quality_score'], ai_sentiment
            )
            
            # 종합 점수 계산 (AI 가중치 추가)
            base_score = (
                neural_result['quality_score'] * 0.30 +      # Neural Quality
                quantum_state['confidence'] * 0.25 +         # Quantum Cycle  
                candidate.get('technical_score', 0.5) * 0.25 +  # Technical
                candidate.get('momentum_score', 0.5) * 0.20     # Momentum
            )
            
            # AI 보너스 점수 (있는 경우)
            ai_bonus = 0.0
            if ai_sentiment:
                sentiment_score = ai_sentiment.get('overall_recommendation', 50) / 100
                attractiveness_score = ai_sentiment.get('investment_attractiveness', 50) / 100
                ai_bonus = (sentiment_score + attractiveness_score) / 2 * 0.1  # 최대 10% 보너스
            
            legendary_score = min(1.0, base_score + ai_bonus)
            
            # 액션 결정 (AI 보정)
            if legendary_score >= 0.70:
                action = 'BUY'
            elif legendary_score <= 0.30:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # AI 분석 결과가 부정적이면 액션 하향 조정
            if ai_sentiment and ai_sentiment.get('overall_recommendation', 50) < 30:
                if action == 'BUY':
                    action = 'HOLD'
                    logger.info(f"🤖 {symbol}: AI 부정적 분석으로 BUY → HOLD 조정")
            
            # 전설급 시그널 생성
            signal = LegendarySignal(
                symbol=symbol,
                action=action,
                confidence=neural_result['confidence'],
                price=price,
                neural_quality=neural_result['quality_score'],
                ai_explanation=neural_result['ai_explanation'],
                quantum_cycle=quantum_state['cycle'],
                cycle_confidence=quantum_state['confidence'],
                kelly_fraction=diamond_result['kelly_fraction'],
                emotion_factor=diamond_result['emotion_factor'],
                total_investment=diamond_result['total_investment'],
                ai_adjustment=diamond_result.get('ai_adjustment', 1.0),
                entry_prices=diamond_result['entry_prices'],
                stage_amounts=diamond_result['stage_amounts'],
                take_profits=diamond_result['take_profits'],
                stop_loss=diamond_result['stop_loss'],
                ai_market_sentiment=ai_sentiment,
                ai_fundamental=neural_result.get('ai_fundamental'),
                ai_enhanced=neural_result.get('ai_enhanced', False) or diamond_result.get('ai_enhanced', False),
                legendary_score=legendary_score,
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"개별 코인 전설급 분석 실패 {candidate['symbol']}: {e}")
            return None
    
    def print_legendary_results(self, signals: List[LegendarySignal]):
        """전설급 결과 출력 (월 5-7% 최적화) + OpenAI"""
        print("\n" + "="*90)
        print("🏆 LEGENDARY QUANT STRATEGY COMPLETE + OpenAI - 월 5-7% 최적화 🏆")
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
        print(f"   AI 강화: {ai_enhanced_count}개 ({self.openai_analyzer.enabled and 'OpenAI 활성화' or 'OpenAI 비활성화'})")
        print(f"   총 투자금: {total_investment:,.0f}원")
        print(f"   포트폴리오 비중: {(total_investment/self.portfolio_value)*100:.1f}%")
        print(f"   운영 모드: {'시뮬레이션' if self.demo_mode else '실제거래'}")
        print(f"   오늘: {today_name} ({'거래일' if is_trading_day else '비거래일'})")
        
        if signals:
            print(f"\n🌊 양자 사이클 상태:")
            print(f"   현재 사이클: {signals[0].quantum_cycle}")
            print(f"   신뢰도: {signals[0].cycle_confidence:.2f}")
        
        print(f"\n🤖 OpenAI 분석 시스템:")
        if self.openai_analyzer.enabled:
            print(f"   • 시장 심리 분석: GPT-3.5-Turbo")
            print(f"   • 펀더멘털 분석: 블록체인 전문가 모드")
            print(f"   • 거래 전략 생성: 퀀트 트레이더 모드")
            print(f"   • AI 강화 비율: {ai_enhanced_count}/{len(buy_signals)}개")
        else:
            print(f"   • 상태: 비활성화 (OPENAI_API_KEY 없음)")
            print(f"   • 기본 분석만 사용")
        
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
            print(f"\n[{i}] {signal.symbol} {ai_mark}")
            print(f"   전설 점수: {signal.legendary_score:.3f}")
            print(f"   AI 품질: {signal.neural_quality:.2f} | {signal.ai_explanation}")
            print(f"   켈리 비중: {signal.kelly_fraction:.1%}")
            if signal.ai_adjustment != 1.0:
                print(f"   AI 조정: {signal.ai_adjustment:.2f}x")
            print(f"   투자금액: {signal.total_investment:,.0f}원")
            print(f"   진입가격: {[f'{p:,.0f}' for p in signal.entry_prices]}")
            print(f"   익절가격: {[f'{p:,.0f}' for p in signal.take_profits]} (0차/1차/2차)")
            print(f"   손절가격: {signal.stop_loss:,.0f}원")
            
            # AI 추가 정보
            if signal.ai_market_sentiment:
                sentiment = signal.ai_market_sentiment
                print(f"   🤖 AI 심리: {sentiment.get('market_sentiment', 0)}/100, 매력도: {sentiment.get('investment_attractiveness', 0)}/100")
                if sentiment.get('risk_factors'):
                    risks = ', '.join(sentiment['risk_factors'][:2])
                    print(f"   ⚠️ 리스크: {risks}")
        
        print(f"\n📈 월 5-7% 달성 전략:")
        print(f"   • 포트폴리오 8개 중 2-3개 대박(50%+) → 월수익 견인")
        print(f"   • 나머지 4-5개 소폭수익(5-25%) → 안정성 확보")
        print(f"   • 1-2개 손실(-5~8%) → 손절로 제한")
        print(f"   • 평균 월수익: 5-7% 목표")
        print(f"   • AI 분석으로 정확도 향상")
        
        print("\n" + "="*90)
        print("⚡ LEGENDARY STRATEGY COMPLETE + OpenAI - 월 5-7% 최적화 ⚡")

# ============================================================================
# 🚀 메인 실행 함수들
# ============================================================================
async def main():
    """전설급 퀀트 전략 메인 실행 (월 5-7% 최적화) + OpenAI"""
    print("⚡ LEGENDARY QUANT STRATEGY COMPLETE + OpenAI - 월 5-7% 최적화 STARTING ⚡")
    print("🧠🌊⚡💎🕸️🎯🚨🎮📊🤖 완전체 시스템 + AI 로딩...")
    
    # OpenAI API 키 확인
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("🤖 OpenAI API 키 감지 - AI 분석 모드 활성화")
    else:
        print("📊 OpenAI API 키 없음 - 기본 분석 모드")
    
    # 전설급 마스터 초기화 (시뮬레이션 모드 + OpenAI)
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

# 단일 코인 분석 함수 (OpenAI 통합)
async def analyze_single_coin(symbol: str):
    """단일 코인 전설급 분석 (월 5-7% 최적화) + OpenAI"""
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
        signal = await master._analyze_legendary_coin(candidate, quantum_state)
        
        if signal:
            ai_status = "🤖 AI 강화" if signal.ai_enhanced else "📊 기본 분석"
            print(f"\n🏆 {symbol} 전설급 분석 결과 (월 5-7% 최적화) {ai_status}:")
            print(f"   액션: {signal.action}")
            print(f"   전설 점수: {signal.legendary_score:.3f}")
            print(f"   AI 설명: {signal.ai_explanation}")
            print(f"   양자 사이클: {signal.quantum_cycle}")
            print(f"   투자 권장: {signal.total_investment:,.0f}원")
            if signal.ai_adjustment != 1.0:
                print(f"   AI 조정: {signal.ai_adjustment:.2f}x")
            print(f"   익절 계획: {[f'{p:,.0f}' for p in signal.take_profits]} (0차/1차/2차)")
            print(f"   손절선: {signal.stop_loss:,.0f}원")
            
            # AI 추가 정보
            if signal.ai_market_sentiment:
                sentiment = signal.ai_market_sentiment
                print(f"\n🤖 OpenAI 분석:")
                print(f"   시장 심리: {sentiment.get('market_sentiment', 0)}/100")
                print(f"   투자 매력도: {sentiment.get('investment_attractiveness', 0)}/100")
                print(f"   종합 추천도: {sentiment.get('overall_recommendation', 0)}/100")
                print(f"   단기 전망: {sentiment.get('short_term_outlook', 'N/A')}")
                if sentiment.get('reasoning'):
                    print(f"   AI 근거: {sentiment['reasoning'][:100]}...")
            
            if signal.ai_fundamental:
                fundamental = signal.ai_fundamental
                print(f"\n🔬 AI 펀더멘털:")
                print(f"   기술 혁신도: {fundamental.get('tech_innovation', 0)}/100")
                print(f"   활용도: {fundamental.get('utility_score', 0)}/100")
                print(f"   경쟁력: {fundamental.get('competitive_advantage', 0)}/100")
                print(f"   성장 잠재력: {fundamental.get('growth_potential', 0)}/100")
                
        return signal
        
    except Exception as e:
        print(f"❌ {symbol} 분석 실패: {e}")
        return None

# 실시간 모니터링 시작 함수
async def start_monitoring():
    """실시간 모니터링 시작 (OpenAI 통합)"""
    openai_key = os.getenv('OPENAI_API_KEY')
    master = LegendaryQuantMaster(openai_api_key=openai_key)
    
    print("🔄 실시간 모니터링 시작 (월금 매매 모드)")
    if master.openai_analyzer.enabled:
        print("🤖 OpenAI 분석 활성화")
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
    
    analyzer = OpenAIMarketAnalyzer(openai_key)
    
    if analyzer.enabled:
        print("✅ OpenAI 연결 성공")
        
        # 간단한 테스트
        try:
            print("🧪 OpenAI 분석 테스트 중...")
            test_data = {
                'price': 50000000,
                'volume_krw': 100000000000,
                'ohlcv': None
            }
            cycle_info = {'cycle': 'accumulation', 'confidence': 0.7}
            
            result = await analyzer.analyze_market_sentiment('KRW-BTC', test_data, cycle_info)
            
            if result and result.get('market_sentiment'):
                print(f"✅ OpenAI 테스트 성공!")
                print(f"   시장 심리: {result['market_sentiment']}/100")
                print(f"   분석 유형: {result.get('analysis_type', 'unknown')}")
            else:
                print("⚠️ OpenAI 응답이 예상과 다릅니다.")
                
        except Exception as e:
            print(f"❌ OpenAI 테스트 실패: {e}")
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
        else:
            print("사용법:")
            print("  python script.py                # 전체 전략 실행")
            print("  python script.py analyze:BTC    # 단일 코인 분석")
            print("  python script.py monitor        # 실시간 모니터링")
            print("  python script.py test-openai    # OpenAI 연결 테스트")
            print("")
            print("OpenAI 설정:")
            print("  export OPENAI_API_KEY='your-api-key-here'")
    else:
        # 기본 실행: 전체 전략
        asyncio.run(main())
