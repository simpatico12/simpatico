#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇯🇵 일본 주식 완전 자동화 전략 - 최고퀸트프로젝트
===========================================================================

🎯 핵심 기능:
- 💱 엔화 자동 매매법 (USD/JPY 기반)
- ⚡ 고급 기술적 지표 (RSI, MACD, 볼린저밴드, 스토캐스틱)
- 💰 분할매매 시스템 (리스크 관리)
- 🔍 20개 종목 자동 선별
- 🛡️ 동적 손절/익절
- 🤖 완전 자동화 (혼자서도 OK)

Author: 최고퀸트팀
Version: 3.0.0 (기술지표+분할매매 통합)
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass
import yfinance as yf
import ta

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class JPStockSignal:
    """일본 주식 시그널 (완전 통합)"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    strategy_source: str
    
    # 기술적 지표
    rsi: float
    macd_signal: str      # 'bullish', 'bearish', 'neutral'
    bollinger_signal: str # 'upper', 'lower', 'middle'
    stoch_signal: str     # 'oversold', 'overbought', 'neutral'
    ma_trend: str         # 'uptrend', 'downtrend', 'sideways'
    
    # 포지션 관리
    position_size: int    # 총 주식 수
    split_buy_plan: List[Dict]  # 분할 매수 계획
    split_sell_plan: List[Dict] # 분할 매도 계획
    
    # 손익 관리
    stop_loss: float
    take_profit: float
    max_hold_days: int
    
    # 기본 정보
    stock_type: str       # 'export', 'domestic'
    yen_signal: str       # 'strong', 'weak', 'neutral'
    sector: str
    reasoning: str
    timestamp: datetime
    additional_data: Optional[Dict] = None

# ========================================================================================
# 📊 기술적 지표 분석 클래스 (찾기 쉽게 분리)
# ========================================================================================
class TechnicalIndicators:
    """🔧 기술적 지표 계산 및 분석"""
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 10) -> float:
        """RSI 계산"""
        try:
            rsi = ta.momentum.RSIIndicator(data['Close'], window=period).rsi()
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame) -> Tuple[str, Dict]:
        """MACD 계산 및 신호 분석"""
        try:
            macd = ta.trend.MACD(data['Close'], window_fast=8, window_slow=21, window_sign=5)
            macd_line = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            macd_histogram = macd.macd_diff().iloc[-1]
            
            # 신호 분석
            if macd_line > macd_signal and macd_histogram > 0:
                signal = 'bullish'
            elif macd_line < macd_signal and macd_histogram < 0:
                signal = 'bearish'
            else:
                signal = 'neutral'
                
            details = {
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'histogram': macd_histogram
            }
            
            return signal, details
        except:
            return 'neutral', {}
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20) -> Tuple[str, Dict]:
        """볼린저 밴드 계산 및 신호 분석"""
        try:
            bb = ta.volatility.BollingerBands(data['Close'], window=window)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_middle = bb.bollinger_mavg().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # 신호 분석
            if current_price >= bb_upper:
                signal = 'upper'  # 과매수 구간
            elif current_price <= bb_lower:
                signal = 'lower'  # 과매도 구간
            else:
                signal = 'middle' # 정상 구간
                
            details = {
                'upper': bb_upper,
                'middle': bb_middle,
                'lower': bb_lower,
                'position': (current_price - bb_lower) / (bb_upper - bb_lower)
            }
            
            return signal, details
        except:
            return 'middle', {}
    
    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, k_period: int = 14) -> Tuple[str, Dict]:
        """스토캐스틱 계산 및 신호 분석"""
        try:
            stoch = ta.momentum.StochasticOscillator(
                data['High'], data['Low'], data['Close'], 
                window=k_period, smooth_window=3
            )
            stoch_k = stoch.stoch().iloc[-1]
            stoch_d = stoch.stoch_signal().iloc[-1]
            
            # 신호 분석
            if stoch_k <= 20 and stoch_d <= 20:
                signal = 'oversold'  # 과매도
            elif stoch_k >= 80 and stoch_d >= 80:
                signal = 'overbought'  # 과매수
            else:
                signal = 'neutral'   # 중립
                
            details = {
                'stoch_k': stoch_k,
                'stoch_d': stoch_d
            }
            
            return signal, details
        except:
            return 'neutral', {}
    
    @staticmethod
    def calculate_moving_averages(data: pd.DataFrame) -> Tuple[str, Dict]:
        """이동평균선 분석"""
        try:
            ma5 = data['Close'].rolling(5).mean().iloc[-1]
            ma20 = data['Close'].rolling(20).mean().iloc[-1]
            ma60 = data['Close'].rolling(60).mean().iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # 추세 분석
            if ma5 > ma20 > ma60 and current_price > ma5:
                trend = 'uptrend'
            elif ma5 < ma20 < ma60 and current_price < ma5:
                trend = 'downtrend'
            else:
                trend = 'sideways'
                
            details = {
                'ma5': ma5,
                'ma20': ma20,
                'ma60': ma60,
                'current_price': current_price
            }
            
            return trend, details
        except:
            return 'sideways', {}

# ========================================================================================
# 💰 분할매매 관리 클래스 (찾기 쉽게 분리)
# ========================================================================================
class PositionManager:
    """🔧 분할매매 및 포지션 관리"""
    
    @staticmethod
    def create_split_buy_plan(total_amount: float, current_price: float, 
                            confidence: float) -> Tuple[int, List[Dict]]:
        """분할 매수 계획 생성"""
        try:
            # 신뢰도에 따른 분할 전략
            if confidence >= 0.8:
                # 높은 신뢰도: 50% + 30% + 20%
                ratios = [0.5, 0.3, 0.2]
                triggers = [0, -0.02, -0.04]  # 0%, -2%, -4% 에서 매수
            elif confidence >= 0.6:
                # 중간 신뢰도: 40% + 35% + 25%
                ratios = [0.4, 0.35, 0.25]
                triggers = [0, -0.03, -0.05]  # 0%, -3%, -5%
            else:
                # 낮은 신뢰도: 30% + 35% + 35%
                ratios = [0.3, 0.35, 0.35]
                triggers = [0, -0.04, -0.06]  # 0%, -4%, -6%
            
            total_shares = int(total_amount / current_price / 100) * 100  # 100주 단위
            split_plan = []
            
            for i, (ratio, trigger) in enumerate(zip(ratios, triggers)):
                shares = int(total_shares * ratio / 100) * 100
                target_price = current_price * (1 + trigger)
                
                split_plan.append({
                    'step': i + 1,
                    'shares': shares,
                    'target_price': target_price,
                    'ratio': ratio,
                    'executed': False
                })
            
            return total_shares, split_plan
            
        except Exception as e:
            logger.error(f"분할 매수 계획 생성 실패: {e}")
            return 0, []
    
    @staticmethod
    def create_split_sell_plan(total_shares: int, current_price: float, 
                             target_price: float, confidence: float) -> List[Dict]:
        """분할 매도 계획 생성"""
        try:
            # 신뢰도에 따른 매도 전략
            if confidence >= 0.8:
                # 높은 신뢰도: 목표가 달성 시 50% 매도, 나머지 홀드
                sell_ratios = [0.5, 0.5]
                price_targets = [target_price, target_price * 1.1]
            else:
                # 일반: 목표가 달성 시 70% 매도, 30% 홀드
                sell_ratios = [0.7, 0.3]
                price_targets = [target_price, target_price * 1.05]
            
            split_plan = []
            remaining_shares = total_shares
            
            for i, (ratio, price_target) in enumerate(zip(sell_ratios, price_targets)):
                shares = int(remaining_shares * ratio / 100) * 100
                remaining_shares -= shares
                
                split_plan.append({
                    'step': i + 1,
                    'shares': shares,
                    'target_price': price_target,
                    'ratio': ratio,
                    'executed': False
                })
            
            return split_plan
            
        except Exception as e:
            logger.error(f"분할 매도 계획 생성 실패: {e}")
            return []

# ========================================================================================
# 🇯🇵 메인 일본 주식 전략 클래스 (핵심 로직)
# ========================================================================================
class JPStrategy:
    """🇯🇵 일본 주식 완전 자동화 전략"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        """전략 초기화"""
        # 설정 로드
        self.config = self._load_config(config_path)
        self.jp_config = self.config.get('jp_strategy', {})
        self.enabled = self.jp_config.get('enabled', True)
        
        # 🎯 20개 종목 자동 선별 설정
        self.target_stocks = 20
        self.min_market_cap = 100000000000  # 1000억엔
        self.min_avg_volume = 1000000       # 100만주
        
        # 💱 엔화 매매 설정
        self.yen_strong_threshold = 105     # 엔화 강세
        self.yen_weak_threshold = 110       # 엔화 약세
        self.current_usd_jpy = 0.0
        
        # ⚡ 기술적 지표 설정
        self.rsi_period = 10
        self.momentum_period = 10
        self.volume_spike_threshold = 1.3
        
        # 🛡️ 손절/익절 설정
        self.base_stop_loss = 0.08         # 기본 8%
        self.base_take_profit = 0.15       # 기본 15%
        self.max_hold_days = 30
        
        # 💰 분할매매 설정
        self.use_split_trading = True       # 분할매매 사용
        self.split_buy_steps = 3           # 3단계 분할 매수
        self.split_sell_steps = 2          # 2단계 분할 매도
        
        # 🔍 자동 선별된 종목들
        self.export_stocks = []    # 수출주 10개
        self.domestic_stocks = []  # 내수주 10개
        
        # 초기화
        if self.enabled:
            logger.info(f"🇯🇵 일본 주식 완전 자동화 전략 초기화")
            # 비동기 초기화는 별도 메서드에서
        else:
            logger.info("🇯🇵 일본 주식 전략이 비활성화되어 있습니다")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패, 기본값 사용: {e}")
            return {}

    # ========================================================================================
    # 🔧 유틸리티 메서드들 (찾기 쉽게 분리)
    # ========================================================================================
    
    async def _update_yen_rate(self):
        """USD/JPY 환율 업데이트"""
        try:
            ticker = yf.Ticker("USDJPY=X")
            data = ticker.history(period="1d")
            if not data.empty:
                self.current_usd_jpy = data['Close'].iloc[-1]
            else:
                self.current_usd_jpy = 107.5  # 기본값
        except Exception as e:
            logger.error(f"환율 조회 오류: {e}")
            self.current_usd_jpy = 107.5

    def _get_yen_signal(self) -> str:
        """엔화 신호 분석"""
        if self.current_usd_jpy <= self.yen_strong_threshold:
            return 'strong'  # 엔화 강세
        elif self.current_usd_jpy >= self.yen_weak_threshold:
            return 'weak'    # 엔화 약세
        else:
            return 'neutral'

    def _get_stock_type(self, symbol: str) -> str:
        """종목 타입 확인"""
        if symbol in self.export_stocks:
            return 'export'
        elif symbol in self.domestic_stocks:
            return 'domestic'
        else:
            return 'unknown'

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """섹터 분류"""
        if symbol in self.export_stocks:
            return 'EXPORT'
        elif symbol in self.domestic_stocks:
            return 'DOMESTIC'
        else:
            return 'UNKNOWN'

    async def _get_stock_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """주식 데이터 수집"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                logger.warning(f"데이터 없음: {symbol}")
                return pd.DataFrame()
            return data
        except Exception as e:
            logger.error(f"주식 데이터 수집 실패 {symbol}: {e}")
            return pd.DataFrame()

    def _set_default_stocks(self):
        """기본 종목 설정"""
        logger.info("기본 종목 리스트 사용")
        
        # 수출주 (제조업, 기술)
        self.export_stocks = [
            '7203.T', '6758.T', '9984.T', '6861.T', '4689.T',
            '6954.T', '7201.T', '6981.T', '8035.T', '6902.T'
        ]
        
        # 내수주 (금융, 소비재, 유틸리티)
        self.domestic_stocks = [
            '8306.T', '8316.T', '8411.T', '9983.T', '2914.T',
            '4568.T', '7974.T', '9432.T', '8267.T', '5020.T'
        ]

    # ========================================================================================
    # 📊 통합 기술적 분석 메서드 (핵심 로직)
    # ========================================================================================
    
    def _analyze_technical_indicators(self, data: pd.DataFrame) -> Tuple[float, Dict]:
        """통합 기술적 지표 분석"""
        try:
            if len(data) < 60:  # 충분한 데이터 필요
                return 0.0, {}
            
            # 각 지표 계산
            rsi = TechnicalIndicators.calculate_rsi(data, self.rsi_period)
            macd_signal, macd_details = TechnicalIndicators.calculate_macd(data)
            bb_signal, bb_details = TechnicalIndicators.calculate_bollinger_bands(data)
            stoch_signal, stoch_details = TechnicalIndicators.calculate_stochastic(data)
            ma_trend, ma_details = TechnicalIndicators.calculate_moving_averages(data)
            
            # 거래량 분석
            volume = data['Volume']
            recent_volume = volume.tail(3).mean()
            avg_volume = volume.tail(15).head(12).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 종합 점수 계산 (각 지표별 가중치)
            total_score = 0.0
            max_score = 0.0
            
            # 1. RSI 점수 (가중치 20%)
            weight = 0.2
            if 30 <= rsi <= 70:
                total_score += weight * 0.8  # 정상 구간
            elif rsi < 30:
                total_score += weight * 1.0  # 과매도 (매수 기회)
            elif rsi > 70:
                total_score += weight * 0.3  # 과매수 (주의)
            max_score += weight
            
            # 2. MACD 점수 (가중치 25%)
            weight = 0.25
            if macd_signal == 'bullish':
                total_score += weight * 1.0
            elif macd_signal == 'bearish':
                total_score += weight * 0.2
            else:
                total_score += weight * 0.5
            max_score += weight
            
            # 3. 볼린저 밴드 점수 (가중치 20%)
            weight = 0.2
            if bb_signal == 'lower':
                total_score += weight * 1.0  # 과매도
            elif bb_signal == 'upper':
                total_score += weight * 0.3  # 과매수
            else:
                total_score += weight * 0.6  # 중간
            max_score += weight
            
            # 4. 스토캐스틱 점수 (가중치 15%)
            weight = 0.15
            if stoch_signal == 'oversold':
                total_score += weight * 1.0
            elif stoch_signal == 'overbought':
                total_score += weight * 0.3
            else:
                total_score += weight * 0.6
            max_score += weight
            
            # 5. 이동평균 추세 점수 (가중치 15%)
            weight = 0.15
            if ma_trend == 'uptrend':
                total_score += weight * 1.0
            elif ma_trend == 'downtrend':
                total_score += weight * 0.2
            else:
                total_score += weight * 0.5
            max_score += weight
            
            # 6. 거래량 보너스 (가중치 5%)
            weight = 0.05
            if volume_ratio >= self.volume_spike_threshold:
                total_score += weight * 1.0
            max_score += weight
            
            # 정규화된 점수
            technical_score = total_score / max_score if max_score > 0 else 0.5
            
            # 상세 정보
            details = {
                'rsi': rsi,
                'macd_signal': macd_signal,
                'macd_details': macd_details,
                'bollinger_signal': bb_signal,
                'bollinger_details': bb_details,
                'stochastic_signal': stoch_signal,
                'stochastic_details': stoch_details,
                'ma_trend': ma_trend,
                'ma_details': ma_details,
                'volume_ratio': volume_ratio,
                'technical_score': technical_score
            }
            
            return technical_score, details
            
        except Exception as e:
            logger.error(f"기술적 지표 분석 실패: {e}")
            return 0.0, {}

    # ========================================================================================
    # 💱 엔화 + 기술적 지표 통합 분석 (핵심 로직)
    # ========================================================================================
    
    def _analyze_yen_technical_signal(self, symbol: str, technical_score: float, 
                                    technical_details: Dict) -> Tuple[str, float, str]:
        """엔화 + 기술적 지표 통합 분석"""
        try:
            yen_signal = self._get_yen_signal()
            stock_type = self._get_stock_type(symbol)
            
            total_score = 0.0
            reasons = []
            
            # 1. 엔화 기반 점수 (40% 가중치)
            yen_score = 0.0
            if yen_signal == 'strong' and stock_type == 'domestic':
                yen_score = 0.4
                reasons.append("엔화강세+내수주")
            elif yen_signal == 'weak' and stock_type == 'export':
                yen_score = 0.4
                reasons.append("엔화약세+수출주")
            elif yen_signal == 'neutral':
                yen_score = 0.2
                reasons.append("엔화중립")
            else:
                yen_score = 0.1
                reasons.append("엔화불리")
            
            total_score += yen_score
            
            # 2. 기술적 지표 점수 (60% 가중치)
            tech_weighted = technical_score * 0.6
            total_score += tech_weighted
            
            # 기술적 지표 설명
            rsi = technical_details.get('rsi', 50)
            macd_signal = technical_details.get('macd_signal', 'neutral')
            bb_signal = technical_details.get('bollinger_signal', 'middle')
            ma_trend = technical_details.get('ma_trend', 'sideways')
            
            tech_reasons = []
            if technical_score >= 0.7:
                tech_reasons.append("기술적강세")
            elif technical_score <= 0.4:
                tech_reasons.append("기술적약세")
            else:
                tech_reasons.append("기술적중립")
            
            tech_reasons.append(f"RSI({rsi:.0f})")
            tech_reasons.append(f"MACD({macd_signal})")
            tech_reasons.append(f"추세({ma_trend})")
            
            reasons.extend(tech_reasons)
            
            # 최종 판단
            if total_score >= 0.65:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= 0.35:
                action = 'sell'
                confidence = min(1 - total_score, 0.95)
            else:
                action = 'hold'
                confidence = 0.5
            
            reasoning = f"엔화({yen_signal})+{stock_type}: " + " | ".join(reasons)
            
            return action, confidence, reasoning
            
        except Exception as e:
            logger.error(f"통합 신호 분석 실패: {e}")
            return 'hold', 0.5, "분석 실패"

    # ========================================================================================
    # 🛡️ 동적 손절/익절 계산 (리스크 관리)
    # ========================================================================================
    
    def _calculate_dynamic_stop_take(self, current_price: float, confidence: float, 
                                   stock_type: str, yen_signal: str) -> Tuple[float, float, int]:
        """동적 손절/익절 계산"""
        try:
            # 기본값
            stop_loss_pct = self.base_stop_loss
            take_profit_pct = self.base_take_profit
            hold_days = self.max_hold_days
            
            # 1. 엔화 기반 조정
            if yen_signal == 'strong' and stock_type == 'domestic':
                stop_loss_pct = 0.06   # 6%
                take_profit_pct = 0.12 # 12%
                hold_days = 25
            elif yen_signal == 'weak' and stock_type == 'export':
                stop_loss_pct = 0.10   # 10%
                take_profit_pct = 0.18 # 18%
                hold_days = 35
            elif yen_signal != 'neutral':
                stop_loss_pct = 0.05   # 5%
                take_profit_pct = 0.08 # 8%
                hold_days = 20
            
            # 2. 신뢰도 기반 조정
            if confidence >= 0.8:
                stop_loss_pct *= 0.8    # 손절 타이트
                take_profit_pct *= 1.3  # 익절 크게
                hold_days += 10
            elif confidence <= 0.6:
                stop_loss_pct *= 0.6    # 손절 매우 타이트
                take_profit_pct *= 0.8  # 익절 작게
                hold_days -= 10
            
            # 3. 범위 제한
            stop_loss_pct = max(0.03, min(0.12, stop_loss_pct))
            take_profit_pct = max(0.05, min(0.25, take_profit_pct))
            hold_days = max(15, min(45, hold_days))
            
            # 4. 최종 가격 계산
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
            
            return stop_loss, take_profit, hold_days
            
        except Exception as e:
            logger.error(f"손절/익절 계산 실패: {e}")
            return (current_price * 0.92, current_price * 1.15, 30)

    # ========================================================================================
    # 🎯 메인 종목 분석 메서드 (모든 기능 통합)
    # ========================================================================================
    
    async def analyze_symbol(self, symbol: str) -> JPStockSignal:
        """개별 종목 완전 분석 (모든 기능 통합)"""
        if not self.enabled:
            return self._create_disabled_signal(symbol)
        
        try:
            # 1. 환율 업데이트
            await self._update_yen_rate()
            
            # 2. 주식 데이터 수집
            data = await self._get_stock_data(symbol)
            if data.empty:
                raise ValueError(f"주식 데이터 없음: {symbol}")
            
            current_price = data['Close'].iloc[-1]
            
            # 3. 📊 기술적 지표 분석
            technical_score, technical_details = self._analyze_technical_indicators(data)
            
            # 4. 💱 엔화 + 기술적 지표 통합 분석
            action, confidence, reasoning = self._analyze_yen_technical_signal(
                symbol, technical_score, technical_details
            )
            
            # 5. 🛡️ 동적 손절/익절 계산
            stop_loss, take_profit, max_hold_days = self
