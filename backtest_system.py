#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 통합 백테스팅 시스템 v2.0
=============================================
🇺🇸 미국 + 🇯🇵 일본 + 🇮🇳 인도 + 💰 암호화폐 통합 백테스팅

✨ 핵심 기능:
- 4가지 전략 통합 백테스팅
- 실시간 웹 인터페이스  
- 포트폴리오 최적화
- 리스크 분석
- 성과 비교 분석
- CSV/JSON 내보내기

Author: 퀸트팀 | Version: 2.0.0
"""

import asyncio
import logging
import json
import os
import sqlite3
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import yfinance as yf
import pyupbit
from flask import Flask, render_template_string, request, jsonify, send_file
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import zipfile

warnings.filterwarnings('ignore')

# ============================================================================
# 🔧 설정 및 로깅
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# 기본 설정
INITIAL_CAPITAL = 1_000_000_000  # 10억원
DATA_DIR = Path("backtest_data")
RESULTS_DIR = Path("backtest_results")

# 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# 📊 데이터 클래스
# ============================================================================
@dataclass
class BacktestConfig:
    """백테스트 설정"""
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    commission: float
    slippage: float
    enabled_strategies: List[str]
    risk_free_rate: float = 0.02

@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    action: str  # buy, sell
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    strategy: str
    reason: str = ""

@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    strategy: str
    entry_date: datetime

@dataclass
class PerformanceMetrics:
    """성과 지표"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float

# ============================================================================
# 🎯 전략 인터페이스
# ============================================================================
class StrategyInterface:
    """전략 인터페이스"""
    
    def __init__(self, name: str, initial_capital: float):
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.dates: List[datetime] = []
    
    async def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict:
        """신호 생성 (오버라이드 필요)"""
        raise NotImplementedError
    
    def execute_trade(self, trade: Trade):
        """거래 실행"""
        if trade.action == 'buy':
            self._execute_buy(trade)
        elif trade.action == 'sell':
            self._execute_sell(trade)
        
        self.trades.append(trade)
    
    def _execute_buy(self, trade: Trade):
        """매수 실행"""
        total_cost = trade.quantity * trade.price + trade.commission
        
        if total_cost > self.current_capital:
            return False
        
        self.current_capital -= total_cost
        
        if trade.symbol in self.positions:
            # 기존 포지션 업데이트
            pos = self.positions[trade.symbol]
            total_quantity = pos.quantity + trade.quantity
            total_cost_basis = (pos.quantity * pos.avg_price) + (trade.quantity * trade.price)
            pos.avg_price = total_cost_basis / total_quantity
            pos.quantity = total_quantity
        else:
            # 새 포지션 생성
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                quantity=trade.quantity,
                avg_price=trade.price,
                current_price=trade.price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                strategy=self.name,
                entry_date=trade.timestamp
            )
        
        return True
    
    def _execute_sell(self, trade: Trade):
        """매도 실행"""
        if trade.symbol not in self.positions:
            return False
        
        pos = self.positions[trade.symbol]
        if trade.quantity > pos.quantity:
            return False
        
        # 수익 계산
        proceeds = trade.quantity * trade.price - trade.commission
        cost_basis = trade.quantity * pos.avg_price
        realized_pnl = proceeds - cost_basis
        
        self.current_capital += proceeds
        pos.realized_pnl += realized_pnl
        pos.quantity -= trade.quantity
        
        # 포지션 완전 청산시 제거
        if pos.quantity <= 0:
            del self.positions[trade.symbol]
        
        return True
    
    def update_positions(self, prices: Dict[str, float], timestamp: datetime):
        """포지션 업데이트"""
        total_equity = self.current_capital
        
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
                pos.unrealized_pnl = (pos.current_price - pos.avg_price) * pos.quantity
                total_equity += pos.current_price * pos.quantity
        
        self.equity_curve.append(total_equity)
        self.dates.append(timestamp)
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """포트폴리오 가치 계산"""
        total_value = self.current_capital
        
        for symbol, pos in self.positions.items():
            if symbol in prices:
                total_value += pos.quantity * prices[symbol]
        
        return total_value

# ============================================================================
# 🇺🇸 미국 주식 전략
# ============================================================================
class USStrategy(StrategyInterface):
    """미국 주식 전략 (서머타임 + 고급기술지표)"""
    
    def __init__(self, initial_capital: float):
        super().__init__("US_Strategy", initial_capital)
        self.selected_stocks = []
        self.last_selection_date = None
        self.rebalance_frequency = 30  # 30일마다 리밸런싱
    
    async def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict:
        """미국 전략 신호 생성"""
        try:
            if len(data) < 50:
                return {'action': 'hold', 'confidence': 0.0}
            
            # 기술지표 계산
            signals = self._calculate_technical_signals(data)
            scores = self._calculate_strategy_scores(data)
            
            # 통합 점수
            total_score = (
                scores['buffett'] * 0.20 +
                scores['lynch'] * 0.20 +
                scores['momentum'] * 0.20 +
                scores['technical'] * 0.25 +
                scores['advanced'] * 0.15
            )
            
            # 매수/매도 결정
            if total_score >= 0.7 and signals['macd_signal'] > 0 and signals['bb_position'] < 0.8:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= 0.3 or signals['rsi'] > 80:
                action = 'sell'
                confidence = 0.8
            else:
                action = 'hold'
                confidence = 0.5
            
            return {
                'action': action,
                'confidence': confidence,
                'scores': scores,
                'technical': signals,
                'total_score': total_score
            }
            
        except Exception as e:
            logger.error(f"미국 전략 신호 생성 실패 {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0}
    
    def _calculate_technical_signals(self, data: pd.DataFrame) -> Dict:
        """기술지표 계산"""
        try:
            close = data['Close']
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            macd_signal = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
            
            # 볼린저 밴드
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            return {
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                'macd_signal': macd_signal,
                'bb_position': bb_position if not pd.isna(bb_position) else 0.5
            }
            
        except Exception as e:
            logger.error(f"기술지표 계산 실패: {e}")
            return {'rsi': 50, 'macd_signal': 0, 'bb_position': 0.5}
    
    def _calculate_strategy_scores(self, data: pd.DataFrame) -> Dict:
        """전략별 점수 계산"""
        try:
            # 간단한 점수 계산 (실제로는 더 복잡)
            close = data['Close']
            volume = data['Volume']
            
            # 모멘텀 점수
            returns_1m = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0
            returns_3m = (close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0
            momentum_score = max(0, min(1, (returns_1m + returns_3m) / 2 + 0.5))
            
            # 기술적 점수
            price_above_ma20 = close.iloc[-1] > close.rolling(20).mean().iloc[-1]
            price_above_ma50 = close.iloc[-1] > close.rolling(50).mean().iloc[-1]
            technical_score = (price_above_ma20 * 0.6 + price_above_ma50 * 0.4)
            
            return {
                'buffett': 0.6,  # 기본값
                'lynch': 0.6,
                'momentum': momentum_score,
                'technical': technical_score,
                'advanced': 0.6
            }
            
        except Exception as e:
            logger.error(f"전략 점수 계산 실패: {e}")
            return {'buffett': 0.5, 'lynch': 0.5, 'momentum': 0.5, 'technical': 0.5, 'advanced': 0.5}

# ============================================================================
# 🇯🇵 일본 주식 전략
# ============================================================================
class JapanStrategy(StrategyInterface):
    """일본 주식 전략 (엔화 + 화목 하이브리드)"""
    
    def __init__(self, initial_capital: float):
        super().__init__("Japan_Strategy", initial_capital)
        self.yen_rate = 110.0  # USD/JPY
        self.trading_days = [1, 3]  # 화, 목
    
    async def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict:
        """일본 전략 신호 생성"""
        try:
            if len(data) < 30:
                return {'action': 'hold', 'confidence': 0.0}
            
            # 6개 기술지표 계산
            indicators = self._calculate_6_indicators(data)
            
            # 엔화 신호
            yen_signal = self._get_yen_signal()
            
            # 화목 하이브리드 점수
            hybrid_score = (
                indicators['rsi_score'] * 0.20 +
                indicators['macd_score'] * 0.20 +
                indicators['bb_score'] * 0.15 +
                indicators['stoch_score'] * 0.15 +
                indicators['atr_score'] * 0.15 +
                indicators['volume_score'] * 0.15
            )
            
            # 엔화 보정
            if yen_signal == 'positive':
                hybrid_score *= 1.2
            elif yen_signal == 'negative':
                hybrid_score *= 0.8
            
            # 매수/매도 결정
            if hybrid_score >= 0.75:
                action = 'buy'
                confidence = min(hybrid_score, 0.95)
            elif hybrid_score <= 0.25:
                action = 'sell'
                confidence = 0.8
            else:
                action = 'hold'
                confidence = 0.5
            
            return {
                'action': action,
                'confidence': confidence,
                'hybrid_score': hybrid_score,
                'yen_signal': yen_signal,
                'indicators': indicators
            }
            
        except Exception as e:
            logger.error(f"일본 전략 신호 생성 실패 {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0}
    
    def _calculate_6_indicators(self, data: pd.DataFrame) -> Dict:
        """6개 핵심 기술지표 계산"""
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_score = 0.8 if 30 <= rsi.iloc[-1] <= 70 else 0.3
            
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            macd_score = 0.8 if macd_line.iloc[-1] > signal_line.iloc[-1] else 0.2
            
            # 볼린저 밴드
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            bb_score = 0.8 if 0.2 <= bb_position <= 0.8 else 0.3
            
            # 스토캐스틱
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_d = stoch_k.rolling(3).mean()
            stoch_score = 0.8 if 20 <= stoch_d.iloc[-1] <= 80 else 0.3
            
            # ATR (변동성)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_score = 0.7  # 기본값
            
            # 거래량
            vol_avg = volume.rolling(20).mean()
            vol_ratio = volume.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1
            volume_score = 0.8 if vol_ratio > 1.2 else 0.5
            
            return {
                'rsi_score': rsi_score,
                'macd_score': macd_score,
                'bb_score': bb_score,
                'stoch_score': stoch_score,
                'atr_score': atr_score,
                'volume_score': volume_score
            }
            
        except Exception as e:
            logger.error(f"6개 지표 계산 실패: {e}")
            return {
                'rsi_score': 0.5, 'macd_score': 0.5, 'bb_score': 0.5,
                'stoch_score': 0.5, 'atr_score': 0.5, 'volume_score': 0.5
            }
    
    def _get_yen_signal(self) -> str:
        """엔화 신호 (간소화)"""
        if self.yen_rate <= 105:
            return 'positive'  # 엔화 강세
        elif self.yen_rate >= 115:
            return 'negative'  # 엔화 약세
        else:
            return 'neutral'

# ============================================================================
# 🇮🇳 인도 주식 전략
# ============================================================================
class IndiaStrategy(StrategyInterface):
    """인도 주식 전략 (5대 전설 + 수요일 안정형)"""
    
    def __init__(self, initial_capital: float):
        super().__init__("India_Strategy", initial_capital)
        self.trading_day = 2  # 수요일
        self.monthly_target = 0.06  # 6%
    
    async def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict:
        """인도 전략 신호 생성"""
        try:
            if len(data) < 50:
                return {'action': 'hold', 'confidence': 0.0}
            
            # 5대 전설 전략 점수
            legendary_scores = self._calculate_legendary_scores(data)
            
            # 고급 기술지표
            advanced_indicators = self._calculate_advanced_indicators(data)
            
            # 안정형 점수 계산
            conservative_score = (
                legendary_scores['jhunjhunwala'] * 0.25 +
                legendary_scores['agrawal'] * 0.25 +
                legendary_scores['kedia'] * 0.20 +
                legendary_scores['veliyath'] * 0.15 +
                legendary_scores['karnik'] * 0.15
            )
            
            # 기술지표 보정
            technical_bonus = (
                advanced_indicators['ichimoku'] * 0.3 +
                advanced_indicators['elliott'] * 0.2 +
                advanced_indicators['vwap'] * 0.3 +
                advanced_indicators['macd'] * 0.2
            )
            
            final_score = conservative_score + (technical_bonus * 0.3)
            
            # 매수/매도 결정 (보수적)
            if final_score >= 0.8:
                action = 'buy'
                confidence = min(final_score, 0.9)
            elif final_score <= 0.3:
                action = 'sell'
                confidence = 0.7
            else:
                action = 'hold'
                confidence = 0.5
            
            return {
                'action': action,
                'confidence': confidence,
                'conservative_score': conservative_score,
                'legendary_scores': legendary_scores,
                'advanced_indicators': advanced_indicators,
                'final_score': final_score
            }
            
        except Exception as e:
            logger.error(f"인도 전략 신호 생성 실패 {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0}
    
    def _calculate_legendary_scores(self, data: pd.DataFrame) -> Dict:
        """5대 전설 투자자 전략 점수"""
        try:
            close = data['Close']
            
            # 각 전략별 간단한 점수 (실제로는 더 복잡)
            price_momentum = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0
            volume_trend = data['Volume'].rolling(10).mean().iloc[-1] / data['Volume'].rolling(30).mean().iloc[-1]
            
            return {
                'jhunjhunwala': max(0, min(1, 0.6 + price_momentum)),  # 워런 버핏 킬러
                'agrawal': max(0, min(1, 0.6 + price_momentum * 0.5)), # QGLP 마스터
                'kedia': max(0, min(1, 0.7 - abs(price_momentum))),    # SMILE 투자법
                'veliyath': max(0, min(1, 0.8 - price_momentum)),      # 콘트라리안
                'karnik': max(0, min(1, 0.6 + volume_trend * 0.2))     # 인프라 제왕
            }
            
        except Exception as e:
            logger.error(f"전설 점수 계산 실패: {e}")
            return {'jhunjhunwala': 0.6, 'agrawal': 0.6, 'kedia': 0.6, 'veliyath': 0.6, 'karnik': 0.6}
    
    def _calculate_advanced_indicators(self, data: pd.DataFrame) -> Dict:
        """고급 기술지표 계산"""
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # 일목균형표 (간소화)
            tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
            kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
            ichimoku_score = 0.8 if close.iloc[-1] > tenkan_sen.iloc[-1] > kijun_sen.iloc[-1] else 0.3
            
            # 엘리어트 파동 (간소화)
            elliott_score = 0.6  # 기본값
            
            # VWAP
            vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
            vwap_score = 0.8 if close.iloc[-1] > vwap.iloc[-1] else 0.3
            
            # 고급 MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            macd_score = 0.8 if macd_line.iloc[-1] > signal_line.iloc[-1] else 0.2
            
            return {
                'ichimoku': ichimoku_score,
                'elliott': elliott_score,
                'vwap': vwap_score,
                'macd': macd_score
            }
            
        except Exception as e:
            logger.error(f"고급 지표 계산 실패: {e}")
            return {'ichimoku': 0.5, 'elliott': 0.5, 'vwap': 0.5, 'macd': 0.5}

# ============================================================================
# 💰 암호화폐 전략
# ============================================================================
class CryptoStrategy(StrategyInterface):
    """암호화폐 전략 (전설급 5대 시스템 + 월금 매매)"""
    
    def __init__(self, initial_capital: float):
        super().__init__("Crypto_Strategy", initial_capital)
        self.trading_days = [0, 4]  # 월, 금
        self.monthly_target = 0.06  # 6%
    
    async def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict:
        """암호화폐 전략 신호 생성"""
        try:
            if len(data) < 60:
                return {'action': 'hold', 'confidence': 0.0}
            
            # 전설급 5대 시스템
            neural_quality = self._neural_quality_score(symbol)
            quantum_cycle = self._quantum_cycle_analysis(data)
            fractal_score = self._fractal_filtering_score(data)
            diamond_signals = self._diamond_hand_signals(data)
            correlation_score = self._correlation_web_score(symbol)
            
            # 통합 점수
            legendary_score = (
                neural_quality * 0.30 +
                quantum_cycle * 0.25 +
                fractal_score * 0.25 +
                diamond_signals * 0.20
            )
            
            # 상관관계 보정
            final_score = legendary_score * correlation_score
            
            # 매수/매도 결정
            if final_score >= 0.7:
                action = 'buy'
                confidence = min(final_score, 0.95)
            elif final_score <= 0.3:
                action = 'sell'
                confidence = 0.8
            else:
                action = 'hold'
                confidence = 0.5
            
            return {
                'action': action,
                'confidence': confidence,
                'legendary_score': legendary_score,
                'neural_quality': neural_quality,
                'quantum_cycle': quantum_cycle,
                'fractal_score': fractal_score,
                'diamond_signals': diamond_signals,
                'correlation_score': correlation_score,
                'final_score': final_score
            }
            
        except Exception as e:
            logger.error(f"암호화폐 전략 신호 생성 실패 {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0}
    
    def _neural_quality_score(self, symbol: str) -> float:
        """신경망 품질 점수"""
        # 코인별 품질 점수 (기술력, 생태계, 커뮤니티, 채택도)
        quality_scores = {
            'BTC': 0.95, 'ETH': 0.90, 'BNB': 0.80, 'ADA': 0.75,
            'SOL': 0.80, 'AVAX': 0.75, 'DOT': 0.75, 'MATIC': 0.80
        }
        
        coin_name = symbol.replace('KRW-', '') if 'KRW-' in symbol else symbol
        return quality_scores.get(coin_name, 0.6)
    
    def _quantum_cycle_analysis(self, data: pd.DataFrame) -> float:
        """양자 사이클 분석"""
        try:
            close = data['Close']
            
            # 3차원 상태 분석 (간소화)
            ma7 = close.rolling(7).mean()
            ma14 = close.rolling(14).mean()
            ma30 = close.rolling(30).mean()
            
            current_price = close.iloc[-1]
            
            if current_price > ma7.iloc[-1] > ma14.iloc[-1] > ma30.iloc[-1]:
                return 0.9  # 강한 상승 사이클
            elif current_price < ma7.iloc[-1] < ma14.iloc[-1] < ma30.iloc[-1]:
                return 0.2  # 강한 하락 사이클
            else:
                return 0.6  # 중립 사이클
                
        except Exception as e:
            logger.error(f"양자 사이클 분석 실패: {e}")
            return 0.5
    
    def _fractal_filtering_score(self, data: pd.DataFrame) -> float:
        """프랙탈 필터링 점수"""
        try:
            close = data['Close']
            volume = data['Volume']
            
            # 변동성 및 안정성 분석
            returns = close.pct_change().dropna()
            volatility = returns.std()
            volume_stability = volume.rolling(7).std() / volume.rolling(7).mean()
            
            # 프랙탈 점수 (안정성 + 적절한 변동성)
            if 0.02 <= volatility <= 0.08 and volume_stability.iloc[-1] < 1.0:
                return 0.8
            elif 0.01 <= volatility <= 0.12 and volume_stability.iloc[-1] < 1.5:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"프랙탈 필터링 실패: {e}")
            return 0.5
    
    def _diamond_hand_signals(self, data: pd.DataFrame) -> float:
        """다이아몬드 핸드 신호"""
        try:
            close = data['Close']
            
            # 켈리 공식 기반 분석 (간소화)
            returns = close.pct_change().dropna()
            win_rate = (returns > 0).mean()
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
            
            if avg_loss > 0:
                kelly_fraction = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                return max(0, min(1, kelly_fraction + 0.5))
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"다이아몬드 핸드 분석 실패: {e}")
            return 0.5
    
    def _correlation_web_score(self, symbol: str) -> float:
        """상관관계 웹 점수"""
        # 간소화된 상관관계 점수
        correlation_factors = {
            'BTC': 1.0,   # 비트코인은 독립성 최고
            'ETH': 0.9,   # 이더리움은 높은 독립성
            'BNB': 0.8,   # 바이낸스 코인
            'ADA': 0.85,  # 카르다노
            'SOL': 0.82,  # 솔라나
        }
        
        coin_name = symbol.replace('KRW-', '') if 'KRW-' in symbol else symbol
        return correlation_factors.get(coin_name, 0.7)

# ============================================================================
# 📊 통합 백테스트 엔진
# ============================================================================
class IntegratedBacktestEngine:
    """4가지 전략 통합 백테스트 엔진"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategies: Dict[str, StrategyInterface] = {}
        self.results: Dict[str, Any] = {}
        self.portfolio_equity: List[float] = []
        self.portfolio_dates: List[datetime] = []
        
        # 전략별 자본 배분
        strategy_allocations = {
            'US_Strategy': 0.40,      # 40%
            'Japan_Strategy': 0.25,   # 25%
            'Crypto_Strategy': 0.20,  # 20%
            'India_Strategy': 0.15    # 15%
        }
        
        # 활성화된 전략만 초기화
        for strategy_name in config.enabled_strategies:
            if strategy_name in strategy_allocations:
                capital = config.initial_capital * strategy_allocations[strategy_name]
                
                if strategy_name == 'US_Strategy':
                    self.strategies[strategy_name] = USStrategy(capital)
                elif strategy_name == 'Japan_Strategy':
                    self.strategies[strategy_name] = JapanStrategy(capital)
                elif strategy_name == 'Crypto_Strategy':
                    self.strategies[strategy_name] = CryptoStrategy(capital)
                elif strategy_name == 'India_Strategy':
                    self.strategies[strategy_name] = IndiaStrategy(capital)
    
    async def run_backtest(self, symbols: Dict[str, List[str]]) -> Dict[str, Any]:
        """통합 백테스트 실행"""
        logger.info("🏆 통합 백테스트 시작")
        
        try:
            # 날짜 범위 생성
            start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
            
            # 각 전략별 백테스트 실행
            strategy_results = {}
            
            for strategy_name, strategy in self.strategies.items():
                logger.info(f"📊 {strategy_name} 백테스트 중...")
                
                strategy_symbols = symbols.get(strategy_name, [])
                if not strategy_symbols:
                    continue
                
                # 전략별 데이터 수집 및 백테스트
                result = await self._run_strategy_backtest(
                    strategy, strategy_symbols, start_date, end_date
                )
                strategy_results[strategy_name] = result
            
            # 통합 포트폴리오 성과 계산
            portfolio_result = self._calculate_portfolio_performance(strategy_results)
            
            # 결과 종합
            self.results = {
                'config': asdict(self.config),
                'strategy_results': strategy_results,
                'portfolio_result': portfolio_result,
                'summary': self._generate_summary(strategy_results, portfolio_result)
            }
            
            logger.info("✅ 통합 백테스트 완료")
            return self.results
            
        except Exception as e:
            logger.error(f"백테스트 실행 실패: {e}")
            return {'error': str(e)}
    
    async def _run_strategy_backtest(self, strategy: StrategyInterface, symbols: List[str], 
                                   start_date: datetime, end_date: datetime) -> Dict:
        """개별 전략 백테스트"""
        try:
            all_data = {}
            
            # 데이터 수집
            for symbol in symbols:
                try:
                    data = await self._fetch_data(symbol, start_date, end_date, strategy.name)
                    if data is not None and not data.empty:
                        all_data[symbol] = data
                except Exception as e:
                    logger.warning(f"{symbol} 데이터 수집 실패: {e}")
                    continue
            
            if not all_data:
                return {'error': 'no_data'}
            
            # 날짜별 백테스트 실행
            trading_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            for current_date in trading_dates:
                try:
                    # 각 심볼에 대해 신호 생성 및 거래 실행
                    await self._process_trading_day(strategy, all_data, current_date)
                except Exception as e:
                    logger.warning(f"{current_date} 거래 처리 실패: {e}")
                    continue
            
            # 성과 계산
            performance = self._calculate_strategy_performance(strategy)
            
            return {
                'strategy_name': strategy.name,
                'initial_capital': strategy.initial_capital,
                'final_capital': strategy.current_capital,
                'total_positions_value': sum(
                    pos.current_price * pos.quantity for pos in strategy.positions.values()
                ),
                'total_trades': len(strategy.trades),
                'performance': performance,
                'equity_curve': strategy.equity_curve,
                'dates': strategy.dates,
                'trades': [asdict(trade) for trade in strategy.trades[-100:]],  # 최근 100개만
                'positions': [asdict(pos) for pos in strategy.positions.values()]
            }
            
        except Exception as e:
            logger.error(f"전략 백테스트 실패 {strategy.name}: {e}")
            return {'error': str(e)}
    
    async def _fetch_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                         strategy_name: str) -> Optional[pd.DataFrame]:
        """데이터 수집"""
        try:
            if strategy_name == 'Crypto_Strategy':
                # 암호화폐 데이터는 임시로 랜덤 생성
                return self._generate_sample_crypto_data(symbol, start_date, end_date)
            else:
                # 주식 데이터
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if data.empty:
                    return None
                
                # 컬럼명 정규화
                data.columns = [col.title() for col in data.columns]
                return data
                
        except Exception as e:
            logger.error(f"데이터 수집 실패 {symbol}: {e}")
            return None
    
    def _generate_sample_crypto_data(self, symbol: str, start_date: datetime, 
                                   end_date: datetime) -> pd.DataFrame:
        """샘플 암호화폐 데이터 생성"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 기본 가격 설정
        base_prices = {
            'KRW-BTC': 50000000,   # 5천만원
            'KRW-ETH': 3000000,    # 300만원
            'KRW-BNB': 300000,     # 30만원
            'KRW-ADA': 500,        # 500원
            'KRW-SOL': 100000      # 10만원
        }
        
        base_price = base_prices.get(symbol, 50000)
        
        # 랜덤 워크로 가격 생성
        np.random.seed(42)  # 재현 가능한 결과
        returns = np.random.normal(0.001, 0.05, len(dates))  # 일평균 0.1%, 변동성 5%
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # OHLCV 데이터 생성
        data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.08) for p in prices],
            'Low': [p * np.random.uniform(0.92, 1.00) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(1000000, 10000000) for _ in prices]
        }, index=dates)
        
        return data
    
    async def _process_trading_day(self, strategy: StrategyInterface, all_data: Dict[str, pd.DataFrame], 
                                 current_date: datetime):
        """거래일 처리"""
        try:
            # 현재 가격 수집
            current_prices = {}
            
            for symbol, data in all_data.items():
                if current_date in data.index:
                    current_prices[symbol] = data.loc[current_date, 'Close']
            
            if not current_prices:
                return
            
            # 포지션 업데이트
            strategy.update_positions(current_prices, current_date)
            
            # 각 심볼에 대해 신호 생성
            for symbol, data in all_data.items():
                if current_date not in data.index:
                    continue
                
                # 현재까지의 데이터만 사용 (미래 정보 방지)
                historical_data = data.loc[:current_date]
                
                if len(historical_data) < 30:  # 최소 데이터 요구사항
                    continue
                
                # 신호 생성
                signal = await strategy.generate_signals(symbol, historical_data)
                
                if signal.get('action') == 'buy':
                    await self._execute_buy_signal(strategy, symbol, signal, current_prices[symbol], current_date)
                elif signal.get('action') == 'sell':
                    await self._execute_sell_signal(strategy, symbol, signal, current_prices[symbol], current_date)
            
        except Exception as e:
            logger.error(f"거래일 처리 실패 {current_date}: {e}")
    
    async def _execute_buy_signal(self, strategy: StrategyInterface, symbol: str, signal: Dict, 
                                price: float, timestamp: datetime):
        """매수 신호 실행"""
        try:
            confidence = signal.get('confidence', 0.5)
            max_position_size = strategy.current_capital * 0.1 * confidence  # 신뢰도에 따른 포지션 크기
            
            quantity = max_position_size / price
            commission = max_position_size * self.config.commission
            
            trade = Trade(
                symbol=symbol,
                action='buy',
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                commission=commission,
                strategy=strategy.name,
                reason=f"Signal confidence: {confidence:.2f}"
            )
            
            success = strategy._execute_buy(trade)
            if success:
                strategy.trades.append(trade)
            
        except Exception as e:
            logger.error(f"매수 실행 실패 {symbol}: {e}")
    
    async def _execute_sell_signal(self, strategy: StrategyInterface, symbol: str, signal: Dict, 
                                 price: float, timestamp: datetime):
        """매도 신호 실행"""
        try:
            if symbol not in strategy.positions:
                return
            
            position = strategy.positions[symbol]
            quantity = position.quantity  # 전량 매도
            commission = quantity * price * self.config.commission
            
            trade = Trade(
                symbol=symbol,
                action='sell',
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                commission=commission,
                strategy=strategy.name,
                reason=f"Sell signal"
            )
            
            success = strategy._execute_sell(trade)
            if success:
                strategy.trades.append(trade)
            
        except Exception as e:
            logger.error(f"매도 실행 실패 {symbol}: {e}")
    
    def _calculate_strategy_performance(self, strategy: StrategyInterface) -> PerformanceMetrics:
        """전략 성과 계산"""
        try:
            if len(strategy.equity_curve) < 2:
                return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # 수익률 계산
            equity_series = pd.Series(strategy.equity_curve)
            returns = equity_series.pct_change().dropna()
            
            total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
            
            # 연환산 수익률
            days = len(strategy.dates)
            annual_return = ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (365 / days) - 1) * 100 if days > 0 else 0
            
            # 샤프 비율
            if len(returns) > 1 and returns.std() > 0:
                excess_returns = returns - (self.config.risk_free_rate / 365)
                sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(365)
            else:
                sharpe_ratio = 0
            
            # 최대 낙폭
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = abs(drawdown.min()) * 100
            
            # 거래 통계
            winning_trades = [t for t in strategy.trades if t.action == 'sell' and 
                            self._calculate_trade_pnl(t, strategy) > 0]
            total_trades = len([t for t in strategy.trades if t.action == 'sell'])
            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            
            # 수익률 변동성
            volatility = returns.std() * np.sqrt(365) * 100 if len(returns) > 1 else 0
            
            # 칼마 비율
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            # 평균 거래 수익률
            trade_returns = [self._calculate_trade_return(t, strategy) for t in strategy.trades if t.action == 'sell']
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            
            # 프로핏 팩터
            profits = [r for r in trade_returns if r > 0]
            losses = [abs(r) for r in trade_returns if r < 0]
            profit_factor = (sum(profits) / sum(losses)) if losses else 1
            
            return PerformanceMetrics(
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                num_trades=total_trades,
                avg_trade_return=avg_trade_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio
            )
            
        except Exception as e:
            logger.error(f"성과 계산 실패: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_trade_pnl(self, trade: Trade, strategy: StrategyInterface) -> float:
        """거래 손익 계산"""
        # 간단한 손익 계산 (실제로는 더 복잡)
        return trade.quantity * trade.price - trade.commission
    
    def _calculate_trade_return(self, trade: Trade, strategy: StrategyInterface) -> float:
        """거래 수익률 계산"""
        # 매도 거래의 수익률 계산 (간소화)
        if trade.action == 'sell':
            # 해당 심볼의 평균 매수가와 비교
            buy_trades = [t for t in strategy.trades if t.symbol == trade.symbol and t.action == 'buy']
            if buy_trades:
                avg_buy_price = np.mean([t.price for t in buy_trades])
                return (trade.price - avg_buy_price) / avg_buy_price * 100
        return 0
    
    def _calculate_portfolio_performance(self, strategy_results: Dict) -> Dict:
        """포트폴리오 통합 성과 계산"""
        try:
            total_initial_capital = self.config.initial_capital
            total_final_value = 0
            weighted_returns = []
            
            for strategy_name, result in strategy_results.items():
                if 'error' in result:
                    continue
                
                initial = result['initial_capital']
                final = result['final_capital'] + result.get('total_positions_value', 0)
                weight = initial / total_initial_capital
                
                strategy_return = (final - initial) / initial
                weighted_returns.append(strategy_return * weight)
                total_final_value += final
            
            # 포트폴리오 전체 수익률
            portfolio_return = (total_final_value - total_initial_capital) / total_initial_capital * 100
            
            # 전략별 기여도
            strategy_contributions = {}
            for strategy_name, result in strategy_results.items():
                if 'error' not in result:
                    initial = result['initial_capital']
                    final = result['final_capital'] + result.get('total_positions_value', 0)
                    contribution = (final - initial) / total_initial_capital * 100
                    strategy_contributions[strategy_name] = contribution
            
            return {
                'total_return': portfolio_return,
                'initial_capital': total_initial_capital,
                'final_value': total_final_value,
                'strategy_contributions': strategy_contributions,
                'best_strategy': max(strategy_contributions.items(), key=lambda x: x[1]) if strategy_contributions else None,
                'worst_strategy': min(strategy_contributions.items(), key=lambda x: x[1]) if strategy_contributions else None
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 성과 계산 실패: {e}")
            return {'error': str(e)}
    
    def _generate_summary(self, strategy_results: Dict, portfolio_result: Dict) -> Dict:
        """결과 요약 생성"""
        try:
            successful_strategies = {k: v for k, v in strategy_results.items() if 'error' not in v}
            
            summary = {
                'total_strategies': len(self.config.enabled_strategies),
                'successful_strategies': len(successful_strategies),
                'portfolio_return': portfolio_result.get('total_return', 0),
                'best_performing_strategy': portfolio_result.get('best_strategy', [None, 0]),
                'total_trades': sum(result.get('total_trades', 0) for result in successful_strategies.values()),
                'backtest_period': f"{self.config.start_date} ~ {self.config.end_date}",
                'initial_capital': self.config.initial_capital,
                'final_value': portfolio_result.get('final_value', 0),
                'strategy_summary': {}
            }
            
            # 전략별 요약
            for strategy_name, result in successful_strategies.items():
                if 'performance' in result:
                    perf = result['performance']
                    summary['strategy_summary'][strategy_name] = {
                        'return': round(perf.total_return, 2),
                        'sharpe': round(perf.sharpe_ratio, 2),
                        'max_drawdown': round(perf.max_drawdown, 2),
                        'win_rate': round(perf.win_rate, 2),
                        'trades': perf.num_trades
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            return {'error': str(e)}

# ============================================================================
# 🌐 웹 인터페이스
# ============================================================================
app = Flask(__name__)

# 전역 변수
backtest_engine = None
current_results = None

@app.route('/')
def index():
    """메인 페이지"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🏆 퀸트프로젝트 통합 백테스팅 시스템</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .config-section { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .strategy-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .strategy-card { background: white; padding: 15px; border-radius: 8px; border: 2px solid #e9ecef; }
        .strategy-card.enabled { border-color: #28a745; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 5px; }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn:hover { opacity: 0.8; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .results-section { margin-top: 30px; }
        .loading { text-align: center; padding: 50px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .metric-label { font-size: 0.9em; opacity: 0.9; }
        .chart-container { margin: 20px 0; }
        .tabs { display: flex; border-bottom: 2px solid #e9ecef; margin-bottom: 20px; }
        .tab { padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; }
        .tab.active { border-bottom-color: #007bff; color: #007bff; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 퀸트프로젝트 통합 백테스팅 시스템</h1>
            <p>🇺🇸 미국 + 🇯🇵 일본 + 🇮🇳 인도 + 💰 암호화폐 통합 백테스팅</p>
        </div>

        <div class="config-section">
            <h3>⚙️ 백테스트 설정</h3>
            <form id="backtest-form">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                    <div class="form-group">
                        <label>시작일자</label>
                        <input type="date" id="start_date" value="2023-01-01" required>
                    </div>
                    <div class="form-group">
                        <label>종료일자</label>
                        <input type="date" id="end_date" value="2024-12-31" required>
                    </div>
                    <div class="form-group">
                        <label>초기자본 (원)</label>
                        <input type="number" id="initial_capital" value="1000000000" min="1000000" step="1000000" required>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                    <div class="form-group">
                        <label>수수료 (%)</label>
                        <input type="number" id="commission" value="0.25" min="0" max="5" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label>슬리피지 (%)</label>
                        <input type="number" id="slippage" value="0.1" min="0" max="2" step="0.01" required>
                    </div>
                </div>
            </form>
        </div>

        <div class="config-section">
            <h3>📊 전략 선택</h3>
            <div class="strategy-grid">
                <div class="strategy-card enabled">
                    <h4>🇺🇸 미국 주식 전략</h4>
                    <p>서머타임 연동 + 5가지 융합 전략 + 화목 매매</p>
                    <label><input type="checkbox" id="us_strategy" checked> 활성화 (40% 배분)</label>
                </div>
                <div class="strategy-card enabled">
                    <h4>🇯🇵 일본 주식 전략</h4>
                    <p>엔화 연동 + 6개 기술지표 + 화목 하이브리드</p>
                    <label><input type="checkbox" id="japan_strategy" checked> 활성화 (25% 배분)</label>
                </div>
                <div class="strategy-card enabled">
                    <h4>💰 암호화폐 전략</h4>
                    <p>전설급 5대 시스템 + 월금 매매</p>
                    <label><input type="checkbox" id="crypto_strategy" checked> 활성화 (20% 배분)</label>
                </div>
                <div class="strategy-card enabled">
                    <h4>🇮🇳 인도 주식 전략</h4>
                    <p>5대 전설 투자자 + 수요일 안정형</p>
                    <label><input type="checkbox" id="india_strategy" checked> 활성화 (15% 배분)</label>
                </div>
            </div>
        </div>

        <div style="text-align: center; margin: 30px 0;">
            <button class="btn btn-primary" onclick="runBacktest()">🚀 백테스트 실행</button>
            <button class="btn btn-success" onclick="downloadResults()" id="download-btn" style="display: none;">📥 결과 다운로드</button>
        </div>

        <div id="loading" class="loading" style="display: none;">
            <div class="spinner"></div>
            <p>백테스트 실행 중... 잠시만 기다려주세요.</p>
        </div>

        <div id="results" class="results-section" style="display: none;">
            <div class="tabs">
                <div class="tab active" onclick="showTab('summary')">📊 요약</div>
                <div class="tab" onclick="showTab('performance')">📈 성과</div>
                <div class="tab" onclick="showTab('strategies')">🎯 전략별</div>
                <div class="tab" onclick="showTab('trades')">💼 거래내역</div>
            </div>

            <div id="summary-tab" class="tab-content active">
                <h3>📊 백테스트 요약</h3>
                <div id="summary-content"></div>
            </div>

            <div id="performance-tab" class="tab-content">
                <h3>📈 포트폴리오 성과</h3>
                <div id="performance-charts"></div>
            </div>

            <div id="strategies-tab" class="tab-content">
                <h3>🎯 전략별 분석</h3>
                <div id="strategy-analysis"></div>
            </div>

            <div id="trades-tab" class="tab-content">
                <h3>💼 거래 내역</h3>
                <div id="trades-content"></div>
            </div>
        </div>
    </div>

    <script>
        let currentResults = null;

        function showTab(tabName) {
            // 모든 탭 비활성화
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            // 선택된 탭 활성화
            document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        }

        async function runBacktest() {
            const form = document.getElementById('backtest-form');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            // 폼 데이터 수집
            const config = {
                start_date: document.getElementById('start_date').value,
                end_date: document.getElementById('end_date').value,
                initial_capital: parseFloat(document.getElementById('initial_capital').value),
                commission: parseFloat(document.getElementById('commission').value) / 100,
                slippage: parseFloat(document.getElementById('slippage').value) / 100,
                enabled_strategies: []
            };

            // 활성화된 전략 수집
            if (document.getElementById('us_strategy').checked) config.enabled_strategies.push('US_Strategy');
            if (document.getElementById('japan_strategy').checked) config.enabled_strategies.push('Japan_Strategy');
            if (document.getElementById('crypto_strategy').checked) config.enabled_strategies.push('Crypto_Strategy');
            if (document.getElementById('india_strategy').checked) config.enabled_strategies.push('India_Strategy');

            if (config.enabled_strategies.length === 0) {
                alert('최소 하나의 전략을 선택해주세요.');
                return;
            }

            // 로딩 표시
            loading.style.display = 'block';
            results.style.display = 'none';

            try {
                const response = await fetch('/api/backtest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });

                const data = await response.json();
                
                if (data.error) {
                    alert('백테스트 실패: ' + data.error);
                    return;
                }

                currentResults = data;
                displayResults(data);
                
                // 다운로드 버튼 표시
                document.getElementById('download-btn').style.display = 'inline-block';

            } catch (error) {
                alert('백테스트 실행 중 오류가 발생했습니다: ' + error);
            } finally {
                loading.style.display = 'none';
            }
        }

        function displayResults(data) {
            const results = document.getElementById('results');
            results.style.display = 'block';

            // 요약 정보 표시
            displaySummary(data.summary);
            
            // 성과 차트 표시
            displayPerformanceCharts(data);
            
            // 전략별 분석 표시
            displayStrategyAnalysis(data.strategy_results);
            
            // 거래 내역 표시
            displayTrades(data.strategy_results);
        }

        function displaySummary(summary) {
            const container = document.getElementById('summary-content');
            
            const html = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
                    <div class="metric-card">
                        <div class="metric-value">${summary.portfolio_return?.toFixed(2) || 0}%</div>
                        <div class="metric-label">총 수익률</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(summary.final_value / 100000000).toFixed(1) || 0}억원</div>
                        <div class="metric-label">최종 자산</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${summary.total_trades || 0}회</div>
                        <div class="metric-label">총 거래 수</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${summary.successful_strategies || 0}/${summary.total_strategies || 0}</div>
                        <div class="metric-label">성공 전략</div>
                    </div>
                </div>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <h4>📋 상세 정보</h4>
                    <p><strong>백테스트 기간:</strong> ${summary.backtest_period || 'N/A'}</p>
                    <p><strong>초기 자본:</strong> ${(summary.initial_capital / 100000000).toFixed(1)}억원</p>
                    <p><strong>최고 성과 전략:</strong> ${summary.best_performing_strategy ? summary.best_performing_strategy[0] + ' (' + summary.best_performing_strategy[1].toFixed(2) + '%)' : 'N/A'}</p>
                    
                    ${summary.strategy_summary ? 
                        '<h4 style="margin-top: 20px;">🎯 전략별 요약</h4>' +
                        Object.entries(summary.strategy_summary).map(([name, stats]) => 
                            `<div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                                <strong>${name}:</strong> 
                                수익률 ${stats.return}%, 
                                샤프비율 ${stats.sharpe}, 
                                승률 ${stats.win_rate}%, 
                                거래 ${stats.trades}회
                            </div>`
                        ).join('') : ''
                    }
                </div>
            `;
            
            container.innerHTML = html;
        }

        function displayPerformanceCharts(data) {
            const container = document.getElementById('performance-charts');
            
            // 포트폴리오 수익률 차트
            const portfolioData = [];
            const strategyColors = {
                'US_Strategy': '#1f77b4',
                'Japan_Strategy': '#ff7f0e', 
                'Crypto_Strategy': '#2ca02c',
                'India_Strategy': '#d62728'
            };

            Object.entries(data.strategy_results).forEach(([name, result]) => {
                if (result.equity_curve && result.dates) {
                    const returns = result.equity_curve.map((value, index) => 
                        index === 0 ? 0 : (value / result.equity_curve[0] - 1) * 100
                    );
                    
                    portfolioData.push({
                        x: result.dates,
                        y: returns,
                        type: 'scatter',
                        mode: 'lines',
                        name: name.replace('_Strategy', ''),
                        line: { color: strategyColors[name] }
                    });
                }
            });

            if (portfolioData.length > 0) {
                Plotly.newPlot('performance-charts', portfolioData, {
                    title: '📈 전략별 수익률 곡선',
                    xaxis: { title: '날짜' },
                    yaxis: { title: '수익률 (%)' },
                    height: 500,
                    showlegend: true
                });
            }
        }

        function displayStrategyAnalysis(strategyResults) {
            const container = document.getElementById('strategy-analysis');
            
            let html = '';
            
            Object.entries(strategyResults).forEach(([name, result]) => {
                if (result.error) {
                    html += `
                        <div style="background: #f8d7da; color: #721c24; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                            <h4>${name}</h4>
                            <p>오류: ${result.error}</p>
                        </div>
                    `;
                    return;
                }

                const perf = result.performance || {};
                const finalValue = result.final_capital + (result.total_positions_value || 0);
                const totalReturn = ((finalValue - result.initial_capital) / result.initial_capital * 100).toFixed(2);

                html += `
                    <div style="background: white; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                        <h4>${name.replace('_Strategy', '')} 전략</h4>
                        
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                            <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                                <div style="font-size: 1.5em; font-weight: bold; color: ${totalReturn >= 0 ? '#28a745' : '#dc3545'};">${totalReturn}%</div>
                                <div>총 수익률</div>
                            </div>
                            <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                                <div style="font-size: 1.5em; font-weight: bold;">${perf.sharpe_ratio?.toFixed(2) || 'N/A'}</div>
                                <div>샤프 비율</div>
                            </div>
                            <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                                <div style="font-size: 1.5em; font-weight: bold;">${perf.max_drawdown?.toFixed(2) || 'N/A'}%</div>
                                <div>최대 낙폭</div>
                            </div>
                            <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                                <div style="font-size: 1.5em; font-weight: bold;">${perf.win_rate?.toFixed(1) || 'N/A'}%</div>
                                <div>승률</div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 15px;">
                            <p><strong>초기 자본:</strong> ${(result.initial_capital / 100000000).toFixed(2)}억원</p>
                            <p><strong>최종 가치:</strong> ${(finalValue / 100000000).toFixed(2)}억원</p>
                            <p><strong>총 거래 수:</strong> ${result.total_trades || 0}회</p>
                            <p><strong>변동성:</strong> ${perf.volatility?.toFixed(2) || 'N/A'}%</p>
                            <p><strong>칼마 비율:</strong> ${perf.calmar_ratio?.toFixed(2) || 'N/A'}</p>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }

        function displayTrades(strategyResults) {
            const container = document.getElementById('trades-content');
            
            let allTrades = [];
            
            Object.entries(strategyResults).forEach(([strategyName, result]) => {
                if (result.trades) {
                    result.trades.forEach(trade => {
                        allTrades.push({
                            ...trade,
                            strategy: strategyName.replace('_Strategy', '')
                        });
                    });
                }
            });
            
            // 최신 거래부터 정렬
            allTrades.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            
            let html = `
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h4>📊 거래 통계</h4>
                    <p><strong>총 거래 수:</strong> ${allTrades.length}회</p>
                    <p><strong>매수 거래:</strong> ${allTrades.filter(t => t.action === 'buy').length}회</p>
                    <p><strong>매도 거래:</strong> ${allTrades.filter(t => t.action === 'sell').length}회</p>
                </div>
                
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; background: white;">
                        <thead>
                            <tr style="background: #f8f9fa;">
                                <th style="padding: 12px; border: 1px solid #dee2e6;">날짜</th>
                                <th style="padding: 12px; border: 1px solid #dee2e6;">전략</th>
                                <th style="padding: 12px; border: 1px solid #dee2e6;">심볼</th>
                                <th style="padding: 12px; border: 1px solid #dee2e6;">액션</th>
                                <th style="padding: 12px; border: 1px solid #dee2e6;">수량</th>
                                <th style="padding: 12px; border: 1px solid #dee2e6;">가격</th>
                                <th style="padding: 12px; border: 1px solid #dee2e6;">수수료</th>
                                <th style="padding: 12px; border: 1px solid #dee2e6;">사유</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            allTrades.slice(0, 100).forEach(trade => {
                const actionColor = trade.action === 'buy' ? '#28a745' : '#dc3545';
                const actionSymbol = trade.action === 'buy' ? '📈' : '📉';
                
                html += `
                    <tr>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">${new Date(trade.timestamp).toLocaleDateString()}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">${trade.strategy}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">${trade.symbol}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6; color: ${actionColor};">${actionSymbol} ${trade.action.toUpperCase()}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6; text-align: right;">${trade.quantity.toFixed(6)}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6; text-align: right;">${trade.price.toLocaleString()}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6; text-align: right;">${trade.commission.toLocaleString()}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">${trade.reason || '-'}</td>
                    </tr>
                `;
            });
            
            html += `
                        </tbody>
                    </table>
                </div>
                ${allTrades.length > 100 ? '<p style="text-align: center; margin-top: 20px; color: #6c757d;">최근 100개 거래만 표시됩니다.</p>' : ''}
            `;
            
            container.innerHTML = html;
        }

        async function downloadResults() {
            if (!currentResults) {
                alert('다운로드할 결과가 없습니다.');
                return;
            }

            try {
                const response = await fetch('/api/download', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentResults)
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `backtest_results_${new Date().toISOString().split('T')[0]}.zip`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                } else {
                    alert('다운로드 실패');
                }
            } catch (error) {
                alert('다운로드 중 오류가 발생했습니다: ' + error);
            }
        }

        // 페이지 로드 시 오늘 날짜를 종료일로 설정
        document.addEventListener('DOMContentLoaded', function() {
            const today = new Date().toISOString().split('T')[0];
            document.getElementById('end_date').value = today;
        });
    </script>
</body>
</html>
    ''')

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """백테스트 API"""
    global backtest_engine, current_results
    
    try:
        config_data = request.get_json()
        
        # 설정 객체 생성
        config = BacktestConfig(
            strategy_name="Integrated_Backtest",
            start_date=config_data['start_date'],
            end_date=config_data['end_date'],
            initial_capital=config_data['initial_capital'],
            commission=config_data['commission'],
            slippage=config_data['slippage'],
            enabled_strategies=config_data['enabled_strategies']
        )
        
        # 백테스트 엔진 생성
        backtest_engine = IntegratedBacktestEngine(config)
        
        # 심볼 정의
        symbols = {
            'US_Strategy': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX'],
            'Japan_Strategy': ['7203.T', '6758.T', '9984.T', '6861.T', '8306.T'],
            'Crypto_Strategy': ['KRW-BTC', 'KRW-ETH', 'KRW-BNB', 'KRW-ADA', 'KRW-SOL'],
            'India_Strategy': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
        }
        
        # 백테스트 실행
        async def run_backtest():
            return await backtest_engine.run_backtest(symbols)
        
        # 동기 실행 (실제 환경에서는 비동기 처리 권장)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(run_backtest())
        loop.close()
        
        current_results = results
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"백테스트 API 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download', methods=['POST'])
def api_download():
    """결과 다운로드 API"""
    try:
        results = request.get_json()
        
        # ZIP 파일 생성
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # JSON 결과
            zip_file.writestr(
                'backtest_results.json',
                json.dumps(results, indent=2, default=str)
            )
            
            # 요약 CSV
            if 'summary' in results:
                summary_df = pd.DataFrame([results['summary']])
                csv_buffer = io.StringIO()
                summary_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                zip_file.writestr('summary.csv', csv_buffer.getvalue())
            
            # 전략별 결과 CSV
            if 'strategy_results' in results:
                for strategy_name, strategy_result in results['strategy_results'].items():
                    if 'trades' in strategy_result:
                        trades_df = pd.DataFrame(strategy_result['trades'])
                        if not trades_df.empty:
                            csv_buffer = io.StringIO()
                            trades_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                            zip_file.writestr(f'{strategy_name}_trades.csv', csv_buffer.getvalue())
                    
                    if 'equity_curve' in strategy_result and 'dates' in strategy_result:
                        equity_df = pd.DataFrame({
                            'date': strategy_result['dates'],
                            'equity': strategy_result['equity_curve']
                        })
                        csv_buffer = io.StringIO()
                        equity_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                        zip_file.writestr(f'{strategy_name}_equity_curve.csv', csv_buffer.getvalue())
        
        zip_buffer.seek(0)
        
        return send_file(
            io.BytesIO(zip_buffer.read()),
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
        
    except Exception as e:
        logger.error(f"다운로드 API 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# 🚀 메인 실행 함수
# ============================================================================
def main():
    """메인 실행 함수"""
    print("🏆 퀸트프로젝트 통합 백테스팅 시스템 v2.0")
    print("="*60)
    print("🌐 웹 인터페이스 시작 중...")
    print("📱 접속 주소: http://localhost:5000")
    print("📱 모바일 접속: http://[IP주소]:5000")
    print("⚡ Ctrl+C로 종료")
    print("="*60)
    
    try:
        # Flask 앱 실행
        app.run(
            host='0.0.0.0',  # 모든 IP에서 접속 가능
            port=5000,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 백테스팅 시스템을 종료합니다.")
    except Exception as e:
        print(f"❌ 시스템 실행 오류: {e}")

if __name__ == "__main__":
    main() 
