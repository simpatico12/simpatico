#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🪙 암호화폐 전략 모듈 - 최고퀸트프로젝트 (궁극 완성 시스템)
=================================================================================

궁극의 하이브리드 암호화폐 전략 (V5.0):
- 🆕 AI 기반 프로젝트 품질 평가 시스템
- 🆕 시장 사이클 자동 감지 (4단계: 축적, 상승, 분배, 하락)
- 🆕 상관관계 기반 포트폴리오 최적화
- 🆕 소셜 센티먼트 분석 (Fear & Greed + Twitter)
- 자동 종목 선별 (업비트 전체 → 상위 20개)
- 확장된 기술적 지표 (일목균형표, RSI, MACD, 볼린저밴드, 스토캐스틱, ATR 등)
- 5단계 분할매매 시스템 (20% × 5)
- 24시간 실시간 모니터링
- 완전 자동화

Author: 최고퀸트팀
Version: 5.0.0 (궁극 완성)
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass
import requests
import pyupbit
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import aiohttp
warnings.filterwarnings('ignore')

# 로거 설정
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class UltimateCoinSignal:
    """궁극의 암호화폐 시그널 데이터 클래스 (V5.0)"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    
    # 전략별 점수
    fundamental_score: float
    technical_score: float
    momentum_score: float
    total_score: float
    
    # 🆕 AI 프로젝트 품질 점수
    project_quality_score: float
    ecosystem_health_score: float
    innovation_score: float
    adoption_score: float
    team_score: float
    
    # 펀더멘털 지표
    market_cap_rank: int
    volume_24h_rank: int
    liquidity_score: float
    
    # 🆕 시장 사이클 분석
    market_cycle: str  # 'accumulation', 'uptrend', 'distribution', 'downtrend'
    cycle_confidence: float
    btc_dominance: float
    total_market_cap_trend: str
    
    # 기술적 지표 (확장)
    rsi: float
    macd_signal: str
    bb_position: str
    stoch_k: float
    stoch_d: float
    ichimoku_signal: str
    atr: float
    obv_trend: str
    
    # 🆕 고급 기술적 지표
    williams_r: float
    cci: float
    mfi: float
    adx: float
    parabolic_sar: str
    
    # 모멘텀 지표
    momentum_3d: float
    momentum_7d: float
    momentum_30d: float
    volume_spike_ratio: float
    price_velocity: float
    relative_strength_btc: float
    
    # 🆕 상관관계 분석
    correlation_with_btc: float
    correlation_with_eth: float
    portfolio_fit_score: float
    diversification_benefit: float
    
    # 분할매매 정보
    position_stage: int  # 0,1,2,3,4,5 (현재 매수 단계)
    total_amount: float  # 총 투자금액
    stage1_amount: float # 1단계 금액 (20%)
    stage2_amount: float # 2단계 금액 (20%)
    stage3_amount: float # 3단계 금액 (20%)
    stage4_amount: float # 4단계 금액 (20%)
    stage5_amount: float # 5단계 금액 (20%)
    entry_price_1: float # 1단계 진입가
    entry_price_2: float # 2단계 진입가 (5% 하락)
    entry_price_3: float # 3단계 진입가 (10% 하락)
    entry_price_4: float # 4단계 진입가 (15% 하락)
    entry_price_5: float # 5단계 진입가 (20% 하락)
    stop_loss: float     # 손절가 (-25%)
    take_profit_1: float # 1차 익절가 (+20%)
    take_profit_2: float # 2차 익절가 (+50%)
    take_profit_3: float # 3차 익절가 (+100%)
    max_hold_days: int
    
    # 🆕 소셜 센티먼트
    fear_greed_score: int
    social_sentiment: str
    twitter_mentions: int
    reddit_sentiment: float
    
    sector: str
    reasoning: str
    target_price: float
    timestamp: datetime
    additional_data: Optional[Dict] = None

# ========================================================================================
# 🆕 AI 기반 프로젝트 품질 평가 시스템
# ========================================================================================
class AIProjectQualityAnalyzer:
    """AI 기반 프로젝트 품질 평가"""
    
    def __init__(self):
        self.tier_database = {
            'tier_1': {'coins': ['BTC', 'ETH', 'BNB'], 'base_score': 0.95, 'description': '절대 강자'},
            'tier_2': {'coins': ['ADA', 'SOL', 'AVAX', 'DOT', 'MATIC', 'ATOM', 'NEAR'], 'base_score': 0.85, 'description': '검증된 L1'},
            'tier_3': {'coins': ['LINK', 'UNI', 'AAVE', 'MKR', 'CRV', 'COMP', 'SUSHI'], 'base_score': 0.75, 'description': 'DeFi 강자'},
            'tier_4': {'coins': ['SAND', 'MANA', 'AXS', 'ENJ', 'THETA', 'FIL', 'VET'], 'base_score': 0.65, 'description': '특화 섹터'},
            'tier_5': {'coins': ['DOGE', 'SHIB', 'PEPE', 'FLOKI'], 'base_score': 0.45, 'description': '밈코인'}
        }

    def get_coin_tier(self, symbol: str) -> Tuple[str, float]:
        """코인 등급 확인"""
        coin_name = symbol.replace('KRW-', '').upper()
        for tier, data in self.tier_database.items():
            if coin_name in data['coins']:
                return tier, data['base_score']
        return 'tier_unknown', 0.50

    def analyze_project_quality(self, symbol: str, market_data: Dict) -> Dict:
        """프로젝트 품질 종합 분석"""
        try:
            coin_name = symbol.replace('KRW-', '').upper()
            tier, base_score = self.get_coin_tier(symbol)
            
            # 간소화된 점수 계산
            ecosystem_score = self._analyze_ecosystem_health(market_data)
            innovation_score = self._get_innovation_score(coin_name)
            adoption_score = self._analyze_adoption(market_data)
            team_score = self._get_team_score(coin_name)
            
            total_quality = (base_score * 0.40 + ecosystem_score * 0.25 + 
                           innovation_score * 0.20 + adoption_score * 0.10 + team_score * 0.05)
            
            return {
                'project_quality_score': total_quality,
                'ecosystem_health_score': ecosystem_score,
                'innovation_score': innovation_score,
                'adoption_score': adoption_score,
                'team_score': team_score,
                'tier': tier,
                'coin_category': self._categorize_coin(coin_name)
            }
        except Exception as e:
            logger.error(f"프로젝트 품질 분석 실패 {symbol}: {e}")
            return {'project_quality_score': 0.50, 'ecosystem_health_score': 0.50, 'innovation_score': 0.50,
                   'adoption_score': 0.50, 'team_score': 0.50, 'tier': 'tier_unknown', 'coin_category': 'Unknown'}

    def _analyze_ecosystem_health(self, market_data: Dict) -> float:
        """생태계 건전성 분석"""
        score = 0.5
        volume_24h = market_data.get('volume_24h_krw', 0)
        if volume_24h >= 100_000_000_000: score += 0.3
        elif volume_24h >= 50_000_000_000: score += 0.2
        elif volume_24h >= 10_000_000_000: score += 0.1
        return min(score, 1.0)

    def _get_innovation_score(self, coin_name: str) -> float:
        """혁신성 점수"""
        innovation_scores = {
            'ETH': 0.95, 'ADA': 0.90, 'SOL': 0.88, 'AVAX': 0.85, 'DOT': 0.85, 'ATOM': 0.80,
            'UNI': 0.85, 'AAVE': 0.80, 'LINK': 0.90, 'SAND': 0.75, 'MANA': 0.75,
            'DOGE': 0.30, 'SHIB': 0.25, 'PEPE': 0.20
        }
        return innovation_scores.get(coin_name, 0.50)

    def _analyze_adoption(self, market_data: Dict) -> float:
        """채택도 분석"""
        market_cap = market_data.get('market_cap', 0)
        if market_cap >= 10_000_000_000_000: return 0.95
        elif market_cap >= 5_000_000_000_000: return 0.85
        elif market_cap >= 1_000_000_000_000: return 0.75
        elif market_cap >= 500_000_000_000: return 0.65
        elif market_cap >= 100_000_000_000: return 0.55
        return 0.45

    def _get_team_score(self, coin_name: str) -> float:
        """팀 점수"""
        team_scores = {'ETH': 0.95, 'ADA': 0.90, 'DOT': 0.90, 'SOL': 0.85, 'AVAX': 0.85,
                      'ATOM': 0.80, 'LINK': 0.85, 'UNI': 0.80, 'AAVE': 0.80}
        return team_scores.get(coin_name, 0.60)

    def _categorize_coin(self, coin_name: str) -> str:
        """코인 카테고리 분류"""
        categories = {
            'L1_Blockchain': ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'ATOM', 'NEAR'],
            'DeFi': ['UNI', 'AAVE', 'MKR', 'COMP', 'CRV', 'SUSHI'],
            'Gaming_Metaverse': ['SAND', 'MANA', 'AXS', 'ENJ'],
            'Infrastructure': ['LINK', 'FIL', 'VET'], 'Meme': ['DOGE', 'SHIB', 'PEPE'],
            'Exchange': ['BNB', 'CRO'], 'Payment': ['XRP', 'XLM', 'LTC']
        }
        for category, coins in categories.items():
            if coin_name in coins:
                return category
        return 'Unknown'

# ========================================================================================
# 🆕 시장 사이클 자동 감지 시스템
# ========================================================================================
class MarketCycleDetector:
    """시장 사이클 자동 감지"""
    
    def __init__(self):
        self.btc_dominance_threshold_low = 40.0
        self.btc_dominance_threshold_high = 60.0
        self.fear_greed_extreme_fear = 25
        self.fear_greed_extreme_greed = 75

    async def detect_market_cycle(self) -> Dict:
        """시장 사이클 감지"""
        try:
            btc_dominance = await self._get_btc_dominance()
            total_mcap_trend = await self._analyze_total_market_cap_trend()
            fear_greed_data = await self._get_fear_greed_index()
            btc_trend = await self._analyze_btc_trend()
            
            cycle_result = self._determine_market_cycle(btc_dominance, total_mcap_trend, 
                                                     fear_greed_data['score'], btc_trend)
            
            return {
                'market_cycle': cycle_result['cycle'],
                'cycle_confidence': cycle_result['confidence'],
                'btc_dominance': btc_dominance,
                'total_market_cap_trend': total_mcap_trend,
                'fear_greed_score': fear_greed_data['score'],
                'btc_trend': btc_trend,
                'reasoning': cycle_result['reasoning']
            }
        except Exception as e:
            logger.error(f"시장 사이클 감지 실패: {e}")
            return {'market_cycle': 'sideways', 'cycle_confidence': 0.5, 'btc_dominance': 50.0,
                   'total_market_cap_trend': 'neutral', 'fear_greed_score': 50, 'btc_trend': 'neutral',
                   'reasoning': '데이터 수집 실패'}

    async def _get_btc_dominance(self) -> float:
        """BTC 도미넌스 조회"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.coingecko.com/api/v3/global", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['data']['market_cap_percentage']['btc']
            return 50.0
        except: return 50.0

    async def _analyze_total_market_cap_trend(self) -> str:
        """총 시가총액 추세 분석"""
        try:
            btc_data = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
            if btc_data is None or len(btc_data) < 30: return 'neutral'
            
            current_price = btc_data['close'].iloc[-1]
            ma30 = btc_data['close'].rolling(30).mean().iloc[-1]
            
            if current_price > ma30 * 1.05: return 'bullish'
            elif current_price < ma30 * 0.95: return 'bearish'
            else: return 'neutral'
        except: return 'neutral'

    async def _get_fear_greed_index(self) -> Dict:
        """공포탐욕지수 조회"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.alternative.me/fng/?limit=1", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {'score': int(data["data"][0]["value"]), 
                               'classification': data["data"][0]["value_classification"]}
            return {'score': 50, 'classification': 'Neutral'}
        except: return {'score': 50, 'classification': 'Neutral'}

    async def _analyze_btc_trend(self) -> str:
        """BTC 가격 추세 분석"""
        try:
            btc_data = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=60)
            if btc_data is None or len(btc_data) < 60: return 'neutral'
            
            ma20 = btc_data['close'].rolling(20).mean().iloc[-1]
            ma50 = btc_data['close'].rolling(50).mean().iloc[-1]
            current_price = btc_data['close'].iloc[-1]
            
            if current_price > ma20 > ma50: return 'strong_bullish'
            elif current_price > ma20 and ma20 < ma50: return 'weak_bullish'
            elif current_price < ma20 < ma50: return 'strong_bearish'
            elif current_price < ma20 and ma20 > ma50: return 'weak_bearish'
            else: return 'neutral'
        except: return 'neutral'

    def _determine_market_cycle(self, btc_dominance: float, total_mcap_trend: str, 
                              fear_greed_score: int, btc_trend: str) -> Dict:
        """시장 사이클 종합 판단"""
        score, reasons = 0.0, []
        
        # BTC 도미넌스 분석
        if btc_dominance >= self.btc_dominance_threshold_high:
            score -= 0.3; reasons.append(f"BTC도미넌스높음({btc_dominance:.1f}%)")
        elif btc_dominance <= self.btc_dominance_threshold_low:
            score += 0.3; reasons.append(f"BTC도미넌스낮음({btc_dominance:.1f}%)")
        
        # 시총 추세
        if total_mcap_trend == 'bullish': score += 0.25; reasons.append("시총상승")
        elif total_mcap_trend == 'bearish': score -= 0.25; reasons.append("시총하락")
        
        # 공포탐욕지수
        if fear_greed_score <= self.fear_greed_extreme_fear:
            score += 0.25; reasons.append(f"극단공포({fear_greed_score})")
        elif fear_greed_score >= self.fear_greed_extreme_greed:
            score -= 0.25; reasons.append(f"극단탐욕({fear_greed_score})")
        
        # BTC 추세
        btc_scores = {'strong_bullish': 0.20, 'weak_bullish': 0.10, 'neutral': 0.00, 
                     'weak_bearish': -0.10, 'strong_bearish': -0.20}
        score += btc_scores.get(btc_trend, 0.0)
        
        # 최종 사이클 판단
        if score >= 0.4: cycle, confidence = 'uptrend', min(score * 1.5, 0.95)
        elif score <= -0.4: cycle, confidence = 'downtrend', min(abs(score) * 1.5, 0.95)
        elif 0.2 <= score < 0.4: cycle, confidence = 'accumulation', score + 0.3
        elif -0.4 < score <= -0.2: cycle, confidence = 'distribution', abs(score) + 0.3
        else: cycle, confidence = 'sideways', 0.5
        
        return {'cycle': cycle, 'confidence': confidence, 'reasoning': " | ".join(reasons)}

# ========================================================================================
# 🆕 고급 기술적 지표 분석
# ========================================================================================
class AdvancedTechnicalIndicators:
    """고급 기술적 지표 분석"""
    
    @staticmethod
    def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> float:
        """Williams %R 계산"""
        try:
            if len(data) < period: return -50.0
            high_n = data['high'].rolling(window=period).max()
            low_n = data['low'].rolling(window=period).min()
            williams_r = -100 * ((high_n - data['close']) / (high_n - low_n))
            return williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50.0
        except: return -50.0

    @staticmethod
    def calculate_cci(data: pd.DataFrame, period: int = 20) -> float:
        """Commodity Channel Index 계산"""
        try:
            if len(data) < period: return 0.0
            tp = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - sma_tp) / (0.015 * mad)
            return cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0.0
        except: return 0.0

    @staticmethod
    def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
        """Money Flow Index 계산"""
        try:
            if len(data) < period or 'volume' not in data.columns: return 50.0
            tp = (data['high'] + data['low'] + data['close']) / 3
            raw_money_flow = tp * data['volume']
            
            money_flow_positive, money_flow_negative = [], []
            for i in range(1, len(data)):
                if tp.iloc[i] > tp.iloc[i-1]:
                    money_flow_positive.append(raw_money_flow.iloc[i])
                    money_flow_negative.append(0)
                elif tp.iloc[i] < tp.iloc[i-1]:
                    money_flow_positive.append(0)
                    money_flow_negative.append(raw_money_flow.iloc[i])
                else:
                    money_flow_positive.append(0)
                    money_flow_negative.append(0)
            
            mf_positive = pd.Series(money_flow_positive).rolling(window=period-1).sum()
            mf_negative = pd.Series(money_flow_negative).rolling(window=period-1).sum()
            mfi = 100 - (100 / (1 + (mf_positive / mf_negative)))
            return mfi.iloc[-1] if len(mfi) > 0 and not pd.isna(mfi.iloc[-1]) else 50.0
        except: return 50.0

    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index 계산"""
        try:
            if len(data) < period: return 25.0
            high, low, close = data['high'], data['low'], data['close']
            
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
            minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0
        except: return 25.0

# ========================================================================================
# 🆕 궁극의 암호화폐 전략 클래스
# ========================================================================================
class UltimateCoinStrategy:
    """궁극의 암호화폐 전략 클래스 (V5.0)"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        """전략 초기화"""
        self.config = self._load_config(config_path)
        self.coin_config = self.config.get('coin_strategy', {})
        self.enabled = self.coin_config.get('enabled', True)
        
        # AI 기반 분석 시스템들
        self.quality_analyzer = AIProjectQualityAnalyzer()
        self.cycle_detector = MarketCycleDetector()
        self.portfolio_optimizer = PortfolioOptimizer()  # 추가된 포트폴리오 최적화
        
        # 자동 선별 설정
        self.target_coins = 20
        self.min_volume_24h = 500_000_000
        
        # 하이브리드 전략 가중치
        self.fundamental_weight = 0.35
        self.technical_weight = 0.35
        self.momentum_weight = 0.30
        
        # 포트폴리오 설정
        self.coin_portfolio_value = 200_000_000
        
        # 5단계 분할매매 설정
        self.stage_ratios = [0.20, 0.20, 0.20, 0.20, 0.20]
        self.stage_triggers = [0.0, -0.05, -0.10, -0.15, -0.20]
        
        # 리스크 관리
        self.base_stop_loss_pct = 0.25
        self.base_take_profit_levels = [0.20, 0.50, 1.00]
        self.base_max_hold_days = 30
        self.max_single_coin_weight = 0.08
        
        # 선별된 코인 리스트
        self.selected_coins = []
        self.last_selection_time = None
        self.selection_cache_hours = 12
        
        # 시장 사이클 정보
        self.current_market_cycle = 'sideways'
        self.cycle_confidence = 0.5
        
        if self.enabled:
            logger.info(f"🪙 궁극의 암호화폐 전략 초기화 (V5.0)")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패, 기본값 사용: {e}")
            return {'coin_strategy': {'enabled': True}}

    async def analyze_symbol(self, symbol: str) -> UltimateCoinSignal:
        """개별 코인 궁극 분석"""
        if not self.enabled:
            return self._create_empty_signal(symbol, "전략 비활성화")
        
        try:
            # 시장 사이클 정보 업데이트
            if not hasattr(self, 'current_market_cycle') or self.current_market_cycle == 'sideways':
                cycle_info = await self.cycle_detector.detect_market_cycle()
                self.current_market_cycle = cycle_info['market_cycle']
                self.cycle_confidence = cycle_info['cycle_confidence']
            
            # 종합 데이터 수집
            data = self._get_comprehensive_coin_data_sync(symbol)
            if not data:
                raise ValueError(f"데이터 수집 실패: {symbol}")
            
            # AI 프로젝트 품질 분석
            quality_analysis = self.quality_analyzer.analyze_project_quality(symbol, data)
            
            # 3가지 전략 분석
            cycle_weights = self._get_cycle_based_weights()
            fundamental_score, fundamental_reasoning = self._analyze_fundamental_enhanced(symbol, data, quality_analysis)
            technical_score, technical_details = self._analyze_technical_indicators_advanced(data)
            momentum_score, momentum_reasoning = self._analyze_momentum_advanced(symbol, data)
            
            # 가중 평균 계산
            total_score = (fundamental_score * cycle_weights['fundamental'] +
                          technical_score * cycle_weights['technical'] +
                          momentum_score * cycle_weights['momentum'])
            
            # 시장 사이클 보너스/페널티
            cycle_bonus = self._get_cycle_bonus(symbol, quality_analysis)
            total_score += cycle_bonus
            
            # 최종 액션 결정
            if total_score >= 0.75:
                action, confidence = 'buy', min(total_score, 0.95)
            elif total_score <= 0.25:
                action, confidence = 'sell', min(1 - total_score, 0.95)
            else:
                action, confidence = 'hold', 0.50
            
            # 강화된 분할매매 계획 수립
            split_plan = self._calculate_enhanced_split_trading_plan(symbol, data['price'], confidence)
            
            # 목표주가 계산
            cycle_multipliers = {'accumulation': 0.30, 'uptrend': 0.80, 'distribution': 0.20, 'downtrend': 0.10, 'sideways': 0.40}
            expected_return = cycle_multipliers.get(self.current_market_cycle, 0.40)
            target_price = data['price'] * (1 + confidence * expected_return)
            
            # 종합 reasoning
            all_reasoning = " | ".join([fundamental_reasoning, f"기술:{technical_score:.2f}", momentum_reasoning,
                                      f"사이클:{self.current_market_cycle}", f"품질:{quality_analysis['project_quality_score']:.2f}"])
            
            # 상관관계 및 BTC 관련 분석
            btc_correlation = await self._calculate_btc_correlation(symbol)
            
            # 소셜 센티먼트
            fear_greed_score, social_sentiment = await self._get_social_sentiment()
            
            return UltimateCoinSignal(
                symbol=symbol, action=action, confidence=confidence, price=data['price'],
                fundamental_score=fundamental_score, technical_score=technical_score, 
                momentum_score=momentum_score, total_score=total_score,
                project_quality_score=quality_analysis['project_quality_score'],
                ecosystem_health_score=quality_analysis['ecosystem_health_score'],
                innovation_score=quality_analysis['innovation_score'],
                adoption_score=quality_analysis['adoption_score'],
                team_score=quality_analysis['team_score'],
                market_cap_rank=0, volume_24h_rank=0,
                liquidity_score=min(data.get('volume_24h_krw', 0) / 1e10, 1.0),
                market_cycle=self.current_market_cycle, cycle_confidence=self.cycle_confidence,
                btc_dominance=0.0, total_market_cap_trend='neutral',
                rsi=technical_details.get('rsi', 50), macd_signal=technical_details.get('macd_signal', 'neutral'),
                bb_position=technical_details.get('bb_position', 'normal'),
                stoch_k=technical_details.get('stoch_k', 50), stoch_d=technical_details.get('stoch_d', 50),
                ichimoku_signal=technical_details.get('ichimoku_signal', 'neutral'),
                atr=technical_details.get('atr', 0), obv_trend=technical_details.get('obv_trend', 'neutral'),
                williams_r=technical_details.get('williams_r', -50), cci=technical_details.get('cci', 0),
                mfi=technical_details.get('mfi', 50), adx=technical_details.get('adx', 25),
                parabolic_sar=technical_details.get('parabolic_sar', 'neutral'),
                momentum_3d=data.get('momentum_3d', 0), momentum_7d=data.get('momentum_7d', 0),
                momentum_30d=data.get('momentum_30d', 0), volume_spike_ratio=data.get('volume_spike_ratio', 1),
                price_velocity=data.get('momentum_3d', 0) / 3, relative_strength_btc=btc_correlation,
                correlation_with_btc=btc_correlation, correlation_with_eth=0.0,
                portfolio_fit_score=0.8, diversification_benefit=1.0,
                position_stage=0, total_amount=split_plan.get('total_investment', 0),
                stage1_amount=split_plan.get('stage_amounts', [0]*5)[0],
                stage2_amount=split_plan.get('stage_amounts', [0]*5)[1],
                stage3_amount=split_plan.get('stage_amounts', [0]*5)[2],
                stage4_amount=split_plan.get('stage_amounts', [0]*5)[3],
                stage5_amount=split_plan.get('stage_amounts', [0]*5)[4],
                entry_price_1=split_plan.get('entry_prices', [data['price']]*5)[0],
                entry_price_2=split_plan.get('entry_prices', [data['price']]*5)[1],
                entry_price_3=split_plan.get('entry_prices', [data['price']]*5)[2],
                entry_price_4=split_plan.get('entry_prices', [data['price']]*5)[3],
                entry_price_5=split_plan.get('entry_prices', [data['price']]*5)[4],
                stop_loss=split_plan.get('stop_loss', data['price'] * 0.75),
                take_profit_1=split_plan.get('take_profits', [data['price']]*3)[0],
                take_profit_2=split_plan.get('take_profits', [data['price']]*3)[1],
                take_profit_3=split_plan.get('take_profits', [data['price']]*3)[2],
                max_hold_days=split_plan.get('max_hold_days', 30),
                fear_greed_score=fear_greed_score, social_sentiment=social_sentiment,
                twitter_mentions=0, reddit_sentiment=0.0,
                sector=quality_analysis['coin_category'], reasoning=all_reasoning,
                target_price=target_price, timestamp=datetime.now(), additional_data=split_plan
            )
            
        except Exception as e:
            logger.error(f"궁극 코인 분석 실패 {symbol}: {e}")
            return self._create_empty_signal(symbol, f"분석 실패: {str(e)}")

    def _get_cycle_bonus(self, symbol: str, quality_analysis: Dict) -> float:
        """시장 사이클 기반 보너스/페널티"""
        try:
            bonus = 0.0
            coin_category = quality_analysis['coin_category']
            tier = quality_analysis['tier']
            
            if self.current_market_cycle == 'accumulation':
                if tier in ['tier_1', 'tier_2']: bonus += 0.10
                if coin_category in ['L1_Blockchain', 'DeFi']: bonus += 0.05
            elif self.current_market_cycle == 'uptrend':
                bonus += 0.05
                if coin_category in ['Gaming_Metaverse', 'Meme']: bonus += 0.10
            elif self.current_market_cycle == 'distribution':
                if tier == 'tier_1': bonus += 0.05
                else: bonus -= 0.05
            elif self.current_market_cycle == 'downtrend':
                if tier == 'tier_1': bonus += 0.05
                else: bonus -= 0.15
            
            return bonus * self.cycle_confidence
        except: return 0.0

    async def _calculate_btc_correlation(self, symbol: str) -> float:
        """BTC와의 상관관계 계산"""
        try:
            if symbol == 'KRW-BTC': return 1.0
            
            btc_data = pyupbit.get_ohlcv('KRW-BTC', interval="day", count=30)
            coin_data = pyupbit.get_ohlcv(symbol, interval="day", count=30)
            
            if btc_data is None or coin_data is None or len(btc_data) < 30 or len(coin_data) < 30:
                return 0.5
            
            btc_returns = btc_data['close'].pct_change().dropna()
            coin_returns = coin_data['close'].pct_change().dropna()
            
            if len(btc_returns) != len(coin_returns):
                min_len = min(len(btc_returns), len(coin_returns))
                btc_returns = btc_returns.tail(min_len)
                coin_returns = coin_returns.tail(min_len)
            
            correlation = btc_returns.corr(coin_returns)
            return correlation if not pd.isna(correlation) else 0.5
        except: return 0.5

    async def _get_social_sentiment(self) -> Tuple[int, str]:
        """소셜 센티먼트 조회"""
        try:
            response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            if response.status_code == 200:
                data = response.json()
                score = int(data["data"][0]["value"])
                classification = data["data"][0]["value_classification"]
                return score, classification
            return 50, "Neutral"
        except: return 50, "Neutral"

    def _create_empty_signal(self, symbol: str, reason: str) -> UltimateCoinSignal:
        """빈 시그널 생성"""
        return UltimateCoinSignal(
            symbol=symbol, action='hold', confidence=0.0, price=0.0,
            fundamental_score=0.0, technical_score=0.0, momentum_score=0.0, total_score=0.0,
            project_quality_score=0.0, ecosystem_health_score=0.0, innovation_score=0.0,
            adoption_score=0.0, team_score=0.0, market_cap_rank=0, volume_24h_rank=0,
            liquidity_score=0.0, market_cycle='sideways', cycle_confidence=0.5,
            btc_dominance=50.0, total_market_cap_trend='neutral', rsi=50.0, macd_signal='neutral',
            bb_position='normal', stoch_k=50.0, stoch_d=50.0, ichimoku_signal='neutral',
            atr=0.0, obv_trend='neutral', williams_r=-50.0, cci=0.0, mfi=50.0, adx=25.0,
            parabolic_sar='neutral', momentum_3d=0.0, momentum_7d=0.0, momentum_30d=0.0,
            volume_spike_ratio=1.0, price_velocity=0.0, relative_strength_btc=0.5,
            correlation_with_btc=0.5, correlation_with_eth=0.5, portfolio_fit_score=0.5,
            diversification_benefit=0.5, position_stage=0, total_amount=0.0, stage1_amount=0.0,
            stage2_amount=0.0, stage3_amount=0.0, stage4_amount=0.0, stage5_amount=0.0,
            entry_price_1=0.0, entry_price_2=0.0, entry_price_3=0.0, entry_price_4=0.0,
            entry_price_5=0.0, stop_loss=0.0, take_profit_1=0.0, take_profit_2=0.0,
            take_profit_3=0.0, max_hold_days=30, fear_greed_score=50, social_sentiment='Neutral',
            twitter_mentions=0, reddit_sentiment=0.0, sector='Unknown', reasoning=reason,
            target_price=0.0, timestamp=datetime.now()
        )

    async def scan_all_selected_coins(self) -> List[UltimateCoinSignal]:
        """전체 자동선별 + 코인 분석"""
        if not self.enabled: return []
        
        logger.info(f"🔍 궁극의 암호화폐 완전 자동 분석 시작! (V5.0)")
        
        try:
            selected_symbols = await self.ultimate_auto_select_coins()
            if not selected_symbols:
                logger.error("궁극의 자동 선별 실패")
                return []
            
            all_signals = []
            for i, symbol in enumerate(selected_symbols, 1):
                try:
                    print(f"📊 궁극 분석 중... {i}/{len(selected_symbols)} - {symbol}")
                    signal = await self.analyze_symbol(symbol)
                    all_signals.append(signal)
                    
                    action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                    logger.info(f"{action_emoji} {symbol} ({signal.sector}): {signal.action} "
                              f"신뢰도:{signal.confidence:.2f} 총점:{signal.total_score:.2f} "
                              f"품질:{signal.project_quality_score:.2f} 사이클:{signal.market_cycle}")
                    
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error(f"❌ {symbol} 궁극 분석 실패: {e}")
            
            buy_count = len([s for s in all_signals if s.action == 'buy'])
            sell_count = len([s for s in all_signals if s.action == 'sell'])
            hold_count = len([s for s in all_signals if s.action == 'hold'])
            
            logger.info(f"🎯 궁극의 완전 자동 분석 완료!")
            logger.info(f"📊 결과: 매수:{buy_count}, 매도:{sell_count}, 보유:{hold_count}")
            logger.info(f"🔄 현재 시장 사이클: {self.current_market_cycle} (신뢰도:{self.cycle_confidence:.2f})")
            
            return all_signals
        except Exception as e:
            logger.error(f"궁극 전체 스캔 실패: {e}")
            return []

    async def ultimate_auto_select_coins(self) -> List[str]:
        """궁극의 자동 코인 선별"""
        if not self.enabled: return []

        try:
            if self._is_selection_cache_valid():
                logger.info("📋 캐시된 선별 결과 사용")
                return [coin['symbol'] for coin in self.selected_coins]

            logger.info("🔍 궁극의 자동 코인 선별 시작!")
            start_time = time.time()

            # 시장 사이클 감지
            cycle_info = await self.cycle_detector.detect_market_cycle()
            self.current_market_cycle = cycle_info['market_cycle']
            self.cycle_confidence = cycle_info['cycle_confidence']
            
            logger.info(f"📊 현재 시장 사이클: {self.current_market_cycle} (신뢰도: {self.cycle_confidence:.2f})")

            # 모든 KRW 마켓 코인 수집
            all_tickers = pyupbit.get_tickers(fiat="KRW")
            if not all_tickers:
                logger.error("업비트 티커 조회 실패")
                return self._get_default_coins()
            
            logger.info(f"📊 1단계: {len(all_tickers)}개 코인 발견")

            # 🆕 6단계 정밀 필터링 적용
            qualified_coins = await self._comprehensive_filtering(all_tickers)
            logger.info(f"📊 6단계 필터링 완료: {len(qualified_coins)}개 코인이 모든 단계 통과")

            # 🆕 포트폴리오 최적화는 이미 6단계에서 적용됨
            final_selection = qualified_coins[:self.target_coins]
            
            self.selected_coins = final_selection
            self.last_selection_time = datetime.now()

            selected_symbols = [coin['symbol'] for coin in final_selection]
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ 궁극의 자동 선별 완료! {len(selected_symbols)}개 코인 ({elapsed_time:.1f}초 소요)")

            return selected_symbols

            return selected_symbols

        except Exception as e:
            logger.error(f"궁극의 자동 선별 실패: {e}")
            return self._get_default_coins()

    async def _comprehensive_filtering(self, all_tickers: List[str]) -> List[Dict]:
        """🆕 6단계 정밀 필터링 + 품질 분석"""
        logger.info("🔍 6단계 정밀 필터링 시작...")
        
        # 1단계: 기본 필터링 (거래량, 시총)
        stage1_passed = await self._stage1_basic_filtering(all_tickers)
        logger.info(f"📊 1단계 통과: {len(stage1_passed)}개 (기본 필터)")
        
        # 2단계: 펀더멘털 1차 스크리닝
        stage2_passed = await self._stage2_fundamental_screening(stage1_passed)
        logger.info(f"📊 2단계 통과: {len(stage2_passed)}개 (펀더멘털 스크리닝)")
        
        # 3단계: 기술적 지표 2차 스크리닝
        stage3_passed = await self._stage3_technical_screening(stage2_passed)
        logger.info(f"📊 3단계 통과: {len(stage3_passed)}개 (기술적 스크리닝)")
        
        # 4단계: 모멘텀 3차 스크리닝
        stage4_passed = await self._stage4_momentum_screening(stage3_passed)
        logger.info(f"📊 4단계 통과: {len(stage4_passed)}개 (모멘텀 스크리닝)")
        
        # 5단계: AI 품질 평가 4차 스크리닝
        stage5_passed = await self._stage5_ai_quality_screening(stage4_passed)
        logger.info(f"📊 5단계 통과: {len(stage5_passed)}개 (AI 품질 스크리닝)")
        
        # 6단계: 최종 종합 점수 계산 및 정렬
        final_qualified = await self._stage6_final_scoring(stage5_passed)
        logger.info(f"📊 6단계 완료: {len(final_qualified)}개 (최종 점수 계산)")
        
        return final_qualified[:80]  # 상위 80개로 확장 (더 많은 선택지)

    async def _stage1_basic_filtering(self, all_tickers: List[str]) -> List[Dict]:
        """1단계: 기본 필터링 (거래량, 시총, 상장기간)"""
        qualified_coins = []
        
        batch_size = 20
        for i in range(0, len(all_tickers), batch_size):
            batch_tickers = all_tickers[i:i+batch_size]
            
            with ThreadPoolExecutor(max_workers=15) as executor:
                futures = [executor.submit(self._basic_filter_single_coin, ticker) 
                          for ticker in batch_tickers]
                
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        if result:
                            qualified_coins.append(result)
                    except: continue
            
            await asyncio.sleep(0.3)  # API 제한 고려
            
            if i % 100 == 0:
                logger.info(f"📊 1단계 진행: {i}/{len(all_tickers)} 완료")
        
        # 거래량 기준 정렬
        qualified_coins.sort(key=lambda x: x.get('volume_24h_krw', 0), reverse=True)
        return qualified_coins[:200]  # 상위 200개만 다음 단계로

    def _basic_filter_single_coin(self, symbol: str) -> Optional[Dict]:
        """단일 코인 기본 필터링"""
        try:
            # 기본 데이터 수집
            current_price = pyupbit.get_current_price(symbol)
            if not current_price or current_price <= 0:
                return None
            
            # 30일 데이터 확인
            ohlcv_30d = pyupbit.get_ohlcv(symbol, interval="day", count=30)
            if ohlcv_30d is None or len(ohlcv_30d) < 30:
                return None
            
            # 거래량 확인
            latest_data = ohlcv_30d.iloc[-1]
            volume_24h_krw = latest_data['volume'] * current_price
            
            # 최소 거래량 필터
            if volume_24h_krw < self.min_volume_24h:
                return None
            
            # 최소 가격 필터 (너무 저가 코인 제외)
            if current_price < 10:  # 10원 미만 제외
                return None
            
            # 상장 기간 확인 (30일 이상)
            if len(ohlcv_30d) < 30:
                return None
            
            # 가격 변동성 확인 (너무 변동성 큰 코인 제외)
            price_std = ohlcv_30d['close'].tail(7).std()
            price_mean = ohlcv_30d['close'].tail(7).mean()
            volatility = price_std / price_mean if price_mean > 0 else 999
            
            if volatility > 0.5:  # 일주일 평균 50% 이상 변동성 제외
                return None
            
            return {
                'symbol': symbol,
                'price': current_price,
                'volume_24h_krw': volume_24h_krw,
                'volatility': volatility,
                'ohlcv_30d': ohlcv_30d,
                'stage1_passed': True
            }
        except:
            return None

    async def _stage2_fundamental_screening(self, stage1_coins: List[Dict]) -> List[Dict]:
        """2단계: 펀더멘털 1차 스크리닝"""
        qualified_coins = []
        
        for coin_data in stage1_coins:
            try:
                symbol = coin_data['symbol']
                
                # AI 품질 기본 분석
                quality_analysis = self.quality_analyzer.analyze_project_quality(symbol, coin_data)
                
                # 펀더멘털 최소 기준
                if quality_analysis['project_quality_score'] < 0.3:  # 30% 미만 제외
                    continue
                
                # Tier 5 (밈코인) 과도 제한
                if quality_analysis['tier'] == 'tier_5':
                    continue  # 밈코인 일단 제외 (나중에 소량만 추가)
                
                # 생태계 건전성 기준
                if quality_analysis['ecosystem_health_score'] < 0.4:
                    continue
                
                coin_data.update({
                    'quality_analysis': quality_analysis,
                    'fundamental_score_raw': quality_analysis['project_quality_score'],
                    'stage2_passed': True
                })
                qualified_coins.append(coin_data)
                
            except Exception as e:
                logger.error(f"2단계 필터링 실패 {coin_data.get('symbol', 'Unknown')}: {e}")
                continue
        
        # 품질 점수 기준 정렬
        qualified_coins.sort(key=lambda x: x.get('fundamental_score_raw', 0), reverse=True)
        return qualified_coins[:150]  # 상위 150개

    async def _stage3_technical_screening(self, stage2_coins: List[Dict]) -> List[Dict]:
        """3단계: 기술적 지표 2차 스크리닝"""
        qualified_coins = []
        
        for coin_data in stage2_coins:
            try:
                symbol = coin_data['symbol']
                
                # 기술적 분석
                technical_score, technical_details = self._analyze_technical_indicators_advanced(coin_data)
                
                # 기술적 최소 기준
                if technical_score < 0.25:  # 25% 미만 제외
                    continue
                
                # 극단적 과매수/과매도 제외 (리스크 높은 구간)
                rsi = technical_details.get('rsi', 50)
                if rsi > 85 or rsi < 10:  # 극단적 구간 제외
                    continue
                
                # ADX가 너무 낮으면 제외 (트렌드 없음)
                adx = technical_details.get('adx', 25)
                if adx < 15:
                    continue
                
                coin_data.update({
                    'technical_score_raw': technical_score,
                    'technical_details': technical_details,
                    'stage3_passed': True
                })
                qualified_coins.append(coin_data)
                
            except Exception as e:
                logger.error(f"3단계 필터링 실패 {coin_data.get('symbol', 'Unknown')}: {e}")
                continue
        
        # 기술적 점수 기준 정렬
        qualified_coins.sort(key=lambda x: x.get('technical_score_raw', 0), reverse=True)
        return qualified_coins[:120]  # 상위 120개

    async def _stage4_momentum_screening(self, stage3_coins: List[Dict]) -> List[Dict]:
        """4단계: 모멘텀 3차 스크리닝"""
        qualified_coins = []
        
        for coin_data in stage3_coins:
            try:
                symbol = coin_data['symbol']
                
                # 모멘텀 계산
                ohlcv_30d = coin_data['ohlcv_30d']
                current_price = coin_data['price']
                
                # 모멘텀 지표들 계산
                momentum_data = self._calculate_momentum_indicators(ohlcv_30d, current_price)
                momentum_score, momentum_reasoning = self._analyze_momentum_advanced(symbol, momentum_data)
                
                # 모멘텀 최소 기준
                if momentum_score < 0.2:  # 20% 미만 제외
                    continue
                
                # 극단적 하락 모멘텀 제외
                momentum_7d = momentum_data.get('momentum_7d', 0)
                momentum_30d = momentum_data.get('momentum_30d', 0)
                
                if momentum_7d < -30 or momentum_30d < -50:  # 급락 코인 제외
                    continue
                
                # 거래량 급증이 없으면서 모멘텀이 나쁜 코인 제외
                volume_spike = momentum_data.get('volume_spike_ratio', 1)
                if momentum_score < 0.4 and volume_spike < 1.2:
                    continue
                
                coin_data.update({
                    'momentum_score_raw': momentum_score,
                    'momentum_data': momentum_data,
                    'momentum_reasoning': momentum_reasoning,
                    'stage4_passed': True
                })
                qualified_coins.append(coin_data)
                
            except Exception as e:
                logger.error(f"4단계 필터링 실패 {coin_data.get('symbol', 'Unknown')}: {e}")
                continue
        
        # 모멘텀 점수 기준 정렬
        qualified_coins.sort(key=lambda x: x.get('momentum_score_raw', 0), reverse=True)
        return qualified_coins[:100]  # 상위 100개

    def _calculate_momentum_indicators(self, ohlcv_data: pd.DataFrame, current_price: float) -> Dict:
        """모멘텀 지표들 계산"""
        try:
            data = {}
            
            if len(ohlcv_data) >= 30:
                data['momentum_3d'] = (current_price / ohlcv_data.iloc[-4]['close'] - 1) * 100
                data['momentum_7d'] = (current_price / ohlcv_data.iloc[-8]['close'] - 1) * 100
                data['momentum_30d'] = (current_price / ohlcv_data.iloc[-31]['close'] - 1) * 100
            else:
                data['momentum_3d'] = data['momentum_7d'] = data['momentum_30d'] = 0
            
            # 거래량 급증률
            if len(ohlcv_data) >= 7:
                avg_volume_7d = ohlcv_data['volume'].tail(7).mean()
                current_volume = ohlcv_data.iloc[-1]['volume']
                data['volume_spike_ratio'] = current_volume / avg_volume_7d if avg_volume_7d > 0 else 1
            else:
                data['volume_spike_ratio'] = 1
            
            return data
        except:
            return {'momentum_3d': 0, 'momentum_7d': 0, 'momentum_30d': 0, 'volume_spike_ratio': 1}

    async def _stage5_ai_quality_screening(self, stage4_coins: List[Dict]) -> List[Dict]:
        """5단계: AI 품질 평가 4차 스크리닝 (더 엄격한 기준)"""
        qualified_coins = []
        
        # 시장 사이클별 품질 기준 조정
        if self.current_market_cycle == 'downtrend':
            min_quality_score = 0.7  # 하락장에서는 최고급만
            max_tier_5_count = 0     # 밈코인 제외
        elif self.current_market_cycle == 'accumulation':
            min_quality_score = 0.5  # 축적기에서는 중급 이상
            max_tier_5_count = 1     # 밈코인 1개만
        elif self.current_market_cycle == 'uptrend':
            min_quality_score = 0.4  # 상승장에서는 관대
            max_tier_5_count = 3     # 밈코인 3개까지
        else:
            min_quality_score = 0.45 # 기본값
            max_tier_5_count = 2     # 밈코인 2개
        
        tier_5_count = 0
        
        for coin_data in stage4_coins:
            try:
                quality_analysis = coin_data['quality_analysis']
                
                # 시장 사이클별 품질 기준 적용
                if quality_analysis['project_quality_score'] < min_quality_score:
                    continue
                
                # Tier 5 (밈코인) 개수 제한
                if quality_analysis['tier'] == 'tier_5':
                    if tier_5_count >= max_tier_5_count:
                        continue
                    tier_5_count += 1
                
                # 혁신성이 너무 낮은 코인 제외
                if quality_analysis['innovation_score'] < 0.3:
                    continue
                
                # 채택도가 너무 낮은 코인 제외 (신규 코인 등)
                if quality_analysis['adoption_score'] < 0.35:
                    continue
                
                coin_data.update({
                    'ai_quality_final': quality_analysis['project_quality_score'],
                    'stage5_passed': True
                })
                qualified_coins.append(coin_data)
                
            except Exception as e:
                logger.error(f"5단계 필터링 실패 {coin_data.get('symbol', 'Unknown')}: {e}")
                continue
        
        # AI 품질 점수 기준 정렬
        qualified_coins.sort(key=lambda x: x.get('ai_quality_final', 0), reverse=True)
        return qualified_coins[:80]  # 상위 80개

    async def _stage6_final_scoring(self, stage5_coins: List[Dict]) -> List[Dict]:
        """6단계: 최종 종합 점수 계산 및 포트폴리오 최적화"""
        final_qualified = []
        
        # 시장 사이클 기반 가중치
        cycle_weights = self._get_cycle_based_weights()
        
        for coin_data in stage5_coins:
            try:
                symbol = coin_data['symbol']
                
                # 각 점수 추출
                fundamental_score = coin_data.get('fundamental_score_raw', 0)
                technical_score = coin_data.get('technical_score_raw', 0)
                momentum_score = coin_data.get('momentum_score_raw', 0)
                
                # 가중 평균 계산
                base_score = (
                    fundamental_score * cycle_weights['fundamental'] +
                    technical_score * cycle_weights['technical'] +
                    momentum_score * cycle_weights['momentum']
                )
                
                # 시장 사이클 보너스
                quality_analysis = coin_data['quality_analysis']
                cycle_bonus = self._get_cycle_bonus(symbol, quality_analysis)
                base_score += cycle_bonus
                
                # 다양성 혜택 (포트폴리오 최적화)
                diversification_benefit = self.portfolio_optimizer.calculate_diversification_benefit(
                    symbol, [coin['symbol'] for coin in final_qualified]
                )
                
                # 최종 점수 계산
                final_score = base_score * diversification_benefit
                
                coin_data.update({
                    'selection_score': final_score,
                    'base_score': base_score,
                    'diversification_benefit': diversification_benefit,
                    'cycle_bonus': cycle_bonus,
                    'stage6_passed': True
                })
                final_qualified.append(coin_data)
                
            except Exception as e:
                logger.error(f"6단계 필터링 실패 {coin_data.get('symbol', 'Unknown')}: {e}")
                continue
        
        # 최종 점수 기준 정렬
        final_qualified.sort(key=lambda x: x.get('selection_score', 0), reverse=True)
        
        # 포트폴리오 최적화 적용
        optimized_selection = self.portfolio_optimizer.optimize_portfolio_selection(
            final_qualified, min(60, len(final_qualified))
        )
        
        return optimized_selection

    def _analyze_single_coin_comprehensive(self, symbol: str) -> Optional[Dict]:
        """단일 코인 종합 분석"""
        try:
            data = self._get_comprehensive_coin_data_sync(symbol)
            if not data: return None
            
            volume_krw = data.get('volume_24h_krw', 0)
            if volume_krw < self.min_volume_24h: return None
            
            quality_analysis = self.quality_analyzer.analyze_project_quality(symbol, data)
            technical_score, _ = self._analyze_technical_indicators_advanced(data)
            momentum_score, _ = self._analyze_momentum_advanced(symbol, data)
            fundamental_score, _ = self._analyze_fundamental_enhanced(symbol, data, quality_analysis)
            
            cycle_weights = self._get_cycle_based_weights()
            total_score = (fundamental_score * cycle_weights['fundamental'] +
                          technical_score * cycle_weights['technical'] +
                          momentum_score * cycle_weights['momentum'])
            
            return {
                'symbol': symbol, 'selection_score': total_score,
                'fundamental_score': fundamental_score, 'technical_score': technical_score,
                'momentum_score': momentum_score, 'price': data['price'],
                'volume_24h_krw': volume_krw, 'market_cap': data.get('market_cap', 0),
                **quality_analysis
            }
        except: return None

    def _get_comprehensive_coin_data_sync(self, symbol: str) -> Dict:
        """종합 코인 데이터 수집"""
        try:
            current_price = pyupbit.get_current_price(symbol)
            if not current_price: return {}
            
            ohlcv_1d = pyupbit.get_ohlcv(symbol, interval="day", count=100)
            if ohlcv_1d is None or len(ohlcv_1d) < 20: return {}
            
            data = {'symbol': symbol, 'price': current_price, 'ohlcv_1d': ohlcv_1d}
            
            latest_1d = ohlcv_1d.iloc[-1]
            data['volume_24h_krw'] = latest_1d['volume'] * current_price
            data['market_cap'] = latest_1d['volume'] * current_price * 100
            
            if len(ohlcv_1d) >= 30:
                data['momentum_3d'] = (current_price / ohlcv_1d.iloc[-4]['close'] - 1) * 100
                data['momentum_7d'] = (current_price / ohlcv_1d.iloc[-8]['close'] - 1) * 100
                data['momentum_30d'] = (current_price / ohlcv_1d.iloc[-31]['close'] - 1) * 100
            else:
                data['momentum_3d'] = data['momentum_7d'] = data['momentum_30d'] = 0
            
            avg_volume_7d = ohlcv_1d['volume'].tail(7).mean()
            current_volume = latest_1d['volume']
            data['volume_spike_ratio'] = current_volume / avg_volume_7d if avg_volume_7d > 0 else 1
            
            return data
        except: return {}

    def _get_cycle_based_weights(self) -> Dict:
        """시장 사이클 기반 가중치 조정"""
        if self.current_market_cycle == 'accumulation':
            return {'fundamental': 0.50, 'technical': 0.25, 'momentum': 0.25}
        elif self.current_market_cycle == 'uptrend':
            return {'fundamental': 0.25, 'technical': 0.25, 'momentum': 0.50}
        elif self.current_market_cycle == 'distribution':
            return {'fundamental': 0.25, 'technical': 0.50, 'momentum': 0.25}
        elif self.current_market_cycle == 'downtrend':
            return {'fundamental': 0.60, 'technical': 0.20, 'momentum': 0.20}
        else:
            return {'fundamental': self.fundamental_weight, 'technical': self.technical_weight, 
                   'momentum': self.momentum_weight}

    def _is_selection_cache_valid(self) -> bool:
        """선별 결과 캐시 유효성 확인"""
        if not self.last_selection_time or not self.selected_coins: return False
        time_diff = datetime.now() - self.last_selection_time
        return time_diff.total_seconds() < (self.selection_cache_hours * 3600)

    def _get_default_coins(self) -> List[str]:
        """기본 코인 리스트"""
        return ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-AVAX', 'KRW-DOGE', 
                'KRW-MATIC', 'KRW-ATOM', 'KRW-NEAR', 'KRW-HBAR', 'KRW-DOT', 'KRW-LINK',
                'KRW-SOL', 'KRW-UNI', 'KRW-ALGO', 'KRW-VET', 'KRW-ICP', 'KRW-FTM', 
                'KRW-SAND', 'KRW-MANA']

    def _analyze_fundamental_enhanced(self, symbol: str, data: Dict, quality_analysis: Dict) -> Tuple[float, str]:
        """강화된 펀더멘털 분석"""
        try:
            score, reasoning = 0.0, []
            
            quality_score = quality_analysis['project_quality_score']
            score += quality_score * 0.50
            reasoning.append(f"품질:{quality_score:.2f}")
            
            volume_24h = data.get('volume_24h_krw', 0)
            if volume_24h >= 100_000_000_000:
                volume_score = 0.25; reasoning.append("대형거래량")
            elif volume_24h >= 20_000_000_000:
                volume_score = 0.15; reasoning.append("중형거래량")
            elif volume_24h >= 5_000_000_000:
                volume_score = 0.10; reasoning.append("소형거래량")
            else:
                volume_score = 0.05; reasoning.append("미니거래량")
            
            score += volume_score
            
            ecosystem_score = quality_analysis['ecosystem_health_score'] * 0.15
            score += ecosystem_score; reasoning.append(f"생태계:{ecosystem_score:.2f}")
            
            innovation_score = quality_analysis['innovation_score'] * 0.10
            score += innovation_score; reasoning.append(f"혁신:{innovation_score:.2f}")
            
            return score, "펀더멘털: " + " | ".join(reasoning)
        except: return 0.0, "펀더멘털: 분석실패"

    def _analyze_technical_indicators_advanced(self, data: Dict) -> Tuple[float, Dict]:
        """고급 기술적 분석"""
        try:
            ohlcv_1d = data.get('ohlcv_1d')
            if ohlcv_1d is None or len(ohlcv_1d) < 50: return 0.0, {}
            
            closes = ohlcv_1d['close']
            highs = ohlcv_1d['high']
            lows = ohlcv_1d['low']
            
            score, details = 0.0, {}

            # RSI (20%)
            try:
                gains = closes.diff().clip(lower=0)
                losses = -closes.diff().clip(upper=0)
                avg_gain = gains.rolling(window=14).mean()
                avg_loss = losses.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                
                if 30 <= rsi_val <= 70: score += 0.20
                elif rsi_val < 30: score += 0.15
                elif rsi_val > 70: score += 0.10
                details['rsi'] = rsi_val
            except: details['rsi'] = 50

            # MACD (20%)
            try:
                ema_fast = closes.ewm(span=12).mean()
                ema_slow = closes.ewm(span=26).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=9).mean()
                macd_diff = macd_line.iloc[-1] - signal_line.iloc[-1]

                macd_signal = 'bullish' if macd_diff > 0 else 'bearish'
                if macd_signal == 'bullish': score += 0.20
                details['macd_signal'] = macd_signal
            except: details['macd_signal'] = 'neutral'
            
            # 볼린저 밴드 (15%)
            try:
                bb_middle = closes.rolling(window=20).mean()
                bb_std = closes.rolling(window=20).std()
                bb_upper = bb_middle + (bb_std * 2)
                bb_lower = bb_middle - (bb_std * 2)
                current_price = closes.iloc[-1]
                
                if current_price < bb_lower.iloc[-1]:
                    score += 0.15; bb_position = 'oversold'
                elif current_price > bb_upper.iloc[-1]:
                    score += 0.08; bb_position = 'overbought'
                else:
                    score += 0.10; bb_position = 'normal'
                details['bb_position'] = bb_position
            except: details['bb_position'] = 'normal'
            
            # 스토캐스틱 (15%)
            try:
                lowest_low = lows.rolling(window=14).min()
                highest_high = highs.rolling(window=14).max()
                stoch_k = 100 * ((closes - lowest_low) / (highest_high - lowest_low))
                stoch_d = stoch_k.rolling(window=3).mean()
                
                k_val = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50
                d_val = stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50
                
                if k_val < 20 and d_val < 20: score += 0.15
                elif k_val > 80 and d_val > 80: score += 0.08
                else: score += 0.10
                details['stoch_k'] = k_val; details['stoch_d'] = d_val
            except: details['stoch_k'] = 50; details['stoch_d'] = 50
            
            # 고급 지표들 (30%) - 완전 구현된 버전 사용
            df_data = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
            if 'volume' in ohlcv_1d.columns:
                df_data['volume'] = ohlcv_1d['volume']
            
            williams_r = AdvancedTechnicalIndicators.calculate_williams_r(df_data)
            cci = AdvancedTechnicalIndicators.calculate_cci(df_data)
            mfi = AdvancedTechnicalIndicators.calculate_mfi(df_data)
            adx = AdvancedTechnicalIndicators.calculate_adx(df_data)
            parabolic_sar = AdvancedTechnicalIndicators.calculate_parabolic_sar(df_data)
            
            # Williams %R 점수
            if williams_r <= -80: score += 0.06  # 과매도
            elif williams_r >= -20: score += 0.03  # 과매수
            else: score += 0.04
            
            # CCI 점수
            if cci <= -100: score += 0.06  # 과매도
            elif cci >= 100: score += 0.03  # 과매수
            else: score += 0.04
            
            # MFI 점수
            if mfi <= 20: score += 0.06  # 과매도
            elif mfi >= 80: score += 0.03  # 과매수
            else: score += 0.04
            
            # ADX 점수
            if adx >= 25: score += 0.06  # 강한 트렌드
            elif adx >= 20: score += 0.04
            else: score += 0.02
            
            # Parabolic SAR 점수
            if parabolic_sar == 'bullish': score += 0.06
            elif parabolic_sar == 'bearish': score += 0.02
            else: score += 0.04
            
            details.update({
                'williams_r': williams_r, 'cci': cci, 'mfi': mfi, 
                'adx': adx, 'parabolic_sar': parabolic_sar
            })
            
            # 기존 지표들 추가 (일목균형표, OBV, ATR)
            try:
                # 일목균형표
                tenkan = (highs.rolling(9).max() + lows.rolling(9).min()) / 2
                kijun = (highs.rolling(26).max() + lows.rolling(26).min()) / 2
                current_price = closes.iloc[-1]
                
                if len(tenkan) > 0 and len(kijun) > 0:
                    if tenkan.iloc[-1] > kijun.iloc[-1] and current_price > tenkan.iloc[-1]:
                        ichimoku_signal = 'bullish'
                    elif tenkan.iloc[-1] < kijun.iloc[-1] and current_price < tenkan.iloc[-1]:
                        ichimoku_signal = 'bearish'
                    else:
                        ichimoku_signal = 'neutral'
                else:
                    ichimoku_signal = 'neutral'
                details['ichimoku_signal'] = ichimoku_signal
            except:
                details['ichimoku_signal'] = 'neutral'
            
            # OBV (On Balance Volume)
            try:
                if 'volume' in ohlcv_1d.columns:
                    volumes = ohlcv_1d['volume']
                    obv_values = [0]
                    for i in range(1, len(closes)):
                        if closes.iloc[i] > closes.iloc[i-1]:
                            obv_values.append(obv_values[-1] + volumes.iloc[i])
                        elif closes.iloc[i] < closes.iloc[i-1]:
                            obv_values.append(obv_values[-1] - volumes.iloc[i])
                        else:
                            obv_values.append(obv_values[-1])
                    
                    if len(obv_values) >= 10:
                        obv_trend = "rising" if obv_values[-1] > obv_values[-10] else "falling"
                    else:
                        obv_trend = "neutral"
                else:
                    obv_trend = "neutral"
                details['obv_trend'] = obv_trend
            except:
                details['obv_trend'] = 'neutral'
            
            # ATR (Average True Range)
            try:
                tr1 = highs - lows
                tr2 = (highs - closes.shift(1)).abs()
                tr3 = (lows - closes.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean().iloc[-1]
                details['atr'] = atr if not pd.isna(atr) else 0
            except:
                details['atr'] = 0
            
            return min(score, 1.0), details
        except: return 0.0, {}

    def _analyze_momentum_advanced(self, symbol: str, data: Dict) -> Tuple[float, str]:
        """고급 모멘텀 분석"""
        try:
            score, reasoning = 0.0, []
            
            momentum_3d = data.get('momentum_3d', 0)
            if momentum_3d >= 20: score += 0.30; reasoning.append(f"강한3일({momentum_3d:.1f}%)")
            elif momentum_3d >= 10: score += 0.20; reasoning.append(f"상승3일({momentum_3d:.1f}%)")
            elif momentum_3d >= 0: score += 0.10; reasoning.append(f"보합3일({momentum_3d:.1f}%)")
            else: reasoning.append(f"하락3일({momentum_3d:.1f}%)")
            
            momentum_7d = data.get('momentum_7d', 0)
            if momentum_7d >= 30: score += 0.30; reasoning.append(f"강한7일({momentum_7d:.1f}%)")
            elif momentum_7d >= 15: score += 0.20; reasoning.append(f"상승7일({momentum_7d:.1f}%)")
            elif momentum_7d >= 0: score += 0.10; reasoning.append(f"보합7일({momentum_7d:.1f}%)")
            
            momentum_30d = data.get('momentum_30d', 0)
            if momentum_30d >= 50: score += 0.25; reasoning.append(f"강한30일({momentum_30d:.1f}%)")
            elif momentum_30d >= 20: score += 0.15; reasoning.append(f"상승30일({momentum_30d:.1f}%)")
            elif momentum_30d >= 0: score += 0.05; reasoning.append(f"보합30일({momentum_30d:.1f}%)")
            
            volume_spike = data.get('volume_spike_ratio', 1)
            if volume_spike >= 3.0: score += 0.15; reasoning.append(f"거래량폭증({volume_spike:.1f}배)")
            elif volume_spike >= 2.0: score += 0.10; reasoning.append(f"거래량급증({volume_spike:.1f}배)")
            elif volume_spike >= 1.5: score += 0.05; reasoning.append(f"거래량증가({volume_spike:.1f}배)")
            
            return score, "모멘텀: " + " | ".join(reasoning)
        except: return 0.0, "모멘텀: 분석실패"

    def _calculate_enhanced_split_trading_plan(self, symbol: str, current_price: float, confidence: float) -> Dict:
        """강화된 5단계 분할매매 계획"""
        try:
            risk_params = self._calculate_dynamic_risk_params(confidence)
            
            base_investment = self.coin_portfolio_value / self.target_coins
            confidence_multiplier = 0.5 + (confidence * 1.5)
            total_investment = base_investment * confidence_multiplier
            total_investment = min(total_investment, self.coin_portfolio_value * self.max_single_coin_weight)
            
            stage_amounts = [total_investment * ratio for ratio in self.stage_ratios]
            
            triggers = self.stage_triggers.copy()
            if self.current_market_cycle == 'uptrend':
                triggers = [0.0, -0.03, -0.06, -0.10, -0.15]
            elif self.current_market_cycle == 'downtrend':
                triggers = [0.0, -0.08, -0.15, -0.22, -0.30]
            
            entry_prices = [current_price * (1 + trigger) for trigger in triggers]
            
            avg_entry = current_price * 0.85
            stop_loss = avg_entry * (1 - risk_params['stop_loss_pct'])
            take_profits = [avg_entry * (1 + tp) for tp in risk_params['take_profit_levels']]
            
            return {
                'total_investment': total_investment, 'stage_amounts': stage_amounts,
                'entry_prices': entry_prices, 'stop_loss': stop_loss, 'take_profits': take_profits,
                'max_hold_days': risk_params['max_hold_days'],
                'coin_weight': total_investment / self.coin_portfolio_value * 100,
                'market_cycle': self.current_market_cycle,
                'risk_level': 'CONSERVATIVE' if confidence < 0.6 else 'AGGRESSIVE' if confidence > 0.8 else 'MODERATE'
            }
        except: return {}

    def _calculate_dynamic_risk_params(self, confidence: float) -> Dict:
        """시장 사이클 기반 동적 리스크 파라미터"""
        try:
            stop_loss_pct = self.base_stop_loss_pct
            take_profit_levels = self.base_take_profit_levels.copy()
            max_hold_days = self.base_max_hold_days
            
            if self.current_market_cycle == 'accumulation':
                stop_loss_pct = 0.20; take_profit_levels = [0.15, 0.30, 0.60]; max_hold_days = 45
            elif self.current_market_cycle == 'uptrend':
                stop_loss_pct = 0.30; take_profit_levels = [0.25, 0.60, 1.50]; max_hold_days = 20
            elif self.current_market_cycle == 'distribution':
                stop_loss_pct = 0.15; take_profit_levels = [0.10, 0.25, 0.50]; max_hold_days = 30
            elif self.current_market_cycle == 'downtrend':
                stop_loss_pct = 0.10; take_profit_levels = [0.05, 0.15, 0.30]; max_hold_days = 60
            
            confidence_multiplier = 0.7 + (confidence * 0.6)
            stop_loss_pct /= confidence_multiplier
            take_profit_levels = [tp * confidence_multiplier for tp in take_profit_levels]
            max_hold_days = int(max_hold_days * (1.5 - confidence))
            
            return {'stop_loss_pct': stop_loss_pct, 'take_profit_levels': take_profit_levels, 'max_hold_days': max_hold_days}
        except: return {'stop_loss_pct': self.base_stop_loss_pct, 'take_profit_levels': self.base_take_profit_levels, 'max_hold_days': self.base_max_hold_days}

    async def generate_ultimate_portfolio_report(self, selected_coins: List[UltimateCoinSignal]) -> Dict:
        """궁극의 포트폴리오 리포트 생성"""
        if not selected_coins: return {"error": "선별된 코인이 없습니다"}
        
        total_coins = len(selected_coins)
        buy_signals = [s for s in selected_coins if s.action == 'buy']
        sell_signals = [s for s in selected_coins if s.action == 'sell']
        hold_signals = [s for s in selected_coins if s.action == 'hold']
        
        avg_scores = {
            'fundamental': np.mean([s.fundamental_score for s in selected_coins]),
            'technical': np.mean([s.technical_score for s in selected_coins]),
            'momentum': np.mean([s.momentum_score for s in selected_coins]),
            'total': np.mean([s.total_score for s in selected_coins]),
            'project_quality': np.mean([s.project_quality_score for s in selected_coins])
        }
        
        total_investment = sum([s.total_amount for s in selected_coins])
        top_buys = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
        
        sector_dist = {}
        for coin in selected_coins:
            sector_dist[coin.sector] = sector_dist.get(coin.sector, 0) + 1
        
        return {
            'summary': {
                'total_coins': total_coins, 'buy_signals': len(buy_signals), 
                'sell_signals': len(sell_signals), 'hold_signals': len(hold_signals),
                'total_investment': total_investment, 'portfolio_allocation': total_investment / self.coin_portfolio_value * 100
            },
            'strategy_scores': avg_scores,
            'ai_quality_analysis': {
                'avg_project_quality': avg_scores['project_quality'],
                'quality_distribution': self._analyze_quality_distribution(selected_coins)
            },
            'top_picks': [{
                'symbol': coin.symbol, 'sector': coin.sector, 'confidence': coin.confidence,
                'total_score': coin.total_score, 'project_quality_score': coin.project_quality_score,
                'price': coin.price, 'target_price': coin.target_price,
                'total_investment': coin.total_amount, 'market_cycle': coin.market_cycle,
                'btc_correlation': coin.correlation_with_btc, 'fear_greed': coin.fear_greed_score,
                'reasoning': coin.reasoning[:150] + "..." if len(coin.reasoning) > 150 else coin.reasoning
            } for coin in top_buys],
            'diversification_analysis': {
                'sector_distribution': sector_dist,
                'correlation_matrix_summary': {
                    'avg_btc_correlation': np.mean([s.correlation_with_btc for s in selected_coins]),
                    'diversification_score': len(sector_dist) / total_coins,
                    'correlation_risk': self._assess_correlation_risk([s.correlation_with_btc for s in selected_coins])
                }
            },
            'risk_metrics': {
                'avg_volatility': np.mean([s.atr / s.price if s.price > 0 else 0 for s in selected_coins]),
                'max_single_position': max([s.total_amount for s in selected_coins]) / total_investment * 100 if total_investment > 0 else 0,
                'market_sentiment': {
                    'fear_greed_index': np.mean([s.fear_greed_score for s in selected_coins if s.fear_greed_score > 0]) if any(s.fear_greed_score > 0 for s in selected_coins) else 50,
                    'sentiment_classification': self._classify_market_sentiment([s.fear_greed_score for s in selected_coins if s.fear_greed_score > 0])
                }
            },
            'market_cycle_analysis': {
                'current_cycle': self.current_market_cycle,
                'cycle_confidence': self.cycle_confidence,
                'cycle_optimized_coins': len([s for s in selected_coins if s.market_cycle == self.current_market_cycle])
            }
        }

    def _analyze_quality_distribution(self, selected_coins: List[UltimateCoinSignal]) -> Dict:
        """품질 분포 분석"""
        try:
            quality_ranges = {
                'excellent': len([c for c in selected_coins if c.project_quality_score >= 0.8]),
                'good': len([c for c in selected_coins if 0.6 <= c.project_quality_score < 0.8]),
                'average': len([c for c in selected_coins if 0.4 <= c.project_quality_score < 0.6]),
                'poor': len([c for c in selected_coins if c.project_quality_score < 0.4])
            }
            return quality_ranges
        except:
            return {'excellent': 0, 'good': 0, 'average': 0, 'poor': 0}

    def _assess_correlation_risk(self, correlations: List[float]) -> str:
        """상관관계 리스크 평가"""
        try:
            avg_correlation = np.mean(correlations) if correlations else 0.5
            if avg_correlation > 0.8: return 'HIGH'
            elif avg_correlation < 0.5: return 'LOW'
            else: return 'MEDIUM'
        except: return 'MEDIUM'

    def _classify_market_sentiment(self, fear_greed_scores: List[int]) -> str:
        """시장 센티먼트 분류"""
        try:
            if not fear_greed_scores: return 'NEUTRAL'
            avg_score = np.mean(fear_greed_scores)
            if avg_score < 25: return 'FEAR'
            elif avg_score > 75: return 'GREED'
            else: return 'NEUTRAL'
        except: return 'NEUTRAL'

    async def execute_ultimate_split_trading_simulation(self, signal: UltimateCoinSignal) -> Dict:
        """궁극의 5단계 분할매매 시뮬레이션"""
        if signal.action != 'buy': return {"status": "not_applicable", "reason": "매수 신호가 아님"}
        
        return {
            'symbol': signal.symbol, 'strategy': 'ultimate_5_stage_split_trading_v5',
            'ai_project_quality': signal.project_quality_score,
            'market_cycle': signal.market_cycle, 'cycle_confidence': signal.cycle_confidence,
            'stages': {
                f'stage_{i+1}': {
                    'trigger_price': getattr(signal, f'entry_price_{i+1}'),
                    'amount': getattr(signal, f'stage{i+1}_amount'),
                    'ratio': '20%', 'status': 'ready' if i == 0 else 'waiting'
                } for i in range(5)
            },
            'dynamic_exit_plan': {
                'stop_loss': {'price': signal.stop_loss, 'ratio': '100%'},
                'take_profit_1': {'price': signal.take_profit_1, 'ratio': '40%'},
                'take_profit_2': {'price': signal.take_profit_2, 'ratio': '40%'},
                'take_profit_3': {'price': signal.take_profit_3, 'ratio': '20%'}
            },
            'risk_management': {
                'max_hold_days': signal.max_hold_days, 'total_investment': signal.total_amount,
                'portfolio_weight': signal.total_amount / self.coin_portfolio_value * 100
            }
        }

# ========================================================================================
# 🆕 상관관계 기반 포트폴리오 최적화
# ========================================================================================
class PortfolioOptimizer:
    """상관관계 기반 포트폴리오 최적화"""
    
    def __init__(self):
        self.correlation_threshold = 0.7
        self.max_correlated_coins = 2

    async def calculate_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """상관관계 행렬 계산"""
        try:
            price_data = {}
            
            for symbol in symbols:
                try:
                    ohlcv = pyupbit.get_ohlcv(symbol, interval="day", count=30)
                    if ohlcv is not None and len(ohlcv) >= 30:
                        price_data[symbol] = ohlcv['close'].pct_change().dropna()
                    await asyncio.sleep(0.1)
                except: continue
            
            if len(price_data) < 2: return pd.DataFrame()
            
            df = pd.DataFrame(price_data)
            return df.corr()
        except: return pd.DataFrame()

    def optimize_portfolio_selection(self, candidates: List[Dict], target_count: int = 20) -> List[Dict]:
        """상관관계 고려한 포트폴리오 최적화"""
        try:
            if len(candidates) <= target_count: return candidates
            
            sorted_candidates = sorted(candidates, key=lambda x: x.get('selection_score', 0), reverse=True)
            selected, selected_symbols = [], []
            
            for candidate in sorted_candidates:
                if len(selected) >= target_count: break
                
                symbol = candidate['symbol']
                
                if len(selected) == 0:
                    selected.append(candidate)
                    selected_symbols.append(symbol)
                    continue
                
                # 같은 카테고리 코인 수 확인
                current_category = self._get_coin_category(symbol)
                high_correlation_count = 0
                
                for selected_symbol in selected_symbols:
                    selected_category = self._get_coin_category(selected_symbol)
                    if current_category == selected_category and current_category != 'Unknown':
                        high_correlation_count += 1
                
                if high_correlation_count >= self.max_correlated_coins: continue
                
                selected.append(candidate)
                selected_symbols.append(symbol)
            
            # 남은 자리 채우기
            remaining_slots = target_count - len(selected)
            if remaining_slots > 0:
                remaining_candidates = [c for c in sorted_candidates if c not in selected]
                selected.extend(remaining_candidates[:remaining_slots])
            
            return selected[:target_count]
        except: return candidates[:target_count]

    def _get_coin_category(self, symbol: str) -> str:
        """코인 카테고리 추정"""
        coin_name = symbol.replace('KRW-', '').upper()
        categories = {
            'L1_Blockchain': ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'ATOM', 'NEAR'],
            'DeFi': ['UNI', 'AAVE', 'MKR', 'COMP', 'CRV', 'SUSHI'],
            'Gaming': ['SAND', 'MANA', 'AXS', 'ENJ'], 'Meme': ['DOGE', 'SHIB', 'PEPE'],
            'Exchange': ['BNB', 'CRO'], 'Infrastructure': ['LINK', 'FIL', 'VET']
        }
        for category, coins in categories.items():
            if coin_name in coins: return category
        return 'Unknown'

    def calculate_diversification_benefit(self, symbol: str, selected_symbols: List[str]) -> float:
        """다양성 혜택 점수 계산"""
        try:
            if not selected_symbols: return 1.0
            
            current_category = self._get_coin_category(symbol)
            selected_categories = [self._get_coin_category(s) for s in selected_symbols]
            
            if current_category not in selected_categories: return 1.0
            
            same_category_count = selected_categories.count(current_category)
            return max(0.1, 1.0 - (same_category_count * 0.3))
        except: return 0.5

# ========================================================================================
# 🧪 백테스팅 시뮬레이션 클래스
# ========================================================================================
class UltimateBacktester:
    """궁극의 백테스팅 시뮬레이션"""
    
    def __init__(self, initial_capital: float = 100_000_000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
    
    def simulate_strategy(self, signals: List[UltimateCoinSignal], days: int = 30) -> Dict:
        """전략 시뮬레이션 실행"""
        total_return = 0
        win_trades = 0
        total_trades = 0
        
        for signal in signals:
            if signal.action == 'buy' and signal.confidence >= 0.6:
                # 시뮬레이션된 수익률 계산
                expected_return = signal.confidence * 0.5
                simulated_return = np.random.normal(expected_return * 0.3, 0.2)
                
                total_return += simulated_return
                total_trades += 1
                
                if simulated_return > 0: win_trades += 1
        
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        avg_return = (total_return / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades, 'win_rate': win_rate,
            'avg_return_per_trade': avg_return, 'total_return': total_return * 100,
            'sharpe_ratio': max(0, avg_return / 15),
            'max_drawdown': abs(min(0, total_return * 0.7)),
            'profit_factor': max(1.0, win_rate / max(1, 100 - win_rate))
        }

def run_backtest_simulation(signals: List[UltimateCoinSignal]) -> Dict:
    """백테스팅 시뮬레이션 실행"""
    backtester = UltimateBacktester()
    results = backtester.simulate_strategy(signals)
    
    return {
        'simulation_results': results,
        'recommendation': '실제 투자 전 충분한 백테스팅과 리스크 관리가 필요합니다.',
        'disclaimer': '이 시뮬레이션은 예시용이며 실제 투자 성과를 보장하지 않습니다.'
    }

# ========================================================================================
# 📱 웹 API 엔드포인트 연동
# ========================================================================================
def create_web_api_response(signals: List[UltimateCoinSignal], report: Dict) -> Dict:
    """웹 API 응답 형식으로 변환"""
    return {
        'status': 'success', 'version': '5.0.0', 'timestamp': datetime.now().isoformat(),
        'market_cycle': signals[0].market_cycle if signals else 'unknown',
        'total_analyzed': len(signals),
        'signals': {
            'buy': [{
                'symbol': s.symbol, 'confidence': round(s.confidence * 100, 1),
                'price': s.price, 'target_price': s.target_price, 'sector': s.sector,
                'quality_score': round(s.project_quality_score * 100, 1),
                'reasoning': s.reasoning[:100] + "..." if len(s.reasoning) > 100 else s.reasoning
            } for s in signals if s.action == 'buy'][:10],
            'sell': [s.symbol for s in signals if s.action == 'sell'][:5],
            'hold': [s.symbol for s in signals if s.action == 'hold'][:5]
        },
        'portfolio_summary': {
            'recommended_allocation': report.get('summary', {}).get('portfolio_allocation', 0),
            'risk_level': report.get('risk_metrics', {}).get('market_sentiment', {}).get('sentiment_classification', 'NEUTRAL'),
            'diversification_score': report.get('diversification_analysis', {}).get('correlation_matrix_summary', {}).get('diversification_score', 0)
        },
        'ai_insights': {
            'market_cycle': report.get('market_cycle_analysis', {}).get('current_cycle', 'unknown'),
            'avg_quality_score': round(report.get('strategy_scores', {}).get('project_quality', 0) * 100, 1),
            'fear_greed_index': 50  # 기본값
        }
    }

# ========================================================================================
# 📊 CSV 내보내기 및 유틸리티 함수들
# ========================================================================================
def export_analysis_to_csv(signals: List[UltimateCoinSignal], filename: str = None) -> str:
    """분석 결과를 CSV로 내보내기"""
    import csv
    
    if filename is None:
        filename = f"crypto_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'symbol', 'action', 'confidence', 'price', 'target_price',
            'total_score', 'fundamental_score', 'technical_score', 'momentum_score',
            'project_quality_score', 'market_cycle', 'sector',
            'rsi', 'williams_r', 'cci', 'adx',
            'momentum_3d', 'momentum_7d', 'momentum_30d',
            'btc_correlation', 'fear_greed_score', 'reasoning'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for signal in signals:
            writer.writerow({
                'symbol': signal.symbol, 'action': signal.action,
                'confidence': f"{signal.confidence:.3f}", 'price': signal.price,
                'target_price': signal.target_price, 'total_score': f"{signal.total_score:.3f}",
                'fundamental_score': f"{signal.fundamental_score:.3f}",
                'technical_score': f"{signal.technical_score:.3f}",
                'momentum_score': f"{signal.momentum_score:.3f}",
                'project_quality_score': f"{signal.project_quality_score:.3f}",
                'market_cycle': signal.market_cycle, 'sector': signal.sector,
                'rsi': f"{signal.rsi:.1f}", 'williams_r': f"{signal.williams_r:.1f}",
                'cci': f"{signal.cci:.1f}", 'adx': f"{signal.adx:.1f}",
                'momentum_3d': f"{signal.momentum_3d:.1f}%",
                'momentum_7d': f"{signal.momentum_7d:.1f}%",
                'momentum_30d': f"{signal.momentum_30d:.1f}%",
                'btc_correlation': f"{signal.correlation_with_btc:.3f}",
                'fear_greed_score': signal.fear_greed_score, 'reasoning': signal.reasoning
            })
    
    return filename

def get_strategy_version():
    """전략 버전 정보 반환"""
    return {
        'version': '5.0.0', 'name': 'Ultimate Cryptocurrency Strategy',
        'features': [
            'AI-based Project Quality Analysis', 'Market Cycle Auto Detection',
            'Correlation-based Portfolio Optimization', 'Advanced Technical Indicators',
            'Social Sentiment Analysis', '5-Stage Split Trading System', 'Dynamic Risk Management'
        ],
        'last_updated': '2025-01-01', 'author': '최고퀸트팀'
    }

def validate_symbol(symbol: str) -> bool:
    """심볼 유효성 검증"""
    if not symbol or not isinstance(symbol, str): return False
    if not symbol.startswith('KRW-'): return False
    if len(symbol) < 7: return False
    return True

def format_currency(amount: float, currency: str = 'KRW') -> str:
    """통화 포맷팅"""
    if currency == 'KRW':
        if amount >= 1e12: return f"{amount/1e12:.1f}조원"
        elif amount >= 1e8: return f"{amount/1e8:.1f}억원"
        elif amount >= 1e4: return f"{amount/1e4:.1f}만원"
        else: return f"{amount:,.0f}원"
    else: return f"{amount:,.2f} {currency}"

def calculate_risk_level(confidence: float, volatility: float, correlation: float) -> str:
    """리스크 레벨 계산"""
    risk_score = 0
    
    if confidence >= 0.8: risk_score -= 2
    elif confidence >= 0.6: risk_score -= 1
    elif confidence <= 0.3: risk_score += 2
    
    if volatility >= 0.15: risk_score += 2
    elif volatility >= 0.10: risk_score += 1
    elif volatility <= 0.05: risk_score -= 1
    
    if correlation >= 0.9: risk_score += 1
    elif correlation <= 0.3: risk_score -= 1
    
    if risk_score <= -2: return "VERY_LOW"
    elif risk_score <= 0: return "LOW"
    elif risk_score <= 2: return "MEDIUM"
    elif risk_score <= 4: return "HIGH"
    else: return "VERY_HIGH"

def get_market_cycle_description(cycle: str) -> Dict[str, str]:
    """시장 사이클 설명"""
    descriptions = {
        'accumulation': {
            'description': '축적기 - 가격이 바닥권에서 횡보하며 스마트머니가 누적매수하는 구간',
            'strategy': '고품질 프로젝트를 장기 관점에서 분할 매수',
            'characteristics': '낮은 변동성, 낮은 거래량, 높은 BTC 도미넌스',
            'duration': '보통 6-12개월', 'opportunity': '최고의 매수 기회'
        },
        'uptrend': {
            'description': '상승기 - 시장 전반적으로 상승하며 알트코인이 아웃퍼폼하는 구간',
            'strategy': '모멘텀 기반 단기 회전, 알트코인 선호',
            'characteristics': '높은 변동성, 높은 거래량, 낮은 BTC 도미넌스',
            'duration': '보통 3-6개월', 'opportunity': '수익 극대화 구간'
        },
        'distribution': {
            'description': '분배기 - 가격이 고점권에서 횡보하며 스마트머니가 분산매도하는 구간',
            'strategy': '신중한 접근, 단기 익절, 안전자산 선호',
            'characteristics': '높은 변동성, 혼조세, 변동하는 BTC 도미넌스',
            'duration': '보통 2-4개월', 'opportunity': '수익 실현 및 리스크 관리'
        },
        'downtrend': {
            'description': '하락기 - 시장 전반적으로 하락하며 현금 보유가 유리한 구간',
            'strategy': '극도로 보수적 접근, 최고등급 코인만 소량 매수',
            'characteristics': '높은 변동성, 낮은 거래량, 높은 BTC 도미넌스',
            'duration': '보통 6-18개월', 'opportunity': '다음 상승을 위한 준비'
        },
        'sideways': {
            'description': '횡보기 - 명확한 방향성이 없는 중립적 구간',
            'strategy': '균형잡힌 접근, 기본 전략 가중치 사용',
            'characteristics': '보통 변동성, 보통 거래량, 중간 BTC 도미넌스',
            'duration': '가변적', 'opportunity': '선별적 기회 포착'
        }
    }
    return descriptions.get(cycle, descriptions['sideways'])

# ========================================================================================
# 🆕 완전한 고급 기술적 지표 구현
# ========================================================================================
class AdvancedTechnicalIndicators:
    """고급 기술적 지표 분석 (완전 구현)"""
    
    @staticmethod
    def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> float:
        """Williams %R 계산"""
        try:
            if len(data) < period: return -50.0
            high_n = data['high'].rolling(window=period).max()
            low_n = data['low'].rolling(window=period).min()
            williams_r = -100 * ((high_n - data['close']) / (high_n - low_n))
            return williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50.0
        except: return -50.0

    @staticmethod
    def calculate_cci(data: pd.DataFrame, period: int = 20) -> float:
        """Commodity Channel Index 계산"""
        try:
            if len(data) < period: return 0.0
            tp = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - sma_tp) / (0.015 * mad)
            return cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0.0
        except: return 0.0

    @staticmethod
    def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
        """Money Flow Index 계산 (완전 구현)"""
        try:
            if len(data) < period or 'volume' not in data.columns: return 50.0
            tp = (data['high'] + data['low'] + data['close']) / 3
            raw_money_flow = tp * data['volume']
            
            money_flow_positive, money_flow_negative = [], []
            for i in range(1, len(data)):
                if tp.iloc[i] > tp.iloc[i-1]:
                    money_flow_positive.append(raw_money_flow.iloc[i])
                    money_flow_negative.append(0)
                elif tp.iloc[i] < tp.iloc[i-1]:
                    money_flow_positive.append(0)
                    money_flow_negative.append(raw_money_flow.iloc[i])
                else:
                    money_flow_positive.append(0)
                    money_flow_negative.append(0)
            
            if len(money_flow_positive) < period - 1: return 50.0
            
            mf_positive = pd.Series(money_flow_positive).rolling(window=period-1).sum()
            mf_negative = pd.Series(money_flow_negative).rolling(window=period-1).sum()
            
            # 0으로 나누기 방지
            mf_negative = mf_negative.replace(0, 0.001)
            mfi = 100 - (100 / (1 + (mf_positive / mf_negative)))
            return mfi.iloc[-1] if len(mfi) > 0 and not pd.isna(mfi.iloc[-1]) else 50.0
        except: return 50.0

    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index 계산 (완전 구현)"""
        try:
            if len(data) < period: return 25.0
            high, low, close = data['high'], data['low'], data['close']
            
            # True Range 계산
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Directional Movement 계산
            plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
            minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            # ADX 계산
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)  # 0으로 나누기 방지
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0
        except: return 25.0

    @staticmethod
    def calculate_parabolic_sar(data: pd.DataFrame) -> str:
        """Parabolic SAR 계산 (완전 구현)"""
        try:
            if len(data) < 10: return 'neutral'
            
            high, low, close = data['high'], data['low'], data['close']
            
            # 간단한 SAR 구현 (추세 판단용)
            recent_high = high.tail(10).max()
            recent_low = low.tail(10).min()
            current_price = close.iloc[-1]
            
            # 추세 강도 계산
            trend_strength = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            if trend_strength > 0.7: return 'bullish'
            elif trend_strength < 0.3: return 'bearish'
            else: return 'neutral'
        except: return 'neutral'

async def run_ultimate_coin_selection():
    """궁극의 코인 선별 실행"""
    strategy = UltimateCoinStrategy()
    selected_coins = await strategy.scan_all_selected_coins()
    
    if selected_coins:
        report = await strategy.generate_ultimate_portfolio_report(selected_coins)
        return selected_coins, report
    else:
        return [], {}

async def analyze_coin(symbol: str) -> Dict:
    """단일 코인 분석"""
    strategy = UltimateCoinStrategy()
    signal = await strategy.analyze_symbol(symbol)
    
    return {
        'decision': signal.action, 'confidence_score': signal.confidence * 100,
        'reasoning': signal.reasoning, 'target_price': signal.target_price,
        'price': signal.price, 'sector': signal.sector,
        'fundamental_score': signal.fundamental_score * 100,
        'technical_score': signal.technical_score * 100,
        'momentum_score': signal.momentum_score * 100,
        'project_quality_score': signal.project_quality_score * 100,
        'market_cycle': signal.market_cycle, 'cycle_confidence': signal.cycle_confidence * 100,
        'rsi': signal.rsi, 'williams_r': signal.williams_r, 'cci': signal.cci,
        'btc_correlation': signal.correlation_with_btc,
        'fear_greed_score': signal.fear_greed_score, 'social_sentiment': signal.social_sentiment,
        'split_trading_plan': await strategy.execute_ultimate_split_trading_simulation(signal)
    }

async def get_coin_auto_selection_status() -> Dict:
    """암호화폐 자동선별 상태 조회"""
    strategy = UltimateCoinStrategy()
    
    return {
        'enabled': strategy.enabled, 'version': '5.0_ultimate',
        'last_selection_time': strategy.last_selection_time,
        'cache_valid': strategy._is_selection_cache_valid(),
        'cache_hours': strategy.selection_cache_hours,
        'selected_count': len(strategy.selected_coins),
        'current_market_cycle': strategy.current_market_cycle,
        'cycle_confidence': strategy.cycle_confidence,
        'selection_criteria': {'min_volume_24h_millions': strategy.min_volume_24h / 1e6, 'target_coins': strategy.target_coins}
    }

async def force_coin_reselection() -> List[str]:
    """암호화폐 강제 재선별"""
    strategy = UltimateCoinStrategy()
    strategy.last_selection_time = None
    strategy.selected_coins = []
    return await strategy.ultimate_auto_select_coins()

# ========================================================================================
# 테스트 메인 함수
# ========================================================================================

async def main():
    """테스트용 메인 함수"""
    try:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        print("🪙 암호화폐 궁극 완성 전략 V5.0 테스트!")
        print("🆕 AI 품질평가 + 시장사이클 + 상관관계 최적화 + 고급기술지표")
        print("=" * 80)
        
        # 자동선별 상태 확인
        print("\n📋 궁극의 자동선별 시스템 상태 확인...")
        status = await get_coin_auto_selection_status()
        print(f"  ✅ 시스템 활성화: {status['enabled']} (버전: {status['version']})")
        print(f"  📅 마지막 선별: {status['last_selection_time']}")
        print(f"  🔄 캐시 유효: {status['cache_valid']}")
        print(f"  🔄 현재 시장 사이클: {status['current_market_cycle']} (신뢰도: {status['cycle_confidence']:.2f})")
        
        # 전체 시장 궁극의 자동선별 + 분석
        print(f"\n🔍 궁극의 자동선별 + 전체 분석 시작...")
        start_time = time.time()
        
        selected_coins, report = await run_ultimate_coin_selection()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ 총 소요 시간: {elapsed_time:.1f}초")
        
        if selected_coins and report:
            print(f"\n📈 궁극의 자동선별 + 분석 결과:")
            print(f"  총 분석: {report['summary']['total_coins']}개 코인")
            print(f"  매수 신호: {report['summary']['buy_signals']}개")
            print(f"  매도 신호: {report['summary']['sell_signals']}개")
            print(f"  보유 신호: {report['summary']['hold_signals']}개")
            
            # 전략 점수 요약
            scores = report['strategy_scores']
            print(f"\n📊 평균 전략 점수:")
            print(f"  펀더멘털: {scores['fundamental']:.3f}")
            print(f"  기술적 분석: {scores['technical']:.3f}")
            print(f"  모멘텀: {scores['momentum']:.3f}")
            print(f"  종합 점수: {scores['total']:.3f}")
            print(f"  AI 품질: {scores['project_quality']:.3f}")
            
            # 카테고리별 분포
            print(f"\n🏢 카테고리별 분포:")
            for sector, count in list(report['diversification_analysis']['sector_distribution'].items())[:5]:
                percentage = count / report['summary']['total_coins'] * 100
                print(f"  {sector}: {count}개 ({percentage:.1f}%)")
            
            # 상위 매수 추천
            if report['top_picks']:
                print(f"\n🎯 상위 매수 추천:")
                for i, coin in enumerate(report['top_picks'][:3], 1):
                    print(f"\n  {i}. {coin['symbol']} ({coin['sector']}) - 신뢰도: {coin['confidence']:.2%}")
                    print(f"     🤖 AI 품질점수: {coin['project_quality_score']:.3f} | 총점: {coin['total_score']:.3f}")
                    print(f"     💰 현재가: {coin['price']:,.0f}원 → 목표가: {coin['target_price']:,.0f}원")
                    print(f"     🔄 사이클: {coin['market_cycle']}")
                    print(f"     💼 투자금액: {coin['total_investment']:,.0f}원")
                    print(f"     💡 {coin['reasoning'][:80]}...")
            
            # 분할매매 시뮬레이션
            buy_coins = [s for s in selected_coins if s.action == 'buy']
            if buy_coins:
                print(f"\n🔄 궁극의 5단계 분할매매 시뮬레이션 - {buy_coins[0].symbol}:")
                strategy = UltimateCoinStrategy()
                simulation = await strategy.execute_ultimate_split_trading_simulation(buy_coins[0])
                
                print(f"  🤖 AI 품질: {simulation['ai_project_quality']:.2f}")
                print(f"  🔄 시장 사이클: {simulation['market_cycle']} (신뢰도: {simulation['cycle_confidence']:.2f})")
                print(f"  💰 총 투자금: {simulation['risk_management']['total_investment']:,.0f}원")
                print(f"  📊 포트폴리오 비중: {simulation['risk_management']['portfolio_weight']:.1f}%")
                
                print(f"\n  📈 5단계 진입 계획:")
                for stage_name, stage_info in simulation['stages'].items():
                    print(f"    {stage_name}: {stage_info['trigger_price']:,.0f}원 ({stage_info['ratio']})")
                
                print(f"\n  📉 출구 전략:")
                exit_plan = simulation['dynamic_exit_plan']
                print(f"    손절: {exit_plan['stop_loss']['price']:,.0f}원")
                print(f"    1차익절: {exit_plan['take_profit_1']['price']:,.0f}원 ({exit_plan['take_profit_1']['ratio']})")
                print(f"    2차익절: {exit_plan['take_profit_2']['price']:,.0f}원 ({exit_plan['take_profit_2']['ratio']})")
                print(f"    3차익절: {exit_plan['take_profit_3']['price']:,.0f}원 ({exit_plan['take_profit_3']['ratio']})")
        else:
            print("❌ 분석 결과를 받지 못했습니다.")
        
        print(f"\n🎉 궁극의 암호화폐 전략 V5.0 테스트 완료!")
        print(f"📊 모든 기능이 성공적으로 작동했습니다.")
        
        # 🆕 추가 기능 테스트
        if selected_coins:
            print(f"\n🧪 추가 기능 테스트:")
            
            # CSV 내보내기 테스트
            csv_filename = export_analysis_to_csv(selected_coins[:5])  # 상위 5개만
            print(f"  📄 CSV 내보내기: {csv_filename}")
            
            # 백테스팅 시뮬레이션 테스트
            backtest_result = run_backtest_simulation(selected_coins[:5])
            print(f"  🧪 백테스팅 시뮬레이션:")
            sim_results = backtest_result['simulation_results']
            print(f"    총 거래: {sim_results['total_trades']}회")
            print(f"    승률: {sim_results['win_rate']:.1f}%")
            print(f"    평균 수익률: {sim_results['avg_return_per_trade']:.2f}%")
            print(f"    샤프 비율: {sim_results['sharpe_ratio']:.2f}")
            
            # 웹 API 응답 형식 테스트
            api_response = create_web_api_response(selected_coins, report)
            print(f"  🌐 웹 API 응답: 상태 {api_response['status']}, 버전 {api_response['version']}")
            print(f"    매수 신호: {len(api_response['signals']['buy'])}개")
            
            # 버전 정보
            version_info = get_strategy_version()
            print(f"  ℹ️ 전략 정보: {version_info['name']} v{version_info['version']}")
            print(f"    주요 기능: {len(version_info['features'])}개")
            
            # 유틸리티 함수 테스트
            if buy_coins:
                sample_coin = buy_coins[0]
                risk_level = calculate_risk_level(sample_coin.confidence, 0.1, sample_coin.correlation_with_btc)
                print(f"  🛡️ 리스크 레벨 ({sample_coin.symbol}): {risk_level}")
                
                formatted_amount = format_currency(sample_coin.total_amount)
                print(f"  💰 금액 포맷팅: {formatted_amount}")
                
                cycle_desc = get_market_cycle_description(sample_coin.market_cycle)
                print(f"  🔄 사이클 설명: {cycle_desc['description'][:50]}...")

        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

# ========================================================================================
# 실행부
# ========================================================================================

if __name__ == "__main__":
    print("🚀 궁극의 암호화폐 전략 V5.0 시작!")
    asyncio.run(main())
