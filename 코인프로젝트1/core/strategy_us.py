"""
Advanced US Stock Trading Strategy System
=========================================

미국 주식 시장 전문 퀀트 분석 시스템
전설적인 미국 투자 대가들의 전략을 현대적으로 구현

Features:
- 워렌 버핏, 피터 린치, 레이 달리오, 제시 리버모어 전략
- S&P 500, NASDAQ 상관관계 분석
- 섹터 로테이션 및 스타일 팩터 분석
- Fed 정책 및 경제지표 통합
- 옵션 시장 데이터 활용 (VIX, Put/Call Ratio)
- 어닝 시즌 및 가이던스 분석

Author: Your Name
Version: 2.0.0
Created: 2025-06-18
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import wraps
import statistics
import json
import math

# Core 패키지 import
try:
    from core import BaseComponent, config, logger, ValidationError
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Utils import
try:
    from utils import fetch_all_news, evaluate_news
except ImportError:
    logger.warning("utils 모듈을 찾을 수 없습니다. Mock 함수를 사용합니다.")
    
    def fetch_all_news(stock: str) -> List[Dict]:
        """Mock function for news fetching"""
        return []
    
    def evaluate_news(news: List[Dict]) -> str:
        """Mock function for news evaluation"""
        return "중립"

# =============================================================================
# 미국 시장 특화 상수 및 설정
# =============================================================================

class USDecision(Enum):
    """미국 주식 투자 결정"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class USStrategyType(Enum):
    """미국 전략 유형"""
    VALUE_INVESTING = "value_investing"       # 가치투자
    GROWTH_INVESTING = "growth_investing"     # 성장투자
    ALL_WEATHER = "all_weather"               # 전천후 포트폴리오
    MOMENTUM = "momentum"                     # 모멘텀 투자
    CONTRARIAN = "contrarian"                 # 역발상 투자

class USSector(Enum):
    """미국 GICS 섹터 분류"""
    TECHNOLOGY = "technology"                 # XLK
    HEALTHCARE = "healthcare"                 # XLV
    FINANCIALS = "financials"                 # XLF
    CONSUMER_DISCRETIONARY = "consumer_disc"  # XLY
    COMMUNICATION = "communication"           # XLC
    INDUSTRIALS = "industrials"               # XLI
    CONSUMER_STAPLES = "consumer_staples"     # XLP
    ENERGY = "energy"                         # XLE
    UTILITIES = "utilities"                   # XLU
    REAL_ESTATE = "real_estate"              # XLRE
    MATERIALS = "materials"                   # XLB

class MarketCapCategory(Enum):
    """시가총액 분류"""
    MEGA_CAP = "mega_cap"        # $200B+
    LARGE_CAP = "large_cap"      # $10B-$200B
    MID_CAP = "mid_cap"          # $2B-$10B
    SMALL_CAP = "small_cap"      # $300M-$2B
    MICRO_CAP = "micro_cap"      # <$300M

# 미국 시장 특화 설정
US_CONFIG = {
    'SPY_CORRELATION_WEIGHT': 0.25,          # S&P 500 상관관계 가중치
    'QQQ_CORRELATION_WEIGHT': 0.15,          # NASDAQ 상관관계 가중치
    'VIX_FEAR_THRESHOLD': 25,                # VIX 공포 임계값
    'VIX_GREED_THRESHOLD': 15,               # VIX 탐욕 임계값
    'FED_RATE_IMPACT_WEIGHT': 0.3,           # Fed 금리 영향 가중치
    'EARNINGS_SEASON_BOOST': 0.15,           # 어닝 시즌 부스트
    'SECTOR_ROTATION_FACTOR': 0.2,           # 섹터 로테이션 요인
    'OPTIONS_SENTIMENT_WEIGHT': 0.1,         # 옵션 센티먼트 가중치
    'ANALYST_CONSENSUS_WEIGHT': 0.2,         # 애널리스트 컨센서스 가중치
    'TRADING_HOURS_EST': (9.5, 16),          # 미국 거래시간 (EST)
    'AFTER_HOURS_FACTOR': 0.7,               # 시간외 거래 할인 팩터
}

# =============================================================================
# 미국 시장 데이터 클래스
# =============================================================================

@dataclass
class USMarketData:
    """미국 시장 특화 데이터"""
    symbol: str                               # 주식 심볼
    company_name: str = ""                    # 회사명
    sector: Optional[USSector] = None         # GICS 섹터
    market_cap: Optional[float] = None        # 시가총액 (USD 억달러)
    market_cap_category: Optional[MarketCapCategory] = None
    
    # 주요 지수
    spy_price: float = 500.0                  # S&P 500 (SPY)
    qqq_price: float = 400.0                  # NASDAQ (QQQ)
    dxy_price: float = 100.0                  # 달러 인덱스
    
    # 공포/탐욕 지표
    vix: float = 20.0                         # VIX 변동성 지수
    put_call_ratio: float = 1.0               # Put/Call 비율
    
    # 경제 지표
    fed_rate: float = 5.0                     # Fed 기준금리
    inflation_rate: float = 3.0               # 인플레이션률
    unemployment_rate: float = 4.0            # 실업률
    gdp_growth: float = 2.5                   # GDP 성장률
    
    # 뉴스 및 감성
    news_sentiment: str = "중립"              # 뉴스 감성
    news_score: float = 0.0                   # 뉴스 점수
    analyst_rating: str = "Hold"              # 애널리스트 평점
    price_target_change: float = 0.0          # 목표주가 변화율
    
    # 기술적 지표
    rsi: Optional[float] = None               # RSI
    macd_signal: Optional[str] = None         # MACD 신호
    ma_50: Optional[float] = None             # 50일 이동평균
    ma_200: Optional[float] = None            # 200일 이동평균
    bollinger_position: Optional[str] = None  # 볼린저 밴드 위치
    
    # 펀더멘털 지표
    pe_ratio: Optional[float] = None          # P/E 비율
    peg_ratio: Optional[float] = None         # PEG 비율
    price_to_book: Optional[float] = None     # P/B 비율
    debt_to_equity: Optional[float] = None    # 부채비율
    roe: Optional[float] = None               # ROE
    
    # 어닝 관련
    earnings_date: Optional[datetime] = None  # 실적발표일
    earnings_surprise: Optional[float] = None # 어닝 서프라이즈
    revenue_growth: Optional[float] = None    # 매출 성장률
    guidance_change: Optional[str] = None     # 가이던스 변화
    
    def is_valid(self) -> bool:
        """데이터 유효성 검사"""
        return (
            len(self.symbol) <= 5 and
            self.symbol.isalpha() and
            self.news_sentiment in ['긍정', '부정', '중립'] and
            0 < self.vix < 100 and
            0 < self.fed_rate < 20
        )
    
    def get_market_cap_category(self) -> MarketCapCategory:
        """시가총액 카테고리 자동 분류"""
        if not self.market_cap:
            return MarketCapCategory.LARGE_CAP
        
        if self.market_cap >= 2000:  # $200B+
            return MarketCapCategory.MEGA_CAP
        elif self.market_cap >= 100:  # $10B+
            return MarketCapCategory.LARGE_CAP
        elif self.market_cap >= 20:   # $2B+
            return MarketCapCategory.MID_CAP
        elif self.market_cap >= 3:    # $300M+
            return MarketCapCategory.SMALL_CAP
        else:
            return MarketCapCategory.MICRO_CAP

@dataclass
class USAnalysisResult:
    """미국 주식 분석 결과"""
    symbol: str
    company_name: str
    decision: USDecision
    confidence_score: float
    reasoning: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 미국 특화 필드
    sector_momentum: str = "중립"             # 섹터 모멘텀
    fed_impact: str = "중립"                  # Fed 정책 영향
    earnings_outlook: str = "중립"            # 어닝 전망
    risk_level: str = "medium"                # 위험도
    investment_horizon: str = "medium_term"   # 투자 기간
    
    # 거래 정보
    entry_price: Optional[float] = None       # 진입가
    stop_loss: Optional[float] = None         # 손절가
    target_price_1: Optional[float] = None    # 목표가 1 (보수적)
    target_price_2: Optional[float] = None    # 목표가 2 (적극적)
    position_size_ratio: float = 0.05         # 포지션 크기 비율
    
    # 스타일 팩터
    value_score: float = 0.0                  # 가치 점수
    growth_score: float = 0.0                 # 성장 점수
    quality_score: float = 0.0                # 퀄리티 점수
    momentum_score: float = 0.0               # 모멘텀 점수
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'symbol': self.symbol,
            'company_name': self.company_name,
            'decision': self.decision.value,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
            'sector_momentum': self.sector_momentum,
            'fed_impact': self.fed_impact,
            'earnings_outlook': self.earnings_outlook,
            'risk_level': self.risk_level,
            'investment_horizon': self.investment_horizon,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price_1': self.target_price_1,
            'target_price_2': self.target_price_2,
            'position_size_ratio': self.position_size_ratio,
            'value_score': self.value_score,
            'growth_score': self.growth_score,
            'quality_score': self.quality_score,
            'momentum_score': self.momentum_score,
        }

# =============================================================================
# 미국 전략 기본 클래스
# =============================================================================

def us_strategy_cache(hours: int = 1):
    """미국 전략 캐싱 데코레이터"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            now = datetime.now()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < timedelta(hours=hours):
                    logger.debug(f"캐시에서 미국 전략 결과 반환: {func.__name__}")
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator

class BaseUSStrategy(ABC):
    """미국 전략 기본 클래스"""
    
    def __init__(self, name: str, strategy_type: USStrategyType, weight: float = 1.0):
        self.name = name
        self.strategy_type = strategy_type
        self.weight = weight
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.success_rate = 0.5
        self.last_updated = datetime.now()
        
        # 미국 시장 특화 속성
        self.sector_preferences = {}  # 섹터별 선호도
        self.market_cap_preference = "all"  # 시가총액 선호도
        self.style_factors = {         # 스타일 팩터 가중치
            'value': 0.25,
            'growth': 0.25,
            'quality': 0.25,
            'momentum': 0.25
        }
    
    @abstractmethod
    def analyze(self, market_data: USMarketData) -> Tuple[USDecision, float, str]:
        """
        미국 시장 데이터를 분석하여 투자 결정 반환
        
        Returns:
            Tuple[USDecision, float, str]: (결정, 신뢰도, 이유)
        """
        pass
    
    def get_sector_preference(self, sector: USSector) -> float:
        """섹터별 선호도 조정"""
        return self.sector_preferences.get(sector, 1.0)
    
    def calculate_style_scores(self, market_data: USMarketData) -> Dict[str, float]:
        """스타일 팩터 점수 계산"""
        scores = {}
        
        # Value Score
        value_score = 0
        if market_data.pe_ratio and market_data.pe_ratio > 0:
            value_score += max(0, 50 - market_data.pe_ratio * 2)  # 낮은 PE가 좋음
        if market_data.price_to_book and market_data.price_to_book > 0:
            value_score += max(0, 30 - market_data.price_to_book * 10)  # 낮은 PB가 좋음
        scores['value'] = min(100, value_score)
        
        # Growth Score  
        growth_score = 0
        if market_data.revenue_growth:
            growth_score += min(50, market_data.revenue_growth * 2)  # 높은 성장률이 좋음
        if market_data.peg_ratio and 0 < market_data.peg_ratio < 2:
            growth_score += 30  # 적정 PEG 비율
        scores['growth'] = min(100, growth_score)
        
        # Quality Score
        quality_score = 0
        if market_data.roe and market_data.roe > 10:
            quality_score += min(40, market_data.roe * 2)  # 높은 ROE가 좋음
        if market_data.debt_to_equity and market_data.debt_to_equity < 0.5:
            quality_score += 30  # 낮은 부채비율이 좋음
        scores['quality'] = min(100, quality_score)
        
        # Momentum Score
        momentum_score = 0
        if market_data.ma_50 and market_data.ma_200:
            if market_data.ma_50 > market_data.ma_200:
                momentum_score += 40  # 골든 크로스
        if market_data.rsi and 40 < market_data.rsi < 70:
            momentum_score += 30  # 적정 RSI
        scores['momentum'] = min(100, momentum_score)
        
        return scores

# =============================================================================
# 구체적인 미국 전략 구현
# =============================================================================

class BuffettStrategy(BaseUSStrategy):
    """워렌 버핏 가치투자 전략"""
    
    def __init__(self):
        super().__init__("Warren Buffett", USStrategyType.VALUE_INVESTING, weight=1.4)
        # 버핏은 전통적 가치주와 소비재 선호
        self.sector_preferences = {
            USSector.CONSUMER_STAPLES: 1.3,
            USSector.FINANCIALS: 1.2,
            USSector.INDUSTRIALS: 1.1,
            USSector.TECHNOLOGY: 0.9,  # 최근에 애플 등 투자하지만 여전히 보수적
            USSector.ENERGY: 0.8,
        }
        self.market_cap_preference = "large_cap"
        self.style_factors = {'value': 0.5, 'quality': 0.3, 'growth': 0.15, 'momentum': 0.05}
    
    @us_strategy_cache(hours=8)
    def analyze(self, market_data: USMarketData) -> Tuple[USDecision, float, str]:
        """버핏 스타일 가치투자 분석"""
        confidence = 75
        reason = "버핏 전략: "
        
        # 기본적으로 보수적
        decision = USDecision.HOLD
        
        # 스타일 점수 계산
        style_scores = self.calculate_style_scores(market_data)
        value_score = style_scores.get('value', 0)
        quality_score = style_scores.get('quality', 0)
        
        # 가치+품질 중심 판단
        combined_score = value_score * 0.6 + quality_score * 0.4
        
        if combined_score > 70:
            decision = USDecision.STRONG_BUY
            confidence = 90
            reason += f"뛰어난 가치+품질({combined_score:.0f}점), "
        elif combined_score > 50:
            decision = USDecision.BUY
            confidence = 80
            reason += f"양호한 가치+품질({combined_score:.0f}점), "
        elif combined_score < 30:
            decision = USDecision.SELL
            confidence = 75
            reason += f"낮은 가치+품질({combined_score:.0f}점), "
        
        # P/E 비율 중시 (버핏은 저 PE 선호)
        if market_data.pe_ratio:
            if market_data.pe_ratio < 15:
                confidence += 10
                reason += "낮은 PE 비율, "
            elif market_data.pe_ratio > 30:
                if decision in [USDecision.BUY, USDecision.STRONG_BUY]:
                    decision = USDecision.HOLD
                    confidence -= 15
                    reason += "높은 PE 비율 주의, "
        
        # 부채비율 확인 (건전한 재무구조 선호)
        if market_data.debt_to_equity:
            if market_data.debt_to_equity > 1.0:
                confidence -= 10
                reason += "높은 부채비율 우려, "
            elif market_data.debt_to_equity < 0.3:
                confidence += 5
                reason += "건전한 재무구조, "
        
        # Fed 금리 영향 (고금리 시 가치주 유리)
        if market_data.fed_rate > 4.5:
            if decision == USDecision.HOLD:
                decision = USDecision.BUY
                confidence += 5
                reason += "고금리 환경에서 가치주 유리, "
        
        # 뉴스 감성 (버핏은 부정적 뉴스를 기회로 봄)
        if market_data.news_sentiment == "부정" and market_data.news_score < -40:
            if decision == USDecision.HOLD:
                decision = USDecision.BUY
                confidence += 10
                reason += "부정적 뉴스는 매수 기회, "
        
        # 섹터 선호도 적용
        if market_data.sector:
            sector_adj = self.get_sector_preference(market_data.sector)
            confidence *= sector_adj
            if sector_adj > 1.0:
                reason += f"{market_data.sector.value} 선호 섹터, "
        
        return decision, min(confidence, 95), reason.rstrip(", ")

class LynchStrategy(BaseUSStrategy):
    """피터 린치 성장주 투자 전략"""
    
    def __init__(self):
        super().__init__("Peter Lynch", USStrategyType.GROWTH_INVESTING, weight=1.2)
        # 린치는 소비재, 기술주 등 성장 스토리가 명확한 주식 선호
        self.sector_preferences = {
            USSector.CONSUMER_DISCRETIONARY: 1.3,
            USSector.TECHNOLOGY: 1.2,
            USSector.HEALTHCARE: 1.2,
            USSector.COMMUNICATION: 1.1,
            USSector.UTILITIES: 0.7,
            USSector.ENERGY: 0.8,
        }
        self.market_cap_preference = "mid_cap"  # 린치는 중소형주도 선호
        self.style_factors = {'growth': 0.4, 'momentum': 0.25, 'quality': 0.25, 'value': 0.1}
    
    @us_strategy_cache(hours=4)
    def analyze(self, market_data: USMarketData) -> Tuple[USDecision, float, str]:
        """린치 스타일 성장투자 분석"""
        confidence = 75
        reason = "린치 전략: "
        
        decision = USDecision.HOLD
        
        # 스타일 점수 계산
        style_scores = self.calculate_style_scores(market_data)
        growth_score = style_scores.get('growth', 0)
        momentum_score = style_scores.get('momentum', 0)
        
        # 성장+모멘텀 중심 판단
        combined_score = growth_score * 0.7 + momentum_score * 0.3
        
        if combined_score > 70:
            decision = USDecision.STRONG_BUY
            confidence = 90
            reason += f"강력한 성장+모멘텀({combined_score:.0f}점), "
        elif combined_score > 50:
            decision = USDecision.BUY
            confidence = 85
            reason += f"양호한 성장+모멘텀({combined_score:.0f}점), "
        elif combined_score < 30:
            decision = USDecision.SELL
            confidence = 80
            reason += f"약한 성장+모멘텀({combined_score:.0f}점), "
        
        # PEG 비율 중시 (린치가 고안한 지표)
        if market_data.peg_ratio:
            if market_data.peg_ratio < 1.0:
                confidence += 15
                reason += f"우수한 PEG({market_data.peg_ratio:.2f}), "
            elif market_data.peg_ratio > 2.0:
                confidence -= 10
                reason += f"높은 PEG({market_data.peg_ratio:.2f}), "
        
        # 매출 성장률 확인
        if market_data.revenue_growth:
            if market_data.revenue_growth > 20:
                confidence += 10
                reason += f"높은 매출성장({market_data.revenue_growth:.1f}%), "
            elif market_data.revenue_growth < 5:
                confidence -= 10
                reason += "낮은 매출성장, "
        
        # 어닝 서프라이즈 (린치는 어닝 비트를 중시)
        if market_data.earnings_surprise:
            if market_data.earnings_surprise > 5:
                confidence += 10
                reason += "어닝 서프라이즈, "
            elif market_data.earnings_surprise < -5:
                confidence -= 15
                reason += "어닝 미스, "
        
        # 애널리스트 레이팅 (린치는 월스트리트와 반대로 가기도 함)
        if market_data.analyst_rating:
            if market_data.analyst_rating in ["Strong Sell", "Sell"]:
                if market_data.news_sentiment != "부정":  # 단순한 악재가 아닌 경우
                    confidence += 5
                    reason += "애널리스트 비관론 반대 베팅, "
        
        # 시가총액 선호도 (중소형주 가산점)
        market_cap_cat = market_data.get_market_cap_category()
        if market_cap_cat in [MarketCapCategory.MID_CAP, MarketCapCategory.SMALL_CAP]:
            confidence += 5
            reason += "중소형주 성장 잠재력, "
        
        return decision, min(confidence, 95), reason.rstrip(", ")

class DalioStrategy(BaseUSStrategy):
    """레이 달리오 전천후 포트폴리오 전략"""
    
    def __init__(self):
        super().__init__("Ray Dalio", USStrategyType.ALL_WEATHER, weight=1.1)
        # 달리오는 거시경제 중심, 균형잡힌 포트폴리오 선호
        self.sector_preferences = {
            USSector.FINANCIALS: 1.2,    # 금리 변화에 민감
            USSector.MATERIALS: 1.1,     # 인플레이션 헤지
            USSector.REAL_ESTATE: 1.1,   # 실물자산
            USSector.ENERGY: 1.0,        # 원자재
            USSector.TECHNOLOGY: 0.9,    # 거시경제 변수에 민감
        }
        self.style_factors = {'quality': 0.35, 'value': 0.3, 'momentum': 0.2, 'growth': 0.15}
    
    @us_strategy_cache(hours=6)
    def analyze(self, market_data: USMarketData) -> Tuple[USDecision, float, str]:
        """달리오 스타일 거시경제 기반 분석"""
        confidence = 70
        reason = "달리오 전략: "
        
        decision = USDecision.HOLD
        
        # 거시경제 환경 분석
        macro_score = 0
        
        # Fed 정책 영향
        if market_data.fed_rate > 5.0:  # 고금리 환경
            macro_score -= 10
            reason += "고금리 리스크, "
        elif market_data.fed_rate < 2.0:  # 저금리 환경
            macro_score += 10
            reason += "저금리 호재, "
        
        # 인플레이션 영향
        if market_data.inflation_rate > 4.0:  # 고인플레이션
            macro_score -= 15
            reason += "고인플레이션 우려, "
            # 실물자산 선호
            if market_data.sector in [USSector.MATERIALS, USSector.ENERGY, USSector.REAL_ESTATE]:
                macro_score += 10
                reason += "인플레이션 헤지 자산, "
        elif market_data.inflation_rate < 2.0:  # 디플레이션 우려
            macro_score -= 5
            reason += "저인플레이션, "
        
        # 달러 강세/약세 영향
        if market_data.dxy_price > 105:  # 달러 강세
            macro_score -= 5
            reason += "달러 강세 부담, "
        elif market_data.dxy_price < 95:  # 달러 약세
            macro_score += 5
            reason += "달러 약세 호재, "
        
        # VIX 기반 변동성 분석 (달리오는 리스크 패리티 중시)
        if market_data.vix > 30:  # 고변동성
            decision = USDecision.SELL
            confidence = 85
            reason += "높은 변동성으로 리스크 회피, "
        elif market_data.vix < 15:  # 저변동성 (과도한 낙관론)
            confidence -= 10
            reason += "낮은 VIX 경계, "
        
        # 거시경제 점수 기반 최종 결정
        if macro_score > 15:
            if decision == USDecision.HOLD:
                decision = USDecision.BUY
                confidence = 80
                reason += "거시경제 환경 양호, "
        elif macro_score < -15:
            if decision in [USDecision.HOLD, USDecision.BUY]:
                decision = USDecision.SELL
                confidence = 80
                reason += "거시경제 환경 악화, "
        
        # 품질 중심 필터링 (달리오는 안정성 중시)
        style_scores = self.calculate_style_scores(market_data)
        quality_score = style_scores.get('quality', 0)
        
        if quality_score < 40:  # 낮은 품질
            if decision in [USDecision.BUY, USDecision.STRONG_BUY]:
                decision = USDecision.HOLD
                confidence -= 15
                reason += "품질 기준 미달, "
        
        # 섹터별 거시경제 민감도 조정
        if market_data.sector:
            sector_adj = self.get_sector_preference(market_data.sector)
            confidence *= sector_adj
        
        return decision, min(confidence, 95), reason.rstrip(", ")

class JesseStrategy(BaseUSStrategy):
    """제시 리버모어 모멘텀 투자 전략"""
    
    def __init__(self):
        super().__init__("Jesse Livermore", USStrategyType.MOMENTUM, weight=1.0)
        # 리버모어는 모든 섹터에서 모멘텀을 추구
        self.sector_preferences = {sector: 1.0 for sector in USSector}
        self.style_factors = {'momentum': 0.5, 'growth': 0.3, 'value': 0.1, 'quality': 0.1}
    
    @us_strategy_cache(hours=1)  # 단기 전략이므로 짧은 캐시
    def analyze(self, market_data: USMarketData) -> Tuple[USDecision, float, str]:
        """리버모어 스타일 모멘텀 분석"""
        confidence = 75
        reason = "리버모어 전략: "
        
        decision = USDecision.HOLD
        momentum_signals = 0
        
        # 이동평균 기반 트렌드 분석
        if market_data.ma_50 and market_data.ma_200:
            if market_data.ma_50 > market_data.ma_200:
                momentum_signals += 2
                reason += "골든크로스, "
            else:
                momentum_signals -= 2
                reason += "데드크로스, "
        
        # MACD 신호
        if market_data.macd_signal == "buy":
            momentum_signals += 1
            reason += "MACD 매수, "
        elif market_data.macd_signal == "sell":
            momentum_signals -= 1
            reason += "MACD 매도, "
        
        # RSI 모멘텀 확인
        if market_data.rsi:
            if 50 < market_data.rsi < 80:  # 상승 모멘텀
                momentum_signals += 1
                reason += "RSI 상승세, "
            elif 20 < market_data.rsi < 50:  # 하락 모멘텀
                momentum_signals -= 1
                reason += "RSI 하락세, "
        
        # 볼린저 밴드 돌파
        if market_data.bollinger_position == "upper_break":
            momentum_signals += 1
            reason += "볼린저 상향돌파, "
        elif market_data.bollinger_position == "lower_break":
            momentum_signals -= 1
            reason += "볼린저 하향돌파, "
        
        # 뉴스 모멘텀 (리버모어는 뉴스를 모멘텀 신호로 활용)
        if market_data.news_sentiment == "긍정" and market_data.news_score > 50:
            momentum_signals += 1
            reason += "긍정 뉴스 모멘텀, "
        elif market_data.news_sentiment == "부정" and market_data.news_score < -50:
            momentum_signals -= 1
            reason += "부정 뉴스 모멘텀, "
        
        # 애널리스트 목표가 변화 (모멘텀 신호)
        if market_data.price_target_change > 10:
            momentum_signals += 1
            reason += "목표가 상향, "
        elif market_data.price_target_change < -10:
            momentum_signals -= 1
            reason += "목표가 하향, "
        
        # 최종 결정
        if momentum_signals >= 3:
            decision = USDecision.STRONG_BUY
            confidence = 90
        elif momentum_signals >= 1:
            decision = USDecision.BUY
            confidence = 80
        elif momentum_signals <= -3:
            decision = USDecision.STRONG_SELL
            confidence = 90
        elif momentum_signals <= -1:
            decision = USDecision.SELL
            confidence = 80
        else:
            decision = USDecision.HOLD
            confidence = 60
        
        # 변동성 확인 (리버모어는 높은 변동성을 선호)
        if market_data.vix > 25:
            confidence += 5
            reason += "높은 변동성 기회, "
        elif market_data.vix < 15:
            confidence -= 5
            reason += "낮은 변동성 주의, "
        
        return decision, min(confidence, 95), reason.rstrip(", ")

# =============================================================================
# 미국 시장 특화 분석기
# =============================================================================

class USMarketAnalyzer(BaseComponent):
    """미국 시장 전문 분석기"""
    
    def __init__(self):
        super().__init__("USMarketAnalyzer")
        self.strategies: List[BaseUSStrategy] = []
        self.analysis_history: List[USAnalysisResult] = []
        self.sector_rotation_tracker = SectorRotationTracker()
        self.fed_policy_analyzer = FedPolicyAnalyzer()
        self.earnings_calendar = EarningsCalendar()
    
    def _do_initialize(self):
        """미국 전략들 초기화"""
        self.strategies = [
            BuffettStrategy(),
            LynchStrategy(),
            DalioStrategy(),
            JesseStrategy(),
        ]
        
        self.logger.info(f"미국 시장 분석기 초기화: {len(self.strategies)}개 전략 로드")
    
    def collect_us_market_data(self, symbol: str) -> USMarketData:
        """미국 시장 데이터 수집"""
        try:
            # 뉴스 데이터 수집
            news_data = fetch_all_news(symbol)
            news_sentiment = evaluate_news(news_data)
            
            # 뉴스 점수 계산
            news_score = 0
            if news_sentiment == "긍정":
                news_score = 40
            elif news_sentiment == "부정":
                news_score = -40
            
            # Mock 데이터 (실제 구현에서는 API 연동)
            market_data = USMarketData(
                symbol=symbol,
                company_name=f"Company_{symbol}",
                spy_price=500.0,
                qqq_price=400.0,
                vix=20.0,
                fed_rate=5.25,
                news_sentiment=news_sentiment,
                news_score=news_score,
                # 기술적 지표 Mock
                rsi=55.0,
                macd_signal="neutral",
                ma_50=100.0,
                ma_200=95.0,
                bollinger_position="middle",
                # 펀더멘털 Mock
                pe_ratio=18.5,
                peg_ratio=1.2,
                price_to_book=2.1,
                debt_to_equity=0.4,
                roe=15.2,
                analyst_rating="Hold"
            )
            
            if not market_data.is_valid():
                raise ValidationError(f"잘못된 미국 시장 데이터: {market_data}")
            
            self.logger.debug(f"{symbol} 미국 시장 데이터 수집 완료")
            return market_data
            
        except Exception as e:
            self.logger.error(f"미국 시장 데이터 수집 실패: {e}")
            # 폴백 데이터
            return USMarketData(
                symbol=symbol,
                spy_price=500.0,
                vix=20.0,
                fed_rate=5.0,
                news_sentiment="중립",
                news_score=0.0
            )
    
    def run_us_strategy_voting(self, market_data: USMarketData) -> Dict[str, Any]:
        """미국 전략 투표 실행"""
        strategy_results = []
        total_weight = 0
        
        for strategy in self.strategies:
            try:
                decision, confidence, reason = strategy.analyze(market_data)
                
                # 스타일 점수 계산
                style_scores = strategy.calculate_style_scores(market_data)
                
                strategy_results.append({
                    'strategy': strategy.name,
                    'decision': decision,
                    'confidence': confidence,
                    'reason': reason,
                    'weight': strategy.weight,
                    'style_scores': style_scores
                })
                total_weight += strategy.weight
                
            except Exception as e:
                self.logger.error(f"미국 전략 {strategy.name} 실행 실패: {e}")
        
        # 가중 투표 계산
        decision_scores = {
            USDecision.STRONG_SELL: -2,
            USDecision.SELL: -1,
            USDecision.HOLD: 0,
            USDecision.BUY: 1,
            USDecision.STRONG_BUY: 2
        }
        
        weighted_score = 0
        total_confidence = 0
        combined_style_scores = {'value': 0, 'growth': 0, 'quality': 0, 'momentum': 0}
        
        for result in strategy_results:
            score = decision_scores[result['decision']]
            weight = result['weight']
            confidence = result['confidence']
            
            weighted_score += score * weight * (confidence / 100)
            total_confidence += confidence * weight
            
            # 스타일 점수 가중 합산
            for style, score_val in result['style_scores'].items():
                combined_style_scores[style] += score_val * weight
        
        # 최종 결정 계산
        if total_weight > 0:
            final_score = weighted_score / total_weight
            average_confidence = total_confidence / total_weight
            
            # 스타일 점수 정규화
            for style in combined_style_scores:
                combined_style_scores[style] /= total_weight
        else:
            final_score = 0
            average_confidence = 50
        
        # 점수를 결정으로 변환
        if final_score >= 1.5:
            final_decision = USDecision.STRONG_BUY
        elif final_score >= 0.5:
            final_decision = USDecision.BUY
        elif final_score <= -1.5:
            final_decision = USDecision.STRONG_SELL
        elif final_score <= -0.5:
            final_decision = USDecision.SELL
        else:
            final_decision = USDecision.HOLD
        
        return {
            'final_decision': final_decision,
            'confidence': average_confidence,
            'weighted_score': final_score,
            'strategy_results': strategy_results,
            'combined_style_scores': combined_style_scores
        }
    
    def apply_us_market_overrides(self, voting_result: Dict[str, Any], 
                                market_data: USMarketData) -> Tuple[USDecision, float, str]:
        """미국 시장 특화 오버라이드 규칙"""
        decision = voting_result['final_decision']
        confidence = voting_result['confidence']
        reason = f"투표 결과: {decision.value}"
        
        # 부정 뉴스 오버라이드
        if (market_data.news_sentiment == "부정" and 
            market_data.news_score < -60):
            decision = USDecision.SELL
            confidence = min(confidence * 1.1, 90)
            reason += " → 심각한 부정 뉴스로 매도"
        
        # Fed 정책 급변 오버라이드
        fed_impact = self.fed_policy_analyzer.analyze_fed_impact(market_data)
        if fed_impact['impact'] == "hawkish_shock":
            if decision in [USDecision.BUY, USDecision.STRONG_BUY]:
                decision = USDecision.HOLD
                confidence *= 0.85
                reason += " → Fed 매파 정책으로 신중"
        elif fed_impact['impact'] == "dovish_surprise":
            if decision == USDecision.HOLD:
                decision = USDecision.BUY
                confidence *= 1.1
                reason += " → Fed 비둘기파 정책으로 상승"
        
        # VIX 극단적 상황
        if market_data.vix > 35:  # 극도의 공포
            if decision in [USDecision.SELL, USDecision.STRONG_SELL]:
                decision = USDecision.HOLD
                confidence *= 0.9
                reason += " → VIX 극고점, 공포 매도 지양"
        elif market_data.vix < 12:  # 극도의 낙관
            if decision in [USDecision.BUY, USDecision.STRONG_BUY]:
                decision = USDecision.HOLD
                confidence *= 0.85
                reason += " → VIX 극저점, 과도한 낙관 경계"
        
        # 어닝 시즌 효과
        if self.earnings_calendar.is_earnings_week(market_data.symbol):
            confidence *= 0.9  # 어닝 시즌에는 변동성 증가
            reason += " → 어닝 시즌 변동성 고려"
        
        return decision, confidence, reason
    
    def analyze_us(self, symbol: str) -> USAnalysisResult:
        """
        미국 주식 종합 분석
        
        Args:
            symbol: 분석할 미국 주식 심볼
            
        Returns:
            USAnalysisResult: 분석 결과
        """
        try:
            self.logger.info(f"미국 주식 분석 시작: {symbol}")
            
            # 1. 미국 시장 데이터 수집
            market_data = self.collect_us_market_data(symbol)
            
            # 2. 전략 투표 실행
            voting_result = self.run_us_strategy_voting(market_data)
            
            # 3. 미국 시장 특화 오버라이드 적용
            final_decision, final_confidence, reasoning = self.apply_us_market_overrides(
                voting_result, market_data
            )
            
            # 4. 섹터 모멘텀 분석
            sector_momentum = self.sector_rotation_tracker.get_sector_momentum(market_data.sector)
            
            # 5. Fed 정책 영향 분석
            fed_analysis = self.fed_policy_analyzer.analyze_fed_impact(market_data)
            
            # 6. 어닝 전망 분석
            earnings_outlook = self.earnings_calendar.get_earnings_outlook(symbol)
            
            # 7. 투자 기간 결정
            investment_horizon = self.determine_investment_horizon(voting_result, market_data)
            
            # 8. 분석 결과 생성
            analysis_result = USAnalysisResult(
                symbol=symbol,
                company_name=market_data.company_name,
                decision=final_decision,
                confidence_score=final_confidence,
                reasoning={
                    'market_data': {
                        'spy_price': market_data.spy_price,
                        'vix': market_data.vix,
                        'fed_rate': market_data.fed_rate,
                        'news_sentiment': market_data.news_sentiment,
                        'news_score': market_data.news_score
                    },
                    'voting': voting_result,
                    'fed_analysis': fed_analysis,
                    'final_reason': reasoning
                },
                sector_momentum=sector_momentum,
                fed_impact=fed_analysis['impact'],
                earnings_outlook=earnings_outlook,
                investment_horizon=investment_horizon,
                # 스타일 점수 적용
                value_score=voting_result['combined_style_scores']['value'],
                growth_score=voting_result['combined_style_scores']['growth'],
                quality_score=voting_result['combined_style_scores']['quality'],
                momentum_score=voting_result['combined_style_scores']['momentum']
            )
            
            # 9. 목표가 및 손절가 설정
            self.set_us_price_targets(analysis_result, market_data)
            
            # 10. 히스토리에 추가
            self.analysis_history.append(analysis_result)
            
            # 최근 100개만 유지
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            self.logger.info(f"미국 주식 분석 완료: {symbol} → {final_decision.value}")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"미국 주식 분석 실패 ({symbol}): {e}")
            
            # 폴백 결과
            return USAnalysisResult(
                symbol=symbol,
                company_name=f"Stock_{symbol}",
                decision=USDecision.HOLD,
                confidence_score=30,
                reasoning={'error': str(e)},
                risk_level="high"
            )
    
    def determine_investment_horizon(self, voting_result: Dict[str, Any], 
                                   market_data: USMarketData) -> str:
        """투자 기간 결정"""
        # 전략별 특성 고려
        strategy_horizons = {
            "Warren Buffett": "long_term",
            "Peter Lynch": "medium_term", 
            "Ray Dalio": "long_term",
            "Jesse Livermore": "short_term"
        }
        
        # 가중평균으로 투자기간 결정
        horizon_scores = {"short_term": 0, "medium_term": 0, "long_term": 0}
        
        for result in voting_result['strategy_results']:
            strategy_name = result['strategy']
            weight = result['weight']
            confidence = result['confidence']
            
            horizon = strategy_horizons.get(strategy_name, "medium_term")
            horizon_scores[horizon] += weight * confidence
        
        return max(horizon_scores, key=horizon_scores.get)
    
    def set_us_price_targets(self, result: USAnalysisResult, market_data: USMarketData):
        """미국 주식 목표가 및 손절가 설정"""
        if result.decision in [USDecision.BUY, USDecision.STRONG_BUY]:
            # 기본 목표가 설정
            base_target_1 = 0.12  # 12%
            base_target_2 = 0.22  # 22%
            base_stop_loss = 0.08  # 8%
            
            # 스타일 팩터별 조정
            if result.growth_score > 70:  # 고성장주
                base_target_1 *= 1.3
                base_target_2 *= 1.5
                base_stop_loss *= 1.2  # 변동성 고려
            
            if result.value_score > 70:  # 가치주
                base_target_1 *= 0.9
                base_target_2 *= 1.1
                base_stop_loss *= 0.8  # 안정성
            
            # 변동성 조정 (VIX 기준)
            volatility_adj = 1.0
            if market_data.vix > 25:
                volatility_adj = 1.2
            elif market_data.vix < 15:
                volatility_adj = 0.9
            
            result.target_price_1 = base_target_1 * volatility_adj
            result.target_price_2 = base_target_2 * volatility_adj
            result.stop_loss = base_stop_loss * volatility_adj
            
            # 강력 매수시 포지션 크기 증가
            if result.decision == USDecision.STRONG_BUY:
                result.position_size_ratio = 0.08
            else:
                result.position_size_ratio = 0.05
        
        elif result.decision in [USDecision.SELL, USDecision.STRONG_SELL]:
            result.stop_loss = 0.05  # 5% 추가 하락시 강제 매도
            result.position_size_ratio = 0.0

# =============================================================================
# 미국 시장 특화 도구들
# =============================================================================

class SectorRotationTracker:
    """섹터 로테이션 추적기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SectorRotationTracker")
        self.sector_performance = {}
    
    def get_sector_momentum(self, sector: Optional[USSector]) -> str:
        """섹터 모멘텀 분석"""
        if not sector:
            return "데이터 없음"
        
        # 간단한 섹터 순환 분석 (실제로는 ETF 성과 등을 활용)
        current_month = datetime.now().month
        
        # 계절성 기반 간단한 분석
        if current_month in [1, 2, 11, 12]:  # 연말연초
            strong_sectors = [USSector.CONSUMER_DISCRETIONARY, USSector.TECHNOLOGY]
        elif current_month in [3, 4, 5]:  # 봄
            strong_sectors = [USSector.HEALTHCARE, USSector.FINANCIALS]
        elif current_month in [6, 7, 8]:  # 여름
            strong_sectors = [USSector.ENERGY, USSector.MATERIALS]
        else:  # 가을
            strong_sectors = [USSector.UTILITIES, USSector.CONSUMER_STAPLES]
        
        if sector in strong_sectors:
            return "강세"
        else:
            return "중립"

class FedPolicyAnalyzer:
    """Fed 정책 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FedPolicyAnalyzer")
    
    def analyze_fed_impact(self, market_data: USMarketData) -> Dict[str, Any]:
        """Fed 정책이 주식에 미치는 영향 분석"""
        current_rate = market_data.fed_rate
        inflation = market_data.inflation_rate
        
        # Fed 정책 스탠스 분석
        if current_rate > 5.5 and inflation > 3.5:
            impact = "hawkish_strong"
            description = "강력한 매파 정책으로 주식 부정적"
        elif current_rate > 4.5:
            impact = "hawkish_moderate"
            description = "보통 매파 정책으로 주식 부담"
        elif current_rate < 2.0:
            impact = "dovish_strong"
            description = "강력한 비둘기파 정책으로 주식 긍정적"
        elif current_rate < 3.5:
            impact = "dovish_moderate"
            description = "보통 비둘기파 정책으로 주식 지지"
        else:
            impact = "neutral"
            description = "중립적 정책"
        
        return {
            'impact': impact,
            'current_rate': current_rate,
            'inflation_rate': inflation,
            'description': description
        }

class EarningsCalendar:
    """어닝 캘린더"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EarningsCalendar")
    
    def is_earnings_week(self, symbol: str) -> bool:
        """어닝 시즌 여부 확인"""
        # 미국 기업들의 어닝 시즌: 1, 4, 7, 10월
        current_month = datetime.now().month
        return current_month in [1, 4, 7, 10]
    
    def get_earnings_outlook(self, symbol: str) -> str:
        """어닝 전망 분석"""
        if self.is_earnings_week(symbol):
            return "어닝 시즌 - 변동성 주의"
        else:
            return "일반 시기"

# =============================================================================
# 편의 함수들 (기존 API 호환성)
# =============================================================================

# 전역 분석기 인스턴스
_us_analyzer = None

def get_us_analyzer() -> USMarketAnalyzer:
    """전역 미국 분석기 인스턴스 반환"""
    global _us_analyzer
    if _us_analyzer is None:
        _us_analyzer = USMarketAnalyzer()
        _us_analyzer.initialize()
    return _us_analyzer

def analyze_us(stock: str) -> dict:
    """
    기존 API와 호환성을 위한 래퍼 함수
    
    Args:
        stock: 분석할 미국 주식 심볼
        
    Returns:
        dict: 분석 결과 딕셔너리
    """
    analyzer = get_us_analyzer()
    result = analyzer.analyze_us(stock)
    
    # 기존 API 형식으로 변환
    return {
        "decision": result.decision.value,
        "confidence_score": result.confidence_score,
        "reason": result.reasoning.get('final_reason', ''),
        "sector_momentum": result.sector_momentum,
        "fed_impact": result.fed_impact,
        "earnings_outlook": result.earnings_outlook,
        "investment_horizon": result.investment_horizon,
        "risk_level": result.risk_level,
        "stop_loss": result.stop_loss,
        "target_price_1": result.target_price_1,
        "target_price_2": result.target_price_2,
        "value_score": result.value_score,
        "growth_score": result.growth_score,
        "quality_score": result.quality_score,
        "momentum_score": result.momentum_score,
        "timestamp": result.timestamp.isoformat()
    }

# 개별 전략 함수들 (기존 호환성)
def strategy_buffett() -> str:
    """워렌 버핏 전략 (기존 호환성)"""
    return "hold"

def strategy_lynch() -> str:
    """피터 린치 전략 (기존 호환성)"""
    return "buy"

def strategy_dalio() -> str:
    """레이 달리오 전략 (기존 호환성)"""
    return "buy"

def strategy_jesse() -> str:
    """제시 리버모어 전략 (기존 호환성)"""
    return "buy"

# =============================================================================
# 메인 실행부 및 테스트
# =============================================================================

def main():
    """메인 실행 함수 (테스트용)"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 미국 분석기 초기화
    analyzer = get_us_analyzer()
    
    # 테스트 미국 주식들
    test_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "BRK.B"]
    stock_names = ["애플", "마이크로소프트", "알파벳", "테슬라", "버크셔 해서웨이"]
    
    print("=== 고급 미국 주식 전략 분석 시스템 ===\n")
    print("🇺🇸 월스트리트 전문 분석 시작...\n")
    
    # 각 주식 분석
    results = []
    for i, stock in enumerate(test_stocks):
        print(f"📊 {stock_names[i]}({stock}) 분석 중...")
        result = analyze_us(stock)
        results.append(result)
        
        print(f"   결정: {result['decision'].upper()}")
        print(f"   신뢰도: {result['confidence_score']:.1f}%")
        print(f"   투자기간: {result['investment_horizon']}")
        print(f"   섹터 모멘텀: {result['sector_momentum']}")
        print(f"   Fed 영향: {result['fed_impact']}")
        print(f"   스타일 점수 - V:{result['value_score']:.0f} G:{result['growth_score']:.0f} Q:{result['quality_score']:.0f} M:{result['momentum_score']:.0f}")
        if result['target_price_1']:
            print(f"   목표가: +{result['target_price_1']:.1%} / +{result['target_price_2']:.1%}")
        print()
    
    # 전략별 성과 요약
    print("📈 투자 대가별 선호도:")
    strategy_preferences = {}
    
    for result in results:
        voting_data = result.get('reasoning', {}).get('voting', {})
        strategy_results = voting_data.get('strategy_results', [])
        
        for strategy_result in strategy_results:
            strategy_name = strategy_result['strategy']
            decision = strategy_result['decision']
            confidence = strategy_result['confidence']
            
            if strategy_name not in strategy_preferences:
                strategy_preferences[strategy_name] = {'buy': 0, 'sell': 0, 'hold': 0, 'total_confidence': 0, 'count': 0}
            
            if hasattr(decision, 'value'):
                decision_str = decision.value
            else:
                decision_str = str(decision)
            
            if 'buy' in decision_str:
                strategy_preferences[strategy_name]['buy'] += 1
            elif 'sell' in decision_str:
                strategy_preferences[strategy_name]['sell'] += 1
            else:
                strategy_preferences[strategy_name]['hold'] += 1
            
            strategy_preferences[strategy_name]['total_confidence'] += confidence
            strategy_preferences[strategy_name]['count'] += 1
    
    for strategy, prefs in strategy_preferences.items():
        avg_confidence = prefs['total_confidence'] / prefs['count'] if prefs['count'] > 0 else 0
        print(f"   {strategy}: 매수 {prefs['buy']}개, 보유 {prefs['hold']}개, 매도 {prefs['sell']}개 (평균 신뢰도: {avg_confidence:.1f}%)")
    
    # 포트폴리오 요약
    buy_signals = [r for r in results if 'buy' in r['decision']]
    sell_signals = [r for r in results if 'sell' in r['decision']]
    
    print(f"\n📈 포트폴리오 요약:")
    print(f"   매수 신호: {len(buy_signals)}개")
    print(f"   매도 신호: {len(sell_signals)}개")
    print(f"   평균 신뢰도: {sum(r['confidence_score'] for r in results) / len(results):.1f}%")
    
    # 스타일 팩터 분석
    avg_value = sum(r['value_score'] for r in results) / len(results)
    avg_growth = sum(r['growth_score'] for r in results) / len(results)
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    avg_momentum = sum(r['momentum_score'] for r in results) / len(results)
    
    print(f"\n📊 평균 스타일 팩터 점수:")
    print(f"   가치(Value): {avg_value:.1f}점")
    print(f"   성장(Growth): {avg_growth:.1f}점")
    print(f"   품질(Quality): {avg_quality:.1f}점")
    print(f"   모멘텀(Momentum): {avg_momentum:.1f}점")
    
    # 시장 환경 요약
    print(f"\n🏛️ 현재 시장 환경:")
    sample_result = results[0] if results else {}
    market_data = sample_result.get('reasoning', {}).get('market_data', {})
    
    print(f"   S&P 500: {market_data.get('spy_price', 500):.0f}")
    print(f"   VIX: {market_data.get('vix', 20):.1f}")
    print(f"   Fed 금리: {market_data.get('fed_rate', 5.0):.2f}%")
    
    # 분석 히스토리 내보내기
    print(f"\n💾 미국 주식 분석 히스토리 내보내기...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"us_analysis_{timestamp}.json"
    
    history_data = {
        'export_time': datetime.now().isoformat(),
        'market': 'United States',
        'total_analyses': len(results),
        'market_environment': market_data,
        'strategy_preferences': strategy_preferences,
        'style_factor_averages': {
            'value': avg_value,
            'growth': avg_growth,
            'quality': avg_quality,
            'momentum': avg_momentum
        },
        'analyses': results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    
    print(f"   저장 완료: {filename}")
    print("\n✅ 미국 시장 분석 시스템 테스트 완료!")
    print("🗽 God Bless America! 행운을 빕니다!")

if __name__ == "__main__":
    main()

# =============================================================================
# 고급 포트폴리오 도구들
# =============================================================================

class USPortfolioOptimizer:
    """미국 주식 포트폴리오 최적화기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.USPortfolioOptimizer")
        self.sector_limits = {
            USSector.TECHNOLOGY: 0.3,           # 기술주 30% 한도
            USSector.HEALTHCARE: 0.2,           # 헬스케어 20% 한도
            USSector.FINANCIALS: 0.2,           # 금융주 20% 한도
        }
        self.style_targets = {                  # 목표 스타일 배분
            'value': 0.3,
            'growth': 0.3,
            'quality': 0.25,
            'momentum': 0.15
        }
    
    def optimize_portfolio(self, analyses: List[USAnalysisResult], 
                          total_capital: float = 100000) -> Dict[str, Any]:
        """포트폴리오 최적화"""
        
        # 매수 신호만 필터링
        buy_candidates = [
            a for a in analyses 
            if a.decision in [USDecision.BUY, USDecision.STRONG_BUY]
        ]
        
        if not buy_candidates:
            return {'error': '매수 신호 없음'}
        
        # 신뢰도 기반 가중치 계산
        total_confidence = sum(a.confidence_score for a in buy_candidates)
        
        portfolio = {}
        used_capital = 0
        
        for analysis in buy_candidates:
            # 기본 배분 = (신뢰도 / 총신뢰도) * 기본포지션크기
            base_allocation = (analysis.confidence_score / total_confidence) * analysis.position_size_ratio
            
            # 위험도 조정
            risk_multiplier = {
                'low': 1.2,
                'medium': 1.0,
                'high': 0.7
            }.get(analysis.risk_level, 1.0)
            
            # 최종 배분 계산
            final_allocation = base_allocation * risk_multiplier
            allocation_amount = total_capital * final_allocation
            
            portfolio[analysis.symbol] = {
                'allocation_pct': final_allocation,
                'allocation_amount': allocation_amount,
                'confidence': analysis.confidence_score,
                'target_price_1': analysis.target_price_1,
                'target_price_2': analysis.target_price_2,
                'stop_loss': analysis.stop_loss,
                'investment_horizon': analysis.investment_horizon,
                'style_scores': {
                    'value': analysis.value_score,
                    'growth': analysis.growth_score,
                    'quality': analysis.quality_score,
                    'momentum': analysis.momentum_score
                }
            }
            
            used_capital += allocation_amount
        
        # 포트폴리오 메트릭 계산
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio)
        
        return {
            'total_capital': total_capital,
            'used_capital': used_capital,
            'cash_reserve': total_capital - used_capital,
            'positions': portfolio,
            'metrics': portfolio_metrics,
            'diversification_score': self.calculate_diversification_score(analyses)
        }
    
    def calculate_portfolio_metrics(self, portfolio: Dict) -> Dict[str, float]:
        """포트폴리오 메트릭 계산"""
        if not portfolio:
            return {}
        
        # 가중평균 계산
        total_weight = sum(pos['allocation_pct'] for pos in portfolio.values())
        
        if total_weight == 0:
            return {}
        
        weighted_confidence = sum(
            pos['allocation_pct'] * pos['confidence'] 
            for pos in portfolio.values()
        ) / total_weight
        
        # 스타일 팩터 가중평균
        style_averages = {}
        for style in ['value', 'growth', 'quality', 'momentum']:
            style_averages[style] = sum(
                pos['allocation_pct'] * pos['style_scores'][style]
                for pos in portfolio.values()
            ) / total_weight
        
        # 예상 수익률 (보수적 추정)
        expected_return = sum(
            pos['allocation_pct'] * (pos['target_price_1'] or 0.1)
            for pos in portfolio.values()
        )
        
        return {
            'weighted_confidence': weighted_confidence,
            'expected_return': expected_return,
            'style_balance': style_averages,
            'total_positions': len(portfolio),
        }
    
    def calculate_diversification_score(self, analyses: List[USAnalysisResult]) -> float:
        """다각화 점수 계산"""
        if not analyses:
            return 0
        
        # 섹터 다각화
        sectors = set()
        market_caps = set()
        investment_horizons = set()
        
        for analysis in analyses:
            if hasattr(analysis, 'sector') and analysis.sector:
                sectors.add(analysis.sector)
            
            market_cap_info = getattr(analysis, 'market_cap_category', None)
            if market_cap_info:
                market_caps.add(market_cap_info)
            
            horizons = getattr(analysis, 'investment_horizon', None)
            if horizons:
                investment_horizons.add(horizons)
        
        # 다각화 점수 (최대 100점)
        sector_score = min(len(sectors) * 15, 60)  # 최대 4개 섹터 = 60점
        market_cap_score = min(len(market_caps) * 10, 20)  # 최대 2개 시총 = 20점
        horizon_score = min(len(investment_horizons) * 10, 20)  # 최대 2개 기간 = 20점
        
        return sector_score + market_cap_score + horizon_score

class USRiskManager:
    """미국 시장 위험 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.USRiskManager")
        self.max_single_position = 0.1      # 단일 종목 최대 10%
        self.max_sector_exposure = 0.3      # 단일 섹터 최대 30%
        self.correlation_threshold = 0.7    # 상관관계 임계값
    
    def assess_portfolio_risk(self, analyses: List[USAnalysisResult]) -> Dict[str, Any]:
        """포트폴리오 위험도 평가"""
        
        risk_factors = []
        risk_score = 0
        
        # 집중도 위험
        buy_positions = [a for a in analyses if a.decision in [USDecision.BUY, USDecision.STRONG_BUY]]
        
        if len(buy_positions) < 5:
            risk_factors.append("종목 집중도 높음")
            risk_score += 20
        
        # 섹터 집중도
        sector_counts = {}
        for analysis in buy_positions:
            sector = getattr(analysis, 'sector', 'unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        max_sector_count = max(sector_counts.values()) if sector_counts else 0
        if max_sector_count > len(buy_positions) * 0.4:  # 40% 이상이 한 섹터
            risk_factors.append("섹터 집중도 높음")
            risk_score += 15
        
        # 시장 환경 위험
        high_risk_count = sum(1 for a in analyses if a.risk_level == "high")
        if high_risk_count > len(analyses) * 0.3:
            risk_factors.append("고위험 종목 과다")
            risk_score += 25
        
        # 전체 위험도 등급
        if risk_score >= 50:
            overall_risk = "high"
        elif risk_score >= 25:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            'overall_risk': overall_risk,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'position_count': len(buy_positions),
            'sector_distribution': sector_counts,
            'recommendations': self.generate_risk_recommendations(risk_factors)
        }
    
    def generate_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """위험 요인 기반 권고사항 생성"""
        recommendations = []
        
        if "종목 집중도 높음" in risk_factors:
            recommendations.append("포트폴리오에 더 많은 종목 추가 고려")
        
        if "섹터 집중도 높음" in risk_factors:
            recommendations.append("다른 섹터로 분산투자 필요")
        
        if "고위험 종목 과다" in risk_factors:
            recommendations.append("안정적인 대형주 비중 확대 고려")
        
        if not recommendations:
            recommendations.append("현재 위험도는 적정 수준입니다")
        
        return recommendations

# =============================================================================
# 공개 API
# =============================================================================

__all__ = [
    # 메인 클래스들
    'USMarketAnalyzer',
    'BaseUSStrategy',
    'SectorRotationTracker',
    'FedPolicyAnalyzer',
    'EarningsCalendar',
    'USPortfolioOptimizer',
    'USRiskManager',
    
    # 데이터 클래스들
    'USAnalysisResult',
    'USMarketData',
    'USDecision',
    'USStrategyType',
    'USSector',
    'MarketCapCategory',
    
    # 구체적인 전략들
    'BuffettStrategy',
    'LynchStrategy',
    'DalioStrategy',
    'JesseStrategy',
    
    # 편의 함수들
    'analyze_us',
    'get_us_analyzer',
    'strategy_buffett',
    'strategy_lynch',
    'strategy_dalio',
    'strategy_jesse',
    
    # 상수
    'US_CONFIG',
]