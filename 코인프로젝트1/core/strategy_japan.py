"""
Advanced Japanese Stock Trading Strategy System
==============================================

일본 주식 시장 전문 퀀트 분석 시스템
일본의 투자 대가들과 전통적인 기술적 분석 기법을 현대적으로 구현

Features:
- 일본 전통 기술적 분석 (혼마 촛대, 일목균형표)
- 현대 일본 투자 대가 전략 (BNF, CIS)
- 일본 시장 특성 반영 (시가총액, 업종별 분석)
- 엔화 강세/약세 영향 분석
- 일본 경제지표 통합 분석

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
# 일본 시장 특화 상수 및 설정
# =============================================================================

class JapaneseDecision(Enum):
    """일본 주식 투자 결정"""
    TSUYOKU_KAIMASU = "tsuyoku_kaimasu"    # 강력 매수 (強く買います)
    KAIMASU = "kaimasu"                    # 매수 (買います)
    MOCHIMASU = "mochimasu"                # 보유 (持ちます)
    URIMASU = "urimasu"                    # 매도 (売ります)
    TSUYOKU_URIMASU = "tsuyoku_urimasu"    # 강력 매도 (強く売ります)
    
    # 영어 매핑
    @property
    def english(self) -> str:
        mapping = {
            "tsuyoku_kaimasu": "strong_buy",
            "kaimasu": "buy", 
            "mochimasu": "hold",
            "urimasu": "sell",
            "tsuyoku_urimasu": "strong_sell"
        }
        return mapping[self.value]

class JapaneseStrategyType(Enum):
    """일본 전략 유형"""
    HONMA_CANDLE = "honma_candle"         # 혼마 촛대 분석
    ICHIMOKU = "ichimoku"                 # 일목균형표
    MODERN_SWING = "modern_swing"         # 현대 스윙 트레이딩
    SCALPING = "scalping"                 # 스캘핑
    NEWS_MOMENTUM = "news_momentum"       # 뉴스 모멘텀

class JapaneseSector(Enum):
    """일본 주요 업종"""
    TECHNOLOGY = "technology"             # 기술주
    AUTOMOTIVE = "automotive"             # 자동차
    FINANCE = "finance"                   # 금융
    RETAIL = "retail"                     # 소매
    MANUFACTURING = "manufacturing"       # 제조업
    REAL_ESTATE = "real_estate"          # 부동산
    UTILITIES = "utilities"               # 유틸리티
    HEALTHCARE = "healthcare"             # 헬스케어

# 일본 시장 특화 설정
JAPAN_CONFIG = {
    'NIKKEI_CORRELATION_WEIGHT': 0.3,     # 니케이지수 상관관계 가중치
    'USDJPY_SENSITIVITY': 0.2,            # 달러엔 환율 민감도
    'SECTOR_ROTATION_FACTOR': 0.15,       # 업종순환 요인
    'NEWS_IMPACT_MULTIPLIER': 1.2,        # 일본 뉴스 임팩트 배수
    'EARNINGS_SEASON_BOOST': 0.1,         # 실적발표 시즌 부스트
    'BOJ_POLICY_WEIGHT': 0.25,            # 일본은행 정책 가중치
    'TRADING_HOURS_JST': (9, 15),         # 일본 거래시간 (JST)
    'LUNCH_BREAK': (11.5, 12.5),          # 점심시간
}

# =============================================================================
# 일본 시장 데이터 클래스
# =============================================================================

@dataclass
class JapaneseMarketData:
    """일본 시장 특화 데이터"""
    stock_code: str                        # 주식 코드 (4자리)
    company_name: str = ""                 # 회사명
    sector: Optional[JapaneseSector] = None # 업종
    market_cap: Optional[float] = None     # 시가총액 (억엔)
    
    # 시장 지표
    nikkei_225: float = 0.0               # 니케이225 지수
    topix: float = 0.0                    # TOPIX 지수
    mothers_index: float = 0.0            # 마더스 지수
    
    # 환율 및 경제지표
    usdjpy_rate: float = 150.0            # 달러엔 환율
    jgb_10y_yield: float = 0.5            # 일본 10년 국채 수익률
    boj_rate: float = -0.1                # 일본은행 기준금리
    
    # 뉴스 및 감성
    news_sentiment: str = "중립"           # 뉴스 감성
    news_score: float = 0.0               # 뉴스 점수
    
    # 기술적 지표
    rsi: Optional[float] = None           # RSI
    macd_signal: Optional[str] = None     # MACD 신호
    bollinger_position: Optional[str] = None # 볼린저 밴드 위치
    
    # 일목균형표 지표
    ichimoku_kumo_position: Optional[str] = None    # 구름 위치
    ichimoku_tenkan_kijun: Optional[str] = None     # 전환선-기준선 관계
    ichimoku_chikou_span: Optional[str] = None      # 지연선 상태
    
    def is_valid(self) -> bool:
        """데이터 유효성 검사"""
        return (
            len(self.stock_code) == 4 and
            self.stock_code.isdigit() and
            self.news_sentiment in ['긍정', '부정', '중립'] and
            100 < self.usdjpy_rate < 200  # 현실적인 환율 범위
        )

@dataclass
class JapaneseAnalysisResult:
    """일본 주식 분석 결과"""
    stock_code: str
    company_name: str
    decision: JapaneseDecision
    confidence_score: float
    reasoning: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 일본 특화 필드
    sector_outlook: str = "중립"           # 업종 전망
    yen_impact: str = "중립"              # 엔화 영향
    technical_pattern: str = ""           # 기술적 패턴
    risk_level: str = "medium"            # 위험도
    
    # 거래 정보
    entry_price: Optional[float] = None    # 진입가
    stop_loss: Optional[float] = None      # 손절가
    target_price_1: Optional[float] = None # 목표가 1
    target_price_2: Optional[float] = None # 목표가 2
    position_size_ratio: float = 0.05      # 포지션 크기 비율
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'stock_code': self.stock_code,
            'company_name': self.company_name,
            'decision': self.decision.value,
            'decision_english': self.decision.english,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
            'sector_outlook': self.sector_outlook,
            'yen_impact': self.yen_impact,
            'technical_pattern': self.technical_pattern,
            'risk_level': self.risk_level,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price_1': self.target_price_1,
            'target_price_2': self.target_price_2,
            'position_size_ratio': self.position_size_ratio,
        }

# =============================================================================
# 일본 전략 기본 클래스
# =============================================================================

def japanese_strategy_cache(hours: int = 1):
    """일본 전략 캐싱 데코레이터"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            now = datetime.now()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < timedelta(hours=hours):
                    logger.debug(f"캐시에서 일본 전략 결과 반환: {func.__name__}")
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator

class BaseJapaneseStrategy(ABC):
    """일본 전략 기본 클래스"""
    
    def __init__(self, name: str, strategy_type: JapaneseStrategyType, weight: float = 1.0):
        self.name = name
        self.strategy_type = strategy_type
        self.weight = weight
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.success_rate = 0.5
        self.last_updated = datetime.now()
        
        # 일본 시장 특화 속성
        self.sector_expertise = {}  # 업종별 전문성
        self.market_cap_preference = "all"  # 시가총액 선호도
    
    @abstractmethod
    def analyze(self, market_data: JapaneseMarketData) -> Tuple[JapaneseDecision, float, str]:
        """
        일본 시장 데이터를 분석하여 투자 결정 반환
        
        Returns:
            Tuple[JapaneseDecision, float, str]: (결정, 신뢰도, 이유)
        """
        pass
    
    def get_sector_adjustment(self, sector: JapaneseSector) -> float:
        """업종별 조정 계수"""
        return self.sector_expertise.get(sector, 1.0)
    
    def get_market_cap_adjustment(self, market_cap: float) -> float:
        """시가총액별 조정 계수"""
        if self.market_cap_preference == "large" and market_cap > 1000:  # 1000억엔 이상
            return 1.2
        elif self.market_cap_preference == "small" and market_cap < 100:  # 100억엔 미만
            return 1.2
        return 1.0

# =============================================================================
# 구체적인 일본 전략 구현
# =============================================================================

class HonmaStrategy(BaseJapaneseStrategy):
    """혼마 무네히사 촛대 분석 전략"""
    
    def __init__(self):
        super().__init__("Honma Munehisa", JapaneseStrategyType.HONMA_CANDLE, weight=1.3)
        # 혼마는 쌀 거래의 대가로 전통적 가치투자 성향
        self.sector_expertise = {
            JapaneseSector.MANUFACTURING: 1.2,
            JapaneseSector.AUTOMOTIVE: 1.1,
            JapaneseSector.FINANCE: 1.0,
        }
        self.market_cap_preference = "large"
    
    @japanese_strategy_cache(hours=4)
    def analyze(self, market_data: JapaneseMarketData) -> Tuple[JapaneseDecision, float, str]:
        """혼마 촛대 분석"""
        confidence = 75
        reason = "혼마 전략: "
        
        # 기본적으로 보수적 접근
        decision = JapaneseDecision.MOCHIMASU
        
        # 뉴스 감성 분석 (혼마는 시장 심리를 중시)
        if market_data.news_sentiment == "긍정" and market_data.news_score > 30:
            decision = JapaneseDecision.KAIMASU
            confidence = 80
            reason += "긍정적 시장 심리, 매수"
        elif market_data.news_sentiment == "부정" and market_data.news_score < -30:
            decision = JapaneseDecision.URIMASU
            confidence = 85
            reason += "부정적 시장 심리, 매도"
        
        # 니케이 지수와의 상관관계 (전체 시장 흐름 중시)
        if market_data.nikkei_225 > 30000:  # 역사적 고점 근처
            if decision == JapaneseDecision.KAIMASU:
                decision = JapaneseDecision.MOCHIMASU
                confidence -= 10
                reason += ", 지수 고점으로 보수적"
        elif market_data.nikkei_225 < 25000:  # 상대적 저점
            if decision == JapaneseDecision.MOCHIMASU:
                decision = JapaneseDecision.KAIMASU
                confidence += 5
                reason += ", 지수 저점으로 매수 기회"
        
        # 업종별 조정
        if market_data.sector:
            sector_adj = self.get_sector_adjustment(market_data.sector)
            confidence *= sector_adj
            if sector_adj > 1.0:
                reason += f", {market_data.sector.value} 업종 전문성 반영"
        
        return decision, min(confidence, 95), reason

class IchimokuStrategy(BaseJapaneseStrategy):
    """일목균형표 전략"""
    
    def __init__(self):
        super().__init__("Ichimoku Kinko Hyo", JapaneseStrategyType.ICHIMOKU, weight=1.2)
        # 일목균형표는 모든 업종에 적용 가능한 범용 기술적 분석
        self.sector_expertise = {sector: 1.0 for sector in JapaneseSector}
    
    @japanese_strategy_cache(hours=2)
    def analyze(self, market_data: JapaneseMarketData) -> Tuple[JapaneseDecision, float, str]:
        """일목균형표 분석"""
        confidence = 70
        reason = "일목균형표: "
        signal_strength = 0  # -2(강매도) ~ +2(강매수)
        
        # 구름(Kumo) 위치 분석
        if market_data.ichimoku_kumo_position == "above":
            signal_strength += 1
            reason += "구름 위 위치(상승세), "
        elif market_data.ichimoku_kumo_position == "below":
            signal_strength -= 1
            reason += "구름 아래 위치(하락세), "
        
        # 전환선-기준선 관계
        if market_data.ichimoku_tenkan_kijun == "golden_cross":
            signal_strength += 1
            confidence += 10
            reason += "전환선 상향돌파, "
        elif market_data.ichimoku_tenkan_kijun == "dead_cross":
            signal_strength -= 1
            confidence += 5
            reason += "전환선 하향돌파, "
        
        # 지연선(Chikou Span) 분석
        if market_data.ichimoku_chikou_span == "above_price":
            signal_strength += 1
            reason += "지연선 강세, "
        elif market_data.ichimoku_chikou_span == "below_price":
            signal_strength -= 1
            reason += "지연선 약세, "
        
        # 최종 결정
        if signal_strength >= 2:
            decision = JapaneseDecision.TSUYOKU_KAIMASU
            confidence = 90
        elif signal_strength == 1:
            decision = JapaneseDecision.KAIMASU
            confidence = 80
        elif signal_strength == -1:
            decision = JapaneseDecision.URIMASU
            confidence = 80
        elif signal_strength <= -2:
            decision = JapaneseDecision.TSUYOKU_URIMASU
            confidence = 90
        else:
            decision = JapaneseDecision.MOCHIMASU
            confidence = 60
        
        # 뉴스 감성 보조 신호
        if market_data.news_sentiment == "부정" and market_data.news_score < -50:
            if decision in [JapaneseDecision.KAIMASU, JapaneseDecision.TSUYOKU_KAIMASU]:
                decision = JapaneseDecision.MOCHIMASU
                confidence -= 15
                reason += "부정 뉴스로 매수 신호 약화"
        
        return decision, min(confidence, 95), reason.rstrip(", ")

class BNFStrategy(BaseJapaneseStrategy):
    """BNF (B・N・F) 스윙 트레이딩 전략"""
    
    def __init__(self):
        super().__init__("BNF", JapaneseStrategyType.MODERN_SWING, weight=1.1)
        # BNF는 소형주, 테마주에 특화
        self.sector_expertise = {
            JapaneseSector.TECHNOLOGY: 1.3,
            JapaneseSector.RETAIL: 1.2,
            JapaneseSector.HEALTHCARE: 1.1,
        }
        self.market_cap_preference = "small"
    
    @japanese_strategy_cache(hours=1)
    def analyze(self, market_data: JapaneseMarketData) -> Tuple[JapaneseDecision, float, str]:
        """BNF 스타일 스윙 분석"""
        confidence = 75
        reason = "BNF 전략: "
        
        # 기본적으로 적극적 매수 성향
        decision = JapaneseDecision.KAIMASU
        
        # 소형주 선호 (시가총액 조정)
        if market_data.market_cap and market_data.market_cap < 500:  # 500억엔 미만
            confidence += 10
            reason += "소형주 선호, "
        
        # 뉴스 모멘텀 중시
        if market_data.news_sentiment == "긍정":
            if market_data.news_score > 60:
                decision = JapaneseDecision.TSUYOKU_KAIMASU
                confidence = 90
                reason += "강력한 긍정 뉴스 모멘텀, "
            elif market_data.news_score > 20:
                confidence += 5
                reason += "긍정 뉴스 모멘텀, "
        elif market_data.news_sentiment == "부정" and market_data.news_score < -40:
            decision = JapaneseDecision.URIMASU
            confidence = 85
            reason += "부정 뉴스로 매도, "
        
        # RSI 오버바잉/오버셀링 확인
        if market_data.rsi:
            if market_data.rsi > 80:
                if decision in [JapaneseDecision.KAIMASU, JapaneseDecision.TSUYOKU_KAIMASU]:
                    decision = JapaneseDecision.MOCHIMASU
                    confidence -= 10
                    reason += "RSI 과매수, "
            elif market_data.rsi < 20:
                if decision == JapaneseDecision.MOCHIMASU:
                    decision = JapaneseDecision.KAIMASU
                    confidence += 10
                    reason += "RSI 과매도 반등, "
        
        # 업종별 조정
        if market_data.sector:
            sector_adj = self.get_sector_adjustment(market_data.sector)
            confidence *= sector_adj
        
        return decision, min(confidence, 95), reason.rstrip(", ")

class CISStrategy(BaseJapaneseStrategy):
    """CIS 스캘핑/데이트레이딩 전략"""
    
    def __init__(self):
        super().__init__("CIS", JapaneseStrategyType.SCALPING, weight=0.9)
        # CIS는 유동성 높은 대형주, 빠른 매매
        self.sector_expertise = {
            JapaneseSector.TECHNOLOGY: 1.1,
            JapaneseSector.FINANCE: 1.2,
            JapaneseSector.AUTOMOTIVE: 1.1,
        }
        self.market_cap_preference = "large"
    
    @japanese_strategy_cache(hours=0.5)  # 30분 캐시 (단기 전략)
    def analyze(self, market_data: JapaneseMarketData) -> Tuple[JapaneseDecision, float, str]:
        """CIS 스타일 단기 분석"""
        confidence = 70
        reason = "CIS 전략: "
        
        # 기본적으로 보수적 (단기 매매라 신중)
        decision = JapaneseDecision.MOCHIMASU
        
        # 대형주 선호
        if market_data.market_cap and market_data.market_cap > 1000:  # 1000억엔 이상
            confidence += 5
            reason += "대형주 유동성, "
        
        # MACD 신호 중시
        if market_data.macd_signal == "buy":
            decision = JapaneseDecision.KAIMASU
            confidence = 80
            reason += "MACD 매수 신호, "
        elif market_data.macd_signal == "sell":
            decision = JapaneseDecision.URIMASU
            confidence = 85
            reason += "MACD 매도 신호, "
        
        # 볼린저 밴드 분석
        if market_data.bollinger_position == "upper":
            if decision == JapaneseDecision.KAIMASU:
                decision = JapaneseDecision.MOCHIMASU
                confidence -= 10
                reason += "볼린저 상단 주의, "
        elif market_data.bollinger_position == "lower":
            if decision == JapaneseDecision.MOCHIMASU:
                decision = JapaneseDecision.KAIMASU
                confidence += 10
                reason += "볼린저 하단 반등, "
        
        # 뉴스는 단기적으로만 반영
        if market_data.news_sentiment == "부정" and market_data.news_score < -70:
            decision = JapaneseDecision.URIMASU
            confidence = 90
            reason += "급격한 부정 뉴스, "
        
        # 시장 시간 확인 (일본 시간 기준)
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour > 15:  # 장외 시간
            confidence *= 0.8
            reason += "장외시간 리스크, "
        
        return decision, min(confidence, 95), reason.rstrip(", ")

# =============================================================================
# 일본 시장 특화 분석기
# =============================================================================

class JapaneseMarketAnalyzer(BaseComponent):
    """일본 시장 전문 분석기"""
    
    def __init__(self):
        super().__init__("JapaneseMarketAnalyzer")
        self.strategies: List[BaseJapaneseStrategy] = []
        self.analysis_history: List[JapaneseAnalysisResult] = []
        self.sector_rotation_tracker = {}
        self.yen_trend_analyzer = YenTrendAnalyzer()
    
    def _do_initialize(self):
        """일본 전략들 초기화"""
        self.strategies = [
            HonmaStrategy(),
            IchimokuStrategy(), 
            BNFStrategy(),
            CISStrategy(),
        ]
        
        self.logger.info(f"일본 시장 분석기 초기화: {len(self.strategies)}개 전략 로드")
    
    def collect_japanese_market_data(self, stock_code: str) -> JapaneseMarketData:
        """일본 시장 데이터 수집"""
        try:
            # 뉴스 데이터 수집
            news_data = fetch_all_news(stock_code)
            news_sentiment = evaluate_news(news_data)
            
            # 뉴스 점수 계산
            news_score = 0
            if news_sentiment == "긍정":
                news_score = 40
            elif news_sentiment == "부정":
                news_score = -40
            
            # Mock 데이터 (실제 구현에서는 API 연동)
            market_data = JapaneseMarketData(
                stock_code=stock_code,
                company_name=f"회사_{stock_code}",
                nikkei_225=28500.0,
                usdjpy_rate=148.5,
                news_sentiment=news_sentiment,
                news_score=news_score,
                # 기술적 지표 Mock
                rsi=50.0,
                macd_signal="neutral",
                bollinger_position="middle",
                ichimoku_kumo_position="above",
                ichimoku_tenkan_kijun="neutral",
                ichimoku_chikou_span="neutral"
            )
            
            if not market_data.is_valid():
                raise ValidationError(f"잘못된 일본 시장 데이터: {market_data}")
            
            self.logger.debug(f"{stock_code} 일본 시장 데이터 수집 완료")
            return market_data
            
        except Exception as e:
            self.logger.error(f"일본 시장 데이터 수집 실패: {e}")
            # 폴백 데이터
            return JapaneseMarketData(
                stock_code=stock_code,
                nikkei_225=28000.0,
                usdjpy_rate=150.0,
                news_sentiment="중립",
                news_score=0.0
            )
    
    def run_japanese_strategy_voting(self, market_data: JapaneseMarketData) -> Dict[str, Any]:
        """일본 전략 투표 실행"""
        strategy_results = []
        total_weight = 0
        
        for strategy in self.strategies:
            try:
                decision, confidence, reason = strategy.analyze(market_data)
                strategy_results.append({
                    'strategy': strategy.name,
                    'decision': decision,
                    'confidence': confidence,
                    'reason': reason,
                    'weight': strategy.weight
                })
                total_weight += strategy.weight
                
            except Exception as e:
                self.logger.error(f"일본 전략 {strategy.name} 실행 실패: {e}")
        
        # 가중 투표 계산
        decision_scores = {
            JapaneseDecision.TSUYOKU_URIMASU: -2,
            JapaneseDecision.URIMASU: -1,
            JapaneseDecision.MOCHIMASU: 0,
            JapaneseDecision.KAIMASU: 1,
            JapaneseDecision.TSUYOKU_KAIMASU: 2
        }
        
        weighted_score = 0
        total_confidence = 0
        
        for result in strategy_results:
            score = decision_scores[result['decision']]
            weight = result['weight']
            confidence = result['confidence']
            
            weighted_score += score * weight * (confidence / 100)
            total_confidence += confidence * weight
        
        # 최종 결정 계산
        if total_weight > 0:
            final_score = weighted_score / total_weight
            average_confidence = total_confidence / total_weight
        else:
            final_score = 0
            average_confidence = 50
        
        # 점수를 결정으로 변환
        if final_score >= 1.5:
            final_decision = JapaneseDecision.TSUYOKU_KAIMASU
        elif final_score >= 0.5:
            final_decision = JapaneseDecision.KAIMASU
        elif final_score <= -1.5:
            final_decision = JapaneseDecision.TSUYOKU_URIMASU
        elif final_score <= -0.5:
            final_decision = JapaneseDecision.URIMASU
        else:
            final_decision = JapaneseDecision.MOCHIMASU
        
        return {
            'final_decision': final_decision,
            'confidence': average_confidence,
            'weighted_score': final_score,
            'strategy_results': strategy_results
        }
    
    def apply_japanese_market_overrides(self, voting_result: Dict[str, Any], 
                                      market_data: JapaneseMarketData) -> Tuple[JapaneseDecision, float, str]:
        """일본 시장 특화 오버라이드 규칙"""
        decision = voting_result['final_decision']
        confidence = voting_result['confidence']
        reason = f"투표 결과: {decision.value}"
        
        # 부정 뉴스 오버라이드 (일본은 뉴스에 민감)
        if (market_data.news_sentiment == "부정" and 
            market_data.news_score < -60):
            decision = JapaneseDecision.URIMASU
            confidence = min(confidence * 1.1, 90)
            reason += " → 심각한 부정 뉴스로 매도"
        
        # 엔화 급변 오버라이드
        yen_impact = self.yen_trend_analyzer.analyze_yen_impact(market_data)
        if yen_impact['impact'] == "negative_strong":
            if decision in [JapaneseDecision.KAIMASU, JapaneseDecision.TSUYOKU_KAIMASU]:
                decision = JapaneseDecision.MOCHIMASU
                confidence *= 0.9
                reason += " → 엔화 급등으로 매수 신호 약화"
        elif yen_impact['impact'] == "positive_strong":
            if decision == JapaneseDecision.MOCHIMASU:
                decision = JapaneseDecision.KAIMASU
                confidence *= 1.05
                reason += " → 엔화 약세로 수출주 유리"
        
        # 니케이 지수 극단적 상황
        if market_data.nikkei_225 > 32000:  # 역사적 고점
            if decision in [JapaneseDecision.KAIMASU, JapaneseDecision.TSUYOKU_KAIMASU]:
                decision = JapaneseDecision.MOCHIMASU
                confidence *= 0.85
                reason += " → 니케이 고점으로 보수적"
        elif market_data.nikkei_225 < 24000:  # 상대적 저점
            if decision == JapaneseDecision.URIMASU:
                decision = JapaneseDecision.MOCHIMASU
                confidence *= 0.9
                reason += " → 니케이 저점, 성급한 매도 방지"
        
        return decision, confidence, reason
    
    def analyze_japan(self, stock_code: str) -> JapaneseAnalysisResult:
        """
        일본 주식 종합 분석
        
        Args:
            stock_code: 4자리 일본 주식 코드
            
        Returns:
            JapaneseAnalysisResult: 분석 결과
        """
        try:
            self.logger.info(f"일본 주식 분석 시작: {stock_code}")
            
            # 1. 일본 시장 데이터 수집
            market_data = self.collect_japanese_market_data(stock_code)
            
            # 2. 전략 투표 실행
            voting_result = self.run_japanese_strategy_voting(market_data)
            
            # 3. 일본 시장 특화 오버라이드 적용
            final_decision, final_confidence, reasoning = self.apply_japanese_market_overrides(
                voting_result, market_data
            )
            
            # 4. 엔화 영향 분석
            yen_analysis = self.yen_trend_analyzer.analyze_yen_impact(market_data)
            
            # 5. 업종 전망 분석
            sector_outlook = self.analyze_sector_outlook(market_data.sector)
            
            # 6. 기술적 패턴 인식
            technical_pattern = self.identify_technical_pattern(market_data)
            
            # 7. 분석 결과 생성
            analysis_result = JapaneseAnalysisResult(
                stock_code=stock_code,
                company_name=market_data.company_name,
                decision=final_decision,
                confidence_score=final_confidence,
                reasoning={
                    'market_data': {
                        'nikkei_225': market_data.nikkei_225,
                        'usdjpy_rate': market_data.usdjpy_rate,
                        'news_sentiment': market_data.news_sentiment,
                        'news_score': market_data.news_score
                    },
                    'voting': voting_result,
                    'yen_analysis': yen_analysis,
                    'final_reason': reasoning
                },
                sector_outlook=sector_outlook,
                yen_impact=yen_analysis['impact'],
                technical_pattern=technical_pattern
            )
            
            # 8. 목표가 및 손절가 설정
            self.set_price_targets(analysis_result, market_data)
            
            # 9. 히스토리에 추가
            self.analysis_history.append(analysis_result)
            
            # 최근 100개만 유지
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            self.logger.info(f"일본 주식 분석 완료: {stock_code} → {final_decision.value}")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"일본 주식 분석 실패 ({stock_code}): {e}")
            
            # 폴백 결과
            return JapaneseAnalysisResult(
                stock_code=stock_code,
                company_name=f"주식_{stock_code}",
                decision=JapaneseDecision.MOCHIMASU,
                confidence_score=30,
                reasoning={'error': str(e)},
                risk_level="high"
            )
    
    def analyze_sector_outlook(self, sector: Optional[JapaneseSector]) -> str:
        """업종 전망 분석"""
        if not sector:
            return "업종 정보 없음"
        
        # 간단한 업종 순환 분석 (실제로는 더 복잡한 로직)
        sector_outlooks = {
            JapaneseSector.TECHNOLOGY: "긍정",
            JapaneseSector.AUTOMOTIVE: "중립",
            JapaneseSector.FINANCE: "보통",
            JapaneseSector.RETAIL: "긍정",
            JapaneseSector.MANUFACTURING: "중립",
        }
        
        return sector_outlooks.get(sector, "중립")
    
    def identify_technical_pattern(self, market_data: JapaneseMarketData) -> str:
        """기술적 패턴 식별"""
        patterns = []
        
        # 일목균형표 패턴
        if (market_data.ichimoku_kumo_position == "above" and 
            market_data.ichimoku_tenkan_kijun == "golden_cross"):
            patterns.append("일목_강세돌파")
        
        # RSI 패턴
        if market_data.rsi:
            if market_data.rsi > 70:
                patterns.append("RSI_과매수")
            elif market_data.rsi < 30:
                patterns.append("RSI_과매도")
        
        # 볼린저 밴드 패턴
        if market_data.bollinger_position == "upper":
            patterns.append("볼린저_상단")
        elif market_data.bollinger_position == "lower":
            patterns.append("볼린저_하단")
        
        return ", ".join(patterns) if patterns else "명확한 패턴 없음"
    
    def set_price_targets(self, result: JapaneseAnalysisResult, market_data: JapaneseMarketData):
        """목표가 및 손절가 설정"""
        # 간단한 목표가 설정 (실제로는 더 정교한 계산)
        if result.decision in [JapaneseDecision.KAIMASU, JapaneseDecision.TSUYOKU_KAIMASU]:
            result.stop_loss = 0.08  # 8% 손절
            result.target_price_1 = 0.15  # 15% 목표가 1
            result.target_price_2 = 0.25  # 25% 목표가 2
            
            # 강력 매수시 더 공격적
            if result.decision == JapaneseDecision.TSUYOKU_KAIMASU:
                result.target_price_1 = 0.20
                result.target_price_2 = 0.35
                result.position_size_ratio = 0.08
        
        elif result.decision in [JapaneseDecision.URIMASU, JapaneseDecision.TSUYOKU_URIMASU]:
            result.stop_loss = 0.05  # 5% 추가 하락시 강제 매도
            result.position_size_ratio = 0.0  # 매도는 포지션 크기 0

# =============================================================================
# 엔화 트렌드 분석기
# =============================================================================

class YenTrendAnalyzer:
    """엔화 트렌드 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.YenTrendAnalyzer")
        self.historical_rates = []  # 과거 환율 데이터
    
    def analyze_yen_impact(self, market_data: JapaneseMarketData) -> Dict[str, Any]:
        """엔화 환율이 주식에 미치는 영향 분석"""
        current_rate = market_data.usdjpy_rate
        
        # 기준 환율 (최근 평균으로 가정)
        baseline_rate = 145.0
        
        rate_change_pct = (current_rate - baseline_rate) / baseline_rate * 100
        
        # 영향도 분석
        if rate_change_pct > 3:  # 3% 이상 엔화 약세
            impact = "positive_strong"
            description = "엔화 약세로 수출주 크게 유리"
        elif rate_change_pct > 1:
            impact = "positive_moderate"
            description = "엔화 약세로 수출주 유리"
        elif rate_change_pct < -3:  # 3% 이상 엔화 강세
            impact = "negative_strong"
            description = "엔화 강세로 수출주 크게 불리"
        elif rate_change_pct < -1:
            impact = "negative_moderate"
            description = "엔화 강세로 수출주 불리"
        else:
            impact = "neutral"
            description = "환율 중립"
        
        return {
            'impact': impact,
            'rate_change_pct': rate_change_pct,
            'current_rate': current_rate,
            'baseline_rate': baseline_rate,
            'description': description
        }

# =============================================================================
# 편의 함수들 (기존 API 호환성)
# =============================================================================

# 전역 분석기 인스턴스
_japanese_analyzer = None

def get_japanese_analyzer() -> JapaneseMarketAnalyzer:
    """전역 일본 분석기 인스턴스 반환"""
    global _japanese_analyzer
    if _japanese_analyzer is None:
        _japanese_analyzer = JapaneseMarketAnalyzer()
        _japanese_analyzer.initialize()
    return _japanese_analyzer

def analyze_japan(stock: str) -> dict:
    """
    기존 API와 호환성을 위한 래퍼 함수
    
    Args:
        stock: 분석할 일본 주식 코드
        
    Returns:
        dict: 분석 결과 딕셔너리
    """
    analyzer = get_japanese_analyzer()
    result = analyzer.analyze_japan(stock)
    
    # 기존 API 형식으로 변환
    return {
        "decision": result.decision.english,  # 영어 결정
        "decision_japanese": result.decision.value,  # 일본어 결정
        "confidence_score": result.confidence_score,
        "reason": result.reasoning.get('final_reason', ''),
        "sector_outlook": result.sector_outlook,
        "yen_impact": result.yen_impact,
        "technical_pattern": result.technical_pattern,
        "risk_level": result.risk_level,
        "stop_loss": result.stop_loss,
        "target_price_1": result.target_price_1,
        "target_price_2": result.target_price_2,
        "timestamp": result.timestamp.isoformat()
    }

# 개별 전략 함수들 (기존 호환성)
def strategy_honma() -> str:
    """혼마 전략 (기존 호환성)"""
    return "buy"

def strategy_ichimoku() -> str:
    """일목균형표 전략 (기존 호환성)"""
    return "buy"

def strategy_bnf() -> str:
    """BNF 전략 (기존 호환성)"""
    return "buy"

def strategy_cis() -> str:
    """CIS 전략 (기존 호환성)"""
    return "hold"

# =============================================================================
# 일본 시장 특화 도구들
# =============================================================================

class JapaneseEarningsCalendar:
    """일본 기업 실적발표 캘린더"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.JapaneseEarningsCalendar")
        self.earnings_schedule = {}
    
    def is_earnings_season(self, stock_code: str) -> bool:
        """실적발표 시즌 여부 확인"""
        # 일본 기업들의 결산 시기: 3월, 9월이 집중
        current_month = datetime.now().month
        
        # 3월 결산 기업 (대부분)
        if current_month in [4, 5]:  # 4-5월 1Q 실적
            return True
        elif current_month in [7, 8]:  # 7-8월 2Q 실적
            return True
        elif current_month in [10, 11]:  # 10-11월 3Q 실적
            return True
        elif current_month in [1, 2]:  # 1-2월 연간 실적
            return True
        
        return False

class JapaneseMarketRegime:
    """일본 시장 국면 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.JapaneseMarketRegime")
    
    def detect_market_regime(self, market_data: JapaneseMarketData) -> str:
        """일본 시장 국면 탐지"""
        nikkei = market_data.nikkei_225
        
        # 간단한 국면 분류
        if nikkei > 30000:
            return "bull_market_high"
        elif nikkei > 27000:
            return "bull_market"
        elif nikkei < 22000:
            return "bear_market"
        elif nikkei < 25000:
            return "consolidation_low"
        else:
            return "consolidation"

class JapanesePortfolioManager:
    """일본 주식 포트폴리오 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.JapanesePortfolioManager")
        self.positions = {}
        self.sector_limits = {
            JapaneseSector.TECHNOLOGY: 0.3,  # 기술주 30% 한도
            JapaneseSector.FINANCE: 0.2,     # 금융주 20% 한도
        }
    
    def calculate_sector_allocation(self, analyses: List[JapaneseAnalysisResult]) -> Dict[str, float]:
        """업종별 포트폴리오 배분 계산"""
        sector_weights = {}
        total_weight = 0
        
        for analysis in analyses:
            if analysis.decision in [JapaneseDecision.KAIMASU, JapaneseDecision.TSUYOKU_KAIMASU]:
                sector = analysis.reasoning.get('market_data', {}).get('sector', 'unknown')
                weight = analysis.confidence_score * analysis.position_size_ratio
                
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += weight
                total_weight += weight
        
        # 정규화
        if total_weight > 0:
            for sector in sector_weights:
                sector_weights[sector] /= total_weight
        
        return sector_weights

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
    
    # 일본 분석기 초기화
    analyzer = get_japanese_analyzer()
    
    # 테스트 일본 주식들 (4자리 코드)
    test_stocks = ["7203", "6758", "9984", "8306"]  # 토요타, 소니, 소프트뱅크, 미쓰비시UFJ
    stock_names = ["토요타", "소니", "소프트뱅크", "미쓰비시UFJ"]
    
    print("=== 고급 일본 주식 전략 분석 시스템 ===\n")
    print("🗾 일본 시장 전문 분석 시작...\n")
    
    # 각 주식 분석
    results = []
    for i, stock in enumerate(test_stocks):
        print(f"📊 {stock_names[i]}({stock}) 분석 중...")
        result = analyze_japan(stock)
        results.append(result)
        
        print(f"   결정: {result['decision'].upper()} ({result['decision_japanese']})")
        print(f"   신뢰도: {result['confidence_score']:.1f}%")
        print(f"   업종 전망: {result['sector_outlook']}")
        print(f"   엔화 영향: {result['yen_impact']}")
        print(f"   기술적 패턴: {result['technical_pattern']}")
        if result['target_price_1']:
            print(f"   목표가1: +{result['target_price_1']:.1%}")
        print()
    
    # 포트폴리오 요약
    buy_signals = [r for r in results if r['decision'] in ['buy', 'strong_buy']]
    sell_signals = [r for r in results if r['decision'] in ['sell', 'strong_sell']]
    
    print("📈 포트폴리오 요약:")
    print(f"   매수 신호: {len(buy_signals)}개")
    print(f"   매도 신호: {len(sell_signals)}개")
    print(f"   평균 신뢰도: {sum(r['confidence_score'] for r in results) / len(results):.1f}%")
    
    # 분석 히스토리 내보내기
    print(f"\n💾 일본 주식 분석 히스토리 내보내기...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"japanese_analysis_{timestamp}.json"
    
    history_data = {
        'export_time': datetime.now().isoformat(),
        'market': 'Japan',
        'total_analyses': len(results),
        'analyses': results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    
    print(f"   저장 완료: {filename}")
    print("\n✅ 일본 시장 분석 시스템 테스트 완료!")
    print("🎌 頑張って! (힘내세요!)")

if __name__ == "__main__":
    main()

# =============================================================================
# 공개 API
# =============================================================================

__all__ = [
    # 메인 클래스들
    'JapaneseMarketAnalyzer',
    'BaseJapaneseStrategy',
    'YenTrendAnalyzer',
    'JapaneseEarningsCalendar',
    'JapaneseMarketRegime',
    'JapanesePortfolioManager',
    
    # 데이터 클래스들
    'JapaneseAnalysisResult',
    'JapaneseMarketData',
    'JapaneseDecision',
    'JapaneseStrategyType',
    'JapaneseSector',
    
    # 구체적인 전략들
    'HonmaStrategy',
    'IchimokuStrategy',
    'BNFStrategy',
    'CISStrategy',
    
    # 편의 함수들
    'analyze_japan',
    'get_japanese_analyzer',
    'strategy_honma',
    'strategy_ichimoku',
    'strategy_bnf',
    'strategy_cis',
    
    # 상수
    'JAPAN_CONFIG',
]