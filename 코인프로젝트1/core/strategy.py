"""
Advanced Cryptocurrency Trading Strategy System
==============================================

퀀트 수준의 코인 거래 전략 분석 시스템
다양한 투자 대가들의 전략을 조합하여 최적의 투자 결정을 제공합니다.

Features:
- 다중 전략 투표 시스템
- Fear & Greed 지수 분석
- 뉴스 감성 분석
- 기술적 지표 통합
- 위험 관리 시스템

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

# Core 패키지 import
try:
    from core import BaseComponent, config, logger, ValidationError
except ImportError:
    # 폴백 로거 (core 패키지가 없을 경우)
    import logging
    logger = logging.getLogger(__name__)

# Utils import
try:
    from utils import get_fear_greed_index, fetch_all_news, evaluate_news
except ImportError:
    logger.warning("utils 모듈을 찾을 수 없습니다. Mock 함수를 사용합니다.")
    
    def get_fear_greed_index() -> float:
        """Mock function for Fear & Greed Index"""
        return 50.0
    
    def fetch_all_news(coin: str) -> List[Dict]:
        """Mock function for news fetching"""
        return []
    
    def evaluate_news(news: List[Dict]) -> str:
        """Mock function for news evaluation"""
        return "중립"

# =============================================================================
# 상수 및 설정
# =============================================================================

class Decision(Enum):
    """투자 결정 열거형"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class StrategyType(Enum):
    """전략 유형"""
    VALUE = "value"           # 가치 투자
    MOMENTUM = "momentum"     # 모멘텀 투자
    CONTRARIAN = "contrarian" # 역발상 투자
    TECHNICAL = "technical"   # 기술적 분석

# 기본 설정값
DEFAULT_CONFIG = {
    'FG_EXTREME_FEAR': 20,      # 극도의 공포 임계값
    'FG_EXTREME_GREED': 80,     # 극도의 탐욕 임계값
    'FG_MODERATE_FEAR': 40,     # 보통 공포 임계값
    'FG_MODERATE_GREED': 60,    # 보통 탐욕 임계값
    'MIN_CONFIDENCE': 50,       # 최소 신뢰도
    'NEWS_WEIGHT': 0.3,         # 뉴스 가중치
    'FG_WEIGHT': 0.4,          # FG 지수 가중치
    'STRATEGY_WEIGHT': 0.3,     # 전략 가중치
    'ENABLE_RISK_MANAGEMENT': True,  # 위험 관리 활성화
}

# =============================================================================
# 데이터 클래스들
# =============================================================================

@dataclass
class AnalysisResult:
    """분석 결과를 담는 데이터 클래스"""
    coin: str
    decision: Decision
    confidence_score: float
    reasoning: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    risk_level: str = "medium"
    expected_return: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def __post_init__(self):
        """후처리: 신뢰도 검증"""
        if not 0 <= self.confidence_score <= 100:
            raise ValidationError(f"신뢰도는 0-100 사이여야 합니다: {self.confidence_score}")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'coin': self.coin,
            'decision': self.decision.value,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
            'risk_level': self.risk_level,
            'expected_return': self.expected_return,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
        }

@dataclass
class MarketData:
    """시장 데이터를 담는 클래스"""
    fear_greed_index: float
    news_sentiment: str
    news_score: float = 0.0
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    volume_analysis: Optional[Dict[str, Any]] = None
    
    def is_valid(self) -> bool:
        """데이터 유효성 검사"""
        return (
            0 <= self.fear_greed_index <= 100 and
            self.news_sentiment in ['긍정', '부정', '중립'] and
            -100 <= self.news_score <= 100
        )

# =============================================================================
# 전략 인터페이스 및 구현
# =============================================================================

def strategy_cache(hours: int = 1):
    """전략 결과 캐싱 데코레이터"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            now = datetime.now()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < timedelta(hours=hours):
                    logger.debug(f"캐시에서 전략 결과 반환: {func.__name__}")
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator

class BaseStrategy(ABC):
    """기본 전략 추상 클래스"""
    
    def __init__(self, name: str, strategy_type: StrategyType, weight: float = 1.0):
        self.name = name
        self.strategy_type = strategy_type
        self.weight = weight
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.success_rate = 0.0  # 과거 성공률
        self.last_updated = datetime.now()
    
    @abstractmethod
    def analyze(self, market_data: MarketData, coin: str) -> Tuple[Decision, float, str]:
        """
        시장 데이터를 분석하여 투자 결정을 반환
        
        Returns:
            Tuple[Decision, float, str]: (결정, 신뢰도, 이유)
        """
        pass
    
    def get_weighted_confidence(self, base_confidence: float) -> float:
        """가중치가 적용된 신뢰도 계산"""
        # 성공률과 가중치를 고려한 조정
        success_modifier = (self.success_rate - 0.5) * 0.2  # -0.1 ~ +0.1
        weighted_confidence = base_confidence * self.weight * (1 + success_modifier)
        return max(0, min(100, weighted_confidence))
    
    def update_success_rate(self, new_rate: float):
        """성공률 업데이트"""
        if 0 <= new_rate <= 1:
            self.success_rate = new_rate
            self.last_updated = datetime.now()
            self.logger.info(f"{self.name} 전략 성공률 업데이트: {new_rate:.2%}")

class BuffettStrategy(BaseStrategy):
    """워렌 버핏 가치투자 전략"""
    
    def __init__(self):
        super().__init__("Buffett", StrategyType.VALUE, weight=1.2)
    
    @strategy_cache(hours=6)
    def analyze(self, market_data: MarketData, coin: str) -> Tuple[Decision, float, str]:
        """장기 가치 투자 관점에서 분석"""
        confidence = 70
        reason = f"버핏 전략: "
        
        # Fear & Greed 지수 기반 역발상 투자
        fg = market_data.fear_greed_index
        
        if fg < DEFAULT_CONFIG['FG_EXTREME_FEAR']:  # 극도의 공포
            decision = Decision.STRONG_BUY
            confidence = 85
            reason += "극도의 공포 시기, 강력 매수"
        elif fg < DEFAULT_CONFIG['FG_MODERATE_FEAR']:  # 공포
            decision = Decision.BUY
            confidence = 75
            reason += "공포 시기, 매수 기회"
        elif fg > DEFAULT_CONFIG['FG_EXTREME_GREED']:  # 극도의 탐욕
            decision = Decision.SELL
            confidence = 80
            reason += "극도의 탐욕 시기, 매도"
        elif fg > DEFAULT_CONFIG['FG_MODERATE_GREED']:  # 탐욕
            decision = Decision.HOLD
            confidence = 65
            reason += "탐욕 시기, 보유"
        else:  # 중립
            decision = Decision.HOLD
            confidence = 60
            reason += "시장 중립, 보유"
        
        # 뉴스 감정 고려 (장기 투자자이므로 단기 뉴스에 덜 민감)
        if market_data.news_sentiment == "부정" and market_data.news_score < -50:
            reason += ", 부정적 뉴스는 매수 기회"
            if decision == Decision.HOLD:
                decision = Decision.BUY
                confidence += 5
        
        return decision, self.get_weighted_confidence(confidence), reason

class JesseStrategy(BaseStrategy):
    """제시 리버모어 모멘텀 전략"""
    
    def __init__(self):
        super().__init__("Jesse Livermore", StrategyType.MOMENTUM, weight=1.0)
    
    @strategy_cache(hours=1)
    def analyze(self, market_data: MarketData, coin: str) -> Tuple[Decision, float, str]:
        """모멘텀과 시장 심리 기반 분석"""
        confidence = 75
        reason = "리버모어 전략: "
        
        fg = market_data.fear_greed_index
        news_score = market_data.news_score
        
        # 모멘텀 기반 결정
        if fg > 60 and news_score > 20:  # 상승 모멘텀
            decision = Decision.BUY
            confidence = 80
            reason += "상승 모멘텀 확인, 매수"
        elif fg < 40 and news_score < -20:  # 하락 모멘텀
            decision = Decision.SELL
            confidence = 85
            reason += "하락 모멘텀 확인, 매도"
        elif 40 <= fg <= 60:  # 횡보
            decision = Decision.HOLD
            confidence = 60
            reason += "모멘텀 부족, 대기"
        else:
            decision = Decision.BUY  # 기본적으로 매수 성향
            confidence = 70
            reason += "시장 진입 신호"
        
        return decision, self.get_weighted_confidence(confidence), reason

class WonyoStrategy(BaseStrategy):
    """원요 개인 전략 (적극적 매수)"""
    
    def __init__(self):
        super().__init__("Wonyo", StrategyType.TECHNICAL, weight=0.8)
    
    @strategy_cache(hours=2)
    def analyze(self, market_data: MarketData, coin: str) -> Tuple[Decision, float, str]:
        """적극적 매수 전략"""
        confidence = 70
        reason = "원요 전략: "
        
        # 기본적으로 매수 성향이지만 상황에 따라 조정
        fg = market_data.fear_greed_index
        
        if market_data.news_sentiment == "부정" and market_data.news_score < -60:
            decision = Decision.SELL
            confidence = 75
            reason += "심각한 부정 뉴스, 매도"
        elif fg < 30:  # 공포 구간에서 더 적극적
            decision = Decision.STRONG_BUY
            confidence = 85
            reason += "공포 구간, 강력 매수"
        elif fg > 75:  # 과열 구간에서는 보수적
            decision = Decision.HOLD
            confidence = 65
            reason += "과열 구간, 보유"
        else:
            decision = Decision.BUY
            confidence = 80
            reason += "적극적 매수 전략"
        
        return decision, self.get_weighted_confidence(confidence), reason

class JimRogersStrategy(BaseStrategy):
    """짐 로저스 원자재/글로벌 전략"""
    
    def __init__(self):
        super().__init__("Jim Rogers", StrategyType.VALUE, weight=1.1)
    
    @strategy_cache(hours=4)
    def analyze(self, market_data: MarketData, coin: str) -> Tuple[Decision, float, str]:
        """글로벌 거시경제 관점에서 분석"""
        confidence = 75
        reason = "짐 로저스 전략: "
        
        fg = market_data.fear_greed_index
        
        # 거시경제적 관점에서 암호화폐를 원자재로 간주
        if fg < 25:  # 극도의 공포 - 원자재처럼 축적
            decision = Decision.STRONG_BUY
            confidence = 90
            reason += "극도의 공포, 자산 축적 시기"
        elif fg < 45:  # 공포 - 매수
            decision = Decision.BUY
            confidence = 80
            reason += "공포 구간, 매수"
        elif fg > 70:  # 탐욕 - 이익 실현
            decision = Decision.SELL
            confidence = 85
            reason += "탐욕 구간, 이익 실현"
        else:  # 중립 - 기본 매수
            decision = Decision.BUY
            confidence = 70
            reason += "중장기 상승 전망, 매수"
        
        # 글로벌 뉴스에 더 민감하게 반응
        if market_data.news_sentiment == "긍정" and market_data.news_score > 40:
            if decision in [Decision.HOLD, Decision.SELL]:
                decision = Decision.BUY
                confidence += 10
                reason += ", 긍정적 글로벌 전망"
        
        return decision, self.get_weighted_confidence(confidence), reason

# =============================================================================
# 위험 관리 시스템
# =============================================================================

class RiskManager:
    """위험 관리 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RiskManager")
        self.max_position_size = 0.1  # 최대 포지션 크기 (10%)
        self.max_daily_loss = 0.05    # 최대 일일 손실 (5%)
        self.stop_loss_ratio = 0.15   # 스탑로스 비율 (15%)
        self.take_profit_ratio = 0.25 # 이익실현 비율 (25%)
    
    def assess_risk(self, analysis_result: AnalysisResult, market_data: MarketData) -> AnalysisResult:
        """위험 평가 및 조정"""
        
        # 위험 레벨 계산
        risk_factors = []
        
        # Fear & Greed 지수 기반 위험도
        fg = market_data.fear_greed_index
        if fg < 20 or fg > 80:
            risk_factors.append("extreme_market")
        
        # 뉴스 감정 기반 위험도
        if market_data.news_sentiment == "부정" and market_data.news_score < -70:
            risk_factors.append("negative_sentiment")
        
        # 신뢰도 기반 위험도
        if analysis_result.confidence_score < 60:
            risk_factors.append("low_confidence")
        
        # 위험 레벨 결정
        if len(risk_factors) >= 2:
            analysis_result.risk_level = "high"
            # 고위험 시 포지션 크기 축소
            if analysis_result.decision in [Decision.BUY, Decision.STRONG_BUY]:
                analysis_result.confidence_score *= 0.8
        elif len(risk_factors) == 1:
            analysis_result.risk_level = "medium"
        else:
            analysis_result.risk_level = "low"
        
        # 스탑로스 및 이익실현 설정
        if analysis_result.decision in [Decision.BUY, Decision.STRONG_BUY]:
            analysis_result.stop_loss = self.stop_loss_ratio
            analysis_result.take_profit = self.take_profit_ratio
            
            # 위험도에 따른 조정
            if analysis_result.risk_level == "high":
                analysis_result.stop_loss *= 0.7  # 더 타이트한 스탑로스
                analysis_result.take_profit *= 0.8  # 더 보수적인 이익실현
        
        self.logger.info(f"위험 평가 완료: {analysis_result.coin}, 위험도: {analysis_result.risk_level}")
        
        return analysis_result

# =============================================================================
# 메인 전략 분석기
# =============================================================================

class AdvancedStrategyAnalyzer(BaseComponent):
    """고급 전략 분석기"""
    
    def __init__(self):
        super().__init__("AdvancedStrategyAnalyzer")
        self.strategies: List[BaseStrategy] = []
        self.risk_manager = RiskManager()
        self.analysis_history: List[AnalysisResult] = []
        self.config_values = DEFAULT_CONFIG.copy()
    
    def _do_initialize(self):
        """전략들 초기화"""
        self.strategies = [
            BuffettStrategy(),
            JesseStrategy(),
            WonyoStrategy(),
            JimRogersStrategy(),
        ]
        
        # 설정값 로드
        if hasattr(self, 'config'):
            for key, default_value in DEFAULT_CONFIG.items():
                self.config_values[key] = self.config.get(key, default_value)
        
        self.logger.info(f"전략 분석기 초기화 완료: {len(self.strategies)}개 전략 로드")
    
    def collect_market_data(self, coin: str) -> MarketData:
        """시장 데이터 수집"""
        try:
            # Fear & Greed 지수
            fg_index = get_fear_greed_index()
            
            # 뉴스 데이터
            news_data = fetch_all_news(coin)
            news_sentiment = evaluate_news(news_data)
            
            # 뉴스 점수 계산 (간단한 예시)
            news_score = 0
            if news_sentiment == "긍정":
                news_score = 50
            elif news_sentiment == "부정":
                news_score = -50
            # 중립이면 0
            
            market_data = MarketData(
                fear_greed_index=fg_index,
                news_sentiment=news_sentiment,
                news_score=news_score
            )
            
            if not market_data.is_valid():
                raise ValidationError(f"잘못된 시장 데이터: {market_data}")
            
            self.logger.debug(f"{coin} 시장 데이터 수집 완료: FG={fg_index}, 뉴스={news_sentiment}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 실패: {e}")
            # 폴백 데이터
            return MarketData(
                fear_greed_index=50.0,
                news_sentiment="중립",
                news_score=0.0
            )
    
    def run_strategy_voting(self, market_data: MarketData, coin: str) -> Dict[str, Any]:
        """전략 투표 실행"""
        strategy_results = []
        total_weight = 0
        
        for strategy in self.strategies:
            try:
                decision, confidence, reason = strategy.analyze(market_data, coin)
                strategy_results.append({
                    'strategy': strategy.name,
                    'decision': decision,
                    'confidence': confidence,
                    'reason': reason,
                    'weight': strategy.weight
                })
                total_weight += strategy.weight
                
            except Exception as e:
                self.logger.error(f"전략 {strategy.name} 실행 실패: {e}")
        
        # 가중 투표 계산
        decision_scores = {
            Decision.STRONG_SELL: -2,
            Decision.SELL: -1,
            Decision.HOLD: 0,
            Decision.BUY: 1,
            Decision.STRONG_BUY: 2
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
            final_decision = Decision.STRONG_BUY
        elif final_score >= 0.5:
            final_decision = Decision.BUY
        elif final_score <= -1.5:
            final_decision = Decision.STRONG_SELL
        elif final_score <= -0.5:
            final_decision = Decision.SELL
        else:
            final_decision = Decision.HOLD
        
        return {
            'final_decision': final_decision,
            'confidence': average_confidence,
            'weighted_score': final_score,
            'strategy_results': strategy_results
        }
    
    def apply_overrides(self, voting_result: Dict[str, Any], market_data: MarketData) -> Tuple[Decision, float, str]:
        """오버라이드 규칙 적용"""
        decision = voting_result['final_decision']
        confidence = voting_result['confidence']
        reason = f"투표 결과: {decision.value}"
        
        # 부정 뉴스 오버라이드
        if (market_data.news_sentiment == "부정" and 
            market_data.news_score < -60):
            decision = Decision.SELL
            confidence = min(confidence * 1.1, 90)
            reason += " → 심각한 부정 뉴스로 매도"
        
        # 극도의 Fear & Greed 오버라이드
        elif market_data.fear_greed_index > self.config_values['FG_EXTREME_GREED']:
            if decision in [Decision.BUY, Decision.STRONG_BUY]:
                decision = Decision.HOLD
                confidence *= 0.9
                reason += " → 극도의 탐욕으로 보유"
        
        elif market_data.fear_greed_index < self.config_values['FG_EXTREME_FEAR']:
            if decision == Decision.SELL:
                decision = Decision.HOLD
                confidence *= 0.9
                reason += " → 극도의 공포, 성급한 매도 방지"
        
        return decision, confidence, reason
    
    def analyze_coin(self, coin: str) -> AnalysisResult:
        """
        코인 하나에 대해 종합적인 분석을 수행합니다.
        
        Args:
            coin: 분석할 코인 심볼
            
        Returns:
            AnalysisResult: 분석 결과
        """
        try:
            self.logger.info(f"코인 분석 시작: {coin}")
            
            # 1. 시장 데이터 수집
            market_data = self.collect_market_data(coin)
            
            # 2. 전략 투표 실행
            voting_result = self.run_strategy_voting(market_data, coin)
            
            # 3. 오버라이드 규칙 적용
            final_decision, final_confidence, reasoning = self.apply_overrides(
                voting_result, market_data
            )
            
            # 4. 분석 결과 생성
            analysis_result = AnalysisResult(
                coin=coin,
                decision=final_decision,
                confidence_score=final_confidence,
                reasoning={
                    'market_data': {
                        'fear_greed_index': market_data.fear_greed_index,
                        'news_sentiment': market_data.news_sentiment,
                        'news_score': market_data.news_score
                    },
                    'voting': voting_result,
                    'final_reason': reasoning
                }
            )
            
            # 5. 위험 관리 적용
            if self.config_values.get('ENABLE_RISK_MANAGEMENT', True):
                analysis_result = self.risk_manager.assess_risk(analysis_result, market_data)
            
            # 6. 히스토리에 추가
            self.analysis_history.append(analysis_result)
            
            # 최근 100개만 유지
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            self.logger.info(f"코인 분석 완료: {coin} → {final_decision.value} (신뢰도: {final_confidence:.1f}%)")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"코인 분석 실패 ({coin}): {e}")
            
            # 폴백 결과
            return AnalysisResult(
                coin=coin,
                decision=Decision.HOLD,
                confidence_score=30,
                reasoning={'error': str(e)},
                risk_level="high"
            )
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """전략별 성과 분석"""
        if not self.analysis_history:
            return {}
        
        performance = {}
        for strategy in self.strategies:
            performance[strategy.name] = {
                'success_rate': strategy.success_rate,
                'weight': strategy.weight,
                'last_updated': strategy.last_updated.isoformat(),
                'strategy_type': strategy.strategy_type.value
            }
        
        return performance
    
    def export_analysis_history(self, filepath: str = None) -> str:
        """분석 히스토리 내보내기"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"analysis_history_{timestamp}.json"
        
        history_data = {
            'export_time': datetime.now().isoformat(),
            'total_analyses': len(self.analysis_history),
            'strategy_performance': self.get_strategy_performance(),
            'analyses': [result.to_dict() for result in self.analysis_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"분석 히스토리 내보내기 완료: {filepath}")
        return filepath

# =============================================================================
# 편의 함수들 (기존 API와의 호환성)
# =============================================================================

# 전역 분석기 인스턴스
_analyzer = None

def get_analyzer() -> AdvancedStrategyAnalyzer:
    """전역 분석기 인스턴스 반환"""
    global _analyzer
    if _analyzer is None:
        _analyzer = AdvancedStrategyAnalyzer()
        _analyzer.initialize()
    return _analyzer

def analyze_coin(coin: str) -> dict:
    """
    기존 API와 호환성을 위한 래퍼 함수
    
    Args:
        coin: 분석할 코인 심볼
        
    Returns:
        dict: 분석 결과 딕셔너리
    """
    analyzer = get_analyzer()
    result = analyzer.analyze_coin(coin)
    
    # 기존 API 형식으로 변환
    return {
        "decision": result.decision.value,
        "confidence_score": result.confidence_score,
        "reason": result.reasoning.get('final_reason', ''),
        "risk_level": result.risk_level,
        "stop_loss": result.stop_loss,
        "take_profit": result.take_profit,
        "timestamp": result.timestamp.isoformat()
    }

# 개별 전략 함수들 (기존 호환성)
def strategy_buffett() -> str:
    """워렌 버핏 전략 (기존 호환성)"""
    return "hold"  # 기본값, 실제로는 AdvancedStrategyAnalyzer 사용 권장

def strategy_jesse() -> str:
    """제시 리버모어 전략 (기존 호환성)"""
    return "buy"   # 기본값, 실제로는 AdvancedStrategyAnalyzer 사용 권장

def strategy_wonyo() -> str:
    """원요 전략 (기존 호환성)"""
    return "buy"   # 기본값, 실제로는 AdvancedStrategyAnalyzer 사용 권장

def strategy_jim_rogers() -> str:
    """짐 로저스 전략 (기존 호환성)"""
    return "buy"   # 기본값, 실제로는 AdvancedStrategyAnalyzer 사용 권장

# =============================================================================
# 고급 분석 도구들
# =============================================================================

class MarketRegimeDetector:
    """시장 국면 탐지기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MarketRegimeDetector")
        self.historical_data = []
    
    def detect_regime(self, market_data: MarketData) -> str:
        """
        현재 시장 국면을 탐지합니다.
        
        Returns:
            str: 'bull_market', 'bear_market', 'sideways', 'volatile'
        """
        fg = market_data.fear_greed_index
        news_score = market_data.news_score
        
        # 간단한 국면 탐지 로직
        if fg > 70 and news_score > 30:
            return "bull_market"
        elif fg < 30 and news_score < -30:
            return "bear_market"
        elif abs(news_score) < 20 and 40 <= fg <= 60:
            return "sideways"
        else:
            return "volatile"
    
    def get_regime_strategy_adjustment(self, regime: str) -> Dict[str, float]:
        """국면별 전략 가중치 조정"""
        adjustments = {
            "bull_market": {
                "Buffett": 0.8,      # 가치투자는 상승장에서 보수적
                "Jesse Livermore": 1.3,  # 모멘텀은 상승장에서 강화
                "Wonyo": 1.2,        # 적극적 매수 강화
                "Jim Rogers": 0.9    # 거시경제 관점에서 보수적
            },
            "bear_market": {
                "Buffett": 1.4,      # 가치투자는 하락장에서 기회
                "Jesse Livermore": 0.7,  # 모멘텀은 하락장에서 약화
                "Wonyo": 0.6,        # 적극적 매수 억제
                "Jim Rogers": 1.2    # 거시경제적 관점에서 기회 포착
            },
            "sideways": {
                "Buffett": 1.0,
                "Jesse Livermore": 0.8,
                "Wonyo": 0.9,
                "Jim Rogers": 1.1
            },
            "volatile": {
                "Buffett": 1.1,      # 변동성 장에서 가치투자 유리
                "Jesse Livermore": 1.2,  # 변동성을 활용한 모멘텀
                "Wonyo": 0.8,        # 변동성 장에서 신중
                "Jim Rogers": 0.9
            }
        }
        
        return adjustments.get(regime, {})

class PortfolioOptimizer:
    """포트폴리오 최적화기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PortfolioOptimizer")
    
    def calculate_position_size(self, analysis_result: AnalysisResult, 
                              portfolio_value: float, 
                              max_position_ratio: float = 0.1) -> float:
        """
        포지션 크기 계산
        
        Args:
            analysis_result: 분석 결과
            portfolio_value: 총 포트폴리오 가치
            max_position_ratio: 최대 포지션 비율
            
        Returns:
            float: 권장 포지션 크기 (금액)
        """
        base_position = portfolio_value * max_position_ratio
        
        # 신뢰도에 따른 조정
        confidence_multiplier = analysis_result.confidence_score / 100
        
        # 위험도에 따른 조정
        risk_multipliers = {
            "low": 1.0,
            "medium": 0.8,
            "high": 0.5
        }
        risk_multiplier = risk_multipliers.get(analysis_result.risk_level, 0.8)
        
        # 결정 강도에 따른 조정
        decision_multipliers = {
            Decision.STRONG_BUY: 1.2,
            Decision.BUY: 1.0,
            Decision.HOLD: 0.0,
            Decision.SELL: -1.0,
            Decision.STRONG_SELL: -1.2
        }
        decision_multiplier = decision_multipliers.get(analysis_result.decision, 0)
        
        final_position = (base_position * confidence_multiplier * 
                         risk_multiplier * abs(decision_multiplier))
        
        self.logger.debug(f"포지션 크기 계산: {analysis_result.coin}, "
                         f"기본={base_position:.2f}, 최종={final_position:.2f}")
        
        return max(0, final_position)

class BacktestEngine:
    """백테스트 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BacktestEngine")
        self.trades = []
        self.performance_metrics = {}
    
    def run_backtest(self, coin: str, start_date: datetime, end_date: datetime,
                    initial_capital: float = 10000) -> Dict[str, Any]:
        """
        백테스트 실행 (시뮬레이션)
        
        Args:
            coin: 테스트할 코인
            start_date: 시작 날짜
            end_date: 종료 날짜
            initial_capital: 초기 자본
            
        Returns:
            Dict: 백테스트 결과
        """
        # 실제 구현에서는 과거 데이터를 사용해야 함
        # 여기서는 시뮬레이션 결과 반환
        
        total_trades = 50
        winning_trades = 35
        win_rate = winning_trades / total_trades
        
        final_capital = initial_capital * (1 + (win_rate - 0.5) * 2)
        total_return = (final_capital - initial_capital) / initial_capital
        
        results = {
            'coin': coin,
            'period': f"{start_date.date()} ~ {end_date.date()}",
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'max_drawdown': 0.15,  # 최대 손실률
            'sharpe_ratio': 1.2,   # 샤프 비율
            'volatility': 0.25     # 변동성
        }
        
        self.logger.info(f"백테스트 완료: {coin}, 수익률: {total_return:.2%}")
        return results

# =============================================================================
# 실시간 모니터링 시스템
# =============================================================================

class RealTimeMonitor:
    """실시간 모니터링 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RealTimeMonitor")
        self.alerts = []
        self.monitoring_coins = set()
        self.last_analysis = {}
    
    def add_coin_to_monitor(self, coin: str, alert_threshold: float = 20):
        """모니터링할 코인 추가"""
        self.monitoring_coins.add(coin)
        self.logger.info(f"모니터링 코인 추가: {coin}")
    
    def remove_coin_from_monitor(self, coin: str):
        """모니터링에서 코인 제거"""
        self.monitoring_coins.discard(coin)
        if coin in self.last_analysis:
            del self.last_analysis[coin]
        self.logger.info(f"모니터링 코인 제거: {coin}")
    
    def check_for_alerts(self, coin: str, current_analysis: AnalysisResult) -> List[str]:
        """알림 조건 확인"""
        alerts = []
        
        # 이전 분석 결과와 비교
        if coin in self.last_analysis:
            prev_analysis = self.last_analysis[coin]
            
            # 결정 변경 알림
            if prev_analysis.decision != current_analysis.decision:
                alerts.append(
                    f"{coin}: 투자 결정 변경 "
                    f"{prev_analysis.decision.value} → {current_analysis.decision.value}"
                )
            
            # 신뢰도 급변 알림
            confidence_change = abs(current_analysis.confidence_score - prev_analysis.confidence_score)
            if confidence_change > 20:
                alerts.append(
                    f"{coin}: 신뢰도 급변 {confidence_change:.1f}%"
                )
            
            # 위험도 변경 알림
            if prev_analysis.risk_level != current_analysis.risk_level:
                alerts.append(
                    f"{coin}: 위험도 변경 "
                    f"{prev_analysis.risk_level} → {current_analysis.risk_level}"
                )
        
        # 극단적 조건 알림
        if current_analysis.confidence_score > 90:
            alerts.append(f"{coin}: 매우 높은 신뢰도 ({current_analysis.confidence_score:.1f}%)")
        
        if current_analysis.risk_level == "high" and current_analysis.decision in [Decision.BUY, Decision.STRONG_BUY]:
            alerts.append(f"{coin}: 고위험 매수 신호 주의")
        
        # 현재 분석 결과 저장
        self.last_analysis[coin] = current_analysis
        
        return alerts
    
    def get_portfolio_summary(self, coins: List[str]) -> Dict[str, Any]:
        """포트폴리오 요약 정보"""
        analyzer = get_analyzer()
        
        summary = {
            'total_coins': len(coins),
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'high_confidence_signals': 0,
            'high_risk_signals': 0,
            'average_confidence': 0,
            'coins_analysis': {}
        }
        
        total_confidence = 0
        
        for coin in coins:
            try:
                analysis = analyzer.analyze_coin(coin)
                
                summary['coins_analysis'][coin] = {
                    'decision': analysis.decision.value,
                    'confidence': analysis.confidence_score,
                    'risk_level': analysis.risk_level
                }
                
                # 통계 업데이트
                if analysis.decision in [Decision.BUY, Decision.STRONG_BUY]:
                    summary['buy_signals'] += 1
                elif analysis.decision in [Decision.SELL, Decision.STRONG_SELL]:
                    summary['sell_signals'] += 1
                else:
                    summary['hold_signals'] += 1
                
                if analysis.confidence_score > 80:
                    summary['high_confidence_signals'] += 1
                
                if analysis.risk_level == "high":
                    summary['high_risk_signals'] += 1
                
                total_confidence += analysis.confidence_score
                
            except Exception as e:
                self.logger.error(f"포트폴리오 요약 중 오류 ({coin}): {e}")
        
        if coins:
            summary['average_confidence'] = total_confidence / len(coins)
        
        return summary

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
    
    # 분석기 초기화
    analyzer = get_analyzer()
    
    # 테스트 코인들
    test_coins = ["BTC", "ETH", "ADA", "SOL"]
    
    print("=== 고급 코인 전략 분석 시스템 ===\n")
    
    # 각 코인 분석
    for coin in test_coins:
        print(f"📊 {coin} 분석 중...")
        result = analyze_coin(coin)
        
        print(f"   결정: {result['decision'].upper()}")
        print(f"   신뢰도: {result['confidence_score']:.1f}%")
        print(f"   위험도: {result['risk_level']}")
        print(f"   이유: {result['reason'][:100]}...")
        print()
    
    # 전략 성과 출력
    print("📈 전략별 성과:")
    performance = analyzer.get_strategy_performance()
    for strategy_name, perf in performance.items():
        print(f"   {strategy_name}: 성공률 {perf['success_rate']:.1%}, 가중치 {perf['weight']}")
    
    # 분석 히스토리 내보내기
    print(f"\n💾 분석 히스토리 내보내기...")
    history_file = analyzer.export_analysis_history()
    print(f"   저장 완료: {history_file}")
    
    print("\n✅ 시스템 테스트 완료!")

if __name__ == "__main__":
    main()

# =============================================================================
# 공개 API
# =============================================================================

__all__ = [
    # 메인 클래스들
    'AdvancedStrategyAnalyzer',
    'BaseStrategy',
    'RiskManager',
    'MarketRegimeDetector',
    'PortfolioOptimizer',
    'BacktestEngine',
    'RealTimeMonitor',
    
    # 데이터 클래스들
    'AnalysisResult',
    'MarketData',
    'Decision',
    'StrategyType',
    
    # 구체적인 전략들
    'BuffettStrategy',
    'JesseStrategy',
    'WonyoStrategy',
    'JimRogersStrategy',
    
    # 편의 함수들
    'analyze_coin',
    'get_analyzer',
    'strategy_buffett',
    'strategy_jesse',
    'strategy_wonyo',
    'strategy_jim_rogers',
    
    # 상수
    'DEFAULT_CONFIG',
]