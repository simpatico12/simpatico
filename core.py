#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 최고퀸트프로젝트 - 핵심 실행 엔진 (완전 통합 버전)
=======================================================

전 세계 시장 통합 매매 시스템:
- 🇺🇸 미국 주식 (버핏 + 린치 + 모멘텀 + 기술분석 전략)
- 🇯🇵 일본 주식 (일목균형표 + 엔화 + 기술분석)
- 🪙 암호화폐 (AI 품질평가 + 시장사이클 + 상관관계 최적화)
- 📊 통합 리스크 관리
- 🔔 실시간 알림 시스템
- 📈 성과 추적 및 리포트
- 🔄 요일별 스케줄링 시스템
- 🤖 완전 자동화 (자동선별 + 분석 + 실행)

Author: 최고퀸트팀
Version: 5.0.0 (완전 통합)
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import yaml
import pandas as pd
from dataclasses import dataclass, asdict
import traceback
import numpy as np
import time

# 프로젝트 모듈 import (자동 폴백 처리)
try:
    from us_strategy import AdvancedUSStrategy, USStockSignal, analyze_us, get_us_auto_selection_status, force_us_reselection
    US_STRATEGY_AVAILABLE = True
    logger.info("✅ 미국 주식 전략 모듈 로드 성공")
except ImportError as e:
    print(f"⚠️ 미국 주식 전략 로드 실패: {e}")
    US_STRATEGY_AVAILABLE = False

try:
    from jp_strategy import JPStrategy, JPStockSignal, analyze_jp, get_jp_auto_selection_status, force_jp_reselection
    JP_STRATEGY_AVAILABLE = True
    logger.info("✅ 일본 주식 전략 모듈 로드 성공")
except ImportError as e:
    print(f"⚠️ 일본 주식 전략 로드 실패: {e}")
    JP_STRATEGY_AVAILABLE = False

try:
    from coin_strategy import UltimateCoinStrategy, UltimateCoinSignal, analyze_coin, get_coin_auto_selection_status, force_coin_reselection
    COIN_STRATEGY_AVAILABLE = True
    logger.info("✅ 암호화폐 전략 모듈 로드 성공")
except ImportError as e:
    print(f"⚠️ 암호화폐 전략 로드 실패: {e}")
    COIN_STRATEGY_AVAILABLE = False

# 선택적 모듈들
try:
    from notifier import send_telegram_message, send_trading_alert, send_market_summary
    NOTIFIER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 알림 모듈 로드 실패: {e}")
    NOTIFIER_AVAILABLE = False

try:
    from scheduler import get_today_strategies, is_trading_time
    SCHEDULER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 스케줄러 모듈 로드 실패: {e}")
    SCHEDULER_AVAILABLE = False

try:
    from trading import TradingExecutor, execute_trade_signal
    TRADING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 매매 모듈 로드 실패: {e}")
    TRADING_AVAILABLE = False

# 로깅 설정
def setup_logging():
    """로깅 시스템 설정"""
    # logs 폴더 생성
    os.makedirs('logs', exist_ok=True)
    
    # 로그 파일명 (날짜별)
    log_filename = f"logs/quant_{datetime.now().strftime('%Y%m%d')}.log"
    
    # 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🏆 최고퀸트프로젝트 로깅 시스템 초기화 완료")
    return logger

@dataclass
class UnifiedTradingSignal:
    """통합 매매 신호 데이터 클래스 (모든 시장 대응)"""
    market: str  # 'US', 'JP', 'COIN'
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    strategy: str
    reasoning: str
    target_price: float
    timestamp: datetime
    sector: Optional[str] = None
    
    # 통합 점수 정보
    total_score: float = 0.0
    selection_score: float = 0.0
    
    # 미국 주식 전용
    buffett_score: Optional[float] = None
    lynch_score: Optional[float] = None
    momentum_score: Optional[float] = None
    technical_score: Optional[float] = None
    
    # 일본 주식 전용
    yen_signal: Optional[str] = None
    stock_type: Optional[str] = None  # 'export', 'domestic'
    
    # 암호화폐 전용
    project_quality_score: Optional[float] = None
    market_cycle: Optional[str] = None
    btc_correlation: Optional[float] = None
    
    # 공통 기술적 지표
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    trend: Optional[str] = None
    
    # 분할매매 정보 (통합)
    position_size: Optional[float] = None  # 실제 매매용 포지션 크기
    total_investment: Optional[float] = None  # 총 투자금액
    split_stages: Optional[int] = None  # 분할 단계 수
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_days: Optional[int] = None
    
    additional_data: Optional[Dict] = None

@dataclass
class TradeExecution:
    """매매 실행 결과 데이터 클래스"""
    signal: UnifiedTradingSignal
    executed: bool
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    quantity: Optional[float] = None
    order_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class MarketSummary:
    """시장별 요약 데이터"""
    market: str
    total_analyzed: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    top_picks: List[UnifiedTradingSignal]
    executed_trades: List[TradeExecution]  # 실행된 거래
    analysis_time: float
    errors: List[str]
    is_trading_day: bool  # 오늘 해당 시장 거래일인지
    
    # 시장별 추가 정보
    market_specific_info: Optional[Dict] = None

class QuantTradingEngine:
    """🏆 최고퀸트프로젝트 메인 엔진 (완전 통합 버전)"""
    
    def __init__(self, config_path: str = "settings.yaml", force_test: bool = False):
        """엔진 초기화"""
        self.logger = setup_logging()
        self.config_path = config_path
        self.config = self._load_config()
        self.force_test = force_test  # 🚀 강제 테스트 모드
        
        # 데이터 폴더 생성
        os.makedirs('data', exist_ok=True)
        
        # 전략 객체 초기화
        self.us_strategy = None
        self.jp_strategy = None
        self.coin_strategy = None
        
        # 오늘 실행할 전략 확인 (스케줄링)
        self.today_strategies = self._get_today_strategies()
        
        # 🔥 강제 테스트 모드 처리
        if self.force_test:
            self.logger.info("🧪 강제 테스트 모드 활성화 - 모든 전략 테스트")
            self.today_strategies = ['US', 'JP', 'COIN']
        
        self._initialize_strategies()
        
        # 매매 실행 설정
        self.trading_config = self.config.get('trading', {})
        self.auto_execution = self.trading_config.get('auto_execution', False)
        self.paper_trading = self.trading_config.get('paper_trading', True)
        
        # 매매 실행기 초기화
        self.trading_executor = None
        if TRADING_AVAILABLE and self.auto_execution:
            try:
                self.trading_executor = TradingExecutor(config_path)
                self.logger.info(f"💰 매매 실행기 초기화 완료 (모의거래: {self.paper_trading})")
            except Exception as e:
                self.logger.error(f"❌ 매매 실행기 초기화 실패: {e}")
        
        # 리스크 관리 설정 (시장별 차등 적용)
        self.risk_config = self.config.get('risk_management', {})
        self.max_position_size = self.risk_config.get('max_position_size', 0.1)
        self.stop_loss = self.risk_config.get('stop_loss', -0.05)
        self.take_profit = self.risk_config.get('take_profit', 0.15)
        self.max_daily_trades = self.risk_config.get('max_daily_trades', 30)  # 🚀 증가
        
        # 시장별 포트폴리오 비중
        self.market_allocation = {
            'US': 0.50,    # 50% 미국 주식
            'JP': 0.30,    # 30% 일본 주식  
            'COIN': 0.20   # 20% 암호화폐
        }
        
        # 실행 통계
        self.daily_trades_count = 0
        self.total_signals_generated = 0
        self.session_start_time = datetime.now()
        
        self.logger.info("🚀 최고퀸트프로젝트 엔진 초기화 완료")
        self.logger.info(f"⚙️ 자동매매: {self.auto_execution}, 모의거래: {self.paper_trading}")
        self.logger.info(f"📊 오늘 활성 전략: {len(self.today_strategies)}개 - {self.today_strategies}")
        self.logger.info(f"💼 시장별 비중: 미국{self.market_allocation['US']*100:.0f}% 일본{self.market_allocation['JP']*100:.0f}% 코인{self.market_allocation['COIN']*100:.0f}%")
        
        if self.force_test:
            self.logger.info("🧪 강제 테스트 모드: 스케줄 무시하고 모든 전략 테스트")

    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"✅ 설정 파일 로드 성공: {self.config_path}")
                return config
        except Exception as e:
            self.logger.error(f"❌ 설정 파일 로드 실패: {e}")
            return {}

    def _get_today_strategies(self) -> List[str]:
        """오늘 실행할 전략 목록 조회"""
        if self.force_test:
            return ['US', 'JP', 'COIN']  # 강제 테스트 시 모든 전략
            
        if SCHEDULER_AVAILABLE:
            try:
                strategies = get_today_strategies(self.config)
                return strategies
            except Exception as e:
                self.logger.error(f"❌ 스케줄러 조회 실패: {e}")
        
        # 현재 요일 확인 (기본 스케줄)
        weekday = datetime.now().weekday()  # 0=월요일, 6=일요일
        schedule_config = self.config.get('schedule', {})
        
        day_mapping = {
            0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday',
            4: 'friday', 5: 'saturday', 6: 'sunday'
        }
        
        today_key = day_mapping.get(weekday, 'monday')
        today_strategies = schedule_config.get(today_key, ['US', 'JP', 'COIN'])  # 기본값
        
        self.logger.info(f"📅 오늘({today_key}): {today_strategies if today_strategies else '휴무'}")
        
        return today_strategies

    def _initialize_strategies(self):
        """전략 객체들 초기화 (스케줄링 고려)"""
        try:
            # 미국 주식 전략
            if US_STRATEGY_AVAILABLE and 'US' in self.today_strategies:
                us_config = self.config.get('us_strategy', {})
                if us_config.get('enabled', True):
                    self.us_strategy = AdvancedUSStrategy(self.config_path)
                    self.logger.info("🇺🇸 미국 주식 전략 활성화 (4가지 전략 융합)")
                else:
                    self.logger.info("🇺🇸 미국 주식 전략 설정에서 비활성화")
            elif 'US' not in self.today_strategies:
                self.logger.info("🇺🇸 미국 주식 전략 오늘 비활성화 (스케줄)")
            
            # 일본 주식 전략
            if JP_STRATEGY_AVAILABLE and 'JP' in self.today_strategies:
                jp_config = self.config.get('jp_strategy', {})
                if jp_config.get('enabled', True):
                    self.jp_strategy = JPStrategy(self.config_path)
                    self.logger.info("🇯🇵 일본 주식 전략 활성화 (엔화+기술분석)")
                else:
                    self.logger.info("🇯🇵 일본 주식 전략 설정에서 비활성화")
            elif 'JP' not in self.today_strategies:
                self.logger.info("🇯🇵 일본 주식 전략 오늘 비활성화 (스케줄)")
            
            # 암호화폐 전략
            if COIN_STRATEGY_AVAILABLE and 'COIN' in self.today_strategies:
                coin_config = self.config.get('coin_strategy', {})
                if coin_config.get('enabled', True):
                    self.coin_strategy = UltimateCoinStrategy(self.config_path)
                    self.logger.info("🪙 암호화폐 전략 활성화 (AI 품질평가+시장사이클)")
                else:
                    self.logger.info("🪙 암호화폐 전략 설정에서 비활성화")
            elif 'COIN' not in self.today_strategies:
                self.logger.info("🪙 암호화폐 전략 오늘 비활성화 (스케줄)")
                    
        except Exception as e:
            self.logger.error(f"❌ 전략 초기화 실패: {e}")

    def _check_trading_time(self) -> bool:
        """현재 시간이 거래 시간인지 확인"""
        try:
            if self.force_test:
                return True  # 강제 테스트 시 항상 거래 시간
                
            if SCHEDULER_AVAILABLE:
                return is_trading_time(self.config)
            else:
                # 기본값: 암호화폐는 24시간, 주식은 평일 거래시간
                current_hour = datetime.now().hour
                weekday = datetime.now().weekday()
                
                # 평일이고 9시-16시면 거래 가능
                if weekday < 5 and 9 <= current_hour <= 16:
                    return True
                # 암호화폐는 24시간
                if 'COIN' in self.today_strategies:
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"거래 시간 확인 실패: {e}")
            return True

    def _convert_to_unified_signal(self, signal: Union[USStockSignal, JPStockSignal, UltimateCoinSignal], 
                                 market: str) -> UnifiedTradingSignal:
        """각 전략의 신호를 통합 신호로 변환"""
        try:
            # 공통 필드
            unified = UnifiedTradingSignal(
                market=market,
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                price=signal.price,
                strategy=getattr(signal, 'strategy_source', signal.strategy if hasattr(signal, 'strategy') else 'unknown'),
                reasoning=signal.reasoning,
                target_price=signal.target_price,
                timestamp=signal.timestamp,
                sector=signal.sector if hasattr(signal, 'sector') else None,
                rsi=signal.rsi if hasattr(signal, 'rsi') else None,
                macd_signal=signal.macd_signal if hasattr(signal, 'macd_signal') else None
            )
            
            # 미국 주식 전용 필드
            if market == 'US' and isinstance(signal, USStockSignal):
                unified.buffett_score = signal.buffett_score
                unified.lynch_score = signal.lynch_score
                unified.momentum_score = signal.momentum_score
                unified.technical_score = signal.technical_score
                unified.total_score = signal.total_score
                unified.selection_score = signal.selection_score
                unified.trend = signal.trend
                unified.position_size = signal.total_shares
                unified.total_investment = signal.additional_data.get('total_investment', 0) if signal.additional_data else 0
                unified.split_stages = 3  # 미국 주식은 3단계
                unified.stop_loss = signal.stop_loss
                unified.take_profit = signal.take_profit_2  # 최종 익절가
                unified.max_hold_days = signal.max_hold_days
                
            # 일본 주식 전용 필드
            elif market == 'JP' and isinstance(signal, JPStockSignal):
                unified.yen_signal = signal.yen_signal
                unified.stock_type = signal.stock_type
                unified.total_score = signal.confidence  # 신뢰도를 총점으로 사용
                unified.selection_score = signal.selection_score
                unified.trend = signal.ma_trend
                unified.position_size = signal.position_size
                unified.split_stages = len(signal.split_buy_plan) if signal.split_buy_plan else 0
                unified.stop_loss = signal.stop_loss
                unified.take_profit = signal.take_profit
                unified.max_hold_days = signal.max_hold_days
                
            # 암호화폐 전용 필드
            elif market == 'COIN' and isinstance(signal, UltimateCoinSignal):
                unified.project_quality_score = signal.project_quality_score
                unified.market_cycle = signal.market_cycle
                unified.btc_correlation = signal.correlation_with_btc
                unified.total_score = signal.total_score
                unified.selection_score = signal.confidence  # 신뢰도를 선별점수로 사용
                unified.position_size = signal.total_amount  # 총 투자금액
                unified.split_stages = 5  # 암호화폐는 5단계
                unified.stop_loss = signal.stop_loss
                unified.take_profit = signal.take_profit_3  # 최종 익절가
                unified.max_hold_days = signal.max_hold_days
                
            return unified
            
        except Exception as e:
            self.logger.error(f"신호 변환 실패 {market}-{signal.symbol}: {e}")
            # 기본 신호 반환
            return UnifiedTradingSignal(
                market=market, symbol=signal.symbol, action='hold', confidence=0.0,
                price=getattr(signal, 'price', 0), strategy='conversion_error',
                reasoning=f"신호 변환 실패: {str(e)}", target_price=0,
                timestamp=datetime.now()
            )

    def _apply_risk_management(self, signals: List[UnifiedTradingSignal]) -> List[UnifiedTradingSignal]:
        """통합 리스크 관리 적용"""
        filtered_signals = []
        
        # 일일 거래 제한 체크
        if self.daily_trades_count >= self.max_daily_trades:
            self.logger.warning(f"⚠️ 일일 거래 한도 도달: {self.daily_trades_count}/{self.max_daily_trades}")
            return filtered_signals
        
        # 시장별 신뢰도 기준 (다르게 적용)
        market_thresholds = {
            'US': {'buy': 0.65, 'sell': 0.60},     # 미국: 높은 기준
            'JP': {'buy': 0.60, 'sell': 0.55},     # 일본: 중간 기준  
            'COIN': {'buy': 0.50, 'sell': 0.45}    # 코인: 낮은 기준 (변동성 고려)
        }
        
        for signal in signals:
            market = signal.market
            thresholds = market_thresholds.get(market, {'buy': 0.60, 'sell': 0.55})
            
            if signal.action == 'buy':
                if signal.confidence >= thresholds['buy']:
                    filtered_signals.append(signal)
                    self.logger.info(f"✅ {market} 매수 신호 통과: {signal.symbol} ({signal.confidence:.2f})")
                else:
                    self.logger.debug(f"낮은 신뢰도로 매수 신호 제외: {signal.symbol} ({signal.confidence:.2f})")
                    
            elif signal.action == 'sell':
                if signal.confidence >= thresholds['sell']:
                    filtered_signals.append(signal)
                    self.logger.info(f"✅ {market} 매도 신호 통과: {signal.symbol} ({signal.confidence:.2f})")
                else:
                    self.logger.debug(f"낮은 신뢰도로 매도 신호 제외: {signal.symbol} ({signal.confidence:.2f})")
        
        return filtered_signals

    async def _execute_trades(self, signals: List[UnifiedTradingSignal]) -> List[TradeExecution]:
        """매매 신호 실행"""
        executed_trades = []
        
        if not self.trading_executor or not self.auto_execution:
            self.logger.info("📊 매매 신호만 생성 (실행 비활성화)")
            # 신호만 생성하고 실행하지 않음
            for signal in signals:
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="자동 매매 비활성화"
                ))
            return executed_trades
        
        # 거래 시간 체크
        if not self._check_trading_time():
            self.logger.info("⏰ 거래 시간이 아님 - 신호만 생성")
            for signal in signals:
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="거래 시간 아님"
                ))
            return executed_trades
        
        # 일일 거래 한도 체크
        if self.daily_trades_count >= self.max_daily_trades:
            self.logger.warning(f"⚠️ 일일 거래 한도 도달: {self.daily_trades_count}/{self.max_daily_trades}")
            for signal in signals:
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="일일 거래 한도 초과"
                ))
            return executed_trades
        
        # 실제 매매 실행 (모의거래)
        for signal in signals:
            if signal.action in ['buy', 'sell']:
                try:
                    self.logger.info(f"💰 {signal.action.upper()} 주문 실행 (모의): {signal.market}-{signal.symbol}")
                    
                    # 시장별 수량 계산
                    if signal.market == 'US':
                        quantity = signal.position_size or 100  # 주식 수
                    elif signal.market == 'JP':
                        quantity = signal.position_size or 100  # 주식 수
                    elif signal.market == 'COIN':
                        quantity = (signal.position_size or 1000000) / signal.price  # 코인 개수
                    else:
                        quantity = 100
                    
                    # 모의 실행 결과 생성
                    execution_result = {
                        'success': True,
                        'price': signal.price,
                        'quantity': quantity,
                        'order_id': f"TEST_{signal.market}_{datetime.now().strftime('%H%M%S')}"
                    }
                    
                    executed_trades.append(TradeExecution(
                        signal=signal,
                        executed=True,
                        execution_price=execution_result['price'],
                        execution_time=datetime.now(),
                        quantity=execution_result['quantity'],
                        order_id=execution_result['order_id']
                    ))
                    
                    self.daily_trades_count += 1
                    
                    # 실행 알림 발송
                    if NOTIFIER_AVAILABLE:
                        await send_trading_alert(
                            signal.market, signal.symbol, signal.action,
                            execution_result['price'], signal.confidence, 
                            f"✅ 모의 매매 완료: {signal.reasoning[:100]}",
                            signal.target_price
                        )
                    
                    self.logger.info(f"✅ 모의 매매 완료: {signal.market}-{signal.symbol} {signal.action}")
                        
                except Exception as e:
                    executed_trades.append(TradeExecution(
                        signal=signal,
                        executed=False,
                        error_message=str(e)
                    ))
                    self.logger.error(f"❌ 매매 실행 중 오류 {signal.symbol}: {e}")
                    
                # API 호출 제한 고려
                await asyncio.sleep(0.3)
            else:
                # hold 신호는 실행하지 않음
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="HOLD 신호"
                ))
        
        return executed_trades

    async def analyze_us_market(self) -> MarketSummary:
        """🇺🇸 미국 시장 분석"""
        start_time = datetime.now()
        errors = []
        signals = []
        executed_trades = []
        
        if not self.us_strategy:
            return MarketSummary(
                market='US', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], analysis_time=0.0, 
                errors=['전략 비활성화'], is_trading_day='US' in self.today_strategies
            )
        
        try:
            self.logger.info("🔍 미국 시장 분석 시작 (4가지 전략 융합)...")
            
            # 전체 시장 스캔 (자동선별 + 분석)
            us_signals = await self.us_strategy.scan_all_selected_stocks()
            
            # UnifiedTradingSignal 형태로 변환
            for signal in us_signals:
                unified_signal = self._convert_to_unified_signal(signal, 'US')
                signals.append(unified_signal)
            
            # 리스크 관리 적용
            filtered_signals = self._apply_risk_management(signals)
            
            # 매매 실행 (매수/매도 신호만)
            trade_signals = [s for s in filtered_signals if s.action in ['buy', 'sell']]
            if trade_signals:
                executed_trades = await self._execute_trades(trade_signals)
            
            # 통계 계산
            buy_signals = len([s for s in signals if s.action == 'buy'])
            sell_signals = len([s for s in signals if s.action == 'sell'])
            hold_signals = len([s for s in signals if s.action == 'hold'])
            
            # 상위 종목 선정 (매수 신호 중 신뢰도 높은 순)
            top_picks = sorted([s for s in signals if s.action == 'buy'], 
                             key=lambda x: x.confidence, reverse=True)[:5]
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # 미국 시장 특화 정보
            market_specific_info = {
                'vix_level': self.us_strategy.stock_selector.current_vix if hasattr(self.us_strategy, 'stock_selector') else 0,
                'avg_buffett_score': np.mean([s.buffett_score for s in signals if s.buffett_score is not None]),
                'avg_lynch_score': np.mean([s.lynch_score for s in signals if s.lynch_score is not None]),
                'avg_momentum_score': np.mean([s.momentum_score for s in signals if s.momentum_score is not None]),
                'avg_technical_score': np.mean([s.technical_score for s in signals if s.technical_score is not None]),
                'sp500_count': len([s for s in signals if s.additional_data and 'S&P500' in s.additional_data.get('index_membership', [])]),
                'nasdaq_count': len([s for s in signals if s.additional_data and 'NASDAQ100' in s.additional_data.get('index_membership', [])])
            }
            
            self.logger.info(f"✅ 미국 시장 분석 완료 - 매수:{buy_signals}, 매도:{sell_signals}, 보유:{hold_signals}")
            if executed_trades:
                executed_count = len([t for t in executed_trades if t.executed])
                self.logger.info(f"💰 실행된 거래: {executed_count}개")
            
            return MarketSummary(
                market='US',
                total_analyzed=len(signals),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                top_picks=top_picks,
                executed_trades=executed_trades,
                analysis_time=analysis_time,
                errors=errors,
                is_trading_day='US' in self.today_strategies,
                market_specific_info=market_specific_info
            )
            
        except Exception as e:
            error_msg = f"미국 시장 분석 실패: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
            
            return MarketSummary(
                market='US', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], 
                analysis_time=(datetime.now() - start_time).total_seconds(),
                errors=errors, is_trading_day='US' in self.today_strategies
            )

    async def analyze_jp_market(self) -> MarketSummary:
        """🇯🇵 일본 시장 분석"""
        start_time = datetime.now()
        errors = []
        signals = []
        executed_trades = []
        
        if not self.jp_strategy:
            return MarketSummary(
                market='JP', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], analysis_time=0.0, 
                errors=['전략 비활성화'], is_trading_day='JP' in self.today_strategies
            )
        
        try:
            self.logger.info("🔍 일본 시장 분석 시작 (엔화+기술분석)...")
            
            # 전체 시장 스캔 (자동선별 + 분석)
            jp_signals = await self.jp_strategy.scan_all_symbols()
            
            # UnifiedTradingSignal 형태로 변환
            for signal in jp_signals:
                unified_signal = self._convert_to_unified_signal(signal, 'JP')
                signals.append(unified_signal)
            
            # 리스크 관리 적용
            filtered_signals = self._apply_risk_management(signals)
            
            # 매매 실행 (매수/매도 신호만)
            trade_signals = [s for s in filtered_signals if s.action in ['buy', 'sell']]
            if trade_signals:
                executed_trades = await self._execute_trades(trade_signals)
            
            # 통계 계산
            buy_signals = len([s for s in signals if s.action == 'buy'])
            sell_signals = len([s for s in signals if s.action == 'sell'])
            hold_signals = len([s for s in signals if s.action == 'hold'])
            
            # 상위 종목 선정
            top_picks = sorted([s for s in signals if s.action == 'buy'], 
                             key=lambda x: x.confidence, reverse=True)[:5]
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # 일본 시장 특화 정보
            market_specific_info = {
                'usd_jpy_rate': self.jp_strategy.current_usd_jpy if hasattr(self.jp_strategy, 'current_usd_jpy') else 0,
                'yen_signal': self.jp_strategy._get_yen_signal() if hasattr(self.jp_strategy, '_get_yen_signal') else 'neutral',
                'export_stocks': len([s for s in signals if s.stock_type == 'export']),
                'domestic_stocks': len([s for s in signals if s.stock_type == 'domestic']),
                'mixed_stocks': len([s for s in signals if s.stock_type == 'mixed']),
                'avg_selection_score': np.mean([s.selection_score for s in signals if s.selection_score is not None])
            }
            
            self.logger.info(f"✅ 일본 시장 분석 완료 - 매수:{buy_signals}, 매도:{sell_signals}, 보유:{hold_signals}")
            if executed_trades:
                executed_count = len([t for t in executed_trades if t.executed])
                self.logger.info(f"💰 실행된 거래: {executed_count}개")
            
            return MarketSummary(
                market='JP',
                total_analyzed=len(signals),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                top_picks=top_picks,
                executed_trades=executed_trades,
                analysis_time=analysis_time,
                errors=errors,
                is_trading_day='JP' in self.today_strategies,
                market_specific_info=market_specific_info
            )
            
        except Exception as e:
            error_msg = f"일본 시장 분석 실패: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
            
            return MarketSummary(
                market='JP', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], 
                analysis_time=(datetime.now() - start_time).total_seconds(),
                errors=errors, is_trading_day='JP' in self.today_strategies
            )

    async def analyze_coin_market(self) -> MarketSummary:
        """🪙 암호화폐 시장 분석"""
        start_time = datetime.now()
        errors = []
        signals = []
        executed_trades = []
        
        if not self.coin_strategy:
            return MarketSummary(
                market='COIN', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], analysis_time=0.0, 
                errors=['전략 비활성화'], is_trading_day='COIN' in self.today_strategies
            )
        
        try:
            self.logger.info("🔍 암호화폐 시장 분석 시작 (AI 품질평가+시장사이클)...")
            
            # 전체 시장 스캔 (자동선별 + 분석)
            coin_signals = await self.coin_strategy.scan_all_selected_coins()
            
            # UnifiedTradingSignal 형태로 변환
            for signal in coin_signals:
                unified_signal = self._convert_to_unified_signal(signal, 'COIN')
                signals.append(unified_signal)
            
            # 리스크 관리 적용
            filtered_signals = self._apply_risk_management(signals)
            
            # 매매 실행 (매수/매도 신호만)
            trade_signals = [s for s in filtered_signals if s.action in ['buy', 'sell']]
            if trade_signals:
                executed_trades = await self._execute_trades(trade_signals)
            
            # 통계 계산
            buy_signals = len([s for s in signals if s.action == 'buy'])
            sell_signals = len([s for s in signals if s.action == 'sell'])
            hold_signals = len([s for s in signals if s.action == 'hold'])
            
            # 상위 종목 선정 (매수 신호 중 신뢰도 높은 순)
            top_picks = sorted([s for s in signals if s.action == 'buy'], 
                             key=lambda x: x.confidence, reverse=True)[:5]
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # 암호화폐 시장 특화 정보
            market_specific_info = {
                'market_cycle': self.coin_strategy.current_market_cycle if hasattr(self.coin_strategy, 'current_market_cycle') else 'sideways',
                'cycle_confidence': self.coin_strategy.cycle_confidence if hasattr(self.coin_strategy, 'cycle_confidence') else 0.5,
                'avg_project_quality': np.mean([s.project_quality_score for s in signals if s.project_quality_score is not None]),
                'avg_btc_correlation': np.mean([s.btc_correlation for s in signals if s.btc_correlation is not None]),
                'tier_1_count': len([s for s in signals if s.additional_data and s.additional_data.get('tier') == 'tier_1']),
                'defi_count': len([s for s in signals if s.sector and 'DeFi' in s.sector]),
                'l1_count': len([s for s in signals if s.sector and 'L1_Blockchain' in s.sector])
            }
            
            self.logger.info(f"✅ 암호화폐 시장 분석 완료 - 매수:{buy_signals}, 매도:{sell_signals}, 보유:{hold_signals}")
            if errors:
                self.logger.warning(f"⚠️ 분석 중 오류 {len(errors)}개")
            if executed_trades:
                executed_count = len([t for t in executed_trades if t.executed])
                self.logger.info(f"💰 실행된 거래: {executed_count}개")
            
            return MarketSummary(
                market='COIN',
                total_analyzed=len(signals),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                top_picks=top_picks,
                executed_trades=executed_trades,
                analysis_time=analysis_time,
                errors=errors,
                is_trading_day='COIN' in self.today_strategies,
                market_specific_info=market_specific_info
            )
            
        except Exception as e:
            error_msg = f"암호화폐 시장 분석 실패: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
            
            return MarketSummary(
                market='COIN', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], 
                analysis_time=(datetime.now() - start_time).total_seconds(),
                errors=errors, is_trading_day='COIN' in self.today_strategies
            )

    async def run_full_analysis(self) -> Dict[str, MarketSummary]:
        """🌍 전체 시장 통합 분석"""
        self.logger.info("🚀 전체 시장 통합 분석 시작...")
        start_time = datetime.now()
        
        # 병렬로 모든 시장 분석
        tasks = []
        
        if self.us_strategy:
            tasks.append(self.analyze_us_market())
        if self.jp_strategy:
            tasks.append(self.analyze_jp_market())
        if self.coin_strategy:
            tasks.append(self.analyze_coin_market())
        
        if not tasks:
            self.logger.warning("⚠️ 활성화된 전략이 없습니다")
            return {}
        
        # 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 정리
        market_summaries = {}
        total_signals = 0
        total_buy_signals = 0
        total_executed = 0
        
        for result in results:
            if isinstance(result, MarketSummary):
                market_summaries[result.market] = result
                total_signals += result.total_analyzed
                total_buy_signals += result.buy_signals
                executed_count = len([t for t in result.executed_trades if t.executed])
                total_executed += executed_count
            elif isinstance(result, Exception):
                self.logger.error(f"❌ 시장 분석 중 오류: {result}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        self.total_signals_generated += total_signals
        
        self.logger.info(f"🎯 전체 분석 완료 - {len(market_summaries)}개 시장, "
                        f"총 {total_signals}개 신호, 매수 {total_buy_signals}개, "
                        f"실행 {total_executed}개, 소요시간: {total_time:.1f}초")
        
        # 결과 저장
        await self._save_analysis_results(market_summaries)
        
        # 알림 발송
        if NOTIFIER_AVAILABLE:
            await self._send_analysis_notification(market_summaries)
        
        return market_summaries

    async def _save_analysis_results(self, market_summaries: Dict[str, MarketSummary]):
        """분석 결과 저장"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/analysis_{timestamp}.json"
            
            # 직렬화 가능한 형태로 변환
            save_data = {
                'timestamp': timestamp,
                'session_info': {
                    'start_time': self.session_start_time.isoformat(),
                    'total_signals_generated': self.total_signals_generated,
                    'daily_trades_count': self.daily_trades_count,
                    'today_strategies': self.today_strategies,
                    'force_test_mode': self.force_test,
                    'market_allocation': self.market_allocation
                },
                'market_summaries': {}
            }
            
            for market, summary in market_summaries.items():
                # top_picks 직렬화
                top_picks_data = []
                for signal in summary.top_picks:
                    signal_dict = asdict(signal)
                    # datetime 객체를 문자열로 변환
                    signal_dict['timestamp'] = signal.timestamp.isoformat()
                    top_picks_data.append(signal_dict)
                
                save_data['market_summaries'][market] = {
                    'market': summary.market,
                    'total_analyzed': summary.total_analyzed,
                    'buy_signals': summary.buy_signals,
                    'sell_signals': summary.sell_signals,
                    'hold_signals': summary.hold_signals,
                    'analysis_time': summary.analysis_time,
                    'errors': summary.errors,
                    'is_trading_day': summary.is_trading_day,
                    'top_picks': top_picks_data,
                    'executed_trades_count': len([t for t in summary.executed_trades if t.executed]),
                    'total_executed_trades': len(summary.executed_trades),
                    'market_specific_info': summary.market_specific_info
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"📊 분석 결과 저장 완료: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ 분석 결과 저장 실패: {e}")

    async def _send_analysis_notification(self, market_summaries: Dict[str, MarketSummary]):
        """분석 결과 알림 발송"""
        try:
            # 알림 설정 확인
            notification_config = self.config.get('notifications', {})
            if not notification_config.get('telegram', {}).get('enabled', False):
                return
            
            # 통합 요약 알림 발송
            await send_market_summary(market_summaries)
            
            self.logger.info("📱 알림 발송 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 알림 발송 실패: {e}")

    async def get_quick_analysis(self, symbols: List[str]) -> List[UnifiedTradingSignal]:
        """빠른 개별 종목 분석"""
        signals = []
        
        for symbol in symbols:
            try:
                # 시장 판별 (간단한 방식)
                if symbol.endswith('.T'):
                    # 일본 주식
                    if self.jp_strategy:
                        result = await analyze_jp(symbol)
                        signal = UnifiedTradingSignal(
                            market='JP', symbol=symbol, action=result['decision'],
                            confidence=result['confidence_score']/100, price=result['current_price'],
                            strategy='jp_quick', reasoning=result['reasoning'],
                            target_price=result.get('target_price', result['current_price']), 
                            timestamp=datetime.now(),
                            yen_signal=result.get('yen_signal'),
                            stock_type=result.get('stock_type'),
                            sector=result.get('sector')
                        )
                        signals.append(signal)
                        
                elif '-' in symbol and 'KRW' in symbol:
                    # 암호화폐
                    if self.coin_strategy:
                        result = await analyze_coin(symbol)
                        signal = UnifiedTradingSignal(
                            market='COIN', symbol=symbol, action=result['decision'],
                            confidence=result['confidence_score']/100, price=result['price'],
                            strategy='coin_quick', reasoning=result['reasoning'],
                            target_price=result.get('target_price', result['price']),
                            timestamp=datetime.now(),
                            project_quality_score=result.get('project_quality_score', 0)/100,
                            market_cycle=result.get('market_cycle'),
                            btc_correlation=result.get('btc_correlation'),
                            sector=result.get('sector')
                        )
                        signals.append(signal)
                        
                else:
                    # 미국 주식
                    if self.us_strategy:
                        result = await analyze_us(symbol)
                        signal = UnifiedTradingSignal(
                            market='US', symbol=symbol, action=result['decision'],
                            confidence=result['confidence_score']/100, price=result['price'],
                            strategy='us_quick', reasoning=result['reasoning'],
                            target_price=result.get('target_price', result['price']),
                            timestamp=datetime.now(),
                            buffett_score=result.get('buffett_score', 0)/100,
                            lynch_score=result.get('lynch_score', 0)/100,
                            momentum_score=result.get('momentum_score', 0)/100,
                            technical_score=result.get('technical_score', 0)/100,
                            sector=result.get('sector')
                        )
                        signals.append(signal)
                        
            except Exception as e:
                self.logger.error(f"❌ {symbol} 빠른 분석 실패: {e}")
        
        return signals

    async def generate_unified_portfolio_report(self, market_summaries: Dict[str, MarketSummary]) -> Dict:
        """📊 통합 포트폴리오 리포트 생성"""
        if not market_summaries:
            return {"error": "분석된 시장이 없습니다"}
        
        # 전체 통계
        total_analyzed = sum([summary.total_analyzed for summary in market_summaries.values()])
        total_buy_signals = sum([summary.buy_signals for summary in market_summaries.values()])
        total_sell_signals = sum([summary.sell_signals for summary in market_summaries.values()])
        total_hold_signals = sum([summary.hold_signals for summary in market_summaries.values()])
        total_executed = sum([len([t for t in summary.executed_trades if t.executed]) for summary in market_summaries.values()])
        
        # 시장별 성과
        market_performance = {}
        top_picks_unified = []
        
        for market, summary in market_summaries.items():
            market_performance[market] = {
                'analyzed': summary.total_analyzed,
                'buy_signals': summary.buy_signals,
                'sell_signals': summary.sell_signals,
                'buy_rate': summary.buy_signals / summary.total_analyzed * 100 if summary.total_analyzed > 0 else 0,
                'executed_trades': len([t for t in summary.executed_trades if t.executed]),
                'analysis_time': summary.analysis_time,
                'errors': len(summary.errors),
                'specific_info': summary.market_specific_info
            }
            
            # 상위 종목들 통합
            top_picks_unified.extend(summary.top_picks)
        
        # 전체 상위 종목 (시장 구분 없이 신뢰도 순)
        top_picks_unified.sort(key=lambda x: x.confidence, reverse=True)
        global_top_picks = top_picks_unified[:10]
        
        # 섹터별 분포 (전체)
        sector_distribution = {}
        for summary in market_summaries.values():
            for signal in summary.top_picks:
                sector = signal.sector or 'Unknown'
                sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
        
        # 리스크 메트릭스
        avg_confidence = np.mean([signal.confidence for summary in market_summaries.values() for signal in summary.top_picks])
        
        # 다양성 점수
        diversification_score = len(market_summaries) / 3  # 최대 3개 시장
        sector_diversity = len(sector_distribution) / max(1, total_buy_signals)
        
        report = {
            'summary': {
                'total_markets': len(market_summaries),
                'active_strategies': self.today_strategies,
                'total_analyzed': total_analyzed,
                'total_buy_signals': total_buy_signals,
                'total_sell_signals': total_sell_signals,
                'total_hold_signals': total_hold_signals,
                'total_executed': total_executed,
                'overall_buy_rate': total_buy_signals / total_analyzed * 100 if total_analyzed > 0 else 0,
                'session_duration': (datetime.now() - self.session_start_time).total_seconds() / 3600,
                'daily_trades_count': self.daily_trades_count,
                'max_daily_trades': self.max_daily_trades
            },
            'market_performance': market_performance,
            'global_top_picks': [
                {
                    'market': signal.market,
                    'symbol': signal.symbol,
                    'sector': signal.sector,
                    'confidence': signal.confidence,
                    'total_score': signal.total_score or signal.confidence,
                    'price': signal.price,
                    'target_price': signal.target_price,
                    'strategy': signal.strategy,
                    'reasoning': signal.reasoning[:100] + "..." if len(signal.reasoning) > 100 else signal.reasoning,
                    
                    # 시장별 특수 정보
                    'market_specific': {
                        'buffett_score': signal.buffett_score,
                        'lynch_score': signal.lynch_score,
                        'yen_signal': signal.yen_signal,
                        'stock_type': signal.stock_type,
                        'project_quality_score': signal.project_quality_score,
                        'market_cycle': signal.market_cycle,
                        'btc_correlation': signal.btc_correlation
                    }
                }
                for signal in global_top_picks
            ],
            'diversification_analysis': {
                'market_diversification': diversification_score,
                'sector_diversification': sector_diversity,
                'sector_distribution': sector_distribution,
                'market_allocation': self.market_allocation
            },
            'risk_metrics': {
                'avg_confidence': avg_confidence,
                'max_position_size': self.max_position_size,
                'daily_trades_utilization': self.daily_trades_count / self.max_daily_trades * 100,
                'error_rate': sum([len(summary.errors) for summary in market_summaries.values()]) / len(market_summaries) * 100,
                'auto_execution': self.auto_execution,
                'paper_trading': self.paper_trading
            },
            'system_info': {
                'version': '5.0.0',
                'force_test_mode': self.force_test,
                'available_strategies': {
                    'US': US_STRATEGY_AVAILABLE,
                    'JP': JP_STRATEGY_AVAILABLE,
                    'COIN': COIN_STRATEGY_AVAILABLE
                },
                'available_modules': {
                    'notifier': NOTIFIER_AVAILABLE,
                    'scheduler': SCHEDULER_AVAILABLE,
                    'trading': TRADING_AVAILABLE
                },
                'last_analysis_time': datetime.now().isoformat()
            }
        }
        
        return report

    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        uptime = (datetime.now() - self.session_start_time).total_seconds()
        
        return {
            'system_status': 'running',
            'version': '5.0.0',
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'strategies_enabled': {
                'us_strategy': self.us_strategy is not None,
                'jp_strategy': self.jp_strategy is not None,
                'coin_strategy': self.coin_strategy is not None
            },
            'today_strategies': self.today_strategies,
            'daily_trades_count': self.daily_trades_count,
            'total_signals_generated': self.total_signals_generated,
            'max_daily_trades': self.max_daily_trades,
            'auto_execution': self.auto_execution,
            'paper_trading': self.paper_trading,
            'force_test_mode': self.force_test,
            'session_start_time': self.session_start_time.isoformat(),
            'last_config_load': self.config_path,
            'current_time': datetime.now().isoformat(),
            'current_weekday': datetime.now().strftime('%A'),
            'market_allocation': self.market_allocation,
            'risk_settings': {
                'max_position_size': self.max_position_size,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'max_daily_trades': self.max_daily_trades
            },
            'module_status': {
                'us_strategy_available': US_STRATEGY_AVAILABLE,
                'jp_strategy_available': JP_STRATEGY_AVAILABLE,
                'coin_strategy_available': COIN_STRATEGY_AVAILABLE,
                'notifier_available': NOTIFIER_AVAILABLE,
                'scheduler_available': SCHEDULER_AVAILABLE,
                'trading_available': TRADING_AVAILABLE
            }
        }

    async def force_reselection_all_markets(self) -> Dict[str, List[str]]:
        """모든 시장 강제 재선별"""
        results = {}
        
        try:
            if self.us_strategy and US_STRATEGY_AVAILABLE:
                us_symbols = await force_us_reselection()
                results['US'] = us_symbols
                self.logger.info(f"🇺🇸 미국 주식 재선별 완료: {len(us_symbols)}개")
        except Exception as e:
            self.logger.error(f"❌ 미국 주식 재선별 실패: {e}")
            results['US'] = []

        try:
            if self.jp_strategy and JP_STRATEGY_AVAILABLE:
                jp_symbols = await force_jp_reselection()
                results['JP'] = jp_symbols
                self.logger.info(f"🇯🇵 일본 주식 재선별 완료: {len(jp_symbols)}개")
        except Exception as e:
            self.logger.error(f"❌ 일본 주식 재선별 실패: {e}")
            results['JP'] = []

        try:
            if self.coin_strategy and COIN_STRATEGY_AVAILABLE:
                coin_symbols = await force_coin_reselection()
                results['COIN'] = coin_symbols
                self.logger.info(f"🪙 암호화폐 재선별 완료: {len(coin_symbols)}개")
        except Exception as e:
            self.logger.error(f"❌ 암호화폐 재선별 실패: {e}")
            results['COIN'] = []

        return results

    async def get_auto_selection_status_all(self) -> Dict[str, Dict]:
        """모든 시장의 자동선별 상태 조회"""
        status = {}
        
        try:
            if US_STRATEGY_AVAILABLE:
                status['US'] = await get_us_auto_selection_status()
        except Exception as e:
            self.logger.error(f"❌ 미국 주식 상태 조회 실패: {e}")
            status['US'] = {'error': str(e)}

        try:
            if JP_STRATEGY_AVAILABLE:
                status['JP'] = await get_jp_auto_selection_status()
        except Exception as e:
            self.logger.error(f"❌ 일본 주식 상태 조회 실패: {e}")
            status['JP'] = {'error': str(e)}

        try:
            if COIN_STRATEGY_AVAILABLE:
                status['COIN'] = await get_coin_auto_selection_status()
        except Exception as e:
            self.logger.error(f"❌ 암호화폐 상태 조회 실패: {e}")
            status['COIN'] = {'error': str(e)}

        return status

# ========================================================================================
# 🎯 편의 함수들 (외부에서 쉽게 사용)
# ========================================================================================

async def run_single_analysis(force_test: bool = False):
    """단일 분석 실행"""
    engine = QuantTradingEngine(force_test=force_test)
    results = await engine.run_full_analysis()
    return results

async def run_full_system_analysis(force_test: bool = False):
    """전체 시스템 분석 + 리포트 생성"""
    engine = QuantTradingEngine(force_test=force_test)
    market_results = await engine.run_full_analysis()
    
    if market_results:
        unified_report = await engine.generate_unified_portfolio_report(market_results)
        return market_results, unified_report
    else:
        return {}, {}

async def analyze_symbols(symbols: List[str]):
    """특정 종목들 분석"""
    engine = QuantTradingEngine()
    signals = await engine.get_quick_analysis(symbols)
    return signals

def get_engine_status():
    """엔진 상태 조회"""
    engine = QuantTradingEngine()
    return engine.get_system_status()

async def force_reselection_all():
    """모든 시장 강제 재선별"""
    engine = QuantTradingEngine()
    return await engine.force_reselection_all_markets()

async def get_all_selection_status():
    """모든 시장 선별 상태 조회"""
    engine = QuantTradingEngine()
    return await engine.get_auto_selection_status_all()

# ========================================================================================
# 🧪 메인 실행 함수 (완전 통합 버전)
# ========================================================================================

async def main():
    """메인 실행 함수 (완전 통합 시스템)"""
    try:
        print("🏆 최고퀸트프로젝트 - 완전 통합 시스템 V5.0!")
        print("=" * 80)
        print("🌍 전 세계 시장 통합 매매 시스템:")
        print("  🇺🇸 미국 주식 (버핏+린치+모멘텀+기술분석)")
        print("  🇯🇵 일본 주식 (엔화+기술분석+자동선별)")
        print("  🪙 암호화폐 (AI품질평가+시장사이클+상관관계)")
        print("  🤖 완전 자동화 (자동선별+분석+실행+알림)")
        print("=" * 80)
        
        # 🚀 강제 테스트 모드 옵션
        import sys
        force_test = '--test' in sys.argv or '--force' in sys.argv
        
        if force_test:
            print("🧪 강제 테스트 모드 활성화")
        
        # 엔진 초기화
        engine = QuantTradingEngine(force_test=force_test)
        
        # 시스템 상태 출력
        status = engine.get_system_status()
        print(f"\n💻 시스템 상태:")
        print(f"  버전: {status['version']}")
        print(f"  가동시간: {status['uptime_hours']:.1f}시간")
        print(f"  활성화된 전략: {sum(status['strategies_enabled'].values())}개")
        print(f"  오늘 실행 전략: {status['today_strategies']}")
        print(f"  일일 거래: {status['daily_trades_count']}/{status['max_daily_trades']}")
        print(f"  자동매매: {status['auto_execution']} (모의거래: {status['paper_trading']})")
        
        # 모듈 가용성 확인
        print(f"\n🔧 모듈 상태:")
        modules = status['module_status']
        for module, available in modules.items():
            status_icon = "✅" if available else "❌"
            print(f"  {status_icon} {module}")
        
        if force_test:
            print(f"  🧪 강제 테스트: {status['force_test_mode']}")
        
        # 자동선별 상태 확인
        print(f"\n📋 자동선별 상태 확인...")
        selection_status = await engine.get_auto_selection_status_all()
        
        for market, info in selection_status.items():
            if 'error' in info:
                print(f"  ❌ {market}: {info['error']}")
            else:
                print(f"  ✅ {market}: 선별 {info.get('selected_count', 0)}개, "
                      f"캐시 {'유효' if info.get('cache_valid', False) else '무효'}")
        
        print()
        
        # 전체 시장 통합 분석 실행
        print(f"🔍 전체 시장 통합 분석 시작...")
        start_time = time.time()
        
        market_results, unified_report = await run_full_system_analysis(force_test)
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ 총 소요 시간: {elapsed_time:.1f}초")
        
        if market_results and unified_report:
            # 전체 요약
            summary = unified_report['summary']
            print(f"\n🎯 통합 분석 결과:")
            print(f"  분석 시장: {summary['total_markets']}개")
            print(f"  총 분석 종목: {summary['total_analyzed']}개")
            print(f"  매수 신호: {summary['total_buy_signals']}개")
            print(f"  매도 신호: {summary['total_sell_signals']}개")
            print(f"  실행된 거래: {summary['total_executed']}개")
            print(f"  전체 매수율: {summary['overall_buy_rate']:.1f}%")
            
            # 시장별 성과
            print(f"\n📊 시장별 성과:")
            performance = unified_report['market_performance']
            for market, perf in performance.items():
                market_name = {'US': '🇺🇸 미국', 'JP': '🇯🇵 일본', 'COIN': '🪙 코인'}.get(market, market)
                print(f"  {market_name}: 분석 {perf['analyzed']}개, "
                      f"매수 {perf['buy_signals']}개 ({perf['buy_rate']:.1f}%), "
                      f"실행 {perf['executed_trades']}개 ({perf['analysis_time']:.1f}초)")
                
                # 시장별 특수 정보
                specific = perf.get('specific_info', {})
                if market == 'US' and specific:
                    print(f"    VIX: {specific.get('vix_level', 0):.1f}, "
                          f"버핏: {specific.get('avg_buffett_score', 0):.2f}, "
                          f"린치: {specific.get('avg_lynch_score', 0):.2f}")
                elif market == 'JP' and specific:
                    print(f"    USD/JPY: {specific.get('usd_jpy_rate', 0):.2f}, "
                          f"엔화신호: {specific.get('yen_signal', 'unknown')}, "
                          f"수출주: {specific.get('export_stocks', 0)}개")
                elif market == 'COIN' and specific:
                    print(f"    사이클: {specific.get('market_cycle', 'unknown')}, "
                          f"품질: {specific.get('avg_project_quality', 0):.2f}, "
                          f"DeFi: {specific.get('defi_count', 0)}개")
            
            # 글로벌 상위 추천
            top_picks = unified_report['global_top_picks']
            if top_picks:
                print(f"\n🏆 글로벌 상위 추천 (전 시장 통합):")
                for i, pick in enumerate(top_picks[:5], 1):
                    market_emoji = {'US': '🇺🇸', 'JP': '🇯🇵', 'COIN': '🪙'}.get(pick['market'], '❓')
                    print(f"\n  {i}. {market_emoji} {pick['symbol']} ({pick['sector']})")
                    print(f"     신뢰도: {pick['confidence']:.2%} | 전략: {pick['strategy']}")
                    print(f"     현재가: {pick['price']:.2f} → 목표가: {pick['target_price']:.2f}")
                    
                    # 시장별 특수 점수
                    ms = pick['market_specific']
                    if pick['market'] == 'US':
                        print(f"     점수: 버핏{ms['buffett_score'] or 0:.2f} 린치{ms['lynch_score'] or 0:.2f}")
                    elif pick['market'] == 'JP':
                        print(f"     엔화: {ms['yen_signal']} | 타입: {ms['stock_type']}")
                    elif pick['market'] == 'COIN':
                        print(f"     품질: {ms['project_quality_score'] or 0:.2f} | 사이클: {ms['market_cycle']}")
                    
                    print(f"     📝 {pick['reasoning'][:80]}...")
            
            # 다양성 분석
            diversity = unified_report['diversification_analysis']
            print(f"\n🎨 포트폴리오 다양성:")
            print(f"  시장 다양성: {diversity['market_diversification']:.2f}")
            print(f"  섹터 다양성: {diversity['sector_diversification']:.2f}")
            print(f"  시장 비중: 미국{diversity['market_allocation']['US']*100:.0f}% "
                  f"일본{diversity['market_allocation']['JP']*100:.0f}% "
                  f"코인{diversity['market_allocation']['COIN']*100:.0f}%")
            
            # 리스크 메트릭스
            risk = unified_report['risk_metrics']
            print(f"\n🛡️ 리스크 관리:")
            print(f"  평균 신뢰도: {risk['avg_confidence']:.2%}")
            print(f"  일일 거래 활용: {risk['daily_trades_utilization']:.1f}%")
            print(f"  오류율: {risk['error_rate']:.1f}%")
            print(f"  자동매매: {risk['auto_execution']} (모의거래: {risk['paper_trading']})")
            
            # 시스템 정보
            system = unified_report['system_info']
            print(f"\n🔧 시스템 정보:")
            print(f"  버전: {system['version']}")
            print(f"  강제테스트: {system['force_test_mode']}")
            strategies = system['available_strategies']
            print(f"  전략: US{'✅' if strategies['US'] else '❌'} "
                  f"JP{'✅' if strategies['JP'] else '❌'} "
                  f"COIN{'✅' if strategies['COIN'] else '❌'}")
        
        else:
            print("❌ 분석 결과가 없습니다.")
        
        print("\n✅ 완전 통합 분석 완료!")
        print("\n🎯 최고퀸트프로젝트 V5.0 완전 통합 시스템 특징:")
        print("  ✅ 🌍 전 세계 3개 시장 완전 통합 (미국+일본+암호화폐)")
        print("  ✅ 🤖 완전 자동화 (자동선별+분석+실행+알림)")
        print("  ✅ 📊 시장별 최적화 전략")
        print("  ✅ 🛡️ 통합 리스크 관리")
        print("  ✅ 📱 실시간 알림 시스템")
        print("  ✅ 🔄 요일별 스케줄링")
        print("  ✅ 💰 분할매매 시스템")
        print("  ✅ 📈 성과 추적 및 리포트")
        print("  ✅ 🎨 포트폴리오 다양성 관리")
        print("  ✅ ⚡ 병렬 처리로 빠른 분석")
        
        print("\n💡 사용법:")
        print("  python main_engine.py              : 정상 실행")
        print("  python main_engine.py --test       : 강제 테스트 모드")
        print("  await run_full_system_analysis()   : 전체 분석")
        print("  await analyze_symbols(['AAPL'])     : 개별 종목 분석")
        print("  await force_reselection_all()      : 모든 시장 재선별")
        print("  get_engine_status()                 : 시스템 상태")
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다")
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        traceback.print_exc()

# ========================================================================================
# 🎯 실제 매매 시뮬레이션 함수들 (추가)
# ========================================================================================

async def run_trading_simulation(days: int = 30):
    """매매 시뮬레이션 실행"""
    print(f"🎮 {days}일간 매매 시뮬레이션 시작...")
    
    total_profit = 0
    total_trades = 0
    winning_trades = 0
    
    for day in range(days):
        try:
            print(f"\n📅 Day {day + 1}/{days}")
            
            # 일일 분석 실행
            engine = QuantTradingEngine(force_test=True)
            market_results = await engine.run_full_analysis()
            
            if market_results:
                day_profit = 0
                day_trades = 0
                
                for market, summary in market_results.items():
                    executed_trades = [t for t in summary.executed_trades if t.executed]
                    
                    for trade in executed_trades:
                        # 간단한 수익률 시뮬레이션 (랜덤)
                        import random
                        profit_rate = random.uniform(-0.1, 0.15)  # -10% ~ +15%
                        
                        if profit_rate > 0:
                            winning_trades += 1
                        
                        day_profit += profit_rate
                        day_trades += 1
                        total_trades += 1
                
                total_profit += day_profit
                print(f"  일일 수익률: {day_profit*100:.1f}%, 거래: {day_trades}건")
            
            # 하루 대기 (시뮬레이션에서는 빠르게)
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"  ❌ Day {day + 1} 오류: {e}")
    
    # 결과 요약
    print(f"\n📊 {days}일 시뮬레이션 결과:")
    print(f"  총 수익률: {total_profit*100:.1f}%")
    print(f"  총 거래: {total_trades}건")
    print(f"  승률: {winning_trades/total_trades*100:.1f}%" if total_trades > 0 else "  승률: 0%")
    print(f"  일평균 수익률: {total_profit/days*100:.1f}%")

async def run_stress_test():
    """시스템 스트레스 테스트"""
    print("⚡ 시스템 스트레스 테스트 시작...")
    
    start_time = time.time()
    
    # 동시에 여러 분석 실행
    tasks = []
    for i in range(5):
        task = run_single_analysis(force_test=True)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed_time = time.time() - start_time
    
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    
    print(f"⚡ 스트레스 테스트 완료:")
    print(f"  동시 실행: 5개")
    print(f"  성공: {success_count}/5")
    print(f"  총 소요시간: {elapsed_time:.1f}초")
    print(f"  평균 소요시간: {elapsed_time/5:.1f}초")

if __name__ == "__main__":
    asyncio.run(main())
