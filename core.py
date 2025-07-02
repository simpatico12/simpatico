#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 최고퀸트프로젝트 - 핵심 실행 엔진 (Enhanced 완전 통합 버전)
==================================================================

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
Version: 6.0.0 (Enhanced 완전 통합)
Project: 최고퀸트프로젝트
Last Updated: 2025-07-01
"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import traceback
import numpy as np
import time
import signal
from pathlib import Path

# 설정 관리 import (안전한 fallback)
try:
    import yaml
    from dotenv import load_dotenv
    YAML_AVAILABLE = True
    DOTENV_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 설정 라이브러리 누락: {e}")
    YAML_AVAILABLE = False
    DOTENV_AVAILABLE = False

# 데이터 분석 import (안전한 fallback)
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 데이터 분석 라이브러리 누락: {e}")
    PANDAS_AVAILABLE = False

# 기본 dataclass import
from dataclasses import dataclass, asdict, field

# 프로젝트 모듈 import (개선된 안전한 fallback 처리)
def safe_import(module_name, description):
    """안전한 모듈 import"""
    try:
        module = __import__(module_name)
        print(f"✅ {description} 모듈 로드 성공")
        return module, True
    except ImportError as e:
        print(f"⚠️ {description} 모듈 로드 실패: {e}")
        print(f"   해결방법: pip install -r requirements.txt 또는 해당 모듈 파일 생성")
        return None, False

# 전략 모듈들
us_strategy_module, US_STRATEGY_AVAILABLE = safe_import('us_strategy', '미국 주식 전략')
jp_strategy_module, JP_STRATEGY_AVAILABLE = safe_import('jp_strategy', '일본 주식 전략')
coin_strategy_module, COIN_STRATEGY_AVAILABLE = safe_import('coin_strategy', '암호화폐 전략')

# 유틸리티 모듈들
notifier_module, NOTIFIER_AVAILABLE = safe_import('notifier', '알림 시스템')
scheduler_module, SCHEDULER_AVAILABLE = safe_import('스케줄러', '스케줄러')
trading_module, TRADING_AVAILABLE = safe_import('trading', '매매 실행')
config_module, CONFIG_AVAILABLE = safe_import('config', '설정 로더')

# 로깅 설정 (개선된 버전)
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """개선된 로깅 시스템 설정"""
    # logs 폴더 생성
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 (날짜별)
    today = datetime.now().strftime('%Y%m%d')
    log_filename = log_dir / f"quant_{today}.log"
    error_log_filename = log_dir / f"error_{today}.log"
    
    # 로거 생성
    logger = logging.getLogger('QuantTradingEngine')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 핸들러가 이미 있으면 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러 (일반 로그)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 파일 핸들러 (에러 로그)
    error_handler = logging.FileHandler(error_log_filename, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
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
    position_size: Optional[float] = None
    total_investment: Optional[float] = None
    split_stages: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_days: Optional[int] = None
    
    additional_data: Optional[Dict] = field(default_factory=dict)

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
    commission: Optional[float] = None
    slippage: Optional[float] = None
    estimated_profit: Optional[float] = None

@dataclass
class MarketSummary:
    """시장별 요약 데이터 (개선된 버전)"""
    market: str
    total_analyzed: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    top_picks: List[UnifiedTradingSignal]
    executed_trades: List[TradeExecution]
    analysis_time: float
    errors: List[str]
    is_trading_day: bool
    
    # 성과 지표 추가
    avg_confidence: float = 0.0
    success_rate: float = 0.0
    total_estimated_profit: float = 0.0
    
    # 시장별 추가 정보
    market_specific_info: Optional[Dict] = field(default_factory=dict)

class QuantTradingEngine:
    """🏆 최고퀸트프로젝트 메인 엔진 (Enhanced 완전 통합 버전)"""
    
    def __init__(self, config_path: str = "settings.yaml", force_test: bool = False):
        """엔진 초기화 (개선된 버전)"""
        # 시그널 핸들러 등록 (안전한 종료)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.config_path = config_path
        self.force_test = force_test
        self.shutdown_requested = False
        
        # 로깅 초기화
        self.logger = setup_logging()
        
        # 환경변수 로드
        self._load_environment()
        
        # 설정 로드
        self.config = self._load_config()
        
        # 필수 디렉토리 생성
        self._create_directories()
        
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
        
        # 전략 초기화
        self._initialize_strategies()
        
        # 매매 설정
        self._initialize_trading_config()
        
        # 리스크 관리 설정
        self._initialize_risk_config()
        
        # 시장별 포트폴리오 비중
        self.market_allocation = self.config.get('portfolio', {}).get('allocation', {
            'us_ratio': 0.50,
            'jp_ratio': 0.30, 
            'coin_ratio': 0.20
        })
        
        # 실행 통계
        self.daily_trades_count = 0
        self.total_signals_generated = 0
        self.session_start_time = datetime.now()
        self.last_analysis_time = None
        self.analysis_count = 0
        
        # 성과 추적
        self.session_stats = {
            'total_profit': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_commission': 0.0
        }
        
        self.logger.info("🚀 최고퀸트프로젝트 엔진 초기화 완료")
        self._log_initialization_summary()

    def _signal_handler(self, signum, frame):
        """시그널 핸들러 (안전한 종료)"""
        self.logger.info(f"🛑 종료 시그널 수신: {signum}")
        self.shutdown_requested = True

    def _load_environment(self):
        """환경변수 로드"""
        if DOTENV_AVAILABLE:
            env_file = Path('.env')
            if env_file.exists():
                load_dotenv(env_file)
                self.logger.info("✅ 환경변수 로드 완료")
            else:
                self.logger.warning("⚠️ .env 파일 없음. .env.example 참조하여 생성하세요")
        else:
            self.logger.warning("⚠️ python-dotenv 패키지 없음")

    def _load_config(self) -> Dict:
        """설정 파일 로드 (개선된 버전)"""
        config = {}
        
        try:
            if not YAML_AVAILABLE:
                self.logger.error("❌ PyYAML 패키지가 필요합니다: pip install PyYAML")
                return self._get_default_config()
            
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.error(f"❌ 설정 파일 없음: {self.config_path}")
                return self._get_default_config()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 환경변수 치환 (${VAR_NAME:-default} 형식)
                import re
                def replace_env_vars(match):
                    var_expr = match.group(1)
                    if ':-' in var_expr:
                        var_name, default_value = var_expr.split(':-', 1)
                        return os.getenv(var_name, default_value)
                    else:
                        return os.getenv(var_expr, '')
                
                content = re.sub(r'\$\{([^}]+)\}', replace_env_vars, content)
                
                # YAML 파싱
                config = yaml.safe_load(content)
                self.logger.info(f"✅ 설정 파일 로드 성공: {self.config_path}")
                
        except Exception as e:
            self.logger.error(f"❌ 설정 파일 로드 실패: {e}")
            config = self._get_default_config()
        
        return config

    def _get_default_config(self) -> Dict:
        """기본 설정값 반환"""
        return {
            'project': {
                'name': '최고퀸트프로젝트',
                'version': '6.0.0',
                'environment': 'development'
            },
            'schedule': {
                'weekly_schedule': {
                    'monday': {'strategies': ['COIN']},
                    'tuesday': {'strategies': ['US', 'JP']},
                    'wednesday': {'strategies': []},
                    'thursday': {'strategies': ['US', 'JP']},
                    'friday': {'strategies': ['COIN']},
                    'saturday': {'strategies': ['COIN', 'US', 'JP']},
                    'sunday': {'strategies': []}
                }
            },
            'trading': {
                'paper_trading': True,
                'auto_execution': False,
                'order_type': 'limit'
            },
            'risk_management': {
                'max_position_size': 0.05,
                'max_daily_trades': 20,
                'stop_loss': -0.03,
                'take_profit': 0.06
            }
        }

    def _create_directories(self):
        """필수 디렉토리 생성"""
        directories = ['data', 'logs', 'reports', 'backups', 'models']
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        self.logger.info("📁 필수 디렉토리 생성 완료")

    def _get_today_strategies(self) -> List[str]:
        """오늘 실행할 전략 목록 조회 (개선된 버전)"""
        if self.force_test:
            return ['US', 'JP', 'COIN']
        
        try:
            if SCHEDULER_AVAILABLE:
                # 스케줄러 모듈 사용
                from scheduler import get_today_strategies
                strategies = get_today_strategies(self.config)
                return strategies
        except Exception as e:
            self.logger.error(f"❌ 스케줄러 모듈 사용 실패: {e}")
        
        # 기본 스케줄 로직
        weekday = datetime.now().weekday()
        day_mapping = {
            0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday',
            4: 'friday', 5: 'saturday', 6: 'sunday'
        }
        
        today_key = day_mapping.get(weekday, 'monday')
        schedule_config = self.config.get('schedule', {}).get('weekly_schedule', {})
        today_config = schedule_config.get(today_key, {'strategies': ['US', 'JP', 'COIN']})
        
        strategies = today_config.get('strategies', [])
        
        self.logger.info(f"📅 오늘({today_key}): {strategies if strategies else '휴무'}")
        return strategies

    def _initialize_strategies(self):
        """전략 객체들 초기화 (개선된 버전)"""
        try:
            # 미국 주식 전략
            if US_STRATEGY_AVAILABLE and 'US' in self.today_strategies:
                us_config = self.config.get('us_strategy', {})
                if us_config.get('enabled', True):
                    try:
                        from us_strategy import AdvancedUSStrategy
                        self.us_strategy = AdvancedUSStrategy(self.config_path)
                        self.logger.info("🇺🇸 미국 주식 전략 활성화 (4가지 전략 융합)")
                    except Exception as e:
                        self.logger.error(f"❌ 미국 주식 전략 초기화 실패: {e}")
                else:
                    self.logger.info("🇺🇸 미국 주식 전략 설정에서 비활성화")
            elif 'US' not in self.today_strategies:
                self.logger.info("🇺🇸 미국 주식 전략 오늘 비활성화 (스케줄)")
            
            # 일본 주식 전략
            if JP_STRATEGY_AVAILABLE and 'JP' in self.today_strategies:
                jp_config = self.config.get('jp_strategy', {})
                if jp_config.get('enabled', True):
                    try:
                        from jp_strategy import JPStrategy
                        self.jp_strategy = JPStrategy(self.config_path)
                        self.logger.info("🇯🇵 일본 주식 전략 활성화 (엔화+기술분석)")
                    except Exception as e:
                        self.logger.error(f"❌ 일본 주식 전략 초기화 실패: {e}")
                else:
                    self.logger.info("🇯🇵 일본 주식 전략 설정에서 비활성화")
            elif 'JP' not in self.today_strategies:
                self.logger.info("🇯🇵 일본 주식 전략 오늘 비활성화 (스케줄)")
            
            # 암호화폐 전략
            if COIN_STRATEGY_AVAILABLE and 'COIN' in self.today_strategies:
                coin_config = self.config.get('coin_strategy', {})
                if coin_config.get('enabled', True):
                    try:
                        from coin_strategy import UltimateCoinStrategy
                        self.coin_strategy = UltimateCoinStrategy(self.config_path)
                        self.logger.info("🪙 암호화폐 전략 활성화 (AI 품질평가+시장사이클)")
                    except Exception as e:
                        self.logger.error(f"❌ 암호화폐 전략 초기화 실패: {e}")
                else:
                    self.logger.info("🪙 암호화폐 전략 설정에서 비활성화")
            elif 'COIN' not in self.today_strategies:
                self.logger.info("🪙 암호화폐 전략 오늘 비활성화 (스케줄)")
                    
        except Exception as e:
            self.logger.error(f"❌ 전략 초기화 실패: {e}")

    def _initialize_trading_config(self):
        """매매 설정 초기화"""
        self.trading_config = self.config.get('trading', {})
        self.auto_execution = self.trading_config.get('auto_execution', False)
        self.paper_trading = self.trading_config.get('paper_trading', True)
        
        # 매매 실행기 초기화
        self.trading_executor = None
        if TRADING_AVAILABLE and self.auto_execution:
            try:
                from trading import TradingExecutor
                self.trading_executor = TradingExecutor(self.config_path)
                self.logger.info(f"💰 매매 실행기 초기화 완료 (모의거래: {self.paper_trading})")
            except Exception as e:
                self.logger.error(f"❌ 매매 실행기 초기화 실패: {e}")

    def _initialize_risk_config(self):
        """리스크 관리 설정 초기화"""
        self.risk_config = self.config.get('risk_management', {})
        self.max_position_size = self.risk_config.get('position_risk', {}).get('max_position_size', 0.05)
        self.stop_loss = self.risk_config.get('position_risk', {}).get('stop_loss', -0.03)
        self.take_profit = self.risk_config.get('position_risk', {}).get('take_profit', 0.06)
        self.max_daily_trades = self.risk_config.get('limits', {}).get('max_daily_trades', 20)

    def _log_initialization_summary(self):
        """초기화 요약 로그"""
        self.logger.info("⚙️ 시스템 설정:")
        self.logger.info(f"  자동매매: {self.auto_execution}")
        self.logger.info(f"  모의거래: {self.paper_trading}")
        self.logger.info(f"  활성 전략: {len(self.today_strategies)}개 - {self.today_strategies}")
        self.logger.info(f"  시장 비중: US:{self.market_allocation.get('us_ratio', 0)*100:.0f}% "
                        f"JP:{self.market_allocation.get('jp_ratio', 0)*100:.0f}% "
                        f"COIN:{self.market_allocation.get('coin_ratio', 0)*100:.0f}%")
        self.logger.info(f"  일일 거래 한도: {self.max_daily_trades}건")
        
        if self.force_test:
            self.logger.info("🧪 강제 테스트: 스케줄 무시하고 모든 전략 테스트")

    def _check_trading_time(self) -> bool:
        """현재 시간이 거래 시간인지 확인 (개선된 버전)"""
        try:
            if self.force_test:
                return True
                
            if SCHEDULER_AVAILABLE:
                from scheduler import is_trading_time
                return is_trading_time(self.config)
            else:
                # 기본 거래시간 로직
                current_hour = datetime.now().hour
                weekday = datetime.now().weekday()
                
                # 암호화폐는 24시간
                if 'COIN' in self.today_strategies:
                    return True
                
                # 주식은 평일 거래시간
                if weekday < 5 and 9 <= current_hour <= 16:
                    return True
                    
                return False
                
        except Exception as e:
            self.logger.error(f"거래 시간 확인 실패: {e}")
            return True

    def _convert_to_unified_signal(self, signal: Any, market: str) -> UnifiedTradingSignal:
        """각 전략의 신호를 통합 신호로 변환 (개선된 버전)"""
        try:
            # 기본 필드 추출 (안전한 방식)
            symbol = getattr(signal, 'symbol', 'UNKNOWN')
            action = getattr(signal, 'action', 'hold')
            confidence = getattr(signal, 'confidence', 0.0)
            price = getattr(signal, 'price', 0.0)
            strategy = getattr(signal, 'strategy', 'unknown')
            reasoning = getattr(signal, 'reasoning', '')
            target_price = getattr(signal, 'target_price', price)
            timestamp = getattr(signal, 'timestamp', datetime.now())
            
            # 통합 신호 생성
            unified = UnifiedTradingSignal(
                market=market,
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=price,
                strategy=strategy,
                reasoning=reasoning,
                target_price=target_price,
                timestamp=timestamp,
                sector=getattr(signal, 'sector', None),
                rsi=getattr(signal, 'rsi', None),
                macd_signal=getattr(signal, 'macd_signal', None)
            )
            
            # 시장별 특수 필드 (안전한 추출)
            if market == 'US':
                unified.buffett_score = getattr(signal, 'buffett_score', None)
                unified.lynch_score = getattr(signal, 'lynch_score', None)
                unified.momentum_score = getattr(signal, 'momentum_score', None)
                unified.technical_score = getattr(signal, 'technical_score', None)
                unified.total_score = getattr(signal, 'total_score', confidence)
                unified.position_size = getattr(signal, 'total_shares', None)
                unified.split_stages = 3
                
            elif market == 'JP':
                unified.yen_signal = getattr(signal, 'yen_signal', None)
                unified.stock_type = getattr(signal, 'stock_type', None)
                unified.total_score = confidence
                unified.split_stages = getattr(signal, 'split_stages', 0)
                
            elif market == 'COIN':
                unified.project_quality_score = getattr(signal, 'project_quality_score', None)
                unified.market_cycle = getattr(signal, 'market_cycle', None)
                unified.btc_correlation = getattr(signal, 'correlation_with_btc', None)
                unified.total_score = getattr(signal, 'total_score', confidence)
                unified.split_stages = 5
            
            # 공통 설정
            unified.stop_loss = getattr(signal, 'stop_loss', self.stop_loss)
            unified.take_profit = getattr(signal, 'take_profit', self.take_profit)
            unified.max_hold_days = getattr(signal, 'max_hold_days', 30)
            
            return unified
            
        except Exception as e:
            self.logger.error(f"신호 변환 실패 {market}-{getattr(signal, 'symbol', 'UNKNOWN')}: {e}")
            
            # 기본 신호 반환
            return UnifiedTradingSignal(
                market=market,
                symbol=getattr(signal, 'symbol', 'UNKNOWN'),
                action='hold',
                confidence=0.0,
                price=getattr(signal, 'price', 0.0),
                strategy='conversion_error',
                reasoning=f"신호 변환 실패: {str(e)}",
                target_price=0.0,
                timestamp=datetime.now()
            )

    def _apply_risk_management(self, signals: List[UnifiedTradingSignal]) -> List[UnifiedTradingSignal]:
        """통합 리스크 관리 적용 (개선된 버전)"""
        filtered_signals = []
        
        # 종료 요청 확인
        if self.shutdown_requested:
            self.logger.info("🛑 종료 요청으로 리스크 관리 중단")
            return filtered_signals
        
        # 일일 거래 제한 체크
        if self.daily_trades_count >= self.max_daily_trades:
            self.logger.warning(f"⚠️ 일일 거래 한도 도달: {self.daily_trades_count}/{self.max_daily_trades}")
            return filtere
            'basic_status': {
                    'uptime_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600,
                    'daily_trades': self.daily_trades_count,
                    'shutdown_requested': self.shutdown_requested
                }
             {
                    'uptime_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600,
                    'daily_trades': self.daily_trades_count,
                    'shutdown_requested': self.shutdown_requested
                }
             {
                    'uptime_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600,
                    'daily_trades': self.daily_trades_count,
                    'shutdown_requested': self.shutdown_requested
                }

    async def force_reselection_all_markets(self) -> Dict[str, List[str]]:
        """모든 시장 강제 재선별 (개선된 버전)"""
        results = {}
        
        if self.shutdown_requested:
            self.logger.info("🛑 종료 요청으로 재선별 중단")
            return results
        
        self.logger.info("🔄 전체 시장 강제 재선별 시작...")
        
        # 미국 주식 재선별
        if self.us_strategy and US_STRATEGY_AVAILABLE:
            try:
                from us_strategy import force_us_reselection
                us_symbols = await force_us_reselection()
                results['US'] = us_symbols
                self.logger.info(f"🇺🇸 미국 주식 재선별 완료: {len(us_symbols)}개")
            except Exception as e:
        print(f"\n❌ 시스템 실행 중 치명적 오류: {e}")
        traceback.print_exc()
    finally:
        print("\n🏁 최고퀸트프로젝트 시스템 종료")

# ========================================================================================
# 🔧 CLI 유틸리티 함수들 (새로 추가)
# ========================================================================================

def print_help():
    """도움말 출력"""
    print("🏆 최고퀸트프로젝트 V5.0 - Enhanced 완전 통합 시스템")
    print("=" * 60)
    print("사용법: python main_engine.py [옵션] [설정파일]")
    print()
    print("📋 기본 옵션:")
    print("  (없음)           : 정상 실행 (스케줄에 따라)")
    print("  --test, --force  : 강제 테스트 모드 (모든 전략 실행)")
    print("  --help, -h       : 이 도움말 표시")
    print()
    print("🔧 고급 옵션:")
    print("  --continuous     : 연속 모니터링 모드")
    print("  --benchmark      : 성능 벤치마크 실행")
    print("  --stress         : 시스템 스트레스 테스트")
    print("  --simulation     : 매매 시뮬레이션 실행")
    print("  --status         : 시스템 상태만 확인")
    print("  --reselect       : 모든 시장 강제 재선별")
    print()
    print("📁 설정 파일:")
    print("  settings.yaml    : 기본 설정 파일")
    print("  custom.yaml      : 사용자 정의 설정")
    print()
    print("💡 사용 예시:")
    print("  python main_engine.py")
    print("  python main_engine.py --test")
    print("  python main_engine.py --continuous")
    print("  python main_engine.py custom.yaml --test")
    print()
    print("📞 지원:")
    print("  - 설정: .env.example → .env 파일 생성")
    print("  - 패키지: pip install -r requirements.txt")
    print("  - 로그: logs/ 폴더 확인")

async def show_status_only(config_path: str = "settings.yaml"):
    """상태만 확인하고 종료"""
    try:
        print("📊 시스템 상태 확인 중...")
        
        engine = QuantTradingEngine(config_path=config_path, force_test=False)
        status = engine.get_system_status()
        
        print("\n💻 시스템 정보:")
        sys_info = status['system_info']
        print(f"  상태: {sys_info['status']}")
        print(f"  버전: {sys_info['version']}")
        print(f"  가동시간: {sys_info['uptime_hours']:.1f}시간")
        print(f"  시작시간: {sys_info['start_time']}")
        
        print("\n🎯 전략 상태:")
        strategies = status['strategies']
        enabled = strategies['enabled_strategies']
        available = strategies['available_modules']
        
        for strategy in ['us_strategy', 'jp_strategy', 'coin_strategy']:
            enabled_icon = "✅" if enabled.get(strategy, False) else "❌"
            available_icon = "📦" if available.get(f"{strategy}_module", False) else "❌"
            market = strategy.split('_')[0].upper()
            print(f"  {market:4s}: {enabled_icon} 활성화 {available_icon} 모듈")
        
        print(f"  오늘 실행: {strategies['today_strategies']}")
        
        print("\n💰 거래 설정:")
        trading = status['trading_status']
        print(f"  자동매매: {trading['auto_execution']}")
        print(f"  모의거래: {trading['paper_trading']}")
        print(f"  일일거래: {trading['daily_trades_count']}/{trading['max_daily_trades']}")
        print(f"  거래활용: {trading['trading_utilization']:.1f}%")
        
        print("\n📈 분석 현황:")
        analysis = status['analysis_status']
        print(f"  총 신호: {analysis['total_signals_generated']}개")
        print(f"  분석횟수: {analysis['analysis_count']}회")
        print(f"  마지막분석: {analysis['last_analysis_time'] or 'None'}")
        
        print("\n🔧 의존성:")
        deps = status['system_dependencies']
        for dep, available in deps.items():
            icon = "✅" if available else "❌"
            print(f"  {icon} {dep}")
        
        print("\n✅ 상태 확인 완료")
        
    except Exception as e:
        print(f"❌ 상태 확인 실패: {e}")

async def run_reselection_only(config_path: str = "settings.yaml"):
    """재선별만 실행하고 종료"""
    try:
        print("🔄 모든 시장 강제 재선별 시작...")
        
        results = await force_reselection_all(config_path)
        
        if results:
            total_symbols = sum(len(symbols) for symbols in results.values())
            print(f"\n✅ 재선별 완료: 총 {total_symbols}개 종목")
            
            for market, symbols in results.items():
                market_name = {'US': '🇺🇸 미국', 'JP': '🇯🇵 일본', 'COIN': '🪙 코인'}.get(market, market)
                print(f"  {market_name}: {len(symbols)}개")
                
                if symbols:
                    print(f"    예시: {', '.join(symbols[:5])}")
                    if len(symbols) > 5:
                        print(f"    ... 외 {len(symbols)-5}개")
        else:
            print("❌ 재선별 결과가 없습니다")
            
    except Exception as e:
        print(f"❌ 재선별 실패: {e}")

# ========================================================================================
# 🚀 메인 진입점 (개선된 CLI 처리)
# ========================================================================================

if __name__ == "__main__":
    # 명령행 인수 파싱
    import sys
    
    # 도움말 확인
    if '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        sys.exit(0)
    
    # 설정 파일 확인
    config_path = "settings.yaml"
    for arg in sys.argv[1:]:
        if arg.endswith('.yaml') or arg.endswith('.yml'):
            config_path = arg
            break
    
    # 설정 파일 존재 확인
    if not Path(config_path).exists():
        print(f"❌ 설정 파일이 없습니다: {config_path}")
        print("💡 해결방법:")
        print("  1. settings.yaml 파일을 생성하세요")
        print("  2. 또는 다른 설정 파일을 지정하세요")
        print("  3. --help 옵션으로 사용법을 확인하세요")
        sys.exit(1)
    
    # 환경변수 파일 확인
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️ .env 파일이 없습니다")
        print("💡 .env.example을 참조하여 .env 파일을 생성하세요")
        print("   주요 설정: API 키, 보안 키, 데이터베이스 등")
        print()
    
    # 특별 모드 확인 및 실행
    try:
        if '--status' in sys.argv:
            asyncio.run(show_status_only(config_path))
        elif '--reselect' in sys.argv:
            asyncio.run(run_reselection_only(config_path))
        else:
            # 메인 시스템 실행
            asyncio.run(main())
            
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류: {e}")
        print("\n🔧 문제해결 방법:")
        print("  1. pip install -r requirements.txt")
        print("  2. .env 파일 설정 확인")
        print("  3. settings.yaml 파일 구문 확인")
        print("  4. logs/ 폴더의 오류 로그 확인")
        print("  5. --help 옵션으로 사용법 확인")
        
        # 개발자 모드에서는 상세 오류 출력
        if '--debug' in sys.argv:
            traceback.print_exc()
    
    finally:
        print("\n🏁 최고퀸트프로젝트 종료")

# ========================================================================================
# 📝 모듈 정보 (docstring)
# ========================================================================================

__version__ = "5.0.0"
__author__ = "최고퀸트팀"
__description__ = "최고퀸트프로젝트 - Enhanced 완전 통합 시스템"

__all__ = [
    # 메인 클래스
    'QuantTradingEngine',
    
    # 데이터 클래스
    'UnifiedTradingSignal',
    'TradeExecution', 
    'MarketSummary',
    
    # 편의 함수
    'run_single_analysis',
    'run_full_system_analysis',
    'analyze_symbols',
    'get_engine_status',
    'force_reselection_all',
    'get_all_selection_status',
    
    # 고급 기능
    'run_continuous_monitoring',
    'run_performance_benchmark',
    'run_stress_test',
    'run_trading_simulation',
    
    # CLI 유틸리티
    'print_help',
    'show_status_only',
    'run_reselection_only'
]

# ========================================================================================
# 🎯 사용 예시 (주석)
# ========================================================================================

"""
📋 사용 예시:

1. 기본 실행:
   python main_engine.py

2. 강제 테스트:
   python main_engine.py --test

3. 연속 모니터링:
   python main_engine.py --continuous

4. 성능 벤치마크:
   python main_engine.py --benchmark

5. 시스템 상태 확인:
   python main_engine.py --status

6. 프로그래매틱 사용:
   ```python
   import asyncio
   from main_engine import run_full_system_analysis
   
   async def my_analysis():
       market_results, report = await run_full_system_analysis(force_test=True)
       return report
   
   result = asyncio.run(my_analysis())
   ```

7. 개별 종목 분석:
   ```python
   from main_engine import analyze_symbols
   
   signals = await analyze_symbols(['AAPL', 'GOOGL', 'KRW-BTC'])
   for signal in signals:
       print(f"{signal.symbol}: {signal.action} ({signal.confidence:.1%})")
   ```

🔧 설정 파일 구조:
- settings.yaml: 메인 설정 (전략, 리스크, 스케줄 등)
- .env: 민감 정보 (API 키, 보안 설정 등)
- requirements.txt: 의존성 패키지
- .gitignore: 보안 파일 제외

📁 디렉토리 구조:
- data/: 분석 결과, 캐시 데이터
- logs/: 실행 로그, 오류 로그  
- reports/: 생성된 리포트
- backups/: 백업 데이터
- models/: ML 모델 (향후)

🚀 시스템 특징:
- 🌍 전 세계 3개 시장 통합 (미국, 일본, 암호화폐)
- 🤖 완전 자동화 (선별, 분석, 실행, 알림)
- ⚡ 고성능 병렬 처리
- 🛡️ 강화된 리스크 관리
- 🔒 안전한 종료 및 오류 처리
- 📊 실시간 성과 추적
- 🎮 시뮬레이션 및 백테스팅
- 📱 다채널 알림 시스템
- 💾 자동 데이터 저장 및 백업
- 🔄 연속 모니터링 지원
"""
                self.logger.error(f"❌ 미국 주식 재선별 실패: {e}")
                results['US'] = []

        # 일본 주식 재선별
        if self.jp_strategy and JP_STRATEGY_AVAILABLE:
            try:
                from jp_strategy import force_jp_reselection
                jp_symbols = await force_jp_reselection()
                results['JP'] = jp_symbols
                self.logger.info(f"🇯🇵 일본 주식 재선별 완료: {len(jp_symbols)}개")
            except Exception as e:
                self.logger.error(f"❌ 일본 주식 재선별 실패: {e}")
                results['JP'] = []

        # 암호화폐 재선별
        if self.coin_strategy and COIN_STRATEGY_AVAILABLE:
            try:
                from coin_strategy import force_coin_reselection
                coin_symbols = await force_coin_reselection()
                results['COIN'] = coin_symbols
                self.logger.info(f"🪙 암호화폐 재선별 완료: {len(coin_symbols)}개")
            except Exception as e:
                self.logger.error(f"❌ 암호화폐 재선별 실패: {e}")
                results['COIN'] = []

        total_symbols = sum(len(symbols) for symbols in results.values())
        self.logger.info(f"🔄 전체 재선별 완료: {total_symbols}개 종목")
        
        return results

    async def get_auto_selection_status_all(self) -> Dict[str, Dict]:
        """모든 시장의 자동선별 상태 조회 (개선된 버전)"""
        status = {}
        
        if self.shutdown_requested:
            return {'error': '시스템 종료 중'}
        
        # 미국 주식 상태
        if US_STRATEGY_AVAILABLE:
            try:
                from us_strategy import get_us_auto_selection_status
                status['US'] = await get_us_auto_selection_status()
            except Exception as e:
                self.logger.error(f"❌ 미국 주식 상태 조회 실패: {e}")
                status['US'] = {'error': str(e), 'available': False}
        else:
            status['US'] = {'error': '모듈 없음', 'available': False}

        # 일본 주식 상태
        if JP_STRATEGY_AVAILABLE:
            try:
                from jp_strategy import get_jp_auto_selection_status
                status['JP'] = await get_jp_auto_selection_status()
            except Exception as e:
                self.logger.error(f"❌ 일본 주식 상태 조회 실패: {e}")
                status['JP'] = {'error': str(e), 'available': False}
        else:
            status['JP'] = {'error': '모듈 없음', 'available': False}

        # 암호화폐 상태
        if COIN_STRATEGY_AVAILABLE:
            try:
                from coin_strategy import get_coin_auto_selection_status
                status['COIN'] = await get_coin_auto_selection_status()
            except Exception as e:
                self.logger.error(f"❌ 암호화폐 상태 조회 실패: {e}")
                status['COIN'] = {'error': str(e), 'available': False}
        else:
            status['COIN'] = {'error': '모듈 없음', 'available': False}

        return status

    def graceful_shutdown(self):
        """안전한 종료"""
        self.logger.info("🛑 안전한 종료 시작...")
        self.shutdown_requested = True
        
        # 진행 중인 작업 완료 대기
        # (실제로는 asyncio 태스크들이 자연스럽게 종료되도록 함)
        
        # 세션 통계 로그
        uptime = (datetime.now() - self.session_start_time).total_seconds()
        self.logger.info(f"📊 세션 종료 통계:")
        self.logger.info(f"  가동시간: {uptime/3600:.1f}시간")
        self.logger.info(f"  총 분석: {self.analysis_count}회")
        self.logger.info(f"  총 신호: {self.total_signals_generated}개")
        self.logger.info(f"  총 거래: {self.daily_trades_count}건")
        self.logger.info(f"  예상 수익: {self.session_stats['total_profit']:,.0f}원")
        
        self.logger.info("✅ 안전한 종료 완료")

# ========================================================================================
# 🎯 편의 함수들 (개선된 버전)
# ========================================================================================

async def run_single_analysis(force_test: bool = False, config_path: str = "settings.yaml"):
    """단일 분석 실행"""
    try:
        engine = QuantTradingEngine(config_path=config_path, force_test=force_test)
        results = await engine.run_full_analysis()
        return results
    except Exception as e:
        print(f"❌ 단일 분석 실행 실패: {e}")
        return {}

async def run_full_system_analysis(force_test: bool = False, config_path: str = "settings.yaml"):
    """전체 시스템 분석 + 리포트 생성"""
    try:
        engine = QuantTradingEngine(config_path=config_path, force_test=force_test)
        market_results = await engine.run_full_analysis()
        
        if market_results:
            unified_report = await engine.generate_unified_portfolio_report(market_results)
            return market_results, unified_report
        else:
            return {}, {}
    except Exception as e:
        print(f"❌ 전체 시스템 분석 실패: {e}")
        return {}, {}

async def analyze_symbols(symbols: List[str], config_path: str = "settings.yaml"):
    """특정 종목들 분석"""
    try:
        engine = QuantTradingEngine(config_path=config_path)
        signals = await engine.get_quick_analysis(symbols)
        return signals
    except Exception as e:
        print(f"❌ 종목 분석 실패: {e}")
        return []

def get_engine_status(config_path: str = "settings.yaml"):
    """엔진 상태 조회"""
    try:
        engine = QuantTradingEngine(config_path=config_path)
        return engine.get_system_status()
    except Exception as e:
        print(f"❌ 상태 조회 실패: {e}")
        return {"error": str(e)}

async def force_reselection_all(config_path: str = "settings.yaml"):
    """모든 시장 강제 재선별"""
    try:
        engine = QuantTradingEngine(config_path=config_path)
        return await engine.force_reselection_all_markets()
    except Exception as e:
        print(f"❌ 재선별 실패: {e}")
        return {}

async def get_all_selection_status(config_path: str = "settings.yaml"):
    """모든 시장 선별 상태 조회"""
    try:
        engine = QuantTradingEngine(config_path=config_path)
        return await engine.get_auto_selection_status_all()
    except Exception as e:
        print(f"❌ 상태 조회 실패: {e}")
        return {}

# ========================================================================================
# 🧪 고급 테스트 및 시뮬레이션 함수들 (새로 추가)
# ========================================================================================

async def run_continuous_monitoring(interval_minutes: int = 60, max_iterations: int = 24):
    """연속 모니터링 실행"""
    print(f"📡 연속 모니터링 시작: {interval_minutes}분 간격, 최대 {max_iterations}회")
    
    for i in range(max_iterations):
        try:
            print(f"\n🔄 모니터링 {i+1}/{max_iterations} 시작...")
            
            # 전체 분석 실행
            market_results, unified_report = await run_full_system_analysis(force_test=False)
            
            if unified_report:
                summary = unified_report.get('summary', {})
                print(f"📊 결과: 분석 {summary.get('total_analyzed', 0)}개, "
                      f"매수 {summary.get('total_buy_signals', 0)}개, "
                      f"실행 {summary.get('total_executed', 0)}개")
            
            # 다음 실행까지 대기
            if i < max_iterations - 1:
                print(f"⏰ {interval_minutes}분 대기...")
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 사용자가 모니터링을 중단했습니다")
            break
        except Exception as e:
            print(f"❌ 모니터링 {i+1} 실행 중 오류: {e}")
            await asyncio.sleep(60)  # 1분 대기 후 계속

    print("✅ 연속 모니터링 완료")

async def run_performance_benchmark(iterations: int = 10):
    """성능 벤치마크 실행"""
    print(f"⚡ 성능 벤치마크 시작: {iterations}회 반복")
    
    times = []
    signal_counts = []
    
    for i in range(iterations):
        try:
            start_time = time.time()
            
            # 강제 테스트 모드로 실행
            market_results = await run_single_analysis(force_test=True)
            
            elapsed_time = time.time() - start_time
            total_signals = sum([len(result.top_picks) for result in market_results.values()]) if market_results else 0
            
            times.append(elapsed_time)
            signal_counts.append(total_signals)
            
            print(f"  {i+1:2d}회: {elapsed_time:.1f}초, {total_signals}개 신호")
            
        except Exception as e:
            print(f"  {i+1:2d}회: 오류 - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_signals = sum(signal_counts) / len(signal_counts)
        
        print(f"\n📊 벤치마크 결과:")
        print(f"  평균 시간: {avg_time:.1f}초")
        print(f"  최소 시간: {min_time:.1f}초")
        print(f"  최대 시간: {max_time:.1f}초")
        print(f"  평균 신호: {avg_signals:.1f}개")
        print(f"  초당 신호: {avg_signals/avg_time:.1f}개/초")

async def run_stress_test(concurrent_analyses: int = 5, iterations: int = 3):
    """시스템 스트레스 테스트"""
    print(f"💪 스트레스 테스트 시작: 동시 {concurrent_analyses}개 분석, {iterations}회 반복")
    
    for iteration in range(iterations):
        print(f"\n🔥 스트레스 테스트 {iteration+1}/{iterations}")
        
        start_time = time.time()
        
        # 동시 분석 실행
        tasks = []
        for i in range(concurrent_analyses):
            task = run_single_analysis(force_test=True)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            elapsed_time = time.time() - start_time
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            
            print(f"  결과: {success_count}/{concurrent_analyses} 성공")
            print(f"  시간: {elapsed_time:.1f}초")
            print(f"  동시 처리 효율: {success_count/elapsed_time:.1f}개/초")
            
            # 실패한 태스크가 있으면 로그
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"  태스크 {i+1} 실패: {result}")
                    
        except Exception as e:
            print(f"  전체 실패: {e}")
        
        # 테스트 간 휴식
        if iteration < iterations - 1:
            await asyncio.sleep(5)

    print("✅ 스트레스 테스트 완료")

async def run_trading_simulation(days: int = 30, initial_capital: float = 10000000):
    """고급 매매 시뮬레이션"""
    print(f"🎮 {days}일간 매매 시뮬레이션 시작 (초기자본: {initial_capital:,.0f}원)")
    
    portfolio = {
        'cash': initial_capital,
        'positions': {},
        'daily_pnl': [],
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0
    }
    
    for day in range(days):
        try:
            print(f"\n📅 Day {day + 1}/{days}")
            
            # 일일 분석 실행
            market_results = await run_single_analysis(force_test=True)
            
            if market_results:
                daily_pnl = 0
                daily_trades = 0
                
                for market, summary in market_results.items():
                    # 매수 신호 처리
                    for signal in summary.top_picks[:2]:  # 상위 2개만
                        if signal.action == 'buy' and signal.confidence > 0.7:
                            # 포지션 크기 계산 (자본의 5%)
                            position_value = portfolio['cash'] * 0.05
                            
                            if position_value > 0:
                                # 매수 실행
                                portfolio['positions'][signal.symbol] = {
                                    'shares': position_value / signal.price,
                                    'entry_price': signal.price,
                                    'market': market,
                                    'entry_day': day
                                }
                                portfolio['cash'] -= position_value
                                daily_trades += 1
                                portfolio['total_trades'] += 1
                                
                                print(f"  매수: {signal.symbol} @ {signal.price:.2f}")
                    
                    # 기존 포지션 평가 및 매도
                    positions_to_close = []
                    for symbol, position in portfolio['positions'].items():
                        # 간단한 수익률 시뮬레이션
                        import random
                        current_price = position['entry_price'] * (1 + random.uniform(-0.1, 0.15))
                        
                        pnl = (current_price - position['entry_price']) * position['shares']
                        
                        # 손익 조건 체크 (±10% 또는 7일 홀딩)
                        pnl_rate = pnl / (position['entry_price'] * position['shares'])
                        holding_days = day - position['entry_day']
                        
                        if abs(pnl_rate) > 0.1 or holding_days > 7:
                            # 매도
                            sell_value = current_price * position['shares']
                            portfolio['cash'] += sell_value
                            daily_pnl += pnl
                            
                            if pnl > 0:
                                portfolio['winning_trades'] += 1
                            else:
                                portfolio['losing_trades'] += 1
                            
                            positions_to_close.append(symbol)
                            daily_trades += 1
                            
                            print(f"  매도: {symbol} @ {current_price:.2f} (수익: {pnl:,.0f}원)")
                    
                    # 포지션 정리
                    for symbol in positions_to_close:
                        del portfolio['positions'][symbol]
                
                portfolio['daily_pnl'].append(daily_pnl)
                
                # 일일 요약
                total_value = portfolio['cash'] + sum([pos['entry_price'] * pos['shares'] 
                                                     for pos in portfolio['positions'].values()])
                
                print(f"  일일 손익: {daily_pnl:,.0f}원")
                print(f"  총 자산: {total_value:,.0f}원")
                print(f"  수익률: {(total_value - initial_capital) / initial_capital * 100:.1f}%")
                print(f"  거래: {daily_trades}건")
            
            # 하루 대기 (시뮬레이션)
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"  ❌ Day {day + 1} 오류: {e}")
    
    # 최종 결과
    final_value = portfolio['cash'] + sum([pos['entry_price'] * pos['shares'] 
                                         for pos in portfolio['positions'].values()])
    total_return = (final_value - initial_capital) / initial_capital * 100
    win_rate = portfolio['winning_trades'] / max(1, portfolio['total_trades']) * 100
    
    print(f"\n📊 {days}일 시뮬레이션 결과:")
    print(f"  초기자본: {initial_capital:,.0f}원")
    print(f"  최종자산: {final_value:,.0f}원")
    print(f"  총 수익률: {total_return:.1f}%")
    print(f"  연환산 수익률: {total_return * 365 / days:.1f}%")
    print(f"  총 거래: {portfolio['total_trades']}건")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  승리: {portfolio['winning_trades']}건")
    print(f"  패배: {portfolio['losing_trades']}건")

# ========================================================================================
# 🧪 메인 실행 함수 (Enhanced 완전 통합 버전)
# ========================================================================================

async def main():
    """메인 실행 함수 (Enhanced 완전 통합 시스템)"""
    try:
        print("🏆 최고퀸트프로젝트 - Enhanced 완전 통합 시스템 V5.0!")
        print("=" * 80)
        print("🌍 전 세계 시장 통합 매매 시스템:")
        print("  🇺🇸 미국 주식 (버핏+린치+모멘텀+기술분석)")
        print("  🇯🇵 일본 주식 (엔화+기술분석+자동선별)")
        print("  🪙 암호화폐 (AI품질평가+시장사이클+상관관계)")
        print("  🤖 완전 자동화 (자동선별+분석+실행+알림)")
        print("  ⚡ Enhanced 기능 (성능최적화+안전종료+오류처리)")
        print("=" * 80)
        
        # 명령행 인수 처리
        import sys
        force_test = '--test' in sys.argv or '--force' in sys.argv
        continuous = '--continuous' in sys.argv or '--monitor' in sys.argv
        benchmark = '--benchmark' in sys.argv or '--perf' in sys.argv
        stress = '--stress' in sys.argv
        simulation = '--simulation' in sys.argv or '--sim' in sys.argv
        
        if force_test:
            print("🧪 강제 테스트 모드 활성화")
        
        # 엔진 초기화
        config_path = "settings.yaml"
        if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
            config_path = sys.argv[1]
        
        engine = QuantTradingEngine(config_path=config_path, force_test=force_test)
        
        # 시스템 상태 출력
        status = engine.get_system_status()
        print(f"\n💻 시스템 상태:")
        print(f"  버전: {status['system_info']['version']}")
        print(f"  상태: {status['system_info']['status']}")
        print(f"  가동시간: {status['system_info']['uptime_hours']:.1f}시간")
        print(f"  활성화된 전략: {sum(status['strategies']['enabled_strategies'].values())}개")
        print(f"  오늘 실행 전략: {status['strategies']['today_strategies']}")
        print(f"  일일 거래: {status['trading_status']['daily_trades_count']}/{status['trading_status']['max_daily_trades']}")
        print(f"  자동매매: {status['trading_status']['auto_execution']} (모의거래: {status['trading_status']['paper_trading']})")
        
        # 의존성 상태
        print(f"\n🔧 시스템 의존성:")
        deps = status['system_dependencies']
        for dep, available in deps.items():
            status_icon = "✅" if available else "❌"
            print(f"  {status_icon} {dep}")
        
        # 모듈 가용성
        print(f"\n📦 모듈 상태:")
        modules = status['strategies']['available_modules']
        for module, available in modules.items():
            status_icon = "✅" if available else "❌"
            print(f"  {status_icon} {module}")
        
        # 특수 모드 실행
        if benchmark:
            await run_performance_benchmark(iterations=5)
            return
        elif stress:
            await run_stress_test(concurrent_analyses=3, iterations=2)
            return
        elif simulation:
            await run_trading_simulation(days=10, initial_capital=10000000)
            return
        elif continuous:
            await run_continuous_monitoring(interval_minutes=30, max_iterations=48)
            return
        
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
        
        try:
            market_results, unified_report = await run_full_system_analysis(force_test, config_path)
            
            elapsed_time = time.time() - start_time
            print(f"\n⏱️ 총 소요 시간: {elapsed_time:.1f}초")
            
            if market_results and unified_report:
                # 결과 출력 (기존 코드와 동일하지만 더 안전하게)
                summary = unified_report.get('summary', {})
                print(f"\n🎯 통합 분석 결과:")
                print(f"  분석 시장: {summary.get('total_markets', 0)}개")
                print(f"  총 분석 종목: {summary.get('total_analyzed', 0)}개")
                print(f"  매수 신호: {summary.get('total_buy_signals', 0)}개")
                print(f"  매도 신호: {summary.get('total_sell_signals', 0)}개")
                print(f"  실행된 거래: {summary.get('total_executed', 0)}개")
                print(f"  전체 매수율: {summary.get('overall_buy_rate', 0):.1f}%")
                print(f"  예상 수익: {summary.get('total_estimated_profit', 0):,.0f}원")
                
                # 시장별 성과 (요약)
                performance = unified_report.get('market_performance', {})
                if performance:
                    print(f"\n📊 시장별 성과:")
                    for market, perf in performance.items():
                        market_name = {'US': '🇺🇸 미국', 'JP': '🇯🇵 일본', 'COIN': '🪙 코인'}.get(market, market)
                        print(f"  {market_name}: 분석 {perf.get('analyzed', 0)}개, "
                              f"매수 {perf.get('buy_signals', 0)}개 ({perf.get('buy_rate', 0):.1f}%), "
                              f"실행 {perf.get('executed_trades', 0)}개")
                
                # 상위 추천 (간략)
                top_picks = unified_report.get('global_top_picks', [])
                if top_picks:
                    print(f"\n🏆 글로벌 상위 추천 TOP 3:")
                    for i, pick in enumerate(top_picks[:3], 1):
                        market_emoji = {'US': '🇺🇸', 'JP': '🇯🇵', 'COIN': '🪙'}.get(pick['market'], '❓')
                        potential_return = pick.get('potential_return', 0)
                        print(f"  {i}. {market_emoji} {pick['symbol']} "
                              f"({pick['confidence']:.1%}, +{potential_return:.1f}%)")
            
            else:
                print("❌ 분석 결과가 없습니다.")
        
        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 중단되었습니다")
            engine.graceful_shutdown()
        except Exception as e:
            print(f"\n❌ 분석 실행 중 오류: {e}")
            traceback.print_exc()
        
        print("\n✅ Enhanced 완전 통합 분석 완료!")
        print("\n🎯 최고퀸트프로젝트 V6.0 Enhanced 시스템 특징:")
        print("  ✅ 🌍 전 세계 3개 시장 완전 통합")
        print("  ✅ 🤖 완전 자동화 시스템")
        print("  ✅ ⚡ 성능 최적화 및 병렬 처리")
        print("  ✅ 🛡️ 강화된 리스크 관리")
        print("  ✅ 🔒 안전한 종료 및 오류 처리")
        print("  ✅ 📊 고급 성과 분석")
        print("  ✅ 🎮 시뮬레이션 및 백테스팅")
        print("  ✅ 📱 실시간 알림 시스템")
        print("  ✅ 💾 자동 데이터 저장")
        print("  ✅ 🔄 연속 모니터링")
        
        print("\n💡 사용법:")
        print("  python main_engine.py                    : 정상 실행")
        print("  python main_engine.py --test             : 강제 테스트 모드")
        print("  python main_engine.py --continuous       : 연속 모니터링")
        print("  python main_engine.py --benchmark        : 성능 벤치마크")
        print("  python main_engine.py --stress           : 스트레스 테스트")
        print("  python main_engine.py --simulation       : 매매 시뮬레이션")
        print("  python main_engine.py custom.yaml        : 커스텀 설정 파일")
        
        # 안전한 종료
        engine.graceful_shutdown()
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다")
    except Exception as e:    async def _save_analysis_results(self, market_summaries: Dict[str, MarketSummary]):olds = {
            'US': {'buy': 0.65, 'sell': 0.60},
            'JP': {'buy': 0.60, 'sell': 0.55},
            'COIN': {'buy': 0.50, 'sell': 0.45}
        }
        
        # 신호 품질 분석
        signal_stats = {
            'total': len(signals),
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0
        }
        
        for signal in signals:
            try:
                market = signal.market
                thresholds = market_thresholds.get(market, {'buy': 0.60, 'sell': 0.55})
                
                # 신뢰도 분류
                if signal.confidence >= 0.8:
                    signal_stats['high_confidence'] += 1
                elif signal.confidence >= 0.6:
                    signal_stats['medium_confidence'] += 1
                else:
                    signal_stats['low_confidence'] += 1
                
                # 매수 신호 필터링
                if signal.action == 'buy':
                    if signal.confidence >= thresholds['buy']:
                        # 추가 검증 로직
                        if self._validate_buy_signal(signal):
                            filtered_signals.append(signal)
                            self.logger.info(f"✅ {market} 매수 신호 통과: {signal.symbol} ({signal.confidence:.2%})")
                        else:
                            self.logger.debug(f"추가 검증 실패: {signal.symbol}")
                    else:
                        self.logger.debug(f"낮은 신뢰도로 매수 신호 제외: {signal.symbol} ({signal.confidence:.2%})")
                
                # 매도 신호 필터링
                elif signal.action == 'sell':
                    if signal.confidence >= thresholds['sell']:
                        if self._validate_sell_signal(signal):
                            filtered_signals.append(signal)
                            self.logger.info(f"✅ {market} 매도 신호 통과: {signal.symbol} ({signal.confidence:.2%})")
                        else:
                            self.logger.debug(f"추가 검증 실패: {signal.symbol}")
                    else:
                        self.logger.debug(f"낮은 신뢰도로 매도 신호 제외: {signal.symbol} ({signal.confidence:.2%})")
                
            except Exception as e:
                self.logger.error(f"❌ 리스크 관리 중 오류 {signal.symbol}: {e}")
        
        # 리스크 관리 통계 로그
        self.logger.info(f"📊 리스크 관리 결과: {len(filtered_signals)}/{len(signals)} 신호 통과")
        self.logger.debug(f"신뢰도 분포: 높음{signal_stats['high_confidence']} "
                         f"중간{signal_stats['medium_confidence']} 낮음{signal_stats['low_confidence']}")
        
        return filtered_signals

    def _validate_buy_signal(self, signal: UnifiedTradingSignal) -> bool:
        """매수 신호 추가 검증"""
        try:
            # 기본 검증
            if signal.price <= 0:
                return False
            
            if signal.target_price <= signal.price:
                return False
            
            # 시장별 특수 검증
            if signal.market == 'US':
                # 미국 주식: 기본적인 재무 건전성 체크
                if signal.buffett_score is not None and signal.buffett_score < 0.3:
                    return False
                    
            elif signal.market == 'JP':
                # 일본 주식: 엔화 신호와의 일치성 체크
                if signal.yen_signal == 'negative' and signal.stock_type == 'export':
                    return False
                    
            elif signal.market == 'COIN':
                # 암호화폐: 프로젝트 품질 최소 기준
                if signal.project_quality_score is not None and signal.project_quality_score < 0.4:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"매수 신호 검증 오류 {signal.symbol}: {e}")
            return False

    def _validate_sell_signal(self, signal: UnifiedTradingSignal) -> bool:
        """매도 신호 추가 검증"""
        try:
            # 기본 검증
            if signal.price <= 0:
                return False
            
            # 급락 상황에서는 매도 신호 보수적 적용
            if signal.confidence < 0.8 and signal.action == 'sell':
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"매도 신호 검증 오류 {signal.symbol}: {e}")
            return False

    async def _execute_trades(self, signals: List[UnifiedTradingSignal]) -> List[TradeExecution]:
        """매매 신호 실행 (개선된 버전)"""
        executed_trades = []
        
        # 종료 요청 확인
        if self.shutdown_requested:
            self.logger.info("🛑 종료 요청으로 매매 실행 중단")
            return executed_trades
        
        if not self.auto_execution:
            self.logger.info("📊 매매 신호만 생성 (자동 실행 비활성화)")
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
        remaining_trades = self.max_daily_trades - self.daily_trades_count
        if remaining_trades <= 0:
            self.logger.warning(f"⚠️ 일일 거래 한도 도달")
            for signal in signals:
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="일일 거래 한도 초과"
                ))
            return executed_trades
        
        # 우선순위 정렬 (신뢰도 + 시장별 가중치)
        market_priority = {'US': 1.0, 'JP': 0.9, 'COIN': 0.8}
        signals.sort(key=lambda x: x.confidence * market_priority.get(x.market, 0.5), reverse=True)
        
        # 실제 매매 실행
        for i, signal in enumerate(signals):
            if self.shutdown_requested:
                break
                
            if self.daily_trades_count >= self.max_daily_trades:
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="일일 거래 한도 도달"
                ))
                continue
            
            if signal.action in ['buy', 'sell']:
                try:
                    self.logger.info(f"💰 {signal.action.upper()} 주문 실행 시도: {signal.market}-{signal.symbol}")
                    
                    # 수량 계산 (개선된 버전)
                    quantity = self._calculate_position_size(signal)
                    
                    # 수수료 및 슬리피지 계산
                    commission = self._calculate_commission(signal, quantity)
                    slippage = self._calculate_slippage(signal)
                    
                    # 실제 실행 가격 (슬리피지 적용)
                    execution_price = signal.price * (1 + slippage if signal.action == 'buy' else 1 - slippage)
                    
                    # 모의거래 실행
                    if self.paper_trading:
                        execution_result = {
                            'success': True,
                            'price': execution_price,
                            'quantity': quantity,
                            'order_id': f"PAPER_{signal.market}_{datetime.now().strftime('%H%M%S')}_{i:03d}",
                            'commission': commission,
                            'slippage': slippage
                        }
                    else:
                        # 실제 매매 (향후 구현)
                        execution_result = await self._execute_real_trade(signal, quantity)
                    
                    if execution_result['success']:
                        # 예상 수익 계산
                        estimated_profit = self._calculate_estimated_profit(signal, execution_result)
                        
                        executed_trades.append(TradeExecution(
                            signal=signal,
                            executed=True,
                            execution_price=execution_result['price'],
                            execution_time=datetime.now(),
                            quantity=execution_result['quantity'],
                            order_id=execution_result['order_id'],
                            commission=execution_result.get('commission', commission),
                            slippage=execution_result.get('slippage', slippage),
                            estimated_profit=estimated_profit
                        ))
                        
                        self.daily_trades_count += 1
                        self.session_stats['total_trades'] += 1
                        self.session_stats['total_commission'] += commission
                        
                        # 성공 알림
                        if NOTIFIER_AVAILABLE:
                            await self._send_trade_notification(signal, execution_result, estimated_profit)
                        
                        self.logger.info(f"✅ 매매 완료: {signal.market}-{signal.symbol} "
                                       f"{signal.action} @ {execution_result['price']:.4f}")
                    else:
                        executed_trades.append(TradeExecution(
                            signal=signal,
                            executed=False,
                            error_message=execution_result.get('error', 'Unknown error')
                        ))
                        
                except Exception as e:
                    executed_trades.append(TradeExecution(
                        signal=signal,
                        executed=False,
                        error_message=f"실행 오류: {str(e)}"
                    ))
                    self.logger.error(f"❌ 매매 실행 중 오류 {signal.symbol}: {e}")
                    
                # API 제한 고려
                await asyncio.sleep(0.5)
            else:
                # hold 신호
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="HOLD 신호"
                ))
        
        return executed_trades

    def _calculate_position_size(self, signal: UnifiedTradingSignal) -> float:
        """포지션 크기 계산"""
        try:
            if signal.position_size:
                return signal.position_size
            
            # 시장별 기본 포지션 크기
            base_amounts = {
                'US': 10000,    # $10,000
                'JP': 1000000,  # ¥1,000,000  
                'COIN': 1000000  # ₩1,000,000
            }
            
            base_amount = base_amounts.get(signal.market, 100000)
            
            # 신뢰도에 따른 조정
            confidence_multiplier = 0.5 + (signal.confidence * 0.5)  # 0.5 ~ 1.0
            
            adjusted_amount = base_amount * confidence_multiplier
            
            if signal.market in ['US', 'JP']:
                # 주식: 주 수 계산
                return max(1, int(adjusted_amount / signal.price))
            else:
                # 암호화폐: 금액 기준
                return adjusted_amount
                
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 오류: {e}")
            return 100

    def _calculate_commission(self, signal: UnifiedTradingSignal, quantity: float) -> float:
        """수수료 계산"""
        try:
            # 시장별 수수료율
            commission_rates = {
                'US': 0.005,    # 0.5%
                'JP': 0.003,    # 0.3%
                'COIN': 0.0005  # 0.05%
            }
            
            rate = commission_rates.get(signal.market, 0.001)
            
            if signal.market in ['US', 'JP']:
                return signal.price * quantity * rate
            else:
                return quantity * rate
                
        except Exception:
            return 0.0

    def _calculate_slippage(self, signal: UnifiedTradingSignal) -> float:
        """슬리피지 계산"""
        try:
            # 시장별 평균 슬리피지
            slippage_rates = {
                'US': 0.001,    # 0.1%
                'JP': 0.002,    # 0.2%
                'COIN': 0.003   # 0.3%
            }
            
            base_slippage = slippage_rates.get(signal.market, 0.001)
            
            # 신뢰도가 낮을수록 슬리피지 증가
            confidence_factor = 2.0 - signal.confidence  # 1.0 ~ 2.0
            
            return base_slippage * confidence_factor
            
        except Exception:
            return 0.001

    def _calculate_estimated_profit(self, signal: UnifiedTradingSignal, execution_result: Dict) -> float:
        """예상 수익 계산"""
        try:
            if signal.action == 'buy':
                # 매수 시: 목표가까지의 수익
                profit_rate = (signal.target_price - execution_result['price']) / execution_result['price']
                return execution_result['price'] * execution_result['quantity'] * profit_rate
            else:
                # 매도 시: 현재 수익 (가정)
                return execution_result['price'] * execution_result['quantity'] * 0.05  # 5% 가정
                
        except Exception:
            return 0.0

    async def _execute_real_trade(self, signal: UnifiedTradingSignal, quantity: float) -> Dict:
        """실제 매매 실행 (향후 구현)"""
        # 실제 브로커 API 연동 부분
        await asyncio.sleep(0.1)  # API 호출 시뮬레이션
        
        return {
            'success': True,
            'price': signal.price,
            'quantity': quantity,
            'order_id': f"REAL_{signal.market}_{datetime.now().strftime('%H%M%S')}",
            'commission': self._calculate_commission(signal, quantity),
            'slippage': self._calculate_slippage(signal)
        }

    async def _send_trade_notification(self, signal: UnifiedTradingSignal, execution_result: Dict, estimated_profit: float):
        """매매 실행 알림 발송"""
        try:
            if NOTIFIER_AVAILABLE:
                from notifier import send_trading_alert
                
                message = f"{'📈' if signal.action == 'buy' else '📉'} {signal.market} 매매 완료\n"
                message += f"종목: {signal.symbol}\n"
                message += f"동작: {signal.action.upper()}\n"
                message += f"가격: {execution_result['price']:.4f}\n"
                message += f"수량: {execution_result['quantity']:.2f}\n"
                message += f"신뢰도: {signal.confidence:.1%}\n"
                message += f"예상수익: {estimated_profit:,.0f}\n"
                message += f"전략: {signal.strategy}"
                
                await send_trading_alert(
                    signal.market, signal.symbol, signal.action,
                    execution_result['price'], signal.confidence,
                    message, signal.target_price
                )
                
        except Exception as e:
            self.logger.error(f"매매 알림 발송 실패: {e}")

    async def analyze_us_market(self) -> MarketSummary:
        """🇺🇸 미국 시장 분석 (개선된 버전)"""
        start_time = datetime.now()
        errors = []
        signals = []
        executed_trades = []
        
        if not self.us_strategy:
            return self._create_empty_market_summary('US', '전략 비활성화')
        
        try:
            self.logger.info("🔍 미국 시장 분석 시작...")
            
            # 전체 시장 스캔
            from us_strategy import USStockSignal
            us_signals = await self.us_strategy.scan_all_selected_stocks()
            
            # 신호 변환
            for signal in us_signals:
                if self.shutdown_requested:
                    break
                unified_signal = self._convert_to_unified_signal(signal, 'US')
                signals.append(unified_signal)
            
            if self.shutdown_requested:
                return self._create_empty_market_summary('US', '사용자 중단')
            
            # 리스크 관리 및 매매 실행
            filtered_signals = self._apply_risk_management(signals)
            trade_signals = [s for s in filtered_signals if s.action in ['buy', 'sell']]
            
            if trade_signals:
                executed_trades = await self._execute_trades(trade_signals)
            
            return self._create_market_summary('US', signals, executed_trades, start_time, errors)
            
        except Exception as e:
            error_msg = f"미국 시장 분석 실패: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
            return self._create_market_summary('US', signals, executed_trades, start_time, errors)

    async def analyze_jp_market(self) -> MarketSummary:
        """🇯🇵 일본 시장 분석 (개선된 버전)"""
        start_time = datetime.now()
        errors = []
        signals = []
        executed_trades = []
        
        if not self.jp_strategy:
            return self._create_empty_market_summary('JP', '전략 비활성화')
        
        try:
            self.logger.info("🔍 일본 시장 분석 시작...")
            
            # 전체 시장 스캔
            from jp_strategy import JPStockSignal
            jp_signals = await self.jp_strategy.scan_all_symbols()
            
            # 신호 변환
            for signal in jp_signals:
                if self.shutdown_requested:
                    break
                unified_signal = self._convert_to_unified_signal(signal, 'JP')
                signals.append(unified_signal)
            
            if self.shutdown_requested:
                return self._create_empty_market_summary('JP', '사용자 중단')
            
            # 리스크 관리 및 매매 실행
            filtered_signals = self._apply_risk_management(signals)
            trade_signals = [s for s in filtered_signals if s.action in ['buy', 'sell']]
            
            if trade_signals:
                executed_trades = await self._execute_trades(trade_signals)
            
            return self._create_market_summary('JP', signals, executed_trades, start_time, errors)
            
        except Exception as e:
            error_msg = f"일본 시장 분석 실패: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
            return self._create_market_summary('JP', signals, executed_trades, start_time, errors)

    async def analyze_coin_market(self) -> MarketSummary:
        """🪙 암호화폐 시장 분석 (개선된 버전)"""
        start_time = datetime.now()
        errors = []
        signals = []
        executed_trades = []
        
        if not self.coin_strategy:
            return self._create_empty_market_summary('COIN', '전략 비활성화')
        
        try:
            self.logger.info("🔍 암호화폐 시장 분석 시작...")
            
            # 전체 시장 스캔
            from coin_strategy import UltimateCoinSignal
            coin_signals = await self.coin_strategy.scan_all_selected_coins()
            
            # 신호 변환
            for signal in coin_signals:
                if self.shutdown_requested:
                    break
                unified_signal = self._convert_to_unified_signal(signal, 'COIN')
                signals.append(unified_signal)
            
            if self.shutdown_requested:
                return self._create_empty_market_summary('COIN', '사용자 중단')
            
            # 리스크 관리 및 매매 실행
            filtered_signals = self._apply_risk_management(signals)
            trade_signals = [s for s in filtered_signals if s.action in ['buy', 'sell']]
            
            if trade_signals:
                executed_trades = await self._execute_trades(trade_signals)
            
            return self._create_market_summary('COIN', signals, executed_trades, start_time, errors)
            
        except Exception as e:
            error_msg = f"암호화폐 시장 분석 실패: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
            return self._create_market_summary('COIN', signals, executed_trades, start_time, errors)

    def _create_empty_market_summary(self, market: str, reason: str) -> MarketSummary:
        """빈 시장 요약 생성"""
        return MarketSummary(
            market=market,
            total_analyzed=0,
            buy_signals=0,
            sell_signals=0,
            hold_signals=0,
            top_picks=[],
            executed_trades=[],
            analysis_time=0.0,
            errors=[reason],
            is_trading_day=market in self.today_strategies,
            avg_confidence=0.0,
            success_rate=0.0,
            total_estimated_profit=0.0
        )

    def _create_market_summary(self, market: str, signals: List[UnifiedTradingSignal], 
                             executed_trades: List[TradeExecution], start_time: datetime, 
                             errors: List[str]) -> MarketSummary:
        """시장 요약 생성"""
        try:
            # 통계 계산
            buy_signals = len([s for s in signals if s.action == 'buy'])
            sell_signals = len([s for s in signals if s.action == 'sell'])
            hold_signals = len([s for s in signals if s.action == 'hold'])
            
            # 상위 종목 선정
            top_picks = sorted([s for s in signals if s.action == 'buy'], 
                             key=lambda x: x.confidence, reverse=True)[:5]
            
            # 성과 지표
            avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0.0
            successful_trades = len([t for t in executed_trades if t.executed and t.estimated_profit and t.estimated_profit > 0])
            success_rate = successful_trades / len(executed_trades) * 100 if executed_trades else 0.0
            total_estimated_profit = sum([t.estimated_profit or 0 for t in executed_trades if t.executed])
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # 시장별 특화 정보
            market_specific_info = self._get_market_specific_info(market, signals)
            
            self.logger.info(f"✅ {market} 시장 분석 완료 - "
                           f"매수:{buy_signals} 매도:{sell_signals} 보유:{hold_signals} "
                           f"실행:{len([t for t in executed_trades if t.executed])}건")
            
            return MarketSummary(
                market=market,
                total_analyzed=len(signals),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                top_picks=top_picks,
                executed_trades=executed_trades,
                analysis_time=analysis_time,
                errors=errors,
                is_trading_day=market in self.today_strategies,
                avg_confidence=avg_confidence,
                success_rate=success_rate,
                total_estimated_profit=total_estimated_profit,
                market_specific_info=market_specific_info
            )
            
        except Exception as e:
            self.logger.error(f"시장 요약 생성 실패 {market}: {e}")
            return self._create_empty_market_summary(market, f"요약 생성 실패: {str(e)}")

    def _get_market_specific_info(self, market: str, signals: List[UnifiedTradingSignal]) -> Dict:
        """시장별 특화 정보 수집"""
        try:
            if market == 'US':
                return {
                    'avg_buffett_score': np.mean([s.buffett_score for s in signals if s.buffett_score is not None]),
                    'avg_lynch_score': np.mean([s.lynch_score for s in signals if s.lynch_score is not None]),
                    'avg_momentum_score': np.mean([s.momentum_score for s in signals if s.momentum_score is not None]),
                    'avg_technical_score': np.mean([s.technical_score for s in signals if s.technical_score is not None]),
                    'high_quality_count': len([s for s in signals if s.buffett_score and s.buffett_score > 0.7])
                }
            elif market == 'JP':
                return {
                    'yen_positive_signals': len([s for s in signals if s.yen_signal == 'positive']),
                    'export_stocks': len([s for s in signals if s.stock_type == 'export']),
                    'domestic_stocks': len([s for s in signals if s.stock_type == 'domestic']),
                    'avg_selection_score': np.mean([s.selection_score for s in signals if s.selection_score is not None])
                }
            elif market == 'COIN':
                return {
                    'avg_project_quality': np.mean([s.project_quality_score for s in signals if s.project_quality_score is not None]),
                    'bull_cycle_signals': len([s for s in signals if s.market_cycle == 'bull']),
                    'high_correlation_count': len([s for s in signals if s.btc_correlation and abs(s.btc_correlation) > 0.7]),
                    'defi_count': len([s for s in signals if s.sector and 'DeFi' in s.sector])
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"시장 특화 정보 수집 실패 {market}: {e}")
            return {}

    async def run_full_analysis(self) -> Dict[str, MarketSummary]:
        """🌍 전체 시장 통합 분석 (개선된 버전)"""
        self.logger.info("🚀 전체 시장 통합 분석 시작...")
        self.last_analysis_time = datetime.now()
        self.analysis_count += 1
        
        # 종료 요청 확인
        if self.shutdown_requested:
            self.logger.info("🛑 사용자 종료 요청으로 분석 중단")
            return {}
        
        start_time = time.time()
        
        # 병렬 분석 태스크 생성
        tasks = []
        task_names = []
        
        if self.us_strategy:
            tasks.append(self.analyze_us_market())
            task_names.append('US')
            
        if self.jp_strategy:
            tasks.append(self.analyze_jp_market())
            task_names.append('JP')
            
        if self.coin_strategy:
            tasks.append(self.analyze_coin_market())
            task_names.append('COIN')
        
        if not tasks:
            self.logger.warning("⚠️ 활성화된 전략이 없습니다")
            return {}
        
        # 병렬 실행
        self.logger.info(f"⚡ {len(tasks)}개 시장 병렬 분석 시작...")
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"❌ 병렬 분석 실행 실패: {e}")
            return {}
        
    async def _save_analysis_results(self, market_summaries: Dict[str, MarketSummary]):
        """분석 결과 저장 (개선된 버전)"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 데이터 디렉토리 확인
            data_dir = Path('data')
            data_dir.mkdir(exist_ok=True)
            
            # 파일명 생성
            filename = data_dir / f"analysis_{timestamp}.json"
            
            # 저장할 데이터 구성
            save_data = {
                'metadata': {
                    'timestamp': timestamp,
                    'analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                    'analysis_count': self.analysis_count,
                    'version': '6.0.0',
                    'force_test_mode': self.force_test
                },
                'session_info': {
                    'start_time': self.session_start_time.isoformat(),
                    'uptime_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600,
                    'total_signals_generated': self.total_signals_generated,
                    'daily_trades_count': self.daily_trades_count,
                    'max_daily_trades': self.max_daily_trades,
                    'today_strategies': self.today_strategies,
                    'market_allocation': self.market_allocation,
                    'session_stats': self.session_stats
                },
                'system_status': {
                    'auto_execution': self.auto_execution,
                    'paper_trading': self.paper_trading,
                    'shutdown_requested': self.shutdown_requested,
                    'module_availability': {
                        'us_strategy': US_STRATEGY_AVAILABLE,
                        'jp_strategy': JP_STRATEGY_AVAILABLE,
                        'coin_strategy': COIN_STRATEGY_AVAILABLE,
                        'notifier': NOTIFIER_AVAILABLE,
                        'scheduler': SCHEDULER_AVAILABLE,
                        'trading': TRADING_AVAILABLE
                    }
                },
                'market_summaries': {}
            }
            
            # 시장별 요약 데이터 변환
            for market, summary in market_summaries.items():
                # top_picks 직렬화
                top_picks_data = []
                for signal in summary.top_picks:
                    signal_dict = asdict(signal)
                    signal_dict['timestamp'] = signal.timestamp.isoformat()
                    top_picks_data.append(signal_dict)
                
                # executed_trades 직렬화
                executed_trades_data = []
                for trade in summary.executed_trades:
                    trade_dict = {
                        'symbol': trade.signal.symbol,
                        'market': trade.signal.market,
                        'action': trade.signal.action,
                        'executed': trade.executed,
                        'execution_price': trade.execution_price,
                        'execution_time': trade.execution_time.isoformat() if trade.execution_time else None,
                        'quantity': trade.quantity,
                        'order_id': trade.order_id,
                        'error_message': trade.error_message,
                        'commission': trade.commission,
                        'slippage': trade.slippage,
                        'estimated_profit': trade.estimated_profit
                    }
                    executed_trades_data.append(trade_dict)
                
                save_data['market_summaries'][market] = {
                    'market': summary.market,
                    'total_analyzed': summary.total_analyzed,
                    'buy_signals': summary.buy_signals,
                    'sell_signals': summary.sell_signals,
                    'hold_signals': summary.hold_signals,
                    'analysis_time': summary.analysis_time,
                    'errors': summary.errors,
                    'is_trading_day': summary.is_trading_day,
                    'avg_confidence': summary.avg_confidence,
                    'success_rate': summary.success_rate,
                    'total_estimated_profit': summary.total_estimated_profit,
                    'top_picks': top_picks_data,
                    'executed_trades': executed_trades_data,
                    'market_specific_info': summary.market_specific_info or {}
                }
            
            # JSON 파일로 저장
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"💾 분석 결과 저장 완료: {filename}")
            
            # 오래된 파일 정리 (30일 이상)
            await self._cleanup_old_files(data_dir, days=30)
            
        except Exception as e:
            self.logger.error(f"❌ 분석 결과 저장 실패: {e}")

    async def _cleanup_old_files(self, directory: Path, days: int = 30):
        """오래된 파일 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_count = 0
            
            for file_path in directory.glob("analysis_*.json"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                self.logger.info(f"🗑️ 오래된 파일 {deleted_count}개 정리 완료")
                
        except Exception as e:
            self.logger.error(f"파일 정리 실패: {e}")

    async def _send_analysis_notification(self, market_summaries: Dict[str, MarketSummary]):
        """분석 결과 알림 발송 (개선된 버전)"""
        try:
            # 알림 설정 확인
            notification_config = self.config.get('notifications', {})
            telegram_config = notification_config.get('telegram', {})
            
            if not telegram_config.get('enabled', False):
                return
            
            # 요약 메시지 생성
            total_signals = sum([s.total_analyzed for s in market_summaries.values()])
            total_buy = sum([s.buy_signals for s in market_summaries.values()])
            total_executed = sum([len([t for t in s.executed_trades if t.executed]) for s in market_summaries.values()])
            
            message = f"🎯 전체 시장 분석 완료\n\n"
            message += f"📊 전체 요약:\n"
            message += f"• 분석 종목: {total_signals}개\n"
            message += f"• 매수 신호: {total_buy}개\n"
            message += f"• 실행 거래: {total_executed}개\n"
            message += f"• 분석 시간: {datetime.now().strftime('%H:%M:%S')}\n\n"
            
            # 시장별 요약
            market_emojis = {'US': '🇺🇸', 'JP': '🇯🇵', 'COIN': '🪙'}
            for market, summary in market_summaries.items():
                emoji = market_emojis.get(market, '❓')
                executed_count = len([t for t in summary.executed_trades if t.executed])
                
                message += f"{emoji} {market}:\n"
                message += f"  분석 {summary.total_analyzed}개 → 매수 {summary.buy_signals}개 → 실행 {executed_count}개\n"
                
                if summary.top_picks:
                    top_pick = summary.top_picks[0]
                    message += f"  🏆 {top_pick.symbol} ({top_pick.confidence:.1%})\n"
                message += "\n"
            
            # 세션 통계
            uptime_hours = (datetime.now() - self.session_start_time).total_seconds() / 3600
            message += f"📈 세션 통계:\n"
            message += f"• 가동시간: {uptime_hours:.1f}시간\n"
            message += f"• 총 신호: {self.total_signals_generated}개\n"
            message += f"• 일일 거래: {self.daily_trades_count}/{self.max_daily_trades}\n"
            message += f"• 예상 수익: {self.session_stats['total_profit']:,.0f}원\n"
            
            # 알림 발송
            from notifier import send_telegram_message
            await send_telegram_message(message)
            
            self.logger.info("📱 알림 발송 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 알림 발송 실패: {e}")

    async def get_quick_analysis(self, symbols: List[str]) -> List[UnifiedTradingSignal]:
        """빠른 개별 종목 분석 (개선된 버전)"""
        signals = []
        
        if self.shutdown_requested:
            return signals
        
        self.logger.info(f"⚡ 빠른 분석 시작: {len(symbols)}개 종목")
        
        for symbol in symbols:
            if self.shutdown_requested:
                break
                
            try:
                # 시장 판별 개선
                market = self._detect_market(symbol)
                
                if market == 'JP' and self.jp_strategy:
                    from jp_strategy import analyze_jp
                    result = await analyze_jp(symbol)
                    signal = self._create_unified_signal_from_result(result, 'JP', symbol)
                    
                elif market == 'COIN' and self.coin_strategy:
                    from coin_strategy import analyze_coin
                    result = await analyze_coin(symbol)
                    signal = self._create_unified_signal_from_result(result, 'COIN', symbol)
                    
                elif market == 'US' and self.us_strategy:
                    from us_strategy import analyze_us
                    result = await analyze_us(symbol)
                    signal = self._create_unified_signal_from_result(result, 'US', symbol)
                    
                else:
                    # 알 수 없는 시장이거나 전략 비활성화
                    signal = UnifiedTradingSignal(
                        market='UNKNOWN',
                        symbol=symbol,
                        action='hold',
                        confidence=0.0,
                        price=0.0,
                        strategy='unknown_market',
                        reasoning=f"시장 판별 실패 또는 전략 비활성화: {market}",
                        target_price=0.0,
                        timestamp=datetime.now()
                    )
                
                signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"❌ {symbol} 빠른 분석 실패: {e}")
                
                # 오류 신호 생성
                error_signal = UnifiedTradingSignal(
                    market='ERROR',
                    symbol=symbol,
                    action='hold',
                    confidence=0.0,
                    price=0.0,
                    strategy='analysis_error',
                    reasoning=f"분석 오류: {str(e)}",
                    target_price=0.0,
                    timestamp=datetime.now()
                )
                signals.append(error_signal)
        
        self.logger.info(f"⚡ 빠른 분석 완료: {len(signals)}개 신호 생성")
        return signals

    def _detect_market(self, symbol: str) -> str:
        """시장 판별 로직"""
        symbol_upper = symbol.upper()
        
        # 일본 주식
        if symbol_upper.endswith('.T') or symbol_upper.endswith('.JP'):
            return 'JP'
        
        # 암호화폐
        if 'KRW-' in symbol_upper or 'USDT-' in symbol_upper or 'BTC-' in symbol_upper:
            return 'COIN'
        
        # 기타 암호화폐 패턴
        crypto_patterns = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'SOL', 'AVAX', 'MATIC', 'LINK', 'UNI']
        if any(pattern in symbol_upper for pattern in crypto_patterns):
            return 'COIN'
        
        # 기본값: 미국 주식
        return 'US'

    def _create_unified_signal_from_result(self, result: Dict, market: str, symbol: str) -> UnifiedTradingSignal:
        """분석 결과를 통합 신호로 변환"""
        try:
            return UnifiedTradingSignal(
                market=market,
                symbol=symbol,
                action=result.get('decision', 'hold'),
                confidence=result.get('confidence_score', 0) / 100,
                price=result.get('price', result.get('current_price', 0)),
                strategy=f"{market.lower()}_quick",
                reasoning=result.get('reasoning', ''),
                target_price=result.get('target_price', result.get('price', result.get('current_price', 0))),
                timestamp=datetime.now(),
                sector=result.get('sector'),
                
                # 시장별 특수 필드
                buffett_score=result.get('buffett_score', 0) / 100 if market == 'US' else None,
                lynch_score=result.get('lynch_score', 0) / 100 if market == 'US' else None,
                yen_signal=result.get('yen_signal') if market == 'JP' else None,
                stock_type=result.get('stock_type') if market == 'JP' else None,
                project_quality_score=result.get('project_quality_score', 0) / 100 if market == 'COIN' else None,
                market_cycle=result.get('market_cycle') if market == 'COIN' else None,
                btc_correlation=result.get('btc_correlation') if market == 'COIN' else None
            )
            
        except Exception as e:
            self.logger.error(f"결과 변환 실패 {symbol}: {e}")
            return UnifiedTradingSignal(
                market=market, symbol=symbol, action='hold', confidence=0.0,
                price=0.0, strategy='conversion_error', reasoning=f"변환 실패: {str(e)}",
                target_price=0.0, timestamp=datetime.now()
            )

    async def generate_unified_portfolio_report(self, market_summaries: Dict[str, MarketSummary]) -> Dict:
        """📊 통합 포트폴리오 리포트 생성 (개선된 버전)"""
        if not market_summaries:
            return {"error": "분석된 시장이 없습니다", "timestamp": datetime.now().isoformat()}
        
        try:
            # 전체 통계 계산
            total_analyzed = sum([summary.total_analyzed for summary in market_summaries.values()])
            total_buy_signals = sum([summary.buy_signals for summary in market_summaries.values()])
            total_sell_signals = sum([summary.sell_signals for summary in market_summaries.values()])
            total_hold_signals = sum([summary.hold_signals for summary in market_summaries.values()])
            total_executed = sum([len([t for t in summary.executed_trades if t.executed]) for summary in market_summaries.values()])
            total_estimated_profit = sum([summary.total_estimated_profit for summary in market_summaries.values()])
            
            # 시장별 성과 분석
            market_performance = {}
            all_top_picks = []
            
            for market, summary in market_summaries.items():
                executed_trades = [t for t in summary.executed_trades if t.executed]
                
                market_performance[market] = {
                    'analyzed': summary.total_analyzed,
                    'buy_signals': summary.buy_signals,
                    'sell_signals': summary.sell_signals,
                    'hold_signals': summary.hold_signals,
                    'buy_rate': summary.buy_signals / summary.total_analyzed * 100 if summary.total_analyzed > 0 else 0,
                    'executed_trades': len(executed_trades),
                    'execution_rate': len(executed_trades) / max(1, summary.buy_signals + summary.sell_signals) * 100,
                    'avg_confidence': summary.avg_confidence,
                    'success_rate': summary.success_rate,
                    'estimated_profit': summary.total_estimated_profit,
                    'analysis_time': summary.analysis_time,
                    'errors': len(summary.errors),
                    'error_rate': len(summary.errors) / max(1, summary.total_analyzed) * 100,
                    'specific_info': summary.market_specific_info or {}
                }
                
                # 상위 종목들 통합
                all_top_picks.extend(summary.top_picks)
            
            # 글로벌 상위 종목 (신뢰도 순)
            all_top_picks.sort(key=lambda x: x.confidence, reverse=True)
            global_top_picks = all_top_picks[:10]
            
            # 섹터별 분포
            sector_distribution = {}
            for summary in market_summaries.values():
                for signal in summary.top_picks:
                    sector = signal.sector or 'Unknown'
                    sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
            
            # 리스크 분석
            if PANDAS_AVAILABLE:
                confidences = [signal.confidence for summary in market_summaries.values() for signal in summary.top_picks]
                avg_confidence = np.mean(confidences) if confidences else 0
                confidence_std = np.std(confidences) if confidences else 0
            else:
                avg_confidence = sum([s.avg_confidence for s in market_summaries.values()]) / len(market_summaries)
                confidence_std = 0
            
            # 다양성 점수
            market_diversity = len(market_summaries) / 3  # 최대 3개 시장
            sector_diversity = len(sector_distribution) / max(1, total_buy_signals)
            
            # 세션 성과
            uptime_hours = (datetime.now() - self.session_start_time).total_seconds() / 3600
            trades_per_hour = self.daily_trades_count / max(0.1, uptime_hours)
            
            # 최종 리포트 구성
            report = {
                'metadata': {
                    'report_time': datetime.now().isoformat(),
                    'analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                    'analysis_count': self.analysis_count,
                    'version': '6.0.0',
                    'force_test_mode': self.force_test
                },
                'summary': {
                    'total_markets': len(market_summaries),
                    'active_strategies': self.today_strategies,
                    'total_analyzed': total_analyzed,
                    'total_buy_signals': total_buy_signals,
                    'total_sell_signals': total_sell_signals,
                    'total_hold_signals': total_hold_signals,
                    'total_executed': total_executed,
                    'overall_buy_rate': total_buy_signals / total_analyzed * 100 if total_analyzed > 0 else 0,
                    'overall_execution_rate': total_executed / max(1, total_buy_signals + total_sell_signals) * 100,
                    'total_estimated_profit': total_estimated_profit,
                    'session_duration_hours': uptime_hours,
                    'daily_trades_count': self.daily_trades_count,
                    'max_daily_trades': self.max_daily_trades,
                    'trades_per_hour': trades_per_hour
                },
                'market_performance': market_performance,
                'global_top_picks': [
                    {
                        'rank': i + 1,
                        'market': signal.market,
                        'symbol': signal.symbol,
                        'sector': signal.sector,
                        'confidence': signal.confidence,
                        'total_score': signal.total_score or signal.confidence,
                        'price': signal.price,
                        'target_price': signal.target_price,
                        'potential_return': (signal.target_price - signal.price) / signal.price * 100 if signal.price > 0 else 0,
                        'strategy': signal.strategy,
                        'reasoning_summary': signal.reasoning[:150] + "..." if len(signal.reasoning) > 150 else signal.reasoning,
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
                    for i, signal in enumerate(global_top_picks)
                ],
                'diversification_analysis': {
                    'market_diversification_score': market_diversity,
                    'sector_diversification_score': sector_diversity,
                    'sector_distribution': sector_distribution,
                    'market_allocation': self.market_allocation,
                    'concentration_risk': max(sector_distribution.values()) / max(1, sum(sector_distribution.values())) if sector_distribution else 0
                },
                'risk_metrics': {
                    'avg_confidence': avg_confidence,
                    'confidence_volatility': confidence_std,
                    'max_position_size': self.max_position_size,
                    'daily_trades_utilization': self.daily_trades_count / self.max_daily_trades * 100,
                    'error_rate': sum([len(summary.errors) for summary in market_summaries.values()]) / len(market_summaries) * 100,
                    'auto_execution': self.auto_execution,
                    'paper_trading': self.paper_trading,
                    'estimated_portfolio_risk': avg_confidence * market_diversity  # 간단한 리스크 점수
                },
                'session_statistics': {
                    'start_time': self.session_start_time.isoformat(),
                    'uptime_hours': uptime_hours,
                    'total_signals_generated': self.total_signals_generated,
                    'analysis_frequency': self.analysis_count / max(0.1, uptime_hours),
                    'session_stats': self.session_stats,
                    'win_rate': self.session_stats['winning_trades'] / max(1, self.session_stats['total_trades']) * 100
                },
                'system_health': {
                    'shutdown_requested': self.shutdown_requested,
                    'module_availability': {
                        'us_strategy': US_STRATEGY_AVAILABLE and self.us_strategy is not None,
                        'jp_strategy': JP_STRATEGY_AVAILABLE and self.jp_strategy is not None,
                        'coin_strategy': COIN_STRATEGY_AVAILABLE and self.coin_strategy is not None,
                        'notifier': NOTIFIER_AVAILABLE,
                        'scheduler': SCHEDULER_AVAILABLE,
                        'trading': TRADING_AVAILABLE
                    },
                    'config_status': {
                        'config_file': self.config_path,
                        'yaml_available': YAML_AVAILABLE,
                        'pandas_available': PANDAS_AVAILABLE,
                        'dotenv_available': DOTENV_AVAILABLE
                    }
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 통합 리포트 생성 실패: {e}")
            return {
                "error": f"리포트 생성 실패: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "basic_summary": {
                    "markets": len(market_summaries),
                    "total_analyzed": sum([s.total_analyzed for s in market_summaries.values()]),
                    "total_buy_signals": sum([s.buy_signals for s in market_summaries.values()])
                }
            }

    def get_system_status(self) -> Dict:
        """시스템 상태 조회 (개선된 버전)"""
        try:
            uptime = (datetime.now() - self.session_start_time).total_seconds()
            
            return {
                'system_info': {
                    'status': 'running' if not self.shutdown_requested else 'shutting_down',
                    'version': '6.0.0',
                    'uptime_seconds': uptime,
                    'uptime_hours': uptime / 3600,
                    'start_time': self.session_start_time.isoformat(),
                    'current_time': datetime.now().isoformat(),
                    'current_weekday': datetime.now().strftime('%A'),
                    'force_test_mode': self.force_test
                },
                'strategies': {
                    'enabled_strategies': {
                        'us_strategy': self.us_strategy is not None,
                        'jp_strategy': self.jp_strategy is not None,
                        'coin_strategy': self.coin_strategy is not None
                    },
                    'today_strategies': self.today_strategies,
                    'available_modules': {
                        'us_strategy_module': US_STRATEGY_AVAILABLE,
                        'jp_strategy_module': JP_STRATEGY_AVAILABLE,
                        'coin_strategy_module': COIN_STRATEGY_AVAILABLE,
                        'notifier_module': NOTIFIER_AVAILABLE,
                        'scheduler_module': SCHEDULER_AVAILABLE,
                        'trading_module': TRADING_AVAILABLE
                    }
                },
                'trading_status': {
                    'auto_execution': self.auto_execution,
                    'paper_trading': self.paper_trading,
                    'daily_trades_count': self.daily_trades_count,
                    'max_daily_trades': self.max_daily_trades,
                    'trades_remaining': self.max_daily_trades - self.daily_trades_count,
                    'trading_utilization': self.daily_trades_count / self.max_daily_trades * 100
                },
                'analysis_status': {
                    'total_signals_generated': self.total_signals_generated,
                    'analysis_count': self.analysis_count,
                    'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                    'signals_per_analysis': self.total_signals_generated / max(1, self.analysis_count)
                },
                'portfolio_config': {
                    'market_allocation': self.market_allocation,
                    'max_position_size': self.max_position_size,
                    'stop_loss': self.stop_loss,
                    'take_profit': self.take_profit
                },
                'session_performance': self.session_stats,
                'system_dependencies': {
                    'yaml_available': YAML_AVAILABLE,
                    'pandas_available': PANDAS_AVAILABLE,
                    'dotenv_available': DOTENV_AVAILABLE,
                    'config_file_exists': Path(self.config_path).exists()
                },
                'health_indicators': {
                    'memory_usage_ok': True,  # 향후 구현
                    'disk_space_ok': True,    # 향후 구현
                    'network_ok': True,       # 향후 구현
                    'all_systems_ok': not self.shutdown_requested
                }
            }
            
        except Exception as e:
            self.logger.error(f"시스템 상태 조회 실패: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'basic_status': {
                    'uptime_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600,
                    'daily_trades': self.daily_trades_count,
                    'shutdown_requested': self.shutdown_requested
                }
            }
