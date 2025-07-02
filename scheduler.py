#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# =====================================
# 🏆 최고퀸트프로젝트 - 통합 스케줄링 시스템
# =====================================
# 
# 완전 통합 스케줄러:
# - 📅 APScheduler 기반 작업 스케줄링
# - 🌍 글로벌 시장 시간대 관리 (US/JP/COIN)
# - 📊 백테스팅 자동 실행 연동
# - ⏰ 크론/인터벌/단발성 작업 지원
# - 🔔 텔레그램/슬랙 알림 통합
# - 🛡️ 오류 방지 및 자동 복구
# - ⚙️ 설정 파일 완전 연동
#
# 설정 파일: settings.yaml 
# 백테스팅 연동: unified_backtester.py
#
# Author: 최고퀸트팀
# Version: 3.0.0 (통합 + 안정성 강화)
# Project: 최고퀸트프로젝트
# =====================================
"""

import asyncio
import logging
import warnings
import yaml
import os
import pytz
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import traceback

# APScheduler 임포트
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

# 경고 숨기기
warnings.filterwarnings('ignore')

# 설정 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 프로젝트 모듈들 (선택적)
try:
    from utils import TimeZoneManager, ScheduleUtils, get_config
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

try:
    from notifier import send_schedule_notification, send_system_alert
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False

try:
    from unified_backtester import UnifiedBacktestEngine, BacktestConfig
    BACKTESTER_AVAILABLE = True
except ImportError:
    BACKTESTER_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scheduler.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ================================================================================================
# 📊 데이터 모델 및 설정
# ================================================================================================

@dataclass
class TradingSession:
    """거래 세션 정보"""
    market: str
    start_time: time
    end_time: time
    timezone: str
    is_active: bool = True
    session_type: str = "regular"  # regular, premarket, aftermarket, 24/7
    
    def __post_init__(self):
        """데이터 검증"""
        if self.session_type != "24/7" and self.start_time >= self.end_time:
            logger.warning(f"⚠️ {self.market} 세션 시간 오류: {self.start_time} >= {self.end_time}")

@dataclass
class ScheduleEvent:
    """스케줄 이벤트"""
    event_type: str  # market_open, market_close, strategy_start, backtest, notification
    market: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    strategies: List[str] = field(default_factory=list)
    description: str = ""
    priority: str = "normal"  # low, normal, high, critical
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        """기본값 설정"""
        if not self.description:
            self.description = f"{self.market} {self.event_type}"

class SafeSchedulerConfig:
    """안전한 스케줄러 설정 로더"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.schedule_config = self.config.get('schedule', {})
        self.trading_config = self.config.get('trading', {})
    
    def _load_config(self) -> Dict:
        """설정 파일 안전 로드"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ 스케줄러 설정 로드: {self.config_path}")
                return config or {}
            else:
                logger.warning(f"⚠️ 설정 파일 없음: {self.config_path}, 기본값 사용")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """기본 설정값"""
        return {
            'schedule': {
                'weekly_schedule': {
                    'monday': ['COIN'],
                    'tuesday': ['US', 'JP'],
                    'wednesday': [],
                    'thursday': ['US', 'JP'],
                    'friday': ['COIN'],
                    'saturday': [],
                    'sunday': []
                },
                'force_enabled_strategies': [],
                'force_disabled_strategies': [],
                'global_trading_hours': {
                    'start_hour': 0,
                    'end_hour': 24
                },
                'strategy_restrictions': {},
                'auto_backtest': {
                    'enabled': True,
                    'cron': '0 18 * * 1-5',  # 평일 오후 6시
                    'strategies': ['US', 'JP', 'COIN']
                },
                'notifications': {
                    'market_open': True,
                    'market_close': True,
                    'backtest_complete': True,
                    'errors': True
                }
            },
            'trading': {},
            'backtest': {
                'initial_capital': 100000.0,
                'start_date': '2023-01-01',
                'end_date': '2024-12-31'
            }
        }
    
    def get(self, key: str, default=None):
        """설정값 안전 조회"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# ================================================================================================
# 📅 기본 스케줄러 (APScheduler 래퍼)
# ================================================================================================

class BasicScheduler:
    """APScheduler 기반 기본 스케줄러 (원본 코드 호환)"""
    
    def __init__(self, use_async: bool = False):
        """스케줄러 초기화"""
        if not APSCHEDULER_AVAILABLE:
            logger.error("❌ APScheduler가 설치되지 않음")
            raise ImportError("APScheduler 설치 필요: pip install apscheduler")
        
        self.use_async = use_async
        
        if use_async:
            self.scheduler = AsyncIOScheduler()
        else:
            self.scheduler = BackgroundScheduler()
        
        # 이벤트 리스너 등록
        self.scheduler.add_listener(self._job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)
        
        # 시작
        try:
            self.scheduler.start()
            logger.info(f"✅ {'비동기' if use_async else '동기'} 스케줄러 시작")
        except Exception as e:
            logger.error(f"❌ 스케줄러 시작 실패: {e}")
            raise
    
    def _job_listener(self, event):
        """작업 이벤트 리스너"""
        try:
            if event.code == EVENT_JOB_EXECUTED:
                logger.debug(f"✅ 작업 완료: {event.job_id}")
            elif event.code == EVENT_JOB_ERROR:
                logger.error(f"❌ 작업 오류: {event.job_id} - {event.exception}")
            elif event.code == EVENT_JOB_MISSED:
                logger.warning(f"⚠️ 작업 누락: {event.job_id}")
        except Exception as e:
            logger.error(f"❌ 작업 이벤트 처리 실패: {e}")
    
    def add_interval_job(self, func: Callable, seconds: int, job_id: str = None, 
                        args: list = None, kwargs: dict = None, **scheduler_kwargs):
        """초 단위 반복 작업 추가 (원본 호환)"""
        try:
            self.scheduler.add_job(
                func,
                IntervalTrigger(seconds=seconds),
                id=job_id,
                args=args or [],
                kwargs=kwargs or {},
                replace_existing=True,
                **scheduler_kwargs
            )
            logger.info(f"📅 인터벌 작업 등록: {job_id} (매 {seconds}초)")
        except Exception as e:
            logger.error(f"❌ 인터벌 작업 등록 실패: {e}")
    
    def add_cron_job(self, func: Callable, cron_expr: str, job_id: str = None,
                    args: list = None, kwargs: dict = None, **scheduler_kwargs):
        """cron 표현식으로 작업 추가 (원본 호환)"""
        try:
            trigger = CronTrigger.from_crontab(cron_expr)
            self.scheduler.add_job(
                func,
                trigger,
                id=job_id,
                args=args or [],
                kwargs=kwargs or {},
                replace_existing=True,
                **scheduler_kwargs
            )
            logger.info(f"📅 크론 작업 등록: {job_id} ({cron_expr})")
        except Exception as e:
            logger.error(f"❌ 크론 작업 등록 실패: {e}")
    
    def add_date_job(self, func: Callable, run_date: datetime, job_id: str = None,
                    args: list = None, kwargs: dict = None, **scheduler_kwargs):
        """특정 날짜에 한 번 실행되는 작업 추가 (원본 호환)"""
        try:
            self.scheduler.add_job(
                func,
                DateTrigger(run_date=run_date),
                id=job_id,
                args=args or [],
                kwargs=kwargs or {},
                replace_existing=True,
                **scheduler_kwargs
            )
            logger.info(f"📅 단발 작업 등록: {job_id} ({run_date})")
        except Exception as e:
            logger.error(f"❌ 단발 작업 등록 실패: {e}")
    
    def list_jobs(self):
        """등록된 모든 작업 나열 (원본 호환)"""
        try:
            jobs = self.scheduler.get_jobs()
            logger.info(f"📋 등록된 작업 수: {len(jobs)}개")
            for job in jobs:
                logger.info(f"  - {job.id}: {job.next_run_time}")
            return jobs
        except Exception as e:
            logger.error(f"❌ 작업 목록 조회 실패: {e}")
            return []
    
    def remove_job(self, job_id: str):
        """특정 작업 제거 (원본 호환)"""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"🗑️ 작업 제거: {job_id}")
        except Exception as e:
            logger.error(f"❌ 작업 제거 실패: {e}")
    
    def shutdown(self, wait: bool = True):
        """스케줄러 종료 (원본 호환)"""
        try:
            self.scheduler.shutdown(wait=wait)
            logger.info("🛑 스케줄러 종료 완료")
        except Exception as e:
            logger.error(f"❌ 스케줄러 종료 실패: {e}")

# ================================================================================================
# 🏆 통합 거래 스케줄러
# ================================================================================================

class UnifiedTradingScheduler:
    """🏆 통합 거래 스케줄러 (고급 기능 + 기본 스케줄러 통합)"""
    
    def __init__(self, config_path: str = "settings.yaml", use_async: bool = True):
        """통합 스케줄러 초기화"""
        try:
            # 설정 로드
            self.config = SafeSchedulerConfig(config_path)
            
            # 기본 스케줄러 초기화
            self.basic_scheduler = BasicScheduler(use_async=use_async)
            
            # 시간대 관리
            self.tz_manager = TimeZoneManager() if UTILS_AVAILABLE else None
            
            # 거래 세션 정의
            self.trading_sessions = self._define_trading_sessions()
            
            # 백테스팅 엔진 (선택적)
            self.backtest_engine = None
            if BACKTESTER_AVAILABLE:
                try:
                    self.backtest_engine = UnifiedBacktestEngine(config_path)
                    logger.info("✅ 백테스팅 엔진 연동 완료")
                except Exception as e:
                    logger.warning(f"⚠️ 백테스팅 엔진 연동 실패: {e}")
            
            # 상태 관리
            self.session_start_time = datetime.now()
            self.last_run_cache = {}
            self.notification_cache = {}
            self.is_running = True
            
            # 자동 작업 등록
            self._register_auto_jobs()
            
            logger.info("🏆 통합 거래 스케줄러 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 통합 스케줄러 초기화 실패: {e}")
            raise
    
    def _define_trading_sessions(self) -> Dict[str, List[TradingSession]]:
        """시장별 거래 세션 정의"""
        try:
            sessions = {
                'US': [
                    TradingSession('US', time(4, 0), time(9, 30), 'US/Eastern', True, 'premarket'),
                    TradingSession('US', time(9, 30), time(16, 0), 'US/Eastern', True, 'regular'),
                    TradingSession('US', time(16, 0), time(20, 0), 'US/Eastern', True, 'aftermarket')
                ],
                'JP': [
                    TradingSession('JP', time(9, 0), time(11, 30), 'Asia/Tokyo', True, 'morning'),
                    TradingSession('JP', time(12, 30), time(15, 0), 'Asia/Tokyo', True, 'afternoon')
                ],
                'COIN': [
                    TradingSession('COIN', time(0, 0), time(23, 59), 'UTC', True, '24/7')
                ]
            }
            
            # 세션 검증
            for market, market_sessions in sessions.items():
                for session in market_sessions:
                    try:
                        pytz.timezone(session.timezone)
                    except pytz.exceptions.UnknownTimeZoneError:
                        logger.warning(f"⚠️ 알 수 없는 시간대: {session.timezone}, UTC로 대체")
                        session.timezone = 'UTC'
            
            return sessions
            
        except Exception as e:
            logger.error(f"❌ 거래 세션 정의 실패: {e}")
            return {
                'US': [TradingSession('US', time(9, 30), time(16, 0), 'US/Eastern')],
                'JP': [TradingSession('JP', time(9, 0), time(15, 0), 'Asia/Tokyo')],
                'COIN': [TradingSession('COIN', time(0, 0), time(23, 59), 'UTC', True, '24/7')]
            }
    
    def _register_auto_jobs(self):
        """자동 작업 등록"""
        try:
            # 1. 자동 백테스팅
            if self.config.get('schedule.auto_backtest.enabled', True):
                cron_expr = self.config.get('schedule.auto_backtest.cron', '0 18 * * 1-5')
                self.basic_scheduler.add_cron_job(
                    self._run_auto_backtest,
                    cron_expr,
                    job_id='auto_backtest',
                    max_instances=1
                )
                logger.info(f"📊 자동 백테스팅 등록: {cron_expr}")
            
            # 2. 시장 개장 알림
            if self.config.get('schedule.notifications.market_open', True):
                self.basic_scheduler.add_cron_job(
                    self._send_market_open_notification,
                    '0 9 * * 1-5',  # 평일 오전 9시
                    job_id='market_open_notification'
                )
            
            # 3. 시장 마감 알림
            if self.config.get('schedule.notifications.market_close', True):
                self.basic_scheduler.add_cron_job(
                    self._send_market_close_notification,
                    '0 18 * * 1-5',  # 평일 오후 6시
                    job_id='market_close_notification'
                )
            
            # 4. 상태 체크 (10분마다)
            self.basic_scheduler.add_interval_job(
                self._health_check,
                seconds=600,
                job_id='health_check'
            )
            
            # 5. 캐시 정리 (매일 자정)
            self.basic_scheduler.add_cron_job(
                self._cleanup_cache,
                '0 0 * * *',
                job_id='cache_cleanup'
            )
            
        except Exception as e:
            logger.error(f"❌ 자동 작업 등록 실패: {e}")
    
    # ============================================================================================
    # 🎯 핵심 스케줄링 기능 (개선된 원본 코드 기반)
    # ============================================================================================
    
    def get_today_strategies(self, config: Optional[Dict] = None) -> List[str]:
        """오늘 실행할 전략 목록 조회"""
        try:
            if config is None:
                config = self.config.config
            
            current_time = datetime.now()
            weekday = current_time.weekday()
            
            # 요일별 스케줄 조회
            schedule_config = config.get('schedule', {})
            weekly_schedule = schedule_config.get('weekly_schedule', {})
            
            weekday_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            today_key = weekday_names[weekday]
            today_strategies = weekly_schedule.get(today_key, []).copy()
            
            # 공휴일 체크 (UTILS 모듈이 있는 경우)
            if UTILS_AVAILABLE:
                for strategy in today_strategies.copy():
                    try:
                        if not ScheduleUtils.is_trading_day(strategy, current_time.strftime('%Y-%m-%d')):
                            today_strategies.remove(strategy)
                            logger.info(f"📅 {strategy} 시장 휴장일로 제외")
                    except Exception:
                        pass  # 에러시 무시하고 계속
            
            # 강제 설정 적용
            force_enabled = schedule_config.get('force_enabled_strategies', [])
            force_disabled = schedule_config.get('force_disabled_strategies', [])
            
            for strategy in force_enabled:
                if strategy not in today_strategies:
                    today_strategies.append(strategy)
                    logger.info(f"⚡ {strategy} 전략 강제 활성화")
            
            for strategy in force_disabled:
                if strategy in today_strategies:
                    today_strategies.remove(strategy)
                    logger.info(f"🚫 {strategy} 전략 강제 비활성화")
            
            weekday_str = ["월", "화", "수", "목", "금", "토", "일"][weekday]
            logger.info(f"📊 오늘({weekday_str}) 활성 전략: {today_strategies}")
            return today_strategies
            
        except Exception as e:
            logger.error(f"❌ 오늘 전략 조회 실패: {e}")
            # Fallback: 기본 스케줄
            weekday = datetime.now().weekday()
            default_schedule = {
                0: ['COIN'], 1: ['US', 'JP'], 2: [], 3: ['US', 'JP'], 
                4: ['COIN'], 5: [], 6: []
            }
            return default_schedule.get(weekday, [])
    
    def is_trading_time(self, config: Optional[Dict] = None, market: Optional[str] = None) -> bool:
        """현재 시간이 거래 시간인지 확인"""
        try:
            if config is None:
                config = self.config.config
            
            current_time = datetime.now()
            
            # 글로벌 거래 시간 제한 체크
            schedule_config = config.get('schedule', {})
            global_hours = schedule_config.get('global_trading_hours', {})
            
            if global_hours:
                start_hour = global_hours.get('start_hour', 0)
                end_hour = global_hours.get('end_hour', 24)
                current_hour = current_time.hour
                
                if not (start_hour <= current_hour < end_hour):
                    logger.debug(f"⏰ 글로벌 거래 시간 외: {current_hour}시")
                    return False
            
            # 특정 시장 지정된 경우
            if market:
                return self._is_market_trading_time(market, current_time)
            
            # 오늘 활성화된 전략들 중 하나라도 거래 시간이면 True
            today_strategies = self.get_today_strategies(config)
            
            if not today_strategies:
                return False
            
            for strategy in today_strategies:
                if self._is_market_trading_time(strategy, current_time):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 거래 시간 확인 실패: {e}")
            return True  # 에러시 기본적으로 허용
    
    def _is_market_trading_time(self, market: str, check_time: Optional[datetime] = None) -> bool:
        """특정 시장의 거래 시간 확인"""
        try:
            if check_time is None:
                check_time = datetime.now()
            
            if market not in self.trading_sessions:
                logger.warning(f"⚠️ 알 수 없는 시장: {market}")
                return True
            
            sessions = self.trading_sessions[market]
            
            for session in sessions:
                if not session.is_active:
                    continue
                
                # 24/7 시장 (암호화폐)
                if session.session_type == "24/7":
                    return True
                
                # 시간대 변환 및 체크
                try:
                    market_tz = pytz.timezone(session.timezone)
                    market_time = check_time.astimezone(market_tz)
                    current_time_only = market_time.time()
                    
                    if session.start_time <= current_time_only <= session.end_time:
                        return True
                        
                except Exception as e:
                    logger.warning(f"⚠️ {market} 시간대 변환 실패: {e}")
                    # 로컬 시간으로 대략 체크
                    current_time_only = check_time.time()
                    if session.start_time <= current_time_only <= session.end_time:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ {market} 시장 거래 시간 확인 실패: {e}")
            return True
    
    def should_run_strategy(self, strategy: str, check_time: Optional[datetime] = None) -> bool:
        """특정 전략을 실행해야 하는지 확인"""
        try:
            if check_time is None:
                check_time = datetime.now()
            
            # 1. 오늘 활성화된 전략인지 확인
            today_strategies = self.get_today_strategies()
            if strategy not in today_strategies:
                return False
            
            # 2. 해당 시장의 거래 시간인지 확인
            if not self._is_market_trading_time(strategy, check_time):
                return False
            
            # 3. 전략별 설정 확인
            strategy_config = self.config.config.get(f'{strategy.lower()}_strategy', {})
            if not strategy_config.get('enabled', True):
                return False
            
            # 4. 실행 제한 확인
            restrictions = self.config.get('schedule.strategy_restrictions', {})
            if strategy in restrictions:
                restriction = restrictions[strategy]
                
                # 허용 시간대 체크
                if 'allowed_hours' in restriction:
                    allowed_hours = restriction['allowed_hours']
                    if check_time.hour not in allowed_hours:
                        return False
                
                # 최소 실행 간격 체크
                if 'min_interval_minutes' in restriction:
                    min_interval = restriction['min_interval_minutes']
                    last_run = self.last_run_cache.get(strategy)
                    
                    if last_run:
                        time_since_last = (check_time - last_run).total_seconds() / 60
                        if time_since_last < min_interval:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {strategy} 전략 실행 가능성 확인 실패: {e}")
            return False
    
    def get_schedule_status(self) -> Dict:
        """스케줄러 상태 조회"""
        try:
            current_time = datetime.now()
            today_strategies = self.get_today_strategies()
            
            status = {
                'scheduler_status': 'running' if self.is_running else 'stopped',
                'current_time': current_time.isoformat(),
                'session_uptime': str(current_time - self.session_start_time).split('.')[0],
                'today_strategies': today_strategies,
                'trading_day': len(today_strategies) > 0,
                'trading_time': self.is_trading_time(),
                'market_status': {},
                'next_session': None,
                'job_count': len(self.basic_scheduler.list_jobs()),
                'config_status': {
                    'config_file_exists': self.config.config_path.exists(),
                    'utils_available': UTILS_AVAILABLE,
                    'notifier_available': NOTIFIER_AVAILABLE,
                    'backtester_available': BACKTESTER_AVAILABLE,
                    'apscheduler_available': APSCHEDULER_AVAILABLE
                }
            }
            
            # 시장별 상태
            for strategy in ['US', 'JP', 'COIN']:
                is_active = strategy in today_strategies
                is_trading = self._is_market_trading_time(strategy, current_time)
                should_run = self.should_run_strategy(strategy, current_time)
                
                sessions_info = []
                if strategy in self.trading_sessions:
                    for session in self.trading_sessions[strategy]:
                        sessions_info.append({
                            'type': session.session_type,
                            'start': session.start_time.strftime('%H:%M'),
                            'end': session.end_time.strftime('%H:%M'),
                            'timezone': session.timezone,
                            'is_active': session.is_active
                        })
                
                status['market_status'][strategy] = {
                    'active_today': is_active,
                    'trading_now': is_trading,
                    'should_run': should_run,
                    'sessions': sessions_info,
                    'last_run': self.last_run_cache.get(strategy, {}).get('timestamp') if isinstance(self.last_run_cache.get(strategy), dict) else str(self.last_run_cache.get(strategy)) if self.last_run_cache.get(strategy) else None
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ 스케줄러 상태 조회 실패: {e}")
            return {
                'scheduler_status': 'error',
                'error': str(e),
                'current_time': datetime.now().isoformat()
            }
    
    def update_last_run(self, strategy: str, run_time: Optional[datetime] = None):
        """전략 마지막 실행 시간 업데이트"""
        if run_time is None:
            run_time = datetime.now()
        self.last_run_cache[strategy] = run_time
        logger.debug(f"📝 {strategy} 전략 마지막 실행 시간 업데이트: {run_time}")
    
    # ============================================================================================
    # 🚀 자동 작업 함수들
    # ============================================================================================
    
    def _run_auto_backtest(self):
        """자동 백테스팅 실행"""
        try:
            if not self.backtest_engine:
                logger.warning("⚠️ 백테스팅 엔진이 없어 자동 백테스팅 건너뜀")
                return
            
            logger.info("🚀 자동 백테스팅 시작")
            
            # 비동기 백테스팅을 동기 함수에서 실행
            import asyncio
            
            async def run_backtest():
                try:
                    # 백테스팅 설정 생성
                    config = BacktestConfig(
                        start_date=self.config.get('backtest.start_date', '2023-01-01'),
                        end_date=self.config.get('backtest.end_date', '2024-12-31'),
                        initial_capital=self.config.get('backtest.initial_capital', 100000.0)
                    )
                    
                    result = await self.backtest_engine.run_backtest(config)
                    
                    # 결과 저장
                    self.backtest_engine.save_results(result)
                    
                    # 알림 발송
                    if NOTIFIER_AVAILABLE and self.config.get('schedule.notifications.backtest_complete', True):
                        await self._send_backtest_complete_notification(result)
                    
                    logger.info("✅ 자동 백테스팅 완료")
                    
                except Exception as e:
                    logger.error(f"❌ 자동 백테스팅 실행 실패: {e}")
                    
                    # 오류 알림
                    if NOTIFIER_AVAILABLE and self.config.get('schedule.notifications.errors', True):
                        await send_system_alert("error", f"자동 백테스팅 실패: {str(e)}", "high")
            
            # 현재 이벤트 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 루프가 있으면 태스크로 추가
                asyncio.create_task(run_backtest())
            except RuntimeError:
                # 실행 중인 루프가 없으면 새로 실행
                asyncio.run(run_backtest())
                
        except Exception as e:
            logger.error(f"❌ 자동 백테스팅 함수 실패: {e}")
    
    def _send_market_open_notification(self):
        """시장 개장 알림 발송"""
        try:
            if not NOTIFIER_AVAILABLE:
                return
            
            today_strategies = self.get_today_strategies()
            
            if not today_strategies:
                logger.info("📅 오늘 활성 전략이 없어 개장 알림 생략")
                return
            
            async def send_notification():
                try:
                    await send_schedule_notification(today_strategies, "start")
                    logger.info("📱 시장 개장 알림 발송 완료")
                except Exception as e:
                    logger.error(f"❌ 시장 개장 알림 발송 실패: {e}")
            
            # 비동기 함수 실행
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(send_notification())
            except RuntimeError:
                asyncio.run(send_notification())
                
        except Exception as e:
            logger.error(f"❌ 시장 개장 알림 함수 실패: {e}")
    
    def _send_market_close_notification(self):
        """시장 마감 알림 발송"""
        try:
            if not NOTIFIER_AVAILABLE:
                return
            
            today_strategies = self.get_today_strategies()
            
            if not today_strategies:
                return
            
            async def send_notification():
                try:
                    await send_schedule_notification(today_strategies, "end")
                    logger.info("📱 시장 마감 알림 발송 완료")
                except Exception as e:
                    logger.error(f"❌ 시장 마감 알림 발송 실패: {e}")
            
            # 비동기 함수 실행
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(send_notification())
            except RuntimeError:
                asyncio.run(send_notification())
                
        except Exception as e:
            logger.error(f"❌ 시장 마감 알림 함수 실패: {e}")
    
    async def _send_backtest_complete_notification(self, result):
        """백테스팅 완료 알림"""
        try:
            if not NOTIFIER_AVAILABLE:
                return
            
            metrics = result.performance_metrics
            message = f"📊 자동 백테스팅 완료\n\n"
            message += f"💰 총 수익률: {metrics.total_return*100:+.2f}%\n"
            message += f"📈 연간 수익률: {metrics.annual_return*100:+.2f}%\n"
            message += f"⚡ 샤프 비율: {metrics.sharpe_ratio:.3f}\n"
            message += f"📉 최대 손실폭: {metrics.max_drawdown*100:.2f}%\n"
            message += f"💼 총 거래: {metrics.total_trades}건\n"
            message += f"🎯 승률: {metrics.win_rate*100:.1f}%"
            
            await send_system_alert("info", message, "normal")
            logger.info("📱 백테스팅 완료 알림 발송")
            
        except Exception as e:
            logger.error(f"❌ 백테스팅 완료 알림 실패: {e}")
    
    def _health_check(self):
        """스케줄러 상태 체크"""
        try:
            current_time = datetime.now()
            
            # 기본 상태 체크
            jobs = self.basic_scheduler.list_jobs()
            job_count = len(jobs)
            
            # 실행 중인 작업 확인
            running_jobs = [job for job in jobs if job.next_run_time]
            
            logger.info(f"💓 스케줄러 상태 체크: {job_count}개 작업 등록, {len(running_jobs)}개 대기 중")
            
            # 메모리 사용량 체크 (선택적)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > 500:  # 500MB 초과시 경고
                    logger.warning(f"⚠️ 메모리 사용량 높음: {memory_mb:.1f}MB")
                    
            except ImportError:
                pass  # psutil이 없으면 건너뜀
            
            # 에러 알림 (필요시)
            if job_count == 0:
                logger.warning("⚠️ 등록된 작업이 없음")
                
        except Exception as e:
            logger.error(f"❌ 상태 체크 실패: {e}")
    
    def _cleanup_cache(self):
        """캐시 정리"""
        try:
            current_time = datetime.now()
            
            # 1일 이전 캐시 정리
            cutoff_time = current_time - timedelta(days=1)
            
            # 마지막 실행 시간 캐시 정리
            for strategy, last_run in list(self.last_run_cache.items()):
                if isinstance(last_run, datetime) and last_run < cutoff_time:
                    del self.last_run_cache[strategy]
                    logger.debug(f"🧹 {strategy} 캐시 정리")
            
            # 알림 캐시 정리
            for key, timestamp in list(self.notification_cache.items()):
                if isinstance(timestamp, datetime) and timestamp < cutoff_time:
                    del self.notification_cache[key]
                    logger.debug(f"🧹 알림 캐시 정리: {key}")
            
            logger.info("🧹 캐시 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 캐시 정리 실패: {e}")
    
    # ============================================================================================
    # 📅 고급 스케줄링 기능
    # ============================================================================================
    
    def add_strategy_job(self, strategy: str, func: Callable, cron_expr: str, 
                        job_id: Optional[str] = None, **kwargs):
        """전략별 작업 등록"""
        try:
            if job_id is None:
                job_id = f"strategy_{strategy}_{func.__name__}"
            
            # 전략 실행 가능성 체크 래퍼
            def strategy_wrapper(*args, **func_kwargs):
                if self.should_run_strategy(strategy):
                    logger.info(f"🚀 {strategy} 전략 작업 실행: {func.__name__}")
                    result = func(*args, **func_kwargs)
                    self.update_last_run(strategy)
                    return result
                else:
                    logger.debug(f"⏸️ {strategy} 전략 작업 건너뜀: {func.__name__}")
                    return None
            
            self.basic_scheduler.add_cron_job(
                strategy_wrapper,
                cron_expr,
                job_id,
                **kwargs
            )
            
            logger.info(f"📊 {strategy} 전략 작업 등록: {job_id} ({cron_expr})")
            
        except Exception as e:
            logger.error(f"❌ {strategy} 전략 작업 등록 실패: {e}")
    
    def add_market_session_job(self, market: str, session_type: str, event_type: str,
                              func: Callable, job_id: Optional[str] = None):
        """시장 세션별 작업 등록"""
        try:
            if market not in self.trading_sessions:
                logger.error(f"❌ 알 수 없는 시장: {market}")
                return
            
            # 해당 세션 찾기
            target_session = None
            for session in self.trading_sessions[market]:
                if session.session_type == session_type:
                    target_session = session
                    break
            
            if not target_session:
                logger.error(f"❌ {market}에서 {session_type} 세션을 찾을 수 없음")
                return
            
            if job_id is None:
                job_id = f"{market}_{session_type}_{event_type}"
            
            # 시간 설정
            if event_type == "open":
                target_time = target_session.start_time
            elif event_type == "close":
                target_time = target_session.end_time
            else:
                logger.error(f"❌ 알 수 없는 이벤트 타입: {event_type}")
                return
            
            # 24/7 시장은 세션 작업 불가
            if target_session.session_type == "24/7":
                logger.warning(f"⚠️ {market}은 24/7 시장으로 세션 작업 등록 불가")
                return
            
            # 크론 표현식 생성 (평일만)
            cron_expr = f"{target_time.minute} {target_time.hour} * * 1-5"
            
            self.basic_scheduler.add_cron_job(
                func,
                cron_expr,
                job_id
            )
            
            logger.info(f"🌍 {market} {session_type} {event_type} 작업 등록: {cron_expr}")
            
        except Exception as e:
            logger.error(f"❌ 시장 세션 작업 등록 실패: {e}")
    
    def schedule_backtest(self, cron_expr: str, strategies: List[str] = None,
                         config: Optional[Dict] = None, job_id: str = "custom_backtest"):
        """사용자 정의 백테스팅 스케줄"""
        try:
            if not self.backtest_engine:
                logger.error("❌ 백테스팅 엔진이 없어 스케줄 등록 불가")
                return
            
            if strategies is None:
                strategies = ['US', 'JP', 'COIN']
            
            def backtest_job():
                logger.info(f"🚀 사용자 정의 백테스팅 시작: {strategies}")
                self._run_auto_backtest()
            
            self.basic_scheduler.add_cron_job(
                backtest_job,
                cron_expr,
                job_id
            )
            
            logger.info(f"📊 사용자 정의 백테스팅 스케줄 등록: {cron_expr}")
            
        except Exception as e:
            logger.error(f"❌ 백테스팅 스케줄 등록 실패: {e}")
    
    def add_conditional_job(self, condition_func: Callable, action_func: Callable,
                           check_interval: int = 60, job_id: Optional[str] = None):
        """조건부 작업 등록"""
        try:
            if job_id is None:
                job_id = f"conditional_{action_func.__name__}"
            
            def conditional_wrapper():
                try:
                    if condition_func():
                        logger.info(f"✅ 조건 충족, 작업 실행: {action_func.__name__}")
                        return action_func()
                    else:
                        logger.debug(f"⏸️ 조건 미충족, 작업 건너뜀: {action_func.__name__}")
                        return None
                except Exception as e:
                    logger.error(f"❌ 조건부 작업 실행 실패: {e}")
            
            self.basic_scheduler.add_interval_job(
                conditional_wrapper,
                check_interval,
                job_id
            )
            
            logger.info(f"🔍 조건부 작업 등록: {job_id} (매 {check_interval}초 체크)")
            
        except Exception as e:
            logger.error(f"❌ 조건부 작업 등록 실패: {e}")
    
    # ============================================================================================
    # 🛠️ 유틸리티 및 관리 기능
    # ============================================================================================
    
    def get_job_status(self, job_id: str) -> Dict:
        """특정 작업 상태 조회"""
        try:
            jobs = self.basic_scheduler.list_jobs()
            target_job = None
            
            for job in jobs:
                if job.id == job_id:
                    target_job = job
                    break
            
            if not target_job:
                return {'found': False, 'error': f'작업을 찾을 수 없음: {job_id}'}
            
            return {
                'found': True,
                'id': target_job.id,
                'name': target_job.name,
                'func': str(target_job.func),
                'trigger': str(target_job.trigger),
                'next_run_time': target_job.next_run_time.isoformat() if target_job.next_run_time else None,
                'coalesce': target_job.coalesce,
                'max_instances': target_job.max_instances,
                'misfire_grace_time': target_job.misfire_grace_time
            }
            
        except Exception as e:
            logger.error(f"❌ 작업 상태 조회 실패: {e}")
            return {'found': False, 'error': str(e)}
    
    def pause_job(self, job_id: str) -> bool:
        """작업 일시정지"""
        try:
            self.basic_scheduler.scheduler.pause_job(job_id)
            logger.info(f"⏸️ 작업 일시정지: {job_id}")
            return True
        except Exception as e:
            logger.error(f"❌ 작업 일시정지 실패: {e}")
            return False
    
    def resume_job(self, job_id: str) -> bool:
        """작업 재개"""
        try:
            self.basic_scheduler.scheduler.resume_job(job_id)
            logger.info(f"▶️ 작업 재개: {job_id}")
            return True
        except Exception as e:
            logger.error(f"❌ 작업 재개 실패: {e}")
            return False
    
    def modify_job(self, job_id: str, **changes) -> bool:
        """작업 수정"""
        try:
            self.basic_scheduler.scheduler.modify_job(job_id, **changes)
            logger.info(f"✏️ 작업 수정: {job_id}")
            return True
        except Exception as e:
            logger.error(f"❌ 작업 수정 실패: {e}")
            return False
    
    def get_all_jobs_summary(self) -> Dict:
        """모든 작업 요약"""
        try:
            jobs = self.basic_scheduler.list_jobs()
            
            summary = {
                'total_jobs': len(jobs),
                'running_jobs': 0,
                'paused_jobs': 0,
                'jobs_by_type': {'cron': 0, 'interval': 0, 'date': 0},
                'next_execution': None,
                'jobs': []
            }
            
            next_times = []
            
            for job in jobs:
                job_info = {
                    'id': job.id,
                    'name': job.name or job.id,
                    'trigger_type': type(job.trigger).__name__,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                    'is_paused': hasattr(job, '_scheduler') and job._scheduler.state == 2
                }
                
                summary['jobs'].append(job_info)
                
                # 통계 업데이트
                if job.next_run_time:
                    summary['running_jobs'] += 1
                    next_times.append(job.next_run_time)
                else:
                    summary['paused_jobs'] += 1
                
                # 트리거 타입별 통계
                trigger_type = type(job.trigger).__name__.lower()
                if 'cron' in trigger_type:
                    summary['jobs_by_type']['cron'] += 1
                elif 'interval' in trigger_type:
                    summary['jobs_by_type']['interval'] += 1
                elif 'date' in trigger_type:
                    summary['jobs_by_type']['date'] += 1
            
            # 다음 실행 시간
            if next_times:
                summary['next_execution'] = min(next_times).isoformat()
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 작업 요약 생성 실패: {e}")
            return {'error': str(e)}
    
    def export_schedule_config(self, output_path: str = "schedule_export.yaml") -> bool:
        """스케줄 설정 내보내기"""
        try:
            jobs = self.basic_scheduler.list_jobs()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'scheduler_config': self.config.config,
                'active_jobs': [],
                'market_sessions': {}
            }
            
            # 활성 작업 정보
            for job in jobs:
                job_data = {
                    'id': job.id,
                    'name': job.name,
                    'trigger': str(job.trigger),
                    'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None
                }
                export_data['active_jobs'].append(job_data)
            
            # 시장 세션 정보
            for market, sessions in self.trading_sessions.items():
                export_data['market_sessions'][market] = []
                for session in sessions:
                    session_data = {
                        'session_type': session.session_type,
                        'start_time': session.start_time.strftime('%H:%M'),
                        'end_time': session.end_time.strftime('%H:%M'),
                        'timezone': session.timezone,
                        'is_active': session.is_active
                    }
                    export_data['market_sessions'][market].append(session_data)
            
            # 파일 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"📤 스케줄 설정 내보내기 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 스케줄 설정 내보내기 실패: {e}")
            return False
    
    # ============================================================================================
    # 🔧 기본 스케줄러 인터페이스 (원본 호환)
    # ============================================================================================
    
    def add_interval_job(self, func: Callable, seconds: int, job_id: str = None, **kwargs):
        """인터벌 작업 추가 (원본 호환)"""
        return self.basic_scheduler.add_interval_job(func, seconds, job_id, **kwargs)
    
    def add_cron_job(self, func: Callable, cron_expr: str, job_id: str = None, **kwargs):
        """크론 작업 추가 (원본 호환)"""
        return self.basic_scheduler.add_cron_job(func, cron_expr, job_id, **kwargs)
    
    def add_date_job(self, func: Callable, run_date: datetime, job_id: str = None, **kwargs):
        """단발 작업 추가 (원본 호환)"""
        return self.basic_scheduler.add_date_job(func, run_date, job_id, **kwargs)
    
    def list_jobs(self):
        """작업 목록 조회 (원본 호환)"""
        return self.basic_scheduler.list_jobs()
    
    def remove_job(self, job_id: str):
        """작업 제거 (원본 호환)"""
        return self.basic_scheduler.remove_job(job_id)
    
    def shutdown(self, wait: bool = True):
        """스케줄러 종료 (원본 호환)"""
        try:
            self.is_running = False
            self.basic_scheduler.shutdown(wait)
            logger.info("🛑 통합 스케줄러 종료 완료")
        except Exception as e:
            logger.error(f"❌ 통합 스케줄러 종료 실패: {e}")

# ================================================================================================
# 📚 편의 함수들 (전역 접근)
# ================================================================================================

_unified_scheduler_instance = None

def get_unified_scheduler() -> UnifiedTradingScheduler:
    """통합 스케줄러 싱글톤 인스턴스"""
    global _unified_scheduler_instance
    if _unified_scheduler_instance is None:
        _unified_scheduler_instance = UnifiedTradingScheduler()
    return _unified_scheduler_instance

def get_today_strategies() -> List[str]:
    """오늘 실행할 전략 목록 (편의 함수)"""
    try:
        scheduler = get_unified_scheduler()
        return scheduler.get_today_strategies()
    except Exception as e:
        logger.error(f"❌ 오늘 전략 조회 실패: {e}")
        weekday = datetime.now().weekday()
        default = {0: ['COIN'], 1: ['US', 'JP'], 2: [], 3: ['US', 'JP'], 4: ['COIN'], 5: [], 6: []}
        return default.get(weekday, [])

def is_trading_time(market: str = None) -> bool:
    """거래 시간 확인 (편의 함수)"""
    try:
        scheduler = get_unified_scheduler()
        return scheduler.is_trading_time(market=market)
    except Exception as e:
        logger.error(f"❌ 거래 시간 확인 실패: {e}")
        return True

def should_run_strategy(strategy: str) -> bool:
    """전략 실행 가능성 확인 (편의 함수)"""
    try:
        scheduler = get_unified_scheduler()
        return scheduler.should_run_strategy(strategy)
    except Exception as e:
        logger.error(f"❌ 전략 실행 가능성 확인 실패: {e}")
        return False

def get_schedule_status() -> Dict:
    """스케줄러 상태 조회 (편의 함수)"""
    try:
        scheduler = get_unified_scheduler()
        return scheduler.get_schedule_status()
    except Exception as e:
        logger.error(f"❌ 스케줄러 상태 조회 실패: {e}")
        return {'error': str(e)}

# ================================================================================================
# 🧪 테스트 및 데모 함수
# ================================================================================================

def demo_task():
    """데모 작업 함수"""
    current_time = datetime.now()
    logger.info(f"🎯 데모 작업 실행: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 현재 활성 전략 출력
    strategies = get_today_strategies()
    logger.info(f"📊 현재 활성 전략: {strategies}")
    
    # 거래 시간 체크
    trading = is_trading_time()
    logger.info(f"⏰ 거래 시간: {'Yes' if trading else 'No'}")

async def test_unified_scheduler():
    """🧪 통합 스케줄러 종합 테스트"""
    print("\n" + "="*80)
    print("🧪 최고퀸트프로젝트 - 통합 스케줄러 테스트")
    print("="*80)
    
    try:
        # 1. 스케줄러 초기화
        print("1️⃣ 통합 스케줄러 초기화...")
        scheduler = UnifiedTradingScheduler()
        print(f"   ✅ 완료 - 기본 스케줄러: {APSCHEDULER_AVAILABLE}")
        print(f"   ✅ 백테스팅 연동: {BACKTESTER_AVAILABLE}")
        print(f"   ✅ 알림 시스템: {NOTIFIER_AVAILABLE}")
        
        # 2. 현재 상태 확인
        print("\n2️⃣ 현재 스케줄 상태...")
        status = scheduler.get_schedule_status()
        print(f"   📅 오늘 전략: {status['today_strategies']}")
        print(f"   ⏰ 거래 시간: {status['trading_time']}")
        print(f"   💼 등록된 작업: {status['job_count']}개")
        
        # 3. 시장별 상세 상태
        print("\n3️⃣ 시장별 상세 상태...")
        for market, market_status in status['market_status'].items():
            market_name = {'US': '🇺🇸미국', 'JP': '🇯🇵일본', 'COIN': '🪙암호화폐'}.get(market, market)
            active = "🟢" if market_status['should_run'] else "🔴"
            print(f"   {market_name}: {active} 활성({market_status['active_today']}) 개장({market_status['trading_now']}) 실행가능({market_status['should_run']})")
            
            # 세션 정보
            for session in market_status['sessions'][:1]:  # 첫 번째 세션만
                print(f"        └─ {session['type']}: {session['start']}-{session['end']} ({session['timezone']})")
        
        # 4. 데모 작업 등록
        print("\n4️⃣ 데모 작업 등록...")
        
        # 간단한 인터벌 작업
        scheduler.add_interval_job(demo_task, seconds=5, job_id='demo_interval')
        print("   ✅ 5초 간격 데모 작업 등록")
        
        # 크론 작업 (매분 0초에 실행)
        scheduler.add_cron_job(demo_task, '0 * * * *', job_id='demo_cron')
        print("   ✅ 매시간 0분 데모 작업 등록")
        
        # 5분 후 단발 작업
        future_time = datetime.now() + timedelta(minutes=5)
        scheduler.add_date_job(demo_task, future_time, job_id='demo_date')
        print(f"   ✅ 5분 후 단발 작업 등록: {future_time.strftime('%H:%M:%S')}")
        
        # 5. 작업 목록 확인
        print("\n5️⃣ 등록된 작업 목록...")
        jobs_summary = scheduler.get_all_jobs_summary()
        print(f"   📊 총 작업: {jobs_summary['total_jobs']}개")
        print(f"   ▶️ 실행 중: {jobs_summary['running_jobs']}개")
        print(f"   ⏸️ 일시정지: {jobs_summary['paused_jobs']}개")
        print(f"   📅 다음 실행: {jobs_summary['next_execution']}")
        
        # 작업 상세 정보
        for job in jobs_summary['jobs'][:5]:  # 처음 5개만
            print(f"     - {job['id']}: {job['next_run'] or 'N/A'}")
        
        # 6. 전략별 작업 등록 데모
        print("\n6️⃣ 전략별 작업 등록 데모...")
        
        def us_strategy_demo():
            if should_run_strategy('US'):
                logger.info("🇺🇸 미국 전략 데모 실행")
            else:
                logger.info("🇺🇸 미국 전략 실행 조건 미충족")
        
        def jp_strategy_demo():
            if should_run_strategy('JP'):
                logger.info("🇯🇵 일본 전략 데모 실행")
            else:
                logger.info("🇯🇵 일본 전략 실행 조건 미충족")
        
        def coin_strategy_demo():
            if should_run_strategy('COIN'):
                logger.info("🪙 코인 전략 데모 실행")
            else:
                logger.info("🪙 코인 전략 실행 조건 미충족")
        
        # 전략별 작업 등록
        scheduler.add_strategy_job('US', us_strategy_demo, '0 9 * * 1-5', 'us_demo')
        scheduler.add_strategy_job('JP', jp_strategy_demo, '0 9 * * 1-5', 'jp_demo')
        scheduler.add_strategy_job('COIN', coin_strategy_demo, '0 */4 * * *', 'coin_demo')
        
        print("   ✅ 미국 전략 작업: 평일 오전 9시")
        print("   ✅ 일본 전략 작업: 평일 오전 9시")
        print("   ✅ 코인 전략 작업: 4시간마다")
        
        # 7. 조건부 작업 데모
        print("\n7️⃣ 조건부 작업 데모...")
        
        def market_condition():
            """시장이 열려있으면 True"""
            return is_trading_time()
        
        def market_action():
            """시장이 열려있을 때 실행되는 작업"""
            logger.info("📈 시장 개장 중 - 조건부 작업 실행")
        
        scheduler.add_conditional_job(market_condition, market_action, 30, 'market_conditional')
        print("   ✅ 시장 개장시에만 실행되는 조건부 작업 등록 (30초마다 체크)")
        
        # 8. 최종 상태 확인
        print("\n8️⃣ 최종 상태 확인...")
        final_summary = scheduler.get_all_jobs_summary()
        print(f"   📊 최종 등록 작업: {final_summary['total_jobs']}개")
        print(f"   🕒 다음 실행: {final_summary['next_execution']}")
        
        # 작업 타입별 분포
        types = final_summary['jobs_by_type']
        print(f"   📋 작업 타입: 크론({types['cron']}) 인터벌({types['interval']}) 단발({types['date']})")
        
        print("\n✅ 모든 테스트 완료!")
        print("\n⏰ 데모 작업들이 백그라운드에서 실행됩니다...")
        print("⚠️ 종료하려면 Ctrl+C를 누르세요")
        
        return scheduler
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        traceback.print_exc()
        return None

def run_scheduler_demo():
    """스케줄러 데모 실행"""
    print("🚀 최고퀸트프로젝트 - 통합 스케줄러 데모 시작")
    
    async def demo_main():
        scheduler = await test_unified_scheduler()
        
        if scheduler:
            try:
                # 무한 대기 (Ctrl+C로 종료)
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 사용자에 의해 중단됨")
                scheduler.shutdown()
                print("✅ 스케줄러 정상 종료")
    
    try:
        asyncio.run(demo_main())
    except KeyboardInterrupt:
        print("\n🛑 프로그램 종료")

# ================================================================================================
# 🔧 CLI 인터페이스
# ================================================================================================

def run_scheduler_cli():
    """CLI 인터페이스"""
    import argparse
    
    parser = argparse.ArgumentParser(description='최고퀸트프로젝트 통합 스케줄러')
    parser.add_argument('--mode', choices=['test', 'demo', 'status', 'export'], 
                       default='demo', help='실행 모드')
    parser.add_argument('--config', default='settings.yaml', 
                       help='설정 파일 경로')
    parser.add_argument('--output', default='schedule_export.yaml',
                       help='내보내기 파일 경로')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'test':
            # 테스트만 실행
            asyncio.run(test_unified_scheduler())
            
        elif args.mode == 'demo':
            # 데모 실행
            run_scheduler_demo()
            
        elif args.mode == 'status':
            # 상태 조회만
            scheduler = UnifiedTradingScheduler(args.config)
            status = scheduler.get_schedule_status()
            
            print("\n📊 스케줄러 상태")
            print("="*50)
            print(f"상태: {status['scheduler_status']}")
            print(f"가동 시간: {status['session_uptime']}")
            print(f"오늘 전략: {status['today_strategies']}")
            print(f"거래 시간: {status['trading_time']}")
            print(f"등록 작업: {status['job_count']}개")
            
            scheduler.shutdown(wait=False)
            
        elif args.mode == 'export':
            # 설정 내보내기
            scheduler = UnifiedTradingScheduler(args.config)
            success = scheduler.export_schedule_config(args.output)
            
            if success:
                print(f"✅ 스케줄 설정 내보내기 완료: {args.output}")
            else:
                print("❌ 내보내기 실패")
                
            scheduler.shutdown(wait=False)
    
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")

# ================================================================================================
# 🚀 메인 실행
# ================================================================================================

if __name__ == "__main__":
    try:
        # 환경 확인
        logger.info("🔍 환경 확인...")
        logger.info(f"✅ APScheduler: {'사용 가능' if APSCHEDULER_AVAILABLE else '설치 필요'}")
        logger.info(f"✅ 백테스팅 연동: {'사용 가능' if BACKTESTER_AVAILABLE else '선택사항'}")
        logger.info(f"✅ 알림 시스템: {'사용 가능' if NOTIFIER_AVAILABLE else '선택사항'}")
        logger.info(f"✅ 유틸리티: {'사용 가능' if UTILS_AVAILABLE else '선택사항'}")
        
        if not APSCHEDULER_AVAILABLE:
            print("❌ APScheduler가 설치되지 않았습니다.")
            print("설치 명령: pip install apscheduler")
            exit(1)
        
        # CLI 실행
        run_scheduler_cli()
        
    except Exception as e:
        logger.error(f"❌ 프로그램 실행 실패: {e}")
        print("\n🔧 문제 해결 방법:")
        print("1. 필수 패키지 설치: pip install apscheduler")
        print("2. 설정 파일 확인: settings.yaml")
        print("3. 간단한 테스트: python unified_scheduler.py --mode test")
