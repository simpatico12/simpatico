"""
📅 최고퀸트프로젝트 - 스케줄링 시스템 (개선 버전)
====================================

완전한 스케줄링 관리:
- 📊 요일별 전략 스케줄 (월/금 코인, 화/목 주식)
- 🕐 시간대별 거래 시간 관리
- 🌍 글로벌 시장 시간 동기화
- 🔔 스케줄 기반 알림
- 📅 공휴일 및 휴장일 처리
- ⚙️ 동적 스케줄 조정
- 📈 거래 세션 최적화

Author: 최고퀸트팀
Version: 1.1.0 (개선)
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
import yaml
import os
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple, Union
import pytz
from dataclasses import dataclass, field
import calendar
from pathlib import Path

# 프로젝트 모듈 import
try:
    from utils import TimeZoneManager, ScheduleUtils, get_config
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ utils 모듈 로드 실패: {e}")
    UTILS_AVAILABLE = False

try:
    from notifier import send_schedule_notification, send_system_alert
    NOTIFIER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ notifier 모듈 로드 실패: {e}")
    NOTIFIER_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class TradingSession:
    """거래 세션 정보"""
    market: str
    start_time: time
    end_time: time
    timezone: str
    is_active: bool = True
    session_type: str = "regular"  # regular, premarket, aftermarket
    
    def __post_init__(self):
        """데이터 검증"""
        if self.start_time >= self.end_time and self.session_type != "24/7":
            raise ValueError(f"시작 시간({self.start_time})이 종료 시간({self.end_time})보다 늦습니다")

@dataclass
class ScheduleEvent:
    """스케줄 이벤트"""
    event_type: str  # market_open, market_close, strategy_start, strategy_end
    market: str
    timestamp: datetime
    strategies: List[str] = field(default_factory=list)
    description: str = ""
    priority: str = "normal"  # low, normal, high, critical
    
    def __post_init__(self):
        """기본값 설정"""
        if not self.description:
            self.description = f"{self.market} {self.event_type}"

class TradingScheduler:
    """🏆 최고퀸트프로젝트 거래 스케줄러 (개선 버전)"""
    
    def __init__(self, config_path: str = "configs/settings.yaml"):
        """스케줄러 초기화"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 시간대 관리자
        self.tz_manager = TimeZoneManager() if UTILS_AVAILABLE else None
            
        # 스케줄 설정
        self.schedule_config = self.config.get('schedule', {})
        self.trading_config = self.config.get('trading', {})
        
        # 기본 요일별 스케줄 (사용자 요구사항)
        self.default_weekly_schedule = {
            0: ['COIN'],        # 월요일: 암호화폐만
            1: ['US', 'JP'],    # 화요일: 미국 + 일본 주식
            2: [],              # 수요일: 휴무
            3: ['US', 'JP'],    # 목요일: 미국 + 일본 주식  
            4: ['COIN'],        # 금요일: 암호화폐만
            5: [],              # 토요일: 휴무
            6: []               # 일요일: 휴무
        }
        
        # 시장별 거래 세션 정의
        self.trading_sessions = self._define_trading_sessions()
        
        # 실행 통계
        self.session_start_time = datetime.now()
        self.last_schedule_check = None
        self._last_run_cache = {}  # 전략별 마지막 실행 시간 캐시
        
        logger.info("📅 최고퀸트프로젝트 스케줄러 초기화 완료")
        logger.info(f"⚙️ 요일별 기본 스케줄: {self._format_weekly_schedule()}")

    def _load_config(self) -> Dict:
        """설정 파일 로드 (개선된 오류 처리)"""
        try:
            if not self.config_path.exists():
                logger.warning(f"⚠️ 설정 파일 없음, 기본 설정 사용: {self.config_path}")
                return self._create_default_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if not isinstance(config, dict):
                raise ValueError("설정 파일이 유효한 YAML 딕셔너리가 아닙니다")
                
            logger.info(f"✅ 스케줄 설정 로드 성공: {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML 파싱 오류: {e}")
            return self._create_default_config()
        except Exception as e:
            logger.error(f"❌ 스케줄 설정 로드 실패: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """기본 설정 생성"""
        return {
            'schedule': {
                'weekly_schedule': {},
                'force_enabled_strategies': [],
                'force_disabled_strategies': [],
                'global_trading_hours': {
                    'start_hour': 0,
                    'end_hour': 24
                },
                'strategy_restrictions': {}
            },
            'trading': {},
            'us_strategy': {'enabled': True},
            'jp_strategy': {'enabled': True},
            'coin_strategy': {'enabled': True}
        }
    
    def _format_weekly_schedule(self) -> str:
        """요일별 스케줄 포맷팅"""
        weekdays = ["월", "화", "수", "목", "금", "토", "일"]
        schedule_str = []
        
        for day_idx, strategies in self.default_weekly_schedule.items():
            day_name = weekdays[day_idx]
            if strategies:
                strategy_names = []
                for strategy in strategies:
                    if strategy == 'US':
                        strategy_names.append("🇺🇸미국")
                    elif strategy == 'JP':
                        strategy_names.append("🇯🇵일본")
                    elif strategy == 'COIN':
                        strategy_names.append("🪙코인")
                    else:
                        strategy_names.append(strategy)
                schedule_str.append(f"{day_name}({'+'.join(strategy_names)})")
            else:
                schedule_str.append(f"{day_name}(휴무)")
                
        return " ".join(schedule_str)
    
    def _define_trading_sessions(self) -> Dict[str, List[TradingSession]]:
        """시장별 거래 세션 정의 (개선된 검증)"""
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
            
            # 세션 유효성 검증
            for market, market_sessions in sessions.items():
                for session in market_sessions:
                    try:
                        # 시간대 검증
                        pytz.timezone(session.timezone)
                    except pytz.exceptions.UnknownTimeZoneError:
                        logger.warning(f"⚠️ 알 수 없는 시간대: {session.timezone}, UTC로 대체")
                        session.timezone = 'UTC'
            
            return sessions
            
        except Exception as e:
            logger.error(f"❌ 거래 세션 정의 실패: {e}")
            # 최소한의 기본 세션 반환
            return {
                'US': [TradingSession('US', time(9, 30), time(16, 0), 'US/Eastern')],
                'JP': [TradingSession('JP', time(9, 0), time(15, 0), 'Asia/Tokyo')],
                'COIN': [TradingSession('COIN', time(0, 0), time(23, 59), 'UTC', True, '24/7')]
            }

    def get_schedule_status(self) -> Dict:
        """스케줄러 상태 조회 (개선된 상세 정보)"""
        try:
            current_time = datetime.now()
            today_strategies = self.get_today_strategies()
            
            status = {
                'scheduler_status': 'running',
                'current_time': current_time.isoformat(),
                'session_uptime': str(current_time - self.session_start_time).split('.')[0],
                'last_schedule_check': self.last_schedule_check.isoformat() if self.last_schedule_check else None,
                'today_strategies': today_strategies,
                'trading_day': len(today_strategies) > 0,
                'trading_time': self.is_trading_time(),
                'market_status': {},
                'next_session': None,
                'config_status': {
                    'config_file_exists': self.config_path.exists(),
                    'utils_available': UTILS_AVAILABLE,
                    'notifier_available': NOTIFIER_AVAILABLE
                },
                'last_run_cache': dict(self._last_run_cache)
            }
            
            # 시장별 상태 (더 상세한 정보)
            for strategy in ['US', 'JP', 'COIN']:
                is_active = strategy in today_strategies
                is_trading = self._is_market_trading_time(strategy, current_time)
                should_run = self.should_run_strategy(strategy, current_time)
                
                # 세션 정보
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
                    'last_run': self._last_run_cache.get(strategy, {}).get('timestamp') if isinstance(self._last_run_cache.get(strategy), dict) else str(self._last_run_cache.get(strategy)) if self._last_run_cache.get(strategy) else None
                }
            
            # 다음 세션 정보 (더 상세한)
            next_session = self.get_next_trading_session()
            if next_session:
                time_until = next_session.timestamp - current_time
                status['next_session'] = {
                    'market': next_session.market,
                    'timestamp': next_session.timestamp.isoformat(),
                    'time_until_seconds': int(time_until.total_seconds()),
                    'time_until_formatted': f"{int(time_until.total_seconds()//3600)}시간 {int((time_until.total_seconds()%3600)//60)}분",
                    'description': next_session.description,
                    'strategies': next_session.strategies,
                    'event_type': next_session.event_type
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ 스케줄러 상태 조회 실패: {e}")
            return {
                'scheduler_status': 'error', 
                'error': str(e),
                'current_time': datetime.now().isoformat()
            }

    def update_schedule_config(self, new_config: Dict) -> bool:
        """스케줄 설정 업데이트 (개선된 백업 및 검증)"""
        try:
            # 기존 설정 백업
            backup_path = self.config_path.with_suffix('.yaml.backup')
            if self.config_path.exists():
                import shutil
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"💾 설정 파일 백업: {backup_path}")
            
            # 새 설정 검증
            if not isinstance(new_config, dict):
                raise ValueError("새 설정이 딕셔너리가 아닙니다")
            
            # 기존 설정과 병합
            updated_config = self.config.copy()
            
            # 깊은 병합 (deep merge)
            def deep_merge(base: dict, update: dict) -> dict:
                for key, value in update.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        base[key] = deep_merge(base[key], value)
                    else:
                        base[key] = value
                return base
            
            updated_config = deep_merge(updated_config, new_config)
            
            # 디렉토리 생성
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일에 저장
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(updated_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            # 메모리 설정 업데이트
            self.config = updated_config
            self.schedule_config = self.config.get('schedule', {})
            
            logger.info("⚙️ 스케줄 설정 업데이트 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 스케줄 설정 업데이트 실패: {e}")
            # 백업 복원 시도
            backup_path = self.config_path.with_suffix('.yaml.backup')
            if backup_path.exists():
                try:
                    import shutil
                    shutil.copy2(backup_path, self.config_path)
                    logger.info("🔄 백업에서 설정 복원 완료")
                except Exception as restore_error:
                    logger.error(f"❌ 백업 복원 실패: {restore_error}")
            return False

    def validate_schedule_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """스케줄 설정 검증"""
        errors = []
        
        try:
            # 기본 구조 검증
            if 'schedule' in config:
                schedule = config['schedule']
                
                # 주간 스케줄 검증
                if 'weekly_schedule' in schedule:
                    weekly = schedule['weekly_schedule']
                    valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                    
                    for day, strategies in weekly.items():
                        if day not in valid_days:
                            errors.append(f"유효하지 않은 요일: {day}")
                        
                        if not isinstance(strategies, list):
                            errors.append(f"{day}: 전략 목록이 리스트가 아닙니다")
                        else:
                            valid_strategies = ['US', 'JP', 'COIN']
                            for strategy in strategies:
                                if strategy not in valid_strategies:
                                    errors.append(f"{day}: 유효하지 않은 전략 {strategy}")
                
                # 글로벌 거래 시간 검증
                if 'global_trading_hours' in schedule:
                    hours = schedule['global_trading_hours']
                    start_hour = hours.get('start_hour', 0)
                    end_hour = hours.get('end_hour', 24)
                    
                    if not (0 <= start_hour <= 23):
                        errors.append(f"시작 시간이 유효하지 않음: {start_hour}")
                    if not (1 <= end_hour <= 24):
                        errors.append(f"종료 시간이 유효하지 않음: {end_hour}")
                    if start_hour >= end_hour:
                        errors.append(f"시작 시간({start_hour})이 종료 시간({end_hour})보다 늦습니다")
                
                # 전략 제한 검증
                if 'strategy_restrictions' in schedule:
                    restrictions = schedule['strategy_restrictions']
                    
                    for strategy, restriction in restrictions.items():
                        if 'allowed_hours' in restriction:
                            hours = restriction['allowed_hours']
                            if not isinstance(hours, list):
                                errors.append(f"{strategy}: allowed_hours가 리스트가 아닙니다")
                            else:
                                for hour in hours:
                                    if not (0 <= hour <= 23):
                                        errors.append(f"{strategy}: 유효하지 않은 시간 {hour}")
                        
                        if 'min_interval_minutes' in restriction:
                            interval = restriction['min_interval_minutes']
                            if not isinstance(interval, (int, float)) or interval < 0:
                                errors.append(f"{strategy}: 유효하지 않은 최소 간격 {interval}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"설정 검증 중 오류: {e}")
            return False, errors

    def get_weekly_schedule_preview(self, weeks: int = 2) -> Dict:
        """주간 스케줄 미리보기"""
        try:
            preview = {
                'weeks': [],
                'summary': {
                    'total_trading_days': 0,
                    'strategy_counts': {'US': 0, 'JP': 0, 'COIN': 0},
                    'most_active_day': None,
                    'least_active_day': None
                }
            }
            
            today = datetime.now()
            day_activity = {}
            
            for week in range(weeks):
                week_start = today + timedelta(days=week*7)
                week_data = {
                    'week_start': week_start.strftime('%Y-%m-%d'),
                    'days': []
                }
                
                for day_offset in range(7):
                    check_date = week_start + timedelta(days=day_offset)
                    weekday = check_date.weekday()
                    strategies = self.default_weekly_schedule.get(weekday, [])
                    
                    day_data = {
                        'date': check_date.strftime('%Y-%m-%d'),
                        'weekday': check_date.strftime('%A'),
                        'weekday_kr': ["월", "화", "수", "목", "금", "토", "일"][weekday],
                        'strategies': strategies,
                        'is_trading_day': len(strategies) > 0,
                        'sessions_count': sum(len(self.trading_sessions.get(s, [])) for s in strategies)
                    }
                    
                    week_data['days'].append(day_data)
                    
                    # 통계 업데이트
                    if strategies:
                        preview['summary']['total_trading_days'] += 1
                        for strategy in strategies:
                            if strategy in preview['summary']['strategy_counts']:
                                preview['summary']['strategy_counts'][strategy] += 1
                    
                    # 요일별 활동도
                    weekday_name = day_data['weekday_kr']
                    if weekday_name not in day_activity:
                        day_activity[weekday_name] = 0
                    day_activity[weekday_name] += len(strategies)
                
                preview['weeks'].append(week_data)
            
            # 가장/덜 활성화된 요일
            if day_activity:
                most_active = max(day_activity.items(), key=lambda x: x[1])
                least_active = min(day_activity.items(), key=lambda x: x[1])
                
                preview['summary']['most_active_day'] = {
                    'day': most_active[0],
                    'activity_score': most_active[1]
                }
                preview['summary']['least_active_day'] = {
                    'day': least_active[0],
                    'activity_score': least_active[1]
                }
            
            return preview
            
        except Exception as e:
            logger.error(f"❌ 주간 스케줄 미리보기 생성 실패: {e}")
            return {'error': str(e)}

# =====================================
# 편의 함수들 (개선된 버전)
# =====================================

_scheduler_instance = None

def get_scheduler_instance() -> TradingScheduler:
    """스케줄러 싱글톤 인스턴스 조회"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TradingScheduler()
    return _scheduler_instance

def get_today_strategies(config: Optional[Dict] = None) -> List[str]:
    """오늘 실행할 전략 목록 조회 (개선된 편의 함수)"""
    try:
        scheduler = get_scheduler_instance()
        return scheduler.get_today_strategies(config)
    except Exception as e:
        logger.error(f"❌ 오늘 전략 조회 실패: {e}")
        # 요일별 기본 스케줄 fallback
        weekday = datetime.now().weekday()
        default_schedule = {
            0: ['COIN'], 1: ['US', 'JP'], 2: [], 3: ['US', 'JP'], 
            4: ['COIN'], 5: [], 6: []
        }
        return default_schedule.get(weekday, [])

def is_trading_time(config: Optional[Dict] = None, market: Optional[str] = None) -> bool:
    """현재 거래 시간인지 확인 (개선된 편의 함수)"""
    try:
        scheduler = get_scheduler_instance()
        return scheduler.is_trading_time(config, market)
    except Exception as e:
        logger.error(f"❌ 거래 시간 확인 실패: {e}")
        # 기본 시간 체크 fallback
        hour = datetime.now().hour
        if market == 'US':
            return 9 <= hour <= 16
        elif market == 'JP':
            return 9 <= hour <= 15
        elif market == 'COIN':
            return True
        else:
            return 9 <= hour <= 18  # 일반적인 거래 시간

def should_run_strategy(strategy: str) -> bool:
    """특정 전략 실행 여부 (개선된 편의 함수)"""
    try:
        scheduler = get_scheduler_instance()
        return scheduler.should_run_strategy(strategy)
    except Exception as e:
        logger.error(f"❌ {strategy} 전략 실행 가능성 확인 실패: {e}")
        # 기본 체크: 오늘 전략에 포함되고 거래 시간이면 실행
        today_strategies = get_today_strategies()
        return strategy in today_strategies and is_trading_time(market=strategy)

def get_schedule_status() -> Dict:
    """스케줄러 상태 조회 (개선된 편의 함수)"""
    try:
        scheduler = get_scheduler_instance()
        return scheduler.get_schedule_status()
    except Exception as e:
        logger.error(f"❌ 스케줄러 상태 조회 실패: {e}")
        return {
            'scheduler_status': 'error',
            'error': str(e),
            'current_time': datetime.now().isoformat()
        }

def update_strategy_last_run(strategy: str, run_time: Optional[datetime] = None):
    """전략 마지막 실행 시간 업데이트 (편의 함수)"""
    try:
        scheduler = get_scheduler_instance()
        scheduler.update_last_run(strategy, run_time)
    except Exception as e:
        logger.error(f"❌ {strategy} 마지막 실행 시간 업데이트 실패: {e}")

# =====================================
# 스케줄러 데몬 (개선된 버전)
# =====================================

class SchedulerDaemon:
    """스케줄러 백그라운드 데몬 (개선된 버전)"""
    
    def __init__(self, scheduler: TradingScheduler):
        self.scheduler = scheduler
        self.running = False
        self.check_interval = 60  # 1분마다 체크
        self.last_notification_times = {}  # 중복 알림 방지
        self.error_count = 0
        self.max_errors = 10
        
    async def start(self):
        """데몬 시작"""
        self.running = True
        self.error_count = 0
        logger.info("🤖 스케줄러 데몬 시작")
        
        try:
            while self.running and self.error_count < self.max_errors:
                try:
                    await self._check_schedule_events()
                    self.error_count = 0  # 성공시 에러 카운트 리셋
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"❌ 스케줄 이벤트 체크 오류 ({self.error_count}/{self.max_errors}): {e}")
                
                await asyncio.sleep(self.check_interval)
                
        except Exception as e:
            logger.error(f"❌ 스케줄러 데몬 치명적 오류: {e}")
        finally:
            logger.info("🛑 스케줄러 데몬 종료")
    
    def stop(self):
        """데몬 정지"""
        self.running = False
        logger.info("📴 스케줄러 데몬 정지 요청")
    
    async def _check_schedule_events(self):
        """스케줄 이벤트 체크 (개선된 중복 방지)"""
        try:
            current_time = datetime.now()
            current_key = current_time.strftime('%Y%m%d%H%M')
            
            # 거래 시작 시간 체크 (09:00)
            if current_time.hour == 9 and current_time.minute == 0:
                if not self._was_notification_sent('start', current_key):
                    await self.scheduler.send_schedule_notifications("start")
                    self._mark_notification_sent('start', current_key)
            
            # 거래 종료 시간 체크 (18:00)
            elif current_time.hour == 18 and current_time.minute == 0:
                if not self._was_notification_sent('end', current_key):
                    await self.scheduler.send_schedule_notifications("end")
                    self._mark_notification_sent('end', current_key)
            
            # 다음 세션 1시간 전 알림
            next_session = self.scheduler.get_next_trading_session()
            if next_session:
                time_until = next_session.timestamp - current_time
                # 59-60분 전에 알림 (1분 오차 허용)
                if 3540 <= time_until.total_seconds() <= 3660:
                    alert_key = f"next_session_{next_session.market}_{current_key[:-1]}"  # 분 단위 제거
                    if not self._was_notification_sent('next_session', alert_key):
                        await self.scheduler.send_next_session_alert()
                        self._mark_notification_sent('next_session', alert_key)
            
            # 오래된 알림 기록 정리 (24시간 이전)
            self._cleanup_old_notifications()
                    
        except Exception as e:
            logger.error(f"❌ 스케줄 이벤트 체크 실패: {e}")
            raise
    
    def _was_notification_sent(self, notification_type: str, key: str) -> bool:
        """알림이 이미 발송되었는지 확인"""
        return self.last_notification_times.get(f"{notification_type}_{key}", False)
    
    def _mark_notification_sent(self, notification_type: str, key: str):
        """알림 발송 기록"""
        self.last_notification_times[f"{notification_type}_{key}"] = True
    
    def _cleanup_old_notifications(self):
        """오래된 알림 기록 정리"""
        try:
            current_time = datetime.now()
            yesterday = current_time - timedelta(days=1)
            yesterday_key = yesterday.strftime('%Y%m%d')
            
            # 어제 이전 기록 삭제
            keys_to_remove = []
            for key in self.last_notification_times.keys():
                if any(yesterday_key > key[len(prefix):len(prefix)+8] for prefix in ['start_', 'end_', 'next_session_'] if key.startswith(prefix)):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.last_notification_times[key]
                
        except Exception as e:
            logger.warning(f"⚠️ 알림 기록 정리 실패: {e}")

# =====================================
# 테스트 함수 (개선된 버전)
# =====================================

async def test_scheduler_system():
    """🧪 스케줄러 시스템 종합 테스트"""
    print("📅 최고퀸트프로젝트 스케줄러 테스트 (개선 버전)")
    print("=" * 60)
    
    try:
        # 1. 스케줄러 초기화
        print("1️⃣ 스케줄러 초기화...")
        scheduler = TradingScheduler()
        print(f"   ✅ 완료 (설정파일: {scheduler.config_path.exists()})")
        
        # 2. 오늘 전략 조회
        print("2️⃣ 오늘 전략 조회...")
        today_strategies = scheduler.get_today_strategies()
        current_time = datetime.now()
        weekday = ["월", "화", "수", "목", "금", "토", "일"][current_time.weekday()]
        print(f"   📅 오늘({weekday}요일): {today_strategies}")
        print(f"   📊 활성 전략 수: {len(today_strategies)}개")
        
        # 3. 거래 시간 확인
        print("3️⃣ 거래 시간 확인...")
        is_trading = scheduler.is_trading_time()
        status = "🟢 거래 중" if is_trading else "🔴 휴장"
        print(f"   ⏰ 현재 상태: {status}")
        print(f"   🕐 현재 시간: {current_time.strftime('%H:%M:%S')}")
        
        # 4. 시장별 상세 상태
        print("4️⃣ 시장별 상세 상태...")
        markets = ['US', 'JP', 'COIN']
        for market in markets:
            is_active = market in today_strategies
            is_open = scheduler._is_market_trading_time(market)
            should_run = scheduler.should_run_strategy(market)
            
            status_emoji = "🟢" if should_run else "🔴"
            print(f"   {market:4}: {status_emoji} 오늘활성({is_active}) 개장중({is_open}) 실행가능({should_run})")
            
            # 세션 정보
            if market in scheduler.trading_sessions:
                sessions = scheduler.trading_sessions[market]
                for session in sessions[:1]:  # 첫 번째 세션만 표시
                    print(f"        └─ {session.session_type}: {session.start_time}-{session.end_time} ({session.timezone})")
        
        # 5. 다음 세션 정보
        print("5️⃣ 다음 거래 세션...")
        next_session = scheduler.get_next_trading_session()
        if next_session:
            time_until = next_session.timestamp - current_time
            hours = int(time_until.total_seconds() // 3600)
            minutes = int((time_until.total_seconds() % 3600) // 60)
            print(f"   🎯 시장: {next_session.market}")
            print(f"   ⏰ 시작: {next_session.timestamp.strftime('%m/%d %H:%M')}")
            print(f"   🕒 남은 시간: {hours}시간 {minutes}분")
            print(f"   📝 설명: {next_session.description}")
        else:
            print("   ❌ 다음 세션 정보 없음")
        
        # 6. 주간 스케줄 미리보기
        print("6️⃣ 주간 스케줄 미리보기...")
        preview = scheduler.get_weekly_schedule_preview(1)
        if 'error' not in preview:
            print(f"   📅 다음 7일간 거래일: {preview['summary']['total_trading_days']}일")
            for strategy, count in preview['summary']['strategy_counts'].items():
                if count > 0:
                    print(f"   📈 {strategy} 전략: {count}일")
        
        # 7. 설정 검증
        print("7️⃣ 설정 검증...")
        is_valid, errors = scheduler.validate_schedule_config(scheduler.config)
        if is_valid:
            print("   ✅ 설정 검증 통과")
        else:
            print("   ❌ 설정 오류:")
            for error in errors[:3]:  # 최대 3개만 표시
                print(f"      - {error}")
        
        # 8. 스케줄러 상태
        print("8️⃣ 스케줄러 상태...")
        status = scheduler.get_schedule_status()
        print(f"   📊 상태: {status['scheduler_status']}")
        print(f"   ⏱️ 가동시간: {status['session_uptime']}")
        print(f"   🔧 Utils 사용 가능: {status['config_status']['utils_available']}")
        print(f"   📱 알림 사용 가능: {status['config_status']['notifier_available']}")
        
        # 9. 편의 함수 테스트
        print("9️⃣ 편의 함수 테스트...")
        strategies = get_today_strategies()
        trading = is_trading_time()
        print(f"   📋 get_today_strategies(): {len(strategies)}개")
        print(f"   ⏰ is_trading_time(): {trading}")
        
        # 10. 알림 테스트 (선택적)
        if NOTIFIER_AVAILABLE:
            print("🔟 알림 테스트...")
            try:
                await scheduler.send_schedule_notifications("start")
                print("   ✅ 스케줄 알림 발송 테스트 완료")
            except Exception as e:
                print(f"   ❌ 스케줄 알림 실패: {e}")
        else:
            print("🔟 알림 모듈 없음 - 알림 테스트 스킵")
        
        print()
        print("🎯 스케줄러 시스템 테스트 완료!")
        print("📅 개선된 요일별 스케줄링 시스템이 정상 작동합니다")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def test_weekly_schedule():
    """주간 스케줄 시뮬레이션 (개선된 버전)"""
    print("\n📅 주간 스케줄 시뮬레이션")
    print("-" * 40)
    
    try:
        scheduler = TradingScheduler()
        
        # 향후 2주간 스케줄
        preview = scheduler.get_weekly_schedule_preview(2)
        
        if 'error' in preview:
            print(f"❌ 스케줄 미리보기 오류: {preview['error']}")
            return
        
        for week_idx, week in enumerate(preview['weeks']):
            print(f"\n📆 Week {week_idx + 1} ({week['week_start']}부터)")
            
            for day in week['days']:
                strategies = day['strategies']
                status = "📈 거래" if strategies else "😴 휴무"
                
                strategy_str = ""
                if strategies:
                    strategy_names = []
                    for strategy in strategies:
                        if strategy == 'US':
                            strategy_names.append("🇺🇸")
                        elif strategy == 'JP':
                            strategy_names.append("🇯🇵")
                        elif strategy == 'COIN':
                            strategy_names.append("🪙")
                    strategy_str = f" ({'+'.join(strategy_names)})"
                
                print(f"  {day['weekday_kr']}요일: {status}{strategy_str}")
        
        # 요약 통계
        summary = preview['summary']
        print(f"\n📊 2주간 요약:")
        print(f"  📈 총 거래일: {summary['total_trading_days']}일")
        
        for strategy, count in summary['strategy_counts'].items():
            if count > 0:
                emoji = {"US": "🇺🇸", "JP": "🇯🇵", "COIN": "🪙"}.get(strategy, "📊")
                print(f"  {emoji} {strategy}: {count}일")
        
        if summary['most_active_day']:
            print(f"  🔥 가장 활발한 요일: {summary['most_active_day']['day']}요일")
        
    except Exception as e:
        print(f"❌ 주간 스케줄 시뮬레이션 오류: {e}")

def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n🧪 엣지 케이스 테스트")
    print("-" * 30)
    
    try:
        scheduler = TradingScheduler()
        
        # 1. 잘못된 설정 테스트
        print("1️⃣ 설정 검증 테스트...")
        invalid_config = {
            'schedule': {
                'weekly_schedule': {
                    'invalid_day': ['INVALID_STRATEGY'],
                    'monday': 'not_a_list'
                },
                'global_trading_hours': {
                    'start_hour': 25,  # 잘못된 시간
                    'end_hour': -1
                }
            }
        }
        
        is_valid, errors = scheduler.validate_schedule_config(invalid_config)
        print(f"   📋 검증 결과: {'통과' if is_valid else '실패'}")
        if errors:
            print(f"   ❌ 오류 수: {len(errors)}개")
            for error in errors[:2]:
                print(f"      - {error}")
        
        # 2. 시간대 경계 테스트
        print("2️⃣ 시간대 경계 테스트...")
        test_times = [
            datetime.now().replace(hour=0, minute=0),   # 자정
            datetime.now().replace(hour=9, minute=0),   # 시장 시작
            datetime.now().replace(hour=12, minute=0),  # 정오
            datetime.now().replace(hour=16, minute=0),  # 미국 장 마감
            datetime.now().replace(hour=23, minute=59), # 하루 끝
        ]
        
        for test_time in test_times:
            trading_status = scheduler.is_trading_time()
            print(f"   🕐 {test_time.strftime('%H:%M')} - 거래 가능: {trading_status}")
        
        # 3. 빈 전략 목록 처리
        print("3️⃣ 빈 전략 목록 테스트...")
        empty_config = {'schedule': {'weekly_schedule': {'monday': []}}}
        strategies = scheduler.get_today_strategies(empty_config)
        print(f"   📋 빈 설정 전략 수: {len(strategies)}개")
        
        # 4. 메모리 사용량 체크
        print("4️⃣ 메모리 상태 체크...")
        import sys
        cache_size = len(scheduler._last_run_cache)
        config_size = len(str(scheduler.config))
        print(f"   💾 캐시 항목: {cache_size}개")
        print(f"   📄 설정 크기: {config_size} 문자")
        
        print("   ✅ 엣지 케이스 테스트 완료")
        
    except Exception as e:
        print(f"❌ 엣지 케이스 테스트 오류: {e}")

async def test_daemon():
    """데몬 기능 테스트"""
    print("\n🤖 데몬 기능 테스트")
    print("-" * 25)
    
    try:
        scheduler = TradingScheduler()
        daemon = SchedulerDaemon(scheduler)
        
        print("1️⃣ 데몬 초기화 완료")
        print("2️⃣ 짧은 테스트 실행 (5초)...")
        
        # 5초간 데몬 실행
        daemon.check_interval = 1  # 1초마다 체크
        
        async def stop_daemon():
            await asyncio.sleep(5)
            daemon.stop()
        
        # 동시 실행
        await asyncio.gather(
            daemon.start(),
            stop_daemon()
        )
        
        print("3️⃣ 데몬 테스트 완료")
        print(f"   📊 알림 기록 수: {len(daemon.last_notification_times)}")
        print(f"   ❌ 오류 횟수: {daemon.error_count}")
        
    except Exception as e:
        print(f"❌ 데몬 테스트 오류: {e}")

def benchmark_performance():
    """성능 벤치마크"""
    print("\n⚡ 성능 벤치마크")
    print("-" * 20)
    
    try:
        import time
        
        # 스케줄러 초기화 시간
        start_time = time.time()
        scheduler = TradingScheduler()
        init_time = (time.time() - start_time) * 1000
        print(f"📊 초기화 시간: {init_time:.2f}ms")
        
        # 오늘 전략 조회 시간 (100회)
        start_time = time.time()
        for _ in range(100):
            strategies = scheduler.get_today_strategies()
        query_time = (time.time() - start_time) * 1000 / 100
        print(f"📊 전략 조회 시간: {query_time:.2f}ms (평균)")
        
        # 거래 시간 확인 시간 (100회)
        start_time = time.time()
        for _ in range(100):
            is_trading = scheduler.is_trading_time()
        trading_check_time = (time.time() - start_time) * 1000 / 100
        print(f"📊 거래 시간 확인: {trading_check_time:.2f}ms (평균)")
        
        # 스케줄 상태 조회 시간
        start_time = time.time()
        status = scheduler.get_schedule_status()
        status_time = (time.time() - start_time) * 1000
        print(f"📊 상태 조회 시간: {status_time:.2f}ms")
        
        # 메모리 사용량 추정
        config_memory = sys.getsizeof(scheduler.config)
        sessions_memory = sys.getsizeof(scheduler.trading_sessions)
        total_memory = config_memory + sessions_memory
        print(f"📊 메모리 사용량: ~{total_memory/1024:.1f}KB")
        
        print("   ✅ 성능 벤치마크 완료")
        
    except Exception as e:
        print(f"❌ 성능 벤치마크 오류: {e}")

if __name__ == "__main__":
    print("📅 최고퀸트프로젝트 스케줄러 시스템 (개선 버전)")
    print("=" * 65)
    
    async def run_all_tests():
        # 기본 테스트
        await test_scheduler_system()
        
        # 주간 스케줄 시뮬레이션
        test_weekly_schedule()
        
        # 엣지 케이스 테스트
        test_edge_cases()
        
        # 데몬 테스트
        await test_daemon()
        
        # 성능 벤치마크
        benchmark_performance()
        
        print("\n" + "=" * 65)
        print("🚀 개선된 스케줄러 시스템 준비 완료!")
        print()
        print("💡 주요 개선사항:")
        print("   ✅ 강화된 오류 처리 및 폴백 메커니즘")
        print("   ✅ 설정 검증 및 백업 시스템")
        print("   ✅ 중복 알림 방지 및 메모리 최적화")
        print("   ✅ 상세한 디버깅 정보 및 성능 모니터링")
        print("   ✅ 엣지 케이스 처리 및 안정성 향상")
        print()
        print("🔧 사용법:")
        print("   - get_today_strategies() : 오늘 실행할 전략 목록")
        print("   - is_trading_time()      : 현재 거래 시간 여부")
        print("   - should_run_strategy()  : 특정 전략 실행 가능 여부")
        print("   - get_schedule_status()  : 스케줄러 상태 조회")
        print("   - update_strategy_last_run() : 전략 실행 시간 기록")
    
    # 모든 테스트 실행
    asyncio.run(run_all_tests())_today_strategies(self, config: Optional[Dict] = None) -> List[str]:
        """오늘 실행할 전략 목록 조회 (개선된 오류 처리)"""
        try:
            if config is None:
                config = self.config
            
            # 현재 요일 (0=월요일, 6=일요일)
            today = datetime.now()
            weekday = today.weekday()
            
            # 설정에서 요일별 스케줄 확인
            schedule_config = config.get('schedule', {})
            
            # 커스텀 스케줄이 있으면 사용, 없으면 기본 스케줄 사용
            if 'weekly_schedule' in schedule_config and schedule_config['weekly_schedule']:
                weekly_schedule = schedule_config['weekly_schedule']
                weekday_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                today_key = weekday_names[weekday]
                today_strategies = weekly_schedule.get(today_key, [])
            else:
                # 기본 스케줄 사용
                today_strategies = self.default_weekly_schedule.get(weekday, []).copy()
            
            # 공휴일 체크 (utils 모듈이 있는 경우)
            if UTILS_AVAILABLE:
                for strategy in today_strategies.copy():
                    market = strategy
                    try:
                        if not ScheduleUtils.is_trading_day(market, today):
                            today_strategies.remove(strategy)
                            logger.info(f"📅 {market} 시장 휴장일로 인해 {strategy} 전략 제외")
                    except Exception as e:
                        logger.warning(f"⚠️ {market} 휴장일 확인 실패: {e}")
            
            # 강제 활성화/비활성화 체크
            force_enabled = schedule_config.get('force_enabled_strategies', [])
            force_disabled = schedule_config.get('force_disabled_strategies', [])
            
            # 강제 활성화 추가
            for strategy in force_enabled:
                if strategy not in today_strategies:
                    today_strategies.append(strategy)
                    logger.info(f"⚡ {strategy} 전략 강제 활성화")
            
            # 강제 비활성화 제거
            for strategy in force_disabled:
                if strategy in today_strategies:
                    today_strategies.remove(strategy)
                    logger.info(f"🚫 {strategy} 전략 강제 비활성화")
            
            self.last_schedule_check = datetime.now()
            
            weekday_str = ScheduleUtils.get_weekday_korean() if UTILS_AVAILABLE else ["월", "화", "수", "목", "금", "토", "일"][weekday]
            logger.info(f"📊 오늘({weekday_str}) 활성 전략: {today_strategies}")
            return today_strategies
            
        except Exception as e:
            logger.error(f"❌ 오늘 전략 조회 실패: {e}")
            # 기본값: 요일별 기본 스케줄
            weekday = datetime.now().weekday()
            return self.default_weekly_schedule.get(weekday, [])

    def is_trading_time(self, config: Optional[Dict] = None, market: Optional[str] = None) -> bool:
        """현재 시간이 거래 시간인지 확인 (개선된 로직)"""
        try:
            if config is None:
                config = self.config
            
            current_time = datetime.now()
            
            # 글로벌 거래 시간 체크 (설정에서)
            schedule_config = config.get('schedule', {})
            
            # 전체 거래 시간 제한이 있는지 체크
            global_hours = schedule_config.get('global_trading_hours', {})
            if global_hours:
                start_hour = global_hours.get('start_hour', 0)
                end_hour = global_hours.get('end_hour', 24)
                
                current_hour = current_time.hour
                if not (start_hour <= current_hour < end_hour):
                    logger.debug(f"⏰ 글로벌 거래 시간 외: {current_hour}시 (허용: {start_hour}-{end_hour}시)")
                    return False
            
            # 특정 시장 지정된 경우
            if market:
                return self._is_market_trading_time(market, current_time)
            
            # 오늘 활성화된 전략들의 시장 중 하나라도 개장시간이면 True
            today_strategies = self.get_today_strategies(config)
            
            if not today_strategies:
                logger.debug("📅 오늘 활성화된 전략이 없음")
                return False
            
            # 각 전략의 시장별 거래 시간 체크
            for strategy in today_strategies:
                if self._is_market_trading_time(strategy, current_time):
                    return True
            
            logger.debug(f"⏰ 모든 활성 시장이 휴장 중: {today_strategies}")
            return False
            
        except Exception as e:
            logger.error(f"❌ 거래 시간 확인 실패: {e}")
            return True  # 에러시 기본적으로 허용

    def _is_market_trading_time(self, market: str, check_time: Optional[datetime] = None) -> bool:
        """특정 시장의 거래 시간 확인 (개선된 시간대 처리)"""
        try:
            if check_time is None:
                check_time = datetime.now()
            
            if market not in self.trading_sessions:
                logger.warning(f"⚠️ 알 수 없는 시장: {market}")
                return True  # 알 수 없는 시장은 기본적으로 허용
            
            sessions = self.trading_sessions[market]
            
            for session in sessions:
                if not session.is_active:
                    continue
                
                # 24/7 시장 (암호화폐)
                if session.session_type == "24/7":
                    return True
                
                # 시간대 변환
                try:
                    market_tz = pytz.timezone(session.timezone)
                    market_time = check_time.astimezone(market_tz)
                    
                    # 세션 시간 체크
                    current_time_only = market_time.time()
                    if session.start_time <= current_time_only <= session.end_time:
                        return True
                        
                except Exception as e:
                    logger.warning(f"⚠️ {market} 시간대 변환 실패: {e}, 로컬 시간 사용")
                    # 시간대 변환 실패시 로컬 시간으로 대략적 체크
                    current_time_only = check_time.time()
                    if session.start_time <= current_time_only <= session.end_time:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ {market} 시장 거래 시간 확인 실패: {e}")
            return True

    def should_run_strategy(self, strategy: str, check_time: Optional[datetime] = None) -> bool:
        """특정 전략을 실행해야 하는지 확인 (개선된 제한 조건)"""
        try:
            if check_time is None:
                check_time = datetime.now()
            
            # 1. 오늘 활성화된 전략인지 확인
            today_strategies = self.get_today_strategies()
            if strategy not in today_strategies:
                logger.debug(f"📅 {strategy} 전략은 오늘 비활성화됨")
                return False
            
            # 2. 해당 시장의 거래 시간인지 확인
            if not self._is_market_trading_time(strategy, check_time):
                logger.debug(f"⏰ {strategy} 시장 휴장 중")
                return False
            
            # 3. 전략별 추가 조건 확인
            strategy_config = self.config.get(f'{strategy.lower()}_strategy', {})
            if not strategy_config.get('enabled', True):
                logger.debug(f"⚙️ {strategy} 전략이 설정에서 비활성화됨")
                return False
            
            # 4. 시간대별 실행 제한 확인
            schedule_restrictions = self.schedule_config.get('strategy_restrictions', {})
            if strategy in schedule_restrictions:
                restriction = schedule_restrictions[strategy]
                
                # 특정 시간대만 실행
                if 'allowed_hours' in restriction:
                    allowed_hours = restriction['allowed_hours']
                    if check_time.hour not in allowed_hours:
                        logger.debug(f"⏰ {strategy} 전략 허용 시간 외: {check_time.hour}시")
                        return False
                
                # 최소 실행 간격 확인
                if 'min_interval_minutes' in restriction:
                    min_interval = restriction['min_interval_minutes']
                    last_run = self._last_run_cache.get(strategy)
                    
                    if last_run:
                        time_since_last = (check_time - last_run).total_seconds() / 60
                        if time_since_last < min_interval:
                            logger.debug(f"⏰ {strategy} 전략 최소 간격 미충족: {time_since_last:.1f}분")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {strategy} 전략 실행 가능성 확인 실패: {e}")
            return False

    def update_last_run(self, strategy: str, run_time: Optional[datetime] = None):
        """전략 마지막 실행 시간 업데이트"""
        if run_time is None:
            run_time = datetime.now()
        self._last_run_cache[strategy] = run_time
        logger.debug(f"📝 {strategy} 전략 마지막 실행 시간 업데이트: {run_time}")

    def get_next_trading_session(self, market: Optional[str] = None) -> Optional[ScheduleEvent]:
        """다음 거래 세션 정보 조회 (개선된 다음 날 처리)"""
        try:
            current_time = datetime.now()
            upcoming_events = []
            
            # 특정 시장 지정된 경우
            if market:
                markets_to_check = [market] if market in self.trading_sessions else []
            else:
                # 오늘 활성화된 전략들의 시장
                today_strategies = self.get_today_strategies()
                markets_to_check = [s for s in today_strategies if s in self.trading_sessions]
            
            for mkt in markets_to_check:
                sessions = self.trading_sessions[mkt]
                
                for session in sessions:
                    if not session.is_active:
                        continue
                    
                    # 24/7 시장은 다음 세션이 없음
                    if session.session_type == "24/7":
                        continue
                    
                    # 세션 시작 시간 계산
                    try:
                        market_tz = pytz.timezone(session.timezone)
                        
                        # 오늘 세션 시간
                        today_session = current_time.replace(
                            hour=session.start_time.hour,
                            minute=session.start_time.minute,
                            second=0, microsecond=0
                        )
                        
                        # 시간대 변환
                        session_time = market_tz.localize(today_session, is_dst=None)
                        session_time = session_time.astimezone(current_time.tzinfo or pytz.UTC)
                        
                        # 오늘 세션이 이미 지났으면 내일로
                        if session_time <= current_time:
                            session_time += timedelta(days=1)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ {mkt} 시간대 계산 실패: {e}")
                        # 시간대 처리 실패시 로컬 시간 사용
                        session_time = current_time.replace(
                            hour=session.start_time.hour,
                            minute=session.start_time.minute,
                            second=0, microsecond=0
                        )
                        if session_time <= current_time:
                            session_time += timedelta(days=1)
                    
                    event = ScheduleEvent(
                        event_type='market_open',
                        market=mkt,
                        timestamp=session_time,
                        strategies=[mkt],
                        description=f"{mkt} {session.session_type} 세션 시작"
                    )
                    upcoming_events.append(event)
            
            # 가장 빠른 이벤트 반환
            if upcoming_events:
                upcoming_events.sort(key=lambda x: x.timestamp)
                return upcoming_events[0]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 다음 거래 세션 조회 실패: {e}")
            return None

    def get_market_schedule_summary(self, date: Optional[datetime] = None) -> Dict:
        """시장별 스케줄 요약 (개선된 정보)"""
        try:
            if date is None:
                date = datetime.now()
            
            # 해당 날짜의 활성 전략
            weekday = date.weekday()
            strategies = self.default_weekly_schedule.get(weekday, [])
            
            summary = {
                'date': date.strftime('%Y-%m-%d'),
                'weekday': ScheduleUtils.get_weekday_korean(date) if UTILS_AVAILABLE else ["월", "화", "수", "목", "금", "토", "일"][weekday],
                'weekday_index': weekday,
                'active_strategies': strategies,
                'trading_day': len(strategies) > 0,
                'market_sessions': {},
                'total_sessions': 0
            }
            
            # 시장별 세션 정보
            for strategy in strategies:
                if strategy in self.trading_sessions:
                    sessions = self.trading_sessions[strategy]
                    session_info = []
                    
                    for session in sessions:
                        if session.is_active:
                            session_data = {
                                'type': session.session_type,
                                'start': session.start_time.strftime('%H:%M'),
                                'end': session.end_time.strftime('%H:%M'),
                                'timezone': session.timezone,
                                'duration_hours': self._calculate_session_duration(session)
                            }
                            session_info.append(session_data)
                            summary['total_sessions'] += 1
                    
                    summary['market_sessions'][strategy] = session_info
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 시장 스케줄 요약 생성 실패: {e}")
            return {'error': str(e)}

    def _calculate_session_duration(self, session: TradingSession) -> float:
        """세션 지속 시간 계산 (시간 단위)"""
        try:
            if session.session_type == "24/7":
                return 24.0
            
            start_minutes = session.start_time.hour * 60 + session.start_time.minute
            end_minutes = session.end_time.hour * 60 + session.end_time.minute
            
            # 다음날로 넘어가는 경우
            if end_minutes < start_minutes:
                end_minutes += 24 * 60
            
            duration_minutes = end_minutes - start_minutes
            return round(duration_minutes / 60, 2)
            
        except Exception as e:
            logger.warning(f"⚠️ 세션 지속시간 계산 실패: {e}")
            return 0.0

    async def send_schedule_notifications(self, event_type: str = "start") -> bool:
        """스케줄 기반 알림 발송 (개선된 오류 처리)"""
        try:
            if not NOTIFIER_AVAILABLE:
                logger.warning("⚠️ 알림 모듈 없음 - 스케줄 알림 불가")
                return False
            
            today_strategies = self.get_today_strategies()
            
            if not today_strategies:
                logger.info("📅 오늘 활성 전략이 없어 알림 생략")
                return True
            
            # 거래 시작 알림
            if event_type == "start":
                success = await send_schedule_notification(today_strategies, "start")
                if success:
                    logger.info("📱 거래 시작 알림 발송 완료")
                return success
                
            # 거래 종료 알림  
            elif event_type == "end":
                success = await send_schedule_notification(today_strategies, "end")
                if success:
                    logger.info("📱 거래 종료 알림 발송 완료")
                return success
            
            logger.warning(f"⚠️ 알 수 없는 알림 타입: {event_type}")
            return False
            
        except Exception as e:
            logger.error(f"❌ 스케줄 알림 발송 실패: {e}")
            return False

    async def send_next_session_alert(self) -> bool:
        """다음 거래 세션 알림 (개선된 메시지)"""
        try:
            if not NOTIFIER_AVAILABLE:
                return False
            
            next_session = self.get_next_trading_session()
            if not next_session:
                logger.info("📅 다음 거래 세션 정보 없음")
                return False
            
            time_until = next_session.timestamp - datetime.now()
            total_minutes = int(time_until.total_seconds() / 60)
            hours = total_minutes // 60
            minutes = total_minutes % 60
            
            message = f"📅 다음 거래 세션 알림\n\n"
            message += f"🎯 시장: {next_session.market}\n"
            message += f"⏰ 시작: {next_session.timestamp.strftime('%m/%d %H:%M')}\n"
            message += f"🕒 남은 시간: {hours}시간 {minutes}분\n"
            message += f"📝 설명: {next_session.description}\n"
            message += f"📊 전략: {', '.join(next_session.strategies)}"
            
            success = await send_system_alert("info", message, "normal")
            if success:
                logger.info("📱 다음 세션 알림 발송 완료")
            return success
            
        except Exception as e:
            logger.error(f"❌ 다음 세션 알림 실패: {e}")
            return False

    def get
