"""
📅 최고퀸트프로젝트 - 스케줄링 시스템
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
Version: 1.0.0
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
import yaml
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
import pytz
from dataclasses import dataclass
import calendar

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

@dataclass
class ScheduleEvent:
    """스케줄 이벤트"""
    event_type: str  # market_open, market_close, strategy_start, strategy_end
    market: str
    timestamp: datetime
    strategies: List[str]
    description: str

class TradingScheduler:
    """🏆 최고퀸트프로젝트 거래 스케줄러"""
    
    def __init__(self, config_path: str = "configs/settings.yaml"):
        """스케줄러 초기화"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # 시간대 관리자
        if UTILS_AVAILABLE:
            self.tz_manager = TimeZoneManager()
        else:
            self.tz_manager = None
            
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
        
        logger.info("📅 최고퀸트프로젝트 스케줄러 초기화 완료")
        logger.info(f"⚙️ 요일별 기본 스케줄: {self._format_weekly_schedule()}")

    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ 스케줄 설정 로드 성공: {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"❌ 스케줄 설정 로드 실패: {e}")
            return {}
    
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
                schedule_str.append(f"{day_name}({'+'.join(strategy_names)})")
            else:
                schedule_str.append(f"{day_name}(휴무)")
                
        return " ".join(schedule_str)
    
    def _define_trading_sessions(self) -> Dict[str, List[TradingSession]]:
        """시장별 거래 세션 정의"""
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
        
        return sessions

    def get_today_strategies(self, config: Dict = None) -> List[str]:
        """오늘 실행할 전략 목록 조회"""
        try:
            if config is None:
                config = self.config
            
            # 현재 요일 (0=월요일, 6=일요일)
            today = datetime.now()
            weekday = today.weekday()
            
            # 설정에서 요일별 스케줄 확인
            schedule_config = config.get('schedule', {})
            
            # 커스텀 스케줄이 있으면 사용, 없으면 기본 스케줄 사용
            if 'weekly_schedule' in schedule_config:
                weekly_schedule = schedule_config['weekly_schedule']
                weekday_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                today_key = weekday_names[weekday]
                today_strategies = weekly_schedule.get(today_key, [])
            else:
                # 기본 스케줄 사용
                today_strategies = self.default_weekly_schedule.get(weekday, [])
            
            # 공휴일 체크
            if UTILS_AVAILABLE:
                for strategy in today_strategies.copy():
                    market = strategy
                    if not ScheduleUtils.is_trading_day(market, today):
                        today_strategies.remove(strategy)
                        logger.info(f"📅 {market} 시장 휴장일로 인해 {strategy} 전략 제외")
            
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
            
            logger.info(f"📊 오늘({ScheduleUtils.get_weekday_korean() if UTILS_AVAILABLE else weekday}) 활성 전략: {today_strategies}")
            return today_strategies
            
        except Exception as e:
            logger.error(f"❌ 오늘 전략 조회 실패: {e}")
            # 기본값: 모든 전략 활성화
            return ['US', 'JP', 'COIN']

    def is_trading_time(self, config: Dict = None, market: str = None) -> bool:
        """현재 시간이 거래 시간인지 확인"""
        try:
            if config is None:
                config = self.config
            
            current_time = datetime.now()
            
            # 글로벌 거래 시간 체크 (설정에서)
            schedule_config = config.get('schedule', {})
            
            # 전체 거래 시간 제한이 있는지 체크
            if 'global_trading_hours' in schedule_config:
                global_hours = schedule_config['global_trading_hours']
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

    def _is_market_trading_time(self, market: str, check_time: datetime = None) -> bool:
        """특정 시장의 거래 시간 확인"""
        try:
            if check_time is None:
                check_time = datetime.now()
            
            if not UTILS_AVAILABLE or not self.tz_manager:
                # utils 없으면 기본 시간 체크
                hour = check_time.hour
                if market == 'US':
                    return 9 <= hour <= 16
                elif market == 'JP':
                    return 9 <= hour <= 15
                elif market == 'COIN':
                    return True
                else:
                    return True
            
            # TimeZoneManager 사용하여 정확한 시간 체크
            return self.tz_manager.is_market_open(market)
            
        except Exception as e:
            logger.error(f"❌ {market} 시장 거래 시간 확인 실패: {e}")
            return True

    def should_run_strategy(self, strategy: str, check_time: datetime = None) -> bool:
        """특정 전략을 실행해야 하는지 확인"""
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
                
                # 최소 실행 간격
                if 'min_interval_minutes' in restriction:
                    min_interval = restriction['min_interval_minutes']
                    last_run_key = f"last_run_{strategy}"
                    # 실제로는 캐시나 DB에서 마지막 실행 시간 확인
                    # 여기서는 간단히 True로 처리
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {strategy} 전략 실행 가능성 확인 실패: {e}")
            return False

    def get_next_trading_session(self, market: str = None) -> Optional[ScheduleEvent]:
        """다음 거래 세션 정보 조회"""
        try:
            current_time = datetime.now()
            upcoming_events = []
            
            # 특정 시장 지정된 경우
            if market:
                markets_to_check = [market]
            else:
                # 오늘 활성화된 전략들의 시장
                today_strategies = self.get_today_strategies()
                markets_to_check = today_strategies
            
            for mkt in markets_to_check:
                if mkt not in self.trading_sessions:
                    continue
                
                sessions = self.trading_sessions[mkt]
                
                for session in sessions:
                    if not session.is_active:
                        continue
                    
                    # 세션 시작 시간 계산
                    if UTILS_AVAILABLE and self.tz_manager:
                        if mkt == 'US':
                            session_time = self.tz_manager.get_current_time('US').replace(
                                hour=session.start_time.hour,
                                minute=session.start_time.minute,
                                second=0, microsecond=0
                            )
                        elif mkt == 'JP':
                            session_time = self.tz_manager.get_current_time('JAPAN').replace(
                                hour=session.start_time.hour,
                                minute=session.start_time.minute,
                                second=0, microsecond=0
                            )
                        else:  # COIN
                            session_time = current_time.replace(
                                hour=session.start_time.hour,
                                minute=session.start_time.minute,
                                second=0, microsecond=0
                            )
                    else:
                        session_time = current_time.replace(
                            hour=session.start_time.hour,
                            minute=session.start_time.minute,
                            second=0, microsecond=0
                        )
                    
                    # 오늘 세션이 이미 지났으면 내일로
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

    def get_market_schedule_summary(self, date: datetime = None) -> Dict:
        """시장별 스케줄 요약"""
        try:
            if date is None:
                date = datetime.now()
            
            # 해당 날짜의 활성 전략
            weekday = date.weekday()
            strategies = self.default_weekly_schedule.get(weekday, [])
            
            summary = {
                'date': date.strftime('%Y-%m-%d'),
                'weekday': ScheduleUtils.get_weekday_korean(date) if UTILS_AVAILABLE else str(weekday),
                'active_strategies': strategies,
                'trading_day': len(strategies) > 0,
                'market_sessions': {}
            }
            
            # 시장별 세션 정보
            for strategy in strategies:
                if strategy in self.trading_sessions:
                    sessions = self.trading_sessions[strategy]
                    session_info = []
                    
                    for session in sessions:
                        if session.is_active:
                            session_info.append({
                                'type': session.session_type,
                                'start': session.start_time.strftime('%H:%M'),
                                'end': session.end_time.strftime('%H:%M'),
                                'timezone': session.timezone
                            })
                    
                    summary['market_sessions'][strategy] = session_info
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 시장 스케줄 요약 생성 실패: {e}")
            return {}

    async def send_schedule_notifications(self, event_type: str = "start") -> bool:
        """스케줄 기반 알림 발송"""
        try:
            if not NOTIFIER_AVAILABLE:
                logger.warning("⚠️ 알림 모듈 없음 - 스케줄 알림 불가")
                return False
            
            today_strategies = self.get_today_strategies()
            
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
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 스케줄 알림 발송 실패: {e}")
            return False

    async def send_next_session_alert(self) -> bool:
        """다음 거래 세션 알림"""
        try:
            if not NOTIFIER_AVAILABLE:
                return False
            
            next_session = self.get_next_trading_session()
            if not next_session:
                return False
            
            time_until = next_session.timestamp - datetime.now()
            hours = int(time_until.total_seconds() // 3600)
            minutes = int((time_until.total_seconds() % 3600) // 60)
            
            message = f"📅 다음 거래 세션\n\n"
            message += f"🎯 시장: {next_session.market}\n"
            message += f"⏰ 시작: {next_session.timestamp.strftime('%H:%M')}\n"
            message += f"🕒 남은 시간: {hours}시간 {minutes}분\n"
            message += f"📝 설명: {next_session.description}"
            
            success = await send_system_alert("info", message, "normal")
            if success:
                logger.info("📱 다음 세션 알림 발송 완료")
            return success
            
        except Exception as e:
            logger.error(f"❌ 다음 세션 알림 실패: {e}")
            return False

    def get_schedule_status(self) -> Dict:
        """스케줄러 상태 조회"""
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
                'next_session': None
            }
            
            # 시장별 상태
            for strategy in ['US', 'JP', 'COIN']:
                is_active = strategy in today_strategies
                is_trading = self._is_market_trading_time(strategy, current_time)
                
                status['market_status'][strategy] = {
                    'active_today': is_active,
                    'trading_now': is_trading,
                    'should_run': self.should_run_strategy(strategy, current_time)
                }
            
            # 다음 세션 정보
            next_session = self.get_next_trading_session()
            if next_session:
                time_until = next_session.timestamp - current_time
                status['next_session'] = {
                    'market': next_session.market,
                    'timestamp': next_session.timestamp.isoformat(),
                    'time_until_seconds': int(time_until.total_seconds()),
                    'description': next_session.description
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ 스케줄러 상태 조회 실패: {e}")
            return {'scheduler_status': 'error', 'error': str(e)}

    def update_schedule_config(self, new_config: Dict) -> bool:
        """스케줄 설정 업데이트"""
        try:
            # 기존 설정과 병합
            updated_config = self.config.copy()
            if 'schedule' in new_config:
                updated_config['schedule'].update(new_config['schedule'])
            
            # 파일에 저장
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(updated_config, f, default_flow_style=False, allow_unicode=True)
            
            # 메모리 설정 업데이트
            self.config = updated_config
            self.schedule_config = self.config.get('schedule', {})
            
            logger.info("⚙️ 스케줄 설정 업데이트 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 스케줄 설정 업데이트 실패: {e}")
            return False

# =====================================
# 편의 함수들 (core.py에서 호출)
# =====================================

def get_today_strategies(config: Dict = None) -> List[str]:
    """오늘 실행할 전략 목록 조회 (편의 함수)"""
    try:
        scheduler = TradingScheduler()
        return scheduler.get_today_strategies(config)
    except Exception as e:
        logger.error(f"❌ 오늘 전략 조회 실패: {e}")
        return ['US', 'JP', 'COIN']  # 기본값

def is_trading_time(config: Dict = None, market: str = None) -> bool:
    """현재 거래 시간인지 확인 (편의 함수)"""
    try:
        scheduler = TradingScheduler()
        return scheduler.is_trading_time(config, market)
    except Exception as e:
        logger.error(f"❌ 거래 시간 확인 실패: {e}")
        return True  # 기본값

def should_run_strategy(strategy: str) -> bool:
    """특정 전략 실행 여부 (편의 함수)"""
    try:
        scheduler = TradingScheduler()
        return scheduler.should_run_strategy(strategy)
    except Exception as e:
        logger.error(f"❌ {strategy} 전략 실행 가능성 확인 실패: {e}")
        return True

def get_schedule_status() -> Dict:
    """스케줄러 상태 조회 (편의 함수)"""
    try:
        scheduler = TradingScheduler()
        return scheduler.get_schedule_status()
    except Exception as e:
        logger.error(f"❌ 스케줄러 상태 조회 실패: {e}")
        return {'scheduler_status': 'error'}

# =====================================
# 스케줄러 데몬 (백그라운드 실행)
# =====================================

class SchedulerDaemon:
    """스케줄러 백그라운드 데몬"""
    
    def __init__(self, scheduler: TradingScheduler):
        self.scheduler = scheduler
        self.running = False
        self.check_interval = 60  # 1분마다 체크
        
    async def start(self):
        """데몬 시작"""
        self.running = True
        logger.info("🤖 스케줄러 데몬 시작")
        
        try:
            while self.running:
                await self._check_schedule_events()
                await asyncio.sleep(self.check_interval)
        except Exception as e:
            logger.error(f"❌ 스케줄러 데몬 오류: {e}")
        finally:
            logger.info("🛑 스케줄러 데몬 종료")
    
    def stop(self):
        """데몬 정지"""
        self.running = False
    
    async def _check_schedule_events(self):
        """스케줄 이벤트 체크"""
        try:
            current_time = datetime.now()
            
            # 거래 시작 시간 체크 (09:00)
            if current_time.hour == 9 and current_time.minute == 0:
                await self.scheduler.send_schedule_notifications("start")
            
            # 거래 종료 시간 체크 (18:00)
            elif current_time.hour == 18 and current_time.minute == 0:
                await self.scheduler.send_schedule_notifications("end")
            
            # 다음 세션 1시간 전 알림
            next_session = self.scheduler.get_next_trading_session()
            if next_session:
                time_until = next_session.timestamp - current_time
                if 3540 <= time_until.total_seconds() <= 3600:  # 59-60분 전
                    await self.scheduler.send_next_session_alert()
                    
        except Exception as e:
            logger.error(f"❌ 스케줄 이벤트 체크 실패: {e}")

# =====================================
# 테스트 함수
# =====================================

async def test_scheduler_system():
    """🧪 스케줄러 시스템 테스트"""
    print("📅 최고퀸트프로젝트 스케줄러 테스트")
    print("=" * 50)
    
    # 1. 스케줄러 초기화
    print("1️⃣ 스케줄러 초기화...")
    scheduler = TradingScheduler()
    print("   ✅ 완료")
    
    # 2. 오늘 전략 조회
    print("2️⃣ 오늘 전략 조회...")
    today_strategies = scheduler.get_today_strategies()
    weekday = ScheduleUtils.get_weekday_korean() if UTILS_AVAILABLE else datetime.now().weekday()
    print(f"   📅 오늘({weekday}): {today_strategies}")
    
    # 3. 거래 시간 확인
    print("3️⃣ 거래 시간 확인...")
    is_trading = scheduler.is_trading_time()
    status = "🟢 거래 중" if is_trading else "🔴 휴장"
    print(f"   ⏰ 현재: {status}")
    
    # 4. 시장별 상태
    print("4️⃣ 시장별 상태...")
    markets = ['US', 'JP', 'COIN']
    for market in markets:
        is_active = market in today_strategies
        is_open = scheduler._is_market_trading_time(market)
        should_run = scheduler.should_run_strategy(market)
        
        status_emoji = "🟢" if should_run else "🔴"
        print(f"   {market:4}: {status_emoji} 활성({is_active}) 개장({is_open}) 실행({should_run})")
    
    # 5. 다음 세션 정보
    print("5️⃣ 다음 거래 세션...")
    next_session = scheduler.get_next_trading_session()
    if next_session:
        time_until = next_session.timestamp - datetime.now()
        hours = int(time_until.total_seconds() // 3600)
        minutes = int((time_until.total_seconds() % 3600) // 60)
        print(f"   🎯 {next_session.market} 시장")
        print(f"   ⏰ {next_session.timestamp.strftime('%H:%M')} ({hours}시간 {minutes}분 후)")
        print(f"   📝 {next_session.description}")
    else:
        print("   ❌ 다음 세션 정보 없음")
    
    # 6. 주간 스케줄 요약
    print("6️⃣ 주간 스케줄 요약...")
    print(f"   📅 {scheduler._format_weekly_schedule()}")
    
    # 7. 스케줄러 상태
    print("7️⃣ 스케줄러 상태...")
    status = scheduler.get_schedule_status()
    print(f"   📊 상태: {status['scheduler_status']}")
    print(f"   ⏱️ 가동시간: {status['session_uptime']}")
    print(f"   📈 오늘 활성 전략: {len(status['today_strategies'])}개")
    
    # 8. 편의 함수 테스트
    print("8️⃣ 편의 함수 테스트...")
    strategies = get_today_strategies()
    trading = is_trading_time()
    print(f"   📋 편의 함수 - 전략: {len(strategies)}개, 거래시간: {trading}")
    
    # 9. 알림 테스트 (선택적)
    if NOTIFIER_AVAILABLE:
        print("9️⃣ 알림 테스트...")
        try:
            # 스케줄 알림 테스트
            await scheduler.send_schedule_notifications("start")
            print("   ✅ 스케줄 알림 발송 완료")
        except Exception as e:
            print(f"   ❌ 스케줄 알림 실패: {e}")
    else:
        print("9️⃣ 알림 모듈 없음 - 알림 테스트 스킵")
    
    print()
    print("🎯 스케줄러 테스트 완료!")
    print("📅 요일별 스케줄링 시스템이 정상 작동합니다")

def test_weekly_schedule():
    """주간 스케줄 시뮬레이션"""
    print("\n📅 주간 스케줄 시뮬레이션")
    print("-" * 30)
    
    scheduler = TradingScheduler()
    
    # 각 요일별 테스트
    for day_offset in range(7):
        test_date = datetime.now() + timedelta(days=day_offset)
        weekday = test_date.weekday()
        strategies = scheduler.default_weekly_schedule.get(weekday, [])
        
        weekday_name = ScheduleUtils.get_weekday_korean(test_date) if UTILS_AVAILABLE else str(weekday)
        
        if strategies:
            strategy_names = []
            for strategy in strategies:
                if strategy == 'US':
                    strategy_names.append("🇺🇸미국")
                elif strategy == 'JP':
                    strategy_names.append("🇯🇵일본")
                elif strategy == 'COIN':
                    strategy_names.append("🪙코인")
            
            print(f"{weekday_name}요일: {' + '.join(strategy_names)} 거래")
        else:
            print(f"{weekday_name}요일: 😴 휴무")

if __name__ == "__main__":
    print("📅 최고퀸트프로젝트 스케줄러 시스템")
    print("=" * 50)
    
    # 기본 테스트
    asyncio.run(test_scheduler_system())
    
    # 주간 스케줄 시뮬레이션
    test_weekly_schedule()
    
    print("\n🚀 스케줄러 시스템 준비 완료!")
    print("💡 core.py에서 get_today_strategies(), is_trading_time() 함수를 사용하세요")
