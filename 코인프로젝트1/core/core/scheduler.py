"""
퀀트 트레이딩 스케줄러
- 주기적 데이터 수집
- 시그널 생성 스케줄링
- 백테스팅 스케줄링
- 리스크 모니터링
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

from logger import get_logger
from api_wrapper import QuantAPIWrapper

logger = get_logger(__name__)

class ScheduleType(Enum):
    """스케줄 타입"""
    MARKET_DATA = "market_data"
    SIGNAL_GENERATION = "signal_generation"
    RISK_MONITORING = "risk_monitoring"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    SYSTEM_HEALTH = "system_health"

@dataclass
class ScheduledTask:
    """스케줄링된 작업"""
    name: str
    schedule_type: ScheduleType
    function: Callable
    interval: int  # seconds
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    error_count: int = 0
    max_errors: int = 5

class InstitutionalTradingScheduler:
    """기관급 트레이딩 스케줄러"""
    
    def __init__(self, api_wrapper: QuantAPIWrapper, config: Dict = None):
        self.api = api_wrapper
        self.config = config or {}
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_thread = None
        
        logger.info("📅 트레이딩 스케줄러 초기화 완료")
        
    def add_task(self, task: ScheduledTask):
        """작업 추가"""
        self.tasks[task.name] = task
        task.next_run = datetime.now() + timedelta(seconds=task.interval)
        logger.info(f"✅ 작업 추가: {task.name} (간격: {task.interval}초)")
        
    def remove_task(self, task_name: str):
        """작업 제거"""
        if task_name in self.tasks:
            del self.tasks[task_name]
            logger.info(f"🗑️ 작업 제거: {task_name}")
    
    def enable_task(self, task_name: str):
        """작업 활성화"""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = True
            logger.info(f"▶️ 작업 활성화: {task_name}")
    
    def disable_task(self, task_name: str):
        """작업 비활성화"""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = False
            logger.info(f"⏸️ 작업 비활성화: {task_name}")
    
    async def run_task(self, task: ScheduledTask):
        """작업 실행"""
        if not task.enabled:
            return
            
        try:
            logger.debug(f"🔄 작업 실행 시작: {task.name}")
            
            # 비동기 함수인지 확인
            if asyncio.iscoroutinefunction(task.function):
                await task.function()
            else:
                task.function()
                
            task.last_run = datetime.now()
            task.next_run = task.last_run + timedelta(seconds=task.interval)
            task.error_count = 0
            
            logger.debug(f"✅ 작업 실행 완료: {task.name}")
            
        except Exception as e:
            task.error_count += 1
            logger.error(f"❌ 작업 실행 실패: {task.name} - {e}")
            
            if task.error_count >= task.max_errors:
                logger.critical(f"🚫 작업 비활성화 (에러 초과): {task.name}")
                task.enabled = False
    
    async def market_data_collector(self):
        """시장 데이터 수집 작업"""
        symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        
        for symbol in symbols:
            try:
                market_data = await self.api.fetch_comprehensive_market_data(symbol)
                logger.debug(f"📊 {symbol} 데이터 수집: ${market_data.price:,.2f}")
            except Exception as e:
                logger.error(f"❌ {symbol} 데이터 수집 실패: {e}")
    
    async def signal_generator(self):
        """트레이딩 시그널 생성 작업"""
        symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        
        for symbol in symbols:
            try:
                market_data = await self.api.fetch_comprehensive_market_data(symbol)
                signals = self.api.generate_trading_signals(market_data)
                
                high_confidence_signals = [s for s in signals if s.confidence > 0.7]
                if high_confidence_signals:
                    logger.info(f"🔥 {symbol} 고신뢰도 시그널 {len(high_confidence_signals)}개 생성")
                    
            except Exception as e:
                logger.error(f"❌ {symbol} 시그널 생성 실패: {e}")
    
    def risk_monitor(self):
        """리스크 모니터링 작업"""
        try:
            # 포트폴리오 리스크 체크
            positions = self.config.get('positions', {})
            if positions:
                logger.info("🛡️ 포트폴리오 리스크 모니터링 실행")
                # 여기에 리스크 계산 로직 추가
            
        except Exception as e:
            logger.error(f"❌ 리스크 모니터링 실패: {e}")
    
    def portfolio_rebalancer(self):
        """포트폴리오 리밸런싱 작업"""
        try:
            logger.info("⚖️ 포트폴리오 리밸런싱 실행")
            # 여기에 리밸런싱 로직 추가
            
        except Exception as e:
            logger.error(f"❌ 포트폴리오 리밸런싱 실패: {e}")
    
    def system_health_check(self):
        """시스템 상태 체크 작업"""
        try:
            status = self.api.get_status()
            
            # 연결 상태 체크
            if not status['exchanges']['binance']:
                logger.warning("⚠️ Binance 연결 상태 이상")
            
            if not status['database']:
                logger.warning("⚠️ 데이터베이스 연결 상태 이상")
            
            # 활성 작업 수 체크
            active_tasks = sum(1 for task in self.tasks.values() if task.enabled)
            logger.info(f"💓 시스템 상태 체크 완료 - 활성 작업: {active_tasks}개")
            
        except Exception as e:
            logger.error(f"❌ 시스템 상태 체크 실패: {e}")
    
    def setup_default_tasks(self):
        """기본 작업들 설정"""
        
        # 시장 데이터 수집 (1분마다)
        self.add_task(ScheduledTask(
            name="market_data_collection",
            schedule_type=ScheduleType.MARKET_DATA,
            function=self.market_data_collector,
            interval=60
        ))
        
        # 시그널 생성 (5분마다)
        self.add_task(ScheduledTask(
            name="signal_generation",
            schedule_type=ScheduleType.SIGNAL_GENERATION,
            function=self.signal_generator,
            interval=300
        ))
        
        # 리스크 모니터링 (10분마다)
        self.add_task(ScheduledTask(
            name="risk_monitoring",
            schedule_type=ScheduleType.RISK_MONITORING,
            function=self.risk_monitor,
            interval=600
        ))
        
        # 포트폴리오 리밸런싱 (1시간마다)
        self.add_task(ScheduledTask(
            name="portfolio_rebalance",
            schedule_type=ScheduleType.PORTFOLIO_REBALANCE,
            function=self.portfolio_rebalancer,
            interval=3600
        ))
        
        # 시스템 상태 체크 (30초마다)
        self.add_task(ScheduledTask(
            name="system_health",
            schedule_type=ScheduleType.SYSTEM_HEALTH,
            function=self.system_health_check,
            interval=30
        ))
        
        logger.info(f"🔧 기본 작업 {len(self.tasks)}개 설정 완료")
    
    async def _scheduler_loop(self):
        """스케줄러 메인 루프"""
        logger.info("🚀 스케줄러 루프 시작")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # 실행할 작업들 찾기
                tasks_to_run = []
                for task in self.tasks.values():
                    if (task.enabled and 
                        task.next_run and 
                        current_time >= task.next_run):
                        tasks_to_run.append(task)
                
                # 작업들 병렬 실행
                if tasks_to_run:
                    await asyncio.gather(
                        *[self.run_task(task) for task in tasks_to_run],
                        return_exceptions=True
                    )
                
                # 1초 대기
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ 스케줄러 루프 에러: {e}")
                await asyncio.sleep(5)
    
    async def start(self):
        """스케줄러 시작"""
        if self.running:
            logger.warning("⚠️ 스케줄러가 이미 실행 중입니다")
            return
        
        self.running = True
        
        # 기본 작업 설정
        if not self.tasks:
            self.setup_default_tasks()
        
        logger.info("🚀 퀀트 트레이딩 스케줄러 시작")
        
        try:
            await self._scheduler_loop()
        except KeyboardInterrupt:
            logger.info("⌨️ 키보드 인터럽트로 스케줄러 중지")
        finally:
            await self.stop()
    
    async def stop(self):
        """스케줄러 중지"""
        self.running = False
        logger.info("🛑 스케줄러 중지")
    
    def get_status(self) -> Dict:
        """스케줄러 상태 반환"""
        return {
            'running': self.running,
            'total_tasks': len(self.tasks),
            'active_tasks': sum(1 for task in self.tasks.values() if task.enabled),
            'tasks': {
                name: {
                    'enabled': task.enabled,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'next_run': task.next_run.isoformat() if task.next_run else None,
                    'error_count': task.error_count,
                    'interval': task.interval
                }
                for name, task in self.tasks.items()
            }
        }

# 편의 함수
async def create_and_start_scheduler(config: Dict = None) -> InstitutionalTradingScheduler:
    """스케줄러 생성 및 시작"""
    api = QuantAPIWrapper(config)
    scheduler = InstitutionalTradingScheduler(api, config)
    
    # 별도 태스크로 실행
    asyncio.create_task(scheduler.start())
    
    return scheduler

# 사용 예제
async def main():
    config = {
        'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        'positions': {'BTC/USDT': 0.1, 'ETH/USDT': 1.0},
        'sandbox': True
    }
    
    # API 래퍼 생성
    api = QuantAPIWrapper(config)
    
    # 스케줄러 생성
    scheduler = InstitutionalTradingScheduler(api, config)
    
    try:
        # 스케줄러 시작 (무한 루프)
        await scheduler.start()
    except KeyboardInterrupt:
        logger.info("프로그램 종료")

if __name__ == "__main__":
    asyncio.run(main())
