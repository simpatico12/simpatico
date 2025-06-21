import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum
from logger import get_logger
from api_wrapper import QuantAPIWrapper

logger = get_logger(__name__)

class ScheduleType(Enum):
    MARKET_DATA = "market_data"
    SIGNAL_GENERATION = "signal_generation"
    RISK_MONITORING = "risk_monitoring"
    EXECUTION = "execution"

@dataclass
class ScheduledTask:
    name: str
    schedule_type: ScheduleType
    function: Callable
    interval: int  # seconds
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None

class TradingScheduler:
    def __init__(self, api: QuantAPIWrapper, config: Dict = None):
        self.api = api
        self.config = config or {}
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        logger.info("✅ 스케줄러 초기화")

    def add_task(self, task: ScheduledTask):
        self.tasks[task.name] = task
        task.next_run = datetime.now() + timedelta(seconds=task.interval)
        logger.info(f"✅ 작업 등록: {task.name} (주기: {task.interval}s)")

    async def run_task(self, task: ScheduledTask):
        if not task.enabled:
            return
        try:
            logger.debug(f"🔄 작업 실행: {task.name}")
            await task.function()
            task.last_run = datetime.now()
            task.next_run = task.last_run + timedelta(seconds=task.interval)
        except Exception as e:
            logger.error(f"❌ {task.name} 실패: {e}")

    async def start(self):
        logger.info("🚀 스케줄러 시작")
        self.running = True
        while self.running:
            now = datetime.now()
            tasks_to_run = [task for task in self.tasks.values() if task.enabled and now >= task.next_run]
            if tasks_to_run:
                await asyncio.gather(*(self.run_task(task) for task in tasks_to_run))
            await asyncio.sleep(1)

    def stop(self):
        self.running = False
        logger.info("🛑 스케줄러 중지")

async def create_scheduler(config: Dict) -> TradingScheduler:
    api = QuantAPIWrapper(config)
    scheduler = TradingScheduler(api, config)

    scheduler.add_task(ScheduledTask(
        name="market_data",
        schedule_type=ScheduleType.MARKET_DATA,
        function=api.collect_market_data,
        interval=60
    ))

    scheduler.add_task(ScheduledTask(
        name="signal_generation",
        schedule_type=ScheduleType.SIGNAL_GENERATION,
        function=api.generate_signals,
        interval=300
    ))

    scheduler.add_task(ScheduledTask(
        name="risk_monitoring",
        schedule_type=ScheduleType.RISK_MONITORING,
        function=api.risk_monitoring,
        interval=600
    ))

    scheduler.add_task(ScheduledTask(
        name="execution",
        schedule_type=ScheduleType.EXECUTION,
        function=api.execute_trades,
        interval=120
    ))

    asyncio.create_task(scheduler.start())
    return scheduler
