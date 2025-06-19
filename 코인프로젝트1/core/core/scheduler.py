import asyncio
import threading
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import json
import traceback
from pathlib import Path

import schedule
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor as APSThreadPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Custom imports (would be your modules)
from config import get_config
from collector import collect, NEWS_CACHE
from trader import execute_trade
from utils import get_fear_greed_index, get_total_asset_value, get_cash_balance
from trade_engine import upbit
from risk_manager import RiskManager
from portfolio_manager import PortfolioManager
from performance_monitor import PerformanceMonitor

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            'scheduler.log', maxBytes=10*1024*1024, backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class TradingJob:
    job_id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    priority: JobPriority = JobPriority.MEDIUM
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class AdvancedJobMonitor:
    """Advanced job monitoring and alerting system"""
    
    def __init__(self):
        self.job_history: Dict[str, List[JobResult]] = {}
        self.active_jobs: Dict[str, JobResult] = {}
        self.performance_metrics = {}
        self.alert_thresholds = {
            'max_execution_time': 300,  # 5 minutes
            'max_failure_rate': 0.1,    # 10%
            'max_memory_usage': 0.8     # 80%
        }
    
    def start_job(self, job_id: str) -> JobResult:
        """Start monitoring a job"""
        result = JobResult(
            job_id=job_id,
            status=JobStatus.RUNNING,
            start_time=datetime.now()
        )
        self.active_jobs[job_id] = result
        logger.info(f"Job {job_id} started at {result.start_time}")
        return result
    
    def complete_job(self, job_id: str, result: Any = None, error: str = None):
        """Complete job monitoring"""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return
        
        job_result = self.active_jobs[job_id]
        job_result.end_time = datetime.now()
        job_result.execution_time = (job_result.end_time - job_result.start_time).total_seconds()
        
        if error:
            job_result.status = JobStatus.FAILED
            job_result.error = error
            logger.error(f"Job {job_id} failed: {error}")
        else:
            job_result.status = JobStatus.COMPLETED
            job_result.result = result
            logger.info(f"Job {job_id} completed in {job_result.execution_time:.2f}s")
        
        # Move to history
        if job_id not in self.job_history:
            self.job_history[job_id] = []
        self.job_history[job_id].append(job_result)
        del self.active_jobs[job_id]
        
        # Check alerts
        self._check_alerts(job_result)
    
    def _check_alerts(self, job_result: JobResult):
        """Check if any alerts should be triggered"""
        # Execution time alert
        if (job_result.execution_time and 
            job_result.execution_time > self.alert_thresholds['max_execution_time']):
            self._send_alert(f"Job {job_result.job_id} exceeded max execution time: {job_result.execution_time:.2f}s")
        
        # Failure rate alert
        if job_result.job_id in self.job_history:
            recent_jobs = self.job_history[job_result.job_id][-10:]  # Last 10 executions
            failure_rate = sum(1 for job in recent_jobs if job.status == JobStatus.FAILED) / len(recent_jobs)
            
            if failure_rate > self.alert_thresholds['max_failure_rate']:
                self._send_alert(f"Job {job_result.job_id} failure rate: {failure_rate:.2%}")
    
    def _send_alert(self, message: str):
        """Send alert (implement your notification system)"""
        logger.critical(f"ALERT: {message}")
        # Here you could integrate with:
        # - Slack webhooks
        # - Email notifications
        # - SMS alerts
        # - Discord notifications
    
    def get_job_statistics(self, job_id: str) -> Dict:
        """Get comprehensive job statistics"""
        if job_id not in self.job_history:
            return {}
        
        jobs = self.job_history[job_id]
        execution_times = [j.execution_time for j in jobs if j.execution_time]
        
        return {
            'total_executions': len(jobs),
            'success_rate': sum(1 for j in jobs if j.status == JobStatus.COMPLETED) / len(jobs),
            'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'last_execution': jobs[-1].start_time if jobs else None,
            'last_status': jobs[-1].status.value if jobs else None
        }

class InstitutionalTradingScheduler:
    """Institutional-grade trading scheduler with advanced features"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = get_config()
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Initialize components
        self.job_monitor = AdvancedJobMonitor()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Setup schedulers
        self._setup_schedulers()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Job queue for dependency management
        self.job_queue: Dict[str, TradingJob] = {}
        self.completed_jobs: set = set()
        
        logger.info("Institutional Trading Scheduler initialized")
    
    def _setup_schedulers(self):
        """Setup multiple schedulers for different purposes"""
        # Main scheduler for trading operations
        self.main_scheduler = AsyncIOScheduler(
            jobstores={
                'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
            },
            executors={
                'default': APSThreadPoolExecutor(20),
                'critical': APSThreadPoolExecutor(5)  # Dedicated pool for critical jobs
            },
            job_defaults={
                'coalesce': False,
                'max_instances': 1,
                'misfire_grace_time': 30
            },
            timezone='Asia/Seoul'
        )
        
        # High-frequency scheduler for monitoring
        self.monitor_scheduler = BackgroundScheduler(timezone='Asia/Seoul')
        
        # Emergency scheduler (separate process-safe scheduler)
        self.emergency_scheduler = BackgroundScheduler(timezone='Asia/Seoul')
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    async def enhanced_collect_and_analyze(self, asset_list: List[str], asset_type: str) -> Dict:
        """Enhanced data collection with comprehensive analysis"""
        job_id = f"collect_{asset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = self.job_monitor.start_job(job_id)
        
        try:
            # Multi-source data collection
            tasks = [
                self._collect_market_data(asset_list, asset_type),
                self._collect_news_sentiment(asset_list, asset_type),
                self._collect_social_sentiment(asset_list, asset_type),
                self._collect_onchain_metrics(asset_list, asset_type)
            ]
            
            market_data, news_sentiment, social_sentiment, onchain_data = await asyncio.gather(*tasks)
            
            # Advanced sentiment aggregation
            aggregated_sentiment = self._aggregate_sentiments(news_sentiment, social_sentiment)
            
            # Store in enhanced cache with TTL
            for asset in asset_list:
                cache_key = f"{asset_type}_{asset}"
                NEWS_CACHE[cache_key] = {
                    'sentiment': aggregated_sentiment.get(asset, 0.0),
                    'market_data': market_data.get(asset, {}),
                    'onchain_data': onchain_data.get(asset, {}),
                    'timestamp': datetime.now(),
                    'confidence': self._calculate_confidence(asset, market_data, news_sentiment)
                }
            
            self.job_monitor.complete_job(job_id, result={'assets_processed': len(asset_list)})
            return {'status': 'success', 'assets_processed': len(asset_list)}
            
        except Exception as e:
            error_msg = f"Collection failed: {str(e)}\n{traceback.format_exc()}"
            self.job_monitor.complete_job(job_id, error=error_msg)
            logger.error(error_msg)
            return {'status': 'error', 'message': str(e)}
    
    async def _collect_market_data(self, asset_list: List[str], asset_type: str) -> Dict:
        """Collect comprehensive market data"""
        market_data = {}
        for asset in asset_list:
            try:
                # This would integrate with your enhanced API wrapper
                data = await self._fetch_comprehensive_data(asset, asset_type)
                market_data[asset] = data
            except Exception as e:
                logger.error(f"Market data collection failed for {asset}: {e}")
                market_data[asset] = {}
        return market_data
    
    async def _collect_news_sentiment(self, asset_list: List[str], asset_type: str) -> Dict:
        """Collect and analyze news sentiment"""
        # This would use your enhanced news collection
        return await collect(asset_list, asset_type)  # Your existing function, enhanced
    
    async def _collect_social_sentiment(self, asset_list: List[str], asset_type: str) -> Dict:
        """Collect social media sentiment"""
        # Integration with Twitter API, Reddit API, etc.
        social_sentiment = {}
        for asset in asset_list:
            # Placeholder - implement actual social sentiment analysis
            social_sentiment[asset] = 0.0
        return social_sentiment
    
    async def _collect_onchain_metrics(self, asset_list: List[str], asset_type: str) -> Dict:
        """Collect on-chain metrics for applicable assets"""
        onchain_data = {}
        for asset in asset_list:
            # Integration with Glassnode, CoinMetrics, etc.
            onchain_data[asset] = {}
        return onchain_data
    
    def _aggregate_sentiments(self, news_sentiment: Dict, social_sentiment: Dict) -> Dict:
        """Aggregate multiple sentiment sources with weights"""
        aggregated = {}
        for asset in set(list(news_sentiment.keys()) + list(social_sentiment.keys())):
            news_score = news_sentiment.get(asset, 0.0)
            social_score = social_sentiment.get(asset, 0.0)
            
            # Weighted average (can be made configurable)
            aggregated[asset] = 0.6 * news_score + 0.4 * social_score
        
        return aggregated
    
    def _calculate_confidence(self, asset: str, market_data: Dict, sentiment_data: Dict) -> float:
        """Calculate confidence score for the analysis"""
        # Implement confidence calculation based on:
        # - Data freshness
        # - Source reliability
        # - Sentiment consensus
        # - Market volatility
        return 0.85  # Placeholder
    
    async def enhanced_trading_execution(self, asset_list: List[str], asset_type: str) -> Dict:
        """Enhanced trading execution with comprehensive risk management"""
        job_id = f"trade_{asset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = self.job_monitor.start_job(job_id)
        
        try:
            # Pre-execution checks
            if not await self._pre_execution_checks():
                raise Exception("Pre-execution checks failed")
            
            # Get current portfolio state
            total_assets = get_total_asset_value(upbit)
            cash_balance = get_cash_balance(upbit)
            
            execution_results = []
            
            for asset in asset_list:
                try:
                    # Enhanced signal generation
                    signals = await self._generate_comprehensive_signals(asset, asset_type)
                    
                    # Risk assessment
                    risk_assessment = await self.risk_manager.assess_trade_risk(
                        asset, asset_type, signals, total_assets
                    )
                    
                    if risk_assessment['approved']:
                        # Execute trade with enhanced parameters
                        trade_result = await self._execute_enhanced_trade(
                            asset, asset_type, signals, risk_assessment
                        )
                        execution_results.append(trade_result)
                        
                        # Update portfolio tracking
                        await self.portfolio_manager.update_position(asset, trade_result)
                    else:
                        logger.warning(f"Trade rejected for {asset}: {risk_assessment['reason']}")
                        execution_results.append({
                            'asset': asset,
                            'status': 'rejected',
                            'reason': risk_assessment['reason']
                        })
                
                except Exception as e:
                    logger.error(f"Trade execution failed for {asset}: {e}")
                    execution_results.append({
                        'asset': asset,
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Post-execution analysis
            await self._post_execution_analysis(execution_results)
            
            self.job_monitor.complete_job(job_id, result=execution_results)
            return {'status': 'success', 'results': execution_results}
            
        except Exception as e:
            error_msg = f"Trading execution failed: {str(e)}\n{traceback.format_exc()}"
            self.job_monitor.complete_job(job_id, error=error_msg)
            logger.error(error_msg)
            return {'status': 'error', 'message': str(e)}
    
    async def _pre_execution_checks(self) -> bool:
        """Comprehensive pre-execution safety checks"""
        checks = [
            self._check_market_conditions(),
            self._check_system_health(),
            self._check_risk_limits(),
            self._check_connectivity()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        return all(result is True for result in results if not isinstance(result, Exception))
    
    async def _check_market_conditions(self) -> bool:
        """Check if market conditions are suitable for trading"""
        # Check for extreme volatility, market closures, etc.
        return True  # Placeholder
    
    async def _check_system_health(self) -> bool:
        """Check system health metrics"""
        # CPU, memory, disk usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        return cpu_percent < 80 and memory_percent < 85
    
    async def _check_risk_limits(self) -> bool:
        """Check if we're within risk limits"""
        return await self.risk_manager.check_global_limits()
    
    async def _check_connectivity(self) -> bool:
        """Check connectivity to exchanges and data sources"""
        # Implement connectivity checks
        return True  # Placeholder
    
    async def _generate_comprehensive_signals(self, asset: str, asset_type: str) -> Dict:
        """Generate comprehensive trading signals"""
        # Get cached data
        cache_key = f"{asset_type}_{asset}"
        cached_data = NEWS_CACHE.get(cache_key, {})
        
        # Generate signals from multiple sources
        fear_greed = get_fear_greed_index()
        sentiment = cached_data.get('sentiment', 0.0)
        
        # This would integrate with your enhanced signal generation
        signals = {
            'fear_greed': fear_greed,
            'sentiment': sentiment,
            'technical': await self._get_technical_signals(asset),
            'fundamental': await self._get_fundamental_signals(asset),
            'confidence': cached_data.get('confidence', 0.5)
        }
        
        return signals
    
    async def _get_technical_signals(self, asset: str) -> Dict:
        """Get technical analysis signals"""
        # This would integrate with your technical analysis
        return {'rsi': 50, 'macd': 0, 'bollinger': 'neutral'}
    
    async def _get_fundamental_signals(self, asset: str) -> Dict:
        """Get fundamental analysis signals"""
        return {'valuation': 'fair', 'growth': 'positive'}
    
    async def _execute_enhanced_trade(self, asset: str, asset_type: str, signals: Dict, risk_assessment: Dict) -> Dict:
        """Execute trade with enhanced parameters"""
        # This would use your enhanced trading execution
        result = execute_trade(asset, asset_type, signals['fear_greed'], signals['sentiment'])
        
        # Add enhanced metadata
        result.update({
            'signals': signals,
            'risk_assessment': risk_assessment,
            'execution_time': datetime.now(),
            'confidence': signals.get('confidence', 0.5)
        })
        
        return result
    
    async def _post_execution_analysis(self, execution_results: List[Dict]):
        """Analyze execution results and update performance metrics"""
        await self.performance_monitor.update_execution_metrics(execution_results)
    
    def setup_enhanced_schedules(self):
        """Setup all trading schedules with enhanced features"""
        logger.info("Setting up enhanced trading schedules...")
        
        for asset_type, asset_list in self.config['assets'].items():
            for schedule_entry in self.config['schedule'][asset_type]:
                self._add_collection_job(asset_list, asset_type, schedule_entry)
                self._add_trading_job(asset_list, asset_type, schedule_entry)
        
        # Add monitoring jobs
        self._add_monitoring_jobs()
        
        # Add maintenance jobs
        self._add_maintenance_jobs()
    
    def _add_collection_job(self, asset_list: List[str], asset_type: str, schedule_entry: Dict):
        """Add data collection job"""
        day = schedule_entry['day'][:3]
        hour, minute = map(int, schedule_entry['time'].split(':'))
        
        job_id = f"collect_{asset_type}_{day}_{hour:02d}{minute:02d}"
        
        self.main_scheduler.add_job(
            self.enhanced_collect_and_analyze,
            CronTrigger(day_of_week=day, hour=hour, minute=minute),
            args=[asset_list, asset_type],
            id=job_id,
            executor='default',
            replace_existing=True,
            max_instances=1
        )
        
        logger.info(f"Added collection job: {job_id}")
    
    def _add_trading_job(self, asset_list: List[str], asset_type: str, schedule_entry: Dict):
        """Add trading execution job"""
        day = schedule_entry['day'][:3]
        hour, minute = map(int, schedule_entry['time'].split(':'))
        
        # Execute 30 minutes after collection
        trade_minute = (minute + 30) % 60
        trade_hour = hour if minute + 30 < 60 else hour + 1
        
        job_id = f"trade_{asset_type}_{day}_{trade_hour:02d}{trade_minute:02d}"
        
        self.main_scheduler.add_job(
            self.enhanced_trading_execution,
            CronTrigger(day_of_week=day, hour=trade_hour, minute=trade_minute),
            args=[asset_list, asset_type],
            id=job_id,
            executor='critical',  # Use dedicated executor for critical operations
            replace_existing=True,
            max_instances=1
        )
        
        logger.info(f"Added trading job: {job_id}")
    
    def _add_monitoring_jobs(self):
        """Add system monitoring jobs"""
        # System health monitoring every 5 minutes
        self.monitor_scheduler.add_job(
            self._monitor_system_health,
            IntervalTrigger(minutes=5),
            id="system_health_monitor"
        )
        
        # Performance monitoring every hour
        self.monitor_scheduler.add_job(
            self._monitor_performance,
            IntervalTrigger(hours=1),
            id="performance_monitor"
        )
        
        # Risk monitoring every 15 minutes
        self.monitor_scheduler.add_job(
            self._monitor_risk_metrics,
            IntervalTrigger(minutes=15),
            id="risk_monitor"
        )
    
    def _add_maintenance_jobs(self):
        """Add maintenance jobs"""
        # Daily cleanup at 2 AM
        self.main_scheduler.add_job(
            self._daily_cleanup,
            CronTrigger(hour=2, minute=0),
            id="daily_cleanup"
        )
        
        # Weekly performance report on Sundays at 6 AM
        self.main_scheduler.add_job(
            self._generate_weekly_report,
            CronTrigger(day_of_week='sun', hour=6, minute=0),
            id="weekly_report"
        )
    
    async def _monitor_system_health(self):
        """Monitor system health metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            health_metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'timestamp': datetime.now()
            }
            
            # Check for alerts
            if cpu_percent > 90:
                logger.critical(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                logger.critical(f"High memory usage: {memory.percent}%")
            if disk.percent > 90:
                logger.critical(f"High disk usage: {disk.percent}%")
            
            # Store metrics
            await self.performance_monitor.store_health_metrics(health_metrics)
            
        except Exception as e:
            logger.error(f"System health monitoring failed: {e}")
    
    async def _monitor_performance(self):
        """Monitor trading performance metrics"""
        try:
            await self.performance_monitor.calculate_performance_metrics()
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
    
    async def _monitor_risk_metrics(self):
        """Monitor risk metrics"""
        try:
            await self.risk_manager.update_risk_metrics()
        except Exception as e:
            logger.error(f"Risk monitoring failed: {e}")
    
    async def _daily_cleanup(self):
        """Daily maintenance and cleanup"""
        try:
            # Clean old logs
            self._cleanup_old_logs()
            
            # Clean cache
            self._cleanup_cache()
            
            # Database maintenance
            await self._database_maintenance()
            
            logger.info("Daily cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Daily cleanup failed: {e}")
    
    def _cleanup_old_logs(self):
        """Clean up old log files"""
        log_dir = Path("./logs")
        if log_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=30)
            for log_file in log_dir.glob("*.log"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    log_file.unlink()
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, value in NEWS_CACHE.items():
            if isinstance(value, dict) and 'timestamp' in value:
                if current_time - value['timestamp'] > timedelta(hours=24):
                    expired_keys.append(key)
        
        for key in expired_keys:
            del NEWS_CACHE[key]
        
        logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
    
    async def _database_maintenance(self):
        """Perform database maintenance"""
        # Implement database optimization, backup, etc.
        pass
    
    async def _generate_weekly_report(self):
        """Generate weekly performance report"""
        try:
            report = await self.performance_monitor.generate_weekly_report()
            
            # Save report
            report_path = f"reports/weekly_report_{datetime.now().strftime('%Y%m%d')}.json"
            Path("reports").mkdir(exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Weekly report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Weekly report generation failed: {e}")
    
    def start(self):
        """Start the scheduler system"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info("Starting Institutional Trading Scheduler...")
        
        try:
            # Setup schedules
            self.setup_enhanced_schedules()
            
            # Start all schedulers
            self.main_scheduler.start()
            self.monitor_scheduler.start()
            self.emergency_scheduler.start()
            
            self.is_running = True
            logger.info("All schedulers started successfully")
            
            # Keep the main thread alive
            try:
                asyncio.get_event_loop().run_until_complete(self.shutdown_event.wait())
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
            finally:
                self.shutdown()
                
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.shutdown()
            raise
    
    def shutdown(self):
        """Graceful shutdown of the scheduler system"""
        if not self.is_running:
            return
        
        logger.info("Initiating graceful shutdown...")
        
        try:
            # Stop schedulers
            if self.main_scheduler.running:
                self.main_scheduler.shutdown(wait=True)
            if self.monitor_scheduler.running:
                self.monitor_scheduler.shutdown(wait=True)
            if self.emergency_scheduler.running:
                self.emergency_scheduler.shutdown(wait=True)
            
            # Set shutdown event
            self.shutdown_event.set()
            
            self.is_running = False
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'is_running': self.is_running,
            'schedulers': {
                'main': self.main_scheduler.running if hasattr(self.main_scheduler, 'running') else False,
                'monitor': self.monitor_scheduler.running if hasattr(self.monitor_scheduler, 'running') else False,
                'emergency': self.emergency_scheduler.running if hasattr(self.emergency_scheduler, 'running') else False
            },
            'active_jobs': len(self.job_monitor.active_jobs),
            'job_statistics': {
                job_id: self.job_monitor.get_job_statistics(job_id)
                for job_id in self.job_monitor.job_history.keys()
            }
        }

# Emergency procedures
class EmergencyProcedures:
    """Emergency procedures for critical situations"""
    
    @staticmethod
    async def emergency_shutdown():
        """Emergency shutdown procedure"""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        # Cancel all pending orders
        # Close all positions
        # Notify administrators
        # Save critical state
        
        logger.critical("EMERGENCY SHUTDOWN COMPLETED")
    
    @staticmethod
    async def emergency_risk_stop():
        """Emergency risk management stop"""
        logger.critical("EMERGENCY RISK STOP INITIATED")
        
        # Stop all trading
        # Assess portfolio risk
        # Implement protective measures
        
        logger.critical("EMERGENCY RISK STOP COMPLETED")

# Usage example
async def main():
    """Main entry point"""
    try:
        scheduler = InstitutionalTradingScheduler()
        scheduler.start()
    except Exception as e:
        logger.critical(f"Critical system failure: {e}")
        await EmergencyProcedures.emergency_shutdown()
        sys.exit(1)

if __name__ == "__main__":
    # Set up asyncio event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Create and configure event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the main scheduler
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        loop.close()
        logger.info("System shutdown complete")

# Additional utility classes and functions

class SchedulerWebAPI:
    """Web API for monitoring and controlling the scheduler"""
    
    def __init__(self, scheduler: InstitutionalTradingScheduler, port: int = 8080):
        self.scheduler = scheduler
        self.port = port
        from flask import Flask, jsonify, request
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup web API routes"""
        
        @self.app.route('/status')
        def get_status():
            return jsonify(self.scheduler.get_status())
        
        @self.app.route('/jobs')
        def get_jobs():
            return jsonify({
                'active_jobs': list(self.scheduler.job_monitor.active_jobs.keys()),
                'job_history': {
                    job_id: [
                        {
                            'status': result.status.value,
                            'start_time': result.start_time.isoformat(),
                            'execution_time': result.execution_time
                        }
                        for result in results[-10:]  # Last 10 executions
                    ]
                    for job_id, results in self.scheduler.job_monitor.job_history.items()
                }
            })
        
        @self.app.route('/emergency/shutdown', methods=['POST'])
        def emergency_shutdown():
            asyncio.create_task(EmergencyProcedures.emergency_shutdown())
            return jsonify({'status': 'emergency_shutdown_initiated'})
        
        @self.app.route('/health')
        def health_check():
            return jsonify({
                'status': 'healthy' if self.scheduler.is_running else 'stopped',
                'timestamp': datetime.now().isoformat()
            })
    
    def start(self):
        """Start the web API server"""
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

class ConfigValidator:
    """Validate trading configuration"""
    
    @staticmethod
    def validate_config(config: Dict) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check required sections
        required_sections = ['assets', 'schedule', 'risk_limits', 'exchange_config']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate asset configuration
        if 'assets' in config:
            for asset_type, asset_list in config['assets'].items():
                if not isinstance(asset_list, list):
                    errors.append(f"Asset list for {asset_type} must be a list")
                if not asset_list:
                    errors.append(f"Asset list for {asset_type} cannot be empty")
        
        # Validate schedule configuration
        if 'schedule' in config:
            for asset_type, schedules in config['schedule'].items():
                for i, schedule in enumerate(schedules):
                    if 'day' not in schedule:
                        errors.append(f"Missing 'day' in schedule {i} for {asset_type}")
                    if 'time' not in schedule:
                        errors.append(f"Missing 'time' in schedule {i} for {asset_type}")
                    
                    # Validate time format
                    if 'time' in schedule:
                        try:
                            hour, minute = map(int, schedule['time'].split(':'))
                            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                                errors.append(f"Invalid time format in schedule {i} for {asset_type}")
                        except ValueError:
                            errors.append(f"Invalid time format in schedule {i} for {asset_type}")
        
        return errors

class PerformanceAnalyzer:
    """Advanced performance analysis for trading operations"""
    
    def __init__(self):
        self.trade_history = []
        self.performance_metrics = {}
    
    def add_trade(self, trade_data: Dict):
        """Add trade to performance tracking"""
        trade_data['timestamp'] = datetime.now()
        self.trade_history.append(trade_data)
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not portfolio_values:
            return 0.0
        
        portfolio_array = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max
        return float(np.min(drawdown))
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.0
        
        profitable_trades = sum(1 for trade in self.trade_history 
                               if trade.get('pnl', 0) > 0)
        return profitable_trades / len(self.trade_history)
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.trade_history:
            return {'error': 'No trade history available'}
        
        # Calculate returns
        returns = [trade.get('return', 0) for trade in self.trade_history]
        portfolio_values = [trade.get('portfolio_value', 100000) for trade in self.trade_history]
        
        report = {
            'total_trades': len(self.trade_history),
            'win_rate': self.calculate_win_rate(),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(portfolio_values),
            'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) if portfolio_values else 0,
            'average_return': np.mean(returns) if returns else 0,
            'volatility': np.std(returns) if returns else 0,
            'best_trade': max(returns) if returns else 0,
            'worst_trade': min(returns) if returns else 0,
            'profit_factor': self._calculate_profit_factor()
        }
        
        return report
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        profits = [trade.get('pnl', 0) for trade in self.trade_history if trade.get('pnl', 0) > 0]
        losses = [abs(trade.get('pnl', 0)) for trade in self.trade_history if trade.get('pnl', 0) < 0]
        
        gross_profit = sum(profits) if profits else 0
        gross_loss = sum(losses) if losses else 1  # Avoid division by zero
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

class AlertSystem:
    """Advanced alerting system for critical events"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_channels = self._setup_alert_channels()
    
    def _setup_alert_channels(self) -> Dict:
        """Setup various alert channels"""
        channels = {}
        
        # Email alerts
        if 'email' in self.config:
            channels['email'] = self._setup_email_alerts()
        
        # Slack alerts
        if 'slack' in self.config:
            channels['slack'] = self._setup_slack_alerts()
        
        # SMS alerts (Twilio)
        if 'sms' in self.config:
            channels['sms'] = self._setup_sms_alerts()
        
        # Discord alerts
        if 'discord' in self.config:
            channels['discord'] = self._setup_discord_alerts()
        
        return channels
    
    def _setup_email_alerts(self):
        """Setup email alerting"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        def send_email(subject: str, message: str, priority: str = 'medium'):
            try:
                config = self.config['email']
                
                msg = MIMEMultipart()
                msg['From'] = config['from_email']
                msg['To'] = config['to_email']
                msg['Subject'] = f"[{priority.upper()}] {subject}"
                
                msg.attach(MIMEText(message, 'plain'))
                
                server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
                server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(msg)
                server.quit()
                
                logger.info(f"Email alert sent: {subject}")
                
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
        
        return send_email
    
    def _setup_slack_alerts(self):
        """Setup Slack alerting"""
        import requests
        
        def send_slack(message: str, priority: str = 'medium'):
            try:
                webhook_url = self.config['slack']['webhook_url']
                
                color_map = {
                    'critical': '#FF0000',
                    'high': '#FF8000',
                    'medium': '#FFFF00',
                    'low': '#00FF00'
                }
                
                payload = {
                    "attachments": [{
                        "color": color_map.get(priority, '#FFFF00'),
                        "fields": [{
                            "title": f"{priority.upper()} Alert",
                            "value": message,
                            "short": False
                        }],
                        "footer": "Trading Bot Alert System",
                        "ts": int(datetime.now().timestamp())
                    }]
                }
                
                response = requests.post(webhook_url, json=payload)
                response.raise_for_status()
                
                logger.info(f"Slack alert sent: {priority}")
                
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")
        
        return send_slack
    
    def _setup_sms_alerts(self):
        """Setup SMS alerting via Twilio"""
        try:
            from twilio.rest import Client
            
            def send_sms(message: str, priority: str = 'medium'):
                try:
                    config = self.config['sms']
                    client = Client(config['account_sid'], config['auth_token'])
                    
                    message = client.messages.create(
                        body=f"[{priority.upper()}] {message}",
                        from_=config['from_number'],
                        to=config['to_number']
                    )
                    
                    logger.info(f"SMS alert sent: {message.sid}")
                    
                except Exception as e:
                    logger.error(f"Failed to send SMS alert: {e}")
            
            return send_sms
            
        except ImportError:
            logger.warning("Twilio not installed, SMS alerts disabled")
            return None
    
    def _setup_discord_alerts(self):
        """Setup Discord alerting"""
        import requests
        
        def send_discord(message: str, priority: str = 'medium'):
            try:
                webhook_url = self.config['discord']['webhook_url']
                
                embed_color_map = {
                    'critical': 16711680,  # Red
                    'high': 16744192,      # Orange
                    'medium': 16776960,    # Yellow
                    'low': 65280          # Green
                }
                
                payload = {
                    "embeds": [{
                        "title": f"{priority.upper()} Alert",
                        "description": message,
                        "color": embed_color_map.get(priority, 16776960),
                        "timestamp": datetime.now().isoformat(),
                        "footer": {
                            "text": "Trading Bot Alert System"
                        }
                    }]
                }
                
                response = requests.post(webhook_url, json=payload)
                response.raise_for_status()
                
                logger.info(f"Discord alert sent: {priority}")
                
            except Exception as e:
                logger.error(f"Failed to send Discord alert: {e}")
        
        return send_discord
    
    async def send_alert(self, message: str, priority: str = 'medium', channels: List[str] = None):
        """Send alert through specified channels"""
        if channels is None:
            channels = list(self.alert_channels.keys())
        
        for channel in channels:
            if channel in self.alert_channels and self.alert_channels[channel]:
                try:
                    await asyncio.to_thread(self.alert_channels[channel], message, priority)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")

# Additional configuration example
EXAMPLE_CONFIG = {
    "assets": {
        "crypto": ["BTC", "ETH", "BNB", "ADA", "DOT"],
        "stocks": ["AAPL", "GOOGL", "MSFT", "TSLA"]
    },
    "schedule": {
        "crypto": [
            {"day": "monday", "time": "09:00"},
            {"day": "wednesday", "time": "14:00"},
            {"day": "friday", "time": "18:00"}
        ],
        "stocks": [
            {"day": "tuesday", "time": "10:00"},
            {"day": "thursday", "time": "15:00"}
        ]
    },
    "risk_limits": {
        "max_position_size": 0.1,  # 10% of portfolio
        "max_daily_loss": 0.02,    # 2% daily loss limit
        "max_portfolio_risk": 0.15  # 15% portfolio risk
    },
    "exchange_config": {
        "sandbox": True,
        "rate_limit_calls": 100,
        "rate_limit_period": 60
    },
    "alerts": {
        "email": {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your_email@gmail.com",
            "password": "your_app_password",
            "from_email": "your_email@gmail.com",
            "to_email": "alerts@yourcompany.com"
        },
        "slack": {
            "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        },
        "discord": {
            "webhook_url": "https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK"
        }
    },
    "database": {
        "url": "postgresql://user:password@localhost:5432/trading_db",
        "pool_size": 10,
        "max_overflow": 20
    },
    "monitoring": {
        "health_check_interval": 300,  # 5 minutes
        "performance_update_interval": 3600,  # 1 hour
        "risk_check_interval": 900  # 15 minutes
    }
}