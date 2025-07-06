#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚨 네트워크 장애 대응 안전장치 시스템
========================================

네트워크 연결이 끊어졌을 때 자동으로 포지션을 보호하는 시스템
- 실시간 네트워크 상태 모니터링
- 연결 장애시 자동 매도 시스템
- 긴급 수동 제어 파일 시스템
- 다중 백업 연결 및 복구 시스템

Author: 전설적퀸트팀
Version: 1.0.0
"""

import asyncio
import aiohttp
import logging
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import requests
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('network_failsafe.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 🚨 네트워크 상태 및 장애 대응 모드 정의
# ============================================================================

class NetworkStatus(Enum):
    """네트워크 상태"""
    CONNECTED = "connected"
    UNSTABLE = "unstable"
    DISCONNECTED = "disconnected"
    CRITICAL = "critical"

class FailsafeMode(Enum):
    """장애 대응 모드"""
    PANIC_SELL = "panic_sell"           # 즉시 전량 시장가 매도
    CONSERVATIVE_SELL = "conservative_sell"  # 손익 고려한 단계적 매도
    HOLD_AND_WAIT = "hold_and_wait"     # 연결 복구까지 대기
    DISABLE_TRADING = "disable_trading"  # 신규 거래만 중단

@dataclass
class NetworkHealth:
    """네트워크 건강 상태"""
    status: NetworkStatus
    last_check: datetime
    success_rate: float
    avg_response_time: float
    consecutive_failures: int
    uptime_percentage: float

@dataclass
class EmergencyAction:
    """긴급 대응 액션"""
    timestamp: datetime
    action_type: str
    strategy: str
    symbol: str
    reason: str
    executed: bool
    error_message: Optional[str] = None

# ============================================================================
# 🔍 네트워크 모니터링 시스템
# ============================================================================

class NetworkMonitor:
    """네트워크 상태 실시간 모니터링"""
    
    def __init__(self):
        self.check_urls = self._load_check_urls()
        self.check_interval = int(os.getenv('NETWORK_CHECK_INTERVAL', '60'))
        self.timeout_threshold = int(os.getenv('NETWORK_TIMEOUT_THRESHOLD', '300'))
        self.retry_count = int(os.getenv('NETWORK_RETRY_COUNT', '5'))
        
        self.health_history: List[NetworkHealth] = []
        self.is_monitoring = False
        
        # 상태 추적
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        self.total_checks = 0
        self.successful_checks = 0
    
    def _load_check_urls(self) -> List[str]:
        """체크할 URL 목록 로드"""
        urls_str = os.getenv('NETWORK_CHECK_URLS', 
                           'https://api.upbit.com/v1/market/all,https://www.google.com,https://api.binance.com/api/v3/ping')
        return [url.strip() for url in urls_str.split(',')]
    
    async def check_single_url(self, session: aiohttp.ClientSession, url: str, timeout: int = 10) -> Dict:
        """단일 URL 연결 상태 체크"""
        try:
            start_time = time.time()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response_time = time.time() - start_time
                
                return {
                    'url': url,
                    'success': response.status == 200,
                    'status_code': response.status,
                    'response_time': response_time,
                    'error': None
                }
        except Exception as e:
            return {
                'url': url,
                'success': False,
                'status_code': 0,
                'response_time': timeout,
                'error': str(e)
            }
    
    async def perform_network_check(self) -> NetworkHealth:
        """종합 네트워크 상태 체크"""
        self.total_checks += 1
        check_results = []
        
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 모든 URL 동시 체크
                tasks = [self.check_single_url(session, url) for url in self.check_urls]
                check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 분석
            successful_checks = sum(1 for result in check_results 
                                  if isinstance(result, dict) and result.get('success', False))
            total_urls = len(self.check_urls)
            success_rate = successful_checks / total_urls if total_urls > 0 else 0
            
            # 평균 응답시간 계산
            valid_times = [result['response_time'] for result in check_results 
                          if isinstance(result, dict) and result.get('success', False)]
            avg_response_time = sum(valid_times) / len(valid_times) if valid_times else 999
            
            # 네트워크 상태 결정
            if success_rate >= 0.8 and avg_response_time < 5:
                status = NetworkStatus.CONNECTED
                self.consecutive_failures = 0
                self.last_success_time = datetime.now()
                self.successful_checks += 1
            elif success_rate >= 0.5:
                status = NetworkStatus.UNSTABLE
                self.consecutive_failures += 1
            elif success_rate > 0:
                status = NetworkStatus.DISCONNECTED
                self.consecutive_failures += 1
            else:
                status = NetworkStatus.CRITICAL
                self.consecutive_failures += 1
            
            # 전체 가동률 계산
            uptime_percentage = (self.successful_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
            
            health = NetworkHealth(
                status=status,
                last_check=datetime.now(),
                success_rate=success_rate,
                avg_response_time=avg_response_time,
                consecutive_failures=self.consecutive_failures,
                uptime_percentage=uptime_percentage
            )
            
            # 히스토리 저장 (최근 100개만)
            self.health_history.append(health)
            if len(self.health_history) > 100:
                self.health_history.pop(0)
            
            # 상세 로그
            logger.info(f"네트워크 체크: {status.value} | 성공률: {success_rate:.1%} | "
                       f"응답시간: {avg_response_time:.1f}s | 연속실패: {self.consecutive_failures}")
            
            return health
            
        except Exception as e:
            logger.error(f"네트워크 체크 실패: {e}")
            
            # 기본 실패 상태
            return NetworkHealth(
                status=NetworkStatus.CRITICAL,
                last_check=datetime.now(),
                success_rate=0.0,
                avg_response_time=999.0,
                consecutive_failures=self.consecutive_failures + 1,
                uptime_percentage=0.0
            )
    
    async def start_monitoring(self, callback: Optional[Callable] = None):
        """네트워크 모니터링 시작"""
        self.is_monitoring = True
        logger.info(f"🔍 네트워크 모니터링 시작 (간격: {self.check_interval}초)")
        
        while self.is_monitoring:
            try:
                health = await self.perform_network_check()
                
                # 콜백 실행 (장애 대응 시스템에 알림)
                if callback:
                    await callback(health)
                
                # 다음 체크까지 대기
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(30)  # 30초 후 재시도
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        logger.info("⏹️ 네트워크 모니터링 중지")
    
    def get_current_status(self) -> Optional[NetworkHealth]:
        """현재 네트워크 상태 조회"""
        return self.health_history[-1] if self.health_history else None
    
    def is_connection_stable(self) -> bool:
        """연결이 안정적인지 확인"""
        if not self.health_history:
            return False
        
        recent_health = self.health_history[-1]
        return (recent_health.status in [NetworkStatus.CONNECTED, NetworkStatus.UNSTABLE] and
                recent_health.consecutive_failures < 3)

# ============================================================================
# 📁 긴급 제어 파일 시스템
# ============================================================================

class EmergencyFileController:
    """긴급 상황 파일 기반 제어 시스템"""
    
    def __init__(self):
        self.emergency_sell_file = Path(os.getenv('EMERGENCY_SELL_ALL_FILE', 'emergency_sell_all.flag'))
        self.emergency_stop_file = Path(os.getenv('EMERGENCY_STOP_TRADING_FILE', 'emergency_stop_trading.flag'))
        self.emergency_enable_file = Path(os.getenv('EMERGENCY_ENABLE_TRADING_FILE', 'emergency_enable_trading.flag'))
        
        # 상태 추적
        self.sell_all_triggered = False
        self.trading_stopped = False
        self.last_file_check = datetime.now()
    
    def check_emergency_files(self) -> Dict[str, bool]:
        """긴급 제어 파일들 상태 체크"""
        try:
            current_time = datetime.now()
            
            # 전량매도 파일 체크
            sell_all_exists = self.emergency_sell_file.exists()
            if sell_all_exists and not self.sell_all_triggered:
                logger.warning(f"🚨 긴급 전량매도 파일 감지: {self.emergency_sell_file}")
                self.sell_all_triggered = True
            
            # 거래중단 파일 체크
            stop_trading_exists = self.emergency_stop_file.exists()
            if stop_trading_exists and not self.trading_stopped:
                logger.warning(f"⏸️ 거래중단 파일 감지: {self.emergency_stop_file}")
                self.trading_stopped = True
            
            # 거래재개 파일 체크
            enable_trading_exists = self.emergency_enable_file.exists()
            if enable_trading_exists and self.trading_stopped:
                logger.info(f"▶️ 거래재개 파일 감지: {self.emergency_enable_file}")
                self.trading_stopped = False
                # 재개 파일 삭제
                self.emergency_enable_file.unlink()
            
            self.last_file_check = current_time
            
            return {
                'sell_all': sell_all_exists and self.sell_all_triggered,
                'stop_trading': stop_trading_exists or self.trading_stopped,
                'enable_trading': enable_trading_exists
            }
            
        except Exception as e:
            logger.error(f"긴급 파일 체크 실패: {e}")
            return {'sell_all': False, 'stop_trading': False, 'enable_trading': False}
    
    def create_emergency_sell_file(self, reason: str = "Manual trigger"):
        """긴급 전량매도 파일 생성"""
        try:
            with open(self.emergency_sell_file, 'w', encoding='utf-8') as f:
                f.write(f"Emergency sell triggered at {datetime.now()}\n")
                f.write(f"Reason: {reason}\n")
            logger.warning(f"🚨 긴급 전량매도 파일 생성: {reason}")
        except Exception as e:
            logger.error(f"긴급 파일 생성 실패: {e}")
    
    def create_emergency_stop_file(self, reason: str = "Manual stop"):
        """거래중단 파일 생성"""
        try:
            with open(self.emergency_stop_file, 'w', encoding='utf-8') as f:
                f.write(f"Trading stopped at {datetime.now()}\n")
                f.write(f"Reason: {reason}\n")
            logger.warning(f"⏸️ 거래중단 파일 생성: {reason}")
        except Exception as e:
            logger.error(f"거래중단 파일 생성 실패: {e}")
    
    def create_emergency_enable_file(self):
        """거래재개 파일 생성"""
        try:
            with open(self.emergency_enable_file, 'w', encoding='utf-8') as f:
                f.write(f"Trading enabled at {datetime.now()}\n")
            logger.info(f"▶️ 거래재개 파일 생성")
        except Exception as e:
            logger.error(f"거래재개 파일 생성 실패: {e}")
    
    def clear_emergency_files(self):
        """모든 긴급 파일 삭제"""
        files_to_clear = [self.emergency_sell_file, self.emergency_stop_file, self.emergency_enable_file]
        
        for file_path in files_to_clear:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"🗑️ 긴급 파일 삭제: {file_path}")
            except Exception as e:
                logger.error(f"파일 삭제 실패 {file_path}: {e}")
        
        # 상태 리셋
        self.sell_all_triggered = False
        self.trading_stopped = False

# ============================================================================
# 💰 전략별 포지션 매니저 인터페이스
# ============================================================================

class StrategyPositionManager:
    """전략별 포지션 관리 인터페이스"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.positions = {}
        self.last_update = datetime.now()
    
    async def get_all_positions(self) -> Dict:
        """모든 포지션 조회 (각 전략에서 구현)"""
        raise NotImplementedError("각 전략에서 구현해야 함")
    
    async def emergency_sell_position(self, symbol: str, mode: FailsafeMode) -> Dict:
        """긴급 포지션 매도 (각 전략에서 구현)"""
        raise NotImplementedError("각 전략에서 구현해야 함")
    
    async def emergency_sell_all(self, mode: FailsafeMode) -> List[Dict]:
        """모든 포지션 긴급 매도"""
        try:
            positions = await self.get_all_positions()
            results = []
            
            for symbol, position_info in positions.items():
                try:
                    result = await self.emergency_sell_position(symbol, mode)
                    results.append(result)
                    logger.info(f"🚨 {self.strategy_name} 긴급매도: {symbol}")
                except Exception as e:
                    logger.error(f"❌ {self.strategy_name} {symbol} 긴급매도 실패: {e}")
                    results.append({'symbol': symbol, 'success': False, 'error': str(e)})
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {self.strategy_name} 전량매도 실패: {e}")
            return []

# ============================================================================
# 🚨 메인 장애 대응 시스템
# ============================================================================

class NetworkFailsafeSystem:
    """네트워크 장애 종합 대응 시스템"""
    
    def __init__(self):
        # 환경변수 로드
        self.enabled = os.getenv('NETWORK_FAILSAFE_ENABLED', 'true').lower() == 'true'
        self.failsafe_mode = FailsafeMode(os.getenv('NETWORK_FAILSAFE_MODE', 'conservative_sell'))
        self.critical_loss_threshold = float(os.getenv('NETWORK_CRITICAL_LOSS_THRESHOLD', '10000'))
        
        # 핵심 컴포넌트
        self.network_monitor = NetworkMonitor()
        self.file_controller = EmergencyFileController()
        
        # 전략별 매니저들 (나중에 등록)
        self.strategy_managers: Dict[str, StrategyPositionManager] = {}
        
        # 상태 추적
        self.emergency_actions: List[EmergencyAction] = []
        self.system_active = False
        self.last_emergency_time = None
        
        # 알림 시스템
        self.telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
    
    def register_strategy_manager(self, strategy_name: str, manager: StrategyPositionManager):
        """전략 매니저 등록"""
        self.strategy_managers[strategy_name] = manager
        logger.info(f"📝 전략 매니저 등록: {strategy_name}")
    
    async def network_status_callback(self, health: NetworkHealth):
        """네트워크 상태 변화 콜백"""
        try:
            # 심각한 연결 장애 감지
            if health.status == NetworkStatus.CRITICAL and health.consecutive_failures >= 5:
                await self._handle_critical_network_failure(health)
            
            # 불안정한 연결 경고
            elif health.status == NetworkStatus.UNSTABLE and health.consecutive_failures >= 3:
                await self._handle_unstable_connection(health)
            
            # 연결 복구 알림
            elif health.status == NetworkStatus.CONNECTED and health.consecutive_failures == 0:
                if self.last_emergency_time and (datetime.now() - self.last_emergency_time).seconds < 3600:
                    await self._send_alert(f"🟢 네트워크 연결 복구됨! 가동률: {health.uptime_percentage:.1f}%")
            
        except Exception as e:
            logger.error(f"네트워크 상태 콜백 오류: {e}")
    
    async def _handle_critical_network_failure(self, health: NetworkHealth):
        """심각한 네트워크 장애 대응"""
        logger.error(f"🚨 심각한 네트워크 장애 감지! 연속실패: {health.consecutive_failures}")
        
        # 중복 실행 방지
        if self.last_emergency_time and (datetime.now() - self.last_emergency_time).seconds < 300:
            return
        
        self.last_emergency_time = datetime.now()
        
        # 알림 전송
        await self._send_alert(
            f"🚨 CRITICAL NETWORK FAILURE 🚨\n"
            f"연속 실패: {health.consecutive_failures}회\n"
            f"성공률: {health.success_rate:.1%}\n"
            f"대응모드: {self.failsafe_mode.value}\n"
            f"자동 대응을 시작합니다..."
        )
        
        # 장애 대응 실행
        if self.failsafe_mode == FailsafeMode.PANIC_SELL:
            await self._execute_panic_sell("Critical network failure")
        elif self.failsafe_mode == FailsafeMode.CONSERVATIVE_SELL:
            await self._execute_conservative_sell("Critical network failure")
        elif self.failsafe_mode == FailsafeMode.DISABLE_TRADING:
            await self._disable_trading("Critical network failure")
        # HOLD_AND_WAIT는 별도 액션 없음
    
    async def _handle_unstable_connection(self, health: NetworkHealth):
        """불안정한 연결 경고"""
        logger.warning(f"⚠️ 불안정한 네트워크 연결: 연속실패 {health.consecutive_failures}회")
        
        # 경고 알림 (5분에 한 번만)
        if not hasattr(self, '_last_unstable_alert'):
            self._last_unstable_alert = datetime.now()
            await self._send_alert(
                f"⚠️ 네트워크 불안정\n"
                f"연속 실패: {health.consecutive_failures}회\n"
                f"성공률: {health.success_rate:.1%}\n"
                f"모니터링 중..."
            )
        elif (datetime.now() - self._last_unstable_alert).seconds > 300:
            self._last_unstable_alert = datetime.now()
            await self._send_alert(f"⚠️ 네트워크 여전히 불안정 (실패: {health.consecutive_failures}회)")
    
    async def _execute_panic_sell(self, reason: str):
        """패닉 매도 실행"""
        logger.error(f"🚨 패닉 매도 실행: {reason}")
        
        results = []
        for strategy_name, manager in self.strategy_managers.items():
            try:
                strategy_results = await manager.emergency_sell_all(FailsafeMode.PANIC_SELL)
                results.extend(strategy_results)
                
                # 액션 기록
                for result in strategy_results:
                    action = EmergencyAction(
                        timestamp=datetime.now(),
                        action_type="panic_sell",
                        strategy=strategy_name,
                        symbol=result.get('symbol', 'ALL'),
                        reason=reason,
                        executed=result.get('success', False),
                        error_message=result.get('error')
                    )
                    self.emergency_actions.append(action)
                
            except Exception as e:
                logger.error(f"❌ {strategy_name} 패닉 매도 실패: {e}")
        
        # 결과 알림
        success_count = sum(1 for r in results if r.get('success', False))
        total_count = len(results)
        
        await self._send_alert(
            f"🚨 패닉 매도 완료\n"
            f"성공: {success_count}/{total_count}\n"
            f"사유: {reason}"
        )
    
    async def _execute_conservative_sell(self, reason: str):
        """보수적 매도 실행 (손익 고려)"""
        logger.warning(f"⚠️ 보수적 매도 실행: {reason}")
        
        results = []
        for strategy_name, manager in self.strategy_managers.items():
            try:
                # 보수적 매도는 각 전략에서 손익을 고려하여 실행
                strategy_results = await manager.emergency_sell_all(FailsafeMode.CONSERVATIVE_SELL)
                results.extend(strategy_results)
                
            except Exception as e:
                logger.error(f"❌ {strategy_name} 보수적 매도 실패: {e}")
        
        success_count = sum(1 for r in results if r.get('success', False))
        await self._send_alert(f"⚠️ 보수적 매도 완료: {success_count}개 포지션")
    
    async def _disable_trading(self, reason: str):
        """신규 거래 중단"""
        logger.warning(f"⏸️ 신규 거래 중단: {reason}")
        
        # 거래중단 파일 생성
        self.file_controller.create_emergency_stop_file(reason)
        
        await self._send_alert(
            f"⏸️ 신규 거래 중단\n"
            f"사유: {reason}\n"
            f"기존 포지션은 유지됩니다."
        )
    
    async def _send_alert(self, message: str):
        """알림 전송"""
        try:
            # 콘솔 출력
            logger.info(f"📢 ALERT: {message}")
            
            # 텔레그램 알림
            if self.telegram_enabled and self.telegram_token and self.telegram_chat_id:
                await self._send_telegram_alert(message)
            
        except Exception as e:
            logger.error(f"알림 전송 실패: {e}")
    
    async def _send_telegram_alert(self, message: str):
        """텔레그램 알림 전송"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': f"🚨 네트워크 안전장치\n{message}",
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.debug("텔레그램 알림 전송 성공")
                    else:
                        logger.error(f"텔레그램 알림 실패: {response.status}")
                        
        except Exception as e:
            logger.error(f"텔레그램 전송 오류: {e}")
    
    async def start_monitoring(self):
        """안전장치 시스템 시작"""
        if not self.enabled:
            logger.info("🚨 네트워크 안전장치 비활성화됨")
            return
        
        self.system_active = True
        logger.info("🚨 네트워크 안전장치 시스템 시작!")
        
        # 시작 알림
        await self._send_alert(
            f"🟢 네트워크 안전장치 시작\n"
            f"모드: {self.failsafe_mode.value}\n"
            f"등록된 전략: {list(self.strategy_managers.keys())}"
        )
        
        # 병렬 실행
        tasks = [
            self.network_monitor.start_monitoring(self.network_status_callback),
            self._file_monitor_loop(),
            self._periodic_health_check()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"안전장치 시스템 오류: {e}")
        finally:
            self.system_active = False
    
    async def _file_monitor_loop(self):
        """파일 기반 긴급 제어 모니터링"""
        while self.system_active:
            try:
                file_status = self.file_controller.check_emergency_files()
                
                # 긴급 전량매도
                if file_status['sell_all']:
                    await self._execute_panic_sell("Emergency file trigger")
                    self.file_controller.sell_all_triggered = False  # 리셋
                
                # 거래 중단/재개
                if file_status['stop_trading']:
                    logger.info("⏸️ 파일 기반 거래 중단 활성")
                elif file_status['enable_trading']:
                    logger.info("▶️ 파일 기반 거래 재개")
                
                await asyncio.sleep(10)  # 10초마다 체크
                
            except Exception as e:
                logger.error(f"파일 모니터링 오류: {e}")
                await asyncio.sleep(30)
    
    async def _periodic_health_check(self):
        """주기적 건강 상태 체크"""
        while self.system_active:
            try:
                await asyncio.sleep(3600)  # 1시간마다
                
                # 시스템 상태 보고
                current_health = self.network_monitor.get_current_status()
                if current_health:
                    logger.info(
                        f"📊 시간별 상태 보고: {current_health.status.value} | "
                        f"가동률: {current_health.uptime_percentage:.1f}% | "
                        f"긴급대응: {len(self.emergency_actions)}회"
                    )
                
            except Exception as e:
                logger.error(f"건강 상태 체크 오류: {e}")
    
    def stop_monitoring(self):
        """안전장치 시스템 중지"""
        self.system_active = False
        self.network_monitor.stop_monitoring()
        logger.info("⏹️ 네트워크 안전장치 시스템 중지")
    
    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        current_health = self.network_monitor.get_current_status()
        
        return {
            'system_active': self.system_active,
            'failsafe_enabled': self.enabled,
            'failsafe_mode': self.failsafe_mode.value,
            'network_status': current_health.status.value if current_health else 'unknown',
            'network_uptime': current_health.uptime_percentage if current_health else 0,
            'consecutive_failures': current_health.consecutive_failures if current_health else 0,
            'registered_strategies': list(self.strategy_managers.keys()),
            'emergency_actions_count': len(self.emergency_actions),
            'last_emergency': self.last_emergency_time.isoformat() if self.last_emergency_time else None
        }

# ============================================================================
# 🛠️ 전략별 구현 예시
# ============================================================================

class USStrategyManager(StrategyPositionManager):
    """미국 주식 전략 매니저 예시"""
    
    def __init__(self):
        super().__init__("US_STRATEGY")
    
    async def get_all_positions(self) -> Dict:
        """미국 주식 포지션 조회"""
        # 실제로는 us_strategy.py의 포지션 매니저에서 조회
        return {
            'AAPL': {'quantity': 100, 'avg_price': 150.0, 'current_pnl': 500.0},
            'MSFT': {'quantity': 50, 'avg_price': 300.0, 'current_pnl': -200.0}
        }
    
    async def emergency_sell_position(self, symbol: str, mode: FailsafeMode) -> Dict:
        """긴급 포지션 매도"""
        try:
            # 실제로는 IBKR API 호출
            if mode == FailsafeMode.PANIC_SELL:
                # 즉시 시장가 매도
                logger.info(f"🚨 {symbol} 패닉 매도 (IBKR)")
                return {'symbol': symbol, 'success': True, 'type': 'market_sell'}
            
            elif mode == FailsafeMode.CONSERVATIVE_SELL:
                # 손익 고려한 매도
                position = (await self.get_all_positions()).get(symbol, {})
                pnl = position.get('current_pnl', 0)
                
                if pnl < -1000:  # 1000달러 이상 손실시만 매도
                    logger.info(f"⚠️ {symbol} 손실 제한 매도 (IBKR)")
                    return {'symbol': symbol, 'success': True, 'type': 'conservative_sell'}
                else:
                    logger.info(f"⏸️ {symbol} 손실 적어 홀딩 유지")
                    return {'symbol': symbol, 'success': False, 'type': 'hold'}
            
            return {'symbol': symbol, 'success': False, 'type': 'unknown_mode'}
            
        except Exception as e:
            return {'symbol': symbol, 'success': False, 'error': str(e)}

class CryptoStrategyManager(StrategyPositionManager):
    """가상화폐 전략 매니저 예시"""
    
    def __init__(self):
        super().__init__("CRYPTO_STRATEGY")
    
    async def get_all_positions(self) -> Dict:
        """가상화폐 포지션 조회"""
        # 실제로는 coin_strategy.py의 포지션 매니저에서 조회
        return {
            'KRW-BTC': {'quantity': 0.1, 'avg_price': 50000000, 'current_pnl': 500000},
            'KRW-ETH': {'quantity': 1.0, 'avg_price': 3000000, 'current_pnl': -100000}
        }
    
    async def emergency_sell_position(self, symbol: str, mode: FailsafeMode) -> Dict:
        """긴급 포지션 매도"""
        try:
            # 실제로는 업비트 API 호출
            if mode == FailsafeMode.PANIC_SELL:
                logger.info(f"🚨 {symbol} 패닉 매도 (업비트)")
                return {'symbol': symbol, 'success': True, 'type': 'market_sell'}
            
            elif mode == FailsafeMode.CONSERVATIVE_SELL:
                position = (await self.get_all_positions()).get(symbol, {})
                pnl = position.get('current_pnl', 0)
                
                if pnl < -500000:  # 50만원 이상 손실시만 매도
                    logger.info(f"⚠️ {symbol} 손실 제한 매도 (업비트)")
                    return {'symbol': symbol, 'success': True, 'type': 'conservative_sell'}
                else:
                    return {'symbol': symbol, 'success': False, 'type': 'hold'}
            
            return {'symbol': symbol, 'success': False, 'type': 'unknown_mode'}
            
        except Exception as e:
            return {'symbol': symbol, 'success': False, 'error': str(e)}

# ============================================================================
# 🚀 메인 실행 함수
# ============================================================================

async def main():
    """네트워크 안전장치 테스트 실행"""
    print("🚨 네트워크 안전장치 시스템 테스트 🚨")
    print("="*60)
    
    # 안전장치 시스템 초기화
    failsafe_system = NetworkFailsafeSystem()
    
    # 전략 매니저들 등록
    us_manager = USStrategyManager()
    crypto_manager = CryptoStrategyManager()
    
    failsafe_system.register_strategy_manager("US_STOCKS", us_manager)
    failsafe_system.register_strategy_manager("CRYPTO", crypto_manager)
    
    # 시스템 상태 출력
    status = failsafe_system.get_system_status()
    print(f"📊 시스템 상태:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print(f"\n🔍 설정된 기능:")
    print(f"   네트워크 모니터링: {failsafe_system.enabled}")
    print(f"   장애 대응 모드: {failsafe_system.failsafe_mode.value}")
    print(f"   등록된 전략: {len(failsafe_system.strategy_managers)}개")
    print(f"   텔레그램 알림: {failsafe_system.telegram_enabled}")
    
    print(f"\n🛠️ 수동 제어 방법:")
    print(f"   긴급 전량매도: touch {failsafe_system.file_controller.emergency_sell_file}")
    print(f"   거래 중단: touch {failsafe_system.file_controller.emergency_stop_file}")
    print(f"   거래 재개: touch {failsafe_system.file_controller.emergency_enable_file}")
    
    print(f"\n🚀 안전장치 시스템 시작 (Ctrl+C로 중지)")
    
    try:
        await failsafe_system.start_monitoring()
    except KeyboardInterrupt:
        print(f"\n👋 안전장치 시스템을 중지합니다")
        failsafe_system.stop_monitoring()
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")

def create_emergency_files():
    """긴급 제어 파일들 생성 (수동 실행용)"""
    controller = EmergencyFileController()
    
    print("🚨 긴급 제어 파일 생성 도구")
    print("1. 긴급 전량매도")
    print("2. 거래 중단")
    print("3. 거래 재개")
    print("4. 모든 파일 삭제")
    
    choice = input("선택 (1-4): ").strip()
    
    if choice == '1':
        reason = input("매도 사유 입력: ").strip() or "Manual emergency sell"
        controller.create_emergency_sell_file(reason)
        print("✅ 긴급 전량매도 파일 생성됨")
    
    elif choice == '2':
        reason = input("중단 사유 입력: ").strip() or "Manual trading stop"
        controller.create_emergency_stop_file(reason)
        print("✅ 거래 중단 파일 생성됨")
    
    elif choice == '3':
        controller.create_emergency_enable_file()
        print("✅ 거래 재개 파일 생성됨")
    
    elif choice == '4':
        controller.clear_emergency_files()
        print("✅ 모든 긴급 파일 삭제됨")
    
    else:
        print("❌ 잘못된 선택")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'emergency':
            # 긴급 제어 파일 생성
            create_emergency_files()
        elif command == 'test':
            # 네트워크 테스트만
            async def test_network():
                monitor = NetworkMonitor()
                health = await monitor.perform_network_check()
                print(f"네트워크 상태: {health.status.value}")
                print(f"성공률: {health.success_rate:.1%}")
                print(f"응답시간: {health.avg_response_time:.1f}초")
            
            asyncio.run(test_network())
        else:
            print("사용법:")
            print("  python network_failsafe.py        # 전체 시스템 실행")
            print("  python network_failsafe.py test   # 네트워크 테스트만")
            print("  python network_failsafe.py emergency  # 긴급 제어 파일 생성")
    else:
        # 기본 실행: 전체 안전장치 시스템
        asyncio.run(main())