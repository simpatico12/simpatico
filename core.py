#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 통합 코어 시스템 (core.py)
=================================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 (4대 전략 통합)

✨ 핵심 기능:
- 4대 전략 통합 관리 시스템
- IBKR 자동 환전 기능 (달러 ↔ 엔/루피)
- 네트워크 모니터링 + 끊김 시 전량 매도
- 통합 포지션 관리 + 리스크 제어
- 실시간 모니터링 + 알림 시스템
- 성과 추적 + 자동 백업
- 🚨 응급 오류 감지 시스템

Author: 퀸트마스터팀
Version: 1.1.0 (IBKR 자동환전 + 응급 오류 감지)
"""

import asyncio
import logging
import sys
import os
import json
import time
import psutil
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
import requests
import aiohttp
import sqlite3
import shutil
import subprocess
import signal

# 전략 모듈 임포트
try:
    from us_strategy import LegendaryQuantStrategy as USStrategy
    US_AVAILABLE = True
except ImportError:
    US_AVAILABLE = False
    print("⚠️ 미국 전략 모듈 없음")

try:
    from jp_strategy import YenHunter as JapanStrategy
    JAPAN_AVAILABLE = True
except ImportError:
    JAPAN_AVAILABLE = False
    print("⚠️ 일본 전략 모듈 없음")

try:
    from inda_strategy import LegendaryIndiaStrategy as IndiaStrategy
    INDIA_AVAILABLE = True
except ImportError:
    INDIA_AVAILABLE = False
    print("⚠️ 인도 전략 모듈 없음")

try:
    from coin_strategy import LegendaryQuantMaster as CryptoStrategy
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("⚠️ 암호화폐 전략 모듈 없음")

# 업비트 연동
try:
    import pyupbit
    UPBIT_AVAILABLE = True
except ImportError:
    UPBIT_AVAILABLE = False
    print("⚠️ 업비트 모듈 없음")

# IBKR 연동
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("⚠️ IBKR 모듈 없음")

# ============================================================================
# 🎯 통합 설정 관리자
# ============================================================================
class CoreConfig:
    """통합 설정 관리"""
    
    def __init__(self):
        load_dotenv()
        
        # 시스템 설정
        self.TOTAL_PORTFOLIO_VALUE = float(os.getenv('TOTAL_PORTFOLIO_VALUE', 1000000000))
        self.MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', 0.05))
        
        # 전략별 활성화
        self.US_ENABLED = os.getenv('US_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.JAPAN_ENABLED = os.getenv('JAPAN_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.INDIA_ENABLED = os.getenv('INDIA_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.CRYPTO_ENABLED = os.getenv('CRYPTO_STRATEGY_ENABLED', 'true').lower() == 'true'
        
        # 전략별 자원 배분
        self.US_ALLOCATION = float(os.getenv('US_STRATEGY_ALLOCATION', 0.40))
        self.JAPAN_ALLOCATION = float(os.getenv('JAPAN_STRATEGY_ALLOCATION', 0.25))
        self.CRYPTO_ALLOCATION = float(os.getenv('CRYPTO_STRATEGY_ALLOCATION', 0.20))
        self.INDIA_ALLOCATION = float(os.getenv('INDIA_STRATEGY_ALLOCATION', 0.15))
        
        # IBKR 설정
        self.IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
        self.IBKR_PORT = int(os.getenv('IBKR_PORT', 7497))
        self.IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', 1))
        
        # 네트워크 모니터링
        self.NETWORK_MONITORING = os.getenv('NETWORK_MONITORING_ENABLED', 'true').lower() == 'true'
        self.NETWORK_CHECK_INTERVAL = int(os.getenv('NETWORK_CHECK_INTERVAL', 30))
        self.NETWORK_DISCONNECT_SELL = os.getenv('NETWORK_DISCONNECT_SELL_ALL', 'false').lower() == 'true'
        
        # 응급 오류 감지
        self.EMERGENCY_SELL_ON_ERROR = os.getenv('EMERGENCY_SELL_ON_ERROR', 'true').lower() == 'true'
        self.EMERGENCY_ERROR_COUNT = int(os.getenv('EMERGENCY_ERROR_COUNT', 5))
        self.EMERGENCY_MEMORY_THRESHOLD = int(os.getenv('EMERGENCY_MEMORY_THRESHOLD', 95))
        self.EMERGENCY_CPU_THRESHOLD = int(os.getenv('EMERGENCY_CPU_THRESHOLD', 90))
        self.EMERGENCY_DISK_THRESHOLD = int(os.getenv('EMERGENCY_DISK_THRESHOLD', 5))
        
        # 업비트 설정
        self.UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
        self.UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')
        self.UPBIT_DEMO_MODE = os.getenv('CRYPTO_DEMO_MODE', 'true').lower() == 'true'
        
        # 알림 설정
        self.TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # 데이터베이스
        self.DB_PATH = os.getenv('DATABASE_PATH', './data/quant_core.db')
        self.BACKUP_PATH = os.getenv('BACKUP_PATH', './backups/')

# ============================================================================
# 🚨 응급 오류 감지 시스템
# ============================================================================
class EmergencyErrorDetector:
    """응급 오류 감지 및 대응 시스템"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        self.cpu_high_start = None
        self.memory_alerts = []
        self.emergency_triggered = False
        
        self.logger = logging.getLogger('EmergencyDetector')
    
    def check_system_health(self) -> Dict[str, Any]:
        """시스템 상태 종합 체크"""
        health_status = {
            'healthy': True,
            'warnings': [],
            'errors': [],
            'emergency_needed': False
        }
        
        try:
            # 메모리 체크
            memory_percent = psutil.virtual_memory().percent
            if memory_percent >= self.config.EMERGENCY_MEMORY_THRESHOLD:
                health_status['emergency_needed'] = True
                health_status['errors'].append(f'메모리 위험: {memory_percent:.1f}%')
            elif memory_percent >= 85:
                health_status['warnings'].append(f'메모리 경고: {memory_percent:.1f}%')
            
            # CPU 체크
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent >= self.config.EMERGENCY_CPU_THRESHOLD:
                if self.cpu_high_start is None:
                    self.cpu_high_start = time.time()
                elif time.time() - self.cpu_high_start > 300:  # 5분 연속
                    health_status['emergency_needed'] = True
                    health_status['errors'].append(f'CPU 위험: {cpu_percent:.1f}% (5분 연속)')
            else:
                self.cpu_high_start = None
            
            # 디스크 체크
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < self.config.EMERGENCY_DISK_THRESHOLD:
                health_status['emergency_needed'] = True
                health_status['errors'].append(f'디스크 위험: {free_gb:.1f}GB 남음')
            elif free_gb < 10:
                health_status['warnings'].append(f'디스크 경고: {free_gb:.1f}GB 남음')
            
            # 네트워크 체크
            network_status = self._check_network()
            if not network_status['connected']:
                health_status['emergency_needed'] = True
                health_status['errors'].append('네트워크 연결 실패')
            
            # 프로세스 체크
            process_status = self._check_processes()
            if process_status['zombie_count'] > 5:
                health_status['warnings'].append(f'좀비 프로세스 {process_status["zombie_count"]}개')
            
            health_status['healthy'] = not health_status['errors']
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"시스템 상태 체크 실패: {e}")
            return {
                'healthy': False,
                'warnings': [],
                'errors': [f'상태 체크 실패: {str(e)}'],
                'emergency_needed': True
            }
    
    def record_error(self, error_type: str, error_msg: str, critical: bool = False):
        """오류 기록 및 응급 상황 판단"""
        current_time = time.time()
        
        self.error_count += 1
        
        # 연속 오류 체크
        if self.last_error_time and current_time - self.last_error_time < 60:
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 1
        
        self.last_error_time = current_time
        
        self.logger.error(f"오류 기록: {error_type} - {error_msg}")
        
        # 응급 상황 판단
        emergency_conditions = [
            critical,
            self.consecutive_errors >= self.config.EMERGENCY_ERROR_COUNT,
            error_type in ['network_failure', 'api_failure', 'system_crash']
        ]
        
        if any(emergency_conditions) and not self.emergency_triggered:
            self.emergency_triggered = True
            self.logger.critical(f"🚨 응급 상황 감지: {error_type}")
            return True
        
        return False
    
    def _check_network(self) -> Dict[str, Any]:
        """네트워크 연결 상태 체크"""
        try:
            response = requests.get('https://www.google.com', timeout=5)
            return {'connected': response.status_code == 200, 'response_time': response.elapsed.total_seconds()}
        except:
            return {'connected': False, 'response_time': None}
    
    def _check_processes(self) -> Dict[str, Any]:
        """프로세스 상태 체크"""
        try:
            processes = list(psutil.process_iter())
            zombie_count = sum(1 for p in processes if p.status() == psutil.STATUS_ZOMBIE)
            
            return {
                'total_processes': len(processes),
                'zombie_count': zombie_count
            }
        except:
            return {'total_processes': 0, 'zombie_count': 0}

# ============================================================================
# 🔗 IBKR 통합 연결 + 자동환전
# ============================================================================
class IBKRManager:
    """IBKR 통합 관리 + 자동환전 기능"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.ib = None
        self.connected = False
        self.account_id = None
        self.positions = {}
        self.balances = {}
        
        self.logger = logging.getLogger('IBKRManager')
    
    async def connect(self) -> bool:
        """IBKR 연결"""
        if not IBKR_AVAILABLE:
            self.logger.error("IBKR 모듈이 설치되지 않았습니다")
            return False
        
        try:
            self.ib = IB()
            await self.ib.connectAsync(
                self.config.IBKR_HOST,
                self.config.IBKR_PORT,
                self.config.IBKR_CLIENT_ID
            )
            
            if self.ib.isConnected():
                self.connected = True
                await self._update_account_info()
                self.logger.info("✅ IBKR 연결 성공")
                return True
            else:
                self.logger.error("IBKR 연결 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"IBKR 연결 오류: {e}")
            return False
    
    async def _update_account_info(self):
        """계좌 정보 업데이트"""
        try:
            # 계좌 정보
            accounts = self.ib.managedAccounts()
            if accounts:
                self.account_id = accounts[0]
            
            # 포지션 정보
            portfolio = self.ib.portfolio()
            self.positions = {}
            for pos in portfolio:
                if pos.position != 0:
                    self.positions[pos.contract.symbol] = {
                        'position': pos.position,
                        'avgCost': pos.avgCost,
                        'marketPrice': pos.marketPrice,
                        'unrealizedPNL': pos.unrealizedPNL,
                        'currency': pos.contract.currency
                    }
            
            # 잔고 정보
            account_values = self.ib.accountValues()
            self.balances = {}
            for av in account_values:
                if av.tag == 'CashBalance':
                    self.balances[av.currency] = float(av.value)
            
        except Exception as e:
            self.logger.error(f"계좌 정보 업데이트 실패: {e}")
    
    async def auto_currency_exchange(self, target_currency: str, required_amount: float) -> bool:
        """자동 환전 기능"""
        if not self.connected:
            return False
        
        try:
            # 현재 잔고 확인
            await self._update_account_info()
            
            current_balance = self.balances.get(target_currency, 0)
            
            if current_balance >= required_amount:
                self.logger.info(f"✅ {target_currency} 잔고 충분: {current_balance:,.2f}")
                return True
            
            # 환전 필요 금액 계산
            needed_amount = required_amount - current_balance
            
            # USD 잔고 확인
            usd_balance = self.balances.get('USD', 0)
            
            if target_currency == 'JPY':
                # 달러 → 엔화 환전
                exchange_rate = await self._get_exchange_rate('USD', 'JPY')
                usd_needed = needed_amount / exchange_rate
                
                if usd_balance >= usd_needed:
                    success = await self._execute_currency_exchange('USD', 'JPY', usd_needed)
                    if success:
                        self.logger.info(f"✅ 환전 완료: ${usd_needed:,.2f} → ¥{needed_amount:,.0f}")
                        return True
                
            elif target_currency == 'INR':
                # 달러 → 루피 환전
                exchange_rate = await self._get_exchange_rate('USD', 'INR')
                usd_needed = needed_amount / exchange_rate
                
                if usd_balance >= usd_needed:
                    success = await self._execute_currency_exchange('USD', 'INR', usd_needed)
                    if success:
                        self.logger.info(f"✅ 환전 완료: ${usd_needed:,.2f} → ₹{needed_amount:,.0f}")
                        return True
            
            self.logger.warning(f"⚠️ 환전 실패: {target_currency} {needed_amount:,.2f} 부족")
            return False
            
        except Exception as e:
            self.logger.error(f"자동 환전 실패: {e}")
            return False
    
    async def _get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """환율 조회"""
        try:
            # IBKR에서 환율 조회
            contract = Forex(f'{from_currency}{to_currency}')
            ticker = self.ib.reqMktData(contract)
            await asyncio.sleep(1)
            
            if ticker.marketPrice():
                rate = float(ticker.marketPrice())
                self.ib.cancelMktData(contract)
                return rate
            
            # 백업: 외부 API 사용
            api_key = os.getenv('EXCHANGE_RATE_API_KEY')
            if api_key:
                url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        data = await response.json()
                        return data['rates'][to_currency]
            
            # 기본값
            default_rates = {'USDJPY': 110.0, 'USDINR': 75.0}
            return default_rates.get(f'{from_currency}{to_currency}', 1.0)
            
        except Exception as e:
            self.logger.error(f"환율 조회 실패: {e}")
            return 1.0
    
    async def _execute_currency_exchange(self, from_currency: str, to_currency: str, amount: float) -> bool:
        """환전 실행"""
        try:
            contract = Forex(f'{from_currency}{to_currency}')
            order = MarketOrder('BUY', amount)
            
            trade = self.ib.placeOrder(contract, order)
            
            # 주문 완료 대기
            for _ in range(30):  # 30초 대기
                await asyncio.sleep(1)
                if trade.isDone():
                    break
            
            if trade.isDone() and trade.orderStatus.status == 'Filled':
                self.logger.info(f"✅ 환전 주문 완료: {from_currency} → {to_currency}")
                return True
            else:
                self.logger.error(f"❌ 환전 주문 실패: {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            self.logger.error(f"환전 실행 오류: {e}")
            return False
    
    async def emergency_sell_all(self) -> Dict[str, bool]:
        """응급 전량 매도"""
        if not self.connected:
            return {}
        
        self.logger.critical("🚨 응급 전량 매도 시작!")
        
        results = {}
        
        try:
            await self._update_account_info()
            
            for symbol, pos_info in self.positions.items():
                if pos_info['position'] > 0:  # 매수 포지션만
                    try:
                        # 계약 생성
                        if pos_info['currency'] == 'USD':
                            contract = Stock(symbol, 'SMART', 'USD')
                        elif pos_info['currency'] == 'JPY':
                            contract = Stock(symbol, 'TSE', 'JPY')
                        elif pos_info['currency'] == 'INR':
                            contract = Stock(symbol, 'NSE', 'INR')
                        else:
                            continue
                        
                        # 시장가 매도 주문
                        order = MarketOrder('SELL', abs(pos_info['position']))
                        trade = self.ib.placeOrder(contract, order)
                        
                        results[symbol] = True
                        self.logger.info(f"🚨 응급 매도: {symbol} {abs(pos_info['position'])}주")
                        
                    except Exception as e:
                        results[symbol] = False
                        self.logger.error(f"응급 매도 실패 {symbol}: {e}")
            
            self.logger.critical(f"🚨 응급 매도 완료: {len(results)}개 종목")
            return results
            
        except Exception as e:
            self.logger.error(f"응급 매도 전체 실패: {e}")
            return {}

# ============================================================================
# 📊 통합 포지션 관리자
# ============================================================================
@dataclass
class UnifiedPosition:
    """통합 포지션 정보"""
    strategy: str
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    currency: str
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_date: datetime
    last_updated: datetime

class UnifiedPositionManager:
    """통합 포지션 관리"""
    
    def __init__(self, config: CoreConfig, ibkr_manager: IBKRManager):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.positions: Dict[str, UnifiedPosition] = {}
        
        self.logger = logging.getLogger('PositionManager')
    
    async def update_all_positions(self):
        """모든 포지션 업데이트"""
        try:
            if self.ibkr_manager.connected:
                await self.ibkr_manager._update_account_info()
                
                # IBKR 포지션을 통합 포지션으로 변환
                for symbol, pos_info in self.ibkr_manager.positions.items():
                    
                    # 전략 추정 (심볼 기반)
                    strategy = self._estimate_strategy(symbol, pos_info['currency'])
                    
                    unified_pos = UnifiedPosition(
                        strategy=strategy,
                        symbol=symbol,
                        quantity=pos_info['position'],
                        avg_cost=pos_info['avgCost'],
                        current_price=pos_info['marketPrice'],
                        currency=pos_info['currency'],
                        unrealized_pnl=pos_info['unrealizedPNL'],
                        unrealized_pnl_pct=(pos_info['marketPrice'] - pos_info['avgCost']) / pos_info['avgCost'] * 100,
                        entry_date=datetime.now(),  # 실제로는 DB에서 로드
                        last_updated=datetime.now()
                    )
                    
                    self.positions[f"{strategy}_{symbol}"] = unified_pos
                
                self.logger.info(f"📊 포지션 업데이트: {len(self.positions)}개")
                
        except Exception as e:
            self.logger.error(f"포지션 업데이트 실패: {e}")
    
    def _estimate_strategy(self, symbol: str, currency: str) -> str:
        """심볼과 통화로 전략 추정"""
        if currency == 'USD':
            return 'US'
        elif currency == 'JPY':
            return 'JAPAN'
        elif currency == 'INR':
            return 'INDIA'
        elif currency == 'KRW':
            return 'CRYPTO'
        else:
            return 'UNKNOWN'
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """포트폴리오 요약"""
        summary = {
            'total_positions': len(self.positions),
            'by_strategy': {},
            'by_currency': {},
            'total_unrealized_pnl': 0,
            'profitable_positions': 0,
            'losing_positions': 0
        }
        
        for pos in self.positions.values():
            # 전략별 집계
            if pos.strategy not in summary['by_strategy']:
                summary['by_strategy'][pos.strategy] = {'count': 0, 'pnl': 0}
            summary['by_strategy'][pos.strategy]['count'] += 1
            summary['by_strategy'][pos.strategy]['pnl'] += pos.unrealized_pnl
            
            # 통화별 집계
            if pos.currency not in summary['by_currency']:
                summary['by_currency'][pos.currency] = {'count': 0, 'pnl': 0}
            summary['by_currency'][pos.currency]['count'] += 1
            summary['by_currency'][pos.currency]['pnl'] += pos.unrealized_pnl
            
            # 전체 집계
            summary['total_unrealized_pnl'] += pos.unrealized_pnl
            
            if pos.unrealized_pnl > 0:
                summary['profitable_positions'] += 1
            else:
                summary['losing_positions'] += 1
        
        return summary

# ============================================================================
# 🌐 네트워크 모니터링 시스템
# ============================================================================
class NetworkMonitor:
    """네트워크 연결 모니터링"""
    
    def __init__(self, config: CoreConfig, ibkr_manager: IBKRManager):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.monitoring = False
        self.connection_failures = 0
        self.last_check_time = None
        
        self.logger = logging.getLogger('NetworkMonitor')
    
    async def start_monitoring(self):
        """네트워크 모니터링 시작"""
        if not self.config.NETWORK_MONITORING:
            return
        
        self.monitoring = True
        self.logger.info("🌐 네트워크 모니터링 시작")
        
        while self.monitoring:
            try:
                await self._check_connections()
                await asyncio.sleep(self.config.NETWORK_CHECK_INTERVAL)
            except Exception as e:
                self.logger.error(f"네트워크 모니터링 오류: {e}")
                await asyncio.sleep(60)
    
    async def _check_connections(self):
        """연결 상태 체크"""
        current_time = time.time()
        
        # 인터넷 연결 체크
        internet_ok = await self._check_internet()
        
        # IBKR 연결 체크
        ibkr_ok = self.ibkr_manager.connected and self.ibkr_manager.ib.isConnected()
        
        # API 서버 체크
        api_ok = await self._check_api_servers()
        
        if not internet_ok or not ibkr_ok or not api_ok:
            self.connection_failures += 1
            
            # IBKR 없이 운영시 더 관대한 기준 적용
            if not self.ibkr_manager.connected and api_ok and internet_ok:
                # IBKR 없어도 API와 인터넷이 되면 경고만
                if self.connection_failures == 1:  # 첫 번째만 로그
                    self.logger.info(f"ℹ️ IBKR 미연결 상태로 운영 중 (암호화폐 전략만 사용)")
                self.connection_failures = 0  # 실패 카운트 리셋
                return
                
            self.logger.warning(f"⚠️ 연결 실패 {self.connection_failures}회: 인터넷={internet_ok}, IBKR={ibkr_ok}, API={api_ok}")
            
            # 연속 실패시 응급 조치 (더 엄격한 기준)
            if self.connection_failures >= 5 and self.config.NETWORK_DISCONNECT_SELL:
                self.logger.critical("🚨 네트워크 연결 실패로 응급 매도 실행!")
                await self.ibkr_manager.emergency_sell_all()
                self.monitoring = False
        else:
            if self.connection_failures > 0:
                self.logger.info("✅ 네트워크 연결 복구")
            self.connection_failures = 0
        
        self.last_check_time = current_time
    
    async def _check_internet(self) -> bool:
        """인터넷 연결 체크"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://www.google.com', timeout=10) as response:
                    return response.status == 200
        except:
            return False
    
    async def _check_api_servers(self) -> bool:
        """API 서버 연결 체크"""
        try:
            servers_to_check = [
                'https://api.upbit.com/v1/market/all',  # 업비트 API
                'https://query1.finance.yahoo.com',     # 야후 파이낸스
            ]
            
            success_count = 0
            for server in servers_to_check:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(server, timeout=5) as response:
                            if response.status == 200:
                                success_count += 1
                except:
                    continue
            
            # 최소 1개 서버라도 연결되면 OK
            return success_count > 0
        except:
            return False

# ============================================================================
# 🔔 통합 알림 시스템
# ============================================================================
class NotificationManager:
    """통합 알림 관리"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.telegram_session = None
        
        self.logger = logging.getLogger('NotificationManager')
    
    async def send_notification(self, message: str, priority: str = 'normal'):
        """통합 알림 전송"""
        try:
            # 우선순위별 이모지
            priority_emojis = {
                'emergency': '🚨',
                'warning': '⚠️',
                'info': 'ℹ️',
                'success': '✅',
                'normal': '📊'
            }
            
            emoji = priority_emojis.get(priority, '📊')
            formatted_message = f"{emoji} {message}"
            
            # 텔레그램 알림
            if self.config.TELEGRAM_ENABLED:
                await self._send_telegram(formatted_message, priority)
            
            # 로그에도 기록
            if priority == 'emergency':
                self.logger.critical(formatted_message)
            elif priority == 'warning':
                self.logger.warning(formatted_message)
            else:
                self.logger.info(formatted_message)
                
        except Exception as e:
            self.logger.error(f"알림 전송 실패: {e}")
    
    async def _send_telegram(self, message: str, priority: str):
        """텔레그램 메시지 전송"""
        if not self.config.TELEGRAM_BOT_TOKEN or not self.config.TELEGRAM_CHAT_ID:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            # 응급 상황시 알림음 설정
            disable_notification = priority not in ['emergency', 'warning']
            
            data = {
                'chat_id': self.config.TELEGRAM_CHAT_ID,
                'text': f"🏆 퀸트프로젝트 알림\n\n{message}",
                'disable_notification': disable_notification,
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.logger.debug("텔레그램 알림 전송 성공")
                    else:
                        self.logger.error(f"텔레그램 알림 실패: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"텔레그램 전송 오류: {e}")

# ============================================================================
# 📈 성과 추적 시스템
# ============================================================================
class PerformanceTracker:
    """통합 성과 추적"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.db_path = config.DB_PATH
        self._init_database()
        
        self.logger = logging.getLogger('PerformanceTracker')
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 거래 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT,
                    symbol TEXT,
                    action TEXT,
                    quantity REAL,
                    price REAL,
                    currency TEXT,
                    timestamp DATETIME,
                    profit_loss REAL,
                    profit_percent REAL,
                    fees REAL
                )
            ''')
            
            # 일일 성과 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date DATE PRIMARY KEY,
                    strategy TEXT,
                    total_value REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    daily_return REAL,
                    positions_count INTEGER
                )
            ''')
            
            # 시스템 로그 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    level TEXT,
                    component TEXT,
                    message TEXT,
                    details TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def record_trade(self, strategy: str, symbol: str, action: str, quantity: float, 
                    price: float, currency: str, profit_loss: float = 0, fees: float = 0):
        """거래 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            profit_percent = 0
            if action == 'SELL' and profit_loss != 0:
                profit_percent = (profit_loss / (quantity * price - profit_loss)) * 100
            
            cursor.execute('''
                INSERT INTO trades 
                (strategy, symbol, action, quantity, price, currency, timestamp, profit_loss, profit_percent, fees)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (strategy, symbol, action, quantity, price, currency, 
                  datetime.now().isoformat(), profit_loss, profit_percent, fees))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"거래 기록: {strategy} {symbol} {action} {quantity}")
            
        except Exception as e:
            self.logger.error(f"거래 기록 실패: {e}")
    
    def record_daily_performance(self, strategy: str, total_value: float, 
                               unrealized_pnl: float, realized_pnl: float, positions_count: int):
        """일일 성과 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            daily_return = (unrealized_pnl + realized_pnl) / total_value * 100 if total_value > 0 else 0
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_performance 
                (date, strategy, total_value, unrealized_pnl, realized_pnl, daily_return, positions_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (today.isoformat(), strategy, total_value, unrealized_pnl, 
                  realized_pnl, daily_return, positions_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"일일 성과 기록 실패: {e}")
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """성과 요약 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).date()
            
            # 기간별 수익률
            cursor.execute('''
                SELECT strategy, SUM(profit_loss) as total_profit, COUNT(*) as trade_count,
                       AVG(profit_percent) as avg_profit_pct, 
                       SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades
                FROM trades 
                WHERE date(timestamp) >= ? AND action = 'SELL'
                GROUP BY strategy
            ''', (start_date.isoformat(),))
            
            strategy_performance = {}
            for row in cursor.fetchall():
                strategy, total_profit, trade_count, avg_profit_pct, winning_trades = row
                win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0
                
                strategy_performance[strategy] = {
                    'total_profit': total_profit,
                    'trade_count': trade_count,
                    'avg_profit_pct': avg_profit_pct,
                    'win_rate': win_rate,
                    'winning_trades': winning_trades
                }
            
            conn.close()
            return strategy_performance
            
        except Exception as e:
            self.logger.error(f"성과 요약 조회 실패: {e}")
            return {}

# ============================================================================
# 🔄 자동 백업 시스템
# ============================================================================
class BackupManager:
    """자동 백업 관리"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.backup_path = Path(config.BACKUP_PATH)
        self.backup_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('BackupManager')
    
    async def perform_backup(self):
        """백업 실행"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.backup_path / f"backup_{timestamp}"
            backup_dir.mkdir(exist_ok=True)
            
            # 데이터베이스 백업
            if os.path.exists(self.config.DB_PATH):
                shutil.copy2(self.config.DB_PATH, backup_dir / "quant_core.db")
            
            # 전략별 데이터베이스 백업
            strategy_dbs = [
                './data/us_performance.db',
                './data/japan_performance.db',
                './data/india_performance.db',
                './data/crypto_performance.db'
            ]
            
            for db_path in strategy_dbs:
                if os.path.exists(db_path):
                    shutil.copy2(db_path, backup_dir / os.path.basename(db_path))
            
            # 설정 파일 백업
            config_files = ['.env', 'settings.yaml', 'positions.json']
            for config_file in config_files:
                if os.path.exists(config_file):
                    shutil.copy2(config_file, backup_dir / config_file)
            
            # 백업 압축
            shutil.make_archive(str(backup_dir), 'zip', str(backup_dir))
            shutil.rmtree(backup_dir)  # 원본 폴더 삭제
            
            self.logger.info(f"✅ 백업 완료: {backup_dir}.zip")
            
            # 오래된 백업 정리
            await self._cleanup_old_backups()
            
        except Exception as e:
            self.logger.error(f"백업 실패: {e}")
    
    async def _cleanup_old_backups(self):
        """오래된 백업 정리"""
        try:
            backup_files = list(self.backup_path.glob("backup_*.zip"))
            
            # 날짜순 정렬
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 최근 30개만 유지
            for old_backup in backup_files[30:]:
                old_backup.unlink()
                self.logger.info(f"오래된 백업 삭제: {old_backup.name}")
                
        except Exception as e:
            self.logger.error(f"백업 정리 실패: {e}")

# ============================================================================
# 🏆 퀸트프로젝트 통합 코어 시스템
# ============================================================================
class QuantProjectCore:
    """퀸트프로젝트 통합 코어 시스템"""
    
    def __init__(self):
        # 설정 로드
        self.config = CoreConfig()
        
        # 로깅 설정 (가장 먼저)
        self._setup_logging()
        
        # 로거 초기화 (로깅 설정 직후)
        self.logger = logging.getLogger('QuantCore')
        
        # 핵심 컴포넌트 초기화
        self.emergency_detector = EmergencyErrorDetector(self.config)
        self.ibkr_manager = IBKRManager(self.config)
        self.position_manager = UnifiedPositionManager(self.config, self.ibkr_manager)
        self.network_monitor = NetworkMonitor(self.config, self.ibkr_manager)
        self.notification_manager = NotificationManager(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        self.backup_manager = BackupManager(self.config)
        
        # 전략 인스턴스
        self.strategies = {}
        self._init_strategies()
        
        # 시스템 상태
        self.running = False
        self.start_time = None
    
    def _setup_logging(self):
        """로깅 설정"""
        log_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # 파일 핸들러
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'quant_core.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    def _init_strategies(self):
        """전략 초기화"""
        try:
            # 미국 전략
            if self.config.US_ENABLED and US_AVAILABLE:
                self.strategies['US'] = USStrategy()
                self.logger.info("✅ 미국 전략 초기화 완료")
            
            # 일본 전략
            if self.config.JAPAN_ENABLED and JAPAN_AVAILABLE:
                self.strategies['JAPAN'] = JapanStrategy()
                self.logger.info("✅ 일본 전략 초기화 완료")
            
            # 인도 전략
            if self.config.INDIA_ENABLED and INDIA_AVAILABLE:
                self.strategies['INDIA'] = IndiaStrategy()
                self.logger.info("✅ 인도 전략 초기화 완료")
            
            # 암호화폐 전략
            if self.config.CRYPTO_ENABLED and CRYPTO_AVAILABLE:
                self.strategies['CRYPTO'] = CryptoStrategy()
                self.logger.info("✅ 암호화폐 전략 초기화 완료")
            
            if not self.strategies:
                self.logger.warning("⚠️ 활성화된 전략이 없습니다")
                
        except Exception as e:
            self.logger.error(f"전략 초기화 실패: {e}")
    
    async def start_system(self):
        """시스템 시작"""
        try:
            self.logger.info("🏆 퀸트프로젝트 통합 시스템 시작!")
            self.start_time = datetime.now()
            self.running = True
            
            # IBKR 연결
            if IBKR_AVAILABLE:
                await self.ibkr_manager.connect()
            
            # 시작 알림
            await self.notification_manager.send_notification(
                f"🚀 퀸트프로젝트 시스템 시작\n"
                f"활성 전략: {', '.join(self.strategies.keys())}\n"
                f"IBKR 연결: {'✅' if self.ibkr_manager.connected else '❌'}\n"
                f"포트폴리오: {self.config.TOTAL_PORTFOLIO_VALUE:,.0f}원",
                'success'
            )
            
            # 백그라운드 태스크 시작
            tasks = [
                asyncio.create_task(self._main_trading_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self.network_monitor.start_monitoring()),
                asyncio.create_task(self._backup_loop())
            ]
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"시스템 시작 실패: {e}")
            await self.emergency_shutdown(f"시스템 시작 실패: {e}")
    
    async def _main_trading_loop(self):
        """메인 거래 루프"""
        while self.running:
            try:
                # 시스템 건강 상태 체크
                health_status = self.emergency_detector.check_system_health()
                
                if health_status['emergency_needed']:
                    await self.emergency_shutdown("시스템 건강 상태 위험")
                    break
                
                # 각 전략 실행 (요일별)
                current_weekday = datetime.now().weekday()
                
                for strategy_name, strategy_instance in self.strategies.items():
                    try:
                        if self._should_run_strategy(strategy_name, current_weekday):
                            await self._execute_strategy(strategy_name, strategy_instance)
                    except Exception as e:
                        error_critical = self.emergency_detector.record_error(
                            f"{strategy_name}_error", str(e), critical=True
                        )
                        if error_critical:
                            await self.emergency_shutdown(f"{strategy_name} 전략 치명적 오류")
                            break
                
                # 포지션 업데이트
                await self.position_manager.update_all_positions()
                
                # 1시간 대기
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"메인 루프 오류: {e}")
                await asyncio.sleep(60)
    
    def _should_run_strategy(self, strategy_name: str, weekday: int) -> bool:
        """전략 실행 여부 판단"""
        # 월요일(0), 화요일(1), 수요일(2), 목요일(3), 금요일(4)
        
        if strategy_name == 'US':
            return weekday in [1, 3]  # 화목
        elif strategy_name == 'JAPAN':
            return weekday in [1, 3]  # 화목
        elif strategy_name == 'INDIA':
            return weekday == 2  # 수요일
        elif strategy_name == 'CRYPTO':
            return weekday in [0, 4]  # 월금
        
        return False
    
    async def _execute_strategy(self, strategy_name: str, strategy_instance):
        """개별 전략 실행"""
        try:
            self.logger.info(f"🎯 {strategy_name} 전략 실행 시작")
            
            # 전략별 필요 통화 환전
            if strategy_name == 'JAPAN':
                await self.ibkr_manager.auto_currency_exchange('JPY', 10000000)  # 1천만엔
            elif strategy_name == 'INDIA':
                await self.ibkr_manager.auto_currency_exchange('INR', 7500000)   # 750만루피
            
            # 전략 실행 (각 전략의 메인 함수 호출)
            if hasattr(strategy_instance, 'run_strategy'):
                result = await strategy_instance.run_strategy()
            elif hasattr(strategy_instance, 'execute_legendary_strategy'):
                result = await strategy_instance.execute_legendary_strategy()
            else:
                self.logger.warning(f"{strategy_name} 전략 실행 메서드를 찾을 수 없음")
                return
            
            self.logger.info(f"✅ {strategy_name} 전략 실행 완료")
            
            # 성과 기록
            if result:
                # 결과에 따른 성과 기록 로직
                pass
                
        except Exception as e:
            self.logger.error(f"{strategy_name} 전략 실행 실패: {e}")
            raise
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.running:
            try:
                # 포지션 모니터링
                portfolio_summary = self.position_manager.get_portfolio_summary()
                
                # 위험 상황 체크
                total_loss_pct = (portfolio_summary['total_unrealized_pnl'] / 
                                 self.config.TOTAL_PORTFOLIO_VALUE * 100)
                
                if total_loss_pct < -self.config.MAX_PORTFOLIO_RISK * 100:
                    await self.notification_manager.send_notification(
                        f"🚨 포트폴리오 손실 한계 초과!\n"
                        f"현재 손실: {total_loss_pct:.2f}%\n"
                        f"한계: {self.config.MAX_PORTFOLIO_RISK * 100:.1f}%",
                        'emergency'
                    )
                
                # 주기적 상태 보고 (6시간마다)
                if datetime.now().hour % 6 == 0 and datetime.now().minute < 5:
                    await self._send_status_report(portfolio_summary)
                
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(60)
    
    async def _backup_loop(self):
        """백업 루프"""
        while self.running:
            try:
                # 매일 새벽 3시에 백업
                now = datetime.now()
                if now.hour == 3 and now.minute < 10:
                    await self.backup_manager.perform_backup()
                    await asyncio.sleep(600)  # 10분 대기
                
                await asyncio.sleep(300)  # 5분마다 체크
                
            except Exception as e:
                self.logger.error(f"백업 루프 오류: {e}")
                await asyncio.sleep(3600)  # 1시간 대기
    
    async def _send_status_report(self, portfolio_summary: Dict):
        """상태 보고서 전송"""
        try:
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            report = (
                f"📊 퀸트프로젝트 상태 보고\n\n"
                f"🕐 가동시간: {uptime}\n"
                f"💼 총 포지션: {portfolio_summary['total_positions']}개\n"
                f"💰 미실현 손익: {portfolio_summary['total_unrealized_pnl']:+,.0f}원\n"
                f"📈 수익 포지션: {portfolio_summary['profitable_positions']}개\n"
                f"📉 손실 포지션: {portfolio_summary['losing_positions']}개\n\n"
                f"전략별 현황:\n"
            )
            
            for strategy, data in portfolio_summary['by_strategy'].items():
                report += f"  {strategy}: {data['count']}개 ({data['pnl']:+,.0f}원)\n"
            
            await self.notification_manager.send_notification(report, 'info')
            
        except Exception as e:
            self.logger.error(f"상태 보고서 전송 실패: {e}")
    
    async def graceful_shutdown(self):
        """정상 종료"""
        try:
            self.logger.info("🛑 시스템 정상 종료 시작")
            
            # 종료 알림
            await self.notification_manager.send_notification(
                f"🛑 시스템 정상 종료\n"
                f"가동시간: {datetime.now() - self.start_time if self.start_time else '알수없음'}",
                'info'
            )
            
            # 네트워크 모니터링 중지
            self.network_monitor.monitoring = False
            
            # 최종 백업
            await self.backup_manager.perform_backup()
            
            # IBKR 연결 해제
            if self.ibkr_manager.connected:
                await self.ibkr_manager.ib.disconnectAsync()
            
            self.running = False
            self.logger.info("✅ 시스템 정상 종료 완료")
            
        except Exception as e:
            self.logger.error(f"정상 종료 실패: {e}")

# ============================================================================
# 🎮 편의 함수들
# ============================================================================
async def get_system_status():
    """시스템 상태 조회"""
    core = QuantProjectCore()
    await core.ibkr_manager.connect()
    await core.position_manager.update_all_positions()
    
    summary = core.position_manager.get_portfolio_summary()
    
    return {
        'strategies': list(core.strategies.keys()),
        'ibkr_connected': core.ibkr_manager.connected,
        'total_positions': summary['total_positions'],
        'total_unrealized_pnl': summary['total_unrealized_pnl'],
        'by_strategy': summary['by_strategy']
    }

async def emergency_sell_all():
    """응급 전량 매도"""
    core = QuantProjectCore()
    await core.ibkr_manager.connect()
    
    if core.ibkr_manager.connected:
        results = await core.ibkr_manager.emergency_sell_all()
        return results
    else:
        return {}

# ============================================================================
# 🏁 메인 실행부
# ============================================================================
async def main():
    """메인 실행 함수"""
    # 신호 핸들러 설정
    def signal_handler(signum, frame):
        print("\n🛑 종료 신호 수신, 정상 종료 중...")
        asyncio.create_task(core.graceful_shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 코어 시스템 생성
    core = QuantProjectCore()
    
    try:
        print("🏆" + "="*70)
        print("🏆 퀸트프로젝트 통합 코어 시스템 v1.1.0")
        print("🏆" + "="*70)
        print("✨ 4대 전략 통합 관리")
        print("✨ IBKR 자동 환전")
        print("✨ 네트워크 모니터링")
        print("✨ 응급 오류 감지")
        print("✨ 통합 포지션 관리")
        print("✨ 실시간 알림")
        print("✨ 자동 백업")
        print("🏆" + "="*70)
        
        # 시스템 시작
        await core.start_system()
        
    except KeyboardInterrupt:
        print("\n👋 사용자 중단")
        await core.graceful_shutdown()
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")
        await core.emergency_shutdown(f"시스템 오류: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 퀸트프로젝트 종료")
        sys.exit(0)
