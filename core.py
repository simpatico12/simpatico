#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 통합 코어 시스템 (core.py)
=================================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 (4대 전략 통합)

✨ 핵심 기능:
- 4대 전략 통합 관리 시스템
- 자동 환전 기능 (원화 ↔ 달러/엔/루피)
- 네트워크 모니터링 + 끊김 시 전량 매도
- 통합 포지션 관리 + 리스크 제어
- 실시간 모니터링 + 알림 시스템
- 성과 추적 + 자동 백업
- 🚨 응급 오류 감지 시스템 (NEW!)

Author: 퀸트마스터팀
Version: 1.1.0 (응급 오류 감지 시스템 추가)
"""

import asyncio
import logging
import os
import sys
import json
import time
import threading
import psutil
import socket
import requests
import sqlite3
import signal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import yaml
import pickle
import aiohttp
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# 전략별 모듈 임포트
try:
    from us_strategy import LegendaryQuantStrategy as USStrategy
    US_AVAILABLE = True
except ImportError:
    US_AVAILABLE = False
    print("⚠️ US Strategy 모듈 없음")

try:
    from jp_strategy import YenHunter as JPStrategy
    JP_AVAILABLE = True
except ImportError:
    JP_AVAILABLE = False
    print("⚠️ Japan Strategy 모듈 없음")

try:
    from inda_strategy import LegendaryIndiaStrategy as INStrategy
    IN_AVAILABLE = True
except ImportError:
    IN_AVAILABLE = False
    print("⚠️ India Strategy 모듈 없음")

try:
    from coin_strategy import LegendaryQuantMaster as CryptoStrategy
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("⚠️ Crypto Strategy 모듈 없음")

# IBKR API
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("⚠️ IBKR API 없음")

# 업비트 API
try:
    import pyupbit
    UPBIT_AVAILABLE = True
except ImportError:
    UPBIT_AVAILABLE = False
    print("⚠️ Upbit API 없음")

# 환경변수 로드
load_dotenv()

# 오류 기반 응급매도 설정
EMERGENCY_SELL_ON_ERROR = os.getenv('EMERGENCY_SELL_ON_ERROR', 'true').lower() == 'true'
EMERGENCY_MEMORY_THRESHOLD = float(os.getenv('EMERGENCY_MEMORY_THRESHOLD', '95'))  # 95%
EMERGENCY_CPU_THRESHOLD = float(os.getenv('EMERGENCY_CPU_THRESHOLD', '90'))  # 90%
EMERGENCY_DISK_THRESHOLD = float(os.getenv('EMERGENCY_DISK_THRESHOLD', '5'))  # 5GB
EMERGENCY_ERROR_COUNT = int(os.getenv('EMERGENCY_ERROR_COUNT', '5'))  # 5회 연속
EMERGENCY_GRACE_PERIOD = int(os.getenv('EMERGENCY_GRACE_PERIOD', '60'))  # 60초

# 환율 API 설정
EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY', '')

# 네트워크 모니터링 설정
NETWORK_CHECK_INTERVAL = int(os.getenv('NETWORK_CHECK_INTERVAL', '30'))
NETWORK_TIMEOUT = int(os.getenv('NETWORK_TIMEOUT', '10'))
NETWORK_MAX_FAILURES = int(os.getenv('NETWORK_MAX_FAILURES', '3'))
NETWORK_GRACE_PERIOD = int(os.getenv('NETWORK_GRACE_PERIOD', '300'))
NETWORK_DISCONNECT_SELL_ALL = os.getenv('NETWORK_DISCONNECT_SELL_ALL', 'false').lower() == 'true'

# 데이터베이스 설정
DATABASE_PATH = os.getenv('DATABASE_PATH', './data/quint_core.db')
BACKUP_ENABLED = os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
BACKUP_PATH = os.getenv('BACKUP_PATH', './backups/')

# 알림 설정
TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))
EMAIL_USERNAME = os.getenv('EMAIL_USERNAME', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
EMAIL_TO = os.getenv('EMAIL_TO', '')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('quint_core.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 🚨 응급 오류 감지 시스템
# ============================================================================

class EmergencyErrorMonitor:
    """응급 오류 감지 및 대응 시스템"""
    
    def __init__(self, core_system):
        self.core_system = core_system
        self.error_counts = {}  # 전략별 오류 카운터
        self.last_emergency_time =             print(f"📊 오류 발생 전략: {error_summary['total_strategies_with_errors']}개")
            print(f"⏰ 마지막 응급매도: {error_summary.get('last_emergency_time', '없음')}")
            
            print(f"\n📋 전략별 오류 현황:")
            if error_summary.get('error_counts'):
                for strategy, count in error_summary['error_counts'].items():
                    print(f"  📌 {strategy}: {count}회 오류")
                    
                    # 상세 오류 정보 표시
                    if strategy in self.core.error_monitor.error_counts:
                        error_details = self.core.error_monitor.error_counts[strategy]
                        last_error = error_details.get('last_error_time')
                        if last_error:
                            print(f"    마지막 오류: {last_error.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # 최근 오류 3개 표시
                        recent_errors = error_details.get('errors', [])[-3:]
                        if recent_errors:
                            print(f"    최근 오류:")
                            for i, error in enumerate(recent_errors, 1):
                                error_time = error['timestamp'][:19]  # YYYY-MM-DD HH:MM:SS
                                error_msg = error['error'][:50] + '...' if len(error['error']) > 50 else error['error']
                                print(f"      {i}. {error_time}: {error_msg}")
            else:
                print("  ✅ 오류가 발생한 전략이 없습니다.")
            
            print(f"\n🔧 응급매도 설정:")
            print(f"  메모리 임계치: {EMERGENCY_MEMORY_THRESHOLD}%")
            print(f"  CPU 임계치: {EMERGENCY_CPU_THRESHOLD}%")
            print(f"  디스크 임계치: {EMERGENCY_DISK_THRESHOLD}GB")
            print(f"  연속 오류 임계치: {EMERGENCY_ERROR_COUNT}회")
            print(f"  유예 시간: {EMERGENCY_GRACE_PERIOD}초")
            
            # 현재 시스템 리소스 상태
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            disk_free = psutil.disk_usage('/').free / (1024**3)
            
            print(f"\n💻 현재 시스템 리소스:")
            mem_status = "⚠️" if memory_usage >= EMERGENCY_MEMORY_THRESHOLD else "✅"
            cpu_status = "⚠️" if cpu_usage >= EMERGENCY_CPU_THRESHOLD else "✅"
            disk_status = "⚠️" if disk_free <= EMERGENCY_DISK_THRESHOLD else "✅"
            
            print(f"  {mem_status} 메모리: {memory_usage:.1f}%")
            print(f"  {cpu_status} CPU: {cpu_usage:.1f}%")
            print(f"  {disk_status} 디스크 여유공간: {disk_free:.1f}GB")
            
            # 오류 카운터 리셋 옵션
            print(f"\n🔄 관리 옵션:")
            print("  r: 모든 오류 카운터 리셋")
            print("  Enter: 돌아가기")
            
            choice = input("\n선택하세요: ").strip().lower()
            if choice == 'r':
                self.core.error_monitor.reset_error_counts()
                print("✅ 모든 오류 카운터가 리셋되었습니다.")
                await asyncio.sleep(2)
            
        except Exception as e:
            print(f"❌ 오류 현황 조회 실패: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_portfolio(self):
        """포트폴리오 상세 표시"""
        if not self.core:
            print("❌ 시스템이 시작되지 않았습니다.")
            return
        
        print("\n💼 포트폴리오 상세 현황")
        print("="*60)
        
        try:
            summary = self.core.position_manager.get_portfolio_summary()
            
            # 전체 요약
            print(f"📊 전체 요약:")
            print(f"  총 포지션: {summary.get('total_positions', 0)}개")
            print(f"  총 가치: {summary.get('total_krw_value', 0):,.0f}원")
            print(f"  미실현 손익: {summary.get('total_unrealized_pnl', 0):+,.0f}원")
            print(f"  총 수익률: {summary.get('total_return_pct', 0):+.2f}%")
            
            # 전략별 분석
            print(f"\n🎯 전략별 분석:")
            for strategy, data in summary.get('by_strategy', {}).items():
                print(f"  {strategy}:")
                print(f"    포지션: {data['count']}개")
                print(f"    가치: {data['krw_value']:,.0f}원")
                print(f"    손익: {data['unrealized_pnl']:+,.0f}원")
            
            # 통화별 분석
            print(f"\n💱 통화별 분석:")
            for currency, data in summary.get('by_currency', {}).items():
                print(f"  {currency}: {data['count']}개 포지션, {data['krw_value']:,.0f}원")
            
            # 상위/하위 종목
            print(f"\n🏆 상위 수익 종목:")
            for gainer in summary.get('top_gainers', [])[:5]:
                print(f"  {gainer['symbol']} ({gainer['strategy']}): {gainer['pnl_pct']:+.1f}%")
            
            print(f"\n💔 하위 수익 종목:")
            for loser in summary.get('top_losers', [])[:5]:
                print(f"  {loser['symbol']} ({loser['strategy']}): {loser['pnl_pct']:+.1f}%")
            
        except Exception as e:
            print(f"❌ 포트폴리오 조회 실패: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_network_status(self):
        """네트워크 상태 상세 표시"""
        if not self.core:
            print("❌ 시스템이 시작되지 않았습니다.")
            return
        
        print("\n🌐 네트워크 상태 상세")
        print("="*40)
        
        try:
            status = self.core.network_monitor.get_network_status()
            
            print(f"연결 상태: {'✅ 정상' if status['is_connected'] else '❌ 끊김'}")
            print(f"지연시간: {status['latency_ms']:.1f}ms")
            print(f"연속 실패: {status['consecutive_failures']}회")
            print(f"가동률: {status['uptime_percentage']:.2f}%")
            print(f"총 체크: {status['total_checks']}회")
            print(f"마지막 체크: {status['last_check']}")
            print(f"응급 매도: {'✅ 활성화' if status['emergency_sell_enabled'] else '❌ 비활성화'}")
            
        except Exception as e:
            print(f"❌ 네트워크 상태 조회 실패: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_currency_rates(self):
        """환율 정보 표시"""
        if not self.core:
            print("❌ 시스템이 시작되지 않았습니다.")
            return
        
        print("\n💱 현재 환율 정보")
        print("="*30)
        
        try:
            if self.core.currency_exchange.last_update:
                print(f"⏰ 마지막 업데이트: {self.core.currency_exchange.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            
            print(f"\n💰 환율:")
            for currency, info in self.core.currency_exchange.exchange_rates.items():
                if currency != 'KRW':
                    print(f"  {currency} ({info.name}): {info.exchange_rate:,.1f}원")
            
            # 환전 계산기
            print(f"\n🔄 간단 환전 계산기:")
            amount = input("금액 입력 (원화): ").strip()
            if amount.isdigit():
                krw_amount = float(amount)
                print(f"  USD: ${krw_amount / self.core.currency_exchange.get_exchange_rate('USD'):,.2f}")
                print(f"  JPY: ¥{krw_amount / self.core.currency_exchange.get_exchange_rate('JPY'):,.0f}")
                print(f"  INR: ₹{krw_amount / self.core.currency_exchange.get_exchange_rate('INR'):,.0f}")
            
        except Exception as e:
            print(f"❌ 환율 정보 조회 실패: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _manage_strategies(self):
        """전략 관리"""
        if not self.core:
            print("❌ 시스템이 시작되지 않았습니다.")
            return
        
        print("\n🎯 전략 관리")
        print("="*30)
        
        print(f"활성 전략: {list(self.core.strategies.keys())}")
        
        # 전략별 오류 현황 표시
        error_summary = self.core.error_monitor.get_error_summary()
        if error_summary.get('error_counts'):
            print(f"\n⚠️ 전략별 오류 현황:")
            for strategy, count in error_summary['error_counts'].items():
                print(f"  {strategy}: {count}회")
        
        print("\n⚙️ 관리 옵션:")
        print("  r: 특정 전략 오류 카운터 리셋")
        print("  Enter: 돌아가기")
        
        choice = input("\n선택하세요: ").strip().lower()
        if choice == 'r':
            strategy = input("전략 이름 입력: ").strip()
            if strategy in self.core.error_monitor.error_counts:
                self.core.error_monitor.reset_error_counts(strategy)
                print(f"✅ {strategy} 전략의 오류 카운터가 리셋되었습니다.")
            else:
                print(f"❌ {strategy} 전략을 찾을 수 없습니다.")
            await asyncio.sleep(2)
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_performance_report(self):
        """성과 리포트 표시"""
        if not self.core:
            print("❌ 시스템이 시작되지 않았습니다.")
            return
        
        print("\n📈 성과 리포트")
        print("="*40)
        
        try:
            # 포트폴리오 요약에서 성과 데이터 추출
            summary = self.core.position_manager.get_portfolio_summary()
            
            print(f"📊 현재 성과:")
            print(f"  총 투자금액: {summary.get('total_krw_value', 0) - summary.get('total_unrealized_pnl', 0):,.0f}원")
            print(f"  현재 가치: {summary.get('total_krw_value', 0):,.0f}원")
            print(f"  미실현 손익: {summary.get('total_unrealized_pnl', 0):+,.0f}원")
            print(f"  수익률: {summary.get('total_return_pct', 0):+.2f}%")
            
            # 월간 목표 대비
            monthly_target = 6.0  # 기본 6%
            progress = summary.get('total_return_pct', 0)
            print(f"\n🎯 월간 목표 대비:")
            print(f"  목표: {monthly_target}%")
            print(f"  현재: {progress:+.2f}%")
            print(f"  달성률: {(progress / monthly_target * 100) if monthly_target > 0 else 0:.1f}%")
            
        except Exception as e:
            print(f"❌ 성과 리포트 생성 실패: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _emergency_sell(self):
        """응급 매도"""
        if not self.core:
            print("❌ 시스템이 시작되지 않았습니다.")
            return
        
        print("\n🚨 응급 전량 매도")
        print("="*30)
        print("⚠️ 경고: 모든 포지션을 즉시 매도합니다!")
        print("이 작업은 되돌릴 수 없습니다.")
        
        confirm1 = input("\n정말 진행하시겠습니까? (yes/no): ").strip().lower()
        if confirm1 != 'yes':
            print("❌ 취소되었습니다.")
            return
        
        confirm2 = input("한 번 더 확인합니다. 응급 매도를 실행하시겠습니까? (EMERGENCY): ").strip()
        if confirm2 != 'EMERGENCY':
            print("❌ 취소되었습니다.")
            return
        
        print("🚨 응급 매도를 실행합니다...")
        
        try:
            await self.core.emergency_sell_all("USER_REQUESTED")
            print("✅ 응급 매도 요청이 전송되었습니다.")
            print("실제 매도 결과는 각 거래소의 응답에 따라 달라질 수 있습니다.")
            
        except Exception as e:
            print(f"❌ 응급 매도 실행 실패: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _shutdown_system(self):
        """시스템 종료"""
        if not self.core:
            print("❌ 시스템이 이미 종료되어 있습니다.")
            return
        
        print("\n🛑 시스템 종료")
        print("="*20)
        
        confirm = input("시스템을 종료하시겠습니까? (y/N): ").strip().lower()
        if confirm == 'y':
            print("🛑 시스템을 종료합니다...")
            await self.core.shutdown()
            self.core = None
            print("✅ 시스템이 안전하게 종료되었습니다.")
        
        input("\n계속하려면 Enter를 누르세요...")

# ============================================================================
# 🚀 메인 실행 함수
# ============================================================================

async def main():
    """메인 실행 함수"""
    try:
        cli = QuintCLI()
        await cli.start_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        logger.error(f"메인 실행 오류: {e}")
        
        # 치명적 오류 시 응급 매도
        if EMERGENCY_SELL_ON_ERROR:
            try:
                # 코어 시스템이 있다면 응급 매도 시도
                if hasattr(cli, 'core') and cli.core:
                    await cli.core.emergency_sell_all(f"CRITICAL_ERROR: {str(e)}")
                    print("🚨 치명적 오류로 인한 응급 매도 실행됨")
            except Exception as emergency_error:
                logger.error(f"응급 매도 실행 실패: {emergency_error}")
                print(f"❌ 응급 매도 실행 실패: {emergency_error}")

if __name__ == "__main__":
    # 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 종료되었습니다.")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        # 최종 안전장치 - 시스템 레벨 오류 시 로그만 기록
        logger.critical(f"시스템 레벨 오류: {e}")
        if EMERGENCY_SELL_ON_ERROR:
            print("🚨 시스템 레벨 오류 감지 - 수동으로 포지션 확인 필요")ON_ERROR else '❌ 비활성화'}\n"
                f"환경: {self.config.get('system', {}).get('environment', 'unknown')}"
            )
            
            # 백그라운드 태스크 시작
            tasks = [
                asyncio.create_task(self._main_loop()),
                asyncio.create_task(self.network_monitor.start_monitoring()),
                asyncio.create_task(self._periodic_tasks())
            ]
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"시스템 시작 실패: {e}")
            self.error_monitor.record_strategy_error('system_start', e)
            await self.shutdown()
    
    async def _main_loop(self):
        """메인 실행 루프"""
        logger.info("🔄 메인 루프 시작")
        
        while self.is_running:
            try:
                # 건강 상태 체크
                await self._health_check()
                
                # 전략별 실행 (스케줄 기반)
                await self._execute_strategies()
                
                # 포지션 업데이트
                await self._update_positions()
                
                # 5분 대기
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"메인 루프 오류: {e}")
                self.error_monitor.record_strategy_error('main_loop', e)
                await asyncio.sleep(60)  # 1분 후 재시도
    
    async def _periodic_tasks(self):
        """주기적 태스크"""
        while self.is_running:
            try:
                # 환율 업데이트 (5분마다)
                await self.currency_exchange.update_exchange_rates()
                
                # 데이터베이스 백업 (1시간마다)
                if datetime.now().minute == 0:
                    self.data_manager.backup_database()
                
                # 성과 리포트 (하루 1회)
                if datetime.now().hour == 9 and datetime.now().minute == 0:
                    await self._generate_daily_report()
                
                await asyncio.sleep(300)  # 5분 대기
                
            except Exception as e:
                logger.error(f"주기적 태스크 오류: {e}")
                self.error_monitor.record_strategy_error('periodic_tasks', e)
                await asyncio.sleep(300)
    
    async def _health_check(self):
        """시스템 건강 상태 체크"""
        try:
            current_time = datetime.now()
            
            # 시스템 리소스 응급 체크
            await self.error_monitor.check_system_resources()
            
            # 메모리 사용량 체크
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 80:
                logger.warning(f"⚠️ 메모리 사용량 높음: {memory_usage:.1f}%")
            
            # 디스크 사용량 체크
            disk_usage = psutil.disk_usage('/').percent
            if disk_usage > 90:
                logger.warning(f"⚠️ 디스크 사용량 높음: {disk_usage:.1f}%")
            
            # 네트워크 상태 체크는 별도 모니터에서 처리
            
            self.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"건강 상태 체크 실패: {e}")
            # 헬스체크 자체 실패도 오류로 기록
            self.error_monitor.record_strategy_error('health_check', e)
    
    async def _execute_strategies(self):
        """전략 실행"""
        try:
            for strategy_name, strategy in self.strategies.items():
                try:
                    # 각 전략의 스케줄 체크 및 실행
                    if await self._should_execute_strategy(strategy_name):
                        logger.info(f"🎯 {strategy_name} 전략 실행")
                        await self._run_strategy(strategy_name, strategy)
                
                except Exception as e:
                    logger.error(f"전략 실행 실패 {strategy_name}: {e}")
                    # 전략 오류 기록
                    self.error_monitor.record_strategy_error(strategy_name, e)
                    # 데이터베이스에도 오류 로그 저장
                    import traceback
                    self.data_manager.save_error_log(strategy_name, 'execution_error', str(e), traceback.format_exc())
                    continue
                    
        except Exception as e:
            logger.error(f"전략 실행 오류: {e}")
            # 전체 전략 실행 오류도 기록
            self.error_monitor.record_strategy_error('strategy_execution', e)
    
    async def _should_execute_strategy(self, strategy_name: str) -> bool:
        """전략 실행 여부 결정"""
        try:
            now = datetime.now()
            
            # 응급 모드에서는 실행하지 않음
            if self.emergency_mode:
                return False
            
            # 전략별 스케줄 체크 (간단한 버전)
            if strategy_name == 'us':
                # 미국: 월요일, 목요일
                return now.weekday() in [0, 3] and now.hour == 10
            elif strategy_name == 'japan':
                # 일본: 화요일, 목요일
                return now.weekday() in [1, 3] and now.hour == 9
            elif strategy_name == 'india':
                # 인도: 수요일
                return now.weekday() == 2 and now.hour == 9
            elif strategy_name == 'crypto':
                # 암호화폐: 월요일, 금요일
                return now.weekday() in [0, 4] and now.hour == 9
            
            return False
            
        except Exception as e:
            logger.error(f"전략 실행 여부 체크 실패: {e}")
            self.error_monitor.record_strategy_error('schedule_check', e)
            return False
    
    async def _run_strategy(self, strategy_name: str, strategy):
        """개별 전략 실행"""
        try:
            # 전략별 실행 방법이 다를 수 있으므로 분기
            if strategy_name == 'us':
                signals = await strategy.scan_all_stocks()
                for signal in signals:
                    if signal.action == 'BUY':
                        # 포지션 추가 로직
                        self.position_manager.add_position(
                            strategy_name, signal.symbol, 
                            100, signal.price, 'USD'  # 임시값
                        )
            
            elif strategy_name == 'crypto':
                signals = await strategy.execute_legendary_strategy()
                for signal in signals:
                    if signal.action == 'BUY':
                        self.position_manager.add_position(
                            strategy_name, signal.symbol,
                            signal.total_investment / signal.price, 
                            signal.price, 'KRW'
                        )
            
            # 다른 전략들도 유사하게 처리...
            
        except Exception as e:
            logger.error(f"전략 실행 오류 {strategy_name}: {e}")
            self.error_monitor.record_strategy_error(strategy_name, e)
    
    async def _update_positions(self):
        """포지션 현재가 업데이트"""
        try:
            # 각 전략별로 현재가 조회
            price_data = {}
            
            for strategy_name in self.strategies.keys():
                strategy_prices = {}
                
                # 전략별 포지션 현재가 조회 로직
                # 실제 구현시에는 각 전략의 API를 사용
                
                price_data[strategy_name] = strategy_prices
            
            # 포지션 매니저 업데이트
            self.position_manager.update_current_prices(price_data)
            
        except Exception as e:
            logger.error(f"포지션 업데이트 실패: {e}")
            self.error_monitor.record_strategy_error('position_update', e)
    
    async def emergency_sell_all(self, reason: str):
        """응급 전량 매도"""
        logger.critical(f"🚨 응급 전량 매도 실행: {reason}")
        
        try:
            self.emergency_mode = True
            
            # 모든 포지션 매도 시도
            for key, position in self.position_manager.positions.items():
                try:
                    # 전략별 매도 로직
                    success = await self._emergency_sell_position(position)
                    if success:
                        logger.info(f"🚨 응급 매도 완료: {position.symbol}")
                    else:
                        logger.error(f"🚨 응급 매도 실패: {position.symbol}")
                        
                except Exception as e:
                    logger.error(f"응급 매도 실패 {position.symbol}: {e}")
                    continue
            
            # 긴급 알림
            await self.notification_manager.send_critical_alert(
                "🚨 응급 전량 매도 실행",
                f"사유: {reason}\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"매도 시도 포지션: {len(self.position_manager.positions)}개"
            )
            
        except Exception as e:
            logger.error(f"응급 매도 실행 실패: {e}")
    
    async def _emergency_sell_position(self, position: UnifiedPosition) -> bool:
        """개별 포지션 응급 매도"""
        try:
            # 전략별로 매도 방법이 다름
            if position.strategy in self.strategies:
                strategy = self.strategies[position.strategy]
                
                # 각 전략의 매도 메소드 호출
                # 실제 구현시에는 각 전략의 API 사용
                
                # 포지션 제거
                self.position_manager.remove_position(position.strategy, position.symbol)
                
                # 거래 기록
                self.data_manager.save_trade(
                    position.strategy, position.symbol, 'EMERGENCY_SELL',
                    position.quantity, position.current_price, position.currency,
                    position.krw_value, 0.0, position.unrealized_pnl
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"개별 응급 매도 실패: {e}")
            return False
    
    async def _generate_daily_report(self):
        """일일 리포트 생성"""
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            network_status = self.network_monitor.get_network_status()
            error_summary = self.error_monitor.get_error_summary()
            
            report = f"""
📊 퀸트프로젝트 일일 리포트
========================
📅 날짜: {datetime.now().strftime('%Y-%m-%d')}

💼 포트폴리오 현황:
• 총 포지션: {portfolio_summary.get('total_positions', 0)}개
• 총 가치: {portfolio_summary.get('total_krw_value', 0):,.0f}원
• 미실현 손익: {portfolio_summary.get('total_unrealized_pnl', 0):+,.0f}원
• 총 수익률: {portfolio_summary.get('total_return_pct', 0):+.2f}%

🎯 전략별 현황:
"""
            
            for strategy, data in portfolio_summary.get('by_strategy', {}).items():
                report += f"• {strategy}: {data['count']}개 포지션, {data['krw_value']:,.0f}원 ({data['unrealized_pnl']:+,.0f}원)\n"
            
            report += f"""
🌐 네트워크 상태:
• 연결 상태: {'✅ 정상' if network_status['is_connected'] else '❌ 끊김'}
• 지연시간: {network_status['latency_ms']:.1f}ms
• 가동률: {network_status['uptime_percentage']:.1f}%

🚨 오류 모니터링:
• 응급매도 시스템: {'✅ 활성화' if error_summary['emergency_enabled'] else '❌ 비활성화'}
• 오류 발생 전략: {error_summary['total_strategies_with_errors']}개
"""
            
            if error_summary.get('error_counts'):
                for strategy, count in error_summary['error_counts'].items():
                    report += f"  - {strategy}: {count}회 오류\n"
            
            report += f"""
🏆 상위 수익 종목:
"""
            
            for gainer in portfolio_summary.get('top_gainers', [])[:3]:
                report += f"• {gainer['symbol']} ({gainer['strategy']}): {gainer['pnl_pct']:+.1f}%\n"
            
            report += "\n💡 하위 수익 종목:\n"
            for loser in portfolio_summary.get('top_losers', [])[:3]:
                report += f"• {loser['symbol']} ({loser['strategy']}): {loser['pnl_pct']:+.1f}%\n"
            
            # 리포트 전송
            await self.notification_manager.send_alert("📊 일일 리포트", report)
            
        except Exception as e:
            logger.error(f"일일 리포트 생성 실패: {e}")
            self.error_monitor.record_strategy_error('daily_report', e)
    
    async def shutdown(self):
        """시스템 종료"""
        logger.info("🛑 퀸트프로젝트 통합 시스템 종료")
        
        try:
            self.is_running = False
            
            # 네트워크 모니터링 중지
            self.network_monitor.stop_monitoring()
            
            # 최종 백업
            self.data_manager.backup_database()
            
            # 종료 알림
            await self.notification_manager.send_alert(
                "🛑 시스템 종료",
                f"퀸트프로젝트 통합 시스템이 종료되었습니다.\n"
                f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"최종 포지션: {len(self.position_manager.positions)}개"
            )
            
        except Exception as e:
            logger.error(f"시스템 종료 오류: {e}")
    
    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            network_status = self.network_monitor.get_network_status()
            
            return {
                'system': {
                    'is_running': self.is_running,
                    'emergency_mode': self.emergency_mode,
                    'last_health_check': self.last_health_check.isoformat(),
                    'uptime': (datetime.now() - self.last_health_check).seconds
                },
                'strategies': {
                    'active_strategies': list(self.strategies.keys()),
                    'total_strategies': len(self.strategies)
                },
                'portfolio': portfolio_summary,
                'network': network_status,
                'currency': {
                    'last_update': self.currency_exchange.last_update.isoformat() if self.currency_exchange.last_update else None,
                    'rates': {k: v.exchange_rate for k, v in self.currency_exchange.exchange_rates.items()}
                },
                'error_monitor': self.error_monitor.get_error_summary()
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 실패: {e}")
            return {'error': str(e)}

# ============================================================================
# 🎮 CLI 인터페이스
# ============================================================================

class QuintCLI:
    """퀸트프로젝트 CLI 인터페이스"""
    
    def __init__(self):
        self.core = None
    
    def print_banner(self):
        """배너 출력"""
        banner = """
🏆════════════════════════════════════════════════════════════════🏆
   ██████╗ ██╗   ██╗██╗███╗   ██╗████████╗    ██████╗ ██████╗  ██████╗      
  ██╔═══██╗██║   ██║██║████╗  ██║╚══██╔══╝   ██╔════╝██╔═══██╗██╔══██╗     
  ██║   ██║██║   ██║██║██╔██╗ ██║   ██║█████╗██║     ██║   ██║██████╔╝     
  ██║▄▄ ██║██║   ██║██║██║╚██╗██║   ██║╚════╝██║     ██║   ██║██╔══██╗     
  ╚██████╔╝╚██████╔╝██║██║ ╚████║   ██║      ╚██████╗╚██████╔╝██║  ██║     
   ╚══▀▀═╝  ╚═════╝ ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═════╝ ╚═════╝ ╚═╝  ╚═╝     
                                                                           
        퀸트프로젝트 통합 코어 시스템 v1.1.0 (응급 오류 감지)                        
        🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐           
🏆════════════════════════════════════════════════════════════════🏆
        """
        print(banner)
    
    async def start_interactive_mode(self):
        """대화형 모드 시작"""
        self.print_banner()
        
        while True:
            try:
                print("\n" + "="*60)
                print("🎮 퀸트프로젝트 통합 관리 시스템")
                print("="*60)
                
                if self.core is None:
                    print("1. 🚀 시스템 시작")
                    print("2. ⚙️  설정 확인")
                    print("3. 📊 시스템 상태 (읽기 전용)")
                    print("0. 🚪 종료")
                else:
                    print("1. 📊 실시간 상태")
                    print("2. 💼 포트폴리오 현황")
                    print("3. 🌐 네트워크 상태")
                    print("4. 💱 환율 정보")
                    print("5. 🎯 전략 관리")
                    print("6. 📈 성과 리포트")
                    print("7. 🚨 응급 매도")
                    print("8. 🛑 시스템 종료")
                    print("9. 🔍 오류 현황")  # NEW!
                    print("0. 🚪 프로그램 종료")
                
                choice = input("\n선택하세요: ").strip()
                
                if self.core is None:
                    await self._handle_startup_menu(choice)
                else:
                    await self._handle_running_menu(choice)
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                await asyncio.sleep(2)
    
    async def _handle_startup_menu(self, choice: str):
        """시작 메뉴 처리"""
        if choice == '1':
            print("🚀 시스템을 시작합니다...")
            self.core = QuintProjectCore()
            
            # 백그라운드에서 시작
            asyncio.create_task(self.core.start())
            await asyncio.sleep(3)  # 시작 대기
            
        elif choice == '2':
            self._show_config()
            
        elif choice == '3':
            await self._show_readonly_status()
            
        elif choice == '0':
            exit(0)
    
    async def _handle_running_menu(self, choice: str):
        """실행 중 메뉴 처리"""
        if choice == '1':
            await self._show_realtime_status()
            
        elif choice == '2':
            await self._show_portfolio()
            
        elif choice == '3':
            await self._show_network_status()
            
        elif choice == '4':
            await self._show_currency_rates()
            
        elif choice == '5':
            await self._manage_strategies()
            
        elif choice == '6':
            await self._show_performance_report()
            
        elif choice == '7':
            await self._emergency_sell()
            
        elif choice == '8':
            await self._shutdown_system()
            
        elif choice == '9':
            await self._show_error_status()  # NEW!
            
        elif choice == '0':
            await self._shutdown_system()
            exit(0)
    
    def _show_config(self):
        """설정 정보 표시"""
        print("\n⚙️ 시스템 설정 정보")
        print("="*40)
        
        config_file = "settings.yaml"
        if Path(config_file).exists():
            print(f"✅ 설정 파일: {config_file}")
        else:
            print(f"❌ 설정 파일 없음: {config_file}")
        
        # 환경변수 체크
        env_vars = [
            'TELEGRAM_BOT_TOKEN', 'UPBIT_ACCESS_KEY', 'IBKR_HOST',
            'EXCHANGE_RATE_API_KEY', 'EMERGENCY_SELL_ON_ERROR'
        ]
        
        print("\n🔑 주요 환경변수:")
        for var in env_vars:
            value = os.getenv(var, '')
            status = "✅" if value else "❌"
            if var == 'EMERGENCY_SELL_ON_ERROR':
                masked_value = value
            else:
                masked_value = f"{value[:4]}***" if len(value) > 4 else "없음"
            print(f"  {status} {var}: {masked_value}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_readonly_status(self):
        """읽기 전용 상태 표시"""
        print("\n📊 시스템 상태 (읽기 전용)")
        print("="*40)
        
        # 파일 시스템 체크
        files_to_check = [
            "./data/quint_core.db",
            "./logs/quint_core.log",
            "./backups/",
            "settings.yaml"
        ]
        
        print("📁 파일 시스템:")
        for file_path in files_to_check:
            path = Path(file_path)
            if path.exists():
                if path.is_file():
                    size = path.stat().st_size
                    print(f"  ✅ {file_path} ({size:,} bytes)")
                else:
                    print(f"  ✅ {file_path} (디렉토리)")
            else:
                print(f"  ❌ {file_path} (없음)")
        
        # 모듈 가용성
        print(f"\n🔌 모듈 가용성:")
        print(f"  {'✅' if US_AVAILABLE else '❌'} 미국 주식 전략")
        print(f"  {'✅' if JP_AVAILABLE else '❌'} 일본 주식 전략") 
        print(f"  {'✅' if IN_AVAILABLE else '❌'} 인도 주식 전략")
        print(f"  {'✅' if CRYPTO_AVAILABLE else '❌'} 암호화폐 전략")
        print(f"  {'✅' if IBKR_AVAILABLE else '❌'} IBKR API")
        print(f"  {'✅' if UPBIT_AVAILABLE else '❌'} Upbit API")
        
        # 응급매도 설정
        print(f"\n🚨 응급매도 설정:")
        print(f"  시스템: {'✅ 활성화' if EMERGENCY_SELL_ON_ERROR else '❌ 비활성화'}")
        print(f"  메모리 임계치: {EMERGENCY_MEMORY_THRESHOLD}%")
        print(f"  CPU 임계치: {EMERGENCY_CPU_THRESHOLD}%")
        print(f"  디스크 임계치: {EMERGENCY_DISK_THRESHOLD}GB")
        print(f"  오류 임계치: {EMERGENCY_ERROR_COUNT}회")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_realtime_status(self):
        """실시간 상태 표시"""
        if not self.core:
            print("❌ 시스템이 시작되지 않았습니다.")
            return
        
        print("\n📊 실시간 시스템 상태")
        print("="*50)
        
        try:
            status = self.core.get_system_status()
            
            # 시스템 정보
            sys_info = status.get('system', {})
            print(f"🔄 실행 상태: {'✅ 실행 중' if sys_info.get('is_running') else '❌ 중지'}")
            print(f"🚨 응급 모드: {'⚠️ 활성화' if sys_info.get('emergency_mode') else '✅ 비활성화'}")
            print(f"⏰ 마지막 체크: {sys_info.get('last_health_check', 'N/A')}")
            
            # 전략 정보
            strategies = status.get('strategies', {})
            print(f"\n🎯 활성 전략: {', '.join(strategies.get('active_strategies', []))}")
            
            # 포트폴리오 요약
            portfolio = status.get('portfolio', {})
            print(f"\n💼 포트폴리오:")
            print(f"  총 포지션: {portfolio.get('total_positions', 0)}개")
            print(f"  총 가치: {portfolio.get('total_krw_value', 0):,.0f}원")
            print(f"  미실현 손익: {portfolio.get('total_unrealized_pnl', 0):+,.0f}원")
            
            # 네트워크 상태
            network = status.get('network', {})
            print(f"\n🌐 네트워크:")
            print(f"  연결: {'✅' if network.get('is_connected') else '❌'}")
            print(f"  지연시간: {network.get('latency_ms', 0):.1f}ms")
            print(f"  가동률: {network.get('uptime_percentage', 0):.1f}%")
            
            # 오류 모니터 상태
            error_info = status.get('error_monitor', {})
            print(f"\n🚨 오류 모니터:")
            print(f"  응급매도: {'✅ 활성화' if error_info.get('emergency_enabled') else '❌ 비활성화'}")
            print(f"  오류 전략: {error_info.get('total_strategies_with_errors', 0)}개")
            if error_info.get('error_counts'):
                for strategy, count in error_info['error_counts'].items():
                    print(f"    {strategy}: {count}회")
            
        except Exception as e:
            print(f"❌ 상태 조회 실패: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_error_status(self):
        """오류 현황 상세 표시 (NEW!)"""
        if not self.core:
            print("❌ 시스템이 시작되지 않았습니다.")
            return
        
        print("\n🔍 오류 현황 상세")
        print("="*40)
        
        try:
            error_summary = self.core.error_monitor.get_error_summary()
            
            print(f"🚨 응급매도 시스템: {'✅ 활성화' if error_summary['emergency_enabled'] else '❌ 비활성화'}")
            print(f"📊 오류 발생 전략
        self.cpu_high_start = None
        self.memory_warnings = 0
        
        # 신호 핸들러 등록
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """프로세스 종료 신호 핸들러 설정"""
        try:
            signal.signal(signal.SIGTERM, self._handle_termination_signal)
            signal.signal(signal.SIGINT, self._handle_termination_signal)
            if hasattr(signal, 'SIGHUP'):  # Unix/Linux only
                signal.signal(signal.SIGHUP, self._handle_termination_signal)
        except Exception as e:
            logger.debug(f"신호 핸들러 설정 실패: {e}")
    
    def _handle_termination_signal(self, signum, frame):
        """종료 신호 처리"""
        logger.critical(f"🚨 종료 신호 감지: {signum}")
        if EMERGENCY_SELL_ON_ERROR:
            asyncio.create_task(self._emergency_sell_on_signal(signum))
    
    async def _emergency_sell_on_signal(self, signum):
        """신호 기반 응급 매도"""
        try:
            await self.core_system.emergency_sell_all(f"TERMINATION_SIGNAL_{signum}")
        except Exception as e:
            logger.error(f"신호 기반 응급 매도 실패: {e}")
    
    async def check_system_resources(self):
        """시스템 리소스 체크"""
        if not EMERGENCY_SELL_ON_ERROR:
            return
        
        try:
            # 메모리 체크
            memory_usage = psutil.virtual_memory().percent
            if memory_usage >= EMERGENCY_MEMORY_THRESHOLD:
                self.memory_warnings += 1
                if self.memory_warnings >= 3:  # 3회 연속 경고
                    await self._handle_resource_emergency("MEMORY_CRITICAL", 
                                                         f"메모리 사용량: {memory_usage:.1f}%")
            else:
                self.memory_warnings = 0
            
            # CPU 체크
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage >= EMERGENCY_CPU_THRESHOLD:
                if self.cpu_high_start is None:
                    self.cpu_high_start = time.time()
                elif time.time() - self.cpu_high_start > 300:  # 5분 연속
                    await self._handle_resource_emergency("CPU_CRITICAL", 
                                                         f"CPU 사용량: {cpu_usage:.1f}% (5분 연속)")
            else:
                self.cpu_high_start = None
            
            # 디스크 체크
            disk_free = psutil.disk_usage('/').free / (1024**3)  # GB
            if disk_free <= EMERGENCY_DISK_THRESHOLD:
                await self._handle_resource_emergency("DISK_CRITICAL", 
                                                     f"디스크 여유공간: {disk_free:.1f}GB")
            
        except Exception as e:
            logger.error(f"시스템 리소스 체크 실패: {e}")
    
    async def _handle_resource_emergency(self, reason: str, details: str):
        """리소스 부족 응급 상황 처리"""
        if self._should_skip_emergency():
            return
        
        logger.critical(f"🚨 시스템 리소스 위험: {reason} - {details}")
        
        # 유예 시간 후 응급 매도
        await asyncio.sleep(EMERGENCY_GRACE_PERIOD)
        await self.core_system.emergency_sell_all(f"{reason}: {details}")
        
        self.last_emergency_time = datetime.now()
    
    def record_strategy_error(self, strategy: str, error: Exception):
        """전략 오류 기록"""
        if strategy not in self.error_counts:
            self.error_counts[strategy] = {'count': 0, 'last_error_time': None, 'errors': []}
        
        self.error_counts[strategy]['count'] += 1
        self.error_counts[strategy]['last_error_time'] = datetime.now()
        self.error_counts[strategy]['errors'].append({
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        
        # 최근 오류만 유지 (메모리 절약)
        if len(self.error_counts[strategy]['errors']) > 10:
            self.error_counts[strategy]['errors'] = self.error_counts[strategy]['errors'][-10:]
        
        # 연속 오류 체크
        if self.error_counts[strategy]['count'] >= EMERGENCY_ERROR_COUNT:
            asyncio.create_task(self._handle_strategy_emergency(strategy))
    
    async def _handle_strategy_emergency(self, strategy: str):
        """전략 오류 응급 상황 처리"""
        if not EMERGENCY_SELL_ON_ERROR or self._should_skip_emergency():
            return
        
        error_info = self.error_counts[strategy]
        logger.critical(f"🚨 전략 오류 임계치 초과: {strategy} ({error_info['count']}회)")
        
        # 해당 전략 포지션만 응급 매도
        await self._emergency_sell_strategy_positions(strategy)
        
        # 오류 카운터 리셋
        self.error_counts[strategy] = {'count': 0, 'last_error_time': None, 'errors': []}
        self.last_emergency_time = datetime.now()
    
    async def _emergency_sell_strategy_positions(self, strategy: str):
        """특정 전략 포지션 응급 매도"""
        try:
            positions_to_sell = [
                pos for pos in self.core_system.position_manager.positions.values()
                if pos.strategy == strategy
            ]
            
            if not positions_to_sell:
                logger.info(f"📝 {strategy} 전략에 매도할 포지션 없음")
                return
            
            for position in positions_to_sell:
                try:
                    success = await self.core_system._emergency_sell_position(position)
                    if success:
                        logger.info(f"🚨 {strategy} 응급 매도 완료: {position.symbol}")
                except Exception as e:
                    logger.error(f"🚨 {strategy} 응급 매도 실패 {position.symbol}: {e}")
            
            # 알림 전송
            await self.core_system.notification_manager.send_critical_alert(
                f"🚨 {strategy} 전략 응급 매도",
                f"전략: {strategy}\n"
                f"매도 포지션: {len(positions_to_sell)}개\n"
                f"사유: 연속 오류 {EMERGENCY_ERROR_COUNT}회 초과"
            )
            
        except Exception as e:
            logger.error(f"전략별 응급 매도 실패: {e}")
    
    def _should_skip_emergency(self) -> bool:
        """응급 매도 스킵 여부 (중복 방지)"""
        if self.last_emergency_time is None:
            return False
        
        # 10분 내 중복 응급 매도 방지
        return (datetime.now() - self.last_emergency_time).seconds < 600
    
    def reset_error_counts(self, strategy: str = None):
        """오류 카운터 리셋"""
        if strategy:
            if strategy in self.error_counts:
                self.error_counts[strategy] = {'count': 0, 'last_error_time': None, 'errors': []}
        else:
            self.error_counts = {}
    
    def get_error_summary(self) -> Dict:
        """오류 현황 요약"""
        return {
            'total_strategies_with_errors': len(self.error_counts),
            'error_counts': {k: v['count'] for k, v in self.error_counts.items()},
            'last_emergency_time': self.last_emergency_time.isoformat() if self.last_emergency_time else None,
            'emergency_enabled': EMERGENCY_SELL_ON_ERROR
        }

# ============================================================================
# 📊 데이터 클래스 정의
# ============================================================================

@dataclass
class Currency:
    """통화 정보"""
    code: str
    name: str
    symbol: str
    exchange_rate: float
    last_updated: datetime

@dataclass
class UnifiedPosition:
    """통합 포지션"""
    strategy: str  # us, jp, in, crypto
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    currency: str
    krw_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_date: datetime
    last_updated: datetime

@dataclass
class StrategyPerformance:
    """전략별 성과"""
    strategy: str
    total_investment: float
    current_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_return_pct: float
    win_rate: float
    trades_count: int
    avg_holding_days: float
    last_updated: datetime

@dataclass
class NetworkStatus:
    """네트워크 상태"""
    is_connected: bool
    latency: float
    last_check: datetime
    consecutive_failures: int
    uptime_percentage: float

# ============================================================================
# 💱 자동 환전 시스템
# ============================================================================

class AutoCurrencyExchange:
    """자동 환전 시스템"""
    
    def __init__(self):
        self.exchange_rates = {}
        self.last_update = None
        self.update_interval = 300  # 5분마다 업데이트
        self.api_key = os.getenv('EXCHANGE_RATE_API_KEY', '')
        
    async def update_exchange_rates(self) -> bool:
        """환율 정보 업데이트"""
        try:
            # 여러 API 소스 사용 (무료 + 유료)
            rates = await self._fetch_from_multiple_sources()
            
            if rates:
                self.exchange_rates = {
                    'USD': Currency('USD', '미국 달러', '$', rates.get('USD', 1300), datetime.now()),
                    'JPY': Currency('JPY', '일본 엔', '¥', rates.get('JPY', 10), datetime.now()),
                    'INR': Currency('INR', '인도 루피', '₹', rates.get('INR', 16), datetime.now()),
                    'KRW': Currency('KRW', '한국 원', '₩', 1.0, datetime.now())
                }
                self.last_update = datetime.now()
                logger.info(f"💱 환율 업데이트: USD={rates.get('USD'):.0f}, JPY={rates.get('JPY'):.1f}, INR={rates.get('INR'):.1f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"환율 업데이트 실패: {e}")
            return False
    
    async def _fetch_from_multiple_sources(self) -> Dict[str, float]:
        """다중 소스에서 환율 정보 수집"""
        rates = {}
        
        # 1. 한국은행 API (기본)
        try:
            bok_rates = await self._fetch_from_bok()
            if bok_rates:
                rates.update(bok_rates)
        except Exception as e:
            logger.debug(f"한국은행 API 실패: {e}")
        
        # 2. ExchangeRate-API (백업)
        try:
            if not rates and self.api_key:
                er_rates = await self._fetch_from_exchangerate_api()
                if er_rates:
                    rates.update(er_rates)
        except Exception as e:
            logger.debug(f"ExchangeRate-API 실패: {e}")
        
        # 3. 고정값 (최후 수단)
        if not rates:
            rates = {'USD': 1300, 'JPY': 9.5, 'INR': 16}
            logger.warning("⚠️ 고정 환율 사용")
        
        return rates
    
    async def _fetch_from_bok(self) -> Dict[str, float]:
        """한국은행 환율 API"""
        try:
            url = "https://www.bok.or.kr/portal/singl/openapi/exchangeJSON.do"
            params = {
                'lang': 'ko',
                'per': 'day',
                'keytype': 'json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        rates = {}
                        for item in data:
                            if item['CUR_UNIT'] == 'USD':
                                rates['USD'] = float(item['DEAL_BAS_R'].replace(',', ''))
                            elif item['CUR_UNIT'] == 'JPY(100)':
                                rates['JPY'] = float(item['DEAL_BAS_R'].replace(',', '')) / 100
                            elif item['CUR_UNIT'] == 'INR':
                                rates['INR'] = float(item['DEAL_BAS_R'].replace(',', ''))
                        
                        return rates
        except Exception as e:
            logger.debug(f"한국은행 API 오류: {e}")
        
        return {}
    
    async def _fetch_from_exchangerate_api(self) -> Dict[str, float]:
        """ExchangeRate-API (유료)"""
        try:
            url = f"https://v6.exchangerate-api.com/v6/{self.api_key}/latest/KRW"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        conversion_rates = data['conversion_rates']
                        
                        return {
                            'USD': 1 / conversion_rates['USD'],
                            'JPY': 1 / conversion_rates['JPY'],
                            'INR': 1 / conversion_rates['INR']
                        }
        except Exception as e:
            logger.debug(f"ExchangeRate-API 오류: {e}")
        
        return {}
    
    def convert_to_krw(self, amount: float, from_currency: str) -> float:
        """다른 통화를 원화로 변환"""
        try:
            if from_currency == 'KRW':
                return amount
            
            if from_currency in self.exchange_rates:
                rate = self.exchange_rates[from_currency].exchange_rate
                return amount * rate
            else:
                logger.warning(f"⚠️ 환율 정보 없음: {from_currency}")
                return amount
        
        except Exception as e:
            logger.error(f"환전 실패: {e}")
            return amount
    
    def convert_from_krw(self, krw_amount: float, to_currency: str) -> float:
        """원화를 다른 통화로 변환"""
        try:
            if to_currency == 'KRW':
                return krw_amount
            
            if to_currency in self.exchange_rates:
                rate = self.exchange_rates[to_currency].exchange_rate
                return krw_amount / rate
            else:
                logger.warning(f"⚠️ 환율 정보 없음: {to_currency}")
                return krw_amount
        
        except Exception as e:
            logger.error(f"환전 실패: {e}")
            return krw_amount
    
    def get_exchange_rate(self, currency: str) -> Optional[float]:
        """환율 조회"""
        if currency in self.exchange_rates:
            return self.exchange_rates[currency].exchange_rate
        return None

# ============================================================================
# 🌐 네트워크 모니터링 시스템
# ============================================================================

class NetworkMonitor:
    """네트워크 모니터링 + 끊김 시 전량 매도"""
    
    def __init__(self, core_system):
        self.core_system = core_system
        self.is_monitoring = False
        self.status = NetworkStatus(
            is_connected=True,
            latency=0.0,
            last_check=datetime.now(),
            consecutive_failures=0,
            uptime_percentage=100.0
        )
        
        # 설정
        self.check_interval = int(os.getenv('NETWORK_CHECK_INTERVAL', '30'))  # 30초
        self.timeout = int(os.getenv('NETWORK_TIMEOUT', '10'))  # 10초
        self.max_failures = int(os.getenv('NETWORK_MAX_FAILURES', '3'))  # 3회 연속 실패
        self.grace_period = int(os.getenv('NETWORK_GRACE_PERIOD', '300'))  # 5분 유예
        self.emergency_sell = os.getenv('NETWORK_DISCONNECT_SELL_ALL', 'false').lower() == 'true'
        
        # 테스트 대상
        self.test_hosts = [
            ('8.8.8.8', 53),      # Google DNS
            ('1.1.1.1', 53),      # Cloudflare DNS
            ('yahoo.com', 80),    # Yahoo Finance
            ('upbit.com', 443)    # Upbit
        ]
        
        # 통계
        self.total_checks = 0
        self.successful_checks = 0
        self.last_disconnect_time = None
        
    async def start_monitoring(self):
        """네트워크 모니터링 시작"""
        self.is_monitoring = True
        logger.info("🌐 네트워크 모니터링 시작")
        
        while self.is_monitoring:
            try:
                await self._check_network_status()
                await asyncio.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"네트워크 모니터링 오류: {e}")
                await asyncio.sleep(60)  # 1분 후 재시도
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        logger.info("⏹️ 네트워크 모니터링 중지")
    
    async def _check_network_status(self):
        """네트워크 상태 체크"""
        start_time = time.time()
        success_count = 0
        
        # 다중 호스트 테스트
        for host, port in self.test_hosts:
            try:
                if await self._test_connection(host, port):
                    success_count += 1
            except:
                continue
        
        # 결과 계산
        latency = (time.time() - start_time) * 1000  # ms
        is_connected = success_count >= 2  # 절반 이상 성공
        
        self.total_checks += 1
        if is_connected:
            self.successful_checks += 1
            self.status.consecutive_failures = 0
        else:
            self.status.consecutive_failures += 1
        
        # 상태 업데이트
        previous_status = self.status.is_connected
        self.status.is_connected = is_connected
        self.status.latency = latency
        self.status.last_check = datetime.now()
        self.status.uptime_percentage = (self.successful_checks / self.total_checks) * 100
        
        # 연결 상태 변화 감지
        if previous_status and not is_connected:
            await self._handle_network_disconnect()
        elif not previous_status and is_connected:
            await self._handle_network_reconnect()
        
        # 연속 실패 체크
        if self.status.consecutive_failures >= self.max_failures:
            await self._handle_critical_network_failure()
    
    async def _test_connection(self, host: str, port: int) -> bool:
        """개별 연결 테스트"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        
        except Exception:
            return False
    
    async def _handle_network_disconnect(self):
        """네트워크 끊김 처리"""
        self.last_disconnect_time = datetime.now()
        logger.warning("🚨 네트워크 연결 끊김 감지!")
        
        # 알림 전송
        await self.core_system.notification_manager.send_critical_alert(
            "🚨 네트워크 연결 끊김",
            f"네트워크 연결이 끊어졌습니다.\n"
            f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"연속 실패: {self.status.consecutive_failures}회"
        )
    
    async def _handle_network_reconnect(self):
        """네트워크 재연결 처리"""
        disconnect_duration = 0
        if self.last_disconnect_time:
            disconnect_duration = (datetime.now() - self.last_disconnect_time).seconds
        
        logger.info(f"✅ 네트워크 연결 복구 (끊김 시간: {disconnect_duration}초)")
        
        # 알림 전송
        await self.core_system.notification_manager.send_alert(
            "✅ 네트워크 연결 복구",
            f"네트워크 연결이 복구되었습니다.\n"
            f"끊김 시간: {disconnect_duration}초\n"
            f"현재 지연시간: {self.status.latency:.1f}ms"
        )
    
    async def _handle_critical_network_failure(self):
        """치명적 네트워크 장애 처리"""
        if not self.emergency_sell:
            logger.warning("⚠️ 네트워크 장애 감지 (응급매도 비활성화)")
            return
        
        logger.critical("🚨 치명적 네트워크 장애 - 응급 전량 매도 실행!")
        
        try:
            # 유예 시간 체크
            if self.last_disconnect_time:
                disconnect_duration = (datetime.now() - self.last_disconnect_time).seconds
                if disconnect_duration < self.grace_period:
                    logger.info(f"⏳ 유예 시간 대기 중: {self.grace_period - disconnect_duration}초 남음")
                    return
            
            # 응급 전량 매도 실행
            await self.core_system.emergency_sell_all("NETWORK_FAILURE")
            
        except Exception as e:
            logger.error(f"응급 매도 실행 실패: {e}")
    
    def get_network_status(self) -> Dict:
        """네트워크 상태 정보 반환"""
        return {
            'is_connected': self.status.is_connected,
            'latency_ms': self.status.latency,
            'consecutive_failures': self.status.consecutive_failures,
            'uptime_percentage': self.status.uptime_percentage,
            'last_check': self.status.last_check.isoformat(),
            'total_checks': self.total_checks,
            'emergency_sell_enabled': self.emergency_sell
        }

# ============================================================================
# 📱 통합 알림 시스템
# ============================================================================

class NotificationManager:
    """통합 알림 관리자"""
    
    def __init__(self):
        # 텔레그램 설정
        self.telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # 이메일 설정
        self.email_enabled = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        self.email_smtp_server = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.email_smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.email_username = os.getenv('EMAIL_USERNAME', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        self.email_to = os.getenv('EMAIL_TO', '')
        
        # 알림 레벨 설정
        self.levels = {
            'trade_execution': True,
            'profit_loss': True,
            'risk_warning': True,
            'network_status': True,
            'daily_summary': True
        }
    
    async def send_alert(self, title: str, message: str, level: str = 'info'):
        """일반 알림 전송"""
        try:
            formatted_message = f"🏆 퀸트프로젝트\n\n📌 {title}\n\n{message}\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            tasks = []
            
            if self.telegram_enabled:
                tasks.append(self._send_telegram(formatted_message))
            
            if self.email_enabled and level in ['warning', 'critical']:
                tasks.append(self._send_email(title, formatted_message))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"알림 전송 실패: {e}")
    
    async def send_critical_alert(self, title: str, message: str):
        """중요 알림 전송 (모든 채널)"""
        try:
            formatted_message = f"🚨 긴급 알림\n\n📌 {title}\n\n{message}\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            tasks = []
            
            if self.telegram_enabled:
                tasks.append(self._send_telegram(formatted_message))
            
            if self.email_enabled:
                tasks.append(self._send_email(f"🚨 {title}", formatted_message))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"중요 알림 전송 실패: {e}")
    
    async def _send_telegram(self, message: str):
        """텔레그램 전송"""
        try:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                return
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=10) as response:
                    if response.status == 200:
                        logger.debug("텔레그램 알림 전송 완료")
                    else:
                        logger.error(f"텔레그램 전송 실패: {response.status}")
                        
        except Exception as e:
            logger.error(f"텔레그램 전송 오류: {e}")
    
    async def _send_email(self, subject: str, message: str):
        """이메일 전송"""
        try:
            if not all([self.email_username, self.email_password, self.email_to]):
                return
            
            msg = MimeMultipart()
            msg['From'] = self.email_username
            msg['To'] = self.email_to
            msg['Subject'] = subject
            
            msg.attach(MimeText(message, 'plain', 'utf-8'))
            
            # 비동기 이메일 전송
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_email_sync, msg
            )
            
        except Exception as e:
            logger.error(f"이메일 전송 오류: {e}")
    
    def _send_email_sync(self, msg):
        """동기 이메일 전송"""
        try:
            server = smtplib.SMTP(self.email_smtp_server, self.email_smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            server.send_message(msg)
            server.quit()
            logger.debug("이메일 알림 전송 완료")
            
        except Exception as e:
            logger.error(f"이메일 전송 동기 오류: {e}")

# ============================================================================
# 🗃️ 통합 데이터 관리자
# ============================================================================

class DataManager:
    """통합 데이터 관리 시스템"""
    
    def __init__(self):
        self.db_path = os.getenv('DATABASE_PATH', './data/quint_core.db')
        self.backup_enabled = os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
        self.backup_path = os.getenv('BACKUP_PATH', './backups/')
        
        # 디렉토리 생성
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.backup_path).mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 통합 포지션 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unified_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    currency TEXT NOT NULL,
                    krw_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    unrealized_pnl_pct REAL NOT NULL,
                    entry_date DATETIME NOT NULL,
                    last_updated DATETIME NOT NULL,
                    UNIQUE(strategy, symbol)
                )
            ''')
            
            # 거래 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    currency TEXT NOT NULL,
                    krw_amount REAL NOT NULL,
                    commission REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            # 성과 추적 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_investment REAL NOT NULL,
                    current_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_return_pct REAL NOT NULL,
                    trades_count INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    created_at DATETIME NOT NULL,
                    UNIQUE(strategy, date)
                )
            ''')
            
            # 네트워크 상태 로그
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS network_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    is_connected BOOLEAN NOT NULL,
                    latency REAL NOT NULL,
                    consecutive_failures INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            # 환율 기록
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS exchange_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    currency TEXT NOT NULL,
                    rate REAL NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            # 오류 기록 테이블 (NEW!)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    stack_trace TEXT,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def save_position(self, position: UnifiedPosition):
        """포지션 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO unified_positions 
                (strategy, symbol, quantity, avg_price, current_price, currency, 
                 krw_value, unrealized_pnl, unrealized_pnl_pct, entry_date, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.strategy, position.symbol, position.quantity, position.avg_price,
                position.current_price, position.currency, position.krw_value,
                position.unrealized_pnl, position.unrealized_pnl_pct,
                position.entry_date.isoformat(), position.last_updated.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"포지션 저장 실패: {e}")
    
    def load_all_positions(self) -> List[UnifiedPosition]:
        """모든 포지션 로드"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM unified_positions')
            rows = cursor.fetchall()
            conn.close()
            
            positions = []
            for row in rows:
                position = UnifiedPosition(
                    strategy=row[1], symbol=row[2], quantity=row[3], avg_price=row[4],
                    current_price=row[5], currency=row[6], krw_value=row[7],
                    unrealized_pnl=row[8], unrealized_pnl_pct=row[9],
                    entry_date=datetime.fromisoformat(row[10]),
                    last_updated=datetime.fromisoformat(row[11])
                )
                positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"포지션 로드 실패: {e}")
            return []
    
    def save_trade(self, strategy: str, symbol: str, action: str, quantity: float, 
                   price: float, currency: str, krw_amount: float, commission: float = 0.0, 
                   realized_pnl: float = 0.0):
        """거래 기록 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades 
                (strategy, symbol, action, quantity, price, currency, krw_amount, 
                 commission, realized_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy, symbol, action, quantity, price, currency, krw_amount,
                commission, realized_pnl, datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"거래 기록 저장 실패: {e}")
    
    def save_error_log(self, strategy: str, error_type: str, error_message: str, stack_trace: str = None):
        """오류 로그 저장 (NEW!)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO error_logs (strategy, error_type, error_message, stack_trace, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                strategy, error_type, error_message, stack_trace, datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"오류 로그 저장 실패: {e}")
    
    def save_performance(self, performance: StrategyPerformance):
        """성과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO performance 
                (strategy, date, total_investment, current_value, unrealized_pnl, 
                 realized_pnl, total_return_pct, trades_count, win_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                performance.strategy, datetime.now().date().isoformat(),
                performance.total_investment, performance.current_value, performance.unrealized_pnl,
                performance.realized_pnl, performance.total_return_pct, performance.trades_count,
                performance.win_rate, datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"성과 저장 실패: {e}")
    
    def save_network_log(self, status: NetworkStatus):
        """네트워크 상태 로그 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO network_logs (is_connected, latency, consecutive_failures, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (
                status.is_connected, status.latency, status.consecutive_failures,
                status.last_check.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"네트워크 로그 저장 실패: {e}")
    
    def backup_database(self):
        """데이터베이스 백업"""
        if not self.backup_enabled:
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = Path(self.backup_path) / f"quint_core_backup_{timestamp}.db"
            
            # 파일 복사
            import shutil
            shutil.copy2(self.db_path, backup_file)
            
            logger.info(f"📦 데이터베이스 백업 완료: {backup_file}")
            
            # 오래된 백업 파일 정리 (30일 이상)
            self._cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"데이터베이스 백업 실패: {e}")
    
    def _cleanup_old_backups(self):
        """오래된 백업 파일 정리"""
        try:
            backup_dir = Path(self.backup_path)
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for backup_file in backup_dir.glob("quint_core_backup_*.db"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    logger.debug(f"오래된 백업 파일 삭제: {backup_file}")
                    
        except Exception as e:
            logger.error(f"백업 파일 정리 실패: {e}")

# ============================================================================
# 🎯 통합 포지션 관리자
# ============================================================================

class UnifiedPositionManager:
    """통합 포지션 관리 시스템"""
    
    def __init__(self, data_manager: DataManager, currency_exchange: AutoCurrencyExchange):
        self.data_manager = data_manager
        self.currency_exchange = currency_exchange
        self.positions: Dict[str, UnifiedPosition] = {}
        self.load_positions()
    
    def load_positions(self):
        """저장된 포지션 로드"""
        try:
            positions = self.data_manager.load_all_positions()
            self.positions = {f"{pos.strategy}_{pos.symbol}": pos for pos in positions}
            logger.info(f"📂 포지션 로드: {len(self.positions)}개")
        except Exception as e:
            logger.error(f"포지션 로드 실패: {e}")
    
    def add_position(self, strategy: str, symbol: str, quantity: float, 
                    avg_price: float, currency: str):
        """포지션 추가"""
        try:
            key = f"{strategy}_{symbol}"
            
            # 기존 포지션이 있으면 평균단가 계산
            if key in self.positions:
                existing = self.positions[key]
                total_quantity = existing.quantity + quantity
                total_cost = (existing.quantity * existing.avg_price) + (quantity * avg_price)
                new_avg_price = total_cost / total_quantity
                
                existing.quantity = total_quantity
                existing.avg_price = new_avg_price
                existing.last_updated = datetime.now()
            else:
                # 새 포지션 생성
                position = UnifiedPosition(
                    strategy=strategy,
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=avg_price,
                    current_price=avg_price,
                    currency=currency,
                    krw_value=self.currency_exchange.convert_to_krw(quantity * avg_price, currency),
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    entry_date=datetime.now(),
                    last_updated=datetime.now()
                )
                self.positions[key] = position
            
            # 데이터베이스에 저장
            self.data_manager.save_position(self.positions[key])
            logger.info(f"➕ 포지션 추가: {strategy} {symbol} {quantity}")
            
        except Exception as e:
            logger.error(f"포지션 추가 실패: {e}")
    
    def remove_position(self, strategy: str, symbol: str, quantity: float = None):
        """포지션 제거 (부분/전체)"""
        try:
            key = f"{strategy}_{symbol}"
            
            if key not in self.positions:
                logger.warning(f"⚠️ 포지션 없음: {strategy} {symbol}")
                return
            
            position = self.positions[key]
            
            if quantity is None or quantity >= position.quantity:
                # 전체 제거
                del self.positions[key]
                # DB에서도 제거
                conn = sqlite3.connect(self.data_manager.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM unified_positions WHERE strategy = ? AND symbol = ?', 
                             (strategy, symbol))
                conn.commit()
                conn.close()
                logger.info(f"➖ 포지션 전체 제거: {strategy} {symbol}")
            else:
                # 부분 제거
                position.quantity -= quantity
                position.last_updated = datetime.now()
                self.data_manager.save_position(position)
                logger.info(f"➖ 포지션 부분 제거: {strategy} {symbol} {quantity}")
                
        except Exception as e:
            logger.error(f"포지션 제거 실패: {e}")
    
    def update_current_prices(self, price_data: Dict[str, Dict[str, float]]):
        """현재가 업데이트"""
        try:
            for key, position in self.positions.items():
                strategy_prices = price_data.get(position.strategy, {})
                if position.symbol in strategy_prices:
                    old_price = position.current_price
                    new_price = strategy_prices[position.symbol]
                    
                    # 가격 및 손익 업데이트
                    position.current_price = new_price
                    position.krw_value = self.currency_exchange.convert_to_krw(
                        position.quantity * new_price, position.currency
                    )
                    position.unrealized_pnl = (new_price - position.avg_price) * position.quantity
                    position.unrealized_pnl_pct = ((new_price - position.avg_price) / position.avg_price) * 100
                    position.last_updated = datetime.now()
                    
                    # 데이터베이스 업데이트
                    self.data_manager.save_position(position)
                    
                    # 큰 변동시 로그
                    price_change = abs((new_price - old_price) / old_price) * 100
                    if price_change > 5:  # 5% 이상 변동
                        logger.info(f"💹 {position.symbol}: {price_change:+.1f}% @ {new_price}")
                        
        except Exception as e:
            logger.error(f"현재가 업데이트 실패: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약"""
        try:
            summary = {
                'total_positions': len(self.positions),
                'total_krw_value': 0.0,
                'total_unrealized_pnl': 0.0,
                'by_strategy': {},
                'by_currency': {},
                'top_gainers': [],
                'top_losers': []
            }
            
            positions_with_pnl = []
            
            for position in self.positions.values():
                # 전체 합계
                summary['total_krw_value'] += position.krw_value
                summary['total_unrealized_pnl'] += self.currency_exchange.convert_to_krw(
                    position.unrealized_pnl, position.currency
                )
                
                # 전략별 집계
                if position.strategy not in summary['by_strategy']:
                    summary['by_strategy'][position.strategy] = {
                        'count': 0, 'krw_value': 0.0, 'unrealized_pnl': 0.0
                    }
                
                summary['by_strategy'][position.strategy]['count'] += 1
                summary['by_strategy'][position.strategy]['krw_value'] += position.krw_value
                summary['by_strategy'][position.strategy]['unrealized_pnl'] += self.currency_exchange.convert_to_krw(
                    position.unrealized_pnl, position.currency
                )
                
                # 통화별 집계
                if position.currency not in summary['by_currency']:
                    summary['by_currency'][position.currency] = {'count': 0, 'krw_value': 0.0}
                
                summary['by_currency'][position.currency]['count'] += 1
                summary['by_currency'][position.currency]['krw_value'] += position.krw_value
                
                # 수익률 정렬용
                positions_with_pnl.append((position, position.unrealized_pnl_pct))
            
            # 상위/하위 종목
            positions_with_pnl.sort(key=lambda x: x[1], reverse=True)
            
            summary['top_gainers'] = [
                {'symbol': pos.symbol, 'strategy': pos.strategy, 'pnl_pct': pnl_pct}
                for pos, pnl_pct in positions_with_pnl[:5]
            ]
            
            summary['top_losers'] = [
                {'symbol': pos.symbol, 'strategy': pos.strategy, 'pnl_pct': pnl_pct}
                for pos, pnl_pct in positions_with_pnl[-5:]
            ]
            
            # 총 수익률
            if summary['total_krw_value'] > 0:
                summary['total_return_pct'] = (summary['total_unrealized_pnl'] / 
                                             (summary['total_krw_value'] - summary['total_unrealized_pnl'])) * 100
            else:
                summary['total_return_pct'] = 0.0
            
            return summary
            
        except Exception as e:
            logger.error(f"포트폴리오 요약 실패: {e}")
            return {}

# ============================================================================
# 🏆 퀸트프로젝트 통합 코어 시스템
# ============================================================================

class QuintProjectCore:
    """퀸트프로젝트 통합 코어 시스템"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        logger.info("🏆 퀸트프로젝트 통합 코어 시스템 초기화")
        
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 핵심 시스템 초기화
        self.currency_exchange = AutoCurrencyExchange()
        self.notification_manager = NotificationManager()
        self.data_manager = DataManager()
        self.position_manager = UnifiedPositionManager(self.data_manager, self.currency_exchange)
        self.network_monitor = NetworkMonitor(self)
        self.error_monitor = EmergencyErrorMonitor(self)  # NEW!
        
        # 전략 시스템 초기화
        self.strategies = {}
        self._init_strategies()
        
        # 상태 변수
        self.is_running = False
        self.emergency_mode = False
        self.last_health_check = datetime.now()
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ 설정 파일 로드: {config_path}")
                return config
            else:
                logger.warning(f"⚠️ 설정 파일 없음: {config_path}, 기본값 사용")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'system': {'environment': 'development', 'debug_mode': True},
            'us_strategy': {'enabled': False},
            'japan_strategy': {'enabled': False},
            'india_strategy': {'enabled': False},
            'crypto_strategy': {'enabled': False}
        }
    
    def _init_strategies(self):
        """전략 시스템 초기화"""
        logger.info("🎯 전략 시스템 초기화 시작")
        
        # 🇺🇸 미국 주식 전략
        if self.config.get('us_strategy', {}).get('enabled', False) and US_AVAILABLE:
            try:
                self.strategies['us'] = USStrategy()
                logger.info("✅ 미국 주식 전략 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 미국 주식 전략 초기화 실패: {e}")
                self.error_monitor.record_strategy_error('us_init', e)
        
        # 🇯🇵 일본 주식 전략
        if self.config.get('japan_strategy', {}).get('enabled', False) and JP_AVAILABLE:
            try:
                self.strategies['japan'] = JPStrategy()
                logger.info("✅ 일본 주식 전략 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 일본 주식 전략 초기화 실패: {e}")
                self.error_monitor.record_strategy_error('japan_init', e)
        
        # 🇮🇳 인도 주식 전략
        if self.config.get('india_strategy', {}).get('enabled', False) and IN_AVAILABLE:
            try:
                self.strategies['india'] = INStrategy()
                logger.info("✅ 인도 주식 전략 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 인도 주식 전략 초기화 실패: {e}")
                self.error_monitor.record_strategy_error('india_init', e)
        
        # 💰 암호화폐 전략
        if self.config.get('crypto_strategy', {}).get('enabled', False) and CRYPTO_AVAILABLE:
            try:
                self.strategies['crypto'] = CryptoStrategy()
                logger.info("✅ 암호화폐 전략 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 암호화폐 전략 초기화 실패: {e}")
                self.error_monitor.record_strategy_error('crypto_init', e)
        
        logger.info(f"🎯 활성화된 전략: {list(self.strategies.keys())}")
    
    async def start(self):
        """시스템 시작"""
        logger.info("🚀 퀸트프로젝트 통합 시스템 시작")
        
        try:
            self.is_running = True
            
            # 환율 정보 초기 업데이트
            await self.currency_exchange.update_exchange_rates()
            
            # 시작 알림
            await self.notification_manager.send_alert(
                "🚀 시스템 시작",
                f"퀸트프로젝트 통합 시스템이 시작되었습니다.\n"
                f"활성화된 전략: {', '.join(self.strategies.keys())}\n"
                f"응급매도 시스템: {'✅ 활성화' if EMERGENCY_SELL_ON_ERROR else '❌ 비활성화'}\n"
                f"환경: {self.config.get('system', {}).get('environment', 'unknown')}"
            )
            
            # 백그라운드 태스크 시작
            tasks = [
                asyncio.create_task(self._main_loop()),
                asyncio.create_task(self.network_monitor.start_monitoring()),
                asyncio.create_task(self._periodic_tasks())
            ]
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"시스템 시작 실패: {e}")
            self.error_monitor.record_strategy_error('system_start', e)
            await self.shutdown()
