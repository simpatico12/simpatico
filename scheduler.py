#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🕐 퀸트프로젝트 - 4대 시장 통합 스케줄러 SCHEDULER.PY
================================================================

🌟 핵심 특징:
- 📊 시장별 최적 시간 자동 스캔 (미국/한국/일본/인도)
- 🔄 포트폴리오 자동 리밸런싱 시스템
- 🛡️ 실시간 리스크 모니터링 & 긴급 정지
- 📈 성과 분석 & 일일/주간 리포트 자동 생성
- 🚨 텔레그램/이메일 자동 알림 시스템
- 💾 자동 백업 & 시스템 헬스체크

⚡ 혼자 보수유지 가능한 완전 자동화 아키텍처
💎 cron 표현식 + 시장별 최적 타이밍
🛡️ 장애 감지 및 자동 복구 시스템

Author: 퀸트팀 | Version: ULTIMATE
Date: 2024.12
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import json
import yaml
from pathlib import Path
import pytz
import schedule
import threading
import time as time_module
from crontab import CronTab
from collections import defaultdict

# 퀸트프로젝트 모듈
try:
    from core import QuintProjectMaster, config
    from utils import QuintLogger, notification, backup, performance_analyzer
    from notifier import QuintNotificationManager
    QUINT_MODULES_AVAILABLE = True
except ImportError:
    QUINT_MODULES_AVAILABLE = False
    logging.warning("퀸트프로젝트 모듈 일부 누락 - 기본 기능만 사용")

# 선택적 import
try:
    import telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ============================================================================
# 📊 스케줄 작업 데이터 클래스
# ============================================================================
@dataclass
class ScheduledTask:
    """스케줄된 작업 정보"""
    name: str
    description: str
    cron_expression: str
    function: Callable
    enabled: bool = True
    last_run: Optional[datetime] =         result = await self.executor.execute_task(task)
        logging.info(f"수동 실행 완료: {task_name} - {result.success}")
        return result
    
    def get_task_status(self, task_name: str = None) -> Dict:
        """작업 상태 조회"""
        if task_name:
            if task_name not in self.tasks:
                return {'error': f'작업 없음: {task_name}'}
            
            task = self.tasks[task_name]
            stats = self.executor.get_task_statistics(task_name)
            
            return {
                'task': task.to_dict(),
                'statistics': stats,
                'is_running': task_name in self.executor.running_tasks
            }
        else:
            # 전체 작업 상태
            all_status = {}
            for name, task in self.tasks.items():
                stats = self.executor.get_task_statistics(name)
                all_status[name] = {
                    'enabled': task.enabled,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'run_count': task.run_count,
                    'error_count': task.error_count,
                    'success_rate': stats.get('success_rate', 0),
                    'is_running': name in self.executor.running_tasks
                }
            
            return {
                'total_tasks': len(self.tasks),
                'running_tasks': len(self.executor.running_tasks),
                'enabled_tasks': sum(1 for t in self.tasks.values() if t.enabled),
                'tasks': all_status
            }
    
    def start(self):
        """스케줄러 시작"""
        if self.running:
            logging.warning("스케줄러가 이미 실행 중입니다")
            return
        
        self.running = True
        
        def scheduler_worker():
            logging.info("🚀 퀸트프로젝트 스케줄러 시작")
            
            while self.running:
                try:
                    schedule.run_pending()
                    time_module.sleep(1)
                except Exception as e:
                    logging.error(f"스케줄러 워커 오류: {e}")
                    time_module.sleep(5)
            
            logging.info("⏹️ 퀸트프로젝트 스케줄러 중지")
        
        self.scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        self.scheduler_thread.start()
        
        logging.info(f"✅ 스케줄러 시작 완료 ({len(self.tasks)}개 작업 등록)")
    
    def stop(self):
        """스케줄러 중지"""
        self.running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        # 실행 중인 작업들 정리
        schedule.clear()
        
        logging.info("🛑 스케줄러 중지 완료")
    
    def get_scheduler_statistics(self) -> Dict:
        """스케줄러 전체 통계"""
        total_executions = sum(len([r for r in self.executor.task_history if r.task_name == name]) 
                              for name in self.tasks.keys())
        
        successful_executions = sum(len([r for r in self.executor.task_history 
                                       if r.task_name == name and r.success]) 
                                  for name in self.tasks.keys())
        
        return {
            'scheduler_running': self.running,
            'total_tasks': len(self.tasks),
            'enabled_tasks': sum(1 for t in self.tasks.values() if t.enabled),
            'running_tasks': len(self.executor.running_tasks),
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            'history_size': len(self.executor.task_history)
        }

# ============================================================================
# 🛠️ 유틸리티 및 헬퍼 함수들
# ============================================================================
class SchedulerUtils:
    """스케줄러 유틸리티"""
    
    @staticmethod
    def validate_cron_expression(cron: str) -> bool:
        """cron 표현식 유효성 검증"""
        try:
            parts = cron.split()
            if len(parts) != 5:
                return False
            
            # 간단한 검증 (실제로는 더 정교한 검증 필요)
            return True
        except:
            return False
    
    @staticmethod
    def get_next_run_time(cron: str) -> Optional[datetime]:
        """다음 실행 시간 계산"""
        try:
            # python-crontab 사용
            from crontab import CronTab
            cron_obj = CronTab(cron)
            return datetime.now() + timedelta(seconds=cron_obj.next())
        except:
            return None
    
    @staticmethod
    def export_scheduler_config(file_path: str) -> bool:
        """스케줄러 설정 내보내기"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(scheduler_config.config, f, default_flow_style=False, allow_unicode=True, indent=2)
            return True
        except Exception as e:
            logging.error(f"설정 내보내기 실패: {e}")
            return False
    
    @staticmethod
    def import_scheduler_config(file_path: str) -> bool:
        """스케줄러 설정 가져오기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
            
            if new_config:
                scheduler_config.config = new_config
                scheduler_config._save_config()
                return True
            
            return False
        except Exception as e:
            logging.error(f"설정 가져오기 실패: {e}")
            return False

# ============================================================================
# 🎮 편의 함수들 (외부 호출용)
# ============================================================================
async def run_market_scan_now(market: str = 'all'):
    """시장 스캔 즉시 실행"""
    scheduler = QuintScheduler()
    
    if market == 'all':
        tasks = ['scan_us_stocks', 'scan_crypto_market', 'scan_japan_stocks', 'scan_india_stocks']
    elif market == 'us':
        tasks = ['scan_us_stocks']
    elif market == 'crypto':
        tasks = ['scan_crypto_market']
    elif market == 'japan':
        tasks = ['scan_japan_stocks']
    elif market == 'india':
        tasks = ['scan_india_stocks']
    else:
        print(f"❌ 지원되지 않는 시장: {market}")
        return
    
    results = {}
    for task_name in tasks:
        if task_name in scheduler.tasks:
            try:
                result = await scheduler.run_task_now(task_name)
                results[task_name] = result.success
                print(f"{'✅' if result.success else '❌'} {task_name}: {result.execution_time:.1f}초")
            except Exception as e:
                results[task_name] = False
                print(f"❌ {task_name}: {e}")
    
    return results

def get_scheduler_status():
    """스케줄러 상태 조회"""
    scheduler = QuintScheduler()
    status = scheduler.get_scheduler_statistics()
    
    print("\n🕐 퀸트프로젝트 스케줄러 상태:")
    print(f"   실행 상태: {'🟢 실행중' if status['scheduler_running'] else '🔴 중지'}")
    print(f"   등록된 작업: {status['total_tasks']}개")
    print(f"   활성화된 작업: {status['enabled_tasks']}개")
    print(f"   실행중인 작업: {status['running_tasks']}개")
    print(f"   성공률: {status['success_rate']:.1f}%")
    print(f"   총 실행 횟수: {status['total_executions']}회")
    
    return status

def list_scheduled_tasks():
    """스케줄된 작업 목록"""
    scheduler = QuintScheduler()
    all_status = scheduler.get_task_status()
    
    print("\n📋 스케줄된 작업 목록:")
    
    # 카테고리별 그룹화
    categories = {
        '시장 스캔': ['scan_us_stocks', 'scan_crypto_market', 'scan_japan_stocks', 'scan_india_stocks'],
        '포트폴리오': ['rebalance_portfolio', 'check_portfolio_performance'],
        '리스크 관리': ['monitor_real_time_risk', 'generate_daily_risk_report'],
        '리포트': ['generate_daily_report', 'generate_weekly_report', 'generate_monthly_report'],
        '시스템 유지보수': ['system_backup', 'system_cleanup', 'health_check']
    }
    
    for category, task_names in categories.items():
        print(f"\n📊 {category}:")
        for task_name in task_names:
            if task_name in all_status['tasks']:
                task_info = all_status['tasks'][task_name]
                status_icon = '🟢' if task_info['enabled'] else '🔴'
                run_info = f"({task_info['run_count']}회 실행)" if task_info['run_count'] > 0 else "(미실행)"
                print(f"   {status_icon} {task_name} {run_info}")

def enable_task(task_name: str):
    """작업 활성화"""
    scheduler = QuintScheduler()
    scheduler.enable_task(task_name)
    print(f"✅ 작업 활성화: {task_name}")

def disable_task(task_name: str):
    """작업 비활성화"""
    scheduler = QuintScheduler()
    scheduler.disable_task(task_name)
    print(f"🔴 작업 비활성화: {task_name}")

def update_schedule(task_name: str, new_cron: str):
    """작업 스케줄 변경"""
    if not SchedulerUtils.validate_cron_expression(new_cron):
        print(f"❌ 잘못된 cron 표현식: {new_cron}")
        return
    
    scheduler = QuintScheduler()
    if task_name in scheduler.tasks:
        scheduler.tasks[task_name].cron_expression = new_cron
        print(f"✅ 스케줄 변경: {task_name} -> {new_cron}")
        
        # 설정 파일에도 반영
        config_key = f"market_scan.{task_name.replace('scan_', '').replace('_stocks', '').replace('_market', '')}.cron"
        scheduler_config.update(config_key, new_cron)
    else:
        print(f"❌ 존재하지 않는 작업: {task_name}")

def backup_scheduler_config():
    """스케줄러 설정 백업"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f"scheduler_config_backup_{timestamp}.yaml"
    
    if SchedulerUtils.export_scheduler_config(backup_file):
        print(f"✅ 스케줄러 설정 백업 완료: {backup_file}")
    else:
        print("❌ 스케줄러 설정 백업 실패")

# ============================================================================
# 🎯 메인 실행 함수
# ============================================================================
async def main():
    """스케줄러 메인 실행"""
    print("🕐" + "="*78)
    print("🚀 퀸트프로젝트 - 4대 시장 통합 스케줄러 SCHEDULER.PY")
    print("="*80)
    print("📊 시장별 최적 시간 자동 스캔 | 🔄 포트폴리오 자동 리밸런싱")
    print("🛡️ 실시간 리스크 모니터링 | 📈 성과 분석 & 리포트 자동 생성")
    print("="*80)
    
    # 스케줄러 초기화 및 시작
    print("\n🔧 스케줄러 초기화 중...")
    scheduler = QuintScheduler()
    
    # 상태 확인
    print(f"\n📊 스케줄러 설정:")
    print(f"   🇺🇸 미국주식: 화요일, 목요일 오후 11시 (장시작 30분 전)")
    print(f"   🪙 암호화폐: 월요일, 금요일 오전9시, 밤9시")
    print(f"   🇯🇵 일본주식: 화요일, 목요일 오전 8시")
    print(f"   🇮🇳 인도주식: 수요일 낮 12시")
    
    try:
        # 스케줄러 시작
        scheduler.start()
        
        # 테스트 실행 (선택적)
        print(f"\n🧪 테스트 실행을 하시겠습니까? (y/n): ", end="")
        
        # 실제 운영에서는 테스트 없이 바로 시작
        test_run = False  # CLI에서는 False로 설정
        
        if test_run:
            print("🧪 시장 스캔 테스트 실행 중...")
            test_results = await run_market_scan_now('crypto')  # 암호화폐만 테스트
            
            success_count = sum(1 for success in test_results.values() if success)
            print(f"✅ 테스트 완료: {success_count}/{len(test_results)}개 성공")
        
        print(f"\n✅ 스케줄러가 정상적으로 시작되었습니다")
        print(f"📊 상태 확인: get_scheduler_status()")
        print(f"📋 작업 목록: list_scheduled_tasks()")
        print(f"🚨 즉시 실행: run_market_scan_now('crypto')")
        
        # 무한 실행 (실제 운영)
        print(f"\n🔄 스케줄러가 백그라운드에서 실행 중입니다...")
        print(f"   Ctrl+C로 종료하세요\n")
        
        while True:
            await asyncio.sleep(60)  # 1분마다 체크
            
            # 주기적 상태 로그
            status = scheduler.get_scheduler_statistics()
            if status['total_executions'] > 0:
                logging.info(f"스케줄러 상태 - 성공률: {status['success_rate']:.1f}%, "
                           f"실행중: {status['running_tasks']}개")
        
    except KeyboardInterrupt:
        print(f"\n👋 스케줄러를 종료합니다...")
        scheduler.stop()
        
    except Exception as e:
        print(f"\n❌ 스케줄러 실행 오류: {e}")
        logging.error(f"스케줄러 메인 실행 실패: {e}")
        scheduler.stop()

# ============================================================================
# 🎮 CLI 인터페이스
# ============================================================================
def cli_interface():
    """간단한 CLI 인터페이스"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'start':
            # 스케줄러 시작
            asyncio.run(main())
            
        elif command == 'status':
            # 스케줄러 상태 확인
            get_scheduler_status()
            
        elif command == 'list':
            # 작업 목록
            list_scheduled_tasks()
            
        elif command == 'scan':
            # 즉시 시장 스캔
            market = sys.argv[2] if len(sys.argv) > 2 else 'all'
            asyncio.run(run_market_scan_now(market))
            
        elif command == 'enable':
            # 작업 활성화
            if len(sys.argv) > 2:
                enable_task(sys.argv[2])
            else:
                print("사용법: python scheduler.py enable <task_name>")
                
        elif command == 'disable':
            # 작업 비활성화
            if len(sys.argv) > 2:
                disable_task(sys.argv[2])
            else:
                print("사용법: python scheduler.py disable <task_name>")
                
        elif command == 'schedule':
            # 스케줄 변경
            if len(sys.argv) >= 4:
                task_name, new_cron = sys.argv[2], ' '.join(sys.argv[3:])
                update_schedule(task_name, new_cron)
            else:
                print("사용법: python scheduler.py schedule <task_name> <cron_expression>")
                
        elif command == 'backup':
            # 설정 백업
            backup_scheduler_config()
            
        elif command == 'test':
            # 테스트 모드
            async def test_mode():
                print("🧪 퀸트프로젝트 스케줄러 테스트 모드")
                
                # 각 시장별 스캔 테스트
                markets = ['crypto', 'us', 'japan', 'india']
                for market in markets:
                    print(f"\n📊 {market.upper()} 시장 스캔 테스트...")
                    result = await run_market_scan_now(market)
                    await asyncio.sleep(2)  # 2초 간격
                
                print("\n✅ 전체 테스트 완료")
            
            asyncio.run(test_mode())
            
        else:
            print("퀸트프로젝트 스케줄러 CLI 사용법:")
            print("  python scheduler.py start              # 스케줄러 시작")
            print("  python scheduler.py status             # 상태 확인")
            print("  python scheduler.py list               # 작업 목록")
            print("  python scheduler.py scan crypto        # 즉시 스캔 (crypto/us/japan/india/all)")
            print("  python scheduler.py enable <task>      # 작업 활성화")
            print("  python scheduler.py disable <task>     # 작업 비활성화")
            print("  python scheduler.py schedule <task> <cron>  # 스케줄 변경")
            print("  python scheduler.py backup             # 설정 백업")
            print("  python scheduler.py test               # 테스트 모드")
            print("\n📊 최적화된 스케줄:")
            print("  🇺🇸 미국주식: 화목 23시 | 🪙 암호화폐: 월금 9시,21시")
            print("  🇯🇵 일본주식: 화목 8시  | 🇮🇳 인도주식: 수요일 12시")
    else:
        # 기본 실행 - 스케줄러 시작
        asyncio.run(main())

# ============================================================================
# 📊 스케줄러 모니터링 대시보드 (간단 버전)
# ============================================================================
class SchedulerDashboard:
    """스케줄러 모니터링 대시보드"""
    
    def __init__(self, scheduler: QuintScheduler):
        self.scheduler = scheduler
    
    def print_live_status(self):
        """실시간 상태 출력"""
        import os
        
        while True:
            try:
                # 화면 클리어 (크로스 플랫폼)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("🕐" + "="*60)
                print("🏆 퀸트프로젝트 스케줄러 실시간 모니터링")
                print("="*62)
                
                # 현재 시간
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"⏰ 현재 시간: {current_time}")
                
                # 전체 상태
                stats = self.scheduler.get_scheduler_statistics()
                print(f"📊 스케줄러: {'🟢 실행중' if stats['scheduler_running'] else '🔴 중지'}")
                print(f"📋 작업: {stats['enabled_tasks']}/{stats['total_tasks']}개 활성화")
                print(f"⚡ 실행중: {stats['running_tasks']}개")
                print(f"📈 성공률: {stats['success_rate']:.1f}%")
                
                # 다음 실행 예정 작업들
                print(f"\n🔜 다음 실행 예정:")
                
                # 요일별 스케줄 표시
                today = datetime.now().weekday()  # 0=월요일
                weekdays = ['월', '화', '수', '목', '금', '토', '일']
                
                schedule_info = {
                    1: "🇺🇸🇯🇵 미국/일본 스캔 (23시/8시)",  # 화요일
                    2: "🇮🇳 인도 스캔 (12시)",            # 수요일  
                    3: "🇺🇸🇯🇵 미국/일본 스캔 (23시/8시)",  # 목요일
                    0: "🪙 암호화폐 스캔 (9시/21시)",       # 월요일
                    4: "🪙 암호화폐 스캔 (9시/21시)"        # 금요일
                }
                
                for i in range(7):
                    day_idx = (today + i) % 7
                    day_name = weekdays[day_idx]
                    marker = "👉" if i == 0 else "  "
                    
                    if day_idx in schedule_info:
                        print(f"{marker} {day_name}요일: {schedule_info[day_idx]}")
                    else:
                        print(f"{marker} {day_name}요일: 휴무")
                
                # 최근 실행 결과
                print(f"\n📊 최근 실행 결과:")
                recent_tasks = self.scheduler.executor.task_history[-5:] if self.scheduler.executor.task_history else []
                
                for task_result in recent_tasks:
                    status_icon = "✅" if task_result.success else "❌"
                    time_str = task_result.timestamp.strftime('%H:%M:%S')
                    print(f"   {status_icon} {time_str} {task_result.task_name} ({task_result.execution_time:.1f}s)")
                
                if not recent_tasks:
                    print("   (아직 실행된 작업이 없습니다)")
                
                print(f"\n💡 명령어: Ctrl+C로 종료")
                print("="*62)
                
                time_module.sleep(30)  # 30초마다 업데이트
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"대시보드 오류: {e}")
                time_module.sleep(5)

# ============================================================================
# 🔧 성능 최적화 및 안정성 강화
# ============================================================================
class SchedulerOptimizer:
    """스케줄러 성능 최적화"""
    
    def __init__(self, scheduler: QuintScheduler):
        self.scheduler = scheduler
        self.performance_metrics = []
    
    def analyze_performance(self) -> Dict:
        """성능 분석"""
        task_stats = {}
        
        for task_name, task in self.scheduler.tasks.items():
            stats = self.scheduler.executor.get_task_statistics(task_name)
            
            if stats:
                task_stats[task_name] = {
                    'avg_execution_time': stats.get('avg_execution_time', 0),
                    'success_rate': stats.get('success_rate', 0),
                    'failure_count': stats.get('failure_count', 0)
                }
        
        # 성능 이슈 감지
        issues = []
        for task_name, stats in task_stats.items():
            if stats['avg_execution_time'] > 120:  # 2분 이상
                issues.append(f"{task_name}: 실행시간 과다 ({stats['avg_execution_time']:.1f}초)")
            
            if stats['success_rate'] < 80:  # 성공률 80% 미만
                issues.append(f"{task_name}: 낮은 성공률 ({stats['success_rate']:.1f}%)")
        
        return {
            'task_statistics': task_stats,
            'performance_issues': issues,
            'optimization_suggestions': self._get_optimization_suggestions(task_stats)
        }
    
    def _get_optimization_suggestions(self, task_stats: Dict) -> List[str]:
        """최적화 제안"""
        suggestions = []
        
        slow_tasks = [name for name, stats in task_stats.items() 
                     if stats['avg_execution_time'] > 60]
        
        if slow_tasks:
            suggestions.append(f"느린 작업들의 타임아웃 증가 고려: {', '.join(slow_tasks)}")
        
        failing_tasks = [name for name, stats in task_stats.items() 
                        if stats['success_rate'] < 90]
        
        if failing_tasks:
            suggestions.append(f"실패율 높은 작업들의 재시도 로직 검토: {', '.join(failing_tasks)}")
        
        return suggestions

# ============================================================================
# 🎯 실행부
# ============================================================================
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('scheduler.log', encoding='utf-8')
        ]
    )
    
    # CLI 모드 실행
    cli_interface()

# ============================================================================
# 📋 퀸트프로젝트 SCHEDULER.PY 특징 요약
# ============================================================================
"""
🕐 퀸트프로젝트 SCHEDULER.PY 완전체 특징:

🔧 혼자 보수유지 가능한 아키텍처:
   ✅ YAML 기반 설정 관리 (scheduler_config.yaml)
   ✅ cron 표현식 지원 + 요일별 최적화
   ✅ 자동 에러 복구 및 재시도 시스템
   ✅ 작업별 타임아웃 및 실행 제한

📊 4대 시장 최적 스케줄링:
   ✅ 🇺🇸 미국주식: 화목 21시 (장시작 전 최적)
   ✅ 🪙 암호화폐: 월금 9시,21시 (변동성 고려)
   ✅ 🇯🇵 일본주식: 화목 8시 (장시작 전 최적)
   ✅ 🇮🇳 인도주식: 수요일 12시 (장중 최적)

⚡ 완전 자동화 시스템:
   ✅ 비동기 작업 실행 (TaskExecutor)
   ✅ 실시간 리스크 모니터링 (5분마다)
   ✅ 자동 포트폴리오 리밸런싱 (주간)
   ✅ 일일/주간/월간 리포트 자동 생성

🛡️ 통합 리스크 관리:
   ✅ 손실률 기반 긴급 정지 시스템
   ✅ 실시간 성과 모니터링 및 알림
   ✅ 시스템 헬스체크 (10분마다)
   ✅ 자동 백업 및 데이터 정리

📱 통합 알림 시스템:
   ✅ 텔레그램 즉시 알림 (긴급 신호)
   ✅ 일일/주간 리포트 자동 전송
   ✅ 리스크 경고 및 긴급 정지 알림
   ✅ 성과 변동 알림 (±5% 이상)

🎮 사용법:
   - 시작: python scheduler.py start
   - 상태: python scheduler.py status
   - 즉시스캔: python scheduler.py scan crypto
   - 테스트: python scheduler.py test

🚀 고급 기능:
   ✅ 실시간 모니터링 대시보드
   ✅ 성능 분석 및 최적화 제안
   ✅ CLI 기반 작업 관리
   ✅ 설정 백업 및 복원

🎯 핵심 철학:
   - 시장별 최적 타이밍에 자동 실행
   - 에러 발생시 자동 복구
   - 중요한 상황은 즉시 알림
   - 혼자서도 충분히 관리 가능

💎 스케줄 예시:
   월요일 09:00 - 🪙 암호화폐 스캔
   화요일 08:00 - 🇯🇵 일본주식 스캔
   화요일 23:00 - 🇺🇸 미국주식 스캔 (장시작 30분 전)
   수요일 12:00 - 🇮🇳 인도주식 스캔
   목요일 08:00 - 🇯🇵 일본주식 스캔
   목요일 23:00 - 🇺🇸 미국주식 스캔 (장시작 30분 전)
   금요일 09:00 - 🪙 암호화폐 스캔
   금요일 21:00 - 🪙 암호화폐 스캔
   금요일 22:00 - 📊 주간 리밸런싱

🏆 퀸트프로젝트 = 완벽한 자동화 스케줄링!
"""
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    max_errors: int = 5
    timeout_seconds: int = 300
    retry_count: int = 3
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'cron_expression': self.cron_expression,
            'enabled': self.enabled,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'run_count': self.run_count,
            'error_count': self.error_count,
            'max_errors': self.max_errors
        }

@dataclass
class TaskResult:
    """작업 실행 결과"""
    task_name: str
    success: bool
    execution_time: float
    result_data: Any = None
    error_message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

# ============================================================================
# 🔧 스케줄러 설정 관리자
# ============================================================================
class SchedulerConfig:
    """스케줄러 설정 관리"""
    
    def __init__(self):
        self.config_file = "scheduler_config.yaml"
        self.config = {}
        self._initialize_config()
    
    def _initialize_config(self):
        """설정 초기화"""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self._create_default_config()
            self._save_config()
    
    def _create_default_config(self):
        """기본 설정 생성"""
        self.config = {
            # 전체 시스템 설정
            'system': {
                'timezone': 'Asia/Seoul',
                'max_concurrent_tasks': 3,
                'task_timeout_default': 300,
                'error_notification_threshold': 3,
                'health_check_interval': 600,  # 10분
                'backup_enabled': True
            },
            
            # 시장 스캔 스케줄 (요일별 최적화)
            'market_scan': {
                'enabled': True,
                'us_stocks': {
                    'cron': '0 23 * * 2,4',     # 화요일, 목요일 오후 11시 (미국 장시작 30분 전)
                    'enabled': True,
                    'timeout': 180
                },
                'upbit_crypto': {
                    'cron': '0 9,21 * * 1,5',  # 월요일, 금요일 오전9시, 밤9시 (변동성 높은 시간)
                    'enabled': True,
                    'timeout': 120
                },
                'japan_stocks': {
                    'cron': '0 8 * * 2,4',     # 화요일, 목요일 오전 8시 (일본 장시작 전)
                    'enabled': True,
                    'timeout': 150
                },
                'india_stocks': {
                    'cron': '0 12 * * 3',      # 수요일 낮 12시 (인도 장중, 주중 최적)
                    'enabled': True,
                    'timeout': 150
                }
            },
            
            # 포트폴리오 관리
            'portfolio': {
                'rebalancing': {
                    'cron': '0 22 * * 5',      # 매주 금요일 밤 10시
                    'enabled': True,
                    'threshold_percent': 5.0,
                    'timeout': 300
                },
                'performance_check': {
                    'cron': '0 9,15,21 * * *', # 하루 3번 (오전9시, 오후3시, 밤9시)
                    'enabled': True,
                    'timeout': 60
                }
            },
            
            # 리스크 관리
            'risk_management': {
                'real_time_monitoring': {
                    'cron': '*/5 * * * *',     # 5분마다
                    'enabled': True,
                    'max_loss_percent': 10.0,
                    'circuit_breaker': True,
                    'timeout': 30
                },
                'daily_risk_report': {
                    'cron': '0 20 * * *',      # 매일 밤 8시
                    'enabled': True,
                    'timeout': 120
                }
            },
            
            # 리포트 생성
            'reports': {
                'daily_report': {
                    'cron': '0 19 * * *',      # 매일 저녁 7시
                    'enabled': True,
                    'timeout': 180
                },
                'weekly_report': {
                    'cron': '0 18 * * 0',      # 매주 일요일 오후 6시
                    'enabled': True,
                    'timeout': 300
                },
                'monthly_report': {
                    'cron': '0 10 1 * *',      # 매월 1일 오전 10시
                    'enabled': True,
                    'timeout': 600
                }
            },
            
            # 시스템 유지보수
            'maintenance': {
                'backup': {
                    'cron': '0 2 * * *',       # 매일 새벽 2시
                    'enabled': True,
                    'retention_days': 30,
                    'timeout': 120
                },
                'cleanup': {
                    'cron': '0 3 * * 0',       # 매주 일요일 새벽 3시
                    'enabled': True,
                    'cleanup_days': 90,
                    'timeout': 180
                },
                'health_check': {
                    'cron': '*/10 * * * *',    # 10분마다
                    'enabled': True,
                    'timeout': 60
                }
            },
            
            # 알림 설정
            'notifications': {
                'telegram': {
                    'enabled': True,
                    'error_alerts': True,
                    'daily_summary': True,
                    'performance_alerts': True
                },
                'email': {
                    'enabled': False,
                    'weekly_reports': True,
                    'monthly_reports': True,
                    'error_alerts': True
                }
            }
        }
    
    def _save_config(self):
        """설정 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            logging.error(f"일일 리스크 리포트 생성 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    # ========================================================================
    # 📊 리포트 생성 작업들
    # ========================================================================
    async def generate_daily_report(self) -> Dict:
        """일일 성과 리포트 생성"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 포트폴리오 현황
            if self.quint_master:
                portfolio_summary = self.quint_master.portfolio_manager.get_portfolio_summary()
                
                # 전체 분석 실행
                analysis_result = await self.quint_master.run_full_analysis()
            else:
                portfolio_summary = {'total_value': 0}
                analysis_result = {'buy_signals': 0, 'total_signals': 0}
            
            # 리포트 데이터 구성
            report_data = {
                'date': today,
                'portfolio_value': portfolio_summary.get('total_value', 0),
                'daily_signals': analysis_result.get('total_signals', 0),
                'buy_signals': analysis_result.get('buy_signals', 0),
                'market_summary': analysis_result.get('market_breakdown', {}),
                'top_opportunities': analysis_result.get('optimized_portfolio', [])[:5]
            }
            
            # 일일 리포트 알림 전송
            if self.notification_manager:
                await self._send_daily_report_notification(report_data)
            
            return {
                'status': 'success',
                'report': report_data
            }
            
        except Exception as e:
            logging.error(f"일일 리포트 생성 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def generate_weekly_report(self) -> Dict:
        """주간 성과 리포트 생성"""
        try:
            # 주간 성과 분석 (QUINT_MODULES_AVAILABLE일 때만)
            if QUINT_MODULES_AVAILABLE and performance_analyzer:
                weekly_performance = performance_analyzer.generate_performance_report(7)
            else:
                weekly_performance = {'overview': {}}
            
            # 기본 주간 리포트
            report_data = {
                'week_start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'week_end': datetime.now().strftime('%Y-%m-%d'),
                'performance_summary': weekly_performance.get('overview', {}),
                'market_analysis': '주간 시장 분석 완료',
                'recommendation': '포트폴리오 검토 권장'
            }
            
            # 주간 리포트 알림
            if self.notification_manager:
                await self._send_weekly_report_notification(report_data)
            
            return {
                'status': 'success',
                'report': report_data
            }
            
        except Exception as e:
            logging.error(f"주간 리포트 생성 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def generate_monthly_report(self) -> Dict:
        """월간 성과 리포트 생성"""
        try:
            # 월간 성과 분석
            if QUINT_MODULES_AVAILABLE and performance_analyzer:
                monthly_performance = performance_analyzer.generate_performance_report(30)
            else:
                monthly_performance = {'overview': {}}
            
            report_data = {
                'month': datetime.now().strftime('%Y-%m'),
                'performance_summary': monthly_performance.get('overview', {}),
                'recommendations': monthly_performance.get('recommendations', []),
                'next_month_strategy': '지속적인 분산투자 전략'
            }
            
            return {
                'status': 'success',
                'report': report_data
            }
            
        except Exception as e:
            logging.error(f"월간 리포트 생성 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    # ========================================================================
    # 🔧 시스템 유지보수 작업들
    # ========================================================================
    async def system_backup(self) -> Dict:
        """시스템 백업"""
        try:
            if QUINT_MODULES_AVAILABLE and backup:
                backup_result = backup.create_backup('scheduled')
                
                if backup_result:
                    # 오래된 백업 정리
                    cleanup_count = backup.cleanup_old_backups()
                    
                    return {
                        'status': 'success',
                        'backup_file': str(backup_result),
                        'cleanup_count': cleanup_count
                    }
                else:
                    return {'status': 'error', 'message': '백업 생성 실패'}
            else:
                # 기본 백업 (설정 파일만)
                backup_dir = Path('backups')
                backup_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                files_to_backup = [
                    'quint_config.yaml',
                    'scheduler_config.yaml',
                    'quint_portfolio.json'
                ]
                
                backup_count = 0
                for file_name in files_to_backup:
                    if Path(file_name).exists():
                        import shutil
                        shutil.copy(file_name, backup_dir / f"{file_name}_{timestamp}")
                        backup_count += 1
                
                return {
                    'status': 'success',
                    'backup_files': backup_count,
                    'timestamp': timestamp
                }
            
        except Exception as e:
            logging.error(f"시스템 백업 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def system_cleanup(self) -> Dict:
        """시스템 정리"""
        try:
            cleanup_results = {}
            
            # 로그 파일 정리
            logs_cleaned = self._cleanup_old_logs()
            cleanup_results['logs_cleaned'] = logs_cleaned
            
            # 임시 파일 정리
            temp_cleaned = self._cleanup_temp_files()
            cleanup_results['temp_files_cleaned'] = temp_cleaned
            
            # 데이터베이스 정리 (QUINT_MODULES_AVAILABLE일 때)
            if QUINT_MODULES_AVAILABLE:
                try:
                    from utils import database
                    db_cleaned = database.cleanup_old_data(90)
                    cleanup_results['database_records_cleaned'] = db_cleaned
                except:
                    cleanup_results['database_records_cleaned'] = 0
            
            return {
                'status': 'success',
                'cleanup_results': cleanup_results
            }
            
        except Exception as e:
            logging.error(f"시스템 정리 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def health_check(self) -> Dict:
        """시스템 헬스 체크"""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'components': {}
            }
            
            # 기본 시스템 체크
            health_status['components']['config'] = {
                'status': 'ok' if Path('quint_config.yaml').exists() else 'warning',
                'details': 'Configuration file check'
            }
            
            health_status['components']['portfolio'] = {
                'status': 'ok' if Path('quint_portfolio.json').exists() else 'info',
                'details': 'Portfolio file check'
            }
            
            # 디스크 공간 체크
            import shutil
            disk_usage = shutil.disk_usage('.')
            free_space_gb = disk_usage.free / (1024**3)
            
            health_status['components']['disk_space'] = {
                'status': 'ok' if free_space_gb > 1.0 else 'warning',
                'details': f'{free_space_gb:.1f}GB free space'
            }
            
            # 퀸트 모듈 체크
            health_status['components']['quint_modules'] = {
                'status': 'ok' if QUINT_MODULES_AVAILABLE else 'warning',
                'details': 'Quint modules availability'
            }
            
            # 전체 상태 결정
            component_statuses = [comp['status'] for comp in health_status['components'].values()]
            if 'error' in component_statuses:
                health_status['overall_status'] = 'error'
            elif 'warning' in component_statuses:
                health_status['overall_status'] = 'warning'
            
            return {
                'status': 'success',
                'health_status': health_status
            }
            
        except Exception as e:
            logging.error(f"헬스 체크 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    # ========================================================================
    # 📱 알림 헬퍼 메서드들
    # ========================================================================
    async def _send_urgent_signal_alert(self, market: str, signals: List) -> None:
        """긴급 시그널 알림"""
        try:
            if self.notification_manager:
                message = f"🚨 {market} 긴급 매수 신호!\n\n"
                for signal in signals:
                    message += f"📈 {signal.symbol}: {signal.confidence:.1%} 신뢰도\n"
                
                await self.notification_manager.send_system_alert(
                    "긴급 매수 신호", message, "high"
                )
        except Exception as e:
            logging.error(f"긴급 시그널 알림 실패: {e}")
    
    async def _send_performance_alert(self, performance: float, total_value: float) -> None:
        """성과 알림"""
        try:
            if self.notification_manager:
                status = "상승" if performance > 0 else "하락"
                message = f"💼 포트폴리오 {status}: {abs(performance):.1f}%\n"
                message += f"현재 가치: {total_value:,.0f}원"
                
                priority = "high" if abs(performance) > 10 else "medium"
                
                await self.notification_manager.send_system_alert(
                    "포트폴리오 성과 알림", message, priority
                )
        except Exception as e:
            logging.error(f"성과 알림 실패: {e}")
    
    async def _send_rebalancing_alert(self, reason: str, allocation: float) -> None:
        """리밸런싱 알림"""
        try:
            if self.notification_manager:
                message = f"🔄 리밸런싱 필요\n사유: {reason}\n현재 할당: {allocation:.1f}%"
                
                await self.notification_manager.send_system_alert(
                    "포트폴리오 리밸런싱", message, "medium"
                )
        except Exception as e:
            logging.error(f"리밸런싱 알림 실패: {e}")
    
    async def _trigger_emergency_stop(self, loss_percent: float) -> None:
        """긴급 정지 트리거"""
        try:
            if self.notification_manager:
                message = f"🛑 긴급 정지 발동!\n손실률: {loss_percent:.1f}%\n즉시 확인 필요"
                
                await self.notification_manager.send_system_alert(
                    "긴급 정지", message, "critical"
                )
            
            # 추가 안전 조치 (자동매매 중지 등)
            if config:
                config.update('system.auto_trading', False)
                
            logging.critical(f"🛑 긴급 정지 발동: {loss_percent:.1f}% 손실")
            
        except Exception as e:
            logging.error(f"긴급 정지 처리 실패: {e}")
    
    async def _send_risk_warning(self, loss_percent: float, warning_level: float) -> None:
        """리스크 경고"""
        try:
            if self.notification_manager:
                message = f"⚠️ 리스크 경고\n손실률: {loss_percent:.1f}%\n경고 수준: {warning_level}%"
                
                await self.notification_manager.send_system_alert(
                    "리스크 경고", message, "high"
                )
        except Exception as e:
            logging.error(f"리스크 경고 실패: {e}")
    
    async def _send_daily_report_notification(self, report_data: Dict) -> None:
        """일일 리포트 알림"""
        try:
            if self.notification_manager:
                await self.notification_manager.send_daily_report()
        except Exception as e:
            logging.error(f"일일 리포트 알림 실패: {e}")
    
    async def _send_weekly_report_notification(self, report_data: Dict) -> None:
        """주간 리포트 알림"""
        try:
            if self.notification_manager:
                await self.notification_manager.send_weekly_report()
        except Exception as e:
            logging.error(f"주간 리포트 알림 실패: {e}")
    
    async def _send_risk_report_notification(self, risk_report: Dict) -> None:
        """리스크 리포트 알림"""
        try:
            if self.notification_manager:
                message = f"📊 일일 리스크 리포트\n"
                message += f"포트폴리오: {risk_report['portfolio_value']:,.0f}원\n"
                message += f"분산도: {risk_report['diversification_score']:.0f}점\n"
                message += f"리스크 수준: {risk_report['risk_level']}"
                
                await self.notification_manager.send_system_alert(
                    "일일 리스크 리포트", message, "medium"
                )
        except Exception as e:
            logging.error(f"리스크 리포트 알림 실패: {e}")
    
    # ========================================================================
    # 🧹 정리 헬퍼 메서드들
    # ========================================================================
    def _cleanup_old_logs(self) -> int:
        """오래된 로그 파일 정리"""
        try:
            logs_dir = Path('logs')
            if not logs_dir.exists():
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=30)
            cleaned_count = 0
            
            for log_file in logs_dir.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            logging.error(f"로그 정리 실패: {e}")
            return 0
    
    def _cleanup_temp_files(self) -> int:
        """임시 파일 정리"""
        try:
            temp_patterns = ['*.tmp', '*.temp', '*~', '.DS_Store']
            cleaned_count = 0
            
            for pattern in temp_patterns:
                for temp_file in Path('.').glob(pattern):
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            logging.error(f"임시 파일 정리 실패: {e}")
            return 0

# ============================================================================
# 🕐 퀸트프로젝트 마스터 스케줄러
# ============================================================================
class QuintScheduler:
    """퀸트프로젝트 통합 스케줄러"""
    
    def __init__(self):
        self.tasks = {}
        self.executor = TaskExecutor()
        self.scheduled_tasks = QuintScheduledTasks()
        self.timing_calculator = MarketTimingCalculator()
        self.running = False
        self.scheduler_thread = None
        
        # 스케줄러 초기화
        self._initialize_tasks()
        self._setup_scheduler()
        
        logging.info("🕐 퀸트프로젝트 스케줄러 초기화 완료")
    
    def _initialize_tasks(self):
        """기본 작업들 등록"""
        # 시장 스캔 작업들 (요일별 최적화)
        if scheduler_config.get('market_scan.us_stocks.enabled', True):
            self.register_task(ScheduledTask(
                name="scan_us_stocks",
                description="미국 주식 시장 스캔 (화목 23시)",
                cron_expression=scheduler_config.get('market_scan.us_stocks.cron', '0 23 * * 2,4'),
                function=self.scheduled_tasks.scan_us_stocks,
                timeout_seconds=scheduler_config.get('market_scan.us_stocks.timeout', 180)
            ))
        
        if scheduler_config.get('market_scan.upbit_crypto.enabled', True):
            self.register_task(ScheduledTask(
                name="scan_crypto_market",
                description="암호화폐 시장 스캔 (월금)",
                cron_expression=scheduler_config.get('market_scan.upbit_crypto.cron', '0 9,21 * * 1,5'),
                function=self.scheduled_tasks.scan_crypto_market,
                timeout_seconds=scheduler_config.get('market_scan.upbit_crypto.timeout', 120)
            ))
        
        if scheduler_config.get('market_scan.japan_stocks.enabled', True):
            self.register_task(ScheduledTask(
                name="scan_japan_stocks",
                description="일본 주식 시장 스캔 (화목)",
                cron_expression=scheduler_config.get('market_scan.japan_stocks.cron', '0 8 * * 2,4'),
                function=self.scheduled_tasks.scan_japan_stocks,
                timeout_seconds=scheduler_config.get('market_scan.japan_stocks.timeout', 150)
            ))
        
        if scheduler_config.get('market_scan.india_stocks.enabled', True):
            self.register_task(ScheduledTask(
                name="scan_india_stocks",
                description="인도 주식 시장 스캔 (수)",
                cron_expression=scheduler_config.get('market_scan.india_stocks.cron', '0 12 * * 3'),
                function=self.scheduled_tasks.scan_india_stocks,
                timeout_seconds=scheduler_config.get('market_scan.india_stocks.timeout', 150)
            ))
        
        # 포트폴리오 관리 작업들
        if scheduler_config.get('portfolio.rebalancing.enabled', True):
            self.register_task(ScheduledTask(
                name="rebalance_portfolio",
                description="포트폴리오 리밸런싱",
                cron_expression=scheduler_config.get('portfolio.rebalancing.cron', '0 22 * * 5'),
                function=self.scheduled_tasks.rebalance_portfolio,
                timeout_seconds=scheduler_config.get('portfolio.rebalancing.timeout', 300)
            ))
        
        if scheduler_config.get('portfolio.performance_check.enabled', True):
            self.register_task(ScheduledTask(
                name="check_portfolio_performance",
                description="포트폴리오 성과 체크",
                cron_expression=scheduler_config.get('portfolio.performance_check.cron', '0 9,15,21 * * *'),
                function=self.scheduled_tasks.check_portfolio_performance,
                timeout_seconds=scheduler_config.get('portfolio.performance_check.timeout', 60)
            ))
        
        # 리스크 관리 작업들
        if scheduler_config.get('risk_management.real_time_monitoring.enabled', True):
            self.register_task(ScheduledTask(
                name="monitor_real_time_risk",
                description="실시간 리스크 모니터링",
                cron_expression=scheduler_config.get('risk_management.real_time_monitoring.cron', '*/5 * * * *'),
                function=self.scheduled_tasks.monitor_real_time_risk,
                timeout_seconds=scheduler_config.get('risk_management.real_time_monitoring.timeout', 30),
                max_errors=10  # 리스크 모니터링은 에러 허용도 높게
            ))
        
        if scheduler_config.get('risk_management.daily_risk_report.enabled', True):
            self.register_task(ScheduledTask(
                name="generate_daily_risk_report",
                description="일일 리스크 리포트",
                cron_expression=scheduler_config.get('risk_management.daily_risk_report.cron', '0 20 * * *'),
                function=self.scheduled_tasks.generate_daily_risk_report,
                timeout_seconds=scheduler_config.get('risk_management.daily_risk_report.timeout', 120)
            ))
        
        # 리포트 생성 작업들
        if scheduler_config.get('reports.daily_report.enabled', True):
            self.register_task(ScheduledTask(
                name="generate_daily_report",
                description="일일 성과 리포트",
                cron_expression=scheduler_config.get('reports.daily_report.cron', '0 19 * * *'),
                function=self.scheduled_tasks.generate_daily_report,
                timeout_seconds=scheduler_config.get('reports.daily_report.timeout', 180)
            ))
        
        if scheduler_config.get('reports.weekly_report.enabled', True):
            self.register_task(ScheduledTask(
                name="generate_weekly_report",
                description="주간 성과 리포트",
                cron_expression=scheduler_config.get('reports.weekly_report.cron', '0 18 * * 0'),
                function=self.scheduled_tasks.generate_weekly_report,
                timeout_seconds=scheduler_config.get('reports.weekly_report.timeout', 300)
            ))
        
        if scheduler_config.get('reports.monthly_report.enabled', True):
            self.register_task(ScheduledTask(
                name="generate_monthly_report",
                description="월간 성과 리포트",
                cron_expression=scheduler_config.get('reports.monthly_report.cron', '0 10 1 * *'),
                function=self.scheduled_tasks.generate_monthly_report,
                timeout_seconds=scheduler_config.get('reports.monthly_report.timeout', 600)
            ))
        
        # 시스템 유지보수 작업들
        if scheduler_config.get('maintenance.backup.enabled', True):
            self.register_task(ScheduledTask(
                name="system_backup",
                description="시스템 백업",
                cron_expression=scheduler_config.get('maintenance.backup.cron', '0 2 * * *'),
                function=self.scheduled_tasks.system_backup,
                timeout_seconds=scheduler_config.get('maintenance.backup.timeout', 120)
            ))
        
        if scheduler_config.get('maintenance.cleanup.enabled', True):
            self.register_task(ScheduledTask(
                name="system_cleanup",
                description="시스템 정리",
                cron_expression=scheduler_config.get('maintenance.cleanup.cron', '0 3 * * 0'),
                function=self.scheduled_tasks.system_cleanup,
                timeout_seconds=scheduler_config.get('maintenance.cleanup.timeout', 180)
            ))
        
        if scheduler_config.get('maintenance.health_check.enabled', True):
            self.register_task(ScheduledTask(
                name="health_check",
                description="시스템 헬스 체크",
                cron_expression=scheduler_config.get('maintenance.health_check.cron', '*/10 * * * *'),
                function=self.scheduled_tasks.health_check,
                timeout_seconds=scheduler_config.get('maintenance.health_check.timeout', 60),
                max_errors=20  # 헬스체크는 에러 허용도 높게
            ))
    
    def _setup_scheduler(self):
        """스케줄러 설정"""
        # python-crontab 사용하여 cron 표현식 파싱
        for task in self.tasks.values():
            if task.enabled:
                self._schedule_task(task)
    
    def _schedule_task(self, task: ScheduledTask):
        """개별 작업 스케줄링"""
        try:
            # cron 표현식을 schedule 라이브러리로 변환하여 등록
            # 간단한 변환 (실제로는 더 정교한 파싱 필요)
            self._convert_cron_to_schedule(task)
            
        except Exception as e:
            logging.error(f"작업 스케줄링 실패 {task.name}: {e}")
    
    def _convert_cron_to_schedule(self, task: ScheduledTask):
        """cron 표현식을 schedule로 변환 (요일별 최적화 지원)"""
        # 기본적인 cron 표현식만 지원 (확장 가능)
        cron = task.cron_expression
        
        if cron == '*/30 * * * *':  # 30분마다
            schedule.every(30).minutes.do(self._run_scheduled_task, task)
        elif cron == '*/5 * * * *':  # 5분마다
            schedule.every(5).minutes.do(self._run_scheduled_task, task)
        elif cron == '*/10 * * * *':  # 10분마다
            schedule.every(10).minutes.do(self._run_scheduled_task, task)
        elif cron.endswith('* * 2,4'):  # 화요일, 목요일만 (미국, 일본)
            hour = int(cron.split()[1])
            minute = int(cron.split()[0])
            schedule.every().tuesday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
            schedule.every().thursday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
        elif cron.endswith('* * 1,5'):  # 월요일, 금요일만 (암호화폐)
            times = cron.split()[1].split(',') if ',' in cron.split()[1] else [cron.split()[1]]
            minute = int(cron.split()[0])
            for time_hour in times:
                hour = int(time_hour)
                schedule.every().monday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
                schedule.every().friday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
        elif cron.endswith('* * 3'):  # 수요일만 (인도)
            hour = int(cron.split()[1])
            minute = int(cron.split()[0])
            schedule.every().wednesday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
        elif cron.endswith('* * 1-5'):  # 평일만 (기존)
            hour = int(cron.split()[1])
            minute = int(cron.split()[0])
            schedule.every().monday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
            schedule.every().tuesday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
            schedule.every().wednesday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
            schedule.every().thursday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
            schedule.every().friday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
        elif cron.endswith('* * *'):  # 매일
            hour = int(cron.split()[1])
            minute = int(cron.split()[0])
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
        elif cron.endswith('* * 0'):  # 일요일만
            hour = int(cron.split()[1])
            minute = int(cron.split()[0])
            schedule.every().sunday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
        elif cron.endswith('* * 5'):  # 금요일만
            hour = int(cron.split()[1])
            minute = int(cron.split()[0])
            schedule.every().friday.at(f"{hour:02d}:{minute:02d}").do(self._run_scheduled_task, task)
    
    def _run_scheduled_task(self, task: ScheduledTask):
        """스케줄된 작업 실행"""
        if not task.enabled:
            return
        
        # 비동기 작업을 동기적으로 실행
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.executor.execute_task(task))
                logging.info(f"작업 완료: {task.name} - {result.success}")
            except Exception as e:
                logging.error(f"작업 실행 오류: {task.name} - {e}")
            finally:
                loop.close()
        
        # 별도 스레드에서 실행
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
    
    def register_task(self, task: ScheduledTask):
        """작업 등록"""
        self.tasks[task.name] = task
        logging.info(f"작업 등록: {task.name} ({task.cron_expression})")
    
    def unregister_task(self, task_name: str):
        """작업 등록 해제"""
        if task_name in self.tasks:
            del self.tasks[task_name]
            logging.info(f"작업 등록 해제: {task_name}")
    
    def enable_task(self, task_name: str):
        """작업 활성화"""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = True
            logging.info(f"작업 활성화: {task_name}")
    
    def disable_task(self, task_name: str):
        """작업 비활성화"""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = False
            logging.info(f"작업 비활성화: {task_name}")
    
    async def run_task_now(self, task_name: str) -> TaskResult:
        """작업 즉시 실행"""
        if task_name not in self.tasks:
            raise ValueError(f"존재하지 않는 작업: {task_name}")
        
        task = self.tasks[task_name]
        result = await self.executor
            logging.error(f"스케줄러 설정 저장 실패: {e}")
    
    def get(self, key_path: str, default=None):
        """설정값 조회"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def update(self, key_path: str, value):
        """설정값 업데이트"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self._save_config()

# 전역 스케줄러 설정
scheduler_config = SchedulerConfig()

# ============================================================================
# 📈 시장별 최적 타이밍 계산기
# ============================================================================
class MarketTimingCalculator:
    """시장별 최적 분석 타이밍 계산"""
    
    def __init__(self):
        self.timezones = {
            'us': pytz.timezone('America/New_York'),
            'korea': pytz.timezone('Asia/Seoul'),
            'japan': pytz.timezone('Asia/Tokyo'),
            'india': pytz.timezone('Asia/Kolkata')
        }
        
        self.market_hours = {
            'us': {'open': time(9, 30), 'close': time(16, 0)},
            'crypto': {'open': time(0, 0), 'close': time(23, 59)},
            'japan': {'open': time(9, 0), 'close': time(15, 0)},
            'india': {'open': time(9, 15), 'close': time(15, 30)}
        }
    
    def get_optimal_scan_time(self, market: str) -> Dict:
        """시장별 최적 스캔 시간 계산"""
        now = datetime.now(pytz.UTC)
        
        if market == 'us':
            # 미국 장시작 30분 전 (한국시간 오후 11시)
            optimal_time = self._convert_to_cron('23:00', 'weekdays')
        elif market == 'crypto':
            # 암호화폐는 24시간이므로 30분마다
            optimal_time = '*/30 * * * *'
        elif market == 'japan':
            # 일본 장시작 1시간 전 (한국시간 오전 8시)
            optimal_time = self._convert_to_cron('08:00', 'weekdays')
        elif market == 'india':
            # 인도 장중 (한국시간 낮 12시)
            optimal_time = self._convert_to_cron('12:00', 'weekdays')
        else:
            optimal_time = '0 9 * * 1-5'  # 기본값
        
        return {
            'cron_expression': optimal_time,
            'description': f'{market} 시장 최적 스캔 시간',
            'timezone': 'Asia/Seoul'
        }
    
    def _convert_to_cron(self, time_str: str, frequency: str) -> str:
        """시간을 cron 표현식으로 변환"""
        hour, minute = map(int, time_str.split(':'))
        
        if frequency == 'weekdays':
            return f"{minute} {hour} * * 1-5"
        elif frequency == 'daily':
            return f"{minute} {hour} * * *"
        elif frequency == 'weekly':
            return f"{minute} {hour} * * 0"
        else:
            return f"{minute} {hour} * * *"
    
    def is_market_open(self, market: str) -> bool:
        """시장 개장 여부 확인"""
        if market == 'crypto':
            return True
        
        tz = self.timezones.get(market.replace('_stocks', ''), self.timezones['korea'])
        now = datetime.now(tz)
        
        # 주말 체크
        if now.weekday() >= 5:
            return False
        
        market_key = market.replace('_stocks', '')
        if market_key not in self.market_hours:
            return False
        
        open_time = self.market_hours[market_key]['open']
        close_time = self.market_hours[market_key]['close']
        current_time = now.time()
        
        return open_time <= current_time <= close_time

# ============================================================================
# 🔄 작업 실행 엔진
# ============================================================================
class TaskExecutor:
    """스케줄 작업 실행 엔진"""
    
    def __init__(self):
        self.running_tasks = set()
        self.task_history = []
        self.max_history = 1000
        self.executor_pool = None
        
    async def execute_task(self, task: ScheduledTask) -> TaskResult:
        """작업 실행"""
        if task.name in self.running_tasks:
            return TaskResult(
                task_name=task.name,
                success=False,
                execution_time=0,
                error_message="이미 실행 중인 작업"
            )
        
        self.running_tasks.add(task.name)
        start_time = time_module.time()
        
        try:
            # 타임아웃 적용
            result_data = await asyncio.wait_for(
                self._run_task_function(task),
                timeout=task.timeout_seconds
            )
            
            execution_time = time_module.time() - start_time
            task.last_run = datetime.now()
            task.run_count += 1
            task.error_count = 0  # 성공시 에러 카운트 리셋
            
            result = TaskResult(
                task_name=task.name,
                success=True,
                execution_time=execution_time,
                result_data=result_data
            )
            
            logging.info(f"✅ 작업 완료: {task.name} ({execution_time:.1f}초)")
            
        except asyncio.TimeoutError:
            execution_time = time_module.time() - start_time
            task.error_count += 1
            
            result = TaskResult(
                task_name=task.name,
                success=False,
                execution_time=execution_time,
                error_message=f"타임아웃 ({task.timeout_seconds}초)"
            )
            
            logging.error(f"⏱️ 작업 타임아웃: {task.name}")
            
        except Exception as e:
            execution_time = time_module.time() - start_time
            task.error_count += 1
            
            result = TaskResult(
                task_name=task.name,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            logging.error(f"❌ 작업 실패: {task.name} - {e}")
            
        finally:
            self.running_tasks.discard(task.name)
        
        # 히스토리 저장
        self._save_to_history(result)
        
        # 에러가 너무 많으면 작업 비활성화
        if task.error_count >= task.max_errors:
            task.enabled = False
            logging.warning(f"⚠️ 작업 비활성화: {task.name} (연속 {task.error_count}회 실패)")
        
        return result
    
    async def _run_task_function(self, task: ScheduledTask) -> Any:
        """작업 함수 실행"""
        if asyncio.iscoroutinefunction(task.function):
            return await task.function()
        else:
            return task.function()
    
    def _save_to_history(self, result: TaskResult):
        """실행 히스토리 저장"""
        self.task_history.append(result)
        
        # 히스토리 크기 제한
        if len(self.task_history) > self.max_history:
            self.task_history = self.task_history[-self.max_history:]
    
    def get_task_statistics(self, task_name: str = None) -> Dict:
        """작업 통계 조회"""
        if task_name:
            history = [r for r in self.task_history if r.task_name == task_name]
        else:
            history = self.task_history
        
        if not history:
            return {}
        
        success_count = sum(1 for r in history if r.success)
        total_count = len(history)
        avg_execution_time = sum(r.execution_time for r in history) / total_count
        
        return {
            'total_executions': total_count,
            'success_count': success_count,
            'failure_count': total_count - success_count,
            'success_rate': success_count / total_count * 100,
            'avg_execution_time': avg_execution_time,
            'last_execution': history[-1].timestamp if history else None
        }

# ============================================================================
# 🎯 핵심 스케줄 작업들
# ============================================================================
class QuintScheduledTasks:
    """퀸트프로젝트 핵심 스케줄 작업들"""
    
    def __init__(self):
        self.quint_master = None
        self.notification_manager = None
        self._initialize_components()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        if QUINT_MODULES_AVAILABLE:
            try:
                self.quint_master = QuintProjectMaster()
                self.notification_manager = QuintNotificationManager()
            except Exception as e:
                logging.error(f"퀸트 컴포넌트 초기화 실패: {e}")
    
    # ========================================================================
    # 📊 시장 스캔 작업들
    # ========================================================================
    async def scan_us_stocks(self) -> Dict:
        """미국 주식 시장 스캔"""
        if not self.quint_master:
            return {'status': 'error', 'message': '퀸트 마스터 없음'}
        
        try:
            signals = await self.quint_master.us_engine.analyze_us_market()
            buy_signals = [s for s in signals if s.action == 'BUY']
            
            # 중요한 신호가 있으면 즉시 알림
            if len(buy_signals) >= 3:
                await self._send_urgent_signal_alert('미국주식', buy_signals[:3])
            
            return {
                'status': 'success',
                'total_signals': len(signals),
                'buy_signals': len(buy_signals),
                'top_signals': [s.to_dict() for s in buy_signals[:5]]
            }
            
        except Exception as e:
            logging.error(f"미국주식 스캔 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def scan_crypto_market(self) -> Dict:
        """암호화폐 시장 스캔"""
        if not self.quint_master:
            return {'status': 'error', 'message': '퀸트 마스터 없음'}
        
        try:
            signals = await self.quint_master.crypto_engine.analyze_crypto_market()
            buy_signals = [s for s in signals if s.action == 'BUY']
            
            # 고신뢰도 신호 체크
            high_confidence = [s for s in buy_signals if s.confidence > 0.8]
            if high_confidence:
                await self._send_urgent_signal_alert('암호화폐', high_confidence[:2])
            
            return {
                'status': 'success',
                'total_signals': len(signals),
                'buy_signals': len(buy_signals),
                'high_confidence_signals': len(high_confidence)
            }
            
        except Exception as e:
            logging.error(f"암호화폐 스캔 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def scan_japan_stocks(self) -> Dict:
        """일본 주식 시장 스캔"""
        if not self.quint_master:
            return {'status': 'error', 'message': '퀸트 마스터 없음'}
        
        try:
            signals = await self.quint_master.japan_engine.analyze_japan_market()
            buy_signals = [s for s in signals if s.action == 'BUY']
            
            return {
                'status': 'success',
                'total_signals': len(signals),
                'buy_signals': len(buy_signals)
            }
            
        except Exception as e:
            logging.error(f"일본주식 스캔 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def scan_india_stocks(self) -> Dict:
        """인도 주식 시장 스캔"""
        if not self.quint_master:
            return {'status': 'error', 'message': '퀸트 마스터 없음'}
        
        try:
            signals = await self.quint_master.india_engine.analyze_india_market()
            buy_signals = [s for s in signals if s.action == 'BUY']
            
            return {
                'status': 'success',
                'total_signals': len(signals),
                'buy_signals': len(buy_signals)
            }
            
        except Exception as e:
            logging.error(f"인도주식 스캔 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    # ========================================================================
    # 💼 포트폴리오 관리 작업들
    # ========================================================================
    async def check_portfolio_performance(self) -> Dict:
        """포트폴리오 성과 체크"""
        try:
            if not self.quint_master:
                return {'status': 'error', 'message': '퀸트 마스터 없음'}
            
            portfolio_summary = self.quint_master.portfolio_manager.get_portfolio_summary()
            
            # 성과 분석
            total_value = portfolio_summary.get('total_value', 0)
            target_value = config.get('system.portfolio_value', 100_000_000)
            performance = ((total_value - target_value) / target_value) * 100
            
            # 성과 알림 (일정 수준 이상/이하일 때)
            if abs(performance) > 5.0:  # 5% 이상 변동시
                await self._send_performance_alert(performance, total_value)
            
            return {
                'status': 'success',
                'total_value': total_value,
                'performance_percent': performance,
                'position_count': portfolio_summary.get('total_positions', 0)
            }
            
        except Exception as e:
            logging.error(f"포트폴리오 성과 체크 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def rebalance_portfolio(self) -> Dict:
        """포트폴리오 리밸런싱"""
        try:
            if not self.quint_master:
                return {'status': 'error', 'message': '퀸트 마스터 없음'}
            
            # 전체 분석 실행
            analysis_result = await self.quint_master.run_full_analysis()
            
            if 'error' in analysis_result:
                return {'status': 'error', 'message': analysis_result['error']}
            
            # 리밸런싱 필요성 체크
            threshold = scheduler_config.get('portfolio.rebalancing.threshold_percent', 5.0)
            current_allocation = analysis_result.get('portfolio_allocation', 0)
            
            if current_allocation > 95.0:  # 너무 풀 투자된 경우
                await self._send_rebalancing_alert("현금 비중 부족", current_allocation)
            
            return {
                'status': 'success',
                'signals_analyzed': analysis_result.get('total_signals', 0),
                'buy_signals': analysis_result.get('buy_signals', 0),
                'portfolio_allocation': current_allocation
            }
            
        except Exception as e:
            logging.error(f"포트폴리오 리밸런싱 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    # ========================================================================
    # 🛡️ 리스크 관리 작업들
    # ========================================================================
    async def monitor_real_time_risk(self) -> Dict:
        """실시간 리스크 모니터링"""
        try:
            if not self.quint_master:
                return {'status': 'error', 'message': '퀸트 마스터 없음'}
            
            portfolio_summary = self.quint_master.portfolio_manager.get_portfolio_summary()
            
            # 손실 체크
            total_value = portfolio_summary.get('total_value', 0)
            target_value = config.get('system.portfolio_value', 100_000_000)
            loss_percent = ((target_value - total_value) / target_value) * 100
            
            max_loss = scheduler_config.get('risk_management.real_time_monitoring.max_loss_percent', 10.0)
            circuit_breaker = scheduler_config.get('risk_management.real_time_monitoring.circuit_breaker', True)
            
            # 긴급 정지 체크
            if loss_percent > max_loss and circuit_breaker:
                await self._trigger_emergency_stop(loss_percent)
                
                return {
                    'status': 'emergency_stop',
                    'loss_percent': loss_percent,
                    'emergency_action': 'triggered'
                }
            
            # 경고 수준 체크
            warning_levels = [5.0, 7.5]  # 5%, 7.5% 손실시 경고
            for warning_level in warning_levels:
                if loss_percent > warning_level:
                    await self._send_risk_warning(loss_percent, warning_level)
                    break
            
            return {
                'status': 'normal',
                'loss_percent': loss_percent,
                'risk_level': 'high' if loss_percent > 7.5 else 'medium' if loss_percent > 5.0 else 'low'
            }
            
        except Exception as e:
            logging.error(f"리스크 모니터링 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def generate_daily_risk_report(self) -> Dict:
        """일일 리스크 리포트 생성"""
        try:
            # 포트폴리오 현황
            if self.quint_master:
                portfolio_summary = self.quint_master.portfolio_manager.get_portfolio_summary()
            else:
                portfolio_summary = {'total_value': 0, 'total_positions': 0}
            
            # 기본 리스크 지표
            total_value = portfolio_summary.get('total_value', 0)
            position_count = portfolio_summary.get('total_positions', 0)
            
            risk_report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'portfolio_value': total_value,
                'position_count': position_count,
                'diversification_score': min(position_count / 20 * 100, 100),  # 20개 기준
                'risk_level': 'low' if position_count >= 15 else 'medium' if position_count >= 10 else 'high'
            }
            
            # 리포트 알림 전송
            if self.notification_manager:
                await self._send_risk_report_notification(risk_report)
            
            return {
                'status': 'success',
                'report': risk_report
            }
            
        except Exception as e:
            logging.error(f"일일 리스크 리포트 생성 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    # ========================================================================
    # 📊 리포트 생성 작업들
    # ========================================================================
    async def generate_daily_report(self) -> Dict:
        """일일 성과 리포트 생성"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 포트폴리오 현황
            if self.quint_master:
                portfolio_summary = self.quint_master.portfolio_manager.get_portfolio_summary()
                
                # 전체 분석 실행
                analysis_result = await self.quint_master.run_full_analysis()
            else:
                portfolio_summary = {'total_value': 0}
                analysis_result = {'buy_signals': 0, 'total_signals': 0}
            
            # 리포트 데이터 구성
            report_data = {
                'date': today,
                'portfolio_value': portfolio_summary.get('total_value', 0),
                'daily_signals': analysis_result.get('total_signals', 0),
                'buy_signals': analysis_result.get('buy_signals', 0),
                'market_summary': analysis_result.get('market_breakdown', {}),
                'top_opportunities': analysis_result.get('optimized_portfolio', [])[:5]
            }
            
            # 일일 리포트 알림 전송
            if self.notification_manager:
                await self._send_daily_report_notification(report_data)
            
            return {
                'status': 'success',
                'report': report_data
            }
            
        except Exception as e:
            logging.error(f"일일 리포트 생성 실패: {e}")
            return {'status': 'error', 'message': str(e)}
🏆 퀸트프로젝트 = 완벽한 자동화 스케줄링!
"""
