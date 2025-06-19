#!/usr/bin/env python3
# scripts/health_check.py
"""
퀀트 트레이딩 시스템 종합 헬스체크
시스템 상태를 진단하고 문제 해결 방안 제시
"""

import os
import sys
import sqlite3
import asyncio
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from db import db_manager, SessionLocal
from logger import logger
from notifier import notifier


class HealthChecker:
    """시스템 헬스체크 클래스"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []
        self.suggestions = []
        
    def print_status(self, status: str, message: str, detail: str = ""):
        """상태 출력"""
        icons = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "info": "ℹ️"
        }
        
        icon = icons.get(status, "")
        print(f"{icon} {message}")
        
        if detail and self.verbose:
            print(f"   └─ {detail}")
        
        # 통계 업데이트
        if status == "success":
            self.checks_passed += 1
        elif status == "error":
            self.checks_failed += 1
            self.errors.append(message)
        elif status == "warning":
            self.warnings.append(message)
    
    def check_database(self) -> bool:
        """데이터베이스 상태 체크"""
        try:
            # DB 파일 존재 확인
            db_url = os.getenv('DATABASE_URL', 'sqlite:///quant.db')
            if 'sqlite' in db_url:
                db_path = db_url.replace('sqlite:///', '')
                if not os.path.exists(db_path):
                    self.print_status("error", "DB 파일 없음", db_path)
                    self.suggestions.append("python -c 'from db import Base, engine; Base.metadata.create_all(bind=engine)'")
                    return False
                
                # 파일 크기 확인
                size_mb = os.path.getsize(db_path) / 1024 / 1024
                if size_mb > 1000:  # 1GB 이상
                    self.print_status("warning", f"DB 크기 과대: {size_mb:.1f}MB")
                    self.suggestions.append("오래된 데이터 정리: db_manager.cleanup_old_records()")
            
            # 연결 테스트
            with SessionLocal() as session:
                result = session.execute("SELECT COUNT(*) FROM trades").scalar()
                self.print_status("success", f"DB 정상 (거래 기록: {result}건)")
            
            # 최근 거래 확인
            recent_trades = db_manager.get_recent_trades(days=1)
            if not recent_trades:
                self.print_status("warning", "24시간 내 거래 없음")
            
            return True
            
        except Exception as e:
            self.print_status("error", f"DB 연결 실패: {str(e)}")
            return False
    
    def check_logs(self) -> bool:
        """로그 상태 체크"""
        log_dir = "logs"
        
        if not os.path.exists(log_dir):
            self.print_status("error", "로그 디렉토리 없음")
            self.suggestions.append(f"mkdir -p {log_dir}")
            return False
        
        # 최신 로그 파일 확인
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        
        if not log_files:
            self.print_status("warning", "로그 파일 없음")
            return True
        
        # 가장 최근 로그 확인
        latest_log = max([os.path.join(log_dir, f) for f in log_files], 
                        key=os.path.getmtime)
        
        # 마지막 수정 시간
        mtime = datetime.fromtimestamp(os.path.getmtime(latest_log))
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        
        if age_hours > 24:
            self.print_status("warning", f"로그 오래됨: {age_hours:.1f}시간 전")
        else:
            self.print_status("success", f"로그 최신 ({age_hours:.1f}시간 전)")
        
        # 로그 크기 확인
        total_size = sum(os.path.getsize(os.path.join(log_dir, f)) 
                        for f in log_files) / 1024 / 1024
        
        if total_size > 100:  # 100MB 이상
            self.print_status("warning", f"로그 크기 과대: {total_size:.1f}MB")
            self.suggestions.append("오래된 로그 압축 또는 삭제 필요")
        
        # 에러 로그 확인
        try:
            with open(latest_log, 'r') as f:
                content = f.read()
                error_count = content.lower().count('error')
                if error_count > 10:
                    self.print_status("warning", f"최근 에러 다수 발생: {error_count}건")
        except:
            pass
        
        return True
    
    def check_processes(self) -> bool:
        """프로세스 상태 체크"""
        critical_processes = {
            'scheduler': 'scheduler.py',
            'main': 'main.py'
        }
        
        all_running = True
        
        for name, pattern in critical_processes.items():
            running = False
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if pattern in cmdline:
                        running = True
                        cpu = proc.cpu_percent(interval=0.1)
                        mem = proc.memory_info().rss / 1024 / 1024
                        
                        self.print_status("success", 
                            f"{name} 실행중 (PID: {proc.info['pid']})",
                            f"CPU: {cpu:.1f}%, MEM: {mem:.1f}MB")
                        
                        # 리소스 사용량 체크
                        if cpu > 80:
                            self.print_status("warning", f"{name} CPU 사용률 높음: {cpu:.1f}%")
                        if mem > 500:
                            self.print_status("warning", f"{name} 메모리 사용량 높음: {mem:.1f}MB")
                        break
                except:
                    continue
            
            if not running:
                self.print_status("error", f"{name} 미실행")
                self.suggestions.append(f"python {pattern} 실행 필요")
                all_running = False
        
        return all_running
    
    async def check_external_services(self) -> bool:
        """외부 서비스 연결 체크"""
        all_good = True
        
        # 1. 텔레그램 체크
        try:
            await notifier.send_message("🏥 Health Check Test")
            self.print_status("success", "텔레그램 정상")
        except Exception as e:
            self.print_status("error", f"텔레그램 연결 실패: {str(e)}")
            self.suggestions.append("텔레그램 봇 토큰 및 채팅 ID 확인")
            all_good = False
        
        # 2. 거래소 API 체크
        try:
            import pyupbit
            cfg = get_config()
            upbit = pyupbit.Upbit(
                cfg['api']['access_key'],
                cfg['api']['secret_key']
            )
            
            # 잔고 조회
            balance = upbit.get_balance("KRW")
            if balance is not None:
                self.print_status("success", f"거래소 API 정상 (잔고: {balance:,.0f}원)")
            else:
                raise Exception("잔고 조회 실패")
                
            # API 호출 제한 체크
            remaining = upbit.get_remaining_req()
            if remaining and remaining.get('min', 0) < 10:
                self.print_status("warning", f"API 호출 한도 임박: {remaining}")
                
        except Exception as e:
            self.print_status("error", f"거래소 API 실패: {str(e)}")
            self.suggestions.append("API 키 확인 필요")
            all_good = False
        
        return all_good
    
    def check_system_resources(self) -> bool:
        """시스템 리소스 체크"""
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            self.print_status("warning", f"CPU 사용률 높음: {cpu_percent:.1f}%")
        else:
            self.print_status("success", f"CPU 정상: {cpu_percent:.1f}%")
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            self.print_status("warning", f"메모리 사용률 높음: {memory.percent:.1f}%")
        else:
            self.print_status("success", f"메모리 정상: {memory.percent:.1f}%")
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            self.print_status("error", f"디스크 공간 부족: {disk.percent:.1f}%")
            self.suggestions.append("디스크 정리 필요")
            return False
        else:
            self.print_status("success", f"디스크 정상: {disk.percent:.1f}%")
        
        return True
    
    def check_performance(self) -> bool:
        """성능 지표 체크"""
        try:
            # 최근 성과 조회
            metrics = db_manager.calculate_portfolio_metrics()
            
            if not metrics:
                self.print_status("info", "성과 데이터 없음")
                return True
            
            # 주요 지표 확인
            total_return = metrics.get('total_return', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            win_rate = metrics.get('win_rate', 0)
            
            self.print_status("info", 
                f"성과: 수익률 {total_return:.2f}%, MDD {max_drawdown:.2f}%, 승률 {win_rate:.1f}%")
            
            # 경고 조건
            if max_drawdown > 20:
                self.print_status("warning", f"최대낙폭 과대: {max_drawdown:.2f}%")
                self.suggestions.append("리스크 관리 전략 재검토 필요")
            
            if win_rate < 40:
                self.print_status("warning", f"승률 저조: {win_rate:.1f}%")
                self.suggestions.append("매매 전략 개선 필요")
            
            return True
            
        except Exception as e:
            self.print_status("error", f"성과 조회 실패: {str(e)}")
            return False
    
    async def run_all_checks(self) -> Dict:
        """모든 체크 실행"""
        print("🏥 퀀트 트레이딩 시스템 헬스체크")
        print("=" * 50)
        
        # 1. 기본 체크
        print("\n📌 기본 시스템 체크")
        self.check_database()
        self.check_logs()
        self.check_processes()
        self.check_system_resources()
        
        # 2. 외부 서비스 체크
        print("\n🌐 외부 서비스 체크")
        await self.check_external_services()
        
        # 3. 성능 체크
        print("\n📊 성능 지표 체크")
        self.check_performance()
        
        # 결과 요약
        print("\n" + "=" * 50)
        print("📋 체크 결과 요약")
        print(f"✅ 성공: {self.checks_passed}개")
        print(f"❌ 실패: {self.checks_failed}개")
        print(f"⚠️  경고: {len(self.warnings)}개")
        
        # 제안사항
        if self.suggestions:
            print("\n💡 개선 제안:")
            for i, suggestion in enumerate(self.suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        # 종합 상태
        if self.checks_failed == 0:
            print("\n✨ 시스템 상태: 정상")
            exit_code = 0
        else:
            print("\n⚠️ 시스템 상태: 문제 발견")
            exit_code = 1
        
        # 결과 저장
        result = {
            "timestamp": datetime.now().isoformat(),
            "passed": self.checks_passed,
            "failed": self.checks_failed,
            "warnings": self.warnings,
            "errors": self.errors,
            "suggestions": self.suggestions,
            "status": "healthy" if self.checks_failed == 0 else "unhealthy"
        }
        
        # 결과 파일 저장
        with open("health_check_result.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='시스템 헬스체크')
    parser.add_argument('-v', '--verbose', action='store_true', help='상세 출력')
    parser.add_argument('--notify', action='store_true', help='텔레그램 알림')
    args = parser.parse_args()
    
    checker = HealthChecker(verbose=args.verbose)
    result = await checker.run_all_checks()
    
    # 텔레그램 알림
    if args.notify and result['failed'] > 0:
        message = f"""
🏥 <b>헬스체크 결과</b>
━━━━━━━━━━━━━━━
상태: {'❌ 문제 발견' if result['failed'] > 0 else '✅ 정상'}
성공: {result['passed']}개
실패: {result['failed']}개
경고: {len(result['warnings'])}개
━━━━━━━━━━━━━━━
"""
        if result['errors']:
            message += "\n<b>에러:</b>\n"
            for error in result['errors'][:5]:
                message += f"• {error}\n"
        
        await notifier.send_message(message)
    
    sys.exit(0 if result['failed'] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())