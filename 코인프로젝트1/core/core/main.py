# main.py
"""
퀸트 트레이딩 시스템 메인 실행 파일
우아한 시작/종료와 상태 모니터링 포함
"""
import asyncio
import signal
import sys
from datetime import datetime
import argparse
import os

from scheduler import scheduler
from notifier import notifier
from logger import logger
from config import get_config
from db import db_manager
from exceptions import handle_errors, error_handler


class QuantTradingSystem:
    """퀀트 트레이딩 시스템 메인 클래스"""
    
    def __init__(self):
        self.cfg = get_config()
        self.is_running = False
        self.start_time = None
        
    async def startup_checks(self) -> bool:
        """시작 전 시스템 체크"""
        logger.info("시스템 시작 전 체크 중...")
        
        checks = {
            "설정 파일": self._check_config(),
            "API 키": self._check_api_keys(),
            "데이터베이스": self._check_database(),
            "텔레그램": await self._check_telegram(),
            "거래소 연결": await self._check_exchange()
        }
        
        # 체크 결과 알림
        status_msg = "🚀 <b>시스템 시작 체크</b>\n"
        all_passed = True
        
        for name, passed in checks.items():
            emoji = "✅" if passed else "❌"
            status_msg += f"{emoji} {name}\n"
            if not passed:
                all_passed = False
        
        await notifier.send_message(status_msg)
        return all_passed
    
    def _check_config(self) -> bool:
        """설정 파일 체크"""
        try:
            required_keys = ['api', 'telegram', 'trading', 'schedule']
            for key in required_keys:
                if key not in self.cfg:
                    logger.error(f"필수 설정 누락: {key}")
                    return False
            return True
        except Exception as e:
            logger.error(f"설정 체크 실패: {e}")
            return False
    
    def _check_api_keys(self) -> bool:
        """API 키 체크"""
        try:
            api_cfg = self.cfg.get('api', {})
            return bool(api_cfg.get('access_key') and api_cfg.get('secret_key'))
        except:
            return False
    
    def _check_database(self) -> bool:
        """데이터베이스 연결 체크"""
        try:
            # 테스트 쿼리
            summary = db_manager.get_daily_summary()
            return True
        except Exception as e:
            logger.error(f"DB 체크 실패: {e}")
            return False
    
    async def _check_telegram(self) -> bool:
        """텔레그램 연결 체크"""
        try:
            await notifier.send_message("🔧 텔레그램 연결 테스트")
            return True
        except:
            return False
    
    async def _check_exchange(self) -> bool:
        """거래소 연결 체크"""
        try:
            import pyupbit
            upbit = pyupbit.Upbit(
                self.cfg['api']['access_key'],
                self.cfg['api']['secret_key']
            )
            balance = upbit.get_balance("KRW")
            return balance is not None
        except:
            return False
    
    async def initialize(self):
        """시스템 초기화"""
        logger.info("퀀트 트레이딩 시스템 초기화 중...")
        
        # 시작 체크
        if not await self.startup_checks():
            raise Exception("시스템 체크 실패")
        
        # 성능 모니터링 시작
        asyncio.create_task(self.monitor_performance())
        
        # 에러 모니터링 시작
        asyncio.create_task(self.monitor_errors())
        
        self.start_time = datetime.now()
        self.is_running = True
        
        # 시작 알림
        await notifier.send_message(
            "🎯 <b>퀀트 트레이딩 시스템 시작</b>\n"
            f"버전: {self.cfg.get('version', '1.0.0')}\n"
            f"환경: {self.cfg.get('environment', 'production')}\n"
            f"시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    async def monitor_performance(self):
        """성능 모니터링 (1시간마다)"""
        while self.is_running:
            await asyncio.sleep(3600)  # 1시간
            
            try:
                # 포트폴리오 지표
                metrics = db_manager.calculate_portfolio_metrics()
                
                # 일일 요약
                summary = db_manager.get_daily_summary()
                
                # 시스템 상태
                uptime = datetime.now() - self.start_time
                
                status_msg = f"""
📊 <b>시스템 상태 리포트</b>
━━━━━━━━━━━━━━━
⏱️ 가동시간: {uptime.days}일 {uptime.seconds//3600}시간
💰 총 수익률: {metrics.get('total_return', 0):.2f}%
📈 승률: {metrics.get('win_rate', 0):.1f}%
📉 최대낙폭: {metrics.get('max_drawdown', 0):.2f}%
🎯 샤프비율: {metrics.get('sharpe_ratio', 0):.2f}

📅 오늘 거래: {summary.get('trades', 0)}건
━━━━━━━━━━━━━━━"""
                
                await notifier.send_message(status_msg)
                
            except Exception as e:
                logger.error(f"성능 모니터링 에러: {e}")
    
    async def monitor_errors(self):
        """에러 모니터링 (5분마다)"""
        while self.is_running:
            await asyncio.sleep(300)  # 5분
            
            try:
                await error_handler.check_critical_errors()
            except Exception as e:
                logger.error(f"에러 모니터링 실패: {e}")
    
    def setup_signal_handlers(self):
        """종료 시그널 핸들러 설정"""
        def signal_handler(sig, frame):
            logger.info(f"종료 시그널 받음: {sig}")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """우아한 종료"""
        logger.info("시스템 종료 중...")
        self.is_running = False
        
        # 스케줄러 정지
        scheduler.stop()
        
        # 종료 통계
        if self.start_time:
            uptime = datetime.now() - self.start_time
            error_stats = error_handler.get_error_stats()
            
            await notifier.send_message(
                f"🛑 <b>시스템 종료</b>\n"
                f"가동 시간: {uptime.days}일 {uptime.seconds//3600}시간\n"
                f"총 에러: {error_stats['total_errors']}건"
            )
        
        # DB 정리
        db_manager.cleanup_old_records(days=180)
        
        logger.info("시스템 종료 완료")
        sys.exit(0)
    
    async def run(self):
        """메인 실행"""
        try:
            # 초기화
            await self.initialize()
            
            # 시그널 핸들러 설정
            self.setup_signal_handlers()
            
            # 스케줄러 시작
            logger.info("스케줄러 시작...")
            scheduler.start()
            
            # 메인 루프
            while self.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("키보드 인터럽트")
            await self.shutdown()
        except Exception as e:
            logger.error(f"치명적 에러: {e}")
            await notifier.send_error_alert(e, "시스템 크래시")
            await self.shutdown()


def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='퀀트 트레이딩 시스템'
    )
    
    parser.add_argument(
        '--env',
        choices=['production', 'development', 'test'],
        default='production',
        help='실행 환경'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='모의 실행 (실제 거래 없음)'
    )
    
    return parser.parse_args()


async def main():
    """메인 진입점"""
    # 명령줄 인자 파싱
    args = parse_arguments()
    
    # 환경 설정
    os.environ['TRADING_ENV'] = args.env
    if args.debug:
        logger.setLevel('DEBUG')
    if args.dry_run:
        os.environ['DRY_RUN'] = 'true'
    
    # 시스템 시작
    system = QuantTradingSystem()
    await system.run()


if __name__ == "__main__":
    # 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("프로그램 종료")
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {e}")
        sys.exit(1)