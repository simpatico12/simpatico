"""
통합 로깅 모듈
- 파일 & 콘솔 동시 출력
- 로그 레벨별 색상 지원
- 에러 발생시 텔레그램 알림
- 성능 모니터링
"""

import os
import sys
import logging
import time
import functools
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Callable, Any

# 로그 폴더 자동 생성
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ANSI 색상 코드 (콘솔 출력용)
class LogColors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    
    LEVEL_COLORS = {
        'DEBUG': BLUE,
        'INFO': GREEN,
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': PURPLE
    }

class ColoredFormatter(logging.Formatter):
    """색상이 있는 로그 포맷터"""
    
    def format(self, record):
        # 콘솔 출력시 색상 추가
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            levelname = record.levelname
            if levelname in LogColors.LEVEL_COLORS:
                record.levelname = f"{LogColors.LEVEL_COLORS[levelname]}{levelname}{LogColors.RESET}"
                record.msg = f"{LogColors.LEVEL_COLORS[levelname]}{record.msg}{LogColors.RESET}"
        
        return super().format(record)

class TelegramHandler(logging.Handler):
    """텔레그램 알림 핸들러 (ERROR 이상만)"""
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        super().__init__()
        self.token = token or os.getenv('TELEGRAM_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        
    def emit(self, record):
        if not self.enabled or record.levelno < logging.ERROR:
            return
            
        try:
            import requests
            message = f"⚠️ {record.levelname}\n"
            message += f"📍 {record.name}\n"
            message += f"💬 {record.getMessage()}\n"
            message += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, data={
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }, timeout=5)
        except Exception:
            # 텔레그램 전송 실패시 무시 (무한 루프 방지)
            pass

def setup_logger(
    name: str = 'quant',
    log_level: str = None,
    log_file: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_telegram: bool = True
) -> logging.Logger:
    """
    통합 로거 설정
    
    Args:
        name: 로거 이름
        log_level: 로그 레벨 (기본값: 환경변수 또는 INFO)
        log_file: 로그 파일 경로
        max_bytes: 파일 최대 크기
        backup_count: 백업 파일 개수
        enable_telegram: 텔레그램 알림 활성화
    """
    
    # 로그 레벨 설정
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    
    # 포맷 설정
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 1. 파일 핸들러 (일별 로테이션)
    if log_file is None:
        log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=30,  # 30일분 보관
        encoding='utf-8'
    )
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # 2. 크기 기반 로테이션 핸들러 (대용량 로그용)
    size_handler = RotatingFileHandler(
        filename=LOG_DIR / f"{name}_debug.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    size_handler.setLevel(logging.DEBUG)
    size_handler.setFormatter(detailed_formatter)
    logger.addHandler(size_handler)
    
    # 3. 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    
    # 프로덕션에서는 INFO 이상만 콘솔 출력
    if os.getenv('ENVIRONMENT') == 'production':
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.DEBUG)
    
    logger.addHandler(console_handler)
    
    # 4. 텔레그램 핸들러 (ERROR 이상)
    if enable_telegram:
        telegram_handler = TelegramHandler()
        telegram_handler.setLevel(logging.ERROR)
        logger.addHandler(telegram_handler)
    
    return logger

# get_logger 함수 추가 (기존 코드 호환성을 위해)
def get_logger(name: str = 'quant') -> logging.Logger:
    """
    간단한 로거 생성 함수 (기존 코드 호환성)
    """
    return setup_logger(name)

# 기본 로거 생성
logger = setup_logger()

# 편의 함수들
def debug(msg: str, *args, **kwargs):
    """디버그 로그"""
    logger.debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    """정보 로그"""
    logger.info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    """경고 로그"""
    logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    """에러 로그 (텔레그램 알림 포함)"""
    logger.error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs):
    """심각한 에러 로그 (텔레그램 알림 포함)"""
    logger.critical(msg, *args, **kwargs)

# 데코레이터
def log_execution_time(func: Callable) -> Callable:
    """함수 실행 시간 로깅 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > 5:  # 5초 이상 걸리면 경고
                warning(f"{func.__name__} 실행 시간: {execution_time:.2f}초 (느림)")
            else:
                debug(f"{func.__name__} 실행 시간: {execution_time:.2f}초")
                
            return result
        except Exception as e:
            error(f"{func.__name__} 실행 중 에러: {str(e)}")
            raise
    return wrapper

def log_errors(func: Callable) -> Callable:
    """에러 자동 로깅 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error(f"{func.__name__} 에러 발생: {str(e)}", exc_info=True)
            raise
    return wrapper

def log_trades(market: str = ""):
    """거래 로깅 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            info(f"[{market}] 거래 시작: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                info(f"[{market}] 거래 완료: {result}")
                return result
            except Exception as e:
                error(f"[{market}] 거래 실패: {str(e)}")
                raise
        return wrapper
    return decorator

# 시스템 상태 로깅
def log_system_status():
    """시스템 상태 로깅 (메모리, CPU 등)"""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info(f"시스템 상태 - CPU: {cpu_percent}%, "
             f"메모리: {memory.percent}%, "
             f"디스크: {disk.percent}%")
        
        # 리소스 부족 경고
        if memory.percent > 80:
            warning(f"메모리 사용률 높음: {memory.percent}%")
        if disk.percent > 90:
            warning(f"디스크 공간 부족: {disk.percent}%")
            
    except ImportError:
        debug("psutil이 설치되지 않아 시스템 상태를 확인할 수 없습니다")

# 로그 정리 함수
def cleanup_old_logs(days: int = 30):
    """오래된 로그 파일 정리"""
    import glob
    from datetime import timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for log_file in glob.glob(str(LOG_DIR / "*.log")):
        file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
        if file_time < cutoff_date:
            try:
                os.remove(log_file)
                info(f"오래된 로그 삭제: {log_file}")
            except Exception as e:
                error(f"로그 삭제 실패: {log_file} - {e}")

# 모듈 로드시 초기 로그
info("="*50)
info("퀀트 트레이딩 시스템 시작")
info(f"로그 레벨: {os.getenv('LOG_LEVEL', 'INFO')}")
info(f"환경: {os.getenv('ENVIRONMENT', 'development')}")
info("="*50)
