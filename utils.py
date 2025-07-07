#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 퀸트프로젝트 유틸리티 모듈 (utils.py)
===============================================
📦 재사용 가능한 공통 함수 및 헬퍼 클래스 모음

✨ 주요 기능:
- 로깅 설정 및 관리
- 환경변수 검증
- 파일 시스템 유틸리티
- 네트워크 연결 테스트
- 데이터 형변환 및 검증
- 시간 관련 유틸리티
- 암호화/보안 관련 함수
- 성능 모니터링 도구
- 메모리 관리 도구

Author: 퀸트마스터팀
Version: 1.0.0
"""

import os
import sys
import json
import time
import logging
import hashlib
import secrets
import socket
import psutil
import sqlite3
import asyncio
import aiohttp
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import yaml
import pandas as pd
import numpy as np
from functools import wraps
import traceback

# ============================================================================
# 🔧 로깅 유틸리티
# ============================================================================

class LoggerManager:
    """통합 로깅 관리자"""
    
    @staticmethod
    def setup_logger(
        name: str,
        log_file: str = None,
        level: str = 'INFO',
        format_string: str = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # 기존 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 포맷 설정
        if format_string is None:
            format_string = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 (선택적)
        if log_file:
            try:
                from logging.handlers import RotatingFileHandler
                
                # 로그 디렉토리 생성
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    log_file, 
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
            except Exception as e:
                logger.error(f"파일 로깅 설정 실패: {e}")
        
        return logger
    
    @staticmethod
    def log_function_call(logger: logging.Logger):
        """함수 호출 로깅 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                logger.debug(f"🔧 {func.__name__} 시작")
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    logger.debug(f"✅ {func.__name__} 완료 ({execution_time:.3f}초)")
                    return result
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"❌ {func.__name__} 실패 ({execution_time:.3f}초): {e}")
                    raise
                    
            return wrapper
        return decorator
    
    @staticmethod
    def log_async_function_call(logger: logging.Logger):
        """비동기 함수 호출 로깅 데코레이터"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                logger.debug(f"🔧 {func.__name__} 시작")
                
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    logger.debug(f"✅ {func.__name__} 완료 ({execution_time:.3f}초)")
                    return result
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"❌ {func.__name__} 실패 ({execution_time:.3f}초): {e}")
                    raise
                    
            return wrapper
        return decorator

# ============================================================================
# 🔐 환경변수 및 설정 유틸리티
# ============================================================================

class ConfigValidator:
    """설정 검증 도구"""
    
    @staticmethod
    def validate_env_vars(required_vars: List[str], optional_vars: Dict[str, Any] = None) -> Dict[str, Any]:
        """환경변수 검증"""
        config = {}
        missing_vars = []
        
        # 필수 환경변수 체크
        for var in required_vars:
            value = os.getenv(var)
            if value is None or value.strip() == '':
                missing_vars.append(var)
            else:
                config[var] = value
        
        if missing_vars:
            raise ValueError(f"필수 환경변수 누락: {', '.join(missing_vars)}")
        
        # 선택적 환경변수 (기본값 포함)
        if optional_vars:
            for var, default in optional_vars.items():
                config[var] = os.getenv(var, default)
        
        return config
    
    @staticmethod
    def load_yaml_config(config_path: str, required_keys: List[str] = None) -> Dict[str, Any]:
        """YAML 설정 파일 로드 및 검증"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"설정 파일 없음: {config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 필수 키 검증
            if required_keys:
                missing_keys = [key for key in required_keys if key not in config]
                if missing_keys:
                    raise ValueError(f"설정 파일 필수 키 누락: {', '.join(missing_keys)}")
            
            return config
            
        except Exception as e:
            raise Exception(f"설정 파일 로드 실패: {e}")
    
    @staticmethod
    def mask_sensitive_data(data: Dict[str, Any], sensitive_keys: List[str] = None) -> Dict[str, Any]:
        """민감한 데이터 마스킹"""
        if sensitive_keys is None:
            sensitive_keys = [
                'password', 'token', 'key', 'secret', 'api_key', 
                'access_key', 'private_key', 'client_secret'
            ]
        
        masked_data = data.copy()
        
        for key, value in masked_data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 4:
                    masked_data[key] = f"{value[:4]}***"
                else:
                    masked_data[key] = "***"
        
        return masked_data

# ============================================================================
# 📁 파일 시스템 유틸리티
# ============================================================================

class FileSystemUtils:
    """파일 시스템 관리 도구"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """디렉토리 존재 보장"""
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def safe_write_json(file_path: Union[str, Path], data: Dict[str, Any], backup: bool = True) -> bool:
        """안전한 JSON 파일 쓰기 (백업 포함)"""
        try:
            file_path = Path(file_path)
            
            # 백업 생성
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(f'.backup_{int(time.time())}.json')
                file_path.rename(backup_path)
            
            # 임시 파일에 쓰기
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 원자적 이동
            temp_path.rename(file_path)
            return True
            
        except Exception as e:
            logging.error(f"JSON 파일 쓰기 실패: {e}")
            return False
    
    @staticmethod
    def safe_read_json(file_path: Union[str, Path], default: Any = None) -> Any:
        """안전한 JSON 파일 읽기"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return default
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logging.error(f"JSON 파일 읽기 실패: {e}")
            return default
    
    @staticmethod
    def cleanup_old_files(directory: Union[str, Path], pattern: str, days: int = 30) -> int:
        """오래된 파일 정리"""
        try:
            directory = Path(directory)
            if not directory.exists():
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=days)
            removed_count = 0
            
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        removed_count += 1
            
            return removed_count
            
        except Exception as e:
            logging.error(f"파일 정리 실패: {e}")
            return 0
    
    @staticmethod
    def get_directory_size(directory: Union[str, Path]) -> int:
        """디렉토리 크기 계산 (바이트)"""
        try:
            directory = Path(directory)
            total_size = 0
            
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size
            
        except Exception as e:
            logging.error(f"디렉토리 크기 계산 실패: {e}")
            return 0

# ============================================================================
# 🌐 네트워크 유틸리티
# ============================================================================

class NetworkUtils:
    """네트워크 관련 유틸리티"""
    
    @staticmethod
    async def test_connection(host: str, port: int, timeout: int = 10) -> Tuple[bool, float]:
        """네트워크 연결 테스트"""
        start_time = time.time()
        
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            
            latency = (time.time() - start_time) * 1000  # ms
            return True, latency
            
        except Exception:
            latency = (time.time() - start_time) * 1000
            return False, latency
    
    @staticmethod
    async def test_multiple_hosts(hosts: List[Tuple[str, int]], timeout: int = 10) -> Dict[str, Dict]:
        """다중 호스트 연결 테스트"""
        results = {}
        
        tasks = []
        for host, port in hosts:
            task = NetworkUtils.test_connection(host, port, timeout)
            tasks.append((f"{host}:{port}", task))
        
        for name, task in tasks:
            try:
                is_connected, latency = await task
                results[name] = {
                    'connected': is_connected,
                    'latency_ms': latency,
                    'status': 'success' if is_connected else 'failed'
                }
            except Exception as e:
                results[name] = {
                    'connected': False,
                    'latency_ms': 0,
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    @staticmethod
    async def http_health_check(url: str, timeout: int = 10) -> Dict[str, Any]:
        """HTTP 상태 체크"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    latency = (time.time() - start_time) * 1000
                    
                    return {
                        'url': url,
                        'status_code': response.status,
                        'latency_ms': latency,
                        'headers': dict(response.headers),
                        'success': 200 <= response.status < 300
                    }
                    
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                'url': url,
                'status_code': 0,
                'latency_ms': latency,
                'error': str(e),
                'success': False
            }

# ============================================================================
# 📊 데이터 유틸리티
# ============================================================================

class DataUtils:
    """데이터 처리 유틸리티"""
    
    @staticmethod
    def safe_float_convert(value: Any, default: float = 0.0) -> float:
        """안전한 float 변환"""
        try:
            if value is None or value == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_int_convert(value: Any, default: int = 0) -> int:
        """안전한 int 변환"""
        try:
            if value is None or value == '':
                return default
            return int(float(value))  # float를 거쳐서 변환 (문자열 "1.0" 처리)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """퍼센트 변화율 계산"""
        try:
            if old_value == 0:
                return 0.0 if new_value == 0 else float('inf')
            return ((new_value - old_value) / old_value) * 100
        except:
            return 0.0
    
    @staticmethod
    def validate_data_range(value: float, min_val: float = None, max_val: float = None) -> bool:
        """데이터 범위 검증"""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    
    @staticmethod
    def clean_symbol(symbol: str) -> str:
        """심볼 정리 (공백, 특수문자 제거)"""
        if not symbol:
            return ''
        return ''.join(c for c in symbol.upper() if c.isalnum())
    
    @staticmethod
    def format_currency(amount: float, currency: str = 'USD', decimal_places: int = 2) -> str:
        """통화 포맷팅"""
        try:
            if currency == 'USD':
                return f"${amount:,.{decimal_places}f}"
            elif currency == 'KRW':
                return f"₩{amount:,.0f}"
            elif currency == 'JPY':
                return f"¥{amount:,.0f}"
            elif currency == 'INR':
                return f"₹{amount:,.{decimal_places}f}"
            else:
                return f"{amount:,.{decimal_places}f} {currency}"
        except:
            return f"{amount} {currency}"

# ============================================================================
# ⏰ 시간 유틸리티
# ============================================================================

class TimeUtils:
    """시간 관련 유틸리티"""
    
    @staticmethod
    def is_market_hours(market: str = 'US') -> bool:
        """시장 시간 체크"""
        now = datetime.now()
        weekday = now.weekday()  # 0=월요일, 6=일요일
        
        # 주말 체크
        if weekday >= 5:  # 토, 일
            return False
        
        hour = now.hour
        
        if market.upper() == 'US':
            # 미국 시장: 22:30 - 05:00 (한국 시간)
            return hour >= 22 or hour < 5
        elif market.upper() == 'JP':
            # 일본 시장: 09:00 - 15:00 (한국 시간)
            return 9 <= hour < 15
        elif market.upper() == 'IN':
            # 인도 시장: 12:15 - 18:30 (한국 시간)
            return 12 <= hour < 19
        elif market.upper() == 'CRYPTO':
            # 암호화폐: 24시간
            return True
        
        return False
    
    @staticmethod
    def get_next_market_open(market: str = 'US') -> datetime:
        """다음 시장 개장 시간"""
        now = datetime.now()
        
        if market.upper() == 'US':
            # 미국 시장 개장: 22:30 (한국 시간)
            target_hour = 22
            target_minute = 30
        elif market.upper() == 'JP':
            # 일본 시장 개장: 09:00
            target_hour = 9
            target_minute = 0
        elif market.upper() == 'IN':
            # 인도 시장 개장: 12:15
            target_hour = 12
            target_minute = 15
        else:
            return now  # 암호화폐는 항상 오픈
        
        # 오늘 개장 시간
        today_open = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        if now < today_open and now.weekday() < 5:
            return today_open
        
        # 다음 평일 개장 시간
        days_ahead = 1
        while (now + timedelta(days=days_ahead)).weekday() >= 5:
            days_ahead += 1
        
        return today_open + timedelta(days=days_ahead)
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """시간 경과 포맷팅"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            return f"{seconds/60:.1f}분"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}시간"
        else:
            return f"{seconds/86400:.1f}일"
    
    @staticmethod
    def get_korean_weekday() -> str:
        """한국어 요일 반환"""
        weekdays = ['월', '화', '수', '목', '금', '토', '일']
        return weekdays[datetime.now().weekday()]

# ============================================================================
# 🔒 보안 유틸리티
# ============================================================================

class SecurityUtils:
    """보안 관련 유틸리티"""
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """API 키 생성"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """비밀번호 해시"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """비밀번호 검증"""
        try:
            salt, password_hash = hashed.split(':')
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash == new_hash.hex()
        except:
            return False
    
    @staticmethod
    def mask_sensitive_string(text: str, show_chars: int = 4) -> str:
        """민감한 문자열 마스킹"""
        if not text or len(text) <= show_chars:
            return "***"
        return f"{text[:show_chars]}***"
    
    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """API 키 형식 검증"""
        if not api_key:
            return False
        
        # 최소 16자, 영숫자+특수문자
        if len(api_key) < 16:
            return False
        
        # 알파벳과 숫자가 모두 포함되어야 함
        has_alpha = any(c.isalpha() for c in api_key)
        has_digit = any(c.isdigit() for c in api_key)
        
        return has_alpha and has_digit

# ============================================================================
# 📈 성능 모니터링 유틸리티
# ============================================================================

class PerformanceMonitor:
    """성능 모니터링 도구"""
    
    def __init__(self):
        self.start_times = {}
        self.metrics = {}
    
    def start_timer(self, name: str):
        """타이머 시작"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """타이머 종료 및 실행 시간 반환"""
        if name not in self.start_times:
            return 0.0
        
        execution_time = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(execution_time)
        del self.start_times[name]
        
        return execution_time
    
    def get_average_time(self, name: str) -> float:
        """평균 실행 시간"""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_system_stats() -> Dict[str, Any]:
        """시스템 통계"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                'boot_time': psutil.boot_time(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

# ============================================================================
# 💾 메모리 관리 유틸리티
# ============================================================================

class MemoryUtils:
    """메모리 관리 도구"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """메모리 사용량 조회"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent,
                'free_gb': memory.free / (1024**3)
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def check_memory_threshold(threshold_percent: float = 90.0) -> bool:
        """메모리 임계값 체크"""
        try:
            memory_percent = psutil.virtual_memory().percent
            return memory_percent >= threshold_percent
        except:
            return False
    
    @staticmethod
    def force_garbage_collection():
        """강제 가비지 컬렉션"""
        import gc
        collected = gc.collect()
        return collected

# ============================================================================
# 📧 알림 유틸리티
# ============================================================================

class NotificationUtils:
    """알림 관련 유틸리티"""
    
    @staticmethod
    async def send_telegram_message(
        bot_token: str, 
        chat_id: str, 
        message: str, 
        parse_mode: str = 'HTML'
    ) -> bool:
        """텔레그램 메시지 전송"""
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=10) as response:
                    return response.status == 200
                    
        except Exception as e:
            logging.error(f"텔레그램 전송 실패: {e}")
            return False
    
    @staticmethod
    def send_email_sync(
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        to_email: str,
        subject: str,
        message: str
    ) -> bool:
        """동기 이메일 전송"""
        try:
            msg = MimeMultipart()
            msg['From'] = username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MimeText(message, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logging.error(f"이메일 전송 실패: {e}")
            return False

# ============================================================================
# 🗄️ 데이터베이스 유틸리티
# ============================================================================

class DatabaseUtils:
    """데이터베이스 관련 유틸리티"""
    
    @staticmethod
    def create_backup(db_path: str, backup_dir: str) -> str:
        """데이터베이스 백업 생성"""
        try:
            import shutil
            
            db_file = Path(db_path)
            if not db_file.exists():
                raise FileNotFoundError(f"데이터베이스 파일 없음: {db_path}")
            
            backup_dir_path = Path(backup_dir)
            backup_dir_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{db_file.stem}_backup_{timestamp}{db_file.suffix}"
            backup_path = backup_dir_path / backup_filename
            
            shutil.copy2(db_path, backup_path)
            
            return str(backup_path)
            
        except Exception as e:
            logging.error(f"데이터베이스 백업 실패: {e}")
            raise
    
    @staticmethod
    def execute_query_safe(db_path: str, query: str, params: tuple = None) -> List[tuple]:
        """안전한 쿼리 실행"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
            conn.commit()
            conn.close()
            
            return results
            
        except Exception as e:
            logging.error(f"쿼리 실행 실패: {e}")
            if 'conn' in locals():
                conn.close()
            raise
    
    @staticmethod
    def get_table_info(db_path: str, table_name: str) -> Dict[str, Any]:
        """테이블 정보 조회"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 테이블 존재 확인
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not cursor.fetchone():
                return {'exists': False}
            
            # 테이블 스키마 정보
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # 행 수 조회
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'exists': True,
                'columns': [
                    {
                        'id': col[0],
                        'name': col[1],
                        'type': col[2],
                        'not_null': bool(col[3]),
                        'default_value': col[4],
                        'primary_key': bool(col[5])
                    }
                    for col in columns
                ],
                'row_count': row_count
            }
            
        except Exception as e:
            logging.error(f"테이블 정보 조회 실패: {e}")
            return {'exists': False, 'error': str(e)}

# ============================================================================
# 🧮 수학 및 통계 유틸리티
# ============================================================================

class MathUtils:
    """수학 및 통계 관련 유틸리티"""
    
    @staticmethod
    def calculate_moving_average(data: List[float], window: int) -> List[float]:
        """이동평균 계산"""
        if len(data) < window:
            return []
        
        moving_averages = []
        for i in range(len(data) - window + 1):
            window_data = data[i:i + window]
            avg = sum(window_data) / window
            moving_averages.append(avg)
        
        return moving_averages
    
    @staticmethod
    def calculate_volatility(prices: List[float]) -> float:
        """변동성 계산 (표준편차)"""
        if len(prices) < 2:
            return 0.0
        
        try:
            import statistics
            return statistics.stdev(prices)
        except:
            # 수동 계산
            mean = sum(prices) / len(prices)
            variance = sum((x - mean) ** 2 for x in prices) / (len(prices) - 1)
            return variance ** 0.5
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        if len(returns) < 2:
            return 0.0
        
        try:
            import statistics
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            
            if std_return == 0:
                return 0.0
            
            return (mean_return - risk_free_rate) / std_return
        except:
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> Dict[str, float]:
        """최대 낙폭 계산"""
        if len(prices) < 2:
            return {'max_drawdown': 0.0, 'peak': 0.0, 'trough': 0.0}
        
        peak = prices[0]
        max_drawdown = 0.0
        peak_price = prices[0]
        trough_price = prices[0]
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                peak_price = peak
                trough_price = price
        
        return {
            'max_drawdown': max_drawdown * 100,  # 퍼센트
            'peak': peak_price,
            'trough': trough_price
        }
    
    @staticmethod
    def calculate_correlation(x: List[float], y: List[float]) -> float:
        """상관계수 계산"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            import statistics
            
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
            
            sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
            sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
            
            denominator = (sum_sq_x * sum_sq_y) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        except:
            return 0.0

# ============================================================================
# 🎯 전략 유틸리티
# ============================================================================

class StrategyUtils:
    """트레이딩 전략 관련 유틸리티"""
    
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """포지션 크기 계산 (리스크 관리)"""
        try:
            if entry_price <= 0 or stop_loss_price <= 0:
                return 0.0
            
            risk_amount = account_balance * (risk_percent / 100)
            price_diff = abs(entry_price - stop_loss_price)
            
            if price_diff == 0:
                return 0.0
            
            position_size = risk_amount / price_diff
            return position_size
            
        except:
            return 0.0
    
    @staticmethod
    def calculate_kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """켈리 공식 계산"""
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            b = avg_win / avg_loss  # 승률 대비 손실 비율
            p = win_rate  # 승률
            q = 1 - win_rate  # 패율
            
            kelly_fraction = (b * p - q) / b
            
            # 켈리 비율을 25%로 제한 (보수적 접근)
            return min(max(kelly_fraction, 0.0), 0.25)
            
        except:
            return 0.0
    
    @staticmethod
    def calculate_risk_reward_ratio(
        entry_price: float,
        target_price: float,
        stop_loss_price: float
    ) -> float:
        """리스크 리워드 비율 계산"""
        try:
            potential_profit = abs(target_price - entry_price)
            potential_loss = abs(entry_price - stop_loss_price)
            
            if potential_loss == 0:
                return float('inf') if potential_profit > 0 else 0.0
            
            return potential_profit / potential_loss
            
        except:
            return 0.0
    
    @staticmethod
    def is_valid_trade_signal(
        current_price: float,
        signal_price: float,
        max_slippage_percent: float = 1.0
    ) -> bool:
        """거래 신호 유효성 검증"""
        try:
            if current_price <= 0 or signal_price <= 0:
                return False
            
            slippage = abs(current_price - signal_price) / signal_price * 100
            return slippage <= max_slippage_percent
            
        except:
            return False

# ============================================================================
# 🔍 에러 처리 유틸리티
# ============================================================================

class ErrorHandlingUtils:
    """에러 처리 관련 유틸리티"""
    
    @staticmethod
    def retry_on_exception(
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        """예외 발생시 재시도 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                retries = 0
                current_delay = delay
                
                while retries < max_retries:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        retries += 1
                        if retries >= max_retries:
                            logging.error(f"최대 재시도 횟수 초과: {func.__name__} - {e}")
                            raise
                        
                        logging.warning(f"재시도 {retries}/{max_retries}: {func.__name__} - {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                
                return None
            return wrapper
        return decorator
    
    @staticmethod
    def async_retry_on_exception(
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        """비동기 함수용 재시도 데코레이터"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                retries = 0
                current_delay = delay
                
                while retries < max_retries:
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        retries += 1
                        if retries >= max_retries:
                            logging.error(f"최대 재시도 횟수 초과: {func.__name__} - {e}")
                            raise
                        
                        logging.warning(f"재시도 {retries}/{max_retries}: {func.__name__} - {e}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                
                return None
            return wrapper
        return decorator
    
    @staticmethod
    def safe_execute(func, default_return=None, log_errors=True):
        """안전한 함수 실행"""
        try:
            return func()
        except Exception as e:
            if log_errors:
                logging.error(f"함수 실행 실패: {func.__name__} - {e}")
            return default_return
    
    @staticmethod
    async def safe_execute_async(func, default_return=None, log_errors=True):
        """안전한 비동기 함수 실행"""
        try:
            return await func()
        except Exception as e:
            if log_errors:
                logging.error(f"비동기 함수 실행 실패: {func.__name__} - {e}")
            return default_return

# ============================================================================
# 📋 시스템 정보 유틸리티
# ============================================================================

class SystemInfoUtils:
    """시스템 정보 관련 유틸리티"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """시스템 정보 조회"""
        try:
            import platform
            
            return {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'hostname': socket.gethostname(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """의존성 모듈 체크"""
        dependencies = {
            'pandas': False,
            'numpy': False,
            'requests': False,
            'aiohttp': False,
            'psutil': False,
            'yaml': False,
            'sqlite3': False,
            'pyupbit': False,
            'ib_insync': False
        }
        
        for module in dependencies.keys():
            try:
                __import__(module)
                dependencies[module] = True
            except ImportError:
                dependencies[module] = False
        
        return dependencies
    
    @staticmethod
    def get_process_info() -> Dict[str, Any]:
        """현재 프로세스 정보"""
        try:
            process = psutil.Process()
            
            return {
                'pid': process.pid,
                'name': process.name(),
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_info_mb': process.memory_info().rss / (1024 * 1024),
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat(),
                'status': process.status(),
                'num_threads': process.num_threads()
            }
        except Exception as e:
            return {'error': str(e)}

# ============================================================================
# 📊 리포팅 유틸리티
# ============================================================================

class ReportUtils:
    """리포트 생성 관련 유틸리티"""
    
    @staticmethod
    def generate_performance_report(
        portfolio_data: Dict[str, Any],
        timeframe: str = 'daily'
    ) -> str:
        """성과 리포트 생성"""
        try:
            report = f"""
📊 퀸트프로젝트 성과 리포트 ({timeframe})
{'='*50}
📅 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💼 포트폴리오 현황:
• 총 포지션: {portfolio_data.get('total_positions', 0)}개
• 총 가치: {DataUtils.format_currency(portfolio_data.get('total_usd_value', 0))}
• 미실현 손익: {DataUtils.format_currency(portfolio_data.get('total_unrealized_pnl', 0))}
• 총 수익률: {portfolio_data.get('total_return_pct', 0):+.2f}%

🎯 전략별 현황:
"""
            
            for strategy, data in portfolio_data.get('by_strategy', {}).items():
                report += f"• {strategy.upper()}: {data.get('count', 0)}개 포지션, "
                report += f"{DataUtils.format_currency(data.get('usd_value', 0))}"
                report += f" ({DataUtils.format_currency(data.get('unrealized_pnl', 0))})\n"
            
            # 상위 수익 종목
            top_gainers = portfolio_data.get('top_gainers', [])
            if top_gainers:
                report += "\n🏆 상위 수익 종목:\n"
                for i, gainer in enumerate(top_gainers[:3], 1):
                    report += f"{i}. {gainer.get('symbol', 'N/A')} "
                    report += f"({gainer.get('strategy', 'N/A')}): "
                    report += f"{gainer.get('pnl_pct', 0):+.1f}%\n"
            
            # 하위 수익 종목
            top_losers = portfolio_data.get('top_losers', [])
            if top_losers:
                report += "\n📉 하위 수익 종목:\n"
                for i, loser in enumerate(top_losers[:3], 1):
                    report += f"{i}. {loser.get('symbol', 'N/A')} "
                    report += f"({loser.get('strategy', 'N/A')}): "
                    report += f"{loser.get('pnl_pct', 0):+.1f}%\n"
            
            return report
            
        except Exception as e:
            return f"리포트 생성 실패: {e}"
    
    @staticmethod
    def generate_system_status_report(system_status: Dict[str, Any]) -> str:
        """시스템 상태 리포트 생성"""
        try:
            report = f"""
🔧 시스템 상태 리포트
{'='*40}
📅 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚙️ 시스템 현황:
• 실행 상태: {'🟢 정상' if system_status.get('system', {}).get('is_running') else '🔴 중지'}
• 응급 모드: {'🚨 활성화' if system_status.get('system', {}).get('emergency_mode') else '⭕ 비활성화'}
• 마지막 체크: {system_status.get('system', {}).get('last_health_check', 'N/A')}

🎯 전략 현황:
• 활성화된 전략: {len(system_status.get('strategies', {}).get('active_strategies', []))}개
• 전략 목록: {', '.join(system_status.get('strategies', {}).get('active_strategies', []))}

🌐 네트워크 상태:
• 연결 상태: {'🟢 연결됨' if system_status.get('network', {}).get('is_connected') else '🔴 끊김'}
• 지연시간: {system_status.get('network', {}).get('latency_ms', 0):.1f}ms
• 가동률: {system_status.get('network', {}).get('uptime_percentage', 0):.1f}%

💱 환율 정보:
• 마지막 업데이트: {system_status.get('ibkr_exchange', {}).get('last_update', 'N/A')}
• 자동 환전: {'🟢 활성화' if system_status.get('ibkr_exchange', {}).get('auto_conversion') else '🔴 비활성화'}
"""
            
            # 환율 정보
            rates = system_status.get('ibkr_exchange', {}).get('rates', {})
            if rates:
                report += "• 현재 환율:\n"
                for currency, rate in rates.items():
                    if currency == 'USD':
                        report += f"  - {currency}/KRW: {rate:,.0f}\n"
                    else:
                        report += f"  - {currency}/KRW: {rate:.3f}\n"
            
            return report
            
        except Exception as e:
            return f"시스템 상태 리포트 생성 실패: {e}"

# ============================================================================
# 🎨 출력 포맷팅 유틸리티
# ============================================================================

class FormatUtils:
    """출력 포맷팅 관련 유틸리티"""
    
    @staticmethod
    def create_table(headers: List[str], rows: List[List[str]], max_width: int = 100) -> str:
        """테이블 형태 문자열 생성"""
        if not headers or not rows:
            return ""
        
        # 컬럼 너비 계산
        col_widths = []
        for i, header in enumerate(headers):
            max_width_for_col = len(header)
            for row in rows:
                if i < len(row):
                    max_width_for_col = max(max_width_for_col, len(str(row[i])))
            col_widths.append(min(max_width_for_col, max_width // len(headers)))
        
        # 테이블 생성
        table = []
        
        # 헤더
        header_row = "| "
        for i, header in enumerate(headers):
            header_row += f"{header:<{col_widths[i]}} | "
        table.append(header_row)
        
        # 구분선
        separator = "+"
        for width in col_widths:
            separator += "-" * (width + 2) + "+"
        table.append(separator)
        
        # 데이터 행
        for row in rows:
            data_row = "| "
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_str = str(cell)[:col_widths[i]]
                    data_row += f"{cell_str:<{col_widths[i]}} | "
            table.append(data_row)
        
        return "\n".join(table)
    
    @staticmethod
    def create_progress_bar(current: int, total: int, width: int = 50) -> str:
        """진행률 바 생성"""
        if total <= 0:
            return "[" + "?" * width + "]"
        
        progress = min(current / total, 1.0)
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = progress * 100
        
        return f"[{bar}] {percentage:.1f}%"
    
    @staticmethod
    def colorize_text(text: str, color: str = 'white') -> str:
        """텍스트 색상 적용 (ANSI 색상 코드)"""
        colors = {
            'black': '\033[30m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'reset': '\033[0m'
        }
        
        color_code = colors.get(color.lower(), colors['white'])
        reset_code = colors['reset']
        
        return f"{color_code}{text}{reset_code}"

# ============================================================================
# 🔄 백그라운드 작업 유틸리티
# ============================================================================

class BackgroundTaskUtils:
    """백그라운드 작업 관련 유틸리티"""
    
    def __init__(self):
        self.tasks = {}
        self.is_running = False
    
    async def start_periodic_task(
        self,
        name: str,
        func,
        interval_seconds: int,
        *args,
        **kwargs
    ):
        """주기적 작업 시작"""
        if name in self.tasks:
            self.stop_task(name)
        
        async def periodic_wrapper():
            while self.is_running:
                try:
                    if asyncio.iscoroutinefunction(func):
                        await func(*args, **kwargs)
                    else:
                        func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"주기적 작업 실패 {name}: {e}")
                
                await asyncio.sleep(interval_seconds)
        
        task = asyncio.create_task(periodic_wrapper())
        self.tasks[name] = task
        self.is_running = True
        
        logging.info(f"주기적 작업 시작: {name} (간격: {interval_seconds}초)")
    
    def stop_task(self, name: str):
        """특정 작업 중지"""
        if name in self.tasks:
            self.tasks[name].cancel()
            del self.tasks[name]
            logging.info(f"작업 중지: {name}")
    
    def stop_all_tasks(self):
        """모든 작업 중지"""
        self.is_running = False
        for name, task in self.tasks.items():
            task.cancel()
            logging.info(f"작업 중지: {name}")
        self.tasks.clear()
    
    def get_task_status(self) -> Dict[str, str]:
        """작업 상태 조회"""
        status = {}
        for name, task in self.tasks.items():
            if task.done():
                if task.exception():
                    status[name] = f"에러: {task.exception()}"
                else:
                    status[name] = "완료"
            elif task.cancelled():
                status[name] = "취소됨"
            else:
                status[name] = "실행 중"
        
        return status

# ============================================================================
# 📤 내보내기/가져오기 유틸리티
# ============================================================================

class ExportImportUtils:
    """데이터 내보내기/가져오기 유틸리티"""
    
    @staticmethod
    def export_to_csv(data: List[Dict], filename: str, encoding: str = 'utf-8-sig') -> bool:
        """딕셔너리 리스트를 CSV로 내보내기"""
        try:
            if not data:
                return False
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding=encoding)
            logging.info(f"CSV 내보내기 완료: {filename}")
            return True
            
        except Exception as e:
            logging.error(f"CSV 내보내기 실패: {e}")
            return False
    
    @staticmethod
    def import_from_csv(filename: str, encoding: str = 'utf-8') -> List[Dict]:
        """CSV에서 딕셔너리 리스트로 가져오기"""
        try:
            df = pd.read_csv(filename, encoding=encoding)
            data = df.to_dict('records')
            logging.info(f"CSV 가져오기 완료: {filename} ({len(data)}행)")
            return data
            
        except Exception as e:
            logging.error(f"CSV 가져오기 실패: {e}")
            return []
    
    @staticmethod
    def export_to_json(data: Any, filename: str, indent: int = 2) -> bool:
        """JSON으로 내보내기"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
            logging.info(f"JSON 내보내기 완료: {filename}")
            return True
            
        except Exception as e:
            logging.error(f"JSON 내보내기 실패: {e}")
            return False
    
    @staticmethod
    def import_from_json(filename: str) -> Any:
        """JSON에서 가져오기"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logging.info(f"JSON 가져오기 완료: {filename}")
            return data
            
        except Exception as e:
            logging.error(f"JSON 가져오기 실패: {e}")
            return None

# ============================================================================
# 🎯 메인 유틸리티 클래스
# ============================================================================

class QuintUtils:
    """퀸트프로젝트 통합 유틸리티 클래스"""
    
    def __init__(self):
        self.logger_manager = LoggerManager()
        self.config_validator = ConfigValidator()
        self.file_utils = FileSystemUtils()
        self.network_utils = NetworkUtils()
        self.data_utils = DataUtils()
        self.time_utils = TimeUtils()
        self.security_utils = SecurityUtils()
        self.performance_monitor = PerformanceMonitor()
        self.memory_utils = MemoryUtils()
        self.notification_utils = NotificationUtils()
        self.database_utils = DatabaseUtils()
        self.math_utils = MathUtils()
        self.strategy_utils = StrategyUtils()
        self.error_handling_utils = ErrorHandlingUtils()
        self.system_info_utils = SystemInfoUtils()
        self.report_utils = ReportUtils()
        self.format_utils = FormatUtils()
        self.background_task_utils = BackgroundTaskUtils()
        self.export_import_utils = ExportImportUtils()
    
    def get_version(self) -> str:
        """유틸리티 버전 반환"""
        return "1.0.0"
    
    def get_available_utilities(self) -> List[str]:
        """사용 가능한 유틸리티 목록"""
        return [
            'logger_manager', 'config_validator', 'file_utils', 'network_utils',
            'data_utils', 'time_utils', 'security_utils', 'performance_monitor',
            'memory_utils', 'notification_utils', 'database_utils', 'math_utils',
            'strategy_utils', 'error_handling_utils', 'system_info_utils',
            'report_utils', 'format_utils', 'background_task_utils', 'export_import_utils'
        ]

# ============================================================================
# 🧪 테스트 도구
# ============================================================================

def run_utility_tests():
    """유틸리티 함수 테스트"""
    print("🧪 퀸트프로젝트 유틸리티 테스트 시작")
    print("=" * 50)
    
    # 1. 데이터 유틸리티 테스트
    print("📊 데이터 유틸리티 테스트:")
    print(f"   safe_float_convert('123.45'): {DataUtils.safe_float_convert('123.45')}")
    print(f"   safe_int_convert('123.67'): {DataUtils.safe_int_convert('123.67')}")
    print(f"   calculate_percentage_change(100, 150): {DataUtils.calculate_percentage_change(100, 150):.1f}%")
    print(f"   format_currency(1234567.89, 'USD'): {DataUtils.format_currency(1234567.89, 'USD')}")
    
    # 2. 시간 유틸리티 테스트
    print("\n⏰ 시간 유틸리티 테스트:")
    print(f"   is_market_hours('US'): {TimeUtils.is_market_hours('US')}")
    print(f"   get_korean_weekday(): {TimeUtils.get_korean_weekday()}")
    print(f"   format_duration(3661): {TimeUtils.format_duration(3661)}")
    
    # 3. 수학 유틸리티 테스트
    print("\n🧮 수학 유틸리티 테스트:")
    test_prices = [100, 105, 103, 108, 110, 107, 112]
    print(f"   test_prices: {test_prices}")
    print(f"   calculate_volatility: {MathUtils.calculate_volatility(test_prices):.2f}")
    
    ma = MathUtils.calculate_moving_average(test_prices, 3)
    print(f"   moving_average(3): {[round(x, 2) for x in ma]}")
    
    drawdown = MathUtils.calculate_max_drawdown(test_prices)
    print(f"   max_drawdown: {drawdown['max_drawdown']:.2f}%")
    
    # 4. 전략 유틸리티 테스트
    print("\n🎯 전략 유틸리티 테스트:")
    position_size = StrategyUtils.calculate_position_size(10000, 2, 100, 95)
    print(f"   position_size(10k, 2%, 100, 95): {position_size:.2f}")
    
    rr_ratio = StrategyUtils.calculate_risk_reward_ratio(100, 110, 95)
    print(f"   risk_reward_ratio(100, 110, 95): {rr_ratio:.2f}")
    
    kelly = StrategyUtils.calculate_kelly_criterion(0.6, 15, 10)
    print(f"   kelly_criterion(60%, 15, 10): {kelly:.3f}")
    
    # 5. 시스템 정보 테스트
    print("\n🔧 시스템 정보 테스트:")
    system_info = SystemInfoUtils.get_system_info()
    print(f"   플랫폼: {system_info.get('platform', 'N/A')}")
    print(f"   CPU 코어: {system_info.get('cpu_count', 'N/A')}")
    print(f"   메모리: {system_info.get('memory_total_gb', 0):.1f}GB")
    
    # 6. 메모리 정보 테스트
    print("\n💾 메모리 정보 테스트:")
    memory_info = MemoryUtils.get_memory_usage()
    print(f"   사용률: {memory_info.get('percent', 0):.1f}%")
    print(f"   사용량: {memory_info.get('used_gb', 0):.1f}GB")
    
    # 7. 성능 모니터 테스트
    print("\n📈 성능 모니터 테스트:")
    perf = PerformanceMonitor()
    perf.start_timer('test_operation')
    time.sleep(0.1)  # 0.1초 대기
    execution_time = perf.end_timer('test_operation')
    print(f"   실행 시간: {execution_time:.3f}초")
    
    # 8. 포맷팅 테스트
    print("\n🎨 포맷팅 테스트:")
    headers = ['Symbol', 'Price', 'Change%']
    rows = [
        ['AAPL', '$150.25', '+2.5%'],
        ['GOOGL', '$2,750.00', '-1.2%'],
        ['TSLA', '$800.50', '+5.8%']
    ]
    table = FormatUtils.create_table(headers, rows)
    print("   테이블 예시:")
    print(table)
    
    progress = FormatUtils.create_progress_bar(75, 100)
    print(f"\n   진행률 바: {progress}")
    
    print("\n✅ 유틸리티 테스트 완료!")
    print("=" * 50)

# ============================================================================
# 🔧 메인 실행부
# ============================================================================

if __name__ == "__main__":
    # 유틸리티 테스트 실행
    run_utility_tests()
    
    print("\n🛠️ 퀸트프로젝트 유틸리티 모듈")
    print("사용 예시:")
    print("from utils import QuintUtils")
    print("utils = QuintUtils()")
    print("logger = utils.logger_manager.setup_logger('my_logger')")
    print("memory_info = utils.memory_utils.get_memory_usage()")
    print("system_stats = utils.performance_monitor.get_system_stats()")
