"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 - 4대 시장 통합 유틸리티 UTILS.PY
================================================================

🌟 핵심 기능:
- 🔧 설정 관리 및 검증 시스템
- 📊 데이터 처리 및 변환 유틸리티
- 🛡️ 보안 및 암호화 시스템
- 📈 기술지표 계산 라이브러리
- 🌐 네트워크 및 API 헬퍼
- 📱 알림 및 로깅 시스템
- 🔄 백업 및 복구 시스템
- 📊 성과 분석 도구

⚡ 혼자 보수유지 가능한 완전 자동화 유틸리티
💎 모듈화된 헬퍼 함수들
🛡️ 에러 핸들링 및 복구 시스템

Author: 퀸트팀 | Version: ULTIMATE
Date: 2024.12
"""

import os
import sys
import json
import yaml
import pickle
import hashlib
import sqlite3
import logging
import asyncio
import aiohttp
import smtplib
import zipfile
import shutil
import psutil
import threading
import functools
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from urllib.parse import urlparse
import traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# 선택적 import (없어도 기본 기능 동작)
try:
    import telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

try:
    import slack_sdk
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# ============================================================================
# 🔐 보안 및 암호화 시스템
# ============================================================================
class QuintSecurity:
    """퀸트프로젝트 보안 관리자"""
    
    def __init__(self):
        self.key_file = ".quint_key"
        self.encrypted_file = ".quint_secrets.enc"
        self._cipher = None
        self._initialize_security()
    
    def _initialize_security(self):
        """보안 시스템 초기화"""
        if not Path(self.key_file).exists():
            self._generate_key()
        self._load_key()
    
    def _generate_key(self):
        """암호화 키 생성"""
        key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(key)
        os.chmod(self.key_file, 0o600)  # 소유자만 읽기 가능
    
    def _load_key(self):
        """암호화 키 로드"""
        try:
            with open(self.key_file, 'rb') as f:
                key = f.read()
            self._cipher = Fernet(key)
        except Exception as e:
            QuintLogger.error(f"암호화 키 로드 실패: {e}")
            self._generate_key()
            self._load_key()
    
    def encrypt_data(self, data: Union[str, dict]) -> bytes:
        """데이터 암호화"""
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            if isinstance(data, str):
                data = data.encode()
            return self._cipher.encrypt(data)
        except Exception as e:
            QuintLogger.error(f"데이터 암호화 실패: {e}")
            return b""
    
    def decrypt_data(self, encrypted_data: bytes) -> Union[str, dict]:
        """데이터 복호화"""
        try:
            decrypted = self._cipher.decrypt(encrypted_data)
            data_str = decrypted.decode()
            try:
                return json.loads(data_str)
            except:
                return data_str
        except Exception as e:
            QuintLogger.error(f"데이터 복호화 실패: {e}")
            return {}
    
    def save_secrets(self, secrets: Dict[str, Any]):
        """비밀 정보 암호화 저장"""
        try:
            encrypted = self.encrypt_data(secrets)
            with open(self.encrypted_file, 'wb') as f:
                f.write(encrypted)
            os.chmod(self.encrypted_file, 0o600)
            QuintLogger.info("비밀 정보 저장 완료")
        except Exception as e:
            QuintLogger.error(f"비밀 정보 저장 실패: {e}")
    
    def load_secrets(self) -> Dict[str, Any]:
        """비밀 정보 복호화 로드"""
        try:
            if not Path(self.encrypted_file).exists():
                return {}
            
            with open(self.encrypted_file, 'rb') as f:
                encrypted = f.read()
            
            return self.decrypt_data(encrypted)
        except Exception as e:
            QuintLogger.error(f"비밀 정보 로드 실패: {e}")
            return {}
    
    def hash_string(self, text: str) -> str:
        """문자열 해시화"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def validate_api_key(self, api_key: str, service: str) -> bool:
        """API 키 유효성 검증"""
        if not api_key or len(api_key) < 10:
            return False
        
        # 서비스별 기본 검증
        validators = {
            'upbit': lambda k: k.startswith('UPBIT') and len(k) >= 32,
            'telegram': lambda k: ':' in k and len(k.split(':')[1]) >= 32,
            'ibkr': lambda k: k.isalnum() and 6 <= len(k) <= 20,
            'openai': lambda k: k.startswith('sk-') and len(k) >= 40
        }
        
        validator = validators.get(service.lower())
        return validator(api_key) if validator else True

# 전역 보안 관리자
security = QuintSecurity()

# ============================================================================
# 📝 로깅 시스템
# ============================================================================
class QuintLogger:
    """퀸트프로젝트 통합 로깅 시스템"""
    
    _loggers = {}
    _handlers_added = False
    
    @classmethod
    def setup(cls, log_level: str = 'INFO', log_file: str = 'quint.log'):
        """로깅 시스템 설정"""
        if cls._handlers_added:
            return
        
        # 로그 디렉토리 생성
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # 로그 파일 경로
        log_path = log_dir / log_file
        
        # 포매터 설정
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 파일 핸들러 (로테이션 지원)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 핸들러 추가
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        cls._handlers_added = True
        cls.info("퀸트프로젝트 로깅 시스템 초기화 완료")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """로거 인스턴스 반환"""
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]
    
    @classmethod
    def debug(cls, message: str, name: str = 'quint'):
        """디버그 로그"""
        cls.get_logger(name).debug(message)
    
    @classmethod
    def info(cls, message: str, name: str = 'quint'):
        """정보 로그"""
        cls.get_logger(name).info(message)
    
    @classmethod
    def warning(cls, message: str, name: str = 'quint'):
        """경고 로그"""
        cls.get_logger(name).warning(message)
    
    @classmethod
    def error(cls, message: str, name: str = 'quint'):
        """에러 로그"""
        cls.get_logger(name).error(message)
    
    @classmethod
    def critical(cls, message: str, name: str = 'quint'):
        """치명적 오류 로그"""
        cls.get_logger(name).critical(message)
    
    @classmethod
    def log_exception(cls, exception: Exception, context: str = "", name: str = 'quint'):
        """예외 로그"""
        error_msg = f"{context}: {str(exception)}\n{traceback.format_exc()}"
        cls.get_logger(name).error(error_msg)
    
    @classmethod
    def log_performance(cls, func_name: str, execution_time: float, name: str = 'performance'):
        """성능 로그"""
        cls.get_logger(name).info(f"{func_name} 실행시간: {execution_time:.4f}초")

# 로깅 시스템 초기화
QuintLogger.setup()

# ============================================================================
# 📊 설정 관리자
# ============================================================================
class QuintConfig:
    """퀸트프로젝트 통합 설정 관리자"""
    
    def __init__(self, config_file: str = "quint_config.yaml"):
        self.config_file = Path(config_file)
        self.config = {}
        self.schema = {}
        self._watchers = []
        self._load_config()
        self._load_schema()
    
    def _load_config(self):
        """설정 파일 로드"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                QuintLogger.info(f"설정 파일 로드 완료: {self.config_file}")
            else:
                self._create_default_config()
                QuintLogger.info("기본 설정 파일 생성 완료")
        except Exception as e:
            QuintLogger.error(f"설정 파일 로드 실패: {e}")
            self.config = {}
    
    def _create_default_config(self):
        """기본 설정 생성"""
        self.config = {
            'system': {
                'environment': 'development',
                'project_mode': 'simulation',
                'debug_mode': False,
                'log_level': 'INFO',
                'timezone': 'UTC',
                'language': 'ko'
            },
            'markets': {
                'us_stocks': {'enabled': True, 'allocation': 40.0},
                'upbit_crypto': {'enabled': True, 'allocation': 30.0},
                'japan_stocks': {'enabled': True, 'allocation': 20.0},
                'india_stocks': {'enabled': True, 'allocation': 10.0}
            },
            'risk_management': {
                'max_total_risk': 20.0,
                'max_daily_loss': 5.0,
                'max_correlation': 0.7,
                'circuit_breaker': True
            },
            'notifications': {
                'telegram': {'enabled': False},
                'discord': {'enabled': False},
                'email': {'enabled': False}
            }
        }
        self.save()
    
    def _load_schema(self):
        """설정 스키마 로드"""
        schema_file = Path("config_schema.yaml")
        if schema_file.exists():
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    self.schema = yaml.safe_load(f) or {}
            except Exception as e:
                QuintLogger.error(f"스키마 로드 실패: {e}")
    
    def get(self, key_path: str, default=None):
        """설정값 조회 (점 표기법)"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        # 환경변수 치환
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, default)
        
        return value
    
    def set(self, key_path: str, value: Any, save: bool = True):
        """설정값 설정"""
        keys = key_path.split('.')
        config = self.config
        
        # 중간 딕셔너리 생성
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 값 설정
        old_value = config.get(keys[-1])
        config[keys[-1]] = value
        
        # 변경 감지 및 알림
        if old_value != value:
            self._notify_watchers(key_path, old_value, value)
        
        if save:
            self.save()
    
    def save(self):
        """설정 파일 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            QuintLogger.info("설정 파일 저장 완료")
        except Exception as e:
            QuintLogger.error(f"설정 파일 저장 실패: {e}")
    
    def validate(self) -> List[str]:
        """설정 유효성 검증"""
        errors = []
        
        # 필수 필드 검사
        required_fields = [
            'system.environment',
            'system.project_mode',
            'markets.us_stocks.enabled',
            'markets.upbit_crypto.enabled'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                errors.append(f"필수 설정 누락: {field}")
        
        # 범위 검사
        range_checks = {
            'markets.us_stocks.allocation': (0, 100),
            'markets.upbit_crypto.allocation': (0, 100),
            'risk_management.max_total_risk': (0, 100),
            'risk_management.max_daily_loss': (0, 50)
        }
        
        for field, (min_val, max_val) in range_checks.items():
            value = self.get(field)
            if value is not None and not (min_val <= value <= max_val):
                errors.append(f"범위 오류 {field}: {value} (범위: {min_val}-{max_val})")
        
        # 할당 비율 합계 검사
        total_allocation = sum([
            self.get('markets.us_stocks.allocation', 0),
            self.get('markets.upbit_crypto.allocation', 0),
            self.get('markets.japan_stocks.allocation', 0),
            self.get('markets.india_stocks.allocation', 0)
        ])
        
        if abs(total_allocation - 100.0) > 0.1:
            errors.append(f"시장 할당 비율 합계 오류: {total_allocation}% (100%여야 함)")
        
        return errors
    
    def add_watcher(self, callback: Callable[[str, Any, Any], None]):
        """설정 변경 감시자 추가"""
        self._watchers.append(callback)
    
    def _notify_watchers(self, key_path: str, old_value: Any, new_value: Any):
        """설정 변경 알림"""
        for watcher in self._watchers:
            try:
                watcher(key_path, old_value, new_value)
            except Exception as e:
                QuintLogger.error(f"설정 변경 감시자 오류: {e}")
    
    def backup(self, backup_dir: str = "backups"):
        """설정 백업"""
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_path / f"config_backup_{timestamp}.yaml"
            
            shutil.copy2(self.config_file, backup_file)
            QuintLogger.info(f"설정 백업 완료: {backup_file}")
            return backup_file
        except Exception as e:
            QuintLogger.error(f"설정 백업 실패: {e}")
            return None
    
    def restore(self, backup_file: Path):
        """설정 복원"""
        try:
            if not backup_file.exists():
                raise FileNotFoundError(f"백업 파일 없음: {backup_file}")
            
            shutil.copy2(backup_file, self.config_file)
            self._load_config()
            QuintLogger.info(f"설정 복원 완료: {backup_file}")
        except Exception as e:
            QuintLogger.error(f"설정 복원 실패: {e}")

# 전역 설정 관리자
config = QuintConfig()

# ============================================================================
# 🌐 네트워크 및 API 헬퍼
# ============================================================================
class QuintNetwork:
    """퀸트프로젝트 네트워크 유틸리티"""
    
    def __init__(self):
        self.session = None
        self.rate_limiters = defaultdict(lambda: {'count': 0, 'reset_time': datetime.now()})
        self.retry_delays = [1, 2, 4, 8, 16]  # 지수 백오프
    
    async def get_session(self) -> aiohttp.ClientSession:
        """비동기 HTTP 세션 반환"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            headers = {
                'User-Agent': 'QuintProject/1.0 (Investment Analysis Bot)',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
            )
        return self.session
    
    async def close_session(self):
        """세션 종료"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def check_rate_limit(self, endpoint: str, limit: int, window_seconds: int = 60) -> bool:
        """API 요청 제한 확인"""
        now = datetime.now()
        limiter = self.rate_limiters[endpoint]
        
        # 윈도우 리셋 확인
        if now - limiter['reset_time'] > timedelta(seconds=window_seconds):
            limiter['count'] = 0
            limiter['reset_time'] = now
        
        # 제한 확인
        if limiter['count'] >= limit:
            return False
        
        limiter['count'] += 1
        return True
    
    async def request_with_retry(self, method: str, url: str, 
                               max_retries: int = 3, **kwargs) -> Optional[Dict]:
        """재시도가 포함된 HTTP 요청"""
        session = await self.get_session()
        
        for attempt in range(max_retries + 1):
            try:
                # Rate limiting
                endpoint = urlparse(url).netloc
                if not self.check_rate_limit(endpoint, 60):  # 분당 60회 제한
                    await asyncio.sleep(1)
                    continue
                
                async with session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    elif response.status == 429:  # Too Many Requests
                        retry_after = int(response.headers.get('Retry-After', 60))
                        QuintLogger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                    elif response.status >= 500:  # Server errors
                        if attempt < max_retries:
                            delay = self.retry_delays[min(attempt, len(self.retry_delays)-1)]
                            QuintLogger.warning(f"Server error {response.status}, retrying in {delay}s")
                            await asyncio.sleep(delay)
                        else:
                            QuintLogger.error(f"Server error {response.status} after {max_retries} retries")
                    else:
                        QuintLogger.error(f"HTTP error {response.status}: {await response.text()}")
                        break
                        
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays)-1)]
                    QuintLogger.warning(f"Timeout, retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    QuintLogger.error(f"Timeout after {max_retries} retries")
            except Exception as e:
                if attempt < max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays)-1)]
                    QuintLogger.warning(f"Request error: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    QuintLogger.error(f"Request failed after {max_retries} retries: {e}")
        
        return None
    
    async def get(self, url: str, **kwargs) -> Optional[Dict]:
        """GET 요청"""
        return await self.request_with_retry('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> Optional[Dict]:
        """POST 요청"""
        return await self.request_with_retry('POST', url, **kwargs)
    
    def check_internet_connection(self) -> bool:
        """인터넷 연결 확인"""
        try:
            response = requests.get('https://www.google.com', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_external_ip(self) -> Optional[str]:
        """외부 IP 주소 조회"""
        try:
            response = requests.get('https://httpbin.org/ip', timeout=10)
            return response.json().get('origin')
        except:
            return None
    
    def ping_host(self, host: str, timeout: int = 5) -> bool:
        """호스트 연결 확인"""
        import subprocess
        import platform
        
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, '1', host]
        
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout)
            return result.returncode == 0
        except:
            return False

# 전역 네트워크 헬퍼
network = QuintNetwork()

# ============================================================================
# 📊 데이터 처리 유틸리티
# ============================================================================
class QuintDataProcessor:
    """퀸트프로젝트 데이터 처리 유틸리티"""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, remove_duplicates: bool = True) -> pd.DataFrame:
        """데이터프레임 정리"""
        if df.empty:
            return df
        
        # 복사본 생성
        cleaned_df = df.copy()
        
        # 중복 제거
        if remove_duplicates:
            cleaned_df = cleaned_df.drop_duplicates()
        
        # 무한값 제거
        cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
        
        # 수치형 컬럼의 이상치 처리
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 이상치를 NaN으로 처리
            cleaned_df.loc[(cleaned_df[col] < lower_bound) | 
                          (cleaned_df[col] > upper_bound), col] = np.nan
        
        return cleaned_df
    
    @staticmethod
    def handle_missing_data(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """결측치 처리"""
        if df.empty:
            return df
        
        if method == 'forward_fill':
            return df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            return df.interpolate(method='linear')
        elif method == 'drop':
            return df.dropna()
        elif method == 'mean':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            return df
        else:
            return df
    
    @staticmethod
    def normalize_data(data: Union[np.ndarray, pd.Series], method: str = 'minmax') -> np.ndarray:
        """데이터 정규화"""
        if isinstance(data, pd.Series):
            data = data.values
        
        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            return scaler.fit_transform(data.reshape(-1, 1)).flatten()
        elif method == 'zscore':
            return (data - np.mean(data)) / np.std(data)
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            return scaler.fit_transform(data.reshape(-1, 1)).flatten()
        else:
            return data
    
    @staticmethod
    def calculate_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """상관관계 매트릭스 계산"""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)
    
    @staticmethod
    def detect_outliers(data: Union[np.ndarray, pd.Series], method: str = 'iqr') -> np.ndarray:
        """이상치 탐지"""
        if isinstance(data, pd.Series):
            data = data.values
        
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return z_scores > 3
        else:
            return np.zeros(len(data), dtype=bool)
    
    @staticmethod
    def resample_timeseries(df: pd.DataFrame, freq: str, agg_func: str = 'mean') -> pd.DataFrame:
        """시계열 데이터 리샘플링"""
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        if agg_func == 'ohlc':
            # OHLC 데이터 처리
            return df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        else:
            return df.resample(freq).agg(agg_func)
    
    @staticmethod
    def calculate_rolling_stats(data: pd.Series, window: int) -> Dict[str, pd.Series]:
        """롤링 통계 계산"""
        return {
            'mean': data.rolling(window).mean(),
            'std': data.rolling(window).std(),
            'min': data.rolling(window).min(),
            'max': data.rolling(window).max(),
            'median': data.rolling(window).median()
        }
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """래그 피처 생성"""
        result_df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    result_df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        return result_df
    
    @staticmethod
    def binning_data(data: pd.Series, bins: int = 10, method: str = 'equal_width') -> pd.Series:
        """데이터 구간화"""
        if method == 'equal_width':
            return pd.cut(data, bins=bins)
        elif method == 'equal_freq':
            return pd.qcut(data, q=bins)
        else:
            return data

# ============================================================================
# 📈 기술지표 계산 라이브러리
# ============================================================================
class QuintTechnicalIndicators:
    """퀸트프로젝트 기술지표 계산 라이브러리"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """단순이동평균 (Simple Moving Average)"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """지수이동평균 (Exponential Moving Average)"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """상대강도지수 (Relative Strength Index)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = QuintTechnicalIndicators.ema(data, fast)
        ema_slow = QuintTechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = QuintTechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """볼린저 밴드 (Bollinger Bands)"""
        sma = QuintTechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """스토캐스틱 (Stochastic Oscillator)"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """평균 참값 범위 (Average True Range)"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """피보나치 되돌림 (Fibonacci Retracement)"""
        diff = high - low
        return {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """윌리엄스 %R (Williams %R)"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """평균 방향 지수 (Average Directional Index)"""
        if TALIB_AVAILABLE:
            import talib
            return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
        else:
            # 간단한 ADX 계산
            tr = QuintTechnicalIndicators.atr(high, low, close, 1)
            dm_plus = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
            dm_minus = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0)
            
            di_plus = 100 * (dm_plus.rolling(period).mean() / tr.rolling(period).mean())
            di_minus = 100 * (dm_minus.rolling(period).mean() / tr.rolling(period).mean())
            
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            return dx.rolling(period).mean()
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series,
                       tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, pd.Series]:
        """일목균형표 (Ichimoku Cloud)"""
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
        chikou_span = close.shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
        """거래량 가중 평균가격 (Volume Weighted Average Price)"""
        return (price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """모멘텀 (Momentum)"""
        return data / data.shift(period) - 1
    
    @staticmethod
    def rate_of_change(data: pd.Series, period: int = 10) -> pd.Series:
        """변화율 (Rate of Change)"""
        return ((data - data.shift(period)) / data.shift(period)) * 100

# ============================================================================
# 📱 알림 시스템
# ============================================================================
class QuintNotification:
    """퀸트프로젝트 통합 알림 시스템"""
    
    def __init__(self):
        self.telegram_bot = None
        self.discord_webhook = None
        self.slack_client = None
        self._initialize_services()
    
    def _initialize_services(self):
        """알림 서비스 초기화"""
        # 텔레그램 초기화
        if config.get('notifications.telegram.enabled') and TELEGRAM_AVAILABLE:
            bot_token = config.get('notifications.telegram.bot_token')
            if bot_token and not bot_token.startswith('${'):
                try:
                    self.telegram_bot = telegram.Bot(token=bot_token)
                    QuintLogger.info("텔레그램 봇 초기화 완료")
                except Exception as e:
                    QuintLogger.error(f"텔레그램 봇 초기화 실패: {e}")
        
        # 디스코드 초기화
        if config.get('notifications.discord.enabled'):
            webhook_url = config.get('notifications.discord.webhook_url')
            if webhook_url:
                self.discord_webhook = webhook_url
                QuintLogger.info("디스코드 웹훅 초기화 완료")
        
        # 슬랙 초기화
        if config.get('notifications.slack.enabled') and SLACK_AVAILABLE:
            token = config.get('notifications.slack.token')
            if token:
                try:
                    self.slack_client = slack_sdk.WebClient(token=token)
                    QuintLogger.info("슬랙 클라이언트 초기화 완료")
                except Exception as e:
                    QuintLogger.error(f"슬랙 클라이언트 초기화 실패: {e}")
    
    async def send_telegram_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """텔레그램 메시지 전송"""
        if not self.telegram_bot:
            return False
        
        try:
            chat_id = config.get('notifications.telegram.chat_id')
            if not chat_id:
                QuintLogger.error("텔레그램 채팅 ID가 설정되지 않았습니다")
                return False
            
            await self.telegram_bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode
            )
            QuintLogger.debug("텔레그램 메시지 전송 완료")
            return True
        except Exception as e:
            QuintLogger.error(f"텔레그램 메시지 전송 실패: {e}")
            return False
    
    async def send_discord_message(self, message: str, embed: Dict = None) -> bool:
        """디스코드 메시지 전송"""
        if not self.discord_webhook:
            return False
        
        try:
            data = {'content': message}
            if embed:
                data['embeds'] = [embed]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=data) as response:
                    if response.status == 204:
                        QuintLogger.debug("디스코드 메시지 전송 완료")
                        return True
                    else:
                        QuintLogger.error(f"디스코드 메시지 전송 실패: {response.status}")
                        return False
        except Exception as e:
            QuintLogger.error(f"디스코드 메시지 전송 실패: {e}")
            return False
    
    def send_slack_message(self, channel: str, message: str) -> bool:
        """슬랙 메시지 전송"""
        if not self.slack_client:
            return False
        
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text=message
            )
            if response['ok']:
                QuintLogger.debug("슬랙 메시지 전송 완료")
                return True
            else:
                QuintLogger.error(f"슬랙 메시지 전송 실패: {response['error']}")
                return False
        except Exception as e:
            QuintLogger.error(f"슬랙 메시지 전송 실패: {e}")
            return False
    
    def send_email(self, to_email: str, subject: str, body: str, html: bool = False) -> bool:
        """이메일 전송"""
        try:
            smtp_server = config.get('notifications.email.smtp_server')
            smtp_port = config.get('notifications.email.smtp_port', 587)
            from_email = config.get('notifications.email.from_email')
            password = config.get('notifications.email.password')
            
            if not all([smtp_server, from_email, password]):
                QuintLogger.error("이메일 설정이 누락되었습니다")
                return False
            
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email
            msg['To'] = to_email
            
            if html:
                msg.attach(MimeText(body, 'html'))
            else:
                msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
            server.quit()
            
            QuintLogger.debug("이메일 전송 완료")
            return True
        except Exception as e:
            QuintLogger.error(f"이메일 전송 실패: {e}")
            return False
    
    async def send_alert(self, alert_type: str, title: str, message: str, 
                        priority: str = 'normal') -> Dict[str, bool]:
        """통합 알림 전송"""
        results = {}
        
        # 조용한 시간 체크
        if self._is_quiet_time():
            if priority != 'critical':
                QuintLogger.info("조용한 시간으로 인해 알림이 연기됩니다")
                return {'delayed': True}
        
        # 알림 타입별 메시지 포맷팅
        formatted_message = self._format_message(alert_type, title, message)
        
        # 텔레그램 전송
        if config.get('notifications.telegram.enabled'):
            results['telegram'] = await self.send_telegram_message(formatted_message)
        
        # 디스코드 전송
        if config.get('notifications.discord.enabled'):
            embed = self._create_discord_embed(alert_type, title, message, priority)
            results['discord'] = await self.send_discord_message(formatted_message, embed)
        
        # 슬랙 전송
        if config.get('notifications.slack.enabled'):
            channel = config.get('notifications.slack.channel', '#general')
            results['slack'] = self.send_slack_message(channel, formatted_message)
        
        # 이메일 전송 (중요한 알림만)
        if config.get('notifications.email.enabled') and priority in ['high', 'critical']:
            to_email = config.get('notifications.email.to_email')
            if to_email:
                results['email'] = self.send_email(to_email, title, message, html=True)
        
        return results
    
    def _is_quiet_time(self) -> bool:
        """조용한 시간 확인"""
        quiet_start = config.get('notifications.quiet_hours.start', '22:00')
        quiet_end = config.get('notifications.quiet_hours.end', '07:00')
        
        if not quiet_start or not quiet_end:
            return False
        
        now = datetime.now().time()
        start_time = datetime.strptime(quiet_start, '%H:%M').time()
        end_time = datetime.strptime(quiet_end, '%H:%M').time()
        
        if start_time <= end_time:
            return start_time <= now <= end_time
        else:  # 자정을 넘나드는 경우
            return now >= start_time or now <= end_time
    
    def _format_message(self, alert_type: str, title: str, message: str) -> str:
        """메시지 포맷팅"""
        emoji_map = {
            'signal': '📊',
            'trade': '💰',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }
        
        emoji = emoji_map.get(alert_type, '📢')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f"{emoji} <b>{title}</b>\n\n{message}\n\n<i>{timestamp}</i>"
    
    def _create_discord_embed(self, alert_type: str, title: str, message: str, priority: str) -> Dict:
        """디스코드 임베드 생성"""
        color_map = {
            'signal': 0x3498db,    # 파랑
            'trade': 0x2ecc71,     # 초록
            'error': 0xe74c3c,     # 빨강
            'warning': 0xf39c12,   # 주황
            'info': 0x9b59b6,      # 보라
            'success': 0x27ae60    # 진한 초록
        }
        
        return {
            'title': title,
            'description': message,
            'color': color_map.get(alert_type, 0x95a5a6),
            'timestamp': datetime.now().isoformat(),
            'footer': {
                'text': f'퀸트프로젝트 | 우선순위: {priority.upper()}'
            }
        }

# 전역 알림 관리자
notification = QuintNotification()

# ============================================================================
# 💾 백업 및 복구 시스템
# ============================================================================
class QuintBackup:
    """퀸트프로젝트 백업 및 복구 시스템"""
    
    def __init__(self):
        self.backup_dir = Path('backups')
        self.backup_dir.mkdir(exist_ok=True)
        self.max_backups = config.get('system.backup.max_backups', 30)
        self.compression_enabled = config.get('system.backup.compression', True)
    
    def create_backup(self, backup_type: str = 'full') -> Optional[Path]:
        """백업 생성"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"quint_backup_{backup_type}_{timestamp}"
            
            if self.compression_enabled:
                backup_file = self.backup_dir / f"{backup_name}.zip"
                return self._create_compressed_backup(backup_file, backup_type)
            else:
                backup_folder = self.backup_dir / backup_name
                backup_folder.mkdir(exist_ok=True)
                return self._create_folder_backup(backup_folder, backup_type)
                
        except Exception as e:
            QuintLogger.error(f"백업 생성 실패: {e}")
            return None
    
    def _create_compressed_backup(self, backup_file: Path, backup_type: str) -> Path:
        """압축 백업 생성"""
        with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 설정 파일들
            config_files = [
                'quint_config.yaml',
                'settings.yaml',
                '.env'
            ]
            
            for file_name in config_files:
                file_path = Path(file_name)
                if file_path.exists():
                    zipf.write(file_path, file_name)
            
            # 데이터 파일들
            data_files = [
                'quint_portfolio.json',
                'quint_trades.json',
                'quint_performance.json'
            ]
            
            for file_name in data_files:
                file_path = Path(file_name)
                if file_path.exists():
                    zipf.write(file_path, file_name)
            
            # 로그 파일들 (선택적)
            if backup_type == 'full':
                logs_dir = Path('logs')
                if logs_dir.exists():
                    for log_file in logs_dir.glob('*.log'):
                        zipf.write(log_file, f"logs/{log_file.name}")
            
            # 암호화된 비밀 파일
            secret_files = ['.quint_secrets.enc', '.quint_key']
            for file_name in secret_files:
                file_path = Path(file_name)
                if file_path.exists():
                    zipf.write(file_path, file_name)
        
        QuintLogger.info(f"압축 백업 생성 완료: {backup_file}")
        return backup_file
    
    def _create_folder_backup(self, backup_folder: Path, backup_type: str) -> Path:
        """폴더 백업 생성"""
        # 설정 파일 복사
        config_files = ['quint_config.yaml', 'settings.yaml', '.env']
        for file_name in config_files:
            src_path = Path(file_name)
            if src_path.exists():
                shutil.copy2(src_path, backup_folder / file_name)
        
        # 데이터 파일 복사
        data_files = ['quint_portfolio.json', 'quint_trades.json', 'quint_performance.json']
        for file_name in data_files:
            src_path = Path(file_name)
            if src_path.exists():
                shutil.copy2(src_path, backup_folder / file_name)
        
        # 로그 디렉토리 복사 (전체 백업시)
        if backup_type == 'full':
            logs_dir = Path('logs')
            if logs_dir.exists():
                shutil.copytree(logs_dir, backup_folder / 'logs', dirs_exist_ok=True)
        
        QuintLogger.info(f"폴더 백업 생성 완료: {backup_folder}")
        return backup_folder
    
    def restore_backup(self, backup_path: Path) -> bool:
        """백업 복원"""
        try:
            if not backup_path.exists():
                QuintLogger.error(f"백업 파일이 존재하지 않습니다: {backup_path}")
                return False
            
            # 현재 파일들 백업
            safety_backup = self.create_backup('safety')
            
            if backup_path.suffix == '.zip':
                return self._restore_compressed_backup(backup_path)
            else:
                return self._restore_folder_backup(backup_path)
                
        except Exception as e:
            QuintLogger.error(f"백업 복원 실패: {e}")
            return False
    
    def _restore_compressed_backup(self, backup_file: Path) -> bool:
        """압축 백업 복원"""
        try:
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                zipf.extractall('.')
            
            QuintLogger.info(f"압축 백업 복원 완료: {backup_file}")
            return True
        except Exception as e:
            QuintLogger.error(f"압축 백업 복원 실패: {e}")
            return False
    
    def _restore_folder_backup(self, backup_folder: Path) -> bool:
        """폴더 백업 복원"""
        try:
            for item in backup_folder.iterdir():
                if item.is_file():
                    shutil.copy2(item, item.name)
                elif item.is_dir():
                    if item.name == 'logs':
                        logs_dir = Path('logs')
                        if logs_dir.exists():
                            shutil.rmtree(logs_dir)
                        shutil.copytree(item, logs_dir)
            
            QuintLogger.info(f"폴더 백업 복원 완료: {backup_folder}")
            return True
        except Exception as e:
            QuintLogger.error(f"폴더 백업 복원 실패: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """백업 목록 조회"""
        backups = []
        
        for backup_path in self.backup_dir.iterdir():
            if backup_path.name.startswith('quint_backup_'):
                try:
                    # 백업 정보 파싱
                    parts = backup_path.stem.split('_')
                    backup_type = parts[2] if len(parts) >= 3 else 'unknown'
                    timestamp_str = parts[3] if len(parts) >= 4 else 'unknown'
                    
                    # 타임스탬프 파싱
                    if timestamp_str != 'unknown':
                        backup_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    else:
                        backup_time = datetime.fromtimestamp(backup_path.stat().st_mtime)
                    
                    # 파일 크기
                    if backup_path.is_file():
                        size = backup_path.stat().st_size
                    else:
                        size = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
                    
                    backups.append({
                        'path': backup_path,
                        'name': backup_path.name,
                        'type': backup_type,
                        'timestamp': backup_time,
                        'size': size,
                        'size_mb': round(size / (1024 * 1024), 2)
                    })
                    
                except Exception as e:
                    QuintLogger.warning(f"백업 정보 파싱 실패 {backup_path}: {e}")
        
        # 시간순 정렬 (최신순)
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        return backups
    
    def cleanup_old_backups(self) -> int:
        """오래된 백업 정리"""
        backups = self.list_backups()
        deleted_count = 0
        
        if len(backups) > self.max_backups:
            old_backups = backups[self.max_backups:]
            
            for backup in old_backups:
                try:
                    backup_path = backup['path']
                    if backup_path.is_file():
                        backup_path.unlink()
                    else:
                        shutil.rmtree(backup_path)
                    
                    deleted_count += 1
                    QuintLogger.info(f"오래된 백업 삭제: {backup['name']}")
                except Exception as e:
                    QuintLogger.error(f"백업 삭제 실패 {backup['name']}: {e}")
        
        return deleted_count
    
    def schedule_auto_backup(self, interval_hours: int = 24):
        """자동 백업 스케줄링"""
        def backup_task():
            while True:
                try:
                    # 백업 생성
                    backup_path = self.create_backup('auto')
                    if backup_path:
                        QuintLogger.info(f"자동 백업 완료: {backup_path}")
                    
                    # 오래된 백업 정리
                    deleted = self.cleanup_old_backups()
                    if deleted > 0:
                        QuintLogger.info(f"오래된 백업 {deleted}개 정리 완료")
                    
                except Exception as e:
                    QuintLogger.error(f"자동 백업 실패: {e}")
                
                # 다음 백업까지 대기
                time.sleep(interval_hours * 3600)
        
        # 백그라운드 스레드로 실행
        backup_thread = threading.Thread(target=backup_task, daemon=True)
        backup_thread.start()
        QuintLogger.info(f"자동 백업 스케줄링 시작 (간격: {interval_hours}시간)")

# 전역 백업 관리자
backup = QuintBackup()

# ============================================================================
# 📊 성과 분석 도구
# ============================================================================
class QuintPerformanceAnalyzer:
    """퀸트프로젝트 성과 분석 도구"""
    
    def __init__(self):
        self.performance_file = "quint_performance.json"
        self.trades_file = "quint_trades.json"
        self.benchmarks = {
            'us': '^GSPC',      # S&P 500
            'crypto': 'BTC-USD', # 비트코인
            'japan': '^N225',    # 니케이 225
            'india': '^NSEI'     # Nifty 50
        }
    
    def calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """포트폴리오 성과 지표 계산"""
        if returns.empty:
            return {}
        
        # 기본 통계
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 샤프 비율 (무위험 수익률 2% 가정)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
        
        # 최대 낙폭 (Maximum Drawdown)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 승률
        win_rate = (returns > 0).mean()
        
        # 평균 수익/손실
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        
        # 수익 팩터
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 칼마 비율
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        
        # 소르티노 비율
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def analyze_market_correlation(self, portfolio_returns: pd.Series, 
                                 market_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """시장과의 상관관계 분석"""
        correlations = {}
        
        for market, returns in market_returns.items():
            # 공통 날짜만 사용
            common_dates = portfolio_returns.index.intersection(returns.index)
            if len(common_dates) > 10:  # 최소 10일 데이터
                port_common = portfolio_returns.loc[common_dates]
                market_common = returns.loc[common_dates]
                correlation = port_common.corr(market_common)
                correlations[market] = correlation
        
        return correlations
    
    def calculate_var_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> Dict[str, float]:
        """VaR (Value at Risk)와 CVaR (Conditional VaR) 계산"""
        if returns.empty:
            return {'var': 0, 'cvar': 0}
        
        # VaR 계산
        var = returns.quantile(confidence_level)
        
        # CVaR 계산 (VaR 이하의 손실들의 평균)
        cvar = returns[returns <= var].mean()
        
        return {
            'var': var,
            'cvar': cvar
        }
    
    def analyze_trade_performance(self, trades: List[Dict]) -> Dict[str, Any]:
        """거래 성과 분석"""
        if not trades:
            return {}
        
        # 거래 데이터 처리
        df = pd.DataFrame(trades)
        
        # 기본 통계
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 수익 통계
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        best_trade = df['pnl'].max()
        worst_trade = df['pnl'].min()
        
        # 승부 분석
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # 연속 승부 분석
        df['win'] = df['pnl'] > 0
        df['streak'] = (df['win'] != df['win'].shift()).cumsum()
        streak_stats = df.groupby(['win', 'streak']).size()
        
        max_win_streak = 0
        max_loss_streak = 0
        
        for (is_win, streak), count in streak_stats.items():
            if is_win:
                max_win_streak = max(max_win_streak, count)
            else:
                max_loss_streak = max(max_loss_streak, count)
        
        # 시장별 성과
        market_performance = df.groupby('market')['pnl'].agg(['sum', 'mean', 'count']).to_dict()
        
        # 월별 성과
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')
            monthly_performance = df.groupby('month')['pnl'].sum().to_dict()
        else:
            monthly_performance = {}
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'market_performance': market_performance,
            'monthly_performance': monthly_performance
        }
    
    def generate_performance_report(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """성과 리포트 생성"""
        try:
            # 거래 데이터 로드
            trades = self.load_trades_data()
            
            # 날짜 필터링
            if start_date or end_date:
                trades = self.filter_trades_by_date(trades, start_date, end_date)
            
            # 성과 분석
            trade_analysis = self.analyze_trade_performance(trades)
            
            # 수익률 시계열 생성
            returns = self.calculate_returns_series(trades)
            portfolio_metrics = self.calculate_portfolio_metrics(returns)
            
            # 리스크 분석
            risk_metrics = self.calculate_var_cvar(returns)
            
            # 벤치마크 비교 (간소화)
            benchmark_comparison = self.compare_with_benchmarks(returns)
            
            report = {
                'report_date': datetime.now().isoformat(),
                'analysis_period': {
                    'start': start_date or (trades[0]['date'] if trades else None),
                    'end': end_date or (trades[-1]['date'] if trades else None)
                },
                'trade_analysis': trade_analysis,
                'portfolio_metrics': portfolio_metrics,
                'risk_metrics': risk_metrics,
                'benchmark_comparison': benchmark_comparison,
                'summary': self.create_performance_summary(trade_analysis, portfolio_metrics)
            }
            
            # 리포트 저장
            self.save_performance_report(report)
            
            return report
            
        except Exception as e:
            QuintLogger.error(f"성과 리포트 생성 실패: {e}")
            return {}
    
    def load_trades_data(self) -> List[Dict]:
        """거래 데이터 로드"""
        try:
            if Path(self.trades_file).exists():
                with open(self.trades_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
                                return []
        except Exception as e:
            QuintLogger.error(f"거래 데이터 로드 실패: {e}")
            return []
    
    def filter_trades_by_date(self, trades: List[Dict], start_date: str, end_date: str) -> List[Dict]:
        """날짜별 거래 필터링"""
        filtered_trades = []
        
        for trade in trades:
            trade_date = trade.get('date')
            if trade_date:
                if start_date and trade_date < start_date:
                    continue
                if end_date and trade_date > end_date:
                    continue
                filtered_trades.append(trade)
        
        return filtered_trades
    
    def calculate_returns_series(self, trades: List[Dict]) -> pd.Series:
        """거래 데이터에서 수익률 시계열 생성"""
        if not trades:
            return pd.Series()
        
        # 거래를 날짜별로 그룹화
        daily_pnl = defaultdict(float)
        
        for trade in trades:
            date = trade.get('date')
            pnl = trade.get('pnl', 0)
            if date:
                daily_pnl[date] += pnl
        
        # 시리즈 생성
        dates = sorted(daily_pnl.keys())
        pnl_values = [daily_pnl[date] for date in dates]
        
        # 수익률로 변환 (포트폴리오 가치 대비)
        portfolio_value = config.get('system.portfolio_value', 100_000_000)
        returns = pd.Series([pnl / portfolio_value for pnl in pnl_values], 
                           index=pd.to_datetime(dates))
        
        return returns
    
    def compare_with_benchmarks(self, returns: pd.Series) -> Dict[str, Dict]:
        """벤치마크 비교 (간소화 버전)"""
        comparison = {}
        
        for market, benchmark in self.benchmarks.items():
            try:
                # 실제로는 yfinance로 벤치마크 데이터를 가져와야 함
                # 여기서는 간소화된 버전
                comparison[market] = {
                    'benchmark_symbol': benchmark,
                    'correlation': 0.0,  # 실제 계산 필요
                    'relative_performance': 0.0,  # 실제 계산 필요
                    'beta': 1.0,  # 실제 계산 필요
                    'alpha': 0.0   # 실제 계산 필요
                }
            except Exception as e:
                QuintLogger.warning(f"벤치마크 {market} 비교 실패: {e}")
        
        return comparison
    
    def create_performance_summary(self, trade_analysis: Dict, portfolio_metrics: Dict) -> Dict[str, str]:
        """성과 요약 생성"""
        total_return = portfolio_metrics.get('total_return', 0)
        win_rate = trade_analysis.get('win_rate', 0)
        sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
        max_drawdown = portfolio_metrics.get('max_drawdown', 0)
        
        # 등급 평가
        performance_grade = 'F'
        if total_return > 0.20:
            performance_grade = 'A+'
        elif total_return > 0.15:
            performance_grade = 'A'
        elif total_return > 0.10:
            performance_grade = 'B+'
        elif total_return > 0.05:
            performance_grade = 'B'
        elif total_return > 0:
            performance_grade = 'C'
        
        # 종합 평가
        if sharpe_ratio > 2.0 and max_drawdown > -0.05:
            overall_rating = "탁월함"
        elif sharpe_ratio > 1.0 and max_drawdown > -0.10:
            overall_rating = "우수함"
        elif sharpe_ratio > 0.5 and max_drawdown > -0.15:
            overall_rating = "양호함"
        elif total_return > 0:
            overall_rating = "보통"
        else:
            overall_rating = "개선 필요"
        
        return {
            'performance_grade': performance_grade,
            'overall_rating': overall_rating,
            'key_strength': self.identify_key_strength(trade_analysis, portfolio_metrics),
            'improvement_area': self.identify_improvement_area(trade_analysis, portfolio_metrics)
        }
    
    def identify_key_strength(self, trade_analysis: Dict, portfolio_metrics: Dict) -> str:
        """주요 강점 식별"""
        win_rate = trade_analysis.get('win_rate', 0)
        sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
        max_drawdown = portfolio_metrics.get('max_drawdown', 0)
        
        if win_rate > 0.7:
            return "높은 승률"
        elif sharpe_ratio > 2.0:
            return "뛰어난 위험조정수익률"
        elif max_drawdown > -0.05:
            return "안정적인 수익 곡선"
        else:
            return "꾸준한 성과"
    
    def identify_improvement_area(self, trade_analysis: Dict, portfolio_metrics: Dict) -> str:
        """개선 영역 식별"""
        win_rate = trade_analysis.get('win_rate', 0)
        sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
        max_drawdown = portfolio_metrics.get('max_drawdown', 0)
        
        if max_drawdown < -0.20:
            return "리스크 관리 강화"
        elif win_rate < 0.4:
            return "매매 신호 정확도 향상"
        elif sharpe_ratio < 0.5:
            return "위험조정수익률 개선"
        else:
            return "전략 최적화"
    
    def save_performance_report(self, report: Dict):
        """성과 리포트 저장"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = Path(f"performance_report_{timestamp}.json")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            QuintLogger.info(f"성과 리포트 저장 완료: {report_file}")
        except Exception as e:
            QuintLogger.error(f"성과 리포트 저장 실패: {e}")

# 전역 성과 분석기
performance_analyzer = QuintPerformanceAnalyzer()

# ============================================================================
# 🔄 시스템 모니터링
# ============================================================================
class QuintSystemMonitor:
    """퀸트프로젝트 시스템 모니터링"""
    
    def __init__(self):
        self.monitoring_active = False
        self.metrics_history = deque(maxlen=1000)  # 최근 1000개 메트릭만 보관
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'network_error_rate': 0.05
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # 네트워크 통계
            network = psutil.net_io_counters()
            
            # 프로세스 정보
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            metrics = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory.available / 1024**3,
                'disk_percent': disk_percent,
                'disk_free_gb': disk.free / 1024**3,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_memory_mb': process_memory,
                'process_cpu_percent': process.cpu_percent()
            }
            
            return metrics
            
        except Exception as e:
            QuintLogger.error(f"시스템 메트릭 수집 실패: {e}")
            return {}
    
    def check_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태 확인"""
        metrics = self.get_system_metrics()
        
        if not metrics:
            return {'status': 'unknown', 'issues': ['메트릭 수집 실패']}
        
        issues = []
        warnings = []
        
        # CPU 사용률 확인
        if metrics['cpu_percent'] > self.alert_thresholds['cpu_percent']:
            issues.append(f"높은 CPU 사용률: {metrics['cpu_percent']:.1f}%")
        elif metrics['cpu_percent'] > self.alert_thresholds['cpu_percent'] * 0.8:
            warnings.append(f"CPU 사용률 주의: {metrics['cpu_percent']:.1f}%")
        
        # 메모리 사용률 확인
        if metrics['memory_percent'] > self.alert_thresholds['memory_percent']:
            issues.append(f"높은 메모리 사용률: {metrics['memory_percent']:.1f}%")
        elif metrics['memory_percent'] > self.alert_thresholds['memory_percent'] * 0.8:
            warnings.append(f"메모리 사용률 주의: {metrics['memory_percent']:.1f}%")
        
        # 디스크 사용률 확인
        if metrics['disk_percent'] > self.alert_thresholds['disk_percent']:
            issues.append(f"높은 디스크 사용률: {metrics['disk_percent']:.1f}%")
        elif metrics['disk_percent'] > self.alert_thresholds['disk_percent'] * 0.8:
            warnings.append(f"디스크 사용률 주의: {metrics['disk_percent']:.1f}%")
        
        # 메모리 부족 확인
        if metrics['memory_available_gb'] < 1.0:
            issues.append(f"사용 가능한 메모리 부족: {metrics['memory_available_gb']:.1f}GB")
        
        # 디스크 공간 부족 확인
        if metrics['disk_free_gb'] < 5.0:
            issues.append(f"사용 가능한 디스크 공간 부족: {metrics['disk_free_gb']:.1f}GB")
        
        # 상태 판정
        if issues:
            status = 'critical'
        elif warnings:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }
    
    def start_monitoring(self, interval: int = 60):
        """시스템 모니터링 시작"""
        def monitoring_loop():
            self.monitoring_active = True
            QuintLogger.info(f"시스템 모니터링 시작 (간격: {interval}초)")
            
            while self.monitoring_active:
                try:
                    health = self.check_system_health()
                    self.metrics_history.append(health)
                    
                    # 문제 발생시 알림
                    if health['status'] == 'critical':
                        asyncio.create_task(self.send_system_alert(health))
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    QuintLogger.error(f"시스템 모니터링 오류: {e}")
                    time.sleep(interval)
        
        # 백그라운드 스레드로 실행
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """시스템 모니터링 중지"""
        self.monitoring_active = False
        QuintLogger.info("시스템 모니터링 중지")
    
    async def send_system_alert(self, health: Dict):
        """시스템 경고 알림 전송"""
        try:
            issues = health.get('issues', [])
            if not issues:
                return
            
            title = "🚨 시스템 경고"
            message = "다음 시스템 문제가 감지되었습니다:\n\n"
            
            for issue in issues:
                message += f"• {issue}\n"
            
            message += f"\n상태: {health['status'].upper()}"
            
            await notification.send_alert('error', title, message, 'high')
            
        except Exception as e:
            QuintLogger.error(f"시스템 경고 알림 전송 실패: {e}")
    
    def get_monitoring_summary(self, hours: int = 24) -> Dict[str, Any]:
        """모니터링 요약 조회"""
        if not self.metrics_history:
            return {'message': '모니터링 데이터 없음'}
        
        # 지정된 시간 내의 데이터만 필터링
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.get('metrics', {}).get('timestamp', datetime.min) > cutoff_time
        ]
        
        if not recent_metrics:
            return {'message': f'최근 {hours}시간 데이터 없음'}
        
        # 통계 계산
        cpu_values = [m['metrics']['cpu_percent'] for m in recent_metrics if 'metrics' in m]
        memory_values = [m['metrics']['memory_percent'] for m in recent_metrics if 'metrics' in m]
        
        status_counts = defaultdict(int)
        for m in recent_metrics:
            status_counts[m.get('status', 'unknown')] += 1
        
        return {
            'period_hours': hours,
            'total_checks': len(recent_metrics),
            'avg_cpu_percent': np.mean(cpu_values) if cpu_values else 0,
            'max_cpu_percent': np.max(cpu_values) if cpu_values else 0,
            'avg_memory_percent': np.mean(memory_values) if memory_values else 0,
            'max_memory_percent': np.max(memory_values) if memory_values else 0,
            'status_distribution': dict(status_counts),
            'uptime_percentage': (status_counts['healthy'] / len(recent_metrics)) * 100 if recent_metrics else 0
        }

# 전역 시스템 모니터
system_monitor = QuintSystemMonitor()

# ============================================================================
# 🎯 데코레이터 및 헬퍼 함수들
# ============================================================================

def measure_time(func):
    """실행 시간 측정 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            QuintLogger.log_performance(func.__name__, execution_time)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            QuintLogger.log_exception(e, f"{func.__name__} 실행 중 오류 (실행시간: {execution_time:.4f}초)")
            raise
    return wrapper

def async_measure_time(func):
    """비동기 실행 시간 측정 데코레이터"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            QuintLogger.log_performance(func.__name__, execution_time)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            QuintLogger.log_exception(e, f"{func.__name__} 실행 중 오류 (실행시간: {execution_time:.4f}초)")
            raise
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """실패시 재시도 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        QuintLogger.error(f"{func.__name__} 최종 실패 (시도: {attempt + 1}회): {e}")
                        raise
                    
                    QuintLogger.warning(f"{func.__name__} 실패 (시도: {attempt + 1}회), {current_delay:.1f}초 후 재시도: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

def cache_result(ttl_seconds: int = 300):
    """결과 캐싱 데코레이터"""
    cache = {}
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            current_time = time.time()
            
            # 캐시 확인
            if cache_key in cache:
                cached_result, cached_time = cache[cache_key]
                if current_time - cached_time < ttl_seconds:
                    QuintLogger.debug(f"{func.__name__} 캐시 적중")
                    return cached_result
            
            # 함수 실행 및 캐시 저장
            result = func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            
            # 오래된 캐시 정리
            expired_keys = [
                key for key, (_, cached_time) in cache.items()
                if current_time - cached_time >= ttl_seconds
            ]
            for key in expired_keys:
                del cache[key]
            
            return result
        return wrapper
    return decorator

def validate_inputs(**validators):
    """입력값 검증 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 매개변수 이름 가져오기
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 검증 실행
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"{func.__name__}의 매개변수 '{param_name}' 검증 실패: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ============================================================================
# 🔧 유틸리티 함수들
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """안전한 나눗셈"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def safe_percentage(value: float, total: float, default: float = 0.0) -> float:
    """안전한 백분율 계산"""
    return safe_divide(value * 100, total, default)

def format_currency(amount: float, currency: str = 'KRW') -> str:
    """통화 포맷팅"""
    if currency == 'KRW':
        if amount >= 100_000_000:  # 1억 이상
            return f"{amount/100_000_000:.1f}억원"
        elif amount >= 10_000:  # 1만 이상
            return f"{amount/10_000:.0f}만원"
        else:
            return f"{amount:,.0f}원"
    elif currency == 'USD':
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float, decimal_places: int = 1) -> str:
    """백분율 포맷팅"""
    return f"{value:.{decimal_places}f}%"

def format_number(value: float, decimal_places: int = 2) -> str:
    """숫자 포맷팅"""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.{decimal_places}f}"

def calculate_compound_growth(principal: float, rate: float, periods: int) -> float:
    """복리 성장 계산"""
    return principal * (1 + rate) ** periods

def calculate_annualized_return(start_value: float, end_value: float, periods: int) -> float:
    """연환산 수익률 계산"""
    if start_value <= 0 or periods <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / periods) - 1

def normalize_symbol(symbol: str, market: str = 'us') -> str:
    """심볼 정규화"""
    symbol = symbol.upper().strip()
    
    if market == 'us':
        # 미국 주식 심볼 정규화
        return symbol
    elif market == 'crypto':
        # 암호화폐 심볼 정규화
        if not symbol.startswith('KRW-'):
            symbol = f"KRW-{symbol}"
        return symbol
    elif market == 'japan':
        # 일본 주식 심볼 정규화
        if not symbol.endswith('.T'):
            symbol = f"{symbol}.T"
        return symbol
    elif market == 'india':
        # 인도 주식 심볼 정규화
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            symbol = f"{symbol}.NS"
        return symbol
    
    return symbol

def parse_timeframe(timeframe: str) -> timedelta:
    """시간 프레임 파싱"""
    timeframe = timeframe.lower().strip()
    
    if timeframe.endswith('d'):
        days = int(timeframe[:-1])
        return timedelta(days=days)
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        return timedelta(hours=hours)
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        return timedelta(minutes=minutes)
    elif timeframe.endswith('w'):
        weeks = int(timeframe[:-1])
        return timedelta(weeks=weeks)
    else:
        raise ValueError(f"지원되지 않는 시간 프레임: {timeframe}")

def get_market_timezone(market: str) -> str:
    """시장별 타임존 반환"""
    timezones = {
        'us': 'America/New_York',
        'crypto': 'UTC',
        'japan': 'Asia/Tokyo',
        'india': 'Asia/Kolkata',
        'korea': 'Asia/Seoul'
    }
    return timezones.get(market.lower(), 'UTC')

def is_market_open(market: str) -> bool:
    """시장 개장 여부 확인"""
    import pytz
    
    timezone_str = get_market_timezone(market)
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)
    
    # 암호화폐는 24시간
    if market.lower() == 'crypto':
        return True
    
    # 주말 체크
    if now.weekday() >= 5:  # 토요일(5), 일요일(6)
        return False
    
    # 시장별 개장 시간
    market_hours = {
        'us': (9, 30, 16, 0),      # 9:30 AM - 4:00 PM EST
        'japan': (9, 0, 15, 0),    # 9:00 AM - 3:00 PM JST
        'india': (9, 15, 15, 30),  # 9:15 AM - 3:30 PM IST
        'korea': (9, 0, 15, 30)    # 9:00 AM - 3:30 PM KST
    }
    
    if market.lower() not in market_hours:
        return False
    
    open_h, open_m, close_h, close_m = market_hours[market.lower()]
    open_time = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    close_time = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    
    return open_time <= now <= close_time

def get_next_market_open(market: str) -> datetime:
    """다음 시장 개장 시간 반환"""
    import pytz
    
    timezone_str = get_market_timezone(market)
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)
    
    # 암호화폐는 항상 열려있음
    if market.lower() == 'crypto':
        return now
    
    market_hours = {
        'us': (9, 30),
        'japan': (9, 0),
        'india': (9, 15),
        'korea': (9, 0)
    }
    
    if market.lower() not in market_hours:
        return now
    
    open_h, open_m = market_hours[market.lower()]
    
    # 오늘 개장 시간
    today_open = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    
    # 오늘 개장하지 않았고 평일이면 오늘
    if now < today_open and now.weekday() < 5:
        return today_open
    
    # 다음 평일 개장 시간
    days_ahead = 1
    while True:
        next_day = now + timedelta(days=days_ahead)
        if next_day.weekday() < 5:  # 평일
            return next_day.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        days_ahead += 1

def create_database_connection(db_path: str = 'quint_data.db') -> sqlite3.Connection:
    """데이터베이스 연결 생성"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
        
        # 기본 테이블 생성
        cursor = conn.cursor()
        
        # 거래 기록 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                pnl REAL,
                strategy TEXT,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 포트폴리오 스냅샷 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                positions TEXT NOT NULL,  -- JSON
                daily_pnl REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 시스템 메트릭 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                network_bytes_sent INTEGER,
                network_bytes_recv INTEGER,
                status TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        return conn
        
    except Exception as e:
        QuintLogger.error(f"데이터베이스 연결 실패: {e}")
        return None

# ============================================================================
# 📊 데이터베이스 헬퍼
# ============================================================================
class QuintDatabase:
    """퀸트프로젝트 데이터베이스 헬퍼"""
    
    def __init__(self, db_path: str = 'quint_data.db'):
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """데이터베이스 초기화"""
        self.conn = create_database_connection(self.db_path)
        if self.conn:
            QuintLogger.info(f"데이터베이스 초기화 완료: {self.db_path}")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """쿼리 실행"""
        try:
            if not self.conn:
                self._initialize_db()
            
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as e:
            QuintLogger.error(f"쿼리 실행 실패: {e}")
            return []
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """INSERT 쿼리 실행"""
        try:
            if not self.conn:
                self._initialize_db()
            
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            QuintLogger.error(f"INSERT 실행 실패: {e}")
            return -1
    
    def save_trade(self, trade_data: Dict) -> int:
        """거래 기록 저장"""
        query = '''
            INSERT INTO trades (timestamp, market, symbol, action, quantity, 
                              price, amount, pnl, strategy, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            trade_data.get('timestamp', datetime.now().isoformat()),
            trade_data.get('market'),
            trade_data.get('symbol'),
            trade_data.get('action'),
            trade_data.get('quantity'),
            trade_data.get('price'),
            trade_data.get('amount'),
            trade_data.get('pnl'),
            trade_data.get('strategy'),
            trade_data.get('confidence')
        )
        
        return self.execute_insert(query, params)
    
    def save_portfolio_snapshot(self, portfolio_data: Dict) -> int:
        """포트폴리오 스냅샷 저장"""
        query = '''
            INSERT INTO portfolio_snapshots (timestamp, total_value, cash_balance, 
                                           positions, daily_pnl)
            VALUES (?, ?, ?, ?, ?)
        '''
        
        params = (
            portfolio_data.get('timestamp', datetime.now().isoformat()),
            portfolio_data.get('total_value'),
            portfolio_data.get('cash_balance'),
            json.dumps(portfolio_data.get('positions', {})),
            portfolio_data.get('daily_pnl')
        )
        
        return self.execute_insert(query, params)
    
    def save_system_metrics(self, metrics: Dict) -> int:
        """시스템 메트릭 저장"""
        query = '''
            INSERT INTO system_metrics (timestamp, cpu_percent, memory_percent,
                                      disk_percent, network_bytes_sent, 
                                      network_bytes_recv, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            metrics.get('timestamp', datetime.now()).isoformat(),
            metrics.get('cpu_percent'),
            metrics.get('memory_percent'),
            metrics.get('disk_percent'),
            metrics.get('network_bytes_sent'),
            metrics.get('network_bytes_recv'),
            metrics.get('status', 'unknown')
        )
        
        return self.execute_insert(query, params)
    
    def get_trades(self, market: str = None, symbol: str = None, 
                   start_date: str = None, end_date: str = None) -> List[Dict]:
        """거래 기록 조회"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if market:
            query += " AND market = ?"
            params.append(market)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        rows = self.execute_query(query, tuple(params))
        return [dict(row) for row in rows]
    
    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """포트폴리오 이력 조회"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = '''
            SELECT * FROM portfolio_snapshots 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC
        '''
        
        rows = self.execute_query(query, (cutoff_date,))
        result = []
        
        for row in rows:
            row_dict = dict(row)
            # JSON 파싱
            try:
                row_dict['positions'] = json.loads(row_dict['positions'])
            except:
                row_dict['positions'] = {}
            result.append(row_dict)
        
        return result
    
    def cleanup_old_data(self, days: int = 90):
        """오래된 데이터 정리"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        tables = ['trades', 'portfolio_snapshots', 'system_metrics']
        deleted_total = 0
        
        for table in tables:
            try:
                cursor = self.conn.cursor()
                cursor.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff_date,))
                deleted_count = cursor.rowcount
                deleted_total += deleted_count
                QuintLogger.info(f"{table} 테이블에서 {deleted_count}개 레코드 삭제")
            except Exception as e:
                QuintLogger.error(f"{table} 정리 실패: {e}")
        
        if deleted_total > 0:
            self.conn.commit()
            # VACUUM으로 데이터베이스 최적화
            self.conn.execute("VACUUM")
            QuintLogger.info(f"총 {deleted_total}개 레코드 정리 완료")
        
        return deleted_total
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            self.conn = None

# 전역 데이터베이스 헬퍼
database = QuintDatabase()

# ============================================================================
# 🚀 초기화 함수
# ============================================================================
def initialize_quint_utils():
    """퀸트프로젝트 유틸리티 시스템 초기화"""
    QuintLogger.info("🚀 퀸트프로젝트 유틸리티 시스템 초기화 시작")
    
    try:
        # 환경변수 로드
        if Path('.env').exists():
            load_dotenv('.env')
            QuintLogger.info("환경변수 로드 완료")
        
        # 설정 검증
        config_errors = config.validate()
        if config_errors:
            QuintLogger.warning(f"설정 검증 경고: {len(config_errors)}개")
            for error in config_errors:
                QuintLogger.warning(f"  - {error}")
        else:
            QuintLogger.info("설정 검증 완료")
        
        # 필수 디렉토리 생성
        essential_dirs = ['logs', 'backups', 'data', 'reports']
        for dir_name in essential_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        QuintLogger.info("필수 디렉토리 생성 완료")
        
        # 보안 시스템 초기화
        security._initialize_security()
        QuintLogger.info("보안 시스템 초기화 완료")
        
        # 알림 시스템 초기화
        notification._initialize_services()
        QuintLogger.info("알림 시스템 초기화 완료")
        
        # 데이터베이스 초기화
        database._initialize_db()
        QuintLogger.info("데이터베이스 초기화 완료")
        
        # 시스템 모니터링 시작 (선택적)
        if config.get('system.monitoring.enabled', False):
            system_monitor.start_monitoring()
            QuintLogger.info("시스템 모니터링 시작")
        
        # 자동 백업 스케줄링 (선택적)
        if config.get('system.backup.auto_enabled', False):
            interval = config.get('system.backup.interval_hours', 24)
            backup.schedule_auto_backup(interval)
            QuintLogger.info(f"자동 백업 스케줄링 시작 (간격: {interval}시간)")
        
        QuintLogger.info("✅ 퀸트프로젝트 유틸리티 시스템 초기화 완료")
        
        return True
        
    except Exception as e:
        QuintLogger.error(f"❌ 유틸리티 시스템 초기화 실패: {e}")
        return False

def cleanup_quint_utils():
    """퀸트프로젝트 유틸리티 시스템 정리"""
    QuintLogger.info("🧹 퀸트프로젝트 유틸리티 시스템 정리 시작")
    
    try:
        # 시스템 모니터링 중지
        system_monitor.stop_monitoring()
        
        # 네트워크 세션 종료
        if network.session and not network.session.closed:
            asyncio.create_task(network.close_session())
        
        # 데이터베이스 연결 종료
        database.close()
        
        # 최종 백업 (선택적)
        if config.get('system.backup.final_backup', True):
            backup.create_backup('final')
        
        QuintLogger.info("✅ 퀸트프로젝트 유틸리티 시스템 정리 완료")
        
    except Exception as e:
        QuintLogger.error(f"❌ 유틸리티 시스템 정리 실패: {e}")

# ============================================================================
# 🎮 편의 함수들 (외부 호출용)
# ============================================================================
def get_system_status() -> Dict[str, Any]:
    """시스템 상태 종합 조회"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'config_valid': len(config.validate()) == 0,
        'database_connected': database.conn is not None,
        'monitoring_active': system_monitor.monitoring_active,
        'network_available': network.check_internet_connection(),
        'disk_usage': {},
        'memory_usage': {},
        'recent_errors': []
    }
    
    # 시스템 메트릭
    try:
        metrics = system_monitor.get_system_metrics()
        status['disk_usage'] = {
            'percent': metrics.get('disk_percent', 0),
            'free_gb': metrics.get('disk_free_gb', 0)
        }
        status['memory_usage'] = {
            'percent': metrics.get('memory_percent', 0),
            'available_gb': metrics.get('memory_available_gb', 0)
        }
    except:
        pass
    
    # 최근 에러 로그 (간소화)
    try:
        log_file = Path('logs/quint.log')
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                error_lines = [line.strip() for line in lines[-100:] if 'ERROR' in line]
                status['recent_errors'] = error_lines[-5:]  # 최근 5개만
    except:
        pass
    
    return status

async def send_test_notification(message: str = "퀸트프로젝트 테스트 알림") -> Dict[str, bool]:
    """테스트 알림 전송"""
    return await notification.send_alert('info', '테스트 알림', message, 'normal')

def create_performance_summary() -> Dict[str, Any]:
    """성과 요약 생성"""
    try:
        trades = database.get_trades()
        if not trades:
            return {'message': '거래 데이터 없음'}
        
        # 간단한 통계
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        
        return {
            'total_trades': total_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'formatted_pnl': format_currency(total_pnl)
        }
    except Exception as e:
        QuintLogger.error(f"성과 요약 생성 실패: {e}")
        return {'error': str(e)}

def backup_system_data() -> Optional[Path]:
    """시스템 데이터 백업"""
    return backup.create_backup('manual')

def cleanup_system_data(days: int = 90) -> Dict[str, int]:
    """시스템 데이터 정리"""
    result = {}
    
    # 데이터베이스 정리
    result['database_records'] = database.cleanup_old_data(days)
    
    # 백업 정리
    result['old_backups'] = backup.cleanup_old_backups()
    
    # 로그 파일 정리 (30일 이상)
    logs_deleted = 0
    try:
        log_dir = Path('logs')
        if log_dir.exists():
            cutoff_time = datetime.now() - timedelta(days=30)
            for log_file in log_dir.glob('*.log.*'):  # 로테이션된 로그만
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_time:
                    log_file.unlink()
                    logs_deleted += 1
    except Exception as e:
        QuintLogger.error(f"로그 파일 정리 실패: {e}")
    
    result['log_files'] = logs_deleted
    
    return result

# ============================================================================
# 🎯 시작시 자동 초기화
# ============================================================================
# 모듈 import시 자동으로 초기화 실행
if __name__ != "__main__":
    initialize_quint_utils()

# ============================================================================
# 📋 퀸트프로젝트 UTILS.PY 완료!
# ============================================================================
"""
🏆 퀸트프로젝트 UTILS.PY 완전체 특징:

🔧 혼자 보수유지 가능한 유틸리티:
   ✅ 통합 설정 관리 시스템
   ✅ 자동 백업 및 복구
   ✅ 시스템 모니터링
   ✅ 에러 핸들링 및 로깅

📊 완전한 데이터 처리:
   ✅ 기술지표 계산 라이브러리
   ✅ 데이터 정제 및 변환
   ✅ 성과 분석 도구
   ✅ 데이터베이스 관리

🛡️ 보안 및 안정성:
   ✅ 암호화 시스템
   ✅ API 키 관리
   ✅ 입력값 검증
   ✅ 예외 처리

📱 통합 알림 시스템:
   ✅ 텔레그램/디스코드/슬랙
   ✅ 이메일 알림
   ✅ 조용한 시간 관리
   ✅ 우선순위별 알림

⚡ 성능 최적화:
   ✅ 캐싱 시스템
   ✅ 비동기 처리
   ✅ 재시도 메커니즘
   ✅ 실행 시간 측정

🎯 사용법:
   - from utils import *
   - get_system_status()
   - send_test_notification()
   - backup_system_data()
   - create_performance_summary()

🚀 퀸트프로젝트 = 완전 자동화 유틸리티!
