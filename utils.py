"""
🛠️ 최고퀸트프로젝트 - 공통 유틸리티 모듈
=========================================

전체 프로젝트에서 사용하는 공통 기능들:
- 📊 데이터 처리 및 변환
- 💰 금융 계산 함수
- 📁 파일 I/O 관리
- 🔄 API 재시도 로직
- 📈 기술적 지표 계산
- 📋 포맷팅 및 검증
- 💾 캐싱 시스템
- 📊 백테스트 유틸리티

Author: 최고퀸트팀
Version: 1.0.0
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
import json
import csv
import os
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
import yaml
import pandas as pd
import numpy as np
from functools import wraps
from pathlib import Path
import requests
from dataclasses import dataclass, asdict
import traceback
import pytz

# 로거 설정
logger = logging.getLogger(__name__)

# ================================
# 🕐 시간대 관리 유틸리티
# ================================

class TimeZoneManager:
    """시간대 관리 전용 클래스"""
    
    def __init__(self):
        """시간대 초기화"""
        self.timezones = {
            'KOR': pytz.timezone('Asia/Seoul'),      # 한국 시간 (KST)
            'US': pytz.timezone('US/Eastern'),       # 미국 동부 (EST/EDT 자동)
            'JP': pytz.timezone('Asia/Tokyo'),       # 일본 시간 (JST)
            'UTC': pytz.UTC                          # 협정 시간
        }
        
        # 시장 운영 시간 (현지 시간 기준)
        self.market_hours = {
            'US': {
                'premarket_open': '04:00',   # 프리마켓 시작
                'regular_open': '09:30',     # 정규 시장 시작
                'regular_close': '16:00',    # 정규 시장 마감
                'aftermarket_close': '20:00' # 애프터마켓 마감
            },
            'JP': {
                'morning_open': '09:00',     # 오전장 시작
                'morning_close': '11:30',    # 오전장 마감
                'afternoon_open': '12:30',   # 오후장 시작
                'afternoon_close': '15:00'   # 오후장 마감
            },
            'COIN': {
                'open': '00:00',             # 24시간 거래
                'close': '23:59'
            }
        }

    def get_current_time(self, timezone: str = 'KOR') -> datetime:
        """특정 시간대의 현재 시간"""
        if timezone not in self.timezones:
            timezone = 'KOR'
        
        utc_now = datetime.now(pytz.UTC)
        local_time = utc_now.astimezone(self.timezones[timezone])
        return local_time

    def convert_time(self, dt: datetime, from_tz: str, to_tz: str) -> datetime:
        """시간대 변환"""
        if from_tz not in self.timezones or to_tz not in self.timezones:
            return dt
        
        # 입력 시간이 naive하면 from_tz를 적용
        if dt.tzinfo is None:
            dt = self.timezones[from_tz].localize(dt)
        
        # 목표 시간대로 변환
        converted = dt.astimezone(self.timezones[to_tz])
        return converted

    def get_all_market_times(self) -> Dict[str, str]:
        """전체 시장 현재 시간"""
        current_times = {}
        
        for market in ['KOR', 'US', 'JP']:
            current = self.get_current_time(market)
            current_times[market] = {
                'datetime': current.strftime('%Y-%m-%d %H:%M:%S'),
                'time_only': current.strftime('%H:%M:%S'),
                'date': current.strftime('%Y-%m-%d'),
                'weekday': current.strftime('%A'),
                'timezone_name': str(current.tzinfo)
            }
        
        return current_times

    def is_weekend(self, timezone: str = 'KOR') -> bool:
        """주말 여부 확인"""
        current = self.get_current_time(timezone)
        return current.weekday() >= 5  # 5=토요일, 6=일요일

    def is_market_open_detailed(self, market: str) -> Dict[str, Any]:
        """상세 시장 개장 정보"""
        market = market.upper()
        
        if market == 'COIN':
            return {
                'is_open': True,
                'session_type': '24시간',
                'status': 'open',
                'next_event': None,
                'current_time': self.get_current_time('UTC').strftime('%H:%M:%S UTC')
            }
        
        # 시간대 매핑
        tz_map = {'US': 'US', 'JP': 'JP', 'KOR': 'KOR'}
        tz = tz_map.get(market, 'KOR')
        
        current = self.get_current_time(tz)
        current_time_str = current.strftime('%H:%M')
        
        # 주말 체크
        if self.is_weekend(tz):
            next_monday = current + timedelta(days=(7 - current.weekday()))
            return {
                'is_open': False,
                'session_type': '주말 휴장',
                'status': 'weekend',
                'next_event': f"월요일 개장까지 {self._get_time_diff(current, next_monday)}",
                'current_time': current.strftime('%H:%M:%S')
            }
        
        # 미국 시장
        if market == 'US':
            hours = self.market_hours['US']
            
            if hours['premarket_open'] <= current_time_str < hours['regular_open']:
                return {
                    'is_open': True,
                    'session_type': '프리마켓',
                    'status': 'premarket',
                    'next_event': f"정규장 시작까지 {self._get_time_until(current, hours['regular_open'])}",
                    'current_time': current.strftime('%H:%M:%S EST/EDT')
                }
            elif hours['regular_open'] <= current_time_str < hours['regular_close']:
                return {
                    'is_open': True,
                    'session_type': '정규장',
                    'status': 'regular',
                    'next_event': f"정규장 마감까지 {self._get_time_until(current, hours['regular_close'])}",
                    'current_time': current.strftime('%H:%M:%S EST/EDT')
                }
            elif hours['regular_close'] <= current_time_str < hours['aftermarket_close']:
                return {
                    'is_open': True,
                    'session_type': '애프터마켓',
                    'status': 'aftermarket',
                    'next_event': f"장 마감까지 {self._get_time_until(current, hours['aftermarket_close'])}",
                    'current_time': current.strftime('%H:%M:%S EST/EDT')
                }
            else:
                return {
                    'is_open': False,
                    'session_type': '휴장',
                    'status': 'closed',
                    'next_event': f"프리마켓 시작까지 {self._get_time_until_next_day(current, hours['premarket_open'])}",
                    'current_time': current.strftime('%H:%M:%S EST/EDT')
                }
        
        # 일본 시장
        elif market == 'JP':
            hours = self.market_hours['JP']
            
            if hours['morning_open'] <= current_time_str < hours['morning_close']:
                return {
                    'is_open': True,
                    'session_type': '오전장',
                    'status': 'morning',
                    'next_event': f"오전장 마감까지 {self._get_time_until(current, hours['morning_close'])}",
                    'current_time': current.strftime('%H:%M:%S JST')
                }
            elif hours['morning_close'] <= current_time_str < hours['afternoon_open']:
                return {
                    'is_open': False,
                    'session_type': '점심시간',
                    'status': 'lunch',
                    'next_event': f"오후장 시작까지 {self._get_time_until(current, hours['afternoon_open'])}",
                    'current_time': current.strftime('%H:%M:%S JST')
                }
            elif hours['afternoon_open'] <= current_time_str < hours['afternoon_close']:
                return {
                    'is_open': True,
                    'session_type': '오후장',
                    'status': 'afternoon',
                    'next_event': f"오후장 마감까지 {self._get_time_until(current, hours['afternoon_close'])}",
                    'current_time': current.strftime('%H:%M:%S JST')
                }
            else:
                return {
                    'is_open': False,
                    'session_type': '휴장',
                    'status': 'closed',
                    'next_event': f"오전장 시작까지 {self._get_time_until_next_day(current, hours['morning_open'])}",
                    'current_time': current.strftime('%H:%M:%S JST')
                }
        
        # 한국 시장 (참고용)
        else:
            return {
                'is_open': False,
                'session_type': '한국 시장 정보 없음',
                'status': 'unknown',
                'next_event': None,
                'current_time': current.strftime('%H:%M:%S KST')
            }

    def _get_time_until(self, current: datetime, target_time_str: str) -> str:
        """현재 시간부터 목표 시간까지 남은 시간"""
        target_hour, target_min = map(int, target_time_str.split(':'))
        target = current.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
        
        if target <= current:
            target += timedelta(days=1)
        
        diff = target - current
        return self._format_timedelta(diff)

    def _get_time_until_next_day(self, current: datetime, target_time_str: str) -> str:
        """다음날 목표 시간까지 남은 시간"""
        target_hour, target_min = map(int, target_time_str.split(':'))
        next_day = current + timedelta(days=1)
        target = next_day.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
        
        diff = target - current
        return self._format_timedelta(diff)

    def _get_time_diff(self, from_time: datetime, to_time: datetime) -> str:
        """두 시간 사이의 차이"""
        diff = to_time - from_time
        return self._format_timedelta(diff)

    def _format_timedelta(self, td: timedelta) -> str:
        """timedelta 포맷팅"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}시간 {minutes}분"
        else:
            return f"{minutes}분"

    def get_market_schedule_today(self) -> Dict[str, List[Dict]]:
        """오늘의 시장 스케줄"""
        schedule = {}
        
        for market in ['US', 'JP']:
            tz = 'US' if market == 'US' else 'JP'
            current = self.get_current_time(tz)
            
            if self.is_weekend(tz):
                schedule[market] = [{'event': '주말 휴장', 'time': '전일'}]
                continue
            
            events = []
            hours = self.market_hours[market]
            
            if market == 'US':
                events = [
                    {'event': '프리마켓 시작', 'time': hours['premarket_open']},
                    {'event': '정규장 시작', 'time': hours['regular_open']},
                    {'event': '정규장 마감', 'time': hours['regular_close']},
                    {'event': '애프터마켓 마감', 'time': hours['aftermarket_close']}
                ]
            elif market == 'JP':
                events = [
                    {'event': '오전장 시작', 'time': hours['morning_open']},
                    {'event': '오전장 마감', 'time': hours['morning_close']},
                    {'event': '오후장 시작', 'time': hours['afternoon_open']},
                    {'event': '오후장 마감', 'time': hours['afternoon_close']}
                ]
            
            schedule[market] = events
        
        return schedule

    def seoul_to_us_time(self, seoul_dt: datetime) -> datetime:
        """서울 시간 → 미국 시간"""
        return self.convert_time(seoul_dt, 'KOR', 'US')

    def seoul_to_japan_time(self, seoul_dt: datetime) -> datetime:
        """서울 시간 → 일본 시간"""
        return self.convert_time(seoul_dt, 'KOR', 'JP')

    def us_to_seoul_time(self, us_dt: datetime) -> datetime:
        """미국 시간 → 서울 시간"""
        return self.convert_time(us_dt, 'US', 'KOR')

    def japan_to_seoul_time(self, jp_dt: datetime) -> datetime:
        """일본 시간 → 서울 시간"""
        return self.convert_time(jp_dt, 'JP', 'KOR')

# ================================
# 📊 데이터 처리 유틸리티
# ================================

class DataProcessor:
    """데이터 처리 전용 클래스"""
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """심볼 정규화"""
        if not symbol:
            return ""
        
        symbol = symbol.upper().strip()
        
        # 암호화폐 처리
        if '-' in symbol and not symbol.endswith('.T'):
            # BTC-KRW, ETH-USDT 등
            return symbol
        
        # 일본 주식 처리
        if symbol.endswith('.T'):
            return symbol
            
        # 미국 주식 처리 (기본)
        return symbol

    @staticmethod
    def detect_market(symbol: str) -> str:
        """심볼로 시장 판별"""
        symbol = DataProcessor.normalize_symbol(symbol)
        
        if symbol.endswith('.T'):
            return 'JP'
        elif '-' in symbol or symbol.endswith('USDT') or symbol.endswith('KRW'):
            return 'COIN'
        else:
            return 'US'

    @staticmethod
    def clean_price_data(data: pd.DataFrame) -> pd.DataFrame:
        """가격 데이터 정리"""
        if data.empty:
            return data
            
        # 결측값 처리
        data = data.dropna()
        
        # 이상값 제거 (3 표준편차 이상)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            mean = data[col].mean()
            std = data[col].std()
            data = data[abs(data[col] - mean) <= 3 * std]
        
        # 인덱스 정렬
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
            
        return data

    @staticmethod
    def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """수익률 계산"""
        return prices.pct_change(periods=periods).fillna(0)

    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """변동성 계산 (Rolling Standard Deviation)"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # 연환산

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        excess_returns = returns.mean() * 252 - risk_free_rate  # 연환산
        volatility = returns.std() * np.sqrt(252)
        
        return excess_returns / volatility if volatility != 0 else 0

# ================================
# 💰 금융 계산 함수
# ================================

class FinanceUtils:
    """금융 계산 전용 클래스"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Dict[str, pd.Series]:
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def calculate_moving_average(prices: pd.Series, period: int) -> pd.Series:
        """이동평균 계산"""
        return prices.rolling(window=period).mean()

    @staticmethod
    def calculate_position_size(capital: float, risk_per_trade: float, 
                              entry_price: float, stop_loss_price: float) -> float:
        """포지션 크기 계산 (고정 리스크 방식)"""
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
            
        risk_amount = capital * risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
            
        return risk_amount / price_risk

    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """켈리 공식으로 최적 베팅 비율 계산"""
        if avg_loss <= 0 or win_rate <= 0:
            return 0
            
        loss_rate = 1 - win_rate
        win_loss_ratio = avg_win / avg_loss
        
        kelly_pct = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        return max(0, min(kelly_pct, 0.25))  # 최대 25%로 제한

# ================================
# 📁 파일 I/O 관리
# ================================

class FileManager:
    """파일 관리 전용 클래스"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self._ensure_directories()

    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = ['data', 'logs', 'configs', 'data/cache', 'data/backups']
        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)

    def save_json(self, data: Any, filename: str, directory: str = "data") -> bool:
        """JSON 파일 저장"""
        try:
            filepath = self.base_path / directory / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"JSON 파일 저장 완료: {filepath}")
            return True
        except Exception as e:
            logger.error(f"JSON 파일 저장 실패: {e}")
            return False

    def load_json(self, filename: str, directory: str = "data") -> Optional[Any]:
        """JSON 파일 로드"""
        try:
            filepath = self.base_path / directory / filename
            if not filepath.exists():
                logger.warning(f"파일이 존재하지 않음: {filepath}")
                return None
                
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"JSON 파일 로드 완료: {filepath}")
            return data
        except Exception as e:
            logger.error(f"JSON 파일 로드 실패: {e}")
            return None

    def save_csv(self, df: pd.DataFrame, filename: str, directory: str = "data") -> bool:
        """CSV 파일 저장"""
        try:
            filepath = self.base_path / directory / filename
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"CSV 파일 저장 완료: {filepath}")
            return True
        except Exception as e:
            logger.error(f"CSV 파일 저장 실패: {e}")
            return False

    def load_csv(self, filename: str, directory: str = "data") -> Optional[pd.DataFrame]:
        """CSV 파일 로드"""
        try:
            filepath = self.base_path / directory / filename
            if not filepath.exists():
                logger.warning(f"파일이 존재하지 않음: {filepath}")
                return None
                
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"CSV 파일 로드 완료: {filepath}")
            return df
        except Exception as e:
            logger.error(f"CSV 파일 로드 실패: {e}")
            return None

    def backup_file(self, filename: str, directory: str = "data") -> bool:
        """파일 백업"""
        try:
            filepath = self.base_path / directory / filename
            if not filepath.exists():
                return False
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
            backup_path = self.base_path / "data" / "backups" / backup_name
            
            import shutil
            shutil.copy2(filepath, backup_path)
            logger.info(f"파일 백업 완료: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"파일 백업 실패: {e}")
            return False

    def cleanup_old_files(self, directory: str = "logs", days: int = 30):
        """오래된 파일 정리"""
        try:
            target_dir = self.base_path / directory
            cutoff_date = datetime.now() - timedelta(days=days)
            
            deleted_count = 0
            for filepath in target_dir.glob("*"):
                if filepath.is_file():
                    file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                    if file_time < cutoff_date:
                        filepath.unlink()
                        deleted_count += 1
            
            logger.info(f"{directory} 폴더에서 {deleted_count}개 파일 정리 완료")
        except Exception as e:
            logger.error(f"파일 정리 실패: {e}")

# ================================
# 🔄 API 재시도 로직
# ================================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """API 호출 재시도 데코레이터"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"함수 {func.__name__} 최대 재시도 횟수 초과: {e}")
                        raise e
                    
                    logger.warning(f"함수 {func.__name__} 재시도 {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"함수 {func.__name__} 최대 재시도 횟수 초과: {e}")
                        raise e
                    
                    logger.warning(f"함수 {func.__name__} 재시도 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class RateLimiter:
    """API 호출 속도 제한"""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0

    async def wait(self):
        """속도 제한 대기"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            wait_time = self.min_interval - time_since_last_call
            await asyncio.sleep(wait_time)
        
        self.last_call_time = time.time()

# ================================
# 📋 포맷팅 및 검증
# ================================

class Formatter:
    """포맷팅 전용 클래스"""
    
    @staticmethod
    def format_price(price: float, decimals: int = 2) -> str:
        """가격 포맷팅"""
        if price >= 1000000:
            return f"${price/1000000:.1f}M"
        elif price >= 1000:
            return f"${price:,.{decimals}f}"
        elif price >= 1:
            return f"${price:.{decimals}f}"
        else:
            return f"${price:.4f}"

    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """퍼센트 포맷팅"""
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.{decimals}f}%"

    @staticmethod
    def format_large_number(value: float) -> str:
        """큰 숫자 포맷팅"""
        if abs(value) >= 1e12:
            return f"{value/1e12:.1f}T"
        elif abs(value) >= 1e9:
            return f"{value/1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.0f}"

    @staticmethod
    def format_duration(seconds: float) -> str:
        """시간 지속 포맷팅"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            return f"{seconds/60:.1f}분"
        else:
            return f"{seconds/3600:.1f}시간"

class Validator:
    """검증 전용 클래스"""
    
    @staticmethod
    def is_valid_symbol(symbol: str) -> bool:
        """심볼 유효성 검사"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        symbol = symbol.strip()
        if len(symbol) < 1 or len(symbol) > 20:
            return False
            
        # 기본적인 패턴 체크
        import re
        patterns = [
            r'^[A-Z]{1,10},           # 미국 주식 (AAPL, MSFT 등)
            r'^[0-9]{4}\.T,           # 일본 주식 (7203.T 등)
            r'^[A-Z]{2,10}-[A-Z]{3,10} # 암호화폐 (BTC-KRW 등)
        ]
        
        return any(re.match(pattern, symbol) for pattern in patterns)

    @staticmethod
    def is_valid_price(price: float) -> bool:
        """가격 유효성 검사"""
        return isinstance(price, (int, float)) and price > 0 and not np.isnan(price)

    @staticmethod
    def is_valid_confidence(confidence: float) -> bool:
        """신뢰도 유효성 검사"""
        return isinstance(confidence, (int, float)) and 0 <= confidence <= 1

# ================================
# 💾 캐싱 시스템
# ================================

class SimpleCache:
    """간단한 메모리 캐시"""
    
    def __init__(self, ttl_seconds: int = 300):  # 5분 기본 TTL
        self.cache = {}
        self.ttl = ttl_seconds

    def _is_expired(self, timestamp: float) -> bool:
        """만료 확인"""
        return time.time() - timestamp > self.ttl

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        if key not in self.cache:
            return None
            
        data, timestamp = self.cache[key]
        if self._is_expired(timestamp):
            del self.cache[key]
            return None
            
        return data

    def set(self, key: str, value: Any):
        """캐시에 값 저장"""
        self.cache[key] = (value, time.time())

    def clear(self):
        """캐시 전체 삭제"""
        self.cache.clear()

    def cleanup(self):
        """만료된 항목 정리"""
        expired_keys = []
        current_time = time.time()
        
        for key, (data, timestamp) in self.cache.items():
            if current_time - timestamp > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]

# ================================
# 📊 백테스트 유틸리티
# ================================

class BacktestUtils:
    """백테스트 관련 유틸리티"""
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """최대 손실폭 계산"""
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """칼마 비율 계산 (연수익률 / 최대손실폭)"""
        annual_return = returns.mean() * 252
        equity_curve = (1 + returns).cumprod()
        max_dd = BacktestUtils.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0
        return annual_return / abs(max_dd)

    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """승률 계산"""
        winning_trades = (returns > 0).sum()
        total_trades = len(returns[returns != 0])
        
        return winning_trades / total_trades if total_trades > 0 else 0

    @staticmethod
    def generate_performance_report(returns: pd.Series) -> Dict[str, float]:
        """성과 리포트 생성"""
        equity_curve = (1 + returns).cumprod()
        
        return {
            'total_return': (equity_curve.iloc[-1] - 1) * 100,
            'annual_return': returns.mean() * 252 * 100,
            'volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': DataProcessor.calculate_sharpe_ratio(returns),
            'calmar_ratio': BacktestUtils.calculate_calmar_ratio(returns),
            'max_drawdown': BacktestUtils.calculate_max_drawdown(equity_curve) * 100,
            'win_rate': BacktestUtils.calculate_win_rate(returns) * 100,
            'total_trades': len(returns[returns != 0])
        }

# ================================
# 누락된 유틸리티 클래스들 (import 오류 해결용)
# ================================

class NewsUtils:
    """뉴스 분석 유틸리티"""
    
    @staticmethod
    def get_news(symbol: str = None, limit: int = 10) -> List[Dict]:
        """뉴스 데이터 조회"""
        try:
            # 실제 뉴스 API 연동 로직이 들어갈 자리
            logger.info(f"뉴스 조회: {symbol}, 제한: {limit}")
            return []
        except Exception as e:
            logger.error(f"뉴스 조회 실패: {e}")
            return []
    
    @staticmethod
    def analyze_sentiment(text: str) -> float:
        """텍스트 센티먼트 분석"""
        try:
            # 간단한 키워드 기반 분석 (실제로는 AI 모델 사용)
            positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit']
            negative_words = ['bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'decline']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count + neg_count == 0:
                return 0.5  # 중립
            
            return pos_count / (pos_count + neg_count)
            
        except Exception as e:
            logger.error(f"센티먼트 분석 실패: {e}")
            return 0.5

    @staticmethod
    def get_market_sentiment(market: str = 'US') -> Dict[str, Any]:
        """시장 전체 센티먼트"""
        try:
            return {
                'market': market,
                'sentiment_score': 0.5,
                'confidence': 0.7,
                'summary': f"{market} 시장 센티먼트 중립",
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"시장 센티먼트 분석 실패: {e}")
            return {}

class ScheduleUtils:
    """스케줄링 유틸리티"""
    
    @staticmethod
    def get_schedule(date: str = None) -> List[Dict]:
        """일정 조회"""
        try:
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # 기본 시장 스케줄 반환
            schedules = []
            
            # 미국 시장 스케줄
            schedules.append({
                'market': 'US',
                'date': date,
                'events': [
                    {'time': '09:30', 'event': '정규장 시작', 'timezone': 'EST/EDT'},
                    {'time': '16:00', 'event': '정규장 마감', 'timezone': 'EST/EDT'}
                ]
            })
            
            # 일본 시장 스케줄
            schedules.append({
                'market': 'JP',
                'date': date,
                'events': [
                    {'time': '09:00', 'event': '오전장 시작', 'timezone': 'JST'},
                    {'time': '11:30', 'event': '오전장 마감', 'timezone': 'JST'},
                    {'time': '12:30', 'event': '오후장 시작', 'timezone': 'JST'},
                    {'time': '15:00', 'event': '오후장 마감', 'timezone': 'JST'}
                ]
            })
            
            return schedules
            
        except Exception as e:
            logger.error(f"스케줄 조회 실패: {e}")
            return []

    @staticmethod
    def is_trading_day(market: str = 'US', date: str = None) -> bool:
        """거래일 여부 확인"""
        try:
            if not date:
                target_date = datetime.now()
            else:
                target_date = datetime.strptime(date, '%Y-%m-%d')
            
            # 주말 체크
            weekday = target_date.weekday()
            if weekday >= 5:  # 토요일(5), 일요일(6)
                return False
            
            # 간단한 공휴일 체크 (실제로는 더 정교한 로직 필요)
            # 여기서는 기본적으로 평일은 거래일로 간주
            return True
            
        except Exception as e:
            logger.error(f"거래일 확인 실패: {e}")
            return True

    @staticmethod
    def get_next_trading_day(market: str = 'US') -> str:
        """다음 거래일 조회"""
        try:
            current_date = datetime.now()
            
            # 최대 7일까지 확인
            for i in range(1, 8):
                next_date = current_date + timedelta(days=i)
                if ScheduleUtils.is_trading_day(market, next_date.strftime('%Y-%m-%d')):
                    return next_date.strftime('%Y-%m-%d')
            
            return current_date.strftime('%Y-%m-%d')  # fallback
            
        except Exception as e:
            logger.error(f"다음 거래일 조회 실패: {e}")
            return datetime.now().strftime('%Y-%m-%d')

class BrokerUtils:
    """브로커 연동 유틸리티"""
    
    @staticmethod
    def connect(broker: str = 'default') -> bool:
        """브로커 연결"""
        try:
            logger.info(f"브로커 연결 시도: {broker}")
            # 실제 브로커 API 연결 로직이 들어갈 자리
            return True
            
        except Exception as e:
            logger.error(f"브로커 연결 실패: {e}")
            return False

    @staticmethod
    def disconnect(broker: str = 'default') -> bool:
        """브로커 연결 해제"""
        try:
            logger.info(f"브로커 연결 해제: {broker}")
            return True
            
        except Exception as e:
            logger.error(f"브로커 연결 해제 실패: {e}")
            return False

    @staticmethod
    def get_account_info(broker: str = 'default') -> Dict[str, Any]:
        """계좌 정보 조회"""
        try:
            # 더미 데이터 반환
            return {
                'broker': broker,
                'account_id': 'DEMO123456',
                'balance': 100000.0,
                'buying_power': 200000.0,
                'positions': [],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"계좌 정보 조회 실패: {e}")
            return {}

    @staticmethod
    def place_order(symbol: str, quantity: float, order_type: str = 'market', 
                   price: float = None) -> Dict[str, Any]:
        """주문 실행"""
        try:
            order_id = f"ORDER_{int(time.time())}"
            
            order_info = {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'status': 'submitted',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"주문 실행: {order_info}")
            return order_info
            
        except Exception as e:
            logger.error(f"주문 실행 실패: {e}")
            return {}

    @staticmethod
    def cancel_order(order_id: str) -> bool:
        """주문 취소"""
        try:
            logger.info(f"주문 취소: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"주문 취소 실패: {e}")
            return False

    @staticmethod
    def get_positions(broker: str = 'default') -> List[Dict]:
        """포지션 조회"""
        try:
            # 더미 포지션 데이터
            return [
                {
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'avg_price': 150.0,
                    'current_price': 155.0,
                    'unrealized_pnl': 500.0,
                    'market_value': 15500.0
                }
            ]
            
        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
            return []

# ================================
# 🔧 편의 함수들
# ================================

# 전역 객체들
file_manager = FileManager()
cache = SimpleCache()
timezone_manager = TimeZoneManager()

def get_config(config_path: str = "configs/settings.yaml") -> Dict:
    """설정 파일 로드 (캐시 적용)"""
    cached = cache.get(f"config_{config_path}")
    if cached:
        return cached
    
    try:
        # YAML 파일이 없는 경우 기본 설정 반환
        if not os.path.exists(config_path):
            logger.warning(f"설정 파일 없음: {config_path}, 기본 설정 사용")
            default_config = {
                'trading': {
                    'risk_limit': 0.02,
                    'max_positions': 10
                },
                'coin_strategy': {
                    'enabled': True,
                    'volume_spike_threshold': 2.0,
                    'symbols': {
                        'MAJOR': ['BTC-KRW', 'ETH-KRW'],
                        'ALTCOIN': ['ADA-KRW', 'DOT-KRW']
                    }
                },
                'us_strategy': {
                    'enabled': True
                },
                'jp_strategy': {
                    'enabled': True
                }
            }
            cache.set(f"config_{config_path}", default_config)
            return default_config
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        cache.set(f"config_{config_path}", config)
        return config
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        return {}

def save_trading_log(log_data: Dict, log_type: str = "trading"):
    """거래 로그 저장"""
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{log_type}_log_{timestamp}.json"
    
    # 기존 로그 로드
    existing_logs = file_manager.load_json(filename, "logs") or []
    
    # 새 로그 추가 (모든 시간대 정보 포함)
    log_data['timestamp'] = datetime.now().isoformat()
    log_data['market_times'] = timezone_manager.get_all_market_times()
    existing_logs.append(log_data)
    
    # 저장
    file_manager.save_json(existing_logs, filename, "logs")

def get_market_hours(market: str = "US") -> Dict[str, Any]:
    """시장 운영 시간 반환 (시간대 정보 포함)"""
    market = market.upper()
    
    base_info = {
        'US': {
            'timezone': 'US/Eastern (EST/EDT)',
            'premarket': '04:00 - 09:30',
            'regular': '09:30 - 16:00', 
            'aftermarket': '16:00 - 20:00',
            'currency': 'USD'
        },
        'JP': {
            'timezone': 'Asia/Tokyo (JST)',
            'morning': '09:00 - 11:30',
            'lunch_break': '11:30 - 12:30',
            'afternoon': '12:30 - 15:00',
            'currency': 'JPY'
        },
        'COIN': {
            'timezone': 'UTC (24시간)',
            'trading': '24시간 연중무휴',
            'currency': 'Various'
        },
        'KOR': {
            'timezone': 'Asia/Seoul (KST)',
            'regular': '09:00 - 15:30',
            'currency': 'KRW'
        }
    }
    
    market_info = base_info.get(market, base_info['US'])
    
    # 현재 시간 정보 추가
    current_time = timezone_manager.get_current_time('KOR' if market == 'KOR' else market)
    market_info['current_local_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
    market_info['current_status'] = timezone_manager.is_market_open_detailed(market)
    
    return market_info

def is_market_open(market: str = "US") -> bool:
    """시장 개장 여부 확인 (정확한 시간대 적용)"""
    market_status = timezone_manager.is_market_open_detailed(market)
    return market_status['is_open']

def get_all_market_status() -> Dict[str, Dict]:
    """전체 시장 상태 조회"""
    markets = ['US', 'JP', 'COIN']
    status = {}
    
    for market in markets:
        status[market] = timezone_manager.is_market_open_detailed(market)
    
    # 서울 시간도 추가
    seoul_time = timezone_manager.get_current_time('KOR')
    status['KOR'] = {
        'current_time': seoul_time.strftime('%Y-%m-%d %H:%M:%S KST'),
        'weekday': seoul_time.strftime('%A'),
        'date': seoul_time.strftime('%Y년 %m월 %d일')
    }
    
    return status

def get_time_until_market_event(market: str = "US") -> Dict[str, str]:
    """다음 시장 이벤트까지 남은 시간"""
    market_status = timezone_manager.is_market_open_detailed(market)
    
    return {
        'market': market,
        'current_status': market_status['status'],
        'next_event': market_status['next_event'],
        'is_open': market_status['is_open'],
        'session_type': market_status['session_type']
    }

def convert_market_times(time_str: str, from_market: str, to_market: str) -> str:
    """시장간 시간 변환 (문자열 입력)"""
    try:
        # 시간 문자열 파싱 (HH:MM 형식 가정)
        hour, minute = map(int, time_str.split(':'))
        
        # 현재 날짜 기준으로 datetime 생성
        from_tz = 'KOR' if from_market == 'KOR' else from_market
        base_date = timezone_manager.get_current_time(from_tz).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        
        # 시간대 변환
        to_tz = 'KOR' if to_market == 'KOR' else to_market
        converted = timezone_manager.convert_time(base_date, from_tz, to_tz)
        
        return converted.strftime('%H:%M')
        
    except Exception as e:
        logger.error(f"시간 변환 실패: {e}")
        return time_str

def get_market_times_comparison() -> Dict[str, str]:
    """현재 시간 기준 모든 시장 시간 비교"""
    times = timezone_manager.get_all_market_times()
    
    comparison = {}
    for market, time_info in times.items():
        market_name = {
            'KOR': '🇰🇷 서울',
            'US': '🇺🇸 뉴욕', 
            'JP': '🇯🇵 도쿄'
        }.get(market, market)
        
        comparison[market_name] = f"{time_info['time_only']} ({time_info['weekday']})"
    
    return comparison

def calculate_portfolio_value(positions: Dict[str, Dict]) -> float:
    """포트폴리오 총 가치 계산"""
    total_value = 0
    
    for symbol, position in positions.items():
        quantity = position.get('quantity', 0)
        current_price = position.get('current_price', 0)
        total_value += quantity * current_price
    
    return total_value

# ================================
# 🧪 테스트 함수
# ================================

def run_utils_test():
    """유틸리티 기능 테스트"""
    print("🛠️ 최고퀸트프로젝트 유틸리티 테스트")
    print("=" * 50)
    
    # 1. 시간대 처리 테스트
    print("🕐 시간대 처리 테스트:")
    current_times = timezone_manager.get_all_market_times()
    for market, time_info in current_times.items():
        market_name = {'KOR': '🇰🇷 서울', 'US': '🇺🇸 뉴욕', 'JP': '🇯🇵 도쿄'}[market]
        print(f"  {market_name}: {time_info['datetime']}")
    
    print("\n📈 시장 개장 상태:")
    market_status = get_all_market_status()
    for market in ['US', 'JP', 'COIN']:
        status = market_status[market]
        market_name = {'US': '🇺🇸 미국', 'JP': '🇯🇵 일본', 'COIN': '🪙 코인'}[market]
        open_status = "🟢 OPEN" if status['is_open'] else "🔴 CLOSED"
        print(f"  {market_name}: {open_status} - {status['session_type']}")
        if status['next_event']:
            print(f"    └─ {status['next_event']}")
    
    # 2. 데이터 처리 테스트
    print("\n📊 데이터 처리 테스트:")
    symbols = ['AAPL', '7203.T', 'BTC-KRW', 'invalid_symbol']
    for symbol in symbols:
        market = DataProcessor.detect_market(symbol)
        is_valid = Validator.is_valid_symbol(symbol)
        print(f"  {symbol}: {market} 시장, 유효성: {is_valid}")
    
    # 3. 시간 변환 테스트
    print("\n🔄 시간 변환 테스트:")
    test_time = "15:30"  # 오후 3시 30분
    
    conversions = [
        ("서울", "뉴욕", convert_market_times(test_time, 'KOR', 'US')),
        ("서울", "도쿄", convert_market_times(test_time, 'KOR', 'JP')),
        ("뉴욕", "서울", convert_market_times(test_time, 'US', 'KOR')),
        ("도쿄", "서울", convert_market_times(test_time, 'JP', 'KOR'))
    ]
    
    for from_city, to_city, converted in conversions:
        print(f"  {from_city} {test_time} → {to_city} {converted}")
    
    # 4. 포맷팅 테스트
    print("\n📋 포맷팅 테스트:")
    prices = [0.0001, 1.23, 123.45, 12345, 1234567]
    for price in prices:
        formatted = Formatter.format_price(price)
        print(f"  ${price} → {formatted}")
    
    # 5. 파일 관리 테스트
    print("\n📁 파일 관리 테스트:")
    test_data = {
        'test': 'data', 
        'timestamp': datetime.now().isoformat(),
        'market_times': timezone_manager.get_all_market_times()
    }
    success = file_manager.save_json(test_data, 'test.json')
    print(f"  JSON 저장: {'성공' if success else '실패'}")
    
    loaded_data = file_manager.load_json('test.json')
    print(f"  JSON 로드: {'성공' if loaded_data else '실패'}")
    
    # 6. 캐시 테스트
    print("\n💾 캐시 테스트:")
    cache.set('test_key', 'test_value')
    cached_value = cache.get('test_key')
    print(f"  캐시 저장/로드: {'성공' if cached_value == 'test_value' else '실패'}")
    
    # 7. 금융 계산 테스트
    print("\n💰 금융 계산 테스트:")
    sample_prices = pd.Series([100, 102, 98, 105, 103, 108, 106, 110])
    rsi = FinanceUtils.calculate_rsi(sample_prices)
    print(f"  RSI 계산: {rsi.iloc[-1]:.2f}")
    
    returns = DataProcessor.calculate_returns(sample_prices)
    sharpe = DataProcessor.calculate_sharpe_ratio(returns)
    print(f"  샤프 비율: {sharpe:.2f}")
    
    # 8. 시장 스케줄 테스트
    print("\n📅 오늘의 시장 스케줄:")
    schedule = timezone_manager.get_market_schedule_today()
    for market, events in schedule.items():
        market_name = {'US': '🇺🇸 미국', 'JP': '🇯🇵 일본'}[market]
        print(f"  {market_name}:")
        for event in events:
            print(f"    {event['time']} - {event['event']}")
    
    # 9. 새로 추가된 유틸리티 테스트
    print("\n📰 뉴스 유틸리티 테스트:")
    news_data = NewsUtils.get_news('AAPL', limit=3)
    print(f"  뉴스 조회: {len(news_data)}건")
    
    sentiment = NewsUtils.analyze_sentiment("This is great news for investors!")
    print(f"  센티먼트 분석: {sentiment:.2f}")
    
    print("\n📅 스케줄 유틸리티 테스트:")
    schedules = ScheduleUtils.get_schedule()
    print(f"  스케줄 조회: {len(schedules)}개 시장")
    
    is_trading = ScheduleUtils.is_trading_day('US')
    print(f"  오늘 거래일 여부: {is_trading}")
    
    next_trading = ScheduleUtils.get_next_trading_day('US')
    print(f"  다음 거래일: {next_trading}")
    
    print("\n🏦 브로커 유틸리티 테스트:")
    broker_connected = BrokerUtils.connect('demo')
    print(f"  브로커 연결: {'성공' if broker_connected else '실패'}")
    
    account_info = BrokerUtils.get_account_info('demo')
    print(f"  계좌 정보: 잔고 ${account_info.get('balance', 0):,.0f}")
    
    positions = BrokerUtils.get_positions('demo')
    print(f"  포지션 조회: {len(positions)}개")
    
    print("\n✅ 모든 테스트 완료!")
    
    # 10. 시간대 비교 요약
    print("\n🌍 현재 시간 비교:")
    time_comparison = get_market_times_comparison()
    for market, time_str in time_comparison.items():
        print(f"  {market}: {time_str}")

if __name__ == "__main__":
    run_utils_test()
