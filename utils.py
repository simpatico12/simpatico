"""
🛠️ 최고퀸트프로젝트 - 공통 유틸리티 모듈 (Enhanced Edition)
========================================================================

전체 프로젝트에서 사용하는 공통 기능들:
- 📊 데이터 처리 및 변환
- 💰 금융 계산 함수
- 📁 파일 I/O 관리
- 🔄 API 재시도 로직
- 📈 기술적 지표 계산
- 📋 포맷팅 및 검증
- 💾 캐싱 시스템
- 📊 백테스트 유틸리티
- 🌍 시간대 관리
- 🔔 알림 시스템
- 🔒 보안 유틸리티
- 📱 텔레그램 통합

Author: 최고퀸트팀
Version: 2.0.0 (Enhanced Edition)
Project: 최고퀸트프로젝트
File: utils.py (프로젝트 루트)
"""

import asyncio
import logging
import json
import csv
import os
import pickle
import hashlib
import time
import secrets
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import yaml
import pandas as pd
import numpy as np
from functools import wraps
from pathlib import Path
import requests
from dataclasses import dataclass, asdict
import traceback
import pytz
from decimal import Decimal, ROUND_HALF_UP

# 설정 파일과 연동
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일 로드
except ImportError:
    pass

# 로거 설정
logger = logging.getLogger(__name__)

# ================================
# 🌐 프로젝트 설정 통합 로더
# ================================

class ConfigManager:
    """설정 파일 통합 관리자 (settings.yaml + .env 연동)"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        self.config_path = config_path
        self.config = None
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """설정 파일 로드 (환경변수 치환 포함)"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"설정 파일 없음: {self.config_path}, 기본 설정 사용")
                self.config = self._get_default_config()
                return self.config
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            # 환경변수 치환
            self.config = self._substitute_env_vars(raw_config)
            logger.info(f"설정 파일 로드 완료: {self.config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            self.config = self._get_default_config()
            return self.config
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """환경변수 치환 (${VAR_NAME:-default} 형식 지원)"""
        if isinstance(obj, str):
            # ${VAR_NAME:-default_value} 형식 처리
            import re
            pattern = r'\$\{([^}]+)\}'
            
            def replace_var(match):
                var_expr = match.group(1)
                if ':-' in var_expr:
                    var_name, default_value = var_expr.split(':-', 1)
                    return os.getenv(var_name, default_value)
                else:
                    return os.getenv(var_expr, match.group(0))
            
            return re.sub(pattern, replace_var, obj)
        
        elif isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        else:
            return obj
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정값"""
        return {
            'project': {
                'name': '최고퀸트프로젝트',
                'version': '2.0.0',
                'environment': 'development',
                'mode': 'safe'
            },
            'trading': {
                'paper_trading': True,
                'max_positions': 10,
                'risk_limit': 0.02
            },
            'api': {
                'upbit': {'enabled': True, 'paper_trading': True},
                'ibkr': {'enabled': True, 'paper_trading': True}
            },
            'notifications': {
                'telegram': {'enabled': False}
            },
            'us_strategy': {'enabled': True, 'confidence_threshold': 0.75},
            'jp_strategy': {'enabled': True, 'confidence_threshold': 0.60},
            'coin_strategy': {
                'enabled': True,
                'confidence_threshold': 0.65,
                'symbols': {
                    'MAJOR': ['KRW-BTC', 'KRW-ETH'],
                    'ALTCOIN': ['KRW-ADA', 'KRW-DOT']
                }
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """점 표기법으로 설정값 가져오기 (예: 'trading.max_positions')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_trading_config(self) -> Dict[str, Any]:
        """거래 관련 설정"""
        return self.get('trading', {})
    
    def get_strategy_config(self, strategy: str) -> Dict[str, Any]:
        """전략별 설정"""
        return self.get(f'{strategy}_strategy', {})
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """API 설정"""
        return self.get(f'api.{api_name}', {})
    
    def is_paper_trading(self) -> bool:
        """모의거래 모드 여부"""
        return self.get('trading.paper_trading', True)
    
    def get_risk_limits(self) -> Dict[str, float]:
        """리스크 제한 설정"""
        return {
            'max_position_size': self.get('trading.max_position_size', 0.05),
            'max_daily_loss': self.get('trading.max_daily_loss', 0.01),
            'max_drawdown': self.get('trading.max_drawdown', 0.1),
            'portfolio_risk': self.get('risk_management.max_portfolio_risk', 0.02)
        }

# 전역 설정 매니저
config_manager = ConfigManager()

# ================================
# 🕐 시간대 관리 유틸리티 (강화)
# ================================

class TimeZoneManager:
    """시간대 관리 전용 클래스 (강화)"""
    
    def __init__(self):
        """시간대 초기화"""
        self.timezones = {
            'KOR': pytz.timezone('Asia/Seoul'),      # 한국 시간 (KST)
            'US': pytz.timezone('US/Eastern'),       # 미국 동부 (EST/EDT 자동)
            'JP': pytz.timezone('Asia/Tokyo'),       # 일본 시간 (JST)
            'UTC': pytz.UTC,                         # 협정 시간
            'EU': pytz.timezone('Europe/London'),    # 유럽 (GMT/BST)
            'CN': pytz.timezone('Asia/Shanghai')     # 중국 (CST)
        }
        
        # 시장 운영 시간 (현지 시간 기준)
        self.market_hours = {
            'US': {
                'premarket_start': '04:00',
                'premarket_end': '09:30',
                'regular_start': '09:30',
                'regular_end': '16:00',
                'aftermarket_start': '16:00',
                'aftermarket_end': '20:00'
            },
            'JP': {
                'morning_start': '09:00',
                'morning_end': '11:30',
                'afternoon_start': '12:30',
                'afternoon_end': '15:00'
            },
            'EU': {
                'regular_start': '08:00',
                'regular_end': '16:30'
            },
            'COIN': {
                'start': '00:00',
                'end': '23:59'
            }
        }
        
        # 공휴일 캐시
        self.holidays_cache = {}

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
        
        for market in ['KOR', 'US', 'JP', 'EU']:
            current = self.get_current_time(market)
            current_times[market] = {
                'datetime': current.strftime('%Y-%m-%d %H:%M:%S'),
                'time_only': current.strftime('%H:%M:%S'),
                'date': current.strftime('%Y-%m-%d'),
                'weekday': current.strftime('%A'),
                'timezone_name': str(current.tzinfo),
                'timestamp': current.timestamp()
            }
        
        return current_times

    def is_weekend(self, timezone: str = 'KOR') -> bool:
        """주말 여부 확인"""
        current = self.get_current_time(timezone)
        return current.weekday() >= 5  # 5=토요일, 6=일요일

    def is_holiday(self, market: str, date: datetime = None) -> bool:
        """공휴일 여부 확인 (간단한 구현)"""
        if date is None:
            date = self.get_current_time(market.upper())
        
        # 주말은 기본적으로 휴일
        if date.weekday() >= 5:
            return True
        
        # 주요 공휴일 체크 (간단한 버전)
        month_day = (date.month, date.day)
        
        common_holidays = [
            (1, 1),   # 신정
            (12, 25), # 크리스마스
        ]
        
        us_holidays = [
            (7, 4),   # 독립기념일
            (11, 11), # 현충일
        ] + common_holidays
        
        jp_holidays = [
            (2, 11),  # 건국기념일
            (4, 29),  # 쇼와의 날
            (5, 3),   # 헌법기념일
            (5, 4),   # 미도리의 날
            (5, 5),   # 어린이날
        ] + common_holidays
        
        if market == 'US' and month_day in us_holidays:
            return True
        elif market == 'JP' and month_day in jp_holidays:
            return True
        elif month_day in common_holidays:
            return True
        
        return False

    def is_market_open_detailed(self, market: str) -> Dict[str, Any]:
        """상세 시장 개장 정보 (강화)"""
        market = market.upper()
        
        if market == 'COIN':
            return {
                'is_open': True,
                'session_type': '24시간',
                'status': 'open',
                'next_event': None,
                'current_time': self.get_current_time('UTC').strftime('%H:%M:%S UTC'),
                'market_phase': 'continuous'
            }
        
        # 시간대 매핑
        tz_map = {'US': 'US', 'JP': 'JP', 'KOR': 'KOR', 'EU': 'EU'}
        tz = tz_map.get(market, 'KOR')
        
        current = self.get_current_time(tz)
        current_time_str = current.strftime('%H:%M')
        
        # 공휴일 체크
        if self.is_holiday(market, current):
            return {
                'is_open': False,
                'session_type': '공휴일',
                'status': 'holiday',
                'next_event': '다음 거래일까지 대기',
                'current_time': current.strftime('%H:%M:%S'),
                'market_phase': 'closed'
            }
        
        # 주말 체크
        if self.is_weekend(tz):
            next_monday = current + timedelta(days=(7 - current.weekday()))
            return {
                'is_open': False,
                'session_type': '주말 휴장',
                'status': 'weekend',
                'next_event': f"월요일 개장까지 {self._get_time_diff(current, next_monday)}",
                'current_time': current.strftime('%H:%M:%S'),
                'market_phase': 'closed'
            }
        
        # 시장별 개장 시간 체크
        return self._check_market_session(market, current, current_time_str)

    def _check_market_session(self, market: str, current: datetime, current_time: str) -> Dict[str, Any]:
        """시장별 세션 체크"""
        hours = self.market_hours.get(market, {})
        
        if market == 'US':
            if self._time_in_range(current_time, hours.get('premarket_start'), hours.get('premarket_end')):
                return self._create_market_status(True, '프리마켓', 'premarket', current, 
                                                hours.get('regular_start'), 'EST/EDT')
            elif self._time_in_range(current_time, hours.get('regular_start'), hours.get('regular_end')):
                return self._create_market_status(True, '정규장', 'regular', current, 
                                                hours.get('regular_end'), 'EST/EDT')
            elif self._time_in_range(current_time, hours.get('aftermarket_start'), hours.get('aftermarket_end')):
                return self._create_market_status(True, '애프터마켓', 'aftermarket', current, 
                                                hours.get('aftermarket_end'), 'EST/EDT')
            else:
                return self._create_market_status(False, '휴장', 'closed', current, 
                                                hours.get('premarket_start'), 'EST/EDT')
        
        elif market == 'JP':
            if self._time_in_range(current_time, hours.get('morning_start'), hours.get('morning_end')):
                return self._create_market_status(True, '오전장', 'morning', current, 
                                                hours.get('morning_end'), 'JST')
            elif self._time_in_range(current_time, hours.get('morning_end'), hours.get('afternoon_start')):
                return self._create_market_status(False, '점심시간', 'lunch', current, 
                                                hours.get('afternoon_start'), 'JST')
            elif self._time_in_range(current_time, hours.get('afternoon_start'), hours.get('afternoon_end')):
                return self._create_market_status(True, '오후장', 'afternoon', current, 
                                                hours.get('afternoon_end'), 'JST')
            else:
                return self._create_market_status(False, '휴장', 'closed', current, 
                                                hours.get('morning_start'), 'JST')
        
        elif market == 'EU':
            if self._time_in_range(current_time, hours.get('regular_start'), hours.get('regular_end')):
                return self._create_market_status(True, '정규장', 'regular', current, 
                                                hours.get('regular_end'), 'GMT/BST')
            else:
                return self._create_market_status(False, '휴장', 'closed', current, 
                                                hours.get('regular_start'), 'GMT/BST')
        
        else:
            return {
                'is_open': False,
                'session_type': '알 수 없음',
                'status': 'unknown',
                'next_event': None,
                'current_time': current.strftime('%H:%M:%S'),
                'market_phase': 'unknown'
            }

    def _time_in_range(self, current_time: str, start_time: str, end_time: str) -> bool:
        """시간 범위 체크"""
        if not start_time or not end_time:
            return False
        
        current = datetime.strptime(current_time, '%H:%M').time()
        start = datetime.strptime(start_time, '%H:%M').time()
        end = datetime.strptime(end_time, '%H:%M').time()
        
        if start <= end:
            return start <= current < end
        else:  # 자정을 넘어가는 경우
            return current >= start or current < end

    def _create_market_status(self, is_open: bool, session_type: str, status: str, 
                            current: datetime, next_time: str, timezone: str) -> Dict[str, Any]:
        """시장 상태 객체 생성"""
        if next_time:
            next_event = f"{session_type} {'마감' if is_open else '시작'}까지 {self._get_time_until(current, next_time)}"
        else:
            next_event = None
        
        return {
            'is_open': is_open,
            'session_type': session_type,
            'status': status,
            'next_event': next_event,
            'current_time': current.strftime(f'%H:%M:%S {timezone}'),
            'market_phase': 'open' if is_open else 'closed'
        }

    def _get_time_until(self, current: datetime, target_time_str: str) -> str:
        """현재 시간부터 목표 시간까지 남은 시간"""
        try:
            target_hour, target_min = map(int, target_time_str.split(':'))
            target = current.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
            
            if target <= current:
                target += timedelta(days=1)
            
            diff = target - current
            return self._format_timedelta(diff)
        except:
            return "계산 불가"

    def _get_time_diff(self, from_time: datetime, to_time: datetime) -> str:
        """두 시간 사이의 차이"""
        diff = to_time - from_time
        return self._format_timedelta(diff)

    def _format_timedelta(self, td: timedelta) -> str:
        """timedelta 포맷팅"""
        total_seconds = int(td.total_seconds())
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        
        if days > 0:
            return f"{days}일 {hours}시간"
        elif hours > 0:
            return f"{hours}시간 {minutes}분"
        else:
            return f"{minutes}분"

    def get_trading_calendar(self, market: str, days: int = 7) -> List[Dict]:
        """향후 거래 일정"""
        calendar = []
        current_date = self.get_current_time(market.upper()).date()
        
        for i in range(days):
            date = current_date + timedelta(days=i)
            date_dt = datetime.combine(date, datetime.min.time())
            
            is_trading = not self.is_holiday(market, date_dt)
            
            calendar.append({
                'date': date.strftime('%Y-%m-%d'),
                'weekday': date.strftime('%A'),
                'is_trading_day': is_trading,
                'market': market.upper(),
                'note': '거래일' if is_trading else '휴장일'
            })
        
        return calendar

# ================================
# 📊 데이터 처리 유틸리티 (강화)
# ================================

class DataProcessor:
    """데이터 처리 전용 클래스 (강화)"""
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """심볼 정규화 (강화)"""
        if not symbol:
            return ""
        
        symbol = symbol.upper().strip()
        
        # 공백 및 특수문자 정리
        symbol = ''.join(c for c in symbol if c.isalnum() or c in '-.')
        
        # 암호화폐 처리
        if '-' in symbol and not symbol.endswith('.T'):
            parts = symbol.split('-')
            if len(parts) == 2:
                base, quote = parts
                # 일반적인 암호화폐 페어 검증
                if base in ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE'] and \
                   quote in ['KRW', 'USDT', 'USD', 'BTC', 'ETH']:
                    return symbol
        
        # 일본 주식 처리
        if symbol.endswith('.T') and len(symbol) >= 6:
            code_part = symbol[:-2]
            if code_part.isdigit() and len(code_part) == 4:
                return symbol
            
        # 미국 주식 처리 (기본)
        if symbol.replace('.', '').isalpha() and 1 <= len(symbol) <= 6:
            return symbol
        
        return symbol

    @staticmethod
    def detect_market(symbol: str) -> str:
        """심볼로 시장 판별 (강화)"""
        symbol = DataProcessor.normalize_symbol(symbol)
        
        if not symbol:
            return 'UNKNOWN'
        
        # 일본 주식
        if symbol.endswith('.T'):
            return 'JP'
        
        # 암호화폐
        if '-' in symbol:
            parts = symbol.split('-')
            if len(parts) == 2:
                base, quote = parts
                crypto_bases = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 
                              'SOL', 'MATIC', 'AVAX', 'ATOM', 'NEAR', 'DOGE', 'SHIB', 'LTC']
                crypto_quotes = ['KRW', 'USDT', 'USD', 'BTC', 'ETH']
                if base in crypto_bases and quote in crypto_quotes:
                    return 'COIN'
        
        # 미국 주식 (기본)
        if symbol.isalpha() and 1 <= len(symbol) <= 6:
            return 'US'
        
        return 'UNKNOWN'

    @staticmethod
    def clean_price_data(data: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
        """가격 데이터 정리 (강화)"""
        if data.empty:
            return data
        
        original_length = len(data)
        
        # 1. 결측값 처리
        data = data.dropna()
        
        # 2. 중복 제거 (인덱스 기준)
        if isinstance(data.index, pd.DatetimeIndex):
            data = data[~data.index.duplicated(keep='first')]
        
        # 3. 음수 가격 제거
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        price_columns = [col for col in numeric_columns if any(price_col in col.lower() 
                        for price_col in ['price', 'open', 'high', 'low', 'close', 'volume'])]
        
        for col in price_columns:
            if 'volume' not in col.lower():  # 거래량은 음수 가능
                data = data[data[col] > 0]
        
        # 4. 이상값 제거 (선택사항)
        if remove_outliers:
            for col in price_columns:
                if 'volume' not in col.lower():
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        # 5. 인덱스 정렬
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        
        cleaned_length = len(data)
        if original_length > 0:
            retention_rate = cleaned_length / original_length
            logger.info(f"데이터 정리 완료: {original_length} → {cleaned_length} ({retention_rate:.1%} 유지)")
        
        return data

    @staticmethod
    def calculate_returns(prices: pd.Series, periods: int = 1, method: str = 'simple') -> pd.Series:
        """수익률 계산 (강화)"""
        if method == 'simple':
            returns = prices.pct_change(periods=periods)
        elif method == 'log':
            returns = np.log(prices / prices.shift(periods))
        else:
            raise ValueError("method는 'simple' 또는 'log'여야 합니다")
        
        return returns.fillna(0)

    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
        """변동성 계산 (강화)"""
        vol = returns.rolling(window=window).std()
        
        if annualize:
            # 252 거래일 기준 연환산
            vol = vol * np.sqrt(252)
        
        return vol.fillna(0)

    @staticmethod
    def calculate_correlation_matrix(data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """상관관계 매트릭스 계산"""
        return data.corr(method=method)

    @staticmethod
    def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """이상값 탐지"""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
        
        else:
            raise ValueError("method는 'iqr' 또는 'zscore'여야 합니다")

    @staticmethod
    def resample_data(data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """데이터 리샘플링"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("데이터 인덱스가 DatetimeIndex여야 합니다")
        
        # OHLCV 데이터 처리
        agg_dict = {}
        for col in data.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                agg_dict[col] = 'first'
            elif 'high' in col_lower:
                agg_dict[col] = 'max'
            elif 'low' in col_lower:
                agg_dict[col] = 'min'
            elif 'close' in col_lower or 'price' in col_lower:
                agg_dict[col] = 'last'
            elif 'volume' in col_lower:
                agg_dict[col] = 'sum'
            else:
                agg_dict[col] = 'last'
        
        return data.resample(freq).agg(agg_dict).dropna()

# ================================
# 💰 금융 계산 함수 (강화)
# ================================

class FinanceUtils:
    """금융 계산 전용 클래스 (강화)"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14, method: str = 'wilder') -> pd.Series:
        """RSI 계산 (강화)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        if method == 'wilder':
            # Wilder's smoothing (원래 RSI 공식)
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        else:
            # Simple Moving Average
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Dict[str, pd.Series]:
        """MACD 계산 (강화)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram,
            'crossover': (macd > signal_line).astype(int).diff() == 1,
            'crossunder': (macd < signal_line).astype(int).diff() == 1
        }

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산 (강화)"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        # 추가 지표
        bb_width = (upper - lower) / sma
        bb_position = (prices - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'width': bb_width,
            'position': bb_position,
            'squeeze': bb_width < bb_width.rolling(20).quantile(0.1)
        }

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """스토캐스틱 계산"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent,
            'oversold': k_percent < 20,
            'overbought': k_percent > 80
        }

    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 14) -> pd.Series:
        """Williams %R 계산"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """Average True Range 계산"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                         tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, pd.Series]:
        """일목균형표 계산"""
        # 전환선 (Tenkan-sen)
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        
        # 기준선 (Kijun-sen)
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        
        # 선행스팬A (Senkou Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # 선행스팬B (Senkou Span B)
        senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
        
        # 후행스팬 (Chikou Span)
        chikou_span = close.shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span,
            'cloud_top': pd.concat([senkou_span_a, senkou_span_b], axis=1).max(axis=1),
            'cloud_bottom': pd.concat([senkou_span_a, senkou_span_b], axis=1).min(axis=1)
        }

    @staticmethod
    def calculate_fibonacci_retracement(high_price: float, low_price: float) -> Dict[str, float]:
        """피보나치 되돌림 계산"""
        diff = high_price - low_price
        
        return {
            'level_0': high_price,
            'level_23.6': high_price - 0.236 * diff,
            'level_38.2': high_price - 0.382 * diff,
            'level_50.0': high_price - 0.500 * diff,
            'level_61.8': high_price - 0.618 * diff,
            'level_78.6': high_price - 0.786 * diff,
            'level_100': low_price
        }

    @staticmethod
    def calculate_position_size(capital: float, risk_per_trade: float, 
                              entry_price: float, stop_loss_price: float,
                              method: str = 'fixed_risk') -> Dict[str, float]:
        """포지션 크기 계산 (강화)"""
        if entry_price <= 0 or stop_loss_price <= 0:
            return {'shares': 0, 'position_value': 0, 'risk_amount': 0}
        
        risk_amount = capital * risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return {'shares': 0, 'position_value': 0, 'risk_amount': 0}
        
        if method == 'fixed_risk':
            shares = risk_amount / price_risk
        elif method == 'fixed_percent':
            shares = (capital * risk_per_trade) / entry_price
        else:
            shares = risk_amount / price_risk
        
        position_value = shares * entry_price
        
        return {
            'shares': round(shares, 2),
            'position_value': round(position_value, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_percent': round((risk_amount / capital) * 100, 2)
        }

    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> Dict[str, float]:
        """켈리 공식 계산 (강화)"""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return {'kelly_percent': 0, 'recommended_percent': 0}
        
        loss_rate = 1 - win_rate
        win_loss_ratio = avg_win / avg_loss
        
        kelly_pct = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # 실용적 제한 (최대 25%)
        recommended_pct = max(0, min(kelly_pct * 0.5, 0.25))  # 켈리의 절반, 최대 25%
        
        return {
            'kelly_percent': round(kelly_pct * 100, 2),
            'recommended_percent': round(recommended_pct * 100, 2),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'expectancy': round((win_rate * avg_win) - (loss_rate * avg_loss), 2)
        }

    @staticmethod
    def calculate_portfolio_metrics(returns: pd.Series, benchmark_returns: pd.Series = None,
                                  risk_free_rate: float = 0.02) -> Dict[str, float]:
        """포트폴리오 성과 지표 계산"""
        if returns.empty:
            return {}
        
        # 기본 통계
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 샤프 비율
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility != 0 else 0
        
        # 최대 손실폭
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # 칼마 비율
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 소르티노 비율
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
        
        # 승률
        win_rate = (returns > 0).mean()
        
        metrics = {
            'total_return': round(total_return * 100, 2),
            'annualized_return': round(annualized_return * 100, 2),
            'volatility': round(volatility * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'calmar_ratio': round(calmar_ratio, 3),
            'sortino_ratio': round(sortino_ratio, 3),
            'max_drawdown': round(max_drawdown * 100, 2),
            'win_rate': round(win_rate * 100, 2),
            'best_day': round(returns.max() * 100, 2),
            'worst_day': round(returns.min() * 100, 2),
            'total_trades': len(returns[returns != 0])
        }
        
        # 벤치마크 대비 지표
        if benchmark_returns is not None and not benchmark_returns.empty:
            # 베타
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # 알파
            benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
            alpha = annualized_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
            
            # 정보 비율
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error != 0 else 0
            
            metrics.update({
                'beta': round(beta, 3),
                'alpha': round(alpha * 100, 2),
                'information_ratio': round(information_ratio, 3),
                'tracking_error': round(tracking_error * 100, 2)
            })
        
        return metrics

# ================================
# 📁 파일 I/O 관리 (강화)
# ================================

class FileManager:
    """파일 관리 전용 클래스 (강화)"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self._ensure_directories()
        self.compression_enabled = config_manager.get('data_management.backup.compression', True)

    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            'data', 'logs', 'data/cache', 'data/backups', 'data/prices', 
            'data/trades', 'data/models', 'reports', 'temp'
        ]
        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)

    def save_json(self, data: Any, filename: str, directory: str = "data", 
                 backup: bool = True, compress: bool = False) -> bool:
        """JSON 파일 저장 (강화)"""
        try:
            filepath = self.base_path / directory / filename
            
            # 기존 파일 백업
            if backup and filepath.exists():
                self.backup_file(filename, directory)
            
            # JSON 직렬화 개선
            if compress:
                import gzip
                with gzip.open(f"{filepath}.gz", 'wt', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str, separators=(',', ':'))
                logger.info(f"압축 JSON 파일 저장 완료: {filepath}.gz")
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                logger.info(f"JSON 파일 저장 완료: {filepath}")
            
            return True
        except Exception as e:
            logger.error(f"JSON 파일 저장 실패: {e}")
            return False

    def load_json(self, filename: str, directory: str = "data", 
                 check_compressed: bool = True) -> Optional[Any]:
        """JSON 파일 로드 (강화)"""
        try:
            filepath = self.base_path / directory / filename
            
            # 압축 파일 우선 확인
            if check_compressed:
                compressed_path = Path(f"{filepath}.gz")
                if compressed_path.exists():
                    import gzip
                    with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"압축 JSON 파일 로드 완료: {compressed_path}")
                    return data
            
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

    def save_csv(self, df: pd.DataFrame, filename: str, directory: str = "data",
                backup: bool = True, compression: str = None) -> bool:
        """CSV 파일 저장 (강화)"""
        try:
            filepath = self.base_path / directory / filename
            
            if backup and filepath.exists():
                self.backup_file(filename, directory)
            
            # 압축 옵션
            if compression:
                df.to_csv(filepath, index=False, encoding='utf-8', compression=compression)
            else:
                df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"CSV 파일 저장 완료: {filepath}")
            return True
        except Exception as e:
            logger.error(f"CSV 파일 저장 실패: {e}")
            return False

    def save_pickle(self, data: Any, filename: str, directory: str = "data") -> bool:
        """Pickle 파일 저장"""
        try:
            filepath = self.base_path / directory / filename
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Pickle 파일 저장 완료: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Pickle 파일 저장 실패: {e}")
            return False

    def load_pickle(self, filename: str, directory: str = "data") -> Optional[Any]:
        """Pickle 파일 로드"""
        try:
            filepath = self.base_path / directory / filename
            if not filepath.exists():
                return None
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Pickle 파일 로드 완료: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Pickle 파일 로드 실패: {e}")
            return None

    def backup_file(self, filename: str, directory: str = "data") -> bool:
        """파일 백업 (강화)"""
        try:
            filepath = self.base_path / directory / filename
            if not filepath.exists():
                return False
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
            backup_path = self.base_path / "data" / "backups" / backup_name
            
            import shutil
            if self.compression_enabled and filepath.suffix in ['.json', '.csv', '.txt']:
                # 백업시 압축
                import gzip
                with open(filepath, 'rb') as f_in:
                    with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                logger.info(f"압축 백업 완료: {backup_path}.gz")
            else:
                shutil.copy2(filepath, backup_path)
                logger.info(f"파일 백업 완료: {backup_path}")
            
            return True
        except Exception as e:
            logger.error(f"파일 백업 실패: {e}")
            return False

    def cleanup_old_files(self, directory: str = "logs", days: int = 30,
                         pattern: str = "*") -> int:
        """오래된 파일 정리 (강화)"""
        try:
            target_dir = self.base_path / directory
            cutoff_date = datetime.now() - timedelta(days=days)
            
            deleted_count = 0
            total_size = 0
            
            for filepath in target_dir.glob(pattern):
                if filepath.is_file():
                    file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_size = filepath.stat().st_size
                        filepath.unlink()
                        deleted_count += 1
                        total_size += file_size
            
            logger.info(f"{directory} 폴더에서 {deleted_count}개 파일 정리 완료 "
                       f"(절약된 용량: {self._format_bytes(total_size)})")
            return deleted_count
        except Exception as e:
            logger.error(f"파일 정리 실패: {e}")
            return 0

    def get_directory_size(self, directory: str = "data") -> Dict[str, Any]:
        """디렉토리 크기 정보"""
        try:
            target_dir = self.base_path / directory
            total_size = 0
            file_count = 0
            
            for filepath in target_dir.rglob("*"):
                if filepath.is_file():
                    total_size += filepath.stat().st_size
                    file_count += 1
            
            return {
                'directory': directory,
                'total_size_bytes': total_size,
                'total_size_formatted': self._format_bytes(total_size),
                'file_count': file_count
            }
        except Exception as e:
            logger.error(f"디렉토리 크기 계산 실패: {e}")
            return {}

    def _format_bytes(self, bytes_value: int) -> str:
        """바이트 크기 포맷팅"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

# ================================
# 🔄 API 재시도 로직 (강화)
# ================================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff: float = 2.0, exceptions: tuple = (Exception,),
                    jitter: bool = True):
    """API 호출 재시도 데코레이터 (강화)"""
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
                    
                    # 지터 추가 (실제 대기 시간에 랜덤성 부여)
                    actual_delay = current_delay
                    if jitter:
                        actual_delay *= (0.5 + 0.5 * secrets.randbelow(100) / 100)
                    
                    logger.warning(f"함수 {func.__name__} 재시도 {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(actual_delay)
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
                    
                    actual_delay = current_delay
                    if jitter:
                        actual_delay *= (0.5 + 0.5 * secrets.randbelow(100) / 100)
                    
                    logger.warning(f"함수 {func.__name__} 재시도 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(actual_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class RateLimiter:
    """API 호출 속도 제한 (강화)"""
    
    def __init__(self, calls_per_second: float = 1.0, burst_limit: int = None):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self.burst_limit = burst_limit or int(calls_per_second * 2)
        self.call_times = []

    async def wait(self):
        """속도 제한 대기 (버스트 지원)"""
        current_time = time.time()
        
        # 버스트 제한 확인
        self.call_times = [t for t in self.call_times if current_time - t < 1.0]
        
        if len(self.call_times) >= self.burst_limit:
            wait_time = 1.0 - (current_time - self.call_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                current_time = time.time()
        
        # 기본 속도 제한
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_interval:
            wait_time = self.min_interval - time_since_last_call
            await asyncio.sleep(wait_time)
            current_time = time.time()
        
        self.last_call_time = current_time
        self.call_times.append(current_time)

    def sync_wait(self):
        """동기 버전 대기"""
        current_time = time.time()
        
        self.call_times = [t for t in self.call_times if current_time - t < 1.0]
        
        if len(self.call_times) >= self.burst_limit:
            wait_time = 1.0 - (current_time - self.call_times[0])
            if wait_time > 0:
                time.sleep(wait_time)
                current_time = time.time()
        
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_interval:
            wait_time = self.min_interval - time_since_last_call
            time.sleep(wait_time)
            current_time = time.time()
        
        self.last_call_time = current_time
        self.call_times.append(current_time)

# ================================
# 📋 포맷팅 및 검증 (강화)
# ================================

class Formatter:
    """포맷팅 전용 클래스 (강화)"""
    
    @staticmethod
    def format_price(price: float, currency: str = 'USD', decimals: int = 2) -> str:
        """가격 포맷팅 (통화별)"""
        if pd.isna(price) or price == 0:
            return f"${0:.{decimals}f}" if currency == 'USD' else f"₩0"
        
        abs_price = abs(price)
        
        if currency == 'KRW':
            if abs_price >= 1000000:
                return f"₩{price/1000000:.1f}M"
            elif abs_price >= 1000:
                return f"₩{price:,.0f}"
            else:
                return f"₩{price:.0f}"
        
        elif currency == 'JPY':
            if abs_price >= 1000000:
                return f"¥{price/1000000:.1f}M"
            else:
                return f"¥{price:,.0f}"
        
        else:  # USD 기본
            if abs_price >= 1000000:
                return f"${price/1000000:.1f}M"
            elif abs_price >= 1000:
                return f"${price:,.{decimals}f}"
            elif abs_price >= 1:
                return f"${price:.{decimals}f}"
            else:
                return f"${price:.4f}"

    @staticmethod
    def format_percentage(value: float, decimals: int = 1, show_sign: bool = True) -> str:
        """퍼센트 포맷팅 (강화)"""
        if pd.isna(value):
            return "N/A"
        
        if show_sign:
            sign = "+" if value > 0 else ""
            return f"{sign}{value:.{decimals}f}%"
        else:
            return f"{value:.{decimals}f}%"

    @staticmethod
    def format_volume(volume: float) -> str:
        """거래량 포맷팅"""
        if pd.isna(volume) or volume == 0:
            return "0"
        
        abs_volume = abs(volume)
        
        if abs_volume >= 1e12:
            return f"{volume/1e12:.1f}T"
        elif abs_volume >= 1e9:
            return f"{volume/1e9:.1f}B"
        elif abs_volume >= 1e6:
            return f"{volume/1e6:.1f}M"
        elif abs_volume >= 1e3:
            return f"{volume/1e3:.1f}K"
        else:
            return f"{volume:.0f}"

    @staticmethod
    def format_duration(seconds: float) -> str:
        """시간 지속 포맷팅 (강화)"""
        if seconds < 0:
            return "0초"
        
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            return f"{seconds/60:.1f}분"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}시간"
        else:
            return f"{seconds/86400:.1f}일"

    @staticmethod
    def format_market_cap(market_cap: float) -> str:
        """시가총액 포맷팅"""
        if pd.isna(market_cap) or market_cap == 0:
            return "N/A"
        
        abs_cap = abs(market_cap)
        
        if abs_cap >= 1e12:
            return f"${market_cap/1e12:.1f}T"
        elif abs_cap >= 1e9:
            return f"${market_cap/1e9:.1f}B"
        elif abs_cap >= 1e6:
            return f"${market_cap/1e6:.1f}M"
        else:
            return f"${market_cap:,.0f}"

    @staticmethod
    def format_datetime(dt: datetime, format_type: str = 'default') -> str:
        """날짜시간 포맷팅"""
        if pd.isna(dt):
            return "N/A"
        
        if format_type == 'short':
            return dt.strftime('%m/%d %H:%M')
        elif format_type == 'date_only':
            return dt.strftime('%Y-%m-%d')
        elif format_type == 'time_only':
            return dt.strftime('%H:%M:%S')
        elif format_type == 'korean':
            return dt.strftime('%Y년 %m월 %d일 %H시 %M분')
        else:  # default
            return dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def format_trading_signal(signal: Dict[str, Any]) -> str:
        """거래 신호 포맷팅"""
        action = signal.get('action', '').upper()
        symbol = signal.get('symbol', '')
        confidence = signal.get('confidence', 0)
        price = signal.get('price', 0)
        
        action_emoji = {
            'BUY': '🟢',
            'SELL': '🔴', 
            'HOLD': '🟡',
            'WAIT': '⚪'
        }.get(action, '❓')
        
        confidence_text = f"{confidence:.1%}" if confidence else "N/A"
        price_text = Formatter.format_price(price) if price else "N/A"
        
        return f"{action_emoji} {action} {symbol} @ {price_text} (신뢰도: {confidence_text})"

class Validator:
    """검증 전용 클래스 (강화)"""
    
    @staticmethod
    def is_valid_symbol(symbol: str) -> bool:
        """심볼 유효성 검사 (강화)"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        symbol = symbol.strip().upper()
        if len(symbol) < 1 or len(symbol) > 20:
            return False
            
        # 패턴별 검증
        import re
        patterns = [
            r'^[A-Z]{1,6},                    # 미국 주식 (AAPL, MSFT 등)
            r'^[0-9]{4}\.T,                   # 일본 주식 (7203.T 등)
            r'^[A-Z]{2,10}-[A-Z]{3,10},       # 암호화폐 (BTC-KRW 등)
            r'^[A-Z]{1,6}\.[A-Z]{1,3}        # 기타 거래소 (TSE, LSE 등)
        ]
        
        return any(re.match(pattern, symbol) for pattern in patterns)

    @staticmethod
    def is_valid_price(price: float) -> bool:
        """가격 유효성 검사 (강화)"""
        return (isinstance(price, (int, float)) and 
                price > 0 and 
                not np.isnan(price) and 
                not np.isinf(price) and
                price < 1e10)  # 상한선 추가

    @staticmethod
    def is_valid_confidence(confidence: float) -> bool:
        """신뢰도 유효성 검사"""
        return (isinstance(confidence, (int, float)) and 
                0 <= confidence <= 1 and 
                not np.isnan(confidence))

    @staticmethod
    def is_valid_quantity(quantity: float, min_qty: float = 0) -> bool:
        """수량 유효성 검사"""
        return (isinstance(quantity, (int, float)) and 
                quantity > min_qty and 
                not np.isnan(quantity) and
                not np.isinf(quantity))

    @staticmethod
    def is_valid_percentage(percentage: float, min_pct: float = -100, max_pct: float = 1000) -> bool:
        """퍼센트 유효성 검사"""
        return (isinstance(percentage, (int, float)) and 
                min_pct <= percentage <= max_pct and 
                not np.isnan(percentage))

    @staticmethod
    def validate_trading_signal(signal: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """거래 신호 유효성 검사"""
        errors = []
        
        # 필수 필드 검사
        required_fields = ['symbol', 'action', 'confidence']
        for field in required_fields:
            if field not in signal:
                errors.append(f"필수 필드 누락: {field}")
        
        # 심볼 검사
        if 'symbol' in signal and not Validator.is_valid_symbol(signal['symbol']):
            errors.append(f"유효하지 않은 심볼: {signal['symbol']}")
        
        # 액션 검사
        valid_actions = ['BUY', 'SELL', 'HOLD', 'WAIT']
        if 'action' in signal and signal['action'].upper() not in valid_actions:
            errors.append(f"유효하지 않은 액션: {signal['action']}")
        
        # 신뢰도 검사
        if 'confidence' in signal and not Validator.is_valid_confidence(signal['confidence']):
            errors.append(f"유효하지 않은 신뢰도: {signal['confidence']}")
        
        # 가격 검사 (있는 경우)
        if 'price' in signal and signal['price'] is not None:
            if not Validator.is_valid_price(signal['price']):
                errors.append(f"유효하지 않은 가격: {signal['price']}")
        
        return len(errors) == 0, errors

    @staticmethod
    def sanitize_input(input_str: str, max_length: int = 100) -> str:
        """입력값 정화"""
        if not isinstance(input_str, str):
            return ""
        
        # 위험한 문자 제거
        import re
        sanitized = re.sub(r'[<>"\';]', '', input_str)
        
        # 길이 제한
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()

# ================================
# 💾 캐싱 시스템 (강화)
# ================================

class SimpleCache:
    """간단한 메모리 캐시 (강화)"""
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.cache = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.access_count = {}
        self.last_cleanup = time.time()

    def _is_expired(self, timestamp: float) -> bool:
        """만료 확인"""
        return time.time() - timestamp > self.ttl

    def _cleanup_if_needed(self):
        """필요시 캐시 정리"""
        current_time = time.time()
        
        # 5분마다 정리
        if current_time - self.last_cleanup > 300:
            self.cleanup()
            self.last_cleanup = current_time
        
        # 크기 제한 초과시 LRU 제거
        if len(self.cache) > self.max_size:
            self._evict_lru()

    def _evict_lru(self):
        """LRU 방식으로 오래된 항목 제거"""
        # 접근 횟수가 적은 항목부터 제거
        sorted_keys = sorted(self.access_count.keys(), key=lambda k: self.access_count[k])
        remove_count = len(self.cache) - self.max_size + 1
        
        for key in sorted_keys[:remove_count]:
            if key in self.cache:
                del self.cache[key]
                del self.access_count[key]

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기 (강화)"""
        self._cleanup_if_needed()
        
        if key not in self.cache:
            return None
            
        data, timestamp = self.cache[key]
        if self._is_expired(timestamp):
            del self.cache[key]
            if key in self.access_count:
                del self.access_count[key]
            return None
        
        # 접근 횟수 증가
        self.access_count[key] = self.access_count.get(key, 0) + 1
        return data

    def set(self, key: str, value: Any, ttl_override: int = None):
        """캐시에 값 저장 (강화)"""
        self._cleanup_if_needed()
        
        expiry_time = time.time() + (ttl_override or self.ttl)
        self.cache[key] = (value, expiry_time)
        self.access_count[key] = self.access_count.get(key, 0) + 1

    def delete(self, key: str):
        """특정 키 삭제"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_count:
            del self.access_count[key]

    def clear(self):
        """캐시 전체 삭제"""
        self.cache.clear()
        self.access_count.clear()

    def cleanup(self):
        """만료된 항목 정리"""
        expired_keys = []
        current_time = time.time()
        
        for key, (data, timestamp) in self.cache.items():
            if current_time - timestamp > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_count:
                del self.access_count[key]
        
        logger.debug(f"캐시 정리 완료: {len(expired_keys)}개 항목 제거")

    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl,
            'usage_percent': (len(self.cache) / self.max_size) * 100,
            'top_accessed': sorted(self.access_count.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
        }

# ================================
# 🔔 알림 시스템 (강화)
# ================================

class NotificationManager:
    """통합 알림 관리자"""
    
    def __init__(self):
        self.telegram_enabled = config_manager.get('notifications.telegram.enabled', False)
        self.email_enabled = config_manager.get('notifications.email.enabled', False)
        self.slack_enabled = config_manager.get('notifications.slack.enabled', False)
        
        # 알림 레벨 설정
        self.notification_levels = {
            'critical': 1,
            'error': 2,
            'warning': 3,
            'info': 4,
            'debug': 5
        }
        
        self.min_level = self.notification_levels.get(
            config_manager.get('notifications.min_level', 'info'), 4
        )

    async def send_notification(self, message: str, level: str = 'info', 
                              channels: List[str] = None) -> Dict[str, bool]:
        """통합 알림 발송"""
        if self.notification_levels.get(level, 4) > self.min_level:
            return {'skipped': True}
        
        results = {}
        
        # 채널 지정이 없으면 활성화된 모든 채널 사용
        if channels is None:
            channels = []
            if self.telegram_enabled:
                channels.append('telegram')
            if self.email_enabled:
                channels.append('email')
            if self.slack_enabled:
                channels.append('slack')
        
        # 레벨별 이모지 추가
        level_emojis = {
            'critical': '🚨',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'debug': '🔍'
        }
        
        formatted_message = f"{level_emojis.get(level, '')} {message}"
        
        # 각 채널별 발송
        for channel in channels:
            try:
                if channel == 'telegram':
                    results[channel] = await self._send_telegram(formatted_message)
                elif channel == 'email':
                    results[channel] = await self._send_email(formatted_message, level)
                elif channel == 'slack':
                    results[channel] = await self._send_slack(formatted_message)
                else:
                    results[channel] = False
            except Exception as e:
                logger.error(f"알림 발송 실패 ({channel}): {e}")
                results[channel] = False
        
        return results

    async def _send_telegram(self, message: str) -> bool:
        """텔레그램 알림 발송"""
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                logger.warning("텔레그램 설정이 누락됨")
                return False
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with requests.Session() as session:
                response = session.post(url, json=payload, timeout=10)
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"텔레그램 발송 실패: {e}")
            return False

    async def _send_email(self, message: str, level: str) -> bool:
        """이메일 알림 발송"""
        try:
            # 이메일 발송 로직 (실제 구현 필요)
            logger.info(f"이메일 발송 시뮬레이션: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"이메일 발송 실패: {e}")
            return False

    async def _send_slack(self, message: str) -> bool:
        """슬랙 알림 발송"""
        try:
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                return False
            
            payload = {'text': message}
            async with requests.Session() as session:
                response = session.post(webhook_url, json=payload, timeout=10)
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"슬랙 발송 실패: {e}")
            return False

    def send_trading_alert(self, signal: Dict[str, Any]):
        """거래 신호 알림"""
        message = Formatter.format_trading_signal(signal)
        asyncio.create_task(self.send_notification(message, 'info'))

    def send_error_alert(self, error_msg: str, context: str = ""):
        """에러 알림"""
        message = f"🚨 에러 발생\n{context}\n{error_msg}"
        asyncio.create_task(self.send_notification(message, 'error'))

    def send_performance_report(self, report: Dict[str, Any]):
        """성과 리포트 알림"""
        message = self._format_performance_report(report)
        asyncio.create_task(self.send_notification(message, 'info'))

    def _format_performance_report(self, report: Dict[str, Any]) -> str:
        """성과 리포트 포맷팅"""
        return f"""
📊 일일 성과 리포트

💰 총 수익률: {Formatter.format_percentage(report.get('total_return', 0))}
📈 연환산 수익률: {Formatter.format_percentage(report.get('annualized_return', 0))}
📉 최대 손실폭: {Formatter.format_percentage(report.get('max_drawdown', 0))}
🎯 승률: {Formatter.format_percentage(report.get('win_rate', 0))}
📊 샤프 비율: {report.get('sharpe_ratio', 0):.2f}
🔢 총 거래 횟수: {report.get('total_trades', 0)}

시간: {Formatter.format_datetime(datetime.now())}
        """.strip()

# ================================
# 🔒 보안 유틸리티
# ================================

class SecurityUtils:
    """보안 관련 유틸리티"""
    
    @staticmethod
    def encrypt_api_key(api_key: str, master_key: str = None) -> str:
        """API 키 암호화"""
        try:
            if not master_key:
                master_key = os.getenv('MASTER_ENCRYPTION_KEY', 'default_key_change_this')
            
            from cryptography.fernet import Fernet
            key = SecurityUtils._derive_key(master_key)
            f = Fernet(key)
            
            encrypted = f.encrypt(api_key.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"API 키 암호화 실패: {e}")
            return api_key

    @staticmethod
    def decrypt_api_key(encrypted_key: str, master_key: str = None) -> str:
        """API 키 복호화"""
        try:
            if not master_key:
                master_key = os.getenv('MASTER_ENCRYPTION_KEY', 'default_key_change_this')
            
            from cryptography.fernet import Fernet
            key = SecurityUtils._derive_key(master_key)
            f = Fernet(key)
            
            decrypted = f.decrypt(encrypted_key.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"API 키 복호화 실패: {e}")
            return encrypted_key

    @staticmethod
    def _derive_key(password: str) -> bytes:
        """비밀번호에서 암호화 키 도출"""
        import base64
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        
        salt = b'quant_project_salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    @staticmethod
    def hash_password(password: str) -> str:
        """비밀번호 해시"""
        import bcrypt
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """비밀번호 검증"""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    @staticmethod
    def generate_api_signature(secret: str, data: str) -> str:
        """API 서명 생성"""
        import hmac
        import hashlib
        
        signature = hmac.new(
            secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature

# ================================
# 📊 백테스트 유틸리티 (강화)
# ================================

class BacktestUtils:
    """백테스트 관련 유틸리티 (강화)"""
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """최대 손실폭 계산"""
        if equity_curve.empty:
            return 0.0
        
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    @staticmethod
    def calculate_underwater_periods(equity_curve: pd.Series) -> pd.DataFrame:
        """수중 기간 분석"""
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        
        # 손실 구간 식별
        underwater = drawdown < -0.01  # 1% 이상 손실
        periods = []
        
        in_drawdown = False
        start_date = None
        
        for date, is_underwater in underwater.items():
            if is_underwater and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif not is_underwater and in_drawdown:
                in_drawdown = False
                if start_date:
                    duration = (date - start_date).days
                    max_dd = drawdown[start_date:date].min()
                    periods.append({
                        'start': start_date,
                        'end': date,
                        'duration_days': duration,
                        'max_drawdown': max_dd
                    })
        
        return pd.DataFrame(periods)

    @staticmethod
    def calculate_rolling_performance(returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """롤링 성과 분석"""
        rolling_return = returns.rolling(window).sum()
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_return * 252 - 0.02) / rolling_vol
        
        return pd.DataFrame({
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe
        })

    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
        """Value at Risk 계산"""
        if returns.empty:
            return 0.0
        
        return returns.quantile(confidence)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.05) -> float:
        """Conditional Value at Risk 계산"""
        var = BacktestUtils.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def monte_carlo_simulation(returns: pd.Series, days: int = 252, 
                             simulations: int = 1000) -> pd.DataFrame:
        """몬테카르로 시뮬레이션"""
        np.random.seed(42)
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        simulated_paths = []
        
        for i in range(simulations):
            random_returns = np.random.normal(mean_return, std_return, days)
            cumulative_returns = (1 + random_returns).cumprod()
            simulated_paths.append(cumulative_returns)
        
        simulation_df = pd.DataFrame(simulated_paths).T
        
        return {
            'paths': simulation_df,
            'final_values': simulation_df.iloc[-1],
            'percentiles': {
                '5%': simulation_df.iloc[-1].quantile(0.05),
                '50%': simulation_df.iloc[-1].quantile(0.50),
                '95%': simulation_df.iloc[-1].quantile(0.95)
            }
        }

# ================================
# 🌐 전역 객체 및 편의 함수
# ================================

# 전역 객체들 초기화
file_manager = FileManager()
cache = SimpleCache(
    ttl_seconds=config_manager.get('performance.caching.ttl_seconds', 300),
    max_size=config_manager.get('performance.caching.max_size', 1000)
)
timezone_manager = TimeZoneManager()
notification_manager = NotificationManager()

def get_config(key_path: str = None, default: Any = None) -> Any:
    """설정값 조회 (캐시 적용)"""
    if key_path is None:
        return config_manager.config
    
    cached_key = f"config_{key_path}"
    cached = cache.get(cached_key)
    if cached is not None:
        return cached
    
    value = config_manager.get(key_path, default)
    cache.set(cached_key, value, ttl_override=600)  # 10분 캐시
    return value

def save_trading_log(log_data: Dict, log_type: str = "trading"):
    """거래 로그 저장 (강화)"""
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{log_type}_log_{timestamp}.json"
    
    # 기존 로그 로드
    existing_logs = file_manager.load_json(filename, "logs") or []
    
    # 새 로그 추가 (모든 시간대 정보 포함)
    enhanced_log = {
        **log_data,
        'timestamp': datetime.now().isoformat(),
        'market_times': timezone_manager.get_all_market_times(),
        'log_id': f"{log_type}_{int(time.time())}_{secrets.randbelow(1000)}",
        'session_id': os.getenv('SESSION_ID', 'default'),
        'environment': config_manager.get('project.environment', 'development')
    }
    
    existing_logs.append(enhanced_log)
    
    # 로그 크기 제한 (최대 1000개)
    if len(existing_logs) > 1000:
        existing_logs = existing_logs[-1000:]
    
    # 저장
    success = file_manager.save_json(existing_logs, filename, "logs")
    
    if success and log_data.get('level') in ['error', 'critical']:
        # 중요한 로그는 알림 발송
        asyncio.create_task(
            notification_manager.send_notification(
                f"로그 기록: {log_data.get('message', '')}", 
                log_data.get('level', 'info')
            )
        )

def get_market_status_summary() -> Dict[str, Any]:
    """시장 상태 요약"""
    all_status = {}
    
    for market in ['US', 'JP', 'COIN']:
        status = timezone_manager.is_market_open_detailed(market)
        all_status[market] = {
            'is_open': status['is_open'],
            'session_type': status['session_type'],
            'next_event': status['next_event']
        }
    
    # 한국 시간 추가
    seoul_time = timezone_manager.get_current_time('KOR')
    all_status['KOR'] = {
        'current_time': seoul_time.strftime('%Y-%m-%d %H:%M:%S KST'),
        'weekday': seoul_time.strftime('%A'),
        'is_weekend': timezone_manager.is_weekend('KOR')
    }
    
    return all_status

def calculate_portfolio_summary(positions: Dict[str, Dict]) -> Dict[str, Any]:
    """포트폴리오 요약 계산"""
    if not positions:
        return {
            'total_value': 0,
            'total_pnl': 0,
            'position_count': 0,
            'markets': {}
        }
    
    total_value = 0
    total_pnl = 0
    market_breakdown = {}
    
    for symbol, position in positions.items():
        quantity = position.get('quantity', 0)
        current_price = position.get('current_price', 0)
        avg_price = position.get('avg_price', 0)
        
        market_value = quantity * current_price
        pnl = quantity * (current_price - avg_price)
        
        total_value += market_value
        total_pnl += pnl
        
        # 시장별 분류
        market = DataProcessor.detect_market(symbol)
        if market not in market_breakdown:
            market_breakdown[market] = {
                'value': 0,
                'pnl': 0,
                'count': 0,
                'symbols': []
            }
        
        market_breakdown[market]['value'] += market_value
        market_breakdown[market]['pnl'] += pnl
        market_breakdown[market]['count'] += 1
        market_breakdown[market]['symbols'].append(symbol)
    
    return {
        'total_value': round(total_value, 2),
        'total_pnl': round(total_pnl, 2),
        'total_pnl_percent': round((total_pnl / (total_value - total_pnl)) * 100, 2) if total_value != total_pnl else 0,
        'position_count': len(positions),
        'markets': market_breakdown,
        'timestamp': datetime.now().isoformat()
    }

def monitor_system_health() -> Dict[str, Any]:
    """시스템 상태 모니터링"""
    import psutil
    
    try:
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # 프로세스 정보
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 네트워크 상태 (간단한 체크)
        network_ok = True
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
        except:
            network_ok = False
        
        health_status = {
            'system': {
                'cpu_percent': round(cpu_percent, 1),
                'memory_percent': round(memory_percent, 1),
                'disk_percent': round(disk_percent, 1),
                'network_ok': network_ok
            },
            'process': {
                'memory_mb': round(process_memory, 1),
                'threads': process.num_threads(),
                'status': process.status()
            },
            'cache': cache.stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 경고 체크
        warnings = []
        if cpu_percent > 80:
            warnings.append(f"높은 CPU 사용률: {cpu_percent:.1f}%")
        if memory_percent > 85:
            warnings.append(f"높은 메모리 사용률: {memory_percent:.1f}%")
        if disk_percent > 90:
            warnings.append(f"높은 디스크 사용률: {disk_percent:.1f}%")
        if not network_ok:
            warnings.append("네트워크 연결 문제")
        
        health_status['warnings'] = warnings
        health_status['status'] = 'warning' if warnings else 'healthy'
        
        return health_status
        
    except Exception as e:
        logger.error(f"시스템 상태 모니터링 실패: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def cleanup_system():
    """시스템 정리"""
    logger.info("시스템 정리 시작...")
    
    # 캐시 정리
    cache.cleanup()
    
    # 오래된 로그 파일 정리
    log_retention_days = config_manager.get('data_management.retention.log_data_days', 30)
    file_manager.cleanup_old_files('logs', log_retention_days, '*.log')
    file_manager.cleanup_old_files('logs', log_retention_days, '*.json')
    
    # 임시 파일 정리
    file_manager.cleanup_old_files('temp', 1, '*')
    
    # 백업 파일 정리
    backup_retention_days = config_manager.get('data_management.backup.retention_days', 30)
    file_manager.cleanup_old_files('data/backups', backup_retention_days)
    
    logger.info("시스템 정리 완료")

# ================================
# 🧪 테스트 및 검증 함수들
# ================================

def validate_environment() -> Dict[str, Any]:
    """환경 검증"""
    validation_results = {
        'config_file': os.path.exists('settings.yaml'),
        'env_file': os.path.exists('.env'),
        'required_dirs': True,
        'python_version': sys.version_info >= (3, 8),
        'required_packages': [],
        'api_keys': {},
        'issues': []
    }
    
    # 필수 디렉토리 확인
    required_dirs = ['data', 'logs', 'data/cache', 'data/backups']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            validation_results['required_dirs'] = False
            validation_results['issues'].append(f"디렉토리 없음: {dir_name}")
    
    # 필수 패키지 확인
    required_packages = ['pandas', 'numpy', 'yaml', 'requests', 'pytz']
    for package in required_packages:
        try:
            __import__(package)
            validation_results['required_packages'].append({'name': package, 'status': 'ok'})
        except ImportError:
            validation_results['required_packages'].append({'name': package, 'status': 'missing'})
            validation_results['issues'].append(f"패키지 없음: {package}")
    
    # API 키 확인
    api_keys_to_check = [
        'UPBIT_ACCESS_KEY', 'UPBIT_SECRET_KEY',
        'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID'
    ]
    
    for key_name in api_keys_to_check:
        key_value = os.getenv(key_name)
        validation_results['api_keys'][key_name] = {
            'configured': bool(key_value),
            'length': len(key_value) if key_value else 0
        }
        
        if not key_value:
            validation_results['issues'].append(f"API 키 없음: {key_name}")
    
    # 전체 상태 결정
    validation_results['overall_status'] = 'ok' if not validation_results['issues'] else 'issues'
    
    return validation_results

def run_comprehensive_test():
    """종합 테스트 실행"""
    print("🛠️ 최고퀸트프로젝트 - 강화된 유틸리티 종합 테스트")
    print("=" * 70)
    
    # 1. 환경 검증
    print("\n🔍 환경 검증:")
    env_validation = validate_environment()
    print(f"  전체 상태: {'✅ 정상' if env_validation['overall_status'] == 'ok' else '⚠️ 문제 있음'}")
    
    if env_validation['issues']:
        print("  발견된 문제:")
        for issue in env_validation['issues'][:5]:  # 최대 5개만 표시
            print(f"    - {issue}")
    
    # 2. 설정 시스템 테스트
    print("\n⚙️ 설정 시스템 테스트:")
    test_config = config_manager.get('project.name', 'Unknown')
    print(f"  프로젝트 이름: {test_config}")
    print(f"  거래 모드: {'모의거래' if config_manager.is_paper_trading() else '실거래'}")
    
    # 3. 시간대 관리 테스트
    print("\n🕐 시간대 관리 테스트:")
    current_times = timezone_manager.get_all_market_times()
    for market, time_info in current_times.items():
        market_name = {'KOR': '🇰🇷 서울', 'US': '🇺🇸 뉴욕', 'JP': '🇯🇵 도쿄', 'EU': '🇪🇺 런던'}[market]
        print(f"  {market_name}: {time_info['datetime']}")
    
    # 4. 시장 개장 상태 테스트
    print("\n📈 시장 상태 테스트:")
    market_status = get_market_status_summary()
    for market in ['US', 'JP', 'COIN']:
        status = market_status[market]
        market_name = {'US': '🇺🇸 미국', 'JP': '🇯🇵 일본', 'COIN': '🪙 암호화폐'}[market]
        open_status = "🟢 개장" if status['is_open'] else "🔴 휴장"
        print(f"  {market_name}: {open_status} - {status['session_type']}")
    
    # 5. 데이터 처리 테스트
    print("\n📊 데이터 처리 테스트:")
    test_symbols = ['AAPL', '7203.T', 'BTC-KRW', 'INVALID']
    for symbol in test_symbols:
        market = DataProcessor.detect_market(symbol)
        is_valid = Validator.is_valid_symbol(symbol)
        print(f"  {symbol}: {market} 시장, 유효성: {'✅' if is_valid else '❌'}")
    
    # 6. 금융 계산 테스트
    print("\n💰 금융 계산 테스트:")
    # 샘플 데이터 생성
    np.random.seed(42)
    sample_prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02))
    
    rsi = FinanceUtils.calculate_rsi(sample_prices)
    macd_data = FinanceUtils.calculate_macd(sample_prices)
    bb_data = FinanceUtils.calculate_bollinger_bands(sample_prices)
    
    print(f"  RSI (마지막): {rsi.iloc[-1]:.2f}")
    print(f"  MACD: {macd_data['macd'].iloc[-1]:.4f}")
    print(f"  볼린저 밴드 폭: {bb_data['width'].iloc[-1]:.4f}")
    
    # 7. 포맷팅 테스트
    print("\n📋 포맷팅 테스트:")
    test_values = [0.0001, 1.23, 1234.56, 1234567.89]
    for value in test_values:
        formatted_usd = Formatter.format_price(value, 'USD')
        formatted_krw = Formatter.format_price(value * 1300, 'KRW')
        print(f"  ${value} → {formatted_usd} / {formatted_krw}")
    
    # 8. 캐시 테스트
    print("\n💾 캐시 시스템 테스트:")
    cache.set('test_key', {'test': 'data', 'timestamp': time.time()})
    cached_value = cache.get('test_key')
    cache_stats = cache.stats()
    print(f"  캐시 저장/로드: {'✅ 성공' if cached_value else '❌ 실패'}")
    print(f"  캐시 사용률: {cache_stats['usage_percent']:.1f}%")
    
    # 9. 파일 관리 테스트
    print("\n📁 파일 관리 테스트:")
    test_data = {
        'test': 'enhanced_data',
        'timestamp': datetime.now().isoformat(),
        'market_times': timezone_manager.get_all_market_times()
    }
    save_success = file_manager.save_json(test_data, 'enhanced_test.json')
    load_success = file_manager.load_json('enhanced_test.json') is not None
    print(f"  JSON 저장: {'✅ 성공' if save_success else '❌ 실패'}")
    print(f"  JSON 로드: {'✅ 성공' if load_success else '❌ 실패'}")
    
    # 10. 보안 테스트
    print("\n🔒 보안 기능 테스트:")
    test_api_key = "test_api_key_12345"
    encrypted = SecurityUtils.encrypt_api_key(test_api_key)
    decrypted = SecurityUtils.decrypt_api_key(encrypted)
    print(f"  암호화/복호화: {'✅ 성공' if decrypted == test_api_key else '❌ 실패'}")
    
    # 11. 시스템 상태 테스트
    print("\n🖥️ 시스템 상태 테스트:")
    health = monitor_system_health()
    if health.get('status') == 'healthy':
        print("  시스템 상태: ✅ 정상")
        print(f"    CPU: {health['system']['cpu_percent']}%")
        print(f"    메모리: {health['system']['memory_percent']}%")
    else:
        print("  시스템 상태: ⚠️ 주의 필요")
        for warning in health.get('warnings', []):
            print(f"    - {warning}")
    
    # 12. 포트폴리오 계산 테스트
    print("\n💼 포트폴리오 계산 테스트:")
    sample_positions = {
        'AAPL': {'quantity': 100, 'current_price': 150, 'avg_price': 145},
        'BTC-KRW': {'quantity': 0.1, 'current_price': 50000000, 'avg_price': 48000000},
        '7203.T': {'quantity': 500, 'current_price': 2500, 'avg_price': 2400}
    }
    
    portfolio_summary = calculate_portfolio_summary(sample_positions)
    print(f"  총 포트폴리오 가치: {Formatter.format_price(portfolio_summary['total_value'])}")
    print(f"  총 손익: {Formatter.format_price(portfolio_summary['total_pnl'])}")
    print(f"  포지션 수: {portfolio_summary['position_count']}개")
    print(f"  시장별 분포: {len(portfolio_summary['markets'])}개 시장")
    
    # 13. 백테스트 유틸리티 테스트
    print("\n📊 백테스트 유틸리티 테스트:")
    sample_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1년치 일일 수익률
    
    max_dd = BacktestUtils.calculate_max_drawdown((1 + sample_returns).cumprod())
    var_5 = BacktestUtils.calculate_var(sample_returns, 0.05)
    cvar_5 = BacktestUtils.calculate_cvar(sample_returns, 0.05)
    
    print(f"  최대 손실폭: {Formatter.format_percentage(max_dd * 100)}")
    print(f"  VaR (5%): {Formatter.format_percentage(var_5 * 100)}")
    print(f"  CVaR (5%): {Formatter.format_percentage(cvar_5 * 100)}")
    
    # 14. 거래 신호 검증 테스트
    print("\n🎯 거래 신호 검증 테스트:")
    test_signals = [
        {'symbol': 'AAPL', 'action': 'BUY', 'confidence': 0.85, 'price': 150.0},
        {'symbol': 'INVALID', 'action': 'BUY', 'confidence': 0.85},
        {'symbol': 'BTC-KRW', 'action': 'HOLD', 'confidence': 1.5}  # 잘못된 신뢰도
    ]
    
    for i, signal in enumerate(test_signals):
        is_valid, errors = Validator.validate_trading_signal(signal)
        status = "✅ 유효" if is_valid else f"❌ 오류: {', '.join(errors)}"
        print(f"  신호 {i+1}: {status}")
    
    # 15. 알림 시스템 테스트
    print("\n🔔 알림 시스템 테스트:")
    print("  텔레그램 활성화:", "✅" if notification_manager.telegram_enabled else "❌")
    print("  이메일 활성화:", "✅" if notification_manager.email_enabled else "❌")
    print("  슬랙 활성화:", "✅" if notification_manager.slack_enabled else "❌")
    
    # 테스트 완료 요약
    print("\n" + "=" * 70)
    print("✅ 강화된 유틸리티 종합 테스트 완료!")
    print(f"🕐 테스트 시간: {Formatter.format_datetime(datetime.now())}")
    
    # 간단한 성능 벤치마크
    print("\n⚡ 성능 벤치마크:")
    
    # 데이터 처리 속도
    start_time = time.time()
    for _ in range(1000):
        DataProcessor.normalize_symbol('AAPL')
    symbol_processing_time = time.time() - start_time
    print(f"  심볼 정규화 (1000회): {symbol_processing_time:.3f}초")
    
    # 캐시 성능
    start_time = time.time()
    for i in range(1000):
        cache.set(f'bench_{i}', f'value_{i}')
        cache.get(f'bench_{i}')
    cache_performance_time = time.time() - start_time
    print(f"  캐시 저장/로드 (1000회): {cache_performance_time:.3f}초")
    
    # 최종 권장사항
    print("\n💡 권장사항:")
    if env_validation['issues']:
        print("  1. 환경 설정 문제를 해결하세요")
    if not config_manager.is_paper_trading():
        print("  2. ⚠️ 실거래 모드입니다 - 신중하게 사용하세요!")
    print("  3. 정기적으로 시스템 상태를 모니터링하세요")
    print("  4. 로그 파일을 주기적으로 확인하세요")
    
    print("\n🚀 시스템이 준비되었습니다!")

# ================================
# 메인 실행부
# ================================

if __name__ == "__main__":
    # 기본 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/utils.log', encoding='utf-8')
        ]
    )
    
    # 종합 테스트 실행
    run_comprehensive_test()
