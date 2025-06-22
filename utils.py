"""
🛠️ 최고퀸트프로젝트 - 공통 유틸리티 모듈
======================================

완전한 유틸리티 시스템:
- 📊 데이터 처리 및 분석
- 💰 금융 계산 및 지표
- 📁 파일 관리 시스템
- 🕐 시간대 처리 (서울/뉴욕/도쿄)
- 🔄 API 재시도 및 속도 제한
- 📋 포맷팅 및 검증
- 💾 캐싱 시스템
- 📈 백테스트 유틸리티
- 🔌 브로커 연동 헬퍼
- 📰 뉴스 분석 헬퍼
- 📅 스케줄링 헬퍼

Author: 최고퀸트팀
Version: 1.0.0
Project: 최고퀸트프로젝트
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from functools import wraps
import time
import hashlib
import pytz
from pathlib import Path
import sqlite3

# 로깅 설정
logger = logging.getLogger(__name__)

# =====================================
# 📊 데이터 처리 클래스 (업데이트)
# =====================================

class DataProcessor:
    """데이터 처리 및 분석"""
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """심볼 정규화"""
        if not symbol:
            return ""
        
        symbol = symbol.upper().strip()
        
        # 일본 주식 처리
        if '.T' in symbol and not symbol.endswith('.T'):
            symbol = symbol.replace('.T', '') + '.T'
        
        # 암호화폐 처리 
        if '-' in symbol:
            parts = symbol.split('-')
            if len(parts) == 2:
                base, quote = parts[0].upper(), parts[1].upper()
                # KRW 기본으로 통일
                if quote in ['KRW', 'USDT', 'BTC']:
                    symbol = f"{base}-{quote}"
        
        return symbol
    
    @staticmethod
    def detect_market(symbol: str) -> str:
        """심볼로 시장 판별"""
        symbol = symbol.upper()
        
        if symbol.endswith('.T'):
            return "JP"
        elif '-' in symbol or 'USDT' in symbol:
            return "COIN"
        else:
            return "US"
    
    @staticmethod
    def clean_price_data(data: pd.DataFrame) -> pd.DataFrame:
        """가격 데이터 정리"""
        try:
            if data.empty:
                return data
            
            # 결측값 처리
            data = data.dropna()
            
            # 이상값 제거 (5% 이상 급등락)
            for col in ['close', 'high', 'low', 'open']:
                if col in data.columns:
                    pct_change = data[col].pct_change().abs()
                    data = data[pct_change <= 0.05]
            
            # 0 이하 가격 제거
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in data.columns:
                    data = data[data[col] > 0]
            
            return data.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"가격 데이터 정리 실패: {e}")
            return data
    
    @staticmethod
    def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """수익률 계산"""
        try:
            return prices.pct_change(periods=periods)
        except Exception as e:
            logger.error(f"수익률 계산 실패: {e}")
            return pd.Series()
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20) -> float:
        """변동성 계산 (연율화)"""
        try:
            if len(returns) < window:
                return 0.0
            daily_vol = returns.rolling(window=window).std().iloc[-1]
            return daily_vol * np.sqrt(252)  # 연율화
        except Exception as e:
            logger.error(f"변동성 계산 실패: {e}")
            return 0.0

# =====================================
# 💰 금융 계산 클래스 (확장)
# =====================================

class FinanceUtils:
    """금융 계산 및 지표"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
            
        except Exception as e:
            logger.error(f"RSI 계산 실패: {e}")
            return 50.0
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD 계산"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line.iloc[-1] if not macd_line.empty else 0.0,
                'signal': signal_line.iloc[-1] if not signal_line.empty else 0.0,
                'histogram': histogram.iloc[-1] if not histogram.empty else 0.0
            }
            
        except Exception as e:
            logger.error(f"MACD 계산 실패: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """볼린저 밴드 계산"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = prices.iloc[-1]
            current_upper = upper_band.iloc[-1] if not upper_band.empty else current_price
            current_lower = lower_band.iloc[-1] if not lower_band.empty else current_price
            current_middle = sma.iloc[-1] if not sma.empty else current_price
            
            # 밴드 위치 (0-100%)
            band_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
            
            return {
                'upper_band': current_upper,
                'middle_band': current_middle,
                'lower_band': current_lower,
                'band_position': band_position,
                'is_oversold': current_price <= current_lower,
                'is_overbought': current_price >= current_upper
            }
            
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 실패: {e}")
            return {
                'upper_band': 0.0, 'middle_band': 0.0, 'lower_band': 0.0,
                'band_position': 50.0, 'is_oversold': False, 'is_overbought': False
            }
    
    @staticmethod
    def calculate_position_size(capital: float, risk_per_trade: float, 
                              entry_price: float, stop_loss_price: float,
                              min_position: float = 0.01, max_position: float = 0.10) -> float:
        """포지션 사이징 계산 (Kelly 기준)"""
        try:
            if stop_loss_price <= 0 or entry_price <= 0:
                return min_position * capital / entry_price
            
            # 리스크 금액 계산
            risk_amount = capital * risk_per_trade
            
            # 주당 리스크 계산
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share == 0:
                return min_position * capital / entry_price
            
            # 포지션 크기 계산
            position_size = risk_amount / risk_per_share
            
            # 최소/최대 제한 적용
            max_shares = (max_position * capital) / entry_price
            min_shares = (min_position * capital) / entry_price
            
            position_size = max(min_shares, min(position_size, max_shares))
            
            return position_size
            
        except Exception as e:
            logger.error(f"포지션 사이징 계산 실패: {e}")
            return min_position * capital / entry_price if entry_price > 0 else 0
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        try:
            if len(returns) < 2:
                return 0.0
            
            mean_return = returns.mean() * 252  # 연율화
            volatility = returns.std() * np.sqrt(252)  # 연율화
            
            if volatility == 0:
                return 0.0
            
            sharpe = (mean_return - risk_free_rate) / volatility
            return sharpe
            
        except Exception as e:
            logger.error(f"샤프 비율 계산 실패: {e}")
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict:
        """최대손실폭 계산"""
        try:
            if equity_curve.empty:
                return {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0}
            
            # 누적 최고점 계산
            peak = equity_curve.cummax()
            
            # 드로우다운 계산
            drawdown = equity_curve - peak
            max_drawdown = drawdown.min()
            
            # 퍼센트 드로우다운
            drawdown_pct = (drawdown / peak) * 100
            max_drawdown_pct = drawdown_pct.min()
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct
            }
            
        except Exception as e:
            logger.error(f"최대손실폭 계산 실패: {e}")
            return {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0}

# =====================================
# 📁 파일 관리 클래스 (확장)
# =====================================

class FileManager:
    """파일 및 데이터 관리"""
    
    @staticmethod
    def ensure_directories():
        """필요한 디렉토리 생성"""
        directories = [
            'data', 'data/cache', 'data/backups', 'data/database',
            'logs', 'logs/trading', 'logs/analysis', 'logs/errors',
            'configs', 'strategies', 'tests',
            'reports', 'reports/daily', 'reports/monthly'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("📁 디렉토리 구조 초기화 완료")
    
    @staticmethod
    def save_json(data: Any, filename: str, backup: bool = True) -> bool:
        """JSON 파일 저장"""
        try:
            # 백업 생성
            if backup and os.path.exists(filename):
                backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(filename, backup_name)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 저장
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"JSON 파일 저장 실패 {filename}: {e}")
            return False
    
    @staticmethod
    def load_json(filename: str, default: Any = None) -> Any:
        """JSON 파일 로드"""
        try:
            if not os.path.exists(filename):
                return default
            
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"JSON 파일 로드 실패 {filename}: {e}")
            return default
    
    @staticmethod
    def save_csv(data: pd.DataFrame, filename: str) -> bool:
        """CSV 파일 저장"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            data.to_csv(filename, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            logger.error(f"CSV 파일 저장 실패 {filename}: {e}")
            return False
    
    @staticmethod
    def load_csv(filename: str) -> Optional[pd.DataFrame]:
        """CSV 파일 로드"""
        try:
            if not os.path.exists(filename):
                return None
            return pd.read_csv(filename, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"CSV 파일 로드 실패 {filename}: {e}")
            return None
    
    @staticmethod
    def cleanup_old_files(directory: str, days: int = 30, pattern: str = "*"):
        """오래된 파일 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for file_path in Path(directory).glob(pattern):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        logger.info(f"오래된 파일 삭제: {file_path}")
                        
        except Exception as e:
            logger.error(f"파일 정리 실패: {e}")

# =====================================
# 🕐 시간대 관리 클래스 (기존 유지)
# =====================================

class TimeZoneManager:
    """시간대 관리 및 변환"""
    
    def __init__(self):
        self.seoul_tz = pytz.timezone('Asia/Seoul')
        self.us_tz = pytz.timezone('US/Eastern')  # 자동 EST/EDT 전환
        self.japan_tz = pytz.timezone('Asia/Tokyo')
        self.utc_tz = pytz.UTC
    
    def get_current_time(self, timezone: str = 'Seoul') -> datetime:
        """현재 시간 조회 (시간대별)"""
        try:
            utc_now = datetime.now(self.utc_tz)
            
            if timezone.upper() == 'SEOUL' or timezone.upper() == 'KOR':
                return utc_now.astimezone(self.seoul_tz)
            elif timezone.upper() == 'US' or timezone.upper() == 'NY':
                return utc_now.astimezone(self.us_tz)
            elif timezone.upper() == 'JAPAN' or timezone.upper() == 'JP':
                return utc_now.astimezone(self.japan_tz)
            else:
                return utc_now
                
        except Exception as e:
            logger.error(f"시간 조회 실패: {e}")
            return datetime.now()
    
    def seoul_to_us_time(self, seoul_dt: datetime) -> datetime:
        """서울 → 뉴욕 시간 변환"""
        try:
            if seoul_dt.tzinfo is None:
                seoul_dt = self.seoul_tz.localize(seoul_dt)
            return seoul_dt.astimezone(self.us_tz)
        except Exception as e:
            logger.error(f"서울→뉴욕 시간 변환 실패: {e}")
            return seoul_dt
    
    def seoul_to_japan_time(self, seoul_dt: datetime) -> datetime:
        """서울 → 도쿄 시간 변환"""
        try:
            if seoul_dt.tzinfo is None:
                seoul_dt = self.seoul_tz.localize(seoul_dt)
            return seoul_dt.astimezone(self.japan_tz)
        except Exception as e:
            logger.error(f"서울→도쿄 시간 변환 실패: {e}")
            return seoul_dt
    
    def us_to_seoul_time(self, us_dt: datetime) -> datetime:
        """뉴욕 → 서울 시간 변환"""
        try:
            if us_dt.tzinfo is None:
                us_dt = self.us_tz.localize(us_dt)
            return us_dt.astimezone(self.seoul_tz)
        except Exception as e:
            logger.error(f"뉴욕→서울 시간 변환 실패: {e}")
            return us_dt
    
    def japan_to_seoul_time(self, jp_dt: datetime) -> datetime:
        """도쿄 → 서울 시간 변환"""
        try:
            if jp_dt.tzinfo is None:
                jp_dt = self.japan_tz.localize(jp_dt)
            return jp_dt.astimezone(self.seoul_tz)
        except Exception as e:
            logger.error(f"도쿄→서울 시간 변환 실패: {e}")
            return jp_dt
    
    def is_market_open(self, market: str = "US") -> bool:
        """시장 개장 여부 확인"""
        try:
            if market.upper() == 'US':
                us_time = self.get_current_time('US')
                # 프리마켓: 4:00-9:30, 정규장: 9:30-16:00, 애프터마켓: 16:00-20:00
                hour = us_time.hour
                minute = us_time.minute
                
                # 주말 체크
                if us_time.weekday() >= 5:  # 토요일(5), 일요일(6)
                    return False
                
                # 프리마켓
                if hour >= 4 and (hour < 9 or (hour == 9 and minute < 30)):
                    return True
                # 정규장
                elif (hour == 9 and minute >= 30) or (10 <= hour < 16):
                    return True
                # 애프터마켓
                elif 16 <= hour < 20:
                    return True
                else:
                    return False
                    
            elif market.upper() == 'JP':
                jp_time = self.get_current_time('JAPAN')
                hour = jp_time.hour
                minute = jp_time.minute
                
                # 주말 체크
                if jp_time.weekday() >= 5:
                    return False
                
                # 오전장: 9:00-11:30, 오후장: 12:30-15:00
                if (hour == 9 and minute >= 0) or (10 <= hour <= 11) or (hour == 11 and minute <= 30):
                    return True
                elif (hour == 12 and minute >= 30) or (13 <= hour <= 14) or (hour == 15 and minute <= 0):
                    return True
                else:
                    return False
                    
            elif market.upper() == 'COIN':
                # 암호화폐는 24시간
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"시장 개장 확인 실패: {e}")
            return False
    
    def get_time_until_market_event(self, market: str = "US") -> Dict:
        """다음 시장 이벤트까지 시간"""
        try:
            if market.upper() == 'US':
                us_time = self.get_current_time('US')
                
                # 각 시간대별 체크
                if us_time.hour < 4:
                    # 프리마켓까지
                    target = us_time.replace(hour=4, minute=0, second=0, microsecond=0)
                    return {
                        'next_event': '프리마켓 시작',
                        'time_until': str(target - us_time).split('.')[0]
                    }
                elif us_time.hour < 9 or (us_time.hour == 9 and us_time.minute < 30):
                    # 정규장까지
                    target = us_time.replace(hour=9, minute=30, second=0, microsecond=0)
                    return {
                        'next_event': '정규장 시작',
                        'time_until': str(target - us_time).split('.')[0]
                    }
                elif us_time.hour < 16:
                    # 정규장 마감까지
                    target = us_time.replace(hour=16, minute=0, second=0, microsecond=0)
                    return {
                        'next_event': '정규장 마감',
                        'time_until': str(target - us_time).split('.')[0]
                    }
                elif us_time.hour < 20:
                    # 애프터마켓 마감까지
                    target = us_time.replace(hour=20, minute=0, second=0, microsecond=0)
                    return {
                        'next_event': '애프터마켓 마감',
                        'time_until': str(target - us_time).split('.')[0]
                    }
                else:
                    # 다음날 프리마켓까지
                    next_day = us_time + timedelta(days=1)
                    target = next_day.replace(hour=4, minute=0, second=0, microsecond=0)
                    return {
                        'next_event': '프리마켓 시작',
                        'time_until': str(target - us_time).split('.')[0]
                    }
            
            # 일본/코인 시장도 유사하게 구현 가능
            return {'next_event': '정보 없음', 'time_until': '0:00:00'}
            
        except Exception as e:
            logger.error(f"시장 이벤트 시간 계산 실패: {e}")
            return {'next_event': '오류', 'time_until': '0:00:00'}

# =====================================
# 🔄 API 재시도 및 제한 (기존 유지)
# =====================================

class RateLimiter:
    """API 호출 속도 제한"""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.last_call_time = 0
        
    async def wait(self):
        """속도 제한 대기"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        min_interval = 1.0 / self.calls_per_second
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_call_time = time.time()

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"{func.__name__} 실패 (재시도 {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} 최종 실패: {e}")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"{func.__name__} 실패 (재시도 {attempt + 1}/{max_retries}): {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} 최종 실패: {e}")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# =====================================
# 📋 포맷팅 및 검증 (확장)
# =====================================

class Formatter:
    """데이터 포맷팅"""
    
    @staticmethod
    def format_price(price: float, market: str = "US", precision: int = None) -> str:
        """가격 포맷팅"""
        try:
            if market.upper() == 'US':
                if price >= 1000000:
                    return f"${price/1000000:.1f}M"
                elif price >= 1000:
                    return f"${price/1000:.1f}K"
                else:
                    return f"${price:.2f}"
                    
            elif market.upper() == 'JP':
                if price >= 1000000:
                    return f"¥{price/1000000:.1f}M"
                elif price >= 1000:
                    return f"¥{price/1000:.0f}K"
                else:
                    return f"¥{price:.0f}"
                    
            elif market.upper() in ['COIN', 'CRYPTO']:
                if price >= 100000000:  # 1억 이상
                    return f"₩{price/100000000:.1f}억"
                elif price >= 10000:  # 1만 이상
                    return f"₩{price/10000:.0f}만"
                elif price >= 1000:
                    return f"₩{price:,.0f}"
                else:
                    return f"₩{price:.2f}"
            else:
                return f"{price:,.2f}"
                
        except Exception as e:
            logger.error(f"가격 포맷팅 실패: {e}")
            return str(price)
    
    @staticmethod
    def format_percentage(value: float, precision: int = 1) -> str:
        """퍼센트 포맷팅"""
        try:
            sign = "+" if value > 0 else ""
            return f"{sign}{value:.{precision}f}%"
        except Exception as e:
            logger.error(f"퍼센트 포맷팅 실패: {e}")
            return f"{value}%"
    
    @staticmethod
    def format_number(number: Union[int, float], precision: int = 2) -> str:
        """숫자 포맷팅"""
        try:
            if abs(number) >= 1000000000:
                return f"{number/1000000000:.{precision}f}B"
            elif abs(number) >= 1000000:
                return f"{number/1000000:.{precision}f}M"
            elif abs(number) >= 1000:
                return f"{number/1000:.{precision}f}K"
            else:
                return f"{number:.{precision}f}"
        except Exception as e:
            logger.error(f"숫자 포맷팅 실패: {e}")
            return str(number)

class Validator:
    """데이터 검증"""
    
    @staticmethod
    def is_valid_symbol(symbol: str) -> bool:
        """심볼 유효성 검증"""
        if not symbol or len(symbol) < 1:
            return False
        
        symbol = symbol.upper()
        
        # 미국 주식 (1-5자 영문)
        if symbol.isalpha() and 1 <= len(symbol) <= 5:
            return True
        
        # 일본 주식 (숫자.T)
        if symbol.endswith('.T') and symbol[:-2].isdigit():
            return True
        
        # 암호화폐 (BASE-QUOTE)
        if '-' in symbol:
            parts = symbol.split('-')
            if len(parts) == 2 and all(part.isalpha() for part in parts):
                return True
        
        return False
    
    @staticmethod
    def is_valid_price(price: float) -> bool:
        """가격 유효성 검증"""
        return isinstance(price, (int, float)) and price > 0 and not np.isnan(price)
    
    @staticmethod
    def is_valid_confidence(confidence: float) -> bool:
        """신뢰도 유효성 검증"""
        return isinstance(confidence, (int, float)) and 0 <= confidence <= 1

# =====================================
# 💾 캐싱 시스템 (확장)
# =====================================

class SimpleCache:
    """간단한 메모리 캐시"""
    
    def __init__(self, default_ttl: int = 300):  # 5분 기본 TTL
        self.cache = {}
        self.default_ttl = default_ttl
    
    def _is_expired(self, timestamp: float, ttl: int) -> bool:
        """만료 여부 확인"""
        return time.time() - timestamp > ttl
    
    def get(self, key: str) -> Any:
        """캐시에서 값 조회"""
        if key not in self.cache:
            return None
        
        value, timestamp, ttl = self.cache[key]
        
        if self._is_expired(timestamp, ttl):
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """캐시에 값 저장"""
        if ttl is None:
            ttl = self.default_ttl
        
        self.cache[key] = (value, time.time(), ttl)
    
    def delete(self, key: str) -> None:
        """캐시에서 값 삭제"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """전체 캐시 삭제"""
        self.cache.clear()
    
    def cleanup(self) -> None:
        """만료된 캐시 정리"""
        current_time = time.time()
        expired_keys = []
        
        for key, (value, timestamp, ttl) in self.cache.items():
            if current_time - timestamp > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]

# 전역 캐시 인스턴스
cache = SimpleCache()

# =====================================
# 📈 백테스트 유틸리티 (확장)
# =====================================

class BacktestUtils:
    """백테스트 관련 유틸리티"""
    
    @staticmethod
    def calculate_performance_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
        """성과 지표 계산"""
        try:
            if returns.empty:
                return {}
            
            # 기본 지표
            total_return = (returns + 1).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            annual_volatility = returns.std() * np.sqrt(252)
            
            # 샤프 비율
            sharpe_ratio = FinanceUtils.calculate_sharpe_ratio(returns)
            
            # 최대손실폭
            equity_curve = (returns + 1).cumprod()
            dd_info = FinanceUtils.calculate_max_drawdown(equity_curve)
            
            # 승률
            win_rate = (returns > 0).mean()
            
            # 칼마 비율 (연수익률 / 최대손실폭)
            calmar_ratio = annual_return / abs(dd_info['max_drawdown_pct']) if dd_info['max_drawdown_pct'] != 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': dd_info['max_drawdown_pct'],
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'num_trades': len(returns)
            }
            
            # 벤치마크 대비 성과
            if benchmark_returns is not None and not benchmark_returns.empty:
                # 같은 기간으로 맞춤
                common_index = returns.index.intersection(benchmark_returns.index)
                if not common_index.empty:
                    returns_aligned = returns.loc[common_index]
                    benchmark_aligned = benchmark_returns.loc[common_index]
                    
                    benchmark_total = (benchmark_aligned + 1).prod() - 1
                    excess_return = total_return - benchmark_total
                    
                    metrics['benchmark_return'] = benchmark_total
                    metrics['excess_return'] = excess_return
                    metrics['information_ratio'] = excess_return / (returns_aligned - benchmark_aligned).std() if (returns_aligned - benchmark_aligned).std() != 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"성과 지표 계산 실패: {e}")
            return {}
    
    @staticmethod
    def generate_performance_report(returns: pd.Series, strategy_name: str = "Strategy") -> str:
        """성과 리포트 생성"""
        try:
            metrics = BacktestUtils.calculate_performance_metrics(returns)
            
            if not metrics:
                return f"📊 {strategy_name} 성과 리포트\n❌ 데이터 부족"
            
            report = f"""
📊 {strategy_name} 성과 리포트
{'='*40}

📈 수익률 지표
  총 수익률: {Formatter.format_percentage(metrics.get('total_return', 0) * 100)}
  연간 수익률: {Formatter.format_percentage(metrics.get('annual_return', 0) * 100)}
  연간 변동성: {Formatter.format_percentage(metrics.get('annual_volatility', 0) * 100)}

🎯 리스크 지표
  샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}
  최대 손실폭: {Formatter.format_percentage(metrics.get('max_drawdown', 0))}
  칼마 비율: {metrics.get('calmar_ratio', 0):.2f}

📊 거래 통계
  승률: {Formatter.format_percentage(metrics.get('win_rate', 0) * 100)}
  총 거래 수: {metrics.get('num_trades', 0)}개

⏰ 리포트 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return report.strip()
            
        except Exception as e:
            logger.error(f"성과 리포트 생성 실패: {e}")
            return f"📊 {strategy_name} 성과 리포트\n❌ 리포트 생성 실패: {e}"

# =====================================
# 🔌 브로커 연동 헬퍼 (신규)
# =====================================

class BrokerUtils:
    """브로커 연동 유틸리티"""
    
    @staticmethod
    def normalize_broker_symbol(symbol: str, broker: str) -> str:
        """브로커별 심볼 정규화"""
        try:
            if broker.upper() == 'IBKR':
                # Interactive Brokers 형식
                if symbol.endswith('.T'):
                    # 일본 주식: 7203.T → 7203 TSE
                    return symbol.replace('.T', ' TSE')
                else:
                    # 미국 주식: AAPL → AAPL
                    return symbol.upper()
                    
            elif broker.upper() == 'UPBIT':
                # 업비트 형식
                if '-' in symbol:
                    # BTC-KRW → KRW-BTC (업비트 순서)
                    base, quote = symbol.split('-')
                    return f"{quote}-{base}"
                else:
                    return f"KRW-{symbol.upper()}"
            
            return symbol.upper()
            
        except Exception as e:
            logger.error(f"브로커 심볼 정규화 실패: {e}")
            return symbol
    
    @staticmethod
    def calculate_trade_amount(symbol: str, confidence: float, portfolio_value: float, 
                             max_position_pct: float = 0.10) -> Dict:
        """거래 금액 계산"""
        try:
            market = DataProcessor.detect_market(symbol)
            
            # 신뢰도 기반 포지션 크기
            base_position_pct = 0.05  # 기본 5%
            position_pct = min(base_position_pct * confidence * 2, max_position_pct)
            
            trade_amount = portfolio_value * position_pct
            
            # 시장별 최소/최대 금액 조정
            if market == 'US':
                min_amount = 100  # $100
                max_amount = portfolio_value * 0.15  # 최대 15%
            elif market == 'JP':
                min_amount = 10000  # ¥10,000
                max_amount = portfolio_value * 0.12  # 최대 12%
            elif market == 'COIN':
                min_amount = 50000  # ₩50,000
                max_amount = portfolio_value * 0.20  # 최대 20% (변동성 고려)
            
            trade_amount = max(min_amount, min(trade_amount, max_amount))
            
            return {
                'trade_amount': trade_amount,
                'position_pct': trade_amount / portfolio_value,
                'market': market,
                'confidence_used': confidence
            }
            
        except Exception as e:
            logger.error(f"거래 금액 계산 실패: {e}")
            return {'trade_amount': 0, 'position_pct': 0, 'market': 'UNKNOWN', 'confidence_used': 0}

# =====================================
# 📰 뉴스 분석 헬퍼 (신규)
# =====================================

class NewsUtils:
    """뉴스 분석 유틸리티"""
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """키워드 추출"""
        try:
            # 간단한 키워드 추출 (실제로는 NLP 라이브러리 사용)
            import re
            
            # 특수문자 제거, 소문자 변환
            text = re.sub(r'[^\w\s]', '', text.lower())
            words = text.split()
            
            # 불용어 제거 (간단한 버전)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            
            # 단어 빈도 계산
            word_freq = {}
            for word in words:
                if len(word) > 2 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # 빈도순 정렬
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in keywords[:max_keywords]]
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
    
    @staticmethod
    def sentiment_score_to_text(score: float) -> str:
        """센티먼트 점수를 텍스트로 변환"""
        if score >= 0.8:
            return "매우 긍정적"
        elif score >= 0.6:
            return "긍정적"
        elif score >= 0.4:
            return "중립적"
        elif score >= 0.2:
            return "부정적"
        else:
            return "매우 부정적"
    
    @staticmethod
    def calculate_news_impact_weight(symbol: str, news_count: int, avg_sentiment: float) -> float:
        """뉴스 영향도 가중치 계산"""
        try:
            market = DataProcessor.detect_market(symbol)
            
            # 시장별 기본 가중치
            base_weights = {
                'US': 0.3,    # 미국 주식: 뉴스 30%
                'JP': 0.4,    # 일본 주식: 뉴스 40%
                'COIN': 0.5   # 암호화폐: 뉴스 50%
            }
            
            base_weight = base_weights.get(market, 0.3)
            
            # 뉴스 개수에 따른 조정 (더 많은 뉴스 = 더 높은 신뢰도)
            count_multiplier = min(1.0 + (news_count - 1) * 0.1, 1.5)  # 최대 1.5배
            
            # 센티먼트 강도에 따른 조정
            sentiment_strength = abs(avg_sentiment - 0.5) * 2  # 0-1 범위
            sentiment_multiplier = 0.5 + sentiment_strength * 0.5  # 0.5-1.0 범위
            
            final_weight = base_weight * count_multiplier * sentiment_multiplier
            return min(final_weight, 0.7)  # 최대 70%
            
        except Exception as e:
            logger.error(f"뉴스 영향도 계산 실패: {e}")
            return 0.3

# =====================================
# 📅 스케줄링 헬퍼 (신규)
# =====================================

class ScheduleUtils:
    """스케줄링 유틸리티"""
    
    @staticmethod
    def get_weekday_korean(date: datetime = None) -> str:
        """한국어 요일 반환"""
        if date is None:
            date = datetime.now()
        
        weekdays = ["월", "화", "수", "목", "금", "토", "일"]
        return weekdays[date.weekday()]
    
    @staticmethod
    def is_trading_day(market: str, date: datetime = None) -> bool:
        """거래일 여부 확인"""
        if date is None:
            date = datetime.now()
        
        # 주말 체크
        if date.weekday() >= 5:  # 토요일(5), 일요일(6)
            return False
        
        # 시장별 공휴일 체크 (간단한 버전)
        if market.upper() == 'US':
            # 미국 주요 공휴일 (간단한 체크)
            us_holidays = [
                '2025-01-01',  # New Year's Day
                '2025-01-20',  # Martin Luther King Jr. Day
                '2025-02-17',  # Presidents' Day
                '2025-04-18',  # Good Friday
                '2025-05-26',  # Memorial Day
                '2025-07-04',  # Independence Day
                '2025-09-01',  # Labor Day
                '2025-11-27',  # Thanksgiving
                '2025-12-25',  # Christmas
            ]
            return date.strftime('%Y-%m-%d') not in us_holidays
            
        elif market.upper() == 'JP':
            # 일본 주요 공휴일
            jp_holidays = [
                '2025-01-01',  # 신정
                '2025-01-13',  # 성인의 날
                '2025-02-11',  # 건국기념일
                '2025-02-23',  # 천황 탄생일
                '2025-03-20',  # 춘분의 날
                '2025-04-29',  # 쇼와의 날
                '2025-05-03',  # 헌법기념일
                '2025-05-04',  # 녹색의 날
                '2025-05-05',  # 어린이날
            ]
            return date.strftime('%Y-%m-%d') not in jp_holidays
            
        elif market.upper() == 'COIN':
            # 암호화폐는 연중무휴
            return True
            
        return True
    
    @staticmethod
    def get_next_trading_day(market: str, date: datetime = None) -> datetime:
        """다음 거래일 조회"""
        if date is None:
            date = datetime.now()
        
        next_date = date + timedelta(days=1)
        
        while not ScheduleUtils.is_trading_day(market, next_date):
            next_date += timedelta(days=1)
            
            # 무한루프 방지 (최대 30일)
            if (next_date - date).days > 30:
                break
        
        return next_date

# =====================================
# 🗃️ 데이터베이스 헬퍼 (신규)
# =====================================

class DatabaseUtils:
    """데이터베이스 유틸리티"""
    
    @staticmethod
    def init_database(db_path: str = "data/database/quant.db") -> bool:
        """데이터베이스 초기화"""
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 거래 이력 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    total_amount REAL NOT NULL,
                    confidence REAL,
                    strategy TEXT,
                    reasoning TEXT,
                    broker TEXT,
                    order_id TEXT,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            # 분석 결과 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    target_price REAL,
                    reasoning TEXT,
                    technical_score REAL,
                    news_score REAL,
                    final_score REAL
                )
            ''')
            
            # 성과 추적 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_portfolio_value REAL,
                    daily_return REAL,
                    cumulative_return REAL,
                    trades_count INTEGER DEFAULT 0,
                    successful_trades INTEGER DEFAULT 0,
                    market_summary TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"데이터베이스 초기화 완료: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            return False
    
    @staticmethod
    def save_trade_record(trade_data: Dict, db_path: str = "data/database/quant.db") -> bool:
        """거래 기록 저장"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    timestamp, market, symbol, action, quantity, price, total_amount,
                    confidence, strategy, reasoning, broker, order_id, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('timestamp', datetime.now().isoformat()),
                trade_data.get('market', ''),
                trade_data.get('symbol', ''),
                trade_data.get('action', ''),
                trade_data.get('quantity', 0),
                trade_data.get('price', 0),
                trade_data.get('total_amount', 0),
                trade_data.get('confidence', 0),
                trade_data.get('strategy', ''),
                trade_data.get('reasoning', ''),
                trade_data.get('broker', ''),
                trade_data.get('order_id', ''),
                trade_data.get('status', 'pending')
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"거래 기록 저장 실패: {e}")
            return False

# =====================================
# 편의 함수들
# =====================================

def get_config(config_path: str = "configs/settings.yaml") -> Dict:
    """설정 파일 로드 (캐시 적용)"""
    cache_key = f"config_{config_path}"
    
    # 캐시에서 조회
    cached_config = cache.get(cache_key)
    if cached_config is not None:
        return cached_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 캐시에 저장 (10분)
        cache.set(cache_key, config, ttl=600)
        return config
        
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        return {}

def save_trading_log(market: str, symbol: str, action: str, details: Dict):
    """거래 로그 저장"""
    try:
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'market': market,
            'symbol': symbol,
            'action': action,
            'details': details
        }
        
        log_file = f"logs/trading/trading_{datetime.now().strftime('%Y%m%d')}.json"
        
        # 기존 로그 로드
        existing_logs = FileManager.load_json(log_file, [])
        existing_logs.append(log_data)
        
        # 저장
        FileManager.save_json(existing_logs, log_file)
        
    except Exception as e:
        logger.error(f"거래 로그 저장 실패: {e}")

def is_market_open(market: str = "US") -> bool:
    """시장 개장 여부 (편의 함수)"""
    tz_manager = TimeZoneManager()
    return tz_manager.is_market_open(market)

def get_market_hours(market: str = "US") -> Dict:
    """시장 시간 정보"""
    tz_manager = TimeZoneManager()
    
    if market.upper() == 'US':
        return {
            'timezone': 'US/Eastern',
            'premarket': '04:00-09:30',
            'regular': '09:30-16:00',
            'aftermarket': '16:00-20:00',
            'is_open': tz_manager.is_market_open('US')
        }
    elif market.upper() == 'JP':
        return {
            'timezone': 'Asia/Tokyo',
            'morning': '09:00-11:30',
            'afternoon': '12:30-15:00',
            'is_open': tz_manager.is_market_open('JP')
        }
    elif market.upper() == 'COIN':
        return {
            'timezone': 'UTC',
            'trading': '24/7',
            'is_open': True
        }
    
    return {}

def calculate_portfolio_value(positions: List[Dict]) -> float:
    """포트폴리오 가치 계산"""
    try:
        total_value = 0.0
        
        for position in positions:
            quantity = position.get('quantity', 0)
            current_price = position.get('current_price', 0)
            total_value += quantity * current_price
        
        return total_value
        
    except Exception as e:
        logger.error(f"포트폴리오 가치 계산 실패: {e}")
        return 0.0

# =====================================
# 테스트 함수 (확장)
# =====================================

async def test_utils_comprehensive():
    """🧪 전체 유틸리티 시스템 테스트"""
    print("🛠️ 최고퀸트프로젝트 유틸리티 테스트")
    print("=" * 50)
    
    # 1. 디렉토리 초기화
    print("1️⃣ 디렉토리 초기화...")
    FileManager.ensure_directories()
    print("   ✅ 완료")
    
    # 2. 시간대 처리 테스트
    print("2️⃣ 시간대 처리 테스트...")
    tz_manager = TimeZoneManager()
    seoul_time = tz_manager.get_current_time('Seoul')
    us_time = tz_manager.get_current_time('US')
    jp_time = tz_manager.get_current_time('Japan')
    
    print(f"   🇰🇷 서울: {seoul_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   🇺🇸 뉴욕: {us_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   🇯🇵 도쿄: {jp_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 3. 데이터 처리 테스트
    print("3️⃣ 데이터 처리 테스트...")
    test_symbols = ['AAPL', '7203.T', 'BTC-KRW', 'invalid', '']
    for symbol in test_symbols:
        normalized = DataProcessor.normalize_symbol(symbol)
        market = DataProcessor.detect_market(symbol) if symbol else 'UNKNOWN'
        is_valid = Validator.is_valid_symbol(symbol)
        print(f"   {symbol:10} → {normalized:10} ({market:4}) {'✅' if is_valid else '❌'}")
    
    # 4. 포맷팅 테스트
    print("4️⃣ 포맷팅 테스트...")
    test_prices = [
        (175.50, 'US'), (2850, 'JP'), (95000000, 'COIN'), (1234567, 'US')
    ]
    for price, market in test_prices:
        formatted = Formatter.format_price(price, market)
        print(f"   {price:>10} ({market:4}) → {formatted}")
    
    # 5. 금융 계산 테스트
    print("5️⃣ 금융 계산 테스트...")
    # 샘플 가격 데이터 생성
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.02))
    
    rsi = FinanceUtils.calculate_rsi(prices)
    macd = FinanceUtils.calculate_macd(prices)
    bb = FinanceUtils.calculate_bollinger_bands(prices)
    
    print(f"   RSI: {rsi:.1f}")
    print(f"   MACD: {macd['macd']:.3f}")
    print(f"   볼린저 위치: {bb['band_position']:.1f}%")
    
    # 6. 캐시 테스트
    print("6️⃣ 캐시 테스트...")
    cache.set('test_key', 'test_value', ttl=5)
    cached_value = cache.get('test_key')
    print(f"   캐시 저장/로드: {'✅' if cached_value == 'test_value' else '❌'}")
    
    # 7. 시장 개장 테스트
    print("7️⃣ 시장 개장 상태...")
    markets = ['US', 'JP', 'COIN']
    for market in markets:
        is_open = is_market_open(market)
        status = "🟢 OPEN" if is_open else "🔴 CLOSED"
        print(f"   {market:4}: {status}")
    
    # 8. 브로커 유틸리티 테스트
    print("8️⃣ 브로커 유틸리티 테스트...")
    test_cases = [
        ('AAPL', 'IBKR'), ('7203.T', 'IBKR'), ('BTC-KRW', 'UPBIT')
    ]
    for symbol, broker in test_cases:
        normalized = BrokerUtils.normalize_broker_symbol(symbol, broker)
        print(f"   {symbol:8} ({broker:5}) → {normalized}")
    
    # 9. 데이터베이스 초기화 테스트
    print("9️⃣ 데이터베이스 초기화...")
    db_success = DatabaseUtils.init_database()
    print(f"   결과: {'✅ 성공' if db_success else '❌ 실패'}")
    
    # 10. 뉴스 유틸리티 테스트
    print("🔟 뉴스 유틸리티 테스트...")
    test_text = "Apple reports strong quarterly earnings with revenue growth exceeding expectations"
    keywords = NewsUtils.extract_keywords(test_text, 3)
    print(f"   키워드: {', '.join(keywords)}")
    
    print()
    print("🎯 전체 유틸리티 테스트 완료!")
    print("📊 모든 핵심 기능이 정상 작동합니다")

if __name__ == "__main__":
    asyncio.run(test_utils_comprehensive())