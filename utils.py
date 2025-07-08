#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 퀸트프로젝트 유틸리티 모듈 (utils.py)
==============================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 공통 유틸리티

✨ 핵심 기능:
- 데이터 전처리 및 변환
- 기술지표 계산
- 리스크 관리 함수
- 시간 및 날짜 유틸리티
- 환율 및 화폐 변환
- 성과 분석 도구
- 로깅 및 알림 헬퍼
- 파일 I/O 유틸리티

Author: 퀸트마스터팀
Version: 1.1.0 (통합 유틸리티)
"""

import asyncio
import logging
import json
import csv
import os
import sqlite3
import hashlib
import time
import math
import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import aiohttp
import pandas as pd
import numpy as np
import pytz
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 🎯 상수 및 설정
# ============================================================================
class Currency(Enum):
    """지원 통화"""
    USD = "USD"
    KRW = "KRW"
    JPY = "JPY"
    INR = "INR"
    BTC = "BTC"
    ETH = "ETH"

class Market(Enum):
    """지원 시장"""
    US_STOCK = "US_STOCK"
    KOREA_STOCK = "KOREA_STOCK"
    JAPAN_STOCK = "JAPAN_STOCK"
    INDIA_STOCK = "INDIA_STOCK"
    CRYPTO = "CRYPTO"

class TimeZones:
    """시간대 상수"""
    UTC = pytz.UTC
    SEOUL = pytz.timezone('Asia/Seoul')
    TOKYO = pytz.timezone('Asia/Tokyo')
    NEW_YORK = pytz.timezone('America/New_York')
    MUMBAI = pytz.timezone('Asia/Kolkata')

# ============================================================================
# 📊 데이터 처리 유틸리티
# ============================================================================
class DataProcessor:
    """데이터 전처리 및 변환"""
    
    @staticmethod
    def clean_numeric_data(data: Union[str, int, float], default: float = 0.0) -> float:
        """숫자 데이터 정리"""
        try:
            if data is None or data == '':
                return default
            
            if isinstance(data, str):
                # 콤마, 공백 제거
                cleaned = data.replace(',', '').replace(' ', '')
                # 퍼센트 기호 제거
                if '%' in cleaned:
                    cleaned = cleaned.replace('%', '')
                    return float(cleaned) / 100
                
                return float(cleaned)
            
            return float(data)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """안전한 나눗셈"""
        try:
            if denominator == 0 or denominator is None:
                return default
            return numerator / denominator
        except (TypeError, ZeroDivisionError):
            return default
    
    @staticmethod
    def normalize_symbol(symbol: str, market: Market) -> str:
        """심볼명 정규화"""
        if not symbol:
            return ""
        
        symbol = symbol.upper().strip()
        
        if market == Market.KOREA_STOCK:
            # 한국 주식 코드 정규화
            return symbol.replace('A', '').zfill(6)
        elif market == Market.CRYPTO:
            # 암호화폐 심볼 정규화
            if '-' not in symbol and symbol != 'KRW':
                return f"KRW-{symbol}"
        
        return symbol
    
    @staticmethod
    def format_number(num: float, decimal_places: int = 2) -> str:
        """숫자 포맷팅"""
        try:
            if abs(num) >= 1_000_000_000:
                return f"{num/1_000_000_000:.1f}B"
            elif abs(num) >= 1_000_000:
                return f"{num/1_000_000:.1f}M"
            elif abs(num) >= 1_000:
                return f"{num/1_000:.1f}K"
            else:
                return f"{num:,.{decimal_places}f}"
        except:
            return "0"
    
    @staticmethod
    def calculate_percentage(value: float, total: float) -> float:
        """퍼센트 계산"""
        return DataProcessor.safe_divide(value * 100, total, 0.0)

# ============================================================================
# 📈 기술지표 계산
# ============================================================================
class TechnicalIndicators:
    """기술지표 계산 함수들"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """단순이동평균 (Simple Moving Average)"""
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(avg)
        
        return sma_values
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """지수이동평균 (Exponential Moving Average)"""
        if len(prices) < period:
            return []
        
        alpha = 2 / (period + 1)
        ema_values = []
        
        # 첫 번째 EMA는 SMA로 계산
        first_sma = sum(prices[:period]) / period
        ema_values.append(first_sma)
        
        # 나머지 EMA 계산
        for i in range(period, len(prices)):
            ema = alpha * prices[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return []
        
        gains = []
        losses = []
        
        # 가격 변화 계산
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        rsi_values = []
        
        # 첫 번째 RSI 계산
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        # 나머지 RSI 계산
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
        """볼린저 밴드"""
        if len(prices) < period:
            return [], [], []
        
        middle_band = TechnicalIndicators.sma(prices, period)
        upper_band = []
        lower_band = []
        
        for i in range(period - 1, len(prices)):
            subset = prices[i - period + 1:i + 1]
            std = statistics.stdev(subset)
            sma_val = middle_band[i - period + 1]
            
            upper_band.append(sma_val + (std_dev * std))
            lower_band.append(sma_val - (std_dev * std))
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return [], [], []
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        # MACD 라인 계산
        macd_line = []
        start_index = slow - fast
        
        for i in range(len(ema_slow)):
            macd_val = ema_fast[i + start_index] - ema_slow[i]
            macd_line.append(macd_val)
        
        # 시그널 라인 계산
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        
        # 히스토그램 계산
        histogram = []
        signal_start = signal - 1
        
        for i in range(len(signal_line)):
            hist_val = macd_line[i + signal_start] - signal_line[i]
            histogram.append(hist_val)
        
        return macd_line, signal_line, histogram

# ============================================================================
# 💰 리스크 관리 유틸리티
# ============================================================================
class RiskManager:
    """리스크 관리 함수들"""
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_per_trade: float, 
                              entry_price: float, stop_loss_price: float) -> float:
        """포지션 크기 계산"""
        try:
            risk_amount = account_balance * risk_per_trade
            price_diff = abs(entry_price - stop_loss_price)
            
            if price_diff == 0:
                return 0
            
            position_size = risk_amount / price_diff
            return max(0, position_size)
        except:
            return 0
    
    @staticmethod
    def calculate_max_drawdown(returns: List[float]) -> float:
        """최대 낙폭 계산"""
        if not returns:
            return 0
        
        cumulative_returns = [1]
        for ret in returns:
            cumulative_returns.append(cumulative_returns[-1] * (1 + ret))
        
        max_drawdown = 0
        peak = cumulative_returns[0]
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """샤프 지수 계산"""
        if len(returns) < 2:
            return 0
        
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0
        
        excess_return = mean_return - risk_free_rate / 252  # 일일 무위험 수익률
        sharpe = excess_return / std_return * math.sqrt(252)  # 연환산
        
        return sharpe
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
        """VaR (Value at Risk) 계산"""
        if not returns:
            return 0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0
    
    @staticmethod
    def is_position_size_valid(position_size: float, account_balance: float, 
                             max_position_pct: float = 0.1) -> bool:
        """포지션 크기 유효성 검사"""
        max_position_value = account_balance * max_position_pct
        return 0 < position_size <= max_position_value

# ============================================================================
# 🕐 시간 및 날짜 유틸리티
# ============================================================================
class TimeUtils:
    """시간 관련 유틸리티"""
    
    @staticmethod
    def get_current_time(timezone_name: str = 'Asia/Seoul') -> datetime:
        """현재 시간 조회"""
        tz = pytz.timezone(timezone_name)
        return datetime.now(tz)
    
    @staticmethod
    def is_market_open(market: Market, current_time: datetime = None) -> bool:
        """시장 개장 여부 확인"""
        if current_time is None:
            current_time = datetime.now(TimeZones.UTC)
        
        # 현지 시간으로 변환
        if market == Market.US_STOCK:
            local_time = current_time.astimezone(TimeZones.NEW_YORK)
            # 월-금, 9:30-16:00 (EST)
            return (local_time.weekday() < 5 and 
                   9.5 <= local_time.hour + local_time.minute/60 <= 16)
        
        elif market == Market.KOREA_STOCK:
            local_time = current_time.astimezone(TimeZones.SEOUL)
            # 월-금, 9:00-15:30 (KST)
            return (local_time.weekday() < 5 and 
                   9 <= local_time.hour + local_time.minute/60 <= 15.5)
        
        elif market == Market.JAPAN_STOCK:
            local_time = current_time.astimezone(TimeZones.TOKYO)
            # 월-금, 9:00-11:30, 12:30-15:00 (JST)
            time_decimal = local_time.hour + local_time.minute/60
            return (local_time.weekday() < 5 and 
                   ((9 <= time_decimal <= 11.5) or (12.5 <= time_decimal <= 15)))
        
        elif market == Market.INDIA_STOCK:
            local_time = current_time.astimezone(TimeZones.MUMBAI)
            # 월-금, 9:15-15:30 (IST)
            return (local_time.weekday() < 5 and 
                   9.25 <= local_time.hour + local_time.minute/60 <= 15.5)
        
        elif market == Market.CRYPTO:
            # 암호화폐는 24시간 거래
            return True
        
        return False
    
    @staticmethod
    def get_next_trading_day(market: Market, from_date: datetime = None) -> datetime:
        """다음 거래일 조회"""
        if from_date is None:
            from_date = datetime.now()
        
        next_day = from_date + timedelta(days=1)
        
        # 암호화폐는 매일 거래
        if market == Market.CRYPTO:
            return next_day
        
        # 주말 건너뛰기
        while next_day.weekday() >= 5:  # 토요일(5), 일요일(6)
            next_day += timedelta(days=1)
        
        return next_day
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
        """날짜 시간 포맷팅"""
        try:
            return dt.strftime(format_str)
        except:
            return ""
    
    @staticmethod
    def parse_datetime(date_str: str, format_str: str = '%Y-%m-%d %H:%M:%S') -> Optional[datetime]:
        """문자열을 datetime으로 변환"""
        try:
            return datetime.strptime(date_str, format_str)
        except:
            return None

# ============================================================================
# 💱 환율 및 화폐 유틸리티
# ============================================================================
class CurrencyConverter:
    """환율 변환 유틸리티"""
    
    def __init__(self):
        self.exchange_rates = {}
        self.last_update = None
        self.cache_duration = timedelta(hours=1)  # 1시간 캐시
    
    async def get_exchange_rate(self, from_currency: Currency, to_currency: Currency) -> float:
        """환율 조회"""
        if from_currency == to_currency:
            return 1.0
        
        # 캐시 확인
        rate_key = f"{from_currency.value}_{to_currency.value}"
        
        if (self.last_update and 
            datetime.now() - self.last_update < self.cache_duration and
            rate_key in self.exchange_rates):
            return self.exchange_rates[rate_key]
        
        # 환율 API 호출
        try:
            rate = await self._fetch_exchange_rate(from_currency.value, to_currency.value)
            self.exchange_rates[rate_key] = rate
            self.last_update = datetime.now()
            return rate
        except:
            # 기본값 반환
            return self._get_default_rate(from_currency, to_currency)
    
    async def _fetch_exchange_rate(self, from_curr: str, to_curr: str) -> float:
        """외부 API에서 환율 조회"""
        try:
            # 여러 API 시도
            apis = [
                f"https://api.exchangerate-api.com/v4/latest/{from_curr}",
                f"https://api.fixer.io/latest?base={from_curr}",
            ]
            
            for api_url in apis:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(api_url, timeout=5) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'rates' in data and to_curr in data['rates']:
                                    return float(data['rates'][to_curr])
                except:
                    continue
            
            raise Exception("모든 API 실패")
            
        except Exception as e:
            raise Exception(f"환율 조회 실패: {e}")
    
    def _get_default_rate(self, from_currency: Currency, to_currency: Currency) -> float:
        """기본 환율 (백업용)"""
        default_rates = {
            ('USD', 'KRW'): 1200.0,
            ('USD', 'JPY'): 110.0,
            ('USD', 'INR'): 75.0,
            ('KRW', 'USD'): 1/1200.0,
            ('JPY', 'USD'): 1/110.0,
            ('INR', 'USD'): 1/75.0,
        }
        
        key = (from_currency.value, to_currency.value)
        reverse_key = (to_currency.value, from_currency.value)
        
        if key in default_rates:
            return default_rates[key]
        elif reverse_key in default_rates:
            return 1 / default_rates[reverse_key]
        else:
            return 1.0
    
    async def convert_amount(self, amount: float, from_currency: Currency, 
                           to_currency: Currency) -> float:
        """금액 환전"""
        rate = await self.get_exchange_rate(from_currency, to_currency)
        return amount * rate

# ============================================================================
# 📊 성과 분석 도구
# ============================================================================
@dataclass
class PerformanceMetrics:
    """성과 지표 데이터 클래스"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int

class PerformanceAnalyzer:
    """성과 분석기"""
    
    @staticmethod
    def calculate_performance_metrics(trades: List[Dict]) -> PerformanceMetrics:
        """성과 지표 계산"""
        if not trades:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # 수익률 계산
        returns = []
        profits = []
        losses = []
        
        for trade in trades:
            if 'profit_loss' in trade and trade['profit_loss'] is not None:
                pnl = float(trade['profit_loss'])
                
                if 'entry_price' in trade and 'quantity' in trade:
                    entry_value = float(trade['entry_price']) * float(trade['quantity'])
                    if entry_value > 0:
                        ret = pnl / entry_value
                        returns.append(ret)
                
                if pnl > 0:
                    profits.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))
        
        # 기본 통계
        total_trades = len(trades)
        winning_trades = len(profits)
        losing_trades = len(losses)
        
        # 수익률 계산
        total_return = sum(returns) if returns else 0
        annual_return = total_return * 252 / len(returns) if returns else 0  # 연환산
        
        # 변동성
        volatility = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0
        
        # 샤프 지수
        sharpe_ratio = RiskManager.calculate_sharpe_ratio(returns)
        
        # 최대 낙폭
        max_drawdown = RiskManager.calculate_max_drawdown(returns)
        
        # 승률
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 프로핏 팩터
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 1  # 0으로 나누기 방지
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades
        )
    
    @staticmethod
    def generate_performance_report(metrics: PerformanceMetrics) -> str:
        """성과 보고서 생성"""
        report = f"""
📊 성과 분석 보고서
{'='*40}

📈 수익률 지표:
  • 총 수익률: {metrics.total_return:.2%}
  • 연환산 수익률: {metrics.annual_return:.2%}
  • 변동성: {metrics.volatility:.2%}

🎯 리스크 지표:
  • 샤프 지수: {metrics.sharpe_ratio:.2f}
  • 최대 낙폭: {metrics.max_drawdown:.2%}

📊 거래 통계:
  • 총 거래 수: {metrics.total_trades}회
  • 승률: {metrics.win_rate:.1f}%
  • 수익 거래: {metrics.winning_trades}회
  • 손실 거래: {metrics.losing_trades}회
  • 프로핏 팩터: {metrics.profit_factor:.2f}

📝 평가:
"""
        
        # 성과 평가
        if metrics.sharpe_ratio > 2:
            report += "  🌟 우수한 위험 대비 수익률\n"
        elif metrics.sharpe_ratio > 1:
            report += "  ✅ 양호한 위험 대비 수익률\n"
        else:
            report += "  ⚠️ 개선 필요한 위험 대비 수익률\n"
        
        if metrics.win_rate > 60:
            report += "  🎯 높은 승률\n"
        elif metrics.win_rate > 40:
            report += "  📊 적정 승률\n"
        else:
            report += "  📉 낮은 승률\n"
        
        if metrics.max_drawdown < 0.1:
            report += "  🛡️ 안정적인 리스크 관리\n"
        elif metrics.max_drawdown < 0.2:
            report += "  ⚖️ 적정한 리스크 수준\n"
        else:
            report += "  ⚠️ 높은 리스크 수준\n"
        
        return report

# ============================================================================
# 📝 로깅 및 알림 헬퍼
# ============================================================================
class LoggingHelper:
    """로깅 헬퍼 클래스"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 중복 핸들러 방지
        if logger.handlers:
            return logger
        
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_trade(logger: logging.Logger, strategy: str, symbol: str, action: str, 
                  quantity: float, price: float, reason: str = ""):
        """거래 로그"""
        message = f"🔄 {strategy} | {action} {symbol} | {quantity}주 @ {price:,.0f} | {reason}"
        logger.info(message)
    
    @staticmethod
    def log_error_with_context(logger: logging.Logger, error: Exception, context: Dict[str, Any]):
        """컨텍스트와 함께 오류 로그"""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        logger.error(f"❌ 오류 발생: {str(error)} | 컨텍스트: {context_str}")

def log_execution_time(func):
    """실행 시간 로그 데코레이터"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger = logging.getLogger(func.__name__)
            logger.info(f"⏱️ {func.__name__} 실행 완료: {execution_time:.2f}초")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger = logging.getLogger(func.__name__)
            logger.error(f"❌ {func.__name__} 실행 실패: {execution_time:.2f}초, 오류: {e}")
            
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# ============================================================================
# 💾 파일 I/O 유틸리티
# ============================================================================
class FileManager:
    """파일 관리 유틸리티"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """디렉토리 생성 확인"""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path], 
                  indent: int = 2) -> bool:
        """JSON 파일 저장"""
        try:
            path_obj = Path(file_path)
            FileManager.ensure_directory(path_obj.parent)
            
            with open(path_obj, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
            return True
        except Exception as e:
            logging.error(f"JSON 저장 실패 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: Union[str, Path], default: Dict[str, Any] = None) -> Dict[str, Any]:
        """JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"JSON 로드 실패 {file_path}: {e}")
            return default or {}
    
    @staticmethod
    def save_csv(data: List[Dict[str, Any]], file_path: Union[str, Path], 
                 fieldnames: List[str] = None) -> bool:
        """CSV 파일 저장"""
        try:
            if not data:
                return False
            
            path_obj = Path(file_path)
            FileManager.ensure_directory(path_obj.parent)
            
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            with open(path_obj, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            return True
        except Exception as e:
            logging.error(f"CSV 저장 실패 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_csv(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """CSV 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logging.warning(f"CSV 로드 실패 {file_path}: {e}")
            return []
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], backup_dir: Union[str, Path] = None) -> Optional[Path]:
        """파일 백업"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return None
            
            if backup_dir is None:
                backup_dir = source_path.parent / 'backups'
            
            backup_path = Path(backup_dir)
            FileManager.ensure_directory(backup_path)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
            backup_file_path = backup_path / backup_filename
            
            import shutil
            shutil.copy2(source_path, backup_file_path)
            
            return backup_file_path
        except Exception as e:
            logging.error(f"파일 백업 실패 {file_path}: {e}")
            return None

# ============================================================================
# 🔐 보안 및 암호화 유틸리티
# ============================================================================
class SecurityUtils:
    """보안 관련 유틸리티"""
    
    @staticmethod
    def hash_string(text: str, algorithm: str = 'sha256') -> str:
        """문자열 해시"""
        try:
            hash_obj = hashlib.new(algorithm)
            hash_obj.update(text.encode('utf-8'))
            return hash_obj.hexdigest()
        except Exception as e:
            logging.error(f"해시 생성 실패: {e}")
            return ""
    
    @staticmethod
    def mask_sensitive_data(data: str, show_chars: int = 4) -> str:
        """민감한 데이터 마스킹"""
        if len(data) <= show_chars * 2:
            return '*' * len(data)
        
        return data[:show_chars] + '*' * (len(data) - show_chars * 2) + data[-show_chars:]
    
    @staticmethod
    def validate_api_key(api_key: str, min_length: int = 16) -> bool:
        """API 키 유효성 검사"""
        if not api_key or len(api_key) < min_length:
            return False
        
        # 기본적인 패턴 검사
        import re
        pattern = r'^[A-Za-z0-9\-_]+$'
        return bool(re.match(pattern, api_key))

# ============================================================================
# 🌐 네트워크 유틸리티
# ============================================================================
class NetworkUtils:
    """네트워크 관련 유틸리티"""
    
    @staticmethod
    async def check_internet_connection(timeout: int = 5) -> bool:
        """인터넷 연결 확인"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://www.google.com', timeout=timeout) as response:
                    return response.status == 200
        except:
            return False
    
    @staticmethod
    async def ping_server(url: str, timeout: int = 5) -> Tuple[bool, float]:
        """서버 핑 테스트"""
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    response_time = time.time() - start_time
                    return response.status == 200, response_time
        except:
            response_time = time.time() - start_time
            return False, response_time
    
    @staticmethod
    async def safe_api_request(url: str, method: str = 'GET', headers: Dict = None, 
                             data: Dict = None, timeout: int = 10, 
                             max_retries: int = 3) -> Optional[Dict]:
        """안전한 API 요청"""
        headers = headers or {}
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method, url, headers=headers, json=data, timeout=timeout
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limit
                            await asyncio.sleep(2 ** attempt)  # 지수 백오프
                            continue
                        else:
                            logging.warning(f"API 요청 실패: {response.status}")
                            return None
            except asyncio.TimeoutError:
                logging.warning(f"API 요청 타임아웃 (시도 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"API 요청 오류: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        return None

# ============================================================================
# 📊 데이터베이스 유틸리티
# ============================================================================
class DatabaseUtils:
    """데이터베이스 관련 유틸리티"""
    
    @staticmethod
    def create_connection(db_path: str) -> Optional[sqlite3.Connection]:
        """SQLite 연결 생성"""
        try:
            FileManager.ensure_directory(Path(db_path).parent)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
            return conn
        except Exception as e:
            logging.error(f"데이터베이스 연결 실패: {e}")
            return None
    
    @staticmethod
    def execute_query(conn: sqlite3.Connection, query: str, 
                     params: Tuple = None) -> Optional[List[sqlite3.Row]]:
        """쿼리 실행"""
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                conn.commit()
                return None
        except Exception as e:
            logging.error(f"쿼리 실행 실패: {e}")
            conn.rollback()
            return None
    
    @staticmethod
    def bulk_insert(conn: sqlite3.Connection, table: str, data: List[Dict[str, Any]]) -> bool:
        """대량 데이터 삽입"""
        if not data:
            return True
        
        try:
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            cursor = conn.cursor()
            values = [[row[col] for col in columns] for row in data]
            cursor.executemany(query, values)
            conn.commit()
            
            return True
        except Exception as e:
            logging.error(f"대량 삽입 실패: {e}")
            conn.rollback()
            return False
    
    @staticmethod
    def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        """테이블 존재 여부 확인"""
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            return cursor.fetchone() is not None
        except:
            return False

# ============================================================================
# 🔧 시스템 모니터링 유틸리티
# ============================================================================
class SystemMonitor:
    """시스템 모니터링 유틸리티"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """시스템 정보 조회"""
        try:
            import psutil
            
            # CPU 정보
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_total = memory.total / (1024**3)  # GB
            memory_available = memory.available / (1024**3)  # GB
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free = disk.free / (1024**3)  # GB
            
            # 네트워크 정보
            network = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'percent': memory_percent,
                    'total_gb': memory_total,
                    'available_gb': memory_available
                },
                'disk': {
                    'percent': disk_percent,
                    'free_gb': disk_free
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                }
            }
        except Exception as e:
            logging.error(f"시스템 정보 조회 실패: {e}")
            return {}
    
    @staticmethod
    def check_system_health() -> Dict[str, Any]:
        """시스템 건강 상태 확인"""
        info = SystemMonitor.get_system_info()
        
        health_status = {
            'healthy': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # CPU 체크
            cpu_percent = info.get('cpu', {}).get('percent', 0)
            if cpu_percent > 90:
                health_status['errors'].append(f'CPU 사용률 위험: {cpu_percent:.1f}%')
                health_status['healthy'] = False
            elif cpu_percent > 75:
                health_status['warnings'].append(f'CPU 사용률 높음: {cpu_percent:.1f}%')
            
            # 메모리 체크
            memory_percent = info.get('memory', {}).get('percent', 0)
            if memory_percent > 95:
                health_status['errors'].append(f'메모리 사용률 위험: {memory_percent:.1f}%')
                health_status['healthy'] = False
            elif memory_percent > 85:
                health_status['warnings'].append(f'메모리 사용률 높음: {memory_percent:.1f}%')
            
            # 디스크 체크
            disk_percent = info.get('disk', {}).get('percent', 0)
            disk_free = info.get('disk', {}).get('free_gb', 0)
            if disk_free < 1:
                health_status['errors'].append(f'디스크 공간 부족: {disk_free:.1f}GB')
                health_status['healthy'] = False
            elif disk_free < 5:
                health_status['warnings'].append(f'디스크 공간 경고: {disk_free:.1f}GB')
            
        except Exception as e:
            health_status['errors'].append(f'건강 상태 체크 실패: {str(e)}')
            health_status['healthy'] = False
        
        return health_status

# ============================================================================
# 🎨 데이터 시각화 헬퍼
# ============================================================================
class VisualizationHelper:
    """데이터 시각화 헬퍼"""
    
    @staticmethod
    def create_ascii_chart(values: List[float], width: int = 50, height: int = 10) -> str:
        """ASCII 차트 생성"""
        if not values:
            return "데이터 없음"
        
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            return "모든 값이 동일함"
        
        # 정규화
        normalized = [(val - min_val) / (max_val - min_val) for val in values]
        
        # 차트 생성
        chart_lines = []
        for y in range(height):
            line = ""
            threshold = 1 - (y / height)
            
            for x in range(min(width, len(normalized))):
                if normalized[x] >= threshold:
                    line += "█"
                else:
                    line += " "
            
            chart_lines.append(line)
        
        # 라벨 추가
        chart = f"최고: {max_val:.2f}\n"
        chart += "\n".join(chart_lines)
        chart += f"\n최저: {min_val:.2f}"
        
        return chart
    
    @staticmethod
    def format_table(data: List[Dict[str, Any]], headers: List[str] = None) -> str:
        """테이블 포맷팅"""
        if not data:
            return "데이터 없음"
        
        if headers is None:
            headers = list(data[0].keys())
        
        # 컬럼 너비 계산
        col_widths = {}
        for header in headers:
            col_widths[header] = len(str(header))
            for row in data:
                if header in row:
                    col_widths[header] = max(col_widths[header], len(str(row[header])))
        
        # 헤더 생성
        header_line = " | ".join(str(header).ljust(col_widths[header]) for header in headers)
        separator_line = "-+-".join("-" * col_widths[header] for header in headers)
        
        # 데이터 행 생성
        data_lines = []
        for row in data:
            line = " | ".join(str(row.get(header, "")).ljust(col_widths[header]) for header in headers)
            data_lines.append(line)
        
        return "\n".join([header_line, separator_line] + data_lines)

# ============================================================================
# 🔄 재시도 및 복구 유틸리티
# ============================================================================
def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                        logging.warning(f"{func.__name__} 실패 (시도 {attempt + 1}/{max_retries}), {wait_time}초 후 재시도: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"{func.__name__} 최종 실패: {e}")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                        logging.warning(f"{func.__name__} 실패 (시도 {attempt + 1}/{max_retries}), {wait_time}초 후 재시도: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"{func.__name__} 최종 실패: {e}")
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# ============================================================================
# 🎯 편의 함수들
# ============================================================================

# 전역 인스턴스
_currency_converter = CurrencyConverter()
_performance_analyzer = PerformanceAnalyzer()

async def convert_currency(amount: float, from_curr: str, to_curr: str) -> float:
    """간편 환율 변환"""
    try:
        from_currency = Currency(from_curr.upper())
        to_currency = Currency(to_curr.upper())
        return await _currency_converter.convert_amount(amount, from_currency, to_currency)
    except:
        return amount

def format_currency(amount: float, currency: str = 'KRW') -> str:
    """통화 포맷팅"""
    if currency == 'KRW':
        return f"₩{amount:,.0f}"
    elif currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'JPY':
        return f"¥{amount:,.0f}"
    elif currency == 'INR':
        return f"₹{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """변화율 계산"""
    return DataProcessor.safe_divide((new_value - old_value) * 100, old_value, 0.0)

def is_trading_time(market: Market) -> bool:
    """거래 시간 확인"""
    return TimeUtils.is_market_open(market)

def get_safe_filename(filename: str) -> str:
    """안전한 파일명 생성"""
    import re
    # 특수문자 제거
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 연속된 언더스코어 제거
    safe_name = re.sub(r'_+', '_', safe_name)
    return safe_name.strip('_')

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """문자열 자르기"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# ============================================================================
# 🎊 시스템 정보 출력
# ============================================================================
def print_system_banner():
    """시스템 배너 출력"""
    banner = """
🛠️ ═══════════════════════════════════════════════════════════════════════════ 🛠️
🛠️                        퀸트프로젝트 유틸리티 모듈 v1.1.0                         🛠️
🛠️ ═══════════════════════════════════════════════════════════════════════════ 🛠️

✨ 핵심 기능:
  📊 데이터 전처리 및 변환      🔢 기술지표 계산 (SMA, EMA, RSI, MACD)
  💰 리스크 관리 도구           🕐 시간 및 날짜 유틸리티
  💱 환율 및 화폐 변환          📈 성과 분석 도구
  📝 로깅 및 알림 헬퍼          💾 파일 I/O 유틸리티
  🔐 보안 및 암호화             🌐 네트워크 유틸리티
  📊 데이터베이스 도구          🔧 시스템 모니터링
  🎨 데이터 시각화              🔄 재시도 및 복구

🎯 지원 시장: 🇺🇸 미국주식 | 🇰🇷 한국주식 | 🇯🇵 일본주식 | 🇮🇳 인도주식 | 💰 암호화폐

🛠️ ═══════════════════════════════════════════════════════════════════════════ 🛠️
"""
    print(banner)

# 모듈 로드시 배너 출력
if __name__ == "__main__":
    print_system_banner()
    
    # 간단한 테스트
    print("🔍 유틸리티 테스트:")
    print(f"  • 숫자 정리: {DataProcessor.clean_numeric_data('1,234.56')}")
    print(f"  • 퍼센트 계산: {calculate_percentage_change(100, 120):.1f}%")
    print(f"  • 통화 포맷: {format_currency(1234567, 'KRW')}")
    print(f"  • 현재 시간: {TimeUtils.get_current_time()}")
    print(f"  • 시스템 정보: CPU {SystemMonitor.get_system_info().get('cpu', {}).get('percent', 0):.1f}%")
    print("✅ 모든 유틸리티 정상 로드 완료!")

            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger = logging.getLogger(func.__name__)
            logger.error(f"❌ {func.__name__} 실행 실패: {execution_time:.2f}초, 오류: {e}")
            
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger = logging.getLogger(func.__name__)
            logger.info(f"⏱️ {func.__name__} 실행 완료: {execution_time:.2f}초")
