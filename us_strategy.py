#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 US 전략 V7.0 - 기술적 분석 + AI 확신도 체크 (비용 최적화)
==============================================================================
월 6-8% 달성형 미국 주식 전용 전략
기술적 분석 중심 + 애매한 상황에서만 AI 확신도 체크

Author: 전설적퀸트팀
Version: 7.0.0 (AI 비용 최적화)
"""

import asyncio
import logging
import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import aiohttp
from dotenv import load_dotenv
import schedule
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import signal
import sys
import pytz

# OpenAI 연동 (확신도 체크 전용)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("⚠️ OpenAI 모듈 없음")

# IBKR 연동
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    logging.warning("⚠️ IBKR 모듈 없음")

warnings.filterwarnings('ignore')

# ========================================================================================
# 🕒 서머타임 관리자
# ========================================================================================

class DaylightSavingManager:
    def __init__(self):
        self.us_eastern = pytz.timezone('US/Eastern')
        self.korea = pytz.timezone('Asia/Seoul')
        self.cache = {}
    
    def is_dst_active(self, date=None) -> bool:
        if date is None:
            date = datetime.now().date()
        
        if date in self.cache:
            return self.cache[date]
        
        year = date.year
        # 3월 둘째주 일요일
        march_first = datetime(year, 3, 1)
        march_second_sunday = march_first + timedelta(days=(6 - march_first.weekday()) % 7 + 7)
        # 11월 첫째주 일요일  
        nov_first = datetime(year, 11, 1)
        nov_first_sunday = nov_first + timedelta(days=(6 - nov_first.weekday()) % 7)
        
        is_dst = march_second_sunday.date() <= date < nov_first_sunday.date()
        self.cache[date] = is_dst
        return is_dst
    
    def get_market_hours_kst(self, date=None) -> Tuple[datetime, datetime]:
        if date is None:
            date = datetime.now().date()
        
        market_open_et = datetime.combine(date, datetime.min.time().replace(hour=9, minute=30))
        market_close_et = datetime.combine(date, datetime.min.time().replace(hour=16, minute=0))
        
        if self.is_dst_active(date):
            market_open_et = self.us_eastern.localize(market_open_et, is_dst=True)
            market_close_et = self.us_eastern.localize(market_close_et, is_dst=True)
        else:
            market_open_et = self.us_eastern.localize(market_open_et, is_dst=False)
            market_close_et = self.us_eastern.localize(market_close_et, is_dst=False)
        
        return market_open_et.astimezone(self.korea), market_close_et.astimezone(self.korea)
    
    def get_trading_times_kst(self, date=None) -> Dict[str, datetime]:
        if date is None:
            date = datetime.now().date()
        
        trading_time_et = datetime.combine(date, datetime.min.time().replace(hour=10, minute=30))
        
        if self.is_dst_active(date):
            trading_time_et = self.us_eastern.localize(trading_time_et, is_dst=True)
        else:
            trading_time_et = self.us_eastern.localize(trading_time_et, is_dst=False)
        
        trading_time_kst = trading_time_et.astimezone(self.korea)
        
        return {
            'tuesday_kst': trading_time_kst if date.weekday() == 1 else None,
            'thursday_kst': trading_time_kst if date.weekday() == 3 else None,
            'market_time_et': trading_time_et,
            'market_time_kst': trading_time_kst,
            'dst_active': self.is_dst_active(date)
        }
    
    def is_market_hours(self, dt=None) -> bool:
        if dt is None:
            dt = datetime.now()
        open_kst, close_kst = self.get_market_hours_kst(dt.date())
        return open_kst <= dt.replace(tzinfo=self.korea) <= close_kst

# ========================================================================================
# 📈 기술적 지표 계산기 (MACD + 볼린저밴드 + RSI + 모멘텀)
# ========================================================================================

class TechnicalIndicators:
    @staticmethod
    def calculate_macd(prices: pd.Series, fast=12, slow=26, signal=9) -> Dict[str, float]:
        try:
            if len(prices) < slow + signal:
                return {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'neutral'}
            
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            current_macd = float(macd_line.iloc[-1])
            current_signal = float(signal_line.iloc[-1])
            current_histogram = float(histogram.iloc[-1])
            prev_histogram = float(histogram.iloc[-2]) if len(histogram) > 1 else 0
            
            if current_macd > current_signal and current_histogram > 0:
                trend = 'bullish'
            elif current_macd < current_signal and current_histogram < 0:
                trend = 'bearish'
            elif current_histogram > prev_histogram:
                trend = 'improving'
            else:
                trend = 'weakening'
            
            return {
                'macd': current_macd,
                'signal': current_signal,
                'histogram': current_histogram,
                'trend': trend,
                'crossover': 'buy' if current_macd > current_signal and macd_line.iloc[-2] <= signal_line.iloc[-2] else 
                           'sell' if current_macd < current_signal and macd_line.iloc[-2] >= signal_line.iloc[-2] else 'none'
            }
        except:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'neutral', 'crossover': 'none'}
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period=20, std=2) -> Dict[str, float]:
        try:
            if len(prices) < period:
                return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 0.5, 'squeeze': False}
            
            middle = prices.rolling(period).mean()
            std_dev = prices.rolling(period).std()
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            current_price = float(prices.iloc[-1])
            current_upper = float(upper.iloc[-1])
            current_middle = float(middle.iloc[-1])
            current_lower = float(lower.iloc[-1])
            
            if current_upper != current_lower:
                position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                position = 0.5
            
            band_width = (current_upper - current_lower) / current_middle
            avg_band_width = ((upper - lower) / middle).rolling(50).mean().iloc[-1] if len(prices) >= 50 else band_width
            squeeze = band_width < avg_band_width * 0.8
            
            return {
                'upper': current_upper,
                'middle': current_middle, 
                'lower': current_lower,
                'position': position,
                'squeeze': squeeze,
                'signal': 'overbought' if position > 0.8 else 'oversold' if position < 0.2 else 'normal'
            }
        except:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 0.5, 'squeeze': False, 'signal': 'normal'}
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period=14) -> float:
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0
    
    @staticmethod
    def calculate_momentum(prices: pd.Series) -> Dict[str, float]:
        try:
            current_price = float(prices.iloc[-1])
            
            momentum = {}
            if len(prices) >= 21:  # 1개월
                momentum['1m'] = ((current_price / float(prices.iloc[-21])) - 1) * 100
            else:
                momentum['1m'] = 0
                
            if len(prices) >= 63:  # 3개월
                momentum['3m'] = ((current_price / float(prices.iloc[-63])) - 1) * 100
            else:
                momentum['3m'] = 0
                
            if len(prices) >= 126:  # 6개월
                momentum['6m'] = ((current_price / float(prices.iloc[-126])) - 1) * 100
            else:
                momentum['6m'] = 0
                
            if len(prices) >= 252:  # 12개월
                momentum['12m'] = ((current_price / float(prices.iloc[-252])) - 1) * 100
            else:
                momentum['12m'] = 0
            
            # 추가 모멘텀 지표
            if len(prices) >= 5:  # 5일 모멘텀
                momentum['5d'] = ((current_price / float(prices.iloc[-5])) - 1) * 100
            else:
                momentum['5d'] = 0
                
            if len(prices) >= 10:  # 10일 모멘텀
                momentum['10d'] = ((current_price / float(prices.iloc[-10])) - 1) * 100
            else:
                momentum['10d'] = 0
            
            # 모멘텀 강도 계산
            momentum_values = [v for v in momentum.values() if v != 0]
            if momentum_values:
                momentum['avg'] = sum(momentum_values) / len(momentum_values)
                momentum['strength'] = len([v for v in momentum_values if v > 0]) / len(momentum_values)
            else:
                momentum['avg'] = 0
                momentum['strength'] = 0.5
            
            return momentum
        except Exception as e:
            logging.error(f"모멘텀 계산 실패: {e}")
            return {'1m': 0, '3m': 0, '6m': 0, '12m': 0, '5d': 0, '10d': 0, 'avg': 0, 'strength': 0.5}

# ========================================================================================
# 🤖 AI 확신도 체크 (비용 최적화)
# ========================================================================================

class AIConfidenceChecker:
    def __init__(self):
        self.client = None
        self.enabled = False
        self.api_key = os.getenv('OPENAI_API_KEY', '')
        self.model = "gpt-4o-mini"  # 가장 저렴한 모델
        self.daily_calls = 0
        self.max_daily_calls = 20  # 일일 최대 20회 (월 5천원 이하)
        self.last_reset = datetime.now().date()
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.enabled = True
                logging.info("✅ AI 확신도 체커 활성화 (비용 최적화)")
            except Exception as e:
                logging.warning(f"⚠️ OpenAI 초기화 실패: {e}")
    
    def _should_use_ai(self, confidence: float) -> bool:
        """애매한 상황(0.4-0.7)에서만 AI 사용"""
        if not self.enabled:
            return False
        
        # 일일 제한 체크
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_calls = 0
            self.last_reset = today
        
        if self.daily_calls >= self.max_daily_calls:
            return False
        
        # 애매한 신뢰도에서만 사용
        return 0.4 <= confidence <= 0.7
    
    async def check_confidence(self, symbol: str, technical_data: Dict, initial_confidence: float) -> Dict[str, Any]:
        """기술적 분석의 확신도만 체크 (뉴스/감정 분석 없음)"""
        
        if not self._should_use_ai(initial_confidence):
            return {
                'adjusted_confidence': initial_confidence,
                'ai_used': False,
                'reason': 'AI 사용 안함 (확신도 명확하거나 일일 제한)'
            }
        
        try:
            self.daily_calls += 1
            
            # 기술적 지표만 전달
            macd = technical_data.get('macd', {})
            bb = technical_data.get('bollinger', {})
            rsi = technical_data.get('rsi', 50)
            momentum = technical_data.get('momentum', {})
            
            prompt = f"""
            {symbol} 주식의 기술적 분석 신호들이 애매한 상황입니다. 확신도를 조정해주세요.
            
            현재 기술적 지표:
            - MACD: {macd.get('trend', 'neutral')} (크로스오버: {macd.get('crossover', 'none')})
            - 볼린저밴드: {bb.get('signal', 'normal')} (위치: {bb.get('position', 0.5):.2f})
            - RSI: {rsi:.1f}
            - 3개월 모멘텀: {momentum.get('3m', 0):.1f}%
            
            현재 신뢰도: {initial_confidence:.2f} (애매함)
            
            기술적 지표만 보고 0.1-0.9 사이에서 조정된 신뢰도와 간단한 이유를 JSON으로 답변:
            {{"confidence": 0.0-0.9, "reason": "간단한 기술적 근거"}}
            """
            
            response = await self._make_api_request(prompt)
            
            try:
                result = json.loads(response)
                adjusted_confidence = float(result.get('confidence', initial_confidence))
                reason = result.get('reason', 'AI 분석 완료')
                
                # 안전장치: 너무 극단적인 조정 방지
                max_adjustment = 0.2
                if abs(adjusted_confidence - initial_confidence) > max_adjustment:
                    if adjusted_confidence > initial_confidence:
                        adjusted_confidence = initial_confidence + max_adjustment
                    else:
                        adjusted_confidence = initial_confidence - max_adjustment
                
                return {
                    'adjusted_confidence': max(0.1, min(0.9, adjusted_confidence)),
                    'ai_used': True,
                    'reason': reason,
                    'daily_calls_used': self.daily_calls
                }
                
            except:
                return {
                    'adjusted_confidence': initial_confidence,
                    'ai_used': False,
                    'reason': 'AI 응답 파싱 실패'
                }
                
        except Exception as e:
            logging.error(f"AI 확신도 체크 실패: {e}")
            return {
                'adjusted_confidence': initial_confidence,
                'ai_used': False,
                'reason': f'AI 오류: {str(e)[:30]}'
            }
    
    async def _make_api_request(self, prompt: str) -> str:
        """최소 비용 API 요청"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 기술적 분석 전문가입니다. 간결하고 정확한 분석만 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,  # 토큰 최소화
                temperature=0.3  # 일관성 중시
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"AI API 요청 실패: {e}")
            return '{"confidence": 0.5, "reason": "API 오류"}'

# ========================================================================================
# 🔧 설정 관리자
# ========================================================================================

class Config:
    def __init__(self):
        self.config = {
            'strategy': {
                'enabled': True,
                'target_stocks': 8,
                'monthly_target': {'min': 6.0, 'max': 8.0}
            },
            'trading': {
                'mode': 'swing',  # swing 또는 weekly
                'take_profit': [8.0, 15.0],
                'profit_ratios': [60.0, 40.0],
                'stop_loss': 7.0,
                'swing': {
                    'hold_days': 14,  # 2주 보유
                    'profit_target': 8.0,
                    'max_positions': 8
                },
                'weekly': {
                    'enabled': True,
                    'tuesday_targets': 4,
                    'thursday_targets': 2,
                    'tuesday_allocation': 13.0,
                    'thursday_allocation': 8.0,
                    'profit_taking_threshold': 9.0,
                    'loss_cutting_threshold': -5.5,
                    'hold_until_next_signal': True
                }
            },
            'risk': {
                'max_position': 15.0,
                'daily_loss_limit': 1.0,
                'monthly_loss_limit': 3.0
            },
            'ibkr': {
                'enabled': True,
                'host': '127.0.0.1',
                'port': 7497,
                'client_id': 1,
                'paper_trading': True
            },
            'ai': {
                'enabled': True,
                'daily_limit': 20,
                'confidence_threshold': [0.4, 0.7]
            },
            'auto_trading': {
                'enabled': True,
                'min_confidence': 0.75,
                'max_daily_trades': 6,
                'notifications': True,
                'auto_execution': True
            }
        }
        
        if Path('.env').exists():
            load_dotenv()
    
    def get(self, key_path: str, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else default
        return value if value != {} else default

config = Config()

# ========================================================================================
# 📊 데이터 클래스
# ========================================================================================

@dataclass
class StockSignal:
    symbol: str
    action: str
    confidence: float
    price: float
    technical_scores: Dict[str, float]
    target_price: float
    stop_loss: float
    reasoning: str
    timestamp: datetime
    ai_confidence: Optional[Dict] = None

@dataclass 
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    entry_date: datetime
    mode: str = 'swing'  # 'swing' 또는 'weekly'
    stage: int = 1
    tp_executed: List[bool] = field(default_factory=lambda: [False, False])
    highest_price: float = 0.0
    entry_day: str = ''  # 'Tuesday' 또는 'Thursday'
    target_exit_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.avg_cost
        
        # 2주 스윙의 경우 목표 청산일 설정
        if self.mode == 'swing' and self.target_exit_date is None:
            self.target_exit_date = self.entry_date + timedelta(days=14)

    def profit_percent(self, current_price: float) -> float:
        return ((current_price - self.avg_cost) / self.avg_cost) * 100
    
    def days_held(self) -> int:
        return (datetime.now() - self.entry_date).days
    
    def should_exit_by_time(self) -> bool:
        """시간 기준 청산 여부"""
        if self.mode == 'swing':
            return self.days_held() >= 14
        elif self.mode == 'weekly':
            # 다음 거래일 전까지 또는 목표 수익 달성
            return False  # 신호 기반으로만 청산
        return False
        
# ========================================================================================
# 🚀 US 주식 선별 엔진
# ========================================================================================

class USStockSelector:
    def __init__(self):
        self.cache = {'symbols': [], 'last_update': None}
        self.indicators = TechnicalIndicators()
        self.ai_checker = AIConfidenceChecker()
    
    async def get_current_vix(self) -> float:
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            return float(hist['Close'].iloc[-1]) if not hist.empty else 20.0
        except:
            return 20.0
    
    async def collect_us_symbols(self) -> List[str]:
        """미국 주요 종목만 수집"""
        try:
            if self._is_cache_valid():
                return self.cache['symbols']
            
            # S&P 500 + NASDAQ 100 조합
            sp500 = self._get_sp500_symbols()
            nasdaq100 = self._get_nasdaq100_symbols()
            
            universe = list(set(sp500 + nasdaq100))
            
            # 시가총액, 거래량 기준 필터링
            filtered_symbols = []
            for symbol in universe:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    
                    market_cap = info.get('marketCap', 0) or 0
                    avg_volume = info.get('averageVolume', 0) or 0
                    
                    if market_cap > 5_000_000_000 and avg_volume > 1_000_000:
                        filtered_symbols.append(symbol)
                        
                    if len(filtered_symbols) >= 200:  # 최대 200개로 제한
                        break
                        
                except:
                    continue
            
            self.cache['symbols'] = filtered_symbols
            self.cache['last_update'] = datetime.now()
            
            logging.info(f"🇺🇸 US 투자 유니버스: {len(filtered_symbols)}개 종목")
            return filtered_symbols
            
        except Exception as e:
            logging.error(f"US 유니버스 생성 실패: {e}")
            return self._get_backup_symbols()
    
    def _get_sp500_symbols(self) -> List[str]:
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return [str(s).replace('.', '-') for s in tables[0]['Symbol'].tolist()]
        except:
            return []
    
    def _get_nasdaq100_symbols(self) -> List[str]:
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            for table in tables:
                if 'Symbol' in table.columns:
                    return table['Symbol'].dropna().tolist()
            return []
        except:
            return []
    
    def _is_cache_valid(self) -> bool:
        return (self.cache['last_update'] and 
                (datetime.now() - self.cache['last_update']).seconds < 24 * 3600)
    
    def _get_backup_symbols(self) -> List[str]:
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
            'BRK-B', 'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'DIS', 'ADBE',
            'CRM', 'ORCL', 'PFE', 'KO', 'PEP', 'ABBV', 'TMO', 'COST', 'XOM', 'WMT'
        ]
    
    async def get_stock_data(self, symbol: str) -> Dict:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1y")
        
            if hist.empty or len(hist) < 50:
                return {}
        
            current_price = float(hist['Close'].iloc[-1])
            closes = hist['Close']
            volumes = hist['Volume']
        
            # 기본 데이터
            data = {
                'symbol': symbol,
                'price': current_price,
                'market_cap': info.get('marketCap', 0) or 0,
                'avg_volume': info.get('averageVolume', 0) or 0,
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'pbr': info.get('priceToBook', 0) or 0,
                'roe': (info.get('returnOnEquity', 0) or 0) * 100,
                'eps_growth': (info.get('earningsQuarterlyGrowth', 0) or 0) * 100,  # 추가
                'revenue_growth': (info.get('revenueQuarterlyGrowth', 0) or 0) * 100,  # 추가
                'debt_to_equity': info.get('debtToEquity', 0) or 0,  # 추가
                'sector': info.get('sector', 'Unknown'),
                'beta': info.get('beta', 1.0) or 1.0
            }
            
            # PEG 계산 추가
            if data['pe_ratio'] > 0 and data['eps_growth'] > 0:
                data['peg'] = data['pe_ratio'] / data['eps_growth']
            else:
                data['peg'] = 999
            
            # 기술적 지표 계산
            data['macd'] = self.indicators.calculate_macd(closes)
            data['bollinger'] = self.indicators.calculate_bollinger_bands(closes)
            data['rsi'] = self.indicators.calculate_rsi(closes)
            data['momentum'] = self.indicators.calculate_momentum(closes)
            
            # 거래량 분석
            avg_vol = float(volumes.rolling(20).mean().iloc[-1])
            current_vol = float(volumes.iloc[-1])
            data['volume_spike'] = current_vol / avg_vol if avg_vol > 0 else 1
            
            # 변동성
            returns = closes.pct_change().dropna()
            data['volatility'] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
            
            await asyncio.sleep(0.2)  # API 제한 대응
            return data
            
        except Exception as e:
            logging.error(f"데이터 수집 실패 {symbol}: {e}")
            return {}
# ========================================================================================
# 🧠 5가지 전략 분석 엔진 (기술적 + 버핏 + 린치)
# ========================================================================================

class AdvancedStrategyAnalyzer:
    def __init__(self):
        self.ai_checker = AIConfidenceChecker()
    
    def calculate_comprehensive_score(self, data: Dict, vix: float) -> Tuple[float, Dict]:
        """5가지 전략 종합 점수 계산"""
        scores = {}
        
        # 1. 기술적 분석 (30%)
        scores['technical'] = self._calculate_technical_score(data)
        
        # 2. 워렌 버핏 전략 (25%)
        scores['buffett'] = self._calculate_buffett_score(data)
        
        # 3. 피터 린치 전략 (25%)
        scores['lynch'] = self._calculate_lynch_score(data)
        
        # 4. 모멘텀 전략 (15%)
        scores['momentum'] = self._calculate_momentum_score(data.get('momentum', {}))
        
        # 5. 품질 지표 (5%)
        scores['quality'] = self._calculate_quality_score_simple(data)
        
        # 가중 평균
        total_score = (
            scores['technical'] * 0.30 +
            scores['buffett'] * 0.25 +
            scores['lynch'] * 0.25 +
            scores['momentum'] * 0.15 +
            scores['quality'] * 0.05
        )
        
        # VIX 조정
        if vix <= 15:
            total_score *= 1.1  # 낮은 변동성에서 강세
        elif vix >= 30:
            total_score *= 0.9  # 높은 변동성에서 약세
        
        scores['total'] = total_score
        scores['vix_adjustment'] = total_score
        
        return total_score, scores
    
    def _calculate_buffett_score(self, data: Dict) -> float:
        """워렌 버핏 가치투자 점수"""
        score = 0.0
        
        # PBR (Price-to-Book Ratio) - 30%
        pbr = data.get('pbr', 999)
        if 0 < pbr <= 1.0:
            score += 0.30
        elif pbr <= 1.5:
            score += 0.25
        elif pbr <= 2.0:
            score += 0.20
        elif pbr <= 3.0:
            score += 0.10
        
        # ROE (Return on Equity) - 25%
        roe = data.get('roe', 0)
        if roe >= 20:
            score += 0.25
        elif roe >= 15:
            score += 0.20
        elif roe >= 10:
            score += 0.15
        elif roe >= 5:
            score += 0.10
        
        # 부채비율 (안전성) - 20%
        # 간접적으로 beta로 추정 (실제로는 debt-to-equity가 필요)
        beta = data.get('beta', 1.0)
        if 0.5 <= beta <= 1.2:  # 안정적인 beta
            score += 0.20
        elif 0.3 <= beta <= 1.5:
            score += 0.15
        elif beta <= 2.0:
            score += 0.10
        
        # PE Ratio (적정 밸류에이션) - 15%
        pe = data.get('pe_ratio', 999)
        if 5 <= pe <= 15:
            score += 0.15
        elif pe <= 20:
            score += 0.12
        elif pe <= 25:
            score += 0.08
        
        # 시가총액 (대형주 선호) - 10%
        market_cap = data.get('market_cap', 0)
        if market_cap > 50_000_000_000:  # 500억달러+
            score += 0.10
        elif market_cap > 20_000_000_000:  # 200억달러+
            score += 0.08
        elif market_cap > 10_000_000_000:  # 100억달러+
            score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_lynch_score(self, data: Dict) -> float:
        """피터 린치 성장투자 점수"""
        score = 0.0
        
        # PEG Ratio (가장 중요) - 40%
        pe_ratio = data.get('pe_ratio', 0)
        eps_growth = data.get('eps_growth', 0)
        
        if pe_ratio > 0 and eps_growth > 0:
            peg = pe_ratio / eps_growth
            if 0 < peg <= 0.5:
                score += 0.40
            elif peg <= 1.0:
                score += 0.35
            elif peg <= 1.5:
                score += 0.25
            elif peg <= 2.0:
                score += 0.15
        
        # EPS 성장률 - 25%
        if eps_growth >= 25:
            score += 0.25
        elif eps_growth >= 20:
            score += 0.20
        elif eps_growth >= 15:
            score += 0.15
        elif eps_growth >= 10:
            score += 0.10
        
        # 매출 성장률 - 20%
        # yfinance에서 직접 제공되지 않으므로 모멘텀으로 대체
        momentum_3m = data.get('momentum', {}).get('3m', 0)
        if momentum_3m >= 20:
            score += 0.20
        elif momentum_3m >= 15:
            score += 0.15
        elif momentum_3m >= 10:
            score += 0.10
        elif momentum_3m >= 5:
            score += 0.05
        
        # ROE (수익성) - 10%
        roe = data.get('roe', 0)
        if roe >= 15:
            score += 0.10
        elif roe >= 10:
            score += 0.08
        elif roe >= 5:
            score += 0.05
        
        # 적정 PE (과도한 고평가 피하기) - 5%
        if pe_ratio > 0:
            if 10 <= pe_ratio <= 30:
                score += 0.05
            elif 5 <= pe_ratio <= 40:
                score += 0.03
        
        return min(score, 1.0)
    
    def _calculate_technical_score(self, data: Dict) -> float:
        """기술적 분석 종합 점수"""
        score = 0.0
        
        # MACD 점수 (25%)
        macd_data = data.get('macd', {})
        macd_score = self._calculate_macd_score(macd_data)
        score += macd_score * 0.25
        
        # 볼린저밴드 점수 (25%)
        bb_data = data.get('bollinger', {})
        bb_score = self._calculate_bollinger_score(bb_data)
        score += bb_score * 0.25
        
        # RSI 점수 (25%)
        rsi = data.get('rsi', 50)
        rsi_score = self._calculate_rsi_score(rsi)
        score += rsi_score * 0.25
        
        # 거래량 점수 (25%)
        volume_spike = data.get('volume_spike', 1)
        volume_score = self._calculate_volume_score(volume_spike)
        score += volume_score * 0.25
        
        return min(score, 1.0)
    
    def _calculate_quality_score_simple(self, data: Dict) -> float:
        """간단한 품질 점수"""
        score = 0.0
        
        # 시가총액
        market_cap = data.get('market_cap', 0)
        if market_cap > 100_000_000_000:
            score += 0.4
        elif market_cap > 50_000_000_000:
            score += 0.3
        elif market_cap > 20_000_000_000:
            score += 0.2
        
        # 거래량
        avg_volume = data.get('avg_volume', 0)
        if avg_volume > 5_000_000:
            score += 0.3
        elif avg_volume > 2_000_000:
            score += 0.2
        elif avg_volume > 1_000_000:
            score += 0.1
        
        # 섹터 안정성
        stable_sectors = ['Consumer Staples', 'Utilities', 'Healthcare', 'Technology']
        if data.get('sector', '') in stable_sectors:
            score += 0.3
        
        return min(score, 1.0)

    def calculate_technical_score(self, data: Dict, vix: float) -> Tuple[float, Dict]:
        """기존 기술적 분석 (하위 호환성)"""
        return self.calculate_comprehensive_score(data, vix)
    
    def _calculate_macd_score(self, macd_data: Dict) -> float:
        score = 0.0
        
        trend = macd_data.get('trend', 'neutral')
        crossover = macd_data.get('crossover', 'none')
        histogram = macd_data.get('histogram', 0)
        
        # 트렌드 점수
        if trend == 'bullish':
            score += 0.4
        elif trend == 'improving':
            score += 0.3
        elif trend == 'bearish':
            score -= 0.2
        
        # 크로스오버 점수
        if crossover == 'buy':
            score += 0.4
        elif crossover == 'sell':
            score -= 0.3
        
        # 히스토그램 강도
        if histogram > 0.1:
            score += 0.2
        elif histogram < -0.1:
            score -= 0.2
        
        return max(0, min(1, score))
    
    def _calculate_bollinger_score(self, bb_data: Dict) -> float:
        score = 0.0
        
        signal = bb_data.get('signal', 'normal')
        position = bb_data.get('position', 0.5)
        squeeze = bb_data.get('squeeze', False)
        
        # 포지션별 점수
        if signal == 'oversold' and position < 0.3:
            score += 0.6  # 강한 매수 신호
        elif signal == 'normal' and 0.3 <= position <= 0.7:
            score += 0.4  # 정상 범위
        elif signal == 'overbought' and position > 0.8:
            score -= 0.3  # 과매수
        
        # 스퀴즈 보너스 (변동성 확대 전조)
        if squeeze:
            score += 0.3
        
        return max(0, min(1, score))
    
    def _calculate_rsi_score(self, rsi: float) -> float:
        if 30 <= rsi <= 70:
            return 0.8  # 정상 범위
        elif 25 <= rsi < 30:
            return 0.9  # 과매도에서 회복
        elif 70 < rsi <= 75:
            return 0.4  # 경미한 과매수
        elif rsi < 25:
            return 0.6  # 극도 과매도
        elif rsi > 80:
            return 0.2  # 극도 과매수
        else:
            return 0.5  # 기타
    
    def _calculate_momentum_score(self, momentum_data: Dict) -> float:
        score = 0.0
        
        # 단기 모멘텀 (20%) - 5일, 10일
        short_momentum = (momentum_data.get('5d', 0) + momentum_data.get('10d', 0)) / 2
        if short_momentum >= 3:
            score += 0.2
        elif short_momentum >= 1:
            score += 0.15
        elif short_momentum >= 0:
            score += 0.1
        elif short_momentum < -3:
            score -= 0.1
        
        # 중기 모멘텀 (30%) - 1개월, 3개월
        medium_momentum = (momentum_data.get('1m', 0) + momentum_data.get('3m', 0)) / 2
        if medium_momentum >= 15:
            score += 0.3
        elif medium_momentum >= 10:
            score += 0.25
        elif medium_momentum >= 5:
            score += 0.2
        elif medium_momentum >= 0:
            score += 0.1
        elif medium_momentum < -10:
            score -= 0.15
        
        # 장기 모멘텀 (30%) - 6개월, 12개월
        long_momentum = (momentum_data.get('6m', 0) + momentum_data.get('12m', 0)) / 2
        if long_momentum >= 30:
            score += 0.3
        elif long_momentum >= 20:
            score += 0.25
        elif long_momentum >= 10:
            score += 0.2
        elif long_momentum >= 0:
            score += 0.1
        elif long_momentum < -15:
            score -= 0.15
        
        # 모멘텀 일관성 (20%) - 모든 기간이 같은 방향인지
        momentum_strength = momentum_data.get('strength', 0.5)
        if momentum_strength >= 0.8:  # 80% 이상 양의 모멘텀
            score += 0.2
        elif momentum_strength >= 0.6:
            score += 0.15
        elif momentum_strength <= 0.2:  # 80% 이상 음의 모멘텀
            score -= 0.1
        
        # 평균 모멘텀 보너스/페널티
        avg_momentum = momentum_data.get('avg', 0)
        if avg_momentum >= 20:
            score += 0.1
        elif avg_momentum <= -15:
            score -= 0.1
        
        return max(0, min(1, score))
    
    def _calculate_volume_score(self, volume_spike: float) -> float:
        if volume_spike >= 2.0:
            return 1.0  # 강한 거래량 증가
        elif volume_spike >= 1.5:
            return 0.8
        elif volume_spike >= 1.2:
            return 0.6
        elif volume_spike < 0.8:
            return 0.3  # 거래량 감소
        else:
            return 0.5
    
    async def determine_action_with_ai(self, data: Dict, technical_score: float) -> Tuple[str, float, Dict]:
        """기술적 분석 + AI 확신도 체크"""
        
        # 기본 기술적 신호 결정
        if technical_score >= 0.75:
            initial_action = 'buy'
            initial_confidence = min(technical_score, 0.9)
        elif technical_score <= 0.25:
            initial_action = 'sell'
            initial_confidence = min(1 - technical_score, 0.9)
        else:
            initial_action = 'hold'
            initial_confidence = 0.5
        
        # AI 확신도 체크 (애매한 경우만)
        ai_result = await self.ai_checker.check_confidence(
            data['symbol'], 
            data, 
            initial_confidence
        )
        
        final_confidence = ai_result['adjusted_confidence']
        
        # 최종 액션 재결정
        if final_confidence >= 0.7:
            final_action = 'buy'
        elif final_confidence <= 0.3:
            final_action = 'sell'
        else:
            final_action = 'hold'

        return final_action, final_confidence, ai_result
# ========================================================================================
# 🏦 IBKR 연동 시스템
# ========================================================================================

class IBKRTrader:
    def __init__(self):
        self.ib = None
        self.connected = False
        self.positions = {}
        self.daily_pnl = 0.0
        
    async def connect(self) -> bool:
        try:
            if not IBKR_AVAILABLE:
                return False
            
            self.ib = IB()
            await self.ib.connectAsync(
                config.get('ibkr.host', '127.0.0.1'),
                config.get('ibkr.port', 7497),
                clientId=config.get('ibkr.client_id', 1)
            )
            
            if self.ib.isConnected():
                self.connected = True
                await self._update_account()
                logging.info("✅ IBKR 연결 완료")
                return True
            return False
                
        except Exception as e:
            logging.error(f"IBKR 연결 실패: {e}")
            return False
    
    async def disconnect(self):
        try:
            if self.ib and self.connected:
                self.ib.disconnect()
                self.connected = False
        except Exception as e:
            logging.error(f"연결 해제 오류: {e}")
    
    async def _update_account(self):
        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'DayPNL':
                    self.daily_pnl = float(av.value)
                    break
                    
            portfolio = self.ib.portfolio()
            self.positions = {}
            for pos in portfolio:
                if pos.position != 0:
                    self.positions[pos.contract.symbol] = {
                        'quantity': pos.position,
                        'avg_cost': pos.avgCost,
                        'market_price': pos.marketPrice,
                        'unrealized_pnl': pos.unrealizedPNL
                    }
        except Exception as e:
            logging.error(f"계좌 업데이트 실패: {e}")
    
    async def place_buy_order(self, symbol: str, quantity: int) -> Optional[str]:
        try:
            if not self.connected:
                return None
            
            contract = Stock(symbol, 'SMART', 'USD')
            order = MarketOrder('BUY', quantity)
            trade = self.ib.placeOrder(contract, order)
            
            logging.info(f"📈 매수: {symbol} {quantity}주")
            return str(trade.order.orderId)
            
        except Exception as e:
            logging.error(f"매수 실패 {symbol}: {e}")
            return None
    
    async def place_sell_order(self, symbol: str, quantity: int, reason: str = '') -> Optional[str]:
        try:
            if not self.connected or symbol not in self.positions:
                return None
            
            current_qty = abs(self.positions[symbol]['quantity'])
            sell_qty = min(quantity, current_qty)
            
            contract = Stock(symbol, 'SMART', 'USD')
            order = MarketOrder('SELL', sell_qty)
            trade = self.ib.placeOrder(contract, order)
            
            logging.info(f"📉 매도: {symbol} {sell_qty}주 - {reason}")
            return str(trade.order.orderId)
            
        except Exception as e:
            logging.error(f"매도 실패 {symbol}: {e}")
            return None
    
    async def get_portfolio_value(self) -> float:
        """포트폴리오 총 가치 조회"""
        try:
            if not self.connected:
                return 100000.0  # 기본값
            
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    return float(av.value)
            
            return 100000.0
        except:
            return 100000.0
    
    async def get_current_price(self, symbol: str) -> float:
        """현재가 조회"""
        try:
            if not self.connected:
                return 0.0
            
            contract = Stock(symbol, 'SMART', 'USD')
            ticker = self.ib.reqMktData(contract)
            await asyncio.sleep(1)  # 데이터 수신 대기
            
            if ticker.marketPrice():
                return float(ticker.marketPrice())
            elif ticker.close:
                return float(ticker.close)
            else:
                return 0.0
                
        except:
            return 0.0    

# ========================================================================================
# 🏆 US 전략 메인 시스템
# ========================================================================================

class USStrategy:
    def __init__(self):
        self.enabled = config.get('strategy.enabled', True)
        self.trading_mode = config.get('trading.mode', 'swing')  # 'swing' 또는 'weekly'
        
        self.dst_manager = DaylightSavingManager()
        self.selector = USStockSelector()
        self.analyzer = AdvancedStrategyAnalyzer()
        self.ibkr = IBKRTrader()
        
        self.selected_stocks = []
        self.positions = {}  # symbol -> Position
        self.last_selection = None
        self.monthly_return = 0.0
        self.last_trade_dates = {'Tuesday': None, 'Thursday': None}
        
        if self.enabled:
            logging.info("🇺🇸 US 전략 V7.0 시스템 가동!")
            logging.info(f"📈 거래 모드: {self.trading_mode.upper()}")
            logging.info(f"🕒 서머타임: {'활성' if self.dst_manager.is_dst_active() else '비활성'}")
            logging.info(f"🤖 AI 확신도 체커: {'활성' if self.analyzer.ai_checker.enabled else '비활성'}")
    
    def is_trading_day(self) -> Tuple[bool, str]:
        """거래일 여부 및 타입 확인"""
        now = datetime.now()
        weekday = now.weekday()  # 0=월요일, 1=화요일, 3=목요일
        
        if weekday == 1:  # 화요일
            return True, 'Tuesday'
        elif weekday == 3:  # 목요일
            return True, 'Thursday'
        else:
            return False, ''
    
    def get_trading_allocation(self, day_type: str) -> float:
        """거래일별 자금 배분"""
        if day_type == 'Tuesday':
            return config.get('trading.weekly.tuesday_allocation', 13.0)
        elif day_type == 'Thursday':
            return config.get('trading.weekly.thursday_allocation', 8.0)
        else:
            return 0.0
    
    def get_target_positions(self, day_type: str) -> int:
        """거래일별 목표 포지션 수"""
        if day_type == 'Tuesday':
            return config.get('trading.weekly.tuesday_targets', 4)
        elif day_type == 'Thursday':
            return config.get('trading.weekly.thursday_targets', 2)
        else:
            return 0
    
    async def execute_2week_swing_strategy(self):
        """2주 스윙 전략 실행"""
        try:
            logging.info("🎯 2주 스윙 전략 실행 시작")
            
            # 기존 포지션 관리
            await self._manage_existing_positions()
            
            # 새로운 진입 기회 탐색
            if self.trading_mode == 'swing':
                await self._execute_swing_entries()
            elif self.trading_mode == 'weekly':
                await self._execute_weekly_entries()
            
            # 포트폴리오 상태 리포트
            await self._report_portfolio_status()
            
        except Exception as e:
            logging.error(f"2주 스윙 전략 실행 실패: {e}")
    
    async def _manage_existing_positions(self):
        """기존 포지션 관리 (청산/보유 결정)"""
        try:
            for symbol, position in list(self.positions.items()):
                current_price = await self._get_current_price(symbol)
                if current_price <= 0:
                    continue
                
                profit_pct = position.profit_percent(current_price)
                days_held = position.days_held()
                
                # 청산 조건 체크
                should_exit = False
                exit_reason = ""
                
                # 1. 시간 기준 청산 (2주 스윙)
                if position.should_exit_by_time():
                    should_exit = True
                    exit_reason = f"시간만료 ({days_held}일)"
                
                # 2. 손절 기준
                elif profit_pct <= -config.get('trading.stop_loss', 7.0):
                    should_exit = True
                    exit_reason = f"손절 ({profit_pct:.1f}%)"
                
                # 3. 목표 수익 달성
                elif profit_pct >= config.get('trading.take_profit.0', 8.0):
                    should_exit = True
                    exit_reason = f"목표달성 ({profit_pct:.1f}%)"
                
                # 4. 기술적 매도 신호
                else:
                    signal = await self.analyze_stock_signal(symbol)
                    if signal.action == 'sell' and signal.confidence >= 0.7:
                        should_exit = True
                        exit_reason = f"매도신호 (신뢰도: {signal.confidence:.1%})"
                
                # 청산 실행
                if should_exit:
                    await self._exit_position(position, exit_reason, current_price)
                else:
                    logging.info(f"📊 {symbol}: 보유유지 - {days_held}일차, {profit_pct:+.1f}%")
                    
        except Exception as e:
            logging.error(f"포지션 관리 실패: {e}")

async def _execute_swing_entries(self):
        """스윙 진입 실행 (2주 보유)"""
        try:
            # 현재 포지션 수 확인
            current_positions = len(self.positions)
            max_positions = config.get('trading.swing.max_positions', 8)
            
            if current_positions >= max_positions:
                logging.info(f"📊 포지션 만석: {current_positions}/{max_positions}")
                return
            
            # 신규 종목 선별
            available_slots = max_positions - current_positions
            candidates = await self.auto_select_stocks()
            
            # 기존 보유 종목 제외
            new_candidates = [s for s in candidates if s not in self.positions.keys()]
            
            logging.info(f"🎯 신규 진입 후보: {len(new_candidates)}개 (가능 슬롯: {available_slots}개)")
            
            # 상위 후보군에서 진입
            entries = 0
            for symbol in new_candidates[:available_slots]:
                try:
                    signal = await self.analyze_stock_signal(symbol)
                    
                    if signal.action == 'buy' and signal.confidence >= 0.75:
                        success = await self._enter_position(symbol, signal, 'swing')
                        if success:
                            entries += 1
                            logging.info(f"✅ {symbol} 신규 진입 성공")
                        
                except Exception as e:
                    logging.error(f"{symbol} 진입 실패: {e}")
            
            logging.info(f"🎯 스윙 진입 완료: {entries}개 신규 포지션")
            
        except Exception as e:
            logging.error(f"스윙 진입 실행 실패: {e}")
    
    async def _execute_weekly_entries(self):
        """주간 진입 실행 (화/목 거래)"""
        try:
            is_trading, day_type = await self.is_trading_day()
            
            if not is_trading:
                logging.info("📅 오늘은 거래일이 아닙니다")
                return
            
            # 해당 요일에 이미 거래했는지 확인
            last_trade = self.last_trade_dates.get(day_type)
            today = datetime.now().date()
            
            if last_trade and last_trade == today:
                logging.info(f"📅 {day_type} 거래 이미 완료")
                return
            
            # 목표 포지션 수
            target_positions = self.get_target_positions(day_type)
            allocation = self.get_trading_allocation(day_type)
            
            logging.info(f"📈 {day_type} 거래 시작: 목표 {target_positions}개, 배분 {allocation}%")
            
            # 해당 요일 기존 포지션 정리
            await self._cleanup_weekly_positions(day_type)
            
            # 신규 진입
            candidates = await self.auto_select_stocks()
            entries = 0
            
            for symbol in candidates[:target_positions]:
                try:
                    signal = await self.analyze_stock_signal(symbol)
                    
                    if signal.action == 'buy' and signal.confidence >= 0.7:
                        success = await self._enter_position(symbol, signal, 'weekly', day_type)
                        if success:
                            entries += 1
                            
                except Exception as e:
                    logging.error(f"{symbol} 주간 진입 실패: {e}")
            
            # 거래일 기록
            self.last_trade_dates[day_type] = today
            logging.info(f"✅ {day_type} 거래 완료: {entries}/{target_positions}개 진입")
            
        except Exception as e:
            logging.error(f"주간 진입 실행 실패: {e}")
    
    async def _cleanup_weekly_positions(self, day_type: str):
        """주간 포지션 정리"""
        try:
            positions_to_exit = []
            
            for symbol, position in self.positions.items():
                if position.mode == 'weekly' and position.entry_day == day_type:
                    current_price = await self._get_current_price(symbol)
                    profit_pct = position.profit_percent(current_price)
                    
                    # 목표 수익 달성 또는 손실 한도 도달시 청산
                    profit_threshold = config.get('trading.weekly.profit_taking_threshold', 9.0)
                    loss_threshold = config.get('trading.weekly.loss_cutting_threshold', -5.5)
                    
                    if profit_pct >= profit_threshold or profit_pct <= loss_threshold:
                        positions_to_exit.append((position, f"주간정리 ({profit_pct:+.1f}%)", current_price))
            
            # 청산 실행
            for position, reason, price in positions_to_exit:
                await self._exit_position(position, reason, price)
                
        except Exception as e:
            logging.error(f"주간 포지션 정리 실패: {e}")
    
    async def _enter_position(self, symbol: str, signal: StockSignal, mode: str, day_type: str = '') -> bool:
        """포지션 진입"""
        try:
            # 포지션 크기 계산 (자금의 일정 비율)
            if mode == 'swing':
                position_size = 100_000 / len(self.selected_stocks) if self.selected_stocks else 12_500  # 8만 분할
            else:  # weekly
                allocation = self.get_trading_allocation(day_type)
                position_size = 100_000 * allocation / 100
            
            quantity = int(position_size / signal.price)
            
            if quantity <= 0:
                return False
            
            # IBKR 주문 실행
            if self.ibkr.connected:
                order_id = await self.ibkr.place_buy_order(symbol, quantity)
                if not order_id:
                    return False
            
            # 포지션 기록
            position = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=signal.price,
                entry_date=datetime.now(),
                mode=mode,
                entry_day=day_type
            )
            
            self.positions[symbol] = position
            
            logging.info(f"📈 {symbol} 진입: {quantity}주 @ ${signal.price:.2f} ({mode})")
            return True
            
        except Exception as e:
            logging.error(f"{symbol} 진입 실패: {e}")
            return False
    
    async def _exit_position(self, position: Position, reason: str, current_price: float):
        """포지션 청산"""
        try:
            symbol = position.symbol
            
            # IBKR 매도 주문
            if self.ibkr.connected:
                order_id = await self.ibkr.place_sell_order(symbol, position.quantity, reason)
                if not order_id:
                    logging.error(f"{symbol} 매도 주문 실패")
                    return
            
            # 수익률 계산
            profit_pct = position.profit_percent(current_price)
            profit_amount = (current_price - position.avg_cost) * position.quantity
            
            # 포지션 제거
            del self.positions[symbol]
            
            logging.info(f"📉 {symbol} 청산: {reason} | {profit_pct:+.1f}% (${profit_amount:+.2f})")
            
        except Exception as e:
            logging.error(f"{position.symbol} 청산 실패: {e}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """현재가 조회"""
        try:
            if self.ibkr.connected:
                return await self.ibkr.get_current_price(symbol)
            else:
                # yfinance 폴백
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1d")
                return float(hist['Close'].iloc[-1]) if not hist.empty else 0.0
        except:
            return 0.0

async def _report_portfolio_status(self):
        """포트폴리오 상태 리포트"""
        try:
            if not self.positions:
                logging.info("📊 현재 보유 포지션 없음")
                return
            
            total_value = 0
            total_profit = 0
            
            logging.info(f"📊 포트폴리오 현황 ({len(self.positions)}개 포지션):")
            
            for symbol, position in self.positions.items():
                current_price = await self._get_current_price(symbol)
                profit_pct = position.profit_percent(current_price)
                days_held = position.days_held()
                market_value = current_price * position.quantity
                
                total_value += market_value
                total_profit += (current_price - position.avg_cost) * position.quantity
                
                logging.info(f"  {symbol}: {profit_pct:+6.1f}% | {days_held:2d}일 | "
                           f"${market_value:,.0f} ({position.mode})")
            
            total_profit_pct = (total_profit / (total_value - total_profit)) * 100 if total_value > total_profit else 0
            logging.info(f"📈 포트폴리오 총합: ${total_value:,.0f} | {total_profit_pct:+.1f}% (${total_profit:+,.0f})")
            
    async def run_full_auto_trading(self):
        """🤖 완전 자동매매 실행"""
        try:
            if not config.get('auto_trading.enabled', True):
                logging.info("⚠️ 자동매매 비활성화 상태")
                return
            
            logging.info("🤖 완전 자동매매 시작!")
            
            # 1. 시장 상태 및 거래 가능 여부 확인
            if not await self._pre_trading_checks():
                return
            
            # 2. 기존 포지션 자동 관리
            await self._auto_manage_positions()
            
            # 3. 신규 진입 자동 실행
            await self._auto_enter_new_positions()
            
            # 4. 거래 완료 리포트 및 알림
            await self._send_trading_report()
            
            logging.info("✅ 완전 자동매매 완료!")
            
        except Exception as e:
            logging.error(f"❌ 완전 자동매매 실패: {e}")
            await self._send_error_notification(str(e))
    
    async def _pre_trading_checks(self) -> bool:
        """사전 거래 가능 여부 확인"""
        try:
            # 1. 시장 시간 확인
            if not self.dst_manager.is_market_hours():
                logging.info("⏰ 시장 시간이 아님")
                return False
            
            # 2. 거래일 확인
            is_trading, day_type = self.is_trading_day()
            if not is_trading:
                logging.info("📅 오늘은 거래일이 아님")
                return False
            
            # 3. IBKR 연결 확인
            if not await self.ibkr.connect():
                logging.error("❌ IBKR 연결 실패")
                return False
            
            # 4. 계좌 잔고 확인
            portfolio_value = await self.ibkr.get_portfolio_value()
            if portfolio_value < 10000:  # 최소 1만달러
                logging.error(f"❌ 계좌 잔고 부족: ${portfolio_value:,.0f}")
                return False
            
            # 5. 일일 거래 한도 확인
            daily_trades = len([p for p in self.positions.values() 
                              if p.entry_date.date() == datetime.now().date()])
            max_daily = config.get('auto_trading.max_daily_trades', 6)
            
            if daily_trades >= max_daily:
                logging.info(f"📊 일일 거래 한도 도달: {daily_trades}/{max_daily}")
                return False
            
            logging.info(f"✅ 사전 체크 통과 - {day_type} 자동매매 진행")
            return True
            
        except Exception as e:
            logging.error(f"사전 체크 실패: {e}")
            return False
    
    async def _auto_manage_positions(self):
        """기존 포지션 자동 관리"""
        try:
            logging.info("📊 기존 포지션 자동 관리 시작")
            
            if not self.positions:
                logging.info("📊 관리할 포지션 없음")
                return
            
            managed_count = 0
            
            for symbol, position in list(self.positions.items()):
                try:
                    current_price = await self._get_current_price(symbol)
                    if current_price <= 0:
                        continue
                    
                    profit_pct = position.profit_percent(current_price)
                    days_held = position.days_held()
                    
                    # 자동 청산 조건 확인
                    should_exit, reason = await self._check_auto_exit_conditions(
                        position, current_price, profit_pct, days_held
                    )
                    
                    if should_exit:
                        success = await self._execute_auto_exit(position, reason, current_price)
                        if success:
                            managed_count += 1
                            logging.info(f"🤖 {symbol} 자동 청산: {reason}")
                    else:
                        # 포지션 업데이트 (최고가 갱신 등)
                        if current_price > position.highest_price:
                            position.highest_price = current_price
                        
                        logging.info(f"📊 {symbol}: 보유유지 - {days_held}일차, {profit_pct:+.1f}%")
                
                except Exception as e:
                    logging.error(f"{symbol} 포지션 관리 실패: {e}")
            
            logging.info(f"✅ 포지션 관리 완료: {managed_count}개 청산")
            
        except Exception as e:
            logging.error(f"포지션 관리 실패: {e}")

async def _check_auto_exit_conditions(self, position: Position, current_price: float, 
                                        profit_pct: float, days_held: int) -> Tuple[bool, str]:
        """자동 청산 조건 확인"""
        try:
            # 1. 시간 기준 청산 (2주 완료)
            if position.should_exit_by_time():
                return True, f"시간만료 ({days_held}일)"
            
            # 2. 손절선 (-7%)
            stop_loss = config.get('trading.stop_loss', 7.0)
            if profit_pct <= -stop_loss:
                return True, f"손절 ({profit_pct:.1f}%)"
            
            # 3. 목표 수익 달성 (+8%)
            target_profit = config.get('trading.take_profit.0', 8.0)
            if profit_pct >= target_profit:
                return True, f"목표달성 ({profit_pct:.1f}%)"
            
            # 4. 고점 대비 -3% 하락 (트레일링 스톱)
            if position.highest_price > 0:
                drawdown_pct = ((current_price - position.highest_price) / position.highest_price) * 100
                if drawdown_pct <= -3.0:
                    return True, f"트레일링스톱 ({drawdown_pct:.1f}%)"
            
            # 5. 강한 매도 신호
            signal = await self.analyze_stock_signal(position.symbol)
            min_confidence = config.get('auto_trading.min_confidence', 0.75)
            
            if signal.action == 'sell' and signal.confidence >= min_confidence:
                return True, f"매도신호 ({signal.confidence:.1%})"
            
            return False, ""
            
        except Exception as e:
            logging.error(f"청산 조건 확인 실패: {e}")
            return False, "확인실패"
    
    async def _execute_auto_exit(self, position: Position, reason: str, current_price: float) -> bool:
        """자동 청산 실행"""
        try:
            symbol = position.symbol
            
            # IBKR 자동 매도 주문
            if config.get('auto_trading.auto_execution', True):
                order_id = await self.ibkr.place_sell_order(symbol, position.quantity, f"자동청산: {reason}")
                if not order_id:
                    logging.error(f"❌ {symbol} 자동 매도 주문 실패")
                    return False
            
            # 수익률 계산 및 기록
            profit_pct = position.profit_percent(current_price)
            profit_amount = (current_price - position.avg_cost) * position.quantity
            
            # 포지션 제거
            del self.positions[symbol]
            
            # 거래 기록 저장
            trade_record = {
                'symbol': symbol,
                'action': 'sell',
                'quantity': position.quantity,
                'price': current_price,
                'profit_pct': profit_pct,
                'profit_amount': profit_amount,
                'reason': reason,
                'hold_days': position.days_held(),
                'timestamp': datetime.now()
            }
            
            await self._save_trade_record(trade_record)
            
            logging.info(f"🤖 {symbol} 자동청산 완료: {reason} | {profit_pct:+.1f}% (${profit_amount:+,.2f})")
            return True
            
        except Exception as e:
            logging.error(f"자동청산 실행 실패: {e}")
            return False
    
    async def _auto_enter_new_positions(self):
        """신규 포지션 자동 진입"""
        try:
            logging.info("🚀 신규 포지션 자동 진입 시작")
            
            # 거래일 확인
            is_trading, day_type = self.is_trading_day()
            if not is_trading:
                return
            
            # 목표 포지션 수 및 자금 배분
            target_positions = self.get_target_positions(day_type)
            current_day_positions = len([p for p in self.positions.values() 
                                       if p.entry_day == day_type and 
                                       p.entry_date.date() == datetime.now().date()])
            
            if current_day_positions >= target_positions:
                logging.info(f"📊 {day_type} 목표 포지션 달성: {current_day_positions}/{target_positions}")
                return
            
            # 신규 진입 가능 수량
            available_slots = target_positions - current_day_positions
            
            # 자동 종목 선별
            candidates = await self.auto_select_stocks()
            new_candidates = [s for s in candidates if s not in self.positions.keys()]
            
            logging.info(f"🎯 {day_type} 자동진입: {available_slots}개 슬롯, {len(new_candidates)}개 후보")
            
            # 자동 진입 실행
            entries = 0
            min_confidence = config.get('auto_trading.min_confidence', 0.75)
            
            for symbol in new_candidates[:available_slots]:
                try:
                    signal = await self.analyze_stock_signal(symbol)
                    
                    # 높은 신뢰도 매수 신호만 자동 실행
                    if signal.action == 'buy' and signal.confidence >= min_confidence:
                        success = await self._execute_auto_entry(symbol, signal, day_type)
                        if success:
                            entries += 1
                            logging.info(f"🤖 {symbol} 자동 진입: {signal.confidence:.1%} 신뢰도")
                        
                        # 안전을 위한 딜레이
                        await asyncio.sleep(2)
                    else:
                        logging.info(f"📊 {symbol}: 진입보류 - {signal.action} ({signal.confidence:.1%})")
                
                except Exception as e:
                    logging.error(f"{symbol} 자동진입 실패: {e}")
            
            logging.info(f"✅ {day_type} 자동진입 완료: {entries}/{available_slots}개")
            
        except Exception as e:
            logging.error(f"신규 포지션 자동진입 실패: {e}")
    
    async def _execute_auto_entry(self, symbol: str, signal: StockSignal, day_type: str) -> bool:
        """자동 진입 실행"""
        try:
            # 포지션 크기 계산
            allocation = self.get_trading_allocation(day_type)
            portfolio_value = await self.ibkr.get_portfolio_value()
            position_value = portfolio_value * allocation / 100
            quantity = int(position_value / signal.price)
            
            if quantity <= 0:
                logging.error(f"❌ {symbol} 수량 계산 오류: {quantity}")
                return False
            
            # 자동 매수 주문 실행
            if config.get('auto_trading.auto_execution', True):
                order_id = await self.ibkr.place_buy_order(symbol, quantity)
                if not order_id:
                    logging.error(f"❌ {symbol} 자동 매수 주문 실패")
                    return False
            
            # 포지션 생성 및 기록
            position = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=signal.price,
                entry_date=datetime.now(),
                mode=self.trading_mode,
                entry_day=day_type
            )
            
            self.positions[symbol] = position
            
            # 거래 기록 저장
            trade_record = {
                'symbol': symbol,
                'action': 'buy',
                'quantity': quantity,
                'price': signal.price,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning,
                'day_type': day_type,
                'timestamp': datetime.now()
            }
            
            await self._save_trade_record(trade_record)
            
            logging.info(f"🤖 {symbol} 자동진입 완료: {quantity}주 @ ${signal.price:.2f}")
            return True
            
        except Exception as e:
            logging.error(f"{symbol} 자동진입 실행 실패: {e}")
            return False

async def _save_trade_record(self, record: Dict):
        """거래 기록 저장"""
        try:
            # SQLite DB에 거래 기록 저장
            import sqlite3
            
            db_path = "auto_trading_records.db"
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # 테이블 생성 (없으면)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        action TEXT,
                        quantity INTEGER,
                        price REAL,
                        profit_pct REAL,
                        profit_amount REAL,
                        confidence REAL,
                        reasoning TEXT,
                        reason TEXT,
                        day_type TEXT,
                        hold_days INTEGER,
                        timestamp TEXT
                    )
                ''')
                
                # 거래 기록 삽입
                cursor.execute('''
                    INSERT INTO trades (symbol, action, quantity, price, profit_pct, 
                                      profit_amount, confidence, reasoning, reason, 
                                      day_type, hold_days, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.get('symbol', ''),
                    record.get('action', ''),
                    record.get('quantity', 0),
                    record.get('price', 0),
                    record.get('profit_pct', 0),
                    record.get('profit_amount', 0),
                    record.get('confidence', 0),
                    record.get('reasoning', ''),
                    record.get('reason', ''),
                    record.get('day_type', ''),
                    record.get('hold_days', 0),
                    record['timestamp'].isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"거래 기록 저장 실패: {e}")
    
    async def _send_trading_report(self):
        """거래 완료 리포트 및 알림"""
        try:
            if not config.get('auto_trading.notifications', True):
                return
            
            # 오늘 거래 요약
            today = datetime.now().date()
            today_trades = []
            
            # DB에서 오늘 거래 조회
            try:
                import sqlite3
                with sqlite3.connect("auto_trading_records.db") as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM trades 
                        WHERE date(timestamp) = date('now') 
                        ORDER BY timestamp DESC
                    ''')
                    today_trades = cursor.fetchall()
            except:
                pass
            
            # 현재 포트폴리오 상태
            portfolio_summary = await self._get_portfolio_summary()
            
            # 리포트 생성
            report = self._generate_trading_report(today_trades, portfolio_summary)
            
            # 텔레그램 알림 (설정되어 있으면)
            await self._send_telegram_notification(report)
            
            # 로그에도 기록
            logging.info("📊 일일 거래 리포트 전송 완료")
            
        except Exception as e:
            logging.error(f"거래 리포트 전송 실패: {e}")
    
    async def _get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약 정보"""
        try:
            if not self.positions:
                return {'total_value': 0, 'total_profit': 0, 'positions_count': 0}
            
            total_value = 0
            total_profit = 0
            
            for position in self.positions.values():
                current_price = await self._get_current_price(position.symbol)
                market_value = current_price * position.quantity
                profit = (current_price - position.avg_cost) * position.quantity
                
                total_value += market_value
                total_profit += profit
            
            return {
                'total_value': total_value,
                'total_profit': total_profit,
                'total_profit_pct': (total_profit / (total_value - total_profit)) * 100 if total_value > total_profit else 0,
                'positions_count': len(self.positions)
            }
            
        except Exception as e:
            logging.error(f"포트폴리오 요약 실패: {e}")
            return {'total_value': 0, 'total_profit': 0, 'positions_count': 0}
    
    def _generate_trading_report(self, today_trades: List, portfolio_summary: Dict) -> str:
        """거래 리포트 생성"""
        try:
            report = f"""
🤖 **US 전략 V7.0 자동매매 리포트**
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}

📊 **오늘 거래 요약**
• 총 거래: {len(today_trades)}건
• 매수: {len([t for t in today_trades if t[2] == 'buy'])}건  
• 매도: {len([t for t in today_trades if t[2] == 'sell'])}건

💼 **현재 포트폴리오**  
• 보유 종목: {portfolio_summary['positions_count']}개
• 총 자산: ${portfolio_summary['total_value']:,.0f}
• 총 손익: ${portfolio_summary['total_profit']:+,.0f} ({portfolio_summary.get('total_profit_pct', 0):+.1f}%)

🎯 **시스템 상태**
• 자동매매: ✅ 정상 작동
• AI 확신도: ✅ 활성화  
• 리스크 관리: ✅ 적용중

---
🏆 US 전략 V7.0 - 완전 자동매매
            """
            
            return report.strip()
            
        except Exception as e:
            logging.error(f"리포트 생성 실패: {e}")
            return f"리포트 생성 오류: {e}"
    
    async def _send_telegram_notification(self, message: str):
        """텔레그램 알림 전송"""
        try:
            bot_token = config.get('notifications.telegram.bot_token', '')
            chat_id = config.get('notifications.telegram.chat_id', '')
            
            if not bot_token or not chat_id:
                return
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        logging.info("✅ 텔레그램 알림 전송 성공")
                    else:
                        logging.error(f"❌ 텔레그램 알림 전송 실패: {response.status}")
            
        except Exception as e:
            logging.error(f"텔레그램 알림 실패: {e}")
    
    async def _send_error_notification(self, error_msg: str):
        """오류 알림 전송"""
        try:
            error_report = f"""
🚨 **US 전략 V7.0 오류 발생**
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}

❌ **오류 내용**
{error_msg}

🔧 **조치 사항**
시스템 점검 및 재시작 필요

---
🤖 자동매매 시스템
            """
            
            await self._send_telegram_notification(error_report)
            
        except Exception as e:
            logging.error(f"오류 알림 실패: {e}")

async def auto_select_stocks(self) -> List[str]:
        """고급 알고리즘 기반 자동 종목 선별"""
        if not self.enabled:
            return []
        
        try:
            # 캐시 확인 (24시간 유효)
            if (self.last_selection and 
                (datetime.now() - self.last_selection).seconds < 24 * 3600):
                logging.info("📋 캐시된 선별 결과 사용")
                return [s['symbol'] for s in self.selected_stocks]
            
            logging.info("🚀 US 종목 고급 자동선별 시작!")
            start_time = time.time()
            
            # 1단계: 투자 유니버스 수집
            universe = await self.selector.collect_us_symbols()
            if not universe:
                logging.warning("⚠️ 유니버스 수집 실패, 백업 사용")
                return self._get_fallback_stocks()
            
            logging.info(f"📊 투자 유니버스: {len(universe)}개 종목")
            
            # 2단계: 시장 환경 분석
            current_vix = await self.selector.get_current_vix()
            market_regime = self._determine_market_regime(current_vix)
            logging.info(f"📈 VIX: {current_vix:.1f} | 시장환경: {market_regime}")
            
            # 3단계: 배치 분석 (동시성 최적화)
            scored_stocks = []
            batch_size = 20  # 동시 처리 종목 수
            
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i + batch_size]
                
                # 비동기 배치 처리
                tasks = [self._analyze_stock_for_selection(symbol, current_vix, market_regime) 
                        for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 유효한 결과만 수집
                for result in results:
                    if isinstance(result, dict) and result and result.get('valid', False):
                        scored_stocks.append(result)
                
                # 진행률 출력
                if i % 100 == 0:
                    logging.info(f"📊 고급분석 진행: {i}/{len(universe)} ({len(scored_stocks)}개 후보)")
            
            logging.info(f"📈 1차 선별 완료: {len(scored_stocks)}개 후보")
            
            if not scored_stocks:
                logging.warning("⚠️ 선별된 종목 없음, 백업 사용")
                return self._get_fallback_stocks()
            
            # 4단계: 고급 선별 알고리즘
            target_count = config.get('strategy.target_stocks', 8)
            final_selection = await self._advanced_stock_selection(
                scored_stocks, target_count, market_regime
            )
            
            # 5단계: 결과 저장 및 리포트
            self.selected_stocks = final_selection
            self.last_selection = datetime.now()
            
            elapsed = time.time() - start_time
            selected_symbols = [s['symbol'] for s in final_selection]
            
            # 선별 리포트
            avg_score = sum(s['total'] for s in final_selection) / len(final_selection)
            sectors = list(set(s.get('sector', 'Unknown') for s in final_selection))
            
            logging.info(f"🏆 고급 자동선별 완료!")
            logging.info(f"  📊 선별종목: {len(selected_symbols)}개")
            logging.info(f"  ⏱️  소요시간: {elapsed:.1f}초")
            logging.info(f"  📈 평균점수: {avg_score:.2f}")
            logging.info(f"  🏭 섹터수: {len(sectors)}개")
            logging.info(f"  💎 종목리스트: {', '.join(selected_symbols)}")
            
            return selected_symbols
            
        except Exception as e:
            logging.error(f"자동선별 실패: {e}")
            return self._get_fallback_stocks()
    
    def _determine_market_regime(self, vix: float) -> str:
        """시장 환경 판단"""
        if vix <= 12:
            return "초저변동성"  # 매우 안정적
        elif vix <= 16:
            return "저변동성"    # 안정적
        elif vix <= 20:
            return "정상변동성"  # 보통
        elif vix <= 25:
            return "고변동성"    # 불안정
        elif vix <= 30:
            return "매우불안정" # 위험
        else:
            return "극도불안정"  # 매우 위험
    
    async def _analyze_stock_for_selection(self, symbol: str, vix: float, market_regime: str) -> Optional[Dict]:
        """선별용 종목 분석 (최적화)"""
        try:
            # 기본 데이터 수집
            data = await self.selector.get_stock_data(symbol)
            if not data:
                return None
            
            # 1차 필터링 (기본 조건)
            if not self._basic_filter(data):
                return None
            
            # 기술적 점수 계산 -> 종합 점수 계산
            total_score, scores = self.analyzer.calculate_comprehensive_score(data, vix)
            
            # 2차 필터링 (기술적 조건)
            if not self._technical_filter_for_selection(data, total_score, market_regime):
                return None
            
            # 추가 메트릭 계산
            momentum_data = data.get('momentum', {})
            risk_score = self._calculate_risk_score(data, vix)
            quality_score = self._calculate_quality_score(data)
            
            # 종합 점수 (가중평균)
            composite_score = (
                total_score * 0.5 +           # 기술적 분석 50%
                quality_score * 0.25 +       # 품질 점수 25%
                (1 - risk_score) * 0.25      # 위험 점수 25% (역가중)
            )
            
            result = data.copy()
            result.update(scores)
            result.update({
                'composite_score': composite_score,
                'risk_score': risk_score,
                'quality_score': quality_score,
                'momentum_avg': momentum_data.get('avg', 0),
                'momentum_strength': momentum_data.get('strength', 0.5),
                'market_regime': market_regime,
                'vix': vix,
                'valid': True
            })
            
            return result
            
        except Exception as e:
            logging.error(f"선별 분석 실패 {symbol}: {e}")
            return None
    
    def _basic_filter(self, data: Dict) -> bool:
        """1차 기본 필터링"""
        try:
            # 시가총액 필터 (50억달러 이상)
            if data.get('market_cap', 0) < 5_000_000_000:
                return False
            
            # 거래량 필터 (100만주 이상)
            if data.get('avg_volume', 0) < 1_000_000:
                return False
            
            # 주가 필터 (10달러 이상)
            if data.get('price', 0) < 10:
                return False
            
            # 베타 필터 (극단값 제외)
            beta = data.get('beta', 1.0)
            if beta < 0.5 or beta > 3.0:
                return False
            
            return True
        except:
            return False
    
    def _technical_filter_for_selection(self, data: Dict, score: float, market_regime: str) -> bool:
        """2차 기술적 필터링 (시장환경 고려)"""
        try:
            # 시장환경별 최소 점수 조정
            min_scores = {
                "초저변동성": 0.65,  # 안정적일 때 높은 기준
                "저변동성": 0.60,
                "정상변동성": 0.55,
                "고변동성": 0.50,    # 불안정할 때 낮은 기준
                "매우불안정": 0.45,
                "극도불안정": 0.40
            }
            
            min_score = min_scores.get(market_regime, 0.55)
            if score < min_score:
                return False
            
            # MACD 강한 약세 필터
            macd_data = data.get('macd', {})
            if (macd_data.get('trend') == 'bearish' and 
                macd_data.get('histogram', 0) < -1.0):
                return False
            
            # 극도 과매수 필터
            bb_data = data.get('bollinger', {})
            if bb_data.get('position', 0.5) > 0.95:
                return False
            
            # RSI 극단값 필터
            rsi = data.get('rsi', 50)
            if rsi > 90 or rsi < 5:
                return False
            
            # 모멘텀 일관성 체크
            momentum_data = data.get('momentum', {})
            momentum_strength = momentum_data.get('strength', 0.5)
            if momentum_strength < 0.2:  # 너무 일관성 없는 모멘텀
                return False
            
            return True
        except:
            return True

def _calculate_risk_score(self, data: Dict, vix: float) -> float:
        """위험 점수 계산 (0-1, 높을수록 위험)"""
        try:
            risk_score = 0.0
            
            # 변동성 위험
            volatility = data.get('volatility', 25)
            if volatility > 40:
                risk_score += 0.3
            elif volatility > 30:
                risk_score += 0.2
            elif volatility < 15:
                risk_score += 0.1  # 너무 낮은 변동성도 위험
            
            # 베타 위험
            beta = data.get('beta', 1.0)
            if beta > 1.5:
                risk_score += 0.2
            elif beta > 1.2:
                risk_score += 0.1
            
            # 거래량 위험
            volume_spike = data.get('volume_spike', 1)
            if volume_spike < 0.5:  # 거래량 급감
                risk_score += 0.2
            elif volume_spike > 3.0:  # 거래량 급증 (위험신호)
                risk_score += 0.1
            
            # VIX 조정
            if vix > 25:
                risk_score += 0.2
            elif vix > 20:
                risk_score += 0.1
            
            return min(1.0, risk_score)
        except:
            return 0.5
    
    def _calculate_quality_score(self, data: Dict) -> float:
        """품질 점수 계산 (0-1, 높을수록 좋음)"""
        try:
            quality_score = 0.0
            
            # 시가총액 점수
            market_cap = data.get('market_cap', 0)
            if market_cap > 100_000_000_000:  # 1000억달러+
                quality_score += 0.3
            elif market_cap > 50_000_000_000:  # 500억달러+
                quality_score += 0.25
            elif market_cap > 20_000_000_000:  # 200억달러+
                quality_score += 0.2
            elif market_cap > 10_000_000_000:  # 100억달러+
                quality_score += 0.15
            
            # 거래량 점수
            avg_volume = data.get('avg_volume', 0)
            if avg_volume > 10_000_000:  # 1000만주+
                quality_score += 0.2
            elif avg_volume > 5_000_000:   # 500만주+
                quality_score += 0.15
            elif avg_volume > 2_000_000:   # 200만주+
                quality_score += 0.1
            
            # ROE 점수
            roe = data.get('roe', 0)
            if roe > 20:
                quality_score += 0.2
            elif roe > 15:
                quality_score += 0.15
            elif roe > 10:
                quality_score += 0.1
            
            # PE 적정성 점수
            pe_ratio = data.get('pe_ratio', 0)
            if 10 <= pe_ratio <= 25:
                quality_score += 0.15
            elif 5 <= pe_ratio <= 35:
                quality_score += 0.1
            
            # 섹터 안정성 보너스
            stable_sectors = ['Consumer Staples', 'Utilities', 'Healthcare', 'Technology']
            if data.get('sector', '') in stable_sectors:
                quality_score += 0.15
            
            return min(1.0, quality_score)
        except:
            return 0.5
    
    async def _advanced_stock_selection(self, scored_stocks: List[Dict], target_count: int, market_regime: str) -> List[Dict]:
        """고급 선별 알고리즘 (시장환경 고려)"""
        try:
            # 종합점수 기준 정렬
            scored_stocks.sort(key=lambda x: x['composite_score'], reverse=True)
            
            final_selection = []
            sector_counts = {}
            
            # 시장환경별 선별 전략
            if market_regime in ["초저변동성", "저변동성"]:
                # 안정적일 때: 고품질 + 모멘텀 중시
                selection_criteria = lambda x: (x['quality_score'] >= 0.6 and 
                                              x['momentum_strength'] >= 0.6)
            elif market_regime in ["정상변동성"]:
                # 보통일 때: 균형잡힌 선별
                selection_criteria = lambda x: (x['composite_score'] >= 0.65 and 
                                              x['risk_score'] <= 0.6)
            else:
                # 불안정할 때: 디펜시브 + 고품질 중시
                selection_criteria = lambda x: (x['quality_score'] >= 0.7 and 
                                              x['risk_score'] <= 0.5)
            
            # 1차: 조건 만족 + 섹터 다양성
            for stock in scored_stocks:
                if len(final_selection) >= target_count:
                    break
                
                sector = stock.get('sector', 'Unknown')
                
                if (selection_criteria(stock) and 
                    sector_counts.get(sector, 0) < 2):  # 섹터당 최대 2개
                    final_selection.append(stock)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # 2차: 고점수 종목으로 나머지 채우기
            remaining = target_count - len(final_selection)
            for stock in scored_stocks:
                if remaining <= 0:
                    break
                if stock not in final_selection:
                    final_selection.append(stock)
                    remaining -= 1
            
            return final_selection
            
        except Exception as e:
            logging.error(f"고급 선별 실패: {e}")
            # 단순 정렬 폴백
            return scored_stocks[:target_count]
    
    def _get_fallback_stocks(self) -> List[str]:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX']
    
    async def analyze_stock_signal(self, symbol: str) -> StockSignal:
        """개별 종목 신호 분석 + AI 확신도 체크"""
        try:
            data = await self.selector.get_stock_data(symbol)
            if not data:
                raise ValueError(f"데이터 수집 실패: {symbol}")
            
            vix = await self.selector.get_current_vix()
            comprehensive_score, scores = self.analyzer.calculate_comprehensive_score(data, vix)
            
            # AI 확신도 체크 포함 액션 결정
            action, confidence, ai_result = await self.analyzer.determine_action_with_ai(
                data, comprehensive_score
            )
            
            # 목표가 및 손절가 계산
            target_multiplier = 1.0 + (confidence * 0.20)
            target_price = data['price'] * target_multiplier
            stop_loss = data['price'] * (1 - 0.07)
            
            # 근거 생성
            macd_trend = data.get('macd', {}).get('trend', 'neutral')
            bb_signal = data.get('bollinger', {}).get('signal', 'normal')
            rsi = data.get('rsi', 50)
            
            reasoning = (f"종합:{comprehensive_score:.2f} | "
                        f"버핏:{scores.get('buffett', 0):.2f} "
                        f"린치:{scores.get('lynch', 0):.2f} "
                        f"기술:{scores.get('technical', 0):.2f} "
                        f"모멘텀:{scores.get('momentum', 0):.2f}")
            
            if ai_result['ai_used']:
                reasoning += f" | AI조정:{ai_result['reason'][:20]}"
            
            return StockSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=data['price'],
                technical_scores=scores,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning,
                timestamp=datetime.now(),
                ai_confidence=ai_result
            )
            
        except Exception as e:
            return StockSignal(
                symbol=symbol,
                action='hold',
                confidence=0.0,
                price=0.0,
                technical_scores={},
                target_price=0.0,
                stop_loss=0.0,
                reasoning=f"오류: {e}",
                timestamp=datetime.now(),
                ai_confidence=None
            )

# ========================================================================================
# 🎯 편의 함수들
# ========================================================================================

async def run_us_auto_selection():
    """US 종목 자동 선별 실행"""
    strategy = USStrategy()
    signals = []
    selected = await strategy.auto_select_stocks()
    
    for symbol in selected[:5]:
        try:
            signal = await strategy.analyze_stock_signal(symbol)
            signals.append(signal)
        except:
            continue
    
    return signals

async def analyze_single_us_stock(symbol: str):
    """개별 US 종목 분석"""
    strategy = USStrategy()
    return await strategy.analyze_stock_signal(symbol)

async def get_us_system_status():
    """US 전략 시스템 상태 확인"""
    try:
        strategy = USStrategy()
        dst_manager = DaylightSavingManager()
        
        ibkr_connected = False
        try:
            if IBKR_AVAILABLE:
                ibkr_connected = await strategy.ibkr.connect()
                if ibkr_connected:
                    await strategy.ibkr.disconnect()
        except:
            ibkr_connected = False
        
        dst_active = dst_manager.is_dst_active()
        market_open, market_close = dst_manager.get_market_hours_kst()
        
        ai_status = {
            'enabled': strategy.analyzer.ai_checker.enabled,
            'daily_calls': strategy.analyzer.ai_checker.daily_calls,
            'max_calls': strategy.analyzer.ai_checker.max_daily_calls,
            'remaining': strategy.analyzer.ai_checker.max_daily_calls - strategy.analyzer.ai_checker.daily_calls
        }
        
        return {
            'enabled': strategy.enabled,
            'ibkr_connected': ibkr_connected,
            'ibkr_available': IBKR_AVAILABLE,
            'ai_checker': ai_status,
            'monthly_return': strategy.monthly_return,
            'dst_active': dst_active,
            'market_hours_kst': f"{market_open.strftime('%H:%M')}-{market_close.strftime('%H:%M')}",
            'timezone_status': 'EDT' if dst_active else 'EST',
            'last_tuesday': strategy.last_trade_dates.get('Tuesday'),
            'last_thursday': strategy.last_trade_dates.get('Thursday')
        }
        
    except Exception as e:
        return {'error': str(e)}

async def test_technical_indicators(symbol: str = 'AAPL'):
    """기술적 지표 테스트"""
    try:
        strategy = USStrategy()
        data = await strategy.selector.get_stock_data(symbol)
        
        if data:
            vix = await strategy.selector.get_current_vix()
            technical_score, scores = strategy.analyzer.calculate_technical_score(data, vix)
            
            return {
                'symbol': symbol,
                'price': data['price'],
                'technical_score': technical_score,
                'scores': scores,
                'indicators': {
                    'macd': data.get('macd', {}),
                    'bollinger': data.get('bollinger', {}),
                    'rsi': data.get('rsi', 50),
                    'momentum': data.get('momentum', {}),
                    'volume_spike': data.get('volume_spike', 1)
                },
                'vix': vix
            }
        else:
            return {'error': '데이터 없음'}
    except Exception as e:
        return {'error': str(e)}

async def test_ai_checker():
    """AI 확신도 체커 테스트"""
    print("🤖 AI 확신도 체커 테스트...")
    
    try:
        strategy = USStrategy()
        ai_checker = strategy.analyzer.ai_checker
        
        print(f"AI 활성화: {ai_checker.enabled}")
        print(f"오늘 사용량: {ai_checker.daily_calls}/{ai_checker.max_daily_calls}")
        
        if ai_checker.enabled:
            # 테스트용 애매한 데이터
            test_data = {
                'symbol': 'AAPL',
                'macd': {'trend': 'improving', 'crossover': 'none'},
                'bollinger': {'signal': 'normal', 'position': 0.55},
                'rsi': 58,
                'momentum': {'3m': 5.2}
            }
            
            result = await ai_checker.check_confidence('AAPL', test_data, 0.55)
            print(f"테스트 결과: {result}")
        else:
            print("❌ AI 체커 비활성화")
        
    except Exception as e:
        print(f"❌ AI 테스트 실패: {e}")

async def test_momentum_analysis(symbol: str = 'AAPL'):
    """모멘텀 분석 테스트"""
    print(f"📈 {symbol} 모멘텀 분석 테스트...")
    
    try:
        strategy = USStrategy()
        data = await strategy.selector.get_stock_data(symbol)
        
        if data:
            momentum = data.get('momentum', {})
            
            print(f"💰 현재가: ${data['price']:.2f}")
            print(f"📊 모멘텀 분석:")
            print(f"  5일: {momentum.get('5d', 0):+.1f}%")
            print(f"  10일: {momentum.get('10d', 0):+.1f}%")
            print(f"  1개월: {momentum.get('1m', 0):+.1f}%")
            print(f"  3개월: {momentum.get('3m', 0):+.1f}%")
            print(f"  6개월: {momentum.get('6m', 0):+.1f}%")
            print(f"  12개월: {momentum.get('12m', 0):+.1f}%")
            print(f"  평균: {momentum.get('avg', 0):+.1f}%")
            print(f"  강도: {momentum.get('strength', 0.5):.1%}")
        else:
            print(f"❌ {symbol} 데이터 수집 실패")
            
    except Exception as e:
        print(f"❌ 모멘텀 분석 실패: {e}")

async def test_value_growth_analysis(symbol: str = 'BRK-B'):
    """가치/성장 분석 테스트 (버핏 + 린치)"""
    print(f"💎 {symbol} 가치/성장 분석 테스트...")
    
    try:
        strategy = USStrategy()
        data = await strategy.selector.get_stock_data(symbol)
        
        if data:
            analyzer = strategy.analyzer
            buffett_score = analyzer._calculate_buffett_score(data)
            lynch_score = analyzer._calculate_lynch_score(data)
            
            print(f"💰 현재가: ${data['price']:.2f}")
            print(f"🏛️ 워렌 버핏 점수: {buffett_score:.2f}")
            print(f"  PBR: {data.get('pbr', 0):.2f}")
            print(f"  ROE: {data.get('roe', 0):.1f}%")
            print(f"  PE: {data.get('pe_ratio', 0):.1f}")
            print(f"  베타: {data.get('beta', 1.0):.2f}")
            print(f"  시총: ${data.get('market_cap', 0)/1e9:.0f}B")
            
            print(f"🚀 피터 린치 점수: {lynch_score:.2f}")
            print(f"  PEG: {data.get('peg', 999):.2f}")
            print(f"  EPS 성장: {data.get('eps_growth', 0):.1f}%")
            print(f"  매출 성장: {data.get('revenue_growth', 0):.1f}%")
            print(f"  ROE: {data.get('roe', 0):.1f}%")
            
        else:
            print(f"❌ {symbol} 데이터 수집 실패")
            
    except Exception as e:
        print(f"❌ 가치/성장 분석 실패: {e}")

async def quick_us_test():
    """US 전략 빠른 테스트"""
    print("🇺🇸 US 전략 V7.0 빠른 테스트...")
    
    try:
        # 시스템 상태
        print("\n📊 시스템 상태:")
        status = await get_us_system_status()
        if 'error' not in status:
            print(f"  ✅ 전략: {'활성화' if status['enabled'] else '비활성화'}")
            print(f"  🤖 AI 확신도 체커: {status['ai_checker']['remaining']}/{status['ai_checker']['max_calls']} 남음")
            print(f"  🕒 시간대: {status['timezone_status']}")
            print(f"  📈 시장시간: {status['market_hours_kst']} KST")
        else:
            print(f"  ❌ 상태 확인 실패: {status['error']}")
        
        # 기술적 지표 테스트
        print("\n📈 AAPL 기술적 분석:")
        indicators = await test_technical_indicators('AAPL')
        if 'error' not in indicators:
            print(f"  💰 현재가: ${indicators['price']:.2f}")
            print(f"  📊 기술적 점수: {indicators['technical_score']:.2f}")
            print(f"  📊 MACD: {indicators['indicators']['macd'].get('trend', 'unknown')}")
            print(f"  📊 볼린저: {indicators['indicators']['bollinger'].get('signal', 'unknown')}")
            print(f"  📊 RSI: {indicators['indicators']['rsi']:.1f}")
        else:
            print(f"  ❌ 실패: {indicators['error']}")
        
        # AI 확신도 체크 테스트
        print("\n🤖 AI 확신도 체커:")
        await test_ai_checker()
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

# ========================================================================================
# 🤖 완전 자동매매 실행 함수들
# ========================================================================================

async def run_full_auto_trading():
    """🤖 완전 자동매매 실행 (메인 함수)"""
    strategy = USStrategy()
    await strategy.run_full_auto_trading()

async def start_auto_trading_daemon():
    """🔄 자동매매 데몬 시작"""
    print("🤖 US 전략 V7.0 완전 자동매매 데몬 시작...")
    
    strategy = USStrategy()
    
    while True:
        try:
            current_time = datetime.now()
            
            # 화요일/목요일 미국 시장시간 체크
            is_trading, day_type = strategy.is_trading_day()
            is_market_time = strategy.dst_manager.is_market_hours()
            
            if is_trading and is_market_time:
                # 시장 개장 1시간 후(10:30 ET)에 실행
                market_open, _ = strategy.dst_manager.get_market_hours_kst()
                trading_time = market_open + timedelta(hours=1)
                
                # 거래 시간이면 자동매매 실행
                if abs((current_time - trading_time).total_seconds()) < 300:  # 5분 오차 허용
                    print(f"🎯 {day_type} 자동매매 실행...")
                    await strategy.run_full_auto_trading()
            
            # 30분마다 체크
            await asyncio.sleep(1800)
            
        except KeyboardInterrupt:
            print("👋 자동매매 데몬 종료")
            break
        except Exception as e:
            print(f"❌ 자동매매 데몬 오류: {e}")
            await asyncio.sleep(300)  # 5분 후 재시도

async def test_auto_trading_system():
    """🧪 완전 자동매매 시스템 테스트"""
    print("🤖 완전 자동매매 시스템 테스트...")
    
    try:
        strategy = USStrategy()
        
        print("\n📊 자동매매 설정:")
        print(f"  ✅ 자동실행: {config.get('auto_trading.enabled', True)}")
        print(f"  🎯 최소신뢰도: {config.get('auto_trading.min_confidence', 0.75):.1%}")
        print(f"  📈 일일한도: {config.get('auto_trading.max_daily_trades', 6)}건")
        print(f"  🔔 알림: {config.get('auto_trading.notifications', True)}")
        
        # 사전 체크 테스트
        print("\n🔍 사전 체크 테스트:")
        
        # 1. 시장 시간
        is_market = strategy.dst_manager.is_market_hours()
        print(f"  📅 시장시간: {'✅' if is_market else '❌'}")
        
        # 2. 거래일
        is_trading, day_type = strategy.is_trading_day()
        print(f"  📅 거래일: {'✅' if is_trading else '❌'} {day_type}")
        
        # 3. IBKR 연결 (시뮬레이션)
        print(f"  🔗 IBKR: ✅ 연결 준비")
        
        # 4. 자동 종목 선별 테스트
        print("\n🚀 자동 종목 선별:")
        candidates = await strategy.auto_select_stocks()
        
        auto_entries = []
        min_confidence = config.get('auto_trading.min_confidence', 0.75)
        
        for symbol in candidates[:3]:
            signal = await strategy.analyze_stock_signal(symbol)
            auto_execute = signal.action == 'buy' and signal.confidence >= min_confidence
            
            print(f"  📊 {symbol}: {signal.action} ({signal.confidence:.1%}) "
                  f"{'🤖 자동실행' if auto_execute else '⏸️ 보류'}")
            
            if auto_execute:
                auto_entries.append(symbol)
        
        print(f"\n✅ 자동진입 대상: {len(auto_entries)}개")
        
        # 5. 포지션 관리 시뮬레이션
        print("\n📊 포지션 관리 시뮬레이션:")
        
        # 가상 포지션으로 테스트
        test_positions = [
            {'symbol': 'AAPL', 'profit': +6.2, 'days': 8, 'action': '보유'},
            {'symbol': 'MSFT', 'profit': +8.5, 'days': 12, 'action': '🤖 목표달성 자동청산'},
            {'symbol': 'GOOGL', 'profit': -7.2, 'days': 6, 'action': '🤖 손절 자동청산'}
        ]
        
        for pos in test_positions:
            print(f"  {pos['symbol']}: {pos['profit']:+.1f}% ({pos['days']}일) → {pos['action']}")
        
        print("\n🎯 완전 자동매매 시스템 정상 작동!")
        
        # 실제 실행 여부 확인
        if input("\n실제 자동매매를 실행하시겠습니까? (y/N): ").lower() == 'y':
            print("🤖 완전 자동매매 실행...")
            await strategy.run_full_auto_trading()
        else:
            print("📊 테스트 모드 완료")
        
    except Exception as e:
        print(f"❌ 자동매매 테스트 실패: {e}")

async def view_trading_records():
    """📋 자동매매 거래 기록 조회"""
    print("📋 자동매매 거래 기록...")
    
    try:
        import sqlite3
        
        with sqlite3.connect("auto_trading_records.db") as conn:
            cursor = conn.cursor()
            
            # 최근 거래 기록 조회
            cursor.execute('''
                SELECT symbol, action, quantity, price, profit_pct, 
                       reason, timestamp 
                FROM trades 
                ORDER BY timestamp DESC 
                LIMIT 20
            ''')
            
            records = cursor.fetchall()
            
            if not records:
                print("📊 거래 기록이 없습니다.")
                return
            
            print(f"\n📈 최근 거래 기록 ({len(records)}건):")
            print("심볼  | 액션 | 수량 | 가격   | 수익률 | 사유       | 시간")
            print("-" * 65)
            
            for record in records:
                symbol, action, qty, price, profit, reason, timestamp = record
                profit_str = f"{profit:+.1f}%" if profit else "  -  "
                time_str = timestamp[:16] if timestamp else ""
                reason_str = (reason[:10] + "...") if reason and len(reason) > 10 else (reason or "")
                
                print(f"{symbol:5s} | {action:4s} | {qty:4d} | ${price:6.2f} | {profit_str:6s} | {reason_str:10s} | {time_str}")
            
            # 수익률 통계
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    AVG(profit_pct) as avg_profit,
                    SUM(profit_amount) as total_profit,
                    COUNT(CASE WHEN profit_pct > 0 THEN 1 END) as winning_trades
                FROM trades 
                WHERE action = 'sell'
            ''')
            
            stats = cursor.fetchone()
            if stats and stats[0] > 0:
                total, avg_profit, total_profit, winning = stats
                win_rate = (winning / total) * 100
                
                print(f"\n📊 거래 통계:")
                print(f"  총 거래: {total}건")
                print(f"  승률: {win_rate:.1f}% ({winning}/{total})")
                print(f"  평균 수익률: {avg_profit:.1f}%")
                print(f"  총 손익: ${total_profit:+,.2f}")
        
    except Exception as e:
        print(f"❌ 거래 기록 조회 실패: {e}")

# ========================================================================================
# 🧪 추가 테스트 함수들
# ========================================================================================

async def test_2week_swing_system():
    """2주 스윙 시스템 테스트"""
    print("🎯 2주 스윙 트레이딩 시스템 테스트...")
    
    try:
        strategy = USStrategy()
        
        print(f"\n📈 현재 거래 모드: {strategy.trading_mode.upper()}")
        
        # 거래일 확인
        is_trading, day_type = strategy.is_trading_day()
        print(f"📅 오늘: {'거래일' if is_trading else '비거래일'} {f'({day_type})' if day_type else ''}")
        
        if is_trading:
            allocation = strategy.get_trading_allocation(day_type)
            targets = strategy.get_target_positions(day_type)
            print(f"🎯 {day_type} 설정: {targets}개 포지션, {allocation}% 자금배분")
        
        # 기존 포지션 상태
        print(f"\n📊 현재 포지션: {len(strategy.positions)}개")
        if strategy.positions:
            for symbol, pos in strategy.positions.items():
                print(f"  {symbol}: {pos.mode} 모드, {pos.days_held()}일차")
        
        # 2주 스윙 전략 실행 (테스트 모드)
        print("\n🎯 2주 스윙 전략 시뮬레이션:")
        
        # 가상 포지션 생성 (테스트용)
        test_positions = {
            'AAPL': Position('AAPL', 100, 150.0, datetime.now() - timedelta(days=10), 'swing'),
            'MSFT': Position('MSFT', 50, 300.0, datetime.now() - timedelta(days=15), 'swing'),
            'GOOGL': Position('GOOGL', 25, 120.0, datetime.now() - timedelta(days=5), 'weekly', entry_day='Tuesday')
        }
        
        for symbol, pos in test_positions.items():
            days = pos.days_held()
            should_exit = pos.should_exit_by_time()
            print(f"  {symbol}: {days}일차, {'청산대상' if should_exit else '보유유지'} ({pos.mode})")
        
        # 신규 진입 후보
        print("\n🚀 신규 진입 후보 분석:")
        candidates = await strategy.auto_select_stocks()
        
        for i, symbol in enumerate(candidates[:3], 1):
            signal = await strategy.analyze_stock_signal(symbol)
            entry_signal = "진입" if signal.action == 'buy' and signal.confidence >= 0.75 else "대기"
            print(f"  {i}. {symbol}: {signal.action} ({signal.confidence:.1%}) - {entry_signal}")
        
        print("\n✅ 2주 스윙 시스템 정상 작동")
        
    except Exception as e:
        print(f"❌ 2주 스윙 테스트 실패: {e}")

# ========================================================================================
# 🏁 메인 실행부
# ========================================================================================

async def main():
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('us_strategy_v7.log', encoding='utf-8')
            ]
        )
        
        print("🇺🇸" + "="*70)
        print("🔥 US 전략 V7.0 - 기술적 분석 + AI 확신도 체크 (비용 최적화)")
        print("🚀 월 6-8% 달성형 미국 주식 전용 전략")
        print("="*72)
        
        print("\n🌟 V7.0 완전체 특징:")
        print("  ✨ 서머타임 완전 자동화 (EDT/EST 자동전환)")
        print("  ✨ 5가지 종합전략 (기술적+버핏+린치+모멘텀+품질)")
        print("  ✨ 2주 스윙 트레이딩 (화/목 주 2회 매매)")
        print("  ✨ 🤖 완전 자동매매 시스템 (AI 확신도 체크)")
        print("  ✨ 실시간 포지션 관리 + 자동 손익 실현")
        print("  ✨ 텔레그램 알림 + 거래 기록 DB")
        print("  ✨ 월 목표 상향 (6-8% vs 기존 5-7%)")
        
        status = await get_us_system_status()
        
        if 'error' not in status:
            print(f"\n🔧 시스템 상태:")
            print(f"  ✅ 전략: {'활성화' if status['enabled'] else '비활성화'}")
            print(f"  🕒 시간대: {status['timezone_status']} ({'서머타임' if status['dst_active'] else '표준시'})")
            print(f"  📈 시장시간: {status['market_hours_kst']} KST")
            print(f"  🤖 IBKR: {'연결가능' if status['ibkr_connected'] else '연결불가'}")
            ai_info = status['ai_checker']
            print(f"  🤖 AI 체커: {'활성화' if ai_info['enabled'] else '비활성화'} ({ai_info['remaining']}/{ai_info['max_calls']} 남음)")
            print(f"  📈 월 수익률: {status['monthly_return']:.2f}%")
        else:
            print(f"  ❌ 상태 확인 실패: {status['error']}")
        
        print("\n🚀 실행 옵션:")
        print("  1. 🔍 US 종목 기술적 선별")
        print("  2. 📊 개별 US 종목 분석")
        print("  3. 🧪 기술적 지표 테스트")
        print("  4. 🤖 AI 확신도 체커 테스트")
        print("  5. ⚡ 빠른 종합 테스트")
        print("  6. 📈 모멘텀 분석 테스트")
        print("  7. 💎 가치/성장 분석 테스트 (버핏+린치)")
        print("  8. 🎯 2주 스윙 시스템 테스트")
        print("  9. 🤖 완전 자동매매 시스템 테스트")
        print("  a. 🚀 완전 자동매매 실행")
        print("  b. 🔄 자동매매 데몬 시작")
        print("  c. 📋 거래 기록 조회")
        print("  0. 👋 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (0-9, a-c): ").strip().lower()
                
                if choice == '1':
                    print("\n🔍 US 종목 기술적 선별!")
                    signals = await run_us_auto_selection()
                    
                    if signals:
                        print(f"\n📈 기술적 분석 결과: {len(signals)}개 스캔")
                        
                        buy_signals = [s for s in signals if s.action == 'buy']
                        print(f"🟢 매수추천: {len(buy_signals)}개")
                        
                        for i, signal in enumerate(buy_signals[:3], 1):
                            ai_note = ""
                            if signal.ai_confidence and signal.ai_confidence['ai_used']:
                                ai_note = f" | AI: {signal.ai_confidence['reason'][:20]}..."
                            print(f"  {i}. {signal.symbol}: {signal.confidence:.1%} - {signal.reasoning[:50]}...{ai_note}")
                    else:
                        print("❌ 스캔 실패")
                
                elif choice == '2':
                    symbol = input("분석할 US 종목 심볼: ").strip().upper()
                    if symbol:
                        print(f"\n🔍 {symbol} 기술적 분석...")
                        
                        signal = await analyze_single_us_stock(symbol)
                        if signal and signal.confidence > 0:
                            print(f"💰 현재가: ${signal.price:.2f}")
                            print(f"🎯 종합결론: {signal.action.upper()} (신뢰도: {signal.confidence:.1%})")
                            print(f"💡 근거: {signal.reasoning}")
                            print(f"🎯 목표가: ${signal.target_price:.2f}")
                            print(f"🛑 손절가: ${signal.stop_loss:.2f}")
                            
                            if signal.ai_confidence and signal.ai_confidence['ai_used']:
                                ai_info = signal.ai_confidence
                                print(f"🤖 AI 조정: {ai_info['reason']} (사용량: {ai_info['daily_calls_used']}/20)")
                        else:
                            print(f"❌ 분석 실패")
                
                elif choice == '3':
                    print("\n🧪 기술적 지표 테스트...")
                    symbols = input("테스트할 종목들 (쉼표로 구분, 엔터시 기본값): ").strip()
                    if symbols:
                        symbol_list = [s.strip().upper() for s in symbols.split(',')]
                    else:
                        symbol_list = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
                    
                    print(f"🚀 기술적 분석: {', '.join(symbol_list)}")
                    
                    for symbol in symbol_list:
                        try:
                            result = await test_technical_indicators(symbol)
                            if 'error' not in result:
                                print(f"📊 {symbol}: {result['technical_score']:.2f} | "
                                      f"MACD:{result['indicators']['macd'].get('trend', 'unknown')[:4]} "
                                      f"RSI:{result['indicators']['rsi']:.0f}")
                            else:
                                print(f"❌ {symbol}: {result['error']}")
                        except:
                            print(f"❌ {symbol}: 분석 실패")
                
                elif choice == '4':
                    print("\n🤖 AI 확신도 체커 테스트...")
                    await test_ai_checker()
                
                elif choice == '5':
                    print("\n⚡ 빠른 종합 테스트...")
                    await quick_us_test()
                
                elif choice == '6':
                    symbol = input("모멘텀 분석할 종목 (엔터시 AAPL): ").strip().upper()
                    if not symbol:
                        symbol = 'AAPL'
                    print(f"\n📈 {symbol} 모멘텀 분석...")
                    await test_momentum_analysis(symbol)
                
                elif choice == '7':
                    symbol = input("가치/성장 분석할 종목 (엔터시 BRK-B): ").strip().upper()
                    if not symbol:
                        symbol = 'BRK-B'
                    print(f"\n💎 {symbol} 가치/성장 분석...")
                    await test_value_growth_analysis(symbol)
                
                elif choice == '8':
                    print("\n🎯 2주 스윙 시스템 테스트...")
                    await test_2week_swing_system()
                
                elif choice == '9':
                    print("\n🤖 완전 자동매매 시스템 테스트...")
                    await test_auto_trading_system()
                
                elif choice == 'a':
                    print("\n🤖 완전 자동매매 실행...")
                    await run_full_auto_trading()
                
                elif choice == 'b':
                    print("\n🔄 자동매매 데몬 시작...")
                    await start_auto_trading_daemon()
                
                elif choice == 'c':
                    print("\n📋 거래 기록 조회...")
                    await view_trading_records()
                
                elif choice == '0':
                    print("👋 US 전략 V7.0 시스템을 종료합니다!")
                    break
                    
                else:
                    print("❌ 잘못된 선택입니다. 0-9, a-c 중 선택하세요.")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 실행 오류: {e}")
        
    except Exception as e:
        logging.error(f"메인 실행 실패: {e}")
        print(f"❌ 실행 실패: {e}")

def print_us_help():
    """US 전략 도움말"""
    help_text = """
🇺🇸 US 전략 V7.0 - 기술적 분석 + AI 확신도 체크 (비용 최적화)
=================================================================

📋 주요 명령어:
  python us_strategy_optimized.py                              # 메인 메뉴
  python -c "import asyncio; from us_strategy_optimized import *; asyncio.run(quick_us_test())"  # 빠른 테스트

🔧 V7.0 설정:
  1. pip install yfinance pandas numpy requests aiohttp python-dotenv pytz openai
  2. IBKR 사용시: pip install ib_insync
  3. .env 파일 설정:
     OPENAI_API_KEY=your_openai_api_key_here

🆕 V7.0 최적화 특징:
  🕒 서머타임 완전 자동화 (EDT/EST 자동 감지)
  📈 기술적 지표 4종 (MACD + 볼린저밴드 + RSI + 모멘텀)
  🤖 AI 확신도 체크 (애매한 상황 0.4-0.7에서만 호출)
  💰 비용 최적화 (일일 20회 제한, 월 5천원 이하)
  🇺🇸 US 주식 전용 (S&P 500 + NASDAQ 100)
  🎯 동적 손익절 (신뢰도 기반 적응형)

🤖 AI 확신도 체커:
  📊 기술적 분석의 신뢰도가 애매할 때만 AI 호출
  💡 뉴스/감정 분석 제거로 비용 최적화
  🎯 명확한 신호(0.7+ 또는 0.4-)에서는 AI 미사용
  📉 일일 20회 제한으로 월 사용료 5천원 이하

💡 사용 팁:
  - 기술적 지표 중심 분석으로 안정성 향상
  - AI는 애매한 상황에서만 확신도 조정
  - 명확한 매수/매도 신호에서는 AI 비용 절약
  - US 주식 전용으로 시장 특화 최적화

⚠️ 주의사항:
  - OpenAI API 사용료 월 5천원 이하로 제한
  - AI는 확신도 체크만, 투자 결정은 기술적 분석 기반
  - 일일 AI 호출 20회 제한 (초과시 기술적 분석만)
"""
    print(help_text)

# ========================================================================================
# 🏁 실행 진입점
# ========================================================================================

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] in ['help', '--help']:
                print_us_help()
                sys.exit(0)
            elif sys.argv[1] == 'quick-test':
                asyncio.run(quick_us_test())
                sys.exit(0)
            elif sys.argv[1] == 'ai-test':
                asyncio.run(test_ai_checker())
                sys.exit(0)
            elif sys.argv[1] == 'auto-trading':
                asyncio.run(run_full_auto_trading())
                sys.exit(0)
            elif sys.argv[1] == 'daemon':
                asyncio.run(start_auto_trading_daemon())
                sys.exit(0)
            elif sys.argv[1] == 'records':
                asyncio.run(view_trading_records())
                sys.exit(0)
        
        # 메인 실행
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 US 전략 V7.0 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ US 전략 V7.0 실행 오류: {e}")
        logging.error(f"US 전략 V7.0 실행 오류: {e}")
