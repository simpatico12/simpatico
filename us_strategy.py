#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 전설적 퀸트프로젝트 V6.4 - 서머타임 + 고급기술지표 + OpenAI GPT (최적화)
==============================================================================
월 6-8% 달성형 주 2회 화목 매매 시스템
서머타임 자동처리 + MACD/볼린저밴드 + GPT 분석

Author: 전설적퀸트팀
Version: 6.4.0 (OpenAI 통합 최적화)
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
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import aiohttp
from dotenv import load_dotenv
import sqlite3
import pytz
from typing import Dict, List, Optional, Tuple, Any

# OpenAI 연동
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
# 📈 고급 기술지표 계산기 (MACD + 볼린저밴드)
# ========================================================================================

class AdvancedIndicators:
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

# ========================================================================================
# 🤖 OpenAI GPT 분석기
# ========================================================================================

class OpenAIAnalyzer:
    def __init__(self):
        self.client = None
        self.enabled = False
        self.api_key = os.getenv('OPENAI_API_KEY', '')
        self.model = "gpt-4o-mini"  # 비용 효율적인 모델
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                # OpenAI 최신 버전 지원
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.enabled = True
                logging.info("✅ OpenAI GPT 분석기 활성화")
            except Exception as e:
                logging.warning(f"⚠️ OpenAI 초기화 실패: {e}")
                # 구버전 OpenAI 지원
                try:
                    openai.api_key = self.api_key
                    self.client = openai
                    self.enabled = True
                    logging.info("✅ OpenAI GPT 분석기 활성화 (구버전)")
                except Exception as e2:
                    logging.warning(f"⚠️ OpenAI 구버전도 실패: {e2}")
        else:
            logging.warning("⚠️ OpenAI API 키 없음 또는 모듈 없음")
    
    async def analyze_market_sentiment(self, market_data: Dict) -> Dict[str, Any]:
        """GPT를 활용한 시장 감정 분석"""
        if not self.enabled:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'reasoning': 'GPT 비활성화'}
        
        try:
            vix = market_data.get('vix', 20)
            spy_momentum = market_data.get('spy_momentum', 0)
            qqq_momentum = market_data.get('qqq_momentum', 0)
            
            prompt = f"""
            미국 주식시장 분석을 해주세요.
            
            현재 시장 데이터:
            - VIX 지수: {vix:.1f}
            - SPY 3개월 모멘텀: {spy_momentum:.1f}%
            - QQQ 3개월 모멘텀: {qqq_momentum:.1f}%
            
            분석 요청:
            1. 시장 감정 (bullish/bearish/neutral)
            2. 신뢰도 (0-1)
            3. 간단한 이유 (한국어, 50자 이내)
            
            JSON 형태로 답변해주세요:
            {{"sentiment": "bullish/bearish/neutral", "confidence": 0.0-1.0, "reasoning": "이유"}}
            """
            
            response = await self._make_gpt_request(prompt)
            
            try:
                result = json.loads(response)
                return {
                    'sentiment': result.get('sentiment', 'neutral'),
                    'confidence': float(result.get('confidence', 0.5)),
                    'reasoning': result.get('reasoning', 'GPT 분석 완료')
                }
            except:
                return {'sentiment': 'neutral', 'confidence': 0.5, 'reasoning': 'GPT 파싱 실패'}
                
        except Exception as e:
            logging.error(f"GPT 시장 분석 실패: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'reasoning': f'GPT 오류: {e}'}
    
    async def analyze_stock_fundamentals(self, stock_data: Dict) -> Dict[str, Any]:
        """GPT를 활용한 개별 종목 펀더멘털 분석"""
        if not self.enabled:
            return {'score': 0.5, 'recommendation': 'hold', 'reasoning': 'GPT 비활성화'}
        
        try:
            symbol = stock_data.get('symbol', 'UNKNOWN')
            pe_ratio = stock_data.get('pe_ratio', 0)
            eps_growth = stock_data.get('eps_growth', 0)
            revenue_growth = stock_data.get('revenue_growth', 0)
            roe = stock_data.get('roe', 0)
            sector = stock_data.get('sector', 'Unknown')
            
            prompt = f"""
            {symbol} 주식의 펀더멘털 분석을 해주세요.
            
            재무 데이터:
            - PE Ratio: {pe_ratio:.1f}
            - EPS 성장률: {eps_growth:.1f}%
            - 매출 성장률: {revenue_growth:.1f}%
            - ROE: {roe:.1f}%
            - 섹터: {sector}
            
            분석 요청:
            1. 펀더멘털 점수 (0-1)
            2. 추천 (buy/sell/hold)
            3. 간단한 이유 (한국어, 60자 이내)
            
            JSON 형태로 답변해주세요:
            {{"score": 0.0-1.0, "recommendation": "buy/sell/hold", "reasoning": "이유"}}
            """
            
            response = await self._make_gpt_request(prompt)
            
            try:
                result = json.loads(response)
                return {
                    'score': float(result.get('score', 0.5)),
                    'recommendation': result.get('recommendation', 'hold'),
                    'reasoning': result.get('reasoning', 'GPT 분석 완료')
                }
            except:
                return {'score': 0.5, 'recommendation': 'hold', 'reasoning': 'GPT 파싱 실패'}
                
        except Exception as e:
            logging.error(f"GPT 종목 분석 실패: {e}")
            return {'score': 0.5, 'recommendation': 'hold', 'reasoning': f'GPT 오류: {e}'}
    
    async def generate_trading_insight(self, portfolio_data: Dict, market_condition: Dict) -> str:
        """GPT를 활용한 거래 인사이트 생성"""
        if not self.enabled:
            return "🤖 GPT 비활성화 상태입니다."
        
        try:
            positions_count = portfolio_data.get('positions_count', 0)
            weekly_return = portfolio_data.get('weekly_return', 0)
            market_sentiment = market_condition.get('sentiment', 'neutral')
            vix = market_condition.get('vix', 20)
            
            prompt = f"""
            현재 포트폴리오와 시장 상황을 분석해서 간단한 거래 인사이트를 한국어로 제공해주세요.
            
            포트폴리오:
            - 보유 종목 수: {positions_count}개
            - 주간 수익률: {weekly_return:.1f}%
            
            시장 상황:
            - 시장 감정: {market_sentiment}
            - VIX: {vix:.1f}
            
            2-3줄로 간단한 조언을 해주세요. 이모지 포함.
            """
            
            response = await self._make_gpt_request(prompt)
            return response.strip()
            
        except Exception as e:
            logging.error(f"GPT 인사이트 생성 실패: {e}")
            return f"🤖 GPT 인사이트 생성 중 오류가 발생했습니다: {e}"
    
    async def _make_gpt_request(self, prompt: str) -> str:
        """GPT API 요청 처리"""
        try:
            if not self.client:
                return "GPT 클라이언트 없음"
            
            # OpenAI 최신 API 방식으로 수정
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 전문적인 주식 투자 분석가입니다. 정확하고 간결한 분석을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"GPT API 요청 실패: {e}")
            return f"GPT API 오류: {str(e)}"

# ========================================================================================
# 🔧 설정 관리자
# ========================================================================================

class Config:
    def __init__(self):
        self.config = {
            'strategy': {
                'enabled': True,
                'mode': 'swing',
                'target_stocks': {'classic': 20, 'swing': 8},
                'monthly_target': {'min': 6.0, 'max': 8.0},
                'weights': {'buffett': 20.0, 'lynch': 20.0, 'momentum': 20.0, 'technical': 20.0, 'advanced': 10.0, 'gpt': 10.0}
            },
            'trading': {
                'swing': {'take_profit': [7.0, 14.0], 'profit_ratios': [60.0, 40.0], 'stop_loss': 7.0},
                'weekly': {
                    'enabled': True, 'tuesday_targets': 4, 'thursday_targets': 2,
                    'tuesday_allocation': 13.0, 'thursday_allocation': 8.0,
                    'profit_taking_threshold': 9.0, 'loss_cutting_threshold': -5.5
                }
            },
            'risk': {'max_position': 15.0, 'daily_loss_limit': 1.0, 'monthly_loss_limit': 3.0},
            'ibkr': {'enabled': True, 'host': '127.0.0.1', 'port': 7497, 'client_id': 1, 'paper_trading': True},
            'openai': {
                'enabled': True,
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'model': 'gpt-4o-mini',
                'max_tokens': 500
            },
            'notifications': {
                'telegram': {
                    'enabled': True,
                    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
                }
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
    scores: Dict[str, float]
    target_price: float
    stop_loss: float
    reasoning: str
    timestamp: datetime
    gpt_insight: str = ""

@dataclass 
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    entry_date: datetime
    mode: str
    stage: int = 1
    tp_executed: List[bool] = field(default_factory=lambda: [False, False])
    highest_price: float = 0.0
    entry_day: str = ''
    
    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.avg_cost

    def profit_percent(self, current_price: float) -> float:
        return ((current_price - self.avg_cost) / self.avg_cost) * 100

# ========================================================================================
# 🚀 주식 선별 엔진
# ========================================================================================

class StockSelector:
    def __init__(self):
        self.cache = {'sp500': [], 'nasdaq': [], 'last_update': None}
        self.indicators = AdvancedIndicators()
        self.gpt_analyzer = OpenAIAnalyzer()
    
    async def get_current_vix(self) -> float:
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            return float(hist['Close'].iloc[-1]) if not hist.empty else 20.0
        except:
            return 20.0
    
    async def collect_symbols(self) -> List[str]:
        try:
            if self._is_cache_valid():
                return self.cache['sp500'] + self.cache['nasdaq']
            
            sp500 = self._get_sp500_symbols()
            nasdaq = self._get_nasdaq_symbols()
            
            self.cache['sp500'] = sp500
            self.cache['nasdaq'] = nasdaq
            self.cache['last_update'] = datetime.now()
            
            universe = list(set(sp500 + nasdaq))
            logging.info(f"🌌 투자 유니버스: {len(universe)}개 종목")
            return universe
            
        except Exception as e:
            logging.error(f"유니버스 생성 실패: {e}")
            return self._get_backup_symbols()
    
    def _get_sp500_symbols(self) -> List[str]:
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return [str(s).replace('.', '-') for s in tables[0]['Symbol'].tolist()]
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ']
    
    def _get_nasdaq_symbols(self) -> List[str]:
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            for table in tables:
                if 'Symbol' in table.columns:
                    return table['Symbol'].dropna().tolist()
            return []
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    
    def _is_cache_valid(self) -> bool:
        return (self.cache['last_update'] and 
                (datetime.now() - self.cache['last_update']).seconds < 24 * 3600)
    
    def _get_backup_symbols(self) -> List[str]:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
                'BRK-B', 'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA']

    async def get_stock_data(self, symbol: str) -> Dict:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1y")
        
            if hist.empty or len(hist) < 50:
                return {}
        
            current_price = float(hist['Close'].iloc[-1])
            closes = hist['Close']
        
            data = {
                'symbol': symbol, 'price': current_price,
                'market_cap': info.get('marketCap', 0) or 0,
                'avg_volume': info.get('averageVolume', 0) or 0,
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'pbr': info.get('priceToBook', 0) or 0,
                'roe': (info.get('returnOnEquity', 0) or 0) * 100,
                'eps_growth': (info.get('earningsQuarterlyGrowth', 0) or 0) * 100,
                'revenue_growth': (info.get('revenueQuarterlyGrowth', 0) or 0) * 100,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'sector': info.get('sector', 'Unknown'),
                'beta': info.get('beta', 1.0) or 1.0
            }
            
            if data['pe_ratio'] > 0 and data['eps_growth'] > 0:
                data['peg'] = data['pe_ratio'] / data['eps_growth']
            else:
                data['peg'] = 999
            
            if len(hist) >= 252:
                data['momentum_3m'] = ((current_price / float(closes.iloc[-63])) - 1) * 100
                data['momentum_6m'] = ((current_price / float(closes.iloc[-126])) - 1) * 100
                data['momentum_12m'] = ((current_price / float(closes.iloc[-252])) - 1) * 100
            else:
                data['momentum_3m'] = data['momentum_6m'] = data['momentum_12m'] = 0
            
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['rsi'] = float(100 - (100 / (1 + rs)).iloc[-1])
            
            # 🆕 고급 기술지표 (MACD + 볼린저)
            data['macd'] = self.indicators.calculate_macd(closes)
            data['bollinger'] = self.indicators.calculate_bollinger_bands(closes)
            
            avg_vol = float(hist['Volume'].rolling(20).mean().iloc[-1])
            current_vol = float(hist['Volume'].iloc[-1])
            data['volume_spike'] = current_vol / avg_vol if avg_vol > 0 else 1
            
            returns = closes.pct_change().dropna()
            data['volatility'] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
            
            # 🤖 GPT 펀더멘털 분석 추가
            if self.gpt_analyzer.enabled:
                gpt_analysis = await self.gpt_analyzer.analyze_stock_fundamentals(data)
                data['gpt_score'] = gpt_analysis['score']
                data['gpt_recommendation'] = gpt_analysis['recommendation']
                data['gpt_reasoning'] = gpt_analysis['reasoning']
            else:
                data['gpt_score'] = 0.5
                data['gpt_recommendation'] = 'hold'
                data['gpt_reasoning'] = 'GPT 비활성화'
            
            await asyncio.sleep(0.3)
            return data
            
        except Exception as e:
            logging.error(f"데이터 수집 실패 {symbol}: {e}")
            return {}

# ========================================================================================
# 🧠 6가지 전략 분석 엔진 (GPT 추가)
# ========================================================================================

class AdvancedStrategyAnalyzer:
    def __init__(self):
        self.gpt_analyzer = OpenAIAnalyzer()
    
    def calculate_scores(self, data: Dict, vix: float) -> Tuple[float, Dict]:
        scores = {}
        
        scores['buffett'] = self._calculate_buffett_score(data)
        scores['lynch'] = self._calculate_lynch_score(data)
        scores['momentum'] = self._calculate_momentum_score(data)
        scores['technical'] = self._calculate_technical_score(data)
        scores['advanced'] = self._calculate_advanced_indicators_score(data)
        scores['gpt'] = self._calculate_gpt_score(data)
        
        weights = config.get('strategy.weights', {})
        total = sum(scores[key] * weights.get(key, 16.67) for key in scores.keys()) / 100
        
        if vix <= 15:
            adjusted = total * 1.15
        elif vix >= 30:
            adjusted = total * 0.85
        else:
            adjusted = total
        
        scores['total'] = adjusted
        scores['vix_adjustment'] = adjusted - total
        
        return adjusted, scores
    
    def _calculate_buffett_score(self, data: Dict) -> float:
        score = 0.0
        
        pbr = data.get('pbr', 999)
        if 0 < pbr <= 1.0: score += 0.30
        elif pbr <= 1.5: score += 0.25
        elif pbr <= 2.0: score += 0.20
        
        roe = data.get('roe', 0)
        if roe >= 20: score += 0.25
        elif roe >= 15: score += 0.20
        elif roe >= 10: score += 0.15
        
        debt_ratio = data.get('debt_to_equity', 999) / 100
        if debt_ratio <= 0.3: score += 0.20
        elif debt_ratio <= 0.5: score += 0.15
        
        pe = data.get('pe_ratio', 999)
        if 5 <= pe <= 15: score += 0.15
        elif pe <= 20: score += 0.10
        
        if data.get('market_cap', 0) > 10_000_000_000: score += 0.10
        
        return min(score, 1.0)
    
    def _calculate_lynch_score(self, data: Dict) -> float:
        score = 0.0
        
        peg = data.get('peg', 999)
        if 0 < peg <= 0.5: score += 0.40
        elif peg <= 1.0: score += 0.35
        elif peg <= 1.5: score += 0.25
        
        eps_growth = data.get('eps_growth', 0)
        if eps_growth >= 25: score += 0.30
        elif eps_growth >= 20: score += 0.25
        elif eps_growth >= 15: score += 0.20
        
        rev_growth = data.get('revenue_growth', 0)
        if rev_growth >= 20: score += 0.20
        elif rev_growth >= 15: score += 0.15
        
        if data.get('roe', 0) >= 15: score += 0.10
        
        return min(score, 1.0)
    
    def _calculate_momentum_score(self, data: Dict) -> float:
        score = 0.0
        
        mom_3m = data.get('momentum_3m', 0)
        if mom_3m >= 20: score += 0.30
        elif mom_3m >= 15: score += 0.25
        elif mom_3m >= 10: score += 0.20
        
        mom_6m = data.get('momentum_6m', 0)
        if mom_6m >= 30: score += 0.25
        elif mom_6m >= 20: score += 0.20
        elif mom_6m >= 15: score += 0.15
        
        mom_12m = data.get('momentum_12m', 0)
        if mom_12m >= 50: score += 0.25
        elif mom_12m >= 30: score += 0.20
        elif mom_12m >= 20: score += 0.15
        
        vol_spike = data.get('volume_spike', 1)
        if vol_spike >= 2.0: score += 0.20
        elif vol_spike >= 1.5: score += 0.15
        
        return min(score, 1.0)
    
    def _calculate_technical_score(self, data: Dict) -> float:
        score = 0.0
        
        rsi = data.get('rsi', 50)
        if 30 <= rsi <= 70: score += 0.40
        elif 25 <= rsi < 30 or 70 < rsi <= 75: score += 0.30
        
        volatility = data.get('volatility', 25)
        if 15 <= volatility <= 30: score += 0.30
        elif 10 <= volatility <= 35: score += 0.20
        
        beta = data.get('beta', 1.0)
        if 0.8 <= beta <= 1.3: score += 0.30
        elif 0.6 <= beta <= 1.5: score += 0.20
        
        return min(score, 1.0)
    
    def _calculate_advanced_indicators_score(self, data: Dict) -> float:
        score = 0.0
        
        # MACD 점수
        macd_data = data.get('macd', {})
        macd_trend = macd_data.get('trend', 'neutral')
        macd_crossover = macd_data.get('crossover', 'none')
        
        if macd_trend == 'bullish': score += 0.30
        elif macd_trend == 'improving': score += 0.25
        elif macd_trend == 'bearish': score -= 0.10
        
        if macd_crossover == 'buy': score += 0.20
        elif macd_crossover == 'sell': score -= 0.15
        
        # 볼린저 밴드 점수
        bb_data = data.get('bollinger', {})
        bb_position = bb_data.get('position', 0.5)
        bb_squeeze = bb_data.get('squeeze', False)
        bb_signal = bb_data.get('signal', 'normal')
        
        if bb_signal == 'oversold' and bb_position < 0.3: score += 0.30
        elif bb_signal == 'normal' and 0.3 <= bb_position <= 0.7: score += 0.20
        elif bb_signal == 'overbought': score -= 0.15
        
        if bb_squeeze: score += 0.20  # 스퀴즈는 폭발적 움직임 전조
        
        # 두 지표 일치성 보너스
        bullish_count = sum([
            1 if macd_trend in ['bullish', 'improving'] else 0,
            1 if bb_signal in ['oversold', 'normal'] else 0
        ])
        
        if bullish_count >= 2: score += 0.20
        
        return max(min(score, 1.0), 0.0)
    
    def _calculate_gpt_score(self, data: Dict) -> float:
        """🤖 GPT 기반 점수 계산"""
        try:
            gpt_score = data.get('gpt_score', 0.5)
            gpt_recommendation = data.get('gpt_recommendation', 'hold')
            
            # GPT 추천에 따른 점수 조정
            if gpt_recommendation == 'buy':
                adjusted_score = min(gpt_score * 1.2, 1.0)
            elif gpt_recommendation == 'sell':
                adjusted_score = max(gpt_score * 0.6, 0.0)
            else:  # hold
                adjusted_score = gpt_score
            
            return adjusted_score
            
        except Exception as e:
            logging.error(f"GPT 점수 계산 실패: {e}")
            return 0.5

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
    
    async def get_current_price(self, symbol: str) -> float:
        try:
            if not self.connected:
                return 0.0
            
            contract = Stock(symbol, 'SMART', 'USD')
            ticker = self.ib.reqMktData(contract)
            await asyncio.sleep(0.5)
            
            price = ticker.marketPrice() or ticker.last or 0.0
            self.ib.cancelMktData(contract)
            return float(price)
            
        except:
            return 0.0
    
    async def get_portfolio_value(self) -> float:
        try:
            await self._update_account()
            return sum(pos['market_price'] * abs(pos['quantity']) for pos in self.positions.values())
        except:
            return 1000000

# ========================================================================================
# 🏆 메인 전략 시스템 (GPT 통합)
# ========================================================================================

class LegendaryQuantStrategy:
    def __init__(self):
        self.enabled = config.get('strategy.enabled', True)
        self.current_mode = config.get('strategy.mode', 'swing')
        
        self.dst_manager = DaylightSavingManager()
        self.selector = StockSelector()
        self.analyzer = AdvancedStrategyAnalyzer()
        self.ibkr = IBKRTrader()
        self.gpt_analyzer = OpenAIAnalyzer()
        
        self.selected_stocks = []
        self.last_selection = None
        self.monthly_return = 0.0
        self.last_trade_dates = {'Tuesday': None, 'Thursday': None}
        
        if self.enabled:
            logging.info("🏆 전설적 퀸트 V6.4+GPT 시스템 가동!")
            logging.info(f"🎯 모드: {self.current_mode.upper()}")
            logging.info(f"🕒 서머타임: {'활성' if self.dst_manager.is_dst_active() else '비활성'}")
            logging.info(f"🤖 GPT 분석: {'활성' if self.gpt_analyzer.enabled else '비활성'}")

    async def auto_select_stocks(self) -> List[str]:
        if not self.enabled:
            return []
        
        try:
            if (self.last_selection and 
                (datetime.now() - self.last_selection).seconds < 24 * 3600):
                return [s['symbol'] for s in self.selected_stocks]
            
            logging.info("🚀 고급 종목 선별 시작! (GPT 포함)")
            start_time = time.time()
            
            universe = await self.selector.collect_symbols()
            if not universe:
                return self._get_fallback_stocks()
            
            current_vix = await self.selector.get_current_vix()
            
            scored_stocks = []
            batch_size = 12  # GPT 때문에 배치 사이즈 약간 감소
            
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i + batch_size]
                tasks = [self._analyze_stock_async(symbol, current_vix) for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result:
                        scored_stocks.append(result)
                
                if i % 60 == 0:
                    logging.info(f"📊 고급분석+GPT: {i}/{len(universe)}")
            
            if not scored_stocks:
                return self._get_fallback_stocks()
            
            target_count = config.get(f'strategy.target_stocks.{self.current_mode}', 8)
            final_selection = self._select_best_stocks_with_gpt(scored_stocks, target_count)
            
            self.selected_stocks = final_selection
            self.last_selection = datetime.now()
            
            elapsed = time.time() - start_time
            selected_symbols = [s['symbol'] for s in final_selection]
            
            logging.info(f"🏆 고급선별+GPT 완료! {len(selected_symbols)}개 ({elapsed:.1f}초)")
            return selected_symbols
            
        except Exception as e:
            logging.error(f"선별 실패: {e}")
            return self._get_fallback_stocks()
    
    async def _analyze_stock_async(self, symbol: str, vix: float) -> Optional[Dict]:
        try:
            data = await self.selector.get_stock_data(symbol)
            if not data:
                return None
            
            if (data.get('market_cap', 0) < 5_000_000_000 or 
                data.get('avg_volume', 0) < 1_000_000):
                return None
            
            total_score, scores = self.analyzer.calculate_scores(data, vix)
            
            if not self._advanced_filter_with_gpt(data, total_score):
                return None
            
            result = data.copy()
            result.update(scores)
            result['vix'] = vix
            
            return result
            
        except Exception as e:
            logging.error(f"종목 분석 실패 {symbol}: {e}")
            return None
    
    def _advanced_filter_with_gpt(self, data: Dict, score: float) -> bool:
        try:
            if score < 0.65:
                return False
            
            macd_data = data.get('macd', {})
            if macd_data.get('trend') == 'bearish' and macd_data.get('histogram', 0) < -0.5:
                return False
            
            bb_data = data.get('bollinger', {})
            if bb_data.get('position', 0.5) > 0.9:
                return False
            
            # 🤖 GPT 필터 추가
            gpt_recommendation = data.get('gpt_recommendation', 'hold')
            if gpt_recommendation == 'sell':
                return False
            
            return True
        except:
            return True
    
    def _select_best_stocks_with_gpt(self, scored_stocks: List[Dict], target_count: int) -> List[Dict]:
        scored_stocks.sort(key=lambda x: x['total'], reverse=True)
        
        final_selection = []
        sector_counts = {}
        
        # 1차: GPT 추천 + 고점수
        for stock in scored_stocks:
            if len(final_selection) >= target_count:
                break
            
            sector = stock.get('sector', 'Unknown')
            gpt_rec = stock.get('gpt_recommendation', 'hold')
            
            if (gpt_rec == 'buy' and 
                sector_counts.get(sector, 0) < 2 and 
                stock['total'] >= 0.75):
                final_selection.append(stock)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # 2차: 일반 섹터 다양성
        for stock in scored_stocks:
            if len(final_selection) >= target_count:
                break
            
            sector = stock.get('sector', 'Unknown')
            if (sector_counts.get(sector, 0) < 2 and 
                stock not in final_selection):
                final_selection.append(stock)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # 3차: 나머지 고점수
        remaining = target_count - len(final_selection)
        for stock in scored_stocks:
            if remaining <= 0:
                break
            if stock not in final_selection:
                final_selection.append(stock)
                remaining -= 1
        
        return final_selection
    
    def _get_fallback_stocks(self) -> List[str]:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX']
    
    async def analyze_stock_signal(self, symbol: str) -> StockSignal:
        try:
            data = await self.selector.get_stock_data(symbol)
            if not data:
                raise ValueError(f"데이터 수집 실패: {symbol}")
            
            vix = await self.selector.get_current_vix()
            total_score, scores = self.analyzer.calculate_scores(data, vix)
            
            confidence_threshold = 0.70
            
            if total_score >= confidence_threshold:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= 0.25:
                action = 'sell'
                confidence = min(1 - total_score, 0.95)
            else:
                action = 'hold'
                confidence = 0.50
            
            advanced_score = scores.get('advanced', 0)
            gpt_score = scores.get('gpt', 0.5)
            target_multiplier = 1.0 + ((advanced_score + gpt_score) * 0.15)
            
            target_price = data['price'] * (1 + confidence * 0.20 * target_multiplier)
            stop_loss = data['price'] * (1 - 0.07)
            
            macd_trend = data.get('macd', {}).get('trend', 'neutral')
            bb_signal = data.get('bollinger', {}).get('signal', 'normal')
            gpt_rec = data.get('gpt_recommendation', 'hold')
            gpt_reasoning = data.get('gpt_reasoning', '')
            
            reasoning = (f"통합:{total_score:.2f} | "
                        f"MACD:{macd_trend} BB:{bb_signal} | "
                        f"GPT:{gpt_rec}")
            
            return StockSignal(
                symbol=symbol, action=action, confidence=confidence, price=data['price'],
                scores=scores, target_price=target_price, stop_loss=stop_loss,
                reasoning=reasoning, timestamp=datetime.now(), gpt_insight=gpt_reasoning
            )
            
        except Exception as e:
            return StockSignal(
                symbol=symbol, action='hold', confidence=0.0, price=0.0,
                scores={}, target_price=0.0, stop_loss=0.0,
                reasoning=f"오류: {e}", timestamp=datetime.now(), gpt_insight=""
            )

# ========================================================================================
# 🎯 편의 함수들
# ========================================================================================

async def run_auto_selection():
    strategy = LegendaryQuantStrategy()
    signals = []
    selected = await strategy.auto_select_stocks()
    
    for symbol in selected[:5]:
        try:
            signal = await strategy.analyze_stock_signal(symbol)
            signals.append(signal)
        except:
            continue
    
    return signals

async def analyze_single_stock(symbol: str):
    strategy = LegendaryQuantStrategy()
    return await strategy.analyze_stock_signal(symbol)

async def get_system_status():
    try:
        strategy = LegendaryQuantStrategy()
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
        
        return {
            'enabled': strategy.enabled,
            'current_mode': strategy.current_mode,
            'ibkr_connected': ibkr_connected,
            'ibkr_available': IBKR_AVAILABLE,
            'openai_enabled': strategy.gpt_analyzer.enabled,
            'openai_available': OPENAI_AVAILABLE,
            'monthly_return': strategy.monthly_return,
            'dst_active': dst_active,
            'market_hours_kst': f"{market_open.strftime('%H:%M')}-{market_close.strftime('%H:%M')}",
            'timezone_status': 'EDT' if dst_active else 'EST',
            'advanced_indicators': True,
            'gpt_analysis': strategy.gpt_analyzer.enabled,
            'last_tuesday': strategy.last_trade_dates.get('Tuesday'),
            'last_thursday': strategy.last_trade_dates.get('Thursday')
        }
        
    except Exception as e:
        return {'error': str(e)}

async def test_advanced_indicators_with_gpt(symbol: str = 'AAPL'):
    try:
        strategy = LegendaryQuantStrategy()
        data = await strategy.selector.get_stock_data(symbol)
        
        if data:
            return {
                'symbol': symbol,
                'price': data['price'],
                'macd': data.get('macd', {}),
                'bollinger': data.get('bollinger', {}),
                'traditional': {'rsi': data.get('rsi', 50), 'volume_spike': data.get('volume_spike', 1)},
                'gpt_analysis': {
                    'score': data.get('gpt_score', 0.5),
                    'recommendation': data.get('gpt_recommendation', 'hold'),
                    'reasoning': data.get('gpt_reasoning', 'GPT 비활성화')
                }
            }
        else:
            return {'error': '데이터 없음'}
    except Exception as e:
        return {'error': str(e)}

async def quick_gpt_test():
    print("🤖 GPT 기능 테스트...")
    
    try:
        strategy = LegendaryQuantStrategy()
        
        # 시장 감정 테스트
        print("\n📊 시장 감정 분석:")
        if strategy.gpt_analyzer.enabled:
            market_data = {
                'vix': await strategy.selector.get_current_vix(),
                'spy_momentum': 0,
                'qqq_momentum': 0
            }
            
            sentiment = await strategy.gpt_analyzer.analyze_market_sentiment(market_data)
            print(f"   감정: {sentiment.get('sentiment', 'unknown')}")
            print(f"   신뢰도: {sentiment.get('confidence', 0):.1%}")
            print(f"   이유: {sentiment.get('reasoning', 'none')}")
        else:
            print("   ❌ GPT 비활성화")
        
        # 종목 분석 테스트
        print("\n📈 AAPL 펀더멘털 분석:")
        indicators = await test_advanced_indicators_with_gpt('AAPL')
        if 'error' not in indicators:
            gpt_analysis = indicators.get('gpt_analysis', {})
            print(f"   점수: {gpt_analysis.get('score', 0):.2f}")
            print(f"   추천: {gpt_analysis.get('recommendation', 'unknown')}")
            print(f"   이유: {gpt_analysis.get('reasoning', 'none')}")
        else:
            print(f"   ❌ 실패: {indicators['error']}")
        
    except Exception as e:
        print(f"❌ GPT 테스트 실패: {e}")

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
                logging.FileHandler('legendary_quant_v64_gpt.log', encoding='utf-8')
            ]
        )
        
        print("🏆" + "="*75)
        print("🔥 전설적 퀸트프로젝트 V6.4 - 서머타임 + 고급기술지표 + OpenAI GPT")
        print("🚀 월 6-8% 달성형 6가지 전략 융합 시스템 (AI 강화)")
        print("="*77)
        
        print("\n🌟 V6.4+GPT 주요기능:")
        print("  ✨ 서머타임 완전 자동화 (EDT/EST 자동전환)")
        print("  ✨ 고급 기술지표 2종 (MACD + 볼린저밴드)")
        print("  ✨ 6가지 전략 융합 (버핏+린치+모멘텀+기술+고급+GPT)")
        print("  ✨ GPT-4 AI 분석 (시장감정+종목분석+거래인사이트)")
        print("  ✨ 동적 손익절 (AI 시그널 기반 적응형)")
        print("  ✨ 월 목표 상향 (6-8% vs 기존 5-7%)")
        
        status = await get_system_status()
        
        if 'error' not in status:
            print(f"\n🔧 시스템 상태:")
            print(f"  ✅ 시스템: {status['current_mode'].upper()}")
            print(f"  🕒 시간대: {status['timezone_status']} ({'서머타임' if status['dst_active'] else '표준시'})")
            print(f"  📈 시장시간: {status['market_hours_kst']} KST")
            print(f"  🤖 IBKR: {'연결가능' if status['ibkr_connected'] else '연결불가'}")
            print(f"  🤖 OpenAI: {'활성화' if status['openai_enabled'] else '비활성화'}")
            print(f"  📊 고급지표: {'활성화' if status['advanced_indicators'] else '비활성화'}")
            print(f"  🎯 GPT 분석: {'활성화' if status['gpt_analysis'] else '비활성화'}")
            print(f"  📈 월 수익률: {status['monthly_return']:.2f}%")
        else:
            print(f"  ❌ 상태 확인 실패: {status['error']}")
        
        print("\n🚀 실행 옵션:")
        print("  1. 🔍 고급 지표+GPT 종목 선별")
        print("  2. 📊 개별 종목 고급+GPT 분석")
        print("  3. 🧪 고급 지표+GPT 테스트")
        print("  4. 🤖 GPT 기능 전용 테스트")
        print("  0. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (0-4): ").strip()
                
                if choice == '1':
                    print("\n🔍 고급 지표+GPT 종목 선별!")
                    signals = await run_auto_selection()
                    
                    if signals:
                        print(f"\n📈 고급분석+GPT 결과: {len(signals)}개 스캔")
                        
                        buy_signals = [s for s in signals if s.action == 'buy']
                        print(f"🟢 매수추천: {len(buy_signals)}개")
                        
                        for i, signal in enumerate(buy_signals[:3], 1):
                            gpt_note = f" | GPT: {signal.gpt_insight[:30]}..." if signal.gpt_insight else ""
                            print(f"  {i}. {signal.symbol}: {signal.confidence:.1%} - {signal.reasoning[:40]}...{gpt_note}")
                    else:
                        print("❌ 스캔 실패")
                
                elif choice == '2':
                    symbol = input("분석할 종목 심볼: ").strip().upper()
                    if symbol:
                        print(f"\n🔍 {symbol} 고급+GPT 분석...")
                        
                        indicators = await test_advanced_indicators_with_gpt(symbol)
                        if 'error' not in indicators:
                            print(f"💰 현재가: ${indicators['price']:.2f}")
                            
                            macd = indicators.get('macd', {})
                            print(f"📊 MACD: {macd.get('trend', 'unknown')} (크로스오버: {macd.get('crossover', 'none')})")
                            
                            bb = indicators.get('bollinger', {})
                            print(f"📊 볼린저: {bb.get('signal', 'unknown')} (위치: {bb.get('position', 0.5):.2f})")
                            
                            gpt_analysis = indicators.get('gpt_analysis', {})
                            print(f"🤖 GPT 점수: {gpt_analysis.get('score', 0):.2f}")
                            print(f"🤖 GPT 추천: {gpt_analysis.get('recommendation', 'unknown')}")
                            print(f"🤖 GPT 사유: {gpt_analysis.get('reasoning', 'none')}")
                            
                            signal = await analyze_single_stock(symbol)
                            if signal and signal.confidence > 0:
                                print(f"\n🎯 종합결론: {signal.action.upper()} (신뢰도: {signal.confidence:.1%})")
                                print(f"💡 근거: {signal.reasoning}")
                                if signal.gpt_insight:
                                    print(f"🤖 GPT: {signal.gpt_insight}")
                        else:
                            print(f"❌ 분석 실패: {indicators['error']}")
                
                elif choice == '3':
                    print("\n🧪 고급 지표+GPT 테스트...")
                    symbols = input("테스트할 종목들 (쉼표로 구분, 엔터시 기본값): ").strip()
                    if symbols:
                        symbol_list = [s.strip().upper() for s in symbols.split(',')]
                    else:
                        symbol_list = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
                    
                    print(f"🚀 고급 분석+GPT: {', '.join(symbol_list)}")
                    
                    strategy = LegendaryQuantStrategy()
                    
                    for symbol in symbol_list:
                        try:
                            signal = await strategy.analyze_stock_signal(symbol)
                            action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                            gpt_note = f" | GPT: {signal.gpt_insight[:20]}..." if signal.gpt_insight else ""
                            print(f"{action_emoji} {symbol}: {signal.action} ({signal.confidence:.1%}) - {signal.reasoning[:40]}...{gpt_note}")
                        except:
                            print(f"❌ {symbol}: 분석 실패")
                
                elif choice == '4':
                    print("\n🤖 GPT 기능 전용 테스트...")
                    await quick_gpt_test()
                
                elif choice == '0':
                    print("👋 V6.4+GPT 시스템을 종료합니다!")
                    break
                    
                else:
                    print("❌ 잘못된 선택입니다. 0-4 중 선택하세요.")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
    except Exception as e:
        logging.error(f"메인 실행 실패: {e}")
        print(f"❌ 실행 실패: {e}")

def print_help():
    help_text = """
🏆 전설적 퀸트프로젝트 V6.4+GPT - 최적화된 미국전략
=======================================================

📋 주요 명령어:
  python legendary_quant_v64_gpt.py                    # 메인 메뉴
  python -c "import asyncio; from legendary_quant_v64_gpt import *; asyncio.run(quick_gpt_test())"  # GPT 테스트

🔧 V6.4+GPT 설정:
  1. pip install yfinance pandas numpy requests aiohttp python-dotenv pytz openai
  2. IBKR 사용시: pip install ib_insync
  3. .env 파일 설정:
     OPENAI_API_KEY=your_openai_api_key_here
     TELEGRAM_BOT_TOKEN=your_telegram_token_here
     TELEGRAM_CHAT_ID=your_chat_id_here

🆕 V6.4+GPT 최적화 특징:
  🕒 서머타임 완전 자동화 (EDT/EST 자동 감지)
  📈 고급 기술지표 2종 (MACD + 볼린저밴드)
  🤖 OpenAI GPT-4 AI 분석 (시장감정+종목분석)
  🧠 6가지 전략 융합 (버핏+린치+모멘텀+기술+고급+GPT)
  🎯 AI 강화 성능 (월 목표 6-8%)

🤖 GPT 기능:
  📊 시장 감정 실시간 분석
  📈 개별 종목 펀더멘털 AI 평가
  💡 맞춤형 거래 인사이트 생성
  🎯 AI 강화 매수/매도 신호

💡 사용 팁:
  - OpenAI API 키 필수 (gpt-4o-mini 사용으로 비용 최적화)
  - GPT 신뢰도 70% 이상시 강한 신호로 활용
  - 기술적 지표 + GPT 일치시 신뢰도 최대
  - 코드 최적화로 안정성 및 성능 향상

⚠️ 주의사항:
  - OpenAI API 사용료 발생 (월 $5-20 예상)
  - GPT 응답 지연 가능성 (3-5초)
  - API 제한 시 기존 지표로 자동 대체
"""
    print(help_text)

# ========================================================================================
# 🏁 실행 진입점
# ========================================================================================

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] in ['help', '--help']:
                print_help()
                sys.exit(0)
            elif sys.argv[1] == 'gpt-test':
                asyncio.run(quick_gpt_test())
                sys.exit(0)
        
        # 메인 실행
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 V6.4+GPT 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ V6.4+GPT 실행 오류: {e}")
        logging.error(f"V6.4+GPT 실행 오류: {e}")
