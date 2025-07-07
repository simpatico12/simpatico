#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 전설적 퀸트프로젝트 - 서머타임 + 고급기술지표 V6.4
=========================================================
월 6-8% 달성형 주 2회 화목 매매 시스템
서머타임 자동처리 + MACD/볼린저밴드

Author: 전설적퀸트팀
Version: 6.4.0 (최적화)
Lines: ~1500
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
                'weights': {'buffett': 20.0, 'lynch': 20.0, 'momentum': 20.0, 'technical': 25.0, 'advanced': 15.0}
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
            
            await asyncio.sleep(0.3)
            return data
            
        except Exception as e:
            logging.error(f"데이터 수집 실패 {symbol}: {e}")
            return {}

# ========================================================================================
# 🧠 5가지 전략 분석 엔진
# ========================================================================================

class AdvancedStrategyAnalyzer:
    def calculate_scores(self, data: Dict, vix: float) -> Tuple[float, Dict]:
        scores = {}
        
        scores['buffett'] = self._calculate_buffett_score(data)
        scores['lynch'] = self._calculate_lynch_score(data)
        scores['momentum'] = self._calculate_momentum_score(data)
        scores['technical'] = self._calculate_technical_score(data)
        scores['advanced'] = self._calculate_advanced_indicators_score(data)
        
        weights = config.get('strategy.weights', {})
        total = sum(scores[key] * weights.get(key, 20) for key in scores.keys()) / 100
        
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
# 🤖 고급 손익절 관리자
# ========================================================================================

class AdvancedStopTakeManager:
    def __init__(self, ibkr_trader: IBKRTrader):
        self.ibkr = ibkr_trader
        self.positions: Dict[str, Position] = {}
        self.monitoring = False
        self.db_path = 'legendary_performance_v64.db'
        self._init_database()
    
    def _init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, action TEXT, quantity INTEGER, price REAL,
                    timestamp DATETIME, profit_loss REAL, profit_percent REAL,
                    mode TEXT, entry_day TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY, quantity INTEGER, avg_cost REAL,
                    entry_date DATETIME, mode TEXT, stage INTEGER,
                    tp_executed TEXT, highest_price REAL, entry_day TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"DB 초기화 실패: {e}")
    
    def add_position(self, symbol: str, quantity: int, avg_cost: float, mode: str, entry_day: str = ''):
        position = Position(
            symbol=symbol, quantity=quantity, avg_cost=avg_cost,
            entry_date=datetime.now(), mode=mode, highest_price=avg_cost, entry_day=entry_day
        )
        
        self.positions[symbol] = position
        self._save_position_to_db(position)
        logging.info(f"➕ 포지션 추가: {symbol} {quantity}주 @${avg_cost:.2f} [{entry_day}]")
    
    def _save_position_to_db(self, position: Position):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol, position.quantity, position.avg_cost,
                position.entry_date.isoformat(), position.mode, position.stage,
                json.dumps(position.tp_executed), position.highest_price, position.entry_day
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"포지션 저장 실패: {e}")
    
    async def start_monitoring(self):
        self.monitoring = True
        logging.info("🔍 고급 손익절 모니터링 시작!")
        
        while self.monitoring:
            try:
                await self._monitor_all_positions()
                await asyncio.sleep(15)
            except Exception as e:
                logging.error(f"모니터링 오류: {e}")
                await asyncio.sleep(30)
    
    def stop_monitoring(self):
        self.monitoring = False
    
    async def _monitor_all_positions(self):
        for symbol, position in list(self.positions.items()):
            try:
                current_price = await self.ibkr.get_current_price(symbol)
                if current_price <= 0:
                    continue
                
                position.highest_price = max(position.highest_price, current_price)
                profit_pct = position.profit_percent(current_price)
                
                await self._advanced_exit_logic(symbol, position, current_price, profit_pct)
                
            except Exception as e:
                logging.error(f"포지션 모니터링 실패 {symbol}: {e}")
    
    async def _advanced_exit_logic(self, symbol: str, position: Position, current_price: float, profit_pct: float):
        try:
            selector = StockSelector()
            current_data = await selector.get_stock_data(symbol)
            
            if not current_data:
                return
            
            tp_levels = config.get('trading.swing.take_profit', [7.0, 14.0])
            stop_loss_pct = config.get('trading.swing.stop_loss', 7.0)
            
            # 고급 지표 기반 조정
            macd_data = current_data.get('macd', {})
            bb_data = current_data.get('bollinger', {})
            
            exit_signals = 0
            hold_signals = 0
            
            # MACD 시그널
            if macd_data.get('trend') == 'bearish' or macd_data.get('crossover') == 'sell':
                exit_signals += 1
            elif macd_data.get('trend') == 'bullish':
                hold_signals += 1
            
            # 볼린저 밴드 시그널
            if bb_data.get('signal') == 'overbought' and bb_data.get('position', 0.5) > 0.8:
                exit_signals += 1
            elif bb_data.get('squeeze', False):
                hold_signals += 1
            
            # 동적 손익절 결정
            if profit_pct >= tp_levels[1]:  # 2차 익절선
                await self._execute_partial_exit(symbol, position, current_price, 0.4, '2차익절', profit_pct)
            elif profit_pct >= tp_levels[0]:  # 1차 익절선
                if exit_signals >= 2:  # 강한 매도 시그널
                    await self._execute_partial_exit(symbol, position, current_price, 0.8, '시그널익절', profit_pct)
                else:
                    await self._execute_partial_exit(symbol, position, current_price, 0.6, '1차익절', profit_pct)
            elif profit_pct <= -stop_loss_pct:  # 손절선
                await self._execute_full_exit(symbol, position, current_price, '손절', profit_pct)
            elif exit_signals >= 2 and profit_pct > 3:  # 강한 시그널 + 소폭 수익
                await self._execute_partial_exit(symbol, position, current_price, 0.5, '시그널보호', profit_pct)
            
            # 트레일링 스톱
            if position.highest_price > position.avg_cost * 1.1:
                trailing_pct = 0.95 if hold_signals > exit_signals else 0.93
                trailing_stop = position.highest_price * trailing_pct
                
                if current_price <= trailing_stop:
                    await self._execute_full_exit(symbol, position, current_price, '트레일링', profit_pct)
                    
        except Exception as e:
            logging.error(f"고급 손익절 로직 실패 {symbol}: {e}")
    
    async def _execute_partial_exit(self, symbol: str, position: Position, price: float, ratio: float, reason: str, profit_pct: float):
        if position.tp_executed[0] and ratio > 0.5:
            ratio = 0.4
        elif position.tp_executed[0]:
            return
        
        sell_qty = int(position.quantity * ratio)
        if sell_qty > 0:
            order_id = await self.ibkr.place_sell_order(symbol, sell_qty, reason)
            if order_id:
                position.quantity -= sell_qty
                position.tp_executed[0] = True
                await self._record_trade(symbol, f'SELL_{reason}', sell_qty, price, profit_pct)
                await self._send_notification(f"💰 {symbol} {reason}! +{profit_pct:.1f}% [{position.entry_day}]")
    
    async def _execute_full_exit(self, symbol: str, position: Position, price: float, reason: str, profit_pct: float):
        order_id = await self.ibkr.place_sell_order(symbol, position.quantity, reason)
        if order_id:
            await self._record_trade(symbol, f'SELL_{reason}', position.quantity, price, profit_pct)
            await self._send_notification(f"🔔 {symbol} {reason}! {profit_pct:+.1f}% [{position.entry_day}]")
            del self.positions[symbol]
            await self._remove_position_from_db(symbol)
    
    async def _record_trade(self, symbol: str, action: str, quantity: int, price: float, profit_pct: float):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            profit_loss = 0.0
            if 'SELL' in action and symbol in self.positions:
                position = self.positions[symbol]
                profit_loss = (price - position.avg_cost) * quantity
            
            cursor.execute('''
                INSERT INTO trades 
                VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, action, quantity, price, datetime.now().isoformat(), 
                  profit_loss, profit_pct, 'swing', ''))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"거래 기록 실패: {e}")
    
    async def _remove_position_from_db(self, symbol: str):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
            conn.commit()
            conn.close()
        except:
            pass
    
    async def _send_notification(self, message: str):
        try:
            logging.info(f"📢 {message}")
            if config.get('notifications.telegram.enabled', False):
                await self._send_telegram(message)
        except:
            pass
    
    async def _send_telegram(self, message: str):
        try:
            token = config.get('notifications.telegram.bot_token', '')
            chat_id = config.get('notifications.telegram.chat_id', '')
            
            if not token or not chat_id:
                return
            
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {'chat_id': chat_id, 'text': f"🏆 전설적퀸트 V6.4\n{message}"}
            
            async with aiohttp.ClientSession() as session:
                await session.post(url, json=data)
        except:
            pass

# ========================================================================================
# 🏆 메인 전략 시스템
# ========================================================================================

class LegendaryQuantStrategy:
    def __init__(self):
        self.enabled = config.get('strategy.enabled', True)
        self.current_mode = config.get('strategy.mode', 'swing')
        
        self.dst_manager = DaylightSavingManager()
        self.selector = StockSelector()
        self.analyzer = AdvancedStrategyAnalyzer()
        self.ibkr = IBKRTrader()
        self.stop_take = AdvancedStopTakeManager(self.ibkr)
        
        self.selected_stocks = []
        self.last_selection = None
        self.monthly_return = 0.0
        self.last_trade_dates = {'Tuesday': None, 'Thursday': None}
        
        if self.enabled:
            logging.info("🏆 전설적 퀸트 V6.4 시스템 가동!")
            logging.info(f"🎯 모드: {self.current_mode.upper()}")
            logging.info(f"🕒 서머타임: {'활성' if self.dst_manager.is_dst_active() else '비활성'}")
    
    async def auto_select_stocks(self) -> List[str]:
        if not self.enabled:
            return []
        
        try:
            if (self.last_selection and 
                (datetime.now() - self.last_selection).seconds < 24 * 3600):
                return [s['symbol'] for s in self.selected_stocks]
            
            logging.info("🚀 고급 종목 선별 시작!")
            start_time = time.time()
            
            universe = await self.selector.collect_symbols()
            if not universe:
                return self._get_fallback_stocks()
            
            current_vix = await self.selector.get_current_vix()
            
            scored_stocks = []
            batch_size = 15
            
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i + batch_size]
                tasks = [self._analyze_stock_async(symbol, current_vix) for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result:
                        scored_stocks.append(result)
                
                if i % 75 == 0:
                    logging.info(f"📊 고급분석: {i}/{len(universe)}")
            
            if not scored_stocks:
                return self._get_fallback_stocks()
            
            target_count = config.get(f'strategy.target_stocks.{self.current_mode}', 8)
            final_selection = self._select_best_stocks(scored_stocks, target_count)
            
            self.selected_stocks = final_selection
            self.last_selection = datetime.now()
            
            elapsed = time.time() - start_time
            selected_symbols = [s['symbol'] for s in final_selection]
            
            logging.info(f"🏆 고급선별 완료! {len(selected_symbols)}개 ({elapsed:.1f}초)")
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
            
            if not self._advanced_filter(data, total_score):
                return None
            
            result = data.copy()
            result.update(scores)
            result['vix'] = vix
            
            return result
            
        except Exception as e:
            logging.error(f"종목 분석 실패 {symbol}: {e}")
            return None
    
    def _advanced_filter(self, data: Dict, score: float) -> bool:
        try:
            if score < 0.65:
                return False
            
            macd_data = data.get('macd', {})
            if macd_data.get('trend') == 'bearish' and macd_data.get('histogram', 0) < -0.5:
                return False
            
            bb_data = data.get('bollinger', {})
            if bb_data.get('position', 0.5) > 0.9:
                return False
            
            return True
        except:
            return True
    
    def _select_best_stocks(self, scored_stocks: List[Dict], target_count: int) -> List[Dict]:
        scored_stocks.sort(key=lambda x: x['total'], reverse=True)
        
        final_selection = []
        sector_counts = {}
        
        for stock in scored_stocks:
            if len(final_selection) >= target_count:
                break
            
            sector = stock.get('sector', 'Unknown')
            if sector_counts.get(sector, 0) < 2:
                final_selection.append(stock)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
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
            target_multiplier = 1.0 + (advanced_score * 0.3)
            
            target_price = data['price'] * (1 + confidence * 0.20 * target_multiplier)
            stop_loss = data['price'] * (1 - 0.07)
            
            macd_trend = data.get('macd', {}).get('trend', 'neutral')
            bb_signal = data.get('bollinger', {}).get('signal', 'normal')
            
            reasoning = (f"통합:{total_score:.2f} | "
                        f"MACD:{macd_trend} BB:{bb_signal}")
            
            return StockSignal(
                symbol=symbol, action=action, confidence=confidence, price=data['price'],
                scores=scores, target_price=target_price, stop_loss=stop_loss,
                reasoning=reasoning, timestamp=datetime.now()
            )
            
        except Exception as e:
            return StockSignal(
                symbol=symbol, action='hold', confidence=0.0, price=0.0,
                scores={}, target_price=0.0, stop_loss=0.0,
                reasoning=f"오류: {e}", timestamp=datetime.now()
            )
    
    # ========================================================================================
    # 🕒 서머타임 연동 주 2회 화목 매매
    # ========================================================================================
    
    async def initialize_trading(self) -> bool:
        try:
            logging.info("🚀 고급 거래 시스템 초기화...")
            
            if not await self.ibkr.connect():
                return False
            
            dst_active = self.dst_manager.is_dst_active()
            market_open, market_close = self.dst_manager.get_market_hours_kst()
            
            logging.info(f"🕒 서머타임: {'활성' if dst_active else '비활성'}")
            logging.info(f"📈 시장시간: {market_open.strftime('%H:%M')} - {market_close.strftime('%H:%M')} KST")
            
            await self._load_existing_positions()
            logging.info("✅ 고급 거래 시스템 초기화 완료!")
            return True
            
        except Exception as e:
            logging.error(f"초기화 실패: {e}")
            return False
    
    async def _load_existing_positions(self):
        try:
            conn = sqlite3.connect(self.stop_take.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM positions')
            rows = cursor.fetchall()
            
            for row in rows:
                tp_executed = json.loads(row[6]) if row[6] else [False, False]
                
                position = Position(
                    symbol=row[0], quantity=row[1], avg_cost=row[2],
                    entry_date=datetime.fromisoformat(row[3]), mode=row[4],
                    stage=row[5], tp_executed=tp_executed,
                    highest_price=row[7], entry_day=row[8]
                )
                
                self.stop_take.positions[position.symbol] = position
            
            conn.close()
            logging.info(f"📂 기존 포지션 로드: {len(self.stop_take.positions)}개")
            
        except Exception as e:
            logging.error(f"포지션 로드 실패: {e}")
    
    async def start_auto_trading(self):
        try:
            logging.info("🎯 서머타임 연동 자동거래 시작!")
            
            monitor_task = asyncio.create_task(self.stop_take.start_monitoring())
            schedule_task = asyncio.create_task(self._run_dst_schedule())
            
            await asyncio.gather(monitor_task, schedule_task)
            
        except Exception as e:
            logging.error(f"자동거래 실행 실패: {e}")
        finally:
            await self.shutdown()
    
    async def _run_dst_schedule(self):
        logging.info("📅 서머타임 연동 화목 매매 스케줄러 시작!")
        
        while True:
            try:
                now = datetime.now(self.dst_manager.korea)
                weekday = now.weekday()
                
                trading_times = self.dst_manager.get_trading_times_kst(now.date())
                dst_status = "EDT" if trading_times['dst_active'] else "EST"
                
                if (weekday == 1 and self._is_trading_time(now, trading_times['market_time_kst']) and
                    self.last_trade_dates['Tuesday'] != now.date() and self._is_trading_day()):
                    
                    logging.info(f"🔥 화요일 매매 시작! ({dst_status})")
                    await self._execute_tuesday_trading()
                    self.last_trade_dates['Tuesday'] = now.date()
                        
                elif (weekday == 3 and self._is_trading_time(now, trading_times['market_time_kst']) and
                      self.last_trade_dates['Thursday'] != now.date() and self._is_trading_day()):
                    
                    logging.info(f"📋 목요일 매매 시작! ({dst_status})")
                    await self._execute_thursday_trading()
                    self.last_trade_dates['Thursday'] = now.date()
                
                if now.hour == 9 and now.minute == 0:
                    await self._perform_daily_check()
                
                if now.hour == 16 and now.minute == 0:
                    await self._generate_enhanced_report()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logging.error(f"스케줄 오류: {e}")
                await asyncio.sleep(60)
    
    def _is_trading_time(self, current_time, target_time) -> bool:
        time_diff = abs((current_time - target_time).total_seconds())
        return time_diff <= 1800
    
    async def _execute_tuesday_trading(self):
        try:
            market_condition = await self._analyze_advanced_market()
            if not market_condition['safe_to_trade']:
                await self.stop_take._send_notification(
                    f"⚠️ 화요일 매매 스킵\n📊 사유: {market_condition['reason']}")
                return
            
            base_targets = config.get('trading.weekly.tuesday_targets', 4)
            signal_strength = market_condition.get('signal_strength', 1.0)
            adjusted_targets = max(2, min(6, int(base_targets * signal_strength)))
            
            selected = await self.auto_select_stocks()
            if not selected:
                return
            
            existing_symbols = list(self.stop_take.positions.keys())
            new_candidates = [s for s in selected if s not in existing_symbols][:adjusted_targets]
            
            portfolio_value = await self.ibkr.get_portfolio_value()
            base_allocation = config.get('trading.weekly.tuesday_allocation', 13.0) / 100
            risk_adjusted_allocation = base_allocation * market_condition.get('risk_factor', 1.0)
            
            new_entries = 0
            total_investment = 0
            
            for symbol in new_candidates:
                try:
                    investment_amount = portfolio_value * risk_adjusted_allocation
                    success = await self._enter_position_with_signals(symbol, investment_amount, 'Tuesday')
                    if success:
                        new_entries += 1
                        total_investment += investment_amount
                    await asyncio.sleep(2)
                except Exception as e:
                    logging.error(f"화요일 {symbol} 진입 실패: {e}")
            
            await self.stop_take._send_notification(
                f"🔥 화요일 공격적 진입 완료!\n"
                f"📊 시장: {market_condition['status']} (VIX: {market_condition.get('vix', 0):.1f})\n"
                f"🎯 시그널강도: {signal_strength:.1f}x\n"
                f"💰 신규진입: {new_entries}/{len(new_candidates)}개\n"
                f"💵 투자금액: ${total_investment:.0f}"
            )
            
        except Exception as e:
            logging.error(f"화요일 매매 실패: {e}")
    
    async def _analyze_advanced_market(self) -> Dict:
        try:
            vix = await self.selector.get_current_vix()
            spy_data = await self.selector.get_stock_data('SPY')
            qqq_data = await self.selector.get_stock_data('QQQ')
            
            if not spy_data or not qqq_data:
                return {'safe_to_trade': False, 'reason': '지수 데이터 없음'}
            
            spy_macd = spy_data.get('macd', {})
            spy_bb = spy_data.get('bollinger', {})
            qqq_macd = qqq_data.get('macd', {})
            qqq_bb = qqq_data.get('bollinger', {})
            
            bullish_signals = 0
            bearish_signals = 0
            
            if spy_macd.get('trend') == 'bullish': bullish_signals += 2
            elif spy_macd.get('trend') == 'bearish': bearish_signals += 2
            if qqq_macd.get('trend') == 'bullish': bullish_signals += 2
            elif qqq_macd.get('trend') == 'bearish': bearish_signals += 2
            
            if spy_bb.get('signal') == 'oversold': bullish_signals += 1
            elif spy_bb.get('signal') == 'overbought': bearish_signals += 1
            if qqq_bb.get('signal') == 'oversold': bullish_signals += 1
            elif qqq_bb.get('signal') == 'overbought': bearish_signals += 1
            
            signal_strength = max(0.5, min(1.5, 1.0 + (bullish_signals - bearish_signals) * 0.1))
            
            condition = {
                'vix': vix,
                'spy_signals': {'macd': spy_macd.get('trend'), 'bb': spy_bb.get('signal')},
                'qqq_signals': {'macd': qqq_macd.get('trend'), 'bb': qqq_bb.get('signal')},
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'signal_strength': signal_strength,
                'safe_to_trade': True,
                'status': 'normal',
                'reason': '',
                'risk_factor': 1.0
            }
            
            if vix > 35:
                condition.update({
                    'safe_to_trade': False,
                    'status': 'high_volatility',
                    'reason': f'VIX 과도함: {vix:.1f}'
                })
            elif vix > 25:
                condition.update({
                    'status': 'volatile',
                    'risk_factor': 0.7,
                    'signal_strength': signal_strength * 0.8
                })
            elif vix < 15:
                condition.update({
                    'status': 'low_volatility',
                    'risk_factor': 1.2,
                    'signal_strength': signal_strength * 1.1
                })
            
            if bearish_signals > bullish_signals + 2:
                condition.update({
                    'safe_to_trade': False,
                    'status': 'bearish_trend',
                    'reason': f'베어리시 신호 과다: {bearish_signals}vs{bullish_signals}'
                })
            
            return condition
            
        except Exception as e:
            logging.error(f"고급 시장 분석 실패: {e}")
            return {'safe_to_trade': False, 'reason': f'분석 실패: {e}', 'signal_strength': 0.5}
    
    async def _enter_position_with_signals(self, symbol: str, investment: float, entry_day: str) -> bool:
        try:
            current_price = await self.ibkr.get_current_price(symbol)
            if current_price <= 0:
                return False
            
            signal = await self.analyze_stock_signal(symbol)
            if signal.action != 'buy' or signal.confidence < 0.7:
                logging.info(f"⚠️ {symbol} 시그널 부족: {signal.action} ({signal.confidence:.2f})")
                return False
            
            quantity = int(investment / current_price)
            if quantity < 1:
                return False
            
            order_id = await self.ibkr.place_buy_order(symbol, quantity)
            
            if order_id:
                self.stop_take.add_position(symbol, quantity, current_price, 'swing', entry_day)
                
                investment_value = quantity * current_price
                await self.stop_take._send_notification(
                    f"🚀 {symbol} 고급시그널 진입! ({entry_day})\n"
                    f"💰 ${investment_value:.0f} ({quantity}주 @${current_price:.2f})\n"
                    f"📊 신뢰도: {signal.confidence:.1%}\n"
                    f"🎯 목표: ${signal.target_price:.2f}"
                )
                
                return True
            
            return False
                
        except Exception as e:
            logging.error(f"고급 포지션 진입 실패 {symbol}: {e}")
            return False
    
    async def _execute_thursday_trading(self):
        try:
            market_condition = await self._analyze_advanced_market()
            weekly_performance = await self._analyze_weekly_performance()
            
            actions_taken = await self._thursday_advanced_review(weekly_performance, market_condition)
            
            if (weekly_performance['weekly_return'] >= 0 and 
                market_condition['safe_to_trade'] and
                market_condition.get('signal_strength', 1.0) > 0.9):
                new_entries = await self._thursday_selective_entry()
                actions_taken['new_entries'] = new_entries
            
            await self.stop_take._send_notification(
                f"📋 목요일 고급정리 완료!\n"
                f"💰 이익실현: {actions_taken.get('profit_taken', 0)}개\n"
                f"🛑 손절청산: {actions_taken.get('stop_losses', 0)}개\n"
                f"📊 신규진입: {actions_taken.get('new_entries', 0)}개\n"
                f"📈 주간수익률: {weekly_performance['weekly_return']:+.2f}%\n"
                f"🎯 시그널강도: {market_condition.get('signal_strength', 1.0):.1f}x"
            )
            
        except Exception as e:
            logging.error(f"목요일 매매 실패: {e}")
    
    async def _thursday_advanced_review(self, weekly_performance: Dict, market_condition: Dict) -> Dict:
        try:
            actions_taken = {'profit_taken': 0, 'stop_losses': 0, 'held_positions': 0}
            
            for symbol, position in list(self.stop_take.positions.items()):
                try:
                    current_price = await self.ibkr.get_current_price(symbol)
                    if current_price <= 0:
                        continue
                    
                    profit_pct = position.profit_percent(current_price)
                    hold_days = (datetime.now() - position.entry_date).days
                    
                    current_data = await self.selector.get_stock_data(symbol)
                    action = self._thursday_advanced_decision(
                        symbol, position, profit_pct, hold_days, 
                        weekly_performance, market_condition, current_data
                    )
                    
                    if action == 'TAKE_PROFIT':
                        sell_qty = int(position.quantity * 0.6)
                        if sell_qty > 0:
                            order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'Thursday-Advanced-Profit')
                            if order_id:
                                actions_taken['profit_taken'] += 1
                                position.quantity -= sell_qty
                                self.stop_take._save_position_to_db(position)
                    
                    elif action == 'FULL_EXIT':
                        order_id = await self.ibkr.place_sell_order(symbol, position.quantity, 'Thursday-Advanced-Exit')
                        if order_id:
                            actions_taken['stop_losses'] += 1
                            del self.stop_take.positions[symbol]
                            await self.stop_take._remove_position_from_db(symbol)
                    
                    else:
                        actions_taken['held_positions'] += 1
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logging.error(f"목요일 {symbol} 리뷰 실패: {e}")
            
            return actions_taken
            
        except Exception as e:
            logging.error(f"목요일 고급 리뷰 실패: {e}")
            return {'profit_taken': 0, 'stop_losses': 0, 'held_positions': 0}
    
    def _thursday_advanced_decision(self, symbol: str, position, profit_pct: float, hold_days: int,
                                   weekly_performance: Dict, market_condition: Dict, current_data: Dict) -> str:
        try:
            profit_threshold = config.get('trading.weekly.profit_taking_threshold', 9.0)
            loss_threshold = config.get('trading.weekly.loss_cutting_threshold', -5.5)
            
            if profit_pct >= profit_threshold:
                return 'TAKE_PROFIT'
            
            if profit_pct <= loss_threshold:
                return 'FULL_EXIT'
            
            if current_data:
                exit_signals = 0
                hold_signals = 0
                
                macd_data = current_data.get('macd', {})
                if macd_data.get('trend') == 'bearish' or macd_data.get('crossover') == 'sell':
                    exit_signals += 2
                elif macd_data.get('trend') == 'bullish':
                    hold_signals += 1
                
                bb_data = current_data.get('bollinger', {})
                if bb_data.get('signal') == 'overbought' and bb_data.get('position', 0.5) > 0.85:
                    exit_signals += 1
                elif bb_data.get('squeeze', False):
                    hold_signals += 1
                
                if exit_signals >= 3 and profit_pct > 2:
                    return 'TAKE_PROFIT'
                elif exit_signals >= 2 and profit_pct < 0:
                    return 'FULL_EXIT'
            
            if weekly_performance['weekly_return'] < -2.0 and profit_pct < 1:
                return 'FULL_EXIT'
            
            if market_condition.get('signal_strength', 1.0) < 0.7 and profit_pct < 2:
                return 'FULL_EXIT'
            
            if hold_days >= 8 and -1 <= profit_pct <= 3:
                return 'FULL_EXIT'
            
            return 'HOLD'
            
        except Exception as e:
            logging.error(f"목요일 고급결정 로직 오류 {symbol}: {e}")
            return 'HOLD'
    
    async def _thursday_selective_entry(self) -> int:
        try:
            max_new_entries = config.get('trading.weekly.thursday_targets', 2)
            
            current_positions = len(self.stop_take.positions)
            if current_positions >= 8:
                return 0
            
            selected = await self.auto_select_stocks()
            if not selected:
                return 0
            
            existing_symbols = list(self.stop_take.positions.keys())
            new_candidates = []
            
            for symbol in selected:
                if symbol not in existing_symbols:
                    signal = await self.analyze_stock_signal(symbol)
                    if signal.confidence >= 0.85:
                        new_candidates.append(symbol)
                        if len(new_candidates) >= max_new_entries:
                            break
            
            portfolio_value = await self.ibkr.get_portfolio_value()
            conservative_allocation = config.get('trading.weekly.thursday_allocation', 8.0) / 100
            
            new_entries = 0
            for symbol in new_candidates:
                try:
                    investment_amount = portfolio_value * conservative_allocation
                    success = await self._enter_position_with_signals(symbol, investment_amount, 'Thursday')
                    if success:
                        new_entries += 1
                    await asyncio.sleep(2)
                except Exception as e:
                    logging.error(f"목요일 {symbol} 진입 실패: {e}")
            
            return new_entries
            
        except Exception as e:
            logging.error(f"목요일 선별적 진입 실패: {e}")
            return 0
    
    async def _analyze_weekly_performance(self) -> Dict:
        try:
            now = datetime.now()
            week_start = now - timedelta(days=now.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            conn = sqlite3.connect(self.stop_take.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT SUM(profit_loss), COUNT(*) FROM trades 
                WHERE timestamp >= ? AND action LIKE 'SELL%'
            ''', (week_start.isoformat(),))
            
            result = cursor.fetchone()
            weekly_profit = result[0] if result[0] else 0.0
            weekly_trades = result[1] if result[1] else 0
            
            portfolio_value = await self.ibkr.get_portfolio_value()
            weekly_return = (weekly_profit / portfolio_value) * 100 if portfolio_value > 0 else 0.0
            
            conn.close()
            
            return {
                'weekly_profit': weekly_profit,
                'weekly_return': weekly_return,
                'weekly_trades': weekly_trades,
                'portfolio_value': portfolio_value
            }
            
        except Exception as e:
            logging.error(f"주간 성과 분석 실패: {e}")
            return {'weekly_profit': 0.0, 'weekly_return': 0.0, 'weekly_trades': 0, 'portfolio_value': 1000000}
    
    def _is_trading_day(self) -> bool:
        today = datetime.now()
        return today.weekday() < 5
    
    async def _perform_daily_check(self):
        try:
            if not self._is_trading_day():
                return
            
            await self.ibkr._update_account()
            await self._calculate_monthly_return()
            
            dst_active = self.dst_manager.is_dst_active()
            if dst_active != self.dst_manager.is_dst_active(datetime.now().date() - timedelta(days=1)):
                await self.stop_take._send_notification(
                    f"🕒 서머타임 변경!\n{'EDT 시작' if dst_active else 'EST 시작'}"
                )
            
        except Exception as e:
            logging.error(f"일일 체크 실패: {e}")
    
    async def _calculate_monthly_return(self):
        try:
            conn = sqlite3.connect(self.stop_take.db_path)
            cursor = conn.cursor()
            
            current_month = datetime.now().strftime('%Y-%m')
            cursor.execute('''
                SELECT SUM(profit_loss) FROM trades 
                WHERE strftime('%Y-%m', timestamp) = ? AND action LIKE 'SELL%'
            ''', (current_month,))
            
            result = cursor.fetchone()
            monthly_profit = result[0] if result[0] else 0.0
            
            portfolio_value = await self.ibkr.get_portfolio_value()
            if portfolio_value > 0:
                self.monthly_return = (monthly_profit / portfolio_value) * 100
            
            conn.close()
        except Exception as e:
            logging.error(f"월 수익률 계산 실패: {e}")
    
    async def _generate_enhanced_report(self):
        try:
            active_positions = len(self.stop_take.positions)
            daily_pnl = self.ibkr.daily_pnl
            
            dst_active = self.dst_manager.is_dst_active()
            market_open, market_close = self.dst_manager.get_market_hours_kst()
            
            today = datetime.now()
            weekday_info = ""
            
            if today.weekday() == 1:
                if self.last_trade_dates.get('Tuesday') == today.date():
                    weekday_info = "🔥 오늘 화요일 고급진입 완료"
            elif today.weekday() == 3:
                if self.last_trade_dates.get('Thursday') == today.date():
                    weekday_info = "📋 오늘 목요일 고급정리 완료"
            
            market_condition = await self._analyze_advanced_market()
            
            report = f"""
🏆 고급 일일 리포트 V6.4
========================
📊 모드: {self.current_mode.upper()} | 🕒 {dst_active and 'EDT' or 'EST'}
📈 시장시간: {market_open.strftime('%H:%M')}-{market_close.strftime('%H:%M')}
💰 일일 P&L: ${daily_pnl:.2f}
📈 월 수익률: {self.monthly_return:.2f}% (목표: 6-8%)
💼 활성 포지션: {active_positions}개
🎯 시그널강도: {market_condition.get('signal_strength', 1.0):.1f}x
📊 시장상태: {market_condition.get('status', 'unknown')}
{weekday_info}
"""
            
            await self.stop_take._send_notification(report)
            
        except Exception as e:
            logging.error(f"고급 리포트 생성 실패: {e}")
    
    async def shutdown(self):
        try:
            logging.info("🔌 고급 시스템 종료 중...")
            self.stop_take.stop_monitoring()
            await self.ibkr.disconnect()
            logging.info("✅ 시스템 종료 완료")
        except Exception as e:
            logging.error(f"종료 실패: {e}")

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

async def run_auto_trading():
    strategy = LegendaryQuantStrategy()
    try:
        if await strategy.initialize_trading():
            await strategy.start_auto_trading()
        else:
            logging.error("❌ 시스템 초기화 실패")
    except KeyboardInterrupt:
        logging.info("⏹️ 사용자 중단")
    except Exception as e:
        logging.error(f"❌ 자동거래 실패: {e}")
    finally:
        await strategy.shutdown()

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
            'monthly_return': strategy.monthly_return,
            'dst_active': dst_active,
            'market_hours_kst': f"{market_open.strftime('%H:%M')}-{market_close.strftime('%H:%M')}",
            'timezone_status': 'EDT' if dst_active else 'EST',
            'advanced_indicators': True,
            'last_tuesday': strategy.last_trade_dates.get('Tuesday'),
            'last_thursday': strategy.last_trade_dates.get('Thursday')
        }
        
    except Exception as e:
        return {'error': str(e)}

# 고급 함수들
async def test_advanced_indicators(symbol: str = 'AAPL'):
    try:
        strategy = LegendaryQuantStrategy()
        data = await strategy.selector.get_stock_data(symbol)
        
        if data:
            return {
                'symbol': symbol,
                'price': data['price'],
                'macd': data.get('macd', {}),
                'bollinger': data.get('bollinger', {}),
                'traditional': {'rsi': data.get('rsi', 50), 'volume_spike': data.get('volume_spike', 1)}
            }
        else:
            return {'error': '데이터 없음'}
    except Exception as e:
        return {'error': str(e)}

async def check_dst_status():
    try:
        dst_manager = DaylightSavingManager()
        now = datetime.now()
        
        dst_active = dst_manager.is_dst_active()
        market_open, market_close = dst_manager.get_market_hours_kst()
        trading_times = dst_manager.get_trading_times_kst()
        
        return {
            'current_time_kst': now.strftime('%Y-%m-%d %H:%M:%S'),
            'dst_active': dst_active,
            'timezone': 'EDT' if dst_active else 'EST',
            'market_open_kst': market_open.strftime('%H:%M'),
            'market_close_kst': market_close.strftime('%H:%M'),
            'tuesday_trading_kst': trading_times['market_time_kst'].strftime('%H:%M') if trading_times['tuesday_kst'] else None,
            'thursday_trading_kst': trading_times['market_time_kst'].strftime('%H:%M') if trading_times['thursday_kst'] else None,
            'is_market_hours': dst_manager.is_market_hours()
        }
    except Exception as e:
        return {'error': str(e)}

async def manual_tuesday_trading():
    strategy = LegendaryQuantStrategy()
    try:
        if await strategy.initialize_trading():
            await strategy._execute_tuesday_trading()
            return {'status': 'success', 'message': '화요일 고급매매 완료'}
        else:
            return {'status': 'error', 'message': '시스템 초기화 실패'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    finally:
        await strategy.shutdown()

async def manual_thursday_trading():
    strategy = LegendaryQuantStrategy()
    try:
        if await strategy.initialize_trading():
            await strategy._execute_thursday_trading()
            return {'status': 'success', 'message': '목요일 고급매매 완료'}
        else:
            return {'status': 'error', 'message': '시스템 초기화 실패'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    finally:
        await strategy.shutdown()

async def get_advanced_market_analysis():
    try:
        strategy = LegendaryQuantStrategy()
        return await strategy._analyze_advanced_market()
    except Exception as e:
        return {'error': str(e)}

async def scan_with_advanced_indicators():
    try:
        signals = await run_auto_selection()
        return {'signals': [{'symbol': s.symbol, 'action': s.action, 'confidence': s.confidence, 
                            'price': s.price, 'reasoning': s.reasoning} for s in signals], 
                'total_scanned': len(signals)}
    except Exception as e:
        return {'error': str(e)}

# 빠른 실행 함수들
async def quick_advanced_analysis(symbols: List[str] = None):
    if not symbols:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    
    print(f"🚀 고급 분석: {', '.join(symbols)}")
    
    strategy = LegendaryQuantStrategy()
    
    for symbol in symbols:
        try:
            signal = await strategy.analyze_stock_signal(symbol)
            action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
            print(f"{action_emoji} {symbol}: {signal.action} ({signal.confidence:.1%}) - {signal.reasoning[:50]}...")
        except:
            print(f"❌ {symbol}: 분석 실패")

async def quick_dst_check():
    print("🕒 서머타임 상태 체크...")
    
    try:
        dst_info = await check_dst_status()
        if 'error' not in dst_info:
            print(f"📅 현재시간: {dst_info['current_time_kst']}")
            print(f"🕒 시간대: {dst_info['timezone']} ({'활성' if dst_info['dst_active'] else '비활성'})")
            print(f"📈 시장시간: {dst_info['market_open_kst']}-{dst_info['market_close_kst']} KST")
            print(f"📊 거래중: {'✅' if dst_info['is_market_hours'] else '❌'}")
        else:
            print(f"❌ 체크 실패: {dst_info['error']}")
    except Exception as e:
        print(f"❌ 체크 실패: {e}")

async def quick_market_signals():
    print("📊 고급 시장 시그널 분석...")
    
    try:
        analysis = await get_advanced_market_analysis()
        if 'error' not in analysis:
            print(f"📊 VIX: {analysis.get('vix', 0):.1f}")
            print(f"🎯 시그널강도: {analysis.get('signal_strength', 1.0):.1f}x")
            print(f"📈 상태: {analysis.get('status', 'unknown')}")
            print(f"💰 매매가능: {'✅' if analysis.get('safe_to_trade') else '❌'}")
            
            spy_signals = analysis.get('spy_signals', {})
            print(f"🔵 SPY: MACD({spy_signals.get('macd', 'unknown')}) BB({spy_signals.get('bb', 'unknown')})")
        else:
            print(f"❌ 분석 실패: {analysis['error']}")
    except Exception as e:
        print(f"❌ 분석 실패: {e}")

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
                logging.FileHandler('legendary_quant_v64.log', encoding='utf-8')
            ]
        )
        
        print("🏆" + "="*70)
        print("🔥 전설적 퀸트프로젝트 V6.4 - 서머타임 + 고급기술지표")
        print("🚀 월 6-8% 달성형 5가지 전략 융합 시스템")
        print("="*72)
        
        print("\n🌟 V6.4 신기능:")
        print("  ✨ 🆕 서머타임 완전 자동화 (EDT/EST 자동전환)")
        print("  ✨ 🆕 고급 기술지표 2종 (MACD + 볼린저밴드)")
        print("  ✨ 🆕 5가지 전략 융합 (버핏+린치+모멘텀+기술+고급)")
        print("  ✨ 🆕 동적 손익절 (시그널 기반 적응형)")
        print("  ✨ 🆕 월 목표 상향 (6-8% vs 기존 5-7%)")
        
        print("\n🕒 서머타임 기능:")
        print("  📅 미국 EDT/EST 자동 감지")
        print("  ⏰ 한국시간 거래시간 동적 계산")
        print("  🔄 3월/11월 전환일 자동 처리")
        print("  📊 시간대별 매매시간 실시간 조정")
        
        print("\n📈 고급 기술지표:")
        print("  📊 MACD: 추세 및 모멘텀 변화 포착")
        print("  📊 볼린저밴드: 과매수/과매도 + 변동성 스퀴즈")
        print("  🎯 2개 지표 종합 시그널 강도 계산")
        
        status = await get_system_status()
        
        if 'error' not in status:
            print(f"\n🔧 시스템 상태:")
            print(f"  ✅ 시스템: {status['current_mode'].upper()}")
            print(f"  🕒 시간대: {status['timezone_status']} ({'서머타임' if status['dst_active'] else '표준시'})")
            print(f"  📈 시장시간: {status['market_hours_kst']} KST")
            print(f"  🤖 IBKR: {'연결가능' if status['ibkr_connected'] else '연결불가'}")
            print(f"  📊 고급지표: {'활성화' if status['advanced_indicators'] else '비활성화'}")
            print(f"  📈 월 수익률: {status['monthly_return']:.2f}%")
        else:
            print(f"  ❌ 상태 확인 실패: {status['error']}")
        
        print("\n🚀 실행 옵션:")
        print("  1. 🏆 완전 자동 서머타임 연동 매매")
        print("  2. 🔥 수동 화요일 고급매매")
        print("  3. 📋 수동 목요일 고급매매")
        print("  4. 🔍 고급 지표 종목 선별")
        print("  5. 📊 개별 종목 고급 분석")
        print("  6. 🕒 서머타임 상태 확인")
        print("  7. 📈 고급 시장 시그널 분석")
        print("  8. 🧪 고급 지표 테스트")
        print("  0. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (0-8): ").strip()
                
                if choice == '1':
                    print("\n🏆 서머타임 연동 완전 자동매매!")
                    print("🕒 EDT/EST 자동전환 + 고급기술지표")
                    print("📊 5가지 전략 융합 + 동적 손익절")
                    print("🎯 월 6-8% 목표 달성형")
                    confirm = input("시작하시겠습니까? (y/N): ").strip().lower()
                    if confirm == 'y':
                        await run_auto_trading()
                    break
                
                elif choice == '2':
                    print("\n🔥 화요일 고급매매!")
                    confirm = input("실행하시겠습니까? (y/N): ").strip().lower()
                    if confirm == 'y':
                        result = await manual_tuesday_trading()
                        print(f"{'✅' if result['status'] == 'success' else '❌'} {result['message']}")
                
                elif choice == '3':
                    print("\n📋 목요일 고급매매!")
                    confirm = input("실행하시겠습니까? (y/N): ").strip().lower()
                    if confirm == 'y':
                        result = await manual_thursday_trading()
                        print(f"{'✅' if result['status'] == 'success' else '❌'} {result['message']}")
                
                elif choice == '4':
                    print("\n🔍 고급 지표 종목 선별!")
                    result = await scan_with_advanced_indicators()
                    
                    if 'error' not in result:
                        signals = result['signals']
                        print(f"\n📈 고급분석 결과: {result['total_scanned']}개 스캔")
                        
                        buy_signals = [s for s in signals if s['action'] == 'buy']
                        print(f"🟢 매수추천: {len(buy_signals)}개")
                        
                        for i, signal in enumerate(buy_signals[:3], 1):
                            print(f"  {i}. {signal['symbol']}: {signal['confidence']:.1%} - {signal['reasoning'][:60]}...")
                    else:
                        print(f"❌ 스캔 실패: {result['error']}")
                
                elif choice == '5':
                    symbol = input("분석할 종목 심볼: ").strip().upper()
                    if symbol:
                        print(f"\n🔍 {symbol} 고급 분석...")
                        
                        indicators = await test_advanced_indicators(symbol)
                        if 'error' not in indicators:
                            print(f"💰 현재가: ${indicators['price']:.2f}")
                            
                            macd = indicators.get('macd', {})
                            print(f"📊 MACD: {macd.get('trend', 'unknown')} (크로스오버: {macd.get('crossover', 'none')})")
                            
                            bb = indicators.get('bollinger', {})
                            print(f"📊 볼린저: {bb.get('signal', 'unknown')} (위치: {bb.get('position', 0.5):.2f})")
                            
                            signal = await analyze_single_stock(symbol)
                            if signal and signal.confidence > 0:
                                print(f"\n🎯 종합결론: {signal.action.upper()} (신뢰도: {signal.confidence:.1%})")
                                print(f"💡 근거: {signal.reasoning}")
                        else:
                            print(f"❌ 분석 실패: {indicators['error']}")
                
                elif choice == '6':
                    print("\n🕒 서머타임 상태 확인...")
                    await quick_dst_check()
                
                elif choice == '7':
                    print("\n📈 고급 시장 시그널 분석...")
                    await quick_market_signals()
                
                elif choice == '8':
                    print("\n🧪 고급 지표 테스트...")
                    symbols = input("테스트할 종목들 (쉼표로 구분, 엔터시 기본값): ").strip()
                    if symbols:
                        symbol_list = [s.strip().upper() for s in symbols.split(',')]
                    else:
                        symbol_list = None
                    await quick_advanced_analysis(symbol_list)
                
                elif choice == '0':
                    print("👋 V6.4 시스템을 종료합니다!")
                    break
                    
                else:
                    print("❌ 잘못된 선택입니다. 0-8 중 선택하세요.")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
    except Exception as e:
        logging.error(f"메인 실행 실패: {e}")
        print(f"❌ 실행 실패: {e}")

def print_v64_help():
    help_text = """
🏆 전설적 퀸트프로젝트 V6.4 - 서머타임 + 고급기술지표 (최적화)
================================================================

📋 주요 명령어:
  python legendary_quant_v64.py                    # 메인 메뉴
  python -c "from legendary_quant_v64 import *; asyncio.run(quick_dst_check())"      # 서머타임 체크
  python -c "from legendary_quant_v64 import *; asyncio.run(quick_market_signals())" # 시장 시그널
  python -c "from legendary_quant_v64 import *; asyncio.run(quick_advanced_analysis())" # 고급 분석

🔧 V6.4 설정:
  1. pip install yfinance pandas numpy requests aiohttp python-dotenv pytz
  2. IBKR 사용시: pip install ib_insync
  3. .env 파일에서 텔레그램 설정

🆕 V6.4 최적화:
  🕒 서머타임 완전 자동화
    - EDT/EST 자동 감지 및 전환
    - 한국시간 기준 거래시간 동적 계산
    - 3월/11월 전환일 특별 처리
    
  📈 고급 기술지표 2종 (최적화)
    - MACD: 추세변화 및 크로스오버 시그널
    - 볼린저밴드: 과매수/과매도 + 변동성 스퀴즈
    
  🧠 5가지 전략 융합
    - 버핏 가치투자: 20%
    - 린치 성장투자: 20%  
    - 모멘텀 전략: 20%
    - 기술적 분석: 25%
    - 🆕 고급지표: 15%
    
  🎯 향상된 성능
    - 월 목표 수익률: 6-8%
    - 동적 손익절: 고급지표 기반 적응형
    - 시그널 강도: 2개 지표 종합 판단

🕒 서머타임 스케줄:
  📅 3월 둘째주 일요일 ~ 11월 첫째주 일요일: EDT (UTC-4)
  📅 나머지 기간: EST (UTC-5)
  
  🔥 화요일 진입: 미국 10:30 AM ET
    - EDT시: 한국 23:30 (당일)
    - EST시: 한국 00:30 (다음날)
    
  📋 목요일 정리: 미국 10:30 AM ET  
    - EDT시: 한국 23:30 (당일)
    - EST시: 한국 00:30 (다음날)

💡 사용 팁:
  - 서머타임 전환 주간에는 시간 확인 필수
  - 고급지표 신뢰도 85% 이상만 진입
  - 2개 지표 일치시 신뢰도 높음
  - VIX 30 이상시 자동 매매 중단
  - 코드 최적화로 안정성 향상
"""
    print(help_text)

# ========================================================================================
# 🏁 실행 진입점
# ========================================================================================

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] in ['help', '--help']:
                print_v64_help()
                sys.exit(0)
            elif sys.argv[1] == 'dst-check':
                asyncio.run(quick_dst_check())
                sys.exit(0)
            elif sys.argv[1] == 'market-signals':
                asyncio.run(quick_market_signals())
                sys.exit(0)
            elif sys.argv[1] == 'advanced-analysis':
                symbols = sys.argv[2:] if len(sys.argv) > 2 else None
                asyncio.run(quick_advanced_analysis(symbols))
                sys.exit(0)
            elif sys.argv[1] == 'tuesday':
                asyncio.run(manual_tuesday_trading())
                sys.exit(0)
            elif sys.argv[1] == 'thursday':
                asyncio.run(manual_thursday_trading())
                sys.exit(0)
        
        # 메인 실행
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 V6.4 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ V6.4 실행 오류: {e}")
        logging.error(f"V6.4 실행 오류: {e}")
