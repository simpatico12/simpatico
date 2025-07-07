#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 전설적 퀸트프로젝트 - 간소화 완성판 V6.3
===============================================
월 5-7% 달성형 주 2회 화목 매매 시스템
기존 기능 100% 유지 + 코드 최적화

Author: 전설적퀸트팀
Version: 6.3.0 (간소화)
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

# 타입 힌트
try:
    from typing import Dict, List, Optional, Tuple, Any
except ImportError:
    Dict = dict
    List = list
    Optional = lambda x: x
    Tuple = tuple
    Any = object

# IBKR 연동
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    logging.warning("⚠️ IBKR 모듈 없음 (pip install ib_insync 필요)")

warnings.filterwarnings('ignore')

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
                'monthly_target': {'min': 5.0, 'max': 7.0},
                'weights': {'buffett': 25.0, 'lynch': 25.0, 'momentum': 25.0, 'technical': 25.0},
                'vix_thresholds': {'low': 15.0, 'high': 30.0}
            },
            'trading': {
                'swing': {'take_profit': [6.0, 12.0], 'profit_ratios': [60.0, 40.0], 'stop_loss': 8.0},
                'classic': {'take_profit': [20.0, 35.0], 'stop_loss': 15.0},
                'weekly': {
                    'enabled': True,
                    'tuesday_targets': 4,
                    'thursday_targets': 2,
                    'tuesday_allocation': 12.5,
                    'thursday_allocation': 8.0,
                    'profit_taking_threshold': 8.0,
                    'loss_cutting_threshold': -6.0,
                    'tuesday_time': '10:30',
                    'thursday_time': '10:30'
                }
            },
            'risk': {
                'max_position': 15.0,
                'daily_loss_limit': 1.0,
                'monthly_loss_limit': 3.0,
                'weekly_loss_limit': 3.0,
                'trailing_stop': True
            },
            'selection': {
                'min_market_cap': 5_000_000_000,
                'min_volume': 1_000_000,
                'excluded_symbols': ['SPXL', 'TQQQ'],
                'refresh_hours': 24
            },
            'ibkr': {
                'enabled': True,
                'host': '127.0.0.1',
                'port': 7497,
                'client_id': 1,
                'paper_trading': True,
                'max_daily_trades': 20
            },
            'notifications': {
                'telegram': {
                    'enabled': True,
                    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
                }
            },
            'performance': {'database_file': 'legendary_performance.db'}
        }
        
        if Path('.env').exists():
            load_dotenv()
    
    def get(self, key_path: str, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

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
    mode: str
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
    tp_executed: List[bool] = field(default_factory=lambda: [False, False, False])
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
            
            # S&P 500
            try:
                tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
                sp500 = [str(s).replace('.', '-') for s in tables[0]['Symbol'].tolist()]
                self.cache['sp500'] = sp500
            except:
                sp500 = self._get_backup_sp500()
            
            # NASDAQ 100
            try:
                tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
                nasdaq = []
                for table in tables:
                    if 'Symbol' in table.columns:
                        nasdaq = table['Symbol'].dropna().tolist()
                        break
                self.cache['nasdaq'] = nasdaq
            except:
                nasdaq = self._get_backup_nasdaq()
            
            self.cache['last_update'] = datetime.now()
            universe = list(set(sp500 + nasdaq))
            excluded = config.get('selection.excluded_symbols', [])
            
            logging.info(f"🌌 투자 유니버스: {len(universe)}개 종목")
            return [s for s in universe if s not in excluded]
            
        except Exception as e:
            logging.error(f"유니버스 생성 실패: {e}")
            return self._get_backup_sp500() + self._get_backup_nasdaq()
    
    def _is_cache_valid(self) -> bool:
        if not self.cache['last_update']:
            return False
        hours = config.get('selection.refresh_hours', 24)
        return (datetime.now() - self.cache['last_update']).seconds < hours * 3600
    
    def _get_backup_sp500(self) -> List[str]:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'ABBV', 'BAC']
    
    def _get_backup_nasdaq(self) -> List[str]:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN']
        async def get_stock_data(self, symbol: str) -> Dict:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                hist = stock.history(period="1y")
            
                if hist.empty:
                    return {}
            
                    current_price = float(hist['Close'].iloc[-1])
            
            # 기본 데이터
            data = {
                'symbol': symbol,
                'price': current_price,
                'market_cap': info.get('marketCap', 0) or 0,
                'avg_volume': info.get('averageVolume', 0) or 0,
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'pbr': info.get('priceToBook', 0) or 0,
                'roe': (info.get('returnOnEquity', 0) or 0) * 100,
                'eps_growth': (info.get('earningsQuarterlyGrowth', 0) or 0) * 100,
                'revenue_growth': (info.get('revenueQuarterlyGrowth', 0) or 0) * 100,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'sector': info.get('sector', 'Unknown'),
                'beta': info.get('beta', 1.0) or 1.0,
                'profit_margins': (info.get('profitMargins', 0) or 0) * 100
            }
            
            # PEG 계산
            if data['pe_ratio'] > 0 and data['eps_growth'] > 0:
                data['peg'] = data['pe_ratio'] / data['eps_growth']
            else:
                data['peg'] = 999
            
            # 모멘텀 지표
            if len(hist) >= 252:
                data['momentum_3m'] = ((current_price / float(hist['Close'].iloc[-63])) - 1) * 100
                data['momentum_6m'] = ((current_price / float(hist['Close'].iloc[-126])) - 1) * 100
                data['momentum_12m'] = ((current_price / float(hist['Close'].iloc[-252])) - 1) * 100
            else:
                data['momentum_3m'] = data['momentum_6m'] = data['momentum_12m'] = 0
            
            # 기술적 지표
            if len(hist) >= 50:
                # RSI
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                data['rsi'] = float(100 - (100 / (1 + rs)).iloc[-1])
                
                # 추세
                ma20 = float(hist['Close'].rolling(20).mean().iloc[-1])
                ma50 = float(hist['Close'].rolling(50).mean().iloc[-1])
                
                if current_price > ma50 > ma20:
                    data['trend'] = 'strong_uptrend'
                elif current_price > ma50:
                    data['trend'] = 'uptrend'
                else:
                    data['trend'] = 'downtrend'
                
                # 거래량
                avg_vol = float(hist['Volume'].rolling(20).mean().iloc[-1])
                current_vol = float(hist['Volume'].iloc[-1])
                data['volume_spike'] = current_vol / avg_vol if avg_vol > 0 else 1
                
                # 변동성
                returns = hist['Close'].pct_change().dropna()
                data['volatility'] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
            else:
                data.update({'rsi': 50, 'trend': 'sideways', 'volume_spike': 1, 'volatility': 25})
            
            await asyncio.sleep(0.3)
            return data
            
        except Exception as e:
            logging.error(f"데이터 수집 실패 {symbol}: {e}")
            return {}

# ========================================================================================
# 🧠 4가지 투자전략 분석 엔진
# ========================================================================================

class StrategyAnalyzer:
    def calculate_scores(self, data: Dict, vix: float) -> Tuple[float, Dict]:
        # 버핏 가치투자 점수
        buffett = 0.0
        pbr = data.get('pbr', 999)
        if 0 < pbr <= 1.0: buffett += 0.30
        elif pbr <= 1.5: buffett += 0.25
        elif pbr <= 2.0: buffett += 0.20
        elif pbr <= 3.0: buffett += 0.10
        
        roe = data.get('roe', 0)
        if roe >= 20: buffett += 0.25
        elif roe >= 15: buffett += 0.20
        elif roe >= 10: buffett += 0.15
        elif roe >= 5: buffett += 0.10
        
        debt_ratio = data.get('debt_to_equity', 999) / 100
        if debt_ratio <= 0.3: buffett += 0.20
        elif debt_ratio <= 0.5: buffett += 0.15
        elif debt_ratio <= 0.7: buffett += 0.10
        
        pe = data.get('pe_ratio', 999)
        if 5 <= pe <= 15: buffett += 0.15
        elif pe <= 20: buffett += 0.10
        elif pe <= 25: buffett += 0.05
        
        margins = data.get('profit_margins', 0)
        if margins >= 15: buffett += 0.10
        elif margins >= 10: buffett += 0.07
        elif margins >= 5: buffett += 0.05
        
        buffett = min(buffett, 1.0)
        
        # 린치 성장투자 점수
        lynch = 0.0
        peg = data.get('peg', 999)
        if 0 < peg <= 0.5: lynch += 0.40
        elif peg <= 1.0: lynch += 0.35
        elif peg <= 1.5: lynch += 0.25
        elif peg <= 2.0: lynch += 0.15
        
        eps_growth = data.get('eps_growth', 0)
        if eps_growth >= 25: lynch += 0.30
        elif eps_growth >= 20: lynch += 0.25
        elif eps_growth >= 15: lynch += 0.20
        elif eps_growth >= 10: lynch += 0.15
        
        rev_growth = data.get('revenue_growth', 0)
        if rev_growth >= 20: lynch += 0.20
        elif rev_growth >= 15: lynch += 0.15
        elif rev_growth >= 10: lynch += 0.10
        
        if roe >= 15: lynch += 0.10
        elif roe >= 10: lynch += 0.07
        
        lynch = min(lynch, 1.0)
        
        # 모멘텀 전략 점수
        momentum = 0.0
        mom_3m = data.get('momentum_3m', 0)
        if mom_3m >= 20: momentum += 0.30
        elif mom_3m >= 15: momentum += 0.25
        elif mom_3m >= 10: momentum += 0.20
        elif mom_3m >= 5: momentum += 0.15
        
        mom_6m = data.get('momentum_6m', 0)
        if mom_6m >= 30: momentum += 0.25
        elif mom_6m >= 20: momentum += 0.20
        elif mom_6m >= 15: momentum += 0.15
        elif mom_6m >= 10: momentum += 0.10
        
        mom_12m = data.get('momentum_12m', 0)
        if mom_12m >= 50: momentum += 0.25
        elif mom_12m >= 30: momentum += 0.20
        elif mom_12m >= 20: momentum += 0.15
        elif mom_12m >= 10: momentum += 0.10
        
        vol_spike = data.get('volume_spike', 1)
        if vol_spike >= 3.0: momentum += 0.20
        elif vol_spike >= 2.0: momentum += 0.15
        elif vol_spike >= 1.5: momentum += 0.10
        
        momentum = min(momentum, 1.0)
        
        # 기술적 분석 점수
        technical = 0.0
        rsi = data.get('rsi', 50)
        if 30 <= rsi <= 70: technical += 0.30
        elif 25 <= rsi < 30: technical += 0.25
        elif 70 < rsi <= 75: technical += 0.20
        
        trend = data.get('trend', 'sideways')
        if trend == 'strong_uptrend': technical += 0.35
        elif trend == 'uptrend': technical += 0.25
        elif trend == 'sideways': technical += 0.10
        
        volatility = data.get('volatility', 25)
        if 15 <= volatility <= 30: technical += 0.20
        elif 10 <= volatility <= 40: technical += 0.15
        elif volatility <= 50: technical += 0.10
        
        if vol_spike >= 1.5: technical += 0.15
        elif vol_spike >= 1.2: technical += 0.10
        
        technical = min(technical, 1.0)
        
        # 가중치 적용
        weights = config.get('strategy.weights', {})
        total = (
            buffett * weights.get('buffett', 25) +
            lynch * weights.get('lynch', 25) +
            momentum * weights.get('momentum', 25) +
            technical * weights.get('technical', 25)
        ) / 100
        
        # VIX 조정
        low_vix = config.get('strategy.vix_thresholds.low', 15.0)
        high_vix = config.get('strategy.vix_thresholds.high', 30.0)
        
        if vix <= low_vix:
            adjusted = total * 1.15
        elif vix >= high_vix:
            adjusted = total * 0.85
        else:
            adjusted = total
        
        scores = {
            'buffett': buffett,
            'lynch': lynch,
            'momentum': momentum,
            'technical': technical,
            'total': adjusted,
            'vix_adjustment': adjusted - total
        }
        
        return adjusted, scores
# ========================================================================================
# 🏦 IBKR 연동 시스템
# ========================================================================================

class IBKRTrader:
    def __init__(self):
        self.ib = None
        self.connected = False
        self.positions = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
    async def connect(self) -> bool:
        try:
            if not IBKR_AVAILABLE:
                logging.error("❌ IBKR 모듈 없음")
                return False
            
            host = config.get('ibkr.host', '127.0.0.1')
            port = config.get('ibkr.port', 7497)
            client_id = config.get('ibkr.client_id', 1)
            
            self.ib = IB()
            await self.ib.connectAsync(host, port, clientId=client_id)
            
            if self.ib.isConnected():
                self.connected = True
                mode = '모의투자' if config.get('ibkr.paper_trading') else '실거래'
                logging.info(f"✅ IBKR 연결 - {mode}")
                await self._update_account()
                return True
            else:
                logging.error("❌ IBKR 연결 실패")
                return False
                
        except Exception as e:
            logging.error(f"❌ IBKR 연결 오류: {e}")
            return False
    
    async def disconnect(self):
        try:
            if self.ib and self.connected:
                self.ib.disconnect()
                self.connected = False
                logging.info("🔌 IBKR 연결 해제")
        except Exception as e:
            logging.error(f"연결 해제 오류: {e}")
    
    async def _update_account(self):
        try:
            account_values = self.ib.accountValues()
            portfolio = self.ib.portfolio()
            
            for av in account_values:
                if av.tag == 'DayPNL':
                    self.daily_pnl = float(av.value)
                    break
            
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
            if not self.connected or not self._safety_check():
                return None
            
            contract = Stock(symbol, 'SMART', 'USD')
            order = MarketOrder('BUY', quantity)
            trade = self.ib.placeOrder(contract, order)
            
            logging.info(f"📈 매수 주문: {symbol} {quantity}주")
            self.daily_trades += 1
            return str(trade.order.orderId)
            
        except Exception as e:
            logging.error(f"❌ 매수 실패 {symbol}: {e}")
            return None
    
    async def place_sell_order(self, symbol: str, quantity: int, reason: str = '') -> Optional[str]:
        try:
            if not self.connected:
                return None
            
            if symbol not in self.positions:
                logging.warning(f"⚠️ {symbol} 포지션 없음")
                return None
            
            current_qty = abs(self.positions[symbol]['quantity'])
            if quantity > current_qty:
                quantity = current_qty
            
            contract = Stock(symbol, 'SMART', 'USD')
            order = MarketOrder('SELL', quantity)
            trade = self.ib.placeOrder(contract, order)
            
            logging.info(f"📉 매도 주문: {symbol} {quantity}주 - {reason}")
            self.daily_trades += 1
            return str(trade.order.orderId)
            
        except Exception as e:
            logging.error(f"❌ 매도 실패 {symbol}: {e}")
            return None
    
    def _safety_check(self) -> bool:
        max_trades = config.get('ibkr.max_daily_trades', 20)
        max_loss = config.get('risk.daily_loss_limit', 1.0) * 10000
        
        if self.daily_trades >= max_trades:
            logging.warning(f"⚠️ 일일 거래 한도 초과: {self.daily_trades}")
            return False
        
        if self.daily_pnl < -max_loss:
            logging.warning(f"⚠️ 일일 손실 한도 초과: ${self.daily_pnl:.2f}")
            return False
        
        return True
    
    async def get_current_price(self, symbol: str) -> float:
        try:
            if not self.connected:
                return 0.0
            
            contract = Stock(symbol, 'SMART', 'USD')
            ticker = self.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(0.5)
            
            price = ticker.marketPrice() or ticker.last or 0.0
            self.ib.cancelMktData(contract)
            return float(price)
            
        except Exception as e:
            logging.error(f"현재가 조회 실패 {symbol}: {e}")
            return 0.0
    
    async def get_portfolio_value(self) -> float:
        try:
            await self._update_account()
            return sum(pos['market_price'] * abs(pos['quantity']) for pos in self.positions.values())
        except:
            return 1000000

# ========================================================================================
# 🤖 자동 손익절 관리자
# ========================================================================================

class StopTakeManager:
    def __init__(self, ibkr_trader: IBKRTrader):
        self.ibkr = ibkr_trader
        self.positions: Dict[str, Position] = {}
        self.monitoring = False
        self.db_path = config.get('performance.database_file', 'legendary_performance.db')
        self._init_database()
    
    def _init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    profit_loss REAL DEFAULT 0.0,
                    profit_percent REAL DEFAULT 0.0,
                    mode TEXT NOT NULL,
                    entry_day TEXT DEFAULT ''
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    entry_date DATETIME NOT NULL,
                    mode TEXT NOT NULL,
                    stage INTEGER DEFAULT 1,
                    tp_executed TEXT DEFAULT '[]',
                    highest_price REAL DEFAULT 0.0,
                    entry_day TEXT DEFAULT ''
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"DB 초기화 실패: {e}")
    
    def add_position(self, symbol: str, quantity: int, avg_cost: float, mode: str, entry_day: str = ''):
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_cost=avg_cost,
            entry_date=datetime.now(),
            mode=mode,
            highest_price=avg_cost,
            entry_day=entry_day
        )
        
        self.positions[symbol] = position
        self._save_position_to_db(position)
        logging.info(f"➕ 포지션 추가: {symbol} {quantity}주 @${avg_cost:.2f} ({mode}) [{entry_day}]")
    
    def _save_position_to_db(self, position: Position):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            tp_json = json.dumps(position.tp_executed)
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                (symbol, quantity, avg_cost, entry_date, mode, stage, tp_executed, highest_price, entry_day)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol, position.quantity, position.avg_cost,
                position.entry_date.isoformat(), position.mode, position.stage,
                tp_json, position.highest_price, position.entry_day
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"포지션 저장 실패: {e}")
    
    async def start_monitoring(self):
        self.monitoring = True
        logging.info("🔍 자동 손익절 모니터링 시작!")
        
        while self.monitoring:
            try:
                await self._monitor_all_positions()
                await asyncio.sleep(15)
            except Exception as e:
                logging.error(f"모니터링 오류: {e}")
                await asyncio.sleep(30)
    
    def stop_monitoring(self):
        self.monitoring = False
        logging.info("⏹️ 모니터링 중지") 
async def _monitor_all_positions(self):
        for symbol, position in list(self.positions.items()):
            try:
                current_price = await self.ibkr.get_current_price(symbol)
                if current_price <= 0:
                    continue
                
                if current_price > position.highest_price:
                    position.highest_price = current_price
                
                profit_pct = position.profit_percent(current_price)
                
                # 익절 체크
                await self._check_take_profit(symbol, position, current_price, profit_pct)
                
                # 손절 체크
                await self._check_stop_loss(symbol, position, current_price, profit_pct)
                
            except Exception as e:
                logging.error(f"포지션 모니터링 실패 {symbol}: {e}")
    
    async def _check_take_profit(self, symbol: str, position: Position, current_price: float, profit_pct: float):
        if position.mode == 'swing':
            tp_levels = config.get('trading.swing.take_profit', [6.0, 12.0])
            ratios = config.get('trading.swing.profit_ratios', [60.0, 40.0])
            
            # 2차 익절 (12%)
            if profit_pct >= tp_levels[1] and not position.tp_executed[1]:
                sell_qty = int(position.quantity * ratios[1] / 100)
                if sell_qty > 0:
                    order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'SWING_TP2')
                    if order_id:
                        position.tp_executed[1] = True
                        position.quantity -= sell_qty
                        await self._record_trade(symbol, 'SELL_SWING_TP2', sell_qty, current_price, profit_pct, position.mode, position.entry_day)
                        await self._send_notification(f"🎉 {symbol} 스윙 2차 익절! +{profit_pct:.1f}% [{position.entry_day}]")
                        
                        if position.quantity <= 0:
                            del self.positions[symbol]
                            await self._remove_position_from_db(symbol)
            
            # 1차 익절 (6%)
            elif profit_pct >= tp_levels[0] and not position.tp_executed[0]:
                sell_qty = int(position.quantity * ratios[0] / 100)
                if sell_qty > 0:
                    order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'SWING_TP1')
                    if order_id:
                        position.tp_executed[0] = True
                        position.quantity -= sell_qty
                        await self._record_trade(symbol, 'SELL_SWING_TP1', sell_qty, current_price, profit_pct, position.mode, position.entry_day)
                        await self._send_notification(f"✅ {symbol} 스윙 1차 익절! +{profit_pct:.1f}% [{position.entry_day}]")
        
        else:  # classic
            tp_levels = config.get('trading.classic.take_profit', [20.0, 35.0])
            
            # 2차 익절 (35%)
            if profit_pct >= tp_levels[1] and not position.tp_executed[1]:
                sell_qty = int(position.quantity * 0.4)
                if sell_qty > 0:
                    order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'CLASSIC_TP2')
                    if order_id:
                        position.tp_executed[1] = True
                        position.quantity -= sell_qty
                        await self._record_trade(symbol, 'SELL_CLASSIC_TP2', sell_qty, current_price, profit_pct, position.mode, position.entry_day)
                        await self._send_notification(f"💰 {symbol} 클래식 2차 익절! +{profit_pct:.1f}%")
            
            # 1차 익절 (20%)
            elif profit_pct >= tp_levels[0] and not position.tp_executed[0]:
                sell_qty = int(position.quantity * 0.6)
                if sell_qty > 0:
                    order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'CLASSIC_TP1')
                    if order_id:
                        position.tp_executed[0] = True
                        position.quantity -= sell_qty
                        await self._record_trade(symbol, 'SELL_CLASSIC_TP1', sell_qty, current_price, profit_pct, position.mode, position.entry_day)
                        await self._send_notification(f"✅ {symbol} 클래식 1차 익절! +{profit_pct:.1f}%")
    
    async def _check_stop_loss(self, symbol: str, position: Position, current_price: float, profit_pct: float):
        stop_pct = config.get(f'trading.{position.mode}.stop_loss', 8.0)
        
        # 고정 손절
        if profit_pct <= -stop_pct:
            order_id = await self.ibkr.place_sell_order(symbol, position.quantity, 'STOP_LOSS')
            if order_id:
                await self._record_trade(symbol, 'SELL_STOP', position.quantity, current_price, profit_pct, position.mode, position.entry_day)
                await self._send_notification(f"🛑 {symbol} {position.mode} 손절! {profit_pct:.1f}% [{position.entry_day}]")
                del self.positions[symbol]
                await self._remove_position_from_db(symbol)
        
        # 트레일링 스톱
        elif (config.get('risk.trailing_stop', True) and position.highest_price > position.avg_cost * 1.1):
            trailing_stop = position.highest_price * 0.95  # 5% 트레일링
            if current_price <= trailing_stop:
                order_id = await self.ibkr.place_sell_order(symbol, position.quantity, 'TRAILING_STOP')
                if order_id:
                    await self._record_trade(symbol, 'SELL_TRAILING', position.quantity, current_price, profit_pct, position.mode, position.entry_day)
                    await self._send_notification(f"📉 {symbol} 트레일링 스톱! {profit_pct:.1f}% [{position.entry_day}]")
                    del self.positions[symbol]
                    await self._remove_position_from_db(symbol)
    
    async def _record_trade(self, symbol: str, action: str, quantity: int, price: float, profit_pct: float, mode: str, entry_day: str = ''):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            profit_loss = 0.0
            if 'SELL' in action and symbol in self.positions:
                position = self.positions[symbol]
                profit_loss = (price - position.avg_cost) * quantity
            
            cursor.execute('''
                INSERT INTO trades 
                (symbol, action, quantity, price, timestamp, profit_loss, profit_percent, mode, entry_day)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, action, quantity, price, datetime.now().isoformat(), profit_loss, profit_pct, mode, entry_day))
            
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
        except Exception as e:
            logging.error(f"포지션 제거 실패: {e}")
    
    async def _send_notification(self, message: str):
        try:
            # 텔레그램 알림
            if config.get('notifications.telegram.enabled', False):
                await self._send_telegram(message)
            logging.info(f"📢 {message}")
        except Exception as e:
            logging.error(f"알림 전송 실패: {e}")
    
    async def _send_telegram(self, message: str):
        try:
            token = config.get('notifications.telegram.bot_token', '')
            chat_id = config.get('notifications.telegram.chat_id', '')
            
            if not token or not chat_id:
                return
            
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {'chat_id': chat_id, 'text': f"🏆 전설적퀸트\n{message}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logging.debug("텔레그램 알림 전송 완료")
        except Exception as e:
            logging.error(f"텔레그램 전송 오류: {e}")

# ========================================================================================
# 🏆 메인 전략 시스템 (주 2회 화목 매매 통합)
# ========================================================================================

class LegendaryQuantStrategy:
    def __init__(self):
        self.enabled = config.get('strategy.enabled', True)
        self.current_mode = config.get('strategy.mode', 'swing')
        self.weekly_mode = config.get('trading.weekly.enabled', True)
        
        # 핵심 컴포넌트들
        self.selector = StockSelector()
        self.analyzer = StrategyAnalyzer()
        self.ibkr = IBKRTrader()
        self.stop_take = StopTakeManager(self.ibkr)
        
        # 캐싱 및 성과 추적
        self.selected_stocks = []
        self.last_selection = None
        self.monthly_return = 0.0
        self.target_min = config.get('strategy.monthly_target.min', 5.0)
        self.target_max = config.get('strategy.monthly_target.max', 7.0)
        self.last_trade_dates = {'Tuesday': None, 'Thursday': None}
        
        if self.enabled:
            logging.info("🏆 전설적 퀸트 전략 시스템 가동!")
            logging.info(f"🎯 현재 모드: {self.current_mode.upper()}")
            if self.weekly_mode:
                logging.info("📅 주 2회 화목 매매 모드 활성화")
    
    async def auto_select_stocks(self) -> List[str]:
        if not self.enabled:
            return []
        
        try:
            # 캐시 확인
            if self.last_selection and (datetime.now() - self.last_selection).seconds < 24 * 3600:
                logging.info("📋 캐시된 선별 결과 사용")
                return [s['symbol'] for s in self.selected_stocks]
            
            logging.info("🚀 종목 자동선별 시작!")
            start_time = time.time()
            
            # 1. 투자 유니버스 생성
            universe = await self.selector.collect_symbols()
            if not universe:
                return self._get_fallback_stocks()
            
            # 2. VIX 조회
            current_vix = await self.selector.get_current_vix()
            
            # 3. 병렬 분석
            scored_stocks = []
            batch_size = 20
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i + batch_size]
                tasks = [self._analyze_stock_async(symbol, current_vix) for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result:
                        scored_stocks.append(result)
                
                if i % 100 == 0:
                    logging.info(f"📊 분석 진행: {i}/{len(universe)}")
            
            if not scored_stocks:
                return self._get_fallback_stocks()
            
            # 4. 상위 종목 선별 (다양성 고려)
            target_count = config.get(f'strategy.target_stocks.{self.current_mode}', 8)
            final_selection = self._select_diversified_stocks(scored_stocks, target_count)
            
            # 5. 결과 저장
            self.selected_stocks = final_selection
            self.last_selection = datetime.now()
            
            selected_symbols = [s['symbol'] for s in final_selection]
            elapsed = time.time() - start_time
            
            logging.info(f"🏆 선별 완료! {len(selected_symbols)}개 종목 ({elapsed:.1f}초)")
            return selected_symbols
            
        except Exception as e:
            logging.error(f"자동선별 실패: {e}")
            return self._get_fallback_stocks()
    
    async def _analyze_stock_async(self, symbol: str, vix: float) -> Optional[Dict]:
        try:
            data = await self.selector.get_stock_data(symbol)
            if not data:
                return None
            
            # 기본 필터링
            min_cap = config.get('selection.min_market_cap', 5_000_000_000)
            min_vol = config.get('selection.min_volume', 1_000_000)
            
            if data.get('market_cap', 0) < min_cap or data.get('avg_volume', 0) < min_vol:
                return None
            
            # 통합 점수 계산
            total_score, scores = self.analyzer.calculate_scores(data, vix)
            
            # 모드별 필터링
            if not self._mode_filter(data, total_score):
                return None
            
            result = data.copy()
            result.update(scores)
            result['symbol'] = symbol
            result['vix'] = vix
            result['mode'] = self.current_mode
            
            return result
            
        except Exception as e:
            logging.error(f"종목 분석 실패 {symbol}: {e}")
            return None
    
    def _mode_filter(self, data: Dict, score: float) -> bool:
        try:
            if self.current_mode == 'classic':
                return (score >= 0.60 and data.get('volatility', 50) <= 40 and data.get('beta', 2.0) <= 1.8)
            elif self.current_mode == 'swing':
                return (score >= 0.65 and 15 <= data.get('volatility', 25) <= 35 and 0.8 <= data.get('beta', 1.0) <= 1.5)
            else:  # hybrid
                return score >= 0.62
        except:
            return True
    
    def _select_diversified_stocks(self, scored_stocks: List[Dict], target_count: int) -> List[Dict]:
        scored_stocks.sort(key=lambda x: x['total'], reverse=True)
        
        final_selection = []
        sector_counts = {}
        max_per_sector = 2 if self.current_mode == 'swing' else 4
        
        for stock in scored_stocks:
            if len(final_selection) >= target_count:
                break
            
            sector = stock.get('sector', 'Unknown')
            if sector_counts.get(sector, 0) < max_per_sector:
                final_selection.append(stock)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # 부족하면 상위 점수로 채움
        remaining = target_count - len(final_selection)
        if remaining > 0:
            for stock in scored_stocks:
                if remaining <= 0:
                    break
                if stock not in final_selection:
                    final_selection.append(stock)
                    remaining -= 1
        
        return final_selection
    
    def _get_fallback_stocks(self) -> List[str]:
        if self.current_mode == 'swing':
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX']
        else:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                    'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'ABBV', 'BAC']
    
    async def analyze_stock_signal(self, symbol: str) -> StockSignal:
        try:
            data = await self.selector.get_stock_data(symbol)
            if not data:
                raise ValueError(f"데이터 수집 실패: {symbol}")
            
            vix = await self.selector.get_current_vix()
            total_score, scores = self.analyzer.calculate_scores(data, vix)
            
            # 액션 결정
            confidence_threshold = 0.70 if self.current_mode == 'classic' else 0.65
            
            if total_score >= confidence_threshold:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= 0.30:
                action = 'sell'
                confidence = min(1 - total_score, 0.95)
            else:
                action = 'hold'
                confidence = 0.50
            
            # 목표가 및 손절가 계산
            max_return = 0.25 if self.current_mode == 'swing' else 0.35
            target_price = data['price'] * (1 + confidence * max_return)
            
            stop_loss_pct = config.get(f'trading.{self.current_mode}.stop_loss', 8.0)
            stop_loss = data['price'] * (1 - stop_loss_pct / 100)
            
            reasoning = (f"버핏:{scores['buffett']:.2f} 린치:{scores['lynch']:.2f} "
                        f"모멘텀:{scores['momentum']:.2f} 기술:{scores['technical']:.2f} "
                        f"VIX:{scores['vix_adjustment']:+.2f}")
            
            return StockSignal(
                symbol=symbol, action=action, confidence=confidence, price=data['price'],
                mode=self.current_mode, scores=scores, target_price=target_price,
                stop_loss=stop_loss, reasoning=reasoning, timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"시그널 분석 실패 {symbol}: {e}")
            return StockSignal(
                symbol=symbol, action='hold', confidence=0.0, price=0.0,
                mode=self.current_mode, scores={}, target_price=0.0,
                stop_loss=0.0, reasoning=f"오류: {e}", timestamp=datetime.now()
            )

        async def scan_all_stocks(self) -> List[StockSignal]:
        if not self.enabled:
            return []
        
        logging.info("🔍 전체 종목 스캔 시작!")
        
        try:
            selected = await self.auto_select_stocks()
            if not selected:
                return []
            
            signals = []
            for symbol in selected:
                try:
                    signal = await self.analyze_stock_signal(symbol)
                    signals.append(signal)
                    
                    emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                    logging.info(f"{emoji} {symbol}: {signal.action} 신뢰도:{signal.confidence:.2f}")
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logging.error(f"❌ {symbol} 분석 실패: {e}")
            
            buy_count = len([s for s in signals if s.action == 'buy'])
            sell_count = len([s for s in signals if s.action == 'sell'])
            hold_count = len([s for s in signals if s.action == 'hold'])
            
            logging.info(f"🏆 스캔 완료! 매수:{buy_count}, 매도:{sell_count}, 보유:{hold_count}")
            return signals
            
        except Exception as e:
            logging.error(f"전체 스캔 실패: {e}")
            return []
    
    # ========================================================================================
    # 🆕 주 2회 화목 매매 시스템
    # ========================================================================================
    
    async def initialize_trading(self) -> bool:
        try:
            logging.info("🚀 거래 시스템 초기화...")
            
            if not await self.ibkr.connect():
                logging.error("❌ IBKR 연결 실패")
                return False
            
            # 기존 포지션 로드
            await self._load_existing_positions()
            logging.info("✅ 거래 시스템 초기화 완료!")
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
                tp_executed = json.loads(row[6]) if row[6] else [False, False, False]
                entry_day = row[8] if len(row) > 8 else ''
                
                position = Position(
                    symbol=row[0], quantity=row[1], avg_cost=row[2],
                    entry_date=datetime.fromisoformat(row[3]), mode=row[4],
                    stage=row[5], tp_executed=tp_executed, highest_price=row[7], entry_day=entry_day
                )
                
                self.stop_take.positions[position.symbol] = position
            
            conn.close()
            logging.info(f"📂 기존 포지션 로드: {len(self.stop_take.positions)}개")
            
        except Exception as e:
            logging.error(f"포지션 로드 실패: {e}")
    
    async def start_auto_trading(self):
        try:
            mode_text = "주 2회 화목 매매" if self.weekly_mode else "일반 모드"
            logging.info(f"🎯 자동거래 시작! ({mode_text})")
            
            # 손익절 모니터링 시작
            monitor_task = asyncio.create_task(self.stop_take.start_monitoring())
            schedule_task = asyncio.create_task(self._run_schedule())
            
            await asyncio.gather(monitor_task, schedule_task)
            
        except Exception as e:
            logging.error(f"자동거래 실행 실패: {e}")
        finally:
            await self.shutdown()
    
    async def _run_schedule(self):
        logging.info("📅 주 2회 화목 매매 스케줄러 시작!")
        
        while True:
            try:
                now = datetime.now()
                weekday = now.weekday()
                current_time = now.time()
                
                if self.weekly_mode:
                    # 화요일 매매 (10:30)
                    if (weekday == 1 and current_time.hour == 10 and current_time.minute == 30 and
                        self.last_trade_dates['Tuesday'] != now.date() and self._is_trading_day()):
                        await self._execute_tuesday_trading()
                        self.last_trade_dates['Tuesday'] = now.date()
                        
                    # 목요일 매매 (10:30)
                    elif (weekday == 3 and current_time.hour == 10 and current_time.minute == 30 and
                          self.last_trade_dates['Thursday'] != now.date() and self._is_trading_day()):
                        await self._execute_thursday_trading()
                        self.last_trade_dates['Thursday'] = now.date()
                
                # 일일 체크들
                if current_time.hour == 9 and current_time.minute == 0:
                    await self._perform_daily_check()
                
                if current_time.hour == 16 and current_time.minute == 0:
                    await self._generate_report()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logging.error(f"스케줄 오류: {e}")
                await asyncio.sleep(60)
    
    async def _execute_tuesday_trading(self):
        try:
            logging.info("🔥 화요일 공격적 진입 시작!")
            
            # 시장 상황 분석
            market_condition = await self._analyze_market_condition()
            if not market_condition['safe_to_trade']:
                await self.stop_take._send_notification(f"⚠️ 화요일 매매 스킵 - {market_condition['reason']}")
                return
            
            # 타겟 종목수 (시장 상황에 따라 조정)
            base_targets = config.get('trading.weekly.tuesday_targets', 4)
            aggressiveness = market_condition.get('aggressiveness', 1.0)
            adjusted_targets = max(2, min(6, int(base_targets * aggressiveness)))
            
            # 종목 선별
            selected = await self.auto_select_stocks()
            if not selected:
                return
            
            # 기존 포지션 제외
            existing_symbols = list(self.stop_take.positions.keys())
            new_candidates = [s for s in selected if s not in existing_symbols][:adjusted_targets]
            
            # 포지션 진입
            portfolio_value = await self.ibkr.get_portfolio_value()
            base_allocation = config.get('trading.weekly.tuesday_allocation', 12.5) / 100
            adjusted_allocation = base_allocation * aggressiveness
            
            new_entries = 0
            for symbol in new_candidates:
                try:
                    investment_amount = portfolio_value * adjusted_allocation
                    success = await self._enter_position_safely(symbol, investment_amount, 'swing', 'Tuesday')
                    if success:
                        new_entries += 1
                    await asyncio.sleep(2)
                except Exception as e:
                    logging.error(f"화요일 {symbol} 진입 실패: {e}")
            
            # 결과 알림
            await self.stop_take._send_notification(
                f"🔥 화요일 공격적 진입 완료!\n"
                f"📊 시장상황: {market_condition['status']} (VIX: {market_condition.get('vix', 0):.1f})\n"
                f"💰 신규진입: {new_entries}/{len(new_candidates)}개"
            )
            
        except Exception as e:
            logging.error(f"화요일 매매 실패: {e}")
    
    async def _execute_thursday_trading(self):
        try:
            logging.info("📋 목요일 포지션 정리 시작!")
            
            # 시장 상황 및 주간 성과 분석
            market_condition = await self._analyze_market_condition()
            weekly_performance = await self._analyze_weekly_performance()
            
            # 기존 포지션 리뷰
            actions_taken = await self._thursday_position_review(weekly_performance)
            
            # 선별적 신규 진입
            if (weekly_performance['weekly_return'] >= 0 and market_condition['safe_to_trade'] and
                market_condition.get('aggressiveness', 1.0) > 0.8):
                new_entries = await self._thursday_selective_entry()
                actions_taken['new_entries'] = new_entries
            
            # 결과 알림
            await self.stop_take._send_notification(
                f"📋 목요일 포지션 정리 완료!\n"
                f"💰 이익실현: {actions_taken.get('profit_taken', 0)}개\n"
                f"🛑 손절청산: {actions_taken.get('stop_losses', 0)}개\n"
                f"📊 신규진입: {actions_taken.get('new_entries', 0)}개\n"
                f"📈 주간수익률: {weekly_performance['weekly_return']:+.2f}%"
            )
            
        except Exception as e:
            logging.error(f"목요일 매매 실패: {e}")
    
    async def _analyze_market_condition(self) -> Dict:
        try:
            # VIX 조회
            vix = await self.selector.get_current_vix()
            
            # SPY 모멘텀 확인
            spy_data = await self.selector.get_stock_data('SPY')
            spy_momentum = spy_data.get('momentum_3m', 0) if spy_data else 0
            
            # QQQ 모멘텀 확인
            qqq_data = await self.selector.get_stock_data('QQQ')
            qqq_momentum = qqq_data.get('momentum_3m', 0) if qqq_data else 0
            
            # 시장 상황 판단
            condition = {
                'vix': vix,
                'spy_momentum': spy_momentum,
                'qqq_momentum': qqq_momentum,
                'market_momentum': (spy_momentum + qqq_momentum) / 2,
                'safe_to_trade': True,
                'status': 'normal',
                'reason': '',
                'aggressiveness': 1.0
            }
            
            # VIX 기반 판단
            if vix > 35:
                condition.update({
                    'safe_to_trade': False,
                    'status': 'high_volatility',
                    'reason': f'VIX 과도하게 높음: {vix:.1f}'
                })
            elif vix > 25:
                condition.update({'status': 'volatile', 'aggressiveness': 0.7})
            elif vix < 15:
                condition.update({'status': 'low_volatility', 'aggressiveness': 1.3})
            
            # 모멘텀 기반 추가 판단
            if condition['market_momentum'] < -10:
                condition['aggressiveness'] *= 0.6
                condition['status'] = 'bearish'
            elif condition['market_momentum'] > 15:
                condition['aggressiveness'] *= 1.2
                condition['status'] = 'bullish'
            
            logging.info(f"📊 시장상황: {condition['status']}, VIX: {vix:.1f}, "
                        f"SPY모멘텀: {spy_momentum:.1f}%, 공격성: {condition['aggressiveness']:.1f}")
            
            return condition
            
        except Exception as e:
            logging.error(f"시장 분석 실패: {e}")
            return {'safe_to_trade': False, 'status': 'error', 'reason': f'분석 실패: {e}', 'aggressiveness': 0.5} 
async def _thursday_position_review(self, weekly_performance: Dict) -> Dict:
        try:
            actions_taken = {'profit_taken': 0, 'stop_losses': 0, 'held_positions': 0}
            
            for symbol, position in list(self.stop_take.positions.items()):
                try:
                    current_price = await self.ibkr.get_current_price(symbol)
                    if current_price <= 0:
                        continue
                    
                    profit_pct = position.profit_percent(current_price)
                    hold_days = (datetime.now() - position.entry_date).days
                    
                    # 목요일 특별 룰
                    action = self._thursday_position_decision(symbol, position, profit_pct, hold_days, weekly_performance)
                    
                    if action == 'TAKE_PROFIT':
                        # 부분 이익실현 (50%)
                        sell_qty = int(position.quantity * 0.5)
                        if sell_qty > 0:
                            order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'Thursday-Profit-Taking')
                            if order_id:
                                actions_taken['profit_taken'] += 1
                                position.quantity -= sell_qty
                                self.stop_take._save_position_to_db(position)
                    
                    elif action == 'FULL_EXIT':
                        # 전량 매도
                        order_id = await self.ibkr.place_sell_order(symbol, position.quantity, 'Thursday-Full-Exit')
                        if order_id:
                            actions_taken['stop_losses'] += 1
                            del self.stop_take.positions[symbol]
                            await self.stop_take._remove_position_from_db(symbol)
                    
                    else:  # HOLD
                        actions_taken['held_positions'] += 1
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logging.error(f"목요일 {symbol} 리뷰 실패: {e}")
            
            return actions_taken
            
        except Exception as e:
            logging.error(f"목요일 포지션 리뷰 실패: {e}")
            return {'profit_taken': 0, 'stop_losses': 0, 'held_positions': 0}
    
    def _thursday_position_decision(self, symbol: str, position, profit_pct: float, hold_days: int, weekly_performance: Dict) -> str:
        try:
            profit_threshold = config.get('trading.weekly.profit_taking_threshold', 8.0)
            loss_threshold = config.get('trading.weekly.loss_cutting_threshold', -6.0)
            
            # 1. 큰 수익 -> 부분 이익실현
            if profit_pct >= profit_threshold:
                return 'TAKE_PROFIT'
            
            # 2. 손실이 크거나 주간 성과가 나쁜 경우 -> 전량 매도
            if profit_pct <= loss_threshold or weekly_performance['weekly_return'] < -3.0:
                return 'FULL_EXIT'
            
            # 3. 보유 기간이 길고 수익이 미미한 경우
            if hold_days >= 7 and -2.0 <= profit_pct <= 2.0:
                return 'FULL_EXIT'
            
            # 4. 나머지는 보유
            return 'HOLD'
            
        except Exception as e:
            logging.error(f"목요일 결정 로직 오류 {symbol}: {e}")
            return 'HOLD'
    
    async def _thursday_selective_entry(self) -> int:
        try:
            max_new_entries = config.get('trading.weekly.thursday_targets', 2)
            
            # 현재 포지션 수 확인
            current_positions = len(self.stop_take.positions)
            target_total = config.get(f'strategy.target_stocks.{self.current_mode}', 8)
            
            if current_positions >= target_total:
                return 0
            
            # 고품질 종목만 선별
            selected = await self.auto_select_stocks()
            if not selected:
                return 0
            
            # 기존 포지션 제외
            existing_symbols = list(self.stop_take.positions.keys())
            new_candidates = [s for s in selected if s not in existing_symbols][:max_new_entries]
            
            portfolio_value = await self.ibkr.get_portfolio_value()
            conservative_allocation = config.get('trading.weekly.thursday_allocation', 8.0) / 100
            
            new_entries = 0
            for symbol in new_candidates:
                try:
                    investment_amount = portfolio_value * conservative_allocation
                    success = await self._enter_position_safely(symbol, investment_amount, 'swing', 'Thursday')
                    if success:
                        new_entries += 1
                    await asyncio.sleep(2)
                except Exception as e:
                    logging.error(f"목요일 보수적 진입 {symbol} 실패: {e}")
            
            return new_entries
            
        except Exception as e:
            logging.error(f"목요일 선별적 진입 실패: {e}")
            return 0
    
    async def _enter_position_safely(self, symbol: str, investment: float
        async def _enter_position_safely(self, symbol: str, investment: float, mode: str, entry_day: str) -> bool:
        try:
            # 현재가 조회
            current_price = await self.ibkr.get_current_price(symbol)
            if current_price <= 0:
                return False
            
            # 수량 계산
            quantity = int(investment / current_price)
            if quantity < 1:
                return False
            
            # 진입 타이밍 체크
            if not await self._check_entry_timing(symbol, current_price):
                return False
            
            # 매수 주문
            order_id = await self.ibkr.place_buy_order(symbol, quantity)
            
            if order_id:
                # 포지션 추가
                self.stop_take.add_position(symbol, quantity, current_price, mode, entry_day)
                
                # 알림
                investment_value = quantity * current_price
                await self.stop_take._send_notification(
                    f"🚀 {symbol} 진입! ({entry_day})\n"
                    f"💰 ${investment_value:.0f} ({quantity}주 @${current_price:.2f})"
                )
                
                return True
            
            return False
                
        except Exception as e:
            logging.error(f"포지션 진입 실패 {symbol}: {e}")
            return False
    
    async def _check_entry_timing(self, symbol: str, current_price: float) -> bool:
        try:
            stock_data = await self.selector.get_stock_data(symbol)
            
            if not stock_data:
                return True
            
            # RSI 체크 (과매수 구간 회피)
            rsi = stock_data.get('rsi', 50)
            if rsi > 80:
                return False
            
            # 당일 변동성 체크
            volume_spike = stock_data.get('volume_spike', 1.0)
            if volume_spike > 5.0:
                return False
            
            return True
            
        except Exception as e:
            return True
    
    async def _analyze_weekly_performance(self) -> Dict:
        try:
            # 이번주 시작일 계산
            now = datetime.now()
            days_since_monday = now.weekday()
            week_start = now - timedelta(days=days_since_monday)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # DB에서 이번주 거래 조회
            conn = sqlite3.connect(self.stop_take.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT SUM(profit_loss), COUNT(*) FROM trades 
                WHERE timestamp >= ? AND action LIKE 'SELL%'
            ''', (week_start.isoformat(),))
            
            result = cursor.fetchone()
            weekly_profit = result[0] if result[0] else 0.0
            weekly_trades = result[1] if result[1] else 0
            
            # 포트폴리오 가치
            portfolio_value = await self.ibkr.get_portfolio_value()
            weekly_return = (weekly_profit / portfolio_value) * 100 if portfolio_value > 0 else 0.0
            
            conn.close()
            
            return {
                'weekly_profit': weekly_profit,
                'weekly_return': weekly_return,
                'weekly_trades': weekly_trades,
                'week_start': week_start,
                'portfolio_value': portfolio_value
            }
            
        except Exception as e:
            logging.error(f"주간 성과 분석 실패: {e}")
            return {'weekly_profit': 0.0, 'weekly_return': 0.0, 'weekly_trades': 0, 'week_start': datetime.now(), 'portfolio_value': 1000000}
    
    def _is_trading_day(self) -> bool:
        today = datetime.now()
        
        # 주말 제외
        if today.weekday() >= 5:
            return False
        
        # 공휴일 체크 (간단한 버전)
        holidays = [
            datetime(today.year, 1, 1),   # 신정
            datetime(today.year, 7, 4),   # 독립기념일
            datetime(today.year, 12, 25), # 크리스마스
        ]
        
        if any(today.date() == holiday.date() for holiday in holidays):
            return False
        
        return True
    
    async def _perform_daily_check(self):
        try:
            if not self._is_trading_day():
                return
            
            logging.info("📊 일일 체크...")
            
            # 계좌 정보 업데이트
            await self.ibkr._update_account()
            
            # 월 수익률 계산
            await self._calculate_monthly_return()
            
            # 리스크 체크
            await self._check_risk_limits()
            
        except Exception as e:
            logging.error(f"일일 체크 실패: {e}")
    
    async def _calculate_monthly_return(self):
        try:
            conn = sqlite3.connect(self.stop_take.db_path)
            cursor = conn.cursor()
            
            current_month = datetime.now().strftime('%Y-%m')
            cursor.execute('''
                SELECT SUM(profit_loss) FROM trades 
                WHERE strftime('%Y-%m', timestamp) = ?
                AND action LIKE 'SELL%'
            ''', (current_month,))
            
            result = cursor.fetchone()
            monthly_profit = result[0] if result[0] else 0.0
            
            # 포트폴리오 가치
            portfolio_value = await self.ibkr.get_portfolio_value()
            if portfolio_value > 0:
                self.monthly_return = (monthly_profit / portfolio_value) * 100
            
            conn.close()
            
        except Exception as e:
            logging.error(f"월 수익률 계산 실패: {e}")
    
    async def _check_risk_limits(self):
        try:
            # 일일 손실 한도
            daily_limit = config.get('risk.daily_loss_limit', 1.0)
            portfolio_value = await self.ibkr.get_portfolio_value()
            
            if self.ibkr.daily_pnl < -(portfolio_value * daily_limit / 100):
                await self._emergency_stop("일일 손실 한도 초과")
                return
            
            # 주간 손실 한도 (주간 모드에서)
            if self.weekly_mode:
                weekly_performance = await self._analyze_weekly_performance()
                weekly_limit = config.get('risk.weekly_loss_limit', 3.0)
                if weekly_performance['weekly_return'] < -weekly_limit:
                    await self._emergency_stop(f"주간 손실 한도 초과: {weekly_performance['weekly_return']:.2f}%")
            
            # 월 손실 한도
            monthly_limit = config.get('risk.monthly_loss_limit', 3.0)
            if self.monthly_return < -monthly_limit:
                await self._emergency_stop(f"월 손실 한도 초과: {self.monthly_return:.2f}%")
            
        except Exception as e:
            logging.error(f"리스크 체크 실패: {e}")
    
    async def _emergency_stop(self, reason: str):
        try:
            logging.warning(f"🚨 비상 정지: {reason}")
            
            # 모든 포지션 정리
            for symbol, position in list(self.stop_take.positions.items()):
                await self.ibkr.place_sell_order(symbol, position.quantity, 'EMERGENCY')
            
            # 포지션 초기화
            self.stop_take.positions.clear()
            
            # 알림
            await self.stop_take._send_notification(f"🚨 시스템 비상 정지!\n📝 사유: {reason}\n💰 모든 포지션 정리")
            
        except Exception as e:
            logging.error(f"비상 정지 실패: {e}")
    
    async def _generate_report(self):
        try:
            # 간단한 일일 리포트
            active_positions = len(self.stop_take.positions)
            daily_pnl = self.ibkr.daily_pnl
            
            # 요일별 특별 정보
            today = datetime.now()
            weekday_info = ""
            
            if self.weekly_mode:
                if today.weekday() == 1:  # 화요일
                    last_tuesday = self.last_trade_dates.get('Tuesday')
                    if last_tuesday == today.date():
                        weekday_info = "🔥 오늘 화요일 진입 완료"
                elif today.weekday() == 3:  # 목요일
                    last_thursday = self.last_trade_dates.get('Thursday')
                    if last_thursday == today.date():
                        weekday_info = "📋 오늘 목요일 정리 완료"
            
            report = f"""
🏆 일일 요약 리포트
==================
📊 현재 모드: {self.current_mode.upper()} {"(주간매매)" if self.weekly_mode else ""}
💰 일일 P&L: ${daily_pnl:.2f}
📈 월 수익률: {self.monthly_return:.2f}% (목표: {self.target_min:.1f}%-{self.target_max:.1f}%)
💼 활성 포지션: {active_positions}개
{weekday_info}
"""
            
            await self.stop_take._send_notification(report)
            
        except Exception as e:
            logging.error(f"리포트 생성 실패: {e}")
    
    async def shutdown(self):
        try:
            logging.info("🔌 시스템 종료 중...")
            
            self.stop_take.stop_monitoring()
            await self.ibkr.disconnect()
            
            logging.info("✅ 시스템 종료 완료")
            
        except Exception as e:
            logging.error(f"종료 실패: {e}")

# ========================================================================================
# 🎯 편의 함수들
# ========================================================================================

async def run_auto_selection():
    try:
        strategy = LegendaryQuantStrategy()
        signals = await strategy.scan_all_stocks()
        return signals
    except Exception as e:
        logging.error(f"자동 선별 실패: {e}")
        return []

async def analyze_single_stock(symbol: str):
    try:
        strategy = LegendaryQuantStrategy()
        signal = await strategy.analyze_stock_signal(symbol)
        return signal
    except Exception as e:
        logging.error(f"종목 분석 실패: {e}")
        return None

async def get_system_status():
    try:
        strategy = LegendaryQuantStrategy()
        
        # IBKR 연결 테스트
        ibkr_connected = False
        try:
            if IBKR_AVAILABLE:
                ibkr_connected = await strategy.ibkr.connect()
                if ibkr_connected:
                    await strategy.ibkr.disconnect()
        except Exception:
            ibkr_connected = False
        
        return {
            'enabled': strategy.enabled,
            'current_mode': strategy.current_mode,
            'weekly_mode': strategy.weekly_mode,
            'ibkr_connected': ibkr_connected,
            'selected_count': len(strategy.selected_stocks),
            'target_min': strategy.target_min,
            'target_max': strategy.target_max,
            'monthly_return': strategy.monthly_return,
            'ibkr_available': IBKR_AVAILABLE,
            'last_tuesday': strategy.last_trade_dates.get('Tuesday'),
            'last_thursday': strategy.last_trade_dates.get('Thursday')
        }
        
    except Exception as e:
        logging.error(f"상태 조회 실패: {e}")
        return {'error': str(e)}
        async def run_auto_trading():
    strategy = LegendaryQuantStrategy()
    
    try:
        if await strategy.initialize_trading():
            await strategy.start_auto_trading()
        else:
            logging.error("❌ 거래 시스템 초기화 실패")
    except KeyboardInterrupt:
        logging.info("⏹️ 사용자 중단")
    except Exception as e:
        logging.error(f"❌ 자동거래 실패: {e}")
    finally:
        await strategy.shutdown()

# 🆕 주간 매매 전용 함수들
async def manual_tuesday_trading():
    strategy = LegendaryQuantStrategy()
    try:
        if await strategy.initialize_trading():
            await strategy._execute_tuesday_trading()
            return {'status': 'success', 'message': '화요일 매매 완료'}
        else:
            return {'status': 'error', 'message': '거래 시스템 초기화 실패'}
    except Exception as e:
        logging.error(f"수동 화요일 매매 실패: {e}")
        return {'status': 'error', 'message': str(e)}
    finally:
        await strategy.shutdown()

async def manual_thursday_trading():
    strategy = LegendaryQuantStrategy()
    try:
        if await strategy.initialize_trading():
            await strategy._execute_thursday_trading()
            return {'status': 'success', 'message': '목요일 매매 완료'}
        else:
            return {'status': 'error', 'message': '거래 시스템 초기화 실패'}
    except Exception as e:
        logging.error(f"수동 목요일 매매 실패: {e}")
        return {'status': 'error', 'message': str(e)}
    finally:
        await strategy.shutdown()

async def get_weekly_performance():
    try:
        strategy = LegendaryQuantStrategy()
        if await strategy.initialize_trading():
            performance = await strategy._analyze_weekly_performance()
            await strategy.shutdown()
            return performance
        else:
            return {'error': '시스템 초기화 실패'}
    except Exception as e:
        logging.error(f"주간 성과 조회 실패: {e}")
        return {'error': str(e)}

async def test_market_condition():
    try:
        strategy = LegendaryQuantStrategy()
        condition = await strategy._analyze_market_condition()
        return condition
    except Exception as e:
        logging.error(f"시장 상황 테스트 실패: {e}")
        return {'error': str(e)}

# ========================================================================================
# 🎯 메인 실행부
# ========================================================================================

async def main():
    try:
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('legendary_quant_v63.log', encoding='utf-8')
            ]
        )
        
        print("🏆" + "="*70)
        print("🔥 전설적 퀸트프로젝트 - 간소화 완성판 V6.3")
        print("🚀 월 5-7% 달성형 주 2회 화목 매매 시스템")
        print("="*72)
        
        print("\n🌟 주요 특징:")
        print("  ✨ 4가지 투자전략 지능형 융합 (버핏+린치+모멘텀+기술)")
        print("  ✨ 실시간 S&P500+NASDAQ 자동선별")
        print("  ✨ VIX 기반 시장상황 자동판단")
        print("  ✨ 월 5-7% 달성형 스윙 + 분할매매 통합")
        print("  ✨ 🆕 주 2회 화목 매매 최적화")
        print("  ✨ IBKR 실거래 연동 + 자동 손익절")
        
        print("\n📅 주 2회 화목 매매:")
        print("  🔥 화요일 10:30: 공격적 신규 진입 (4-6개 종목)")
        print("  📋 목요일 10:30: 포지션 정리 + 이익실현")
        print("  📊 VIX 연동 시장 상황별 자동 조정")
        
        # 시스템 상태 확인
        print("\n🔧 시스템 상태 확인...")
        status = await get_system_status()
        
        if 'error' not in status:
            print(f"  ✅ 시스템 활성화: {status['enabled']}")
            print(f"  ✅ 현재 모드: {status['current_mode'].upper()}")
            print(f"  ✅ 주간 매매: {'활성화' if status.get('weekly_mode', False) else '비활성화'}")
            
            # IBKR 상태 표시
            if status.get('ibkr_available', False):
                ibkr_status = '연결 가능' if status['ibkr_connected'] else '연결 불가'
                print(f"  ✅ IBKR 상태: {ibkr_status}")
            else:
                print(f"  ⚠️  IBKR 모듈: 미설치 (pip install ib_insync)")
            
            print(f"  ✅ 월 목표: {status['target_min']:.1f}%-{status['target_max']:.1f}%")
            print(f"  ✅ 월 수익률: {status['monthly_return']:.2f}%")
            print(f"  ✅ 선별된 종목: {status['selected_count']}개")
            
            # 주간 매매 정보
            if status.get('last_tuesday'):
                print(f"  🔥 마지막 화요일: {status['last_tuesday']}")
            if status.get('last_thursday'):
                print(f"  📋 마지막 목요일: {status['last_thursday']}")
        else:
            print(f"  ❌ 상태 확인 실패: {status['error']}")
        
        print("\n🚀 실행 옵션:")
        print("  1. 🏆 완전 자동 주 2회 화목 매매")
        print("  2. 🔥 수동 화요일 매매 (공격적 진입)")
        print("  3. 📋 수동 목요일 매매 (포지션 정리)")
        print("  4. 🔍 종목 자동선별 + 분석")
        print("  5. 📊 개별 종목 분석")
        print("  6. 📈 주간 성과 + 시장 상황")
        print("  7. 📊 시스템 상태")
        print("  0. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (0-7): ").strip()
                
                if choice == '1':
                    print("\n🏆 완전 자동 주 2회 화목 매매 시작!")
                    print("⚠️  IBKR TWS/Gateway가 실행 중인지 확인하세요!")
                    print("📅 화요일 10:30 - 공격적 진입 (VIX 연동)")
                    print("📅 목요일 10:30 - 포지션 정리 + 이익실현")
                    print("🎯 월 5-7% 목표 달성형 최적화")
                    confirm = input("계속하시겠습니까? (y/N): ").strip().lower()
                    if confirm == 'y':
                        await run_auto_trading()
                    break
                
                elif choice == '2':
                    print("\n🔥 수동 화요일 매매 실행!")
                    print("📊 시장 상황 분석 후 공격적 진입")
                    confirm = input("화요일 공격적 진입을 실행하시겠습니까? (y/N): ").strip().lower()
                    if confirm == 'y':
                        result = await manual_tuesday_trading()
                        if result['status'] == 'success':
                            print("✅ 화요일 매매 완료!")
                        else:
                            print(f"❌ 화요일 매매 실패: {result.get('message', '알 수 없는 오류')}")
                
                elif choice == '3':
                    print("\n📋 수동 목요일 매매 실행!")
                    print("💰 이익실현 + 손절 + 선별적 신규진입")
                    confirm = input("목요일 포지션 정리를 실행하시겠습니까? (y/N): ").strip().lower()
                    if confirm == 'y':
                        result = await manual_thursday_trading()
                        if result['status'] == 'success':
                            print("✅ 목요일 매매 완료!")
                        else:
                            print(f"❌ 목요일 매매 실패: {result.get('message', '알 수 없는 오류')}")
                
                elif choice == '4':
                    print("\n🔍 종목 자동선별 + 분석 시작!")
                    signals = await run_auto_selection()
                    
                    if signals:
                        print(f"\n📈 분석 결과:")
                        buy_signals = [s for s in signals if s.action == 'buy']
                        sell_signals = [s for s in signals if s.action == 'sell']
                        
                        print(f"  🟢 매수 추천: {len(buy_signals)}개")
                        print(f"  🔴 매도 추천: {len(sell_signals)}개")
                        print(f"  ⚪ 보유 추천: {len(signals) - len(buy_signals) - len(sell_signals)}개")
                        
                        # 상위 매수 추천
                        top_buys = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
                        if top_buys:
                            print(f"\n🏆 상위 매수 추천 (화요일 진입 후보):")
                            for i, signal in enumerate(top_buys, 1):
                                print(f"  {i}. {signal.symbol}: 신뢰도 {signal.confidence:.1%}, "
                                      f"목표가 ${signal.target_price:.2f}")
                    else:
                        print("❌ 분석 결과 없음")
                
                elif choice == '5':
                    symbol = input("분석할 종목 심볼: ").strip().upper()
                    if symbol:
                        print(f"\n🔍 {symbol} 분석중...")
                        signal = await analyze_single_stock(symbol)
                        
                        if signal and signal.confidence > 0:
                            print(f"\n📊 {symbol} 분석 결과:")
                            print(f"  🎯 결정: {signal.action.upper()}")
                            print(f"  💯 신뢰도: {signal.confidence:.1%}")
                            print(f"  💰 현재가: ${signal.price:.2f}")
                            print(f"  🛑 손절가: ${signal.stop_loss:.2f}")
                            print(f"  📈 모드: {signal.mode.upper()}")
                            print(f"  💡 근거: {signal.reasoning}")
                        else:
                            print(f"❌ {symbol} 분석 실패")
                
                elif choice == '6':
                    print("\n📈 주간 성과 + 시장 상황 분석...")
                    
                    # 시장 상황
                    print("📊 현재 시장 상황:")
                    condition = await test_market_condition()
                    if 'error' not in condition:
                        print(f"  VIX: {condition['vix']:.1f}")
                        print(f"  SPY 모멘텀: {condition['spy_momentum']:.1f}%")
                        print(f"  QQQ 모멘텀: {condition['qqq_momentum']:.1f}%")
                        print(f"  시장 상태: {condition['status']}")
                        print(f"  공격성 지수: {condition['aggressiveness']:.1f}")
                        print(f"  매매 가능: {'✅' if condition['safe_to_trade'] else '❌'}")
                    
                    # 주간 성과
                    print("\n📈 주간 성과:")
                    performance = await get_weekly_performance()
                    if 'error' not in performance:
                        print(f"  주간 수익률: {performance['weekly_return']:.2f}%")
                        print(f"  주간 P&L: ${performance['weekly_profit']:.2f}")
                        print(f"  주간 거래: {performance['weekly_trades']}회")
                        print(f"  포트폴리오: ${performance['portfolio_value']:.0f}")
                    else:
                        print(f"  ❌ 성과 조회 실패: {performance['error']}")
                
                elif choice == '7':
                    print("\n📊 시스템 상세 상태:")
                    status = await get_system_status()
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                
                elif choice == '0':
                    print("👋 시스템을 종료합니다!")
                    break
                    
                else:
                    print("❌ 잘못된 선택입니다. 0-7 중 선택하세요.")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
    except Exception as e:
        logging.error(f"메인 실행 실패: {e}")
        print(f"❌ 실행 실패: {e}")

# ========================================================================================
# 🏃‍♂️ 빠른 시작 함수들
# ========================================================================================

async def quick_analysis(symbols: List[str] = None):
    if not symbols:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    print(f"🚀 빠른 분석: {', '.join(symbols)}")
    
    strategy = LegendaryQuantStrategy()
    
    for symbol in symbols:
        try:
            signal = await strategy.analyze_stock_signal(symbol)
            action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
            print(f"{action_emoji} {symbol}: {signal.action} ({signal.confidence:.1%})")
        except Exception as e:
            print(f"❌ {symbol}: 분석 실패")

async def quick_scan():
    print("🔍 빠른 전체 스캔...")
    
    try:
        signals = await run_auto_selection()
        
        if signals:
            buy_signals = [s for s in signals if s.action == 'buy']
            
            print(f"\n📊 스캔 결과: 총 {len(signals)}개 종목")
            print(f"🟢 매수 추천: {len(buy_signals)}개")
            
            # 상위 5개
            top_5 = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
            if top_5:
                print("\n🏆 TOP 5 매수 추천:")
                for i, signal in enumerate(top_5, 1):
                    print(f"  {i}. {signal.symbol}: {signal.confidence:.1%}")
        else:
            print("❌ 스캔 결과 없음")
            
    except Exception as e:
        print(f"❌ 스캔 실패: {e}")

async def quick_weekly_status():
    print("📅 빠른 주간 상태 체크...")
    
    try:
        # 시장 상황
        condition = await test_market_condition()
        print(f"📊 시장: {condition.get('status', 'unknown')} (VIX: {condition.get('vix', 0):.1f})")
        
        # 주간 성과
        performance = await get_weekly_performance()
        if 'error' not in performance:
            print(f"📈 주간: {performance['weekly_return']:+.2f}% ({performance['weekly_trades']}회 거래)")
        
        # 시스템 상태
        status = await get_system_status()
        if 'error' not in status:
            weekly_text = "ON" if status.get('weekly_mode') else "OFF"
            print(f"⚙️  시스템: {status['current_mode'].upper()} | 주간모드: {weekly_text}")
            
    except Exception as e:
        print(f"❌ 상태 체크 실패: {e}")

def print_help():
    help_text = """
🏆 전설적 퀸트프로젝트 V6.3 - 주 2회 화목 매매 시스템
=======================================================

📋 주요 명령어:
  python legendary_quant_v63.py        # 메인 메뉴 실행
  python -c "from legendary_quant_v63 import *; asyncio.run(quick_weekly_status())"  # 빠른 상태
  python -c "from legendary_quant_v63 import *; asyncio.run(quick_scan())"  # 빠른 스캔
  python -c "from legendary_quant_v63 import *; asyncio.run(quick_analysis())"  # 빠른 분석

🔧 초기 설정:
  1. pip install yfinance pandas numpy requests beautifulsoup4 aiohttp python-dotenv
  2. IBKR 사용시: pip install ib_insync
  3. .env 파일에서 텔레그램/IBKR 설정

📅 주 2회 화목 매매 스케줄:
  🔥 화요일 10:30: 공격적 신규 진입
    - 시장 상황 분석 (VIX, SPY/QQQ 모멘텀)
    - 공격성 지수 적용 (0.6x - 1.3x)
    - 4-6개 종목 선별적 진입
    - 12.5% 기본 포지션 (시장 상황에 따라 조정)
    
  📋 목요일 10:30: 포지션 정리 및 최적화
    - 8% 이상 수익 → 50% 부분 이익실현
    - -6% 이하 손실 → 전량 청산
    - 장기 보유 미수익 종목 정리
    - 선별적 보수적 신규 진입 (최대 2개)

🎯 월 5-7% 목표 달성 전략:
  - VIX < 15: 공격성 1.3x (저변동성 기회 활용)
  - VIX 15-25: 표준 전략 (1.0x)
  - VIX 25-30: 보수적 접근 (0.7x)
  - VIX > 30: 매매 중단 (고위험 회피)

💡 4가지 전략 융합:
  - 버핏 가치투자: PBR, ROE, 부채비율 중심 (25%)
  - 린치 성장투자: PEG, EPS성장률 중심 (25%)
  - 모멘텀 전략: 3/6/12개월 수익률 중심 (25%)
  - 기술적 분석: RSI, 추세, 변동성 중심 (25%)

🛡️ 통합 리스크 관리:
  - 일일 손실 한도: 1%
  - 주간 손실 한도: 3%
  - 월간 손실 한도: 3%
  - 개별 종목 최대: 15%
  - 자동 트레일링 스톱

📱 실시간 알림:
  - 텔레그램: 진입/청산/성과 알림
  - 주간 리포트: 목요일 성과 요약
"""
    print(help_text)

# ========================================================================================
# 🏁 실행 진입점
# ========================================================================================

if __name__ == "__main__":
    try:
        # 명령행 인자 처리
        if len(sys.argv) > 1:
            if sys.argv[1] == 'help' or sys.argv[1] == '--help':
                print_help()
                sys.exit(0)
            elif sys.argv[1] == 'quick-scan':
                asyncio.run(quick_scan())
                sys.exit(0)
            elif sys.argv[1] == 'quick-analysis':
                symbols = sys.argv[2:] if len(sys.argv) > 2 else None
                asyncio.run(quick_analysis(symbols))
                sys.exit(0)
            elif sys.argv[1] == 'quick-weekly':
                asyncio.run(quick_weekly_status())
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
        print("\n👋 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        logging.error(f"실행 오류: {e}")
