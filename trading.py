#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 전설적 퀸트프로젝트 TRADING 시스템 - 완전통합판
================================================================

실제 거래 실행 + 포지션 관리 + 리스크 제어 + 자동화 시스템
- 🇺🇸 미국주식: 스윙+클래식 전략 (IBKR)
- 🇯🇵 일본주식: 화목 하이브리드 (IBKR) 
- 🇮🇳 인도주식: 수요일 전용 안정형 (IBKR)
- 🪙 가상화폐: 월금 매매 (업비트)
- 🛡️ 통합 리스크 관리 시스템
- 📊 실시간 포지션 모니터링
- 🔔 자동 알림 연동

Author: 전설적퀸트팀 | Version: TRADING v1.0
"""

import asyncio
import logging
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import uuid
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import schedule

# 전략 모듈들 임포트
try:
    from core import QuintCore, Signal, Position, PerformanceMetrics
    from us_strategy import LegendaryQuantStrategy as USStrategy
    from jp_strategy import YenHunter as JPStrategy
    from inda_strategy import LegendaryIndiaStrategy as INStrategy
    from coin_strategy import LegendaryQuantMaster as CoinStrategy
    from notifier import CoreNotificationInterface
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    logging.warning(f"⚠️ 전략 모듈 임포트 실패: {e}")

# 거래소 API 임포트
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    logging.warning("⚠️ IBKR API 없음")

try:
    import pyupbit
    UPBIT_AVAILABLE = True
except ImportError:
    UPBIT_AVAILABLE = False
    logging.warning("⚠️ Upbit API 없음")

from dotenv import load_dotenv
load_dotenv()

# ========================================================================================
# 📊 거래 시스템 데이터 클래스
# ========================================================================================

class MarketType(Enum):
    """시장 유형"""
    US_STOCK = "us_stock"
    JP_STOCK = "jp_stock"
    IN_STOCK = "in_stock"
    CRYPTO = "crypto"

class OrderType(Enum):
    """주문 유형"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class TradingConfig:
    """거래 설정"""
    # 기본 설정
    demo_mode: bool = True
    auto_trading: bool = False
    max_positions_per_strategy: int = 8
    
    # 자금 관리
    total_capital: float = 1_000_000_000  # 10억원
    max_daily_loss: float = 0.02  # 일일 최대 손실 2%
    position_size_limit: float = 0.15  # 포지션당 최대 15%
    
    # 거래 시간
    trading_start: str = "09:00"
    trading_end: str = "15:30"
    
    # 전략별 활성화
    us_enabled: bool = True
    jp_enabled: bool = True
    in_enabled: bool = True
    crypto_enabled: bool = True
    
    # 알림 설정
    notifications_enabled: bool = True
    telegram_alerts: bool = True
    
    # 리스크 관리
    stop_loss_enabled: bool = True
    take_profit_enabled: bool = True
    trailing_stop_enabled: bool = True

@dataclass
class TradingOrder:
    """거래 주문"""
    id: str
    symbol: str
    market: MarketType
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    strategy: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, REJECTED
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class TradingPosition:
    """거래 포지션"""
    id: str
    symbol: str
    market: MarketType
    strategy: str
    quantity: float
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_date: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class TradingPerformance:
    """거래 성과"""
    strategy: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    current_positions: int
    last_updated: datetime = field(default_factory=datetime.now)

# ========================================================================================
# 🏦 IBKR 거래 인터페이스
# ========================================================================================

class IBKRTradingInterface:
    """IBKR 거래 인터페이스"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.ib = None
        self.connected = False
        self.account_id = os.getenv('IBKR_ACCOUNT_ID', '')
        self.host = os.getenv('IBKR_HOST', '127.0.0.1')
        self.port = int(os.getenv('IBKR_PORT', '7497'))
        self.client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))
        
    async def connect(self) -> bool:
        """IBKR 연결"""
        try:
            if not IBKR_AVAILABLE:
                logging.error("❌ IBKR 라이브러리 없음")
                return False
            
            self.ib = IB()
            await self.ib.connectAsync(self.host, self.port, self.client_id)
            
            if self.ib.isConnected():
                self.connected = True
                mode = "Paper" if self.config.demo_mode else "Live"
                logging.info(f"✅ IBKR 연결 성공 ({mode})")
                return True
            else:
                logging.error("❌ IBKR 연결 실패")
                return False
                
        except Exception as e:
            logging.error(f"❌ IBKR 연결 오류: {e}")
            return False
    
    async def disconnect(self):
        """연결 해제"""
        try:
            if self.ib and self.connected:
                self.ib.disconnect()
                self.connected = False
                logging.info("🔌 IBKR 연결 해제")
        except Exception as e:
            logging.error(f"연결 해제 오류: {e}")
    
    def create_contract(self, symbol: str, market: MarketType) -> Optional[Contract]:
        """계약 생성"""
        try:
            contract = Contract()
            
            if market == MarketType.US_STOCK:
                contract.symbol = symbol
                contract.secType = 'STK'
                contract.exchange = 'SMART'
                contract.currency = 'USD'
                
            elif market == MarketType.JP_STOCK:
                contract.symbol = symbol.replace('.T', '')
                contract.secType = 'STK'
                contract.exchange = 'TSE'
                contract.currency = 'JPY'
                
            elif market == MarketType.IN_STOCK:
                contract.symbol = symbol
                contract.secType = 'STK'
                contract.exchange = 'NSE'
                contract.currency = 'INR'
                
            else:
                logging.error(f"지원하지 않는 시장: {market}")
                return None
            
            return contract
            
        except Exception as e:
            logging.error(f"계약 생성 실패 {symbol}: {e}")
            return None
    
    async def place_order(self, trading_order: TradingOrder) -> bool:
        """주문 실행"""
        try:
            if not self.connected:
                logging.error("❌ IBKR 연결되지 않음")
                return False
            
            contract = self.create_contract(trading_order.symbol, trading_order.market)
            if not contract:
                return False
            
            # 주문 생성
            order = Order()
            order.action = trading_order.side.value.upper()
            order.totalQuantity = trading_order.quantity
            
            if trading_order.order_type == OrderType.MARKET:
                order.orderType = 'MKT'
            elif trading_order.order_type == OrderType.LIMIT:
                order.orderType = 'LMT'
                order.lmtPrice = trading_order.price
            elif trading_order.order_type == OrderType.STOP:
                order.orderType = 'STP'
                order.auxPrice = trading_order.stop_price
            
            order.tif = trading_order.time_in_force
            
            if self.config.demo_mode:
                # 시뮬레이션 모드
                trading_order.status = "FILLED"
                trading_order.filled_at = datetime.now()
                trading_order.filled_price = trading_order.price or 0
                trading_order.filled_quantity = trading_order.quantity
                trading_order.broker_order_id = f"SIM_{uuid.uuid4().hex[:8]}"
                
                logging.info(f"📈 [시뮬] {trading_order.side.value.upper()} {trading_order.symbol} "
                           f"{trading_order.quantity} @ {trading_order.filled_price}")
                return True
            
            else:
                # 실제 주문
                trade = self.ib.placeOrder(contract, order)
                trading_order.broker_order_id = str(trade.order.orderId)
                trading_order.status = "PENDING"
                
                logging.info(f"📈 [실제] {trading_order.side.value.upper()} {trading_order.symbol} "
                           f"{trading_order.quantity} 주문 실행")
                return True
            
        except Exception as e:
            logging.error(f"❌ 주문 실행 실패: {e}")
            trading_order.status = "REJECTED"
            trading_order.error_message = str(e)
            return False
    
    async def get_current_price(self, symbol: str, market: MarketType) -> Optional[float]:
        """현재가 조회"""
        try:
            if not self.connected:
                return None
            
            contract = self.create_contract(symbol, market)
            if not contract:
                return None
            
            if self.config.demo_mode:
                # 시뮬레이션 가격 (실제로는 외부 API에서 가져와야 함)
                return 100.0 + np.random.normal(0, 5)
            
            ticker = self.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(1)
            
            price = ticker.marketPrice() or ticker.last or ticker.close
            self.ib.cancelMktData(contract)
            
            return float(price) if price and price > 0 else None
            
        except Exception as e:
            logging.error(f"현재가 조회 실패 {symbol}: {e}")
            return None
    
    async def get_positions(self) -> List[Dict]:
        """포지션 조회"""
        try:
            if not self.connected:
                return []
            
            if self.config.demo_mode:
                # 시뮬레이션 포지션 (실제로는 DB에서 로드)
                return []
            
            positions = self.ib.positions()
            result = []
            
            for position in positions:
                if position.position != 0:
                    result.append({
                        'symbol': position.contract.symbol,
                        'quantity': position.position,
                        'avg_cost': position.avgCost,
                        'market_price': position.marketPrice,
                        'unrealized_pnl': position.unrealizedPNL
                    })
            
            return result
            
        except Exception as e:
            logging.error(f"포지션 조회 실패: {e}")
            return []

# ========================================================================================
# 🪙 업비트 거래 인터페이스
# ========================================================================================

class UpbitTradingInterface:
    """업비트 거래 인터페이스"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.upbit = None
        self.connected = False
        
        if UPBIT_AVAILABLE:
            access_key = os.getenv('UPBIT_ACCESS_KEY', '')
            secret_key = os.getenv('UPBIT_SECRET_KEY', '')
            
            if access_key and secret_key:
                self.upbit = pyupbit.Upbit(access_key, secret_key)
                self.connected = True
    
    async def place_order(self, trading_order: TradingOrder) -> bool:
        """주문 실행"""
        try:
            if not self.connected and not self.config.demo_mode:
                logging.error("❌ 업비트 연결되지 않음")
                return False
            
            symbol = trading_order.symbol
            side = trading_order.side
            quantity = trading_order.quantity
            
            if self.config.demo_mode:
                # 시뮬레이션 모드
                current_price = pyupbit.get_current_price(symbol)
                
                trading_order.status = "FILLED"
                trading_order.filled_at = datetime.now()
                trading_order.filled_price = current_price or trading_order.price or 0
                trading_order.filled_quantity = quantity
                trading_order.broker_order_id = f"UPB_SIM_{uuid.uuid4().hex[:8]}"
                
                logging.info(f"🪙 [시뮬] {side.value.upper()} {symbol} "
                           f"{quantity:.6f} @ {trading_order.filled_price:,.0f}원")
                return True
            
            else:
                # 실제 주문
                if side == OrderSide.BUY:
                    # 시장가 매수 (원화 기준)
                    result = self.upbit.buy_market_order(symbol, quantity)
                else:
                    # 시장가 매도 (코인 수량 기준)
                    result = self.upbit.sell_market_order(symbol, quantity)
                
                if result:
                    trading_order.status = "PENDING"
                    trading_order.broker_order_id = result.get('uuid', '')
                    
                    logging.info(f"🪙 [실제] {side.value.upper()} {symbol} "
                               f"{quantity:.6f} 주문 실행")
                    return True
                else:
                    trading_order.status = "REJECTED"
                    return False
            
        except Exception as e:
            logging.error(f"❌ 업비트 주문 실행 실패: {e}")
            trading_order.status = "REJECTED"
            trading_order.error_message = str(e)
            return False
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회"""
        try:
            price = pyupbit.get_current_price(symbol)
            return float(price) if price else None
        except Exception as e:
            logging.error(f"업비트 현재가 조회 실패 {symbol}: {e}")
            return None
    
    async def get_balances(self) -> List[Dict]:
        """잔고 조회"""
        try:
            if not self.connected:
                return []
            
            balances = self.upbit.get_balances()
            result = []
            
            for balance in balances:
                if float(balance['balance']) > 0:
                    result.append({
                        'currency': balance['currency'],
                        'balance': float(balance['balance']),
                        'locked': float(balance['locked']),
                        'avg_buy_price': float(balance['avg_buy_price'])
                    })
            
            return result
            
        except Exception as e:
            logging.error(f"업비트 잔고 조회 실패: {e}")
            return []

# ========================================================================================
# 🗄️ 포지션 관리자
# ========================================================================================

class PositionManager:
    """통합 포지션 관리자"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.positions: Dict[str, TradingPosition] = {}
        self.orders: Dict[str, TradingOrder] = {}
        self.db_path = "trading_data.db"
        self._init_database()
        self.load_positions()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 포지션 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    entry_date TEXT NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    trailing_stop REAL,
                    last_updated TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # 주문 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    stop_price REAL,
                    strategy TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    filled_at TEXT,
                    status TEXT NOT NULL,
                    filled_price REAL,
                    filled_quantity REAL,
                    broker_order_id TEXT,
                    error_message TEXT
                )
            ''')
            
            # 성과 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    strategy TEXT PRIMARY KEY,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    avg_win REAL NOT NULL,
                    avg_loss REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    current_positions INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("📊 거래 데이터베이스 초기화 완료")
            
        except Exception as e:
            logging.error(f"DB 초기화 실패: {e}")
    
    def add_position(self, position: TradingPosition):
        """포지션 추가"""
        try:
            self.positions[position.id] = position
            self.save_position(position)
            
            logging.info(f"➕ 포지션 추가: {position.symbol} {position.quantity} @ {position.avg_cost}")
            
        except Exception as e:
            logging.error(f"포지션 추가 실패: {e}")
    
    def update_position(self, position_id: str, **kwargs):
        """포지션 업데이트"""
        try:
            if position_id in self.positions:
                position = self.positions[position_id]
                
                for key, value in kwargs.items():
                    if hasattr(position, key):
                        setattr(position, key, value)
                
                position.last_updated = datetime.now()
                self.save_position(position)
                
        except Exception as e:
            logging.error(f"포지션 업데이트 실패: {e}")
    
    def remove_position(self, position_id: str):
        """포지션 제거"""
        try:
            if position_id in self.positions:
                position = self.positions[position_id]
                del self.positions[position_id]
                
                # DB에서 제거
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM positions WHERE id = ?', (position_id,))
                conn.commit()
                conn.close()
                
                logging.info(f"➖ 포지션 제거: {position.symbol}")
                
        except Exception as e:
            logging.error(f"포지션 제거 실패: {e}")
    
    def save_position(self, position: TradingPosition):
        """포지션 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                (id, symbol, market, strategy, quantity, avg_cost, current_price, 
                 unrealized_pnl, realized_pnl, entry_date, stop_loss, take_profit, 
                 trailing_stop, last_updated, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.id, position.symbol, position.market.value, position.strategy,
                position.quantity, position.avg_cost, position.current_price,
                position.unrealized_pnl, position.realized_pnl, 
                position.entry_date.isoformat(), position.stop_loss, position.take_profit,
                position.trailing_stop, position.last_updated.isoformat(),
                json.dumps(position.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"포지션 저장 실패: {e}")
    
    def load_positions(self):
        """포지션 로드"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM positions')
            rows = cursor.fetchall()
            
            for row in rows:
                position = TradingPosition(
                    id=row[0],
                    symbol=row[1],
                    market=MarketType(row[2]),
                    strategy=row[3],
                    quantity=row[4],
                    avg_cost=row[5],
                    current_price=row[6],
                    unrealized_pnl=row[7],
                    realized_pnl=row[8],
                    entry_date=datetime.fromisoformat(row[9]),
                    stop_loss=row[10],
                    take_profit=row[11],
                    trailing_stop=row[12],
                    last_updated=datetime.fromisoformat(row[13]),
                    metadata=json.loads(row[14]) if row[14] else {}
                )
                
                self.positions[position.id] = position
            
            conn.close()
            logging.info(f"📂 포지션 로드 완료: {len(self.positions)}개")
            
        except Exception as e:
            logging.error(f"포지션 로드 실패: {e}")
    
    def get_positions_by_strategy(self, strategy: str) -> List[TradingPosition]:
        """전략별 포지션 조회"""
        return [pos for pos in self.positions.values() if pos.strategy == strategy]
    
    def get_positions_by_market(self, market: MarketType) -> List[TradingPosition]:
        """시장별 포지션 조회"""
        return [pos for pos in self.positions.values() if pos.market == market]
    
    def calculate_total_exposure(self) -> float:
        """총 노출 금액 계산"""
        total = 0.0
        for position in self.positions.values():
            total += abs(position.quantity * position.current_price)
        return total
    
    def calculate_unrealized_pnl(self) -> float:
        """총 미실현 손익 계산"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

# ========================================================================================
# 🎯 통합 거래 실행기
# ========================================================================================

class TradingExecutor:
    """통합 거래 실행기"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.position_manager = PositionManager(config)
        
        # 거래소 인터페이스
        self.ibkr = IBKRTradingInterface(config)
        self.upbit = UpbitTradingInterface(config)
        
        # 전략 시스템들
        self.us_strategy = None
        self.jp_strategy = None
        self.in_strategy = None
        self.crypto_strategy = None
        
        # 알림 시스템
        self.notifier = None
        if config.notifications_enabled:
            try:
                self.notifier = CoreNotificationInterface()
            except Exception as e:
                logging.warning(f"알림 시스템 초기화 실패: {e}")
        
        # 실행 상태
        self.running = False
        self.last_execution_time = {}
        
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """전략 시스템 초기화"""
        try:
            if CORE_AVAILABLE:
                if self.config.us_enabled:
                    self.us_strategy = USStrategy()
                    logging.info("✅ 미국 전략 시스템 초기화")
                
                if self.config.jp_enabled:
                    self.jp_strategy = JPStrategy()
                    logging.info("✅ 일본 전략 시스템 초기화")
                
                if self.config.in_enabled:
                    self.in_strategy = INStrategy()
                    logging.info("✅ 인도 전략 시스템 초기화")
                
                if self.config.crypto_enabled:
                    self.crypto_strategy = CoinStrategy()
                    logging.info("✅ 암호화폐 전략 시스템 초기화")
            
        except Exception as e:
            logging.error(f"전략 시스템 초기화 실패: {e}")
    
    async def initialize(self) -> bool:
        """거래 시스템 초기화"""
        try:
            logging.info("🚀 거래 시스템 초기화 시작...")
            
            # IBKR 연결
            if self.config.us_enabled or self.config.jp_enabled or self.config.in_enabled:
                if not await self.ibkr.connect():
                    logging.warning("⚠️ IBKR 연결 실패 - 주식 거래 불가")
            
            # 업비트 연결 확인
            if self.config.crypto_enabled:
                if not self.upbit.connected and not self.config.demo_mode:
                    logging.warning("⚠️ 업비트 연결 실패 - 암호화폐 거래 불가")
            
            logging.info("✅ 거래 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logging.error(f"❌ 거래 시스템 초기화 실패: {e}")
            return False
    
    async def execute_strategy_signals(self, strategy: str, signals: List) -> Dict:
        """전략 신호 실행"""
        results = {
            'strategy': strategy,
            'total_signals': len(signals),
            'executed_orders': 0,
            'failed_orders': 0,
            'orders': []
        }
        
        try:
            buy_signals = [s for s in signals if hasattr(s, 'action') and s.action == 'buy']
            
            for signal in buy_signals:
                try:
                    # 거래 주문 생성
                    order = await self._create_order_from_signal(signal, strategy)
                    if not order:
                        continue
                    
                    # 주문 실행
                    success = await self._execute_order(order)
                    if success:
                        results['executed_orders'] += 1
                        
                        # 포지션 생성
                        if order.status == "FILLED":
                            await self._create_position_from_order(order, signal, strategy)
                    else:
                        results['failed_orders'] += 1
                    
                    results['orders'].append(order)
                    
                except Exception as e:
                    logging.error(f"신호 실행 실패 {getattr(signal, 'symbol', 'N/A')}: {e}")
                    results['failed_orders'] += 1
            
            # 알림 전송
            if self.notifier and results['executed_orders'] > 0:
                await self.notifier.notify_signals({strategy: buy_signals})
            
            logging.info(f"📊 {strategy} 전략 실행: {results['executed_orders']}개 주문 성공, "
                        f"{results['failed_orders']}개 실패")
            
        except Exception as e:
            logging.error(f"전략 신호 실행 실패 {strategy}: {e}")
        
        return results
    
    async def _create_order_from_signal(self, signal, strategy: str) -> Optional[TradingOrder]:
        """신호에서 주문 생성"""
        try:
            symbol = getattr(signal, 'symbol', '')
            price = getattr(signal, 'price', 0)
            confidence = getattr(signal, 'confidence', 0.5)
            
            if not symbol or price <= 0:
                return None
            
            # 시장 타입 결정
            market = self._determine_market_type(symbol, strategy)
            if not market:
                return None
            
            # 포지션 크기 계산
            quantity = self._calculate_position_size(signal, strategy, market)
            if quantity <= 0:
                return None
            
            # 주문 생성
            order = TradingOrder(
                id=str(uuid.uuid4()),
                symbol=symbol,
                market=market,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=price,
                strategy=strategy
            )
            
            return order
            
        except Exception as e:
            logging.error(f"주문 생성 실패: {e}")
            return None
    
    def _determine_market_type(self, symbol: str, strategy: str) -> Optional[MarketType]:
        """시장 타입 결정"""
        if strategy == 'us':
            return MarketType.US_STOCK
        elif strategy == 'japan':
            return MarketType.JP_STOCK
        elif strategy == 'india':
            return MarketType.IN_STOCK
        elif strategy == 'crypto':
            return MarketType.CRYPTO
        else:
            return None
    
    def _calculate_position_size(self, signal, strategy: str, market: MarketType) -> float:
        """포지션 크기 계산"""
        try:
            price = getattr(signal, 'price', 0)
            confidence = getattr(signal, 'confidence', 0.5)
            
            if price <= 0:
                return 0.0
            
            # 전략별 기본 할당
            base_allocation = {
                'us': 0.40,      # 40%
                'japan': 0.25,   # 25%
                'india': 0.20,   # 20%
                'crypto': 0.10   # 10%
            }
            
            strategy_allocation = base_allocation.get(strategy, 0.1)
            
            # 신뢰도 조정
            confidence_multiplier = 0.5 + (confidence * 0.5)
            
            # 목표 투자 금액
            target_investment = self.config.total_capital * strategy_allocation * confidence_multiplier
            
            # 포지션 크기 제한
            max_investment = self.config.total_capital * self.config.position_size_limit
            target_investment = min(target_investment, max_investment)
            
            # 수량 계산
            if market == MarketType.CRYPTO:
                # 암호화폐는 원화 기준
                return target_investment
            else:
                # 주식은 주수 계산
                return int(target_investment / price)
            
        except Exception as e:
            logging.error(f"포지션 크기 계산 실패: {e}")
            return 0.0
    
    async def _execute_order(self, order: TradingOrder) -> bool:
        """주문 실행"""
        try:
            if order.market in [MarketType.US_STOCK, MarketType.JP_STOCK, MarketType.IN_STOCK]:
                return await self.ibkr.place_order(order)
            elif order.market == MarketType.CRYPTO:
                return await self.upbit.place_order(order)
            else:
                logging.error(f"지원하지 않는 시장: {order.market}")
                return False
                
        except Exception as e:
            logging.error(f"주문 실행 실패: {e}")
            return False
    
    async def _create_position_from_order(self, order: TradingOrder, signal, strategy: str):
        """주문에서 포지션 생성"""
        try:
            # 손절가/익절가 계산
            stop_loss = getattr(signal, 'stop_loss', None)
            take_profit = getattr(signal, 'target_price', None)
            
            position = TradingPosition(
                id=str(uuid.uuid4()),
                symbol=order.symbol,
                market=order.market,
                strategy=strategy,
                quantity=order.filled_quantity,
                avg_cost=order.filled_price,
                current_price=order.filled_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_date=order.filled_at or datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'entry_signal': getattr(signal, 'reasoning', ''),
                    'confidence': getattr(signal, 'confidence', 0.5),
                    'order_id': order.id
                }
            )
            
            self.position_manager.add_position(position)
            
        except Exception as e:
            logging.error(f"포지션 생성 실패: {e}")
    
    async def monitor_positions(self):
        """포지션 모니터링"""
        try:
            if not self.position_manager.positions:
                return
            
            for position_id, position in list(self.position_manager.positions.items()):
                try:
                    # 현재가 업데이트
                    await self._update_position_price(position)
                    
                    # 리스크 관리 체크
                    exit_signal = await self._check_exit_conditions(position)
                    
                    if exit_signal:
                        await self._execute_exit_order(position, exit_signal)
                    
                except Exception as e:
                    logging.error(f"포지션 모니터링 실패 {position.symbol}: {e}")
            
        except Exception as e:
            logging.error(f"포지션 모니터링 오류: {e}")
    
    async def _update_position_price(self, position: TradingPosition):
        """포지션 현재가 업데이트"""
        try:
            if position.market in [MarketType.US_STOCK, MarketType.JP_STOCK, MarketType.IN_STOCK]:
                current_price = await self.ibkr.get_current_price(position.symbol, position.market)
            elif position.market == MarketType.CRYPTO:
                current_price = await self.upbit.get_current_price(position.symbol)
            else:
                return
            
            if current_price and current_price > 0:
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
                position.last_updated = datetime.now()
                
                # DB 업데이트
                self.position_manager.save_position(position)
            
        except Exception as e:
            logging.error(f"가격 업데이트 실패 {position.symbol}: {e}")
    
    async def _check_exit_conditions(self, position: TradingPosition) -> Optional[str]:
        """청산 조건 체크"""
        try:
            current_price = position.current_price
            avg_cost = position.avg_cost
            
            if current_price <= 0 or avg_cost <= 0:
                return None
            
            pnl_ratio = (current_price - avg_cost) / avg_cost
            
            # 손절 체크
            if position.stop_loss and current_price <= position.stop_loss:
                return f"stop_loss_{pnl_ratio*100:.1f}%"
            
            # 익절 체크
            if position.take_profit and current_price >= position.take_profit:
                return f"take_profit_{pnl_ratio*100:.1f}%"
            
            # 전략별 시간 기반 청산
            holding_days = (datetime.now() - position.entry_date).days
            
            if position.strategy == 'crypto':
                # 암호화폐: 월금 매매 (2주 홀딩)
                if holding_days >= 14:
                    return f"time_limit_{holding_days}days"
            
            elif position.strategy == 'japan':
                # 일본: 화목 매매 (1주 홀딩)
                if holding_days >= 7:
                    return f"time_limit_{holding_days}days"
            
            elif position.strategy == 'india':
                # 인도: 수요일 매매 (1주 홀딩)
                if holding_days >= 7:
                    return f"time_limit_{holding_days}days"
            
            elif position.strategy == 'us':
                # 미국: 2-4주 홀딩
                if holding_days >= 28:
                    return f"time_limit_{holding_days}days"
            
            # 큰 손실 방지 (-10%)
            if pnl_ratio <= -0.10:
                return f"emergency_stop_{pnl_ratio*100:.1f}%"
            
            return None
            
        except Exception as e:
            logging.error(f"청산 조건 체크 실패: {e}")
            return None
    
    async def _execute_exit_order(self, position: TradingPosition, reason: str):
        """청산 주문 실행"""
        try:
            # 매도 주문 생성
            order = TradingOrder(
                id=str(uuid.uuid4()),
                symbol=position.symbol,
                market=position.market,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=abs(position.quantity),
                strategy=position.strategy
            )
            
            # 주문 실행
            success = await self._execute_order(order)
            
            if success:
                # 실현 손익 계산
                realized_pnl = position.unrealized_pnl
                pnl_ratio = (position.current_price - position.avg_cost) / position.avg_cost * 100
                
                # 포지션 제거
                self.position_manager.remove_position(position.id)
                
                # 알림 전송
                if self.notifier:
                    await self.notifier.notify_system_event(
                        "포지션 청산",
                        f"{position.symbol} 청산: {realized_pnl:+,.0f}원 ({pnl_ratio:+.1f}%) - {reason}",
                        "info" if realized_pnl >= 0 else "warning"
                    )
                
                logging.info(f"🔚 포지션 청산: {position.symbol} {realized_pnl:+,.0f}원 ({pnl_ratio:+.1f}%) - {reason}")
            
        except Exception as e:
            logging.error(f"청산 주문 실행 실패: {e}")
    
    async def run_daily_strategy(self):
        """일일 전략 실행"""
        try:
            today = datetime.now()
            weekday = today.weekday()  # 0=월요일, 6=일요일
            
            logging.info(f"📅 일일 전략 실행: {today.strftime('%Y-%m-%d %A')}")
            
            results = {}
            
            # 미국 전략 (매일)
            if self.config.us_enabled and self.us_strategy:
                try:
                    signals = await self.us_strategy.scan_all_stocks()
                    if signals:
                        results['us'] = await self.execute_strategy_signals('us', signals)
                except Exception as e:
                    logging.error(f"미국 전략 실행 실패: {e}")
            
            # 일본 전략 (화요일, 목요일)
            if self.config.jp_enabled and self.jp_strategy and weekday in [1, 3]:
                try:
                    signals = await self.jp_strategy.hunt_and_analyze()
                    if signals:
                        results['japan'] = await self.execute_strategy_signals('japan', signals)
                except Exception as e:
                    logging.error(f"일본 전략 실행 실패: {e}")
            
            # 인도 전략 (수요일)
            if self.config.in_enabled and self.in_strategy and weekday == 2:
                try:
                    sample_df = self.in_strategy.create_sample_data()
                    strategy_result = self.in_strategy.run_strategy(sample_df, enable_trading=True)
                    
                    if strategy_result.get('selected_stocks') is not None:
                        # 선별된 종목을 신호로 변환
                        signals = self._convert_india_signals(strategy_result['selected_stocks'])
                        if signals:
                            results['india'] = await self.execute_strategy_signals('india', signals)
                except Exception as e:
                    logging.error(f"인도 전략 실행 실패: {e}")
            
            # 암호화폐 전략 (월요일, 금요일)
            if self.config.crypto_enabled and self.crypto_strategy and weekday in [0, 4]:
                try:
                    signals = await self.crypto_strategy.execute_legendary_strategy()
                    if signals:
                        results['crypto'] = await self.execute_strategy_signals('crypto', signals)
                except Exception as e:
                    logging.error(f"암호화폐 전략 실행 실패: {e}")
            
            # 포지션 모니터링 (매일)
            await self.monitor_positions()
            
            # 일일 리포트
            await self._generate_daily_report(results)
            
            return results
            
        except Exception as e:
            logging.error(f"일일 전략 실행 실패: {e}")
            return {}
    
    def _convert_india_signals(self, selected_stocks) -> List:
        """인도 선별 종목을 신호로 변환"""
        signals = []
        try:
            if hasattr(selected_stocks, 'iterrows'):
                for _, stock in selected_stocks.iterrows():
                    signal = type('Signal', (), {
                        'symbol': stock.get('ticker', ''),
                        'action': 'buy',
                        'confidence': stock.get('final_score', 0) / 100,
                        'price': stock.get('close', 0),
                        'target_price': stock.get('conservative_take_profit', 0),
                        'stop_loss': stock.get('conservative_stop_loss', 0),
                        'reasoning': '인도 안정형 전략'
                    })()
                    signals.append(signal)
        except Exception as e:
            logging.error(f"인도 신호 변환 실패: {e}")
        
        return signals
    
    async def _generate_daily_report(self, results: Dict):
        """일일 리포트 생성"""
        try:
            total_orders = sum(r.get('executed_orders', 0) for r in results.values())
            total_positions = len(self.position_manager.positions)
            total_pnl = self.position_manager.calculate_unrealized_pnl()
            
            report = f"📊 일일 거래 리포트\n"
            report += f"{'='*30}\n"
            report += f"📅 일자: {datetime.now().strftime('%Y-%m-%d')}\n"
            report += f"📈 신규 주문: {total_orders}개\n"
            report += f"💼 활성 포지션: {total_positions}개\n"
            report += f"💰 미실현 손익: {total_pnl:+,.0f}원\n\n"
            
            # 전략별 결과
            for strategy, result in results.items():
                emoji = {'us': '🇺🇸', 'japan': '🇯🇵', 'india': '🇮🇳', 'crypto': '🪙'}.get(strategy, '📊')
                report += f"{emoji} {strategy.upper()}: {result.get('executed_orders', 0)}개 주문 실행\n"
            
            # 알림 전송
            if self.notifier:
                await self.notifier.notify_system_event("일일 리포트", report, "info")
            
            logging.info(f"📊 일일 리포트: {total_orders}개 주문, {total_positions}개 포지션, {total_pnl:+,.0f}원")
            
        except Exception as e:
            logging.error(f"일일 리포트 생성 실패: {e}")
    
    async def start_automated_trading(self):
        """자동화 거래 시작"""
        try:
            if not self.config.auto_trading:
                logging.warning("⚠️ 자동 거래가 비활성화되어 있습니다")
                return
            
            logging.info("🤖 자동화 거래 시스템 시작")
            self.running = True
            
            # 스케줄 설정
            schedule.every().day.at("09:00").do(lambda: asyncio.create_task(self.run_daily_strategy()))
            schedule.every(15).minutes.do(lambda: asyncio.create_task(self.monitor_positions()))
            schedule.every().day.at("16:00").do(lambda: asyncio.create_task(self._generate_daily_report({})))
            
            while self.running:
                try:
                    schedule.run_pending()
                    await asyncio.sleep(60)  # 1분마다 체크
                    
                except KeyboardInterrupt:
                    logging.info("🛑 사용자 중단 요청")
                    break
                except Exception as e:
                    logging.error(f"자동화 루프 오류: {e}")
                    await asyncio.sleep(300)  # 5분 후 재시도
            
        except Exception as e:
            logging.error(f"자동화 거래 실행 실패: {e}")
        finally:
            self.running = False
            await self.shutdown()
    
    def stop_automated_trading(self):
        """자동화 거래 중지"""
        self.running = False
        logging.info("⏹️ 자동화 거래 중지 요청")
    
    async def shutdown(self):
        """시스템 종료"""
        try:
            logging.info("🔌 거래 시스템 종료 중...")
            
            # 연결 해제
            await self.ibkr.disconnect()
            
            # 알림 시스템 정리
            if self.notifier:
                self.notifier.cleanup()
            
            logging.info("✅ 거래 시스템 종료 완료")
            
        except Exception as e:
            logging.error(f"시스템 종료 오류: {e}")
    
    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        try:
            total_positions = len(self.position_manager.positions)
            total_exposure = self.position_manager.calculate_total_exposure()
            total_pnl = self.position_manager.calculate_unrealized_pnl()
            
            # 전략별 포지션 수
            strategy_positions = {}
            for strategy in ['us', 'japan', 'india', 'crypto']:
                positions = self.position_manager.get_positions_by_strategy(strategy)
                strategy_positions[strategy] = len(positions)
            
            return {
                'running': self.running,
                'demo_mode': self.config.demo_mode,
                'auto_trading': self.config.auto_trading,
                'total_positions': total_positions,
                'total_exposure': total_exposure,
                'total_pnl': total_pnl,
                'strategy_positions': strategy_positions,
                'ibkr_connected': self.ibkr.connected,
                'upbit_connected': self.upbit.connected,
                'last_execution': max(self.last_execution_time.values()) if self.last_execution_time else None
            }
            
        except Exception as e:
            logging.error(f"시스템 상태 조회 실패: {e}")
            return {'error': str(e)}

# ========================================================================================
# 🎮 편의 함수들
# ========================================================================================

async def quick_trading_test():
    """빠른 거래 테스트"""
    print("🧪 거래 시스템 테스트")
    print("=" * 50)
    
    # 설정
    config = TradingConfig(
        demo_mode=True,
        auto_trading=False,
        total_capital=100_000_000  # 1억원
    )
    
    # 거래 시스템 초기화
    executor = TradingExecutor(config)
    
    try:
        # 초기화
        if await executor.initialize():
            print("✅ 거래 시스템 초기화 성공")
            
            # 시스템 상태
            status = executor.get_system_status()
            print(f"📊 현재 포지션: {status['total_positions']}개")
            print(f"💰 총 노출: {status['total_exposure']:,.0f}원")
            print(f"📈 미실현 손익: {status['total_pnl']:+,.0f}원")
            
            # 포지션 모니터링 테스트
            print("\n🔍 포지션 모니터링 테스트...")
            await executor.monitor_positions()
            
        else:
            print("❌ 거래 시스템 초기화 실패")
            
    finally:
        await executor.shutdown()

def create_trading_config_file():
    """거래 설정 파일 생성"""
    config_data = {
        "trading": {
            "demo_mode": True,
            "auto_trading": False,
            "total_capital": 1000000000,
            "max_daily_loss": 0.02,
            "position_size_limit": 0.15
        },
        "strategies": {
            "us_enabled": True,
            "jp_enabled": True,
            "in_enabled": True,
            "crypto_enabled": True
        },
        "risk_management": {
            "stop_loss_enabled": True,
            "take_profit_enabled": True,
            "trailing_stop_enabled": True
        },
        "notifications": {
            "enabled": True,
            "telegram_alerts": True
        }
    }
    
    try:
        with open('trading_config.json', 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        print("✅ 거래 설정 파일 생성: trading_config.json")
    except Exception as e:
        print(f"❌ 설정 파일 생성 실패: {e}")

def create_trading_env_file():
    """거래용 환경변수 파일 생성"""
    env_content = """# QuintProject Trading 환경변수 설정

# IBKR 설정 (주식)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
IBKR_ACCOUNT_ID=your_ibkr_account_id

# Upbit 설정 (암호화폐)
UPBIT_ACCESS_KEY=your_upbit_access_key
UPBIT_SECRET_KEY=your_upbit_secret_key

# 텔레그램 알림
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# 거래 설정
DEMO_MODE=true
AUTO_TRADING=false
TOTAL_CAPITAL=1000000000
"""
    
    try:
        with open('.env.trading', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ 거래 환경변수 파일 생성: .env.trading")
        print("\n📋 다음 단계:")
        print("1. IBKR TWS/Gateway 실행")
        print("2. Upbit API 키 발급")
        print("3. .env.trading 파일 편집")
        print("4. python trading.py test 실행")
    except Exception as e:
        print(f"❌ 환경변수 파일 생성 실패: {e}")

def print_trading_help():
    """거래 시스템 도움말"""
    help_text = """
🏆 QuintProject Trading System v1.0
===================================

📋 주요 기능:
  - 🇺🇸 미국 주식 자동매매 (IBKR)
  - 🇯🇵 일본 주식 화목 전략 (IBKR)
  - 🇮🇳 인도 주식 수요일 전략 (IBKR)
  - 🪙 암호화폐 월금 전략 (업비트)
  - 🛡️ 통합 리스크 관리
  - 📊 실시간 포지션 모니터링

🚀 명령어:
  python trading.py test         # 거래 테스트
  python trading.py run          # 일일 전략 실행
  python trading.py auto         # 자동화 거래 시작
  python trading.py status       # 시스템 상태
  python trading.py positions    # 포지션 조회
  python trading.py setup        # 초기 설정

🔧 필수 설정:
  - IBKR TWS/Gateway 실행 필요
  - 업비트 API 키 설정
  - 텔레그램 봇 설정 (알림용)

📊 전략 일정:
  - 월요일: 🇺🇸 미국 + 🪙 암호화폐
  - 화요일: 🇺🇸 미국 + 🇯🇵 일본
  - 수요일: 🇺🇸 미국 + 🇮🇳 인도
  - 목요일: 🇺🇸 미국 + 🇯🇵 일본
  - 금요일: 🇺🇸 미국 + 🪙 암호화폐

⚠️ 주의사항:
  - 데모 모드로 충분히 테스트 후 실거래
  - 리스크 관리 설정 필수
  - 정기적인 포지션 모니터링
"""
    print(help_text)

# ========================================================================================
# 🏁 메인 실행부
# ========================================================================================

async def main():
    """메인 실행 함수"""
    print("🏆" + "=" * 70)
    print("🔥 전설적 퀸트프로젝트 TRADING 시스템")
    print("🤖 자동화 거래 + 포지션 관리 + 리스크 제어")
    print("=" * 72)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            print("🧪 거래 시스템 테스트 시작...")
            await quick_trading_test()
            
        elif command == 'run':
            print("🚀 일일 전략 실행...")
            config = TradingConfig(demo_mode=True)
            executor = TradingExecutor(config)
            
            try:
                if await executor.initialize():
                    results = await executor.run_daily_strategy()
                    
                    total_orders = sum(r.get('executed_orders', 0) for r in results.values())
                    print(f"✅ 일일 전략 완료: {total_orders}개 주문 실행")
                    
                    for strategy, result in results.items():
                        emoji = {'us': '🇺🇸', 'japan': '🇯🇵', 'india': '🇮🇳', 'crypto': '🪙'}.get(strategy, '📊')
                        print(f"  {emoji} {strategy.upper()}: {result.get('executed_orders', 0)}개")
                else:
                    print("❌ 시스템 초기화 실패")
            finally:
                await executor.shutdown()
                
        elif command == 'auto':
            print("🤖 자동화 거래 시작...")
            print("⚠️ 실제 자금이 사용될 수 있습니다!")
            
            mode = input("모드 선택 - [d]emo / [l]ive: ").lower()
            demo_mode = mode != 'l'
            
            if not demo_mode:
                confirm = input("실거래 모드로 진행하시겠습니까? (YES 입력): ")
                if confirm != 'YES':
                    print("❌ 취소되었습니다")
                    return
            
            config = TradingConfig(
                demo_mode=demo_mode,
                auto_trading=True
            )
            
            executor = TradingExecutor(config)
            
            try:
                if await executor.initialize():
                    print(f"✅ {'시뮬레이션' if demo_mode else '실거래'} 모드로 자동화 시작")
                    print("Ctrl+C로 중지할 수 있습니다.")
                    await executor.start_automated_trading()
                else:
                    print("❌ 시스템 초기화 실패")
            except KeyboardInterrupt:
                print("\n⏹️ 자동화 거래 중지")
                executor.stop_automated_trading()
            finally:
                await executor.shutdown()
                
        elif command == 'status':
            print("📊 시스템 상태 조회...")
            config = TradingConfig()
            executor = TradingExecutor(config)
            
            try:
                if await executor.initialize():
                    status = executor.get_system_status()
                    
                    print(f"\n🔍 거래 시스템 현황")
                    print("=" * 40)
                    print(f"🔄 실행 상태: {'실행중' if status['running'] else '중지'}")
                    print(f"🎮 모드: {'시뮬레이션' if status['demo_mode'] else '실거래'}")
                    print(f"🤖 자동거래: {'활성화' if status['auto_trading'] else '비활성화'}")
                    print(f"💼 총 포지션: {status['total_positions']}개")
                    print(f"💰 총 노출: {status['total_exposure']:,.0f}원")
                    print(f"📈 미실현 손익: {status['total_pnl']:+,.0f}원")
                    
                    print(f"\n📊 전략별 포지션:")
                    for strategy, count in status['strategy_positions'].items():
                        emoji = {'us': '🇺🇸', 'japan': '🇯🇵', 'india': '🇮🇳', 'crypto': '🪙'}.get(strategy, '📊')
                        print(f"  {emoji} {strategy.upper()}: {count}개")
                    
                    print(f"\n🔗 연결 상태:")
                    print(f"  📈 IBKR: {'✅ 연결됨' if status['ibkr_connected'] else '❌ 연결 안됨'}")
                    print(f"  🪙 업비트: {'✅ 연결됨' if status['upbit_connected'] else '❌ 연결 안됨'}")
                    
                else:
                    print("❌ 시스템 상태 조회 실패")
            finally:
                await executor.shutdown()
                
        elif command == 'positions':
            print("💼 포지션 조회...")
            config = TradingConfig()
            executor = TradingExecutor(config)
            
            try:
                if await executor.initialize():
                    positions = executor.position_manager.positions
                    
                    if not positions:
                        print("📭 보유 포지션이 없습니다")
                    else:
                        print(f"\n💼 보유 포지션 ({len(positions)}개)")
                        print("=" * 80)
                        
                        for pos in positions.values():
                            pnl_pct = (pos.current_price - pos.avg_cost) / pos.avg_cost * 100
                            emoji = "📈" if pnl_pct >= 0 else "📉"
                            
                            market_emoji = {
                                MarketType.US_STOCK: '🇺🇸',
                                MarketType.JP_STOCK: '🇯🇵', 
                                MarketType.IN_STOCK: '🇮🇳',
                                MarketType.CRYPTO: '🪙'
                            }.get(pos.market, '📊')
                            
                            print(f"{market_emoji} {pos.symbol} | {pos.strategy.upper()}")
                            print(f"  💰 수량: {pos.quantity:,.2f} | 평단: {pos.avg_cost:,.0f} | 현재: {pos.current_price:,.0f}")
                            print(f"  {emoji} 손익: {pos.unrealized_pnl:+,.0f}원 ({pnl_pct:+.1f}%)")
                            print(f"  📅 진입: {pos.entry_date.strftime('%Y-%m-%d')}")
                            print("-" * 80)
                else:
                    print("❌ 포지션 조회 실패")
            finally:
                await executor.shutdown()
                
        elif command == 'setup':
            print("🔧 초기 설정 시작...")
            create_trading_config_file()
            create_trading_env_file()
            
        elif command == 'help' or command == '--help':
            print_trading_help()
            
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print("사용법: python trading.py [test|run|auto|status|positions|setup|help]")
    
    else:
        # 인터랙티브 모드
        print("\n🚀 거래 시스템 옵션:")
        print("  1. 거래 시스템 테스트")
        print("  2. 일일 전략 실행")
        print("  3. 자동화 거래 시작")
        print("  4. 시스템 상태 조회")
        print("  5. 포지션 조회")
        print("  6. 설정 파일 생성")
        print("  7. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (1-7): ").strip()
                
                if choice == '1':
                    print("\n🧪 거래 시스템 테스트...")
                    await quick_trading_test()
                
                elif choice == '2':
                    print("\n🚀 일일 전략 실행...")
                    
                    mode = input("모드 선택 - [d]emo / [l]ive (기본값: demo): ").lower() or 'd'
                    demo_mode = mode != 'l'
                    
                    config = TradingConfig(demo_mode=demo_mode)
                    executor = TradingExecutor(config)
                    
                    try:
                        if await executor.initialize():
                            print(f"✅ {'시뮬레이션' if demo_mode else '실거래'} 모드로 전략 실행")
                            
                            results = await executor.run_daily_strategy()
                            
                            total_orders = sum(r.get('executed_orders', 0) for r in results.values())
                            print(f"\n📊 실행 결과: 총 {total_orders}개 주문")
                            
                            for strategy, result in results.items():
                                emoji = {'us': '🇺🇸', 'japan': '🇯🇵', 'india': '🇮🇳', 'crypto': '🪙'}.get(strategy, '📊')
                                executed = result.get('executed_orders', 0)
                                failed = result.get('failed_orders', 0)
                                print(f"  {emoji} {strategy.upper()}: {executed}개 성공, {failed}개 실패")
                        else:
                            print("❌ 시스템 초기화 실패")
                    finally:
                        await executor.shutdown()
                
                elif choice == '3':
                    print("\n🤖 자동화 거래 시작...")
                    
                    mode = input("모드 선택 - [d]emo / [l]ive: ").lower()
                    demo_mode = mode != 'l'
                    
                    if not demo_mode:
                        print("⚠️ 실거래 모드 선택됨!")
                        confirm = input("정말로 실거래를 시작하시겠습니까? (YES 입력): ")
                        if confirm != 'YES':
                            print("❌ 취소되었습니다")
                            continue
                    
                    config = TradingConfig(
                        demo_mode=demo_mode,
                        auto_trading=True
                    )
                    
                    executor = TradingExecutor(config)
                    
                    try:
                        if await executor.initialize():
                            print(f"🚀 {'시뮬레이션' if demo_mode else '실거래'} 자동화 시작")
                            print("📋 스케줄:")
                            print("  - 09:00: 일일 전략 실행")
                            print("  - 15분마다: 포지션 모니터링")
                            print("  - 16:00: 일일 리포트")
                            print("\nCtrl+C로 중지할 수 있습니다.")
                            
                            await executor.start_automated_trading()
                        else:
                            print("❌ 시스템 초기화 실패")
                    except KeyboardInterrupt:
                        print("\n⏹️ 자동화 거래 중지")
                        executor.stop_automated_trading()
                    finally:
                        await executor.shutdown()
                        break
                
                elif choice == '4':
                    print("\n📊 시스템 상태 조회...")
                    config = TradingConfig()
                    executor = TradingExecutor(config)
                    
                    try:
                        if await executor.initialize():
                            status = executor.get_system_status()
                            
                            print(f"\n🎯 거래 시스템 종합 현황")
                            print("=" * 50)
                            print(f"🔄 실행: {'실행중' if status['running'] else '중지'}")
                            print(f"🎮 모드: {'시뮬레이션' if status['demo_mode'] else '실거래'}")
                            print(f"🤖 자동: {'활성화' if status['auto_trading'] else '비활성화'}")
                            print(f"💼 포지션: {status['total_positions']}개")
                            print(f"💰 노출: {status['total_exposure']:,.0f}원")
                            print(f"📈 손익: {status['total_pnl']:+,.0f}원")
                            
                            print(f"\n📊 전략별 현황:")
                            strategies = {'us': '🇺🇸 미국', 'japan': '🇯🇵 일본', 'india': '🇮🇳 인도', 'crypto': '🪙 암호화폐'}
                            for strategy, name in strategies.items():
                                count = status['strategy_positions'].get(strategy, 0)
                                enabled = getattr(executor.config, f'{strategy}_enabled', False)
                                status_text = f"{count}개 포지션" if enabled else "비활성화"
                                print(f"  {name}: {status_text}")
                            
                            print(f"\n🔗 거래소 연결:")
                            print(f"  📈 IBKR: {'✅ 연결됨' if status['ibkr_connected'] else '❌ 연결 안됨'}")
                            print(f"  🪙 업비트: {'✅ 연결됨' if status['upbit_connected'] else '❌ 연결 안됨'}")
                        else:
                            print("❌ 상태 조회 실패")
                    finally:
                        await executor.shutdown()
                
                elif choice == '5':
                    print("\n💼 포지션 상세 조회...")
                    config = TradingConfig()
                    executor = TradingExecutor(config)
                    
                    try:
                        if await executor.initialize():
                            positions = executor.position_manager.positions
                            
                            if not positions:
                                print("📭 현재 보유 포지션이 없습니다")
                            else:
                                print(f"\n💼 포지션 상세 ({len(positions)}개)")
                                print("=" * 90)
                                
                                total_pnl = 0
                                for i, pos in enumerate(positions.values(), 1):
                                    pnl_pct = (pos.current_price - pos.avg_cost) / pos.avg_cost * 100
                                    total_pnl += pos.unrealized_pnl
                                    
                                    market_emoji = {
                                        MarketType.US_STOCK: '🇺🇸',
                                        MarketType.JP_STOCK: '🇯🇵', 
                                        MarketType.IN_STOCK: '🇮🇳',
                                        MarketType.CRYPTO: '🪙'
                                    }.get(pos.market, '📊')
                                    
                                    pnl_emoji = "📈" if pnl_pct >= 0 else "📉"
                                    holding_days = (datetime.now() - pos.entry_date).days
                                    
                                    print(f"{i:2d}. {market_emoji} {pos.symbol} ({pos.strategy.upper()})")
                                    print(f"    💰 수량: {pos.quantity:,.2f} | 평단가: {pos.avg_cost:,.0f}")
                                    print(f"    📊 현재가: {pos.current_price:,.0f} | 보유일: {holding_days}일")
                                    print(f"    {pnl_emoji} 손익: {pos.unrealized_pnl:+,.0f}원 ({pnl_pct:+.1f}%)")
                                    
                                    if pos.stop_loss:
                                        print(f"    🛑 손절가: {pos.stop_loss:,.0f}")
                                    if pos.take_profit:
                                        print(f"    🎯 익절가: {pos.take_profit:,.0f}")
                                        
                                    print("-" * 90)
                                
                                print(f"💰 총 미실현 손익: {total_pnl:+,.0f}원")
                        else:
                            print("❌ 포지션 조회 실패")
                    finally:
                        await executor.shutdown()
                
                elif choice == '6':
                    print("\n🔧 설정 파일 생성...")
                    create_trading_config_file()
                    create_trading_env_file()
                    print("\n✅ 설정 파일 생성 완료!")
                    print("📋 다음 단계:")
                    print("1. .env.trading 파일 편집 (API 키 설정)")
                    print("2. trading_config.json 파일 편집 (거래 설정)")
                    print("3. IBKR TWS/Gateway 실행")
                    print("4. python trading.py test 로 테스트")
                
                elif choice == '7':
                    print("👋 거래 시스템을 종료합니다!")
                    break
                
                else:
                    print("❌ 잘못된 선택입니다. 1-7 중 선택하세요.")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def setup_logging():
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'trading.log', encoding='utf-8'),
            logging.FileHandler(log_dir / f'trading_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
        ]
    )

def check_dependencies():
    """의존성 체크"""
    required_modules = ['core', 'us_strategy', 'jp_strategy', 'inda_strategy', 'coin_strategy', 'notifier']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ 누락된 모듈: {', '.join(missing_modules)}")
        print("📋 필요한 파일들이 같은 폴더에 있는지 확인하세요:")
        for module in required_modules:
            print(f"  - {module}.py")
        return False
    
    print("✅ 모든 전략 모듈이 준비되었습니다")
    return True

# ========================================================================================
# 🔄 스케줄러 유틸리티
# ========================================================================================

class TradingScheduler:
    """거래 스케줄러"""
    
    @staticmethod
    def should_trade_today(strategy: str) -> bool:
        """오늘 거래 가능 여부"""
        today = datetime.now()
        weekday = today.weekday()  # 0=월요일
        
        # 주말 제외
        if weekday >= 5:
            return False
        
        if strategy == 'us':
            return True  # 매일
        elif strategy == 'japan':
            return weekday in [1, 3]  # 화, 목
        elif strategy == 'india':
            return weekday == 2  # 수
        elif strategy == 'crypto':
            return weekday in [0, 4]  # 월, 금
        
        return False
    
    @staticmethod
    def get_trading_schedule() -> Dict:
        """거래 스케줄 조회"""
        schedule = {}
        days = ['월', '화', '수', '목', '금', '토', '일']
        
        for i in range(7):
            day_strategies = []
            
            if i < 5:  # 평일만
                if TradingScheduler.should_trade_today('us'):
                    day_strategies.append('🇺🇸 미국')
                
                if i in [1, 3]:  # 화, 목
                    day_strategies.append('🇯🇵 일본')
                
                if i == 2:  # 수
                    day_strategies.append('🇮🇳 인도')
                
                if i in [0, 4]:  # 월, 금
                    day_strategies.append('🪙 암호화폐')
            
            schedule[days[i]] = day_strategies
        
        return schedule

# ========================================================================================
# 🏁 최종 실행부
# ========================================================================================

if __name__ == "__main__":
    try:
        # 로깅 설정
        setup_logging()
        
        # 의존성 체크
        if not check_dependencies():
            print("\n⚠️ 전략 모듈들을 먼저 준비해주세요.")
            sys.exit(1)
        
        # 메인 실행
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 거래 시스템이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        logging.error(f"실행 오류: {e}")

# ========================================================================================
# 🏁 최종 익스포트
# ========================================================================================

__all__ = [
    'TradingConfig',
    'TradingOrder', 
    'TradingPosition',
    'TradingPerformance',
    'IBKRTradingInterface',
    'UpbitTradingInterface',
    'PositionManager',
    'TradingExecutor',
    'TradingScheduler',
    'MarketType',
    'OrderType',
    'OrderSide',
    'quick_trading_test'
]

"""
🏆 QuintProject Trading System 사용 예시:

# 1. 기본 사용
from trading import TradingExecutor, TradingConfig

config = TradingConfig(demo_mode=True)
executor = TradingExecutor(config)

# 2. 일일 전략 실행
await executor.initialize()
results = await executor.run_daily_strategy()

# 3. 자동화 거래
config.auto_trading = True
await executor.start_automated_trading()

# 4. 포지션 모니터링
await executor.monitor_positions()

# 5. 시스템 상태
status = executor.get_system_status()

# 6. 빠른 테스트
python trading.py test

# 7. 자동화 실행
python trading.py auto

# 8. 포지션 조회
python trading.py positions
"""
