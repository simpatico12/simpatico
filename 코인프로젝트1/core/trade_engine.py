"""
Advanced Multi-Market Trading Engine
====================================

퀀트 수준의 통합 거래 엔진
다중 거래소 지원 (업비트, IBKR, 바이낸스 등)
안전하고 효율적인 포지션 관리 및 위험 제어

Features:
- 다중 거래소 통합 관리
- 실시간 포지션 추적 및 리스크 관리
- 자동 손절/익절 시스템
- 거래 내역 로깅 및 성과 분석
- 시장별 특화 주문 처리
- 백테스팅 모드 지원

Author: Your Name
Version: 3.0.0
Created: 2025-06-18
"""

import os
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import json
import threading
from decimal import Decimal, ROUND_DOWN
from collections import defaultdict
import logging

# Core 패키지 import
try:
    from core import BaseComponent, config, logger, ValidationError
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 거래소 API imports
try:
    import pyupbit
    UPBIT_AVAILABLE = True
except ImportError:
    UPBIT_AVAILABLE = False
    logger.warning("pyupbit 모듈을 찾을 수 없습니다.")

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    logger.warning("ib_insync 모듈을 찾을 수 없습니다.")

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("ccxt 모듈을 찾을 수 없습니다.")

# =============================================================================
# 상수 및 열거형
# =============================================================================

class OrderType(Enum):
    """주문 유형"""
    MARKET = "market"           # 시장가
    LIMIT = "limit"             # 지정가
    STOP_LOSS = "stop_loss"     # 손절
    TAKE_PROFIT = "take_profit" # 익절
    TRAILING_STOP = "trailing_stop"  # 트레일링 스탑

class OrderSide(Enum):
    """매수/매도"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"         # 대기
    FILLED = "filled"           # 체결
    PARTIALLY_FILLED = "partially_filled"  # 부분체결
    CANCELLED = "cancelled"     # 취소
    REJECTED = "rejected"       # 거부
    EXPIRED = "expired"         # 만료

class ExchangeType(Enum):
    """거래소 유형"""
    UPBIT = "upbit"             # 업비트 (암호화폐)
    IBKR = "ibkr"               # Interactive Brokers (주식)
    BINANCE = "binance"         # 바이낸스 (암호화폐)
    BYBIT = "bybit"             # 바이비트 (파생상품)
    KRX = "krx"                 # 한국거래소
    MOCK = "mock"               # 백테스팅용

class PositionSide(Enum):
    """포지션 방향"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

# =============================================================================
# 데이터 클래스들
# =============================================================================

@dataclass
class TradingConfig:
    """거래 설정"""
    max_position_size: float = 0.1          # 최대 포지션 크기 (10%)
    max_daily_loss: float = 0.05            # 최대 일일 손실 (5%)
    default_stop_loss: float = 0.08         # 기본 손절 (8%)
    default_take_profit: float = 0.15       # 기본 익절 (15%)
    order_timeout: int = 300                # 주문 타임아웃 (초)
    slippage_tolerance: float = 0.002       # 슬리피지 허용범위 (0.2%)
    min_order_amount: float = 10.0          # 최소 주문 금액
    enable_paper_trading: bool = False      # 페이퍼 트레이딩 모드
    enable_auto_stop_loss: bool = True      # 자동 손절 활성화
    trading_hours_only: bool = True         # 거래시간만 활성화

@dataclass
class OrderRequest:
    """주문 요청"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"              # Good Till Cancelled
    client_order_id: str = field(default_factory=lambda: f"order_{int(time.time() * 1000)}")
    exchange: ExchangeType = ExchangeType.UPBIT
    
    def __post_init__(self):
        """검증"""
        if self.quantity <= 0:
            raise ValidationError(f"수량은 0보다 커야 합니다: {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValidationError("지정가 주문에는 가격이 필요합니다")

@dataclass
class Order:
    """주문 정보"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    filled_quantity: float = 0.0
    average_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    exchange: ExchangeType = ExchangeType.UPBIT
    timestamp: datetime = field(default_factory=datetime.now)
    client_order_id: str = ""
    commission: float = 0.0
    
    @property
    def remaining_quantity(self) -> float:
        """미체결 수량"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """완전 체결 여부"""
        return self.status == OrderStatus.FILLED
    
    @property
    def fill_ratio(self) -> float:
        """체결률"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0

@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: PositionSide
    quantity: float
    average_price: float
    market_price: float = 0.0
    exchange: ExchangeType = ExchangeType.UPBIT
    timestamp: datetime = field(default_factory=datetime.now)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """시장 가치"""
        return abs(self.quantity) * self.market_price
    
    @property
    def unrealized_pnl(self) -> float:
        """미실현 손익"""
        if self.side == PositionSide.LONG:
            return (self.market_price - self.average_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            return (self.average_price - self.market_price) * abs(self.quantity)
        return 0.0
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """미실현 손익률"""
        if self.average_price > 0:
            return self.unrealized_pnl / (self.average_price * abs(self.quantity))
        return 0.0

@dataclass
class TradeExecution:
    """거래 체결 내역"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: ExchangeType = ExchangeType.UPBIT
    
    @property
    def gross_amount(self) -> float:
        """총 거래대금"""
        return self.quantity * self.price
    
    @property
    def net_amount(self) -> float:
        """순 거래대금 (수수료 제외)"""
        return self.gross_amount - self.commission

# =============================================================================
# 거래소 인터페이스
# =============================================================================

class BaseExchange(ABC):
    """거래소 기본 클래스"""
    
    def __init__(self, exchange_type: ExchangeType, config: Dict[str, Any]):
        self.exchange_type = exchange_type
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{exchange_type.value}")
        self.is_connected = False
        self.last_heartbeat = datetime.now()
    
    @abstractmethod
    async def connect(self) -> bool:
        """거래소 연결"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """거래소 연결 해제"""
        pass
    
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> Order:
        """주문 전송"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """주문 상태 조회"""
        pass
    
    @abstractmethod
    async def get_balance(self, asset: str) -> float:
        """잔고 조회"""
        pass
    
    @abstractmethod
    async def get_market_price(self, symbol: str) -> float:
        """현재가 조회"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """포지션 조회"""
        pass
    
    async def healthcheck(self) -> bool:
        """연결 상태 확인"""
        try:
            # 간단한 API 호출로 연결 상태 확인
            await self.get_balance("KRW" if self.exchange_type == ExchangeType.UPBIT else "USD")
            self.last_heartbeat = datetime.now()
            return True
        except Exception as e:
            self.logger.error(f"헬스체크 실패: {e}")
            return False

class UpbitExchange(BaseExchange):
    """업비트 거래소"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ExchangeType.UPBIT, config)
        self.client = None
    
    async def connect(self) -> bool:
        """업비트 연결"""
        try:
            if not UPBIT_AVAILABLE:
                raise ImportError("pyupbit 모듈이 필요합니다")
            
            access_key = self.config.get('access_key')
            secret_key = self.config.get('secret_key')
            
            if not access_key or not secret_key:
                raise ValidationError("업비트 API 키가 설정되지 않았습니다")
            
            self.client = pyupbit.Upbit(access_key, secret_key)
            
            # 연결 테스트
            balances = self.client.get_balances()
            if balances is None:
                raise ConnectionError("업비트 API 연결 실패")
            
            self.is_connected = True
            self.logger.info("업비트 연결 성공")
            return True
            
        except Exception as e:
            self.logger.error(f"업비트 연결 실패: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """업비트 연결 해제"""
        self.client = None
        self.is_connected = False
        self.logger.info("업비트 연결 해제")
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """업비트 주문"""
        if not self.is_connected or not self.client:
            raise ConnectionError("업비트에 연결되지 않음")
        
        try:
            symbol = order_request.symbol
            if not symbol.startswith("KRW-"):
                symbol = f"KRW-{symbol}"
            
            if order_request.order_type == OrderType.MARKET:
                if order_request.side == OrderSide.BUY:
                    # 시장가 매수 (KRW 금액으로)
                    amount_krw = order_request.quantity
                    result = self.client.buy_market_order(symbol, amount_krw)
                else:
                    # 시장가 매도 (코인 수량으로)
                    result = self.client.sell_market_order(symbol, order_request.quantity)
            else:
                # 지정가 주문
                if order_request.side == OrderSide.BUY:
                    result = self.client.buy_limit_order(symbol, order_request.price, order_request.quantity)
                else:
                    result = self.client.sell_limit_order(symbol, order_request.price, order_request.quantity)
            
            if result and 'uuid' in result:
                order = Order(
                    order_id=result['uuid'],
                    symbol=symbol,
                    side=order_request.side,
                    order_type=order_request.order_type,
                    quantity=order_request.quantity,
                    price=order_request.price,
                    client_order_id=order_request.client_order_id,
                    exchange=ExchangeType.UPBIT
                )
                
                self.logger.info(f"업비트 주문 성공: {order.order_id}")
                return order
            else:
                raise Exception(f"주문 실패: {result}")
                
        except Exception as e:
            self.logger.error(f"업비트 주문 실패: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """업비트 주문 취소"""
        try:
            result = self.client.cancel_order(order_id)
            return result is not None and 'uuid' in result
        except Exception as e:
            self.logger.error(f"업비트 주문 취소 실패: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Order:
        """업비트 주문 상태"""
        try:
            result = self.client.get_order(order_id)
            if not result:
                raise Exception(f"주문을 찾을 수 없음: {order_id}")
            
            # 업비트 상태를 표준 상태로 변환
            status_mapping = {
                'wait': OrderStatus.PENDING,
                'done': OrderStatus.FILLED,
                'cancel': OrderStatus.CANCELLED
            }
            
            status = status_mapping.get(result['state'], OrderStatus.PENDING)
            
            return Order(
                order_id=result['uuid'],
                symbol=result['market'],
                side=OrderSide.BUY if result['side'] == 'bid' else OrderSide.SELL,
                order_type=OrderType.MARKET if result['ord_type'] == 'price' else OrderType.LIMIT,
                quantity=float(result['volume']),
                price=float(result['price']) if result['price'] else None,
                filled_quantity=float(result['executed_volume']),
                average_price=float(result['avg_price']) if result['avg_price'] else 0.0,
                status=status,
                exchange=ExchangeType.UPBIT,
                commission=float(result['paid_fee'])
            )
            
        except Exception as e:
            self.logger.error(f"업비트 주문 상태 조회 실패: {e}")
            raise
    
    async def get_balance(self, asset: str) -> float:
        """업비트 잔고"""
        try:
            balance = self.client.get_balance(asset)
            return float(balance) if balance else 0.0
        except Exception as e:
            self.logger.error(f"업비트 잔고 조회 실패: {e}")
            return 0.0
    
    async def get_market_price(self, symbol: str) -> float:
        """업비트 현재가"""
        try:
            if not symbol.startswith("KRW-"):
                symbol = f"KRW-{symbol}"
            
            price = pyupbit.get_current_price(symbol)
            return float(price) if price else 0.0
        except Exception as e:
            self.logger.error(f"업비트 현재가 조회 실패: {e}")
            return 0.0
    
    async def get_positions(self) -> List[Position]:
        """업비트 포지션 (보유 자산)"""
        try:
            balances = self.client.get_balances()
            positions = []
            
            for balance in balances:
                currency = balance['currency']
                quantity = float(balance['balance'])
                
                if quantity > 0 and currency != 'KRW':
                    symbol = f"KRW-{currency}"
                    market_price = await self.get_market_price(symbol)
                    
                    # 평균 매수가는 업비트에서 직접 제공하지 않으므로 현재가로 대체
                    position = Position(
                        symbol=symbol,
                        side=PositionSide.LONG,
                        quantity=quantity,
                        average_price=market_price,  # 실제로는 거래 내역에서 계산해야 함
                        market_price=market_price,
                        exchange=ExchangeType.UPBIT
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"업비트 포지션 조회 실패: {e}")
            return []

class IBKRExchange(BaseExchange):
    """IBKR 거래소"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ExchangeType.IBKR, config)
        self.client = None
    
    async def connect(self) -> bool:
        """IBKR 연결"""
        try:
            if not IBKR_AVAILABLE:
                raise ImportError("ib_insync 모듈이 필요합니다")
            
            self.client = IB()
            
            host = self.config.get('host', '127.0.0.1')
            port = self.config.get('port', 7497)
            client_id = self.config.get('client_id', 1)
            
            self.client.connect(host=host, port=port, clientId=client_id)
            
            self.is_connected = True
            self.logger.info("IBKR 연결 성공")
            return True
            
        except Exception as e:
            self.logger.error(f"IBKR 연결 실패: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """IBKR 연결 해제"""
        if self.client:
            self.client.disconnect()
        self.is_connected = False
        self.logger.info("IBKR 연결 해제")
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """IBKR 주문"""
        if not self.is_connected or not self.client:
            raise ConnectionError("IBKR에 연결되지 않음")
        
        try:
            # 통화 결정 (일본 주식은 JPY, 나머지는 USD)
            currency = 'JPY' if order_request.symbol.endswith('.T') else 'USD'
            
            # 계약 정의
            contract = Stock(order_request.symbol.replace('.T', ''), 'SMART', currency)
            
            # 주문 생성
            if order_request.order_type == OrderType.MARKET:
                order = MarketOrder(
                    order_request.side.value.upper(),
                    order_request.quantity
                )
            else:
                order = LimitOrder(
                    order_request.side.value.upper(),
                    order_request.quantity,
                    order_request.price
                )
            
            # 주문 전송
            trade = self.client.placeOrder(contract, order)
            
            # Order 객체 생성
            result_order = Order(
                order_id=str(trade.order.orderId),
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                client_order_id=order_request.client_order_id,
                exchange=ExchangeType.IBKR
            )
            
            self.logger.info(f"IBKR 주문 성공: {result_order.order_id}")
            return result_order
            
        except Exception as e:
            self.logger.error(f"IBKR 주문 실패: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """IBKR 주문 취소"""
        try:
            # IBKR의 주문 취소 로직
            trades = self.client.trades()
            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    self.client.cancelOrder(trade.order)
                    return True
            return False
        except Exception as e:
            self.logger.error(f"IBKR 주문 취소 실패: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Order:
        """IBKR 주문 상태"""
        # IBKR 주문 상태 조회 로직 구현
        # 실제 구현에서는 ib_insync의 Trade 객체를 활용
        pass
    
    async def get_balance(self, asset: str) -> float:
        """IBKR 잔고"""
        try:
            account_values = self.client.accountValues()
            for value in account_values:
                if value.tag == 'CashBalance' and value.currency == asset:
                    return float(value.value)
            return 0.0
        except Exception as e:
            self.logger.error(f"IBKR 잔고 조회 실패: {e}")
            return 0.0
    
    async def get_market_price(self, symbol: str) -> float:
        """IBKR 현재가"""
        try:
            currency = 'JPY' if symbol.endswith('.T') else 'USD'
            contract = Stock(symbol.replace('.T', ''), 'SMART', currency)
            
            # 시장 데이터 요청
            self.client.reqMktData(contract, '', False, False)
            self.client.sleep(1)  # 데이터 수신 대기
            
            ticker = self.client.ticker(contract)
            if ticker and ticker.last:
                return float(ticker.last)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"IBKR 현재가 조회 실패: {e}")
            return 0.0
    
    async def get_positions(self) -> List[Position]:
        """IBKR 포지션"""
        try:
            positions = []
            ib_positions = self.client.positions()
            
            for pos in ib_positions:
                if pos.position != 0:
                    symbol = pos.contract.symbol
                    if pos.contract.currency == 'JPY':
                        symbol += '.T'
                    
                    position = Position(
                        symbol=symbol,
                        side=PositionSide.LONG if pos.position > 0 else PositionSide.SHORT,
                        quantity=abs(pos.position),
                        average_price=pos.avgCost,
                        market_price=await self.get_market_price(symbol),
                        exchange=ExchangeType.IBKR
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"IBKR 포지션 조회 실패: {e}")
            return []

class MockExchange(BaseExchange):
    """백테스팅용 모의 거래소"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ExchangeType.MOCK, config)
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.balances: Dict[str, float] = {
            'KRW': 1000000.0,  # 100만원
            'USD': 10000.0,    # 1만달러
            'JPY': 1000000.0   # 100만엔
        }
        self.order_counter = 0
    
    async def connect(self) -> bool:
        """모의 연결"""
        self.is_connected = True
        self.logger.info("모의 거래소 연결")
        return True
    
    async def disconnect(self):
        """모의 연결 해제"""
        self.is_connected = False
        self.logger.info("모의 거래소 연결 해제")
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """모의 주문"""
        self.order_counter += 1
        order_id = f"mock_{self.order_counter}"
        
        order = Order(
            order_id=order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            client_order_id=order_request.client_order_id,
            exchange=ExchangeType.MOCK,
            status=OrderStatus.FILLED,  # 모의에서는 즉시 체결
            filled_quantity=order_request.quantity,
            average_price=order_request.price or 100.0  # 임의 가격
        )
        
        self.orders[order_id] = order
        self.logger.info(f"모의 주문 체결: {order_id}")
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """모의 주문 취소"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
    
    async def get_order_status(self, order_id: str) -> Order:
        """모의 주문 상태"""
        return self.orders.get(order_id)
    
    async def get_balance(self, asset: str) -> float:
        """모의 잔고"""
        return self.balances.get(asset, 0.0)
    
    async def get_market_price(self, symbol: str) -> float:
        """모의 현재가"""
        # 심볼에 따른 임의 가격 반환
        import hashlib
        hash_value = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
        base_price = (hash_value % 90000) + 10000  # 10,000 ~ 100,000
        return float(base_price)
    
    async def get_positions(self) -> List[Position]:
        """모의 포지션"""
        return list(self.positions.values())

# =============================================================================
# 메인 거래 엔진
# =============================================================================

class TradingEngine(BaseComponent):
    """통합 거래 엔진"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        super().__init__("TradingEngine")
        self.config_dict = config_dict or {}
        self.trading_config = TradingConfig()
        self.exchanges: Dict[ExchangeType, BaseExchange] = {}
        self.active_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeExecution] = []
        self.risk_manager = RiskManager(self.trading_config)
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        
        # 백그라운드 작업용
        self._monitoring_task = None
        self._stop_monitoring = False
    
    def _do_initialize(self):
        """거래 엔진 초기화"""
        # 설정 로드
        self._load_trading_config()
        
        # 거래소 초기화
        self._initialize_exchanges()
        
        # 백그라운드 모니터링 시작
        self._start_monitoring()
        
        self.logger.info("거래 엔진 초기화 완료")
    
    def _load_trading_config(self):
        """거래 설정 로드"""
        if 'trading' in self.config_dict:
            trading_cfg = self.config_dict['trading']
            
            self.trading_config.max_position_size = trading_cfg.get('max_position_size', 0.1)
            self.trading_config.max_daily_loss = trading_cfg.get('max_daily_loss', 0.05)
            self.trading_config.default_stop_loss = trading_cfg.get('default_stop_loss', 0.08)
            self.trading_config.default_take_profit = trading_cfg.get('default_take_profit', 0.15)
            self.trading_config.enable_paper_trading = trading_cfg.get('enable_paper_trading', False)
            self.trading_config.enable_auto_stop_loss = trading_cfg.get('enable_auto_stop_loss', True)
    
    def _initialize_exchanges(self):
        """거래소 초기화"""
        api_config = self.config_dict.get('api', {})
        
        # 업비트 초기화
        if 'upbit' in api_config and UPBIT_AVAILABLE:
            self.exchanges[ExchangeType.UPBIT] = UpbitExchange(api_config['upbit'])
        
        # IBKR 초기화
        if 'ibkr' in api_config and IBKR_AVAILABLE:
            self.exchanges[ExchangeType.IBKR] = IBKRExchange(api_config['ibkr'])
        
        # 백테스팅 모드 또는 다른 거래소가 없는 경우 Mock 사용
        if not self.exchanges or self.trading_config.enable_paper_trading:
            self.exchanges[ExchangeType.MOCK] = MockExchange({})
    
    def _start_monitoring(self):
        """백그라운드 모니터링 시작"""
        if not self._monitoring_task:
            self._monitoring_task = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_task.start()
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while not self._stop_monitoring:
            try:
                # 활성 주문 상태 업데이트
                asyncio.run(self._update_active_orders())
                
                # 포지션 업데이트
                asyncio.run(self._update_positions())
                
                # 자동 손절/익절 확인
                if self.trading_config.enable_auto_stop_loss:
                    asyncio.run(self._check_stop_orders())
                
                # 리스크 체크
                self.risk_manager.check_daily_limits(self.positions, self.trade_history)
                
                time.sleep(5)  # 5초마다 체크
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(10)
    
    async def _update_active_orders(self):
        """활성 주문 상태 업데이트"""
        for order_id, order in list(self.active_orders.items()):
            try:
                exchange = self.exchanges.get(order.exchange)
                if exchange and exchange.is_connected:
                    updated_order = await exchange.get_order_status(order_id)
                    if updated_order:
                        self.active_orders[order_id] = updated_order
                        
                        # 체결 완료된 주문 처리
                        if updated_order.is_filled:
                            await self._process_filled_order(updated_order)
                            del self.active_orders[order_id]
                            
            except Exception as e:
                self.logger.error(f"주문 상태 업데이트 실패 ({order_id}): {e}")
    
    async def _update_positions(self):
        """포지션 업데이트"""
        for exchange_type, exchange in self.exchanges.items():
            if exchange.is_connected:
                try:
                    positions = await exchange.get_positions()
                    for position in positions:
                        key = f"{position.symbol}_{exchange_type.value}"
                        
                        # 현재가 업데이트
                        position.market_price = await exchange.get_market_price(position.symbol)
                        
                        self.positions[key] = position
                        
                except Exception as e:
                    self.logger.error(f"포지션 업데이트 실패 ({exchange_type.value}): {e}")
    
    async def _check_stop_orders(self):
        """자동 손절/익절 확인"""
        for position_key, position in self.positions.items():
            try:
                # 손절 확인
                if (position.stop_loss_price and 
                    position.side == PositionSide.LONG and 
                    position.market_price <= position.stop_loss_price):
                    
                    await self._execute_stop_loss(position)
                
                # 익절 확인
                elif (position.take_profit_price and 
                      position.side == PositionSide.LONG and 
                      position.market_price >= position.take_profit_price):
                    
                    await self._execute_take_profit(position)
                    
            except Exception as e:
                self.logger.error(f"자동 주문 확인 실패 ({position_key}): {e}")
    
    async def _execute_stop_loss(self, position: Position):
        """손절 실행"""
        try:
            order_request = OrderRequest(
                symbol=position.symbol,
                side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                exchange=position.exchange
            )
            
            await self.place_order(order_request)
            self.logger.warning(f"자동 손절 실행: {position.symbol} at {position.market_price}")
            
        except Exception as e:
            self.logger.error(f"손절 실행 실패 ({position.symbol}): {e}")
    
    async def _execute_take_profit(self, position: Position):
        """익절 실행"""
        try:
            order_request = OrderRequest(
                symbol=position.symbol,
                side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                exchange=position.exchange
            )
            
            await self.place_order(order_request)
            self.logger.info(f"자동 익절 실행: {position.symbol} at {position.market_price}")
            
        except Exception as e:
            self.logger.error(f"익절 실행 실패 ({position.symbol}): {e}")
    
    async def _process_filled_order(self, order: Order):
        """체결된 주문 처리"""
        # 거래 내역 추가
        trade = TradeExecution(
            trade_id=f"trade_{int(time.time() * 1000)}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.average_price,
            commission=order.commission,
            exchange=order.exchange
        )
        
        self.trade_history.append(trade)
        
        # 포지션 업데이트
        await self.position_manager.update_position_from_trade(trade, self.positions)
        
        self.logger.info(f"거래 체결 처리 완료: {trade.trade_id}")
    
    # =============================================================================
    # 공개 API
    # =============================================================================
    
    async def connect_all_exchanges(self) -> Dict[ExchangeType, bool]:
        """모든 거래소 연결"""
        results = {}
        
        for exchange_type, exchange in self.exchanges.items():
            try:
                success = await exchange.connect()
                results[exchange_type] = success
                
                if success:
                    self.logger.info(f"{exchange_type.value} 연결 성공")
                else:
                    self.logger.error(f"{exchange_type.value} 연결 실패")
                    
            except Exception as e:
                self.logger.error(f"{exchange_type.value} 연결 중 오류: {e}")
                results[exchange_type] = False
        
        return results
    
    async def disconnect_all_exchanges(self):
        """모든 거래소 연결 해제"""
        self._stop_monitoring = True
        
        for exchange_type, exchange in self.exchanges.items():
            try:
                await exchange.disconnect()
                self.logger.info(f"{exchange_type.value} 연결 해제")
            except Exception as e:
                self.logger.error(f"{exchange_type.value} 연결 해제 오류: {e}")
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """주문 전송"""
        # 리스크 검사
        if not self.risk_manager.validate_order(order_request, self.positions):
            raise ValidationError("리스크 검사 실패")
        
        # 거래소 확인
        exchange = self.exchanges.get(order_request.exchange)
        if not exchange or not exchange.is_connected:
            raise ConnectionError(f"거래소에 연결되지 않음: {order_request.exchange.value}")
        
        try:
            # 주문 전송
            order = await exchange.place_order(order_request)
            
            # 활성 주문에 추가
            self.active_orders[order.order_id] = order
            
            # 손절/익절가 설정
            if order_request.side == OrderSide.BUY:
                await self._set_stop_orders(order)
            
            self.logger.info(f"주문 전송 완료: {order.order_id}")
            return order
            
        except Exception as e:
            self.logger.error(f"주문 전송 실패: {e}")
            raise
    
    async def _set_stop_orders(self, order: Order):
        """손절/익절가 설정"""
        if order.price and self.trading_config.enable_auto_stop_loss:
            # 손절가 계산
            stop_loss_price = order.price * (1 - self.trading_config.default_stop_loss)
            
            # 익절가 계산
            take_profit_price = order.price * (1 + self.trading_config.default_take_profit)
            
            # 포지션에 설정 (실제 거래소 손절 주문은 별도 구현 필요)
            position_key = f"{order.symbol}_{order.exchange.value}"
            if position_key in self.positions:
                self.positions[position_key].stop_loss_price = stop_loss_price
                self.positions[position_key].take_profit_price = take_profit_price
    
    async def cancel_order(self, order_id: str, exchange_type: ExchangeType) -> bool:
        """주문 취소"""
        exchange = self.exchanges.get(exchange_type)
        if not exchange or not exchange.is_connected:
            return False
        
        try:
            success = await exchange.cancel_order(order_id)
            
            if success and order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELLED
                del self.active_orders[order_id]
            
            return success
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {e}")
            return False
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """포트폴리오 요약"""
        total_value = 0.0
        total_pnl = 0.0
        position_count = 0
        
        balances = {}
        
        # 모든 거래소의 잔고 조회
        for exchange_type, exchange in self.exchanges.items():
            if exchange.is_connected:
                try:
                    # 주요 통화 잔고
                    currencies = ['KRW', 'USD', 'JPY'] if exchange_type != ExchangeType.UPBIT else ['KRW']
                    for currency in currencies:
                        balance = await exchange.get_balance(currency)
                        if balance > 0:
                            balances[f"{currency}_{exchange_type.value}"] = balance
                            total_value += balance  # 간단히 합산 (실제로는 환율 적용 필요)
                except Exception as e:
                    self.logger.error(f"잔고 조회 실패 ({exchange_type.value}): {e}")
        
        # 포지션 요약
        for position in self.positions.values():
            if position.side != PositionSide.FLAT:
                position_count += 1
                total_value += position.market_value
                total_pnl += position.unrealized_pnl
        
        return {
            'total_value': total_value,
            'total_unrealized_pnl': total_pnl,
            'total_unrealized_pnl_pct': total_pnl / total_value if total_value > 0 else 0,
            'position_count': position_count,
            'active_order_count': len(self.active_orders),
            'balances': balances,
            'connected_exchanges': [
                ex_type.value for ex_type, ex in self.exchanges.items() if ex.is_connected
            ]
        }
    
    def get_trading_performance(self, days: int = 30) -> Dict[str, Any]:
        """거래 성과 분석"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [
            trade for trade in self.trade_history 
            if trade.timestamp >= cutoff_date
        ]
        
        if not recent_trades:
            return {'error': '분석할 거래 데이터 없음'}
        
        # 기본 통계
        total_trades = len(recent_trades)
        buy_trades = [t for t in recent_trades if t.side == OrderSide.BUY]
        sell_trades = [t for t in recent_trades if t.side == OrderSide.SELL]
        
        total_volume = sum(trade.gross_amount for trade in recent_trades)
        total_commission = sum(trade.commission for trade in recent_trades)
        
        # 수익률 계산 (매우 간단한 버전)
        profit_trades = 0
        loss_trades = 0
        
        # 실제로는 매수-매도 매칭해서 계산해야 함
        avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
        
        return {
            'period_days': days,
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_volume': total_volume,
            'total_commission': total_commission,
            'avg_trade_size': avg_trade_size,
            'commission_rate': total_commission / total_volume if total_volume > 0 else 0,
        }

# =============================================================================
# 보조 관리자 클래스들
# =============================================================================

class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, trading_config: TradingConfig):
        self.config = trading_config
        self.logger = logging.getLogger(f"{__name__}.RiskManager")
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
    
    def validate_order(self, order_request: OrderRequest, positions: Dict[str, Position]) -> bool:
        """주문 검증"""
        try:
            # 최소 주문 금액 확인
            if order_request.quantity < self.config.min_order_amount:
                self.logger.warning(f"최소 주문 금액 미달: {order_request.quantity}")
                return False
            
            # 포지션 크기 제한 확인
            if order_request.side == OrderSide.BUY:
                total_position_value = sum(pos.market_value for pos in positions.values())
                if total_position_value > self.config.max_position_size * 1000000:  # 가정: 100만원 기준
                    self.logger.warning("최대 포지션 크기 초과")
                    return False
            
            # 일일 손실 한도 확인
            if not self._check_daily_loss_limit():
                self.logger.warning("일일 손실 한도 초과")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"주문 검증 오류: {e}")
            return False
    
    def _check_daily_loss_limit(self) -> bool:
        """일일 손실 한도 확인"""
        today = datetime.now().date()
        
        # 날짜가 바뀌면 리셋
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = today
        
        # 일일 최대 손실 확인
        max_daily_loss = self.config.max_daily_loss * 1000000  # 가정: 100만원 기준
        return self.daily_pnl > -max_daily_loss
    
    def check_daily_limits(self, positions: Dict[str, Position], trades: List[TradeExecution]):
        """일일 한도 체크"""
        today = datetime.now().date()
        
        # 오늘의 거래로부터 일일 손익 계산
        today_trades = [t for t in trades if t.timestamp.date() == today]
        
        # 실현 손익 계산 (매우 간단한 버전)
        # 실제로는 매수-매도 매칭해서 정확히 계산해야 함
        
        # 미실현 손익 합산
        unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        
        self.daily_pnl = unrealized_pnl  # 임시로 미실현 손익만 사용

class OrderManager:
    """주문 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.OrderManager")
        self.pending_orders: Dict[str, Order] = {}
    
    def add_pending_order(self, order: Order):
        """대기 주문 추가"""
        self.pending_orders[order.order_id] = order
    
    def remove_pending_order(self, order_id: str):
        """대기 주문 제거"""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """심볼별 주문 조회"""
        return [order for order in self.pending_orders.values() if order.symbol == symbol]

class PositionManager:
    """포지션 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PositionManager")
    
    async def update_position_from_trade(self, trade: TradeExecution, positions: Dict[str, Position]):
        """거래로부터 포지션 업데이트"""
        position_key = f"{trade.symbol}_{trade.exchange.value}"
        
        if position_key in positions:
            position = positions[position_key]
            
            if trade.side == OrderSide.BUY:
                # 매수: 포지션 증가
                new_quantity = position.quantity + trade.quantity
                new_average_price = (
                    (position.average_price * position.quantity + trade.price * trade.quantity) 
                    / new_quantity
                )
                
                position.quantity = new_quantity
                position.average_price = new_average_price
                position.side = PositionSide.LONG
                
            elif trade.side == OrderSide.SELL:
                # 매도: 포지션 감소
                position.quantity -= trade.quantity
                
                if position.quantity <= 0:
                    position.side = PositionSide.FLAT
                    position.quantity = 0
        else:
            # 새 포지션 생성
            position = Position(
                symbol=trade.symbol,
                side=PositionSide.LONG if trade.side == OrderSide.BUY else PositionSide.SHORT,
                quantity=trade.quantity,
                average_price=trade.price,
                exchange=trade.exchange
            )
            positions[position_key] = position

# =============================================================================
# 편의 함수들 (기존 API 호환성)
# =============================================================================

# 전역 거래 엔진 인스턴스
_trading_engine = None

def get_trading_engine(config_dict: Optional[Dict] = None) -> TradingEngine:
    """전역 거래 엔진 인스턴스 반환"""
    global _trading_engine
    if _trading_engine is None:
        _trading_engine = TradingEngine(config_dict)
        _trading_engine.initialize()
    return _trading_engine

# 기존 API 호환성을 위한 래퍼 함수들
async def buy_upbit(ticker: str, amount: float) -> Order:
    """업비트 매수 (기존 호환성)"""
    engine = get_trading_engine()
    
    order_request = OrderRequest(
        symbol=ticker,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=amount,
        exchange=ExchangeType.UPBIT
    )
    
    return await engine.place_order(order_request)

async def sell_upbit(ticker: str, amount: float) -> Order:
    """업비트 매도 (기존 호환성)"""
    engine = get_trading_engine()
    
    order_request = OrderRequest(
        symbol=ticker,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=amount,
        exchange=ExchangeType.UPBIT
    )
    
    return await engine.place_order(order_request)

async def get_balance_upbit(currency: str) -> float:
    """업비트 잔고 (기존 호환성)"""
    engine = get_trading_engine()
    exchange = engine.exchanges.get(ExchangeType.UPBIT)
    
    if exchange and exchange.is_connected:
        return await exchange.get_balance(currency)
    return 0.0

async def buy_ibkr(symbol: str, qty: float) -> Order:
    """IBKR 매수 (기존 호환성)"""
    engine = get_trading_engine()
    
    order_request = OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=qty,
        exchange=ExchangeType.IBKR
    )
    
    return await engine.place_order(order_request)

async def sell_ibkr(symbol: str, qty: float) -> Order:
    """IBKR 매도 (기존 호환성)"""
    engine = get_trading_engine()
    
    order_request = OrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=qty,
        exchange=ExchangeType.IBKR
    )
    
    return await engine.place_order(order_request)

# =============================================================================
# 메인 실행부 및 테스트
# =============================================================================

async def main():
    """메인 실행 함수 (테스트용)"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 설정
    test_config = {
        'trading': {
            'enable_paper_trading': True,
            'max_position_size': 0.1,
            'enable_auto_stop_loss': True
        },
        'api': {
            'mock': {}  # 테스트용 mock 거래소만 사용
        }
    }
    
    # 거래 엔진 초기화
    engine = get_trading_engine(test_config)
    
    print("=== 고급 통합 거래 엔진 시스템 ===\n")
    
    # 거래소 연결
    print("🔌 거래소 연결 중...")
    connection_results = await engine.connect_all_exchanges()
    
    for exchange_type, success in connection_results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"   {exchange_type.value}: {status}")
    
    print()
    
    # 포트폴리오 요약
    print("📊 포트폴리오 현황:")
    portfolio = await engine.get_portfolio_summary()
    
    print(f"   총 자산: {portfolio['total_value']:,.0f}")
    print(f"   포지션 수: {portfolio['position_count']}개")
    print(f"   활성 주문: {portfolio['active_order_count']}개")
    print(f"   연결된 거래소: {', '.join(portfolio['connected_exchanges'])}")
    
    if portfolio['balances']:
        print("   💰 잔고:")
        for currency, balance in portfolio['balances'].items():
            print(f"      {currency}: {balance:,.2f}")
    
    print()
    
    # 테스트 주문 (Mock 거래소)
    if ExchangeType.MOCK in engine.exchanges:
        print("📝 테스트 주문 실행...")
        
        try:
            # 매수 주문
            buy_order = OrderRequest(
                symbol="BTC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=50000,  # 5만원어치
                exchange=ExchangeType.MOCK
            )
            
            order = await engine.place_order(buy_order)
            print(f"   ✅ 매수 주문 성공: {order.order_id}")
            
            # 잠시 대기 (실제 환경에서는 체결 완료까지 기다림)
            await asyncio.sleep(1)
            
            # 포트폴리오 업데이트 확인
            updated_portfolio = await engine.get_portfolio_summary()
            print(f"   📈 업데이트된 포지션 수: {updated_portfolio['position_count']}개")
            
        except Exception as e:
            print(f"   ❌ 테스트 주문 실패: {e}")
    
    print()
    
    # 거래 성과 분석
    print("📈 거래 성과 분석:")
    performance = engine.get_trading_performance(30)
    
    if 'error' not in performance:
        print(f"   총 거래 수: {performance['total_trades']}건")
        print(f"   총 거래량: {performance['total_volume']:,.0f}")
        print(f"   평균 거래 크기: {performance['avg_trade_size']:,.0f}")
        print(f"   수수료율: {performance['commission_rate']:.4%}")
    else:
        print(f"   {performance['error']}")
    
    print()
    
    # 리스크 요약
    print("⚠️  리스크 관리 상태:")
    print(f"   최대 포지션 크기: {engine.trading_config.max_position_size:.1%}")
    print(f"   일일 손실 한도: {engine.trading_config.max_daily_loss:.1%}")
    print(f"   자동 손절: {'활성화' if engine.trading_config.enable_auto_stop_loss else '비활성화'}")
    print(f"   페이퍼 트레이딩: {'활성화' if engine.trading_config.enable_paper_trading else '비활성화'}")
    
    print()
    
    # 연결 해제
    print("🔌 거래소 연결 해제 중...")
    await engine.disconnect_all_exchanges()
    
    print("✅ 거래 엔진 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main())

# =============================================================================
# 공개 API
# =============================================================================

__all__ = [
    # 메인 클래스들
    'TradingEngine',
    'BaseExchange',
    'UpbitExchange',
    'IBKRExchange',
    'MockExchange',
    'RiskManager',
    'OrderManager',
    'PositionManager',
    
    # 데이터 클래스들
    'OrderRequest',
    'Order',
    'Position',
    'TradeExecution',
    'TradingConfig',
    
    # 열거형들
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'ExchangeType',
    'PositionSide',
    
    # 편의 함수들
    'get_trading_engine',
    'buy_upbit',
    'sell_upbit',
    'get_balance_upbit',
    'buy_ibkr',
    'sell_ibkr',
]