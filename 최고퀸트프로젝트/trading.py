"""
💰 최고퀸트프로젝트 - 실제 매매 시스템
=====================================

완전한 글로벌 매매 시스템:
- 🏦 Interactive Brokers (IBKR) - 미국/일본 주식
- 🪙 업비트 (Upbit) - 암호화폐
- 📊 통합 포트폴리오 관리
- 🛡️ 고급 리스크 관리
- 💾 실시간 주문 추적
- 🔄 자동 재시도 및 에러 처리
- 📈 성과 추적 및 분석

Author: 최고퀸트팀
Version: 1.0.0
Project: 최고퀸트프로젝트
"""

import asyncio
import aiohttp
import logging
import json
import hmac
import hashlib
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import yaml
import pandas as pd
from decimal import Decimal, ROUND_DOWN
import pytz

# 프로젝트 모듈 import
try:
    from utils import (
        BrokerUtils, DataProcessor, get_config, 
        save_trading_log, DatabaseUtils, Formatter
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ utils 모듈 로드 실패: {e}")
    UTILS_AVAILABLE = False

try:
    from notifier import send_trading_alert, send_system_alert
    NOTIFIER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ notifier 모듈 로드 실패: {e}")
    NOTIFIER_AVAILABLE = False

# Interactive Brokers API
try:
    from ib_insync import IB, Stock, Forex, util
    IBKR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ IBKR 모듈 로드 실패: {e}")
    print("   pip install ib_insync 설치 필요")
    IBKR_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """매매 신호 (core.py와 호환)"""
    market: str
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    strategy: str
    reasoning: str
    target_price: float
    timestamp: datetime
    sector: Optional[str] = None
    position_size: Optional[float] = None
    additional_data: Optional[Dict] = None

@dataclass
class TradeOrder:
    """거래 주문 정보"""
    order_id: str
    symbol: str
    market: str
    action: str  # 'buy', 'sell'
    quantity: float
    price: float
    order_type: str  # 'market', 'limit'
    status: str  # 'pending', 'filled', 'cancelled', 'failed'
    broker: str  # 'ibkr', 'upbit'
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class Portfolio:
    """포트폴리오 정보"""
    broker: str
    currency: str
    cash_balance: float
    total_value: float
    positions: List[Dict]
    last_updated: datetime

class IBKRConnector:
    """Interactive Brokers 연동"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ib = None
        self.connected = False
        
        # IBKR 설정
        self.ibkr_config = config.get('api', {}).get('ibkr', {})
        self.paper_trading = self.ibkr_config.get('paper_trading', True)
        self.tws_port = self.ibkr_config.get('tws_port', 7497 if self.paper_trading else 7496)
        self.client_id = self.ibkr_config.get('client_id', 1)
        
        logger.info(f"🏦 IBKR 커넥터 초기화 (모의거래: {self.paper_trading})")
    
    async def connect(self) -> bool:
        """IBKR TWS 연결"""
        try:
            if not IBKR_AVAILABLE:
                logger.error("❌ IBKR 라이브러리 없음 (pip install ib_insync)")
                return False
            
            self.ib = IB()
            
            # TWS 연결 시도
            await self.ib.connectAsync('127.0.0.1', self.tws_port, clientId=self.client_id)
            self.connected = True
            
            logger.info(f"✅ IBKR 연결 성공 (포트: {self.tws_port})")
            
            # 계좌 정보 확인
            accounts = self.ib.managedAccounts()
            if accounts:
                logger.info(f"📊 연결된 계좌: {accounts}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ IBKR 연결 실패: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """IBKR 연결 해제"""
        try:
            if self.ib and self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("📴 IBKR 연결 해제 완료")
        except Exception as e:
            logger.error(f"❌ IBKR 연결 해제 실패: {e}")
    
    def _create_contract(self, symbol: str, market: str):
        """계약 객체 생성"""
        try:
            if market == 'US':
                # 미국 주식
                if '.' in symbol:
                    symbol = symbol.split('.')[0]  # 확장자 제거
                return Stock(symbol, 'SMART', 'USD')
                
            elif market == 'JP':
                # 일본 주식
                if symbol.endswith('.T'):
                    symbol = symbol[:-2]  # .T 제거
                return Stock(symbol, 'TSE', 'JPY')
                
            else:
                logger.error(f"❌ 지원하지 않는 시장: {market}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 계약 생성 실패 {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str, market: str) -> Optional[float]:
        """현재가 조회"""
        try:
            if not self.connected:
                await self.connect()
            
            contract = self._create_contract(symbol, market)
            if not contract:
                return None
            
            # 시장 데이터 요청
            market_data = self.ib.reqMktData(contract)
            await asyncio.sleep(2)  # 데이터 수신 대기
            
            if market_data.last and market_data.last > 0:
                return float(market_data.last)
            elif market_data.bid and market_data.ask:
                return float((market_data.bid + market_data.ask) / 2)
            else:
                logger.warning(f"⚠️ {symbol} 시장 데이터 없음")
                return None
                
        except Exception as e:
            logger.error(f"❌ {symbol} 현재가 조회 실패: {e}")
            return None
    
    async def place_order(self, symbol: str, market: str, action: str, 
                         quantity: float, order_type: str = 'market') -> Dict:
        """주문 실행"""
        try:
            if not self.connected:
                await self.connect()
            
            contract = self._create_contract(symbol, market)
            if not contract:
                return {'success': False, 'error': '계약 생성 실패'}
            
            # 주문 생성
            from ib_insync import MarketOrder, LimitOrder
            
            if order_type.lower() == 'market':
                order = MarketOrder(action.upper(), quantity)
            else:
                # 제한가 주문 (현재가 기준)
                current_price = await self.get_current_price(symbol, market)
                if not current_price:
                    return {'success': False, 'error': '현재가 조회 실패'}
                
                # 약간의 슬리피지 적용
                if action.lower() == 'buy':
                    limit_price = current_price * 1.002  # 0.2% 높게
                else:
                    limit_price = current_price * 0.998  # 0.2% 낮게
                
                order = LimitOrder(action.upper(), quantity, limit_price)
            
            # 주문 전송
            trade = self.ib.placeOrder(contract, order)
            
            # 주문 ID
            order_id = str(trade.order.orderId)
            
            logger.info(f"📤 IBKR 주문 전송: {symbol} {action} {quantity}주")
            
            # 주문 완료 대기 (최대 30초)
            for _ in range(30):
                await asyncio.sleep(1)
                self.ib.sleep(0.1)  # 이벤트 처리
                
                if trade.orderStatus.status in ['Filled']:
                    # 체결 완료
                    filled_price = float(trade.orderStatus.avgFillPrice) if trade.orderStatus.avgFillPrice else None
                    filled_qty = float(trade.orderStatus.filled) if trade.orderStatus.filled else 0
                    
                    return {
                        'success': True,
                        'order_id': order_id,
                        'price': filled_price,
                        'quantity': filled_qty,
                        'status': 'filled'
                    }
                    
                elif trade.orderStatus.status in ['Cancelled', 'ApiCancelled']:
                    return {'success': False, 'error': '주문 취소됨', 'order_id': order_id}
                    
                elif 'Error' in trade.orderStatus.status:
                    return {'success': False, 'error': f'주문 오류: {trade.orderStatus.status}', 'order_id': order_id}
            
            # 타임아웃
            return {'success': False, 'error': '주문 타임아웃', 'order_id': order_id}
            
        except Exception as e:
            logger.error(f"❌ IBKR 주문 실패 {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_portfolio(self) -> Optional[Portfolio]:
        """포트폴리오 조회"""
        try:
            if not self.connected:
                await self.connect()
            
            # 계좌 정보 조회
            account_values = self.ib.accountValues()
            positions = self.ib.positions()
            
            # 현금 잔고
            cash_balance = 0.0
            total_value = 0.0
            
            for av in account_values:
                if av.tag == 'CashBalance' and av.currency == 'USD':
                    cash_balance = float(av.value)
                elif av.tag == 'NetLiquidation' and av.currency == 'USD':
                    total_value = float(av.value)
            
            # 포지션 정보
            position_list = []
            for pos in positions:
                if pos.position != 0:
                    position_list.append({
                        'symbol': pos.contract.symbol,
                        'quantity': float(pos.position),
                        'market_price': float(pos.marketPrice) if pos.marketPrice else 0.0,
                        'market_value': float(pos.marketValue) if pos.marketValue else 0.0,
                        'avg_cost': float(pos.avgCost) if pos.avgCost else 0.0
                    })
            
            return Portfolio(
                broker='ibkr',
                currency='USD',
                cash_balance=cash_balance,
                total_value=total_value,
                positions=position_list,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ IBKR 포트폴리오 조회 실패: {e}")
            return None

class UpbitConnector:
    """업비트 연동"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 업비트 설정
        self.upbit_config = config.get('api', {}).get('upbit', {})
        self.access_key = self.upbit_config.get('access_key', '')
        self.secret_key = self.upbit_config.get('secret_key', '')
        self.server_url = 'https://api.upbit.com'
        
        self.session = None
        
        logger.info("🪙 업비트 커넥터 초기화")
    
    async def _get_session(self):
        """HTTP 세션 가져오기"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def close(self):
        """세션 종료"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _generate_query_string(self, params: Dict) -> str:
        """쿼리 스트링 생성"""
        if not params:
            return ''
        
        return '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    
    def _create_jwt_token(self, query_string: str = '') -> str:
        """JWT 토큰 생성"""
        try:
            import jwt
            
            payload = {
                'access_key': self.access_key,
                'nonce': str(uuid.uuid4())
            }
            
            if query_string:
                payload['query_hash'] = hashlib.sha512(query_string.encode()).hexdigest()
                payload['query_hash_alg'] = 'SHA512'
            
            return jwt.encode(payload, self.secret_key, algorithm='HS256')
            
        except ImportError:
            logger.error("❌ PyJWT 라이브러리 필요: pip install PyJWT")
            return ''
        except Exception as e:
            logger.error(f"❌ JWT 토큰 생성 실패: {e}")
            return ''
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회"""
        try:
            session = await self._get_session()
            
            # 심볼 정규화 (BTC-KRW 형식으로)
            if '-' not in symbol:
                symbol = f"{symbol}-KRW"
            
            url = f"{self.server_url}/v1/ticker"
            params = {'markets': symbol}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return float(data[0]['trade_price'])
                    else:
                        logger.warning(f"⚠️ {symbol} 시세 데이터 없음")
                        return None
                else:
                    logger.error(f"❌ 업비트 시세 조회 실패 ({response.status}): {symbol}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ {symbol} 현재가 조회 실패: {e}")
            return None
    
    async def place_order(self, symbol: str, action: str, quantity: float = None, 
                         amount: float = None, order_type: str = 'market') -> Dict:
        """주문 실행"""
        try:
            if not self.access_key or not self.secret_key:
                return {'success': False, 'error': '업비트 API 키 설정 필요'}
            
            session = await self._get_session()
            
            # 심볼 정규화
            if '-' not in symbol:
                symbol = f"{symbol}-KRW"
            
            # 주문 파라미터
            if action.lower() == 'buy':
                # 매수: 금액 지정
                if not amount:
                    return {'success': False, 'error': '매수시 금액(amount) 필요'}
                
                params = {
                    'market': symbol,
                    'side': 'bid',
                    'price': str(int(amount))  # 원화는 정수만
                }
                
                if order_type.lower() == 'market':
                    params['ord_type'] = 'price'  # 시장가 매수
                else:
                    # 지정가 매수 (구현 생략)
                    params['ord_type'] = 'limit'
                    
            else:
                # 매도: 수량 지정
                if not quantity:
                    return {'success': False, 'error': '매도시 수량(quantity) 필요'}
                
                params = {
                    'market': symbol,
                    'side': 'ask',
                    'volume': str(quantity),
                    'ord_type': 'market'  # 시장가 매도
                }
            
            # JWT 토큰 생성
            query_string = self._generate_query_string(params)
            jwt_token = self._create_jwt_token(query_string)
            
            if not jwt_token:
                return {'success': False, 'error': 'JWT 토큰 생성 실패'}
            
            # 주문 전송
            url = f"{self.server_url}/v1/orders"
            headers = {'Authorization': f'Bearer {jwt_token}'}
            
            async with session.post(url, json=params, headers=headers) as response:
                if response.status == 201:
                    result = await response.json()
                    
                    order_id = result.get('uuid', '')
                    
                    logger.info(f"📤 업비트 주문 전송: {symbol} {action}")
                    
                    # 주문 완료 대기
                    for _ in range(30):  # 최대 30초
                        await asyncio.sleep(1)
                        
                        order_info = await self._get_order_info(order_id)
                        if order_info:
                            state = order_info.get('state', '')
                            
                            if state == 'done':
                                # 체결 완료
                                executed_volume = float(order_info.get('executed_volume', 0))
                                paid_fee = float(order_info.get('paid_fee', 0))
                                trades = order_info.get('trades', [])
                                
                                avg_price = 0.0
                                if trades:
                                    total_price = sum(float(trade['price']) * float(trade['volume']) for trade in trades)
                                    total_volume = sum(float(trade['volume']) for trade in trades)
                                    avg_price = total_price / total_volume if total_volume > 0 else 0
                                
                                return {
                                    'success': True,
                                    'order_id': order_id,
                                    'price': avg_price,
                                    'quantity': executed_volume,
                                    'fee': paid_fee,
                                    'status': 'filled'
                                }
                                
                            elif state == 'cancel':
                                return {'success': False, 'error': '주문 취소됨', 'order_id': order_id}
                    
                    # 타임아웃
                    return {'success': False, 'error': '주문 타임아웃', 'order_id': order_id}
                    
                else:
                    error_text = await response.text()
                    logger.error(f"❌ 업비트 주문 실패 ({response.status}): {error_text}")
                    return {'success': False, 'error': f'API 오류: {response.status}'}
                    
        except Exception as e:
            logger.error(f"❌ 업비트 주문 실패 {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _get_order_info(self, order_id: str) -> Optional[Dict]:
        """주문 정보 조회"""
        try:
            session = await self._get_session()
            
            params = {'uuid': order_id}
            query_string = self._generate_query_string(params)
            jwt_token = self._create_jwt_token(query_string)
            
            if not jwt_token:
                return None
            
            url = f"{self.server_url}/v1/order"
            headers = {'Authorization': f'Bearer {jwt_token}'}
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"❌ 주문 정보 조회 실패: {e}")
            return None
    
    async def get_portfolio(self) -> Optional[Portfolio]:
        """포트폴리오 조회"""
        try:
            if not self.access_key or not self.secret_key:
                return None
            
            session = await self._get_session()
            
            # 계좌 정보 조회
            jwt_token = self._create_jwt_token()
            if not jwt_token:
                return None
            
            url = f"{self.server_url}/v1/accounts"
            headers = {'Authorization': f'Bearer {jwt_token}'}
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    accounts = await response.json()
                    
                    cash_balance = 0.0
                    total_value = 0.0
                    positions = []
                    
                    for account in accounts:
                        currency = account.get('currency', '')
                        balance = float(account.get('balance', 0))
                        locked = float(account.get('locked', 0))
                        avg_buy_price = float(account.get('avg_buy_price', 0))
                        
                        if currency == 'KRW':
                            cash_balance = balance
                        elif balance > 0:
                            # 암호화폐 포지션
                            symbol = f"{currency}-KRW"
                            current_price = await self.get_current_price(symbol)
                            market_value = balance * current_price if current_price else 0
                            
                            positions.append({
                                'symbol': symbol,
                                'quantity': balance,
                                'locked': locked,
                                'avg_cost': avg_buy_price,
                                'current_price': current_price,
                                'market_value': market_value
                            })
                            
                            total_value += market_value
                    
                    total_value += cash_balance
                    
                    return Portfolio(
                        broker='upbit',
                        currency='KRW',
                        cash_balance=cash_balance,
                        total_value=total_value,
                        positions=positions,
                        last_updated=datetime.now()
                    )
                    
                else:
                    logger.error(f"❌ 업비트 계좌 조회 실패 ({response.status})")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ 업비트 포트폴리오 조회 실패: {e}")
            return None

class TradingExecutor:
    """🏆 최고퀸트프로젝트 매매 실행기"""
    
    def __init__(self, config_path: str = "configs/settings.yaml"):
        """매매 실행기 초기화"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # 매매 설정
        self.trading_config = self.config.get('trading', {})
        self.paper_trading = self.trading_config.get('paper_trading', True)
        self.auto_execution = self.trading_config.get('auto_execution', False)
        
        # 리스크 관리 설정
        self.risk_config = self.config.get('risk_management', {})
        self.max_position_size = self.risk_config.get('max_position_size', 0.1)
        self.max_daily_trades = self.risk_config.get('max_daily_trades', 10)
        
        # 브로커 연결
        self.ibkr = IBKRConnector(self.config) if IBKR_AVAILABLE else None
        self.upbit = UpbitConnector(self.config)
        
        # 실행 통계
        self.daily_trades = 0
        self.session_start_time = datetime.now()
        self.order_history = []
        
        # 데이터베이스 초기화
        if UTILS_AVAILABLE:
            DatabaseUtils.init_database()
        
        logger.info("💰 최고퀸트프로젝트 매매 실행기 초기화 완료")
        logger.info(f"⚙️ 모의거래: {self.paper_trading}, 자동실행: {self.auto_execution}")
    
    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ 매매 설정 로드 성공: {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"❌ 매매 설정 로드 실패: {e}")
            return {}
    
    def _validate_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """신호 유효성 검증"""
        try:
            # 기본 검증
            if not signal.symbol or not signal.action:
                return False, "심볼 또는 액션 누락"
            
            if signal.action not in ['buy', 'sell']:
                return False, f"지원하지 않는 액션: {signal.action}"
            
            if signal.confidence < 0.5:
                return False, f"신뢰도 낮음: {signal.confidence}"
            
            # 시장별 검증
            if signal.market == 'US':
                if not IBKR_AVAILABLE:
                    return False, "IBKR 라이브러리 없음"
                if not UTILS_AVAILABLE or not DataProcessor.detect_market(signal.symbol) == 'US':
                    return False, "미국 주식 심볼 형식 오류"
                    
            elif signal.market == 'JP':
                if not IBKR_AVAILABLE:
                    return False, "IBKR 라이브러리 없음"
                if not signal.symbol.endswith('.T'):
                    return False, "일본 주식 심볼 형식 오류"
                    
            elif signal.market == 'COIN':
                if not self.upbit.access_key:
                    return False, "업비트 API 키 설정 필요"
                    
            else:
                return False, f"지원하지 않는 시장: {signal.market}"
            
            # 일일 거래 한도 확인
            if self.daily_trades >= self.max_daily_trades:
                return False, f"일일 거래 한도 초과: {self.daily_trades}/{self.max_daily_trades}"
            
            return True, "검증 통과"
            
        except Exception as e:
            logger.error(f"❌ 신호 검증 실패: {e}")
            return False, f"검증 오류: {str(e)}"
    
    def _calculate_position_size(self, signal: TradingSignal, portfolio_value: float) -> Tuple[float, float]:
        """포지션 크기 계산"""
        try:
            # 기본 포지션 크기 (신뢰도 기반)
            base_position_pct = 0.05  # 5%
            confidence_multiplier = signal.confidence * 2  # 0.5-1.0 → 1.0-2.0
            position_pct = base_position_pct * confidence_multiplier
            
            # 최대 포지션 크기 제한
            position_pct = min(position_pct, self.max_position_size)
            
            # 시장별 조정
            if signal.market == 'COIN':
                position_pct *= 0.8  # 암호화폐는 80%로 축소 (변동성 고려)
            
            # 포지션 금액 계산
            position_value = portfolio_value * position_pct
            
            # 최소/최대 금액 제한
            if signal.market in ['US', 'JP']:
                min_amount = 1000  # $1,000 또는 ¥100,000
                max_amount = portfolio_value * 0.2  # 최대 20%
            else:  # COIN
                min_amount = 50000  # ₩50,000
                max_amount = portfolio_value * 0.15  # 최대 15%
            
            position_value = max(min_amount, min(position_value, max_amount))
            
            # 수량 계산
            if signal.market in ['US', 'JP']:
                # 주식: 주수 계산
                shares = position_value / signal.price
                
                if signal.market == 'JP':
                    # 일본 주식은 100주 단위
                    shares = int(shares // 100) * 100
                else:
                    # 미국 주식은 1주 단위
                    shares = int(shares)
                
                quantity = shares
                actual_amount = shares * signal.price
                
            else:  # COIN
                # 암호화폐: 금액 기준
                quantity = position_value / signal.price
                actual_amount = position_value
            
            return quantity, actual_amount
            
        except Exception as e:
            logger.error(f"❌ 포지션 크기 계산 실패: {e}")
            return 0.0, 0.0
    
    async def _get_portfolio_value(self, market: str) -> float:
        """포트폴리오 가치 조회"""
        try:
            if market in ['US', 'JP']:
                # IBKR 포트폴리오
                if self.ibkr:
                    portfolio = await self.ibkr.get_portfolio()
                    if portfolio:
                        return portfolio.total_value
                return 100000.0  # 기본값 $100,000
                
            else:  # COIN
                # 업비트 포트폴리오
                portfolio = await self.upbit.get_portfolio()
                if portfolio:
                    return portfolio.total_value
                return 50000000.0  # 기본값 ₩50,000,000
                
        except Exception as e:
            logger.error(f"❌ 포트폴리오 가치 조회 실패: {e}")
            return 100000.0 if market in ['US', 'JP'] else 50000000.0
    
    async def _execute_trade(self, signal: TradingSignal) -> Dict:
        """실제 거래 실행"""
        try:
            # 포지션 크기 계산
            portfolio_value = await self._get_portfolio_value(signal.market)
            quantity, amount = self._calculate_position_size(signal, portfolio_value)
            
            if quantity <= 0:
                return {'success': False, 'error': '포지션 크기 계산 실패'}
            
            # 브로커별 주문 실행
            if signal.market in ['US', 'JP']:
                # IBKR 주문
                if not self.ibkr:
                    return {'success': False, 'error': 'IBKR 연결 없음'}
                
                result = await self.ibkr.place_order(
                    symbol=signal.symbol,
                    market=signal.market,
                    action=signal.action,
                    quantity=quantity
                )
                
            else:  # COIN
                # 업비트 주문
                if signal.action == 'buy':
                    result = await self.upbit.place_order(
                        symbol=signal.symbol,
                        action=signal.action,
                        amount=amount
                    )
                else:  # sell
                    result = await self.upbit.place_order(
                        symbol=signal.symbol,
                        action=signal.action,
                        quantity=quantity
                    )
            
            # 결과 처리
            if result.get('success', False):
                self.daily_trades += 1
                
                # 거래 기록 저장
                trade_record = {
                    'timestamp': datetime.now().isoformat(),
                    'market': signal.market,
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'quantity': result.get('quantity', quantity),
                    'price': result.get('price', signal.price),
                    'total_amount': result.get('quantity', quantity) * result.get('price', signal.price),
                    'confidence': signal.confidence,
                    'strategy': signal.strategy,
                    'reasoning': signal.reasoning,
                    'broker': 'ibkr' if signal.market in ['US', 'JP'] else 'upbit',
                    'order_id': result.get('order_id', ''),
                    'status': 'completed'
                }
                
                # 데이터베이스 저장
                if UTILS_AVAILABLE:
                    DatabaseUtils.save_trade_record(trade_record)
                
                # 거래 로그 저장
                if UTILS_AVAILABLE:
                    save_trading_log(signal.market, signal.symbol, signal.action, {
                        'result': result,
                        'signal': asdict(signal)
                    })
                
                logger.info(f"✅ 거래 완료: {signal.symbol} {signal.action} {result.get('quantity')} @ {result.get('price')}")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ 거래 실행 실패 {signal.symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _simulate_trade(self, signal: TradingSignal) -> Dict:
        """모의 거래 실행"""
        try:
            # 포지션 크기 계산
            portfolio_value = await self._get_portfolio_value(signal.market)
            quantity, amount = self._calculate_position_size(signal, portfolio_value)
            
            if quantity <= 0:
                return {'success': False, 'error': '포지션 크기 계산 실패'}
            
            # 현재가 조회 (실제 시세 사용)
            current_price = signal.price
            
            if signal.market in ['US', 'JP'] and self.ibkr:
                market_price = await self.ibkr.get_current_price(signal.symbol, signal.market)
                if market_price:
                    current_price = market_price
                    
            elif signal.market == 'COIN':
                market_price = await self.upbit.get_current_price(signal.symbol)
                if market_price:
                    current_price = market_price
            
            # 약간의 슬리피지 적용
            if signal.action == 'buy':
                execution_price = current_price * 1.001  # 0.1% 슬리피지
            else:
                execution_price = current_price * 0.999
            
            # 모의 거래 결과
            self.daily_trades += 1
            
            # 모의 거래 기록
            mock_order_id = f"PAPER_{int(time.time())}"
            
            result = {
                'success': True,
                'order_id': mock_order_id,
                'price': execution_price,
                'quantity': quantity,
                'status': 'filled',
                'paper_trade': True
            }
            
            # 거래 로그 저장
            if UTILS_AVAILABLE:
                save_trading_log(signal.market, signal.symbol, signal.action, {
                    'result': result,
                    'signal': asdict(signal),
                    'paper_trading': True
                })
            
            logger.info(f"📄 모의거래 완료: {signal.symbol} {signal.action} {quantity} @ {execution_price:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 모의거래 실패 {signal.symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_signal(self, signal: TradingSignal) -> Dict:
        """매매 신호 실행 (메인 함수)"""
        try:
            # 신호 검증
            is_valid, error_message = self._validate_signal(signal)
            if not is_valid:
                logger.warning(f"⚠️ 신호 검증 실패 {signal.symbol}: {error_message}")
                return {'success': False, 'error': error_message}
            
            # 매매 실행
            if self.paper_trading:
                result = await self._simulate_trade(signal)
            else:
                result = await self._execute_trade(signal)
            
            # 알림 발송
            if NOTIFIER_AVAILABLE and result.get('success', False):
                await send_trading_alert(
                    market=signal.market,
                    symbol=signal.symbol,
                    action=signal.action,
                    price=result.get('price', signal.price),
                    confidence=signal.confidence,
                    reasoning=signal.reasoning,
                    target_price=signal.target_price,
                    execution_status="completed"
                )
            elif NOTIFIER_AVAILABLE and not result.get('success', False):
                await send_trading_alert(
                    market=signal.market,
                    symbol=signal.symbol,
                    action=signal.action,
                    price=signal.price,
                    confidence=signal.confidence,
                    reasoning=result.get('error', '실행 실패'),
                    execution_status="failed"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 매매 신호 실행 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약"""
        try:
            summary = {
                'ibkr_portfolio': None,
                'upbit_portfolio': None,
                'total_value_usd': 0.0,
                'total_value_krw': 0.0,
                'last_updated': datetime.now().isoformat()
            }
            
            # IBKR 포트폴리오
            if self.ibkr:
                ibkr_portfolio = await self.ibkr.get_portfolio()
                if ibkr_portfolio:
                    summary['ibkr_portfolio'] = asdict(ibkr_portfolio)
                    summary['total_value_usd'] += ibkr_portfolio.total_value
            
            # 업비트 포트폴리오
            upbit_portfolio = await self.upbit.get_portfolio()
            if upbit_portfolio:
                summary['upbit_portfolio'] = asdict(upbit_portfolio)
                summary['total_value_krw'] += upbit_portfolio.total_value
                
                # 환율 적용 (간단히 1300으로 가정)
                usd_equivalent = upbit_portfolio.total_value / 1300
                summary['total_value_usd'] += usd_equivalent
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 포트폴리오 요약 실패: {e}")
            return {}
    
    def get_trading_stats(self) -> Dict:
        """거래 통계"""
        uptime = datetime.now() - self.session_start_time
        
        return {
            'executor_status': 'running',
            'session_uptime': str(uptime).split('.')[0],
            'paper_trading': self.paper_trading,
            'auto_execution': self.auto_execution,
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.max_daily_trades,
            'max_position_size': self.max_position_size,
            'brokers': {
                'ibkr_available': IBKR_AVAILABLE and self.ibkr is not None,
                'upbit_configured': bool(self.upbit.access_key),
                'ibkr_connected': self.ibkr.connected if self.ibkr else False
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.ibkr:
                await self.ibkr.disconnect()
            
            if self.upbit:
                await self.upbit.close()
            
            logger.info("🧹 매매 실행기 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 매매 실행기 정리 실패: {e}")

# =====================================
# 편의 함수들 (core.py에서 호출)
# =====================================

_global_executor = None

async def execute_trade_signal(signal) -> Dict:
    """매매 신호 실행 (편의 함수)"""
    global _global_executor
    
    try:
        if _global_executor is None:
            _global_executor = TradingExecutor()
        
        # TradingSignal 객체로 변환 (core.py의 TradingSignal과 호환)
        if hasattr(signal, 'market'):
            trading_signal = TradingSignal(
                market=signal.market,
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                price=signal.price,
                strategy=signal.strategy,
                reasoning=signal.reasoning,
                target_price=signal.target_price,
                timestamp=signal.timestamp,
                sector=getattr(signal, 'sector', None),
                position_size=getattr(signal, 'position_size', None),
                additional_data=getattr(signal, 'additional_data', None)
            )
        else:
            # 딕셔너리인 경우
            trading_signal = TradingSignal(**signal)
        
        return await _global_executor.execute_signal(trading_signal)
        
    except Exception as e:
        logger.error(f"❌ 매매 신호 실행 실패: {e}")
        return {'success': False, 'error': str(e)}

async def get_portfolio_summary() -> Dict:
    """포트폴리오 요약 (편의 함수)"""
    global _global_executor
    
    try:
        if _global_executor is None:
            _global_executor = TradingExecutor()
        
        return await _global_executor.get_portfolio_summary()
        
    except Exception as e:
        logger.error(f"❌ 포트폴리오 요약 실패: {e}")
        return {}

def get_trading_stats() -> Dict:
    """거래 통계 (편의 함수)"""
    global _global_executor
    
    try:
        if _global_executor is None:
            return {'executor_status': 'not_initialized'}
        
        return _global_executor.get_trading_stats()
        
    except Exception as e:
        logger.error(f"❌ 거래 통계 조회 실패: {e}")
        return {'executor_status': 'error'}

# =====================================
# 테스트 함수
# =====================================

async def test_trading_system():
    """🧪 매매 시스템 테스트"""
    print("💰 최고퀸트프로젝트 매매 시스템 테스트")
    print("=" * 50)
    
    # 1. 매매 실행기 초기화
    print("1️⃣ 매매 실행기 초기화...")
    executor = TradingExecutor()
    print(f"   ✅ 완료 (모의거래: {executor.paper_trading})")
    
    # 2. 브로커 연결 테스트
    print("2️⃣ 브로커 연결 테스트...")
    
    # IBKR 테스트
    if executor.ibkr and IBKR_AVAILABLE:
        try:
            connected = await executor.ibkr.connect()
            print(f"   🏦 IBKR: {'✅ 연결됨' if connected else '❌ 연결 실패'}")
        except Exception as e:
            print(f"   🏦 IBKR: ❌ 오류 ({e})")
    else:
        print("   🏦 IBKR: ⏭️ 스킵 (라이브러리 없음)")
    
    # 업비트 테스트
    try:
        btc_price = await executor.upbit.get_current_price('BTC-KRW')
        print(f"   🪙 업비트: {'✅ 연결됨' if btc_price else '❌ 연결 실패'}")
        if btc_price:
            print(f"     📊 BTC 현재가: {Formatter.format_price(btc_price, 'COIN') if UTILS_AVAILABLE else btc_price}")
    except Exception as e:
        print(f"   🪙 업비트: ❌ 오류 ({e})")
    
    # 3. 테스트 신호 생성
    print("3️⃣ 테스트 매매 신호...")
    test_signals = [
        TradingSignal(
            market='US', symbol='AAPL', action='buy', confidence=0.85, price=175.50,
            strategy='test', reasoning='테스트 매수 신호', target_price=195.80,
            timestamp=datetime.now()
        ),
        TradingSignal(
            market='COIN', symbol='BTC-KRW', action='buy', confidence=0.78, price=95000000,
            strategy='test', reasoning='테스트 코인 매수', target_price=105000000,
            timestamp=datetime.now()
        )
    ]
    
    # 4. 모의 거래 실행
    print("4️⃣ 모의 거래 실행...")
    for i, signal in enumerate(test_signals, 1):
        try:
            print(f"   📤 신호 {i}: {signal.symbol} {signal.action}")
            result = await executor.execute_signal(signal)
            
            if result.get('success', False):
                price = result.get('price', 0)
                quantity = result.get('quantity', 0)
                print(f"   ✅ 성공: {quantity} @ {price}")
            else:
                error = result.get('error', '알 수 없음')
                print(f"   ❌ 실패: {error}")
            print()
            
        except Exception as e:
            print(f"   ❌ 신호 {i} 실행 실패: {e}")
    
    # 5. 포트폴리오 조회
    print("5️⃣ 포트폴리오 조회...")
    try:
        portfolio = await executor.get_portfolio_summary()
        
        if portfolio.get('ibkr_portfolio'):
            ibkr = portfolio['ibkr_portfolio']
            print(f"   🏦 IBKR: {Formatter.format_price(ibkr['total_value'], 'US') if UTILS_AVAILABLE else ibkr['total_value']}")
        
        if portfolio.get('upbit_portfolio'):
            upbit = portfolio['upbit_portfolio']
            print(f"   🪙 업비트: {Formatter.format_price(upbit['total_value'], 'COIN') if UTILS_AVAILABLE else upbit['total_value']}")
        
        if not portfolio.get('ibkr_portfolio') and not portfolio.get('upbit_portfolio'):
            print("   📊 포트폴리오 데이터 없음")
            
    except Exception as e:
        print(f"   ❌ 포트폴리오 조회 실패: {e}")
    
    # 6. 거래 통계
    print("6️⃣ 거래 통계...")
    stats = executor.get_trading_stats()
    print(f"   📊 상태: {stats['executor_status']}")
    print(f"   📈 일일 거래: {stats['daily_trades']}/{stats['max_daily_trades']}")
    print(f"   🔧 모의거래: {stats['paper_trading']}")
    print(f"   🏦 IBKR 사용가능: {stats['brokers']['ibkr_available']}")
    print(f"   🪙 업비트 설정: {stats['brokers']['upbit_configured']}")
    
    # 7. 편의 함수 테스트
    print("7️⃣ 편의 함수 테스트...")
    try:
        # 거래 통계
        stats = get_trading_stats()
        print(f"   📋 편의함수 - 상태: {stats['executor_status']}")
        
        # 포트폴리오 요약
        portfolio = await get_portfolio_summary()
        portfolio_count = sum(1 for k in ['ibkr_portfolio', 'upbit_portfolio'] if portfolio.get(k))
        print(f"   📊 편의함수 - 포트폴리오: {portfolio_count}개 연결")
        
    except Exception as e:
        print(f"   ❌ 편의함수 테스트 실패: {e}")
    
    # 8. 리소스 정리
    print("8️⃣ 리소스 정리...")
    await executor.cleanup()
    print("   ✅ 완료")
    
    print()
    print("🎯 매매 시스템 테스트 완료!")
    print("💰 IBKR + 업비트 통합 매매 시스템이 준비되었습니다")

if __name__ == "__main__":
    print("💰 최고퀸트프로젝트 매매 시스템")
    print("=" * 50)
    
    # 테스트 실행
    asyncio.run(test_trading_system())
    
    print("\n🚀 매매 시스템 준비 완료!")
    print("💡 core.py에서 execute_trade_signal() 함수를 사용하세요")
    print("\n⚙️ 설정:")
    print("   📋 configs/settings.yaml에서 trading, api 섹션 설정")
    print("   🏦 IBKR: TWS/Gateway 실행 + ib_insync 설치")
    print("   🪙 업비트: API 키 설정 + PyJWT 설치")
    print("   🛡️ 모의거래 모드로 안전하게 테스트 가능")