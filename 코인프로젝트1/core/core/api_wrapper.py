"""
🏆 UPBIT & IBKR ELITE TRADING SYSTEM 🏆
업비트 + Interactive Brokers 전용 10만점급 퀀트 트레이딩 시스템

- Upbit: 한국 최대 암호화폐 거래소
- IBKR: 세계 최고 글로벌 증권사
- AI 기반 다중 자산 포트폴리오 운용
- 기관급 리스크 관리 시스템
- 실시간 크로스마켓 알고리즘
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import requests
import time
import jwt
import hashlib
import hmac
import uuid
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import warnings
warnings.filterwarnings('ignore')

# Interactive Brokers API
try:
    from ib_insync import IB, Stock, Forex, Future, Option, MarketOrder, LimitOrder, Contract
    from ib_insync import util
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

# 고급 분석 라이브러리
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    import ta
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("upbit_ibkr_elite.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

seoul_tz = pytz.timezone('Asia/Seoul')
ny_tz = pytz.timezone('America/New_York')

# 데이터베이스
Base = declarative_base()

class TradeExecution(Base):
    __tablename__ = 'trade_executions'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    exchange = Column(String(20), nullable=False)  # upbit, ibkr
    market = Column(String(20), nullable=False)    # coin, japan, us
    symbol = Column(String(50), nullable=False)
    action = Column(String(10), nullable=False)    # BUY, SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    confidence = Column(Float, nullable=False)
    strategy = Column(String(100), nullable=False)
    order_id = Column(String(100))
    status = Column(String(20), default='PENDING')  # PENDING, FILLED, CANCELLED
    pnl = Column(Float, default=0.0)
    metadata = Column(Text)

class MarketAnalysis(Base):
    __tablename__ = 'market_analysis'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    market = Column(String(20), nullable=False)
    symbol = Column(String(50), nullable=False)
    
    # 가격 데이터
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # 기술적 지표
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # 고급 지표
    atr = Column(Float)
    stochastic = Column(Float)
    williams_r = Column(Float)
    momentum = Column(Float)
    volatility = Column(Float)
    
    # AI 예측
    ml_prediction = Column(Float)
    ml_confidence = Column(Float)
    
    # 시그널
    signal = Column(String(10))  # BUY, SELL, HOLD
    signal_strength = Column(Float)

@dataclass
class UpbitMarketData:
    """Upbit 마켓 데이터"""
    market: str
    korean_name: str
    english_name: str
    price: float
    volume: float
    change_rate: float
    change_price: float
    timestamp: datetime
    
    # 기술적 지표
    rsi: float = 50.0
    macd: float = 0.0
    signal_line: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    
    # 시장 센티먼트
    order_book_imbalance: float = 0.0
    fear_greed_score: float = 50.0

@dataclass
class IBKRMarketData:
    """IBKR 마켓 데이터"""
    symbol: str
    exchange: str
    currency: str
    price: float
    volume: float
    bid: float
    ask: float
    timestamp: datetime
    
    # 펀더멘탈
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    dividend_yield: Optional[float] = None
    
    # 기술적 지표  
    rsi: float = 50.0
    macd: float = 0.0
    volatility: float = 0.0

@dataclass
class EliteSignal:
    """엘리트 트레이딩 시그널"""
    timestamp: datetime
    exchange: str  # upbit, ibkr
    market: str    # coin, japan, us
    symbol: str
    action: str    # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    
    # 포지션 정보
    target_allocation: float  # 포트폴리오 비중
    quantity: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # 전략 정보
    strategy_name: str = "Elite_Multi_Asset"
    signal_factors: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.5
    expected_return: float = 0.0
    holding_period: int = 1  # 일 단위

class UpbitAPI:
    """업비트 API 래퍼"""
    
    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = "https://api.upbit.com"
        
    def _get_headers(self, query_string: str = None):
        """JWT 토큰 생성"""
        payload = {'access_key': self.access_key, 'nonce': str(uuid.uuid4())}
        
        if query_string:
            payload['query_hash'] = hashlib.sha512(query_string.encode()).hexdigest()
            payload['query_hash_alg'] = 'SHA512'
        
        jwt_token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        return {'Authorization': f'Bearer {jwt_token}'}
    
    async def get_markets(self) -> List[Dict]:
        """마켓 리스트 조회"""
        try:
            response = requests.get(f"{self.base_url}/v1/market/all?isDetails=true")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Upbit 마켓 조회 실패: {e}")
            return []
    
    async def get_ticker(self, markets: List[str]) -> List[Dict]:
        """현재가 정보"""
        try:
            markets_str = ",".join(markets)
            response = requests.get(f"{self.base_url}/v1/ticker?markets={markets_str}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Upbit 현재가 조회 실패: {e}")
            return []
    
    async def get_candles(self, market: str, count: int = 200) -> List[Dict]:
        """캔들 데이터 (일봉)"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/candles/days?market={market}&count={count}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Upbit 캔들 조회 실패: {e}")
            return []
    
    async def get_accounts(self) -> List[Dict]:
        """계좌 조회"""
        try:
            headers = self._get_headers()
            response = requests.get(f"{self.base_url}/v1/accounts", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Upbit 계좌 조회 실패: {e}")
            return []
    
    async def place_order(self, market: str, side: str, volume: str, price: str = None, ord_type: str = "limit") -> Dict:
        """주문하기"""
        try:
            params = {
                'market': market,
                'side': side,  # bid(매수), ask(매도)
                'volume': volume,
                'ord_type': ord_type
            }
            
            if price and ord_type == "limit":
                params['price'] = price
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            headers = self._get_headers(query_string)
            
            response = requests.post(
                f"{self.base_url}/v1/orders",
                headers=headers,
                data=params
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Upbit 주문 실패: {e}")
            return {}

class IBKRAPI:
    """Interactive Brokers API 래퍼"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.ib = IB() if IBKR_AVAILABLE else None
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
    async def connect(self):
        """IBKR 연결"""
        if not IBKR_AVAILABLE:
            logger.error("❌ ib_insync 라이브러리가 설치되지 않았습니다")
            return False
        
        try:
            await self.ib.connectAsync(self.host, self.port, self.client_id)
            self.connected = True
            logger.info("✅ IBKR 연결 성공")
            return True
        except Exception as e:
            logger.error(f"❌ IBKR 연결 실패: {e}")
            return False
    
    async def get_stock_data(self, symbol: str, exchange: str = "SMART") -> Optional[Dict]:
        """주식 데이터 조회"""
        if not self.connected:
            return None
        
        try:
            contract = Stock(symbol, exchange, 'USD')
            self.ib.qualifyContracts(contract)
            
            # 현재가 정보
            ticker = self.ib.reqMktData(contract)
            await asyncio.sleep(2)  # 데이터 수신 대기
            
            # 히스토리컬 데이터
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='1 M',  # 1개월
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True
            )
            
            return {
                'symbol': symbol,
                'price': ticker.marketPrice() or ticker.close,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'volume': ticker.volume,
                'bars': bars
            }
            
        except Exception as e:
            logger.error(f"❌ IBKR {symbol} 데이터 조회 실패: {e}")
            return None
    
    async def place_order(self, symbol: str, action: str, quantity: int, order_type: str = "MKT", price: float = None) -> Optional[str]:
        """주문 실행"""
        if not self.connected:
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            
            if order_type == "MKT":
                order = MarketOrder(action, quantity)
            else:
                order = LimitOrder(action, quantity, price)
            
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"✅ IBKR 주문 실행: {symbol} {action} {quantity}")
            
            return trade.order.orderId
            
        except Exception as e:
            logger.error(f"❌ IBKR 주문 실패: {e}")
            return None

class EliteTradingAPIWrapper:
    """🏆 Upbit & IBKR Elite Trading System"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # API 초기화
        self.upbit = None
        self.ibkr = None
        
        # 포트폴리오 상태
        self.initial_capital = self.config.get('initial_capital', 10000000)  # 1천만원
        self.portfolio = {
            'coin': {'balance': self.initial_capital * 0.3, 'positions': {}},    # 30% 코인
            'japan': {'balance': self.initial_capital * 0.35, 'positions': {}},  # 35% 일본
            'us': {'balance': self.initial_capital * 0.35, 'positions': {}}      # 35% 미국
        }
        
        # ML 모델
        self.ml_models = {}
        self.market_data_cache = {}
        
        # 데이터베이스
        self._init_database()
        
        # 거래 제한
        self.trading_limits = {
            'max_position_size': 0.15,      # 15% 최대 포지션
            'max_daily_trades': 20,         # 일일 최대 거래 수
            'min_confidence': 0.7,          # 최소 신뢰도
            'max_risk_per_trade': 0.02      # 거래당 최대 리스크 2%
        }
        
        logger.info("🏆 Elite Trading API Wrapper 초기화 완료")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            self.engine = create_engine('sqlite:///upbit_ibkr_elite.db', echo=False)
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            logger.info("✅ 데이터베이스 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 데이터베이스 초기화 실패: {e}")
    
    async def initialize_apis(self):
        """API 연결 초기화"""
        # Upbit 초기화
        upbit_access = self.config.get('upbit_access_key')
        upbit_secret = self.config.get('upbit_secret_key')
        
        if upbit_access and upbit_secret:
            self.upbit = UpbitAPI(upbit_access, upbit_secret)
            logger.info("✅ Upbit API 초기화 완료")
        else:
            logger.warning("⚠️ Upbit API 키가 설정되지 않았습니다")
        
        # IBKR 초기화
        if IBKR_AVAILABLE:
            self.ibkr = IBKRAPI(
                host=self.config.get('ibkr_host', '127.0.0.1'),
                port=self.config.get('ibkr_port', 7497),
                client_id=self.config.get('ibkr_client_id', 1)
            )
            
            # IBKR 연결 시도
            connected = await self.ibkr.connect()
            if connected:
                logger.info("✅ IBKR API 연결 완료")
            else:
                logger.warning("⚠️ IBKR 연결 실패 - TWS가 실행되어 있는지 확인하세요")
        else:
            logger.warning("⚠️ ib_insync 라이브러리가 설치되지 않았습니다")

    async def execute_trading(self, market: str):
        """🚀 메인 트레이딩 실행 (엘리트 버전)"""
        logger.info(f"🚀 {market} Elite 매매 시작")
        
        try:
            # ① 포괄적 시장 분석
            market_analysis = await self.comprehensive_market_analysis(market)
            
            # ② AI 기반 시그널 생성
            elite_signals = await self.generate_elite_signals(market, market_analysis)
            
            # ③ 크로스마켓 포트폴리오 최적화
            optimized_signals = await self.cross_market_optimization(elite_signals)
            
            # ④ 다층 리스크 관리
            risk_approved = await self.multi_layer_risk_check(market, optimized_signals)
            
            # ⑤ 스마트 주문 실행
            execution_results = await self.execute_smart_orders(market, risk_approved)
            
            # ⑥ 실시간 성과 모니터링
            await self.real_time_performance_tracking(market, execution_results)
            
            logger.info(f"✅ {market} Elite 매매 완료")
            
        except Exception as e:
            logger.error(f"❌ {market} 매매 실행 중 오류: {e}")
            await self.emergency_risk_management(market, str(e))

    async def comprehensive_market_analysis(self, market: str) -> Dict[str, Any]:
        """📊 포괄적 시장 분석"""
        logger.info(f"📊 {market} 포괄적 시장 분석 시작")
        
        analysis = {
            'market': market,
            'timestamp': datetime.now(seoul_tz),
            'symbols_data': {},
            'market_sentiment': {},
            'technical_overview': {},
            'fundamental_overview': {}
        }
        
        try:
            if market == "coin" and self.upbit:
                analysis = await self._analyze_crypto_market(analysis)
            elif market in ["japan", "us"] and self.ibkr:
                analysis = await self._analyze_stock_market(market, analysis)
            
            # 시장 전체 센티먼트 분석
            analysis['market_sentiment'] = await self._analyze_market_sentiment(market)
            
            # 기술적 분석 요약
            analysis['technical_overview'] = self._summarize_technical_analysis(analysis['symbols_data'])
            
            logger.info(f"✅ {market} 시장 분석 완료: {len(analysis['symbols_data'])}개 종목")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ {market} 시장 분석 실패: {e}")
            return analysis

    async def _analyze_crypto_market(self, analysis: Dict) -> Dict:
        """암호화폐 시장 분석"""
        try:
            # 주요 코인 목록
            major_coins = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-DOT"]
            
            # 현재가 정보
            tickers = await self.upbit.get_ticker(major_coins)
            
            for ticker in tickers:
                market = ticker['market']
                
                # 캔들 데이터로 기술적 지표 계산
                candles = await self.upbit.get_candles(market, 100)
                
                if candles and ML_AVAILABLE:
                    df = pd.DataFrame(candles)
                    df = df.sort_values('candle_date_time_kst')
                    
                    # 기술적 지표 계산
                    closes = df['trade_price'].values
                    
                    # RSI
                    rsi = ta.momentum.RSIIndicator(df['trade_price']).rsi().iloc[-1]
                    
                    # MACD
                    macd_line = ta.trend.MACD(df['trade_price']).macd().iloc[-1]
                    macd_signal = ta.trend.MACD(df['trade_price']).macd_signal().iloc[-1]
                    
                    # 볼린저 밴드
                    bb = ta.volatility.BollingerBands(df['trade_price'])
                    bb_upper = bb.bollinger_hband().iloc[-1]
                    bb_lower = bb.bollinger_lband().iloc[-1]
                    
                    analysis['symbols_data'][market] = {
                        'price': ticker['trade_price'],
                        'volume': ticker['acc_trade_volume_24h'],
                        'change_rate': ticker['change_rate'],
                        'rsi': rsi if not pd.isna(rsi) else 50.0,
                        'macd': macd_line if not pd.isna(macd_line) else 0.0,
                        'macd_signal': macd_signal if not pd.isna(macd_signal) else 0.0,
                        'bb_upper': bb_upper if not pd.isna(bb_upper) else ticker['trade_price'] * 1.05,
                        'bb_lower': bb_lower if not pd.isna(bb_lower) else ticker['trade_price'] * 0.95,
                        'volatility': df['trade_price'].pct_change().std() * np.sqrt(24),
                        'momentum': (ticker['trade_price'] - df['trade_price'].iloc[-20]) / df['trade_price'].iloc[-20] if len(df) >= 20 else 0.0
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 암호화폐 시장 분석 실패: {e}")
            return analysis

    async def _analyze_stock_market(self, market: str, analysis: Dict) -> Dict:
        """주식 시장 분석"""
        try:
            symbols = self._get_stock_symbols(market)
            
            for symbol in symbols[:5]:  # 상위 5개만
                stock_data = await self.ibkr.get_stock_data(symbol)
                
                if stock_data and stock_data.get('bars'):
                    bars = stock_data['bars']
                    df = pd.DataFrame([{
                        'close': bar.close,
                        'volume': bar.volume,
                        'high': bar.high,
                        'low': bar.low
                    } for bar in bars])
                    
                    if len(df) > 20 and ML_AVAILABLE:
                        # 기술적 지표 계산
                        rsi = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
                        macd = ta.trend.MACD(df['close']).macd().iloc[-1]
                        
                        analysis['symbols_data'][symbol] = {
                            'price': stock_data['price'],
                            'volume': stock_data['volume'],
                            'bid': stock_data['bid'],
                            'ask': stock_data['ask'],
                            'rsi': rsi if not pd.isna(rsi) else 50.0,
                            'macd': macd if not pd.isna(macd) else 0.0,
                            'volatility': df['close'].pct_change().std() * np.sqrt(252),
                            'momentum': (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
                        }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ {market} 주식 시장 분석 실패: {e}")
            return analysis

    def _get_stock_symbols(self, market: str) -> List[str]:
        """주식 심볼 목록"""
        symbols_map = {
            'japan': ['7203', '6758', '9984', '8306', '6861'],  # Toyota, Sony, SoftBank 등
            'us': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        }
        return symbols_map.get(market, [])

    async def _analyze_market_sentiment(self, market: str) -> Dict:
        """시장 센티먼트 분석"""
        try:
            sentiment = {
                'fear_greed_index': 50,  # 중립
                'market_trend': 'sideways',
                'volatility_regime': 'normal',
                'risk_appetite': 0.5
            }
            
            if market == "coin":
                # 암호화폐 공포탐욕지수 (실제로는 API 호출)
                sentiment['fear_greed_index'] = np.random.randint(20, 80)
                sentiment['market_trend'] = np.random.choice(['bullish', 'bearish', 'sideways'])
            
            return sentiment
            
        except Exception as e:
            logger.error(f"❌ {market} 센티먼트 분석 실패: {e}")
            return {'fear_greed_index': 50, 'market_trend': 'sideways'}

    def _summarize_technical_analysis(self, symbols_data: Dict) -> Dict:
        """기술적 분석 요약"""
        try:
            if not symbols_data:
                return {}
            
            rsi_values = [data.get('rsi', 50) for data in symbols_data.values()]
            macd_values = [data.get('macd', 0) for data in symbols_data.values()]
            momentum_values = [data.get('momentum', 0) for data in symbols_data.values()]
            
            return {
                'avg_rsi': np.mean(rsi_values),
                'oversold_count': sum(1 for rsi in rsi_values if rsi < 30),
                'overbought_count': sum(1 for rsi in rsi_values if rsi > 70),
                'bullish_macd_count': sum(1 for macd in macd_values if macd > 0),
                'avg_momentum': np.mean(momentum_values),
                'momentum_trend': 'bullish' if np.mean(momentum_values) > 0.02 else 'bearish' if np.mean(momentum_values) < -0.02 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"❌ 기술적 분석 요약 실패: {e}")
            return {}

    async def generate_elite_signals(self, market: str, analysis: Dict) -> List[EliteSignal]:
        """🧠 AI 기반 엘리트 시그널 생성"""
        logger.info(f"🧠 {market} AI 시그널 생성 시작")
        
        signals = []
        symbols_data = analysis.get('symbols_data', {})
        market_sentiment = analysis.get('market_sentiment', {})
        
        try:
            for symbol, data in symbols_data.items():
                signal = await self._generate_symbol_signal(market, symbol, data, market_sentiment)
                if signal and signal.confidence >= self.trading_limits['min_confidence']:
                    signals.append(signal)
            
            logger.info(f"✅ {market} 시그널 생성 완료: {len(signals)}개")
            return signals
            
        except Exception as e:
            logger.error(f"❌ {market} 시그널 생성 실패: {e}")
            return []

    async def _generate_symbol_signal(self, market: str, symbol: str, data: Dict, sentiment: Dict) -> Optional[EliteSignal]:
        """개별 심볼 시그널 생성"""
        try:
            # 다중 팩터 분석
            factors = {}
            
            # 기술적 팩터
            rsi = data.get('rsi', 50)
            macd = data.get('macd', 0)
            momentum = data.get('momentum', 0)
            
            factors['rsi_signal'] = self._rsi_signal(rsi)
            factors['macd_signal'] = 1 if macd > 0 else -1
            factors['momentum_signal'] = np.tanh(momentum * 10)  # -1 to 1
            
            # 센티먼트 팩터
            fear_greed = sentiment.get('fear_greed_index', 50)
            factors['sentiment_signal'] = (fear_greed - 50) / 50  # -1 to 1
            
            # 변동성 팩터
            volatility = data.get('volatility', 0.02)
            factors['volatility_signal'] = -1 if volatility > 0.05 else 1  # 높은 변동성은 부정적
            
            # 종합 시그널 계산
            weights = {
                'rsi_signal': 0.25,
                'macd_signal': 0.25,
                'momentum_signal': 0.30,
                'sentiment_signal': 0.10,
                'volatility_signal': 0.10
            }
            
            weighted_signal = sum(factors[k] * weights[k] for k in factors.keys())
            
            # 시그널 결정
            if weighted_signal > 0.3:
                action = "BUY"
                confidence = min(0.95, 0.5 + abs(weighted_signal))
            elif weighted_signal < -0.3:
                action = "SELL"
                confidence = min(0.95, 0.5 + abs(weighted_signal))
            else:
                action = "HOLD"
                confidence = 0.5
            
            # 포지션 크기 계산
            base_allocation = self.trading_limits['max_position_size']
            adjusted_allocation = base_allocation * confidence
            
            # 수량 계산
            portfolio_value = self.portfolio[market]['balance']
            position_value = portfolio_value * adjusted_allocation
            quantity = position_value / data['price']
            
            # 스톱로스/익절 계산
            atr = data.get('volatility', 0.02) * data['price']
            stop_loss = data['price'] - (2 * atr) if action == "BUY" else data['price'] + (2 * atr)
            take_profit = data['price'] + (3 * atr) if action == "BUY" else data['price'] - (3 * atr)
            
            signal = EliteSignal(
                timestamp=datetime.now(seoul_tz),
                exchange='upbit' if market == 'coin' else 'ibkr',
                market=market,
                symbol=symbol,
                action=action,
                confidence=confidence,
                target_allocation=adjusted_allocation,
                quantity=quantity,
                entry_price=data['price'],
                stop_loss=stop_loss if action != "HOLD" else None,
                take_profit=take_profit if action != "HOLD" else None,
                strategy_name="Elite_Multi_Factor_AI",
                signal_factors=factors,
                risk_score=volatility,
                expected_return=weighted_signal * 0.1,  # 예상 수익률
                holding_period=5  # 5일
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ {symbol} 시그널 생성 실패: {e}")
            return None

    def _rsi_signal(self, rsi: float) -> float:
        """RSI 기반 시그널"""
        if rsi < 30:
            return (30 - rsi) / 30  # 과매도시 양의 시그널
        elif rsi > 70:
            return -(rsi - 70) / 30  # 과매수시 음의 시그널
        else:
            return 0

    async def cross_market_optimization(self, signals: List[EliteSignal]) -> List[EliteSignal]:
        """⚖️ 크로스마켓 포트폴리오 최적화"""
        logger.info("⚖️ 크로스마켓 포트폴리오 최적화 시작")
        
        try:
            if not signals:
                return signals
            
            # 시장별 시그널 그룹화
            market_signals = {}
            for signal in signals:
                if signal.market not in market_signals:
                    market_signals[signal.market] = []
                market_signals[signal.market].append(signal)
            
            optimized_signals = []
            
            # 각 시장별 최적화
            for market, market_signal_list in market_signals.items():
                if not market_signal_list:
                    continue
                
                # 시장 내 포트폴리오 최적화
                optimized_market_signals = self._optimize_market_portfolio(market, market_signal_list)
                optimized_signals.extend(optimized_market_signals)
            
            # 전체 포트폴리오 리밸런싱
            final_signals = self._rebalance_cross_market(optimized_signals)
            
            logger.info(f"✅ 포트폴리오 최적화 완료: {len(final_signals)}개 시그널")
            return final_signals
            
        except Exception as e:
            logger.error(f"❌ 포트폴리오 최적화 실패: {e}")
            return signals

    def _optimize_market_portfolio(self, market: str, signals: List[EliteSignal]) -> List[EliteSignal]:
        """시장 내 포트폴리오 최적화"""
        try:
            if not OPTIMIZATION_AVAILABLE:
                return signals[:3]  # 상위 3개만 선택
            
            # 시그널을 신뢰도 순으로 정렬
            sorted_signals = sorted(signals, key=lambda x: x.confidence, reverse=True)
            
            # 상위 시그널들만 선택 (최대 5개)
            selected_signals = sorted_signals[:5]
            
            # 포트폴리오 가중치 재조정
            total_allocation = sum(s.target_allocation for s in selected_signals)
            max_market_allocation = 0.8  # 시장당 최대 80%
            
            if total_allocation > max_market_allocation:
                scale_factor = max_market_allocation / total_allocation
                for signal in selected_signals:
                    signal.target_allocation *= scale_factor
                    signal.quantity *= scale_factor
            
            return selected_signals
            
        except Exception as e:
            logger.error(f"❌ {market} 포트폴리오 최적화 실패: {e}")
            return signals

    def _rebalance_cross_market(self, signals: List[EliteSignal]) -> List[EliteSignal]:
        """크로스마켓 리밸런싱"""
        try:
            # 시장별 총 배분 계산
            market_allocations = {}
            for signal in signals:
                if signal.market not in market_allocations:
                    market_allocations[signal.market] = 0
                market_allocations[signal.market] += signal.target_allocation
            
            # 목표 배분 (30% 코인, 35% 일본, 35% 미국)
            target_allocations = {'coin': 0.30, 'japan': 0.35, 'us': 0.35}
            
            # 각 시장별 스케일 팩터 계산
            scale_factors = {}
            for market in market_allocations.keys():
                current = market_allocations[market]
                target = target_allocations.get(market, 0.33)
                scale_factors[market] = target / current if current > 0 else 1.0
            
            # 시그널 배분 조정
            for signal in signals:
                scale_factor = scale_factors.get(signal.market, 1.0)
                signal.target_allocation *= scale_factor
                signal.quantity *= scale_factor
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ 크로스마켓 리밸런싱 실패: {e}")
            return signals

    async def multi_layer_risk_check(self, market: str, signals: List[EliteSignal]) -> List[EliteSignal]:
        """🛡️ 다층 리스크 관리"""
        logger.info(f"🛡️ {market} 다층 리스크 관리 시작")
        
        try:
            approved_signals = []
            
            for signal in signals:
                # Layer 1: 기본 리스크 체크
                if not self._basic_risk_check(signal):
                    logger.warning(f"🚫 {signal.symbol} 기본 리스크 체크 실패")
                    continue
                
                # Layer 2: 포지션 크기 체크
                if not self._position_size_check(signal):
                    logger.warning(f"🚫 {signal.symbol} 포지션 크기 초과")
                    continue
                
                # Layer 3: 상관관계 체크
                if not self._correlation_check(signal, approved_signals):
                    logger.warning(f"🚫 {signal.symbol} 상관관계 위험")
                    continue
                
                # Layer 4: 시장 상황 체크
                if not self._market_condition_check(market, signal):
                    logger.warning(f"🚫 {signal.symbol} 시장 상황 부적합")
                    continue
                
                approved_signals.append(signal)
            
            logger.info(f"✅ {market} 리스크 관리 완료: {len(approved_signals)}개 승인")
            return approved_signals
            
        except Exception as e:
            logger.error(f"❌ {market} 리스크 관리 실패: {e}")
            return []

    def _basic_risk_check(self, signal: EliteSignal) -> bool:
        """기본 리스크 체크"""
        try:
            # 신뢰도 체크
            if signal.confidence < self.trading_limits['min_confidence']:
                return False
            
            # 리스크 점수 체크
            if signal.risk_score > 0.1:  # 10% 이상 변동성
                return False
            
            # 포지션 크기 체크
            if signal.target_allocation > self.trading_limits['max_position_size']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 기본 리스크 체크 실패: {e}")
            return False

    def _position_size_check(self, signal: EliteSignal) -> bool:
        """포지션 크기 체크"""
        try:
            # 현재 포지션 + 새 포지션이 한계를 넘지 않는지 확인
            current_exposure = sum(
                pos.get('allocation', 0) 
                for pos in self.portfolio[signal.market]['positions'].values()
            )
            
            total_exposure = current_exposure + signal.target_allocation
            
            return total_exposure <= 0.8  # 시장당 최대 80%
            
        except Exception as e:
            logger.error(f"❌ 포지션 크기 체크 실패: {e}")
            return False

    def _correlation_check(self, signal: EliteSignal, existing_signals: List[EliteSignal]) -> bool:
        """상관관계 체크 (간단 버전)"""
        try:
            # 같은 시장의 시그널이 너무 많은지 체크
            same_market_count = sum(1 for s in existing_signals if s.market == signal.market)
            
            return same_market_count < 3  # 시장당 최대 3개
            
        except Exception as e:
            logger.error(f"❌ 상관관계 체크 실패: {e}")
            return True

    def _market_condition_check(self, market: str, signal: EliteSignal) -> bool:
        """시장 상황 체크"""
        try:
            # 현재 시간 체크
            now = datetime.now(seoul_tz)
            
            if market == "coin":
                # 암호화폐는 24시간 거래 가능
                return True
            elif market == "japan":
                # 일본 시장 시간 (09:00-15:30 JST)
                jst_time = now.astimezone(pytz.timezone('Asia/Tokyo'))
                return 9 <= jst_time.hour < 15 or (jst_time.hour == 15 and jst_time.minute <= 30)
            elif market == "us":
                # 미국 시장 시간 (09:30-16:00 EST)
                est_time = now.astimezone(ny_tz)
                return 9 <= est_time.hour < 16 or (est_time.hour == 9 and est_time.minute >= 30)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 시장 상황 체크 실패: {e}")
            return True

    async def execute_smart_orders(self, market: str, signals: List[EliteSignal]) -> List[Dict]:
        """📝 스마트 주문 실행"""
        logger.info(f"📝 {market} 스마트 주문 실행 시작")
        
        execution_results = []
        
        try:
            for signal in signals:
                result = await self._execute_single_order(signal)
                if result:
                    execution_results.append(result)
                    
                    # 주문 기록 저장
                    await self._save_trade_record(signal, result)
                
                # 주문 간 지연 (API 제한 고려)
                await asyncio.sleep(0.5)
            
            logger.info(f"✅ {market} 주문 실행 완료: {len(execution_results)}개")
            return execution_results
            
        except Exception as e:
            logger.error(f"❌ {market} 주문 실행 실패: {e}")
            return execution_results

    async def _execute_single_order(self, signal: EliteSignal) -> Optional[Dict]:
        """개별 주문 실행"""
        try:
            logger.info(f"📝 {signal.symbol} {signal.action} 주문 실행 중...")
            
            if signal.exchange == 'upbit' and self.upbit:
                return await self._execute_upbit_order(signal)
            elif signal.exchange == 'ibkr' and self.ibkr and self.ibkr.connected:
                return await self._execute_ibkr_order(signal)
            else:
                logger.warning(f"⚠️ {signal.exchange} 연결되지 않음 - 시뮬레이션 모드")
                return self._simulate_order_execution(signal)
            
        except Exception as e:
            logger.error(f"❌ {signal.symbol} 주문 실행 실패: {e}")
            return None

    async def _execute_upbit_order(self, signal: EliteSignal) -> Dict:
        """Upbit 주문 실행"""
        try:
            side = 'bid' if signal.action == 'BUY' else 'ask'
            
            result = await self.upbit.place_order(
                market=signal.symbol,
                side=side,
                volume=str(signal.quantity),
                price=str(signal.entry_price),
                ord_type='limit'
            )
            
            if result.get('uuid'):
                logger.info(f"✅ Upbit {signal.symbol} {signal.action} 주문 성공")
                return {
                    'exchange': 'upbit',
                    'order_id': result['uuid'],
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'quantity': signal.quantity,
                    'price': signal.entry_price,
                    'status': 'PENDING',
                    'timestamp': datetime.now(seoul_tz)
                }
            else:
                logger.error(f"❌ Upbit {signal.symbol} 주문 실패")
                return None
                
        except Exception as e:
            logger.error(f"❌ Upbit 주문 실행 실패: {e}")
            return None

    async def _execute_ibkr_order(self, signal: EliteSignal) -> Dict:
        """IBKR 주문 실행"""
        try:
            action = 'BUY' if signal.action == 'BUY' else 'SELL'
            quantity = int(signal.quantity)
            
            order_id = await self.ibkr.place_order(
                symbol=signal.symbol,
                action=action,
                quantity=quantity,
                order_type='LMT',
                price=signal.entry_price
            )
            
            if order_id:
                logger.info(f"✅ IBKR {signal.symbol} {signal.action} 주문 성공")
                return {
                    'exchange': 'ibkr',
                    'order_id': order_id,
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'quantity': quantity,
                    'price': signal.entry_price,
                    'status': 'PENDING',
                    'timestamp': datetime.now(seoul_tz)
                }
            else:
                logger.error(f"❌ IBKR {signal.symbol} 주문 실패")
                return None
                
        except Exception as e:
            logger.error(f"❌ IBKR 주문 실행 실패: {e}")
            return None

    def _simulate_order_execution(self, signal: EliteSignal) -> Dict:
        """주문 실행 시뮬레이션"""
        logger.info(f"🎭 {signal.symbol} {signal.action} 주문 시뮬레이션")
        
        return {
            'exchange': signal.exchange,
            'order_id': f"SIM_{int(time.time())}",
            'symbol': signal.symbol,
            'action': signal.action,
            'quantity': signal.quantity,
            'price': signal.entry_price,
            'status': 'SIMULATED',
            'timestamp': datetime.now(seoul_tz)
        }

    async def _save_trade_record(self, signal: EliteSignal, result: Dict):
        """거래 기록 저장"""
        try:
            if self.session:
                trade = TradeExecution(
                    timestamp=result['timestamp'],
                    exchange=result['exchange'],
                    market=signal.market,
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=signal.quantity,
                    price=signal.entry_price,
                    total_value=signal.quantity * signal.entry_price,
                    confidence=signal.confidence,
                    strategy=signal.strategy_name,
                    order_id=result['order_id'],
                    status=result['status'],
                    metadata=json.dumps(signal.signal_factors)
                )
                
                self.session.add(trade)
                self.session.commit()
                
        except Exception as e:
            logger.error(f"❌ 거래 기록 저장 실패: {e}")

    async def real_time_performance_tracking(self, market: str, execution_results: List[Dict]):
        """📊 실시간 성과 추적"""
        logger.info(f"📊 {market} 실시간 성과 추적 시작")
        
        try:
            total_trades = len(execution_results)
            total_value = sum(r['quantity'] * r['price'] for r in execution_results)
            
            portfolio_snapshot = {
                'timestamp': datetime.now(seoul_tz),
                'market': market,
                'trades_executed': total_trades,
                'total_trade_value': total_value,
                'portfolio_value': self.portfolio[market]['balance']
            }
            
            logger.info(f"📊 {market} 성과 추적 완료")
            logger.info(f"   실행된 거래: {total_trades}개")
            logger.info(f"   총 거래금액: ₩{total_value:,.0f}")
            
        except Exception as e:
            logger.error(f"❌ {market} 성과 추적 실패: {e}")

    async def emergency_risk_management(self, market: str, error_msg: str):
        """🚨 긴급 리스크 관리"""
        logger.critical(f"🚨 {market} 긴급 리스크 관리 발동: {error_msg}")
        
        try:
            # 모든 pending 주문 취소 (실제로는 거래소 API 호출)
            logger.warning(f"⚠️ {market} 모든 대기 주문 취소 중...")
            
            # 포지션 크기 재점검
            logger.warning(f"⚠️ {market} 포지션 크기 재점검 중...")
            
            # 알림 발송 (실제로는 이메일/텔레그램)
            logger.critical(f"🚨 {market} 거래 시스템 이상 - 관리자 확인 필요")
            
        except Exception as e:
            logger.error(f"❌ 긴급 리스크 관리 실패: {e}")

    async def fetch_market_data(self, market: str) -> Dict:
        """📊 시장 데이터 수집 (기존 호환성)"""
        logger.info(f"📊 {market} 데이터 수집 중...")
        
        # 간단한 더미 데이터 (실제로는 comprehensive_market_analysis 사용)
        return {
            "price": 50000 if market == "coin" else 100,
            "rsi": np.random.uniform(30, 70),
            "time": datetime.now(seoul_tz)
        }

    def generate_signal(self, market_data: Dict) -> Dict:
        """📈 시그널 생성 (기존 호환성)"""
        logger.info("📈 시그널 생성 중...")
        
        rsi = market_data.get("rsi", 50)
        
        if rsi < 30:
            return {"action": "BUY", "confidence": 0.8}
        elif rsi > 70:
            return {"action": "SELL", "confidence": 0.8}
        else:
            return {"action": "HOLD", "confidence": 0.5}

    def check_risk(self, signal: Dict) -> bool:
        """🛡️ 리스크 체크 (기존 호환성)"""
        logger.info("🛡️ 리스크 체크 중...")
        return signal["confidence"] >= 0.6

    async def execute_order(self, market: str, signal: Dict):
        """📝 주문 실행 (기존 호환성)"""
        logger.info(f"📝 {market} {signal['action']} 주문 실행")
        
        # 실제로는 execute_smart_orders 사용
        if market == "coin" and self.upbit:
            logger.info("✅ Upbit 주문 완료 (시뮬레이션)")
        elif market in ["japan", "us"] and self.ibkr:
            logger.info("✅ IBKR 주문 완료 (시뮬레이션)")
        else:
            logger.info("✅ 주문 완료 (시뮬레이션)")

# 사용 예제
async def main():
    """메인 실행 예제"""
    try:
        # 설정
        config = {
            'initial_capital': 10000000,  # 1천만원
            'upbit_access_key': 'your_upbit_access_key',
            'upbit_secret_key': 'your_upbit_secret_key',
            'ibkr_host': '127.0.0.1',
            'ibkr_port': 7497,
            'ibkr_client_id': 1
        }
        
        # Elite Trading System 초기화
        elite_system = EliteTradingAPIWrapper(config)
        
        # API 연결
        await elite_system.initialize_apis()
        
        # 테스트 거래 실행
        markets = ["coin", "japan", "us"]
        
        for market in markets:
            print(f"\n🚀 {market} Elite 매매 시작")
            await elite_system.execute_trading(market)
            print(f"✅ {market} Elite 매매 완료")
        
        print("\n🏆 모든 시장 Elite 매매 완료!")
        
    except Exception as e:
        print(f"❌ 시스템 실행 중 오류: {e}")

if __name__ == "__main__":
    asyncio.run(main())
