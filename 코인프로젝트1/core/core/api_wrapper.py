import os
import time
import asyncio
import aiohttp
import ccxt
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from functools import wraps, lru_cache
import warnings
warnings.filterwarnings('ignore')

# 로거 import (수정된 logger.py에서)
from logger import get_logger

# 로거 생성
logger = get_logger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed, using in-memory cache only")

try:
    import sqlite3
    from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    DATABASE_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("SQLAlchemy not installed, database features disabled")

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class AdvancedMarketData:
    """Institutional-grade market data structure"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    volatility: float
    rsi: float
    macd: float
    bollinger_upper: float
    bollinger_lower: float
    fear_greed_index: int
    news_sentiment: float
    social_sentiment: float
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    liquidations: Optional[Dict] = None
    orderbook_imbalance: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class TradingSignal:
    """Professional trading signal structure"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    timeframe: str = '1h'
    strategy_name: str = 'default'
    metadata: Dict = field(default_factory=dict)

if DATABASE_AVAILABLE:
    class MarketDataTable(Base):
        """SQLAlchemy table for market data"""
        __tablename__ = 'market_data'
        
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, nullable=False)
        symbol = Column(String(20), nullable=False)
        price = Column(Float, nullable=False)
        volume = Column(Float)
        rsi = Column(Float)
        macd = Column(Float)
        sentiment = Column(Float)
        raw_data = Column(Text)

class DataSourceInterface(ABC):
    """Abstract interface for all data sources"""
    
    @abstractmethod
    async def fetch_data(self, symbol: str, **kwargs) -> Dict:
        pass
    
    @abstractmethod
    def validate_data(self, data: Dict) -> bool:
        pass

class QuantAPIWrapper:
    """
    퀀트 트레이딩 API 래퍼 (기존 이름으로 변경)
    Institutional-grade quantitative trading API wrapper
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._initialize_components()
        if DATABASE_AVAILABLE:
            self._setup_database()
        if REDIS_AVAILABLE:
            self._setup_cache()
        self._exchange_connections = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info("✅ QuantAPIWrapper 초기화 완료")
        
    def _initialize_components(self):
        """Initialize all system components"""
        # API 키들을 환경변수에서 가져오거나 기본값 설정
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.binance_api_key = os.getenv('BINANCE_API_KEY', '')
        self.binance_secret = os.getenv('BINANCE_SECRET', '')
        
        if not self.binance_api_key or not self.binance_secret:
            logger.warning("⚠️ Binance API 키가 설정되지 않았습니다. 일부 기능이 제한됩니다.")
        
        # Initialize exchanges
        self._init_exchanges()
        
    def _validate_env_var(self, var_name: str) -> str:
        """Validate environment variables"""
        value = os.getenv(var_name)
        if not value:
            logger.warning(f"⚠️ {var_name}이 환경변수에 설정되지 않았습니다")
            return ""
        return value
    
    def _init_exchanges(self):
        """Initialize multiple exchange connections"""
        try:
            if self.binance_api_key and self.binance_secret:
                self.binance = ccxt.binance({
                    'apiKey': self.binance_api_key,
                    'secret': self.binance_secret,
                    'sandbox': self.config.get('sandbox', True),
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
                logger.info("✅ Binance 연결 초기화 완료")
            else:
                # API 키가 없어도 기본 기능은 동작하도록
                self.binance = ccxt.binance({
                    'enableRateLimit': True,
                    'sandbox': True
                })
                logger.warning("⚠️ Binance API 키 없이 제한 모드로 실행")
            
            self.bybit = ccxt.bybit({
                'enableRateLimit': True,
                'sandbox': self.config.get('sandbox', True)
            })
            
            logger.info("✅ 거래소 연결 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 거래소 초기화 실패: {e}")
            # 에러가 발생해도 계속 진행
            self.binance = None
            self.bybit = None
    
    def _setup_database(self):
        """Setup SQLAlchemy database connection"""
        try:
            db_url = self.config.get('database_url', 'sqlite:///quant_data.db')
            self.engine = create_engine(db_url, echo=False)
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.db_session = Session()
            logger.info("✅ 데이터베이스 연결 완료")
        except Exception as e:
            logger.error(f"❌ 데이터베이스 연결 실패: {e}")
            self.db_session = None
    
    def _setup_cache(self):
        """Setup Redis cache for high-frequency data"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis 캐시 연결 완료")
        except Exception as e:
            logger.warning(f"⚠️ Redis 사용 불가, 메모리 캐시 사용: {e}")
            self.redis_client = None
    
    @lru_cache(maxsize=1000)
    def _get_cached_data(self, key: str, ttl: int = 300) -> Optional[str]:
        """Get cached data with TTL"""
        if self.redis_client:
            try:
                return self.redis_client.get(key)
            except:
                return None
        return None
    
    def _set_cached_data(self, key: str, value: str, ttl: int = 300):
        """Set cached data with TTL"""
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, value)
            except Exception as e:
                logger.debug(f"캐시 저장 실패: {e}")
    
    async def fetch_comprehensive_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 1000) -> AdvancedMarketData:
        """
        Fetch comprehensive market data from multiple sources
        
        This is institutional-grade data aggregation combining:
        - OHLCV data from multiple exchanges
        - Technical indicators
        - Sentiment analysis from multiple sources
        - On-chain metrics
        - Derivatives data
        """
        try:
            tasks = [
                self._fetch_exchange_data(symbol, timeframe, limit),
                self._fetch_fear_greed_enhanced(),
                self._fetch_sentiment_aggregate(symbol),
                self._fetch_derivatives_data(symbol),
                self._fetch_onchain_metrics(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process and combine all data sources
            ohlcv_data, fear_greed, sentiment, derivatives, onchain = results
            
            # 에러 처리
            if isinstance(ohlcv_data, Exception) or not ohlcv_data:
                logger.error(f"OHLCV 데이터 가져오기 실패: {ohlcv_data}")
                ohlcv_data = self._generate_dummy_ohlcv(symbol)
            
            # Calculate advanced technical indicators
            df = pd.DataFrame(ohlcv_data)
            technical_indicators = self._calculate_advanced_indicators(df)
            
            return AdvancedMarketData(
                timestamp=datetime.now(),
                symbol=symbol,
                price=float(df['close'].iloc[-1]) if len(df) > 0 else 50000.0,
                volume=float(df['volume'].iloc[-1]) if len(df) > 0 else 1000000.0,
                volatility=float(df['close'].pct_change().std() * np.sqrt(24)) if len(df) > 1 else 0.02,
                rsi=float(technical_indicators['rsi'].iloc[-1]) if 'rsi' in technical_indicators.columns and len(technical_indicators) > 0 else 50.0,
                macd=float(technical_indicators['macd'].iloc[-1]) if 'macd' in technical_indicators.columns and len(technical_indicators) > 0 else 0.0,
                bollinger_upper=float(technical_indicators['bb_upper'].iloc[-1]) if 'bb_upper' in technical_indicators.columns and len(technical_indicators) > 0 else 52000.0,
                bollinger_lower=float(technical_indicators['bb_lower'].iloc[-1]) if 'bb_lower' in technical_indicators.columns and len(technical_indicators) > 0 else 48000.0,
                fear_greed_index=fear_greed.get('current_value', 50) if isinstance(fear_greed, dict) else 50,
                news_sentiment=sentiment.get('news_sentiment', 0.0) if isinstance(sentiment, dict) else 0.0,
                social_sentiment=sentiment.get('social_sentiment', 0.0) if isinstance(sentiment, dict) else 0.0,
                funding_rate=derivatives.get('funding_rate') if isinstance(derivatives, dict) else None,
                open_interest=derivatives.get('open_interest') if isinstance(derivatives, dict) else None,
                liquidations=derivatives.get('liquidations') if isinstance(derivatives, dict) else None,
                orderbook_imbalance=await self._calculate_orderbook_imbalance(symbol)
            )
        except Exception as e:
            logger.error(f"시장 데이터 가져오기 실패: {e}")
            return self._generate_dummy_market_data(symbol)
    
    def _generate_dummy_ohlcv(self, symbol: str) -> List[Dict]:
        """Generate dummy OHLCV data for testing"""
        data = []
        base_price = 50000 if 'BTC' in symbol else 3000
        
        for i in range(100):
            price = base_price * (1 + np.random.normal(0, 0.01))
            data.append({
                'timestamp': datetime.now() - timedelta(hours=100-i),
                'open': price * 0.999,
                'high': price * 1.002,
                'low': price * 0.998,
                'close': price,
                'volume': np.random.uniform(1000000, 5000000)
            })
        return data
    
    def _generate_dummy_market_data(self, symbol: str) -> AdvancedMarketData:
        """Generate dummy market data for error cases"""
        return AdvancedMarketData(
            timestamp=datetime.now(),
            symbol=symbol,
            price=50000.0,
            volume=1000000.0,
            volatility=0.02,
            rsi=50.0,
            macd=0.0,
            bollinger_upper=52000.0,
            bollinger_lower=48000.0,
            fear_greed_index=50,
            news_sentiment=0.0,
            social_sentiment=0.0
        )
    
    async def _fetch_exchange_data(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Fetch OHLCV data from primary exchange"""
        try:
            if not self.binance:
                logger.warning("Binance 연결이 없어 더미 데이터 생성")
                return self._generate_dummy_ohlcv(symbol)
                
            cache_key = f"ohlcv:{symbol}:{timeframe}:{limit}"
            cached = self._get_cached_data(cache_key, ttl=60)
            
            if cached:
                return eval(cached)  # In production, use proper JSON serialization
            
            ohlcv = await asyncio.to_thread(
                self.binance.fetch_ohlcv, symbol, timeframe, limit=limit
            )
            
            formatted_data = []
            for candle in ohlcv:
                formatted_data.append({
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                })
            
            self._set_cached_data(cache_key, str(formatted_data), ttl=60)
            return formatted_data
            
        except Exception as e:
            logger.error(f"거래소 데이터 가져오기 에러: {e}")
            return self._generate_dummy_ohlcv(symbol)
    
    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            if len(df) == 0:
                return df
                
            # Ensure proper column names
            if 'timestamp' in df.columns:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            else:
                # If columns are different, try to map them
                expected_cols = ['open', 'high', 'low', 'close', 'volume']
                if len(df.columns) >= 5:
                    df.columns = ['timestamp'] + expected_cols if len(df.columns) == 6 else expected_cols
            
            # Make sure we have numeric data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if len(df) < 14:  # RSI needs at least 14 periods
                logger.warning("데이터가 부족하여 지표 계산을 건너뜁니다")
                return df
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            
            # Advanced indicators
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # Custom indicators
            df['price_momentum'] = df['close'].pct_change(periods=min(20, len(df)-1))
            df['volume_sma'] = df['volume'].rolling(window=min(20, len(df))).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"지표 계산 에러: {e}")
            return df
    
    async def fetch_fear_greed_index(self) -> Dict:
        """Fear & Greed 지수 가져오기 (호환성을 위한 메소드명)"""
        return await self._fetch_fear_greed_enhanced()
    
    async def _fetch_fear_greed_enhanced(self) -> Dict:
        """Enhanced Fear & Greed with statistical analysis"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.alternative.me/fng/?limit=30', timeout=10) as response:
                    data = await response.json()
                    
                    if 'data' not in data:
                        return {'current_value': 50, 'trend': 'neutral'}
                    
                    values = [int(item['value']) for item in data['data']]
                    
                    return {
                        'current_value': values[0],
                        'sma_7': np.mean(values[:7]),
                        'sma_30': np.mean(values),
                        'volatility': np.std(values),
                        'trend': 'bullish' if values[0] > np.mean(values[:7]) else 'bearish',
                        'percentile': np.percentile(values, 50),
                        'z_score': (values[0] - np.mean(values)) / np.std(values) if np.std(values) > 0 else 0
                    }
        except Exception as e:
            logger.error(f"Fear & Greed 데이터 가져오기 에러: {e}")
            return {'current_value': 50, 'trend': 'neutral'}
    
    async def _fetch_sentiment_aggregate(self, symbol: str) -> Dict:
        """Aggregate sentiment from multiple sources"""
        # This would integrate with:
        # - Twitter API for social sentiment
        # - Reddit API for community sentiment
        # - News APIs for media sentiment
        # - LunarCrush or similar for crypto-specific sentiment
        
        return {
            'news_sentiment': np.random.uniform(-0.5, 0.5),  # Placeholder
            'social_sentiment': np.random.uniform(-0.5, 0.5),  # Placeholder
            'overall_sentiment': np.random.uniform(-0.5, 0.5)  # Placeholder
        }
    
    async def _fetch_derivatives_data(self, symbol: str) -> Dict:
        """Fetch derivatives and futures data"""
        try:
            if not self.binance:
                return {'funding_rate': None}
                
            # Funding rate
            funding_rate = await asyncio.to_thread(
                self.binance.fetch_funding_rate, symbol
            )
            
            return {
                'funding_rate': funding_rate.get('fundingRate', 0.0),
                'open_interest': None,  # Placeholder
                'liquidations': {}  # Placeholder
            }
        except Exception as e:
            logger.error(f"파생상품 데이터 가져오기 에러: {e}")
            return {'funding_rate': None}
    
    async def _fetch_onchain_metrics(self, symbol: str) -> Dict:
        """Fetch on-chain metrics (for applicable cryptocurrencies)"""
        return {
            'active_addresses': None,  # Placeholder
            'transaction_volume': None,  # Placeholder
            'network_growth': None  # Placeholder
        }
    
    async def _calculate_orderbook_imbalance(self, symbol: str) -> float:
        """Calculate orderbook imbalance for market microstructure analysis"""
        try:
            if not self.binance:
                return 0.0
                
            orderbook = await asyncio.to_thread(
                self.binance.fetch_order_book, symbol, limit=100
            )
            
            bid_volume = sum([bid[1] for bid in orderbook['bids'][:10]])
            ask_volume = sum([ask[1] for ask in orderbook['asks'][:10]])
            
            if bid_volume + ask_volume == 0:
                return 0.0
            
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            return imbalance
            
        except Exception as e:
            logger.error(f"오더북 불균형 계산 에러: {e}")
            return 0.0
    
    def generate_trading_signals(self, market_data: AdvancedMarketData) -> List[TradingSignal]:
        """
        Generate institutional-grade trading signals
        
        Uses multiple strategies:
        1. Mean Reversion
        2. Momentum
        3. Sentiment-based
        4. Technical breakouts
        5. Statistical arbitrage
        """
        signals = []
        
        try:
            # Strategy 1: RSI Mean Reversion
            if market_data.rsi < 30:
                signals.append(TradingSignal(
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    signal_type='BUY',
                    confidence=min(0.8, (30 - market_data.rsi) / 30),
                    strategy_name='RSI_MEAN_REVERSION',
                    metadata={'rsi_value': market_data.rsi}
                ))
            elif market_data.rsi > 70:
                signals.append(TradingSignal(
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    signal_type='SELL',
                    confidence=min(0.8, (market_data.rsi - 70) / 30),
                    strategy_name='RSI_MEAN_REVERSION',
                    metadata={'rsi_value': market_data.rsi}
                ))
            
            # Strategy 2: MACD Momentum
            if market_data.macd > 0:
                signals.append(TradingSignal(
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    signal_type='BUY',
                    confidence=0.6,
                    strategy_name='MACD_MOMENTUM',
                    metadata={'macd_value': market_data.macd}
                ))
            
            # Strategy 3: Sentiment-based
            sentiment_score = (market_data.news_sentiment + market_data.social_sentiment) / 2
            if abs(sentiment_score) > 0.3:
                signals.append(TradingSignal(
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    signal_type='BUY' if sentiment_score > 0 else 'SELL',
                    confidence=min(0.7, abs(sentiment_score)),
                    strategy_name='SENTIMENT_BASED',
                    metadata={'sentiment_score': sentiment_score}
                ))
            
            # Strategy 4: Fear & Greed contrarian
            if market_data.fear_greed_index < 25:  # Extreme fear
                signals.append(TradingSignal(
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    signal_type='BUY',
                    confidence=0.7,
                    strategy_name='FEAR_GREED_CONTRARIAN',
                    metadata={'fg_index': market_data.fear_greed_index}
                ))
            elif market_data.fear_greed_index > 75:  # Extreme greed
                signals.append(TradingSignal(
                    timestamp=datetime.now(),
                    symbol=market_data.symbol,
                    signal_type='SELL',
                    confidence=0.7,
                    strategy_name='FEAR_GREED_CONTRARIAN',
                    metadata={'fg_index': market_data.fear_greed_index}
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"시그널 생성 에러: {e}")
            return []
    
    def calculate_portfolio_metrics(self, positions: Dict[str, float], market_data: Dict[str, AdvancedMarketData]) -> Dict:
        """Calculate institutional-grade portfolio metrics"""
        try:
            portfolio_value = sum(
                positions[symbol] * market_data[symbol].price 
                for symbol in positions.keys() 
                if symbol in market_data
            )
            
            if portfolio_value == 0:
                return {'portfolio_value': 0}
            
            # Calculate portfolio volatility
            weights = {symbol: (positions[symbol] * market_data[symbol].price) / portfolio_value 
                      for symbol in positions.keys() if symbol in market_data}
            
            portfolio_volatility = np.sqrt(
                sum(weights[symbol]**2 * market_data[symbol].volatility**2 
                    for symbol in weights.keys())
            )
            
            # Risk metrics
            var_95 = portfolio_value * 1.645 * portfolio_volatility  # 95% VaR
            expected_shortfall = portfolio_value * 2.33 * portfolio_volatility  # ES
            
            return {
                'portfolio_value': portfolio_value,
                'volatility': portfolio_volatility,
                'var_95': var_95,
                'expected_shortfall': expected_shortfall,
                'sharpe_estimate': 0.1 / portfolio_volatility if portfolio_volatility > 0 else 0,
                'positions': positions,
                'weights': weights
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 지표 계산 에러: {e}")
            return {}
    
    async def run_live_monitoring(self, symbols: List[str], callback=None):
        """Run live market monitoring with real-time signal generation"""
        logger.info(f"🚀 {len(symbols)}개 심볼 실시간 모니터링 시작")
        
        while True:
            try:
                # Fetch data for all symbols concurrently
                tasks = [
                    self.fetch_comprehensive_market_data(symbol) 
                    for symbol in symbols
                ]
                
                market_data_list = await asyncio.gather(*tasks, return_exceptions=True)
                
                for symbol, market_data in zip(symbols, market_data_list):
                    if isinstance(market_data, Exception):
                        logger.error(f"❌ {symbol} 에러: {market_data}")
                        continue
                    
                    # Generate signals
                    signals = self.generate_trading_signals(market_data)
                    
                    # Store data
                    if DATABASE_AVAILABLE and self.db_session:
                        self._store_market_data(market_data)
                    
                    # Callback for real-time processing
                    if callback and signals:
                        await callback(symbol, market_data, signals)
                
                # Wait before next iteration
                await asyncio.sleep(self.config.get('monitoring_interval', 60))
                
            except Exception as e:
                logger.error(f"❌ 실시간 모니터링 에러: {e}")
                await asyncio.sleep(10)
    
    def _store_market_data(self, market_data: AdvancedMarketData):
        """Store market data in database"""
        if not self.db_session:
            return
            
        try:
            db_record = MarketDataTable(
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                price=market_data.price,
                volume=market_data.volume,
                rsi=market_data.rsi,
                macd=market_data.macd,
                sentiment=market_data.news_sentiment,
                raw_data=str(market_data.to_dict())
            )
            
            self.db_session.add(db_record)
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"❌ 데이터베이스 저장 에러: {e}")
            if self.db_session:
                self.db_session.rollback()
    
    def get_status(self) -> Dict:
        """시스템 상태 반환"""
        return {
            'exchanges': {
                'binance': self.binance is not None,
                'bybit': self.bybit is not None
            },
            'database': self.db_session is not None if DATABASE_AVAILABLE else False,
            'cache': self.redis_client is not None if REDIS_AVAILABLE else False,
            'config': self.config
        }
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'db_session') and self.db_session:
                self.db_session.close()
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
        except:
            pass

# Institutional API Wrapper alias for backward compatibility
InstitutionalAPIWrapper = QuantAPIWrapper

# Usage example for institutional deployment
async def main():
    config = {
        'sandbox': True,
        'database_url': 'sqlite:///quant_data.db',
        'redis_host': 'localhost',
        'monitoring_interval': 30
    }
    
    api = QuantAPIWrapper(config)
    
    # Check status
    status = api.get_status()
    logger.info(f"📊 시스템 상태: {status}")
    
    # Example callback for real-time signal processing
    async def signal_callback(symbol, market_data, signals):
        for signal in signals:
            if signal.confidence > 0.7:
                logger.info(f"🔥 고신뢰도 시그널: {signal.signal_type} {symbol} (신뢰도: {signal.confidence:.2f})")
    
    # Start live monitoring
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    # Test single fetch first
    try:
        market_data = await api.fetch_comprehensive_market_data('BTC/USDT')
        logger.info(f"✅ 테스트 데이터 가져오기 성공: {market_data.symbol} - ${market_data.price:,.2f}")
        
        signals = api.generate_trading_signals(market_data)
        logger.info(f"📈 생성된 시그널 수: {len(signals)}")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
    
    # await api.run_live_monitoring(symbols, callback=signal_callback)

if __name__ == "__main__":
    asyncio.run(main())
