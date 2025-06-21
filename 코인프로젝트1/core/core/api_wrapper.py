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
import redis
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

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

class InstitutionalAPIWrapper:
    """Institutional-grade quantitative trading API wrapper"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialize_components()
        self._setup_database()
        self._setup_cache()
        self._exchange_connections = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    def _initialize_components(self):
        """Initialize all system components"""
        self.openai_api_key = self._validate_env_var('OPENAI_API_KEY')
        self.binance_api_key = self._validate_env_var('BINANCE_API_KEY')
        self.binance_secret = self._validate_env_var('BINANCE_SECRET')
        
        # Initialize exchanges
        self._init_exchanges()
        
    def _validate_env_var(self, var_name: str) -> str:
        """Validate environment variables"""
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"{var_name} not found in environment variables")
        return value
    
    def _init_exchanges(self):
        """Initialize multiple exchange connections"""
        try:
            self.binance = ccxt.binance({
                'apiKey': self.binance_api_key,
                'secret': self.binance_secret,
                'sandbox': self.config.get('sandbox', True),
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            
            self.bybit = ccxt.bybit({
                'enableRateLimit': True,
                'sandbox': self.config.get('sandbox', True)
            })
            
            logger.info("Exchange connections initialized successfully")
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
            raise
    
    def _setup_database(self):
        """Setup SQLAlchemy database connection"""
        db_url = self.config.get('database_url', 'sqlite:///quant_data.db')
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        logger.info("Database connection established")
    
    def _setup_cache(self):
        """Setup Redis cache for high-frequency data"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis cache connection established")
        except:
            logger.warning("Redis not available, using in-memory cache")
            self.redis_client = None
    
    @lru_cache(maxsize=1000)
    def _get_cached_data(self, key: str, ttl: int = 300) -> Optional[str]:
        """Get cached data with TTL"""
        if self.redis_client:
            return self.redis_client.get(key)
        return None
    
    def _set_cached_data(self, key: str, value: str, ttl: int = 300):
        """Set cached data with TTL"""
        if self.redis_client:
            self.redis_client.setex(key, ttl, value)
    
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
        
        # Calculate advanced technical indicators
        df = pd.DataFrame(ohlcv_data)
        technical_indicators = self._calculate_advanced_indicators(df)
        
        return AdvancedMarketData(
            timestamp=datetime.now(),
            symbol=symbol,
            price=float(df['close'].iloc[-1]),
            volume=float(df['volume'].iloc[-1]),
            volatility=float(df['close'].pct_change().std() * np.sqrt(24)),
            rsi=float(technical_indicators['rsi'].iloc[-1]),
            macd=float(technical_indicators['macd'].iloc[-1]),
            bollinger_upper=float(technical_indicators['bb_upper'].iloc[-1]),
            bollinger_lower=float(technical_indicators['bb_lower'].iloc[-1]),
            fear_greed_index=fear_greed.get('current_value', 50),
            news_sentiment=sentiment.get('news_sentiment', 0.0),
            social_sentiment=sentiment.get('social_sentiment', 0.0),
            funding_rate=derivatives.get('funding_rate'),
            open_interest=derivatives.get('open_interest'),
            liquidations=derivatives.get('liquidations'),
            orderbook_imbalance=await self._calculate_orderbook_imbalance(symbol)
        )
    
    async def _fetch_exchange_data(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Fetch OHLCV data from primary exchange"""
        try:
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
            logger.error(f"Error fetching exchange data: {e}")
            return []
    
    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            # Ensure proper column names
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
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
            df['price_momentum'] = df['close'].pct_change(periods=20)
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    async def _fetch_fear_greed_enhanced(self) -> Dict:
        """Enhanced Fear & Greed with statistical analysis"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.alternative.me/fng/?limit=30') as response:
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
                        'z_score': (values[0] - np.mean(values)) / np.std(values)
                    }
        except Exception as e:
            logger.error(f"Fear & Greed fetch error: {e}")
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
            # Funding rate
            funding_rate = await asyncio.to_thread(
                self.binance.fetch_funding_rate, symbol
            )
            
            # Open interest (if available)
            # This would require specific exchange endpoints
            
            return {
                'funding_rate': funding_rate.get('fundingRate', 0.0),
                'open_interest': None,  # Placeholder
                'liquidations': {}  # Placeholder
            }
        except Exception as e:
            logger.error(f"Derivatives data fetch error: {e}")
            return {'funding_rate': None}
    
    async def _fetch_onchain_metrics(self, symbol: str) -> Dict:
        """Fetch on-chain metrics (for applicable cryptocurrencies)"""
        # This would integrate with:
        # - Glassnode API
        # - CoinMetrics API
        # - Messari API
        # for metrics like:
        # - Network hash rate
        # - Active addresses
        # - Transaction volume
        # - MVRV ratio
        # - etc.
        
        return {
            'active_addresses': None,  # Placeholder
            'transaction_volume': None,  # Placeholder
            'network_growth': None  # Placeholder
        }
    
    async def _calculate_orderbook_imbalance(self, symbol: str) -> float:
        """Calculate orderbook imbalance for market microstructure analysis"""
        try:
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
            logger.error(f"Orderbook imbalance calculation error: {e}")
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
            logger.error(f"Signal generation error: {e}")
            return []
    
    def calculate_portfolio_metrics(self, positions: Dict[str, float], market_data: Dict[str, AdvancedMarketData]) -> Dict:
        """Calculate institutional-grade portfolio metrics"""
        try:
            portfolio_value = sum(
                positions[symbol] * market_data[symbol].price 
                for symbol in positions.keys() 
                if symbol in market_data
            )
            
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
            logger.error(f"Portfolio metrics calculation error: {e}")
            return {}
    
    async def run_live_monitoring(self, symbols: List[str], callback=None):
        """Run live market monitoring with real-time signal generation"""
        logger.info(f"Starting live monitoring for {len(symbols)} symbols")
        
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
                        logger.error(f"Error for {symbol}: {market_data}")
                        continue
                    
                    # Generate signals
                    signals = self.generate_trading_signals(market_data)
                    
                    # Store data
                    self._store_market_data(market_data)
                    
                    # Callback for real-time processing
                    if callback and signals:
                        await callback(symbol, market_data, signals)
                
                # Wait before next iteration
                await asyncio.sleep(self.config.get('monitoring_interval', 60))
                
            except Exception as e:
                logger.error(f"Live monitoring error: {e}")
                await asyncio.sleep(10)
    
    def _store_market_data(self, market_data: AdvancedMarketData):
        """Store market data in database"""
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
            logger.error(f"Database storage error: {e}")
            self.db_session.rollback()
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'db_session'):
                self.db_session.close()
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
        except:
            pass

# Usage example for institutional deployment
async def main():
    config = {
        'sandbox': True,
        'database_url': 'postgresql://user:pass@localhost/quant_db',
        'redis_host': 'localhost',
        'monitoring_interval': 30
    }
    
    api = InstitutionalAPIWrapper(config)
    
    # Example callback for real-time signal processing
    async def signal_callback(symbol, market_data, signals):
        for signal in signals:
            if signal.confidence > 0.7:
                logger.info(f"HIGH CONFIDENCE SIGNAL: {signal.signal_type} {symbol} at {signal.confidence:.2f}")
    
    # Start live monitoring
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    await api.run_live_monitoring(symbols, callback=signal_callback)

if __name__ == "__main__":
    asyncio.run(main())
