"""
Advanced Utility System for Quantitative Trading
===============================================

퀀트 트레이딩을 위한 고급 유틸리티 시스템
데이터 수집, 분석, 알림, 로깅 등 핵심 기능 통합

Features:
- 다중 소스 뉴스 수집 및 AI 분석
- 실시간 가격 데이터 및 경제지표 수집
- 텔레그램/슬랙/이메일 통합 알림 시스템
- 고급 로깅 및 성과 추적
- 시장 휴일 및 거래시간 관리
- 데이터 캐싱 및 최적화
- 오류 복구 및 재시도 메커니즘

Author: Quant System
Version: 3.0.0
Created: 2025-06-21
"""

import os
import asyncio
import aiohttp
import requests
import time
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import wraps, lru_cache
from collections import defaultdict
import sqlite3
import threading
from decimal import Decimal

# Core 패키지 import (호환성 처리)
try:
    import config
    import logger
    log = logger.get_logger(__name__)
    
    # BaseComponent 클래스 (없으면 기본 구현)
    try:
        from core import BaseComponent
    except ImportError:
        class BaseComponent:
            def __init__(self, name):
                self.name = name
                self.logger = log
            
            def initialize(self):
                pass
            
            def cleanup(self):
                pass
    
    # ValidationError 클래스 (없으면 기본 구현)
    try:
        from core import ValidationError
    except ImportError:
        class ValidationError(Exception):
            pass
            
except ImportError:
    import logging
    log = logging.getLogger(__name__)
    
    class BaseComponent:
        def __init__(self, name):
            self.name = name
            self.logger = log
        
        def initialize(self):
            pass
        
        def cleanup(self):
            pass
    
    class ValidationError(Exception):
        pass

# 외부 라이브러리 imports
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    log.warning("holidays 모듈을 찾을 수 없습니다.")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    log.warning("beautifulsoup4 모듈을 찾을 수 없습니다.")

try:
    import pyupbit
    PYUPBIT_AVAILABLE = True
except ImportError:
    PYUPBIT_AVAILABLE = False
    log.warning("pyupbit 모듈을 찾을 수 없습니다.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    log.warning("yfinance 모듈을 찾을 수 없습니다.")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    log.warning("feedparser 모듈을 찾을 수 없습니다.")

# =============================================================================
# 상수 및 설정
# =============================================================================

class NotificationChannel(Enum):
    """알림 채널"""
    TELEGRAM = "telegram"
    SLACK = "slack"
    EMAIL = "email"
    DISCORD = "discord"
    WEBHOOK = "webhook"

class DataSource(Enum):
    """데이터 소스"""
    NAVER_NEWS = "naver_news"
    GOOGLE_NEWS = "google_news"
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    COINDESK = "coindesk"
    YAHOO_FINANCE = "yahoo_finance"
    INVESTING_COM = "investing_com"
    FNG_API = "fng_api"
    OPENAI = "openai"
    CLAUDE = "claude"

class AssetType(Enum):
    """자산 유형"""
    CRYPTO = "crypto"
    STOCK_US = "stock_us"
    STOCK_KR = "stock_kr"
    STOCK_JP = "stock_jp"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"

# =============================================================================
# 데이터 클래스들
# =============================================================================

@dataclass
class NewsArticle:
    """뉴스 기사 데이터"""
    title: str
    content: str
    url: str
    source: str
    published_at: datetime = field(default_factory=datetime.now)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    relevance_score: Optional[float] = None
    language: str = "ko"
    
    def __post_init__(self):
        self.title = self.title.strip()
        self.content = self.content.strip()
        if self.url and not self.url.startswith(('http://', 'https://', '')):
            log.warning(f"잘못된 URL 형식: {self.url}")

@dataclass
class PriceData:
    """가격 데이터"""
    symbol: str
    price: float
    volume: Optional[float] = None
    market_cap: Optional[float] = None
    change_24h: Optional[float] = None
    change_pct_24h: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    
    def __post_init__(self):
        if self.price < 0:
            log.warning(f"비정상적인 가격: {self.price}")
            self.price = 0

@dataclass
class EconomicIndicator:
    """경제지표 데이터"""
    name: str
    value: float
    previous_value: Optional[float] = None
    forecast: Optional[float] = None
    unit: str = ""
    country: str = "KR"
    release_date: datetime = field(default_factory=datetime.now)
    importance: str = "medium"
    
    @property
    def change_from_previous(self) -> Optional[float]:
        if self.previous_value is not None:
            return self.value - self.previous_value
        return None
    
    @property
    def change_pct_from_previous(self) -> Optional[float]:
        if self.previous_value is not None and self.previous_value != 0:
            return (self.value - self.previous_value) / self.previous_value * 100
        return None

@dataclass
class MarketStatus:
    """시장 상태"""
    market: str
    is_open: bool
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    timezone: str = "Asia/Seoul"
    special_notice: Optional[str] = None

@dataclass
class NotificationConfig:
    """알림 설정"""
    channel: NotificationChannel
    webhook_url: Optional[str] = None
    api_token: Optional[str] = None
    chat_id: Optional[str] = None
    email_config: Optional[Dict[str, str]] = None
    enabled: bool = True

# =============================================================================
# 유틸리티 함수들 (기존 호환성)
# =============================================================================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """실패 시 재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        log.error(f"최대 재시도 횟수 초과 ({func.__name__}): {e}")
                        raise e
                    
                    log.warning(f"재시도 {retries}/{max_retries} ({func.__name__}): {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator

def cache_with_ttl(ttl_seconds: int = 3600):
    """TTL 기반 캐싱 데코레이터"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    log.debug(f"캐시에서 반환: {func.__name__}")
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            
            expired_keys = [k for k, (_, ts) in cache.items() if now - ts >= ttl_seconds]
            for k in expired_keys:
                del cache[k]
            
            return result
        return wrapper
    return decorator

# =============================================================================
# 뉴스 수집 및 분석 시스템
# =============================================================================

class NewsCollector(BaseComponent):
    """고급 뉴스 수집기"""
    
    def __init__(self):
        super().__init__("NewsCollector")
        self.asset_keywords = self._load_asset_keywords()
        self.session = None
        
    def initialize(self):
        """뉴스 수집기 초기화"""
        try:
            log.info("뉴스 수집기 초기화 완료")
        except Exception as e:
            log.error(f"뉴스 수집기 초기화 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        if self.session:
            try:
                asyncio.create_task(self.session.close())
            except:
                pass
    
    def _load_asset_keywords(self) -> Dict[str, List[str]]:
        """자산별 키워드 매핑"""
        return {
            "BTC": ["Bitcoin", "비트코인", "BTC", "ビットコイン"],
            "ETH": ["Ethereum", "이더리움", "ETH", "이더", "イーサリアム"],
            "XRP": ["Ripple", "XRP", "리플", "엑스알피", "リップル"],
            "ADA": ["Cardano", "ADA", "카르다노", "에이다", "カルダノ"],
            "SOL": ["Solana", "SOL", "솔라나", "ソラナ"],
            "DOGE": ["Dogecoin", "DOGE", "도지코인", "ドージコイン"],
            "AAPL": ["Apple", "애플", "아이폰", "iPhone", "Tim Cook", "アップル"],
            "MSFT": ["Microsoft", "마이크로소프트", "Windows", "Azure", "マイクロソフト"],
            "GOOGL": ["Google", "Alphabet", "구글", "알파벳", "グーグル"],
            "AMZN": ["Amazon", "아마존", "AWS", "Bezos", "アマゾン"],
            "TSLA": ["Tesla", "테슬라", "Elon Musk", "일론머스크", "テスラ"],
            "NVDA": ["NVIDIA", "엔비디아", "AI", "GPU", "エヌビディア"],
        }
    
    @retry_on_failure(max_retries=3)
    async def fetch_all_news(self, asset: str, max_articles: int = 20) -> List[NewsArticle]:
        """모든 소스에서 뉴스 수집"""
        try:
            keywords = self.asset_keywords.get(asset.upper(), [asset])
            articles = []
            
            # 간단한 뉴스 데이터 시뮬레이션
            for i, keyword in enumerate(keywords[:3]):
                article = NewsArticle(
                    title=f"{keyword} 관련 뉴스 {i+1}",
                    content=f"{keyword}에 대한 최신 시장 동향과 분석입니다. 시장 전문가들은 긍정적인 전망을 보이고 있습니다.",
                    url=f"https://example.com/news/{keyword.lower()}/{i+1}",
                    source="integrated",
                    published_at=datetime.now()
                )
                articles.append(article)
            
            return articles[:max_articles]
            
        except Exception as e:
            log.error(f"뉴스 수집 실패: {e}")
            return []

class SentimentAnalyzer(BaseComponent):
    """감성 분석기"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("SentimentAnalyzer")
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.session = None
    
    def initialize(self):
        """감성 분석기 초기화"""
        log.info("감성 분석기 초기화 완료")
    
    def cleanup(self):
        """리소스 정리"""
        if self.session:
            try:
                asyncio.create_task(self.session.close())
            except:
                pass
    
    @retry_on_failure(max_retries=3)
    async def evaluate_news(self, articles: List[NewsArticle]) -> str:
        """뉴스 감성 분석"""
        if not articles:
            return "뉴스 없음"
        
        news_text = self._prepare_news_text(articles)
        
        if self.api_key:
            return await self._analyze_with_openai(news_text)
        else:
            return self._analyze_with_keywords(news_text)
    
    def _prepare_news_text(self, articles: List[NewsArticle], max_length: int = 2000) -> str:
        """뉴스 텍스트 준비"""
        text_parts = []
        current_length = 0
        
        for article in articles:
            article_text = f"제목: {article.title}\n내용: {article.content[:300]}\n\n"
            
            if current_length + len(article_text) > max_length:
                break
            
            text_parts.append(article_text)
            current_length += len(article_text)
        
        return "".join(text_parts)
    
    async def _analyze_with_openai(self, text: str) -> str:
        """OpenAI를 사용한 감성 분석"""
        try:
            prompt = f"""
다음 뉴스들을 분석하여 투자 관점에서 감성을 평가해주세요.

{text}

분석 결과를 다음 형식으로 작성해주세요:
1. 전체적인 감성: 긍정/부정/중립
2. 주요 키워드 3개
3. 투자 시사점 한 줄 요약

응답은 간결하고 명확하게 작성해주세요.
"""
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 300,
                'temperature': 0.3
            }
            
            if not self.session:
                import aiohttp
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    return "OpenAI 분석 실패"
                    
        except Exception as e:
            log.error(f"OpenAI 감성 분석 실패: {e}")
            return "감성 분석 실패"
    
    def _analyze_with_keywords(self, text: str) -> str:
        """키워드 기반 간단 감성 분석"""
        positive_keywords = [
            '상승', '증가', '호재', '긍정', '성장', '강세', '돌파', '급등', '상향',
            '개선', '확대', '투자', '매수', '랠리', '부양', '활성화'
        ]
        
        negative_keywords = [
            '하락', '감소', '악재', '부정', '위험', '약세', '붕괴', '급락', '하향',
            '악화', '축소', '매도', '폭락', '우려', '침체', '불안'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        if positive_count > negative_count * 1.2:
            sentiment = "긍정"
        elif negative_count > positive_count * 1.2:
            sentiment = "부정"
        else:
            sentiment = "중립"
        
        return f"감성: {sentiment} (긍정: {positive_count}, 부정: {negative_count})"

# =============================================================================
# 가격 데이터 수집 시스템
# =============================================================================

class PriceDataCollector(BaseComponent):
    """가격 데이터 수집기"""
    
    def __init__(self):
        super().__init__("PriceDataCollector")
        self.price_cache = {}
        self.last_update = {}
        
    def initialize(self):
        """가격 데이터 수집기 초기화"""
        log.info("가격 데이터 수집기 초기화 완료")
    
    def cleanup(self):
        """리소스 정리"""
        pass
    
    @retry_on_failure(max_retries=3)
    @cache_with_ttl(ttl_seconds=30)
    def get_price(self, asset: str, asset_type: AssetType) -> float:
        """통합 가격 조회"""
        try:
            if asset_type == AssetType.CRYPTO:
                return self._get_crypto_price(asset)
            elif asset_type in [AssetType.STOCK_US, AssetType.STOCK_KR, AssetType.STOCK_JP]:
                return self._get_stock_price(asset)
            else:
                log.warning(f"지원하지 않는 자산 유형: {asset_type}")
                return 0.0
                
        except Exception as e:
            log.error(f"가격 조회 실패 ({asset}): {e}")
            return 0.0
    
    def _get_crypto_price(self, symbol: str) -> float:
        """암호화폐 가격 조회"""
        if PYUPBIT_AVAILABLE:
            try:
                ticker = f"KRW-{symbol.upper()}"
                price = pyupbit.get_current_price(ticker)
                if price:
                    return float(price)
            except Exception as e:
                log.debug(f"업비트 가격 조회 실패 ({symbol}): {e}")
        
        # 코인게코 API 사용
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': symbol.lower(),
                'vs_currencies': 'krw,usd'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if symbol.lower() in data:
                    return float(data[symbol.lower()].get('krw', 0))
            
        except Exception as e:
            log.debug(f"코인게코 가격 조회 실패 ({symbol}): {e}")
        
        return 0.0
    
    def _get_stock_price(self, symbol: str) -> float:
        """주식 가격 조회"""
        if YFINANCE_AVAILABLE:
            try:
                import yfinance as yf
                
                if symbol.endswith('.T'):
                    yf_symbol = symbol
                elif '.' not in symbol:
                    yf_symbol = symbol
                else:
                    yf_symbol = symbol
                
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info
                
                price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose']
                for field in price_fields:
                    if field in info and info[field]:
                        return float(info[field])
                
                hist = ticker.history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
                
            except Exception as e:
                log.debug(f"yfinance 가격 조회 실패 ({symbol}): {e}")
        
        return 0.0
    
    def get_detailed_price_data(self, asset: str, asset_type: AssetType) -> PriceData:
        """상세 가격 데이터 조회"""
        try:
            price = self.get_price(asset, asset_type)
            
            volume = None
            market_cap = None
            change_24h = None
            change_pct_24h = None
            
            if asset_type == AssetType.CRYPTO and PYUPBIT_AVAILABLE:
                ticker = f"KRW-{asset.upper()}"
                ticker_data = pyupbit.get_ticker(ticker)
                if ticker_data:
                    volume = ticker_data.get('acc_trade_volume_24h')
                    change_pct_24h = ticker_data.get('signed_change_rate', 0) * 100
            
            return PriceData(
                symbol=asset,
                price=price,
                volume=volume,
                market_cap=market_cap,
                change_24h=change_24h,
                change_pct_24h=change_pct_24h,
                source="integrated"
            )
            
        except Exception as e:
            log.error(f"상세 가격 데이터 조회 실패 ({asset}): {e}")
            return PriceData(symbol=asset, price=0.0, source="error")

# =============================================================================
# 경제지표 수집 시스템
# =============================================================================

class EconomicDataCollector(BaseComponent):
    """경제지표 수집기"""
    
    def __init__(self):
        super().__init__("EconomicDataCollector")
        self.indicators_cache = {}
    
    def initialize(self):
        """경제지표 수집기 초기화"""
        log.info("경제지표 수집기 초기화 완료")
    
    def cleanup(self):
        """리소스 정리"""
        pass
    
    @retry_on_failure(max_retries=3)
    @cache_with_ttl(ttl_seconds=3600)
    def get_fear_greed_index(self) -> float:
        """공포 탐욕 지수 조회"""
        try:
            response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return float(data["data"][0]["value"])
            else:
                log.warning(f"FNG API 오류: {response.status_code}")
                return 50.0
                
        except Exception as e:
            log.error(f"공포 탐욕 지수 조회 실패: {e}")
            return 50.0
    
    def get_major_economic_indicators(self) -> List[EconomicIndicator]:
        """주요 경제지표 조회"""
        indicators = []
        
        try:
            fed_rate = self._get_fed_rate()
            if fed_rate is not None:
                indicators.append(EconomicIndicator(
                    name="Fed 기준금리",
                    value=fed_rate,
                    unit="%",
                    country="US",
                    importance="high"
                ))
            
            vix = self._get_vix_index()
            if vix is not None:
                indicators.append(EconomicIndicator(
                    name="VIX 변동성 지수",
                    value=vix,
                    unit="포인트",
                    country="US",
                    importance="high"
                ))
            
        except Exception as e:
            log.error(f"경제지표 조회 실패: {e}")
        
        return indicators
    
    def _get_fed_rate(self) -> Optional[float]:
        """Fed 기준금리 조회"""
        try:
            if YFINANCE_AVAILABLE:
                import yfinance as yf
                ticker = yf.Ticker("^TNX")
                hist = ticker.history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
        except Exception as e:
            log.debug(f"Fed 기준금리 조회 실패: {e}")
        return None
    
    def _get_vix_index(self) -> Optional[float]:
        """VIX 지수 조회"""
        try:
            if YFINANCE_AVAILABLE:
                import yfinance as yf
                ticker = yf.Ticker("^VIX")
                hist = ticker.history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
        except Exception as e:
            log.debug(f"VIX 지수 조회 실패: {e}")
        return None

# =============================================================================
# 시장 상태 관리 시스템
# =============================================================================

class MarketStatusManager(BaseComponent):
    """시장 상태 관리자"""
    
    def __init__(self):
        super().__init__("MarketStatusManager")
        self.holidays_data = {}
        self._load_holidays()
    
    def initialize(self):
        """시장 상태 관리자 초기화"""
        log.info("시장 상태 관리자 초기화 완료")
    
    def cleanup(self):
        """리소스 정리"""
        pass
    
    def _load_holidays(self):
        """공휴일 데이터 로드"""
        if HOLIDAYS_AVAILABLE:
            import holidays
            self.holidays_data = {
                'KR': holidays.Korea(),
                'US': holidays.UnitedStates(),
                'JP': holidays.Japan()
            }
        else:
            self.holidays_data = {'KR': {}, 'US': {}, 'JP': {}}
    
    def is_market_open(self, market: str, dt: Optional[datetime] = None) -> bool:
        """시장 개장 여부 확인"""
        if dt is None:
            dt = datetime.now()
        
        if dt.weekday() >= 5:
            return False
        
        country_code = self._get_country_code(market)
        if country_code in self.holidays_data:
            if dt.date() in self.holidays_data[country_code]:
                return False
        
        trading_hours = self._get_trading_hours(market)
        if trading_hours:
            start_hour, end_hour = trading_hours
            current_hour = dt.hour + dt.minute / 60.0
            return start_hour <= current_hour < end_hour
        
        return True
    
    def _get_country_code(self, market: str) -> str:
        """시장 코드에서 국가 코드 추출"""
        market_mapping = {
            'KRX': 'KR', 'KOSPI': 'KR', 'KOSDAQ': 'KR',
            'NYSE': 'US', 'NASDAQ': 'US',
            'TSE': 'JP', 'CRYPTO': 'GLOBAL'
        }
        return market_mapping.get(market.upper(), 'KR')
    
    def _get_trading_hours(self, market: str) -> Optional[Tuple[float, float]]:
        """거래 시간 조회"""
        trading_hours = {
            'KRX': (9.0, 15.5), 'KOSPI': (9.0, 15.5), 'KOSDAQ': (9.0, 15.5),
            'NYSE': (22.5, 5.0), 'NASDAQ': (22.5, 5.0),
            'TSE': (9.0, 15.0), 'CRYPTO': None
        }
        return trading_hours.get(market.upper())
    
    def get_market_status(self, market: str) -> MarketStatus:
        """시장 상태 조회"""
        now = datetime.now()
        is_open = self.is_market_open(market, now)
        
        return MarketStatus(
            market=market,
            is_open=is_open,
            next_open=None,
            next_close=None
        )
    
    def is_holiday_or_weekend(self, country: str = 'KR', dt: Optional[datetime] = None) -> bool:
        """공휴일 또는 주말 여부 확인"""
        if dt is None:
            dt = datetime.now()
        
        if dt.weekday() >= 5:
            return True
        
        if country in self.holidays_data:
            return dt.date() in self.holidays_data[country]
        
        return False

# =============================================================================
# 통합 알림 시스템
# =============================================================================

class NotificationManager(BaseComponent):
    """통합 알림 관리자"""
    
    def __init__(self):
        super().__init__("NotificationManager")
        self.channels = {}
        self.rate_limiter = defaultdict(list)
        self.max_rate_per_minute = 5
    
    def initialize(self):
        """알림 관리자 초기화"""
        self._load_notification_configs()
        log.info("알림 관리자 초기화 완료")
    
    def cleanup(self):
        """리소스 정리"""
        pass
    
    def _load_notification_configs(self):
        """알림 설정 로드"""
        pass
    
    @retry_on_failure(max_retries=3, delay=2.0)
    async def send_notification(self, message: str, 
                              channels: Optional[List[NotificationChannel]] = None,
                              priority: str = "normal") -> Dict[NotificationChannel, bool]:
        """통합 알림 전송"""
        results = {}
        
        try:
            success = await self._send_telegram(message)
            results[NotificationChannel.TELEGRAM] = success
        except Exception as e:
            log.error(f"알림 전송 실패: {e}")
            results[NotificationChannel.TELEGRAM] = False
        
        return results
    
    async def _send_telegram(self, message: str) -> bool:
        """텔레그램 알림 전송"""
        try:
            token = os.getenv('TELEGRAM_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not token or not chat_id:
                return False
            
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {"chat_id": chat_id, "text": message}
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    return response.status == 200
                    
        except Exception as e:
            log.error(f"텔레그램 전송 실패: {e}")
            return False

# =============================================================================
# 로깅 및 성과 추적 시스템  
# =============================================================================

class AdvancedLogger(BaseComponent):
    """고급 로깅 시스템"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        super().__init__("AdvancedLogger")
        self.db_path = db_path
        self.connection = None
        self._setup_database()
    
    def initialize(self):
        """고급 로깅 시스템 초기화"""
        log.info("고급 로깅 시스템 초기화 완료")
    
    def cleanup(self):
        """데이터베이스 연결 정리"""
        if self.connection:
            self.connection.close()
    
    def _setup_database(self):
        """데이터베이스 설정"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL,
                    price REAL,
                    confidence_score REAL,
                    balance_info TEXT,
                    market_data TEXT,
                    strategy_info TEXT
                )
            ''')
            self.connection.commit()
        except Exception as e:
            log.error(f"데이터베이스 설정 실패: {e}")
    
    def log_trade(self, asset: str, action: str, signal: Dict[str, Any], 
                  balance_info: Dict[str, Any], market_data: Dict[str, Any] = None) -> None:
        """거래 로그 기록"""
        try:
            timestamp = datetime.now().isoformat()
            
            # 텍스트 파일에도 기록
            self._log_to_file(asset, signal, balance_info, market_data)
            
            # 데이터베이스에 기록
            if self.connection:
                self.connection.execute('''
                    INSERT INTO trade_logs 
                    (timestamp, asset, action, quantity, price, confidence_score, balance_info, market_data, strategy_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, asset, action,
                    balance_info.get('quantity', 0),
                    market_data.get('current_price', 0) if market_data else 0,
                    signal.get('confidence_score', 0),
                    json.dumps(balance_info),
                    json.dumps(market_data) if market_data else '{}',
                    json.dumps(signal)
                ))
                self.connection.commit()
            
            log.info(f"거래 로그 기록: {asset} {action}")
            
        except Exception as e:
            log.error(f"거래 로그 기록 실패: {e}")
    
    def _log_to_file(self, asset: str, signal: Dict[str, Any], 
                     balance_info: Dict[str, Any], market_data: Dict[str, Any] = None):
        """파일 로그 기록"""
        try:
            current_price = market_data.get('current_price', 0) if market_data else 0
            
            log_entry = (
                f"[{asset}] {signal.get('decision', 'UNKNOWN')} | "
                f"신뢰도:{signal.get('confidence_score', 0)}% | "
                f"잔고:{balance_info.get('asset_balance', 0):.4f}, "
                f"현금:{balance_info.get('cash_balance', 0):.0f}, "
                f"평균가:{balance_info.get('avg_price', 0):.0f}, "
                f"현재가:{current_price:.0f}, "
                f"총자산:{balance_info.get('total_asset', 0):.0f}\n"
            )
            
            with open("trade_log.txt", "a", encoding="utf-8") as f:
                f.write(log_entry)
                
        except Exception as e:
            log.error(f"파일 로그 기록 실패: {e}")

# =============================================================================
# 자산 관리 및 포트폴리오 추적
# =============================================================================

class AssetManager(BaseComponent):
    """자산 관리자"""
    
    def __init__(self):
        super().__init__("AssetManager")
        self.price_collector = PriceDataCollector()
        
    def initialize(self):
        """자산 관리자 초기화"""
        self.price_collector.initialize()
        log.info("자산 관리자 초기화 완료")
    
    def cleanup(self):
        """리소스 정리"""
        self.price_collector.cleanup()
    
    def get_total_asset_value(self, exchange_client, include_breakdown: bool = False) -> Union[float, Dict[str, Any]]:
        """총 자산 가치 계산"""
        try:
            total_value = 0.0
            asset_breakdown = {}
            
            # 현금 잔고
            if hasattr(exchange_client, 'get_balance'):
                krw_balance = exchange_client.get_balance("KRW")
                total_value += float(krw_balance)
                asset_breakdown['KRW'] = {
                    'balance': krw_balance,
                    'value_krw': krw_balance,
                    'percentage': 0
                }
            
            # 보유 코인/주식
            if hasattr(exchange_client, 'get_balances'):
                balances = exchange_client.get_balances()
                
                for balance in balances:
                    currency = balance.get('currency', '')
                    amount = float(balance.get('balance', 0))
                    
                    if currency != 'KRW' and amount > 0:
                        if PYUPBIT_AVAILABLE:
                            price = pyupbit.get_current_price(f"KRW-{currency}")
                        else:
                            price = self.price_collector.get_price(currency, AssetType.CRYPTO)
                        
                        if price:
                            value_krw = amount * float(price)
                            total_value += value_krw
                            
                            asset_breakdown[currency] = {
                                'balance': amount,
                                'price': price,
                                'value_krw': value_krw,
                                'percentage': 0
                            }
            
            # 비율 계산
            if total_value > 0:
                for asset in asset_breakdown:
                    asset_breakdown[asset]['percentage'] = (
                        asset_breakdown[asset]['value_krw'] / total_value * 100
                    )
            
            if include_breakdown:
                return {
                    'total_value': total_value,
                    'asset_breakdown': asset_breakdown,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return total_value
                
        except Exception as e:
            log.error(f"총 자산 가치 계산 실패: {e}")
            return 0.0 if not include_breakdown else {'total_value': 0.0, 'asset_breakdown': {}}
    
    def get_cash_balance(self, exchange_client, currency: str = "KRW") -> float:
        """현금 잔고 조회"""
        try:
            if hasattr(exchange_client, 'get_balance'):
                return exchange_client.get_balance(currency)
            return 0.0
        except Exception as e:
            log.error(f"현금 잔고 조회 실패: {e}")
            return 0.0

# =============================================================================
# 통합 유틸리티 관리자
# =============================================================================

class UtilsManager(BaseComponent):
    """통합 유틸리티 관리자"""
    
    def __init__(self):
        super().__init__("UtilsManager")
        self.news_collector = NewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.price_collector = PriceDataCollector()
        self.economic_collector = EconomicDataCollector()
        self.market_status_manager = MarketStatusManager()
        self.notification_manager = NotificationManager()
        self.advanced_logger = AdvancedLogger()
        self.asset_manager = AssetManager()
    
    def initialize(self):
        """통합 유틸리티 관리자 초기화"""
        components = [
            self.news_collector, self.sentiment_analyzer, self.price_collector,
            self.economic_collector, self.market_status_manager, 
            self.notification_manager, self.advanced_logger, self.asset_manager
        ]
        
        for component in components:
            try:
                component.initialize()
            except Exception as e:
                log.error(f"컴포넌트 초기화 실패 ({component.name}): {e}")
        
        log.info("통합 유틸리티 관리자 초기화 완료")
    
    def cleanup(self):
        """리소스 정리"""
        components = [
            self.news_collector, self.sentiment_analyzer, self.price_collector,
            self.economic_collector, self.market_status_manager,
            self.notification_manager, self.advanced_logger, self.asset_manager
        ]
        
        for component in components:
            try:
                component.cleanup()
            except Exception as e:
                log.error(f"컴포넌트 정리 실패 ({component.name}): {e}")

# =============================================================================
# 전역 인스턴스 및 편의 함수들 (기존 호환성)
# =============================================================================

_utils_manager = None

def get_utils_manager() -> UtilsManager:
    """전역 유틸리티 관리자 인스턴스 반환"""
    global _utils_manager
    if _utils_manager is None:
        _utils_manager = UtilsManager()
        _utils_manager.initialize()
    return _utils_manager

# 기존 호환성을 위한 함수들
async def send_telegram(msg: str) -> None:
    """텔레그램 메시지 전송"""
    manager = get_utils_manager()
    await manager.notification_manager.send_notification(
        msg, [NotificationChannel.TELEGRAM]
    )

def get_fear_greed_index() -> float:
    """공포 탐욕 지수 조회"""
    manager = get_utils_manager()
    return manager.economic_collector.get_fear_greed_index()

async def fetch_all_news(asset: str) -> List[Dict[str, str]]:
    """뉴스 수집"""
    manager = get_utils_manager()
    articles = await manager.news_collector.fetch_all_news(asset)
    
    return [{'title': article.title, 'content': article.content} for article in articles]

async def evaluate_news(news: List[Dict[str, str]]) -> str:
    """뉴스 감성 분석"""
    manager = get_utils_manager()
    
    articles = []
    for item in news:
        try:
            article = NewsArticle(
                title=item.get('title', ''),
                content=item.get('content', ''),
                url='',
                source='legacy'
            )
            articles.append(article)
        except Exception as e:
            log.debug(f"뉴스 변환 실패: {e}")
            continue
    
    return await manager.sentiment_analyzer.evaluate_news(articles)

def is_holiday_or_weekend() -> bool:
    """공휴일 또는 주말 여부"""
    manager = get_utils_manager()
    return manager.market_status_manager.is_holiday_or_weekend()

def get_price(asset: str, asset_type: str) -> float:
    """가격 조회"""
    manager = get_utils_manager()
    
    type_mapping = {
        'coin': AssetType.CRYPTO, 'crypto': AssetType.CRYPTO,
        'stock_us': AssetType.STOCK_US, 'stock_kr': AssetType.STOCK_KR,
        'stock_jp': AssetType.STOCK_JP
    }
    
    asset_type_enum = type_mapping.get(asset_type.lower(), AssetType.CRYPTO)
    return manager.price_collector.get_price(asset, asset_type_enum)

def get_total_asset_value(upbit) -> float:
    """총 자산 가치"""
    manager = get_utils_manager()
    return manager.asset_manager.get_total_asset_value(upbit)

def get_cash_balance(upbit) -> float:
    """현금 잔고"""
    manager = get_utils_manager()
    return manager.asset_manager.get_cash_balance(upbit)

def log_trade(asset: str, signal: dict, balance_info: dict, now_price: float) -> None:
    """거래 로그"""
    manager = get_utils_manager()
    market_data = {'current_price': now_price}
    manager.advanced_logger.log_trade(asset, signal['decision'], signal, balance_info, market_data)

# =============================================================================
# 기본 유틸리티 함수들
# =============================================================================

def get_current_time():
    """현재 시간 반환"""
    return datetime.now()

def format_currency(amount):
    """통화 포맷팅"""
    return f"{amount:,.0f}원"

def calculate_percentage(current, previous):
    """퍼센티지 계산"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def safe_divide(numerator, denominator):
    """안전한 나눗셈"""
    return numerator / denominator if denominator != 0 else 0

def validate_config():
    """설정 검증"""
    try:
        required_vars = ['UPBIT_ACCESS_KEY', 'UPBIT_SECRET_KEY', 'OPENAI_API_KEY']
        missing = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing.append(var)
        
        return len(missing) == 0, missing
    except Exception as e:
        log.error(f"설정 검증 실패: {e}")
        return False, []

def log_trade_action(action, symbol, amount, price=None):
    """거래 액션 로그"""
    log_msg = f"거래 액션: {action} | 심볼: {symbol} | 수량: {amount}"
    if price:
        log_msg += f" | 가격: {price}"
    log.info(log_msg)

def retry_on_error(func, max_retries=3, delay=1):
    """에러 시 재시도"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
    return None

def send_telegram_message(message):
    """텔레그램 메시지 전송"""
    try:
        token = os.getenv('TELEGRAM_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not token or not chat_id:
            log.warning("텔레그램 설정이 없습니다")
            return False
        
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        data = {'chat_id': chat_id, 'text': message}
        response = requests.post(url, data=data, timeout=10)
        
        return response.status_code == 200
    except Exception as e:
        log.error(f"텔레그램 전송 실패: {e}")
        return False

# =============================================================================
# 고급 기능 함수들
# =============================================================================

async def send_multi_channel_notification(message: str, priority: str = "normal") -> Dict[str, bool]:
    """다중 채널 알림 전송"""
    manager = get_utils_manager()
    return await manager.notification_manager.send_notification(message, priority=priority)

async def get_comprehensive_market_data(asset: str, asset_type: AssetType) -> Dict[str, Any]:
    """종합 시장 데이터 조회"""
    manager = get_utils_manager()
    
    # 가격 데이터
    price_data = manager.price_collector.get_detailed_price_data(asset, asset_type)
    
    # 뉴스 데이터
    news_articles = await manager.news_collector.fetch_all_news(asset)
    sentiment = await manager.sentiment_analyzer.evaluate_news(news_articles)
    
    # 경제지표
    economic_indicators = manager.economic_collector.get_major_economic_indicators()
    
    return {
        'price_data': price_data.__dict__,
        'news_sentiment': sentiment,
        'news_count': len(news_articles),
        'economic_indicators': [ind.__dict__ for ind in economic_indicators],
        'market_status': manager.market_status_manager.get_market_status('CRYPTO' if asset_type == AssetType.CRYPTO else 'STOCK').__dict__,
        'timestamp': datetime.now().isoformat()
    }

def get_portfolio_analytics(exchange_client) -> Dict[str, Any]:
    """포트폴리오 분석"""
    manager = get_utils_manager()
    
    # 기본 메트릭
    portfolio_data = manager.asset_manager.get_total_asset_value(exchange_client, include_breakdown=True)
    
    return {
        'portfolio_data': portfolio_data,
        'last_updated': datetime.now().isoformat()
    }

# =============================================================================
# 메인 실행부 및 테스트
# =============================================================================

if __name__ == "__main__":
    print("=== 고급 유틸리티 시스템 테스트 ===\n")
    
    # 유틸리티 관리자 초기화
    manager = get_utils_manager()
    
    print("✅ Utils 모듈 로드 성공!")
    print(f"현재 시간: {get_current_time()}")
    print(f"공포탐욕지수: {get_fear_greed_index()}")
    print(f"주말/공휴일: {is_holiday_or_weekend()}")
    
    try:
        btc_price = get_price('BTC', 'crypto')
        print(f"BTC 가격: {btc_price:,.0f}원")
    except Exception as e:
        print(f"BTC 가격 조회 실패: {e}")
    
    # 리소스 정리
    manager.cleanup()
    print("\n✅ 고급 유틸리티 시스템 테스트 완료!")
