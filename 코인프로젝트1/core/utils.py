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

Author: Your Name
Version: 3.0.0
Created: 2025-06-18
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

# Core 패키지 import
try:
    from core import BaseComponent, config, logger, ValidationError
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 외부 라이브러리 imports
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    logger.warning("holidays 모듈을 찾을 수 없습니다.")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 모듈을 찾을 수 없습니다.")

try:
    import pyupbit
    PYUPBIT_AVAILABLE = True
except ImportError:
    PYUPBIT_AVAILABLE = False
    logger.warning("pyupbit 모듈을 찾을 수 없습니다.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance 모듈을 찾을 수 없습니다.")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("feedparser 모듈을 찾을 수 없습니다.")

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

# 기본 설정값
DEFAULT_CONFIG = {
    'news': {
        'max_articles_per_source': 10,
        'content_max_length': 1000,
        'cache_duration_hours': 2,
        'sentiment_analysis_provider': 'openai',
        'enable_translation': True,
        'timeout_seconds': 10
    },
    'notifications': {
        'rate_limit_per_minute': 5,
        'retry_attempts': 3,
        'retry_delay_seconds': 2,
        'enable_markdown': True
    },
    'data': {
        'cache_size': 1000,
        'price_update_interval': 30,  # 초
        'enable_historical_data': True,
        'max_retry_attempts': 3
    }
}

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
        # 제목과 내용 정제
        self.title = self.title.strip()
        self.content = self.content.strip()
        
        # URL 검증
        if not self.url.startswith(('http://', 'https://')):
            raise ValidationError(f"잘못된 URL 형식: {self.url}")

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
        if self.price <= 0:
            raise ValidationError(f"가격은 0보다 커야 합니다: {self.price}")

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
    importance: str = "medium"  # low, medium, high
    
    @property
    def change_from_previous(self) -> Optional[float]:
        """이전값 대비 변화"""
        if self.previous_value is not None:
            return self.value - self.previous_value
        return None
    
    @property
    def change_pct_from_previous(self) -> Optional[float]:
        """이전값 대비 변화율"""
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
    
    def __post_init__(self):
        if self.channel == NotificationChannel.TELEGRAM and not self.api_token:
            raise ValidationError("텔레그램 알림에는 API 토큰이 필요합니다")
        if self.channel == NotificationChannel.WEBHOOK and not self.webhook_url:
            raise ValidationError("웹훅 알림에는 웹훅 URL이 필요합니다")

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
                        logger.error(f"최대 재시도 횟수 초과 ({func.__name__}): {e}")
                        raise e
                    
                    logger.warning(f"재시도 {retries}/{max_retries} ({func.__name__}): {e}")
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
            # 캐시 키 생성
            key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            now = time.time()
            
            # 캐시 확인
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    logger.debug(f"캐시에서 반환: {func.__name__}")
                    return result
            
            # 함수 실행 및 캐시 저장
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            
            # 오래된 캐시 정리
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
        self.news_sources = self._initialize_sources()
        self.session = None
        
    def _do_initialize(self):
        """뉴스 수집기 초기화"""
        import aiohttp
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        self.logger.info("뉴스 수집기 초기화 완료")
    
    def _do_cleanup(self):
        """리소스 정리"""
        if self.session:
            asyncio.create_task(self.session.close())
    
    def _load_asset_keywords(self) -> Dict[str, List[str]]:
        """자산별 키워드 매핑"""
        return {
            # 암호화폐
            "BTC": ["Bitcoin", "비트코인", "BTC", "비트코인", "ビットコイン"],
            "ETH": ["Ethereum", "이더리움", "ETH", "이더", "イーサリアム"],
            "XRP": ["Ripple", "XRP", "리플", "엑스알피", "リップル"],
            "ADA": ["Cardano", "ADA", "카르다노", "에이다", "カルダノ"],
            "SOL": ["Solana", "SOL", "솔라나", "ソラナ"],
            "DOGE": ["Dogecoin", "DOGE", "도지코인", "ドージコイン"],
            
            # 미국 주식
            "AAPL": ["Apple", "애플", "아이폰", "iPhone", "Tim Cook", "アップル"],
            "MSFT": ["Microsoft", "마이크로소프트", "Windows", "Azure", "マイクロソフト"],
            "GOOGL": ["Google", "Alphabet", "구글", "알파벳", "グーグル"],
            "AMZN": ["Amazon", "아마존", "AWS", "Bezos", "アマゾン"],
            "TSLA": ["Tesla", "테슬라", "Elon Musk", "일론머스크", "テスラ"],
            "NVDA": ["NVIDIA", "엔비디아", "AI", "GPU", "エヌビディア"],
            "META": ["Meta", "Facebook", "메타", "페이스북", "フェイスブック"],
            "NFLX": ["Netflix", "넷플릭스", "스트리밍", "ネットフリックス"],
            
            # 일본 주식
            "7203.T": ["Toyota", "토요타", "도요타", "トヨタ", "豊田"],
            "6758.T": ["Sony", "소니", "ソニー", "PlayStation", "플레이스테이션"],
            "9984.T": ["SoftBank", "소프트뱅크", "ソフトバンク", "손정의"],
            "8306.T": ["MUFG", "미쓰비시UFJ", "三菱UFJ", "은행"],
            "6861.T": ["Keyence", "키엔스", "キーエンス", "센서"],
            
            # 경제지표
            "FED": ["Fed", "Federal Reserve", "연준", "FOMC", "금리", "파월"],
            "ECB": ["ECB", "European Central Bank", "유럽중앙은행", "라가르드"],
            "BOJ": ["BOJ", "Bank of Japan", "일본은행", "우에다", "植田"],
            "BOK": ["BOK", "Bank of Korea", "한국은행", "이창용"],
        }
    
    def _initialize_sources(self) -> Dict[DataSource, Dict[str, str]]:
        """뉴스 소스 초기화"""
        return {
            DataSource.NAVER_NEWS: {
                'base_url': 'https://search.naver.com/search.naver',
                'params': {'where': 'news', 'sm': 'tab_jum', 'sort': '1'}  # 최신순
            },
            DataSource.GOOGLE_NEWS: {
                'base_url': 'https://news.google.com/rss/search',
                'params': {'hl': 'ko', 'gl': 'KR', 'ceid': 'KR:ko'}
            },
            DataSource.YAHOO_FINANCE: {
                'base_url': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'params': {}
            },
            DataSource.COINDESK: {
                'base_url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'params': {}
            }
        }
    
    @retry_on_failure(max_retries=3)
    @cache_with_ttl(ttl_seconds=7200)  # 2시간 캐시
    async def fetch_all_news(self, asset: str, max_articles: int = 20) -> List[NewsArticle]:
        """모든 소스에서 뉴스 수집"""
        if not self.session:
            raise RuntimeError("뉴스 수집기가 초기화되지 않았습니다")
        
        keywords = self.asset_keywords.get(asset.upper(), [asset])
        all_articles = []
        
        # 병렬로 여러 소스에서 수집
        tasks = []
        for keyword in keywords[:3]:  # 최대 3개 키워드
            tasks.append(self._collect_from_naver(keyword))
            if FEEDPARSER_AVAILABLE:
                tasks.append(self._collect_from_google_news(keyword))
                tasks.append(self._collect_from_coindesk(keyword))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"뉴스 수집 중 오류: {result}")
        
        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        # 최신순 정렬 및 개수 제한
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)
        return unique_articles[:max_articles]
    
    async def _collect_from_naver(self, keyword: str) -> List[NewsArticle]:
        """네이버 뉴스 수집"""
        if not BS4_AVAILABLE:
            return []
        
        articles = []
        try:
            url = f"https://search.naver.com/search.naver?where=news&query={keyword}&sort=1"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    for item in soup.select('.news_tit')[:5]:  # 최대 5개
                        try:
                            title = item.get_text().strip()
                            link = item.get('href', '')
                            
                            # 본문 추출 (간단 버전)
                            content = await self._extract_article_content(link)
                            
                            article = NewsArticle(
                                title=title,
                                content=content,
                                url=link,
                                source="naver",
                                published_at=datetime.now()
                            )
                            articles.append(article)
                            
                        except Exception as e:
                            self.logger.debug(f"네이버 뉴스 파싱 오류: {e}")
                            continue
            
        except Exception as e:
            self.logger.error(f"네이버 뉴스 수집 실패: {e}")
        
        return articles
    
    async def _collect_from_google_news(self, keyword: str) -> List[NewsArticle]:
        """구글 뉴스 RSS 수집"""
        if not FEEDPARSER_AVAILABLE:
            return []
        
        articles = []
        try:
            import feedparser
            
            url = f"https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:5]:  # 최대 5개
                try:
                    article = NewsArticle(
                        title=entry.title,
                        content=entry.get('summary', ''),
                        url=entry.link,
                        source="google_news",
                        published_at=datetime.now()
                    )
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.debug(f"구글 뉴스 파싱 오류: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"구글 뉴스 수집 실패: {e}")
        
        return articles
    
    async def _collect_from_coindesk(self, keyword: str) -> List[NewsArticle]:
        """코인데스크 RSS 수집"""
        if not FEEDPARSER_AVAILABLE:
            return []
        
        articles = []
        try:
            import feedparser
            
            url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:10]:  # 최대 10개에서 키워드 필터링
                if any(kw.lower() in entry.title.lower() for kw in [keyword]):
                    try:
                        article = NewsArticle(
                            title=entry.title,
                            content=entry.get('summary', ''),
                            url=entry.link,
                            source="coindesk",
                            published_at=datetime.now()
                        )
                        articles.append(article)
                        
                    except Exception as e:
                        self.logger.debug(f"코인데스크 파싱 오류: {e}")
                        continue
            
        except Exception as e:
            self.logger.error(f"코인데스크 수집 실패: {e}")
        
        return articles
    
    async def _extract_article_content(self, url: str, max_length: int = 500) -> str:
        """기사 본문 추출"""
        if not BS4_AVAILABLE:
            return "(본문 추출 불가)"
        
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 다양한 본문 선택자 시도
                    content_selectors = [
                        'article p',
                        '.article-body p',
                        '.news-content p',
                        '.content p',
                        'p'
                    ]
                    
                    content = ""
                    for selector in content_selectors:
                        paragraphs = soup.select(selector)
                        if paragraphs:
                            content = " ".join(p.get_text().strip() for p in paragraphs[:3])
                            break
                    
                    return content[:max_length] if content else "(본문 추출 실패)"
                    
        except Exception as e:
            self.logger.debug(f"본문 추출 실패 ({url}): {e}")
            return "(본문 추출 실패)"

class SentimentAnalyzer(BaseComponent):
    """감성 분석기"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("SentimentAnalyzer")
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.session = None
    
    def _do_initialize(self):
        """감성 분석기 초기화"""
        import aiohttp
        self.session = aiohttp.ClientSession()
        self.logger.info("감성 분석기 초기화 완료")
    
    def _do_cleanup(self):
        """리소스 정리"""
        if self.session:
            asyncio.create_task(self.session.close())
    
    @retry_on_failure(max_retries=3)
    async def evaluate_news(self, articles: List[NewsArticle]) -> str:
        """뉴스 감성 분석"""
        if not articles:
            return "뉴스 없음"
        
        # 기사 요약
        news_text = self._prepare_news_text(articles)
        
        # OpenAI API 호출
        if self.api_key:
            return await self._analyze_with_openai(news_text)
        else:
            # 폴백: 간단한 키워드 기반 분석
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
            
            async with self.session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    error_text = await response.text()
                    self.logger.error(f"OpenAI API 오류: {error_text}")
                    return "OpenAI 분석 실패"
                    
        except Exception as e:
            self.logger.error(f"OpenAI 감성 분석 실패: {e}")
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
        
    def _do_initialize(self):
        """가격 데이터 수집기 초기화"""
        self.logger.info("가격 데이터 수집기 초기화 완료")
    
    @retry_on_failure(max_retries=3)
    @cache_with_ttl(ttl_seconds=30)  # 30초 캐시
    def get_price(self, asset: str, asset_type: AssetType) -> float:
        """통합 가격 조회"""
        try:
            if asset_type == AssetType.CRYPTO:
                return self._get_crypto_price(asset)
            elif asset_type in [AssetType.STOCK_US, AssetType.STOCK_KR, AssetType.STOCK_JP]:
                return self._get_stock_price(asset)
            else:
                self.logger.warning(f"지원하지 않는 자산 유형: {asset_type}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"가격 조회 실패 ({asset}): {e}")
            return 0.0
    
    def _get_crypto_price(self, symbol: str) -> float:
        """암호화폐 가격 조회"""
        if PYUPBIT_AVAILABLE:
            try:
                # 업비트 가격 조회
                ticker = f"KRW-{symbol.upper()}"
                price = pyupbit.get_current_price(ticker)
                if price:
                    return float(price)
            except Exception as e:
                self.logger.debug(f"업비트 가격 조회 실패 ({symbol}): {e}")
        
        # 코인게코 API 사용
        try:
            import requests
            
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
            self.logger.debug(f"코인게코 가격 조회 실패 ({symbol}): {e}")
        
        return 0.0
    
    def _get_stock_price(self, symbol: str) -> float:
        """주식 가격 조회"""
        if YFINANCE_AVAILABLE:
            try:
                import yfinance as yf
                
                # 심볼 형식 조정
                if symbol.endswith('.T'):  # 일본 주식
                    yf_symbol = symbol
                elif '.' not in symbol:  # 미국 주식
                    yf_symbol = symbol
                else:
                    yf_symbol = symbol
                
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info
                
                # 현재가 조회 시도 (여러 필드)
                price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose']
                for field in price_fields:
                    if field in info and info[field]:
                        return float(info[field])
                
                # 최근 거래 데이터에서 조회
                hist = ticker.history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
                
            except Exception as e:
                self.logger.debug(f"yfinance 가격 조회 실패 ({symbol}): {e}")
        
        return 0.0
    
    @retry_on_failure(max_retries=3)
    def get_detailed_price_data(self, asset: str, asset_type: AssetType) -> PriceData:
        """상세 가격 데이터 조회"""
        try:
            price = self.get_price(asset, asset_type)
            
            # 추가 데이터 조회
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
            
            elif asset_type in [AssetType.STOCK_US, AssetType.STOCK_KR, AssetType.STOCK_JP] and YFINANCE_AVAILABLE:
                import yfinance as yf
                ticker = yf.Ticker(asset)
                info = ticker.info
                
                volume = info.get('volume')
                market_cap = info.get('marketCap')
                
                # 전일 대비 변화율
                current_price = info.get('currentPrice', price)
                previous_close = info.get('previousClose')
                if current_price and previous_close:
                    change_pct_24h = ((current_price - previous_close) / previous_close) * 100
            
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
            self.logger.error(f"상세 가격 데이터 조회 실패 ({asset}): {e}")
            return PriceData(symbol=asset, price=0.0, source="error")

# =============================================================================
# 경제지표 수집 시스템
# =============================================================================

class EconomicDataCollector(BaseComponent):
    """경제지표 수집기"""
    
    def __init__(self):
        super().__init__("EconomicDataCollector")
        self.indicators_cache = {}
    
    def _do_initialize(self):
        """경제지표 수집기 초기화"""
        self.logger.info("경제지표 수집기 초기화 완료")
    
    @retry_on_failure(max_retries=3)
    @cache_with_ttl(ttl_seconds=3600)  # 1시간 캐시
    def get_fear_greed_index(self) -> float:
        """공포 탐욕 지수 조회"""
        try:
            response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return float(data["data"][0]["value"])
            else:
                self.logger.warning(f"FNG API 오류: {response.status_code}")
                return 50.0  # 중립값
                
        except Exception as e:
            self.logger.error(f"공포 탐욕 지수 조회 실패: {e}")
            return 50.0
    
    @retry_on_failure(max_retries=3)
    def get_major_economic_indicators(self) -> List[EconomicIndicator]:
        """주요 경제지표 조회"""
        indicators = []
        
        try:
            # Fed 기준금리 (FRED API 또는 scraping)
            fed_rate = self._get_fed_rate()
            if fed_rate is not None:
                indicators.append(EconomicIndicator(
                    name="Fed 기준금리",
                    value=fed_rate,
                    unit="%",
                    country="US",
                    importance="high"
                ))
            
            # VIX 지수
            vix = self._get_vix_index()
            if vix is not None:
                indicators.append(EconomicIndicator(
                    name="VIX 변동성 지수",
                    value=vix,
                    unit="포인트",
                    country="US",
                    importance="high"
                ))
            
            # 달러 인덱스
            dxy = self._get_dollar_index()
            if dxy is not None:
                indicators.append(EconomicIndicator(
                    name="달러 인덱스 (DXY)",
                    value=dxy,
                    unit="포인트",
                    country="US",
                    importance="medium"
                ))
            
        except Exception as e:
            self.logger.error(f"경제지표 조회 실패: {e}")
        
        return indicators
    
    def _get_fed_rate(self) -> Optional[float]:
        """Fed 기준금리 조회"""
        try:
            # Yahoo Finance를 통한 조회
            if YFINANCE_AVAILABLE:
                import yfinance as yf
                
                # 10년 국채 수익률로 대체 (Fed 금리와 상관관계 높음)
                ticker = yf.Ticker("^TNX")
                hist = ticker.history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            
        except Exception as e:
            self.logger.debug(f"Fed 기준금리 조회 실패: {e}")
        
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
            self.logger.debug(f"VIX 지수 조회 실패: {e}")
        
        return None
    
    def _get_dollar_index(self) -> Optional[float]:
        """달러 인덱스 조회"""
        try:
            if YFINANCE_AVAILABLE:
                import yfinance as yf
                
                ticker = yf.Ticker("DX-Y.NYB")
                hist = ticker.history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            
        except Exception as e:
            self.logger.debug(f"달러 인덱스 조회 실패: {e}")
        
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
    
    def _do_initialize(self):
        """시장 상태 관리자 초기화"""
        self.logger.info("시장 상태 관리자 초기화 완료")
    
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
            # 기본 공휴일 (간단 버전)
            self.holidays_data = {
                'KR': {},
                'US': {},
                'JP': {}
            }
    
    def is_market_open(self, market: str, dt: Optional[datetime] = None) -> bool:
        """시장 개장 여부 확인"""
        if dt is None:
            dt = datetime.now()
        
        # 주말 확인
        if dt.weekday() >= 5:  # 토요일(5), 일요일(6)
            return False
        
        # 공휴일 확인
        country_code = self._get_country_code(market)
        if country_code in self.holidays_data:
            if dt.date() in self.holidays_data[country_code]:
                return False
        
        # 거래 시간 확인
        trading_hours = self._get_trading_hours(market)
        if trading_hours:
            start_hour, end_hour = trading_hours
            current_hour = dt.hour + dt.minute / 60.0
            return start_hour <= current_hour < end_hour
        
        return True  # 기본값
    
    def _get_country_code(self, market: str) -> str:
        """시장 코드에서 국가 코드 추출"""
        market_mapping = {
            'KRX': 'KR',
            'KOSPI': 'KR',
            'KOSDAQ': 'KR',
            'NYSE': 'US',
            'NASDAQ': 'US',
            'TSE': 'JP',
            'CRYPTO': 'GLOBAL'  # 24/7
        }
        
        return market_mapping.get(market.upper(), 'KR')
    
    def _get_trading_hours(self, market: str) -> Optional[Tuple[float, float]]:
        """거래 시간 조회 (시.분 형태)"""
        trading_hours = {
            'KRX': (9.0, 15.5),      # 09:00 - 15:30
            'KOSPI': (9.0, 15.5),
            'KOSDAQ': (9.0, 15.5),
            'NYSE': (22.5, 5.0),     # 22:30 - 05:00 (한국시간)
            'NASDAQ': (22.5, 5.0),
            'TSE': (9.0, 15.0),      # 09:00 - 15:00 (일본시간)
            'CRYPTO': None           # 24/7
        }
        
        return trading_hours.get(market.upper())
    
    def get_market_status(self, market: str) -> MarketStatus:
        """시장 상태 조회"""
        now = datetime.now()
        is_open = self.is_market_open(market, now)
        
        # 다음 개장/마감 시간 계산
        next_open = None
        next_close = None
        
        if not is_open:
            next_open = self._calculate_next_open(market, now)
        else:
            next_close = self._calculate_next_close(market, now)
        
        return MarketStatus(
            market=market,
            is_open=is_open,
            next_open=next_open,
            next_close=next_close
        )
    
    def _calculate_next_open(self, market: str, current_time: datetime) -> Optional[datetime]:
        """다음 개장 시간 계산"""
        # 간단한 구현 - 실제로는 더 정교한 로직 필요
        trading_hours = self._get_trading_hours(market)
        if not trading_hours:
            return None
        
        start_hour, _ = trading_hours
        
        # 오늘의 개장 시간
        today_open = current_time.replace(
            hour=int(start_hour),
            minute=int((start_hour % 1) * 60),
            second=0,
            microsecond=0
        )
        
        # 오늘 개장 시간이 지났거나 오늘이 휴장일이면 다음 거래일
        if current_time >= today_open or not self.is_market_open(market, today_open):
            # 다음 거래일 찾기 (최대 7일)
            for days_ahead in range(1, 8):
                next_day = current_time + timedelta(days=days_ahead)
                next_open = next_day.replace(
                    hour=int(start_hour),
                    minute=int((start_hour % 1) * 60),
                    second=0,
                    microsecond=0
                )
                
                if self.is_market_open(market, next_open):
                    return next_open
        
        return today_open
    
    def _calculate_next_close(self, market: str, current_time: datetime) -> Optional[datetime]:
        """다음 마감 시간 계산"""
        trading_hours = self._get_trading_hours(market)
        if not trading_hours:
            return None
        
        _, end_hour = trading_hours
        
        # 오늘의 마감 시간
        today_close = current_time.replace(
            hour=int(end_hour),
            minute=int((end_hour % 1) * 60),
            second=0,
            microsecond=0
        )
        
        if current_time < today_close:
            return today_close
        
        return None
    
    def is_holiday_or_weekend(self, country: str = 'KR', dt: Optional[datetime] = None) -> bool:
        """공휴일 또는 주말 여부 확인 (기존 호환성)"""
        if dt is None:
            dt = datetime.now()
        
        # 주말 확인
        if dt.weekday() >= 5:
            return True
        
        # 공휴일 확인
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
        self.channels: Dict[NotificationChannel, NotificationConfig] = {}
        self.rate_limiter = defaultdict(list)
        self.max_rate_per_minute = 5
    
    def _do_initialize(self):
        """알림 관리자 초기화"""
        self._load_notification_configs()
        self.logger.info("알림 관리자 초기화 완료")
    
    def _load_notification_configs(self):
        """알림 설정 로드"""
        # 환경변수에서 설정 로드
        if os.getenv('TELEGRAM_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
            self.channels[NotificationChannel.TELEGRAM] = NotificationConfig(
                channel=NotificationChannel.TELEGRAM,
                api_token=os.getenv('TELEGRAM_TOKEN'),
                chat_id=os.getenv('TELEGRAM_CHAT_ID')
            )
        
        if os.getenv('SLACK_WEBHOOK_URL'):
            self.channels[NotificationChannel.SLACK] = NotificationConfig(
                channel=NotificationChannel.SLACK,
                webhook_url=os.getenv('SLACK_WEBHOOK_URL')
            )
        
        if os.getenv('DISCORD_WEBHOOK_URL'):
            self.channels[NotificationChannel.DISCORD] = NotificationConfig(
                channel=NotificationChannel.DISCORD,
                webhook_url=os.getenv('DISCORD_WEBHOOK_URL')
            )
    
    def add_notification_channel(self, config: NotificationConfig):
        """알림 채널 추가"""
        self.channels[config.channel] = config
        self.logger.info(f"알림 채널 추가: {config.channel.value}")
    
    @retry_on_failure(max_retries=3, delay=2.0)
    async def send_notification(self, message: str, 
                              channels: Optional[List[NotificationChannel]] = None,
                              priority: str = "normal") -> Dict[NotificationChannel, bool]:
        """통합 알림 전송"""
        if channels is None:
            channels = list(self.channels.keys())
        
        results = {}
        
        for channel in channels:
            if channel not in self.channels:
                results[channel] = False
                continue
            
            config = self.channels[channel]
            if not config.enabled:
                results[channel] = False
                continue
            
            # 속도 제한 확인
            if not self._check_rate_limit(channel):
                self.logger.warning(f"속도 제한 초과: {channel.value}")
                results[channel] = False
                continue
            
            try:
                success = await self._send_to_channel(message, config, priority)
                results[channel] = success
                
                if success:
                    self.logger.debug(f"알림 전송 성공: {channel.value}")
                else:
                    self.logger.warning(f"알림 전송 실패: {channel.value}")
                    
            except Exception as e:
                self.logger.error(f"알림 전송 오류 ({channel.value}): {e}")
                results[channel] = False
        
        return results
    
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """속도 제한 확인"""
        now = time.time()
        minute_ago = now - 60
        
        # 1분 이내 전송 기록 정리
        self.rate_limiter[channel] = [
            timestamp for timestamp in self.rate_limiter[channel]
            if timestamp > minute_ago
        ]
        
        # 제한 확인
        if len(self.rate_limiter[channel]) >= self.max_rate_per_minute:
            return False
        
        # 전송 기록 추가
        self.rate_limiter[channel].append(now)
        return True
    
    async def _send_to_channel(self, message: str, config: NotificationConfig, priority: str) -> bool:
        """특정 채널로 알림 전송"""
        try:
            if config.channel == NotificationChannel.TELEGRAM:
                return await self._send_telegram(message, config)
            elif config.channel == NotificationChannel.SLACK:
                return await self._send_slack(message, config)
            elif config.channel == NotificationChannel.DISCORD:
                return await self._send_discord(message, config)
            elif config.channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook(message, config)
            else:
                self.logger.warning(f"지원하지 않는 알림 채널: {config.channel}")
                return False
                
        except Exception as e:
            self.logger.error(f"채널별 알림 전송 실패 ({config.channel.value}): {e}")
            return False
    
    async def _send_telegram(self, message: str, config: NotificationConfig) -> bool:
        """텔레그램 알림 전송"""
        url = f"https://api.telegram.org/bot{config.api_token}/sendMessage"
        
        # 마크다운 형식 적용
        formatted_message = self._format_message_for_telegram(message)
        
        data = {
            "chat_id": config.chat_id,
            "text": formatted_message,
            "parse_mode": "Markdown"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"텔레그램 전송 실패: {response.status} - {error_text}")
                    return False
    
    async def _send_slack(self, message: str, config: NotificationConfig) -> bool:
        """슬랙 알림 전송"""
        formatted_message = self._format_message_for_slack(message)
        
        payload = {
            "text": formatted_message,
            "username": "TradingBot",
            "icon_emoji": ":chart_with_upwards_trend:"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.webhook_url, json=payload) as response:
                return response.status == 200
    
    async def _send_discord(self, message: str, config: NotificationConfig) -> bool:
        """디스코드 알림 전송"""
        formatted_message = self._format_message_for_discord(message)
        
        payload = {
            "content": formatted_message,
            "username": "TradingBot"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.webhook_url, json=payload) as response:
                return response.status == 204
    
    async def _send_webhook(self, message: str, config: NotificationConfig) -> bool:
        """웹훅 알림 전송"""
        payload = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "source": "trading_system"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.webhook_url, json=payload) as response:
                return response.status in [200, 201, 204]
    
    def _format_message_for_telegram(self, message: str) -> str:
        """텔레그램용 메시지 포맷팅"""
        # 간단한 마크다운 적용
        formatted = message
        formatted = formatted.replace('**', '*')  # 볼드
        formatted = formatted.replace('📊', '📊')  # 이모지 유지
        return formatted
    
    def _format_message_for_slack(self, message: str) -> str:
        """슬랙용 메시지 포맷팅"""
        # 슬랙 마크다운 적용
        formatted = message
        formatted = formatted.replace('**', '*')
        return formatted
    
    def _format_message_for_discord(self, message: str) -> str:
        """디스코드용 메시지 포맷팅"""
        # 디스코드 마크다운 적용
        return message

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
    
    def _do_initialize(self):
        """고급 로깅 시스템 초기화"""
        self.logger.info("고급 로깅 시스템 초기화 완료")
    
    def _do_cleanup(self):
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
            
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_value REAL,
                    daily_pnl REAL,
                    daily_pnl_pct REAL,
                    positions_count INTEGER,
                    trades_count INTEGER,
                    win_rate REAL
                )
            ''')
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 설정 실패: {e}")
    
    def log_trade(self, asset: str, action: str, signal: Dict[str, Any], 
                  balance_info: Dict[str, Any], market_data: Dict[str, Any] = None) -> None:
        """거래 로그 기록"""
        try:
            timestamp = datetime.now().isoformat()
            
            # 텍스트 파일에도 기록 (기존 호환성)
            self._log_to_file(asset, signal, balance_info, market_data)
            
            # 데이터베이스에 기록
            if self.connection:
                self.connection.execute('''
                    INSERT INTO trade_logs 
                    (timestamp, asset, action, quantity, price, confidence_score, balance_info, market_data, strategy_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    asset,
                    action,
                    balance_info.get('quantity', 0),
                    market_data.get('current_price', 0) if market_data else 0,
                    signal.get('confidence_score', 0),
                    json.dumps(balance_info),
                    json.dumps(market_data) if market_data else '{}',
                    json.dumps(signal)
                ))
                self.connection.commit()
            
            self.logger.info(f"거래 로그 기록: {asset} {action}")
            
        except Exception as e:
            self.logger.error(f"거래 로그 기록 실패: {e}")
    
    def _log_to_file(self, asset: str, signal: Dict[str, Any], 
                     balance_info: Dict[str, Any], market_data: Dict[str, Any] = None):
        """파일 로그 기록 (기존 호환성)"""
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
            self.logger.error(f"파일 로그 기록 실패: {e}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """성과 지표 기록"""
        try:
            if self.connection:
                date = datetime.now().date().isoformat()
                
                self.connection.execute('''
                    INSERT OR REPLACE INTO performance_metrics
                    (date, total_value, daily_pnl, daily_pnl_pct, positions_count, trades_count, win_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date,
                    metrics.get('total_value', 0),
                    metrics.get('daily_pnl', 0),
                    metrics.get('daily_pnl_pct', 0),
                    metrics.get('positions_count', 0),
                    metrics.get('trades_count', 0),
                    metrics.get('win_rate', 0)
                ))
                self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"성과 지표 기록 실패: {e}")
    
    def get_trading_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """거래 내역 조회"""
        try:
            if not self.connection:
                return []
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor = self.connection.execute('''
                SELECT * FROM trade_logs 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            ''', (cutoff_date,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'asset': row[2],
                    'action': row[3],
                    'quantity': row[4],
                    'price': row[5],
                    'confidence_score': row[6],
                    'balance_info': json.loads(row[7]) if row[7] else {},
                    'market_data': json.loads(row[8]) if row[8] else {},
                    'strategy_info': json.loads(row[9]) if row[9] else {}
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"거래 내역 조회 실패: {e}")
            return []
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """성과 요약 조회"""
        try:
            if not self.connection:
                return {}
            
            cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()
            
            cursor = self.connection.execute('''
                SELECT 
                    COUNT(*) as total_days,
                    AVG(total_value) as avg_total_value,
                    SUM(daily_pnl) as total_pnl,
                    AVG(daily_pnl_pct) as avg_daily_return,
                    AVG(win_rate) as avg_win_rate,
                    MAX(total_value) as max_value,
                    MIN(total_value) as min_value
                FROM performance_metrics 
                WHERE date >= ?
            ''', (cutoff_date,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'period_days': days,
                    'total_days_recorded': row[0],
                    'avg_total_value': row[1] or 0,
                    'total_pnl': row[2] or 0,
                    'avg_daily_return_pct': row[3] or 0,
                    'avg_win_rate': row[4] or 0,
                    'max_value': row[5] or 0,
                    'min_value': row[6] or 0,
                    'total_return_pct': ((row[5] or 0) - (row[6] or 0)) / (row[6] or 1) * 100 if row[6] else 0
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"성과 요약 조회 실패: {e}")
            return {}

# =============================================================================
# 자산 관리 및 포트폴리오 추적
# =============================================================================

class AssetManager(BaseComponent):
    """자산 관리자"""
    
    def __init__(self):
        super().__init__("AssetManager")
        self.price_collector = PriceDataCollector()
        
    def _do_initialize(self):
        """자산 관리자 초기화"""
        self.price_collector.initialize()
        self.logger.info("자산 관리자 초기화 완료")
    
    def get_total_asset_value(self, exchange_client, include_breakdown: bool = False) -> Union[float, Dict[str, Any]]:
        """총 자산 가치 계산 (기존 호환성 + 고급 기능)"""
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
                    'percentage': 0  # 나중에 계산
                }
            
            # 보유 코인/주식
            if hasattr(exchange_client, 'get_balances'):
                balances = exchange_client.get_balances()
                
                for balance in balances:
                    currency = balance.get('currency', '')
                    amount = float(balance.get('balance', 0))
                    
                    if currency != 'KRW' and amount > 0:
                        # 가격 조회
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
                                'percentage': 0  # 나중에 계산
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
            self.logger.error(f"총 자산 가치 계산 실패: {e}")
            return 0.0 if not include_breakdown else {'total_value': 0.0, 'asset_breakdown': {}}
    
    def get_cash_balance(self, exchange_client, currency: str = "KRW") -> float:
        """현금 잔고 조회 (기존 호환성)"""
        try:
            if hasattr(exchange_client, 'get_balance'):
                return exchange_client.get_balance(currency)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"현금 잔고 조회 실패: {e}")
            return 0.0
    
    def calculate_portfolio_metrics(self, exchange_client) -> Dict[str, Any]:
        """포트폴리오 메트릭 계산"""
        try:
            portfolio_data = self.get_total_asset_value(exchange_client, include_breakdown=True)
            
            if not isinstance(portfolio_data, dict):
                return {}
            
            total_value = portfolio_data['total_value']
            asset_breakdown = portfolio_data['asset_breakdown']
            
            # 다각화 지수 계산 (간단 버전)
            num_assets = len([a for a in asset_breakdown if a != 'KRW'])
            diversification_score = min(num_assets * 20, 100)  # 최대 5개 자산 = 100점
            
            # 현금 비율
            cash_ratio = asset_breakdown.get('KRW', {}).get('percentage', 0)
            
            # 최대 보유 자산 비율
            max_asset_ratio = 0
            max_asset_name = ""
            for asset, data in asset_breakdown.items():
                if asset != 'KRW' and data['percentage'] > max_asset_ratio:
                    max_asset_ratio = data['percentage']
                    max_asset_name = asset
            
            return {
                'total_value': total_value,
                'num_assets': num_assets,
                'diversification_score': diversification_score,
                'cash_ratio': cash_ratio,
                'max_asset_ratio': max_asset_ratio,
                'max_asset_name': max_asset_name,
                'risk_level': self._assess_portfolio_risk(asset_breakdown),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"포트폴리오 메트릭 계산 실패: {e}")
            return {}
    
    def _assess_portfolio_risk(self, asset_breakdown: Dict[str, Any]) -> str:
        """포트폴리오 위험도 평가"""
        try:
            # 현금 비율
            cash_ratio = asset_breakdown.get('KRW', {}).get('percentage', 0)
            
            # 집중도 (최대 보유 자산 비율)
            max_ratio = max(
                [data.get('percentage', 0) for asset, data in asset_breakdown.items() if asset != 'KRW'],
                default=0
            )
            
            # 자산 수
            num_assets = len([a for a in asset_breakdown if a != 'KRW'])
            
            # 위험도 계산
            risk_score = 0
            
            if cash_ratio < 10:  # 현금 부족
                risk_score += 20
            elif cash_ratio > 50:  # 현금 과다
                risk_score += 10
            
            if max_ratio > 70:  # 과도한 집중
                risk_score += 30
            elif max_ratio > 50:
                risk_score += 15
            
            if num_assets < 3:  # 다각화 부족
                risk_score += 25
            elif num_assets > 10:  # 과도한 분산
                risk_score += 10
            
            if risk_score >= 50:
                return "high"
            elif risk_score >= 25:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.debug(f"위험도 평가 실패: {e}")
            return "unknown"

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
    
    def _do_initialize(self):
        """통합 유틸리티 관리자 초기화"""
        # 모든 하위 컴포넌트 초기화
        components = [
            self.news_collector,
            self.sentiment_analyzer,
            self.price_collector,
            self.economic_collector,
            self.market_status_manager,
            self.notification_manager,
            self.advanced_logger,
            self.asset_manager
        ]
        
        for component in components:
            try:
                component.initialize()
            except Exception as e:
                self.logger.error(f"컴포넌트 초기화 실패 ({component.name}): {e}")
        
        self.logger.info("통합 유틸리티 관리자 초기화 완료")
    
    def _do_cleanup(self):
        """리소스 정리"""
        components = [
            self.news_collector,
            self.sentiment_analyzer,
            self.price_collector,
            self.economic_collector,
            self.market_status_manager,
            self.notification_manager,
            self.advanced_logger,
            self.asset_manager
        ]
        
        for component in components:
            try:
                component.cleanup()
            except Exception as e:
                self.logger.error(f"컴포넌트 정리 실패 ({component.name}): {e}")

# =============================================================================
# 전역 인스턴스 및 편의 함수들 (기존 호환성)
# =============================================================================

# 전역 관리자 인스턴스
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
    """텔레그램 메시지 전송 (기존 호환성)"""
    manager = get_utils_manager()
    await manager.notification_manager.send_notification(
        msg, [NotificationChannel.TELEGRAM]
    )

def get_fear_greed_index() -> float:
    """공포 탐욕 지수 조회 (기존 호환성)"""
    manager = get_utils_manager()
    return manager.economic_collector.get_fear_greed_index()

async def fetch_all_news(asset: str) -> List[Dict[str, str]]:
    """뉴스 수집 (기존 호환성)"""
    manager = get_utils_manager()
    articles = await manager.news_collector.fetch_all_news(asset)
    
    # 기존 형식으로 변환
    return [
        {
            'title': article.title,
            'content': article.content
        }
        for article in articles
    ]

async def evaluate_news(news: List[Dict[str, str]]) -> str:
    """뉴스 감성 분석 (기존 호환성)"""
    manager = get_utils_manager()
    
    # 기존 형식을 NewsArticle로 변환
    articles = []
    for item in news:
        try:
            article = NewsArticle(
                title=item.get('title', ''),
                content=item.get('content', ''),
                url='',  # 기존 데이터에는 URL이 없음
                source='legacy'
            )
            articles.append(article)
        except Exception as e:
            logger.debug(f"뉴스 변환 실패: {e}")
            continue
    
    return await manager.sentiment_analyzer.evaluate_news(articles)

def is_holiday_or_weekend() -> bool:
    """공휴일 또는 주말 여부 (기존 호환성)"""
    manager = get_utils_manager()
    return manager.market_status_manager.is_holiday_or_weekend()

def get_price(asset: str, asset_type: str) -> float:
    """가격 조회 (기존 호환성)"""
    manager = get_utils_manager()
    
    # 기존 문자열을 AssetType으로 변환
    type_mapping = {
        'coin': AssetType.CRYPTO,
        'crypto': AssetType.CRYPTO,
        'stock_us': AssetType.STOCK_US,
        'stock_kr': AssetType.STOCK_KR,
        'stock_jp': AssetType.STOCK_JP
    }
    
    asset_type_enum = type_mapping.get(asset_type.lower(), AssetType.CRYPTO)
    return manager.price_collector.get_price(asset, asset_type_enum)

def get_total_asset_value(upbit) -> float:
    """총 자산 가치 (기존 호환성)"""
    manager = get_utils_manager()
    return manager.asset_manager.get_total_asset_value(upbit)

def get_cash_balance(upbit) -> float:
    """현금 잔고 (기존 호환성)"""
    manager = get_utils_manager()
    return manager.asset_manager.get_cash_balance(upbit)

def log_trade(asset: str, signal: dict, balance_info: dict, now_price: float) -> None:
    """거래 로그 (기존 호환성)"""
    manager = get_utils_manager()
    market_data = {'current_price': now_price}
    manager.advanced_logger.log_trade(asset, signal['decision'], signal, balance_info, market_data)

# =============================================================================
# 고급 기능 함수들 (새로운 API)
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
    
    # 경제지표 (관련성 있는 것만)
    economic_indicators = manager.economic_collector.get_major_economic_indicators()
    
    return {
        'price_data': price_data.to_dict() if hasattr(price_data, 'to_dict') else price_data.__dict__,
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
    metrics = manager.asset_manager.calculate_portfolio_metrics(exchange_client)
    
    # 성과 요약
    performance = manager.advanced_logger.get_performance_summary(30)
    
    # 거래 내역
    recent_trades = manager.advanced_logger.get_trading_history(7)
    
    return {
        'portfolio_metrics': metrics,
        'performance_summary': performance,
        'recent_trades_count': len(recent_trades),
        'last_updated': datetime.now().isoformat()
    }

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
    
    print("=== 고급 유틸리티 시스템 테스트 ===\n")
    
    # 유틸리티 관리자 초기화
    manager = get_utils_manager()
    
    # 1. 뉴스 수집 및 감성 분석 테스트
    print("📰 뉴스 수집 및 감성 분석 테스트:")
    try:
        news = await fetch_all_news("BTC")
        print(f"   수집된 뉴스: {len(news)}개")
        
        sentiment = await evaluate_news(news)
        print(f"   감성 분석: {sentiment[:100]}...")
    except Exception as e:
        print(f"   ❌ 뉴스 테스트 실패: {e}")
    
    print()
    
    # 2. 가격 데이터 테스트
    print("💰 가격 데이터 테스트:")
    test_assets = [
        ("BTC", AssetType.CRYPTO),
        ("AAPL", AssetType.STOCK_US),
        ("7203.T", AssetType.STOCK_JP)
    ]
    
    for asset, asset_type in test_assets:
        try:
            price = manager.price_collector.get_price(asset, asset_type)
            print(f"   {asset}: {price:,.0f}")
        except Exception as e:
            print(f"   ❌ {asset} 가격 조회 실패: {e}")
    
    print()
    
    # 3. 경제지표 테스트
    print("📊 경제지표 테스트:")
    try:
        fg_index = get_fear_greed_index()
        print(f"   공포탐욕지수: {fg_index}")
        
        indicators = manager.economic_collector.get_major_economic_indicators()
        print(f"   경제지표 수: {len(indicators)}개")
        
        for indicator in indicators[:3]:  # 상위 3개만 출력
            print(f"   - {indicator.name}: {indicator.value}{indicator.unit}")
    except Exception as e:
        print(f"   ❌ 경제지표 테스트 실패: {e}")
    
    print()
    
    # 4. 시장 상태 테스트
    print("🏛️ 시장 상태 테스트:")
    markets = ['KRX', 'NYSE', 'CRYPTO']
    
    for market in markets:
        try:
            status = manager.market_status_manager.get_market_status(market)
            status_text = "🟢 개장" if status.is_open else "🔴 휴장"
            print(f"   {market}: {status_text}")
        except Exception as e:
            print(f"   ❌ {market} 상태 확인 실패: {e}")
    
    print()
    
    # 5. 알림 시스템 테스트
    print("📨 알림 시스템 테스트:")
    try:
        test_message = "🚀 고급 유틸리티 시스템 테스트 완료!"
        results = await send_multi_channel_notification(test_message)
        
        for channel, success in results.items():
            status = "✅ 성공" if success else "❌ 실패"
            print(f"   {channel.value}: {status}")
    except Exception as e:
        print(f"   ❌ 알림 테스트 실패: {e}")
    
    print()
    
    # 6. 종합 시장 데이터 테스트
    print("🔍 종합 시장 데이터 테스트:")
    try:
        comprehensive_data = await get_comprehensive_market_data("BTC", AssetType.CRYPTO)
        
        print(f"   가격: {comprehensive_data['price_data']['price']:,.0f}")
        print(f"   뉴스 개수: {comprehensive_data['news_count']}")
        print(f"   경제지표: {len(comprehensive_data['economic_indicators'])}개")
        print(f"   감성: {comprehensive_data['news_sentiment'][:50]}...")
    except Exception as e:
        print(f"   ❌ 종합 데이터 테스트 실패: {e}")
    
    print()
    
    # 리소스 정리
    print("🧹 리소스 정리 중...")
    manager.cleanup()
    
    print("✅ 고급 유틸리티 시스템 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main())

# =============================================================================
# 공개 API
# =============================================================================

__all__ = [
    # 메인 클래스들
    'UtilsManager',
    'NewsCollector',
    'SentimentAnalyzer',
    'PriceDataCollector',
    'EconomicDataCollector',
    'MarketStatusManager',
    'NotificationManager',
    'AdvancedLogger',
    'AssetManager',
    
    # 데이터 클래스들
    'NewsArticle',
    'PriceData',
    'EconomicIndicator',
    'MarketStatus',
    'NotificationConfig',
    
    # 열거형들
    'NotificationChannel',
    'DataSource',
    'AssetType',
    
    # 기존 호환 함수들
    'send_telegram',
    'get_fear_greed_index',
    'fetch_all_news',
    'evaluate_news',
    'is_holiday_or_weekend',
    'get_price',
    'get_total_asset_value',
    'get_cash_balance',
    'log_trade',
    
    # 새로운 고급 함수들
    'get_utils_manager',
    'send_multi_channel_notification',
    'get_comprehensive_market_data',
    'get_portfolio_analytics',
    
    # 유틸리티 함수들
    'retry_on_failure',
    'cache_with_ttl',
]
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

Author: Your Name
Version: 3.0.0
Created: 2025-06-18
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

# Core 패키지 import
try:
    from core import BaseComponent, config, logger, ValidationError
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 외부 라이브러리 imports
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    logger.warning("holidays 모듈을 찾을 수 없습니다.")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 모듈을 찾을 수 없습니다.")

try:
    import pyupbit
    PYUPBIT_AVAILABLE = True
except ImportError:
    PYUPBIT_AVAILABLE = False
    logger.warning("pyupbit 모듈을 찾을 수 없습니다.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance 모듈을 찾을 수 없습니다.")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("feedparser 모듈을 찾을 수 없습니다.")

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

# 기본 설정값
DEFAULT_CONFIG = {
    'news': {
        'max_articles_per_source': 10,
        'content_max_length': 1000,
        'cache_duration_hours': 2,
        'sentiment_analysis_provider': 'openai',
        'enable_translation': True,
        'timeout_seconds': 10
    },
    'notifications': {
        'rate_limit_per_minute': 5,
        'retry_attempts': 3,
        'retry_delay_seconds': 2,
        'enable_markdown': True
    },
    'data': {
        'cache_size': 1000,
        'price_update_interval': 30,  # 초
        'enable_historical_data': True,
        'max_retry_attempts': 3
    }
}

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
        # 제목과 내용 정제
        self.title = self.title.strip()
        self.content = self.content.strip()
        
        # URL 검증
        if not self.url.startswith(('http://', 'https://')):
            raise ValidationError(f"잘못된 URL 형식: {self.url}")

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
        if self.price <= 0:
            raise ValidationError(f"가격은 0보다 커야 합니다: {self.price}")

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
    importance: str = "medium"  # low, medium, high
    
    @property
    def change_from_previous(self) -> Optional[float]:
        """이전값 대비 변화"""
        if self.previous_value is not None:
            return self.value - self.previous_value
        return None
    
    @property
    def change_pct_from_previous(self) -> Optional[float]:
        """이전값 대비 변화율"""
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
    
    def __post_init__(self):
        if self.channel == NotificationChannel.TELEGRAM and not self.api_token:
            raise ValidationError("텔레그램 알림에는 API 토큰이 필요합니다")
        if self.channel == NotificationChannel.WEBHOOK and not self.webhook_url:
            raise ValidationError("웹훅 알림에는 웹훅 URL이 필요합니다")

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
                        logger.error(f"최대 재시도 횟수 초과 ({func.__name__}): {e}")
                        raise e
                    
                    logger.warning(f"재시도 {retries}/{max_retries} ({func.__name__}): {e}")
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
            # 캐시 키 생성
            key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            now = time.time()
            
            # 캐시 확인
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    logger.debug(f"캐시에서 반환: {func.__name__}")
                    return result
            
            # 함수 실행 및 캐시 저장
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            
            # 오래된 캐시 정리
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
        self.news_sources = self._initialize_sources()
        self.session = None
        
    def _do_initialize(self):
        """뉴스 수집기 초기화"""
        import aiohttp
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        self.logger.info("뉴스 수집기 초기화 완료")
    
    def _do_cleanup(self):
        """리소스 정리"""
        if self.session:
            asyncio.create_task(self.session.close())
    
    def _load_asset_keywords(self) -> Dict[str, List[str]]:
        """자산별 키워드 매핑"""
        return {
            # 암호화폐
            "BTC": ["Bitcoin", "비트코인", "BTC", "비트코인", "ビットコイン"],
            "ETH": ["Ethereum", "이더리움", "ETH", "이더", "イーサリアム"],
            "XRP": ["Ripple", "XRP", "리플", "엑스알피", "リップル"],
            "ADA": ["Cardano", "ADA", "카르다노", "에이다", "カルダノ"],
            "SOL": ["Solana", "SOL", "솔라나", "ソラナ"],
            "DOGE": ["Dogecoin", "DOGE", "도지코인", "ドージコイン"],
            
            # 미국 주식
            "AAPL": ["Apple", "애플", "아이폰", "iPhone", "Tim Cook", "アップル"],
            "MSFT": ["Microsoft", "마이크로소프트", "Windows", "Azure", "マイクロソフト"],
            "GOOGL": ["Google", "Alphabet", "구글", "알파벳", "グーグル"],
            "AMZN": ["Amazon", "아마존", "AWS", "Bezos", "アマゾン"],
            "TSLA": ["Tesla", "테슬라", "Elon Musk", "일론머스크", "テスラ"],
            "NVDA": ["NVIDIA", "엔비디아", "AI", "GPU", "エヌビディア"],
            "META": ["Meta", "Facebook", "메타", "페이스북", "フェイスブック"],
            "NFLX": ["Netflix", "넷플릭스", "스트리밍", "ネットフリックス"],
            
            # 일본 주식
            "7203.T": ["Toyota", "토요타", "도요타", "トヨタ", "豊田"],
            "6758.T": ["Sony", "소니", "ソニー", "PlayStation", "플레이스테이션"],
            "9984.T": ["SoftBank", "소프트뱅크", "ソフトバンク", "손정의"],
            "8306.T": ["MUFG", "미쓰비시UFJ", "三菱UFJ", "은행"],
            "6861.T": ["Keyence", "키엔스", "キーエンス", "센서"],
            
            # 경제지표
            "FED": ["Fed", "Federal Reserve", "연준", "FOMC", "금리", "파월"],
            "ECB": ["ECB", "European Central Bank", "유럽중앙은행", "라가르드"],
            "BOJ": ["BOJ", "Bank of Japan", "일본은행", "우에다", "植田"],
            "BOK": ["BOK", "Bank of Korea", "한국은행", "이창용"],
        }
    
    def _initialize_sources(self) -> Dict[DataSource, Dict[str, str]]:
        """뉴스 소스 초기화"""
        return {
            DataSource.NAVER_NEWS: {
                'base_url': 'https://search.naver.com/search.naver',
                'params': {'where': 'news', 'sm': 'tab_jum', 'sort': '1'}  # 최신순
            },
            DataSource.GOOGLE_NEWS: {
                'base_url': 'https://news.google.com/rss/search',
                'params': {'hl': 'ko', 'gl': 'KR', 'ceid': 'KR:ko'}
            },
            DataSource.YAHOO_FINANCE: {
                'base_url': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'params': {}
            },
            DataSource.COINDESK: {
                'base_url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'params': {}
            }
        }
    
    @retry_on_failure(max_retries=3)
    @cache_with_ttl(ttl_seconds=7200)  # 2시간 캐시
    async def fetch_all_news(self, asset: str, max_articles: int = 20) -> List[NewsArticle]:
        """모든 소스에서 뉴스 수집"""
        if not self.session:
            raise RuntimeError("뉴스 수집기가 초기화되지 않았습니다")
        
        keywords = self.asset_keywords.get(asset.upper(), [asset])
        all_articles = []
        
        # 병렬로 여러 소스에서 수집
        tasks = []
        for keyword in keywords[:3]:  # 최대 3개 키워드
            tasks.append(self._collect_from_naver(keyword))
            if FEEDPARSER_AVAILABLE:
                tasks.append(self._collect_from_google_news(keyword))
                tasks.append(self._collect_from_coindesk(keyword))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"뉴스 수집 중 오류: {result}")
        
        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        # 최신순 정렬 및 개수 제한
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)
        return unique_articles[:max_articles]
    
    async def _collect_from_naver(self, keyword: str) -> List[NewsArticle]:
        """네이버 뉴스 수집"""
        if not BS4_AVAILABLE:
            return []
        
        articles = []
        try:
            url = f"https://search.naver.com/search.naver?where=news&query={keyword}&sort=1"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    for item in soup.select('.news_tit')[:5]:  # 최대 5개
                        try:
                            title = item.get_text().strip()
                            link = item.get('href', '')
                            
                            # 본문 추출 (간단 버전)
                            content = await self._extract_article_content(link)
                            
                            article = NewsArticle(
                                title=title,
                                content=content,
                                url=link,
                                source="naver",
                                published_at=datetime.now()
                            )
                            articles.append(article)
                            
                        except Exception as e:
                            self.logger.debug(f"네이버 뉴스 파싱 오류: {e}")
                            continue
            
        except Exception as e:
            self.logger.error(f"네이버 뉴스 수집 실패: {e}")
        
        return articles
    
    async def _collect_from_google_news(self, keyword: str) -> List[NewsArticle]:
        """구글 뉴스 RSS 수집"""
        if not FEEDPARSER_AVAILABLE:
            return []
        
        articles = []
        try:
            import feedparser
            
            url = f"https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:5]:  # 최대 5개
                try:
                    article = NewsArticle(
                        title=entry.title,
                        content=entry.get('summary', ''),
                        url=entry.link,
                        source="google_news",
                        published_at=datetime.now()
                    )
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.debug(f"구글 뉴스 파싱 오류: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"구글 뉴스 수집 실패: {e}")
        
        return articles
    
    async def _collect_from_coindesk(self, keyword: str) -> List[NewsArticle]:
        """코인데스크 RSS 수집"""
        if not FEEDPARSER_AVAILABLE:
            return []
        
        articles = []
        try:
            import feedparser
            
            url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:10]:  # 최대 10개에서 키워드 필터링
                if any(kw.lower() in entry.title.lower() for kw in [keyword]):
                    try:
                        article = NewsArticle(
                            title=entry.title,
                            content=entry.get('summary', ''),
                            url=entry.link,
                            source="coindesk",
                            published_at=datetime.now()
                        )
                        articles.append(article)
                        
                    except Exception as e:
                        self.logger.debug(f"코인데스크 파싱 오류: {e}")
                        continue
            
        except Exception as e:
            self.logger.error(f"코인데스크 수집 실패: {e}")
        
        return articles
    
    async def _extract_article_content(self, url: str, max_length: int = 500) -> str:
        """기사 본문 추출"""
        if not BS4_AVAILABLE:
            return "(본문 추출 불가)"
        
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 다양한 본문 선택자 시도
                    content_selectors = [
                        'article p',
                        '.article-body p',
                        '.news-content p',
                        '.content p',
                        'p'
                    ]
                    
                    content = ""
                    for selector in content_selectors:
                        paragraphs = soup.select(selector)
                        if paragraphs:
                            content = " ".join(p.get_text().strip() for p in paragraphs[:3])
                            break
                    
                    return content[:max_length] if content else "(본문 추출 실패)"
                    
        except Exception as e:
            self.logger.debug(f"본문 추출 실패 ({url}): {e}")
            return "(본문 추출 실패)"

class SentimentAnalyzer(BaseComponent):
    """감성 분석기"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("SentimentAnalyzer")
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.session = None
    
    def _do_initialize(self):
        """감성 분석기 초기화"""
        import aiohttp
        self.session = aiohttp.ClientSession()
        self.logger.info("감성 분석기 초기화 완료")
    
    def _do_cleanup(self):
        """리소스 정리"""
        if self.session:
            asyncio.create_task(self.session.close())
    
    @retry_on_failure(max_retries=3)
    async def evaluate_news(self, articles: List[NewsArticle]) -> str:
        """뉴스 감성 분석"""
        if not articles:
            return "뉴스 없음"
        
        # 기사 요약
        news_text = self._prepare_news_text(articles)
        
        # OpenAI API 호출
        if self.api_key:
            return await self._analyze_with_openai(news_text)
        else:
            # 폴백: 간단한 키워드 기반 분석
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
            
            async with self.session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    error_text = await response.text()
                    self.logger.error(f"OpenAI API 오류: {error_text}")
                    return "OpenAI 분석 실패"
                    
        except Exception as e:
            self.logger.error(f"OpenAI 감성 분석 실패: {e}")
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
        
    def _do_initialize(self):
        """가격 데이터 수집기 초기화"""
        self.logger.info("가격 데이터 수집기 초기화 완료")
    
    @retry_on_failure(max_retries=3)
    @cache_with_ttl(ttl_seconds=30)  # 30초 캐시
    def get_price(self, asset: str, asset_type: AssetType) -> float:
        """통합 가격 조회"""
        try:
            if asset_type == AssetType.CRYPTO:
                return self._get_crypto_price(asset)
            elif asset_type in [AssetType.STOCK_US, AssetType.STOCK_KR, AssetType.STOCK_JP]:
                return self._get_stock_price(asset)
            else:
                self.logger.warning(f"지원하지 않는 자산 유형: {asset_type}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"가격 조회 실패 ({asset}): {e}")
            return 0.0
    
    def _get_crypto_price(self, symbol: str) -> float:
        """암호화폐 가격 조회"""
        if PYUPBIT_AVAILABLE:
            try:
                # 업비트 가격 조회
                ticker = f"KRW-{symbol.upper()}"
                price = pyupbit.get_current_price(ticker)
                if price:
                    return float(price)
            except Exception as e:
                self.logger.debug(f"업비트 가격 조회 실패 ({symbol}): {e}")
        
        # 코인게코 API 사용
        try:
            import requests