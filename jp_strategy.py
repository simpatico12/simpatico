#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇯🇵 일본 주식 완전 자동화 전략 - 최고퀸트프로젝트 (설정파일 완전 통합)
===========================================================================

🎯 핵심 기능:
- 📊 실시간 닛케이225 + TOPIX 구성종목 크롤링
- 💱 엔화 자동 매매법 (USD/JPY 기반)
- ⚡ 고급 기술적 지표 (RSI, MACD, 볼린저밴드, 스토캐스틱)
- 💰 분할매매 시스템 (리스크 관리)
- 🔍 실시간 자동 20개 종목 선별
- 🛡️ 동적 손절/익절
- 🤖 완전 자동화 (혼자서도 OK)
- 🔗 settings.yaml + .env 완전 연동

Author: 최고퀸트팀
Version: 5.0.0 (설정파일 완전 통합)
Project: 최고퀸트프로젝트
Dependencies: requirements.txt와 완전 호환
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
warnings.filterwarnings('ignore')

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/jp_strategy.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================================================================================
# 🔧 ta 라이브러리 안전 import 및 대체 함수들
# ========================================================================================

# ta 라이브러리 안전 import
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("⚠️ ta-lib 없음, 기본 계산 사용")

def safe_rsi(prices, period=14):
    """ta-lib 없이도 작동하는 RSI 계산"""
    if HAS_TA:
        try:
            return ta.momentum.RSIIndicator(prices, window=period).rsi()
        except:
            pass
    
    # 직접 RSI 계산
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except:
        return pd.Series([50] * len(prices))  # 기본값 50

def safe_macd(prices):
    """ta-lib 없이도 작동하는 MACD 계산"""
    if HAS_TA:
        try:
            macd_indicator = ta.trend.MACD(prices)
            return macd_indicator.macd(), macd_indicator.macd_signal(), macd_indicator.macd_diff()
        except:
            pass
    
    # 직접 MACD 계산
    try:
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return macd, signal, histogram
    except:
        return pd.Series([0] * len(prices)), pd.Series([0] * len(prices)), pd.Series([0] * len(prices))

def safe_bollinger_bands(prices, window=20):
    """ta-lib 없이도 작동하는 볼린저 밴드 계산"""
    if HAS_TA:
        try:
            bb = ta.volatility.BollingerBands(prices, window=window)
            return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
        except:
            pass
    
    # 직접 볼린저 밴드 계산
    try:
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, sma, lower
    except:
        return prices * 1.02, prices, prices * 0.98

def safe_stochastic(high, low, close, k_period=14):
    """ta-lib 없이도 작동하는 스토캐스틱 계산"""
    if HAS_TA:
        try:
            stoch = ta.momentum.StochasticOscillator(high, low, close, window=k_period, smooth_window=3)
            return stoch.stoch(), stoch.stoch_signal()
        except:
            pass
    
    # 직접 스토캐스틱 계산
    try:
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(3).mean()
        return k_percent, d_percent
    except:
        return pd.Series([50] * len(close)), pd.Series([50] * len(close))

# 안전한 데이터 타입 변환
def safe_upper(value):
    """안전한 upper() 호출"""
    if value is None:
        return "UNKNOWN"
    try:
        return str(value).upper()
    except:
        return "UNKNOWN"

def safe_float(value, default=0.0):
    """안전한 float 변환"""
    try:
        if value is None:
            return default
        return float(value)
    except:
        return default

# 유효한 일본 주식 심볼 목록
VALID_JP_SYMBOLS = [
    "6758.T",  # 소니
    "9984.T",  # 소프트뱅크
    "7974.T",  # 닌텐도
    "6861.T",  # 키엔스
    "8316.T",  # 미쓰비시UFJ
    "9432.T",  # NTT
    "4063.T",  # 신에츠화학
    "6367.T",  # 다이킨공업
]

@dataclass
class JPStockSignal:
    """일본 주식 시그널 (완전 통합)"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    strategy_source: str

    # 기술적 지표
    rsi: float
    macd_signal: str      # 'bullish', 'bearish', 'neutral'
    bollinger_signal: str # 'upper', 'lower', 'middle'
    stoch_signal: str     # 'oversold', 'overbought', 'neutral'
    ma_trend: str         # 'uptrend', 'downtrend', 'sideways'

    # 포지션 관리
    position_size: int    # 총 주식 수
    split_buy_plan: List[Dict]  # 분할 매수 계획
    split_sell_plan: List[Dict] # 분할 매도 계획

    # 손익 관리
    stop_loss: float
    take_profit: float
    max_hold_days: int

    # 기본 정보
    stock_type: str       # 'export', 'domestic'
    yen_signal: str       # 'strong', 'weak', 'neutral'
    sector: str
    reasoning: str
    timestamp: datetime
    
    # 자동선별 추가 정보
    market_cap: float
    selection_score: float  # 선별 점수
    quality_rank: int      # 품질 순위
    additional_data: Optional[Dict] = None

# ========================================================================================
# 🔧 설정 로더 클래스 (NEW!)
# ========================================================================================
class ConfigLoader:
    """🔧 설정 파일 로더 (settings.yaml + .env 통합)"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        self.config_path = config_path
        self.config = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """모든 설정 로드 (YAML + 환경변수)"""
        try:
            # 1. YAML 파일 로드
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"✅ 설정 파일 로드: {self.config_path}")
            else:
                logger.warning(f"⚠️ 설정 파일 없음: {self.config_path}")
                self.config = {}
            
            # 2. 환경변수로 덮어쓰기 (우선순위 높음)
            self._override_with_env_vars()
            
            # 3. 기본값 적용
            self._apply_defaults()
            
            logger.info("🔧 설정 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ 설정 로드 실패: {e}")
            self._apply_defaults()
    
    def _override_with_env_vars(self):
        """환경변수로 설정 덮어쓰기"""
        try:
            # 일본 전략 관련 환경변수
            jp_config = self.config.setdefault('jp_strategy', {})
            
            # 활성화 상태
            if os.getenv('JP_STRATEGY_ENABLED'):
                jp_config['enabled'] = os.getenv('JP_STRATEGY_ENABLED').lower() == 'true'
            
            # 신뢰도 임계값
            if os.getenv('JP_CONFIDENCE_THRESHOLD'):
                jp_config['confidence_threshold'] = float(os.getenv('JP_CONFIDENCE_THRESHOLD'))
            
            # 엔화 설정
            yen_config = jp_config.setdefault('yen_signals', {})
            if os.getenv('JP_YEN_STRONG_THRESHOLD'):
                yen_config['strong_threshold'] = float(os.getenv('JP_YEN_STRONG_THRESHOLD'))
            if os.getenv('JP_YEN_WEAK_THRESHOLD'):
                yen_config['weak_threshold'] = float(os.getenv('JP_YEN_WEAK_THRESHOLD'))
            
            # 선별 설정
            selection_config = jp_config.setdefault('stock_selection', {})
            if os.getenv('JP_MIN_MARKET_CAP'):
                selection_config['min_market_cap'] = int(os.getenv('JP_MIN_MARKET_CAP'))
            if os.getenv('JP_TARGET_STOCKS'):
                selection_config['target_stocks'] = int(os.getenv('JP_TARGET_STOCKS'))
            
            # 리스크 관리 설정
            risk_config = jp_config.setdefault('risk_management', {})
            if os.getenv('JP_BASE_STOP_LOSS'):
                risk_config['base_stop_loss'] = float(os.getenv('JP_BASE_STOP_LOSS'))
            if os.getenv('JP_BASE_TAKE_PROFIT'):
                risk_config['base_take_profit'] = float(os.getenv('JP_BASE_TAKE_PROFIT'))
            
            logger.info("🔧 환경변수 설정 적용 완료")
            
        except Exception as e:
            logger.error(f"❌ 환경변수 처리 실패: {e}")
    
    def _apply_defaults(self):
        """기본값 적용"""
        try:
            # 일본 전략 기본값
            jp_defaults = {
                'enabled': True,
                'confidence_threshold': 0.65,
                'yen_signals': {
                    'strong_threshold': 105.0,
                    'weak_threshold': 110.0,
                    'update_interval': 300  # 5분
                },
                'technical_indicators': {
                    'rsi_period': 10,
                    'momentum_period': 10,
                    'volume_spike_threshold': 1.3,
                    'ma_periods': [5, 20, 60]
                },
                'stock_selection': {
                    'min_market_cap': 500000000000,  # 5000억엔
                    'min_avg_volume': 1000000,       # 100만주
                    'target_stocks': 20,
                    'cache_hours': 24,
                    'sector_diversity': True
                },
                'risk_management': {
                    'base_stop_loss': 0.08,      # 8%
                    'base_take_profit': 0.15,    # 15%
                    'max_hold_days': 30,
                    'position_sizing': 'dynamic'
                },
                'split_trading': {
                    'enabled': True,
                    'buy_steps': 3,
                    'sell_steps': 2,
                    'ratios': {
                        'high_confidence': [0.5, 0.3, 0.2],
                        'medium_confidence': [0.4, 0.35, 0.25],
                        'low_confidence': [0.3, 0.35, 0.35]
                    }
                },
                'data_sources': {
                    'primary': 'yfinance',
                    'backup': 'fallback_list',
                    'timeout': 30,
                    'max_retries': 3
                }
            }
            
            # 기본값 병합 (기존 값 우선)
            if 'jp_strategy' not in self.config:
                self.config['jp_strategy'] = {}
            
            self._deep_merge(self.config['jp_strategy'], jp_defaults)
            
            logger.info("🔧 기본값 적용 완료")
            
        except Exception as e:
            logger.error(f"❌ 기본값 적용 실패: {e}")
    
    def _deep_merge(self, target: Dict, source: Dict):
        """딕셔너리 깊은 병합"""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
    
    def get(self, key_path: str, default=None):
        """점 표기법으로 설정값 조회"""
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def get_jp_config(self) -> Dict:
        """일본 전략 설정 전체 반환"""
        return self.config.get('jp_strategy', {})

# ========================================================================================
# 📊 기술적 지표 분석 클래스 (설정 연동)
# ========================================================================================
class TechnicalIndicators:
    """🔧 기술적 지표 계산 및 분석 (설정 기반)"""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.rsi_period = config.get('jp_strategy.technical_indicators.rsi_period', 10)
        self.volume_threshold = config.get('jp_strategy.technical_indicators.volume_spike_threshold', 1.3)
        self.ma_periods = config.get('jp_strategy.technical_indicators.ma_periods', [5, 20, 60])

    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> float:
        """RSI 계산"""
        try:
            period = period or self.rsi_period
            rsi = safe_rsi(data['Close'], period)
            return safe_float(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0)
        except:
            return 50.0

    def calculate_macd(self, data: pd.DataFrame) -> Tuple[str, Dict]:
        """MACD 계산 및 신호 분석"""
        try:
            macd_line, macd_signal, macd_histogram = safe_macd(data['Close'])
            
            macd_value = safe_float(macd_line.iloc[-1])
            signal_value = safe_float(macd_signal.iloc[-1])
            histogram_value = safe_float(macd_histogram.iloc[-1])

            # 신호 분석
            if macd_value > signal_value and histogram_value > 0:
                signal = 'bullish'
            elif macd_value < signal_value and histogram_value < 0:
                signal = 'bearish'
            else:
                signal = 'neutral'

            details = {
                'macd_line': macd_value,
                'macd_signal': signal_value,
                'histogram': histogram_value
            }

            return signal, details
        except:
            return 'neutral', {}

    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20) -> Tuple[str, Dict]:
        """볼린저 밴드 계산 및 신호 분석"""
        try:
            bb_upper, bb_middle, bb_lower = safe_bollinger_bands(data['Close'], window)
            
            upper_value = safe_float(bb_upper.iloc[-1])
            middle_value = safe_float(bb_middle.iloc[-1])
            lower_value = safe_float(bb_lower.iloc[-1])
            current_price = safe_float(data['Close'].iloc[-1])

            # 신호 분석
            if current_price >= upper_value:
                signal = 'upper'  # 과매수 구간
            elif current_price <= lower_value:
                signal = 'lower'  # 과매도 구간
            else:
                signal = 'middle' # 정상 구간

            details = {
                'upper': upper_value,
                'middle': middle_value,
                'lower': lower_value,
                'position': (current_price - lower_value) / (upper_value - lower_value) if upper_value != lower_value else 0.5
            }

            return signal, details
        except:
            return 'middle', {}

    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14) -> Tuple[str, Dict]:
        """스토캐스틱 계산 및 신호 분석"""
        try:
            stoch_k, stoch_d = safe_stochastic(data['High'], data['Low'], data['Close'], k_period)
            
            k_value = safe_float(stoch_k.iloc[-1])
            d_value = safe_float(stoch_d.iloc[-1])

            # 신호 분석
            if k_value <= 20 and d_value <= 20:
                signal = 'oversold'  # 과매도
            elif k_value >= 80 and d_value >= 80:
                signal = 'overbought'  # 과매수
            else:
                signal = 'neutral'   # 중립

            details = {
                'stoch_k': k_value,
                'stoch_d': d_value
            }

            return signal, details
        except:
            return 'neutral', {}

    def calculate_moving_averages(self, data: pd.DataFrame) -> Tuple[str, Dict]:
        """이동평균선 분석"""
        try:
            ma5 = safe_float(data['Close'].rolling(self.ma_periods[0]).mean().iloc[-1])
            ma20 = safe_float(data['Close'].rolling(self.ma_periods[1]).mean().iloc[-1])
            ma60 = safe_float(data['Close'].rolling(self.ma_periods[2]).mean().iloc[-1])
            current_price = safe_float(data['Close'].iloc[-1])

            # 추세 분석
            if ma5 > ma20 > ma60 and current_price > ma5:
                trend = 'uptrend'
            elif ma5 < ma20 < ma60 and current_price < ma5:
                trend = 'downtrend'
            else:
                trend = 'sideways'

            details = {
                'ma5': ma5,
                'ma20': ma20,
                'ma60': ma60,
                'current_price': current_price
            }

            return trend, details
        except:
            return 'sideways', {}

# ========================================================================================
# 🆕 실시간 종목 수집 및 선별 클래스 (설정 연동)
# ========================================================================================
class RealTimeJPStockSelector:
    """🆕 실시간 일본 주식 종목 수집 및 선별 (설정 기반)"""
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 설정에서 로드
        selection_config = config.get_jp_config().get('stock_selection', {})
        self.min_market_cap = selection_config.get('min_market_cap', 500_000_000_000)
        self.min_avg_volume = selection_config.get('min_avg_volume', 1_000_000)
        self.target_stocks = selection_config.get('target_stocks', 20)
        self.timeout = config.get('jp_strategy.data_sources.timeout', 30)
        self.max_retries = config.get('jp_strategy.data_sources.max_retries', 3)
        
    async def get_nikkei225_constituents(self) -> List[str]:
        """닛케이225 구성종목 실시간 수집 (설정 기반)"""
        try:
            logger.info("🔍 닛케이225 구성종목 실시간 수집 시작...")
            
            symbols = []
            
            # 소스 1: Yahoo Finance Japan (재시도 로직)
            for attempt in range(self.max_retries):
                try:
                    url = "https://finance.yahoo.com/quote/%5EN225/components"
                    response = self.session.get(url, timeout=self.timeout)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        for link in soup.find_all('a', href=True):
                            href = link.get('href', '')
                            if '/quote/' in href and '.T' in href:
                                symbol = href.split('/quote/')[-1].split('?')[0]
                                if symbol.endswith('.T') and len(symbol) <= 8:
                                    symbols.append(symbol)
                    
                    logger.info(f"✅ Yahoo Finance에서 {len(symbols)}개 종목 수집 (시도 {attempt + 1})")
                    break
                    
                except Exception as e:
                    logger.warning(f"Yahoo Finance 수집 실패 (시도 {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        logger.error("모든 시도 실패, 백업 리스트 사용")
            
            # 소스 2: 백업 종목 리스트 (설정에서 확장 가능)
            backup_symbols = self._get_backup_symbols()
            
            # 중복 제거 및 병합
            all_symbols = list(set(symbols + backup_symbols))
            
            logger.info(f"📊 총 {len(all_symbols)}개 후보 종목 수집 완료")
            return all_symbols
            
        except Exception as e:
            logger.error(f"닛케이225 구성종목 수집 실패: {e}")
            return self._get_backup_symbols()

    def _get_backup_symbols(self) -> List[str]:
        """백업 종목 리스트 (설정에서 확장 가능)"""
        return [
            # 자동차
            '7203.T', '7267.T', '7201.T', 
            # 전자/기술
            '6758.T', '6861.T', '9984.T', '4689.T', '6954.T', '6981.T', '8035.T', '6902.T',
            # 금융
            '8306.T', '8316.T', '8411.T', '8604.T', '7182.T',
            # 통신
            '9432.T', '9433.T', '9437.T',
            # 소매/유통
            '9983.T', '3382.T', '8267.T',
            # 의료/제약
            '4568.T', '4502.T', '4506.T',
            # 에너지/유틸리티
            '5020.T', '9501.T', '9502.T',
            # 화학/소재
            '4063.T', '3407.T', '5401.T',
            # 부동산
            '8801.T', '8802.T',
            # 기타 대형주
            '2914.T', '7974.T', '4578.T'
        ]

    async def get_stock_fundamental_data(self, symbol: str) -> Dict:
        """개별 종목 기본 데이터 수집 (재시도 로직)"""
        for attempt in range(self.max_retries):
            try:
                stock = yf.Ticker(symbol)
                
                # 기본 정보
                info = stock.info
                
                # 가격 데이터 (6개월)
                hist = stock.history(period="6mo")
                if hist.empty:
                    continue
                
                current_price = safe_float(hist['Close'].iloc[-1])
                
                # 기본 재무 지표
                data = {
                    'symbol': symbol,
                    'price': current_price,
                    'market_cap': safe_float(info.get('marketCap', 0)),
                    'avg_volume': safe_float(info.get('averageVolume', 0)),
                    'pe_ratio': safe_float(info.get('trailingPE', 0)),
                    'pbr': safe_float(info.get('priceToBook', 0)),
                    'roe': safe_float(info.get('returnOnEquity', 0)) * 100,
                    'debt_to_equity': safe_float(info.get('debtToEquity', 0)),
                    'revenue_growth': safe_float(info.get('revenueQuarterlyGrowth', 0)) * 100,
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                }
                
                # 기술적 지표 추가
                if len(hist) >= 30:
                    rsi = safe_rsi(hist['Close'])
                    data['rsi'] = safe_float(rsi.iloc[-1] if len(rsi) > 0 else 50)
                    data['ma20'] = safe_float(hist['Close'].rolling(20).mean().iloc[-1])
                    recent_vol = hist['Volume'].tail(5).mean()
                    avg_vol = hist['Volume'].tail(20).mean()
                    data['volume_ratio'] = recent_vol / avg_vol if avg_vol > 0 else 1
                
                return data
                
            except Exception as e:
                logger.warning(f"종목 데이터 수집 실패 {symbol} (시도 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)  # 재시도 전 대기
                    
        return {}

    def calculate_selection_score(self, data: Dict) -> float:
        """종목 선별 점수 계산 (설정 기반 가중치)"""
        try:
            score = 0.0
            
            # 가중치 (설정에서 조정 가능)
            weights = {
                'market_cap': 0.25,
                'volume': 0.20,
                'financial': 0.25,
                'technical': 0.20,
                'sector': 0.10
            }
            
            # 1. 시가총액 점수
            market_cap = data.get('market_cap', 0)
            if market_cap >= 2_000_000_000_000:  # 2조엔 이상
                score += weights['market_cap']
            elif market_cap >= 1_000_000_000_000:  # 1조엔 이상
                score += weights['market_cap'] * 0.8
            elif market_cap >= 500_000_000_000:   # 5000억엔 이상
                score += weights['market_cap'] * 0.6
            
            # 2. 거래량 점수
            avg_volume = data.get('avg_volume', 0)
            if avg_volume >= 5_000_000:   # 500만주 이상
                score += weights['volume']
            elif avg_volume >= 2_000_000: # 200만주 이상
                score += weights['volume'] * 0.75
            elif avg_volume >= 1_000_000: # 100만주 이상
                score += weights['volume'] * 0.5
            
            # 3. 재무 건전성 점수
            pe_ratio = data.get('pe_ratio', 999)
            pbr = data.get('pbr', 999)
            roe = data.get('roe', 0)
            
            financial_score = 0
            # PE 점수
            if 5 <= pe_ratio <= 20:
                financial_score += 0.4
            elif 20 < pe_ratio <= 30:
                financial_score += 0.25
            
            # PBR 점수
            if 0.5 <= pbr <= 2.0:
                financial_score += 0.4
            elif 2.0 < pbr <= 3.0:
                financial_score += 0.25
            
            # ROE 점수
            if roe >= 15:
                financial_score += 0.2
            elif roe >= 10:
                financial_score += 0.15
            elif roe >= 5:
                financial_score += 0.1
            
            score += weights['financial'] * financial_score
            
            # 4. 기술적 지표 점수
            rsi = data.get('rsi', 50)
            volume_ratio = data.get('volume_ratio', 1)
            
            technical_score = 0
            # RSI 점수
            if 30 <= rsi <= 70:
                technical_score += 0.5
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                technical_score += 0.25
            
            # 거래량 추세 점수
            if volume_ratio >= 1.2:
                technical_score += 0.5
            elif volume_ratio >= 1.0:
                technical_score += 0.25
            
            score += weights['technical'] * technical_score
            
            # 5. 섹터 보너스
            sector = data.get('sector', '')
            if sector in ['Technology', 'Consumer Cyclical', 'Industrials']:
                score += weights['sector']
            elif sector in ['Healthcare', 'Financial Services']:
                score += weights['sector'] * 0.7
            else:
                score += weights['sector'] * 0.5
            
            return min(score, 1.0)  # 최대 1.0으로 제한
            
        except Exception as e:
            logger.error(f"선별 점수 계산 실패: {e}")
            return 0.0

    def classify_stock_type(self, data: Dict) -> str:
        """종목 타입 분류 (수출주/내수주)"""
        try:
            sector = data.get('sector', '').lower()
            industry = data.get('industry', '').lower()
            
            # 수출주 키워드
            export_keywords = [
                'technology', 'automotive', 'electronics', 'machinery', 
                'industrial', 'semiconductor', 'chemical', 'materials'
            ]
            
            # 내수주 키워드
            domestic_keywords = [
                'financial', 'banking', 'insurance', 'retail', 'utilities',
                'telecommunications', 'real estate', 'healthcare', 'consumer'
            ]
            
            sector_industry = f"{sector} {industry}"
            
            for keyword in export_keywords:
                if keyword in sector_industry:
                    return 'export'
            
            for keyword in domestic_keywords:
                if keyword in sector_industry:
                    return 'domestic'
            
            return 'mixed'  # 분류 불분명
            
        except:
            return 'mixed'

    async def select_top_stocks(self, candidate_symbols: List[str]) -> List[Dict]:
        """상위 종목 선별 (설정 기반 병렬 처리)"""
        logger.info(f"🎯 {len(candidate_symbols)}개 후보에서 상위 {self.target_stocks}개 선별 시작...")
        
        scored_stocks = []
        
        # 병렬 처리로 속도 향상
        max_workers = min(10, len(candidate_symbols))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for symbol in candidate_symbols:
                future = executor.submit(self._process_single_stock, symbol)
                futures.append(future)
            
            for i, future in enumerate(futures, 1):
                try:
                    result = future.result(timeout=self.timeout)
                    if result:
                        scored_stocks.append(result)
                        
                    # 진행상황 표시
                    if i % 10 == 0:
                        logger.info(f"📊 진행상황: {i}/{len(candidate_symbols)} 완료")
                        
                except Exception as e:
                    logger.warning(f"종목 처리 실패: {e}")
                    continue
        
        if not scored_stocks:
            logger.error("선별된 종목이 없습니다!")
            return []
        
        # 점수 기준 정렬
        scored_stocks.sort(key=lambda x: x['selection_score'], reverse=True)
        
        # 섹터 다양성 고려하여 최종 선별
        final_selection = self._ensure_sector_diversity(scored_stocks)
        
        logger.info(f"🏆 최종 {len(final_selection)}개 종목 선별 완료!")
        
        # 선별 결과 로그
        for i, stock in enumerate(final_selection[:10], 1):
            logger.info(f"  {i}. {stock['symbol']}: 점수 {stock['selection_score']:.3f} "
                       f"시총 {stock['market_cap']/1e12:.2f}조엔 ({stock['stock_type']})")
        
        return final_selection

    def _process_single_stock(self, symbol: str) -> Optional[Dict]:
        """단일 종목 처리"""
        try:
            # 데이터 수집
            data = asyncio.run(self.get_stock_fundamental_data(symbol))
            if not data:
                return None
            
            # 기본 필터링
            market_cap = data.get('market_cap', 0)
            avg_volume = data.get('avg_volume', 0)
            
            if market_cap < self.min_market_cap or avg_volume < self.min_avg_volume:
                return None
            
            # 선별 점수 계산
            selection_score = self.calculate_selection_score(data)
            
            # 종목 타입 분류
            stock_type = self.classify_stock_type(data)
            
            result = data.copy()
            result.update({
                'selection_score': selection_score,
                'stock_type': stock_type
            })
            
            return result
            
        except Exception as e:
            logger.error(f"종목 처리 실패 {symbol}: {e}")
            return None

    def _ensure_sector_diversity(self, scored_stocks: List[Dict]) -> List[Dict]:
        """섹터 다양성 확보 (설정 기반)"""
        try:
            final_selection = []
            sector_counts = {}
            max_per_sector = max(1, self.target_stocks // 5)  # 섹터당 최대
            
            # 설정에서 다양성 옵션 확인
            diversity_enabled = self.config.get('jp_strategy.stock_selection.sector_diversity', True)
            
            if not diversity_enabled:
                # 다양성 무시하고 점수 순으로 선별
                return scored_stocks[:self.target_stocks]
            
            # 수출주/내수주 균형 (50:50 목표)
            export_stocks = [s for s in scored_stocks if s['stock_type'] == 'export']
            domestic_stocks = [s for s in scored_stocks if s['stock_type'] == 'domestic']
            mixed_stocks = [s for s in scored_stocks if s['stock_type'] == 'mixed']
            
            target_export = self.target_stocks // 2
            target_domestic = self.target_stocks // 2
            
            # 수출주 선별
            for stock in export_stocks:
                if len([s for s in final_selection if s['stock_type'] == 'export']) < target_export:
                    sector = stock.get('sector', 'Unknown')
                    if sector_counts.get(sector, 0) < max_per_sector:
                        final_selection.append(stock)
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # 내수주 선별
            for stock in domestic_stocks:
                if len([s for s in final_selection if s['stock_type'] == 'domestic']) < target_domestic:
                    sector = stock.get('sector', 'Unknown')
                    if sector_counts.get(sector, 0) < max_per_sector:
                        final_selection.append(stock)
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # 부족한 부분을 mixed로 채움
            remaining_slots = self.target_stocks - len(final_selection)
            for stock in mixed_stocks:
                if remaining_slots <= 0:
                    break
                if stock not in final_selection:
                    final_selection.append(stock)
                    remaining_slots -= 1
            
            # 아직 부족하면 점수 순으로 채움
            remaining_slots = self.target_stocks - len(final_selection)
            if remaining_slots > 0:
                for stock in scored_stocks:
                    if remaining_slots <= 0:
                        break
                    if stock not in final_selection:
                        final_selection.append(stock)
                        remaining_slots -= 1
            
            return final_selection[:self.target_stocks]
            
        except Exception as e:
            logger.error(f"섹터 다양성 확보 실패: {e}")
            return scored_stocks[:self.target_stocks]

# ========================================================================================
# 💰 분할매매 관리 클래스 (설정 연동)
# ========================================================================================
class PositionManager:
    """🔧 분할매매 및 포지션 관리 (설정 기반)"""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.enabled = config.get('jp_strategy.split_trading.enabled', True)
        self.buy_steps = config.get('jp_strategy.split_trading.buy_steps', 3)
        self.sell_steps = config.get('jp_strategy.split_trading.sell_steps', 2)
        self.ratios = config.get('jp_strategy.split_trading.ratios', {})

    def create_split_buy_plan(self, total_amount: float, current_price: float, 
                            confidence: float) -> Tuple[int, List[Dict]]:
        """분할 매수 계획 생성 (설정 기반)"""
        try:
            if not self.enabled:
                total_shares = int(total_amount / current_price / 100) * 100
                return total_shares, []

            # 신뢰도에 따른 분할 전략 (설정에서 로드)
            if confidence >= 0.8:
                ratios = self.ratios.get('high_confidence', [0.5, 0.3, 0.2])
                triggers = [0, -0.02, -0.04]  # 0%, -2%, -4%
            elif confidence >= 0.6:
                ratios = self.ratios.get('medium_confidence', [0.4, 0.35, 0.25])
                triggers = [0, -0.03, -0.05]  # 0%, -3%, -5%
            else:
                ratios = self.ratios.get('low_confidence', [0.3, 0.35, 0.35])
                triggers = [0, -0.04, -0.06]  # 0%, -4%, -6%

            total_shares = int(total_amount / current_price / 100) * 100  # 100주 단위
            split_plan = []

            for i, (ratio, trigger) in enumerate(zip(ratios, triggers)):
                shares = int(total_shares * ratio / 100) * 100
                target_price = current_price * (1 + trigger)

                split_plan.append({
                    'step': i + 1,
                    'shares': shares,
                    'target_price': target_price,
                    'ratio': ratio,
                    'executed': False
                })

            return total_shares, split_plan

        except Exception as e:
            logger.error(f"분할 매수 계획 생성 실패: {e}")
            return 0, []

    def create_split_sell_plan(self, total_shares: int, current_price: float, 
                             target_price: float, confidence: float) -> List[Dict]:
        """분할 매도 계획 생성 (설정 기반)"""
        try:
            if not self.enabled:
                return []

            # 신뢰도에 따른 매도 전략
            if confidence >= 0.8:
                sell_ratios = [0.5, 0.5]
                price_targets = [target_price, target_price * 1.1]
            else:
                sell_ratios = [0.7, 0.3]
                price_targets = [target_price, target_price * 1.05]

            split_plan = []
            remaining_shares = total_shares

            for i, (ratio, price_target) in enumerate(zip(sell_ratios, price_targets)):
                shares = int(remaining_shares * ratio / 100) * 100
                remaining_shares -= shares

                split_plan.append({
                    'step': i + 1,
                    'shares': shares,
                    'target_price': price_target,
                    'ratio': ratio,
                    'executed': False
                })

            return split_plan

        except Exception as e:
            logger.error(f"분할 매도 계획 생성 실패: {e}")
            return []

# ========================================================================================
# 🇯🇵 메인 일본 주식 전략 클래스 (완전 통합)
# ========================================================================================
class JPStrategy:
    """🇯🇵 일본 주식 완전 자동화 전략 (설정 완전 통합)"""

    def __init__(self, config_path: str = "settings.yaml"):
        """전략 초기화 (설정 파일 기반)"""
    def __init__(self, config_path: str = "settings.yaml"):
        """전략 초기화 (설정 파일 기반)"""
        # 설정 로드
        self.config = ConfigLoader(config_path)
        self.jp_config = self.config.get_jp_config()
        self.enabled = self.jp_config.get('enabled', True)

        # 🆕 실시간 종목 선별기 (설정 기반)
        self.stock_selector = RealTimeJPStockSelector(self.config)
        
        # 📊 기술적 지표 분석기 (설정 기반)
        self.technical_analyzer = TechnicalIndicators(self.config)
        
        # 💰 포지션 매니저 (설정 기반)
        self.position_manager = PositionManager(self.config)

        # 💱 엔화 매매 설정 (설정에서 로드)
        yen_config = self.jp_config.get('yen_signals', {})
        self.yen_strong_threshold = yen_config.get('strong_threshold', 105)
        self.yen_weak_threshold = yen_config.get('weak_threshold', 110)
        self.current_usd_jpy = 0.0

        # 🛡️ 손절/익절 설정 (설정에서 로드)
        risk_config = self.jp_config.get('risk_management', {})
        self.base_stop_loss = risk_config.get('base_stop_loss', 0.08)
        self.base_take_profit = risk_config.get('base_take_profit', 0.15)
        self.max_hold_days = risk_config.get('max_hold_days', 30)

        # 🔍 자동 선별된 종목들 (동적으로 업데이트)
        self.selected_stocks = []
        self.last_selection_time = None
        self.selection_cache_hours = self.jp_config.get('stock_selection', {}).get('cache_hours', 24)

        # 로그 디렉토리 생성
        Path('logs').mkdir(exist_ok=True)

        if self.enabled:
            logger.info(f"🇯🇵 일본 주식 완전 자동화 전략 초기화 (V5.0 - 설정 통합)")
            logger.info(f"🔧 설정 파일: {config_path}")
            logger.info(f"🆕 실시간 닛케이225 + TOPIX 자동 선별 시스템")
            logger.info(f"📊 설정 기반 기술적 지표 + 엔화 + 펀더멘털 종합 분석")
            logger.info(f"💰 설정 기반 분할매매 + 동적 손절익절")
            logger.info(f"🎯 목표 종목: {self.stock_selector.target_stocks}개")
            logger.info(f"💱 엔화 임계값: 강세({self.yen_strong_threshold}) 약세({self.yen_weak_threshold})")
        else:
            logger.info("🇯🇵 일본 주식 전략이 비활성화되어 있습니다")

    # ========================================================================================
    # 🆕 실시간 자동 선별 메서드들 (설정 통합)
    # ========================================================================================

    async def auto_select_stocks(self) -> List[str]:
        """🆕 실시간 주식 자동 선별 (설정 기반)"""
        if not self.enabled:
            logger.warning("일본 주식 전략이 비활성화되어 있습니다")
            return []

        try:
            # 캐시 확인
            if self._is_selection_cache_valid():
                logger.info("📋 캐시된 선별 결과 사용")
                return [stock['symbol'] for stock in self.selected_stocks]

            logger.info("🔍 실시간 일본 주식 자동 선별 시작!")
            start_time = time.time()

            # 1단계: 닛케이225 구성종목 수집
            candidate_symbols = await self.stock_selector.get_nikkei225_constituents()
            if not candidate_symbols:
                logger.error("후보 종목 수집 실패")
                return self._get_fallback_stocks()

            # 2단계: 상위 종목 선별
            selected_data = await self.stock_selector.select_top_stocks(candidate_symbols)
            if not selected_data:
                logger.error("종목 선별 실패")
                return self._get_fallback_stocks()

            # 3단계: 선별 결과 저장
            self.selected_stocks = selected_data
            self.last_selection_time = datetime.now()

            # 결과 정리
            selected_symbols = [stock['symbol'] for stock in selected_data]
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ 자동 선별 완료! {len(selected_symbols)}개 종목 ({elapsed_time:.1f}초 소요)")

            # 선별 결과 요약
            export_count = len([s for s in selected_data if s['stock_type'] == 'export'])
            domestic_count = len([s for s in selected_data if s['stock_type'] == 'domestic'])
            mixed_count = len([s for s in selected_data if s['stock_type'] == 'mixed'])

            logger.info(f"📊 종목 구성: 수출주 {export_count}개, 내수주 {domestic_count}개, 혼합 {mixed_count}개")

            # 평균 선별 점수
            avg_score = np.mean([s['selection_score'] for s in selected_data])
            logger.info(f"🎯 평균 선별 점수: {avg_score:.3f}")

            return selected_symbols

        except Exception as e:
            logger.error(f"자동 선별 실패: {e}")
            return self._get_fallback_stocks()

    def _is_selection_cache_valid(self) -> bool:
        """선별 결과 캐시 유효성 확인"""
        if not self.last_selection_time or not self.selected_stocks:
            return False
        
        time_diff = datetime.now() - self.last_selection_time
        return time_diff.total_seconds() < (self.selection_cache_hours * 3600)

    def _get_fallback_stocks(self) -> List[str]:
        """백업 종목 리스트 (설정에서 확장 가능)"""
        fallback_symbols = [
            # 대형 수출주
            '7203.T', '7267.T', '6758.T', '6861.T', '9984.T', 
            '6954.T', '7201.T', '6981.T', '8035.T', '6902.T',
            # 대형 내수주
            '8306.T', '8316.T', '8411.T', '9432.T', '9433.T',
            '9983.T', '3382.T', '4568.T', '8801.T', '5020.T'
        ]
        logger.info("📋 백업 종목 리스트 사용")
        return fallback_symbols

    async def get_selected_stock_info(self, symbol: str) -> Dict:
        """선별된 종목의 상세 정보 조회"""
        try:
            # 선별 데이터에서 찾기
            for stock in self.selected_stocks:
                if stock['symbol'] == symbol:
                    return stock
            
            # 없으면 실시간 조회
            return await self.stock_selector.get_stock_fundamental_data(symbol)
            
        except Exception as e:
            logger.error(f"종목 정보 조회 실패 {symbol}: {e}")
            return {}

    def _get_stock_type(self, symbol: str) -> str:
        """종목 타입 확인 (선별 데이터 기반)"""
        try:
            for stock in self.selected_stocks:
                if stock['symbol'] == symbol:
                    return stock.get('stock_type', 'mixed')
            
            # 백업 로직
            export_symbols = ['7203.T', '6758.T', '7974.T', '6861.T', '9984.T', '6954.T', '7201.T']
            domestic_symbols = ['8306.T', '8316.T', '8411.T', '9432.T', '9433.T', '9983.T', '3382.T']
            
            if symbol in export_symbols:
                return 'export'
            elif symbol in domestic_symbols:
                return 'domestic'
            else:
                return 'mixed'
        except:
            return 'mixed'

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """섹터 분류 (선별 데이터 기반)"""
        try:
            for stock in self.selected_stocks:
                if stock['symbol'] == symbol:
                    return stock.get('sector', 'UNKNOWN')
            
            # 백업 매핑
            sector_map = {
                '7203.T': 'Automotive', '6758.T': 'Technology', '7974.T': 'Technology',
                '6861.T': 'Technology', '9984.T': 'Technology', '8306.T': 'Financial',
                '8316.T': 'Financial', '9432.T': 'Telecommunications', '9983.T': 'Retail'
            }
            return sector_map.get(symbol, 'UNKNOWN')
        except:
            return 'UNKNOWN'

        # 💱 엔화 매매 설정 (설정에서 로드)
        yen_config = self.jp_config.get('yen_signals', {})
        self.yen_strong_threshold = yen_config.get('strong_threshold', 105)
        self.yen_weak_threshold = yen_config.get('weak_threshold', 110)
        self.current_usd_jpy = 0.0

        # 🛡️ 손절/익절 설정 (설정에서 로드)
        risk_config = self.jp_config.get('risk_management', {})
        self.base_stop_loss = risk_config.get('base_stop_loss', 0.08)
        self.base_take_profit = risk_config.get('base_take_profit', 0.15)
        self.max_hold_days = risk_config.get('max_hold_days', 30)

        # 로그 디렉토리 생성
        Path('logs').mkdir(exist_ok=True)

        if self.enabled:
            logger.info(f"🇯🇵 일본 주식 완전 자동화 전략 초기화 (V5.0 - 설정 통합)")
            logger.info(f"🔧 설정 파일: {config_path}")
            logger.info(f"💱 엔화 임계값: 강세({self.yen_strong_threshold}) 약세({self.yen_weak_threshold})")
        else:
            logger.info("🇯🇵 일본 주식 전략이 비활성화되어 있습니다")

    # ========================================================================================
    # 🔧 유틸리티 메서드들 (설정 통합)
    # ========================================================================================

    async def _update_yen_rate(self):
        """USD/JPY 환율 업데이트"""
        try:
            ticker = yf.Ticker("USDJPY=X")
            data = ticker.history(period="1d")
            if not data.empty:
                self.current_usd_jpy = safe_float(data['Close'].iloc[-1], 107.5)
            else:
                self.current_usd_jpy = 107.5  # 기본값
        except Exception as e:
            logger.error(f"환율 조회 오류: {e}")
            self.current_usd_jpy = 107.5

    def _get_yen_signal(self) -> str:
        """엔화 신호 분석 (설정 기반)"""
        if self.current_usd_jpy <= self.yen_strong_threshold:
            return 'strong'  # 엔화 강세
        elif self.current_usd_jpy >= self.yen_weak_threshold:
            return 'weak'    # 엔화 약세
        else:
            return 'neutral'

    def _get_stock_type(self, symbol: str) -> str:
        """종목 타입 확인 (간단 분류)"""
        # 간단한 분류 로직 (실제로는 더 정교해야 함)
        export_symbols = ['7203.T', '6758.T', '7974.T', '6861.T', '9984.T', '6954.T', '7201.T']
        domestic_symbols = ['8306.T', '8316.T', '8411.T', '9432.T', '9433.T', '9983.T', '3382.T']
        
        if symbol in export_symbols:
            return 'export'
        elif symbol in domestic_symbols:
            return 'domestic'
        else:
            return 'mixed'

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """섹터 분류 (간단 매핑)"""
        sector_map = {
            '7203.T': 'Automotive', '6758.T': 'Technology', '7974.T': 'Technology',
            '6861.T': 'Technology', '9984.T': 'Technology', '8306.T': 'Financial',
            '8316.T': 'Financial', '9432.T': 'Telecommunications', '9983.T': 'Retail'
        }
        return sector_map.get(symbol, 'UNKNOWN')

    async def _get_stock_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """주식 데이터 수집 (재시도 로직)"""
        max_retries = self.config.get('jp_strategy.data_sources.max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                if not data.empty:
                    return data
                
            except Exception as e:
                logger.warning(f"주식 데이터 수집 실패 {symbol} (시도 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        logger.error(f"주식 데이터 수집 완전 실패: {symbol}")
        return pd.DataFrame()

    # ========================================================================================
    # 📊 통합 기술적 분석 메서드 (설정 기반)
    # ========================================================================================

    def _analyze_technical_indicators(self, data: pd.DataFrame) -> Tuple[float, Dict]:
        """통합 기술적 지표 분석 (설정 기반)"""
        try:
            if len(data) < 60:  # 충분한 데이터 필요
                return 0.0, {}

            # 각 지표 계산
            rsi = self.technical_analyzer.calculate_rsi(data)
            macd_signal, macd_details = self.technical_analyzer.calculate_macd(data)
            bb_signal, bb_details = self.technical_analyzer.calculate_bollinger_bands(data)
            stoch_signal, stoch_details = self.technical_analyzer.calculate_stochastic(data)
            ma_trend, ma_details = self.technical_analyzer.calculate_moving_averages(data)

            # 거래량 분석
            volume = data['Volume']
            recent_volume = volume.tail(3).mean()
            avg_volume = volume.tail(15).head(12).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            volume_threshold = self.technical_analyzer.volume_threshold

            # 종합 점수 계산 (설정 기반 가중치)
            weights = {
                'rsi': 0.2,
                'macd': 0.25,
                'bollinger': 0.2,
                'stochastic': 0.15,
                'ma_trend': 0.15,
                'volume': 0.05
            }

            total_score = 0.0

            # 1. RSI 점수
            if 30 <= rsi <= 70:
                total_score += weights['rsi'] * 0.8
            elif rsi < 30:
                total_score += weights['rsi'] * 1.0  # 과매도 (매수 기회)
            elif rsi > 70:
                total_score += weights['rsi'] * 0.3  # 과매수 (주의)

            # 2. MACD 점수
            if macd_signal == 'bullish':
                total_score += weights['macd'] * 1.0
            elif macd_signal == 'bearish':
                total_score += weights['macd'] * 0.2
            else:
                total_score += weights['macd'] * 0.5

            # 3. 볼린저 밴드 점수
            if bb_signal == 'lower':
                total_score += weights['bollinger'] * 1.0  # 과매도
            elif bb_signal == 'upper':
                total_score += weights['bollinger'] * 0.3  # 과매수
            else:
                total_score += weights['bollinger'] * 0.6  # 중간

            # 4. 스토캐스틱 점수
            if stoch_signal == 'oversold':
                total_score += weights['stochastic'] * 1.0
            elif stoch_signal == 'overbought':
                total_score += weights['stochastic'] * 0.3
            else:
                total_score += weights['stochastic'] * 0.6

            # 5. 이동평균 추세 점수
            if ma_trend == 'uptrend':
                total_score += weights['ma_trend'] * 1.0
            elif ma_trend == 'downtrend':
                total_score += weights['ma_trend'] * 0.2
            else:
                total_score += weights['ma_trend'] * 0.5

            # 6. 거래량 보너스
            if volume_ratio >= volume_threshold:
                total_score += weights['volume'] * 1.0

            # 정규화된 점수
            technical_score = min(total_score, 1.0)

            # 상세 정보
            details = {
                'rsi': rsi,
                'macd_signal': macd_signal,
                'macd_details': macd_details,
                'bollinger_signal': bb_signal,
                'bollinger_details': bb_details,
                'stochastic_signal': stoch_signal,
                'stochastic_details': stoch_details,
                'ma_trend': ma_trend,
                'ma_details': ma_details,
                'volume_ratio': volume_ratio,
                'technical_score': technical_score
            }

            return technical_score, details

        except Exception as e:
            logger.error(f"기술적 지표 분석 실패: {e}")
            return 0.0, {}

    # ========================================================================================
    # 💱 엔화 + 기술적 지표 통합 분석 (설정 기반)
    # ========================================================================================

    def _analyze_yen_technical_signal(self, symbol: str, technical_score: float, 
                                    technical_details: Dict) -> Tuple[str, float, str]:
        """엔화 + 기술적 지표 통합 분석 (설정 기반)"""
        try:
            yen_signal = self._get_yen_signal()
            stock_type = self._get_stock_type(symbol)
            confidence_threshold = self.jp_config.get('confidence_threshold', 0.65)

            total_score = 0.0
            reasons = []

            # 1. 엔화 기반 점수 (40% 가중치)
            yen_score = 0.0
            if yen_signal == 'strong' and stock_type == 'domestic':
                yen_score = 0.4
                reasons.append("엔화강세+내수주")
            elif yen_signal == 'weak' and stock_type == 'export':
                yen_score = 0.4
                reasons.append("엔화약세+수출주")
            elif yen_signal == 'neutral':
                yen_score = 0.2
                reasons.append("엔화중립")
            else:
                yen_score = 0.1
                reasons.append("엔화불리")

            total_score += yen_score

            # 2. 기술적 지표 점수 (60% 가중치)
            tech_weighted = technical_score * 0.6
            total_score += tech_weighted

            # 기술적 지표 설명
            rsi = technical_details.get('rsi', 50)
            macd_signal = technical_details.get('macd_signal', 'neutral')
            bb_signal = technical_details.get('bollinger_signal', 'middle')
            ma_trend = technical_details.get('ma_trend', 'sideways')

            tech_reasons = []
            if technical_score >= 0.7:
                tech_reasons.append("기술적강세")
            elif technical_score <= 0.4:
                tech_reasons.append("기술적약세")
            else:
                tech_reasons.append("기술적중립")

            tech_reasons.append(f"RSI({rsi:.0f})")
            tech_reasons.append(f"MACD({macd_signal})")
            tech_reasons.append(f"추세({ma_trend})")

            reasons.extend(tech_reasons)

            # 최종 판단 (설정 기반 임계값)
            buy_threshold = confidence_threshold
            sell_threshold = 1 - confidence_threshold

            if total_score >= buy_threshold:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= sell_threshold:
                action = 'sell'
                confidence = min(1 - total_score, 0.95)
            else:
                action = 'hold'
                confidence = 0.5

            reasoning = f"엔화({yen_signal})+{stock_type}: " + " | ".join(reasons)

            return action, confidence, reasoning

        except Exception as e:
            logger.error(f"통합 신호 분석 실패: {e}")
            return 'hold', 0.5, "분석 실패"

    # ========================================================================================
    # 🛡️ 동적 손절/익절 계산 (설정 기반)
    # ========================================================================================

    def _calculate_dynamic_stop_take(self, current_price: float, confidence: float, 
                                   stock_type: str, yen_signal: str) -> Tuple[float, float, int]:
        """동적 손절/익절 계산 (설정 기반)"""
        try:
            # 기본값 (설정에서 로드)
            stop_loss_pct = self.base_stop_loss
            take_profit_pct = self.base_take_profit
            hold_days = self.max_hold_days

            # 1. 엔화 기반 조정
            if yen_signal == 'strong' and stock_type == 'domestic':
                stop_loss_pct = 0.06   # 6%
                take_profit_pct = 0.12 # 12%
                hold_days = 25
            elif yen_signal == 'weak' and stock_type == 'export':
                stop_loss_pct = 0.10   # 10%
                take_profit_pct = 0.18 # 18%
                hold_days = 35
            elif yen_signal != 'neutral':
                stop_loss_pct = 0.05   # 5%
                take_profit_pct = 0.08 # 8%
                hold_days = 20

            # 2. 신뢰도 기반 조정
            if confidence >= 0.8:
                stop_loss_pct *= 0.8    # 손절 타이트
                take_profit_pct *= 1.3  # 익절 크게
                hold_days += 10
            elif confidence <= 0.6:
                stop_loss_pct *= 0.6    # 손절 매우 타이트
                take_profit_pct *= 0.8  # 익절 작게
                hold_days -= 10

            # 3. 범위 제한
            stop_loss_pct = max(0.03, min(0.12, stop_loss_pct))
            take_profit_pct = max(0.05, min(0.25, take_profit_pct))
            hold_days = max(15, min(45, hold_days))

            # 4. 최종 가격 계산
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)

            return stop_loss, take_profit, hold_days

        except Exception as e:
            logger.error(f"손절/익절 계산 실패: {e}")
            return (current_price * 0.92, current_price * 1.15, 30)

    # ========================================================================================
    # 🎯 메인 종목 분석 메서드 (완전 통합)
    # ========================================================================================

    async def analyze_symbol(self, symbol: str) -> JPStockSignal:
        """개별 종목 완전 분석 (설정 기반)"""
        if not self.enabled:
            return self._create_disabled_signal(symbol)

        try:
            # 1. 환율 업데이트
            await self._update_yen_rate()

            # 2. 주식 데이터 수집
            data = await self._get_stock_data(symbol)
            if data.empty:
                raise ValueError(f"주식 데이터 없음: {symbol}")

            current_price = safe_float(data['Close'].iloc[-1])

            # 3. 📊 기술적 지표 분석
            technical_score, technical_details = self._analyze_technical_indicators(data)

            # 4. 💱 엔화 + 기술적 지표 통합 분석
            action, confidence, reasoning = self._analyze_yen_technical_signal(
                symbol, technical_score, technical_details
            )
            
            # 5. 🛡️ 동적 손절/익절 계산
            stop_loss, take_profit, max_hold_days = self._calculate_dynamic_stop_take(
                current_price, confidence, self._get_stock_type(symbol), self._get_yen_signal()
            )
            
            # 6. 💰 분할매매 계획 생성 (설정 기반)
            if action == 'buy':
                # 총 투자금액 (신뢰도에 따라 조정)
                base_amount = 1000000  # 100만엔 기본
                total_amount = base_amount * confidence
                
                position_size, split_buy_plan = self.position_manager.create_split_buy_plan(
                    total_amount, current_price, confidence
                )
                
                split_sell_plan = self.position_manager.create_split_sell_plan(
                    position_size, current_price, take_profit, confidence
                )
            else:
                position_size = 0
                split_buy_plan = []
                split_sell_plan = []
            
            # 7. 📊 선별 정보 추가
            stock_info = await self.get_selected_stock_info(symbol)
            market_cap = stock_info.get('market_cap', 0)
            selection_score = stock_info.get('selection_score', technical_score)
            quality_rank = 0  # 추후 계산
            
            # 8. 📊 JPStockSignal 생성 (모든 정보 포함)
            return JPStockSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                strategy_source='integrated_auto_selection_v5',
                
                # 기술적 지표
                rsi=technical_details.get('rsi', 50.0),
                macd_signal=technical_details.get('macd_signal', 'neutral'),
                bollinger_signal=technical_details.get('bollinger_signal', 'middle'),
                stoch_signal=technical_details.get('stochastic_signal', 'neutral'),
                ma_trend=technical_details.get('ma_trend', 'sideways'),
                
                # 포지션 관리
                position_size=position_size,
                split_buy_plan=split_buy_plan,
                split_sell_plan=split_sell_plan,
                
                # 손익 관리
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_hold_days=max_hold_days,
                
                # 기본 정보
                stock_type=self._get_stock_type(symbol),
                yen_signal=self._get_yen_signal(),
                sector=self._get_sector_for_symbol(symbol),
                reasoning=reasoning,
                timestamp=datetime.now(),
                
                # 자동선별 추가 정보
                market_cap=market_cap,
                selection_score=selection_score,
                quality_rank=quality_rank,
                additional_data={
                    'usd_jpy_rate': self.current_usd_jpy,
                    'technical_score': technical_score,
                    'technical_details': technical_details,
                    'stop_loss_pct': (current_price - stop_loss) / current_price * 100,
                    'take_profit_pct': (take_profit - current_price) / current_price * 100,
                    'selection_method': 'config_based_auto_selection',
                    'config_version': '5.0'
                }
            )
            
        except Exception as e:
            logger.error(f"종목 분석 실패 {symbol}: {e}")
            return self._create_error_signal(symbol, str(e))

    def _create_disabled_signal(self, symbol: str) -> JPStockSignal:
        """비활성화 신호 생성"""
        return JPStockSignal(
            symbol=symbol, action='hold', confidence=0.0, price=0.0,
            strategy_source='disabled', rsi=50.0, macd_signal='neutral',
            bollinger_signal='middle', stoch_signal='neutral', ma_trend='sideways',
            position_size=0, split_buy_plan=[], split_sell_plan=[],
            stop_loss=0.0, take_profit=0.0, max_hold_days=30,
            stock_type='unknown', yen_signal='neutral', sector='UNKNOWN',
            reasoning="전략 비활성화", timestamp=datetime.now(),
            market_cap=0, selection_score=0, quality_rank=0
        )

    def _create_error_signal(self, symbol: str, error_msg: str) -> JPStockSignal:
        """오류 신호 생성"""
        return JPStockSignal(
            symbol=symbol, action='hold', confidence=0.0, price=0.0,
            strategy_source='error', rsi=50.0, macd_signal='neutral',
            bollinger_signal='middle', stoch_signal='neutral', ma_trend='sideways',
            position_size=0, split_buy_plan=[], split_sell_plan=[],
            stop_loss=0.0, take_profit=0.0, max_hold_days=30,
            stock_type='unknown', yen_signal='neutral', sector='UNKNOWN',
            reasoning=f"분석 실패: {error_msg}", timestamp=datetime.now(),
            market_cap=0, selection_score=0, quality_rank=0
        )

    # ========================================================================================
    # 🔍 전체 시장 스캔 (설정 기반 완전 자동화)
    # ========================================================================================
    
    async def scan_all_symbols(self) -> List[JPStockSignal]:
        """전체 자동선별 + 종목 분석 (설정 기반 완전 자동화)"""
        if not self.enabled:
            return []
        
        logger.info(f"🔍 일본 주식 완전 자동 분석 시작! (설정 기반 V5.0)")
        logger.info(f"🔧 설정 파일: {self.config.config_path}")
        logger.info(f"🎯 목표 종목: {self.stock_selector.target_stocks}개")
        logger.info(f"💱 엔화 임계값: 강세({self.yen_strong_threshold}) 약세({self.yen_weak_threshold})")
        
        try:
            # 1단계: 실시간 자동 선별
            selected_symbols = await self.auto_select_stocks()
            if not selected_symbols:
                logger.error("자동 선별 실패")
                return []
            
            # 2단계: 선별된 종목들 상세 분석
            all_signals = []
            
            for i, symbol in enumerate(selected_symbols, 1):
                try:
                    print(f"📊 분석 중... {i}/{len(selected_symbols)} - {symbol}")
                    signal = await self.analyze_symbol(symbol)
                    all_signals.append(signal)
                    
                    # 결과 로그
                    action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                    logger.info(f"{action_emoji} {symbol} ({signal.stock_type}): {signal.action} "
                              f"신뢰도:{signal.confidence:.2f} RSI:{signal.rsi:.0f} "
                              f"선별점수:{signal.selection_score:.3f}")
                    
                    # API 호출 제한
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"❌ {symbol} 분석 실패: {e}")
            
            # 결과 요약
            buy_count = len([s for s in all_signals if s.action == 'buy'])
            sell_count = len([s for s in all_signals if s.action == 'sell'])
            hold_count = len([s for s in all_signals if s.action == 'hold'])
            
            logger.info(f"🎯 설정 기반 완전 자동 분석 완료!")
            logger.info(f"📊 결과: 매수:{buy_count}, 매도:{sell_count}, 보유:{hold_count}")
            logger.info(f"💱 현재 USD/JPY: {self.current_usd_jpy:.2f} ({self._get_yen_signal()})")
            logger.info(f"🆕 자동선별 시간: {self.last_selection_time}")
            logger.info(f"🔧 설정 캐시: {self.selection_cache_hours}시간")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"전체 스캔 실패: {e}")
            return []

    async def scan_symbols(self, symbols: List[str] = None) -> List[JPStockSignal]:
        """선택된 종목들 분석 (설정 기반) - 기존 호환성 유지"""
        if symbols is None:
            # 자동 선별된 종목들 사용
            return await self.scan_all_symbols()
        
        if not self.enabled:
            return []
        
        logger.info(f"🔍 일본 주식 분석 시작! {len(symbols)}개 종목")
        
        try:
            all_signals = []
            
            for i, symbol in enumerate(symbols, 1):
                try:
                    print(f"📊 분석 중... {i}/{len(symbols)} - {symbol}")
                    signal = await self.analyze_symbol(symbol)
                    all_signals.append(signal)
                    
                    # 결과 로그
                    action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                    logger.info(f"{action_emoji} {symbol} ({signal.stock_type}): {signal.action} "
                              f"신뢰도:{signal.confidence:.2f} RSI:{signal.rsi:.0f}")
                    
                    # API 호출 제한
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"❌ {symbol} 분석 실패: {e}")
            
            # 결과 요약
            buy_count = len([s for s in all_signals if s.action == 'buy'])
            sell_count = len([s for s in all_signals if s.action == 'sell'])
            hold_count = len([s for s in all_signals if s.action == 'hold'])
            
            logger.info(f"🎯 분석 완료!")
            logger.info(f"📊 결과: 매수:{buy_count}, 매도:{sell_count}, 보유:{hold_count}")
            logger.info(f"💱 현재 USD/JPY: {self.current_usd_jpy:.2f} ({self._get_yen_signal()})")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"스캔 실패: {e}")
            return []

# ========================================================================================
# 🎯 편의 함수들 (설정 통합)
# ========================================================================================

async def analyze_jp(symbol: str, config_path: str = "settings.yaml") -> Dict:
    """단일 일본 주식 완전 분석 (설정 기반)"""
    # 심볼 검증 및 정리
    if not symbol or symbol in [None, "", "test", "invalid"]:
        symbol = "6758.T"  # 기본값으로 소니 사용
    
    # 심볼이 숫자만 있으면 .T 추가
    if isinstance(symbol, (int, float)):
        symbol = f"{int(symbol)}.T"
    elif isinstance(symbol, str) and symbol.isdigit():
        symbol = f"{symbol}.T"
    
    symbol = safe_upper(symbol)
    
    try:
        strategy = JPStrategy(config_path)
        signal = await strategy.analyze_symbol(symbol)
        
        return {
            'decision': signal.action,
            'confidence': signal.confidence * 100,
            'reasoning': signal.reasoning,
            'current_price': signal.price,
            
            # 기술적 지표
            'rsi': signal.rsi,
            'macd_signal': signal.macd_signal,
            'bollinger_signal': signal.bollinger_signal,
            'stochastic_signal': signal.stoch_signal,
            'ma_trend': signal.ma_trend,
            
            # 포지션 관리
            'position_size': signal.position_size,
            'split_buy_plan': signal.split_buy_plan,
            'split_sell_plan': signal.split_sell_plan,
            
            # 손익 관리
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'max_hold_days': signal.max_hold_days,
            
            # 기본 정보
            'stock_type': signal.stock_type,
            'yen_signal': signal.yen_signal,
            'sector': signal.sector,
            
            # 설정 정보
            'config_version': '5.0',
            'strategy_source': signal.strategy_source
        }
        
    except Exception as e:
        logger.error(f"analyze_jp 실행 오류: {str(e)}")
        return {
            'decision': 'HOLD',
            'confidence': 30,
            'reasoning': f'분석 오류: {str(e)[:100]}',
            'current_price': 0.0,
            'rsi': 50,
            'macd_signal': 'NEUTRAL',
            'bollinger_signal': 'middle',
            'stochastic_signal': 'neutral',
            'ma_trend': 'sideways',
            'position_size': 0,
            'split_buy_plan': [],
            'split_sell_plan': [],
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'max_hold_days': 30,
            'stock_type': 'unknown',
            'yen_signal': 'neutral',
            'sector': 'UNKNOWN',
            'config_version': '5.0',
            'strategy_source': 'error'
        }

async def scan_jp_market(config_path: str = "settings.yaml") -> Dict:
    """일본 시장 전체 자동선별 + 스캔 (설정 기반)"""
    strategy = JPStrategy(config_path)
    signals = await strategy.scan_all_symbols()
    
    buy_signals = [s for s in signals if s.action == 'buy']
    sell_signals = [s for s in signals if s.action == 'sell']
    
    return {
        'total_analyzed': len(signals),
        'buy_count': len(buy_signals),
        'sell_count': len(sell_signals),
        'current_usd_jpy': strategy.current_usd_jpy,
        'yen_signal': strategy._get_yen_signal(),
        'selection_method': 'config_based_auto_selection',
        'last_selection_time': strategy.last_selection_time,
        'config_path': config_path,
        'top_buys': sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5],
        'top_sells': sorted(sell_signals, key=lambda x: x.confidence, reverse=True)[:5],
        'selection_summary': {
            'export_stocks': len([s for s in signals if s.stock_type == 'export']),
            'domestic_stocks': len([s for s in signals if s.stock_type == 'domestic']),
            'mixed_stocks': len([s for s in signals if s.stock_type == 'mixed']),
            'avg_selection_score': np.mean([s.selection_score for s in signals]) if signals else 0,
            'avg_market_cap': np.mean([s.market_cap for s in signals]) / 1e12 if signals else 0,
            'config_version': '5.0'
        }
    }

async def get_jp_config_status(config_path: str = "settings.yaml") -> Dict:
    """일본 주식 설정 상태 조회"""
    strategy = JPStrategy(config_path)
    
    return {
        'config_loaded': True,
        'config_path': config_path,
        'enabled': strategy.enabled,
        'last_selection_time': strategy.last_selection_time,
        'cache_valid': strategy._is_selection_cache_valid(),
        'cache_hours': strategy.selection_cache_hours,
        'selected_count': len(strategy.selected_stocks),
        'current_usd_jpy': strategy.current_usd_jpy,
        'yen_signal': strategy._get_yen_signal(),
        'thresholds': {
            'yen_strong': strategy.yen_strong_threshold,
            'yen_weak': strategy.yen_weak_threshold,
            'confidence': strategy.jp_config.get('confidence_threshold', 0.65)
        },
        'selection_criteria': {
            'min_market_cap': strategy.stock_selector.min_market_cap / 1e12,  # 조엔
            'min_avg_volume': strategy.stock_selector.min_avg_volume / 1e6,   # 백만주
            'target_stocks': strategy.stock_selector.target_stocks
        },
        'risk_management': {
            'base_stop_loss': strategy.base_stop_loss,
            'base_take_profit': strategy.base_take_profit,
            'max_hold_days': strategy.max_hold_days
        },
        'version': '5.0'
    }

# ========================================================================================
# 🧪 테스트 메인 함수 (설정 통합)
# ========================================================================================

async def main():
    """테스트용 메인 함수 (설정 완전 통합 V5.0)"""
    try:
        print("🇯🇵 일본 주식 완전 자동화 전략 V5.0 테스트!")
        print("🔧 settings.yaml + .env + requirements.txt 완전 통합")
        print("📊 기능: 설정기반 엔화+기술지표+분할매매+동적손절익절")
        print("="*80)
        
        # 개별 종목 테스트
        test_symbol = "6758.T"  # 소니
        print(f"\n📊 개별 종목 분석 테스트 - {test_symbol}:")
        result = await analyze_jp(test_symbol)
        
        print(f"  🎯 액션: {result['decision']} (신뢰도: {result['confidence']:.1f}%)")
        print(f"  💰 현재가: {result['current_price']:,.0f}엔")
        print(f"  📊 기술지표:")
        print(f"    - RSI: {result['rsi']:.1f}")
        print(f"    - MACD: {result['macd_signal']}")
        print(f"    - 볼린저밴드: {result['bollinger_signal']}")
        print(f"    - 추세: {result['ma_trend']}")
        print(f"  💱 엔화 정보:")
        print(f"    - 종목타입: {result['stock_type']}")
        print(f"    - 엔화신호: {result['yen_signal']}")
        print(f"  🛡️ 리스크 관리:")
        print(f"    - 손절가: {result['stop_loss']:,.0f}엔")
        print(f"    - 익절가: {result['take_profit']:,.0f}엔")
        print(f"  💡 이유: {result['reasoning']}")
        
        # 전체 시장 자동선별 + 분석
        print(f"\n🔍 설정 기반 실시간 자동선별 + 전체 분석 시작...")
        start_time = time.time()
        
        market_result = await scan_jp_market()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ 총 소요 시간: {elapsed_time:.1f}초")
        
        print(f"\n💱 현재 환율 정보:")
        print(f"  USD/JPY: {market_result['current_usd_jpy']:.2f}")
        print(f"  엔화 신호: {market_result['yen_signal']}")
        print(f"  선별 방식: {market_result['selection_method']}")
        print(f"  선별 시간: {market_result['last_selection_time']}")
        print(f"  설정 파일: {market_result['config_path']}")
        
        print(f"\n📈 설정 기반 자동선별 + 분석 결과:")
        print(f"  총 분석: {market_result['total_analyzed']}개 종목 (설정 기반 자동선별)")
        print(f"  매수 신호: {market_result['buy_count']}개")
        print(f"  매도 신호: {market_result['sell_count']}개")
        print(f"  버전: {market_result['selection_summary']['config_version']}")
        
        # 선별 요약
        summary = market_result['selection_summary']
        print(f"\n🎯 설정 기반 선별 구성:")
        print(f"  수출주: {summary['export_stocks']}개")
        print(f"  내수주: {summary['domestic_stocks']}개") 
        print(f"  혼합주: {summary['mixed_stocks']}개")
        print(f"  평균 선별점수: {summary['avg_selection_score']:.3f}")
        print(f"  평균 시가총액: {summary['avg_market_cap']:.2f}조엔")
        
        # 상위 매수 추천 (상세 정보)
        if market_result['top_buys']:
            print(f"\n🎯 상위 매수 추천 (설정 기반 자동선별):")
            for i, signal in enumerate(market_result['top_buys'][:3], 1):
                print(f"\n  {i}. {signal.symbol} ({signal.stock_type}) - 신뢰도: {signal.confidence:.2%}")
                print(f"     🏆 선별점수: {signal.selection_score:.3f} | 시총: {signal.market_cap/1e12:.2f}조엔")
                print(f"     📊 기술지표: RSI({signal.rsi:.0f}) MACD({signal.macd_signal}) 추세({signal.ma_trend})")
                print(f"     💰 포지션: {signal.position_size:,}주 ({len(signal.split_buy_plan)}단계 분할매수)")
                print(f"     🛡️ 손절: {signal.stop_loss:,.0f}엔 익절: {signal.take_profit:,.0f}엔")
                print(f"     ⏰ 최대보유: {signal.max_hold_days}일")
                print(f"     🔧 전략: {signal.strategy_source}")
                print(f"     💡 이유: {signal.reasoning}")
        
        # 개별 종목 상세 분석 (설정 기반)
        if market_result['total_analyzed'] > 0:
            test_symbol = market_result['top_buys'][0].symbol if market_result['top_buys'] else None
            if test_symbol:
                print(f"\n📊 설정 기반 개별 종목 상세 분석 - {test_symbol}:")
                detailed_result = await analyze_jp(test_symbol)
                print(f"  🎯 액션: {detailed_result['decision']} (신뢰도: {detailed_result['confidence']:.1f}%)")
                print(f"  🔧 전략소스: {detailed_result['strategy_source']}")
                print(f"  📊 기술지표:")
                print(f"    - RSI: {detailed_result['rsi']:.1f}")
                print(f"    - MACD: {detailed_result['macd_signal']}")
                print(f"    - 볼린저밴드: {detailed_result['bollinger_signal']}")
                print(f"    - 스토캐스틱: {detailed_result['stochastic_signal']}")
                print(f"    - 추세: {detailed_result['ma_trend']}")
                print(f"  💰 설정 기반 분할매매:")
                print(f"    - 총 포지션: {detailed_result['position_size']:,}주")
                print(f"    - 매수 계획: {len(detailed_result['split_buy_plan'])}단계")
                print(f"    - 매도 계획: {len(detailed_result['split_sell_plan'])}단계")
                print(f"  🛡️ 설정 기반 리스크 관리:")
                print(f"    - 손절가: {detailed_result['stop_loss']:,.0f}엔")
                print(f"    - 익절가: {detailed_result['take_profit']:,.0f}엔")
                print(f"    - 최대보유: {detailed_result['max_hold_days']}일")
                print(f"  💱 엔화 정보:")
                print(f"    - 종목타입: {detailed_result['stock_type']}")
                print(f"    - 엔화신호: {detailed_result['yen_signal']}")
                print(f"    - 섹터: {detailed_result['sector']}")
        
        print("\n✅ 설정 통합 테스트 완료!")
        print("\n🎯 일본 주식 V5.0 설정 완전 통합 전략 특징:")
        print("  ✅ 🔧 settings.yaml + .env + requirements.txt 완전 연동")
        print("  ✅ 🆕 설정 기반 실시간 닛케이225 크롤링")
        print("  ✅ 📊 설정 기반 펀더멘털 + 기술적 + 엔화 종합 선별")
        print("  ✅ 💱 USD/JPY 환율 실시간 반영 (설정 임계값)")
        print("  ✅ 💰 설정 기반 분할매매 시스템")
        print("  ✅ 🛡️ 설정 기반 동적 손절/익절")
        print("  ✅ 🔍 설정 기반 상위 N개 종목 완전 자동 선별")
        print("  ✅ 🤖 완전 자동화 (설정 기반 캐시 + 실시간 업데이트)")
        print("  ✅ 📱 웹 대시보드 연동 준비")
        print("\n💡 사용법:")
        print("  - python jp_strategy.py : 설정 기반 전체 자동선별 + 분석")
        print("  - await analyze_jp('7203.T') : 설정 기반 개별 종목 분석")
        print("  - await scan_jp_market() : 설정 기반 시장 전체 스캔")
        print("  - await get_jp_config_status() : 설정 상태 확인")
        print("\n🔧 설정 연동 완료:")
        print("  📁 settings.yaml : 메인 설정 (엔화 임계값, 리스크 관리, 선별 기준)")
        print("  🔐 .env : 환경변수 (API 키, 민감 정보)")
        print("  📦 requirements.txt : 의존성 패키지")
        print("  🚫 .gitignore : 보안 파일 제외")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"테스트 실행 실패: {e}")

if __name__ == "__main__":
    asyncio.run(main())
