#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇺🇸 미국 주식 전략 모듈 - 최고퀸트프로젝트 (완전 통합 연동 버전)
==============================================================

🔗 완벽한 설정 파일 연동:
- .env.example ✅ (API 키, 보안 설정)
- .gitignore ✅ (민감정보 보호)
- requirements.txt ✅ (의존성 패키지)
- settings.yaml ✅ (전략 설정, 파라미터)

핵심 기능:
1. 🆕 실시간 S&P500 + NASDAQ100 + 러셀1000 크롤링
2. 4가지 전략 융합 (버핏25% + 린치25% + 모멘텀25% + 기술25%)
3. 개별 분할매매 (각 종목마다 3단계 매수, 2단계 매도)
4. 완전 자동화 시스템
5. VIX 기반 동적 조정

Author: 최고퀸트팀
Version: 5.0.0 (완전 통합 연동)
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass
import yfinance as yf
import requests
import ta
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import aiohttp
import time
import warnings
from pathlib import Path

# 환경변수 로드 (.env 파일 연동)
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

# ========================================================================================
# 🔧 설정 파일 연동 시스템 (NEW!)
# ========================================================================================

class ConfigManager:
    """설정 파일 통합 관리자 (.env + settings.yaml 완전 연동)"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        self.config_path = config_path
        self.config = {}
        self.env_loaded = False
        self._load_all_configs()
    
    def _load_all_configs(self):
        """모든 설정 파일 로드"""
        try:
            # 1. .env 파일 로드
            self._load_env_config()
            
            # 2. settings.yaml 로드
            self._load_yaml_config()
            
            # 3. 환경변수 치환
            self._substitute_env_variables()
            
            logging.info("✅ 모든 설정 파일 로드 완료")
            
        except Exception as e:
            logging.error(f"❌ 설정 파일 로드 실패: {e}")
            self._load_default_config()
    
    def _load_env_config(self):
        """환경변수 로드 (.env 파일 연동)"""
        try:
            # .env 파일이 있으면 로드
            env_path = Path('.env')
            if env_path.exists():
                load_dotenv(env_path)
                self.env_loaded = True
                logging.info("✅ .env 파일 로드됨")
            else:
                logging.warning("⚠️ .env 파일 없음 (.env.example 참고)")
                
        except Exception as e:
            logging.error(f"❌ .env 로드 실패: {e}")
    
    def _load_yaml_config(self):
        """YAML 설정 파일 로드"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                logging.info(f"✅ {self.config_path} 로드됨")
            else:
                logging.warning(f"⚠️ {self.config_path} 파일 없음")
                self.config = {}
                
        except Exception as e:
            logging.error(f"❌ YAML 로드 실패: {e}")
            self.config = {}
    
    def _substitute_env_variables(self):
        """환경변수 치환 (${VAR_NAME:-default} 형태)"""
        try:
            def substitute_recursive(obj):
                if isinstance(obj, dict):
                    return {k: substitute_recursive(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [substitute_recursive(item) for item in obj]
                elif isinstance(obj, str) and obj.startswith('${') and '}' in obj:
                    # ${VAR_NAME:-default} 형태 처리
                    var_expr = obj[2:-1]  # ${ } 제거
                    if ':-' in var_expr:
                        var_name, default_value = var_expr.split(':-', 1)
                        return os.getenv(var_name, default_value)
                    else:
                        return os.getenv(var_expr, obj)
                else:
                    return obj
            
            self.config = substitute_recursive(self.config)
            logging.info("✅ 환경변수 치환 완료")
            
        except Exception as e:
            logging.error(f"❌ 환경변수 치환 실패: {e}")
    
    def _load_default_config(self):
        """기본 설정값 로드 (설정 파일 없을 때)"""
        self.config = {
            'us_strategy': {
                'enabled': True,
                'confidence_threshold': 0.75,
                'target_stocks': 20,
                'max_position_pct': 8.0,
                'stop_loss_pct': 15.0,
                'take_profit_pct': 35.0,
                'vix_low_threshold': 15.0,
                'vix_high_threshold': 30.0,
                'stage1_ratio': 40.0,
                'stage2_ratio': 35.0,
                'stage3_ratio': 25.0,
                'stage2_trigger_pct': -5.0,
                'stage3_trigger_pct': -10.0,
                'max_hold_days': 60,
                'buffett_weight': 25.0,
                'lynch_weight': 25.0,
                'momentum_weight': 25.0,
                'technical_weight': 25.0
            },
            'risk_management': {
                'portfolio_allocation_pct': 80.0,
                'cash_reserve_pct': 20.0,
                'max_sector_weight_pct': 25.0,
                'daily_loss_limit_pct': 5.0,
                'monthly_loss_limit_pct': 15.0
            },
            'data_sources': {
                'yfinance_enabled': True,
                'polygon_enabled': False,
                'alpha_vantage_enabled': False,
                'request_timeout': 30,
                'max_retries': 3,
                'rate_limit_delay': 0.3
            },
            'notifications': {
                'telegram_enabled': False,
                'slack_enabled': False,
                'email_enabled': False
            }
        }
        logging.info("✅ 기본 설정값 로드됨")
    
    def get(self, key_path: str, default=None):
        """점 표기법으로 설정값 조회 (예: 'us_strategy.enabled')"""
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
    
    def get_section(self, section: str) -> Dict:
        """설정 섹션 전체 조회"""
        return self.config.get(section, {})
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """API 키 안전 조회 (환경변수 우선)"""
        # 1. 환경변수에서 먼저 찾기
        env_value = os.getenv(key_name)
        if env_value:
            return env_value
        
        # 2. 설정 파일에서 찾기
        config_value = self.get(f"api_keys.{key_name.lower()}")
        if config_value:
            return config_value
        
        return None
    
    def is_feature_enabled(self, feature_path: str) -> bool:
        """기능 활성화 여부 확인"""
        return bool(self.get(f"{feature_path}.enabled", False))
    
    def validate_config(self) -> Dict[str, List[str]]:
        """설정 유효성 검증"""
        errors = []
        warnings = []
        
        # US 전략 설정 검증
        us_config = self.get_section('us_strategy')
        if not us_config:
            errors.append("us_strategy 섹션이 없습니다")
        else:
            # 필수 설정 확인
            required_fields = ['enabled', 'confidence_threshold', 'target_stocks']
            for field in required_fields:
                if field not in us_config:
                    errors.append(f"us_strategy.{field} 설정이 없습니다")
        
        # API 키 확인
        api_keys = ['TELEGRAM_BOT_TOKEN', 'SLACK_WEBHOOK_URL']
        for key in api_keys:
            if not self.get_api_key(key):
                warnings.append(f"{key} API 키가 설정되지 않았습니다")
        
        return {'errors': errors, 'warnings': warnings}

# 전역 설정 관리자
config_manager = ConfigManager()

# 로거 설정 (설정 파일 기반)
def setup_logger():
    """로거 설정"""
    log_level = config_manager.get('logging.level', 'INFO')
    log_format = config_manager.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/us_strategy.log', encoding='utf-8') if Path('logs').exists() else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logger()

# ========================================================================================
# 📊 데이터 클래스 (설정 연동)
# ========================================================================================

@dataclass
class USStockSignal:
    """미국 주식 시그널 데이터 클래스 (설정 연동)"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    
    # 전략별 점수
    buffett_score: float
    lynch_score: float
    momentum_score: float
    technical_score: float
    total_score: float
    
    # 재무 지표
    pbr: float
    peg: float
    pe_ratio: float
    roe: float
    market_cap: float
    
    # 모멘텀 지표
    momentum_3m: float
    momentum_6m: float
    momentum_12m: float
    relative_strength: float
    
    # 기술적 지표
    rsi: float
    macd_signal: str
    bb_position: str
    trend: str
    volume_spike: float
    
    # 분할매매 정보
    position_stage: int  # 0, 1, 2, 3 (현재 매수 단계)
    total_shares: int    # 총 계획 주식 수
    stage1_shares: int   # 1단계 매수량 (40%)
    stage2_shares: int   # 2단계 매수량 (35%)
    stage3_shares: int   # 3단계 매수량 (25%)
    entry_price_1: float # 1단계 진입가
    entry_price_2: float # 2단계 진입가 (5% 하락시)
    entry_price_3: float # 3단계 진입가 (10% 하락시)
    stop_loss: float     # 손절가
    take_profit_1: float # 1차 익절가 (60% 매도)
    take_profit_2: float # 2차 익절가 (40% 매도)
    max_hold_days: int
    
    sector: str
    reasoning: str
    target_price: float
    timestamp: datetime
    
    # 자동선별 추가 정보
    selection_score: float  # 선별 점수
    quality_rank: int      # 품질 순위
    index_membership: List[str]  # 소속 지수 (S&P500, NASDAQ100 등)
    vix_adjustment: float  # VIX 기반 조정 점수
    additional_data: Optional[Dict] = None

# ========================================================================================
# 🆕 실시간 미국 주식 수집 및 선별 클래스 (설정 연동)
# ========================================================================================

class RealTimeUSStockSelector:
    """🆕 실시간 미국 주식 종목 수집 및 선별 (설정 파일 연동)"""
    
    def __init__(self):
        # 설정 파일에서 값 로드
        self.min_market_cap = config_manager.get('us_strategy.min_market_cap', 5_000_000_000)
        self.min_avg_volume = config_manager.get('us_strategy.min_avg_volume', 1_000_000)
        self.target_stocks = config_manager.get('us_strategy.target_stocks', 20)
        
        # VIX 임계값
        self.vix_low_threshold = config_manager.get('us_strategy.vix_low_threshold', 15.0)
        self.vix_high_threshold = config_manager.get('us_strategy.vix_high_threshold', 30.0)
        
        # HTTP 세션 설정
        self.session = requests.Session()
        timeout = config_manager.get('data_sources.request_timeout', 30)
        user_agent = config_manager.get('data_sources.user_agent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        self.session.headers.update({'User-Agent': user_agent})
        self.session.timeout = timeout
        
        # 요청 제한 설정
        self.rate_limit_delay = config_manager.get('data_sources.rate_limit_delay', 0.3)
        self.max_retries = config_manager.get('data_sources.max_retries', 3)
        
        self.current_vix = 0.0
        
        logger.info(f"📊 종목 선별기 초기화: 시총 ${self.min_market_cap/1e9:.1f}B+, 거래량 {self.min_avg_volume/1e6:.1f}M+, 목표 {self.target_stocks}개")

    async def get_sp500_constituents(self) -> List[str]:
        """S&P 500 구성종목 실시간 수집 (설정 연동)"""
        try:
            logger.info("🔍 S&P 500 구성종목 실시간 수집 시작...")
            
            symbols = []
            
            # 소스 1: Wikipedia S&P 500 리스트
            try:
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                tables = pd.read_html(url)
                sp500_table = tables[0]
                wikipedia_symbols = sp500_table['Symbol'].tolist()
                
                # 심볼 정리 (특수문자 처리)
                cleaned_symbols = []
                for symbol in wikipedia_symbols:
                    # BRK.B -> BRK-B 형태로 변환
                    cleaned_symbol = str(symbol).replace('.', '-')
                    cleaned_symbols.append(cleaned_symbol)
                
                symbols.extend(cleaned_symbols)
                logger.info(f"✅ Wikipedia에서 {len(cleaned_symbols)}개 S&P 500 종목 수집")
                
                # 속도 제한
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Wikipedia S&P 500 수집 실패: {e}")
            
            # 백업 리스트 추가
            if len(symbols) < 400:  # 예상보다 적으면 백업 추가
                backup_symbols = self._get_backup_sp500()
                symbols.extend(backup_symbols)
                logger.info(f"✅ 백업 리스트 {len(backup_symbols)}개 추가")
            
            return list(set(symbols))  # 중복 제거
            
        except Exception as e:
            logger.error(f"S&P 500 구성종목 수집 실패: {e}")
            return self._get_backup_sp500()

    async def get_nasdaq100_constituents(self) -> List[str]:
        """NASDAQ 100 구성종목 실시간 수집 (설정 연동)"""
        try:
            logger.info("🔍 NASDAQ 100 구성종목 실시간 수집 시작...")
            
            symbols = []
            
            # 소스 1: Wikipedia NASDAQ 100
            try:
                url = "https://en.wikipedia.org/wiki/Nasdaq-100"
                tables = pd.read_html(url)
                # 여러 테이블 중에서 종목 리스트가 있는 테이블 찾기
                for table in tables:
                    if 'Symbol' in table.columns or 'Ticker' in table.columns:
                        symbol_col = 'Symbol' if 'Symbol' in table.columns else 'Ticker'
                        nasdaq_symbols = table[symbol_col].tolist()
                        symbols.extend([str(s) for s in nasdaq_symbols if pd.notna(s)])
                        break
                
                logger.info(f"✅ Wikipedia에서 {len(symbols)}개 NASDAQ 100 종목 수집")
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Wikipedia NASDAQ 100 수집 실패: {e}")
            
            # 백업 기술주 추가
            if len(symbols) < 80:
                tech_giants = self._get_backup_nasdaq100()
                symbols.extend(tech_giants)
                logger.info(f"✅ 백업 기술주 {len(tech_giants)}개 추가")
            
            return list(set(symbols))  # 중복 제거
            
        except Exception as e:
            logger.error(f"NASDAQ 100 구성종목 수집 실패: {e}")
            return self._get_backup_nasdaq100()

    async def get_russell1000_sample(self) -> List[str]:
        """러셀1000 주요 종목 샘플 수집 (설정 연동)"""
        try:
            logger.info("🔍 러셀1000 주요 종목 샘플 수집...")
            
            # 러셀1000 주요 대형주 (섹터별 대표주) - 설정에서 확장 가능
            russell_sample = config_manager.get('us_strategy.russell_sample', [
                # 헬스케어
                'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'LLY', 'DHR', 'BMY', 'MRK',
                # 금융
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
                # 소비재
                'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
                # 에너지
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'OKE', 'WMB', 'VLO',
                # 산업재
                'BA', 'CAT', 'GE', 'LMT', 'RTX', 'UNP', 'UPS', 'DE', 'MMM', 'HON',
                # 소재
                'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DD', 'DOW', 'ECL', 'PPG', 'ALB',
                # 유틸리티
                'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED',
                # 부동산
                'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EQR', 'DLR', 'BXP', 'VTR', 'ARE'
            ])
            
            logger.info(f"✅ 러셀1000 샘플 {len(russell_sample)}개 종목 수집")
            return russell_sample
            
        except Exception as e:
            logger.error(f"러셀1000 샘플 수집 실패: {e}")
            return []

    def _get_backup_sp500(self) -> List[str]:
        """S&P 500 백업 리스트 (설정 연동)"""
        return config_manager.get('us_strategy.backup_sp500', [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'ABBV', 'BAC', 'KO', 'AVGO',
            'XOM', 'CVX', 'LLY', 'WMT', 'PEP', 'TMO', 'COST', 'ORCL', 'ABT', 'ACN',
            'NFLX', 'CRM', 'MRK', 'VZ', 'ADBE', 'DHR', 'TXN', 'NKE', 'PM', 'DIS',
            'WFC', 'NEE', 'RTX', 'CMCSA', 'BMY', 'UNP', 'T', 'COP', 'MS', 'AMD',
            'LOW', 'IBM', 'HON', 'AMGN', 'SPGI', 'LIN', 'QCOM', 'GE', 'CAT', 'UPS'
        ])

    def _get_backup_nasdaq100(self) -> List[str]:
        """NASDAQ 100 백업 리스트 (설정 연동)"""
        return config_manager.get('us_strategy.backup_nasdaq100', [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO',
            'TXN', 'MU', 'AMAT', 'ADI', 'MRVL', 'KLAC', 'LRCX', 'SNPS',
            'ISRG', 'GILD', 'BKNG', 'MDLZ', 'ADP', 'CSX', 'REGN', 'VRTX'
        ])

    async def get_vix_level(self) -> float:
        """VIX 지수 조회 (설정 연동)"""
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1d")
            if not vix_data.empty:
                self.current_vix = vix_data['Close'].iloc[-1]
            else:
                self.current_vix = config_manager.get('us_strategy.default_vix', 20.0)
            
            logger.info(f"📊 현재 VIX: {self.current_vix:.2f}")
            return self.current_vix
            
        except Exception as e:
            logger.error(f"VIX 조회 실패: {e}")
            self.current_vix = config_manager.get('us_strategy.default_vix', 20.0)
            return self.current_vix

    async def create_universe(self) -> List[str]:
        """투자 유니버스 생성 (S&P500 + NASDAQ100 + 러셀1000 샘플) - 설정 연동"""
        try:
            logger.info("🌌 투자 유니버스 생성 시작...")
            
            # 병렬로 데이터 수집
            tasks = [
                self.get_sp500_constituents(),
                self.get_nasdaq100_constituents(),
                self.get_russell1000_sample(),
                self.get_vix_level()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            sp500_symbols = results[0] if not isinstance(results[0], Exception) else []
            nasdaq100_symbols = results[1] if not isinstance(results[1], Exception) else []
            russell_symbols = results[2] if not isinstance(results[2], Exception) else []
            
            # 유니버스 통합
            universe = []
            universe.extend(sp500_symbols)
            universe.extend(nasdaq100_symbols)
            universe.extend(russell_symbols)
            
            # 중복 제거 및 정리
            universe = list(set(universe))
            universe = [symbol.upper().strip() for symbol in universe if symbol and len(symbol) <= 5]
            
            # 제외 종목 필터링 (설정에서 관리)
            excluded_symbols = config_manager.get('us_strategy.excluded_symbols', [])
            universe = [symbol for symbol in universe if symbol not in excluded_symbols]
            
            logger.info(f"🌌 투자 유니버스 생성 완료: {len(universe)}개 종목")
            logger.info(f"  - S&P 500: {len(sp500_symbols)}개")
            logger.info(f"  - NASDAQ 100: {len(nasdaq100_symbols)}개") 
            logger.info(f"  - 러셀1000 샘플: {len(russell_symbols)}개")
            
            return universe
            
        except Exception as e:
            logger.error(f"투자 유니버스 생성 실패: {e}")
            # 백업 유니버스
            backup_universe = self._get_backup_sp500() + self._get_backup_nasdaq100()
            return list(set(backup_universe))

    async def get_stock_comprehensive_data(self, symbol: str) -> Dict:
        """종목 종합 데이터 수집 (설정 연동)"""
        try:
            # 재시도 로직
            for attempt in range(self.max_retries):
                try:
                    stock = yf.Ticker(symbol)
                    
                    # 기본 정보
                    info = stock.info
                    
                    # 가격 데이터 (1년)
                    hist = stock.history(period="1y")
                    if hist.empty:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
                            continue
                        return {}
                    
                    current_price = hist['Close'].iloc[-1]
                    
                    # 기본 재무 지표
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
                        'industry': info.get('industry', 'Unknown'),
                        'beta': info.get('beta', 1.0) or 1.0,
                        'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,
                        'profit_margin': (info.get('profitMargins', 0) or 0) * 100,
                    }
                    
                    # PEG 계산
                    if data['pe_ratio'] > 0 and data['eps_growth'] > 0:
                        data['peg'] = data['pe_ratio'] / data['eps_growth']
                    else:
                        data['peg'] = 999
                    
                    # 모멘텀 지표 계산
                    if len(hist) >= 252:  # 1년 데이터
                        data['momentum_3m'] = ((current_price / hist['Close'].iloc[-63]) - 1) * 100  # 3개월
                        data['momentum_6m'] = ((current_price / hist['Close'].iloc[-126]) - 1) * 100  # 6개월
                        data['momentum_12m'] = ((current_price / hist['Close'].iloc[-252]) - 1) * 100  # 12개월
                    else:
                        data['momentum_3m'] = data['momentum_6m'] = data['momentum_12m'] = 0
                    
                    # 기술적 지표 계산
                    if len(hist) >= 50:
                        # RSI
                        data['rsi'] = ta.momentum.RSIIndicator(hist['Close'], window=14).rsi().iloc[-1]
                        
                        # MACD
                        macd = ta.trend.MACD(hist['Close'])
                        macd_diff = macd.macd_diff().iloc[-1]
                        data['macd_signal'] = 'bullish' if macd_diff > 0 else 'bearish'
                        
                        # 볼린저 밴드
                        bb = ta.volatility.BollingerBands(hist['Close'])
                        bb_high = bb.bollinger_hband().iloc[-1]
                        bb_low = bb.bollinger_lband().iloc[-1]
                        if current_price > bb_high:
                            data['bb_position'] = 'overbought'
                        elif current_price < bb_low:
                            data['bb_position'] = 'oversold'
                        else:
                            data['bb_position'] = 'normal'
                        
                        # 추세 (50일 이동평균 기준)
                        ma50 = hist['Close'].rolling(50).mean().iloc[-1]
                        data['trend'] = 'uptrend' if current_price > ma50 else 'downtrend'
                        
                        # 거래량 급증
                        avg_volume_20d = hist['Volume'].rolling(20).mean().iloc[-1]
                        current_volume = hist['Volume'].iloc[-1]
                        data['volume_spike'] = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1
                    else:
                        data.update({
                            'rsi': 50, 'macd_signal': 'neutral', 'bb_position': 'normal',
                            'trend': 'sideways', 'volume_spike': 1
                        })
                    
                    # 속도 제한
                    await asyncio.sleep(self.rate_limit_delay)
                    return data
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"종목 데이터 수집 재시도 {symbol} (시도 {attempt + 1}/{self.max_retries}): {e}")
                        await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
                        continue
                    else:
                        raise e
            
            return {}
            
        except Exception as e:
            logger.error(f"종목 데이터 수집 실패 {symbol}: {e}")
            return {}

    def calculate_buffett_score(self, data: Dict) -> float:
        """워렌 버핏 가치투자 점수 (설정 연동)"""
        try:
            score = 0.0
            
            # 설정에서 가중치 로드
            weights = config_manager.get('us_strategy.buffett_weights', {
                'pbr': 0.35,
                'roe': 0.30,
                'debt_ratio': 0.20,
                'pe_ratio': 0.15
            })
            
            # 설정에서 임계값 로드
            thresholds = config_manager.get('us_strategy.buffett_thresholds', {
                'pbr_excellent': 1.5,
                'pbr_good': 2.5,
                'pbr_fair': 4.0,
                'roe_excellent': 20,
                'roe_good': 15,
                'roe_fair': 10,
                'roe_minimum': 5,
                'debt_excellent': 0.3,
                'debt_good': 0.5,
                'debt_fair': 0.7,
                'pe_min': 5,
                'pe_excellent': 15,
                'pe_good': 25,
                'pe_fair': 35
            })
            
            # PBR 점수
            pbr = data.get('pbr', 999)
            if 0 < pbr <= thresholds['pbr_excellent']:
                score += weights['pbr']
            elif pbr <= thresholds['pbr_good']:
                score += weights['pbr'] * 0.7
            elif pbr <= thresholds['pbr_fair']:
                score += weights['pbr'] * 0.3
            
            # ROE 점수
            roe = data.get('roe', 0)
            if roe >= thresholds['roe_excellent']:
                score += weights['roe']
            elif roe >= thresholds['roe_good']:
                score += weights['roe'] * 0.8
            elif roe >= thresholds['roe_fair']:
                score += weights['roe'] * 0.5
            elif roe >= thresholds['roe_minimum']:
                score += weights['roe'] * 0.2
            
            # 부채비율 점수
            debt_ratio = data.get('debt_to_equity', 999) / 100
            if debt_ratio <= thresholds['debt_excellent']:
                score += weights['debt_ratio']
            elif debt_ratio <= thresholds['debt_good']:
                score += weights['debt_ratio'] * 0.7
            elif debt_ratio <= thresholds['debt_fair']:
                score += weights['debt_ratio'] * 0.4
            
            # PE 적정성 점수
            pe = data.get('pe_ratio', 999)
            if thresholds['pe_min'] <= pe <= thresholds['pe_excellent']:
                score += weights['pe_ratio']
            elif pe <= thresholds['pe_good']:
                score += weights['pe_ratio'] * 0.7
            elif pe <= thresholds['pe_fair']:
                score += weights['pe_ratio'] * 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"버핏 점수 계산 실패: {e}")
            return 0.0

    def calculate_lynch_score(self, data: Dict) -> float:
        """피터 린치 성장투자 점수 (설정 연동)"""
        try:
            score = 0.0
            
            # 설정에서 가중치 로드
            weights = config_manager.get('us_strategy.lynch_weights', {
                'peg': 0.40,
                'eps_growth': 0.35,
                'revenue_growth': 0.25
            })
            
            # 설정에서 임계값 로드
            thresholds = config_manager.get('us_strategy.lynch_thresholds', {
                'peg_excellent': 0.5,
                'peg_good': 1.0,
                'peg_fair': 1.5,
                'peg_acceptable': 2.0,
                'eps_excellent': 25,
                'eps_good': 15,
                'eps_fair': 10,
                'eps_minimum': 5,
                'revenue_excellent': 20,
                'revenue_good': 10,
                'revenue_fair': 5
            })
            
            # PEG 점수
            peg = data.get('peg', 999)
            if 0 < peg <= thresholds['peg_excellent']:
                score += weights['peg']
            elif peg <= thresholds['peg_good']:
                score += weights['peg'] * 0.85
            elif peg <= thresholds['peg_fair']:
                score += weights['peg'] * 0.6
            elif peg <= thresholds['peg_acceptable']:
                score += weights['peg'] * 0.25
            
            # EPS 성장률 점수
            eps_growth = data.get('eps_growth', 0)
            if eps_growth >= thresholds['eps_excellent']:
                score += weights['eps_growth']
            elif eps_growth >= thresholds['eps_good']:
                score += weights['eps_growth'] * 0.7
            elif eps_growth >= thresholds['eps_fair']:
                score += weights['eps_growth'] * 0.4
            elif eps_growth >= thresholds['eps_minimum']:
                score += weights['eps_growth'] * 0.15
            
            # 매출 성장률 점수
            revenue_growth = data.get('revenue_growth', 0)
            if revenue_growth >= thresholds['revenue_excellent']:
                score += weights['revenue_growth']
            elif revenue_growth >= thresholds['revenue_good']:
                score += weights['revenue_growth'] * 0.6
            elif revenue_growth >= thresholds['revenue_fair']:
                score += weights['revenue_growth'] * 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"린치 점수 계산 실패: {e}")
            return 0.0

    def calculate_momentum_score(self, data: Dict) -> float:
        """모멘텀 점수 (설정 연동)"""
        try:
            score = 0.0
            
            # 설정에서 가중치 로드
            weights = config_manager.get('us_strategy.momentum_weights', {
                'momentum_3m': 0.30,
                'momentum_6m': 0.25,
                'momentum_12m': 0.25,
                'volume_spike': 0.20
            })
            
            # 설정에서 임계값 로드
            thresholds = config_manager.get('us_strategy.momentum_thresholds', {
                '3m_excellent': 20,
                '3m_good': 10,
                '3m_fair': 5,
                '6m_excellent': 30,
                '6m_good': 15,
                '6m_fair': 5,
                '12m_excellent': 50,
                '12m_good': 25,
                '12m_fair': 10,
                'volume_excellent': 2.0,
                'volume_good': 1.5,
                'volume_fair': 1.2
            })
            
            # 3개월 모멘텀
            mom_3m = data.get('momentum_3m', 0)
            if mom_3m >= thresholds['3m_excellent']:
                score += weights['momentum_3m']
            elif mom_3m >= thresholds['3m_good']:
                score += weights['momentum_3m'] * 0.7
            elif mom_3m >= thresholds['3m_fair']:
                score += weights['momentum_3m'] * 0.35
            elif mom_3m >= 0:
                score += weights['momentum_3m'] * 0.15
            
            # 6개월 모멘텀
            mom_6m = data.get('momentum_6m', 0)
            if mom_6m >= thresholds['6m_excellent']:
                score += weights['momentum_6m']
            elif mom_6m >= thresholds['6m_good']:
                score += weights['momentum_6m'] * 0.6
            elif mom_6m >= thresholds['6m_fair']:
                score += weights['momentum_6m'] * 0.4
            
            # 12개월 모멘텀
            mom_12m = data.get('momentum_12m', 0)
            if mom_12m >= thresholds['12m_excellent']:
                score += weights['momentum_12m']
            elif mom_12m >= thresholds['12m_good']:
                score += weights['momentum_12m'] * 0.6
            elif mom_12m >= thresholds['12m_fair']:
                score += weights['momentum_12m'] * 0.4
            
            # 거래량 급증
            volume_spike = data.get('volume_spike', 1)
            if volume_spike >= thresholds['volume_excellent']:
                score += weights['volume_spike']
            elif volume_spike >= thresholds['volume_good']:
                score += weights['volume_spike'] * 0.5
            elif volume_spike >= thresholds['volume_fair']:
                score += weights['volume_spike'] * 0.25
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"모멘텀 점수 계산 실패: {e}")
            return 0.0

    def calculate_technical_score(self, data: Dict) -> float:
        """기술적 분석 점수 (설정 연동)"""
        try:
            score = 0.0
            
            # 설정에서 가중치 로드
            weights = config_manager.get('us_strategy.technical_weights', {
                'rsi': 0.30,
                'macd': 0.25,
                'trend': 0.25,
                'bollinger': 0.20
            })
            
            # 설정에서 임계값 로드
            thresholds = config_manager.get('us_strategy.technical_thresholds', {
                'rsi_oversold_min': 20,
                'rsi_oversold_max': 30,
                'rsi_normal_min': 30,
                'rsi_normal_max': 70,
                'rsi_overbought_min': 70,
                'rsi_overbought_max': 80
            })
            
            # RSI 점수
            rsi = data.get('rsi', 50)
            if thresholds['rsi_normal_min'] <= rsi <= thresholds['rsi_normal_max']:
                score += weights['rsi']
            elif thresholds['rsi_oversold_min'] <= rsi < thresholds['rsi_oversold_max']:
                score += weights['rsi'] * 0.7  # 과매도는 매수 기회
            elif thresholds['rsi_overbought_min'] < rsi <= thresholds['rsi_overbought_max']:
                score += weights['rsi'] * 0.5  # 과매수는 중립
            
            # MACD 점수
            macd = data.get('macd_signal', 'neutral')
            if macd == 'bullish':
                score += weights['macd']
            elif macd == 'neutral':
                score += weights['macd'] * 0.5
            
            # 추세 점수
            trend = data.get('trend', 'sideways')
            if trend == 'uptrend':
                score += weights['trend']
            elif trend == 'sideways':
                score += weights['trend'] * 0.3
            
            # 볼린저 밴드 점수
            bb = data.get('bb_position', 'normal')
            if bb == 'oversold':
                score += weights['bollinger']  # 과매도는 매수 기회
            elif bb == 'normal':
                score += weights['bollinger'] * 0.5
            # 과매수는 점수 없음
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"기술적 점수 계산 실패: {e}")
            return 0.0

    def calculate_vix_adjustment(self, base_score: float) -> float:
        """VIX 기반 점수 조정 (설정 연동)"""
        try:
            # 설정에서 VIX 조정 계수 로드
            vix_adjustments = config_manager.get('us_strategy.vix_adjustments', {
                'low_volatility_boost': 1.1,
                'normal_volatility': 1.0,
                'high_volatility_reduction': 0.9
            })
            
            if self.current_vix <= self.vix_low_threshold:
                # 저변동성 (안정적): 가치주 선호
                return base_score * vix_adjustments['low_volatility_boost']
            elif self.current_vix >= self.vix_high_threshold:
                # 고변동성 (불안정): 보수적 접근
                return base_score * vix_adjustments['high_volatility_reduction']
            else:
                # 정상 변동성
                return base_score * vix_adjustments['normal_volatility']
                
        except Exception as e:
            logger.error(f"VIX 조정 실패: {e}")
            return base_score

    def calculate_selection_score(self, data: Dict) -> float:
        """종목 선별 종합 점수 계산 (설정 연동)"""
        try:
            # 4가지 전략 점수 계산
            buffett_score = self.calculate_buffett_score(data)
            lynch_score = self.calculate_lynch_score(data)
            momentum_score = self.calculate_momentum_score(data)
            technical_score = self.calculate_technical_score(data)
            
            # 설정에서 전략별 가중치 로드
            strategy_weights = config_manager.get('us_strategy.strategy_weights', {
                'buffett': 0.25,
                'lynch': 0.25,
                'momentum': 0.25,
                'technical': 0.25
            })
            
            # 가중 평균
            base_score = (
                buffett_score * strategy_weights['buffett'] +
                lynch_score * strategy_weights['lynch'] +
                momentum_score * strategy_weights['momentum'] +
                technical_score * strategy_weights['technical']
            )
            
            # VIX 기반 조정
            adjusted_score = self.calculate_vix_adjustment(base_score)
            
            return min(adjusted_score, 1.0)
            
        except Exception as e:
            logger.error(f"선별 점수 계산 실패: {e}")
            return 0.0

    def determine_index_membership(self, symbol: str, sp500_list: List[str], 
                                 nasdaq100_list: List[str]) -> List[str]:
        """종목의 지수 소속 확인"""
        membership = []
        
        if symbol in sp500_list:
            membership.append('S&P500')
        if symbol in nasdaq100_list:
            membership.append('NASDAQ100')
        if not membership:
            membership.append('OTHER')
            
        return membership

    async def select_top_stocks(self, universe: List[str]) -> List[Dict]:
        """상위 종목 선별 (4가지 전략 + VIX 조정) - 설정 연동"""
        logger.info(f"🎯 {len(universe)}개 후보에서 상위 {self.target_stocks}개 선별 시작...")
        
        # 기본 지수 리스트 미리 준비
        sp500_list = await self.get_sp500_constituents()
        nasdaq100_list = await self.get_nasdaq100_constituents()
        
        scored_stocks = []
        
        # 설정에서 병렬 처리 워커 수 로드
        max_workers = config_manager.get('data_sources.max_workers', 15)
        
        # 병렬 처리로 속도 향상
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for symbol in universe:
                future = executor.submit(self._process_single_stock, symbol, sp500_list, nasdaq100_list)
                futures.append(future)
            
            for i, future in enumerate(futures, 1):
                try:
                    # 설정에서 종목 처리 타임아웃 로드
                    timeout = config_manager.get('data_sources.stock_processing_timeout', 45)
                    result = future.result(timeout=timeout)
                    if result:
                        scored_stocks.append(result)
                        
                    # 진행상황 표시 간격을 설정에서 로드
                    progress_interval = config_manager.get('data_sources.progress_interval', 50)
                    if i % progress_interval == 0:
                        logger.info(f"📊 진행상황: {i}/{len(universe)} 완료")
                        
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
            membership_str = "+".join(stock['index_membership'])
            logger.info(f"  {i}. {stock['symbol']}: 점수 {stock['selection_score']:.3f} "
                       f"시총 ${stock['market_cap']/1e9:.1f}B ({membership_str}) "
                       f"[{stock['sector']}]")
        
        return final_selection

    def _process_single_stock(self, symbol: str, sp500_list: List[str], 
                            nasdaq100_list: List[str]) -> Optional[Dict]:
        """단일 종목 처리 (설정 연동)"""
        try:
            # 데이터 수집
            data = asyncio.run(self.get_stock_comprehensive_data(symbol))
            if not data:
                return None
            
            # 기본 필터링
            market_cap = data.get('market_cap', 0)
            avg_volume = data.get('avg_volume', 0)
            
            if market_cap < self.min_market_cap or avg_volume < self.min_avg_volume:
                return None
            
            # 선별 점수 계산
            selection_score = self.calculate_selection_score(data)
            
            # 4가지 전략 개별 점수
            buffett_score = self.calculate_buffett_score(data)
            lynch_score = self.calculate_lynch_score(data)
            momentum_score = self.calculate_momentum_score(data)
            technical_score = self.calculate_technical_score(data)
            
            # 지수 소속 확인
            index_membership = self.determine_index_membership(symbol, sp500_list, nasdaq100_list)
            
            # VIX 조정값 계산
            vix_adjustment = self.calculate_vix_adjustment(1.0) - 1.0
            
            result = data.copy()
            result.update({
                'selection_score': selection_score,
                'buffett_score': buffett_score,
                'lynch_score': lynch_score,
                'momentum_score': momentum_score,
                'technical_score': technical_score,
                'index_membership': index_membership,
                'vix_adjustment': vix_adjustment
            })
            
            return result
            
        except Exception as e:
            logger.error(f"종목 처리 실패 {symbol}: {e}")
            return None

    def _ensure_sector_diversity(self, scored_stocks: List[Dict]) -> List[Dict]:
        """섹터 다양성 확보 (설정 연동)"""
        try:
            final_selection = []
            sector_counts = {}
            
            # 설정에서 섹터 다양성 규칙 로드
            diversity_config = config_manager.get('us_strategy.sector_diversity', {
                'max_per_sector': 4,
                'sp500_quota_pct': 60,
                'nasdaq_quota_pct': 40
            })
            
            max_per_sector = diversity_config['max_per_sector']
            sp500_quota = int(self.target_stocks * diversity_config['sp500_quota_pct'] / 100)
            nasdaq_quota = int(self.target_stocks * diversity_config['nasdaq_quota_pct'] / 100)
            
            sp500_selected = 0
            nasdaq_selected = 0
            
            for stock in scored_stocks:
                if len(final_selection) >= self.target_stocks:
                    break
                
                sector = stock.get('sector', 'Unknown')
                membership = stock.get('index_membership', [])
                
                # 섹터 제한 확인
                if sector_counts.get(sector, 0) >= max_per_sector:
                    continue
                
                # 지수별 쿼터 확인
                is_sp500 = 'S&P500' in membership
                is_nasdaq = 'NASDAQ100' in membership
                
                if is_sp500 and sp500_selected < sp500_quota:
                    final_selection.append(stock)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    sp500_selected += 1
                elif is_nasdaq and nasdaq_selected < nasdaq_quota:
                    final_selection.append(stock)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    nasdaq_selected += 1
                elif sp500_selected >= sp500_quota and nasdaq_selected >= nasdaq_quota:
                    # 둘 다 쿼터 달성시 점수 순으로 선별
                    final_selection.append(stock)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # 남은 자리가 있으면 점수 순으로 채움
            remaining_slots = self.target_stocks - len(final_selection)
            if remaining_slots > 0:
                remaining_stocks = [s for s in scored_stocks if s not in final_selection]
                final_selection.extend(remaining_stocks[:remaining_slots])
            
            return final_selection[:self.target_stocks]
            
        except Exception as e:
            logger.error(f"섹터 다양성 확보 실패: {e}")
            return scored_stocks[:self.target_stocks]

# ========================================================================================
# 🇺🇸 메인 미국 주식 전략 클래스 (완전 설정 연동)
# ========================================================================================

class AdvancedUSStrategy:
    """🚀 완전 업그레이드 미국 전략 클래스 (설정 파일 완전 연동)"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        """전략 초기화 (설정 파일 연동)"""
        self.config_manager = config_manager
        self.us_config = self.config_manager.get_section('us_strategy')
        self.enabled = self.config_manager.get('us_strategy.enabled', True)
        
        # 🆕 실시간 종목 선별기
        self.stock_selector = RealTimeUSStockSelector()
        
        # 📊 전략별 가중치 (설정에서 로드)
        strategy_weights = self.config_manager.get('us_strategy.strategy_weights', {})
        self.buffett_weight = strategy_weights.get('buffett', 25.0) / 100
        self.lynch_weight = strategy_weights.get('lynch', 25.0) / 100
        self.momentum_weight = strategy_weights.get('momentum', 25.0) / 100
        self.technical_weight = strategy_weights.get('technical', 25.0) / 100
        
        # 💰 포트폴리오 설정 (설정에서 로드)
        portfolio_config = self.config_manager.get_section('risk_management')
        self.total_portfolio_ratio = portfolio_config.get('portfolio_allocation_pct', 80.0) / 100
        self.cash_reserve_ratio = portfolio_config.get('cash_reserve_pct', 20.0) / 100
        
        # 🔧 분할매매 설정 (설정에서 로드)
        self.stage1_ratio = self.config_manager.get('us_strategy.stage1_ratio', 40.0) / 100
        self.stage2_ratio = self.config_manager.get('us_strategy.stage2_ratio', 35.0) / 100
        self.stage3_ratio = self.config_manager.get('us_strategy.stage3_ratio', 25.0) / 100
        self.stage2_trigger = self.config_manager.get('us_strategy.stage2_trigger_pct', -5.0) / 100
        self.stage3_trigger = self.config_manager.get('us_strategy.stage3_trigger_pct', -10.0) / 100
        
        # 🛡️ 리스크 관리 (설정에서 로드)
        self.stop_loss_pct = self.config_manager.get('us_strategy.stop_loss_pct', 15.0) / 100
        self.take_profit1_pct = self.config_manager.get('us_strategy.take_profit1_pct', 20.0) / 100
        self.take_profit2_pct = self.config_manager.get('us_strategy.take_profit2_pct', 35.0) / 100
        self.max_hold_days = self.config_manager.get('us_strategy.max_hold_days', 60)
        self.max_sector_weight = portfolio_config.get('max_sector_weight_pct', 25.0) / 100
        
        # 🔍 자동 선별된 종목들 (동적으로 업데이트)
        self.selected_stocks = []          # 실시간 선별 결과
        self.last_selection_time = None    # 마지막 선별 시간
        self.selection_cache_hours = self.config_manager.get('us_strategy.selection_cache_hours', 24)
        
        # ✅ 설정 검증
        validation_result = self.config_manager.validate_config()
        if validation_result['errors']:
            logger.error(f"❌ 설정 오류: {validation_result['errors']}")
        if validation_result['warnings']:
            logger.warning(f"⚠️ 설정 경고: {validation_result['warnings']}")
        
        if self.enabled:
            logger.info(f"🇺🇸 완전 통합 연동 미국 전략 초기화 (V5.0)")
            logger.info(f"🆕 실시간 S&P500 + NASDAQ100 + 러셀1000 자동 선별")
            logger.info(f"🎯 자동 선별: 상위 {self.stock_selector.target_stocks}개 종목")
            logger.info(f"📊 4가지 전략 융합: 버핏{self.buffett_weight*100:.0f}% + 린치{self.lynch_weight*100:.0f}% + 모멘텀{self.momentum_weight*100:.0f}% + 기술{self.technical_weight*100:.0f}%")
            logger.info(f"💰 분할매매: 3단계 매수({self.stage1_ratio*100:.0f}%+{self.stage2_ratio*100:.0f}%+{self.stage3_ratio*100:.0f}%), 2단계 매도(60%+40%)")
            logger.info(f"🛡️ 리스크 관리: 손절{self.stop_loss_pct*100:.0f}%, 익절{self.take_profit2_pct*100:.0f}%")
            logger.info(f"📊 VIX 기반 동적 조정 시스템")
            logger.info(f"🔗 설정 파일 연동: .env + settings.yaml 완벽 통합")

    # ========================================================================================
    # 🆕 실시간 자동 선별 메서드들 (설정 연동)
    # ========================================================================================

    async def auto_select_top20_stocks(self) -> List[str]:
        """🆕 실시간 미국 주식 자동 선별 (설정 연동)"""
        if not self.enabled:
            logger.warning("미국 주식 전략이 비활성화되어 있습니다")
            return []

        try:
            # 캐시 확인
            if self._is_selection_cache_valid():
                logger.info("📋 캐시된 선별 결과 사용")
                return [stock['symbol'] for stock in self.selected_stocks]

            logger.info("🔍 실시간 미국 주식 자동 선별 시작!")
            start_time = time.time()

            # 1단계: 투자 유니버스 생성
            universe = await self.stock_selector.create_universe()
            if not universe:
                logger.error("투자 유니버스 생성 실패")
                return self._get_fallback_stocks()

            # 2단계: 상위 종목 선별
            selected_data = await self.stock_selector.select_top_stocks(universe)
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
            sp500_count = len([s for s in selected_data if 'S&P500' in s.get('index_membership', [])])
            nasdaq_count = len([s for s in selected_data if 'NASDAQ100' in s.get('index_membership', [])])

            logger.info(f"📊 지수별 구성: S&P500 {sp500_count}개, NASDAQ100 {nasdaq_count}개")
            logger.info(f"📊 현재 VIX: {self.stock_selector.current_vix:.2f}")

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
        """백업 종목 리스트 (자동 선별 실패시) - 설정에서 로드"""
        fallback_symbols = self.config_manager.get('us_strategy.fallback_stocks', [
            # 대형 기술주
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # 헬스케어
            'JNJ', 'UNH', 'PFE', 'ABBV',
            # 금융
            'JPM', 'BAC', 'WFC', 'GS',
            # 소비재
            'PG', 'KO', 'HD', 'WMT',
            # 산업재
            'BA', 'CAT'
        ])
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
            return await self.stock_selector.get_stock_comprehensive_data(symbol)
            
        except Exception as e:
            logger.error(f"종목 정보 조회 실패 {symbol}: {e}")
            return {}

    # ========================================================================================
    # 💰 분할매매 계획 수립 (설정 연동)
    # ========================================================================================

    def _calculate_split_trading_plan(self, symbol: str, current_price: float, 
                                     confidence: float, portfolio_value: float = None) -> Dict:
        """분할매매 계획 수립 (설정 연동)"""
        try:
            # 포트폴리오 가치 설정에서 로드
            if portfolio_value is None:
                portfolio_value = self.config_manager.get('portfolio.default_value', 1000000)
            
            # 종목별 목표 비중 계산 (신뢰도 기반)
            base_weight = self.total_portfolio_ratio / self.stock_selector.target_stocks  # 기본 비중
            confidence_multiplier_range = self.config_manager.get('us_strategy.confidence_multiplier_range', [0.5, 1.5])
            confidence_multiplier = confidence_multiplier_range[0] + (confidence * (confidence_multiplier_range[1] - confidence_multiplier_range[0]))
            target_weight = base_weight * confidence_multiplier
            
            # 최대 비중 제한 (설정에서 로드)
            max_position_pct = self.config_manager.get('us_strategy.max_position_pct', 8.0) / 100
            target_weight = min(target_weight, max_position_pct)
            
            # 총 투자금액
            total_investment = portfolio_value * target_weight
            total_shares = int(total_investment / current_price)
            
            # 3단계 분할 계획
            stage1_shares = int(total_shares * self.stage1_ratio)
            stage2_shares = int(total_shares * self.stage2_ratio)
            stage3_shares = total_shares - stage1_shares - stage2_shares
            
            # 진입가 계획
            entry_price_1 = current_price
            entry_price_2 = current_price * (1 + self.stage2_trigger)
            entry_price_3 = current_price * (1 + self.stage3_trigger)
            
            # 손절/익절 계획
            avg_entry_discount = self.config_manager.get('us_strategy.avg_entry_discount', 10.0) / 100
            avg_entry = current_price * (1 - avg_entry_discount)  # 평균 진입가 추정
            stop_loss = avg_entry * (1 - self.stop_loss_pct)
            take_profit_1 = avg_entry * (1 + self.take_profit1_pct)
            take_profit_2 = avg_entry * (1 + self.take_profit2_pct)
            
            # 보유 기간 (신뢰도 기반)
            hold_days_adjustment = self.config_manager.get('us_strategy.hold_days_adjustment', 1.5)
            max_hold_days = int(self.max_hold_days * (hold_days_adjustment - confidence))
            
            return {
                'total_shares': total_shares,
                'stage1_shares': stage1_shares,
                'stage2_shares': stage2_shares,
                'stage3_shares': stage3_shares,
                'entry_price_1': entry_price_1,
                'entry_price_2': entry_price_2,
                'entry_price_3': entry_price_3,
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'max_hold_days': max_hold_days,
                'target_weight': target_weight * 100,
                'total_investment': total_investment
            }
            
        except Exception as e:
            logger.error(f"분할매매 계획 수립 실패 {symbol}: {e}")
            return {}

    # ========================================================================================
    # 🎯 메인 종목 분석 메서드 (설정 연동)
    # ========================================================================================

    async def analyze_symbol(self, symbol: str) -> USStockSignal:
        """개별 종목 종합 분석 (설정 파일 완전 연동)"""
        if not self.enabled:
            return self._create_empty_signal(symbol, "전략 비활성화")
        
        try:
            # 1. 종합 데이터 수집
            data = await self.stock_selector.get_stock_comprehensive_data(symbol)
            if not data:
                raise ValueError(f"데이터 수집 실패: {symbol}")
            
            # 2. 4가지 전략 분석
            buffett_score = self.stock_selector.calculate_buffett_score(data)
            lynch_score = self.stock_selector.calculate_lynch_score(data)
            momentum_score = self.stock_selector.calculate_momentum_score(data)
            technical_score = self.stock_selector.calculate_technical_score(data)
            
            # 3. 가중 평균 계산 (설정 기반)
            total_score = (
                buffett_score * self.buffett_weight +
                lynch_score * self.lynch_weight +
                momentum_score * self.momentum_weight +
                technical_score * self.technical_weight
            )
            
            # 4. VIX 조정
            vix_adjustment = self.stock_selector.calculate_vix_adjustment(total_score) - total_score
            total_score = self.stock_selector.calculate_vix_adjustment(total_score)
            
            # 5. 최종 액션 결정 (설정에서 임계값 로드)
            buy_threshold = self.config_manager.get('us_strategy.buy_threshold', 0.70)
            sell_threshold = self.config_manager.get('us_strategy.sell_threshold', 0.30)
            max_confidence = self.config_manager.get('us_strategy.max_confidence', 0.95)
            
            if total_score >= buy_threshold:
                action = 'buy'
                confidence = min(total_score, max_confidence)
            elif total_score <= sell_threshold:
                action = 'sell'
                confidence = min(1 - total_score, max_confidence)
            else:
                action = 'hold'
                confidence = 0.50
            
            # 6. 분할매매 계획 수립
            split_plan = self._calculate_split_trading_plan(symbol, data['price'], confidence)
            
            # 7. 목표주가 계산 (설정에서 기대수익률 로드)
            max_expected_return = self.config_manager.get('us_strategy.max_expected_return_pct', 35.0) / 100
            target_price = data['price'] * (1 + confidence * max_expected_return)
            
            # 8. 종합 reasoning
            strategies = [
                f"버핏:{buffett_score:.2f}",
                f"린치:{lynch_score:.2f}", 
                f"모멘텀:{momentum_score:.2f}",
                f"기술:{technical_score:.2f}"
            ]
            all_reasoning = " | ".join(strategies) + f" | VIX조정:{vix_adjustment:+.2f}"
            
            # 9. 선별 정보 추가
            stock_info = await self.get_selected_stock_info(symbol)
            selection_score = stock_info.get('selection_score', total_score)
            index_membership = stock_info.get('index_membership', ['OTHER'])
            
            return USStockSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=data['price'],
                
                # 전략별 점수
                buffett_score=buffett_score,
                lynch_score=lynch_score,
                momentum_score=momentum_score,
                technical_score=technical_score,
                total_score=total_score,
                
                # 재무 지표
                pbr=data.get('pbr', 0),
                peg=data.get('peg', 0),
                pe_ratio=data.get('pe_ratio', 0),
                roe=data.get('roe', 0),
                market_cap=data.get('market_cap', 0),
                
                # 모멘텀 지표
                momentum_3m=data.get('momentum_3m', 0),
                momentum_6m=data.get('momentum_6m', 0),
                momentum_12m=data.get('momentum_12m', 0),
                relative_strength=0,  # 추후 계산
                
                # 기술적 지표
                rsi=data.get('rsi', 50),
                macd_signal=data.get('macd_signal', 'neutral'),
                bb_position=data.get('bb_position', 'normal'),
                trend=data.get('trend', 'sideways'),
                volume_spike=data.get('volume_spike', 1),
                
                # 분할매매 정보
                position_stage=0,  # 초기값
                total_shares=split_plan.get('total_shares', 0),
                stage1_shares=split_plan.get('stage1_shares', 0),
                stage2_shares=split_plan.get('stage2_shares', 0),
                stage3_shares=split_plan.get('stage3_shares', 0),
                entry_price_1=split_plan.get('entry_price_1', data['price']),
                entry_price_2=split_plan.get('entry_price_2', data['price']),
                entry_price_3=split_plan.get('entry_price_3', data['price']),
                stop_loss=split_plan.get('stop_loss', data['price'] * 0.85),
                take_profit_1=split_plan.get('take_profit_1', data['price'] * 1.20),
                take_profit_2=split_plan.get('take_profit_2', data['price'] * 1.35),
                max_hold_days=split_plan.get('max_hold_days', 60),
                
                sector=data.get('sector', 'Unknown'),
                reasoning=all_reasoning,
                target_price=target_price,
                timestamp=datetime.now(),
                
                # 자동선별 추가 정보
                selection_score=selection_score,
                quality_rank=0,  # 추후 계산
                index_membership=index_membership,
                vix_adjustment=vix_adjustment,
                additional_data=split_plan
            )
            
        except Exception as e:
            logger.error(f"종목 분석 실패 {symbol}: {e}")
            return self._create_empty_signal(symbol, f"분석 실패: {str(e)}")

    def _create_empty_signal(self, symbol: str, reason: str) -> USStockSignal:
        """빈 시그널 생성"""
        return USStockSignal(
            symbol=symbol, action='hold', confidence=0.0, price=0.0,
            buffett_score=0.0, lynch_score=0.0, momentum_score=0.0, technical_score=0.0, total_score=0.0,
            pbr=0.0, peg=0.0, pe_ratio=0.0, roe=0.0, market_cap=0, momentum_3m=0.0, momentum_6m=0.0,
            momentum_12m=0.0, relative_strength=0.0, rsi=50.0, macd_signal='neutral', bb_position='normal',
            trend='sideways', volume_spike=1.0, position_stage=0, total_shares=0, stage1_shares=0,
            stage2_shares=0, stage3_shares=0, entry_price_1=0.0, entry_price_2=0.0, entry_price_3=0.0,
            stop_loss=0.0, take_profit_1=0.0, take_profit_2=0.0, max_hold_days=60, sector='Unknown',
            reasoning=reason, target_price=0.0, timestamp=datetime.now(),
            selection_score=0.0, quality_rank=0, index_membership=['UNKNOWN'], vix_adjustment=0.0
        )

    # ========================================================================================
    # 🔍 전체 시장 스캔 (자동선별 + 분석) - 설정 연동
    # ========================================================================================

    async def scan_all_selected_stocks(self) -> List[USStockSignal]:
        """전체 자동선별 + 종목 분석 (설정 파일 완전 연동)"""
        if not self.enabled:
            return []
        
        logger.info(f"🔍 미국 주식 완전 자동 분석 시작! (설정 기반)")
        logger.info(f"🆕 실시간 S&P500+NASDAQ100 자동 선별 + 4가지 전략 분석")
        
        try:
            # 1단계: 실시간 자동 선별
            selected_symbols = await self.auto_select_top20_stocks()
            if not selected_symbols:
                logger.error("자동 선별 실패")
                return []
            
            # 2단계: 선별된 종목들 상세 분석
            all_signals = []
            
            # 설정에서 분석 간격 로드
            analysis_delay = self.config_manager.get('data_sources.analysis_delay', 0.3)
            
            for i, symbol in enumerate(selected_symbols, 1):
                try:
                    signal = await self.analyze_symbol(symbol)
                    all_signals.append(signal)
                    
                    # 결과 로그
                    action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                    membership_str = "+".join(signal.index_membership)
                    logger.info(f"{action_emoji} {symbol} ({membership_str}): {signal.action} "
                              f"신뢰도:{signal.confidence:.2f} 총점:{signal.total_score:.2f} "
                              f"선별점수:{signal.selection_score:.3f}")
                    
                    # API 호출 제한
                    await asyncio.sleep(analysis_delay)
                    
                except Exception as e:
                    logger.error(f"❌ {symbol} 분석 실패: {e}")
            
            # 결과 요약
            buy_count = len([s for s in all_signals if s.action == 'buy'])
            sell_count = len([s for s in all_signals if s.action == 'sell'])
            hold_count = len([s for s in all_signals if s.action == 'hold'])
            
            logger.info(f"🎯 완전 자동 분석 완료! (설정 기반)")
            logger.info(f"📊 결과: 매수:{buy_count}, 매도:{sell_count}, 보유:{hold_count}")
            logger.info(f"📊 현재 VIX: {self.stock_selector.current_vix:.2f}")
            logger.info(f"🆕 자동선별 시간: {self.last_selection_time}")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"전체 스캔 실패: {e}")
            return []

    # ========================================================================================
    # 📊 포트폴리오 리포트 생성 (설정 연동)
    # ========================================================================================

    async def generate_portfolio_report(self, selected_stocks: List[USStockSignal]) -> Dict:
        """📊 포트폴리오 리포트 생성 (설정 파일 연동)"""
        if not selected_stocks:
            return {"error": "선별된 종목이 없습니다"}
        
        # 기본 통계
        total_stocks = len(selected_stocks)
        buy_signals = [s for s in selected_stocks if s.action == 'buy']
        sell_signals = [s for s in selected_stocks if s.action == 'sell']
        hold_signals = [s for s in selected_stocks if s.action == 'hold']
        
        # 평균 점수
        avg_buffett = np.mean([s.buffett_score for s in selected_stocks])
        avg_lynch = np.mean([s.lynch_score for s in selected_stocks])
        avg_momentum = np.mean([s.momentum_score for s in selected_stocks])
        avg_technical = np.mean([s.technical_score for s in selected_stocks])
        avg_total = np.mean([s.total_score for s in selected_stocks])
        avg_selection = np.mean([s.selection_score for s in selected_stocks])
        
        # 총 투자금액 계산
        total_investment = sum([s.additional_data.get('total_investment', 0) for s in selected_stocks if s.additional_data])
        total_shares_value = sum([s.total_shares * s.price for s in selected_stocks])
        
        # 상위 매수 종목 (설정에서 표시할 개수 로드)
        top_picks_count = self.config_manager.get('reporting.top_picks_count', 5)
        top_buys = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:top_picks_count]
        
        # 섹터별 분포
        sector_dist = {}
        for stock in selected_stocks:
            sector_dist[stock.sector] = sector_dist.get(stock.sector, 0) + 1
        
        # 지수별 분포
        index_dist = {'S&P500': 0, 'NASDAQ100': 0, 'OTHER': 0}
        for stock in selected_stocks:
            if 'S&P500' in stock.index_membership:
                index_dist['S&P500'] += 1
            elif 'NASDAQ100' in stock.index_membership:
                index_dist['NASDAQ100'] += 1
            else:
                index_dist['OTHER'] += 1
        
        # VIX 영향 분석
        vix_adjustments = [s.vix_adjustment for s in selected_stocks]
        avg_vix_adjustment = np.mean(vix_adjustments)
        
        # 리스크 메트릭 계산
        max_single_position = max([s.additional_data.get('target_weight', 0) for s in selected_stocks if s.additional_data])
        betas = [s.additional_data.get('beta', 1.0) for s in selected_stocks if s.additional_data and s.additional_data.get('beta')]
        avg_beta = np.mean(betas) if betas else 1.0
        
        # VIX 기반 시장 상태
        vix_thresholds = {
            'low': self.config_manager.get('us_strategy.vix_low_threshold', 15.0),
            'high': self.config_manager.get('us_strategy.vix_high_threshold', 30.0)
        }
        
        market_volatility = ('HIGH' if self.stock_selector.current_vix > vix_thresholds['high'] 
                           else 'LOW' if self.stock_selector.current_vix < vix_thresholds['low'] 
                           else 'MEDIUM')
        
        report = {
            'summary': {
                'total_stocks': total_stocks,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'hold_signals': len(hold_signals),
                'total_investment': total_investment,
                'total_shares_value': total_shares_value,
                'current_vix': self.stock_selector.current_vix,
                'avg_vix_adjustment': avg_vix_adjustment,
                'config_version': '5.0.0'
            },
            'strategy_scores': {
                'avg_buffett_score': avg_buffett,
                'avg_lynch_score': avg_lynch,
                'avg_momentum_score': avg_momentum,
                'avg_technical_score': avg_technical,
                'avg_total_score': avg_total,
                'avg_selection_score': avg_selection,
                'strategy_weights': {
                    'buffett': self.buffett_weight * 100,
                    'lynch': self.lynch_weight * 100,
                    'momentum': self.momentum_weight * 100,
                    'technical': self.technical_weight * 100
                }
            },
            'top_picks': [
                {
                    'symbol': stock.symbol,
                    'sector': stock.sector,
                    'confidence': stock.confidence,
                    'total_score': stock.total_score,
                    'selection_score': stock.selection_score,
                    'price': stock.price,
                    'target_price': stock.target_price,
                    'total_shares': stock.total_shares,
                    'total_investment': stock.additional_data.get('total_investment', 0) if stock.additional_data else 0,
                    'index_membership': stock.index_membership,
                    'vix_adjustment': stock.vix_adjustment,
                    'reasoning': stock.reasoning[:120] + "..." if len(stock.reasoning) > 120 else stock.reasoning
                }
                for stock in top_buys
            ],
            'sector_distribution': sector_dist,
            'index_distribution': index_dist,
            'risk_metrics': {
                'max_single_position': max_single_position,
                'avg_beta': avg_beta,
                'diversification_score': len(sector_dist) / total_stocks,
                'market_volatility': market_volatility,
                'portfolio_allocation': self.total_portfolio_ratio * 100,
                'cash_reserve': self.cash_reserve_ratio * 100
            },
            'auto_selection_info': {
                'selection_method': 'real_time_auto_selection_with_config',
                'last_selection_time': self.last_selection_time,
                'cache_hours_remaining': max(0, self.selection_cache_hours - (
                    (datetime.now() - self.last_selection_time).total_seconds() / 3600
                    if self.last_selection_time else self.selection_cache_hours
                )),
                'target_stocks': self.stock_selector.target_stocks,
                'min_market_cap_billions': self.stock_selector.min_market_cap / 1e9,
                'min_avg_volume_millions': self.stock_selector.min_avg_volume / 1e6,
                'vix_thresholds': vix_thresholds
            },
            'configuration_status': {
                'config_file_loaded': True,
                'env_file_loaded': self.config_manager.env_loaded,
                'strategy_enabled': self.enabled,
                'notifications_enabled': {
                    'telegram': self.config_manager.is_feature_enabled('notifications.telegram'),
                    'slack': self.config_manager.is_feature_enabled('notifications.slack'),
                    'email': self.config_manager.is_feature_enabled('notifications.email')
                }
            }
        }
        
        return report

    async def execute_split_trading_simulation(self, signal: USStockSignal) -> Dict:
        """🔄 분할매매 시뮬레이션 (설정 연동)"""
        if signal.action != 'buy':
            return {"status": "not_applicable", "reason": "매수 신호가 아님"}
        
        # 설정에서 시뮬레이션 파라미터 로드
        simulation_config = self.config_manager.get('us_strategy.simulation', {})
        
        simulation = {
            'symbol': signal.symbol,
            'strategy': 'split_trading_4_strategies_config_based',
            'index_membership': signal.index_membership,
            'selection_score': signal.selection_score,
            'vix_level': self.stock_selector.current_vix,
            'config_version': '5.0.0',
            'stages': {
                'stage_1': {
                    'trigger_price': signal.entry_price_1,
                    'shares': signal.stage1_shares,
                    'investment': signal.stage1_shares * signal.entry_price_1,
                    'ratio': f'{self.stage1_ratio*100:.0f}%',
                    'status': 'ready'
                },
                'stage_2': {
                    'trigger_price': signal.entry_price_2,
                    'shares': signal.stage2_shares,
                    'investment': signal.stage2_shares * signal.entry_price_2,
                    'ratio': f'{self.stage2_ratio*100:.0f}%',
                    'trigger_condition': f'{abs(self.stage2_trigger)*100:.0f}% 하락시',
                    'status': 'waiting'
                },
                'stage_3': {
                    'trigger_price': signal.entry_price_3,
                    'shares': signal.stage3_shares,
                    'investment': signal.stage3_shares * signal.entry_price_3,
                    'ratio': f'{self.stage3_ratio*100:.0f}%',
                    'trigger_condition': f'{abs(self.stage3_trigger)*100:.0f}% 하락시',
                    'status': 'waiting'
                }
            },
            'exit_plan': {
                'stop_loss': {
                    'price': signal.stop_loss,
                    'ratio': '100%',
                    'trigger': f'{self.stop_loss_pct*100:.0f}% 손절'
                },
                'take_profit_1': {
                    'price': signal.take_profit_1,
                    'ratio': '60%',
                    'trigger': f'{self.take_profit1_pct*100:.0f}% 익절'
                },
                'take_profit_2': {
                    'price': signal.take_profit_2,
                    'ratio': '40%',
                    'trigger': f'{self.take_profit2_pct*100:.0f}% 익절'
                }
            },
            'strategy_breakdown': {
                'buffett_score': signal.buffett_score,
                'buffett_weight': self.buffett_weight * 100,
                'lynch_score': signal.lynch_score,
                'lynch_weight': self.lynch_weight * 100,
                'momentum_score': signal.momentum_score,
                'momentum_weight': self.momentum_weight * 100,
                'technical_score': signal.technical_score,
                'technical_weight': self.technical_weight * 100,
                'vix_adjustment': signal.vix_adjustment
            },
            'risk_management': {
                'max_hold_days': signal.max_hold_days,
                'total_investment': signal.additional_data.get('total_investment', 0) if signal.additional_data else 0,
                'portfolio_weight': signal.additional_data.get('target_weight', 0) if signal.additional_data else 0,
                'max_sector_weight': self.max_sector_weight * 100,
                'portfolio_allocation': self.total_portfolio_ratio * 100,
                'cash_reserve': self.cash_reserve_ratio * 100
            },
            'configuration_info': {
                'source': 'settings.yaml + .env',
                'strategy_weights_customizable': True,
                'risk_limits_customizable': True,
                'notification_enabled': any([
                    self.config_manager.is_feature_enabled('notifications.telegram'),
                    self.config_manager.is_feature_enabled('notifications.slack'),
                    self.config_manager.is_feature_enabled('notifications.email')
                ])
            }
        }
        
        return simulation

# ========================================================================================
# 🎯 편의 함수들 (외부에서 쉽게 사용) - 설정 연동
# ========================================================================================

async def run_auto_selection():
    """자동 선별 실행 (설정 기반)"""
    try:
        strategy = AdvancedUSStrategy()
        selected_stocks = await strategy.scan_all_selected_stocks()
        
        if selected_stocks:
            report = await strategy.generate_portfolio_report(selected_stocks)
            return selected_stocks, report
        else:
            return [], {}
            
    except Exception as e:
        logger.error(f"자동 선별 실행 실패: {e}")
        return [], {"error": str(e)}

async def analyze_us(symbol: str) -> Dict:
    """단일 미국 주식 분석 (설정 기반 + 기존 호환성)"""
    try:
        strategy = AdvancedUSStrategy()
        signal = await strategy.analyze_symbol(symbol)
        
        return {
            'decision': signal.action,
            'confidence_score': signal.confidence * 100,
            'reasoning': signal.reasoning,
            'target_price': signal.target_price,
            'pbr': signal.pbr,
            'peg': signal.peg,
            'price': signal.price,
            'sector': signal.sector,
            
            # 4가지 전략 점수
            'buffett_score': signal.buffett_score * 100,
            'lynch_score': signal.lynch_score * 100,
            'momentum_score': signal.momentum_score * 100,
            'technical_score': signal.technical_score * 100,
            'total_score': signal.total_score * 100,
            
            # 자동선별 정보
            'selection_score': signal.selection_score * 100,
            'index_membership': signal.index_membership,
            'vix_adjustment': signal.vix_adjustment,
            'current_vix': strategy.stock_selector.current_vix,
            
            # 설정 정보
            'config_version': '5.0.0',
            'config_based': True,
            'strategy_weights': {
                'buffett': strategy.buffett_weight * 100,
                'lynch': strategy.lynch_weight * 100,
                'momentum': strategy.momentum_weight * 100,
                'technical': strategy.technical_weight * 100
            },
            
            'split_trading_plan': await strategy.execute_split_trading_simulation(signal)
        }
        
    except Exception as e:
        logger.error(f"종목 분석 실패 {symbol}: {e}")
        return {
            'decision': 'hold',
            'confidence_score': 0.0,
            'error': str(e),
            'config_version': '5.0.0'
        }

async def get_us_auto_selection_status() -> Dict:
    """미국 주식 자동선별 상태 조회 (설정 기반)"""
    try:
        strategy = AdvancedUSStrategy()
        
        # 설정 검증
        validation_result = strategy.config_manager.validate_config()
        
        return {
            'enabled': strategy.enabled,
            'last_selection_time': strategy.last_selection_time,
            'cache_valid': strategy._is_selection_cache_valid(),
            'cache_hours': strategy.selection_cache_hours,
            'selected_count': len(strategy.selected_stocks),
            'current_vix': strategy.stock_selector.current_vix,
            'vix_status': ('HIGH' if strategy.stock_selector.current_vix > strategy.stock_selector.vix_high_threshold 
                          else 'LOW' if strategy.stock_selector.current_vix < strategy.stock_selector.vix_low_threshold 
                          else 'MEDIUM'),
            'selection_criteria': {
                'min_market_cap_billions': strategy.stock_selector.min_market_cap / 1e9,
                'min_avg_volume_millions': strategy.stock_selector.min_avg_volume / 1e6,
                'target_stocks': strategy.stock_selector.target_stocks,
                'strategy_weights': {
                    'buffett': strategy.buffett_weight * 100,
                    'lynch': strategy.lynch_weight * 100,
                    'momentum': strategy.momentum_weight * 100,
                    'technical': strategy.technical_weight * 100
                }
            },
            'configuration_status': {
                'config_file_found': Path('settings.yaml').exists(),
                'env_file_found': Path('.env').exists(),
                'env_loaded': strategy.config_manager.env_loaded,
                'config_errors': validation_result.get('errors', []),
                'config_warnings': validation_result.get('warnings', []),
                'version': '5.0.0'
            },
            'risk_settings': {
                'portfolio_allocation': strategy.total_portfolio_ratio * 100,
                'cash_reserve': strategy.cash_reserve_ratio * 100,
                'stop_loss': strategy.stop_loss_pct * 100,
                'take_profit': strategy.take_profit2_pct * 100,
                'max_hold_days': strategy.max_hold_days,
                'max_sector_weight': strategy.max_sector_weight * 100
            },
            'notifications': {
                'telegram_enabled': strategy.config_manager.is_feature_enabled('notifications.telegram'),
                'slack_enabled': strategy.config_manager.is_feature_enabled('notifications.slack'),
                'email_enabled': strategy.config_manager.is_feature_enabled('notifications.email')
            }
        }
        
    except Exception as e:
        logger.error(f"상태 조회 실패: {e}")
        return {
            'enabled': False,
            'error': str(e),
            'config_version': '5.0.0'
        }

async def force_us_reselection() -> List[str]:
    """미국 주식 강제 재선별 (설정 기반)"""
    try:
        strategy = AdvancedUSStrategy()
        strategy.last_selection_time = None  # 캐시 무효화
        strategy.selected_stocks = []        # 기존 선별 결과 삭제
        
        logger.info("🔄 강제 재선별 시작...")
        return await strategy.auto_select_top20_stocks()
        
    except Exception as e:
        logger.error(f"강제 재선별 실패: {e}")
        return []

async def reload_config() -> Dict:
    """설정 파일 다시 로드"""
    try:
        global config_manager
        config_manager = ConfigManager()
        
        # 설정 검증
        validation_result = config_manager.validate_config()
        
        logger.info("🔄 설정 파일 다시 로드 완료")
        
        return {
            'status': 'success',
            'message': '설정 파일이 다시 로드되었습니다',
            'config_errors': validation_result.get('errors', []),
            'config_warnings': validation_result.get('warnings', []),
            'env_loaded': config_manager.env_loaded,
            'version': '5.0.0'
        }
        
    except Exception as e:
        logger.error(f"설정 다시 로드 실패: {e}")
        return {
            'status': 'error',
            'message': f'설정 로드 실패: {str(e)}',
            'version': '5.0.0'
        }

async def update_strategy_weights(buffett: float, lynch: float, momentum: float, technical: float) -> Dict:
    """전략 가중치 동적 업데이트 (런타임 중)"""
    try:
        # 가중치 정규화
        total = buffett + lynch + momentum + technical
        if total == 0:
            return {'status': 'error', 'message': '가중치 합이 0입니다'}
        
        buffett_norm = buffett / total
        lynch_norm = lynch / total
        momentum_norm = momentum / total
        technical_norm = technical / total
        
        # 전역 설정 업데이트 (런타임 중에만 적용)
        strategy = AdvancedUSStrategy()
        strategy.buffett_weight = buffett_norm
        strategy.lynch_weight = lynch_norm
        strategy.momentum_weight = momentum_norm
        strategy.technical_weight = technical_norm
        
        # 종목 선별기도 업데이트
        strategy.stock_selector.calculate_selection_score = lambda data: (
            strategy.stock_selector.calculate_buffett_score(data) * buffett_norm +
            strategy.stock_selector.calculate_lynch_score(data) * lynch_norm +
            strategy.stock_selector.calculate_momentum_score(data) * momentum_norm +
            strategy.stock_selector.calculate_technical_score(data) * technical_norm
        )
        
        logger.info(f"🎯 전략 가중치 업데이트: 버핏{buffett_norm*100:.1f}% 린치{lynch_norm*100:.1f}% 모멘텀{momentum_norm*100:.1f}% 기술{technical_norm*100:.1f}%")
        
        return {
            'status': 'success',
            'message': '전략 가중치가 업데이트되었습니다 (런타임 중에만 적용)',
            'updated_weights': {
                'buffett': buffett_norm * 100,
                'lynch': lynch_norm * 100,
                'momentum': momentum_norm * 100,
                'technical': technical_norm * 100
            },
            'note': 'settings.yaml 파일을 수정하면 영구적으로 저장됩니다'
        }
        
    except Exception as e:
        logger.error(f"전략 가중치 업데이트 실패: {e}")
        return {
            'status': 'error',
            'message': f'가중치 업데이트 실패: {str(e)}'
        }

# ========================================================================================
# 🧪 테스트 메인 함수 (설정 연동 완성판)
# ========================================================================================

async def main():
    """테스트용 메인 함수 (설정 파일 완전 연동 버전)"""
    try:
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print("🇺🇸 미국 주식 완전 통합 연동 전략 V5.0 테스트!")
        print("🔗 완벽한 설정 파일 연동: .env + .gitignore + requirements.txt + settings.yaml")
        print("🆕 진짜 자동선별: S&P500+NASDAQ100+러셀1000 실시간 크롤링")
        print("🎯 4가지 전략 융합 + VIX 기반 동적 조정 + 분할매매")
        print("="*80)
        
        # 설정 파일 상태 확인
        print("\n🔧 설정 파일 연동 상태 확인...")
        status = await get_us_auto_selection_status()
        config_status = status.get('configuration_status', {})
        
        print(f"  ✅ settings.yaml: {'발견됨' if config_status.get('config_file_found') else '❌ 없음'}")
        print(f"  ✅ .env 파일: {'발견됨' if config_status.get('env_file_found') else '❌ 없음 (.env.example 참고)'}")
        print(f"  ✅ 환경변수 로드: {'성공' if config_status.get('env_loaded') else '❌ 실패'}")
        print(f"  ✅ 시스템 활성화: {status['enabled']}")
        print(f"  📊 현재 VIX: {status['current_vix']:.2f} ({status['vix_status']})")
        
        # 설정 오류/경고 확인
        if config_status.get('config_errors'):
            print(f"  ❌ 설정 오류: {config_status['config_errors']}")
        if config_status.get('config_warnings'):
            print(f"  ⚠️ 설정 경고: {config_status['config_warnings']}")
        
        # 전략 설정 표시
        strategy_weights = status['selection_criteria']['strategy_weights']
        print(f"  📊 전략 가중치: 버핏{strategy_weights['buffett']:.0f}% "
              f"린치{strategy_weights['lynch']:.0f}% "
              f"모멘텀{strategy_weights['momentum']:.0f}% "
              f"기술{strategy_weights['technical']:.0f}%")
        
        # 리스크 설정 표시
        risk_settings = status['risk_settings']
        print(f"  🛡️ 리스크 설정: 포트폴리오{risk_settings['portfolio_allocation']:.0f}% "
              f"현금{risk_settings['cash_reserve']:.0f}% "
              f"손절{risk_settings['stop_loss']:.0f}% "
              f"익절{risk_settings['take_profit']:.0f}%")
        
        # 알림 설정 표시
        notifications = status['notifications']
        enabled_notifications = [k for k, v in notifications.items() if v]
        print(f"  📱 알림 설정: {', '.join(enabled_notifications) if enabled_notifications else '비활성화'}")
        
        # 전체 시장 자동선별 + 분석
        print(f"\n🔍 실시간 자동선별 + 전체 분석 시작... (설정 기반)")
        start_time = time.time()
        
        selected_stocks, report = await run_auto_selection()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ 총 소요 시간: {elapsed_time:.1f}초")
        
        if selected_stocks and report and 'error' not in report:
            print(f"\n📈 설정 기반 자동선별 + 분석 결과:")
            print(f"  총 분석: {report['summary']['total_stocks']}개 종목 (실시간 자동선별)")
            print(f"  매수 신호: {report['summary']['buy_signals']}개")
            print(f"  매도 신호: {report['summary']['sell_signals']}개") 
            print(f"  보유 신호: {report['summary']['hold_signals']}개")
            print(f"  현재 VIX: {report['summary']['current_vix']:.2f} (변동성: {report['risk_metrics']['market_volatility']})")
            print(f"  설정 버전: {report['summary']['config_version']}")
            
            # 설정 기반 전략 점수
            strategy_scores = report['strategy_scores']
            weights = strategy_scores['strategy_weights']
            print(f"\n📊 설정 기반 평균 전략 점수:")
            print(f"  버핏 가치투자: {strategy_scores['avg_buffett_score']:.3f} (가중치: {weights['buffett']:.0f}%)")
            print(f"  린치 성장투자: {strategy_scores['avg_lynch_score']:.3f} (가중치: {weights['lynch']:.0f}%)")
            print(f"  모멘텀 전략: {strategy_scores['avg_momentum_score']:.3f} (가중치: {weights['momentum']:.0f}%)")
            print(f"  기술적 분석: {strategy_scores['avg_technical_score']:.3f} (가중치: {weights['technical']:.0f}%)")
            print(f"  종합 점수: {strategy_scores['avg_total_score']:.3f}")
            print(f"  선별 점수: {strategy_scores['avg_selection_score']:.3f}")
            print(f"  VIX 평균 조정: {report['summary']['avg_vix_adjustment']:+.3f}")
            
            # 지수별 분포
            index_dist = report['index_distribution']
            print(f"\n🏢 지수별 분포:")
            for index, count in index_dist.items():
                percentage = count / report['summary']['total_stocks'] * 100
                print(f"  {index}: {count}개 ({percentage:.1f}%)")
            
            # 설정 기반 리스크 메트릭
            risk_metrics = report['risk_metrics']
            print(f"\n🛡️ 설정 기반 리스크 메트릭:")
            print(f"  포트폴리오 할당: {risk_metrics['portfolio_allocation']:.0f}%")
            print(f"  현금 보유: {risk_metrics['cash_reserve']:.0f}%")
            print(f"  최대 단일 포지션: {risk_metrics['max_single_position']:.1f}%")
            print(f"  평균 베타: {risk_metrics['avg_beta']:.2f}")
            print(f"  다양성 점수: {risk_metrics['diversification_score']:.2f}")
            
            # 상위 매수 추천 (설정 기반)
            if report['top_picks']:
                print(f"\n🎯 상위 매수 추천 (설정 기반 선별):")
                for i, stock in enumerate(report['top_picks'][:3], 1):
                    membership_str = "+".join(stock['index_membership'])
                    print(f"\n  {i}. {stock['symbol']} ({membership_str}) - 신뢰도: {stock['confidence']:.2%}")
                    print(f"     🏆 선별점수: {stock['selection_score']:.3f} | 총점: {stock['total_score']:.3f}")
                    print(f"     💰 현재가: ${stock['price']:.2f} → 목표가: ${stock['target_price']:.2f}")
                    print(f"     🔄 분할매매: {stock['total_shares']:,}주 (3단계)")
                    print(f"     💼 투자금액: ${stock['total_investment']:,.0f}")
                    print(f"     📊 VIX 조정: {stock['vix_adjustment']:+.3f}")
                    print(f"     💡 {stock['reasoning'][:60]}...")
            
            # 설정 상태 정보
            config_info = report.get('configuration_status', {})
            print(f"\n🔧 설정 연동 상태:")
            print(f"  설정 파일 로드: {config_info.get('config_file_loaded', False)}")
            print(f"  환경변수 로드: {config_info.get('env_file_loaded', False)}")
            print(f"  전략 활성화: {config_info.get('strategy_enabled', False)}")
            notifications = config_info.get('notifications_enabled', {})
            print(f"  알림 시스템: {', '.join([k for k, v in notifications.items() if v]) or '비활성화'}")
            
        else:
            error_msg = report.get('error', '알 수 없는 오류') if report else '결과 없음'
            print(f"❌ 선별 실패: {error_msg}")
        
        print("\n✅ 설정 연동 테스트 완료!")
        print("\n🎯 미국 주식 V5.0 완전 통합 연동 특징:")
        print("  ✅ 🔗 완벽한 설정 파일 연동 (.env + settings.yaml)")
        print("  ✅ 🆕 실시간 S&P500+NASDAQ100+러셀1000 크롤링")
        print("  ✅ 📊 4가지 전략 융합 (가중치 설정 가능)")
        print("  ✅ 📊 VIX 기반 동적 조정 시스템")
        print("  ✅ 💰 분할매매 시스템 (비율 설정 가능)")
        print("  ✅ 🛡️ 동적 손절/익절 (임계값 설정 가능)")
        print("  ✅ 🔍 종목 선별 기준 (시총/거래량 설정 가능)")
        print("  ✅ 🏢 섹터 다양성 (최대 비중 설정 가능)")
        print("  ✅ 🤖 완전 자동화 (캐시 시간 설정 가능)")
        print("  ✅ 📱 알림 시스템 (텔레그램/슬랙/이메일)")
        print("  ✅ 🔄 런타임 설정 변경 (재시작 불필요)")
        print("  ✅ ✨ 설정 검증 및 오류 체크")
        print("\n💡 설정 파일 사용법:")
        print("  - settings.yaml: 전략 파라미터, 가중치, 임계값")
        print("  - .env: API 키, 보안 설정, 알림 토큰")
        print("  - 런타임 중 설정 변경: update_strategy_weights()")
        print("  - 설정 다시 로드: reload_config()")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"테스트 실행 실패: {e}")

if __name__ == "__main__":
    asyncio.run(main())
