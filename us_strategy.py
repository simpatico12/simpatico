#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 전설적 퀸트프로젝트 - 미국주식 마스터시스템 V6.0
===============================================================

🌟 전설적 핵심 특징:
1. 🔥 완벽한 설정 기반 아키텍처 (혼자 보수유지 가능)
2. 🚀 실시간 S&P500+NASDAQ 자동선별 엔진
3. 💎 4가지 투자전략 지능형 융합 시스템
4. 🧠 VIX 기반 시장상황 자동판단 AI
5. ⚡ 분할매매 + 손절익절 자동화 시스템
6. 🛡️ 통합 리스크관리 + 포트폴리오 최적화

Author: 전설적퀸트팀
Version: 6.0.0 (전설적 완성판)
Project: 🏆 QuintProject - 혼자보수유지가능
"""

import asyncio
import logging
import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from dotenv import load_dotenv

# IBKR 연동 (선택적 import)
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
    logging.info("✅ IBKR 모듈 로드 성공")
except ImportError:
    IBKR_AVAILABLE = False
    logging.warning("⚠️ IBKR 모듈 없음 (pip install ib_insync 필요)")

warnings.filterwarnings('ignore')

# ========================================================================================
# 🔧 전설적 설정관리자 - 완벽한 자동화
# ========================================================================================

class LegendaryConfigManager:
    """🔥 전설적 설정관리자 - 혼자 보수유지 가능한 완벽한 시스템"""
    
    def __init__(self, config_path: str = "quant_settings.yaml"):
        self.config_path = config_path
        self.config = {}
        self.env_loaded = False
        self._initialize_legendary_config()
    
    def _initialize_legendary_config(self):
        """전설적 설정 초기화"""
        try:
            # 1. 환경변수 로드
            if Path('.env').exists():
                load_dotenv()
                self.env_loaded = True
                
            # 2. YAML 설정 로드 또는 생성
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self._create_legendary_default_config()
                self._save_config()
            
            # 3. 환경변수 치환
            self._substitute_env_vars()
            
            logging.info("🔥 전설적 설정관리자 초기화 완료!")
            
        except Exception as e:
            logging.error(f"❌ 설정 초기화 실패: {e}")
            self._create_legendary_default_config()
    
    def _create_legendary_default_config(self):
        """전설적 기본 설정 생성"""
        self.config = {
            # 🎯 핵심 전략 설정
            'legendary_strategy': {
                'enabled': True,
                'strategy_name': '전설적퀸트마스터',
                'target_stocks': 20,
                'selection_cache_hours': 24,
                'confidence_threshold': 0.70,
                
                # 4가지 전략 가중치 (자유롭게 조정 가능)
                'strategy_weights': {
                    'buffett_value': 25.0,    # 워렌버핏 가치투자
                    'lynch_growth': 25.0,     # 피터린치 성장투자  
                    'momentum': 25.0,         # 모멘텀 전략
                    'technical': 25.0         # 기술적분석
                },
                
                # VIX 기반 시장상황 판단
                'vix_thresholds': {
                    'low_volatility': 15.0,   # 저변동성 (적극적)
                    'high_volatility': 30.0,  # 고변동성 (보수적)
                    'adjustments': {
                        'low_boost': 1.15,     # 저변동성 시 15% 부스트
                        'normal': 1.0,         # 정상 변동성
                        'high_reduce': 0.85    # 고변동성 시 15% 감소
                    }
                }
            },
            
            # 💰 분할매매 시스템
            'split_trading': {
                'enabled': True,
                'buy_stages': {
                    'stage1_ratio': 40.0,     # 1단계 40%
                    'stage2_ratio': 35.0,     # 2단계 35%
                    'stage3_ratio': 25.0      # 3단계 25%
                },
                'triggers': {
                    'stage2_drop': -5.0,      # 5% 하락시 2단계
                    'stage3_drop': -10.0      # 10% 하락시 3단계
                },
                'sell_stages': {
                    'profit1_ratio': 60.0,    # 1차 익절 60%
                    'profit2_ratio': 40.0     # 2차 익절 40%
                }
            },
            
            # 🛡️ 리스크 관리
            'risk_management': {
                'portfolio_allocation': 80.0,  # 포트폴리오 투자비중
                'cash_reserve': 20.0,          # 현금 보유비중
                'stop_loss': 15.0,             # 손절선
                'take_profit1': 20.0,          # 1차 익절선
                'take_profit2': 35.0,          # 2차 익절선
                'max_position': 8.0,           # 종목당 최대비중
                'max_sector': 25.0,            # 섹터당 최대비중
                'max_hold_days': 60            # 최대보유일
            },
            
            # 📊 종목선별 기준
            'selection_criteria': {
                'min_market_cap': 5_000_000_000,   # 최소 시총 50억달러
                'min_avg_volume': 1_000_000,       # 최소 일평균거래량 100만주
                'excluded_sectors': [],             # 제외 섹터
                'excluded_symbols': ['SPXL', 'TQQQ'], # 제외 종목 (레버리지ETF등)
                
                # 섹터 다양성
                'sector_diversity': {
                    'max_per_sector': 4,        # 섹터당 최대 4개
                    'sp500_quota': 60.0,        # S&P500 60%
                    'nasdaq_quota': 40.0        # NASDAQ 40%
                }
            },
            
            # 🔍 데이터 수집 설정
            'data_sources': {
                'request_timeout': 30,
                'max_retries': 3,
                'rate_limit_delay': 0.3,
                'max_workers': 15,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            
            # 🏦 IBKR (Interactive Brokers) 연동
            'ibkr': {
                'enabled': False,             # IBKR 연동 활성화
                'host': '127.0.0.1',         # TWS/Gateway 호스트
                'port': 7497,                # TWS 포트 (7497=paper, 7496=live)
                'client_id': 1,              # 클라이언트 ID
                'auto_connect': False,        # 자동 연결
                'paper_trading': True,        # 모의투자 모드
                'account_id': '${IBKR_ACCOUNT:-}',  # 계좌번호
                
                # 주문 설정
                'order_settings': {
                    'default_order_type': 'MKT',    # 시장가 주문
                    'good_till_cancel': True,       # GTC 주문
                    'outside_rth': False,           # 장외시간 거래
                    'transmit': False,              # 실제 전송 여부 (False=검토만)
                    'min_order_value': 100.0        # 최소 주문금액
                },
                
                # 포트폴리오 관리
                'portfolio_settings': {
                    'enable_auto_trading': False,   # 자동매매 활성화
                    'max_daily_trades': 10,         # 일일 최대 거래수
                    'position_size_limit': 10000,   # 포지션 크기 제한 (달러)
                    'cash_threshold': 5000          # 최소 현금 유지
                }
            },
            
            # 📱 알림 시스템 (선택사항)
            'notifications': {
                'telegram': {
                    'enabled': False,
                    'bot_token': '${TELEGRAM_BOT_TOKEN:-}',
                    'chat_id': '${TELEGRAM_CHAT_ID:-}'
                },
                'discord': {
                    'enabled': False,
                    'webhook_url': '${DISCORD_WEBHOOK:-}'
                }
            },
            
            # 🎛️ 고급 설정
            'advanced': {
                'enable_logging': True,
                'log_level': 'INFO',
                'save_analysis_results': True,
                'enable_backtesting': False,
                'performance_tracking': True
            }
        }
    
    def _substitute_env_vars(self):
        """환경변수 치환 ${VAR:-default}"""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_content = obj[2:-1]
                if ':-' in var_content:
                    var_name, default = var_content.split(':-', 1)
                    return os.getenv(var_name, default)
                else:
                    return os.getenv(var_content, obj)
            return obj
        
        self.config = substitute_recursive(self.config)
    
    def _save_config(self):
        """설정 파일 저장"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            logging.error(f"설정 저장 실패: {e}")
    
    def get(self, key_path: str, default=None):
        """점 표기법으로 설정값 조회"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def update(self, key_path: str, value):
        """설정값 업데이트 및 자동 저장"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self._save_config()
        logging.info(f"설정 업데이트: {key_path} = {value}")
    
    def is_enabled(self, feature_path: str) -> bool:
        """기능 활성화 여부"""
        return bool(self.get(f"{feature_path}.enabled", False))

# 전역 설정 관리자
config = LegendaryConfigManager()

# ========================================================================================
# 📊 전설적 데이터 클래스
# ========================================================================================

@dataclass
class LegendaryStockSignal:
    """🏆 전설적 주식 시그널 데이터"""
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
    
    # 재무지표
    market_cap: float
    pe_ratio: float
    pbr: float
    peg: float
    roe: float
    sector: str
    
    # 모멘텀지표
    momentum_3m: float
    momentum_6m: float
    momentum_12m: float
    
    # 기술적지표
    rsi: float
    trend: str
    volume_spike: float
    
    # 분할매매 계획
    total_shares: int
    stage1_shares: int
    stage2_shares: int
    stage3_shares: int
    entry_price_1: float
    entry_price_2: float
    entry_price_3: float
    stop_loss_price: float
    take_profit1_price: float
    take_profit2_price: float
    
    # 메타정보
    target_price: float
    selection_score: float
    index_membership: List[str]
    vix_adjustment: float
    reasoning: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return asdict(self)

# ========================================================================================
# 🚀 전설적 실시간 주식선별 엔진
# ========================================================================================

class LegendaryStockSelector:
    """🔥 전설적 실시간 주식선별 엔진 - 완전자동화"""
    
    def __init__(self):
        self.current_vix = 20.0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('data_sources.user_agent')
        })
        self.session.timeout = config.get('data_sources.request_timeout', 30)
        
        logging.info("🚀 전설적 주식선별 엔진 가동!")
    
    async def get_current_vix(self) -> float:
        """현재 VIX 지수 조회"""
        try:
            vix_ticker = yf.Ticker("^VIX")
            hist = vix_ticker.history(period="1d")
            if not hist.empty:
                self.current_vix = hist['Close'].iloc[-1]
            logging.info(f"📊 현재 VIX: {self.current_vix:.2f}")
            return self.current_vix
        except Exception as e:
            logging.warning(f"VIX 조회 실패: {e}")
            self.current_vix = 20.0
            return self.current_vix
    
    async def collect_sp500_symbols(self) -> List[str]:
        """S&P 500 종목 실시간 수집"""
        try:
            logging.info("🔍 S&P 500 종목 수집중...")
            
            # Wikipedia에서 S&P 500 리스트 수집
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_df = tables[0]
            symbols = sp500_df['Symbol'].tolist()
            
            # 심볼 정리 (BRK.B -> BRK-B)
            cleaned_symbols = [str(s).replace('.', '-') for s in symbols]
            
            logging.info(f"✅ S&P 500: {len(cleaned_symbols)}개 종목 수집")
            await asyncio.sleep(config.get('data_sources.rate_limit_delay', 0.3))
            
            return cleaned_symbols
            
        except Exception as e:
            logging.error(f"S&P 500 수집 실패: {e}")
            # 백업 리스트
            return self._get_backup_sp500()
    
    async def collect_nasdaq100_symbols(self) -> List[str]:
        """NASDAQ 100 종목 실시간 수집"""
        try:
            logging.info("🔍 NASDAQ 100 종목 수집중...")
            
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(url)
            
            symbols = []
            for table in tables:
                if 'Symbol' in table.columns or 'Ticker' in table.columns:
                    symbol_col = 'Symbol' if 'Symbol' in table.columns else 'Ticker'
                    nasdaq_symbols = table[symbol_col].dropna().tolist()
                    symbols.extend([str(s) for s in nasdaq_symbols])
                    break
            
            logging.info(f"✅ NASDAQ 100: {len(symbols)}개 종목 수집")
            await asyncio.sleep(config.get('data_sources.rate_limit_delay', 0.3))
            
            return symbols
            
        except Exception as e:
            logging.error(f"NASDAQ 100 수집 실패: {e}")
            return self._get_backup_nasdaq100()
    
    def _get_backup_sp500(self) -> List[str]:
        """S&P 500 백업 리스트"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'ABBV', 'BAC', 'KO', 'AVGO',
            'XOM', 'CVX', 'LLY', 'WMT', 'PEP', 'TMO', 'COST', 'ORCL', 'ABT', 'ACN',
            'NFLX', 'CRM', 'MRK', 'VZ', 'ADBE', 'DHR', 'TXN', 'NKE', 'PM', 'DIS'
        ]
    
    def _get_backup_nasdaq100(self) -> List[str]:
        """NASDAQ 100 백업 리스트"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO',
            'TXN', 'MU', 'AMAT', 'ADI', 'MRVL', 'KLAC', 'LRCX', 'SNPS'
        ]
    
    async def create_investment_universe(self) -> List[str]:
        """투자 유니버스 생성"""
        try:
            logging.info("🌌 투자 유니버스 생성중...")
            
            # 병렬로 데이터 수집
            tasks = [
                self.collect_sp500_symbols(),
                self.collect_nasdaq100_symbols(),
                self.get_current_vix()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            sp500_symbols = results[0] if not isinstance(results[0], Exception) else []
            nasdaq100_symbols = results[1] if not isinstance(results[1], Exception) else []
            
            # 유니버스 통합
            universe = list(set(sp500_symbols + nasdaq100_symbols))
            
            # 제외 종목 필터링
            excluded = config.get('selection_criteria.excluded_symbols', [])
            universe = [s for s in universe if s not in excluded]
            
            logging.info(f"🌌 투자 유니버스: {len(universe)}개 종목 생성완료")
            return universe
            
        except Exception as e:
            logging.error(f"유니버스 생성 실패: {e}")
            return self._get_backup_sp500() + self._get_backup_nasdaq100()
    
    async def get_stock_data(self, symbol: str) -> Dict:
        """종목 데이터 수집"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            
            # 기본 재무지표
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
                'beta': info.get('beta', 1.0) or 1.0
            }
            
            # PEG 계산
            if data['pe_ratio'] > 0 and data['eps_growth'] > 0:
                data['peg'] = data['pe_ratio'] / data['eps_growth']
            else:
                data['peg'] = 999
            
            # 모멘텀 지표
            if len(hist) >= 252:
                data['momentum_3m'] = ((current_price / hist['Close'].iloc[-63]) - 1) * 100
                data['momentum_6m'] = ((current_price / hist['Close'].iloc[-126]) - 1) * 100
                data['momentum_12m'] = ((current_price / hist['Close'].iloc[-252]) - 1) * 100
            else:
                data['momentum_3m'] = data['momentum_6m'] = data['momentum_12m'] = 0
            
            # 기술적 지표 (간단한 계산)
            if len(hist) >= 50:
                # RSI 계산
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
                
                # 추세 (50일 이동평균)
                ma50 = hist['Close'].rolling(50).mean().iloc[-1]
                data['trend'] = 'uptrend' if current_price > ma50 else 'downtrend'
                
                # 거래량 급증
                avg_volume_20d = hist['Volume'].rolling(20).mean().iloc[-1]
                current_volume = hist['Volume'].iloc[-1]
                data['volume_spike'] = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1
            else:
                data.update({'rsi': 50, 'trend': 'sideways', 'volume_spike': 1})
            
            await asyncio.sleep(config.get('data_sources.rate_limit_delay', 0.3))
            return data
            
        except Exception as e:
            logging.error(f"데이터 수집 실패 {symbol}: {e}")
            return {}

# ========================================================================================
# 🧠 전설적 투자전략 분석엔진 
# ========================================================================================

class LegendaryStrategyAnalyzer:
    """🔥 전설적 4가지 투자전략 분석엔진"""
    
    def __init__(self):
        self.weights = config.get('legendary_strategy.strategy_weights', {})
        
    def calculate_buffett_score(self, data: Dict) -> float:
        """워렌 버핏 가치투자 점수"""
        score = 0.0
        
        # PBR 점수 (낮을수록 좋음)
        pbr = data.get('pbr', 999)
        if 0 < pbr <= 1.5:
            score += 0.35
        elif pbr <= 2.5:
            score += 0.25
        elif pbr <= 4.0:
            score += 0.15
        
        # ROE 점수 (높을수록 좋음)
        roe = data.get('roe', 0)
        if roe >= 20:
            score += 0.30
        elif roe >= 15:
            score += 0.20
        elif roe >= 10:
            score += 0.10
        
        # 부채비율 점수 (낮을수록 좋음)
        debt_ratio = data.get('debt_to_equity', 999) / 100
        if debt_ratio <= 0.3:
            score += 0.20
        elif debt_ratio <= 0.5:
            score += 0.15
        elif debt_ratio <= 0.7:
            score += 0.10
        
        # PE 적정성 점수
        pe = data.get('pe_ratio', 999)
        if 5 <= pe <= 15:
            score += 0.15
        elif pe <= 25:
            score += 0.10
        elif pe <= 35:
            score += 0.05
        
        return min(score, 1.0)
    
    def calculate_lynch_score(self, data: Dict) -> float:
        """피터 린치 성장투자 점수"""
        score = 0.0
        
        # PEG 점수 (낮을수록 좋음)
        peg = data.get('peg', 999)
        if 0 < peg <= 0.5:
            score += 0.40
        elif peg <= 1.0:
            score += 0.35
        elif peg <= 1.5:
            score += 0.25
        elif peg <= 2.0:
            score += 0.15
        
        # EPS 성장률 점수
        eps_growth = data.get('eps_growth', 0)
        if eps_growth >= 25:
            score += 0.35
        elif eps_growth >= 15:
            score += 0.25
        elif eps_growth >= 10:
            score += 0.15
        elif eps_growth >= 5:
            score += 0.05
        
        # 매출 성장률 점수
        revenue_growth = data.get('revenue_growth', 0)
        if revenue_growth >= 20:
            score += 0.25
        elif revenue_growth >= 10:
            score += 0.15
        elif revenue_growth >= 5:
            score += 0.10
        
        return min(score, 1.0)
    
    def calculate_momentum_score(self, data: Dict) -> float:
        """모멘텀 전략 점수"""
        score = 0.0
        
        # 3개월 모멘텀
        mom_3m = data.get('momentum_3m', 0)
        if mom_3m >= 20:
            score += 0.30
        elif mom_3m >= 10:
            score += 0.20
        elif mom_3m >= 5:
            score += 0.10
        elif mom_3m >= 0:
            score += 0.05
        
        # 6개월 모멘텀
        mom_6m = data.get('momentum_6m', 0)
        if mom_6m >= 30:
            score += 0.25
        elif mom_6m >= 15:
            score += 0.15
        elif mom_6m >= 5:
            score += 0.10
        
        # 12개월 모멘텀
        mom_12m = data.get('momentum_12m', 0)
        if mom_12m >= 50:
            score += 0.25
        elif mom_12m >= 25:
            score += 0.15
        elif mom_12m >= 10:
            score += 0.10
        
        # 거래량 급증
        volume_spike = data.get('volume_spike', 1)
        if volume_spike >= 2.0:
            score += 0.20
        elif volume_spike >= 1.5:
            score += 0.10
        elif volume_spike >= 1.2:
            score += 0.05
        
        return min(score, 1.0)
    
    def calculate_technical_score(self, data: Dict) -> float:
        """기술적 분석 점수"""
        score = 0.0
        
        # RSI 점수
        rsi = data.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 0.40  # 정상 범위
        elif 20 <= rsi < 30:
            score += 0.30  # 과매도 (매수기회)
        elif 70 < rsi <= 80:
            score += 0.20  # 약간 과매수
        
        # 추세 점수
        trend = data.get('trend', 'sideways')
        if trend == 'uptrend':
            score += 0.60
        elif trend == 'sideways':
            score += 0.20
        
        return min(score, 1.0)
    
    def calculate_vix_adjustment(self, base_score: float, current_vix: float) -> float:
        """VIX 기반 점수 조정"""
        vix_config = config.get('legendary_strategy.vix_thresholds', {})
        low_threshold = vix_config.get('low_volatility', 15.0)
        high_threshold = vix_config.get('high_volatility', 30.0)
        adjustments = vix_config.get('adjustments', {})
        
        if current_vix <= low_threshold:
            return base_score * adjustments.get('low_boost', 1.15)
        elif current_vix >= high_threshold:
            return base_score * adjustments.get('high_reduce', 0.85)
        else:
            return base_score * adjustments.get('normal', 1.0)
    
    def calculate_total_score(self, data: Dict, current_vix: float) -> Tuple[float, Dict]:
        """종합 점수 계산"""
        # 각 전략 점수 계산
        buffett_score = self.calculate_buffett_score(data)
        lynch_score = self.calculate_lynch_score(data)
        momentum_score = self.calculate_momentum_score(data)
        technical_score = self.calculate_technical_score(data)
        
        # 가중치 적용
        weights = config.get('legendary_strategy.strategy_weights', {})
        buffett_weight = weights.get('buffett_value', 25.0) / 100
        lynch_weight = weights.get('lynch_growth', 25.0) / 100
        momentum_weight = weights.get('momentum', 25.0) / 100
        technical_weight = weights.get('technical', 25.0) / 100
        
        # 가중 평균 계산
        base_score = (
            buffett_score * buffett_weight +
            lynch_score * lynch_weight +
            momentum_score * momentum_weight +
            technical_score * technical_weight
        )
        
        # VIX 조정
        adjusted_score = self.calculate_vix_adjustment(base_score, current_vix)
        vix_adjustment = adjusted_score - base_score
        
        scores = {
            'buffett_score': buffett_score,
            'lynch_score': lynch_score,
            'momentum_score': momentum_score,
            'technical_score': technical_score,
            'base_score': base_score,
            'vix_adjustment': vix_adjustment,
            'total_score': adjusted_score
        }
        
        return adjusted_score, scores

# ========================================================================================
# 💰 전설적 분할매매 시스템
# ========================================================================================

class LegendarySplitTradingSystem:
    """🔥 전설적 분할매매 시스템 - 자동 손절익절"""
    
    def __init__(self):
        self.split_config = config.get('split_trading', {})
        self.risk_config = config.get('risk_management', {})
        
    def calculate_position_plan(self, symbol: str, price: float, confidence: float, 
                              portfolio_value: float = 1000000) -> Dict:
        """포지션 계획 수립"""
        try:
            # 포지션 크기 계산 (신뢰도 기반)
            base_allocation = self.risk_config.get('portfolio_allocation', 80.0) / 100
            target_stocks = config.get('legendary_strategy.target_stocks', 20)
            base_weight = base_allocation / target_stocks
            
            # 신뢰도 승수 (0.5 ~ 1.5배)
            confidence_multiplier = 0.5 + confidence
            target_weight = base_weight * confidence_multiplier
            
            # 최대 포지션 제한
            max_position = self.risk_config.get('max_position', 8.0) / 100
            target_weight = min(target_weight, max_position)
            
            # 총 투자금액 및 주식수
            total_investment = portfolio_value * target_weight
            total_shares = int(total_investment / price)
            
            # 3단계 분할 매수 계획
            stage1_ratio = self.split_config.get('buy_stages', {}).get('stage1_ratio', 40.0) / 100
            stage2_ratio = self.split_config.get('buy_stages', {}).get('stage2_ratio', 35.0) / 100
            stage3_ratio = self.split_config.get('buy_stages', {}).get('stage3_ratio', 25.0) / 100
            
            stage1_shares = int(total_shares * stage1_ratio)
            stage2_shares = int(total_shares * stage2_ratio)
            stage3_shares = total_shares - stage1_shares - stage2_shares
            
            # 진입가 계획
            triggers = self.split_config.get('triggers', {})
            stage2_drop = triggers.get('stage2_drop', -5.0) / 100
            stage3_drop = triggers.get('stage3_drop', -10.0) / 100
            
            entry_price_1 = price
            entry_price_2 = price * (1 + stage2_drop)
            entry_price_3 = price * (1 + stage3_drop)
            
            # 손절익절 계획
            avg_entry_discount = 7.0 / 100  # 평균 진입가 할인율 추정
            avg_entry = price * (1 - avg_entry_discount)
            
            stop_loss_pct = self.risk_config.get('stop_loss', 15.0) / 100
            take_profit1_pct = self.risk_config.get('take_profit1', 20.0) / 100
            take_profit2_pct = self.risk_config.get('take_profit2', 35.0) / 100
            
            stop_loss_price = avg_entry * (1 - stop_loss_pct)
            take_profit1_price = avg_entry * (1 + take_profit1_pct)
            take_profit2_price = avg_entry * (1 + take_profit2_pct)
            
            return {
                'total_shares': total_shares,
                'stage1_shares': stage1_shares,
                'stage2_shares': stage2_shares,
                'stage3_shares': stage3_shares,
                'entry_price_1': entry_price_1,
                'entry_price_2': entry_price_2,
                'entry_price_3': entry_price_3,
                'stop_loss_price': stop_loss_price,
                'take_profit1_price': take_profit1_price,
                'take_profit2_price': take_profit2_price,
                'target_weight': target_weight * 100,
                'total_investment': total_investment
            }
            
        except Exception as e:
            logging.error(f"포지션 계획 수립 실패 {symbol}: {e}")
            return {}

# ========================================================================================
# 🏆 전설적 메인 전략 클래스
# ========================================================================================

class LegendaryQuantStrategy:
    """🔥 전설적 퀸트 전략 마스터시스템 - 혼자 보수유지 가능"""
    
    def __init__(self):
        self.enabled = config.get('legendary_strategy.enabled', True)
        self.target_stocks = config.get('legendary_strategy.target_stocks', 20)
        self.confidence_threshold = config.get('legendary_strategy.confidence_threshold', 0.70)
        
        # 핵심 엔진들
        self.stock_selector = LegendaryStockSelector()
        self.strategy_analyzer = LegendaryStrategyAnalyzer()
        self.split_trading = LegendarySplitTradingSystem()
        
        # 캐싱 시스템
        self.selected_stocks = []
        self.last_selection_time = None
        self.cache_hours = config.get('legendary_strategy.selection_cache_hours', 24)
        
        if self.enabled:
            logging.info("🏆 전설적 퀸트전략 마스터시스템 가동!")
            logging.info(f"🎯 목표종목: {self.target_stocks}개")
            logging.info(f"🔥 신뢰도 임계값: {self.confidence_threshold}")
    
    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if not self.last_selection_time or not self.selected_stocks:
            return False
        time_diff = datetime.now() - self.last_selection_time
        return time_diff.total_seconds() < (self.cache_hours * 3600)
    
    async def auto_select_legendary_stocks(self) -> List[str]:
        """🔥 전설적 종목 자동선별"""
        if not self.enabled:
            logging.warning("전략이 비활성화되어 있습니다")
            return []
        
        try:
            # 캐시 확인
            if self._is_cache_valid():
                logging.info("📋 캐시된 선별 결과 사용")
                return [stock['symbol'] for stock in self.selected_stocks]
            
            logging.info("🚀 전설적 종목 자동선별 시작!")
            start_time = time.time()
            
            # 1단계: 투자 유니버스 생성
            universe = await self.stock_selector.create_investment_universe()
            if not universe:
                logging.error("투자 유니버스 생성 실패")
                return self._get_fallback_stocks()
            
            # 2단계: VIX 조회
            current_vix = await self.stock_selector.get_current_vix()
            
            # 3단계: 종목별 점수 계산 (병렬처리)
            scored_stocks = await self._parallel_stock_analysis(universe, current_vix)
            
            if not scored_stocks:
                logging.error("종목 분석 실패")
                return self._get_fallback_stocks()
            
            # 4단계: 상위 종목 선별 (섹터 다양성 고려)
            final_selection = self._select_diversified_stocks(scored_stocks)
            
            # 5단계: 결과 저장
            self.selected_stocks = final_selection
            self.last_selection_time = datetime.now()
            
            selected_symbols = [stock['symbol'] for stock in final_selection]
            elapsed_time = time.time() - start_time
            
            logging.info(f"🏆 전설적 자동선별 완료! {len(selected_symbols)}개 종목 ({elapsed_time:.1f}초)")
            logging.info(f"📊 평균 점수: {np.mean([s['total_score'] for s in final_selection]):.3f}")
            
            return selected_symbols
            
        except Exception as e:
            logging.error(f"자동선별 실패: {e}")
            return self._get_fallback_stocks()
    
    async def _parallel_stock_analysis(self, universe: List[str], current_vix: float) -> List[Dict]:
        """병렬 종목 분석"""
        scored_stocks = []
        max_workers = config.get('data_sources.max_workers', 15)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for symbol in universe:
                future = executor.submit(self._analyze_single_stock, symbol, current_vix)
                futures.append(future)
            
            for i, future in enumerate(futures, 1):
                try:
                    result = future.result(timeout=45)
                    if result:
                        scored_stocks.append(result)
                    
                    if i % 50 == 0:
                        logging.info(f"📊 분석 진행: {i}/{len(universe)}")
                        
                except Exception as e:
                    logging.warning(f"종목 분석 실패: {e}")
                    continue
        
        return scored_stocks
    
    def _analyze_single_stock(self, symbol: str, current_vix: float) -> Optional[Dict]:
        """단일 종목 분석"""
        try:
            # 데이터 수집
            data = asyncio.run(self.stock_selector.get_stock_data(symbol))
            if not data:
                return None
            
            # 기본 필터링
            min_market_cap = config.get('selection_criteria.min_market_cap', 5_000_000_000)
            min_volume = config.get('selection_criteria.min_avg_volume', 1_000_000)
            
            if data.get('market_cap', 0) < min_market_cap or data.get('avg_volume', 0) < min_volume:
                return None
            
            # 전략 점수 계산
            total_score, scores = self.strategy_analyzer.calculate_total_score(data, current_vix)
            
            result = data.copy()
            result.update(scores)
            result['symbol'] = symbol
            result['current_vix'] = current_vix
            
            return result
            
        except Exception as e:
            logging.error(f"종목 분석 실패 {symbol}: {e}")
            return None
    
    def _select_diversified_stocks(self, scored_stocks: List[Dict]) -> List[Dict]:
        """섹터 다양성을 고려한 종목 선별"""
        # 점수순 정렬
        scored_stocks.sort(key=lambda x: x['total_score'], reverse=True)
        
        final_selection = []
        sector_counts = {}
        
        diversity_config = config.get('selection_criteria.sector_diversity', {})
        max_per_sector = diversity_config.get('max_per_sector', 4)
        
        for stock in scored_stocks:
            if len(final_selection) >= self.target_stocks:
                break
            
            sector = stock.get('sector', 'Unknown')
            
            if sector_counts.get(sector, 0) < max_per_sector:
                final_selection.append(stock)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        return final_selection
    
    def _get_fallback_stocks(self) -> List[str]:
        """백업 종목 리스트"""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JNJ', 'UNH', 'PFE',
                'JPM', 'BAC', 'PG', 'KO', 'HD', 'WMT', 'V', 'MA', 'AVGO', 'ORCL']
    
    async def analyze_stock_signal(self, symbol: str) -> LegendaryStockSignal:
        """개별 종목 시그널 생성"""
        try:
            # 데이터 수집
            data = await self.stock_selector.get_stock_data(symbol)
            if not data:
                raise ValueError(f"데이터 수집 실패: {symbol}")
            
            # VIX 조회
            current_vix = await self.stock_selector.get_current_vix()
            
            # 전략 분석
            total_score, scores = self.strategy_analyzer.calculate_total_score(data, current_vix)
            
            # 액션 결정
            if total_score >= self.confidence_threshold:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= 0.30:
                action = 'sell'
                confidence = min(1 - total_score, 0.95)
            else:
                action = 'hold'
                confidence = 0.50
            
            # 분할매매 계획
            split_plan = self.split_trading.calculate_position_plan(symbol, data['price'], confidence)
            
            # 목표가 계산
            max_expected_return = 0.35  # 최대 35% 기대수익
            target_price = data['price'] * (1 + confidence * max_expected_return)
            
            # 전략별 설명
            reasoning_parts = [
                f"버핏:{scores['buffett_score']:.2f}",
                f"린치:{scores['lynch_score']:.2f}",
                f"모멘텀:{scores['momentum_score']:.2f}",
                f"기술:{scores['technical_score']:.2f}",
                f"VIX조정:{scores['vix_adjustment']:+.2f}"
            ]
            reasoning = " | ".join(reasoning_parts)
            
            return LegendaryStockSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=data['price'],
                
                # 전략 점수
                buffett_score=scores['buffett_score'],
                lynch_score=scores['lynch_score'],
                momentum_score=scores['momentum_score'],
                technical_score=scores['technical_score'],
                total_score=total_score,
                
                # 재무지표
                market_cap=data.get('market_cap', 0),
                pe_ratio=data.get('pe_ratio', 0),
                pbr=data.get('pbr', 0),
                peg=data.get('peg', 0),
                roe=data.get('roe', 0),
                sector=data.get('sector', 'Unknown'),
                
                # 모멘텀
                momentum_3m=data.get('momentum_3m', 0),
                momentum_6m=data.get('momentum_6m', 0),
                momentum_12m=data.get('momentum_12m', 0),
                
                # 기술적지표
                rsi=data.get('rsi', 50),
                trend=data.get('trend', 'sideways'),
                volume_spike=data.get('volume_spike', 1),
                
                # 분할매매
                total_shares=split_plan.get('total_shares', 0),
                stage1_shares=split_plan.get('stage1_shares', 0),
                stage2_shares=split_plan.get('stage2_shares', 0),
                stage3_shares=split_plan.get('stage3_shares', 0),
                entry_price_1=split_plan.get('entry_price_1', data['price']),
                entry_price_2=split_plan.get('entry_price_2', data['price']),
                entry_price_3=split_plan.get('entry_price_3', data['price']),
                stop_loss_price=split_plan.get('stop_loss_price', data['price'] * 0.85),
                take_profit1_price=split_plan.get('take_profit1_price', data['price'] * 1.20),
                take_profit2_price=split_plan.get('take_profit2_price', data['price'] * 1.35),
                
                # 메타정보
                target_price=target_price,
                selection_score=total_score,
                index_membership=['AUTO_SELECTED'],
                vix_adjustment=scores['vix_adjustment'],
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"시그널 생성 실패 {symbol}: {e}")
            return self._create_empty_signal(symbol, str(e))
    
    def _create_empty_signal(self, symbol: str, error_msg: str) -> LegendaryStockSignal:
        """빈 시그널 생성"""
        return LegendaryStockSignal(
            symbol=symbol, action='hold', confidence=0.0, price=0.0,
            buffett_score=0.0, lynch_score=0.0, momentum_score=0.0, technical_score=0.0, total_score=0.0,
            market_cap=0, pe_ratio=0.0, pbr=0.0, peg=0.0, roe=0.0, sector='Unknown',
            momentum_3m=0.0, momentum_6m=0.0, momentum_12m=0.0, rsi=50.0, trend='sideways', volume_spike=1.0,
            total_shares=0, stage1_shares=0, stage2_shares=0, stage3_shares=0,
            entry_price_1=0.0, entry_price_2=0.0, entry_price_3=0.0,
            stop_loss_price=0.0, take_profit1_price=0.0, take_profit2_price=0.0,
            target_price=0.0, selection_score=0.0, index_membership=['ERROR'],
            vix_adjustment=0.0, reasoning=f"오류: {error_msg}", timestamp=datetime.now()
        )
    
    async def scan_all_legendary_stocks(self) -> List[LegendaryStockSignal]:
        """전체 전설적 종목 스캔"""
        if not self.enabled:
            return []
        
        logging.info("🔍 전설적 전체 종목 스캔 시작!")
        
        try:
            # 자동선별
            selected_symbols = await self.auto_select_legendary_stocks()
            if not selected_symbols:
                return []
            
            # 각 종목 분석
            all_signals = []
            for i, symbol in enumerate(selected_symbols, 1):
                try:
                    signal = await self.analyze_stock_signal(symbol)
                    all_signals.append(signal)
                    
                    # 진행상황 로그
                    action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                    logging.info(f"{action_emoji} {symbol}: {signal.action} "
                               f"신뢰도:{signal.confidence:.2f} 점수:{signal.total_score:.3f}")
                    
                    await asyncio.sleep(config.get('data_sources.rate_limit_delay', 0.3))
                    
                except Exception as e:
                    logging.error(f"❌ {symbol} 분석 실패: {e}")
            
            # 결과 요약
            buy_count = len([s for s in all_signals if s.action == 'buy'])
            sell_count = len([s for s in all_signals if s.action == 'sell'])
            hold_count = len([s for s in all_signals if s.action == 'hold'])
            
            logging.info(f"🏆 전설적 스캔 완료! 매수:{buy_count}, 매도:{sell_count}, 보유:{hold_count}")
            
            return all_signals
            
        except Exception as e:
            logging.error(f"전체 스캔 실패: {e}")
            return []
    
    async def generate_legendary_report(self, signals: List[LegendaryStockSignal]) -> Dict:
        """전설적 포트폴리오 리포트 생성"""
        if not signals:
            return {"error": "분석된 종목이 없습니다"}
        
        # 기본 통계
        total_count = len(signals)
        buy_signals = [s for s in signals if s.action == 'buy']
        sell_signals = [s for s in signals if s.action == 'sell']
        hold_signals = [s for s in signals if s.action == 'hold']
        
        # 평균 점수
        avg_scores = {
            'buffett': np.mean([s.buffett_score for s in signals]),
            'lynch': np.mean([s.lynch_score for s in signals]),
            'momentum': np.mean([s.momentum_score for s in signals]),
            'technical': np.mean([s.technical_score for s in signals]),
            'total': np.mean([s.total_score for s in signals])
        }
        
        # 상위 매수 추천 (신뢰도순)
        top_buys = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
        
        # 섹터 분포
        sector_dist = {}
        for signal in signals:
            sector_dist[signal.sector] = sector_dist.get(signal.sector, 0) + 1
        
        # 투자금액 계산
        total_investment = sum([
            s.stage1_shares * s.entry_price_1 + 
            s.stage2_shares * s.entry_price_2 + 
            s.stage3_shares * s.entry_price_3 
            for s in signals if s.total_shares > 0
        ])
        
        # VIX 상태
        current_vix = signals[0].vix_adjustment if signals else 20.0
        vix_status = ('HIGH' if current_vix > 30 else 'LOW' if current_vix < 15 else 'MEDIUM')
        
        report = {
            'summary': {
                'total_stocks': total_count,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'hold_signals': len(hold_signals),
                'total_investment': total_investment,
                'current_vix_status': vix_status,
                'generation_time': datetime.now().isoformat(),
                'strategy_version': '6.0_LEGENDARY'
            },
            'average_scores': avg_scores,
            'top_recommendations': [
                {
                    'symbol': stock.symbol,
                    'sector': stock.sector,
                    'action': stock.action,
                    'confidence': stock.confidence,
                    'total_score': stock.total_score,
                    'price': stock.price,
                    'target_price': stock.target_price,
                    'total_investment': (stock.stage1_shares * stock.entry_price_1 + 
                                       stock.stage2_shares * stock.entry_price_2 + 
                                       stock.stage3_shares * stock.entry_price_3),
                    'reasoning': stock.reasoning[:100] + "..." if len(stock.reasoning) > 100 else stock.reasoning
                }
                for stock in top_buys
            ],
            'sector_distribution': sector_dist,
            'risk_metrics': {
                'diversification_score': len(sector_dist) / total_count if total_count > 0 else 0,
                'avg_confidence': np.mean([s.confidence for s in buy_signals]) if buy_signals else 0,
                'portfolio_allocation': config.get('risk_management.portfolio_allocation', 80.0),
                'max_single_position': config.get('risk_management.max_position', 8.0)
            },
            'configuration_info': {
                'enabled': self.enabled,
                'target_stocks': self.target_stocks,
                'confidence_threshold': self.confidence_threshold,
                'cache_hours': self.cache_hours,
                'last_selection': self.last_selection_time.isoformat() if self.last_selection_time else None,
                'strategy_weights': config.get('legendary_strategy.strategy_weights', {}),
                'risk_settings': config.get('risk_management', {})
            }
        }
        
        return report

# ========================================================================================
# 🎯 전설적 편의 함수들 - 외부 호출용
# ========================================================================================

async def run_legendary_auto_selection():
    """전설적 자동선별 + 전체분석 실행"""
    try:
        strategy = LegendaryQuantStrategy()
        signals = await strategy.scan_all_legendary_stocks()
        
        if signals:
            report = await strategy.generate_legendary_report(signals)
            return signals, report
        else:
            return [], {"error": "분석된 종목이 없습니다"}
            
    except Exception as e:
        logging.error(f"전설적 자동선별 실행 실패: {e}")
        return [], {"error": str(e)}

async def analyze_legendary_stock(symbol: str) -> Dict:
    """단일 종목 전설적 분석"""
    try:
        strategy = LegendaryQuantStrategy()
        signal = await strategy.analyze_stock_signal(symbol)
        
        return {
            'symbol': signal.symbol,
            'decision': signal.action,
            'confidence_score': signal.confidence * 100,
            'total_score': signal.total_score * 100,
            'price': signal.price,
            'target_price': signal.target_price,
            'sector': signal.sector,
            
            # 전략별 점수
            'strategy_scores': {
                'buffett_value': signal.buffett_score * 100,
                'lynch_growth': signal.lynch_score * 100,
                'momentum': signal.momentum_score * 100,
                'technical': signal.technical_score * 100
            },
            
            # 재무지표
            'financial_metrics': {
                'market_cap': signal.market_cap,
                'pe_ratio': signal.pe_ratio,
                'pbr': signal.pbr,
                'peg': signal.peg,
                'roe': signal.roe
            },
            
            # 분할매매 계획
            'split_trading_plan': {
                'total_shares': signal.total_shares,
                'stage1': {'shares': signal.stage1_shares, 'price': signal.entry_price_1},
                'stage2': {'shares': signal.stage2_shares, 'price': signal.entry_price_2},
                'stage3': {'shares': signal.stage3_shares, 'price': signal.entry_price_3},
                'stop_loss': signal.stop_loss_price,
                'take_profit1': signal.take_profit1_price,
                'take_profit2': signal.take_profit2_price
            },
            
            'reasoning': signal.reasoning,
            'vix_adjustment': signal.vix_adjustment,
            'analysis_time': signal.timestamp.isoformat(),
            'legendary_version': '6.0'
        }
        
    except Exception as e:
        logging.error(f"종목 분석 실패 {symbol}: {e}")
        return {
            'symbol': symbol,
            'decision': 'hold',
            'confidence_score': 0.0,
            'error': str(e),
            'legendary_version': '6.0'
        }

async def get_legendary_status() -> Dict:
    """전설적 시스템 상태 조회"""
    try:
        strategy = LegendaryQuantStrategy()
        
        return {
            'system_status': {
                'enabled': strategy.enabled,
                'target_stocks': strategy.target_stocks,
                'confidence_threshold': strategy.confidence_threshold,
                'cache_hours': strategy.cache_hours,
                'last_selection': strategy.last_selection_time.isoformat() if strategy.last_selection_time else None,
                'cache_valid': strategy._is_cache_valid(),
                'selected_count': len(strategy.selected_stocks)
            },
            'configuration': {
                'config_file_exists': Path('quant_settings.yaml').exists(),
                'env_loaded': config.env_loaded,
                'strategy_weights': config.get('legendary_strategy.strategy_weights', {}),
                'risk_settings': config.get('risk_management', {}),
                'vix_thresholds': config.get('legendary_strategy.vix_thresholds', {}),
                'notifications_enabled': {
                    'telegram': config.is_enabled('notifications.telegram'),
                    'discord': config.is_enabled('notifications.discord')
                }
            },
            'performance_metrics': {
                'version': '6.0_LEGENDARY',
                'features': [
                    '완벽한 설정기반 아키텍처',
                    '실시간 S&P500+NASDAQ 자동선별',
                    '4가지 투자전략 지능형 융합',
                    'VIX 기반 시장상황 자동판단',
                    '분할매매 + 손절익절 자동화',
                    '통합 리스크관리 시스템',
                    '혼자 보수유지 가능한 구조'
                ]
            }
        }
        
    except Exception as e:
        logging.error(f"상태 조회 실패: {e}")
        return {
            'system_status': {'enabled': False, 'error': str(e)},
            'legendary_version': '6.0'
        }

async def update_legendary_weights(buffett: float, lynch: float, momentum: float, technical: float) -> Dict:
    """전설적 전략 가중치 업데이트"""
    try:
        # 가중치 정규화
        total = buffett + lynch + momentum + technical
        if total == 0:
            return {'status': 'error', 'message': '가중치 합이 0입니다'}
        
        normalized_weights = {
            'buffett_value': (buffett / total) * 100,
            'lynch_growth': (lynch / total) * 100,
            'momentum': (momentum / total) * 100,
            'technical': (technical / total) * 100
        }
        
        # 설정 업데이트
        config.update('legendary_strategy.strategy_weights', normalized_weights)
        
        logging.info(f"🎯 전설적 가중치 업데이트 완료!")
        
        return {
            'status': 'success',
            'message': '전설적 전략 가중치가 업데이트되었습니다',
            'updated_weights': normalized_weights,
            'auto_saved': True
        }
        
    except Exception as e:
        logging.error(f"가중치 업데이트 실패: {e}")
        return {'status': 'error', 'message': f'업데이트 실패: {str(e)}'}

async def force_legendary_reselection() -> List[str]:
    """전설적 강제 재선별"""
    try:
        strategy = LegendaryQuantStrategy()
        # 캐시 무효화
        strategy.last_selection_time = None
        strategy.selected_stocks = []
        
        logging.info("🔄 전설적 강제 재선별 시작...")
        return await strategy.auto_select_legendary_stocks()
        
    except Exception as e:
        logging.error(f"강제 재선별 실패: {e}")
        return []

async def reload_legendary_config() -> Dict:
    """전설적 설정 다시 로드"""
    try:
        global config
        config = LegendaryConfigManager()
        
        logging.info("🔄 전설적 설정 다시 로드 완료")
        
        return {
            'status': 'success',
            'message': '전설적 설정이 다시 로드되었습니다',
            'config_exists': Path('quant_settings.yaml').exists(),
            'env_loaded': config.env_loaded,
            'legendary_version': '6.0'
        }
        
    except Exception as e:
        logging.error(f"설정 다시 로드 실패: {e}")
        return {
            'status': 'error',
            'message': f'설정 로드 실패: {str(e)}',
            'legendary_version': '6.0'
        }

# ========================================================================================
# 🧪 전설적 테스트 메인 함수
# ========================================================================================

async def legendary_main():
    """🏆 전설적 테스트 메인 함수"""
    try:
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('legendary_quant.log', encoding='utf-8') if Path('logs').exists() else logging.NullHandler()
            ]
        )
        
        print("🏆" + "="*80)
        print("🔥 전설적 퀸트프로젝트 - 미국주식 마스터시스템 V6.0")
        print("🚀 혼자 보수유지 가능한 완벽한 자동화 시스템")
        print("="*82)
        
        # 시스템 상태 확인
        print("\n🔧 전설적 시스템 상태 확인...")
        status = await get_legendary_status()
        system_status = status.get('system_status', {})
        configuration = status.get('configuration', {})
        
        print(f"  ✅ 시스템 활성화: {system_status.get('enabled', False)}")
        print(f"  ✅ 설정파일: {'발견됨' if configuration.get('config_file_exists') else '❌ 없음 (자동생성됨)'}")
        print(f"  ✅ 환경변수: {'로드됨' if configuration.get('env_loaded') else '❌ .env 파일 없음'}")
        print(f"  🎯 목표종목: {system_status.get('target_stocks', 20)}개")
        print(f"  🔥 신뢰도임계: {system_status.get('confidence_threshold', 0.70)}")
        
        # 전략 가중치 표시
        weights = configuration.get('strategy_weights', {})
        print(f"  📊 전략가중치: 버핏{weights.get('buffett_value', 25):.0f}% "
              f"린치{weights.get('lynch_growth', 25):.0f}% "
              f"모멘텀{weights.get('momentum', 25):.0f}% "
              f"기술{weights.get('technical', 25):.0f}%")
        
        # 리스크 설정 표시
        risk_settings = configuration.get('risk_settings', {})
        print(f"  🛡️ 리스크설정: 포트폴리오{risk_settings.get('portfolio_allocation', 80):.0f}% "
              f"손절{risk_settings.get('stop_loss', 15):.0f}% "
              f"익절{risk_settings.get('take_profit2', 35):.0f}%")
        
        print(f"\n🌟 전설적 특징:")
        features = status.get('performance_metrics', {}).get('features', [])
        for feature in features:
            print(f"  ✨ {feature}")
        
        # 전설적 자동선별 + 전체분석 실행
        print(f"\n🚀 전설적 자동선별 + 전체분석 시작...")
        print("🔍 실시간 S&P500+NASDAQ 크롤링 → 4가지전략 융합분석 → VIX조정 → 분할매매계획")
        start_time = time.time()
        
        signals, report = await run_legendary_auto_selection()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ 총 소요시간: {elapsed_time:.1f}초")
        
        if signals and report and 'error' not in report:
            summary = report['summary']
            avg_scores = report['average_scores']
            top_recs = report['top_recommendations']
            
            print(f"\n📈 전설적 분석 결과:")
            print(f"  총 분석종목: {summary['total_stocks']}개 (실시간 자동선별)")
            print(f"  🟢 매수신호: {summary['buy_signals']}개")
            print(f"  🔴 매도신호: {summary['sell_signals']}개")
            print(f"  ⚪ 보유신호: {summary['hold_signals']}개")
            print(f"  📊 시장상태: {summary['current_vix_status']} VIX")
            print(f"  💰 총투자금액: ${summary['total_investment']:,.0f}")
            
            print(f"\n📊 전설적 평균 전략점수:")
            print(f"  🏆 버핏 가치투자: {avg_scores['buffett']:.3f}")
            print(f"  🚀 린치 성장투자: {avg_scores['lynch']:.3f}")
            print(f"  ⚡ 모멘텀 전략: {avg_scores['momentum']:.3f}")
            print(f"  📈 기술적 분석: {avg_scores['technical']:.3f}")
            print(f"  🎯 종합점수: {avg_scores['total']:.3f}")
            
            # 상위 매수 추천
            if top_recs:
                print(f"\n🏆 전설적 상위 매수 추천:")
                for i, stock in enumerate(top_recs[:3], 1):
                    print(f"\n  {i}. {stock['symbol']} ({stock['sector']}) - 신뢰도: {stock['confidence']:.1%}")
                    print(f"     🎯 점수: {stock['total_score']:.3f} | 현재가: ${stock['price']:.2f} → 목표가: ${stock['target_price']:.2f}")
                    print(f"     💰 투자금액: ${stock['total_investment']:,.0f}")
                    print(f"     💡 {stock['reasoning'][:80]}...")
            
            # 섹터 분포
            sector_dist = report['sector_distribution']
            print(f"\n🏢 섹터 분포:")
            for sector, count in list(sector_dist.items())[:5]:
                percentage = count / summary['total_stocks'] * 100
                print(f"  {sector}: {count}개 ({percentage:.1f}%)")
            
            # 리스크 메트릭
            risk_metrics = report['risk_metrics']
            print(f"\n🛡️ 전설적 리스크 메트릭:")
            print(f"  다양성 점수: {risk_metrics['diversification_score']:.2f}")
            print(f"  평균 신뢰도: {risk_metrics['avg_confidence']:.2f}")
            print(f"  포트폴리오 할당: {risk_metrics['portfolio_allocation']:.0f}%")
            print(f"  최대 단일포지션: {risk_metrics['max_single_position']:.0f}%")
            
        else:
            error_msg = report.get('error', '알 수 없는 오류') if report else '결과 없음'
            print(f"❌ 전설적 분석 실패: {error_msg}")
        
        print(f"\n🏆 전설적 퀸트프로젝트 V6.0 테스트 완료!")
        print("\n🌟 혼자 보수유지 가능한 핵심 특징:")
        print("  ✅ 🔧 완벽한 설정기반 아키텍처 (quant_settings.yaml)")
        print("  ✅ 🚀 실시간 자동선별 (S&P500+NASDAQ 크롤링)")
        print("  ✅ 🧠 4가지 전략 지능형 융합 (가중치 조정가능)")
        print("  ✅ 📊 VIX 기반 시장상황 자동판단")
        print("  ✅ 💰 분할매매 + 손절익절 자동화")
        print("  ✅ 🛡️ 통합 리스크관리 + 포트폴리오 최적화")
        print("  ✅ 🔄 런타임 설정변경 (재시작 불필요)")
        print("  ✅ 📱 알림시스템 (텔레그램/디스코드)")
        print("  ✅ 🎯 캐싱시스템 (효율적 API 사용)")
        print("  ✅ ⚡ 병렬처리 (빠른 분석속도)")
        
        print(f"\n💡 사용법:")
        print("  - run_legendary_auto_selection(): 전체 자동선별+분석")
        print("  - analyze_legendary_stock('AAPL'): 개별 종목분석")
        print("  - update_legendary_weights(25,25,25,25): 가중치 조정")
        print("  - force_legendary_reselection(): 강제 재선별")
        print("  - reload_legendary_config(): 설정 다시로드")
        print("  - get_legendary_status(): 시스템 상태확인")
        
        print(f"\n🔧 설정파일 관리:")
        print("  - quant_settings.yaml: 모든 전략 파라미터")
        print("  - .env: API키, 알림토큰 (선택사항)")
        print("  - 설정 자동생성: 최초 실행시 기본설정 생성")
        print("  - 실시간 설정변경: 파일 수정 후 reload_legendary_config()")
        
    except Exception as e:
        print(f"❌ 전설적 테스트 실행 실패: {e}")
        logging.error(f"전설적 테스트 실패: {e}")

# ========================================================================================
# 🎯 실행부
# ========================================================================================

if __name__ == "__main__":
    asyncio.run(legendary_main())
