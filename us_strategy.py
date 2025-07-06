#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 전설적 퀸트프로젝트 - 통합 완성판 V6.2 (오류 수정)
===============================================

🌟 완전 통합 특징:
1. 🔥 4가지 투자전략 지능형 융합 (버핏+린치+모멘텀+기술)
2. 🚀 실시간 S&P500+NASDAQ 자동선별 엔진
3. 💎 VIX 기반 시장상황 자동판단 AI
4. 🎯 월 5-7% 달성형 스윙 + 분할매매 통합
5. ⚡ IBKR 실거래 완전 연동 + 자동 손익절
6. 🛡️ 통합 리스크관리 + 포트폴리오 최적화
7. 🧠 혼자 보수유지 가능한 완벽한 아키텍처

Author: 전설적퀸트팀
Version: 6.2.1 (오류 수정)
"""

import asyncio
import logging
import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
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
import sqlite3
from threading import Thread

# 타입 힌트를 안전하게 처리
try:
    from typing import Dict, List, Optional, Tuple, Any, Union
except ImportError:
    # 구버전 Python 대응
    Dict = dict
    List = list
    Optional = lambda x: x
    Tuple = tuple
    Any = object
    Union = object

# IBKR 연동
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
    logging.info("✅ IBKR 모듈 로드 성공")
except ImportError:
    IBKR_AVAILABLE = False
    logging.warning("⚠️ IBKR 모듈 없음 (pip install ib_insync 필요)")

warnings.filterwarnings('ignore')

# ========================================================================================
# 🔧 통합 설정관리자 - 완전 자동화
# ========================================================================================

class LegendaryConfig:
    """🔥 전설적 통합 설정관리자"""
    
    def __init__(self, config_path: str = "legendary_unified_settings.yaml"):
        self.config_path = config_path
        self.config = {}
        self.env_loaded = False
        self._initialize_config()
    
    def _initialize_config(self):
        """설정 초기화"""
        try:
            if Path('.env').exists():
                load_dotenv()
                self.env_loaded = True
                
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self._create_default_config()
                self._save_config()
            
            self._substitute_env_vars()
            logging.info("🔥 설정관리자 초기화 완료!")
            
        except Exception as e:
            logging.error(f"❌ 설정 초기화 실패: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """기본 설정 생성"""
        self.config = {
            # 🎯 통합 전략 설정
            'strategy': {
                'enabled': True,
                'mode': 'swing',  # 'classic', 'swing', 'hybrid'
                'target_stocks': {'classic': 20, 'swing': 8},
                'monthly_target': {'min': 5.0, 'max': 7.0},
                'weights': {
                    'buffett': 25.0, 'lynch': 25.0, 
                    'momentum': 25.0, 'technical': 25.0
                },
                'vix_thresholds': {'low': 15.0, 'high': 30.0}
            },
            
            # 💰 매매 설정
            'trading': {
                'classic': {
                    'stages': [40.0, 35.0, 25.0],  # 분할매매 비율
                    'triggers': [-5.0, -10.0],     # 추가매수 조건
                    'take_profit': [20.0, 35.0]    # 익절 조건
                },
                'swing': {
                    'take_profit': [6.0, 12.0],    # 2단계 익절
                    'profit_ratios': [60.0, 40.0], # 매도 비율
                    'stop_loss': 8.0               # 손절
                }
            },
            
            # 🛡️ 리스크 관리
            'risk': {
                'portfolio_allocation': 80.0,
                'max_position': 8.0,
                'max_sector': 25.0,
                'stop_loss': {'classic': 15.0, 'swing': 8.0},
                'trailing_stop': True,
                'daily_loss_limit': 1.0,
                'monthly_loss_limit': 3.0
            },
            
            # 📊 종목 선별
            'selection': {
                'min_market_cap': 5_000_000_000,
                'min_volume': 1_000_000,
                'excluded_symbols': ['SPXL', 'TQQQ'],
                'refresh_hours': 24,
                'sp500_quota': 60.0,
                'nasdaq_quota': 40.0
            },
            
            # 🏦 IBKR 설정
            'ibkr': {
                'enabled': True,
                'host': '127.0.0.1',
                'port': 7497,
                'client_id': 1,
                'paper_trading': True,
                'account_id': '${IBKR_ACCOUNT:-}',
                'max_daily_trades': 20,
                'order_type': 'MKT'
            },
            
            # 🤖 자동화
            'automation': {
                'monitoring_interval': 15,
                'weekend_shutdown': True,
                'holiday_shutdown': True,
                'morning_scan': '09:00',
                'evening_report': '16:00'
            },
            
            # 📱 알림
            'notifications': {
                'telegram': {
                    'enabled': True,
                    'bot_token': '${TELEGRAM_BOT_TOKEN:-}',
                    'chat_id': '${TELEGRAM_CHAT_ID:-}'
                }
            },
            
            # 📊 성과 추적
            'performance': {
                'database_file': 'legendary_performance.db',
                'benchmarks': ['SPY', 'QQQ'],
                'detailed_metrics': True
            }
        }
    
    def _substitute_env_vars(self):
        """환경변수 치환"""
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
        """설정 저장"""
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
        """설정값 업데이트"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self._save_config()
        logging.info(f"설정 업데이트: {key_path} = {value}")

# 전역 설정 인스턴스
config = LegendaryConfig()

# ========================================================================================
# 📊 데이터 클래스
# ========================================================================================

@dataclass
class StockSignal:
    """주식 시그널"""
    symbol: str
    action: str  # buy/sell/hold
    confidence: float
    price: float
    mode: str
    scores: Dict[str, float]
    financials: Dict[str, float]
    target_price: float
    stop_loss: float
    reasoning: str
    timestamp: datetime

@dataclass 
class Position:
    """포지션"""
    symbol: str
    quantity: int
    avg_cost: float
    entry_date: datetime
    mode: str
    stage: int = 1
    tp_executed: List[bool] = field(default_factory=lambda: [False, False, False])
    highest_price: float = 0.0
    
    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.avg_cost

    def profit_percent(self, current_price: float) -> float:
        """수익률 계산"""
        return ((current_price - self.avg_cost) / self.avg_cost) * 100

# ========================================================================================
# 🚀 주식 선별 엔진
# ========================================================================================

class StockSelector:
    """실시간 주식 선별 엔진"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.session.timeout = 30
        self.cache = {'sp500': [], 'nasdaq': [], 'last_update': None}
    
    async def get_current_vix(self) -> float:
        """VIX 조회"""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            return float(hist['Close'].iloc[-1]) if not hist.empty else 20.0
        except:
            return 20.0
    
    async def collect_sp500_symbols(self) -> List[str]:
        """S&P 500 심볼 수집"""
        try:
            if self._is_cache_valid():
                return self.cache['sp500']
            
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            symbols = tables[0]['Symbol'].tolist()
            cleaned = [str(s).replace('.', '-') for s in symbols]
            
            self.cache['sp500'] = cleaned
            self.cache['last_update'] = datetime.now()
            
            logging.info(f"✅ S&P 500: {len(cleaned)}개 수집")
            return cleaned
        except Exception as e:
            logging.error(f"S&P 500 수집 실패: {e}")
            return self._get_backup_sp500()
    
    async def collect_nasdaq_symbols(self) -> List[str]:
        """NASDAQ 100 심볼 수집"""
        try:
            if self._is_cache_valid():
                return self.cache['nasdaq']
            
            tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            symbols = []
            for table in tables:
                if 'Symbol' in table.columns:
                    symbols = table['Symbol'].dropna().tolist()
                    break
            
            self.cache['nasdaq'] = symbols
            return symbols
        except Exception as e:
            logging.error(f"NASDAQ 수집 실패: {e}")
            return self._get_backup_nasdaq()
    
    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if not self.cache['last_update']:
            return False
        hours = config.get('selection.refresh_hours', 24)
        return (datetime.now() - self.cache['last_update']).seconds < hours * 3600
    
    def _get_backup_sp500(self) -> List[str]:
        """S&P 500 백업"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'ABBV', 'BAC'
        ]
    
    def _get_backup_nasdaq(self) -> List[str]:
        """NASDAQ 백업"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN'
        ]
    
    async def create_universe(self) -> List[str]:
        """투자 유니버스 생성"""
        try:
            sp500, nasdaq = await asyncio.gather(
                self.collect_sp500_symbols(),
                self.collect_nasdaq_symbols()
            )
            
            universe = list(set(sp500 + nasdaq))
            excluded = config.get('selection.excluded_symbols', [])
            universe = [s for s in universe if s not in excluded]
            
            logging.info(f"🌌 투자 유니버스: {len(universe)}개 종목")
            return universe
        except Exception as e:
            logging.error(f"유니버스 생성 실패: {e}")
            return self._get_backup_sp500() + self._get_backup_nasdaq()
    
    async def get_stock_data(self, symbol: str) -> Dict:
        """종목 데이터 수집"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty:
                return {}
            
            current_price = float(hist['Close'].iloc[-1])
            
            # 기본 데이터
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
                'beta': info.get('beta', 1.0) or 1.0,
                'profit_margins': (info.get('profitMargins', 0) or 0) * 100
            }
            
            # PEG 계산
            if data['pe_ratio'] > 0 and data['eps_growth'] > 0:
                data['peg'] = data['pe_ratio'] / data['eps_growth']
            else:
                data['peg'] = 999
            
            # 모멘텀 지표
            if len(hist) >= 252:
                data['momentum_3m'] = ((current_price / float(hist['Close'].iloc[-63])) - 1) * 100
                data['momentum_6m'] = ((current_price / float(hist['Close'].iloc[-126])) - 1) * 100
                data['momentum_12m'] = ((current_price / float(hist['Close'].iloc[-252])) - 1) * 100
            else:
                data['momentum_3m'] = data['momentum_6m'] = data['momentum_12m'] = 0
            
            # 기술적 지표
            if len(hist) >= 50:
                # RSI
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                data['rsi'] = float(100 - (100 / (1 + rs)).iloc[-1])
                
                # 추세
                ma20 = float(hist['Close'].rolling(20).mean().iloc[-1])
                ma50 = float(hist['Close'].rolling(50).mean().iloc[-1])
                
                if current_price > ma50 > ma20:
                    data['trend'] = 'strong_uptrend'
                elif current_price > ma50:
                    data['trend'] = 'uptrend'
                else:
                    data['trend'] = 'downtrend'
                
                # 거래량
                avg_vol = float(hist['Volume'].rolling(20).mean().iloc[-1])
                current_vol = float(hist['Volume'].iloc[-1])
                data['volume_spike'] = current_vol / avg_vol if avg_vol > 0 else 1
                
                # 변동성
                returns = hist['Close'].pct_change().dropna()
                data['volatility'] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
            else:
                data.update({
                    'rsi': 50, 'trend': 'sideways', 'volume_spike': 1, 'volatility': 25
                })
            
            await asyncio.sleep(0.3)
            return data
            
        except Exception as e:
            logging.error(f"데이터 수집 실패 {symbol}: {e}")
            return {}

# ========================================================================================
# 🧠 4가지 투자전략 분석 엔진
# ========================================================================================

class StrategyAnalyzer:
    """4가지 투자전략 분석 엔진"""
    
    def calculate_buffett_score(self, data: Dict) -> float:
        """버핏 가치투자 점수"""
        score = 0.0
        
        # PBR (30%)
        pbr = data.get('pbr', 999)
        if 0 < pbr <= 1.0: score += 0.30
        elif pbr <= 1.5: score += 0.25
        elif pbr <= 2.0: score += 0.20
        elif pbr <= 3.0: score += 0.10
        
        # ROE (25%)
        roe = data.get('roe', 0)
        if roe >= 20: score += 0.25
        elif roe >= 15: score += 0.20
        elif roe >= 10: score += 0.15
        elif roe >= 5: score += 0.10
        
        # 부채비율 (20%)
        debt_ratio = data.get('debt_to_equity', 999) / 100
        if debt_ratio <= 0.3: score += 0.20
        elif debt_ratio <= 0.5: score += 0.15
        elif debt_ratio <= 0.7: score += 0.10
        
        # PE (15%)
        pe = data.get('pe_ratio', 999)
        if 5 <= pe <= 15: score += 0.15
        elif pe <= 20: score += 0.10
        elif pe <= 25: score += 0.05
        
        # 이익률 (10%)
        margins = data.get('profit_margins', 0)
        if margins >= 15: score += 0.10
        elif margins >= 10: score += 0.07
        elif margins >= 5: score += 0.05
        
        return min(score, 1.0)
    
    def calculate_lynch_score(self, data: Dict) -> float:
        """린치 성장투자 점수"""
        score = 0.0
        
        # PEG (40%)
        peg = data.get('peg', 999)
        if 0 < peg <= 0.5: score += 0.40
        elif peg <= 1.0: score += 0.35
        elif peg <= 1.5: score += 0.25
        elif peg <= 2.0: score += 0.15
        
        # EPS 성장 (30%)
        eps_growth = data.get('eps_growth', 0)
        if eps_growth >= 25: score += 0.30
        elif eps_growth >= 20: score += 0.25
        elif eps_growth >= 15: score += 0.20
        elif eps_growth >= 10: score += 0.15
        
        # 매출 성장 (20%)
        rev_growth = data.get('revenue_growth', 0)
        if rev_growth >= 20: score += 0.20
        elif rev_growth >= 15: score += 0.15
        elif rev_growth >= 10: score += 0.10
        
        # ROE (10%)
        roe = data.get('roe', 0)
        if roe >= 15: score += 0.10
        elif roe >= 10: score += 0.07
        
        return min(score, 1.0)
    
    def calculate_momentum_score(self, data: Dict) -> float:
        """모멘텀 전략 점수"""
        score = 0.0
        
        # 3개월 모멘텀 (30%)
        mom_3m = data.get('momentum_3m', 0)
        if mom_3m >= 20: score += 0.30
        elif mom_3m >= 15: score += 0.25
        elif mom_3m >= 10: score += 0.20
        elif mom_3m >= 5: score += 0.15
        
        # 6개월 모멘텀 (25%)
        mom_6m = data.get('momentum_6m', 0)
        if mom_6m >= 30: score += 0.25
        elif mom_6m >= 20: score += 0.20
        elif mom_6m >= 15: score += 0.15
        elif mom_6m >= 10: score += 0.10
        
        # 12개월 모멘텀 (25%)
        mom_12m = data.get('momentum_12m', 0)
        if mom_12m >= 50: score += 0.25
        elif mom_12m >= 30: score += 0.20
        elif mom_12m >= 20: score += 0.15
        elif mom_12m >= 10: score += 0.10
        
        # 거래량 (20%)
        vol_spike = data.get('volume_spike', 1)
        if vol_spike >= 3.0: score += 0.20
        elif vol_spike >= 2.0: score += 0.15
        elif vol_spike >= 1.5: score += 0.10
        
        return min(score, 1.0)
    
    def calculate_technical_score(self, data: Dict) -> float:
        """기술적 분석 점수"""
        score = 0.0
        
        # RSI (30%)
        rsi = data.get('rsi', 50)
        if 30 <= rsi <= 70: score += 0.30
        elif 25 <= rsi < 30: score += 0.25
        elif 70 < rsi <= 75: score += 0.20
        
        # 추세 (35%)
        trend = data.get('trend', 'sideways')
        if trend == 'strong_uptrend': score += 0.35
        elif trend == 'uptrend': score += 0.25
        elif trend == 'sideways': score += 0.10
        
        # 변동성 (20%)
        volatility = data.get('volatility', 25)
        if 15 <= volatility <= 30: score += 0.20
        elif 10 <= volatility <= 40: score += 0.15
        elif volatility <= 50: score += 0.10
        
        # 거래량 (15%)
        vol_spike = data.get('volume_spike', 1)
        if vol_spike >= 1.5: score += 0.15
        elif vol_spike >= 1.2: score += 0.10
        
        return min(score, 1.0)
    
    def calculate_vix_adjustment(self, base_score: float, vix: float) -> float:
        """VIX 조정"""
        low_vix = config.get('strategy.vix_thresholds.low', 15.0)
        high_vix = config.get('strategy.vix_thresholds.high', 30.0)
        
        if vix <= low_vix:
            return base_score * 1.15  # 저변동성 시 부스트
        elif vix >= high_vix:
            return base_score * 0.85  # 고변동성 시 감소
        else:
            return base_score
    
    def calculate_total_score(self, data: Dict, vix: float) -> Tuple[float, Dict]:
        """통합 점수 계산"""
        # 각 전략 점수
        buffett = self.calculate_buffett_score(data)
        lynch = self.calculate_lynch_score(data)
        momentum = self.calculate_momentum_score(data)
        technical = self.calculate_technical_score(data)
        
        # 가중치 적용
        weights = config.get('strategy.weights', {})
        total = (
            buffett * weights.get('buffett', 25) +
            lynch * weights.get('lynch', 25) +
            momentum * weights.get('momentum', 25) +
            technical * weights.get('technical', 25)
        ) / 100
        
        # VIX 조정
        adjusted = self.calculate_vix_adjustment(total, vix)
        
        scores = {
            'buffett': buffett,
            'lynch': lynch,
            'momentum': momentum,
            'technical': technical,
            'total': adjusted,
            'vix_adjustment': adjusted - total
        }
        
        return adjusted, scores

# ========================================================================================
# 💰 매매 시스템
# ========================================================================================

class TradingSystem:
    """통합 매매 시스템"""
    
    def calculate_position_size(self, price: float, confidence: float, 
                              mode: str, portfolio_value: float = 1000000) -> Dict:
        """포지션 크기 계산"""
        try:
            if mode == 'swing':
                target_stocks = config.get('strategy.target_stocks.swing', 8)
                base_weight = 100 / target_stocks  # 12.5%
            else:  # classic
                target_stocks = config.get('strategy.target_stocks.classic', 20)
                base_weight = 80 / target_stocks  # 4%
            
            # 신뢰도 조정
            confidence_multiplier = 0.8 + (confidence * 0.4)
            target_weight = (base_weight / 100) * confidence_multiplier
            
            # 최대 포지션 제한
            max_pos = config.get('risk.max_position', 8.0) / 100
            target_weight = min(target_weight, max_pos)
            
            # 투자금액 및 주식수
            investment = portfolio_value * target_weight
            shares = int(investment / price)
            
            return {
                'total_shares': shares,
                'investment': investment,
                'weight': target_weight * 100
            }
            
        except Exception as e:
            logging.error(f"포지션 계산 실패: {e}")
            return {'total_shares': 0, 'investment': 0, 'weight': 0}
    
    def calculate_take_profit_levels(self, price: float, mode: str) -> Dict:
        """익절 레벨 계산"""
        if mode == 'swing':
            tp_levels = config.get('trading.swing.take_profit', [6.0, 12.0])
            ratios = config.get('trading.swing.profit_ratios', [60.0, 40.0])
            
            return {
                'tp1_price': price * (1 + tp_levels[0] / 100),
                'tp2_price': price * (1 + tp_levels[1] / 100),
                'tp1_ratio': 0.6,  # 60%
                'tp2_ratio': 0.4   # 40%
            }
    
    def calculate_stop_loss(self, price: float, mode: str) -> float:
        """손절가 계산"""
        if mode == 'swing':
            stop_pct = config.get('trading.swing.stop_loss', 8.0)
        else:
            stop_pct = config.get('risk.stop_loss.classic', 15.0)
        
        return price * (1 - stop_pct / 100)

# ========================================================================================
# 🏦 IBKR 연동 시스템
# ========================================================================================

class IBKRTrader:
    """IBKR 실거래 시스템"""
    
    def __init__(self):
        self.ib = None
        self.connected = False
        self.positions = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
    async def connect(self) -> bool:
        """IBKR 연결"""
        try:
            if not IBKR_AVAILABLE:
                logging.error("❌ IBKR 모듈 없음")
                return False
            
            host = config.get('ibkr.host', '127.0.0.1')
            port = config.get('ibkr.port', 7497)
            client_id = config.get('ibkr.client_id', 1)
            
            self.ib = IB()
            await self.ib.connectAsync(host, port, clientId=client_id)
            
            if self.ib.isConnected():
                self.connected = True
                mode = '모의투자' if config.get('ibkr.paper_trading') else '실거래'
                logging.info(f"✅ IBKR 연결 - {mode}")
                await self._update_account()
                return True
            else:
                logging.error("❌ IBKR 연결 실패")
                return False
                
        except Exception as e:
            logging.error(f"❌ IBKR 연결 오류: {e}")
            return False
    
    async def disconnect(self):
        """연결 해제"""
        try:
            if self.ib and self.connected:
                self.ib.disconnect()
                self.connected = False
                logging.info("🔌 IBKR 연결 해제")
        except Exception as e:
            logging.error(f"연결 해제 오류: {e}")
    
    async def _update_account(self):
        """계좌 정보 업데이트"""
        try:
            account_values = self.ib.accountValues()
            portfolio = self.ib.portfolio()
            
            # 일일 P&L
            for av in account_values:
                if av.tag == 'DayPNL':
                    self.daily_pnl = float(av.value)
                    break
            
            # 포지션 정보
            self.positions = {}
            for pos in portfolio:
                if pos.position != 0:
                    self.positions[pos.contract.symbol] = {
                        'quantity': pos.position,
                        'avg_cost': pos.avgCost,
                        'market_price': pos.marketPrice,
                        'unrealized_pnl': pos.unrealizedPNL
                    }
            
            logging.info(f"📊 계좌 업데이트 - PnL: ${self.daily_pnl:.2f}")
            
        except Exception as e:
            logging.error(f"계좌 업데이트 실패: {e}")
    
    async def place_buy_order(self, symbol: str, quantity: int) -> Optional[str]:
        """매수 주문"""
        try:
            if not self.connected or not self._safety_check():
                return None
            
            contract = Stock(symbol, 'SMART', 'USD')
            order_type = config.get('ibkr.order_type', 'MKT')
            
            if order_type == 'MKT':
                order = MarketOrder('BUY', quantity)
            else:
                order = MarketOrder('BUY', quantity)
            
            trade = self.ib.placeOrder(contract, order)
            order_id = str(trade.order.orderId)
            
            logging.info(f"📈 매수 주문: {symbol} {quantity}주 (ID: {order_id})")
            self.daily_trades += 1
            
            return order_id
            
        except Exception as e:
            logging.error(f"❌ 매수 실패 {symbol}: {e}")
            return None
    
    async def place_sell_order(self, symbol: str, quantity: int, reason: str = '') -> Optional[str]:
        """매도 주문"""
        try:
            if not self.connected:
                return None
            
            # 보유 수량 확인
            if symbol not in self.positions:
                logging.warning(f"⚠️ {symbol} 포지션 없음")
                return None
            
            current_qty = abs(self.positions[symbol]['quantity'])
            if quantity > current_qty:
                quantity = current_qty
            
            contract = Stock(symbol, 'SMART', 'USD')
            order = MarketOrder('SELL', quantity)
            
            trade = self.ib.placeOrder(contract, order)
            order_id = str(trade.order.orderId)
            
            logging.info(f"📉 매도 주문: {symbol} {quantity}주 - {reason}")
            self.daily_trades += 1
            
            return order_id
            
        except Exception as e:
            logging.error(f"❌ 매도 실패 {symbol}: {e}")
            return None
    
    def _safety_check(self) -> bool:
        """안전장치 체크"""
        max_trades = config.get('ibkr.max_daily_trades', 20)
        max_loss = config.get('risk.daily_loss_limit', 1.0) * 10000
        
        if self.daily_trades >= max_trades:
            logging.warning(f"⚠️ 일일 거래 한도 초과: {self.daily_trades}")
            return False
        
        if self.daily_pnl < -max_loss:
            logging.warning(f"⚠️ 일일 손실 한도 초과: ${self.daily_pnl:.2f}")
            return False
        
        return True
    
    async def get_current_price(self, symbol: str) -> float:
        """현재가 조회"""
        try:
            if not self.connected:
                return 0.0
            
            contract = Stock(symbol, 'SMART', 'USD')
            ticker = self.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(0.5)
            
            price = ticker.marketPrice() or ticker.last or 0.0
            self.ib.cancelMktData(contract)
            
            return float(price)
            
        except Exception as e:
            logging.error(f"현재가 조회 실패 {symbol}: {e}")
            return 0.0
    
    async def get_portfolio_value(self) -> float:
        """포트폴리오 가치"""
        try:
            await self._update_account()
            return sum(pos['market_price'] * abs(pos['quantity']) 
                      for pos in self.positions.values())
        except:
            return 0.0

# ========================================================================================
# 🤖 자동 손익절 관리자
# ========================================================================================

class StopTakeManager:
    """자동 손익절 관리자"""
    
    def __init__(self, ibkr_trader: IBKRTrader):
        self.ibkr = ibkr_trader
        self.positions: Dict[str, Position] = {}
        self.monitoring = False
        self.db_path = config.get('performance.database_file', 'legendary_performance.db')
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    profit_loss REAL DEFAULT 0.0,
                    profit_percent REAL DEFAULT 0.0,
                    mode TEXT NOT NULL,
                    reason TEXT DEFAULT ''
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    entry_date DATETIME NOT NULL,
                    mode TEXT NOT NULL,
                    stage INTEGER DEFAULT 1,
                    tp_executed TEXT DEFAULT '[]',
                    highest_price REAL DEFAULT 0.0
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("📊 데이터베이스 초기화 완료")
            
        except Exception as e:
            logging.error(f"DB 초기화 실패: {e}")
    
    def add_position(self, symbol: str, quantity: int, avg_cost: float, mode: str):
        """포지션 추가"""
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_cost=avg_cost,
            entry_date=datetime.now(),
            mode=mode,
            highest_price=avg_cost
        )
        
        self.positions[symbol] = position
        self._save_position_to_db(position)
        
        logging.info(f"➕ 포지션 추가: {symbol} {quantity}주 @${avg_cost:.2f} ({mode})")
    
    def _save_position_to_db(self, position: Position):
        """포지션 DB 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            tp_json = json.dumps(position.tp_executed)
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                (symbol, quantity, avg_cost, entry_date, mode, stage, tp_executed, highest_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol, position.quantity, position.avg_cost,
                position.entry_date.isoformat(), position.mode, position.stage,
                tp_json, position.highest_price
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"포지션 저장 실패: {e}")
    
    async def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring = True
        logging.info("🔍 자동 손익절 모니터링 시작!")
        
        while self.monitoring:
            try:
                await self._monitor_all_positions()
                interval = config.get('automation.monitoring_interval', 15)
                await asyncio.sleep(interval)
                
            except Exception as e:
                logging.error(f"모니터링 오류: {e}")
                await asyncio.sleep(30)
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        logging.info("⏹️ 모니터링 중지")
    
    async def _monitor_all_positions(self):
        """전체 포지션 모니터링"""
        for symbol, position in list(self.positions.items()):
            try:
                await self._monitor_single_position(symbol, position)
            except Exception as e:
                logging.error(f"포지션 모니터링 실패 {symbol}: {e}")
    
    async def _monitor_single_position(self, symbol: str, position: Position):
        """개별 포지션 모니터링"""
        try:
            current_price = await self.ibkr.get_current_price(symbol)
            if current_price <= 0:
                return
            
            # 최고가 업데이트
            if current_price > position.highest_price:
                position.highest_price = current_price
            
            profit_pct = position.profit_percent(current_price)
            hold_days = (datetime.now() - position.entry_date).days
            
            # 모드별 손익절 체크
            if position.mode == 'swing':
                await self._check_swing_exit(symbol, position, current_price, profit_pct, hold_days)
            else:  # classic
                await self._check_classic_exit(symbol, position, current_price, profit_pct, hold_days)
            
            # 공통 손절 체크
            await self._check_stop_loss(symbol, position, current_price, profit_pct)
            
        except Exception as e:
            logging.error(f"개별 모니터링 실패 {symbol}: {e}")
    
    async def _check_swing_exit(self, symbol: str, position: Position, 
                               current_price: float, profit_pct: float, hold_days: int):
        """스윙 익절 체크"""
        tp_levels = config.get('trading.swing.take_profit', [6.0, 12.0])
        ratios = config.get('trading.swing.profit_ratios', [60.0, 40.0])
        
        # 2차 익절 (12%)
        if profit_pct >= tp_levels[1] and not position.tp_executed[1]:
            sell_qty = int(position.quantity * ratios[1] / 100)
            if sell_qty > 0:
                order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'SWING_TP2')
                if order_id:
                    position.tp_executed[1] = True
                    position.quantity -= sell_qty
                    
                    await self._record_trade(symbol, 'SELL_SWING_TP2', sell_qty, 
                                           current_price, profit_pct, position.mode)
                    await self._send_notification(
                        f"🎉 {symbol} 스윙 2차 익절! +{profit_pct:.1f}% "
                        f"(${sell_qty * current_price:.0f})"
                    )
                    
                    if position.quantity <= 0:
                        del self.positions[symbol]
                        await self._remove_position_from_db(symbol)
        
        # 1차 익절 (6%)
        elif profit_pct >= tp_levels[0] and not position.tp_executed[0]:
            sell_qty = int(position.quantity * ratios[0] / 100)
            if sell_qty > 0:
                order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'SWING_TP1')
                if order_id:
                    position.tp_executed[0] = True
                    position.quantity -= sell_qty
                    
                    await self._record_trade(symbol, 'SELL_SWING_TP1', sell_qty,
                                           current_price, profit_pct, position.mode)
                    await self._send_notification(
                        f"✅ {symbol} 스윙 1차 익절! +{profit_pct:.1f}% "
                        f"(${sell_qty * current_price:.0f})"
                    )
    
    async def _check_classic_exit(self, symbol: str, position: Position,
                                 current_price: float, profit_pct: float, hold_days: int):
        """클래식 익절 체크"""
        tp_levels = config.get('trading.classic.take_profit', [20.0, 35.0])
        
        # 2차 익절 (35%)
        if profit_pct >= tp_levels[1] and not position.tp_executed[1]:
            sell_qty = int(position.quantity * 0.4)  # 40%
            if sell_qty > 0:
                order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'CLASSIC_TP2')
                if order_id:
                    position.tp_executed[1] = True
                    position.quantity -= sell_qty
                    
                    await self._record_trade(symbol, 'SELL_CLASSIC_TP2', sell_qty,
                                           current_price, profit_pct, position.mode)
                    await self._send_notification(
                        f"💰 {symbol} 클래식 2차 익절! +{profit_pct:.1f}% "
                        f"(${sell_qty * current_price:.0f})"
                    )
        
        # 1차 익절 (20%)
        elif profit_pct >= tp_levels[0] and not position.tp_executed[0]:
            sell_qty = int(position.quantity * 0.6)  # 60%
            if sell_qty > 0:
                order_id = await self.ibkr.place_sell_order(symbol, sell_qty, 'CLASSIC_TP1')
                if order_id:
                    position.tp_executed[0] = True
                    position.quantity -= sell_qty
                    
                    await self._record_trade(symbol, 'SELL_CLASSIC_TP1', sell_qty,
                                           current_price, profit_pct, position.mode)
                    await self._send_notification(
                        f"✅ {symbol} 클래식 1차 익절! +{profit_pct:.1f}% "
                        f"(${sell_qty * current_price:.0f})"
                    )
    
    async def _check_stop_loss(self, symbol: str, position: Position,
                              current_price: float, profit_pct: float):
        """손절 체크"""
        # 모드별 손절 기준
        if position.mode == 'swing':
            stop_pct = config.get('trading.swing.stop_loss', 8.0)
        else:
            stop_pct = config.get('risk.stop_loss.classic', 15.0)
        
        # 고정 손절
        if profit_pct <= -stop_pct:
            order_id = await self.ibkr.place_sell_order(symbol, position.quantity, 'STOP_LOSS')
            if order_id:
                await self._record_trade(symbol, 'SELL_STOP', position.quantity,
                                       current_price, profit_pct, position.mode)
                await self._send_notification(
                    f"🛑 {symbol} {position.mode} 손절! {profit_pct:.1f}% "
                    f"(${position.quantity * current_price:.0f})"
                )
                
                del self.positions[symbol]
                await self._remove_position_from_db(symbol)
        
        # 트레일링 스톱
        elif (config.get('risk.trailing_stop', True) and 
              position.highest_price > position.avg_cost * 1.1):
            trailing_distance = 0.05  # 5%
            trailing_stop = position.highest_price * (1 - trailing_distance)
            
            if current_price <= trailing_stop:
                order_id = await self.ibkr.place_sell_order(symbol, position.quantity, 'TRAILING_STOP')
                if order_id:
                    await self._record_trade(symbol, 'SELL_TRAILING', position.quantity,
                                           current_price, profit_pct, position.mode)
                    await self._send_notification(
                        f"📉 {symbol} 트레일링 스톱! {profit_pct:.1f}% "
                        f"(최고: ${position.highest_price:.2f})"
                    )
                    
                    del self.positions[symbol]
                    await self._remove_position_from_db(symbol)
    
    async def _record_trade(self, symbol: str, action: str, quantity: int,
                           price: float, profit_pct: float, mode: str):
        """거래 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            profit_loss = 0.0
            if 'SELL' in action and symbol in self.positions:
                position = self.positions[symbol]
                profit_loss = (price - position.avg_cost) * quantity
            
            cursor.execute('''
                INSERT INTO trades 
                (symbol, action, quantity, price, timestamp, profit_loss, profit_percent, mode, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, action, quantity, price, datetime.now().isoformat(), 
                  profit_loss, profit_pct, mode, action))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"거래 기록 실패: {e}")
    
    async def _remove_position_from_db(self, symbol: str):
        """DB에서 포지션 제거"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"포지션 제거 실패: {e}")
    
    async def _send_notification(self, message: str):
        """알림 전송"""
        try:
            # 텔레그램 알림
            if config.get('notifications.telegram.enabled', False):
                await self._send_telegram(message)
            
            # 로그 출력
            logging.info(f"📢 {message}")
            
        except Exception as e:
            logging.error(f"알림 전송 실패: {e}")
    
    async def _send_telegram(self, message: str):
        """텔레그램 메시지 전송"""
        try:
            token = config.get('notifications.telegram.bot_token', '')
            chat_id = config.get('notifications.telegram.chat_id', '')
            
            if not token or not chat_id:
                return
            
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': f"🏆 전설적퀸트\n{message}",
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logging.debug("텔레그램 알림 전송 완료")
                        
        except Exception as e:
            logging.error(f"텔레그램 전송 오류: {e}")

# ========================================================================================
# 🏆 메인 전략 시스템
# ========================================================================================

class LegendaryQuantStrategy:
    """🔥 전설적 퀸트 통합 전략 시스템"""
    
    def __init__(self):
        self.enabled = config.get('strategy.enabled', True)
        self.current_mode = config.get('strategy.mode', 'swing')
        
        # 핵심 컴포넌트들
        self.selector = StockSelector()
        self.analyzer = StrategyAnalyzer()
        self.trading = TradingSystem()
        self.ibkr = IBKRTrader()
        self.stop_take = StopTakeManager(self.ibkr)
        
        # 캐싱
        self.selected_stocks = []
        self.last_selection = None
        self.cache_hours = config.get('selection.refresh_hours', 24)
        
        # 성과 추적
        self.monthly_return = 0.0
        self.target_min = config.get('strategy.monthly_target.min', 5.0)
        self.target_max = config.get('strategy.monthly_target.max', 7.0)
        
        if self.enabled:
            logging.info("🏆 전설적 퀸트 전략 시스템 가동!")
            logging.info(f"🎯 현재 모드: {self.current_mode.upper()}")
    
    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if not self.last_selection or not self.selected_stocks:
            return False
        hours_passed = (datetime.now() - self.last_selection).seconds / 3600
        return hours_passed < self.cache_hours
    
    async def auto_select_stocks(self) -> List[str]:
        """자동 종목 선별"""
        if not self.enabled:
            return []
        
        try:
            # 캐시 확인
            if self._is_cache_valid():
                logging.info("📋 캐시된 선별 결과 사용")
                return [s['symbol'] for s in self.selected_stocks]
            
            logging.info("🚀 종목 자동선별 시작!")
            start_time = time.time()
            
            # 1. 투자 유니버스 생성
            universe = await self.selector.create_universe()
            if not universe:
                return self._get_fallback_stocks()
            
            # 2. VIX 조회
            current_vix = await self.selector.get_current_vix()
            
            # 3. 병렬 분석
            scored_stocks = await self._parallel_analysis(universe, current_vix)
            
            if not scored_stocks:
                return self._get_fallback_stocks()
            
            # 4. 상위 종목 선별
            target_count = self._get_target_count()
            final_selection = self._select_diversified_stocks(scored_stocks, target_count)
            
            # 5. 결과 저장
            self.selected_stocks = final_selection
            self.last_selection = datetime.now()
            
            selected_symbols = [s['symbol'] for s in final_selection]
            elapsed = time.time() - start_time
            
            logging.info(f"🏆 선별 완료! {len(selected_symbols)}개 종목 ({elapsed:.1f}초)")
            
            return selected_symbols
            
        except Exception as e:
            logging.error(f"자동선별 실패: {e}")
            return self._get_fallback_stocks()
    
    async def _parallel_analysis(self, universe: List[str], vix: float) -> List[Dict]:
        """병렬 종목 분석"""
        scored_stocks = []
        
        # 배치 처리로 메모리 효율성 개선
        batch_size = 20
        for i in range(0, len(universe), batch_size):
            batch = universe[i:i + batch_size]
            tasks = [self._analyze_stock_async(symbol, vix) for symbol in batch]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and result:
                    scored_stocks.append(result)
            
            if i % 100 == 0:
                logging.info(f"📊 분석 진행: {i}/{len(universe)}")
        
        return scored_stocks
    
    async def _analyze_stock_async(self, symbol: str, vix: float) -> Optional[Dict]:
        """비동기 종목 분석"""
        try:
            # 데이터 수집
            data = await self.selector.get_stock_data(symbol)
            if not data:
                return None
            
            # 기본 필터링
            min_cap = config.get('selection.min_market_cap', 5_000_000_000)
            min_vol = config.get('selection.min_volume', 1_000_000)
            
            if data.get('market_cap', 0) < min_cap or data.get('avg_volume', 0) < min_vol:
                return None
            
            # 통합 점수 계산
            total_score, scores = self.analyzer.calculate_total_score(data, vix)
            
            # 모드별 필터링
            if not self._mode_filter(data, total_score):
                return None
            
            result = data.copy()
            result.update(scores)
            result['symbol'] = symbol
            result['vix'] = vix
            result['mode'] = self.current_mode
            
            return result
            
        except Exception as e:
            logging.error(f"종목 분석 실패 {symbol}: {e}")
            return None
    
    def _mode_filter(self, data: Dict, score: float) -> bool:
        """모드별 필터링"""
        try:
            if self.current_mode == 'classic':
                return (score >= 0.60 and 
                        data.get('volatility', 50) <= 40 and
                        data.get('beta', 2.0) <= 1.8)
            elif self.current_mode == 'swing':
                return (score >= 0.65 and 
                        15 <= data.get('volatility', 25) <= 35 and
                        0.8 <= data.get('beta', 1.0) <= 1.5)
            else:  # hybrid
                return score >= 0.62
        except:
            return True
    
    def _get_target_count(self) -> int:
        """목표 종목수"""
        if self.current_mode == 'swing':
            return config.get('strategy.target_stocks.swing', 8)
        else:
            return config.get('strategy.target_stocks.classic', 20)
    
    def _select_diversified_stocks(self, scored_stocks: List[Dict], target_count: int) -> List[Dict]:
        """다양성 고려 선별"""
        scored_stocks.sort(key=lambda x: x['total'], reverse=True)
        
        final_selection = []
        sector_counts = {}
        max_per_sector = 2 if self.current_mode == 'swing' else 4
        
        for stock in scored_stocks:
            if len(final_selection) >= target_count:
                break
            
            sector = stock.get('sector', 'Unknown')
            
            if sector_counts.get(sector, 0) < max_per_sector:
                final_selection.append(stock)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # 부족하면 상위 점수로 채움
        remaining = target_count - len(final_selection)
        if remaining > 0:
            for stock in scored_stocks:
                if remaining <= 0:
                    break
                if stock not in final_selection:
                    final_selection.append(stock)
                    remaining -= 1
        
        return final_selection
    
    def _get_fallback_stocks(self) -> List[str]:
        """백업 종목"""
        if self.current_mode == 'swing':
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX']
        else:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                    'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'ABBV', 'BAC']
    
    async def analyze_stock_signal(self, symbol: str) -> StockSignal:
        """개별 종목 시그널 분석"""
        try:
            # 데이터 수집
            data = await self.selector.get_stock_data(symbol)
            if not data:
                raise ValueError(f"데이터 수집 실패: {symbol}")
            
            # VIX 조회
            vix = await self.selector.get_current_vix()
            
            # 점수 계산
            total_score, scores = self.analyzer.calculate_total_score(data, vix)
            
            # 액션 결정
            confidence_threshold = 0.70 if self.current_mode == 'classic' else 0.65
            
            if total_score >= confidence_threshold:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= 0.30:
                action = 'sell'
                confidence = min(1 - total_score, 0.95)
            else:
                action = 'hold'
                confidence = 0.50
            
            # 목표가 계산
            max_return = 0.25 if self.current_mode == 'swing' else 0.35
            target_price = data['price'] * (1 + confidence * max_return)
            
            # 손절가 계산
            stop_loss = self.trading.calculate_stop_loss(data['price'], self.current_mode)
            
            # 근거 생성
            reasoning = (f"버핏:{scores['buffett']:.2f} 린치:{scores['lynch']:.2f} "
                        f"모멘텀:{scores['momentum']:.2f} 기술:{scores['technical']:.2f} "
                        f"VIX:{scores['vix_adjustment']:+.2f} 모드:{self.current_mode}")
            
            return StockSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=data['price'],
                mode=self.current_mode,
                scores=scores,
                financials={
                    'market_cap': data.get('market_cap', 0),
                    'pe_ratio': data.get('pe_ratio', 0),
                    'pbr': data.get('pbr', 0),
                    'peg': data.get('peg', 0),
                    'roe': data.get('roe', 0)
                },
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"시그널 분석 실패 {symbol}: {e}")
            return self._create_empty_signal(symbol, str(e))
    
    def _create_empty_signal(self, symbol: str, error: str) -> StockSignal:
        """빈 시그널 생성"""
        return StockSignal(
            symbol=symbol, action='hold', confidence=0.0, price=0.0,
            mode=self.current_mode, scores={}, financials={},
            target_price=0.0, stop_loss=0.0, reasoning=f"오류: {error}",
            timestamp=datetime.now()
        )
    
    async def scan_all_stocks(self) -> List[StockSignal]:
        """전체 종목 스캔"""
        if not self.enabled:
            return []
        
        logging.info("🔍 전체 종목 스캔 시작!")
        
        try:
            # 종목 선별
            selected = await self.auto_select_stocks()
            if not selected:
                return []
            
            # 각 종목 분석
            signals = []
            for i, symbol in enumerate(selected, 1):
                try:
                    signal = await self.analyze_stock_signal(symbol)
                    signals.append(signal)
                    
                    # 진행상황
                    emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                    logging.info(f"{emoji} {symbol}: {signal.action} 신뢰도:{signal.confidence:.2f}")
                    
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logging.error(f"❌ {symbol} 분석 실패: {e}")
            
            # 결과 요약
            buy_count = len([s for s in signals if s.action == 'buy'])
            sell_count = len([s for s in signals if s.action == 'sell'])
            hold_count = len([s for s in signals if s.action == 'hold'])
            
            logging.info(f"🏆 스캔 완료! 매수:{buy_count}, 매도:{sell_count}, 보유:{hold_count}")
            
            return signals
            
        except Exception as e:
            logging.error(f"전체 스캔 실패: {e}")
            return []
    
    async def initialize_trading(self) -> bool:
        """거래 시스템 초기화"""
        try:
            logging.info("🚀 거래 시스템 초기화...")
            
            # IBKR 연결
            if not await self.ibkr.connect():
                logging.error("❌ IBKR 연결 실패")
                return False
            
            # 기존 포지션 로드
            await self._load_existing_positions()
            
            logging.info("✅ 거래 시스템 초기화 완료!")
            return True
            
        except Exception as e:
            logging.error(f"초기화 실패: {e}")
            return False
    
    async def _load_existing_positions(self):
        """기존 포지션 로드"""
        try:
            conn = sqlite3.connect(self.stop_take.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM positions')
            rows = cursor.fetchall()
            
            for row in rows:
                tp_executed = json.loads(row[6]) if row[6] else [False, False, False]
                
                position = Position(
                    symbol=row[0],
                    quantity=row[1],
                    avg_cost=row[2],
                    entry_date=datetime.fromisoformat(row[3]),
                    mode=row[4],
                    stage=row[5],
                    tp_executed=tp_executed,
                    highest_price=row[7]
                )
                
                self.stop_take.positions[position.symbol] = position
            
            conn.close()
            logging.info(f"📂 기존 포지션 로드: {len(self.stop_take.positions)}개")
            
        except Exception as e:
            logging.error(f"포지션 로드 실패: {e}")
    
    async def start_auto_trading(self):
        """자동거래 시작"""
        try:
            logging.info(f"🎯 자동거래 시작! (모드: {self.current_mode.upper()})")
            
            # 손익절 모니터링 시작
            monitor_task = asyncio.create_task(self.stop_take.start_monitoring())
            
            # 스케줄 실행
            schedule_task = asyncio.create_task(self._run_schedule())
            
            # 병렬 실행
            await asyncio.gather(monitor_task, schedule_task)
            
        except Exception as e:
            logging.error(f"자동거래 실행 실패: {e}")
        finally:
            await self.shutdown()
    
    async def _run_schedule(self):
        """스케줄 실행"""
        while True:
            try:
                now = datetime.now()
                
                # 스윙 모드 스케줄
                if self.current_mode == 'swing':
                    if now.weekday() == 1 and now.hour == 10 and now.minute == 30:  # 화요일
                        await self._swing_entry()
                    elif now.weekday() == 3 and now.hour == 10 and now.minute == 30:  # 목요일
                        await self._swing_entry()
                
                # 클래식 모드 스케줄
                elif self.current_mode == 'classic':
                    if now.hour == 10 and now.minute == 0:  # 매일 10시
                        await self._classic_entry()
                
                # 일일 체크
                if now.hour == 9 and now.minute == 0:
                    await self._perform_daily_check()
                
                # 성과 리포트
                if now.hour == 16 and now.minute == 0:
                    await self._generate_report()
                
                await asyncio.sleep(60)  # 1분마다 체크
                
            except Exception as e:
                logging.error(f"스케줄 오류: {e}")
                await asyncio.sleep(60)
    
    async def _swing_entry(self):
        """스윙 진입"""
        try:
            if not self._is_trading_day():
                return
            
            day = datetime.now().strftime("%A")
            logging.info(f"📅 {day} 스윙 진입...")
            
            # 종목 선별
            target_count = 4 if day == 'Tuesday' else 2
            selected = await self.auto_select_stocks()
            
            # 기존 포지션 제외
            existing = list(self.stop_take.positions.keys())
            new_stocks = [s for s in selected if s not in existing][:target_count]
            
            # 포지션 크기 계산
            portfolio_value = await self.ibkr.get_portfolio_value() or 1000000
            per_stock = portfolio_value * 0.125  # 12.5%
            
            for symbol in new_stocks:
                await self._enter_position(symbol, per_stock, 'swing', day)
            
        except Exception as e:
            logging.error(f"스윙 진입 실패: {e}")
    
    async def _classic_entry(self):
        """클래식 진입"""
        try:
            if not self._is_trading_day():
                return
            
            logging.info("📅 클래식 진입 체크...")
            
            # 포지션 한도 체크
            current_count = len(self.stop_take.positions)
            max_positions = config.get('strategy.target_stocks.classic', 20)
            
            if current_count >= max_positions:
                return
            
            # 종목 선별
            selected = await self.auto_select_stocks()
            existing = list(self.stop_take.positions.keys())
            new_stocks = [s for s in selected if s not in existing][:3]
            
            # 포지션 크기 계산
            portfolio_value = await self.ibkr.get_portfolio_value() or 1000000
            per_stock = portfolio_value * 0.04  # 4%
            
            for symbol in new_stocks:
                await self._enter_position(symbol, per_stock, 'classic', 'Daily')
            
        except Exception as e:
            logging.error(f"클래식 진입 실패: {e}")
    
    async def _enter_position(self, symbol: str, investment: float, mode: str, context: str):
        """포지션 진입"""
        try:
            # 현재가 조회
            current_price = await self.ibkr.get_current_price(symbol)
            if current_price <= 0:
                logging.warning(f"⚠️ {symbol} 현재가 조회 실패")
                return
            
            # 수량 계산
            quantity = int(investment / current_price)
            if quantity < 1:
                logging.warning(f"⚠️ {symbol} 매수 수량 부족")
                return
            
            # 매수 주문
            order_id = await self.ibkr.place_buy_order(symbol, quantity)
            if order_id:
                # 포지션 추가
                self.stop_take.add_position(symbol, quantity, current_price, mode)
                
                # 알림
                investment_value = quantity * current_price
                await self.stop_take._send_notification(
                    f"🚀 {symbol} {mode.upper()} 진입!\n"
                    f"📅 {context}\n"
                    f"💰 ${investment_value:.0f} ({quantity}주 @${current_price:.2f})"
                )
                
                logging.info(f"✅ {symbol} {mode} 포지션 진입: {quantity}주 @${current_price:.2f}")
            
        except Exception as e:
            logging.error(f"포지션 진입 실패 {symbol}: {e}")
    
    def _is_trading_day(self) -> bool:
        """거래일 확인"""
        today = datetime.now()
        
        # 주말 제외
        if today.weekday() >= 5:
            return False
        
        # 공휴일 체크 (간단한 버전)
        holidays = [
            datetime(today.year, 1, 1),   # 신정
            datetime(today.year, 7, 4),   # 독립기념일
            datetime(today.year, 12, 25), # 크리스마스
        ]
        
        if any(today.date() == holiday.date() for holiday in holidays):
            return False
        
        return True
    
    async def _perform_daily_check(self):
        """일일 체크"""
        try:
            if not self._is_trading_day():
                return
            
            logging.info("📊 일일 체크...")
            
            # 계좌 정보 업데이트
            await self.ibkr._update_account()
            
            # 월 수익률 계산
            await self._calculate_monthly_return()
            
            # 리스크 체크
            await self._check_risk_limits()
            
        except Exception as e:
            logging.error(f"일일 체크 실패: {e}")
    
    async def _calculate_monthly_return(self):
        """월 수익률 계산"""
        try:
            conn = sqlite3.connect(self.stop_take.db_path)
            cursor = conn.cursor()
            
            current_month = datetime.now().strftime('%Y-%m')
            cursor.execute('''
                SELECT SUM(profit_loss) FROM trades 
                WHERE strftime('%Y-%m', timestamp) = ?
                AND action LIKE 'SELL%'
            ''', (current_month,))
            
            result = cursor.fetchone()
            monthly_profit = result[0] if result[0] else 0.0
            
            # 포트폴리오 가치
            portfolio_value = await self.ibkr.get_portfolio_value()
            if portfolio_value > 0:
                self.monthly_return = (monthly_profit / portfolio_value) * 100
            
            conn.close()
            
            logging.info(f"📈 월 수익률: {self.monthly_return:.2f}%")
            
        except Exception as e:
            logging.error(f"월 수익률 계산 실패: {e}")
    
    async def _check_risk_limits(self):
        """리스크 한도 체크"""
        try:
            # 일일 손실 한도
            daily_limit = config.get('risk.daily_loss_limit', 1.0)
            portfolio_value = await self.ibkr.get_portfolio_value()
            
            if self.ibkr.daily_pnl < -(portfolio_value * daily_limit / 100):
                await self._emergency_stop("일일 손실 한도 초과")
                return
            
            # 월 손실 한도
            if self.current_mode == 'swing':
                monthly_limit = config.get('risk.monthly_loss_limit', 3.0)
                if self.monthly_return < -monthly_limit:
                    await self._emergency_stop(f"월 손실 한도 초과: {self.monthly_return:.2f}%")
            
        except Exception as e:
            logging.error(f"리스크 체크 실패: {e}")
    
    async def _emergency_stop(self, reason: str):
        """비상 정지"""
        try:
            logging.warning(f"🚨 비상 정지: {reason}")
            
            # 모든 포지션 정리
            for symbol, position in list(self.stop_take.positions.items()):
                await self.ibkr.place_sell_order(symbol, position.quantity, 'EMERGENCY')
            
            # 포지션 초기화
            self.stop_take.positions.clear()
            
            # 알림
            await self.stop_take._send_notification(
                f"🚨 시스템 비상 정지!\n📝 사유: {reason}\n💰 모든 포지션 정리"
            )
            
        except Exception as e:
            logging.error(f"비상 정지 실패: {e}")
    
    async def _generate_report(self):
        """성과 리포트 생성"""
        try:
            # 간단한 일일 리포트
            active_positions = len(self.stop_take.positions)
            daily_pnl = self.ibkr.daily_pnl
            
            report = f"""
🏆 일일 요약 리포트
==================
📊 현재 모드: {self.current_mode.upper()}
💰 일일 P&L: ${daily_pnl:.2f}
📈 월 수익률: {self.monthly_return:.2f}%
📋 활성 포지션: {active_positions}개
"""
            
            await self.stop_take._send_notification(report)
            
        except Exception as e:
            logging.error(f"리포트 생성 실패: {e}")
    
    async def shutdown(self):
        """시스템 종료"""
        try:
            logging.info("🔌 시스템 종료 중...")
            
            self.stop_take.stop_monitoring()
            await self.ibkr.disconnect()
            
            logging.info("✅ 시스템 종료 완료")
            
        except Exception as e:
            logging.error(f"종료 실패: {e}")

# ========================================================================================
# 🎯 편의 함수들
# ========================================================================================

async def run_auto_selection():
    """자동 선별 실행"""
    try:
        strategy = LegendaryQuantStrategy()
        signals = await strategy.scan_all_stocks()
        return signals
    except Exception as e:
        logging.error(f"자동 선별 실패: {e}")
        return []

async def analyze_single_stock(symbol: str):
    """단일 종목 분석"""
    try:
        strategy = LegendaryQuantStrategy()
        signal = await strategy.analyze_stock_signal(symbol)
        return signal
    except Exception as e:
        logging.error(f"종목 분석 실패: {e}")
        return None

async def get_system_status():
    """시스템 상태 조회"""
    try:
        strategy = LegendaryQuantStrategy()
        
        # IBKR 연결 테스트
        ibkr_connected = False
        try:
            if IBKR_AVAILABLE:
                ibkr_connected = await strategy.ibkr.connect()
                if ibkr_connected:
                    await strategy.ibkr.disconnect()
        except Exception as e:
            logging.warning(f"IBKR 연결 테스트 실패: {e}")
            ibkr_connected = False
        
        return {
            'enabled': strategy.enabled,
            'current_mode': strategy.current_mode,
            'ibkr_connected': ibkr_connected,
            'selected_count': len(strategy.selected_stocks),
            'target_min': strategy.target_min,
            'target_max': strategy.target_max,
            'monthly_return': strategy.monthly_return,
            'ibkr_available': IBKR_AVAILABLE
        }
        
    except Exception as e:
        logging.error(f"상태 조회 실패: {e}")
        return {'error': str(e)}

async def switch_mode(mode: str):
    """모드 전환"""
    try:
        if mode in ['classic', 'swing', 'hybrid']:
            config.update('strategy.mode', mode)
            return {'status': 'success', 'message': f'모드가 {mode}로 전환되었습니다'}
        else:
            return {'status': 'error', 'message': '유효한 모드: classic, swing, hybrid'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

async def run_auto_trading():
    """자동거래 실행"""
    strategy = LegendaryQuantStrategy()
    
    try:
        if await strategy.initialize_trading():
            await strategy.start_auto_trading()
        else:
            logging.error("❌ 거래 시스템 초기화 실패")
    except KeyboardInterrupt:
        logging.info("⏹️ 사용자 중단")
    except Exception as e:
        logging.error(f"❌ 자동거래 실패: {e}")
    finally:
        await strategy.shutdown()

# ========================================================================================
# 🎯 메인 실행부
# ========================================================================================

async def main():
    """메인 실행 함수"""
    try:
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('legendary_quant.log', encoding='utf-8')
            ]
        )
        
        print("🏆" + "="*60)
        print("🔥 전설적 퀸트프로젝트 - 통합 완성판 V6.2")
        print("🚀 4가지 전략 융합 + 실시간 크롤링 + IBKR 연동")
        print("="*62)
        
        print("\n🌟 주요 특징:")
        print("  ✨ 4가지 투자전략 지능형 융합 (버핏+린치+모멘텀+기술)")
        print("  ✨ 실시간 S&P500+NASDAQ 자동선별")
        print("  ✨ VIX 기반 시장상황 자동판단")
        print("  ✨ 스윙 2단계 익절 + 클래식 분할매매")
        print("  ✨ IBKR 실거래 연동 + 자동 손익절")
        print("  ✨ 완전 자동화 + 보수유지 최적화")
        
        # 시스템 상태 확인
        print("\n🔧 시스템 상태 확인...")
        status = await get_system_status()
        
        if 'error' not in status:
            print(f"  ✅ 시스템 활성화: {status['enabled']}")
            print(f"  ✅ 현재 모드: {status['current_mode'].upper()}")
            
            # IBKR 상태 표시 개선
            if status.get('ibkr_available', False):
                ibkr_status = '연결 가능' if status['ibkr_connected'] else '연결 불가'
                print(f"  ✅ IBKR 상태: {ibkr_status}")
            else:
                print(f"  ⚠️  IBKR 모듈: 미설치 (pip install ib_insync)")
            
            print(f"  ✅ 월 목표: {status['target_min']:.1f}%-{status['target_max']:.1f}%")
            print(f"  ✅ 월 수익률: {status['monthly_return']:.2f}%")
            print(f"  ✅ 선별된 종목: {status['selected_count']}개")
        else:
            print(f"  ❌ 상태 확인 실패: {status['error']}")
        
        print("\n🚀 실행 옵션:")
        print("  1. 종목 자동선별 + 분석")
        print("  2. 완전 자동거래 시작")
        print("  3. 개별 종목 분석")
        print("  4. 모드 전환")
        print("  5. 시스템 상태")
        print("  6. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (1-6): ").strip()
                
                if choice == '1':
                    print("\n🔍 종목 자동선별 + 분석 시작!")
                    signals = await run_auto_selection()
                    
                    if signals:
                        print(f"\n📈 분석 결과:")
                        buy_signals = [s for s in signals if s.action == 'buy']
                        sell_signals = [s for s in signals if s.action == 'sell']
                        
                        print(f"  🟢 매수 추천: {len(buy_signals)}개")
                        print(f"  🔴 매도 추천: {len(sell_signals)}개")
                        print(f"  ⚪ 보유 추천: {len(signals) - len(buy_signals) - len(sell_signals)}개")
                        
                        # 상위 매수 추천
                        top_buys = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
                        if top_buys:
                            print(f"\n🏆 상위 매수 추천:")
                            for i, signal in enumerate(top_buys, 1):
                                print(f"  {i}. {signal.symbol}: 신뢰도 {signal.confidence:.1%}, "
                                      f"목표가 ${signal.target_price:.2f}")
                    else:
                        print("❌ 분석 결과 없음")
                
                elif choice == '2':
                    print("\n🎯 완전 자동거래 시작!")
                    print("⚠️  IBKR TWS/Gateway가 실행 중인지 확인하세요!")
                    confirm = input("계속하시겠습니까? (y/N): ").strip().lower()
                    if confirm == 'y':
                        await run_auto_trading()
                    break
                
                elif choice == '3':
                    symbol = input("분석할 종목 심볼: ").strip().upper()
                    if symbol:
                        print(f"\n🔍 {symbol} 분석중...")
                        signal = await analyze_single_stock(symbol)
                        
                        if signal and signal.confidence > 0:
                            print(f"\n📊 {symbol} 분석 결과:")
                            print(f"  🎯 결정: {signal.action.upper()}")
                            print(f"  💯 신뢰도: {signal.confidence:.1%}")
                            print(f"  💰 현재가: ${signal.price:.2f}")
                            print(f"  🎯 목표가: ${signal.target_price:.2f}")
                            print(f"  🛑 손절가: ${signal.stop_loss:.2f}")
                            print(f"  📈 모드: {signal.mode.upper()}")
                            print(f"  💡 근거: {signal.reasoning}")
                        else:
                            print(f"❌ {symbol} 분석 실패")
                
                elif choice == '4':
                    print("\n🔄 모드 전환:")
                    print("  1. CLASSIC (클래식 분할매매)")
                    print("  2. SWING (스윙 2단계 익절)")
                    print("  3. HYBRID (하이브리드)")
                    
                    mode_choice = input("모드 선택 (1-3): ").strip()
                    mode_map = {'1': 'classic', '2': 'swing', '3': 'hybrid'}
                    
                    if mode_choice in mode_map:
                        result = await switch_mode(mode_map[mode_choice])
                        print(f"✅ {result['message']}")
                    else:
                        print("❌ 잘못된 선택")
                
                elif choice == '5':
                    print("\n📊 시스템 상세 상태:")
                    status = await get_system_status()
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                
                elif choice == '6':
                    print("👋 시스템을 종료합니다!")
                    break
                    
                else:
                    print("❌ 잘못된 선택입니다. 1-6 중 선택하세요.")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
    except Exception as e:
        logging.error(f"메인 실행 실패: {e}")
        print(f"❌ 실행 실패: {e}")

# ========================================================================================
# 🔧 유틸리티 함수들
# ========================================================================================

def create_default_env_file():
    """기본 .env 파일 생성"""
    env_content = """# 전설적 퀸트프로젝트 환경변수 설정
# IBKR 설정
IBKR_ACCOUNT=YOUR_ACCOUNT_ID

# 텔레그램 알림 설정
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID

# 기타 설정
# EMAIL_USERNAME=your_email@gmail.com
# EMAIL_PASSWORD=your_app_password
# EMAIL_RECIPIENT=recipient@gmail.com
"""
    
    if not Path('.env').exists():
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("📝 .env 파일이 생성되었습니다. 설정을 입력해주세요.")

def check_dependencies():
    """의존성 패키지 확인"""
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'requests', 'beautifulsoup4',
        'aiohttp', 'pyyaml', 'python-dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ 누락된 패키지: {', '.join(missing)}")
        print(f"설치 명령: pip install {' '.join(missing)}")
        return False
    
    return True

def setup_system():
    """시스템 초기 설정"""
    print("🔧 시스템 초기 설정...")
    
    # 의존성 확인
    if not check_dependencies():
        return False
    
    # .env 파일 생성
    create_default_env_file()
    
    # 로그 디렉토리 생성
    log_dir = Path('logs')
    if not log_dir.exists():
        log_dir.mkdir()
    
    # 데이터 디렉토리 생성
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir()
    
    print("✅ 시스템 초기 설정 완료!")
    return True

# ========================================================================================
# 🏃‍♂️ 빠른 시작 함수들
# ========================================================================================

async def quick_analysis(symbols: List[str] = None):
    """빠른 분석"""
    if not symbols:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    print(f"🚀 빠른 분석: {', '.join(symbols)}")
    
    strategy = LegendaryQuantStrategy()
    
    for symbol in symbols:
        try:
            signal = await strategy.analyze_stock_signal(symbol)
            action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
            print(f"{action_emoji} {symbol}: {signal.action} ({signal.confidence:.1%})")
        except Exception as e:
            print(f"❌ {symbol}: 분석 실패")

async def quick_scan():
    """빠른 스캔"""
    print("🔍 빠른 전체 스캔...")
    
    try:
        signals = await run_auto_selection()
        
        if signals:
            buy_signals = [s for s in signals if s.action == 'buy']
            
            print(f"\n📊 스캔 결과: 총 {len(signals)}개 종목")
            print(f"🟢 매수 추천: {len(buy_signals)}개")
            
            # 상위 5개
            top_5 = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
            if top_5:
                print("\n🏆 TOP 5 매수 추천:")
                for i, signal in enumerate(top_5, 1):
                    print(f"  {i}. {signal.symbol}: {signal.confidence:.1%}")
        else:
            print("❌ 스캔 결과 없음")
            
    except Exception as e:
        print(f"❌ 스캔 실패: {e}")

def print_help():
    """도움말 출력"""
    help_text = """
🏆 전설적 퀸트프로젝트 V6.2 - 사용법
=====================================

📋 주요 명령어:
  python legendary_quint_project_v6.py        # 메인 메뉴 실행
  python -c "from legendary_quint_project_v6 import *; asyncio.run(quick_scan())"  # 빠른 스캔
  python -c "from legendary_quint_project_v6 import *; asyncio.run(quick_analysis())"  # 빠른 분석

🔧 초기 설정:
  1. pip install yfinance pandas numpy requests beautifulsoup4 aiohttp pyyaml python-dotenv
  2. IBKR 사용시: pip install ib_insync
  3. .env 파일에서 텔레그램/IBKR 설정
  4. legendary_unified_settings.yaml에서 상세 설정

💡 모드 설명:
  - SWING: 월 5-7% 목표, 8개 종목, 2단계 익절 (6%/12%)
  - CLASSIC: 장기 성장, 20개 종목, 분할매매 (20%/35% 익절)
  - HYBRID: 신뢰도에 따라 자동 전환

🎯 4가지 전략:
  - 버핏 가치투자: PBR, ROE, 부채비율 중심
  - 린치 성장투자: PEG, EPS성장률 중심  
  - 모멘텀 전략: 3/6/12개월 수익률, 거래량 중심
  - 기술적 분석: RSI, 추세, 변동성 중심

📊 VIX 조정:
  - VIX < 15: 점수 15% 부스트 (적극적)
  - VIX > 30: 점수 15% 감소 (보수적)

🛡️ 리스크 관리:
  - 스윙: 8% 손절, 트레일링 스톱
  - 클래식: 15% 손절, 섹터 분산
  - 일일/월간 손실 한도 자동 체크

📱 알림 설정:
  - 텔레그램: 실시간 진입/청산 알림
  - 성과 리포트: 일일/주간/월간

🚀 자동화 기능:
  - 실시간 S&P500/NASDAQ 크롤링
  - 자동 종목 선별 (24시간 캐싱)
  - IBKR 자동 주문 + 손익절
  - 시장 상황별 적응형 전략
"""
    print(help_text)

# ========================================================================================
# 🏁 실행 진입점
# ========================================================================================

if __name__ == "__main__":
    try:
        # 명령행 인자 처리
        if len(sys.argv) > 1:
            if sys.argv[1] == 'help' or sys.argv[1] == '--help':
                print_help()
                sys.exit(0)
            elif sys.argv[1] == 'setup':
                setup_system()
                sys.exit(0)
            elif sys.argv[1] == 'quick-scan':
                asyncio.run(quick_scan())
                sys.exit(0)
            elif sys.argv[1] == 'quick-analysis':
                symbols = sys.argv[2:] if len(sys.argv) > 2 else None
                asyncio.run(quick_analysis(symbols))
                sys.exit(0)
        
        # 메인 실행
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        logging.error(f"실행 오류: {e}")
        
        'tp1_price': price * (1 + tp_levels[0] / 100),
        'tp2_price': price * (1 + tp_levels[1] / 100),
        'tp1_ratio': ratios[0] / 100,
        'tp2_ratio': ratios[1] / 100
    }
    else:  # classic
        tp_levels = config.get('trading.classic.take_profit', [20.0, 35.0])
        return {
            'tp1_price': price * (1 + tp_levels[0] / 100),
            'tp2_price': price * (1 + tp_levels[1] / 100),
            'tp1_ratio': 0.6,  # 60%
            'tp2_ratio': 0.4   # 40%
        }
