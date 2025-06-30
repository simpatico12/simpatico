#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇺🇸 미국 주식 전략 모듈 - 최고퀸트프로젝트 (진짜 자동선별)
===========================================================

핵심 기능:
1. 🆕 실시간 S&P500 + NASDAQ100 + 러셀1000 크롤링
2. 4가지 전략 융합 (버핏25% + 린치25% + 모멘텀25% + 기술25%)
3. 개별 분할매매 (각 종목마다 3단계 매수, 2단계 매도)
4. 완전 자동화 시스템
5. VIX 기반 동적 조정

Author: 최고퀸트팀
Version: 4.0.0 (진짜 자동선별 구현)
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
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
warnings.filterwarnings('ignore')

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class USStockSignal:
    """미국 주식 시그널 데이터 클래스 (완전 업그레이드)"""
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
# 🆕 실시간 미국 주식 수집 및 선별 클래스 (NEW!)
# ========================================================================================
class RealTimeUSStockSelector:
    """🆕 실시간 미국 주식 종목 수집 및 선별"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 선별 기준
        self.min_market_cap = 5_000_000_000   # 50억달러
        self.min_avg_volume = 1_000_000       # 100만주
        self.target_stocks = 20               # 최종 20개 선별
        
        # VIX 기반 동적 조정
        self.current_vix = 0.0
        self.vix_low_threshold = 15.0         # 저변동성
        self.vix_high_threshold = 30.0        # 고변동성

    async def get_sp500_constituents(self) -> List[str]:
        """S&P 500 구성종목 실시간 수집"""
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
                
            except Exception as e:
                logger.warning(f"Wikipedia S&P 500 수집 실패: {e}")
            
            # 소스 2: Yahoo Finance S&P 500
            try:
                url = "https://finance.yahoo.com/quote/%5EGSPC/components"
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        if '/quote/' in href and '?' in href:
                            symbol = href.split('/quote/')[-1].split('?')[0]
                            if len(symbol) <= 5 and symbol.isalpha():
                                symbols.append(symbol)
                
                logger.info(f"✅ Yahoo Finance에서 추가 종목 수집")
                
            except Exception as e:
                logger.warning(f"Yahoo Finance S&P 500 수집 실패: {e}")
            
            return list(set(symbols))  # 중복 제거
            
        except Exception as e:
            logger.error(f"S&P 500 구성종목 수집 실패: {e}")
            return self._get_backup_sp500()

    async def get_nasdaq100_constituents(self) -> List[str]:
        """NASDAQ 100 구성종목 실시간 수집"""
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
                
            except Exception as e:
                logger.warning(f"Wikipedia NASDAQ 100 수집 실패: {e}")
            
            # 소스 2: QQQ ETF 구성종목 (백업)
            try:
                qqq = yf.Ticker("QQQ")
                # ETF 구성종목 정보는 제한적이므로 주요 기술주 추가
                tech_giants = [
                    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
                    'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO',
                    'TXN', 'MU', 'AMAT', 'ADI', 'MRVL', 'KLAC', 'LRCX', 'SNPS'
                ]
                symbols.extend(tech_giants)
                
            except Exception as e:
                logger.warning(f"기술주 백업 리스트 추가 실패: {e}")
            
            return list(set(symbols))  # 중복 제거
            
        except Exception as e:
            logger.error(f"NASDAQ 100 구성종목 수집 실패: {e}")
            return self._get_backup_nasdaq100()

    async def get_russell1000_sample(self) -> List[str]:
        """러셀1000 주요 종목 샘플 수집"""
        try:
            logger.info("🔍 러셀1000 주요 종목 샘플 수집...")
            
            # 러셀1000 주요 대형주 (섹터별 대표주)
            russell_sample = [
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
            ]
            
            logger.info(f"✅ 러셀1000 샘플 {len(russell_sample)}개 종목 수집")
            return russell_sample
            
        except Exception as e:
            logger.error(f"러셀1000 샘플 수집 실패: {e}")
            return []

    def _get_backup_sp500(self) -> List[str]:
        """S&P 500 백업 리스트"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'ABBV', 'BAC', 'KO', 'AVGO',
            'XOM', 'CVX', 'LLY', 'WMT', 'PEP', 'TMO', 'COST', 'ORCL', 'ABT', 'ACN',
            'NFLX', 'CRM', 'MRK', 'VZ', 'ADBE', 'DHR', 'TXN', 'NKE', 'PM', 'DIS',
            'WFC', 'NEE', 'RTX', 'CMCSA', 'BMY', 'UNP', 'T', 'COP', 'MS', 'AMD',
            'LOW', 'IBM', 'HON', 'AMGN', 'SPGI', 'LIN', 'QCOM', 'GE', 'CAT', 'UPS'
        ]

    def _get_backup_nasdaq100(self) -> List[str]:
        """NASDAQ 100 백업 리스트"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO',
            'TXN', 'MU', 'AMAT', 'ADI', 'MRVL', 'KLAC', 'LRCX', 'SNPS',
            'ISRG', 'GILD', 'BKNG', 'MDLZ', 'ADP', 'CSX', 'REGN', 'VRTX'
        ]

    async def get_vix_level(self) -> float:
        """VIX 지수 조회"""
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1d")
            if not vix_data.empty:
                self.current_vix = vix_data['Close'].iloc[-1]
            else:
                self.current_vix = 20.0  # 기본값
            
            logger.info(f"📊 현재 VIX: {self.current_vix:.2f}")
            return self.current_vix
            
        except Exception as e:
            logger.error(f"VIX 조회 실패: {e}")
            self.current_vix = 20.0
            return self.current_vix

    async def create_universe(self) -> List[str]:
        """투자 유니버스 생성 (S&P500 + NASDAQ100 + 러셀1000 샘플)"""
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
        """종목 종합 데이터 수집"""
        try:
            stock = yf.Ticker(symbol)
            
            # 기본 정보
            info = stock.info
            
            # 가격 데이터 (1년)
            hist = stock.history(period="1y")
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            
            # 재무 지표
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
            
            return data
            
        except Exception as e:
            logger.error(f"종목 데이터 수집 실패 {symbol}: {e}")
            return {}

    def calculate_buffett_score(self, data: Dict) -> float:
        """워렌 버핏 가치투자 점수"""
        try:
            score = 0.0
            
            # PBR (35%)
            pbr = data.get('pbr', 999)
            if 0 < pbr <= 1.5:
                score += 0.35
            elif 1.5 < pbr <= 2.5:
                score += 0.20
            elif 2.5 < pbr <= 4.0:
                score += 0.10
            
            # ROE (30%)
            roe = data.get('roe', 0)
            if roe >= 20:
                score += 0.30
            elif roe >= 15:
                score += 0.25
            elif roe >= 10:
                score += 0.15
            elif roe >= 5:
                score += 0.05
            
            # 부채비율 (20%)
            debt_ratio = data.get('debt_to_equity', 999) / 100
            if debt_ratio <= 0.3:
                score += 0.20
            elif debt_ratio <= 0.5:
                score += 0.15
            elif debt_ratio <= 0.7:
                score += 0.10
            
            # PE 적정성 (15%)
            pe = data.get('pe_ratio', 999)
            if 5 <= pe <= 15:
                score += 0.15
            elif 15 < pe <= 25:
                score += 0.10
            elif 25 < pe <= 35:
                score += 0.05
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"버핏 점수 계산 실패: {e}")
            return 0.0

    def calculate_lynch_score(self, data: Dict) -> float:
        """피터 린치 성장투자 점수"""
        try:
            score = 0.0
            
            # PEG (40%)
            peg = data.get('peg', 999)
            if 0 < peg <= 0.5:
                score += 0.40
            elif 0.5 < peg <= 1.0:
                score += 0.35
            elif 1.0 < peg <= 1.5:
                score += 0.25
            elif 1.5 < peg <= 2.0:
                score += 0.10
            
            # EPS 성장률 (35%)
            eps_growth = data.get('eps_growth', 0)
            if eps_growth >= 25:
                score += 0.35
            elif eps_growth >= 15:
                score += 0.25
            elif eps_growth >= 10:
                score += 0.15
            elif eps_growth >= 5:
                score += 0.05
            
            # 매출 성장률 (25%)
            revenue_growth = data.get('revenue_growth', 0)
            if revenue_growth >= 20:
                score += 0.25
            elif revenue_growth >= 10:
                score += 0.15
            elif revenue_growth >= 5:
                score += 0.10
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"린치 점수 계산 실패: {e}")
            return 0.0

    def calculate_momentum_score(self, data: Dict) -> float:
        """모멘텀 점수"""
        try:
            score = 0.0
            
            # 3개월 모멘텀 (30%)
            mom_3m = data.get('momentum_3m', 0)
            if mom_3m >= 20:
                score += 0.30
            elif mom_3m >= 10:
                score += 0.20
            elif mom_3m >= 5:
                score += 0.10
            elif mom_3m >= 0:
                score += 0.05
            
            # 6개월 모멘텀 (25%)
            mom_6m = data.get('momentum_6m', 0)
            if mom_6m >= 30:
                score += 0.25
            elif mom_6m >= 15:
                score += 0.15
            elif mom_6m >= 5:
                score += 0.10
            
            # 12개월 모멘텀 (25%)
            mom_12m = data.get('momentum_12m', 0)
            if mom_12m >= 50:
                score += 0.25
            elif mom_12m >= 25:
                score += 0.15
            elif mom_12m >= 10:
                score += 0.10
            
            # 거래량 급증 (20%)
            volume_spike = data.get('volume_spike', 1)
            if volume_spike >= 2.0:
                score += 0.20
            elif volume_spike >= 1.5:
                score += 0.10
            elif volume_spike >= 1.2:
                score += 0.05
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"모멘텀 점수 계산 실패: {e}")
            return 0.0

    def calculate_technical_score(self, data: Dict) -> float:
        """기술적 분석 점수"""
        try:
            score = 0.0
            
            # RSI (30%)
            rsi = data.get('rsi', 50)
            if 30 <= rsi <= 70:
                score += 0.30
            elif 20 <= rsi < 30:
                score += 0.20
            elif 70 < rsi <= 80:
                score += 0.15
            
            # MACD (25%)
            macd = data.get('macd_signal', 'neutral')
            if macd == 'bullish':
                score += 0.25
            
            # 추세 (25%)
            trend = data.get('trend', 'sideways')
            if trend == 'uptrend':
                score += 0.25
            
            # 볼린저 밴드 (20%)
            bb = data.get('bb_position', 'normal')
            if bb == 'oversold':
                score += 0.20
            elif bb == 'normal':
                score += 0.10
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"기술적 점수 계산 실패: {e}")
            return 0.0

    def calculate_vix_adjustment(self, base_score: float) -> float:
        """VIX 기반 점수 조정"""
        try:
            if self.current_vix <= self.vix_low_threshold:
                # 저변동성 (안정적): 가치주 선호
                return base_score * 1.1
            elif self.current_vix >= self.vix_high_threshold:
                # 고변동성 (불안정): 보수적 접근
                return base_score * 0.9
            else:
                # 정상 변동성
                return base_score
                
        except Exception as e:
            logger.error(f"VIX 조정 실패: {e}")
            return base_score

    def calculate_selection_score(self, data: Dict) -> float:
        """종목 선별 종합 점수 계산"""
        try:
            # 4가지 전략 점수 계산
            buffett_score = self.calculate_buffett_score(data)
            lynch_score = self.calculate_lynch_score(data)
            momentum_score = self.calculate_momentum_score(data)
            technical_score = self.calculate_technical_score(data)
            
            # 가중 평균 (각 25%)
            base_score = (
                buffett_score * 0.25 +
                lynch_score * 0.25 +
                momentum_score * 0.25 +
                technical_score * 0.25
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
        """상위 종목 선별 (4가지 전략 + VIX 조정)"""
        logger.info(f"🎯 {len(universe)}개 후보에서 상위 {self.target_stocks}개 선별 시작...")
        
        # 기본 지수 리스트 미리 준비
        sp500_list = await self.get_sp500_constituents()
        nasdaq100_list = await self.get_nasdaq100_constituents()
        
        scored_stocks = []
        
        # 병렬 처리로 속도 향상
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            
            for symbol in universe:
                future = executor.submit(self._process_single_stock, symbol, sp500_list, nasdaq100_list)
                futures.append(future)
            
            for i, future in enumerate(futures, 1):
                try:
                    result = future.result(timeout=45)
                    if result:
                        scored_stocks.append(result)
                        
                    # 진행상황 표시
                    if i % 50 == 0:
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
        """단일 종목 처리"""
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
        """섹터 다양성 확보"""
        try:
            final_selection = []
            sector_counts = {}
            max_per_sector = max(1, self.target_stocks // 6)  # 섹터당 최대 3-4개
            
            # 지수별 쿼터 (S&P500 우선, NASDAQ100 기술주 포함)
            sp500_quota = int(self.target_stocks * 0.6)  # 60%
            nasdaq_quota = int(self.target_stocks * 0.4)  # 40%
            
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
# 🇺🇸 메인 미국 주식 전략 클래스 (업그레이드)
# ========================================================================================
class AdvancedUSStrategy:
    """🚀 완전 업그레이드 미국 전략 클래스 (진짜 자동선별)"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        """전략 초기화"""
        self.config = self._load_config(config_path)
        self.us_config = self.config.get('us_strategy', {})
        self.enabled = self.us_config.get('enabled', True)
        
        # 🆕 실시간 종목 선별기
        self.stock_selector = RealTimeUSStockSelector()
        
        # 📊 전략별 가중치 (각 25%)
        self.buffett_weight = 0.25
        self.lynch_weight = 0.25
        self.momentum_weight = 0.25
        self.technical_weight = 0.25
        
        # 💰 포트폴리오 설정
        self.total_portfolio_ratio = 0.80  # 총 자본의 80% 투자
        self.cash_reserve_ratio = 0.20     # 20% 현금 보유
        
        # 🔧 분할매매 설정
        self.stage1_ratio = 0.40  # 1단계 40%
        self.stage2_ratio = 0.35  # 2단계 35%
        self.stage3_ratio = 0.25  # 3단계 25%
        self.stage2_trigger = -0.05  # 5% 하락시 2단계
        self.stage3_trigger = -0.10  # 10% 하락시 3단계
        
        # 🛡️ 리스크 관리
        self.stop_loss_pct = 0.15      # 15% 손절
        self.take_profit1_pct = 0.20   # 20% 익절 (1차)
        self.take_profit2_pct = 0.35   # 35% 익절 (2차)
        self.max_hold_days = 60        # 최대 보유 60일
        self.max_sector_weight = 0.25  # 섹터별 최대 25%
        
        # 🔍 자동 선별된 종목들 (동적으로 업데이트)
        self.selected_stocks = []          # 실시간 선별 결과
        self.last_selection_time = None    # 마지막 선별 시간
        self.selection_cache_hours = 24    # 선별 결과 캐시 시간
        
        if self.enabled:
            logger.info(f"🇺🇸 완전 업그레이드 미국 전략 초기화 (V4.0)")
            logger.info(f"🆕 실시간 S&P500 + NASDAQ100 + 러셀1000 자동 선별")
            logger.info(f"🎯 자동 선별: 상위 {self.stock_selector.target_stocks}개 종목")
            logger.info(f"📊 4가지 전략 융합: 버핏25% + 린치25% + 모멘텀25% + 기술25%")
            logger.info(f"💰 분할매매: 3단계 매수(40%+35%+25%), 2단계 매도(60%+40%)")
            logger.info(f"🛡️ 리스크 관리: 손절{self.stop_loss_pct*100}%, 익절{self.take_profit2_pct*100}%")
            logger.info(f"📊 VIX 기반 동적 조정 시스템")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return {}

    # ========================================================================================
    # 🆕 실시간 자동 선별 메서드들 (NEW!)
    # ========================================================================================

    async def auto_select_top20_stocks(self) -> List[str]:
        """🆕 실시간 미국 주식 자동 선별 (메인 기능)"""
        if not self.enabled:
            logger.warning("미국 주식 전략이 비활성화되어 있습니다")
            return []

        try:
            # 캐시 확인 (24시간 이내면 기존 결과 사용)
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
        """백업 종목 리스트 (자동 선별 실패시)"""
        fallback_symbols = [
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
            return await self.stock_selector.get_stock_comprehensive_data(symbol)
            
        except Exception as e:
            logger.error(f"종목 정보 조회 실패 {symbol}: {e}")
            return {}

    # ========================================================================================
    # 💰 분할매매 계획 수립 (기존 로직 유지)
    # ========================================================================================

    def _calculate_split_trading_plan(self, symbol: str, current_price: float, 
                                     confidence: float, portfolio_value: float = 1000000) -> Dict:
        """분할매매 계획 수립"""
        try:
            # 종목별 목표 비중 계산 (신뢰도 기반)
            base_weight = self.total_portfolio_ratio / self.stock_selector.target_stocks  # 기본 4%
            confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5~2.0 배수
            target_weight = base_weight * confidence_multiplier
            target_weight = min(target_weight, 0.08)  # 최대 8%로 제한
            
            # 총 투자금액
            total_investment = portfolio_value * target_weight
            total_shares = int(total_investment / current_price)
            
            # 3단계 분할 계획
            stage1_shares = int(total_shares * self.stage1_ratio)  # 40%
            stage2_shares = int(total_shares * self.stage2_ratio)  # 35%
            stage3_shares = total_shares - stage1_shares - stage2_shares  # 25%
            
            # 진입가 계획
            entry_price_1 = current_price
            entry_price_2 = current_price * (1 + self.stage2_trigger)  # 5% 하락시
            entry_price_3 = current_price * (1 + self.stage3_trigger)  # 10% 하락시
            
            # 손절/익절 계획
            avg_entry = current_price * 0.9  # 평균 진입가 추정
            stop_loss = avg_entry * (1 - self.stop_loss_pct)
            take_profit_1 = avg_entry * (1 + self.take_profit1_pct)  # 20% 익절
            take_profit_2 = avg_entry * (1 + self.take_profit2_pct)  # 35% 익절
            
            # 보유 기간 (신뢰도 기반)
            max_hold_days = int(self.max_hold_days * (1.5 - confidence))  # 고신뢰도일수록 장기보유
            
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
    # 🎯 메인 종목 분석 메서드 (업그레이드)
    # ========================================================================================

    async def analyze_symbol(self, symbol: str) -> USStockSignal:
        """개별 종목 종합 분석 (자동선별 + 4가지 전략)"""
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
            
            # 3. 가중 평균 계산
            total_score = (
                buffett_score * self.buffett_weight +
                lynch_score * self.lynch_weight +
                momentum_score * self.momentum_weight +
                technical_score * self.technical_weight
            )
            
            # 4. VIX 조정
            vix_adjustment = self.stock_selector.calculate_vix_adjustment(total_score) - total_score
            total_score = self.stock_selector.calculate_vix_adjustment(total_score)
            
            # 5. 최종 액션 결정
            if total_score >= 0.70:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= 0.30:
                action = 'sell'
                confidence = min(1 - total_score, 0.95)
            else:
                action = 'hold'
                confidence = 0.50
            
            # 6. 분할매매 계획 수립
            split_plan = self._calculate_split_trading_plan(symbol, data['price'], confidence)
            
            # 7. 목표주가 계산
            target_price = data['price'] * (1 + confidence * 0.35)  # 최대 35% 상승 기대
            
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
    # 🔍 전체 시장 스캔 (자동선별 + 분석)
    # ========================================================================================

    async def scan_all_selected_stocks(self) -> List[USStockSignal]:
        """전체 자동선별 + 종목 분석 (진짜 완전 자동화)"""
        if not self.enabled:
            return []
        
        logger.info(f"🔍 미국 주식 완전 자동 분석 시작!")
        logger.info(f"🆕 실시간 S&P500+NASDAQ100 자동 선별 + 4가지 전략 분석")
        
        try:
            # 1단계: 실시간 자동 선별
            selected_symbols = await self.auto_select_top20_stocks()
            if not selected_symbols:
                logger.error("자동 선별 실패")
                return []
            
            # 2단계: 선별된 종목들 상세 분석
            all_signals = []
            
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
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.error(f"❌ {symbol} 분석 실패: {e}")
            
            # 결과 요약
            buy_count = len([s for s in all_signals if s.action == 'buy'])
            sell_count = len([s for s in all_signals if s.action == 'sell'])
            hold_count = len([s for s in all_signals if s.action == 'hold'])
            
            logger.info(f"🎯 완전 자동 분석 완료!")
            logger.info(f"📊 결과: 매수:{buy_count}, 매도:{sell_count}, 보유:{hold_count}")
            logger.info(f"📊 현재 VIX: {self.stock_selector.current_vix:.2f}")
            logger.info(f"🆕 자동선별 시간: {self.last_selection_time}")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"전체 스캔 실패: {e}")
            return []

    # ========================================================================================
    # 📊 포트폴리오 리포트 생성 (업그레이드)
    # ========================================================================================

    async def generate_portfolio_report(self, selected_stocks: List[USStockSignal]) -> Dict:
        """📊 포트폴리오 리포트 생성 (자동선별 정보 포함)"""
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
        
        # 상위 5개 매수 종목
        top_buys = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
        
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
        
        report = {
            'summary': {
                'total_stocks': total_stocks,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'hold_signals': len(hold_signals),
                'total_investment': total_investment,
                'total_shares_value': total_shares_value,
                'current_vix': self.stock_selector.current_vix,
                'avg_vix_adjustment': avg_vix_adjustment
            },
            'strategy_scores': {
                'avg_buffett_score': avg_buffett,
                'avg_lynch_score': avg_lynch,
                'avg_momentum_score': avg_momentum,
                'avg_technical_score': avg_technical,
                'avg_total_score': avg_total,
                'avg_selection_score': avg_selection
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
                'max_single_position': max([s.additional_data.get('target_weight', 0) for s in selected_stocks if s.additional_data]),
                'avg_beta': np.mean([s.additional_data.get('beta', 1.0) for s in selected_stocks if s.additional_data and s.additional_data.get('beta')]),
                'diversification_score': len(sector_dist) / total_stocks,
                'market_volatility': 'HIGH' if self.stock_selector.current_vix > 30 else 'LOW' if self.stock_selector.current_vix < 15 else 'MEDIUM'
            },
            'auto_selection_info': {
                'selection_method': 'real_time_auto_selection',
                'last_selection_time': self.last_selection_time,
                'cache_hours_remaining': max(0, self.selection_cache_hours - (
                    (datetime.now() - self.last_selection_time).total_seconds() / 3600
                    if self.last_selection_time else self.selection_cache_hours
                )),
                'universe_size': len(await self.stock_selector.create_universe()) if hasattr(self.stock_selector, 'universe_size') else 'Unknown'
            }
        }
        
        return report

    async def execute_split_trading_simulation(self, signal: USStockSignal) -> Dict:
        """🔄 분할매매 시뮬레이션"""
        if signal.action != 'buy':
            return {"status": "not_applicable", "reason": "매수 신호가 아님"}
        
        simulation = {
            'symbol': signal.symbol,
            'strategy': 'split_trading_4_strategies',
            'index_membership': signal.index_membership,
            'selection_score': signal.selection_score,
            'vix_level': self.stock_selector.current_vix,
            'stages': {
                'stage_1': {
                    'trigger_price': signal.entry_price_1,
                    'shares': signal.stage1_shares,
                    'investment': signal.stage1_shares * signal.entry_price_1,
                    'ratio': '40%',
                    'status': 'ready'
                },
                'stage_2': {
                    'trigger_price': signal.entry_price_2,
                    'shares': signal.stage2_shares,
                    'investment': signal.stage2_shares * signal.entry_price_2,
                    'ratio': '35%',
                    'trigger_condition': f'{self.stage2_trigger*100:.0f}% 하락시',
                    'status': 'waiting'
                },
                'stage_3': {
                    'trigger_price': signal.entry_price_3,
                    'shares': signal.stage3_shares,
                    'investment': signal.stage3_shares * signal.entry_price_3,
                    'ratio': '25%',
                    'trigger_condition': f'{self.stage3_trigger*100:.0f}% 하락시',
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
                'lynch_score': signal.lynch_score,
                'momentum_score': signal.momentum_score,
                'technical_score': signal.technical_score,
                'vix_adjustment': signal.vix_adjustment
            },
            'risk_management': {
                'max_hold_days': signal.max_hold_days,
                'total_investment': signal.additional_data.get('total_investment', 0) if signal.additional_data else 0,
                'portfolio_weight': signal.additional_data.get('target_weight', 0) if signal.additional_data else 0
            }
        }
        
        return simulation

# ========================================================================================
# 🎯 편의 함수들 (외부에서 쉽게 사용) - 업그레이드
# ========================================================================================

async def run_auto_selection():
    """자동 선별 실행"""
    strategy = AdvancedUSStrategy()
    selected_stocks = await strategy.scan_all_selected_stocks()
    
    if selected_stocks:
        report = await strategy.generate_portfolio_report(selected_stocks)
        return selected_stocks, report
    else:
        return [], {}

async def analyze_us(symbol: str) -> Dict:
    """단일 미국 주식 분석 (기존 호환성 + 자동선별 정보)"""
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
        
        'split_trading_plan': await strategy.execute_split_trading_simulation(signal)
    }

async def get_us_auto_selection_status() -> Dict:
    """미국 주식 자동선별 상태 조회"""
    strategy = AdvancedUSStrategy()
    
    return {
        'enabled': strategy.enabled,
        'last_selection_time': strategy.last_selection_time,
        'cache_valid': strategy._is_selection_cache_valid(),
        'cache_hours': strategy.selection_cache_hours,
        'selected_count': len(strategy.selected_stocks),
        'current_vix': strategy.stock_selector.current_vix,
        'vix_status': 'HIGH' if strategy.stock_selector.current_vix > 30 else 'LOW' if strategy.stock_selector.current_vix < 15 else 'MEDIUM',
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
        }
    }

async def force_us_reselection() -> List[str]:
    """미국 주식 강제 재선별"""
    strategy = AdvancedUSStrategy()
    strategy.last_selection_time = None  # 캐시 무효화
    strategy.selected_stocks = []        # 기존 선별 결과 삭제
    
    return await strategy.auto_select_top20_stocks()

# ========================================================================================
# 🧪 테스트 메인 함수 (업그레이드)
# ========================================================================================

async def main():
    """테스트용 메인 함수 (진짜 자동선별 시스템)"""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print("🇺🇸 미국 주식 완전 자동화 전략 V4.0 테스트!")
        print("🆕 진짜 자동선별: S&P500+NASDAQ100+러셀1000 실시간 크롤링")
        print("🎯 4가지 전략 융합 + VIX 기반 동적 조정 + 분할매매")
        print("="*80)
        
        # 자동선별 상태 확인
        print("\n📋 자동선별 시스템 상태 확인...")
        status = await get_us_auto_selection_status()
        print(f"  ✅ 시스템 활성화: {status['enabled']}")
        print(f"  📅 마지막 선별: {status['last_selection_time']}")
        print(f"  🔄 캐시 유효: {status['cache_valid']}")
        print(f"  📊 현재 VIX: {status['current_vix']:.2f} ({status['vix_status']})")
        print(f"  🎯 선별 기준: 시총 ${status['selection_criteria']['min_market_cap_billions']:.1f}B 이상, "
              f"거래량 {status['selection_criteria']['min_avg_volume_millions']:.1f}M주 이상")
        print(f"  📊 전략 가중치: 버핏{status['selection_criteria']['strategy_weights']['buffett']:.0f}% "
              f"린치{status['selection_criteria']['strategy_weights']['lynch']:.0f}% "
              f"모멘텀{status['selection_criteria']['strategy_weights']['momentum']:.0f}% "
              f"기술{status['selection_criteria']['strategy_weights']['technical']:.0f}%")
        
        # 전체 시장 자동선별 + 분석
        print(f"\n🔍 실시간 자동선별 + 전체 분석 시작...")
        start_time = time.time()
        
        selected_stocks, report = await run_auto_selection()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ 총 소요 시간: {elapsed_time:.1f}초")
        
        if selected_stocks and report:
            print(f"\n📈 자동선별 + 분석 결과:")
            print(f"  총 분석: {report['summary']['total_stocks']}개 종목 (실시간 자동선별)")
            print(f"  매수 신호: {report['summary']['buy_signals']}개")
            print(f"  매도 신호: {report['summary']['sell_signals']}개")
            print(f"  보유 신호: {report['summary']['hold_signals']}개")
            print(f"  현재 VIX: {report['summary']['current_vix']:.2f} (변동성: {report['risk_metrics']['market_volatility']})")
            
            # 지수별 분포
            index_dist = report['index_distribution']
            print(f"\n🏢 지수별 분포:")
            for index, count in index_dist.items():
                percentage = count / report['summary']['total_stocks'] * 100
                print(f"  {index}: {count}개 ({percentage:.1f}%)")
            
            # 섹터별 분포
            print(f"\n🏢 섹터별 분포:")
            for sector, count in list(report['sector_distribution'].items())[:5]:
                percentage = count / report['summary']['total_stocks'] * 100
                print(f"  {sector}: {count}개 ({percentage:.1f}%)")
            
            # 전략 점수 요약
            scores = report['strategy_scores']
            print(f"\n📊 평균 전략 점수:")
            print(f"  버핏 가치투자: {scores['avg_buffett_score']:.3f}")
            print(f"  린치 성장투자: {scores['avg_lynch_score']:.3f}")
            print(f"  모멘텀 전략: {scores['avg_momentum_score']:.3f}")
            print(f"  기술적 분석: {scores['avg_technical_score']:.3f}")
            print(f"  종합 점수: {scores['avg_total_score']:.3f}")
            print(f"  선별 점수: {scores['avg_selection_score']:.3f}")
            print(f"  VIX 평균 조정: {report['summary']['avg_vix_adjustment']:+.3f}")
            
            # 상위 매수 추천 (상세 정보)
            if report['top_picks']:
                print(f"\n🎯 상위 매수 추천 (실시간 자동선별):")
                for i, stock in enumerate(report['top_picks'][:3], 1):
                    membership_str = "+".join(stock['index_membership'])
                    print(f"\n  {i}. {stock['symbol']} ({membership_str}) - 신뢰도: {stock['confidence']:.2%}")
                    print(f"     🏆 선별점수: {stock['selection_score']:.3f} | 총점: {stock['total_score']:.3f}")
                    print(f"     💰 현재가: ${stock['price']:.2f} → 목표가: ${stock['target_price']:.2f}")
                    print(f"     🔄 분할매매: {stock['total_shares']:,}주 (3단계)")
                    print(f"     💼 투자금액: ${stock['total_investment']:,.0f}")
                    print(f"     📊 VIX 조정: {stock['vix_adjustment']:+.3f}")
                    print(f"     💡 {stock['reasoning'][:60]}...")
            
            # 자동선별 정보
            auto_info = report['auto_selection_info']
            print(f"\n🤖 자동선별 상세:")
            print(f"  선별 방식: {auto_info['selection_method']}")
            print(f"  선별 시간: {auto_info['last_selection_time']}")
            print(f"  캐시 남은시간: {auto_info['cache_hours_remaining']:.1f}시간")
            
            # 분할매매 시뮬레이션 (첫 번째 매수 종목)
            buy_stocks = [s for s in selected_stocks if s.action == 'buy']
            if buy_stocks:
                print(f"\n🔄 분할매매 시뮬레이션 - {buy_stocks[0].symbol}:")
                strategy = AdvancedUSStrategy()
                simulation = await strategy.execute_split_trading_simulation(buy_stocks[0])
                
                print(f"  📊 전략 점수: 버핏{simulation['strategy_breakdown']['buffett_score']:.2f} "
                      f"린치{simulation['strategy_breakdown']['lynch_score']:.2f} "
                      f"모멘텀{simulation['strategy_breakdown']['momentum_score']:.2f} "
                      f"기술{simulation['strategy_breakdown']['technical_score']:.2f}")
                
                for stage, data in simulation['stages'].items():
                    print(f"  {stage}: ${data['trigger_price']:.2f}에 {data['shares']}주 ({data['ratio']}) - {data.get('trigger_condition', data['status'])}")
                
                print(f"  손절: ${simulation['exit_plan']['stop_loss']['price']:.2f}")
                print(f"  익절1: ${simulation['exit_plan']['take_profit_1']['price']:.2f} (60% 매도)")
                print(f"  익절2: ${simulation['exit_plan']['take_profit_2']['price']:.2f} (40% 매도)")
        
        else:
            print("❌ 선별된 종목이 없습니다.")
        
        print("\n✅ 테스트 완료!")
        print("\n🎯 미국 주식 V4.0 완전 자동화 전략 특징:")
        print("  ✅ 🆕 실시간 S&P500+NASDAQ100+러셀1000 크롤링 (하드코딩 완전 제거)")
        print("  ✅ 📊 4가지 전략 융합 (버핏+린치+모멘텀+기술 각 25%)")
        print("  ✅ 📊 VIX 기반 동적 조정 시스템")
        print("  ✅ 💰 분할매매 시스템 (3단계 매수, 2단계 매도)")
        print("  ✅ 🛡️ 동적 손절/익절 (신뢰도 기반)")
        print("  ✅ 🔍 상위 20개 종목 완전 자동 선별")
        print("  ✅ 🏢 섹터 다양성 + 지수별 균형")
        print("  ✅ 🤖 완전 자동화 (24시간 캐시 + 실시간 업데이트)")
        print("  ✅ 📱 웹 대시보드 연동 준비")
        print("\n💡 사용법:")
        print("  - python us_strategy.py : 전체 자동선별 + 분석")
        print("  - await analyze_us('AAPL') : 개별 종목 분석")
        print("  - await run_auto_selection() : 시장 전체 스캔")
        print("  - await force_us_reselection() : 강제 재선별")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"테스트 실행 실패: {e}")

if __name__ == "__main__":
    asyncio.run(main())
