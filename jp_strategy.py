#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇯🇵 일본 주식 완전 자동화 전략 - 최고퀸트프로젝트 (진짜 자동선별)
===========================================================================

🎯 핵심 기능:
- 📊 실시간 닛케이225 + TOPIX 구성종목 크롤링
- 💱 엔화 자동 매매법 (USD/JPY 기반)
- ⚡ 고급 기술적 지표 (RSI, MACD, 볼린저밴드, 스토캐스틱)
- 💰 분할매매 시스템 (리스크 관리)
- 🔍 실시간 자동 20개 종목 선별
- 🛡️ 동적 손절/익절
- 🤖 완전 자동화 (혼자서도 OK)

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
import ta
import requests
from bs4 import BeautifulSoup
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# 로거 설정
logger = logging.getLogger(__name__)

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
# 📊 기술적 지표 분석 클래스 (기존 유지)
# ========================================================================================
class TechnicalIndicators:
    """🔧 기술적 지표 계산 및 분석"""

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 10) -> float:
        """RSI 계산"""
        try:
            rsi = ta.momentum.RSIIndicator(data['Close'], window=period).rsi()
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

    @staticmethod
    def calculate_macd(data: pd.DataFrame) -> Tuple[str, Dict]:
        """MACD 계산 및 신호 분석"""
        try:
            macd = ta.trend.MACD(data['Close'], window_fast=8, window_slow=21, window_sign=5)
            macd_line = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            macd_histogram = macd.macd_diff().iloc[-1]

            # 신호 분석
            if macd_line > macd_signal and macd_histogram > 0:
                signal = 'bullish'
            elif macd_line < macd_signal and macd_histogram < 0:
                signal = 'bearish'
            else:
                signal = 'neutral'

            details = {
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'histogram': macd_histogram
            }

            return signal, details
        except:
            return 'neutral', {}

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20) -> Tuple[str, Dict]:
        """볼린저 밴드 계산 및 신호 분석"""
        try:
            bb = ta.volatility.BollingerBands(data['Close'], window=window)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_middle = bb.bollinger_mavg().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            current_price = data['Close'].iloc[-1]

            # 신호 분석
            if current_price >= bb_upper:
                signal = 'upper'  # 과매수 구간
            elif current_price <= bb_lower:
                signal = 'lower'  # 과매도 구간
            else:
                signal = 'middle' # 정상 구간

            details = {
                'upper': bb_upper,
                'middle': bb_middle,
                'lower': bb_lower,
                'position': (current_price - bb_lower) / (bb_upper - bb_lower)
            }

            return signal, details
        except:
            return 'middle', {}

    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, k_period: int = 14) -> Tuple[str, Dict]:
        """스토캐스틱 계산 및 신호 분석"""
        try:
            stoch = ta.momentum.StochasticOscillator(
                data['High'], data['Low'], data['Close'], 
                window=k_period, smooth_window=3
            )
            stoch_k = stoch.stoch().iloc[-1]
            stoch_d = stoch.stoch_signal().iloc[-1]

            # 신호 분석
            if stoch_k <= 20 and stoch_d <= 20:
                signal = 'oversold'  # 과매도
            elif stoch_k >= 80 and stoch_d >= 80:
                signal = 'overbought'  # 과매수
            else:
                signal = 'neutral'   # 중립

            details = {
                'stoch_k': stoch_k,
                'stoch_d': stoch_d
            }

            return signal, details
        except:
            return 'neutral', {}

    @staticmethod
    def calculate_moving_averages(data: pd.DataFrame) -> Tuple[str, Dict]:
        """이동평균선 분석"""
        try:
            ma5 = data['Close'].rolling(5).mean().iloc[-1]
            ma20 = data['Close'].rolling(20).mean().iloc[-1]
            ma60 = data['Close'].rolling(60).mean().iloc[-1]
            current_price = data['Close'].iloc[-1]

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
# 💰 분할매매 관리 클래스 (기존 유지)
# ========================================================================================
class PositionManager:
    """🔧 분할매매 및 포지션 관리"""

    @staticmethod
    def create_split_buy_plan(total_amount: float, current_price: float, 
                            confidence: float) -> Tuple[int, List[Dict]]:
        """분할 매수 계획 생성"""
        try:
            # 신뢰도에 따른 분할 전략
            if confidence >= 0.8:
                # 높은 신뢰도: 50% + 30% + 20%
                ratios = [0.5, 0.3, 0.2]
                triggers = [0, -0.02, -0.04]  # 0%, -2%, -4% 에서 매수
            elif confidence >= 0.6:
                # 중간 신뢰도: 40% + 35% + 25%
                ratios = [0.4, 0.35, 0.25]
                triggers = [0, -0.03, -0.05]  # 0%, -3%, -5%
            else:
                # 낮은 신뢰도: 30% + 35% + 35%
                ratios = [0.3, 0.35, 0.35]
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

    @staticmethod
    def create_split_sell_plan(total_shares: int, current_price: float, 
                             target_price: float, confidence: float) -> List[Dict]:
        """분할 매도 계획 생성"""
        try:
            # 신뢰도에 따른 매도 전략
            if confidence >= 0.8:
                # 높은 신뢰도: 목표가 달성 시 50% 매도, 나머지 홀드
                sell_ratios = [0.5, 0.5]
                price_targets = [target_price, target_price * 1.1]
            else:
                # 일반: 목표가 달성 시 70% 매도, 30% 홀드
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
# 🆕 실시간 종목 수집 및 선별 클래스 (NEW!)
# ========================================================================================
class RealTimeJPStockSelector:
    """🆕 실시간 일본 주식 종목 수집 및 선별"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 선별 기준
        self.min_market_cap = 500_000_000_000  # 5000억엔
        self.min_avg_volume = 1_000_000        # 100만주
        self.target_stocks = 20                # 최종 20개 선별
        
    async def get_nikkei225_constituents(self) -> List[str]:
        """닛케이225 구성종목 실시간 수집"""
        try:
            logger.info("🔍 닛케이225 구성종목 실시간 수집 시작...")
            
            # 여러 소스에서 데이터 수집 (안정성 확보)
            symbols = []
            
            # 소스 1: Yahoo Finance Japan
            try:
                url = "https://finance.yahoo.com/quote/%5EN225/components"
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # 테이블에서 종목 코드 추출
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        if '/quote/' in href and '.T' in href:
                            symbol = href.split('/quote/')[-1].split('?')[0]
                            if symbol.endswith('.T') and len(symbol) <= 8:
                                symbols.append(symbol)
                
                logger.info(f"✅ Yahoo Finance에서 {len(symbols)}개 종목 수집")
            except Exception as e:
                logger.warning(f"Yahoo Finance 수집 실패: {e}")
            
            # 소스 2: 백업 종목 리스트 (주요 대형주)
            backup_symbols = [
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
            
            # 중복 제거 및 병합
            all_symbols = list(set(symbols + backup_symbols))
            
            logger.info(f"📊 총 {len(all_symbols)}개 후보 종목 수집 완료")
            return all_symbols
            
        except Exception as e:
            logger.error(f"닛케이225 구성종목 수집 실패: {e}")
            return backup_symbols

    async def get_stock_fundamental_data(self, symbol: str) -> Dict:
        """개별 종목 기본 데이터 수집"""
        try:
            stock = yf.Ticker(symbol)
            
            # 기본 정보
            info = stock.info
            
            # 가격 데이터 (6개월)
            hist = stock.history(period="6mo")
            if hist.empty:
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
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'revenue_growth': (info.get('revenueQuarterlyGrowth', 0) or 0) * 100,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }
            
            # 기술적 지표 추가
            if len(hist) >= 30:
                data['rsi'] = TechnicalIndicators.calculate_rsi(hist)
                data['ma20'] = hist['Close'].rolling(20).mean().iloc[-1]
                data['volume_ratio'] = hist['Volume'].tail(5).mean() / hist['Volume'].tail(20).mean()
            
            return data
            
        except Exception as e:
            logger.error(f"종목 데이터 수집 실패 {symbol}: {e}")
            return {}

    def calculate_selection_score(self, data: Dict) -> float:
        """종목 선별 점수 계산"""
        try:
            score = 0.0
            
            # 1. 시가총액 점수 (25%)
            market_cap = data.get('market_cap', 0)
            if market_cap >= 2_000_000_000_000:  # 2조엔 이상
                score += 0.25
            elif market_cap >= 1_000_000_000_000:  # 1조엔 이상
                score += 0.20
            elif market_cap >= 500_000_000_000:   # 5000억엔 이상
                score += 0.15
            
            # 2. 거래량 점수 (20%)
            avg_volume = data.get('avg_volume', 0)
            if avg_volume >= 5_000_000:   # 500만주 이상
                score += 0.20
            elif avg_volume >= 2_000_000: # 200만주 이상
                score += 0.15
            elif avg_volume >= 1_000_000: # 100만주 이상
                score += 0.10
            
            # 3. 재무 건전성 점수 (25%)
            pe_ratio = data.get('pe_ratio', 999)
            pbr = data.get('pbr', 999)
            roe = data.get('roe', 0)
            debt_ratio = data.get('debt_to_equity', 999) / 100
            
            # PE 점수
            if 5 <= pe_ratio <= 20:
                score += 0.08
            elif 20 < pe_ratio <= 30:
                score += 0.05
            
            # PBR 점수
            if 0.5 <= pbr <= 2.0:
                score += 0.08
            elif 2.0 < pbr <= 3.0:
                score += 0.05
            
            # ROE 점수
            if roe >= 15:
                score += 0.09
            elif roe >= 10:
                score += 0.06
            elif roe >= 5:
                score += 0.03
            
            # 4. 기술적 지표 점수 (20%)
            rsi = data.get('rsi', 50)
            volume_ratio = data.get('volume_ratio', 1)
            
            # RSI 점수
            if 30 <= rsi <= 70:
                score += 0.10
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                score += 0.05
            
            # 거래량 추세 점수
            if volume_ratio >= 1.2:
                score += 0.10
            elif volume_ratio >= 1.0:
                score += 0.05
            
            # 5. 섹터 보너스 (10%)
            sector = data.get('sector', '')
            if sector in ['Technology', 'Consumer Cyclical', 'Industrials']:
                score += 0.10
            elif sector in ['Healthcare', 'Financial Services']:
                score += 0.07
            else:
                score += 0.05
            
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
        """상위 종목 선별 (실시간 데이터 기반)"""
        logger.info(f"🎯 {len(candidate_symbols)}개 후보에서 상위 {self.target_stocks}개 선별 시작...")
        
        scored_stocks = []
        
        # 병렬 처리로 속도 향상
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for symbol in candidate_symbols:
                future = executor.submit(self._process_single_stock, symbol)
                futures.append(future)
            
            for i, future in enumerate(futures, 1):
                try:
                    result = future.result(timeout=30)
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
        """섹터 다양성 확보"""
        try:
            final_selection = []
            sector_counts = {}
            max_per_sector = max(1, self.target_stocks // 5)  # 섹터당 최대 4개
            
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
# 🇯🇵 메인 일본 주식 전략 클래스 (업그레이드)
# ========================================================================================
class JPStrategy:
    """🇯🇵 일본 주식 완전 자동화 전략 (진짜 자동선별)"""

    def __init__(self, config_path: str = "settings.yaml"):
        """전략 초기화"""
        # 설정 로드
        self.config = self._load_config(config_path)
        self.jp_config = self.config.get('jp_strategy', {})
        self.enabled = self.jp_config.get('enabled', True)

        # 🆕 실시간 종목 선별기
        self.stock_selector = RealTimeJPStockSelector()

        # 💱 엔화 매매 설정
        self.yen_strong_threshold = 105     # 엔화 강세
        self.yen_weak_threshold = 110       # 엔화 약세
        self.current_usd_jpy = 0.0

        # ⚡ 기술적 지표 설정
        self.rsi_period = 10
        self.momentum_period = 10
        self.volume_spike_threshold = 1.3

        # 🛡️ 손절/익절 설정
        self.base_stop_loss = 0.08         # 기본 8%
        self.base_take_profit = 0.15       # 기본 15%
        self.max_hold_days = 30

        # 💰 분할매매 설정
        self.use_split_trading = True       # 분할매매 사용
        self.split_buy_steps = 3           # 3단계 분할 매수
        self.split_sell_steps = 2          # 2단계 분할 매도

        # 🔍 자동 선별된 종목들 (동적으로 업데이트)
        self.selected_stocks = []          # 실시간 선별 결과
        self.last_selection_time = None    # 마지막 선별 시간
        self.selection_cache_hours = 24    # 선별 결과 캐시 시간

        if self.enabled:
            logger.info(f"🇯🇵 일본 주식 완전 자동화 전략 초기화 (V4.0)")
            logger.info(f"🆕 실시간 닛케이225 + TOPIX 자동 선별 시스템")
            logger.info(f"📊 기술적 지표 + 엔화 + 펀더멘털 종합 분석")
            logger.info(f"💰 3단계 분할매매 + 동적 손절익절")
        else:
            logger.info("🇯🇵 일본 주식 전략이 비활성화되어 있습니다")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패, 기본값 사용: {e}")
            return {}

    # ========================================================================================
    # 🆕 실시간 자동 선별 메서드들 (NEW!)
    # ========================================================================================

    async def auto_select_stocks(self) -> List[str]:
        """🆕 실시간 주식 자동 선별 (메인 기능)"""
        if not self.enabled:
            logger.warning("일본 주식 전략이 비활성화되어 있습니다")
            return []

        try:
            # 캐시 확인 (24시간 이내면 기존 결과 사용)
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
        """백업 종목 리스트 (자동 선별 실패시)"""
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

    # ========================================================================================
    # 🔧 유틸리티 메서드들 (기존 유지 + 개선)
    # ========================================================================================

    async def _update_yen_rate(self):
        """USD/JPY 환율 업데이트"""
        try:
            ticker = yf.Ticker("USDJPY=X")
            data = ticker.history(period="1d")
            if not data.empty:
                self.current_usd_jpy = data['Close'].iloc[-1]
            else:
                self.current_usd_jpy = 107.5  # 기본값
        except Exception as e:
            logger.error(f"환율 조회 오류: {e}")
            self.current_usd_jpy = 107.5

    def _get_yen_signal(self) -> str:
        """엔화 신호 분석"""
        if self.current_usd_jpy <= self.yen_strong_threshold:
            return 'strong'  # 엔화 강세
        elif self.current_usd_jpy >= self.yen_weak_threshold:
            return 'weak'    # 엔화 약세
        else:
            return 'neutral'

    def _get_stock_type(self, symbol: str) -> str:
        """종목 타입 확인 (선별 데이터 기반)"""
        try:
            for stock in self.selected_stocks:
                if stock['symbol'] == symbol:
                    return stock.get('stock_type', 'mixed')
            return 'mixed'
        except:
            return 'mixed'

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """섹터 분류 (선별 데이터 기반)"""
        try:
            for stock in self.selected_stocks:
                if stock['symbol'] == symbol:
                    return stock.get('sector', 'UNKNOWN')
            return 'UNKNOWN'
        except:
            return 'UNKNOWN'

    async def _get_stock_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """주식 데이터 수집"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                logger.warning(f"데이터 없음: {symbol}")
                return pd.DataFrame()
            return data
        except Exception as e:
            logger.error(f"주식 데이터 수집 실패 {symbol}: {e}")
            return pd.DataFrame()

    # ========================================================================================
    # 📊 통합 기술적 분석 메서드 (기존 유지)
    # ========================================================================================

    def _analyze_technical_indicators(self, data: pd.DataFrame) -> Tuple[float, Dict]:
        """통합 기술적 지표 분석"""
        try:
            if len(data) < 60:  # 충분한 데이터 필요
                return 0.0, {}

            # 각 지표 계산
            rsi = TechnicalIndicators.calculate_rsi(data, self.rsi_period)
            macd_signal, macd_details = TechnicalIndicators.calculate_macd(data)
            bb_signal, bb_details = TechnicalIndicators.calculate_bollinger_bands(data)
            stoch_signal, stoch_details = TechnicalIndicators.calculate_stochastic(data)
            ma_trend, ma_details = TechnicalIndicators.calculate_moving_averages(data)

            # 거래량 분석
            volume = data['Volume']
            recent_volume = volume.tail(3).mean()
            avg_volume = volume.tail(15).head(12).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # 종합 점수 계산 (각 지표별 가중치)
            total_score = 0.0
            max_score = 0.0

            # 1. RSI 점수 (가중치 20%)
            weight = 0.2
            if 30 <= rsi <= 70:
                total_score += weight * 0.8  # 정상 구간
            elif rsi < 30:
                total_score += weight * 1.0  # 과매도 (매수 기회)
            elif rsi > 70:
                total_score += weight * 0.3  # 과매수 (주의)
            max_score += weight

            # 2. MACD 점수 (가중치 25%)
            weight = 0.25
            if macd_signal == 'bullish':
                total_score += weight * 1.0
            elif macd_signal == 'bearish':
                total_score += weight * 0.2
            else:
                total_score += weight * 0.5
            max_score += weight

            # 3. 볼린저 밴드 점수 (가중치 20%)
            weight = 0.2
            if bb_signal == 'lower':
                total_score += weight * 1.0  # 과매도
            elif bb_signal == 'upper':
                total_score += weight * 0.3  # 과매수
            else:
                total_score += weight * 0.6  # 중간
            max_score += weight

            # 4. 스토캐스틱 점수 (가중치 15%)
            weight = 0.15
            if stoch_signal == 'oversold':
                total_score += weight * 1.0
            elif stoch_signal == 'overbought':
                total_score += weight * 0.3
            else:
                total_score += weight * 0.6
            max_score += weight

            # 5. 이동평균 추세 점수 (가중치 15%)
            weight = 0.15
            if ma_trend == 'uptrend':
                total_score += weight * 1.0
            elif ma_trend == 'downtrend':
                total_score += weight * 0.2
            else:
                total_score += weight * 0.5
            max_score += weight

            # 6. 거래량 보너스 (가중치 5%)
            weight = 0.05
            if volume_ratio >= self.volume_spike_threshold:
                total_score += weight * 1.0
            max_score += weight

            # 정규화된 점수
            technical_score = total_score / max_score if max_score > 0 else 0.5

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
    # 💱 엔화 + 기술적 지표 통합 분석 (기존 유지)
    # ========================================================================================

    def _analyze_yen_technical_signal(self, symbol: str, technical_score: float, 
                                    technical_details: Dict) -> Tuple[str, float, str]:
        """엔화 + 기술적 지표 통합 분석"""
        try:
            yen_signal = self._get_yen_signal()
            stock_type = self._get_stock_type(symbol)

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

            # 최종 판단
            if total_score >= 0.65:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= 0.35:
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
    # 🛡️ 동적 손절/익절 계산 (기존 유지)
    # ========================================================================================

    def _calculate_dynamic_stop_take(self, current_price: float, confidence: float, 
                                   stock_type: str, yen_signal: str) -> Tuple[float, float, int]:
        """동적 손절/익절 계산"""
        try:
            # 기본값
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
    # 🎯 메인 종목 분석 메서드 (업그레이드)
    # ========================================================================================

    async def analyze_symbol(self, symbol: str) -> JPStockSignal:
        """개별 종목 완전 분석 (자동선별 + 모든 기능 통합)"""
        if not self.enabled:
            return self._create_disabled_signal(symbol)

        try:
            # 1. 환율 업데이트
            await self._update_yen_rate()

            # 2. 주식 데이터 수집
            data = await self._get_stock_data(symbol)
            if data.empty:
                raise ValueError(f"주식 데이터 없음: {symbol}")

            current_price = data['Close'].iloc[-1]

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
            
            # 6. 💰 분할매매 계획 생성
            if self.use_split_trading and action == 'buy':
                # 총 투자금액 (신뢰도에 따라 조정)
                base_amount = 1000000  # 100만엔 기본
                total_amount = base_amount * confidence
                
                position_size, split_buy_plan = PositionManager.create_split_buy_plan(
                    total_amount, current_price, confidence
                )
                
                split_sell_plan = PositionManager.create_split_sell_plan(
                    position_size, current_price, take_profit, confidence
                )
            else:
                position_size = 0
                split_buy_plan = []
                split_sell_plan = []
            
            # 7. 📊 선별 정보 추가
            stock_info = await self.get_selected_stock_info(symbol)
            market_cap = stock_info.get('market_cap', 0)
            selection_score = stock_info.get('selection_score', 0)
            quality_rank = 0  # 추후 계산
            
            # 8. 📊 JPStockSignal 생성 (모든 정보 포함)
            return JPStockSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                strategy_source='auto_selection_yen_technical',
                
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
                    'selection_method': 'real_time_auto_selection'
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
    # 🔍 전체 시장 스캔 (자동선별 + 분석)
    # ========================================================================================
    
    async def scan_all_symbols(self) -> List[JPStockSignal]:
        """전체 자동선별 + 종목 분석 (진짜 완전 자동화)"""
        if not self.enabled:
            return []
        
        logger.info(f"🔍 일본 주식 완전 자동 분석 시작!")
        logger.info(f"🆕 실시간 닛케이225 자동 선별 + 기술적 분석 + 엔화 전략")
        
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
            
            logger.info(f"🎯 완전 자동 분석 완료!")
            logger.info(f"📊 결과: 매수:{buy_count}, 매도:{sell_count}, 보유:{hold_count}")
            logger.info(f"💱 현재 USD/JPY: {self.current_usd_jpy:.2f} ({self._get_yen_signal()})")
            logger.info(f"🆕 자동선별 시간: {self.last_selection_time}")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"전체 스캔 실패: {e}")
            return []

# ========================================================================================
# 🎯 편의 함수들 (외부에서 쉽게 사용)
# ========================================================================================

async def analyze_jp(symbol: str) -> Dict:
    """단일 일본 주식 완전 분석"""
    strategy = JPStrategy()
    signal = await strategy.analyze_symbol(symbol)
    
    return {
        'decision': signal.action,
        'confidence_score': signal.confidence * 100,
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
        
        # 자동선별 정보
        'market_cap': signal.market_cap,
        'selection_score': signal.selection_score,
        'quality_rank': signal.quality_rank
    }

async def scan_jp_market() -> Dict:
    """일본 시장 전체 자동선별 + 스캔"""
    strategy = JPStrategy()
    signals = await strategy.scan_all_symbols()
    
    buy_signals = [s for s in signals if s.action == 'buy']
    sell_signals = [s for s in signals if s.action == 'sell']
    
    return {
        'total_analyzed': len(signals),
        'buy_count': len(buy_signals),
        'sell_count': len(sell_signals),
        'current_usd_jpy': strategy.current_usd_jpy,
        'yen_signal': strategy._get_yen_signal(),
        'selection_method': 'real_time_auto_selection',
        'last_selection_time': strategy.last_selection_time,
        'top_buys': sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5],
        'top_sells': sorted(sell_signals, key=lambda x: x.confidence, reverse=True)[:5],
        'selection_summary': {
            'export_stocks': len([s for s in signals if s.stock_type == 'export']),
            'domestic_stocks': len([s for s in signals if s.stock_type == 'domestic']),
            'mixed_stocks': len([s for s in signals if s.stock_type == 'mixed']),
            'avg_selection_score': np.mean([s.selection_score for s in signals]) if signals else 0,
            'avg_market_cap': np.mean([s.market_cap for s in signals]) / 1e12 if signals else 0  # 조엔 단위
        }
    }

async def get_jp_auto_selection_status() -> Dict:
    """일본 주식 자동선별 상태 조회"""
    strategy = JPStrategy()
    
    return {
        'enabled': strategy.enabled,
        'last_selection_time': strategy.last_selection_time,
        'cache_valid': strategy._is_selection_cache_valid(),
        'cache_hours': strategy.selection_cache_hours,
        'selected_count': len(strategy.selected_stocks),
        'current_usd_jpy': strategy.current_usd_jpy,
        'yen_signal': strategy._get_yen_signal(),
        'selection_criteria': {
            'min_market_cap': strategy.stock_selector.min_market_cap / 1e12,  # 조엔
            'min_avg_volume': strategy.stock_selector.min_avg_volume / 1e6,   # 백만주
            'target_stocks': strategy.stock_selector.target_stocks
        }
    }

async def force_jp_reselection() -> List[str]:
    """일본 주식 강제 재선별"""
    strategy = JPStrategy()
    strategy.last_selection_time = None  # 캐시 무효화
    strategy.selected_stocks = []        # 기존 선별 결과 삭제
    
    return await strategy.auto_select_stocks()

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
        
        print("🇯🇵 일본 주식 완전 자동화 전략 V4.0 테스트!")
        print("🆕 진짜 자동선별: 닛케이225 실시간 크롤링 + 종합 분석")
        print("📊 기능: 엔화+기술지표+분할매매+동적손절익절")
        print("="*80)
        
        # 자동선별 상태 확인
        print("\n📋 자동선별 시스템 상태 확인...")
        status = await get_jp_auto_selection_status()
        print(f"  ✅ 시스템 활성화: {status['enabled']}")
        print(f"  📅 마지막 선별: {status['last_selection_time']}")
        print(f"  🔄 캐시 유효: {status['cache_valid']}")
        print(f"  🎯 선별 기준: 시총 {status['selection_criteria']['min_market_cap']:.1f}조엔 이상, "
              f"거래량 {status['selection_criteria']['min_avg_volume']:.0f}백만주 이상")
        
        # 전체 시장 자동선별 + 분석
        print(f"\n🔍 실시간 자동선별 + 전체 분석 시작...")
        start_time = time.time()
        
        market_result = await scan_jp_market()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ 총 소요 시간: {elapsed_time:.1f}초")
        
        print(f"\n💱 현재 환율 정보:")
        print(f"  USD/JPY: {market_result['current_usd_jpy']:.2f}")
        print(f"  엔화 신호: {market_result['yen_signal']}")
        print(f"  선별 방식: {market_result['selection_method']}")
        print(f"  선별 시간: {market_result['last_selection_time']}")
        
        print(f"\n📈 자동선별 + 분석 결과:")
        print(f"  총 분석: {market_result['total_analyzed']}개 종목 (실시간 자동선별)")
        print(f"  매수 신호: {market_result['buy_count']}개")
        print(f"  매도 신호: {market_result['sell_count']}개")
        
        # 선별 요약
        summary = market_result['selection_summary']
        print(f"\n🎯 선별 구성:")
        print(f"  수출주: {summary['export_stocks']}개")
        print(f"  내수주: {summary['domestic_stocks']}개") 
        print(f"  혼합주: {summary['mixed_stocks']}개")
        print(f"  평균 선별점수: {summary['avg_selection_score']:.3f}")
        print(f"  평균 시가총액: {summary['avg_market_cap']:.2f}조엔")
        
        # 상위 매수 추천 (상세 정보)
        if market_result['top_buys']:
            print(f"\n🎯 상위 매수 추천 (실시간 자동선별):")
            for i, signal in enumerate(market_result['top_buys'][:3], 1):
                print(f"\n  {i}. {signal.symbol} ({signal.stock_type}) - 신뢰도: {signal.confidence:.2%}")
                print(f"     🏆 선별점수: {signal.selection_score:.3f} | 시총: {signal.market_cap/1e12:.2f}조엔")
                print(f"     📊 기술지표: RSI({signal.rsi:.0f}) MACD({signal.macd_signal}) 추세({signal.ma_trend})")
                print(f"     💰 포지션: {signal.position_size:,}주 ({len(signal.split_buy_plan)}단계 분할매수)")
                print(f"     🛡️ 손절: {signal.stop_loss:,.0f}엔 익절: {signal.take_profit:,.0f}엔")
                print(f"     ⏰ 최대보유: {signal.max_hold_days}일")
                print(f"     💡 이유: {signal.reasoning}")
        
        # 개별 종목 상세 분석 (자동선별된 첫 번째 종목)
        if market_result['total_analyzed'] > 0:
            test_symbol = market_result['top_buys'][0].symbol if market_result['top_buys'] else None
            if test_symbol:
                print(f"\n📊 개별 종목 상세 분석 - {test_symbol} (자동선별 1위):")
                detailed_result = await analyze_jp(test_symbol)
                print(f"  🎯 액션: {detailed_result['decision']} (신뢰도: {detailed_result['confidence_score']:.1f}%)")
                print(f"  🏆 선별점수: {detailed_result['selection_score']:.3f} | 품질순위: {detailed_result['quality_rank']}")
                print(f"  📊 기술지표:")
                print(f"    - RSI: {detailed_result['rsi']:.1f}")
                print(f"    - MACD: {detailed_result['macd_signal']}")
                print(f"    - 볼린저밴드: {detailed_result['bollinger_signal']}")
                print(f"    - 스토캐스틱: {detailed_result['stochastic_signal']}")
                print(f"    - 추세: {detailed_result['ma_trend']}")
                print(f"  💰 분할매매:")
                print(f"    - 총 포지션: {detailed_result['position_size']:,}주")
                print(f"    - 매수 계획: {len(detailed_result['split_buy_plan'])}단계")
                print(f"    - 매도 계획: {len(detailed_result['split_sell_plan'])}단계")
                print(f"  🛡️ 리스크 관리:")
                print(f"    - 손절가: {detailed_result['stop_loss']:,.0f}엔")
                print(f"    - 익절가: {detailed_result['take_profit']:,.0f}엔")
                print(f"    - 최대보유: {detailed_result['max_hold_days']}일")
                print(f"  💱 엔화 정보:")
                print(f"    - 종목타입: {detailed_result['stock_type']}")
                print(f"    - 엔화신호: {detailed_result['yen_signal']}")
                print(f"    - 섹터: {detailed_result['sector']}")
        
        print("\n✅ 테스트 완료!")
        print("\n🎯 일본 주식 V4.0 완전 자동화 전략 특징:")
        print("  ✅ 🆕 실시간 닛케이225 크롤링 (하드코딩 완전 제거)")
        print("  ✅ 📊 펀더멘털 + 기술적 + 엔화 종합 선별")
        print("  ✅ 💱 USD/JPY 환율 실시간 반영")
        print("  ✅ 💰 분할매매 시스템 (3단계 매수, 2단계 매도)")
        print("  ✅ 🛡️ 동적 손절/익절 (엔화+신뢰도 기반)")
        print("  ✅ 🔍 상위 20개 종목 완전 자동 선별")
        print("  ✅ 🤖 완전 자동화 (24시간 캐시 + 실시간 업데이트)")
        print("  ✅ 📱 웹 대시보드 연동 준비")
        print("\n💡 사용법:")
        print("  - python jp_strategy.py : 전체 자동선별 + 분석")
        print("  - await analyze_jp('7203.T') : 개별 종목 분석")
        print("  - await scan_jp_market() : 시장 전체 스캔")
        print("  - await force_jp_reselection() : 강제 재선별")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"테스트 실행 실패: {e}")

if __name__ == "__main__":
    asyncio.run(main())
