"""
🇺🇸 미국 주식 전략 모듈 - 최고퀸트프로젝트 (완전 업그레이드)
===========================================================

핵심 기능:
1. 자동 종목 선별 (S&P500 + NASDAQ100 → 상위 20개)
2. 4가지 전략 융합 (버핏25% + 린치25% + 모멘텀25% + 기술25%)
3. 개별 분할매매 (각 종목마다 3단계 매수, 2단계 매도)
4. 완전 자동화 시스템

Author: 최고퀸트팀
Version: 2.0.0 (대폭 업그레이드)
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
    additional_data: Optional[Dict] = None

class AdvancedUSStrategy:
    """🚀 완전 업그레이드 미국 전략 클래스"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        """전략 초기화"""
        self.config = self._load_config(config_path)
        self.us_config = self.config.get('us_strategy', {})
        self.enabled = self.us_config.get('enabled', True)
        
        # 🎯 자동 선별 설정
        self.target_stocks = 20  # 상위 20개 종목 선별
        self.min_market_cap = 10_000_000_000  # 100억 달러 이상
        self.min_avg_volume = 1_000_000  # 일평균 100만주 이상
        
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
        self.max_sector_weight = 0.30  # 섹터별 최대 30%
        
        # 📈 S&P 500 종목 리스트 (상위 100개 대형주)
        self.sp500_top100 = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'ABBV', 'BAC', 'KO', 'AVGO',
            'XOM', 'CVX', 'LLY', 'WMT', 'PEP', 'TMO', 'COST', 'ORCL', 'ABT', 'ACN',
            'NFLX', 'CRM', 'MRK', 'VZ', 'ADBE', 'DHR', 'TXN', 'NKE', 'PM', 'DIS',
            'WFC', 'NEE', 'RTX', 'CMCSA', 'BMY', 'UNP', 'T', 'COP', 'MS', 'AMD',
            'LOW', 'IBM', 'HON', 'AMGN', 'SPGI', 'LIN', 'QCOM', 'GE', 'CAT', 'UPS',
            'BA', 'SBUX', 'AXP', 'BLK', 'MDT', 'GS', 'NOW', 'BKNG', 'AMAT', 'ADI',
            'GILD', 'SYK', 'MMC', 'TJX', 'CVS', 'MO', 'ZTS', 'AON', 'MDLZ', 'C',
            'PYPL', 'CI', 'SO', 'ISRG', 'DUK', 'PLD', 'TGT', 'SCHW', 'MU', 'USB',
            'AMT', 'INTC', 'CB', 'CL', 'PNC', 'DE', 'BSX', 'INTU', 'SHW', 'FIS'
        ]
        
        if self.enabled:
            logger.info(f"🇺🇸 완전 업그레이드 미국 전략 초기화")
            logger.info(f"🎯 자동 선별: 상위 {self.target_stocks}개 종목")
            logger.info(f"📊 4가지 전략 융합: 버핏25% + 린치25% + 모멘텀25% + 기술25%")
            logger.info(f"💰 분할매매: 3단계 매수(40%+35%+25%), 2단계 매도(60%+40%)")
            logger.info(f"🛡️ 리스크 관리: 손절{self.stop_loss_pct*100}%, 익절{self.take_profit2_pct*100}%")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return {}

    async def _get_comprehensive_data(self, symbol: str) -> Dict:
        """종합 데이터 수집 (재무 + 가격 + 기술적)"""
        try:
            stock = yf.Ticker(symbol)
            
            # 기본 정보
            info = stock.info
            
            # 6개월 가격 데이터
            hist = stock.history(period="6mo")
            if hist.empty:
                return {}
            
            # 현재가
            current_price = hist['Close'].iloc[-1]
            
            # 재무 지표
            data = {
                'symbol': symbol,
                'price': current_price,
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'pbr': info.get('priceToBook', 0) or 0,
                'roe': (info.get('returnOnEquity', 0) or 0) * 100,
                'eps_growth': (info.get('earningsQuarterlyGrowth', 0) or 0) * 100,
                'revenue_growth': (info.get('revenueQuarterlyGrowth', 0) or 0) * 100,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'market_cap': info.get('marketCap', 0) or 0,
                'avg_volume': info.get('averageVolume', 0) or 0,
                'sector': info.get('sector', 'Unknown'),
                'beta': info.get('beta', 1.0) or 1.0,
            }
            
            # PEG 계산
            if data['pe_ratio'] > 0 and data['eps_growth'] > 0:
                data['peg'] = data['pe_ratio'] / data['eps_growth']
            else:
                data['peg'] = 999
            
            # 모멘텀 지표 계산
            if len(hist) >= 60:
                data['momentum_3m'] = ((current_price / hist['Close'].iloc[-60]) - 1) * 100
            else:
                data['momentum_3m'] = 0
                
            if len(hist) >= 125:
                data['momentum_6m'] = ((current_price / hist['Close'].iloc[-125]) - 1) * 100
            else:
                data['momentum_6m'] = 0
            
            # 기술적 지표 계산
            if len(hist) >= 20:
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
                
                # 추세 (20일 이동평균 기준)
                ma20 = hist['Close'].rolling(20).mean().iloc[-1]
                data['trend'] = 'uptrend' if current_price > ma20 else 'downtrend'
                
                # 거래량 급증
                avg_volume_10d = hist['Volume'].rolling(10).mean().iloc[-1]
                current_volume = hist['Volume'].iloc[-1]
                data['volume_spike'] = current_volume / avg_volume_10d if avg_volume_10d > 0 else 1
            else:
                data.update({
                    'rsi': 50, 'macd_signal': 'neutral', 'bb_position': 'normal',
                    'trend': 'sideways', 'volume_spike': 1
                })
            
            return data
            
        except Exception as e:
            logger.error(f"데이터 수집 실패 {symbol}: {e}")
            return {}

    def _buffett_analysis(self, data: Dict) -> Tuple[float, str]:
        """워렌 버핏 가치투자 분석 (25%)"""
        score = 0.0
        reasoning = []
        
        # PBR (35%)
        pbr = data.get('pbr', 999)
        if 0 < pbr <= 1.5:
            score += 0.35
            reasoning.append(f"저PBR({pbr:.2f})")
        elif 1.5 < pbr <= 2.5:
            score += 0.15
            reasoning.append(f"적정PBR({pbr:.2f})")
        
        # ROE (30%)
        roe = data.get('roe', 0)
        if roe >= 15:
            score += 0.30
            reasoning.append(f"고ROE({roe:.1f}%)")
        elif roe >= 10:
            score += 0.15
            reasoning.append(f"적정ROE({roe:.1f}%)")
        
        # 부채비율 (20%)
        debt_ratio = data.get('debt_to_equity', 999) / 100 if data.get('debt_to_equity') else 999
        if debt_ratio <= 0.4:
            score += 0.20
            reasoning.append("저부채")
        elif debt_ratio <= 0.6:
            score += 0.10
            reasoning.append("적정부채")
        
        # PE 적정성 (15%)
        pe = data.get('pe_ratio', 999)
        if 0 < pe <= 15:
            score += 0.15
            reasoning.append(f"저PE({pe:.1f})")
        elif 15 < pe <= 25:
            score += 0.05
            reasoning.append(f"적정PE({pe:.1f})")
        
        return score, "버핏: " + " | ".join(reasoning)

    def _lynch_analysis(self, data: Dict) -> Tuple[float, str]:
        """피터 린치 성장투자 분석 (25%)"""
        score = 0.0
        reasoning = []
        
        # PEG (40%)
        peg = data.get('peg', 999)
        if 0 < peg <= 1.0:
            score += 0.40
            reasoning.append(f"저PEG({peg:.2f})")
        elif 1.0 < peg <= 1.5:
            score += 0.20
            reasoning.append(f"적정PEG({peg:.2f})")
        
        # EPS 성장률 (35%)
        eps_growth = data.get('eps_growth', 0)
        if eps_growth >= 20:
            score += 0.35
            reasoning.append(f"고성장({eps_growth:.1f}%)")
        elif eps_growth >= 10:
            score += 0.20
            reasoning.append(f"성장({eps_growth:.1f}%)")
        
        # 매출 성장률 (25%)
        revenue_growth = data.get('revenue_growth', 0)
        if revenue_growth >= 15:
            score += 0.25
            reasoning.append(f"매출급증({revenue_growth:.1f}%)")
        elif revenue_growth >= 5:
            score += 0.10
            reasoning.append(f"매출성장({revenue_growth:.1f}%)")
        
        return score, "린치: " + " | ".join(reasoning)

    def _momentum_analysis(self, data: Dict) -> Tuple[float, str]:
        """모멘텀 분석 (25%)"""
        score = 0.0
        reasoning = []
        
        # 3개월 모멘텀 (40%)
        mom_3m = data.get('momentum_3m', 0)
        if mom_3m >= 20:
            score += 0.40
            reasoning.append(f"강한3M({mom_3m:.1f}%)")
        elif mom_3m >= 10:
            score += 0.20
            reasoning.append(f"상승3M({mom_3m:.1f}%)")
        elif mom_3m >= 0:
            score += 0.05
            reasoning.append(f"보합3M({mom_3m:.1f}%)")
        
        # 6개월 모멘텀 (35%)
        mom_6m = data.get('momentum_6m', 0)
        if mom_6m >= 30:
            score += 0.35
            reasoning.append(f"강한6M({mom_6m:.1f}%)")
        elif mom_6m >= 15:
            score += 0.20
            reasoning.append(f"상승6M({mom_6m:.1f}%)")
        
        # 거래량 급증 (25%)
        volume_spike = data.get('volume_spike', 1)
        if volume_spike >= 2.0:
            score += 0.25
            reasoning.append(f"거래량급증({volume_spike:.1f}x)")
        elif volume_spike >= 1.5:
            score += 0.10
            reasoning.append(f"거래량증가({volume_spike:.1f}x)")
        
        return score, "모멘텀: " + " | ".join(reasoning)

    def _technical_analysis(self, data: Dict) -> Tuple[float, str]:
        """기술적 분석 (25%)"""
        score = 0.0
        reasoning = []
        
        # RSI (30%)
        rsi = data.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 0.30
            reasoning.append(f"RSI적정({rsi:.0f})")
        elif 20 <= rsi < 30:
            score += 0.15
            reasoning.append(f"RSI과매도({rsi:.0f})")
        elif rsi > 70:
            reasoning.append(f"RSI과매수({rsi:.0f})")
        
        # MACD (25%)
        macd = data.get('macd_signal', 'neutral')
        if macd == 'bullish':
            score += 0.25
            reasoning.append("MACD상승")
        
        # 추세 (25%)
        trend = data.get('trend', 'sideways')
        if trend == 'uptrend':
            score += 0.25
            reasoning.append("상승추세")
        
        # 볼린저 밴드 (20%)
        bb = data.get('bb_position', 'normal')
        if bb == 'oversold':
            score += 0.20
            reasoning.append("BB과매도")
        elif bb == 'normal':
            score += 0.10
            reasoning.append("BB정상")
        
        return score, "기술적: " + " | ".join(reasoning)

    def _calculate_split_trading_plan(self, symbol: str, current_price: float, 
                                     confidence: float, portfolio_value: float = 1000000) -> Dict:
        """분할매매 계획 수립"""
        try:
            # 종목별 목표 비중 계산 (신뢰도 기반)
            base_weight = self.total_portfolio_ratio / self.target_stocks  # 기본 4%
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

    async def analyze_symbol(self, symbol: str) -> USStockSignal:
        """개별 종목 종합 분석"""
        if not self.enabled:
            return self._create_empty_signal(symbol, "전략 비활성화")
        
        try:
            # 종합 데이터 수집
            data = await self._get_comprehensive_data(symbol)
            if not data:
                raise ValueError(f"데이터 수집 실패: {symbol}")
            
            # 4가지 전략 분석
            buffett_score, buffett_reasoning = self._buffett_analysis(data)
            lynch_score, lynch_reasoning = self._lynch_analysis(data)
            momentum_score, momentum_reasoning = self._momentum_analysis(data)
            technical_score, technical_reasoning = self._technical_analysis(data)
            
            # 가중 평균 계산
            total_score = (
                buffett_score * self.buffett_weight +
                lynch_score * self.lynch_weight +
                momentum_score * self.momentum_weight +
                technical_score * self.technical_weight
            )
            
            # 최종 액션 결정
            if total_score >= 0.70:
                action = 'buy'
                confidence = min(total_score, 0.95)
            elif total_score <= 0.30:
                action = 'sell'
                confidence = min(1 - total_score, 0.95)
            else:
                action = 'hold'
                confidence = 0.50
            
            # 분할매매 계획 수립
            split_plan = self._calculate_split_trading_plan(symbol, data['price'], confidence)
            
            # 목표주가 계산
            target_price = data['price'] * (1 + confidence * 0.30)  # 최대 30% 상승 기대
            
            # 종합 reasoning
            all_reasoning = " | ".join([buffett_reasoning, lynch_reasoning, 
                                      momentum_reasoning, technical_reasoning])
            
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
                momentum_12m=0,  # 12개월은 별도 계산 필요
                relative_strength=0,  # 별도 계산 필요
                
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
            reasoning=reason, target_price=0.0, timestamp=datetime.now()
        )

    async def auto_select_top20_stocks(self) -> List[USStockSignal]:
        """🎯 자동 종목 선별: S&P500에서 상위 20개 종목 선별"""
        if not self.enabled:
            logger.warning("미국 주식 전략이 비활성화되어 있습니다")
            return []
        
        logger.info(f"🔍 자동 종목 선별 시작 - {len(self.sp500_top100)}개 종목에서 상위 {self.target_stocks}개 선별")
        
        # 1단계: 기본 필터링 (시가총액, 거래량)
        logger.info("📊 1단계: 기본 필터링 (시가총액, 거래량)")
        filtered_symbols = []
        
        for symbol in self.sp500_top100:
            try:
                data = await self._get_comprehensive_data(symbol)
                if data:
                    market_cap = data.get('market_cap', 0)
                    avg_volume = data.get('avg_volume', 0)
                    
                    if market_cap >= self.min_market_cap and avg_volume >= self.min_avg_volume:
                        filtered_symbols.append(symbol)
                        logger.debug(f"✅ {symbol}: 시총 ${market_cap/1e9:.1f}B, 거래량 {avg_volume/1e6:.1f}M")
                    else:
                        logger.debug(f"❌ {symbol}: 필터링 제외")
                
                # API 제한 고려
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 필터링 실패: {e}")
        
        logger.info(f"📊 1단계 완료: {len(filtered_symbols)}개 종목 통과")
        
        # 2단계: 종합 분석 및 스코어링
        logger.info("🎯 2단계: 종합 분석 및 스코어링")
        all_signals = []
        
        for i, symbol in enumerate(filtered_symbols, 1):
            try:
                logger.info(f"📊 분석 중... {i}/{len(filtered_symbols)} - {symbol}")
                signal = await self.analyze_symbol(symbol)
                all_signals.append(signal)
                
                # 진행 상황 로그
                if signal.action == 'buy':
                    logger.info(f"🟢 {symbol}: 매수 신뢰도:{signal.confidence:.2f} 총점:{signal.total_score:.2f}")
                elif signal.action == 'sell':
                    logger.info(f"🔴 {symbol}: 매도 신뢰도:{signal.confidence:.2f} 총점:{signal.total_score:.2f}")
                else:
                    logger.info(f"⚪ {symbol}: 보유 신뢰도:{signal.confidence:.2f} 총점:{signal.total_score:.2f}")
                
                # API 제한 고려
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 분석 실패: {e}")
        
        # 3단계: 상위 20개 선별
        logger.info("🏆 3단계: 상위 20개 종목 선별")
        
        # 총점 기준으로 정렬
        sorted_signals = sorted(all_signals, key=lambda x: x.total_score, reverse=True)
        
        # 섹터 다양성 고려 (같은 섹터 최대 30%)
        selected_signals = []
        sector_counts = {}
        max_per_sector = int(self.target_stocks * self.max_sector_weight)
        
        for signal in sorted_signals:
            sector = signal.sector
            sector_count = sector_counts.get(sector, 0)
            
            if len(selected_signals) < self.target_stocks:
                if sector_count < max_per_sector:
                    selected_signals.append(signal)
                    sector_counts[sector] = sector_count + 1
                    logger.info(f"✅ 선별: {signal.symbol} ({signal.sector}) 점수:{signal.total_score:.2f}")
        
        # 남은 자리가 있으면 점수 순으로 채움
        remaining_slots = self.target_stocks - len(selected_signals)
        if remaining_slots > 0:
            remaining_signals = [s for s in sorted_signals if s not in selected_signals]
            selected_signals.extend(remaining_signals[:remaining_slots])
        
        logger.info(f"🎯 자동 선별 완료: {len(selected_signals)}개 종목 선별")
        
        # 선별 결과 요약
        buy_count = len([s for s in selected_signals if s.action == 'buy'])
        sell_count = len([s for s in selected_signals if s.action == 'sell'])
        hold_count = len([s for s in selected_signals if s.action == 'hold'])
        
        logger.info(f"📊 선별 결과: 매수 {buy_count}개, 매도 {sell_count}개, 보유 {hold_count}개")
        
        # 섹터별 분포
        sector_dist = {}
        for signal in selected_signals:
            sector_dist[signal.sector] = sector_dist.get(signal.sector, 0) + 1
        
        logger.info("🏢 섹터별 분포:")
        for sector, count in sector_dist.items():
            logger.info(f"  {sector}: {count}개 ({count/len(selected_signals)*100:.1f}%)")
        
        return selected_signals

    async def generate_portfolio_report(self, selected_stocks: List[USStockSignal]) -> Dict:
        """📊 포트폴리오 리포트 생성"""
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
        
        # 총 투자금액 계산
        total_investment = sum([s.additional_data.get('total_investment', 0) for s in selected_stocks if s.additional_data])
        total_shares_value = sum([s.total_shares * s.price for s in selected_stocks])
        
        # 상위 5개 매수 종목
        top_buys = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
        
        # 섹터별 분포
        sector_dist = {}
        for stock in selected_stocks:
            sector_dist[stock.sector] = sector_dist.get(stock.sector, 0) + 1
        
        # 리스크 지표
        portfolio_beta = np.mean([s.additional_data.get('beta', 1.0) for s in selected_stocks if s.additional_data])
        
        report = {
            'summary': {
                'total_stocks': total_stocks,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'hold_signals': len(hold_signals),
                'total_investment': total_investment,
                'total_shares_value': total_shares_value,
                'portfolio_beta': portfolio_beta
            },
            'strategy_scores': {
                'avg_buffett_score': avg_buffett,
                'avg_lynch_score': avg_lynch,
                'avg_momentum_score': avg_momentum,
                'avg_technical_score': avg_technical,
                'avg_total_score': avg_total
            },
            'top_picks': [
                {
                    'symbol': stock.symbol,
                    'sector': stock.sector,
                    'confidence': stock.confidence,
                    'total_score': stock.total_score,
                    'price': stock.price,
                    'target_price': stock.target_price,
                    'total_shares': stock.total_shares,
                    'total_investment': stock.additional_data.get('total_investment', 0) if stock.additional_data else 0,
                    'reasoning': stock.reasoning[:100] + "..." if len(stock.reasoning) > 100 else stock.reasoning
                }
                for stock in top_buys
            ],
            'sector_distribution': sector_dist,
            'risk_metrics': {
                'max_single_position': max([s.additional_data.get('target_weight', 0) for s in selected_stocks if s.additional_data]),
                'portfolio_beta': portfolio_beta,
                'diversification_score': len(sector_dist) / total_stocks
            }
        }
        
        return report

    async def execute_split_trading_simulation(self, signal: USStockSignal) -> Dict:
        """🔄 분할매매 시뮬레이션"""
        if signal.action != 'buy':
            return {"status": "not_applicable", "reason": "매수 신호가 아님"}
        
        simulation = {
            'symbol': signal.symbol,
            'strategy': 'split_trading',
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
            'risk_management': {
                'max_hold_days': signal.max_hold_days,
                'total_investment': signal.additional_data.get('total_investment', 0) if signal.additional_data else 0,
                'portfolio_weight': signal.additional_data.get('target_weight', 0) if signal.additional_data else 0
            }
        }
        
        return simulation

# 메인 실행 함수들
async def run_auto_selection():
    """자동 선별 실행"""
    strategy = AdvancedUSStrategy()
    selected_stocks = await strategy.auto_select_top20_stocks()
    
    if selected_stocks:
        report = await strategy.generate_portfolio_report(selected_stocks)
        return selected_stocks, report
    else:
        return [], {}

async def analyze_us(symbol: str) -> Dict:
    """단일 미국 주식 분석 (기존 호환성)"""
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
        'split_trading_plan': await strategy.execute_split_trading_simulation(signal)
    }

if __name__ == "__main__":
    async def main():
        print("🇺🇸 최고퀸트프로젝트 - 완전 업그레이드 미국 전략 테스트!")
        print("🎯 자동 종목 선별 + 4가지 전략 융합 + 분할매매 시스템")
        print("="*60)
        
        # 자동 선별 실행
        print("\n🔍 자동 종목 선별 시작...")
        selected_stocks, report = await run_auto_selection()
        
        if selected_stocks:
            print(f"\n🎯 선별 완료! 상위 {len(selected_stocks)}개 종목:")
            print("="*60)
            
            # 상위 5개 종목 상세 표시
            top_5 = sorted(selected_stocks, key=lambda x: x.total_score, reverse=True)[:5]
            
            for i, stock in enumerate(top_5, 1):
                print(f"\n{i}. {stock.symbol} ({stock.sector})")
                print(f"   🎯 액션: {stock.action} | 신뢰도: {stock.confidence:.1%}")
                print(f"   📊 종합점수: {stock.total_score:.2f} (버핏:{stock.buffett_score:.2f} + 린치:{stock.lynch_score:.2f} + 모멘텀:{stock.momentum_score:.2f} + 기술:{stock.technical_score:.2f})")
                print(f"   💰 현재가: ${stock.price:.2f} → 목표가: ${stock.target_price:.2f}")
                print(f"   🔄 분할매매: {stock.total_shares}주 (1단계:{stock.stage1_shares} + 2단계:{stock.stage2_shares} + 3단계:{stock.stage3_shares})")
                print(f"   🛡️ 손절: ${stock.stop_loss:.2f} | 익절: ${stock.take_profit_1:.2f} → ${stock.take_profit_2:.2f}")
                print(f"   💡 {stock.reasoning[:80]}...")
            
            # 포트폴리오 요약
            print(f"\n📊 포트폴리오 요약:")
            print(f"   총 종목: {report['summary']['total_stocks']}개")
            print(f"   매수: {report['summary']['buy_signals']}개 | 보유: {report['summary']['hold_signals']}개")
            print(f"   총 투자금액: ${report['summary']['total_investment']:,.0f}")
            print(f"   포트폴리오 베타: {report['summary']['portfolio_beta']:.2f}")
            
            # 섹터 분포
            print(f"\n🏢 섹터 분포:")
            for sector, count in report['sector_distribution'].items():
                percentage = count / report['summary']['total_stocks'] * 100
                print(f"   {sector}: {count}개 ({percentage:.1f}%)")
            
            # 분할매매 시뮬레이션 (첫 번째 매수 종목)
            buy_stocks = [s for s in selected_stocks if s.action == 'buy']
            if buy_stocks:
                print(f"\n🔄 분할매매 시뮬레이션 - {buy_stocks[0].symbol}:")
                strategy = AdvancedUSStrategy()
                simulation = await strategy.execute_split_trading_simulation(buy_stocks[0])
                
                for stage, data in simulation['stages'].items():
                    print(f"   {stage}: ${data['trigger_price']:.2f}에 {data['shares']}주 ({data['ratio']}) - {data['status']}")
                
                print(f"   손절: ${simulation['exit_plan']['stop_loss']['price']:.2f}")
                print(f"   익절1: ${simulation['exit_plan']['take_profit_1']['price']:.2f} (60% 매도)")
                print(f"   익절2: ${simulation['exit_plan']['take_profit_2']['price']:.2f} (40% 매도)")
            
        else:
            print("❌ 선별된 종목이 없습니다.")
        
        print("\n✅ 테스트 완료!")
        print("🚀 28살 월 1-3억 목표 달성을 위한 완전 자동화 시스템 준비 완료!")

    asyncio.run(main())
