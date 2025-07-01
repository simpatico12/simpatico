#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 최고퀸트프로젝트 - 통합 백테스팅 시스템 + 웹 대시보드 (완성)
================================================================

전체 시스템 백테스팅 + 모바일 최적화 웹 인터페이스:
- 🇺🇸 미국 주식 전략 백테스팅 (S&P500 자동선별 + 4가지 전략)
- 🇯🇵 일본 주식 전략 백테스팅 (닛케이225 + 엔화전략)  
- 🪙 암호화폐 전략 백테스팅 (업비트 + AI품질평가)
- 📊 통합 포트폴리오 시뮬레이션 (6:2:2 비율)
- 📱 모바일 최적화 웹 대시보드
- 🌐 EC2 탄력적 IP 접속 지원

실행: python backtest_system.py
접속: http://탄력적IP:8080

Author: 최고퀸트팀
Version: 1.0.0
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import json
import os
from dataclasses import dataclass, asdict
import traceback
import time

# FastAPI 웹 프레임워크
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# 차트 및 시각화
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils

# 프로젝트 모듈들
try:
    from core import QuantTradingEngine, UnifiedTradingSignal
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ core 모듈 로드 실패: {e}")
    CORE_AVAILABLE = False

try:
    from utils import (
        DataProcessor, FinanceUtils, TimeZoneManager, 
        Formatter, get_config, FileManager
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ utils 모듈 로드 실패: {e}")
    UTILS_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================================================
# 📊 백테스팅 데이터 모델들
# ================================================================================================

@dataclass
class BacktestConfig:
    """백테스팅 설정"""
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100000.0  # $100,000
    
    # 포트폴리오 비중
    us_allocation: float = 0.6    # 60%
    jp_allocation: float = 0.2    # 20%
    coin_allocation: float = 0.2  # 20%
    
    # 리밸런싱
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly
    
    # 수수료
    commission_rate: float = 0.001  # 0.1%
    slippage: float = 0.0005       # 0.05%

@dataclass  
class TradeRecord:
    """거래 기록"""
    date: str
    market: str  # US, JP, COIN
    symbol: str
    action: str  # buy, sell
    quantity: float
    price: float
    amount: float
    commission: float
    confidence: float
    strategy: str
    reasoning: str

@dataclass
class PerformanceMetrics:
    """성과 지표"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float

@dataclass
class BacktestResult:
    """백테스팅 결과"""
    config: BacktestConfig
    equity_curve: pd.DataFrame
    trade_records: List[TradeRecord]
    performance_metrics: PerformanceMetrics
    market_performance: Dict[str, Dict]
    daily_returns: pd.Series
    monthly_returns: pd.Series
    drawdown_series: pd.Series
    benchmark_comparison: Dict
    
# ================================================================================================
# 📈 통합 백테스팅 엔진
# ================================================================================================

class IntegratedBacktestEngine:
    """🏆 통합 백테스팅 엔진 (3개 시장 + 실제 전략)"""
    
    def __init__(self):
        self.config = BacktestConfig()
        self.engine = None
        self.timezone_manager = TimeZoneManager() if UTILS_AVAILABLE else None
        self.file_manager = FileManager() if UTILS_AVAILABLE else None
        
        # 백테스팅 상태
        self.is_running = False
        self.progress = 0.0
        self.current_status = "대기 중"
        self.start_time = None
        
        logger.info("🚀 통합 백테스팅 엔진 초기화 완료")
    
    def _generate_sample_data(self, symbol: str, market: str, 
                            start_date: str, end_date: str) -> pd.DataFrame:
        """샘플 가격 데이터 생성 (실제 데이터 대체용)"""
        try:
            # 날짜 범위 생성
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 시장별 초기 가격 설정
            if market == 'US':
                base_price = 150.0  # 미국 주식
                volatility = 0.02
            elif market == 'JP':
                base_price = 2500.0  # 일본 주식 (엔)
                volatility = 0.025
            else:  # COIN
                base_price = 50000000.0  # 암호화폐 (원)
                volatility = 0.05
            
            # 랜덤 워크로 가격 생성
            np.random.seed(hash(symbol) % 2**32)  # 심볼별 고정 시드
            returns = np.random.normal(0.0005, volatility, len(date_range))  # 일일 수익률
            
            # 가격 계산
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # DataFrame 생성
            df = pd.DataFrame({
                'Date': date_range,
                'Open': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(100000, 1000000) for _ in prices]
            })
            
            df.set_index('Date', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"샘플 데이터 생성 실패 {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_market_data(self, symbol: str, market: str) -> pd.DataFrame:
        """시장 데이터 수집 (실제 데이터 또는 샘플 데이터)"""
        try:
            # 실제 데이터 수집 시도
            import yfinance as yf
            
            if market == 'US':
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.config.start_date, 
                                    end=self.config.end_date)
            elif market == 'JP':
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.config.start_date,
                                    end=self.config.end_date)
            else:  # COIN - 샘플 데이터 사용
                data = self._generate_sample_data(symbol, market, 
                                                self.config.start_date, 
                                                self.config.end_date)
            
            if data.empty:
                # 실제 데이터가 없으면 샘플 데이터 생성
                data = self._generate_sample_data(symbol, market,
                                                self.config.start_date,
                                                self.config.end_date)
            
            return data
            
        except Exception as e:
            logger.warning(f"실제 데이터 수집 실패 {symbol}, 샘플 데이터 사용: {e}")
            return self._generate_sample_data(symbol, market,
                                            self.config.start_date,
                                            self.config.end_date)
    
    def _simulate_strategy_signals(self, data: pd.DataFrame, market: str, 
                                 symbol: str) -> List[Dict]:
        """전략 신호 시뮬레이션 (실제 전략 로직 기반)"""
        signals = []
        
        try:
            # 기술적 지표 계산
            if len(data) < 30:
                return signals
            
            closes = data['Close']
            
            # RSI 계산
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            rsi = calculate_rsi(closes)
            
            # 이동평균
            ma20 = closes.rolling(20).mean()
            ma50 = closes.rolling(50).mean()
            
            # 볼린저 밴드
            std = closes.rolling(20).std()
            bb_upper = ma20 + (std * 2)
            bb_lower = ma20 - (std * 2)
            
            # 시장별 전략 적용
            for i in range(50, len(data)):  # 50일 후부터 신호 생성
                date = data.index[i]
                price = closes.iloc[i]
                
                # 신호 생성 조건
                confidence = 0.5
                action = 'hold'
                reasoning = "기본"
                
                if market == 'US':
                    # 미국 주식: 버핏 + 린치 전략 시뮬레이션
                    if (rsi.iloc[i] < 40 and 
                        closes.iloc[i] > ma20.iloc[i] and 
                        ma20.iloc[i] > ma50.iloc[i]):
                        action = 'buy'
                        confidence = 0.75
                        reasoning = "버핏+린치: RSI과매도+상승추세"
                    elif rsi.iloc[i] > 70:
                        action = 'sell'
                        confidence = 0.65
                        reasoning = "RSI과매수"
                        
                elif market == 'JP':
                    # 일본 주식: 엔화 + 기술분석
                    if (closes.iloc[i] < bb_lower.iloc[i] and
                        rsi.iloc[i] < 35):
                        action = 'buy'
                        confidence = 0.70
                        reasoning = "볼린저밴드하한+RSI과매도"
                    elif closes.iloc[i] > bb_upper.iloc[i]:
                        action = 'sell'
                        confidence = 0.60
                        reasoning = "볼린저밴드상한돌파"
                        
                else:  # COIN
                    # 암호화폐: AI 품질 + 시장사이클 시뮬레이션
                    if (rsi.iloc[i] < 30 and 
                        closes.iloc[i] > ma20.iloc[i]):
                        action = 'buy'
                        confidence = 0.80
                        reasoning = "AI품질+사이클: 강한매수신호"
                    elif rsi.iloc[i] > 75:
                        action = 'sell'
                        confidence = 0.70
                        reasoning = "시장사이클: 분배단계"
                
                # 신호 저장 (매수/매도만)
                if action in ['buy', 'sell']:
                    signals.append({
                        'date': date,
                        'action': action,
                        'price': price,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'rsi': rsi.iloc[i],
                        'ma20': ma20.iloc[i],
                        'ma50': ma50.iloc[i]
                    })
            
            logger.info(f"{market}-{symbol}: {len(signals)}개 신호 생성")
            return signals
            
        except Exception as e:
            logger.error(f"신호 시뮬레이션 실패 {market}-{symbol}: {e}")
            return []
    
    def _execute_backtest_simulation(self, market_data: Dict, 
                                   market_signals: Dict) -> Tuple[pd.DataFrame, List[TradeRecord]]:
        """백테스팅 시뮬레이션 실행"""
        try:
            # 초기 설정
            initial_capital = self.config.initial_capital
            portfolio_value = initial_capital
            positions = {}  # {symbol: {quantity, avg_price}}
            cash = initial_capital
            trade_records = []
            
            # 시장별 할당
            market_allocations = {
                'US': self.config.us_allocation,
                'JP': self.config.jp_allocation, 
                'COIN': self.config.coin_allocation
            }
            
            # 모든 날짜 수집 및 정렬
            all_dates = set()
            for market, symbols_data in market_data.items():
                for symbol, data in symbols_data.items():
                    all_dates.update(data.index)
            
            all_dates = sorted(list(all_dates))
            
            # 일별 포트폴리오 가치 추적
            equity_curve = []
            
            for date in all_dates:
                daily_portfolio_value = cash
                
                # 각 시장의 신호 처리
                for market, symbols_signals in market_signals.items():
                    market_allocation = market_allocations.get(market, 0)
                    
                    for symbol, signals in symbols_signals.items():
                        # 해당 날짜의 신호 찾기
                        day_signals = [s for s in signals if s['date'].date() == date.date()]
                        
                        if day_signals and symbol in market_data[market]:
                            signal = day_signals[0]  # 첫 번째 신호 사용
                            data = market_data[market][symbol]
                            
                            if date in data.index:
                                current_price = data.loc[date, 'Close']
                                
                                if signal['action'] == 'buy' and signal['confidence'] > 0.6:
                                    # 매수
                                    available_cash = cash * market_allocation * 0.2  # 20% 씩 투자
                                    if available_cash > 1000:  # 최소 투자금액
                                        commission = available_cash * self.config.commission_rate
                                        net_amount = available_cash - commission
                                        quantity = net_amount / current_price
                                        
                                        # 포지션 업데이트
                                        if symbol not in positions:
                                            positions[symbol] = {'quantity': 0, 'avg_price': 0}
                                        
                                        old_quantity = positions[symbol]['quantity']
                                        old_avg_price = positions[symbol]['avg_price']
                                        new_quantity = old_quantity + quantity
                                        new_avg_price = ((old_quantity * old_avg_price) + (quantity * current_price)) / new_quantity
                                        
                                        positions[symbol] = {
                                            'quantity': new_quantity,
                                            'avg_price': new_avg_price
                                        }
                                        
                                        cash -= available_cash
                                        
                                        # 거래 기록
                                        trade_records.append(TradeRecord(
                                            date=date.strftime('%Y-%m-%d'),
                                            market=market,
                                            symbol=symbol,
                                            action='buy',
                                            quantity=quantity,
                                            price=current_price,
                                            amount=available_cash - commission,
                                            commission=commission,
                                            confidence=signal['confidence'],
                                            strategy=f"{market}_strategy",
                                            reasoning=signal['reasoning']
                                        ))
                                
                                elif signal['action'] == 'sell' and symbol in positions:
                                    # 매도
                                    if positions[symbol]['quantity'] > 0:
                                        quantity = positions[symbol]['quantity'] * 0.5  # 절반 매도
                                        gross_amount = quantity * current_price
                                        commission = gross_amount * self.config.commission_rate
                                        net_amount = gross_amount - commission
                                        
                                        positions[symbol]['quantity'] -= quantity
                                        cash += net_amount
                                        
                                        # 거래 기록
                                        trade_records.append(TradeRecord(
                                            date=date.strftime('%Y-%m-%d'),
                                            market=market,
                                            symbol=symbol,
                                            action='sell',
                                            quantity=quantity,
                                            price=current_price,
                                            amount=net_amount,
                                            commission=commission,
                                            confidence=signal['confidence'],
                                            strategy=f"{market}_strategy",
                                            reasoning=signal['reasoning']
                                        ))
                
                # 포지션 가치 계산
                for symbol, position in positions.items():
                    if position['quantity'] > 0:
                        # 해당 심볼의 현재가 찾기
                        for market, symbols_data in market_data.items():
                            if symbol in symbols_data and date in symbols_data[symbol].index:
                                current_price = symbols_data[symbol].loc[date, 'Close']
                                daily_portfolio_value += position['quantity'] * current_price
                                break
                
                # 일별 기록
                equity_curve.append({
                    'Date': date,
                    'Portfolio_Value': daily_portfolio_value,
                    'Cash': cash,
                    'Positions_Value': daily_portfolio_value - cash
                })
            
            # DataFrame 변환
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('Date', inplace=True)
            
            logger.info(f"백테스팅 완료: {len(trade_records)}건 거래, 최종 가치: ${daily_portfolio_value:,.0f}")
            
            return equity_df, trade_records
            
        except Exception as e:
            logger.error(f"백테스팅 시뮬레이션 실패: {e}")
            return pd.DataFrame(), []
    
    def _calculate_performance_metrics(self, equity_curve: pd.DataFrame, 
                                     trade_records: List[TradeRecord]) -> PerformanceMetrics:
        """성과 지표 계산"""
        try:
            if equity_curve.empty:
                return PerformanceMetrics(
                    total_return=0, annual_return=0, volatility=0, sharpe_ratio=0,
                    max_drawdown=0, calmar_ratio=0, win_rate=0, total_trades=0,
                    winning_trades=0, losing_trades=0, avg_win=0, avg_loss=0, profit_factor=0
                )
            
            # 기본 계산
            initial_value = self.config.initial_capital
            final_value = equity_curve['Portfolio_Value'].iloc[-1]
            total_return = (final_value / initial_value) - 1
            
            # 일일 수익률
            daily_returns = equity_curve['Portfolio_Value'].pct_change().dropna()
            
            # 연환산 수익률
            trading_days = len(daily_returns)
            years = trading_days / 252  # 연간 252 거래일
            annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            
            # 변동성 (연환산)
            volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
            
            # 샤프 비율 (무위험 수익률 2% 가정)
            excess_return = annual_return - 0.02
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # 최대 손실폭 (MDD)
            peak = equity_curve['Portfolio_Value'].cummax()
            drawdown = (equity_curve['Portfolio_Value'] - peak) / peak
            max_drawdown = drawdown.min()
            
            # 칼마 비율
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # 거래 분석
            total_trades = len(trade_records)
            buy_trades = [t for t in trade_records if t.action == 'buy']
            sell_trades = [t for t in trade_records if t.action == 'sell']
            
            # 수익/손실 거래 분석 (매수-매도 페어로)
            winning_trades = 0
            losing_trades = 0
            wins = []
            losses = []
            
            # 간단한 거래 수익성 계산
            for sell_trade in sell_trades:
                # 해당 심볼의 이전 매수 거래 찾기
                buy_trade = None
                for buy in reversed(buy_trades):
                    if (buy.symbol == sell_trade.symbol and 
                        buy.date <= sell_trade.date):
                        buy_trade = buy
                        break
                
                if buy_trade:
                    profit = (sell_trade.price - buy_trade.price) / buy_trade.price
                    if profit > 0:
                        winning_trades += 1
                        wins.append(profit)
                    else:
                        losing_trades += 1
                        losses.append(abs(profit))
            
            win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = (winning_trades * avg_win) / (losing_trades * avg_loss) if (losing_trades > 0 and avg_loss > 0) else 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor
            )
            
        except Exception as e:
            logger.error(f"성과 지표 계산 실패: {e}")
            return PerformanceMetrics(
                total_return=0, annual_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, calmar_ratio=0, win_rate=0, total_trades=0,
                winning_trades=0, losing_trades=0, avg_win=0, avg_loss=0, profit_factor=0
            )
    
    async def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """통합 백테스팅 실행"""
        try:
            self.is_running = True
            self.progress = 0.0
            self.current_status = "백테스팅 시작"
            self.start_time = time.time()
            self.config = config
            
            logger.info(f"🚀 통합 백테스팅 시작: {config.start_date} ~ {config.end_date}")
            
            # 1단계: 테스트 종목 정의
            self.current_status = "테스트 종목 수집 중..."
            self.progress = 10.0
            
            test_symbols = {
                'US': ['AAPL', 'MSFT', 'GOOGL'],          # 미국 대표주
                'JP': ['7203.T', '6758.T', '9984.T'],     # 일본 대표주 (토요타, 소니, 소프트뱅크)
                'COIN': ['BTC-KRW', 'ETH-KRW']            # 암호화폐
            }
            
            # 2단계: 시장 데이터 수집
            self.current_status = "시장 데이터 수집 중..."
            self.progress = 30.0
            
            market_data = {}
            for market, symbols in test_symbols.items():
                market_data[market] = {}
                for symbol in symbols:
                    data = await self._get_market_data(symbol, market)
                    if not data.empty:
                        market_data[market][symbol] = data
                    await asyncio.sleep(0.1)  # API 제한 고려
            
            # 3단계: 전략 신호 생성
            self.current_status = "전략 신호 생성 중..."
            self.progress = 50.0
            
            market_signals = {}
            for market, symbols_data in market_data.items():
                market_signals[market] = {}
                for symbol, data in symbols_data.items():
                    signals = self._simulate_strategy_signals(data, market, symbol)
                    market_signals[market][symbol] = signals
            
            # 4단계: 백테스팅 시뮬레이션
            self.current_status = "백테스팅 시뮬레이션 실행 중..."
            self.progress = 70.0
            
            equity_curve, trade_records = self._execute_backtest_simulation(
                market_data, market_signals
            )
            
            # 5단계: 성과 분석
            self.current_status = "성과 지표 계산 중..."
            self.progress = 90.0
            
            performance_metrics = self._calculate_performance_metrics(equity_curve, trade_records)
            
            # 6단계: 결과 정리
            self.current_status = "결과 정리 중..."
            self.progress = 95.0
            
            # 시장별 성과
            market_performance = {}
            for market in ['US', 'JP', 'COIN']:
                market_trades = [t for t in trade_records if t.market == market]
                market_performance[market] = {
                    'trades': len(market_trades),
                    'allocation': getattr(config, f"{market.lower()}_allocation", 0),
                    'symbols': list(test_symbols.get(market, []))
                }
            
            # 월별 수익률
            monthly_returns = equity_curve['Portfolio_Value'].resample('M').last().pct_change().dropna()
            daily_returns = equity_curve['Portfolio_Value'].pct_change().dropna()
            
            # 드로우다운 계산
            peak = equity_curve['Portfolio_Value'].cummax()
            drawdown_series = (equity_curve['Portfolio_Value'] - peak) / peak
            
            # 벤치마크 비교 (간단한 시장 수익률)
            benchmark_comparison = {
                'strategy_return': performance_metrics.total_return,
                'market_return': 0.08,  # 가정: 8% 시장 수익률
                'outperformance': performance_metrics.total_return - 0.08
            }
            
            self.current_status = "완료"
            self.progress = 100.0
            
            result = BacktestResult(
                config=config,
                equity_curve=equity_curve,
                trade_records=trade_records,
                performance_metrics=performance_metrics,
                market_performance=market_performance,
                daily_returns=daily_returns,
                monthly_returns=monthly_returns,
                drawdown_series=drawdown_series,
                benchmark_comparison=benchmark_comparison
            )
            
            elapsed_time = time.time() - self.start_time
            logger.info(f"✅ 백테스팅 완료 ({elapsed_time:.1f}초)")
            logger.info(f"📊 총 수익률: {performance_metrics.total_return*100:.1f}%, "
                       f"샤프비율: {performance_metrics.sharpe_ratio:.2f}")
            
            return result
            
        except Exception as e:
            self.current_status = f"오류: {str(e)