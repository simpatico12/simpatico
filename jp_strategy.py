#!/usr/bin/env python3
"""
🏆 YEN-HUNTER: 전설적인 일본 주식 퀸트 전략 + IBKR 실거래
==========================================================
🎯 핵심: 엔화가 모든 것을 지배한다
⚡ 원칙: 단순함이 최고다  
🚀 목표: 자동화가 승리한다
💰 실거래: IBKR 연동 완료

Version: LEGENDARY 2.0 (IBKR 통합)
Author: 퀸트팀 & Claude
"""

import asyncio
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from dotenv import load_dotenv

# IBKR 연결 (선택적 import)
try:
    from ib_insync import *
    import ib_insync as ib
    HAS_IBKR = True
    print("🔗 IBKR 라이브러리 로드 성공")
except ImportError:
    HAS_IBKR = False
    print("⚠️ IBKR 라이브러리 없음 (가상거래만 가능)")

# 환경변수 로드
load_dotenv()

# ============================================================================
# 🔧 전설의 설정 (IBKR 통합)
# ============================================================================
class Config:
    """전설적인 설정 (IBKR + 환경변수 통합)"""
    # 엔화 임계값 (핵심)
    YEN_STRONG = float(os.getenv('YEN_STRONG', 105.0))    # 이하면 내수주 폭탄
    YEN_WEAK = float(os.getenv('YEN_WEAK', 110.0))        # 이상이면 수출주 전력
    
    # 선별 기준
    MIN_MARKET_CAP = float(os.getenv('MIN_MARKET_CAP', 5e11))   # 5000억엔 이상
    TARGET_STOCKS = int(os.getenv('TARGET_STOCKS', 15))         # 탑15 선별
    
    # 매매 임계값
    BUY_THRESHOLD = float(os.getenv('BUY_THRESHOLD', 0.7))      # 70% 이상이면 매수
    
    # 🚀 IBKR 실거래 설정
    LIVE_TRADING = os.getenv('LIVE_TRADING', 'false').lower() == 'true'
    IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
    IBKR_PORT = int(os.getenv('IBKR_PORT', 7497))  # 7497=live, 7496=paper
    IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', 1))
    
    # 💰 포지션 크기 설정
    BASE_POSITION_SIZE = float(os.getenv('BASE_POSITION_SIZE', 1000000))  # 100만엔
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))  # 최대 5개 포지션
    
    # 🛡️ 리스크 관리
    BASE_STOP_LOSS = float(os.getenv('BASE_STOP_LOSS', 0.08))      # 8%
    BASE_TAKE_PROFIT = float(os.getenv('BASE_TAKE_PROFIT', 0.15))  # 15%
    MAX_HOLD_DAYS = int(os.getenv('MAX_HOLD_DAYS', 30))           # 30일
    
    # 백테스팅
    BACKTEST_PERIOD = "1y"  # 1년 백테스트

# ============================================================================
# 🔗 IBKR 연결 클래스
# ============================================================================
class IBKRConnector:
    """🔗 IBKR 실거래 연결"""
    
    def __init__(self):
        self.ib = None
        self.connected = False
        
        if not HAS_IBKR:
            print("⚠️ IBKR 라이브러리 없음: pip install ib_insync")
            return
            
        if not Config.LIVE_TRADING:
            print("💡 가상거래 모드 (실거래 원하면 LIVE_TRADING=true)")
            return
            
        self.ib = IB()
        
    async def connect(self) -> bool:
        """IBKR 연결"""
        if not self.ib or not Config.LIVE_TRADING:
            return False
            
        try:
            await self.ib.connectAsync(
                host=Config.IBKR_HOST,
                port=Config.IBKR_PORT,
                clientId=Config.IBKR_CLIENT_ID,
                timeout=20
            )
            self.connected = True
            print(f"✅ IBKR 연결 성공: {Config.IBKR_HOST}:{Config.IBKR_PORT}")
            
            # 계좌 정보 확인
            accounts = self.ib.managedAccounts()
            if accounts:
                print(f"📊 연결된 계좌: {accounts[0]}")
            
            return True
            
        except Exception as e:
            print(f"❌ IBKR 연결 실패: {e}")
            print("💡 TWS나 IB Gateway가 실행중인지 확인하세요")
            self.connected = False
            return False
    
    async def disconnect(self):
        """IBKR 연결 해제"""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            print("🔌 IBKR 연결 해제")
    
    def create_jp_stock_contract(self, symbol: str) -> Contract:
        """일본 주식 계약 생성"""
        # 7203.T -> 7203
        clean_symbol = symbol.replace('.T', '')
        
        contract = Stock(
            symbol=clean_symbol,
            exchange='TSE',  # Tokyo Stock Exchange
            currency='JPY'
        )
        return contract
    
    async def place_buy_order(self, symbol: str, shares: int, order_type: str = 'MKT') -> bool:
        """매수 주문"""
        if not self.connected:
            print(f"⚠️ IBKR 연결 안됨 - {symbol} 가상매수만")
            return False
            
        try:
            contract = self.create_jp_stock_contract(symbol)
            
            if order_type == 'MKT':
                order = MarketOrder('BUY', shares)
            else:
                # 추후 지정가 주문 등 추가 가능
                order = MarketOrder('BUY', shares)
            
            trade = self.ib.placeOrder(contract, order)
            print(f"📈 IBKR 매수 주문: {symbol} {shares:,}주")
            
            # 주문 상태 확인 (간단히)
            await asyncio.sleep(1)
            if trade.orderStatus.status in ['Filled', 'Submitted']:
                print(f"✅ {symbol} 매수 주문 성공")
                return True
            else:
                print(f"⚠️ {symbol} 매수 주문 대기중")
                return True
                
        except Exception as e:
            print(f"❌ IBKR 매수 주문 실패 {symbol}: {e}")
            return False
    
    async def place_sell_order(self, symbol: str, shares: int, order_type: str = 'MKT') -> bool:
        """매도 주문"""
        if not self.connected:
            print(f"⚠️ IBKR 연결 안됨 - {symbol} 가상매도만")
            return False
            
        try:
            contract = self.create_jp_stock_contract(symbol)
            
            if order_type == 'MKT':
                order = MarketOrder('SELL', shares)
            else:
                order = MarketOrder('SELL', shares)
            
            trade = self.ib.placeOrder(contract, order)
            print(f"📉 IBKR 매도 주문: {symbol} {shares:,}주")
            
            await asyncio.sleep(1)
            if trade.orderStatus.status in ['Filled', 'Submitted']:
                print(f"✅ {symbol} 매도 주문 성공")
                return True
            else:
                print(f"⚠️ {symbol} 매도 주문 대기중")
                return True
                
        except Exception as e:
            print(f"❌ IBKR 매도 주문 실패 {symbol}: {e}")
            return False
    
    async def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약"""
        if not self.connected:
            return {}
            
        try:
            positions = self.ib.positions()
            portfolio = []
            
            for pos in positions:
                if pos.contract.exchange == 'TSE':  # 일본 주식만
                    portfolio.append({
                        'symbol': f"{pos.contract.symbol}.T",
                        'shares': pos.position,
                        'avg_cost': pos.avgCost,
                        'market_value': pos.marketValue,
                        'unrealized_pnl': pos.unrealizedPNL
                    })
            
            return {
                'positions': portfolio,
                'total_positions': len(portfolio),
                'total_value': sum([p['market_value'] for p in portfolio])
            }
            
        except Exception as e:
            print(f"❌ 포트폴리오 조회 실패: {e}")
            return {}
class LegendaryIndicators:
    """🏆 전설적인 고급 기술지표 (직접 계산)"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> float:
        """전설의 RSI (30이하 매수, 70이상 매도)"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[str, Dict]:
        """🚀 전설의 MACD (추세 전환의 신)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            current_macd = float(macd_line.iloc[-1])
            current_signal = float(signal_line.iloc[-1])
            current_hist = float(histogram.iloc[-1])
            prev_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0
            
            # 신호 판정
            if current_macd > current_signal and current_hist > 0:
                if prev_hist <= 0:  # 골든크로스
                    signal_type = "GOLDEN_CROSS"
                else:
                    signal_type = "BULLISH"
            elif current_macd < current_signal and current_hist < 0:
                if prev_hist >= 0:  # 데드크로스
                    signal_type = "DEAD_CROSS"
                else:
                    signal_type = "BEARISH"
            else:
                signal_type = "NEUTRAL"
            
            details = {
                'macd': current_macd,
                'signal': current_signal,
                'histogram': current_hist,
                'strength': abs(current_hist)
            }
            
            return signal_type, details
        except:
            return "NEUTRAL", {}
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[str, Dict]:
        """💎 전설의 볼린저밴드 (변동성의 마법사)"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = float(prices.iloc[-1])
            upper = float(upper_band.iloc[-1])
            middle = float(sma.iloc[-1])
            lower = float(lower_band.iloc[-1])
            
            # 밴드 위치 계산 (0~1)
            band_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
            
            # 신호 판정
            if current_price >= upper:
                signal = "UPPER_BREAK"  # 상단 돌파 (과매수)
            elif current_price <= lower:
                signal = "LOWER_BREAK"  # 하단 돌파 (과매도 = 매수기회!)
            elif band_position >= 0.8:
                signal = "UPPER_ZONE"   # 상단 근접
            elif band_position <= 0.2:
                signal = "LOWER_ZONE"   # 하단 근접 (매수 관심)
            else:
                signal = "MIDDLE_ZONE"  # 중간대
            
            # 밴드 폭 (변동성 측정)
            band_width = (upper - lower) / middle if middle != 0 else 0
            
            details = {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'position': band_position,
                'width': band_width,
                'squeeze': band_width < 0.1  # 볼린저 스퀴즈
            }
            
            return signal, details
        except:
            return "MIDDLE_ZONE", {}
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[str, Dict]:
        """⚡ 전설의 스토캐스틱 (모멘텀의 황제)"""
        try:
            lowest_low = low.rolling(k_period).min()
            highest_high = high.rolling(k_period).max()
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
            d_percent = k_percent.rolling(d_period).mean()
            
            current_k = float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else 50
            current_d = float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else 50
            prev_k = float(k_percent.iloc[-2]) if len(k_percent) > 1 and not pd.isna(k_percent.iloc[-2]) else current_k
            
            # 신호 판정
            if current_k <= 20 and current_d <= 20:
                signal = "OVERSOLD"      # 과매도 (강력한 매수 신호!)
            elif current_k >= 80 and current_d >= 80:
                signal = "OVERBOUGHT"    # 과매수 (매도 신호)
            elif current_k > current_d and prev_k <= current_d:
                signal = "BULLISH_CROSS" # 골든크로스
            elif current_k < current_d and prev_k >= current_d:
                signal = "BEARISH_CROSS" # 데드크로스
            else:
                signal = "NEUTRAL"
            
            details = {
                'k_percent': current_k,
                'd_percent': current_d,
                'momentum': current_k - prev_k
            }
            
            return signal, details
        except:
            return "NEUTRAL", {}
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """🔥 전설의 ATR (변동성의 척도)"""
        try:
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = true_range.rolling(period).mean()
            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def fibonacci_levels(prices: pd.Series, period: int = 50) -> Dict:
        """🌟 전설의 피보나치 (황금비율의 마법)"""
        try:
            recent_data = prices.tail(period)
            high_price = float(recent_data.max())
            low_price = float(recent_data.min())
            current_price = float(prices.iloc[-1])
            
            # 피보나치 되돌림 레벨
            diff = high_price - low_price
            levels = {
                'high': high_price,
                'low': low_price,
                'fib_23.6': high_price - (diff * 0.236),
                'fib_38.2': high_price - (diff * 0.382),
                'fib_50.0': high_price - (diff * 0.500),
                'fib_61.8': high_price - (diff * 0.618),
                'fib_78.6': high_price - (diff * 0.786)
            }
            
            # 현재가가 어느 레벨 근처인지 확인
            tolerance = diff * 0.02  # 2% 허용 오차
            near_level = None
            
            for level_name, level_price in levels.items():
                if abs(current_price - level_price) <= tolerance:
                    near_level = level_name
                    break
            
            return {
                'levels': levels,
                'current_price': current_price,
                'near_level': near_level,
                'trend_direction': 'UP' if current_price > levels['fib_50.0'] else 'DOWN'
            }
        except:
            return {}
    
    @staticmethod
    def momentum_oscillator(prices: pd.Series, period: int = 10) -> Tuple[str, float]:
        """🚀 전설의 모멘텀 (가속도의 신)"""
        try:
            momentum = ((prices / prices.shift(period)) - 1) * 100
            current_momentum = float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0
            
            # 모멘텀 시그널
            if current_momentum > 5:
                signal = "STRONG_BULLISH"
            elif current_momentum > 2:
                signal = "BULLISH"
            elif current_momentum < -5:
                signal = "STRONG_BEARISH"
            elif current_momentum < -2:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"
            
            return signal, current_momentum
        except:
            return "NEUTRAL", 0.0
    
    @staticmethod
    def volume_analysis(prices: pd.Series, volumes: pd.Series) -> Dict:
        """📊 전설의 볼륨 분석 (돈의 흐름을 읽는다)"""
        try:
            # 가격 변화량
            price_change = prices.pct_change()
            
            # On-Balance Volume (OBV)
            obv = (volumes * np.sign(price_change)).cumsum()
            obv_trend = "UP" if obv.iloc[-1] > obv.iloc[-10] else "DOWN"
            
            # Volume Weighted Average Price (VWAP) 근사
            vwap = (prices * volumes).rolling(20).sum() / volumes.rolling(20).sum()
            current_vwap = float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else 0
            current_price = float(prices.iloc[-1])
            
            # 거래량 급증
            recent_vol = volumes.tail(3).mean()
            avg_vol = volumes.tail(20).head(17).mean()
            volume_spike = recent_vol > avg_vol * 1.5
            volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            
            # 가격-거래량 발산
            price_up = price_change.iloc[-1] > 0
            volume_up = volumes.iloc[-1] > volumes.iloc[-2]
            
            if price_up and volume_up:
                pv_signal = "BULLISH_CONFIRM"     # 상승 + 거래량 증가 (강세 확인)
            elif not price_up and volume_up:
                pv_signal = "BEARISH_VOLUME"      # 하락 + 거래량 증가 (약세 확인)
            elif price_up and not volume_up:
                pv_signal = "WEAK_RALLY"          # 상승하지만 거래량 부족 (약한 상승)
            else:
                pv_signal = "NEUTRAL"
            
            return {
                'obv_trend': obv_trend,
                'vwap': current_vwap,
                'price_vs_vwap': 'ABOVE' if current_price > current_vwap else 'BELOW',
                'volume_spike': volume_spike,
                'volume_ratio': volume_ratio,
                'price_volume_signal': pv_signal
            }
        except:
            return {}
    
    @staticmethod
    def trend_signal(prices: pd.Series) -> str:
        """전설의 추세 체크 (5일 > 20일 > 60일)"""
        try:
            ma5 = prices.rolling(5).mean().iloc[-1]
            ma20 = prices.rolling(20).mean().iloc[-1]
            ma60 = prices.rolling(60).mean().iloc[-1]
            current = prices.iloc[-1]
            
            if ma5 > ma20 > ma60 and current > ma5:
                return "STRONG_UP"
            elif ma5 < ma20 < ma60 and current < ma5:
                return "STRONG_DOWN"
            else:
                return "SIDEWAYS"
        except:
            return "SIDEWAYS"
    
    @staticmethod
    def volume_spike(volumes: pd.Series) -> bool:
        """거래량 급증 체크"""
        try:
            recent = volumes.tail(3).mean()
            avg = volumes.tail(20).head(17).mean()
            return recent > avg * 1.5
        except:
            return False

# ============================================================================
# 🔍 전설의 종목 헌터 (닛케이225 실시간)
# ============================================================================
class StockHunter:
    """전설적인 종목 헌터"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; YenHunter/1.0)'
        })
        
        # 백업 탑종목 (크롤링 실패시)
        self.backup_stocks = [
            '7203.T', '6758.T', '9984.T', '6861.T', '8306.T',  # 대형주
            '7974.T', '9432.T', '8316.T', '6367.T', '4063.T',  # 우량주
            '9983.T', '8411.T', '6954.T', '7201.T', '6981.T'   # 안정주
        ]
    
    async def hunt_nikkei225(self) -> List[str]:
        """닛케이225 실시간 헌팅"""
        try:
            url = "https://finance.yahoo.com/quote/%5EN225/components"
            response = self.session.get(url, timeout=15)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            symbols = set()
            
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if '/quote/' in href and '.T' in href:
                    symbol = href.split('/quote/')[-1].split('?')[0]
                    if symbol.endswith('.T') and len(symbol) <= 8:
                        symbols.add(symbol)
            
            return list(symbols)[:50] if symbols else self.backup_stocks
            
        except Exception as e:
            print(f"⚠️ 크롤링 실패, 백업 사용: {e}")
            return self.backup_stocks
    
    async def select_legends(self, symbols: List[str]) -> List[Dict]:
        """전설급 종목들만 선별"""
        legends = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                market_cap = info.get('marketCap', 0)
                avg_volume = info.get('averageVolume', 0)
                
                # 기준 통과 체크
                if market_cap >= Config.MIN_MARKET_CAP and avg_volume >= 1e6:
                    score = self._calculate_legend_score(market_cap, avg_volume, info)
                    
                    legends.append({
                        'symbol': symbol,
                        'market_cap': market_cap,
                        'score': score,
                        'sector': info.get('sector', 'Unknown')
                    })
                    
            except:
                continue
        
        # 점수순 정렬 후 탑15 선별
        legends.sort(key=lambda x: x['score'], reverse=True)
        return legends[:Config.TARGET_STOCKS]
    
    def _calculate_legend_score(self, market_cap: float, volume: float, info: Dict) -> float:
        """전설 점수 계산"""
        score = 0.0
        
        # 시가총액 점수 (40%)
        if market_cap >= 2e12:      # 2조엔 이상
            score += 0.4
        elif market_cap >= 1e12:    # 1조엔 이상  
            score += 0.3
        else:                       # 5000억엔 이상
            score += 0.2
            
        # 거래량 점수 (30%)
        if volume >= 5e6:           # 500만주 이상
            score += 0.3
        elif volume >= 2e6:         # 200만주 이상
            score += 0.2
        else:                       # 100만주 이상
            score += 0.1
            
        # 재무 건전성 (30%)
        pe = info.get('trailingPE', 999)
        if 5 <= pe <= 25:
            score += 0.3
        elif 25 < pe <= 40:
            score += 0.15
            
        return min(score, 1.0)

# ============================================================================
# 🎯 전설의 신호 생성기
# ============================================================================
@dataclass
class LegendarySignal:
    """전설적인 매매 신호"""
    symbol: str
    action: str         # BUY/SELL/HOLD
    confidence: float   # 0.0 ~ 1.0
    price: float
    reason: str
    yen_rate: float
    
    # 🏆 전설의 기술지표들
    rsi: float
    trend: str
    macd_signal: str
    macd_strength: float
    bb_signal: str
    bb_position: float
    stoch_signal: str
    stoch_k: float
    atr: float
    momentum_signal: str
    momentum_value: float
    volume_signal: str
    fibonacci_level: str
    
    timestamp: datetime
    
    # 🛡️ 리스크 관리
    stop_loss: float    # 손절가
    take_profit1: float # 1차 익절가 (50% 매도)
    take_profit2: float # 2차 익절가 (나머지 매도)
    max_hold_days: int  # 최대 보유일
    position_size: int  # 포지션 크기

@dataclass 
class Position:
    """포지션 관리"""
    symbol: str
    buy_price: float
    shares: int
    buy_date: datetime
    stop_loss: float
    take_profit1: float
    take_profit2: float
    max_hold_date: datetime
    shares_sold_1st: int = 0  # 1차 익절 매도량

class SignalGenerator:
    """전설적인 신호 생성기"""
    
    def __init__(self):
        self.current_usd_jpy = 107.5
        self.indicators = LegendaryIndicators()
    
    def calculate_risk_levels(self, price: float, confidence: float, stock_type: str, yen_signal: str, atr: float = 0) -> Tuple[float, float, float, int]:
        """🛡️ 전설적인 리스크 관리 계산 (ATR + 환경변수 기반)"""
        
        # 환경변수 기반 기본값
        base_stop = Config.BASE_STOP_LOSS
        base_profit1 = Config.BASE_TAKE_PROFIT
        base_profit2 = Config.BASE_TAKE_PROFIT * 1.7  # 25%
        base_days = Config.MAX_HOLD_DAYS
        
        # ATR 기반 조정 (변동성 고려)
        if atr > 0:
            atr_ratio = atr / price
            if atr_ratio > 0.03:  # 고변동성
                base_stop *= 1.3  # 손절 넓게
                base_profit1 *= 1.4  # 익절도 크게
                base_profit2 *= 1.5
            elif atr_ratio < 0.015:  # 저변동성
                base_stop *= 0.8  # 손절 타이트
                base_profit1 *= 0.9
                base_profit2 *= 0.95
        
        # 신뢰도에 따른 조정
        if confidence >= 0.8:
            # 고신뢰도: 손절 넓게, 익절 크게
            stop_rate = base_stop * 1.2
            profit1_rate = base_profit1 * 1.3
            profit2_rate = base_profit2 * 1.4
            max_days = base_days + 15
        elif confidence >= 0.6:
            # 중신뢰도: 기본값
            stop_rate = base_stop
            profit1_rate = base_profit1
            profit2_rate = base_profit2
            max_days = base_days
        else:
            # 저신뢰도: 손절 타이트, 익절 작게
            stop_rate = base_stop * 0.7
            profit1_rate = base_profit1 * 0.8
            profit2_rate = base_profit2 * 0.9
            max_days = base_days - 10
        
        # 엔화 + 종목타입에 따른 조정
        if (yen_signal == "STRONG" and stock_type == "DOMESTIC") or \
           (yen_signal == "WEAK" and stock_type == "EXPORT"):
            # 유리한 조건: 손절 넓게, 익절 크게
            stop_rate *= 0.8
            profit1_rate *= 1.2
            profit2_rate *= 1.3
            max_days += 10
        elif yen_signal == "NEUTRAL":
            # 중립: 기본값 유지
            pass
        else:
            # 불리한 조건: 손절 타이트, 익절 작게
            stop_rate *= 1.2
            profit1_rate *= 0.9
            profit2_rate *= 0.95
            max_days -= 5
        
        # 최종 가격 계산
        stop_loss = price * (1 - stop_rate)
        take_profit1 = price * (1 + profit1_rate)
        take_profit2 = price * (1 + profit2_rate)
        
        # 범위 제한
        max_days = max(15, min(60, max_days))
        
        return stop_loss, take_profit1, take_profit2, max_days
    
    async def update_yen(self):
        """USD/JPY 업데이트"""
        try:
            ticker = yf.Ticker("USDJPY=X")
            data = ticker.history(period="1d")
            if not data.empty:
                self.current_usd_jpy = float(data['Close'].iloc[-1])
        except:
            pass  # 기본값 유지
    
    def get_yen_signal(self) -> Tuple[str, float]:
        """엔화 신호 분석"""
        if self.current_usd_jpy <= Config.YEN_STRONG:
            return "STRONG", 0.4  # 엔화 강세 = 내수주 유리
        elif self.current_usd_jpy >= Config.YEN_WEAK:
            return "WEAK", 0.4    # 엔화 약세 = 수출주 유리
        else:
            return "NEUTRAL", 0.2
    
    def classify_stock_type(self, symbol: str) -> str:
        """수출주/내수주 분류"""
        # 간단 분류 (실제로는 섹터 정보 활용)
        export_symbols = ['7203.T', '6758.T', '7974.T', '6861.T', '9984.T', 
                         '6954.T', '7201.T', '6981.T', '4063.T']
        
        return "EXPORT" if symbol in export_symbols else "DOMESTIC"
    
    def _calculate_legendary_score(self, symbol: str, rsi: float, trend: str, 
                                 macd_signal: str, macd_details: Dict,
                                 bb_signal: str, bb_details: Dict,
                                 stoch_signal: str, stoch_details: Dict,
                                 momentum_signal: str, momentum_value: float,
                                 volume_analysis: Dict) -> float:
        """🏆 전설의 종합 점수 계산 (고급 지표 통합)"""
        score = 0.0
        
        # 1. 엔화 기반 점수 (25%) - 여전히 핵심
        yen_signal, yen_score = self.get_yen_signal()
        stock_type = self.classify_stock_type(symbol)
        
        if (yen_signal == "STRONG" and stock_type == "DOMESTIC") or \
           (yen_signal == "WEAK" and stock_type == "EXPORT"):
            score += 0.25
        else:
            score += 0.125
        
        # 2. RSI 점수 (15%)
        if rsi <= 30:           # 과매도 = 매수기회
            score += 0.15
        elif 30 < rsi <= 50:    # 건전한 수준
            score += 0.12
        elif 50 < rsi <= 70:    # 상승 중
            score += 0.08
        else:                   # 과매수 = 위험
            score += 0.03
        
        # 3. MACD 점수 (15%)
        if macd_signal == "GOLDEN_CROSS":
            score += 0.15  # 골든크로스 = 최고점수
        elif macd_signal == "BULLISH":
            score += 0.12
        elif macd_signal == "DEAD_CROSS":
            score += 0.02  # 데드크로스 = 최저점수
        elif macd_signal == "BEARISH":
            score += 0.05
        else:  # NEUTRAL
            score += 0.08
        
        # 4. 볼린저밴드 점수 (12%)
        if bb_signal == "LOWER_BREAK":
            score += 0.12  # 하단 돌파 = 매수기회
        elif bb_signal == "LOWER_ZONE":
            score += 0.10
        elif bb_signal == "UPPER_BREAK":
            score += 0.03  # 상단 돌파 = 과매수
        elif bb_signal == "UPPER_ZONE":
            score += 0.05
        else:  # MIDDLE_ZONE
            score += 0.08
        
        # 5. 스토캐스틱 점수 (10%)
        if stoch_signal == "OVERSOLD":
            score += 0.10  # 과매도 = 매수기회
        elif stoch_signal == "BULLISH_CROSS":
            score += 0.08
        elif stoch_signal == "OVERBOUGHT":
            score += 0.02  # 과매수 = 위험
        elif stoch_signal == "BEARISH_CROSS":
            score += 0.03
        else:  # NEUTRAL
            score += 0.06
        
        # 6. 모멘텀 점수 (8%)
        if momentum_signal == "STRONG_BULLISH":
            score += 0.08
        elif momentum_signal == "BULLISH":
            score += 0.06
        elif momentum_signal == "STRONG_BEARISH":
            score += 0.01
        elif momentum_signal == "BEARISH":
            score += 0.03
        else:  # NEUTRAL
            score += 0.05
        
        # 7. 추세 점수 (10%)
        if trend == "STRONG_UP":
            score += 0.10
        elif trend == "SIDEWAYS":
            score += 0.05
        else:  # STRONG_DOWN
            score += 0.02
        
        # 8. 거래량 점수 (5%)
        volume_signal = volume_analysis.get('price_volume_signal', 'NEUTRAL')
        if volume_signal == "BULLISH_CONFIRM":
            score += 0.05
        elif volume_signal == "WEAK_RALLY":
            score += 0.02
        elif volume_signal == "BEARISH_VOLUME":
            score += 0.01
        else:
            score += 0.03
        
        return min(score, 1.0)
    
    def _generate_legendary_reason(self, symbol: str, rsi: float, trend: str,
                                 macd_signal: str, bb_signal: str, stoch_signal: str,
                                 momentum_signal: str, volume_analysis: Dict) -> str:
        """🎯 전설의 이유 생성 (고급 지표 포함)"""
        reasons = []
        
        # 엔화 이유
        yen_signal, _ = self.get_yen_signal()
        stock_type = self.classify_stock_type(symbol)
        
        if yen_signal == "STRONG" and stock_type == "DOMESTIC":
            reasons.append("엔화강세+내수주")
        elif yen_signal == "WEAK" and stock_type == "EXPORT":
            reasons.append("엔화약세+수출주")
        else:
            reasons.append(f"엔화{yen_signal.lower()}")
        
        # 핵심 지표들
        if macd_signal == "GOLDEN_CROSS":
            reasons.append("MACD골든크로스")
        elif macd_signal == "DEAD_CROSS":
            reasons.append("MACD데드크로스")
        else:
            reasons.append(f"MACD{macd_signal.lower()}")
        
        if bb_signal == "LOWER_BREAK":
            reasons.append("볼린저하단돌파")
        elif bb_signal == "UPPER_BREAK":
            reasons.append("볼린저상단돌파")
        else:
            reasons.append(f"볼린저{bb_signal.lower()}")
        
        if stoch_signal == "OVERSOLD":
            reasons.append("스토캐스틱과매도")
        elif stoch_signal == "OVERBOUGHT":
            reasons.append("스토캐스틱과매수")
        elif "CROSS" in stoch_signal:
            reasons.append(f"스토캐스틱{stoch_signal.lower()}")
        
        # RSI
        if rsi <= 30:
            reasons.append(f"RSI과매도({rsi:.0f})")
        elif rsi >= 70:
            reasons.append(f"RSI과매수({rsi:.0f})")
        else:
            reasons.append(f"RSI({rsi:.0f})")
        
        # 추세
        reasons.append(f"추세{trend.lower()}")
        
        # 거래량
        volume_signal = volume_analysis.get('price_volume_signal', 'NEUTRAL')
        if volume_signal == "BULLISH_CONFIRM":
            reasons.append("거래량확인")
        elif volume_signal == "WEAK_RALLY":
            reasons.append("거래량부족")
        
        return " | ".join(reasons[:6])  # 최대 6개까지만

    async def generate_signal(self, symbol: str) -> LegendarySignal:
        """🏆 전설적인 신호 생성 (고급 지표 통합)"""
        try:
            # 엔화 업데이트
            await self.update_yen()
            
            # 주식 데이터 수집
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            
            if data.empty:
                raise ValueError("데이터 없음")
            
            current_price = float(data['Close'].iloc[-1])
            
            # 🏆 전설의 고급 기술지표 계산
            rsi = self.indicators.rsi(data['Close'])
            trend = self.indicators.trend_signal(data['Close'])
            macd_signal, macd_details = self.indicators.macd(data['Close'])
            bb_signal, bb_details = self.indicators.bollinger_bands(data['Close'])
            stoch_signal, stoch_details = self.indicators.stochastic(data['High'], data['Low'], data['Close'])
            atr_value = self.indicators.atr(data['High'], data['Low'], data['Close'])
            momentum_signal, momentum_value = self.indicators.momentum_oscillator(data['Close'])
            volume_analysis = self.indicators.volume_analysis(data['Close'], data['Volume'])
            fib_analysis = self.indicators.fibonacci_levels(data['Close'])
            
            # 🎯 전설의 종합 점수 계산
            total_score = self._calculate_legendary_score(
                symbol, rsi, trend, macd_signal, macd_details,
                bb_signal, bb_details, stoch_signal, stoch_details,
                momentum_signal, momentum_value, volume_analysis
            )
            
            # 최종 판단
            if total_score >= Config.BUY_THRESHOLD:
                action = "BUY"
                confidence = min(total_score, 0.95)
            elif total_score <= 0.3:
                action = "SELL" 
                confidence = min(1 - total_score, 0.95)
            else:
                action = "HOLD"
                confidence = 0.5
            
            # 🛡️ 리스크 관리 계산 (ATR 기반 개선)
            yen_signal, _ = self.get_yen_signal()
            stock_type = self.classify_stock_type(symbol)
            
            if action == "BUY":
                stop_loss, take_profit1, take_profit2, max_hold_days = self.calculate_risk_levels(
                    current_price, confidence, stock_type, yen_signal, atr_value
                )
                # 포지션 크기 (환경변수 기반)
                base_amount = Config.BASE_POSITION_SIZE
                position_size = int((base_amount * confidence) / current_price / 100) * 100  # 100주 단위
            else:
                stop_loss = take_profit1 = take_profit2 = 0.0
                max_hold_days = 0
                position_size = 0
            
            # 🎯 전설의 이유 생성
            reason = self._generate_legendary_reason(
                symbol, rsi, trend, macd_signal, bb_signal, 
                stoch_signal, momentum_signal, volume_analysis
            )
            
            return LegendarySignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                reason=reason,
                yen_rate=self.current_usd_jpy,
                
                # 🏆 전설의 기술지표
                rsi=rsi,
                trend=trend,
                macd_signal=macd_signal,
                macd_strength=macd_details.get('strength', 0),
                bb_signal=bb_signal,
                bb_position=bb_details.get('position', 0.5),
                stoch_signal=stoch_signal,
                stoch_k=stoch_details.get('k_percent', 50),
                atr=atr_value,
                momentum_signal=momentum_signal,
                momentum_value=momentum_value,
                volume_signal=volume_analysis.get('price_volume_signal', 'NEUTRAL'),
                fibonacci_level=fib_analysis.get('near_level', 'NONE'),
                
                timestamp=datetime.now(),
                stop_loss=stop_loss,
                take_profit1=take_profit1,
                take_profit2=take_profit2,
                max_hold_days=max_hold_days,
                position_size=position_size
            )
            
        except Exception as e:
            return LegendarySignal(
                symbol=symbol, action="HOLD", confidence=0.0, price=0.0,
                reason=f"분석실패: {e}", yen_rate=self.current_usd_jpy,
                rsi=50.0, trend="UNKNOWN", macd_signal="NEUTRAL", macd_strength=0,
                bb_signal="MIDDLE_ZONE", bb_position=0.5, stoch_signal="NEUTRAL", stoch_k=50,
                atr=0, momentum_signal="NEUTRAL", momentum_value=0, volume_signal="NEUTRAL",
                fibonacci_level="NONE", timestamp=datetime.now(),
                stop_loss=0.0, take_profit1=0.0, take_profit2=0.0, max_hold_days=0, position_size=0
            )

# ============================================================================
# 🛡️ 전설의 포지션 매니저 (IBKR 통합)
# ============================================================================
class PositionManager:
    """전설적인 포지션 관리 (IBKR 실거래 + 가상거래)"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.closed_positions = []
        
        # 🔗 IBKR 연결
        self.ibkr = IBKRConnector()
        self.ibkr_connected = False
        
    async def initialize_ibkr(self):
        """IBKR 초기화"""
        if Config.LIVE_TRADING and HAS_IBKR:
            self.ibkr_connected = await self.ibkr.connect()
            if self.ibkr_connected:
                print("🚀 IBKR 실거래 모드 활성화")
            else:
                print("⚠️ IBKR 연결 실패, 가상거래로 전환")
        else:
            print("💡 가상거래 모드")
    
    async def open_position(self, signal: LegendarySignal):
        """포지션 오픈 (IBKR 실거래 + 가상거래)"""
        if signal.action != "BUY" or signal.position_size <= 0:
            return
            
        # 최대 포지션 수 체크
        if len(self.positions) >= Config.MAX_POSITIONS:
            print(f"⚠️ 최대 포지션 수({Config.MAX_POSITIONS}) 초과, {signal.symbol} 매수 취소")
            return
        
        # 🔗 IBKR 실거래 시도
        ibkr_success = False
        if self.ibkr_connected:
            try:
                ibkr_success = await self.ibkr.place_buy_order(signal.symbol, signal.position_size)
            except Exception as e:
                print(f"❌ IBKR 매수 실패 {signal.symbol}: {e}")
        
        # 포지션 기록 (실거래 성공 여부와 관계없이)
        position = Position(
            symbol=signal.symbol,
            buy_price=signal.price,
            shares=signal.position_size,
            buy_date=signal.timestamp,
            stop_loss=signal.stop_loss,
            take_profit1=signal.take_profit1,
            take_profit2=signal.take_profit2,
            max_hold_date=signal.timestamp + pd.Timedelta(days=signal.max_hold_days)
        )
        self.positions[signal.symbol] = position
        
        # 결과 출력
        trade_type = "🚀 실거래" if ibkr_success else "💡 가상거래"
        print(f"✅ {trade_type} {signal.symbol} 포지션 오픈: {signal.position_size:,}주 @ {signal.price:,.0f}엔")
        print(f"   🛡️ 손절: {signal.stop_loss:,.0f}엔 (-{((signal.price-signal.stop_loss)/signal.price*100):.1f}%)")
        print(f"   🎯 1차익절: {signal.take_profit1:,.0f}엔 (+{((signal.take_profit1-signal.price)/signal.price*100):.1f}%)")
        print(f"   🚀 2차익절: {signal.take_profit2:,.0f}엔 (+{((signal.take_profit2-signal.price)/signal.price*100):.1f}%)")
    
    async def close_position_with_ibkr(self, symbol: str, shares: int, reason: str) -> bool:
        """IBKR 매도 실행"""
        if not self.ibkr_connected:
            return False
            
        try:
            success = await self.ibkr.place_sell_order(symbol, shares)
            if success:
                print(f"🚀 IBKR 실거래 매도: {symbol} {shares:,}주 ({reason})")
            return success
        except Exception as e:
            print(f"❌ IBKR 매도 실패 {symbol}: {e}")
            return False
    
    async def check_positions(self) -> List[Dict]:
        """포지션 체크 및 매도 신호"""
        actions = []
        current_time = datetime.now()
        
        for symbol, position in list(self.positions.items()):
            try:
                # 현재가 조회
                stock = yf.Ticker(symbol)
                current_data = stock.history(period="1d")
                if current_data.empty:
                    continue
                
                current_price = float(current_data['Close'].iloc[-1])
                
                # 손절 체크
                if current_price <= position.stop_loss:
                    pnl = (current_price - position.buy_price) / position.buy_price * 100
                    shares_to_sell = position.shares - position.shares_sold_1st
                    
                    # IBKR 실거래 매도 시도
                    ibkr_success = await self.close_position_with_ibkr(symbol, shares_to_sell, "손절")
                    
                    actions.append({
                        'action': 'STOP_LOSS',
                        'symbol': symbol,
                        'shares': shares_to_sell,
                        'price': current_price,
                        'pnl': pnl,
                        'reason': f'손절 실행 ({pnl:.1f}%)',
                        'ibkr_executed': ibkr_success
                    })
                    self._close_position(symbol, current_price, 'STOP_LOSS')
                    
                # 1차 익절 체크 (50% 매도)
                elif current_price >= position.take_profit1 and position.shares_sold_1st == 0:
                    shares_to_sell = position.shares // 2
                    pnl = (current_price - position.buy_price) / position.buy_price * 100
                    
                    # IBKR 실거래 매도 시도
                    ibkr_success = await self.close_position_with_ibkr(symbol, shares_to_sell, "1차익절")
                    
                    actions.append({
                        'action': 'TAKE_PROFIT_1',
                        'symbol': symbol,
                        'shares': shares_to_sell,
                        'price': current_price,
                        'pnl': pnl,
                        'reason': f'1차 익절 ({pnl:.1f}%) - 50% 매도',
                        'ibkr_executed': ibkr_success
                    })
                    position.shares_sold_1st = shares_to_sell
                    
                # 2차 익절 체크 (나머지 전량 매도)
                elif current_price >= position.take_profit2:
                    remaining_shares = position.shares - position.shares_sold_1st
                    pnl = (current_price - position.buy_price) / position.buy_price * 100
                    
                    # IBKR 실거래 매도 시도
                    ibkr_success = await self.close_position_with_ibkr(symbol, remaining_shares, "2차익절")
                    
                    actions.append({
                        'action': 'TAKE_PROFIT_2',
                        'symbol': symbol,
                        'shares': remaining_shares,
                        'price': current_price,
                        'pnl': pnl,
                        'reason': f'2차 익절 ({pnl:.1f}%) - 전량 매도',
                        'ibkr_executed': ibkr_success
                    })
                    self._close_position(symbol, current_price, 'TAKE_PROFIT_2')
                    
                # 최대 보유기간 초과
                elif current_time >= position.max_hold_date:
                    remaining_shares = position.shares - position.shares_sold_1st
                    pnl = (current_price - position.buy_price) / position.buy_price * 100
                    
                    # IBKR 실거래 매도 시도
                    ibkr_success = await self.close_position_with_ibkr(symbol, remaining_shares, "시간만료")
                    
                    actions.append({
                        'action': 'TIME_EXIT',
                        'symbol': symbol,
                        'shares': remaining_shares,
                        'price': current_price,
                        'pnl': pnl,
                        'reason': f'보유기간 만료 ({pnl:.1f}%)',
                        'ibkr_executed': ibkr_success
                    })
                    self._close_position(symbol, current_price, 'TIME_EXIT')
                
                # 트레일링 스톱 (고급 기능)
                else:
                    # 수익이 15% 이상일 때 손절가를 매수가 근처로 올림
                    if current_price >= position.buy_price * 1.15:
                        new_stop = position.buy_price * 1.02  # 매수가 +2%로 손절 조정
                        if new_stop > position.stop_loss:
                            position.stop_loss = new_stop
                            print(f"📈 {symbol} 트레일링 스톱 조정: {new_stop:,.0f}엔")
                            
            except Exception as e:
                print(f"⚠️ {symbol} 포지션 체크 실패: {e}")
                continue
        
        return actions
    
    def _close_position(self, symbol: str, exit_price: float, exit_reason: str):
        """포지션 종료"""
        if symbol in self.positions:
            position = self.positions[symbol]
            pnl = (exit_price - position.buy_price) / position.buy_price * 100
            
            self.closed_positions.append({
                'symbol': symbol,
                'buy_price': position.buy_price,
                'exit_price': exit_price,
                'shares': position.shares,
                'pnl': pnl,
                'exit_reason': exit_reason,
                'hold_days': (datetime.now() - position.buy_date).days,
                'exit_date': datetime.now()
            })
            
            del self.positions[symbol]
            print(f"🔚 {symbol} 포지션 종료: {pnl:.1f}% ({exit_reason})")
    
    def get_portfolio_status(self) -> Dict:
        """포트폴리오 현황"""
        total_value = 0
        total_pnl = 0
        
        for position in self.positions.values():
            # 간단 계산 (실제로는 실시간 가격 조회 필요)
            value = position.buy_price * position.shares
            total_value += value
        
        if self.closed_positions:
            total_pnl = sum([pos['pnl'] for pos in self.closed_positions]) / len(self.closed_positions)
        
        return {
            'open_positions': len(self.positions),
            'closed_trades': len(self.closed_positions),
            'total_value': total_value,
            'avg_pnl': total_pnl,
            'positions': list(self.positions.keys())
        }

# ============================================================================
# 🏆 전설의 메인 엔진
# ============================================================================
class YenHunter:
    """전설적인 YEN-HUNTER 메인 엔진"""
    
    def __init__(self):
        self.hunter = StockHunter()
        self.signal_gen = SignalGenerator()
        self.position_mgr = PositionManager()
        self.selected_stocks = []
        
        print("🏆 YEN-HUNTER 초기화 완료!")
        print(f"💱 엔화 임계값: 강세({Config.YEN_STRONG}) 약세({Config.YEN_WEAK})")
        print(f"🎯 선별 기준: 시총{Config.MIN_MARKET_CAP/1e11:.0f}천억엔+ 탑{Config.TARGET_STOCKS}개")
        print("🛡️ 리스크 관리: ATR 기반 동적 손절/익절 + 트레일링 스톱")
    
    async def full_trading_cycle(self) -> Dict:
        """🚀 완전한 매매 사이클 (신호 + 포지션 관리)"""
        print("\n🔥 전설적인 완전 매매 사이클 시작!")
        
        # 1단계: 기존 포지션 체크
        print("🛡️ 기존 포지션 체크...")
        position_actions = await self.position_mgr.check_positions()
        
        for action in position_actions:
            emoji = "🛑" if action['action'] == 'STOP_LOSS' else "💰" if 'PROFIT' in action['action'] else "⏰"
            print(f"{emoji} {action['symbol']}: {action['reason']}")
        
        # 2단계: 새로운 매수 기회 탐색
        print("\n🔍 새로운 기회 탐색...")
        signals = await self.hunt_and_analyze()
        
        # 3단계: 매수 신호 실행
        buy_signals = [s for s in signals if s.action == 'BUY' and s.symbol not in self.position_mgr.positions]
        
        executed_buys = []
        for signal in buy_signals[:3]:  # 최대 3개까지만
            self.position_mgr.open_position(signal)
            executed_buys.append(signal)
        
        # 4단계: 포트폴리오 현황
        portfolio = self.position_mgr.get_portfolio_status()
        
        return {
            'timestamp': datetime.now(),
            'position_actions': position_actions,
            'new_signals': len(signals),
            'executed_buys': len(executed_buys),
            'portfolio': portfolio,
            'top_signals': signals[:5] if signals else []
        }
    
    async def monitor_positions(self):
        """포지션 모니터링 (실시간 실행용)"""
        print("👁️ 포지션 모니터링 시작...")
        
        while True:
            try:
                actions = await self.position_mgr.check_positions()
                
                if actions:
                    print(f"\n⚡ {len(actions)}개 액션 발생:")
                    for action in actions:
                        print(f"  {action['symbol']}: {action['reason']}")
                
                # 포트폴리오 현황
                portfolio = self.position_mgr.get_portfolio_status()
                if portfolio['open_positions'] > 0:
                    print(f"📊 현재 포트폴리오: {portfolio['open_positions']}개 포지션, 평균 수익률: {portfolio['avg_pnl']:.1f}%")
                
                await asyncio.sleep(300)  # 5분마다 체크
                
            except KeyboardInterrupt:
                print("🛑 모니터링 종료")
                break
            except Exception as e:
                print(f"⚠️ 모니터링 오류: {e}")
                await asyncio.sleep(60)  # 1분 후 재시도
    
    async def hunt_and_analyze(self) -> List[LegendarySignal]:
        """전설적인 헌팅 + 분석"""
        print("\n🔍 전설적인 종목 헌팅 시작...")
        start_time = time.time()
        
        try:
            # 1단계: 닛케이225 헌팅
            symbols = await self.hunter.hunt_nikkei225()
            print(f"📡 닛케이225 수집: {len(symbols)}개")
            
            # 2단계: 전설급 선별
            legends = await self.hunter.select_legends(symbols)
            print(f"🏆 전설급 선별: {len(legends)}개")
            
            self.selected_stocks = legends
            
            # 3단계: 신호 생성
            signals = []
            for i, stock in enumerate(legends, 1):
                print(f"⚡ 분석 중... {i}/{len(legends)} - {stock['symbol']}")
                signal = await self.signal_gen.generate_signal(stock['symbol'])
                signals.append(signal)
                await asyncio.sleep(0.1)  # API 제한
            
            elapsed = time.time() - start_time
            
            # 결과 요약
            buy_count = len([s for s in signals if s.action == 'BUY'])
            sell_count = len([s for s in signals if s.action == 'SELL'])
            
            print(f"\n🎯 전설적인 분석 완료! ({elapsed:.1f}초)")
            print(f"📊 매수:{buy_count} 매도:{sell_count} 보유:{len(signals)-buy_count-sell_count}")
            print(f"💱 USD/JPY: {self.signal_gen.current_usd_jpy:.2f}")
            
            return signals
            
        except Exception as e:
            print(f"❌ 헌팅 실패: {e}")
            return []
    
    async def analyze_single(self, symbol: str) -> LegendarySignal:
        """단일 종목 분석"""
        return await self.signal_gen.generate_signal(symbol)
    
    def get_top_signals(self, signals: List[LegendarySignal], action: str = "BUY", top: int = 5) -> List[LegendarySignal]:
        """상위 신호 추출"""
        filtered = [s for s in signals if s.action == action]
        return sorted(filtered, key=lambda x: x.confidence, reverse=True)[:top]

# ============================================================================
# 📈 간단 백테스터
# ============================================================================
class SimpleBacktester:
    """전설적인 간단 백테스터"""
    
    @staticmethod
    async def backtest_symbol(symbol: str, period: str = "1y") -> Dict:
        """단일 종목 백테스트"""
        try:
            # 과거 데이터
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if len(data) < 60:
                return {"error": "데이터 부족"}
            
            # 간단 전략: RSI 30 매수, 70 매도
            indicators = LegendaryIndicators()
            
            returns = []
            position = 0
            buy_price = 0
            
            for i in range(60, len(data)):
                current_data = data.iloc[:i+1]
                rsi = indicators.rsi(current_data['Close'])
                price = current_data['Close'].iloc[-1]
                
                # 매수 신호
                if rsi <= 30 and position == 0:
                    position = 1
                    buy_price = price
                
                # 매도 신호  
                elif rsi >= 70 and position == 1:
                    ret = (price - buy_price) / buy_price
                    returns.append(ret)
                    position = 0
            
            if returns:
                total_return = np.prod([1 + r for r in returns]) - 1
                win_rate = len([r for r in returns if r > 0]) / len(returns)
                avg_return = np.mean(returns)
                
                return {
                    "symbol": symbol,
                    "total_return": total_return * 100,
                    "win_rate": win_rate * 100,
                    "avg_return": avg_return * 100,
                    "trades": len(returns)
                }
            else:
                return {"error": "거래 없음"}
                
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# 🎮 편의 함수들
# ============================================================================
async def hunt_jp_legends() -> List[LegendarySignal]:
    """일본 전설급 종목 헌팅"""
    hunter = YenHunter()
    return await hunter.hunt_and_analyze()

async def analyze_jp_single(symbol: str) -> LegendarySignal:
    """단일 종목 분석"""
    hunter = YenHunter()
    return await hunter.analyze_single(symbol)

async def backtest_jp(symbol: str) -> Dict:
    """백테스트 실행"""
    return await SimpleBacktester.backtest_symbol(symbol)

# ============================================================================
# 🧪 테스트 실행
# ============================================================================
async def main():
    """전설적인 테스트"""
    print("🏆 YEN-HUNTER 전설적인 테스트 시작!")
    print("="*50)
    
    # 전체 헌팅 + 분석
    signals = await hunt_jp_legends()
    
    if signals:
        # 상위 매수 추천
        top_buys = YenHunter().get_top_signals(signals, "BUY", 3)
        
        print(f"\n🎯 전설적인 매수 추천:")
        for i, signal in enumerate(top_buys, 1):
            print(f"{i}. {signal.symbol}: {signal.confidence:.1%} 신뢰도")
            print(f"   💰 {signal.price:,.0f}엔 | 포지션: {signal.position_size:,}주")
            print(f"   🛡️ 손절: {signal.stop_loss:,.0f}엔 (-{((signal.price-signal.stop_loss)/signal.price*100):.1f}%)")
            print(f"   🎯 1차익절: {signal.take_profit1:,.0f}엔 (+{((signal.take_profit1-signal.price)/signal.price*100):.1f}%)")
            print(f"   🚀 2차익절: {signal.take_profit2:,.0f}엔 (+{((signal.take_profit2-signal.price)/signal.price*100):.1f}%)")
            print(f"   ⏰ 최대보유: {signal.max_hold_days}일")
            print(f"   🏆 고급지표: RSI({signal.rsi:.0f}) MACD({signal.macd_signal}) BB({signal.bb_signal})")
            print(f"   📊 스토캐스틱({signal.stoch_signal}) 모멘텀({signal.momentum_signal}) ATR({signal.atr:.1f})")
            print(f"   📈 추세({signal.trend}) 거래량({signal.volume_signal}) 피보나치({signal.fibonacci_level})")
            print(f"   💡 {signal.reason}")
        
        # 실제 포지션 관리 테스트
        print(f"\n🛡️ 포지션 관리 시스템 테스트:")
        hunter = YenHunter()
        
        # 가상 매수 실행
        if top_buys:
            print(f"   📝 {top_buys[0].symbol} 가상 포지션 오픈:")
            hunter.position_mgr.open_position(top_buys[0])
            
            # 포트폴리오 현황
            portfolio = hunter.position_mgr.get_portfolio_status()
            print(f"   📊 포트폴리오: {portfolio['open_positions']}개 포지션")
            
            # 포지션 체크 시뮬레이션
            print(f"   🔍 포지션 체크 시뮬레이션:")
            actions = await hunter.position_mgr.check_positions()
            if actions:
                print(f"      ⚡ {len(actions)}개 액션 발생")
            else:
                print(f"      ✅ 모든 포지션 정상")
        
        # 완전 매매 사이클 테스트
        print(f"\n🚀 완전 매매 사이클 테스트:")
        cycle_result = await hunter.full_trading_cycle()
        print(f"   📊 새 신호: {cycle_result['new_signals']}개")
        print(f"   💰 실행된 매수: {cycle_result['executed_buys']}개")
        print(f"   🛡️ 포지션 액션: {len(cycle_result['position_actions'])}개")
        
        # 백테스트 (첫 번째 종목)
        if top_buys:
            print(f"\n📈 {top_buys[0].symbol} 백테스트:")
            backtest_result = await backtest_jp(top_buys[0].symbol)
            if "error" not in backtest_result:
                print(f"   📊 총수익: {backtest_result['total_return']:.1f}%")
                print(f"   🎯 승률: {backtest_result['win_rate']:.1f}%")
                print(f"   💹 거래횟수: {backtest_result['trades']}회")
    
    print("\n✅ 전설적인 테스트 완료!")
    print("\n🚀 YEN-HUNTER 특징:")
    print("  ⚡ 800라인 전설급 완전체")
    print("  💱 엔화 기반 수출/내수 매칭")
    print("  🔍 닛케이225 실시간 헌팅")  
    print("  🏆 전설의 고급 기술지표 (8개)")
    print("    - RSI + MACD + 볼린저밴드")
    print("    - 스토캐스틱 + ATR + 모멘텀")
    print("    - 피보나치 + 고급거래량분석")
    print("  🛡️ ATR 기반 동적 손절/익절")
    print("  💰 분할 익절 (1차 50%, 2차 전량)")
    print("  ⏰ 변동성 고려 보유기간 관리")
    print("  🤖 완전 자동화 포지션 관리")
    print("  📈 백테스트 내장")
    print("\n💡 실전 사용법:")
    print("  - await hunter.full_trading_cycle() : 완전 매매 사이클")
    print("  - await hunter.monitor_positions() : 실시간 포지션 모니터링")
    print("  - hunter.position_mgr.get_portfolio_status() : 포트폴리오 현황")
    print("\n🎯 전설의 리스크 관리:")
    print("  - 신뢰도별 동적 손절률 (5-15%)")
    print("  - ATR 기반 변동성 조정")
    print("  - 2단계 분할 익절 (15%, 25%)")
    print("  - 엔화 상황별 목표 조정")
    print("  - 트레일링 스톱으로 수익 보호")
    print("  - 최대 보유기간으로 리스크 제한")
    print("\n🏆 전설의 고급 지표 시스템:")
    print("  - MACD 골든/데드크로스 감지")
    print("  - 볼린저밴드 돌파 포착")
    print("  - 스토캐스틱 과매수/과매도")
    print("  - 피보나치 레벨 지지/저항")
    print("  - 거래량-가격 발산 분석")
    print("  - ATR 변동성 측정")
    print("  - 모멘텀 가속도 추적")

if __name__ == "__main__":
    asyncio.run(main())
