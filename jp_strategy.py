#!/usr/bin/env python3
"""
🏆 YEN-HUNTER v2.1 OPTIMIZED: 비용최적화 AI 확신도 체크
===============================================================================
🎯 핵심: 엔화 + 화목 집중 + 3차 익절 + 최소 AI 활용
⚡ 원칙: 기술적 분석 우선 + AI는 애매한 상황에서만
🚀 목표: 월 14% (화 2.5% + 목 1.5%) × 4주
💰 AI 비용: 월 5천원 이하 최적화

AI 역할 최소화:
- 신뢰도 0.4-0.7 구간에서만 AI 호출
- 뉴스분석/시장심리분석 완전 제거
- 순수 기술적 분석 위주
- 확신도 체크만 수행
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

# OpenAI 연동 (최소 사용)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI 시뮬레이션 모드 (pip install openai)")

# IBKR 연동
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("⚠️ IBKR 시뮬레이션 모드")

# ============================================================================
# 🔧 화목 하이브리드 설정
# ============================================================================
class Config:
    # 엔화 임계값
    YEN_STRONG = 105.0
    YEN_WEAK = 110.0
    
    # 선별 기준
    MIN_MARKET_CAP = 5e11
    TARGET_STOCKS = 15
    
    # 화목 스케줄
    TRADING_DAYS = [1, 3]  # 화, 목
    TUESDAY_MAX_HOLD = 5
    THURSDAY_MAX_HOLD = 2
    MAX_TUESDAY_TRADES = 2
    MAX_THURSDAY_TRADES = 3
    
    # 월간 목표
    JAPAN_MONTHLY_TARGET = 0.14
    JAPAN_MONTHLY_SAFE = 0.10
    JAPAN_MONTHLY_LIMIT = -0.05
    
    # 매수 임계값
    BUY_THRESHOLD_TUESDAY = 0.75
    BUY_THRESHOLD_THURSDAY = 0.65
    
    # AI 최적화 설정
    AI_CONFIDENCE_MIN = 0.4  # 이 이하는 AI 호출 안함
    AI_CONFIDENCE_MAX = 0.7  # 이 이상은 AI 호출 안함
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = "gpt-3.5-turbo"  # 비용 절약
    
    # IBKR
    IBKR_HOST = '127.0.0.1'
    IBKR_PORT = 7497
    IBKR_CLIENT_ID = 1
    
    # 데이터
    DATA_DIR = Path("yen_hunter_data")

# ============================================================================
# 🤖 최소 AI 확신도 체커 (비용 최적화)
# ============================================================================
class OptimizedAIChecker:
    def __init__(self):
        self.available = OPENAI_AVAILABLE and Config.OPENAI_API_KEY
        self.call_count = 0
        self.monthly_cost = 0.0
        
        if self.available:
            openai.api_key = Config.OPENAI_API_KEY
        else:
            print("⚠️ OpenAI API 키 없음 - 시뮬레이션 모드")
    
    def should_use_ai(self, confidence: float) -> bool:
        """AI 사용 여부 결정 (비용 최적화)"""
        # 애매한 구간에서만 AI 사용
        if Config.AI_CONFIDENCE_MIN <= confidence <= Config.AI_CONFIDENCE_MAX:
            return True
        return False
    
    async def check_confidence(self, symbol: str, technical_data: Dict, confidence: float) -> Dict:
        """확신도 체크만 (매우 간단한 AI 호출)"""
        if not self.available or not self.should_use_ai(confidence):
            return {
                'ai_adjustment': 0.0,
                'final_confidence': confidence,
                'ai_used': False,
                'reason': 'AI 호출 불필요' if not self.should_use_ai(confidence) else 'AI 비활성화'
            }
        
        try:
            # 매우 간단한 프롬프트 (토큰 절약)
            prompt = f"""기술분석 확신도 체크:
{symbol}: RSI {technical_data.get('rsi', 50):.0f}, MACD {technical_data.get('macd_signal', 'N/A')}, 
BB {technical_data.get('bb_signal', 'N/A')}, 현재 확신도 {confidence:.1%}

매수 확신도를 -0.2 ~ +0.2 범위로 조정하세요. 숫자만 답하세요."""
            
            response = openai.ChatCompletion.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "기술적 분석 확신도 조정 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,  # 매우 제한적
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # 숫자 추출
            try:
                adjustment = float(ai_response.replace('%', '').replace('+', ''))
                adjustment = max(-0.2, min(0.2, adjustment))  # 범위 제한
            except:
                adjustment = 0.0
            
            final_confidence = max(0.0, min(1.0, confidence + adjustment))
            
            self.call_count += 1
            self.monthly_cost += 0.002  # 대략적인 비용
            
            return {
                'ai_adjustment': adjustment,
                'final_confidence': final_confidence,
                'ai_used': True,
                'reason': f'AI 조정: {adjustment:+.1%}'
            }
            
        except Exception as e:
            print(f"⚠️ AI 확신도 체크 실패: {e}")
            return {
                'ai_adjustment': 0.0,
                'final_confidence': confidence,
                'ai_used': False,
                'reason': 'AI 오류'
            }
    
    def get_usage_stats(self) -> Dict:
        """사용 통계"""
        return {
            'monthly_calls': self.call_count,
            'estimated_cost': self.monthly_cost,
            'avg_cost_per_call': self.monthly_cost / self.call_count if self.call_count > 0 else 0
        }

# ============================================================================
# 📊 핵심 기술지표 6개 (trend_analysis 오류 수정)
# ============================================================================
class Indicators:
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
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
        """MACD 계산"""
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
            
            if current_macd > current_signal and current_hist > 0:
                if prev_hist <= 0:
                    signal_type = "GOLDEN_CROSS"
                else:
                    signal_type = "BULLISH"
            elif current_macd < current_signal and current_hist < 0:
                if prev_hist >= 0:
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
        """볼린저밴드 계산"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = float(prices.iloc[-1])
            upper = float(upper_band.iloc[-1])
            middle = float(sma.iloc[-1])
            lower = float(lower_band.iloc[-1])
            
            band_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
            
            if current_price >= upper:
                signal = "UPPER_BREAK"
            elif current_price <= lower:
                signal = "LOWER_BREAK"
            elif band_position >= 0.8:
                signal = "UPPER_ZONE"
            elif band_position <= 0.2:
                signal = "LOWER_ZONE"
            else:
                signal = "MIDDLE_ZONE"
            
            band_width = (upper - lower) / middle if middle != 0 else 0
            
            details = {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'position': band_position,
                'width': band_width,
                'squeeze': band_width < 0.1
            }
            
            return signal, details
        except:
            return "MIDDLE_ZONE", {}
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[str, Dict]:
        """스토캐스틱 계산"""
        try:
            lowest_low = low.rolling(k_period).min()
            highest_high = high.rolling(k_period).max()
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
            d_percent = k_percent.rolling(d_period).mean()
            
            current_k = float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else 50
            current_d = float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else 50
            prev_k = float(k_percent.iloc[-2]) if len(k_percent) > 1 and not pd.isna(k_percent.iloc[-2]) else current_k
            
            if current_k <= 20 and current_d <= 20:
                signal = "OVERSOLD"
            elif current_k >= 80 and current_d >= 80:
                signal = "OVERBOUGHT"
            elif current_k > current_d and prev_k <= current_d:
                signal = "BULLISH_CROSS"
            elif current_k < current_d and prev_k >= current_d:
                signal = "BEARISH_CROSS"
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
        """ATR 계산"""
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
    def volume_analysis(prices: pd.Series, volumes: pd.Series) -> Dict:
        """거래량 분석"""
        try:
            price_change = prices.pct_change()
            
            # On-Balance Volume
            obv = (volumes * np.sign(price_change)).cumsum()
            obv_trend = "UP" if obv.iloc[-1] > obv.iloc[-10] else "DOWN"
            
            # Volume spike
            recent_vol = volumes.tail(3).mean()
            avg_vol = volumes.tail(20).head(17).mean()
            volume_spike = recent_vol > avg_vol * 1.5
            volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            
            # Price-Volume divergence
            price_up = price_change.iloc[-1] > 0
            volume_up = volumes.iloc[-1] > volumes.iloc[-2]
            
            if price_up and volume_up:
                pv_signal = "BULLISH_CONFIRM"
            elif not price_up and volume_up:
                pv_signal = "BEARISH_VOLUME"
            elif price_up and not volume_up:
                pv_signal = "WEAK_RALLY"
            else:
                pv_signal = "NEUTRAL"
            
            return {
                'obv_trend': obv_trend,
                'volume_spike': volume_spike,
                'volume_ratio': volume_ratio,
                'price_volume_signal': pv_signal
            }
        except:
            return {}
    
    @staticmethod
    def trend_signal(prices: pd.Series) -> str:
        """추세 신호 (오류 수정)"""
        try:
            # 데이터 길이 체크
            if len(prices) < 60:
                return "SIDEWAYS"
            
            ma5 = prices.rolling(5).mean()
            ma20 = prices.rolling(20).mean()
            ma60 = prices.rolling(60).mean()
            
            # NaN 값 체크
            if pd.isna(ma5.iloc[-1]) or pd.isna(ma20.iloc[-1]) or pd.isna(ma60.iloc[-1]):
                return "SIDEWAYS"
            
            current = prices.iloc[-1]
            ma5_val = ma5.iloc[-1]
            ma20_val = ma20.iloc[-1]
            ma60_val = ma60.iloc[-1]
            
            if ma5_val > ma20_val > ma60_val and current > ma5_val:
                return "STRONG_UP"
            elif ma5_val < ma20_val < ma60_val and current < ma5_val:
                return "STRONG_DOWN"
            else:
                return "SIDEWAYS"
        except Exception as e:
            print(f"⚠️ 추세 분석 오류: {e}")
            return "SIDEWAYS"

# ============================================================================
# 🔍 3개 지수 통합 종목 헌터
# ============================================================================
class StockHunter:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; YenHunter/2.1)'})
        
        self.backup_stocks = [
            '7203.T', '6758.T', '9984.T', '6861.T', '8306.T',
            '7974.T', '9432.T', '8316.T', '6367.T', '4063.T',
            '9983.T', '8411.T', '6954.T', '7201.T', '6981.T'
        ]
    
    async def hunt_japanese_stocks(self) -> List[str]:
        """3개 지수 통합 헌팅"""
        all_symbols = set()
        
        # 1. 닛케이225
        nikkei_symbols = await self.hunt_nikkei225()
        all_symbols.update(nikkei_symbols)
        print(f"📡 닛케이225: {len(nikkei_symbols)}개")
        
        # 2. TOPIX
        topix_symbols = await self.hunt_topix()
        all_symbols.update(topix_symbols)
        print(f"📊 TOPIX: {len(topix_symbols)}개")
        
        # 3. JPX400
        jpx400_symbols = await self.hunt_jpx400()
        all_symbols.update(jpx400_symbols)
        print(f"🏆 JPX400: {len(jpx400_symbols)}개")
        
        final_symbols = list(all_symbols)
        print(f"🎯 총 수집: {len(final_symbols)}개 종목")
        
        return final_symbols
    
    async def hunt_nikkei225(self) -> List[str]:
        """닛케이225 헌팅"""
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
            print(f"⚠️ 닛케이225 실패, 백업 사용: {e}")
            return self.backup_stocks
    
    async def hunt_topix(self) -> List[str]:
        """TOPIX 헌팅"""
        try:
            symbols = set()
            
            # TOPIX 대형주 추가
            topix_large_caps = [
                '6758.T', '9984.T', '4689.T', '6861.T', '6954.T', '4704.T',
                '7203.T', '7267.T', '7201.T', '7269.T',
                '8306.T', '8316.T', '8411.T', '8604.T', '7182.T', '8766.T',
                '9432.T', '9433.T', '9437.T',
                '9983.T', '3382.T', '8267.T', '3086.T',
                '5020.T', '9501.T', '9502.T', '9503.T',
                '4063.T', '3407.T', '5401.T', '4188.T',
                '4568.T', '4502.T', '4506.T', '4523.T'
            ]
            symbols.update(topix_large_caps)
            
            return list(symbols)[:80]
            
        except Exception as e:
            print(f"⚠️ TOPIX 실패: {e}")
            return []
    
    async def hunt_jpx400(self) -> List[str]:
        """JPX400 헌팅"""
        try:
            symbols = set()
            
            # JPX400 우량주 (수익성 우수)
            jpx400_quality = [
                '6758.T', '6861.T', '9984.T', '4689.T', '6954.T', '4704.T', '8035.T',
                '7203.T', '7267.T', '7269.T',
                '8306.T', '8316.T', '8411.T', '7182.T',
                '4063.T', '3407.T', '4188.T', '5401.T', '4042.T',
                '2914.T', '4911.T', '9983.T', '3382.T',
                '4568.T', '4502.T', '4506.T', '4523.T',
                '1803.T', '8801.T', '8802.T',
                '9432.T', '9433.T', '4307.T', '6367.T',
                '6326.T', '6473.T', '7013.T', '6301.T'
            ]
            symbols.update(jpx400_quality)
            
            return list(symbols)[:60]
            
        except Exception as e:
            print(f"⚠️ JPX400 실패: {e}")
            return []
    
    async def select_legends(self, symbols: List[str]) -> List[Dict]:
        """전설급 자동선별"""
        legends = []
        
        print(f"🔍 {len(symbols)}개 종목 자동선별 시작...")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                if i % 10 == 0:
                    print(f"   ⚡ 선별 진행: {i}/{len(symbols)}")
                
                # 기본 데이터 수집
                stock = yf.Ticker(symbol)
                info = stock.info
                hist = stock.history(period="3mo")
                
                if hist.empty or len(hist) < 60:
                    continue
                
                # 기본 필터링
                market_cap = info.get('marketCap', 0)
                avg_volume = info.get('averageVolume', 0)
                current_price = float(hist['Close'].iloc[-1])
                
                if market_cap < Config.MIN_MARKET_CAP or avg_volume < 1e6 or current_price < 100:
                    continue
                
                # 자동선별 점수 계산
                auto_score = await self._calculate_auto_selection_score(symbol, info, hist)
                
                if auto_score >= 0.6:  # 자동선별 임계값
                    legends.append({
                        'symbol': symbol,
                        'market_cap': market_cap,
                        'score': auto_score,
                        'sector': info.get('sector', 'Unknown'),
                        'current_price': current_price,
                        'avg_volume': avg_volume,
                        'selection_reason': self._get_selection_reason(symbol, info, hist)
                    })
                    
            except Exception as e:
                continue
        
        # 점수순 정렬
        legends.sort(key=lambda x: x['score'], reverse=True)
        selected = legends[:Config.TARGET_STOCKS]
        
        print(f"✅ 자동선별 완료: {len(selected)}개 전설급 종목")
        for i, stock in enumerate(selected[:5], 1):
            print(f"   {i}. {stock['symbol']} (점수: {stock['score']:.2f}) - {stock['selection_reason']}")
        
        return selected
    
    async def _calculate_auto_selection_score(self, symbol: str, info: Dict, hist: pd.DataFrame) -> float:
        """자동선별 종합점수 계산"""
        score = 0.0
        
        try:
            current_price = float(hist['Close'].iloc[-1])
            market_cap = info.get('marketCap', 0)
            avg_volume = info.get('averageVolume', 0)
            
            # 1. 기술적 점수 (40%)
            tech_score = self._calculate_technical_score(hist)
            score += tech_score * 0.4
            
            # 2. 펀더멘털 점수 (30%)
            fundamental_score = self._calculate_fundamental_score(info)
            score += fundamental_score * 0.3
            
            # 3. 거래량/유동성 점수 (20%)
            liquidity_score = self._calculate_liquidity_score(avg_volume, market_cap)
            score += liquidity_score * 0.2
            
            # 4. 엔화 적합성 점수 (10%)
            yen_score = self._calculate_yen_fitness_score(symbol)
            score += yen_score * 0.1
            
            return min(score, 1.0)
            
        except:
            return 0.0
    
    def _calculate_technical_score(self, hist: pd.DataFrame) -> float:
        """기술적 분석 점수"""
        try:
            indicators = Indicators()
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']
            
            score = 0.0
            
            # RSI (25%)
            rsi = indicators.rsi(close)
            if 20 <= rsi <= 40:  # 매수 적정 구간
                score += 0.25
            elif 40 < rsi <= 60:
                score += 0.15
            
            # MACD (25%)
            macd_signal, _ = indicators.macd(close)
            if macd_signal in ["GOLDEN_CROSS", "BULLISH"]:
                score += 0.25
            elif macd_signal == "NEUTRAL":
                score += 0.15
            
            # 볼린저밴드 (20%)
            bb_signal, bb_details = indicators.bollinger_bands(close)
            if bb_signal in ["LOWER_BREAK", "LOWER_ZONE"]:
                score += 0.20
            elif bb_signal == "MIDDLE_ZONE":
                score += 0.15
            
            # 추세 (15%)
            trend = indicators.trend_signal(close)
            if trend == "STRONG_UP":
                score += 0.15
            elif trend == "SIDEWAYS":
                score += 0.10
            
            # 거래량 (15%)
            vol_analysis = indicators.volume_analysis(close, volume)
            if vol_analysis.get('price_volume_signal') == 'BULLISH_CONFIRM':
                score += 0.15
            elif vol_analysis.get('volume_spike', False):
                score += 0.10
            
            return min(score, 1.0)
            
        except:
            return 0.5
    
    def _calculate_fundamental_score(self, info: Dict) -> float:
        """펀더멘털 점수"""
        try:
            score = 0.0
            
            # PER (30%)
            pe_ratio = info.get('trailingPE', 999)
            if 5 <= pe_ratio <= 15:
                score += 0.30
            elif 15 < pe_ratio <= 25:
                score += 0.20
            elif 25 < pe_ratio <= 40:
                score += 0.10
            
            # PBR (25%)
            pb_ratio = info.get('priceToBook', 999)
            if 0.5 <= pb_ratio <= 1.5:
                score += 0.25
            elif 1.5 < pb_ratio <= 3.0:
                score += 0.15
            
            # ROE (20%)
            roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            if roe >= 15:
                score += 0.20
            elif roe >= 10:
                score += 0.15
            elif roe >= 5:
                score += 0.10
            
            # 부채비율 (15%)
            debt_ratio = info.get('debtToEquity', 999)
            if debt_ratio <= 50:
                score += 0.15
            elif debt_ratio <= 100:
                score += 0.10
            elif debt_ratio <= 200:
                score += 0.05
            
            # 배당수익률 (10%)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            if dividend_yield >= 3:
                score += 0.10
            elif dividend_yield >= 1:
                score += 0.05
            
            return min(score, 1.0)
            
        except:
            return 0.5
    
    def _calculate_liquidity_score(self, avg_volume: float, market_cap: float) -> float:
        """유동성 점수"""
        try:
            score = 0.0
            
            # 거래량 점수 (60%)
            if avg_volume >= 10e6:
                score += 0.60
            elif avg_volume >= 5e6:
                score += 0.45
            elif avg_volume >= 2e6:
                score += 0.30
            elif avg_volume >= 1e6:
                score += 0.15
            
            # 시가총액 점수 (40%)
            if market_cap >= 5e12:  # 5조엔
                score += 0.40
            elif market_cap >= 2e12:  # 2조엔
                score += 0.30
            elif market_cap >= 1e12:  # 1조엔
                score += 0.20
            elif market_cap >= 5e11:  # 5000억엔
                score += 0.10
            
            return min(score, 1.0)
            
        except:
            return 0.5
    
    def _calculate_yen_fitness_score(self, symbol: str) -> float:
        """엔화 적합성 점수"""
        try:
            # 수출주 리스트 (엔약세 수혜)
            export_stocks = [
                '7203.T', '6758.T', '7974.T', '6861.T', '9984.T',
                '7267.T', '7269.T', '6326.T', '6473.T', '7013.T',
                '4063.T', '6954.T', '8035.T'
            ]
            
            # 내수주 리스트 (엔강세 수혜)  
            domestic_stocks = [
                '8306.T', '8316.T', '8411.T', '9432.T', '9433.T',
                '3382.T', '8267.T', '9983.T', '2914.T', '4911.T',
                '8801.T', '8802.T', '5401.T'
            ]
            
            if symbol in export_stocks:
                return 1.0  # 수출주 우대
            elif symbol in domestic_stocks:
                return 0.8  # 내수주 적정
            else:
                return 0.6  # 기타
                
        except:
            return 0.6
    
    def _get_selection_reason(self, symbol: str, info: Dict, hist: pd.DataFrame) -> str:
        """선별 이유"""
        try:
            reasons = []
            
            # 시가총액
            market_cap = info.get('marketCap', 0)
            if market_cap >= 2e12:
                reasons.append("대형주")
            elif market_cap >= 1e12:
                reasons.append("중견주")
            
            # PER
            pe = info.get('trailingPE', 999)
            if pe <= 15:
                reasons.append("저PER")
            
            # 기술적
            indicators = Indicators()
            rsi = indicators.rsi(hist['Close'])
            if rsi <= 30:
                reasons.append("과매도")
            
            macd_signal, _ = indicators.macd(hist['Close'])
            if macd_signal == "GOLDEN_CROSS":
                reasons.append("MACD골든")
            
            # 엔화 적합성
            export_stocks = ['7203.T', '6758.T', '7974.T', '6861.T', '9984.T']
            if symbol in export_stocks:
                reasons.append("수출주")
            
            return " | ".join(reasons[:3]) if reasons else "기본선별"
            
        except:
            return "자동선별"

# ============================================================================
# 📈 화목 월간 목표 관리자
# ============================================================================
class JapanMonthlyManager:
    """화목 월간 목표 관리"""
    
    def __init__(self):
        self.data_dir = Config.DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        
        self.current_month = datetime.now().strftime('%Y-%m')
        self.monthly_data = self.load_monthly_data()
        
    def load_monthly_data(self) -> Dict:
        """월간 데이터 로드"""
        try:
            performance_file = self.data_dir / "japan_monthly.json"
            if performance_file.exists():
                with open(performance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get(self.current_month, {
                        'tuesday_trades': [],
                        'thursday_trades': [],
                        'total_pnl': 0.0,
                        'tuesday_pnl': 0.0,
                        'thursday_pnl': 0.0,
                        'trade_count': 0,
                        'win_count': 0,
                        'target_reached': False
                    })
            return {
                'tuesday_trades': [],
                'thursday_trades': [],
                'total_pnl': 0.0,
                'tuesday_pnl': 0.0,
                'thursday_pnl': 0.0,
                'trade_count': 0,
                'win_count': 0,
                'target_reached': False
            }
        except:
            return {
                'tuesday_trades': [],
                'thursday_trades': [],
                'total_pnl': 0.0,
                'tuesday_pnl': 0.0,
                'thursday_pnl': 0.0,
                'trade_count': 0,
                'win_count': 0,
                'target_reached': False
            }
    
    def save_monthly_data(self):
        """월간 데이터 저장"""
        try:
            performance_file = self.data_dir / "japan_monthly.json"
            all_data = {}
            if performance_file.exists():
                with open(performance_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            
            all_data[self.current_month] = self.monthly_data
            
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ 월간 데이터 저장 실패: {e}")
    
    def add_trade(self, symbol: str, pnl: float, entry_price: float, exit_price: float, day_type: str):
        """거래 기록 추가"""
        trade = {
            'symbol': symbol,
            'pnl': pnl,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'timestamp': datetime.now().isoformat(),
            'win': pnl > 0,
            'day_type': day_type
        }
        
        # 요일별 분류
        if day_type == "TUESDAY":
            self.monthly_data['tuesday_trades'].append(trade)
            self.monthly_data['tuesday_pnl'] += pnl
        elif day_type == "THURSDAY":
            self.monthly_data['thursday_trades'].append(trade)
            self.monthly_data['thursday_pnl'] += pnl
        
        # 전체 집계
        self.monthly_data['total_pnl'] += pnl
        self.monthly_data['trade_count'] += 1
        if pnl > 0:
            self.monthly_data['win_count'] += 1
        
        # 목표 달성 체크
        if self.monthly_data['total_pnl'] >= Config.JAPAN_MONTHLY_TARGET:
            self.monthly_data['target_reached'] = True
        
        self.save_monthly_data()
    
    def get_trading_intensity(self) -> str:
        """거래 강도 결정"""
        current_pnl = self.monthly_data['total_pnl']
        days_passed = datetime.now().day
        progress = days_passed / 30
        pnl_progress = current_pnl / Config.JAPAN_MONTHLY_TARGET if Config.JAPAN_MONTHLY_TARGET > 0 else 0
        
        # 손실 제한
        if current_pnl <= Config.JAPAN_MONTHLY_LIMIT:
            return "STOP_TRADING"
        
        # 목표 달성
        if current_pnl >= Config.JAPAN_MONTHLY_TARGET:
            return "CONSERVATIVE"
        
        # 화목만 하니까 더 공격적
        if progress > 0.75 and pnl_progress < 0.6:
            return "VERY_AGGRESSIVE"
        elif progress > 0.5 and pnl_progress < 0.4:
            return "AGGRESSIVE"
        
        return "NORMAL"
    
    def adjust_position_size(self, base_size: int, confidence: float, day_type: str) -> int:
        """포지션 크기 조절"""
        intensity = self.get_trading_intensity()
        
        if intensity == "STOP_TRADING":
            return 0
        
        # 요일별 조정
        base_multiplier = 1.2 if day_type == "TUESDAY" else 0.8
        
        # 거래 강도별 조정
        if intensity == "VERY_AGGRESSIVE":
            multiplier = base_multiplier * 2.0
        elif intensity == "AGGRESSIVE":
            multiplier = base_multiplier * 1.5
        elif intensity == "CONSERVATIVE":
            multiplier = base_multiplier * 0.6
        else:
            multiplier = base_multiplier
        
        return int(base_size * multiplier * confidence)
    
    def get_status(self) -> Dict:
        """월간 현황"""
        current_pnl = self.monthly_data['total_pnl']
        tuesday_pnl = self.monthly_data['tuesday_pnl']
        thursday_pnl = self.monthly_data['thursday_pnl']
        trade_count = self.monthly_data['trade_count']
        win_count = self.monthly_data['win_count']
        win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
        
        tuesday_trades = len(self.monthly_data['tuesday_trades'])
        thursday_trades = len(self.monthly_data['thursday_trades'])
        
        return {
            'month': self.current_month,
            'total_pnl': current_pnl,
            'tuesday_pnl': tuesday_pnl,
            'thursday_pnl': thursday_pnl,
            'target_progress': (current_pnl / Config.JAPAN_MONTHLY_TARGET * 100) if Config.JAPAN_MONTHLY_TARGET > 0 else 0,
            'trade_count': trade_count,
            'tuesday_trades': tuesday_trades,
            'thursday_trades': thursday_trades,
            'win_rate': win_rate,
            'trading_intensity': self.get_trading_intensity(),
            'target_reached': self.monthly_data['target_reached']
        }

# ============================================================================
# 🎯 화목 신호 생성기 (비용 최적화 AI)
# ============================================================================
@dataclass
class Signal:
    symbol: str
    action: str
    confidence: float
    price: float
    reason: str
    yen_rate: float
    rsi: float
    macd_signal: str
    bb_signal: str
    stoch_signal: str
    atr: float
    volume_signal: str
    stop_loss: float
    take_profit1: float
    take_profit2: float
    take_profit3: float
    max_hold_days: int
    position_size: int
    ai_confidence_check: str = ""
    ai_adjustment: float = 0.0
    ai_used: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass 
class Position:
    symbol: str
    buy_price: float
    shares: int
    buy_date: datetime
    stop_loss: float
    take_profit1: float
    take_profit2: float
    take_profit3: float
    max_hold_date: datetime
    shares_sold_1st: int = 0
    shares_sold_2nd: int = 0
    shares_sold_3rd: int = 0

    def get_remaining_shares(self) -> int:
        return self.shares - self.shares_sold_1st - self.shares_sold_2nd - self.shares_sold_3rd

class SignalGenerator:
    def __init__(self):
        self.current_usd_jpy = 107.5
        self.indicators = Indicators()
        self.target_manager = JapanMonthlyManager()
        self.ai_checker = OptimizedAIChecker()
    
    async def update_yen(self):
        try:
            ticker = yf.Ticker("USDJPY=X")
            data = ticker.history(period="1d")
            if not data.empty:
                self.current_usd_jpy = float(data['Close'].iloc[-1])
        except:
            pass
    
    def get_yen_signal(self) -> str:
        if self.current_usd_jpy <= Config.YEN_STRONG:
            return "STRONG"
        elif self.current_usd_jpy >= Config.YEN_WEAK:
            return "WEAK"
        else:
            return "NEUTRAL"
    
    def classify_stock_type(self, symbol: str) -> str:
        """종목 유형 분류"""
        export_symbols = [
            '7203.T', '6758.T', '7974.T', '6861.T', '9984.T',
            '7267.T', '7269.T', '6326.T', '6473.T', '7013.T'
        ]
        return "EXPORT" if symbol in export_symbols else "DOMESTIC"
    
    def calculate_hybrid_risk_levels(self, price: float, confidence: float, day_type: str, atr: float = 0) -> Tuple[float, float, float, float, int]:
        """화목 하이브리드 리스크 계산"""
        if day_type == "TUESDAY":  # 화요일 메인
            base_stop, base_p1, base_p2, base_p3 = 0.03, 0.04, 0.07, 0.12
            base_days = Config.TUESDAY_MAX_HOLD
        else:  # 목요일 보완
            base_stop, base_p1, base_p2, base_p3 = 0.02, 0.015, 0.03, 0.05
            base_days = Config.THURSDAY_MAX_HOLD
        
        # 신뢰도별 조정
        if confidence >= 0.8:
            multiplier = 1.3
        elif confidence >= 0.6:
            multiplier = 1.0
        else:
            multiplier = 0.8
        
        # ATR 기반 변동성 조정
        if atr > 0:
            atr_ratio = atr / price
            if atr_ratio > 0.03:  # 고변동성
                multiplier *= 1.3
            elif atr_ratio < 0.015:  # 저변동성
                multiplier *= 0.8
        
        stop_loss = price * (1 - base_stop * (2 - multiplier))
        take_profit1 = price * (1 + base_p1 * multiplier)
        take_profit2 = price * (1 + base_p2 * multiplier)
        take_profit3 = price * (1 + base_p3 * multiplier)
        
        return stop_loss, take_profit1, take_profit2, take_profit3, base_days
    
    def calculate_hybrid_score(self, symbol: str, rsi: float, macd_signal: str, macd_details: Dict,
                              bb_signal: str, bb_details: Dict, stoch_signal: str, stoch_details: Dict,
                              atr: float, volume_analysis: Dict, trend: str, day_type: str) -> float:
        """6개 지표 통합 화목 점수"""
        score = 0.0
        
        # 1. 엔화 기반 (35%)
        yen_signal = self.get_yen_signal()
        stock_type = self.classify_stock_type(symbol)
        
        if (yen_signal == "STRONG" and stock_type == "DOMESTIC") or \
           (yen_signal == "WEAK" and stock_type == "EXPORT"):
            score += 0.35
        else:
            score += 0.20
        
        # 2. 요일별 차등 지표
        if day_type == "TUESDAY":  # 화요일 - MACD/추세 중시
            # MACD (20%)
            if macd_signal == "GOLDEN_CROSS":
                score += 0.20
            elif macd_signal == "BULLISH":
                score += 0.15
            elif macd_signal == "DEAD_CROSS":
                score += 0.03
            else:
                score += 0.10
            
            # 추세 (15%)
            if trend == "STRONG_UP":
                score += 0.15
            elif trend == "SIDEWAYS":
                score += 0.08
            else:
                score += 0.03
            
            # RSI (10%)
            if rsi <= 30:
                score += 0.10
            elif rsi <= 45:
                score += 0.08
            elif rsi >= 70:
                score += 0.02
            else:
                score += 0.05
        
        else:  # 목요일 - 단기 지표 중시
            # 스토캐스틱 (20%)
            if stoch_signal == "OVERSOLD":
                score += 0.20
            elif stoch_signal == "BULLISH_CROSS":
                score += 0.15
            elif stoch_signal == "OVERBOUGHT":
                score += 0.02
            else:
                score += 0.08
            
            # 볼린저밴드 (15%)
            if bb_signal == "LOWER_BREAK":
                score += 0.15
            elif bb_signal == "LOWER_ZONE":
                score += 0.12
            elif bb_signal == "UPPER_BREAK":
                score += 0.02
            else:
                score += 0.06
            
            # RSI 극값 중시 (10%)
            if rsi <= 25:
                score += 0.10
            elif rsi <= 35:
                score += 0.08
            elif rsi >= 75:
                score += 0.02
            else:
                score += 0.05
        
        # 3. 공통 지표
        # ATR 변동성 (5%)
        if atr > 0:
            atr_ratio = atr / self.current_usd_jpy if self.current_usd_jpy > 0 else 0
            if 0.01 <= atr_ratio <= 0.03:  # 적당한 변동성
                score += 0.05
            elif atr_ratio > 0.03:  # 고변동성 - 기회
                score += 0.03
            else:  # 저변동성
                score += 0.02
        
        # 거래량 (10%)
        volume_signal = volume_analysis.get('price_volume_signal', 'NEUTRAL')
        if volume_signal == "BULLISH_CONFIRM":
            score += 0.10
        elif volume_analysis.get('volume_spike', False):
            score += 0.08
        elif volume_signal == "WEAK_RALLY":
            score += 0.03
        else:
            score += 0.05
        
        # 볼린저 스퀴즈 보너스 (5%)
        if bb_details.get('squeeze', False):
            score += 0.05  # 변동성 돌파 기대
        
        return min(score, 1.0)
    
    def generate_hybrid_reason(self, symbol: str, rsi: float, macd_signal: str, bb_signal: str,
                              stoch_signal: str, volume_analysis: Dict, day_type: str) -> str:
        """화목 하이브리드 이유 생성"""
        reasons = []
        
        # 엔화
        yen_signal = self.get_yen_signal()
        stock_type = self.classify_stock_type(symbol)
        
        if yen_signal == "STRONG" and stock_type == "DOMESTIC":
            reasons.append("엔강세내수주")
        elif yen_signal == "WEAK" and stock_type == "EXPORT":
            reasons.append("엔약세수출주")
        else:
            reasons.append(f"엔{yen_signal}")
        
        # 요일별 핵심 이유
        day_name = "화메인" if day_type == "TUESDAY" else "목보완"
        reasons.append(day_name)
        
        if day_type == "TUESDAY":  # 화요일
            if macd_signal == "GOLDEN_CROSS":
                reasons.append("MACD골든")
            elif rsi <= 30:
                reasons.append(f"RSI과매도({rsi:.0f})")
        else:  # 목요일
            if stoch_signal == "OVERSOLD":
                reasons.append("스토과매도")
            elif bb_signal == "LOWER_BREAK":
                reasons.append("볼린저돌파")
            elif rsi <= 25:
                reasons.append(f"극과매도({rsi:.0f})")
        
        # 추가 근거
        if volume_analysis.get('volume_spike', False):
            reasons.append("거래량급증")
        
        return " | ".join(reasons[:5])
    
    async def generate_signal(self, symbol: str) -> Signal:
        """화목 하이브리드 신호 생성 (비용 최적화 AI)"""
        try:
            await self.update_yen()
            
            # 화목 체크
            today = datetime.now()
            if today.weekday() == 1:
                day_type = "TUESDAY"
            elif today.weekday() == 3:
                day_type = "THURSDAY"
            else:
                return Signal(symbol, "HOLD", 0.0, 0.0, "비거래일", self.current_usd_jpy, 50.0, 
                            "NEUTRAL", "MIDDLE_ZONE", "NEUTRAL", 0, "NEUTRAL", 
                            0, 0, 0, 0, 0, 0)
            
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            if data.empty:
                raise ValueError("데이터 없음")
            
            current_price = float(data['Close'].iloc[-1])
            
            # 6개 기술지표 계산
            rsi = self.indicators.rsi(data['Close'])
            macd_signal, macd_details = self.indicators.macd(data['Close'])
            bb_signal, bb_details = self.indicators.bollinger_bands(data['Close'])
            stoch_signal, stoch_details = self.indicators.stochastic(data['High'], data['Low'], data['Close'])
            atr_value = self.indicators.atr(data['High'], data['Low'], data['Close'])
            volume_analysis = self.indicators.volume_analysis(data['Close'], data['Volume'])
            trend = self.indicators.trend_signal(data['Close'])
            
            # 하이브리드 점수 계산
            technical_score = self.calculate_hybrid_score(
                symbol, rsi, macd_signal, macd_details, bb_signal, bb_details,
                stoch_signal, stoch_details, atr_value, volume_analysis, trend, day_type
            )
            
            # AI 확신도 체크 (애매한 구간에서만)
            ai_result = await self.ai_checker.check_confidence(
                symbol, 
                {
                    'rsi': rsi,
                    'macd_signal': macd_signal,
                    'bb_signal': bb_signal,
                    'stoch_signal': stoch_signal,
                    'price': current_price
                },
                technical_score
            )
            
            final_confidence = ai_result['final_confidence']
            
            # 월간 목표 고려
            intensity = self.target_manager.get_trading_intensity()
            
            # 요일별 임계값
            threshold = Config.BUY_THRESHOLD_TUESDAY if day_type == "TUESDAY" else Config.BUY_THRESHOLD_THURSDAY
            
            if intensity == "STOP_TRADING":
                action = "HOLD"
                confidence = 0.0
            else:
                # 거래 강도별 임계값 조정
                if intensity == "VERY_AGGRESSIVE":
                    threshold *= 0.75
                elif intensity == "AGGRESSIVE":
                    threshold *= 0.85
                elif intensity == "CONSERVATIVE":
                    threshold *= 1.15
                
                if final_confidence >= threshold:
                    action = "BUY"
                    confidence = min(final_confidence, 0.95)
                else:
                    action = "HOLD"
                    confidence = final_confidence
            
            # 리스크 계산
            if action == "BUY":
                stop_loss, tp1, tp2, tp3, max_days = self.calculate_hybrid_risk_levels(
                    current_price, confidence, day_type, atr_value
                )
                
                base_amount = 1000000
                position_size = self.target_manager.adjust_position_size(
                    int(base_amount / current_price / 100) * 100,
                    confidence,
                    day_type
                )
            else:
                stop_loss = tp1 = tp2 = tp3 = 0.0
                max_days = position_size = 0
            
            # 이유 생성
            reason = self.generate_hybrid_reason(
                symbol, rsi, macd_signal, bb_signal, stoch_signal, volume_analysis, day_type
            )
            
            return Signal(
                symbol=symbol, action=action, confidence=confidence, price=current_price,
                reason=reason, yen_rate=self.current_usd_jpy, rsi=rsi,
                macd_signal=macd_signal, bb_signal=bb_signal, stoch_signal=stoch_signal,
                atr=atr_value, volume_signal=volume_analysis.get('price_volume_signal', 'NEUTRAL'),
                stop_loss=stop_loss, take_profit1=tp1, take_profit2=tp2, take_profit3=tp3,
                max_hold_days=max_days, position_size=position_size,
                ai_confidence_check=ai_result['reason'], ai_adjustment=ai_result['ai_adjustment'],
                ai_used=ai_result['ai_used']
            )
            
        except Exception as e:
            return Signal(symbol, "HOLD", 0.0, 0.0, f"실패:{e}", self.current_usd_jpy, 50.0,
                        "NEUTRAL", "MIDDLE_ZONE", "NEUTRAL", 0, "NEUTRAL",
                        0, 0, 0, 0, 0, 0)

# ============================================================================
# 🛡️ 포지션 매니저 (3차 익절)
# ============================================================================
class PositionManager:
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.closed_positions = []
        self.target_manager = JapanMonthlyManager()
        self.load_positions()
    
    def load_positions(self):
        try:
            positions_file = Config.DATA_DIR / "positions.json"
            if positions_file.exists():
                with open(positions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for symbol, pos_data in data.items():
                        self.positions[symbol] = Position(
                            symbol=pos_data['symbol'],
                            buy_price=pos_data['buy_price'],
                            shares=pos_data['shares'],
                            buy_date=datetime.fromisoformat(pos_data['buy_date']),
                            stop_loss=pos_data['stop_loss'],
                            take_profit1=pos_data['take_profit1'],
                            take_profit2=pos_data['take_profit2'],
                            take_profit3=pos_data.get('take_profit3', 0),
                            max_hold_date=datetime.fromisoformat(pos_data['max_hold_date']),
                            shares_sold_1st=pos_data.get('shares_sold_1st', 0),
                            shares_sold_2nd=pos_data.get('shares_sold_2nd', 0),
                            shares_sold_3rd=pos_data.get('shares_sold_3rd', 0)
                        )
        except Exception as e:
            print(f"⚠️ 포지션 로드 실패: {e}")
    
    def save_positions(self):
        try:
            Config.DATA_DIR.mkdir(exist_ok=True)
            positions_file = Config.DATA_DIR / "positions.json"
            data = {}
            for symbol, position in self.positions.items():
                data[symbol] = {
                    'symbol': position.symbol,
                    'buy_price': position.buy_price,
                    'shares': position.shares,
                    'buy_date': position.buy_date.isoformat(),
                    'stop_loss': position.stop_loss,
                    'take_profit1': position.take_profit1,
                    'take_profit2': position.take_profit2,
                    'take_profit3': position.take_profit3,
                    'max_hold_date': position.max_hold_date.isoformat(),
                    'shares_sold_1st': position.shares_sold_1st,
                    'shares_sold_2nd': position.shares_sold_2nd,
                    'shares_sold_3rd': position.shares_sold_3rd
                }
            
            with open(positions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 포지션 저장 실패: {e}")
    
    def open_position(self, signal: Signal):
        if signal.action == "BUY" and signal.position_size > 0:
            position = Position(
                symbol=signal.symbol,
                buy_price=signal.price,
                shares=signal.position_size,
                buy_date=signal.timestamp,
                stop_loss=signal.stop_loss,
                take_profit1=signal.take_profit1,
                take_profit2=signal.take_profit2,
                take_profit3=signal.take_profit3,
                max_hold_date=signal.timestamp + timedelta(days=signal.max_hold_days)
            )
            self.positions[signal.symbol] = position
            self.save_positions()
            
            day_name = "화요일" if signal.timestamp.weekday() == 1 else "목요일"
            ai_info = f" (AI {signal.ai_adjustment:+.1%})" if signal.ai_used else ""
            print(f"✅ {signal.symbol} {day_name} 포지션 오픈: {signal.position_size:,}주 @ {signal.price:,.0f}엔{ai_info}")
            print(f"   🛡️ 손절: {signal.stop_loss:,.0f}엔")
            print(f"   🎯 익절: {signal.take_profit1:,.0f}→{signal.take_profit2:,.0f}→{signal.take_profit3:,.0f}엔")
    
    async def check_positions(self) -> List[Dict]:
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
                
                # 손절
                if current_price <= position.stop_loss:
                    remaining = position.get_remaining_shares()
                    if remaining > 0:
                        pnl = (current_price - position.buy_price) / position.buy_price
                        actions.append({
                            'action': 'STOP_LOSS',
                            'symbol': symbol,
                            'shares': remaining,
                            'pnl': pnl * 100,
                            'reason': f'손절 ({pnl*100:.1f}%)'
                        })
                        
                        day_type = "TUESDAY" if position.buy_date.weekday() == 1 else "THURSDAY"
                        self.target_manager.add_trade(symbol, pnl, position.buy_price, current_price, day_type)
                        self._close_position(symbol, current_price, 'STOP_LOSS')
                        continue
                
                # 3차 익절
                if current_price >= position.take_profit3 and position.shares_sold_3rd == 0:
                    remaining = position.get_remaining_shares()
                    if remaining > 0:
                        pnl = (current_price - position.buy_price) / position.buy_price
                        actions.append({
                            'action': 'TAKE_PROFIT_3',
                            'symbol': symbol,
                            'shares': remaining,
                            'pnl': pnl * 100,
                            'reason': f'3차 익절 ({pnl*100:.1f}%) - 대박!'
                        })
                        
                        day_type = "TUESDAY" if position.buy_date.weekday() == 1 else "THURSDAY"
                        self.target_manager.add_trade(symbol, pnl, position.buy_price, current_price, day_type)
                        self._close_position(symbol, current_price, 'TAKE_PROFIT_3')
                        continue
                
                # 2차 익절
                elif current_price >= position.take_profit2 and position.shares_sold_2nd == 0:
                    remaining = position.get_remaining_shares()
                    shares_to_sell = int(remaining * 0.67)
                    if shares_to_sell > 0:
                        pnl = (current_price - position.buy_price) / position.buy_price
                        actions.append({
                            'action': 'TAKE_PROFIT_2',
                            'symbol': symbol,
                            'shares': shares_to_sell,
                            'pnl': pnl * 100,
                            'reason': f'2차 익절 ({pnl*100:.1f}%) - 67% 매도'
                        })
                        
                        position.shares_sold_2nd = shares_to_sell
                        self.save_positions()
                
                # 1차 익절
                elif current_price >= position.take_profit1 and position.shares_sold_1st == 0:
                    shares_to_sell = int(position.shares * 0.4)
                    if shares_to_sell > 0:
                        pnl = (current_price - position.buy_price) / position.buy_price
                        actions.append({
                            'action': 'TAKE_PROFIT_1',
                            'symbol': symbol,
                            'shares': shares_to_sell,
                            'pnl': pnl * 100,
                            'reason': f'1차 익절 ({pnl*100:.1f}%) - 40% 매도'
                        })
                        
                        position.shares_sold_1st = shares_to_sell
                        self.save_positions()
                
                # 화목 강제 청산
                elif self._should_force_exit(position, current_time):
                    remaining = position.get_remaining_shares()
                    if remaining > 0:
                        pnl = (current_price - position.buy_price) / position.buy_price
                        reason = self._get_exit_reason(position, current_time)
                        
                        actions.append({
                            'action': 'FORCE_EXIT',
                            'symbol': symbol,
                            'shares': remaining,
                            'pnl': pnl * 100,
                            'reason': reason
                        })
                        
                        day_type = "TUESDAY" if position.buy_date.weekday() == 1 else "THURSDAY"
                        self.target_manager.add_trade(symbol, pnl, position.buy_price, current_price, day_type)
                        self._close_position(symbol, current_price, 'FORCE_EXIT')
                
                # 트레일링 스톱
                else:
                    self._update_trailing_stop(position, current_price)
                
            except Exception as e:
                print(f"⚠️ {symbol} 체크 실패: {e}")
                continue
        
        return actions
    
    def _should_force_exit(self, position: Position, current_time: datetime) -> bool:
        if current_time >= position.max_hold_date:
            return True
        # 화→목, 목→월 청산
        if position.buy_date.weekday() == 1 and current_time.weekday() == 3:  # 화→목
            return (current_time - position.buy_date).days >= 2
        if position.buy_date.weekday() == 3 and current_time.weekday() == 0:  # 목→월
            return True
        return False
    
    def _get_exit_reason(self, position: Position, current_time: datetime) -> str:
        if current_time >= position.max_hold_date:
            return "최대 보유기간 만료"
        elif position.buy_date.weekday() == 1 and current_time.weekday() == 3:
            return "화→목 중간 청산"
        elif position.buy_date.weekday() == 3:
            return "목→월 주말 청산"
        else:
            return "화목 규칙 청산"
    
    def _update_trailing_stop(self, position: Position, current_price: float):
        # 화요일: 5% 수익시 +1%
        if position.buy_date.weekday() == 1:
            if current_price >= position.buy_price * 1.05:
                new_stop = position.buy_price * 1.01
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    self.save_positions()
        # 목요일: 2% 수익시 매수가
        else:
            if current_price >= position.buy_price * 1.02:
                new_stop = position.buy_price * 1.001
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    self.save_positions()
    
    def _close_position(self, symbol: str, exit_price: float, reason: str):
        if symbol in self.positions:
            position = self.positions[symbol]
            pnl = (exit_price - position.buy_price) / position.buy_price * 100
            
            self.closed_positions.append({
                'symbol': symbol, 'pnl': pnl, 'reason': reason,
                'exit_date': datetime.now().isoformat(),
                'buy_day': '화요일' if position.buy_date.weekday() == 1 else '목요일'
            })
            
            del self.positions[symbol]
            self.save_positions()
            print(f"🔚 {symbol} 종료: {pnl:.1f}% ({reason})")

# ============================================================================
# 🔗 IBKR 연동
# ============================================================================
class IBKRConnector:
    def __init__(self):
        self.ib = None
        self.connected = False
        self.available = IBKR_AVAILABLE
    
    async def connect(self) -> bool:
        if not self.available:
            self.connected = True
            return True
        try:
            self.ib = IB()
            await self.ib.connectAsync(Config.IBKR_HOST, Config.IBKR_PORT, Config.IBKR_CLIENT_ID)
            self.connected = True
            print("🔗 IBKR 연결 완료")
            return True
        except Exception as e:
            print(f"❌ IBKR 실패: {e}, 시뮬레이션 모드")
            self.connected = True
            return True
    
    async def place_order(self, symbol: str, action: str, quantity: int) -> Dict:
        if not self.available:
            print(f"🎭 시뮬레이션: {action} {symbol} {quantity}주")
            return {'status': 'success', 'simulation': True}
        
        try:
            contract = Stock(symbol.replace('.T', ''), 'TSE', 'JPY')
            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)
            print(f"📝 IBKR: {action} {symbol} {quantity}주")
            return {'status': 'success', 'orderId': trade.order.orderId}
        except Exception as e:
            print(f"❌ 주문 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def disconnect(self):
        if self.ib and self.available:
            self.ib.disconnect()
        print("🔌 IBKR 연결 해제")

# ============================================================================
# 🏆 YEN-HUNTER v2.1 메인 (비용 최적화)
# ============================================================================
class YenHunter:
    def __init__(self):
        self.hunter = StockHunter()
        self.signal_gen = SignalGenerator()
        self.position_mgr = PositionManager()
        self.ibkr = IBKRConnector()
        
        print("🏆 YEN-HUNTER v2.1 OPTIMIZED 초기화!")
        print("📅 화목 하이브리드 | 🎯 월 14% | 💰 6개 지표 | 🤖 최소 AI | 🔗 IBKR")
        
        # 현황
        status = self.position_mgr.target_manager.get_status()
        print(f"📊 {status['month']} 진행률: {status['target_progress']:.1f}%")
    
    def should_trade_today(self) -> bool:
        return datetime.now().weekday() in Config.TRADING_DAYS
    
    async def hunt_and_analyze(self) -> List[Signal]:
        if not self.should_trade_today():
            print("😴 오늘은 비거래일")
            return []
        
        day_type = "화요일" if datetime.now().weekday() == 1 else "목요일"
        print(f"\n🔍 {day_type} 헌팅 시작...")
        start_time = time.time()
        
        # 3개 지수 종목 수집
        symbols = await self.hunter.hunt_japanese_stocks()
        legends = await self.hunter.select_legends(symbols)
        print(f"🏆 {len(legends)}개 전설급 선별")
        
        # 6개 지표 + 최소 AI 신호 생성
        signals = []
        ai_call_count = 0
        for i, stock in enumerate(legends, 1):
            print(f"⚡ 분석 {i}/{len(legends)} - {stock['symbol']}")
            signal = await self.signal_gen.generate_signal(stock['symbol'])
            signals.append(signal)
            
            if signal.ai_used:
                ai_call_count += 1
            
            await asyncio.sleep(0.05)
        
        elapsed = time.time() - start_time
        buy_count = len([s for s in signals if s.action == 'BUY'])
        
        print(f"🎯 {day_type} 완료! ({elapsed:.1f}초) 매수: {buy_count}개, AI 호출: {ai_call_count}회")
        return signals
    
    async def run_trading_session(self):
        """화목 거래 세션 (비용 최적화)"""
        today = datetime.now()
        if not self.should_trade_today():
            print("😴 오늘은 비거래일")
            return
        
        day_name = "화요일" if today.weekday() == 1 else "목요일"
        print(f"\n🏯 {day_name} 거래 세션 시작 (비용 최적화)")
        
        # 1. 포지션 체크
        actions = await self.position_mgr.check_positions()
        if actions:
            for action in actions:
                emoji = "🛑" if 'STOP' in action['action'] else "💰" if 'PROFIT' in action['action'] else "⏰"
                print(f"{emoji} {action['symbol']}: {action['reason']}")
        
        # 2. 새로운 기회 (최소 AI 활용)
        signals = await self.hunt_and_analyze()
        buy_signals = [s for s in signals if s.action == 'BUY' and s.symbol not in self.position_mgr.positions]
        
        if buy_signals:
            # 신뢰도 정렬
            buy_signals.sort(key=lambda x: x.confidence, reverse=True)
            max_trades = Config.MAX_TUESDAY_TRADES if today.weekday() == 1 else Config.MAX_THURSDAY_TRADES
            
            executed = 0
            for signal in buy_signals[:max_trades]:
                if signal.position_size > 0:
                    ai_info = f" (AI: {signal.ai_adjustment:+.1%})" if signal.ai_used else ""
                    print(f"💰 {signal.symbol} 매수: {signal.confidence:.1%}{ai_info}")
                    
                    # IBKR 주문
                    if self.ibkr.connected:
                        result = await self.ibkr.place_order(signal.symbol, 'BUY', signal.position_size)
                        if result['status'] == 'success':
                            self.position_mgr.open_position(signal)
                            executed += 1
                    else:
                        await self.ibkr.connect()
                        self.position_mgr.open_position(signal)
                        executed += 1
            
            print(f"✅ {day_name} {executed}개 매수 실행")
        else:
            print(f"😴 {day_name} 매수 기회 없음")
        
        # AI 비용 현황
        ai_stats = self.signal_gen.ai_checker.get_usage_stats()
        print(f"💰 AI 사용량: {ai_stats['monthly_calls']}회 (약 {ai_stats['estimated_cost']:.3f}$)")
        
        # 현황
        status = self.get_status()
        print(f"📊 현재: {status['open_positions']}개 포지션 | 월 진행률: {self.position_mgr.target_manager.get_status()['target_progress']:.1f}%")
    
    async def monitor_positions(self):
        """포지션 모니터링"""
        print("👁️ 포지션 모니터링 시작...")
        
        while True:
            try:
                actions = await self.position_mgr.check_positions()
                if actions:
                    for action in actions:
                        emoji = "🛑" if 'STOP' in action['action'] else "💰"
                        print(f"⚡ {emoji} {action['symbol']}: {action['reason']}")
                
                await asyncio.sleep(300)  # 5분마다
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ 모니터링 오류: {e}")
                await asyncio.sleep(60)
    
    def get_status(self) -> Dict:
        """현황 반환"""
        total_positions = len(self.position_mgr.positions)
        closed_trades = len(self.position_mgr.closed_positions)
        
        if self.position_mgr.closed_positions:
            avg_pnl = sum([t['pnl'] for t in self.position_mgr.closed_positions]) / closed_trades
            win_rate = len([t for t in self.position_mgr.closed_positions if t['pnl'] > 0]) / closed_trades * 100
        else:
            avg_pnl = win_rate = 0
        
        monthly = self.position_mgr.target_manager.get_status()
        ai_stats = self.signal_gen.ai_checker.get_usage_stats()
        
        return {
            'open_positions': total_positions,
            'closed_trades': closed_trades,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'positions': list(self.position_mgr.positions.keys()),
            'monthly_progress': monthly['target_progress'],
            'monthly_pnl': monthly['total_pnl'] * 100,
            'tuesday_pnl': monthly['tuesday_pnl'] * 100,
            'thursday_pnl': monthly['thursday_pnl'] * 100,
            'trading_intensity': monthly['trading_intensity'],
            'ai_calls': ai_stats['monthly_calls'],
            'ai_cost': ai_stats['estimated_cost']
        }

# ============================================================================
# 🧪 비용 최적화 편의 함수들
# ============================================================================
async def hunt_signals() -> List[Signal]:
    """신호 헌팅 (비용 최적화)"""
    hunter = YenHunter()
    return await hunter.hunt_and_analyze()

async def analyze_single(symbol: str) -> Signal:
    """단일 분석 (비용 최적화)"""
    hunter = YenHunter()
    return await hunter.signal_gen.generate_signal(symbol)

async def analyze_with_ai_check(symbol: str) -> Dict:
    """AI 확신도 체크 포함 분석"""
    hunter = YenHunter()
    
    print(f"🤖 {symbol} AI 확신도 체크 분석...")
    
    signal = await hunter.signal_gen.generate_signal(symbol)
    
    return {
        'signal': signal,
        'technical_confidence': signal.confidence - signal.ai_adjustment,
        'ai_adjustment': signal.ai_adjustment,
        'final_confidence': signal.confidence,
        'ai_used': signal.ai_used,
        'ai_reason': signal.ai_confidence_check,
        'recommendation': signal.action
    }

async def run_auto_selection() -> List[Dict]:
    """자동선별 실행"""
    hunter = YenHunter()
    
    print("🤖 자동선별 시스템 시작!")
    print("="*50)
    
    # 3개 지수 종목 수집
    symbols = await hunter.hunter.hunt_japanese_stocks()
    print(f"📡 총 수집: {len(symbols)}개 종목")
    
    # 자동선별 실행
    legends = await hunter.hunter.select_legends(symbols)
    
    print(f"\n🏆 자동선별 결과: {len(legends)}개 전설급")
    print("="*50)
    
    for i, stock in enumerate(legends, 1):
        print(f"{i:2d}. {stock['symbol']} | 점수: {stock['score']:.2f}")
        print(f"    💰 시총: {stock['market_cap']/1e12:.1f}조엔 | 섹터: {stock['sector']}")
        print(f"    📊 현재가: {stock['current_price']:,.0f}엔 | 거래량: {stock['avg_volume']/1e6:.1f}M")
        print(f"    💡 이유: {stock['selection_reason']}")
        print()
    
    return legends

async def analyze_auto_selected() -> List[Signal]:
    """자동선별 종목들 분석 (비용 최적화)"""
    hunter = YenHunter()
    
    # 자동선별 실행
    legends = await run_auto_selection()
    
    if not hunter.should_trade_today():
        print("😴 오늘은 비거래일이지만 분석은 진행합니다.")
    
    print("\n🔍 자동선별 종목 신호 분석 (비용 최적화)")
    print("="*50)
    
    signals = []
    ai_call_count = 0
    
    for i, stock in enumerate(legends, 1):
        print(f"⚡ 분석 {i}/{len(legends)} - {stock['symbol']}")
        signal = await hunter.signal_gen.generate_signal(stock['symbol'])
        signals.append(signal)
        
        if signal.ai_used:
            ai_call_count += 1
        
        # 간단한 결과 출력
        if signal.action == 'BUY':
            ai_info = f" (AI: {signal.ai_adjustment:+.1%})" if signal.ai_used else ""
            print(f"   ✅ 매수신호! {signal.confidence:.1%}{ai_info}")
            print(f"   📊 {signal.reason}")
        else:
            print(f"   ⏸️ 대기 ({signal.confidence:.1%})")
    
    buy_signals = [s for s in signals if s.action == 'BUY']
    print(f"\n🎯 매수 추천: {len(buy_signals)}개 / {len(signals)}개")
    print(f"💰 AI 호출: {ai_call_count}회 (비용 최적화)")
    
    return signals

async def run_auto_trading():
    """자동매매 실행 (비용 최적화)"""
    hunter = YenHunter()
    
    try:
        await hunter.ibkr.connect()
        print("🚀 화목 비용최적화 자동매매 시작 (Ctrl+C로 종료)")
        
        while True:
            now = datetime.now()
            
            # 화목 09시에 거래
            if now.weekday() in [1, 3] and now.hour == 9 and now.minute == 0:
                await hunter.run_trading_session()
                await asyncio.sleep(60)
            else:
                # 포지션 모니터링
                actions = await hunter.position_mgr.check_positions()
                if actions:
                    for action in actions:
                        print(f"⚡ {action['symbol']}: {action['reason']}")
                await asyncio.sleep(300)
                
    except KeyboardInterrupt:
        print("🛑 비용최적화 자동매매 종료")
    finally:
        await hunter.ibkr.disconnect()

def show_status():
    """현황 출력"""
    hunter = YenHunter()
    status = hunter.get_status()
    monthly = hunter.position_mgr.target_manager.get_status()
    
    print(f"\n📊 YEN-HUNTER v2.1 OPTIMIZED 현황")
    print("="*60)
    print(f"💼 오픈 포지션: {status['open_positions']}개")
    print(f"🎲 완료 거래: {status['closed_trades']}회")
    print(f"📈 평균 수익: {status['avg_pnl']:.1f}%")
    print(f"🏆 승률: {status['win_rate']:.1f}%")
    print(f"\n📅 {monthly['month']} 월간 현황:")
    print(f"🎯 목표 진행: {monthly['target_progress']:.1f}% / 14%")
    print(f"💰 총 수익: {monthly['total_pnl']*100:.2f}%")
    print(f"📊 화요일: {monthly['tuesday_pnl']*100:.2f}% ({monthly['tuesday_trades']}회)")
    print(f"📊 목요일: {monthly['thursday_pnl']*100:.2f}% ({monthly['thursday_trades']}회)")
    print(f"⚡ 거래 모드: {monthly['trading_intensity']}")
    
    # AI 비용 현황
    ai_status = "활성화" if OPENAI_AVAILABLE and Config.OPENAI_API_KEY else "시뮬레이션"
    print(f"🤖 AI 상태: {ai_status}")
    print(f"💰 AI 호출: {status['ai_calls']}회 (약 {status['ai_cost']:.3f}$)")
    
    if status['positions']:
        print(f"📋 보유: {', '.join(status['positions'])}")

# ============================================================================
# 📈 백테스터 (비용 최적화)
# ============================================================================
class OptimizedBacktester:
    @staticmethod
    async def backtest_symbol(symbol: str, period: str = "6mo") -> Dict:
        """화목 하이브리드 백테스트 (AI 없이)"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if len(data) < 100:
                return {"error": "데이터 부족"}
            
            indicators = Indicators()
            tuesday_trades = []
            thursday_trades = []
            
            for i in range(60, len(data)):
                current_data = data.iloc[:i+1]
                current_date = current_data.index[-1]
                weekday = current_date.weekday()
                
                # 화목만 거래
                if weekday not in [1, 3]:
                    continue
                
                # 기술지표
                rsi = indicators.rsi(current_data['Close'])
                macd_signal, _ = indicators.macd(current_data['Close'])
                bb_signal, _ = indicators.bollinger_bands(current_data['Close'])
                stoch_signal, _ = indicators.stochastic(current_data['High'], current_data['Low'], current_data['Close'])
                
                price = current_data['Close'].iloc[-1]
                
                # 화목별 매수 조건 (기술적 분석만)
                should_buy = False
                if weekday == 1:  # 화요일
                    if rsi <= 35 and macd_signal == "GOLDEN_CROSS":
                        should_buy = True
                elif weekday == 3:  # 목요일
                    if (rsi <= 25 or bb_signal == "LOWER_BREAK" or stoch_signal == "OVERSOLD"):
                        should_buy = True
                
                if should_buy:
                    # 매도 조건
                    if weekday == 1:  # 화요일
                        hold_target, profit_target, stop_loss = 5, 0.07, 0.03
                    else:  # 목요일
                        hold_target, profit_target, stop_loss = 2, 0.03, 0.02
                    
                    # 결과 계산
                    future_data = data.iloc[i:i+hold_target+1]
                    if len(future_data) > 1:
                        for j, (future_date, future_row) in enumerate(future_data.iterrows()):
                            if j == 0:
                                continue
                                
                            future_price = future_row['Close']
                            pnl = (future_price - price) / price
                            
                            if pnl >= profit_target or pnl <= -stop_loss or j == len(future_data) - 1:
                                trade_info = {
                                    'return': pnl * 100,
                                    'day_type': '화요일' if weekday == 1 else '목요일'
                                }
                                
                                if weekday == 1:
                                    tuesday_trades.append(trade_info)
                                else:
                                    thursday_trades.append(trade_info)
                                break
            
            all_trades = tuesday_trades + thursday_trades
            if all_trades:
                returns = [t['return']/100 for t in all_trades]
                total_return = np.prod([1 + r for r in returns]) - 1
                
                return {
                    "symbol": symbol,
                    "total_return": total_return * 100,
                    "total_trades": len(all_trades),
                    "win_rate": len([r for r in returns if r > 0]) / len(returns) * 100,
                    "tuesday_trades": len(tuesday_trades),
                    "thursday_trades": len(thursday_trades),
                    "tuesday_avg": np.mean([t['return'] for t in tuesday_trades]) if tuesday_trades else 0,
                    "thursday_avg": np.mean([t['return'] for t in thursday_trades]) if thursday_trades else 0,
                }
            else:
                return {"error": "거래 없음"}
                
        except Exception as e:
            return {"error": str(e)}

async def backtest_optimized(symbol: str) -> Dict:
    """백테스트"""
    return await OptimizedBacktester.backtest_symbol(symbol)

# ============================================================================
# 🧪 최적화 테스트 실행
# ============================================================================
async def main():
    """YEN-HUNTER v2.1 OPTIMIZED 테스트"""
    print("🏆 YEN-HUNTER v2.1 OPTIMIZED 테스트!")
    print("="*70)
    print("📅 화목 하이브리드 전략")
    print("🎯 월 14% 목표 (화 2.5% + 목 1.5%)")
    print("💰 6개 핵심 지표 + 3개 지수 헌팅")
    print("🤖 최소 AI (확신도 0.4-0.7에서만 호출)")
    print("💵 월 비용 5천원 이하 최적화")
    print("🔗 IBKR 연동 + 완전 자동화")
    
    # AI 상태 확인
    ai_status = "활성화" if OPENAI_AVAILABLE and Config.OPENAI_API_KEY else "시뮬레이션"
    print(f"🤖 AI 상태: {ai_status}")
    if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
        print(f"💰 AI 설정: GPT-3.5-turbo, 확신도 {Config.AI_CONFIDENCE_MIN:.1f}-{Config.AI_CONFIDENCE_MAX:.1f}에서만 호출")
    
    # 현황 출력
    show_status()
    
    # 거래일 체크
    hunter = YenHunter()
    if not hunter.should_trade_today():
        print(f"\n😴 오늘은 비거래일 (월,수,금,토,일)")
        
        # 비거래일에도 간단한 분석 제공 (AI 사용 없이)
        print("\n📊 간단 시장 체크...")
        await hunter.signal_gen.update_yen()
        yen_signal = hunter.signal_gen.get_yen_signal()
        print(f"💴 엔/달러: {hunter.signal_gen.current_usd_jpy:.2f} ({yen_signal})")
        return
    
    # 최적화된 신호 헌팅
    signals = await hunt_signals()
    
    if signals:
        buy_signals = [s for s in signals if s.action == 'BUY']
        buy_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"\n🎯 비용최적화 매수 추천 TOP 3:")
        ai_used_count = sum(1 for s in buy_signals if s.ai_used)
        print(f"💰 AI 활용: {ai_used_count}/{len(buy_signals)}개 종목")
        
        for i, signal in enumerate(buy_signals[:3], 1):
            profit1_pct = ((signal.take_profit1 - signal.price) / signal.price * 100)
            profit2_pct = ((signal.take_profit2 - signal.price) / signal.price * 100)
            profit3_pct = ((signal.take_profit3 - signal.price) / signal.price * 100)
            stop_pct = ((signal.price - signal.stop_loss) / signal.price * 100)
            
            ai_info = ""
            if signal.ai_used:
                ai_info = f" (AI: {signal.ai_adjustment:+.1%})"
            
            print(f"\n{i}. {signal.symbol} (신뢰도: {signal.confidence:.1%}{ai_info})")
            print(f"   💰 {signal.price:,.0f}엔 | {signal.position_size:,}주")
            print(f"   🛡️ 손절: -{stop_pct:.1f}%")
            print(f"   🎯 익절: +{profit1_pct:.1f}% → +{profit2_pct:.1f}% → +{profit3_pct:.1f}%")
            print(f"   📊 지표: RSI({signal.rsi:.0f}) {signal.macd_signal} {signal.bb_signal} {signal.stoch_signal}")
            print(f"   💡 {signal.reason}")
            if signal.ai_used:
                print(f"   🤖 AI 체크: {signal.ai_confidence_check}")
        
        # 백테스트
        if buy_signals:
            print(f"\n📈 백테스트 ({buy_signals[0].symbol}):")
            backtest_result = await backtest_optimized(buy_signals[0].symbol)
            if "error" not in backtest_result:
                print(f"   📊 총 수익: {backtest_result['total_return']:.1f}%")
                print(f"   🏆 승률: {backtest_result['win_rate']:.1f}%")
                print(f"   📅 화요일: {backtest_result['tuesday_trades']}회 (평균 {backtest_result['tuesday_avg']:.1f}%)")
                print(f"   📅 목요일: {backtest_result['thursday_trades']}회 (평균 {backtest_result['thursday_avg']:.1f}%)")
    
    print("\n✅ YEN-HUNTER v2.1 OPTIMIZED 테스트 완료!")
    print("\n🚀 핵심 특징 (비용 최적화):")
    print("  📊 기술지표: 6개 핵심 (RSI, MACD, 볼린저, 스토캐스틱, ATR, 거래량)")
    print("  🔍 종목헌팅: 3개 지수 통합 (닛케이225 + TOPIX + JPX400)")
    print("  🤖 최소 AI: 확신도 0.4-0.7에서만 호출 (월 5천원 이하)")
    print("  📈 월간관리: 핵심 목표 추적 + 적응형 강도 조절")
    print("  💰 3차 익절: 40% → 67% → 나머지 분할")
    print("  🛡️ 동적 손절: ATR + 신뢰도 기반")
    print("  🔗 IBKR 연동: 실제 거래 + 시뮬레이션")
    
    print(f"\n💡 사용법 (비용 최적화):")
    print("  🤖 자동선별: await run_auto_selection()")
    print("  🔍 선별+분석: await analyze_auto_selected()")
    print("  🎯 AI확신도체크: await analyze_with_ai_check('7203.T')")
    print("  🚀 자동매매: await run_auto_trading()")
    print("  📊 현황: show_status()")
    print("  🔍 단일분석: await analyze_single('7203.T')")
    print("  📈 백테스트: await backtest_optimized('7203.T')")
    
    print(f"\n🔧 AI 최적화 설정:")
    print(f"  📁 데이터: {Config.DATA_DIR}")
    print(f"  🤖 AI 모델: {Config.OPENAI_MODEL}")
    print(f"  💰 AI 호출 구간: {Config.AI_CONFIDENCE_MIN:.1f} - {Config.AI_CONFIDENCE_MAX:.1f}")
    print(f"  🔑 API 키: {'설정됨' if Config.OPENAI_API_KEY else '미설정 (환경변수 OPENAI_API_KEY)'}")
    
    print(f"\n🎯 주요 개선사항:")
    print("  ✅ trend_analysis 오류 완전 수정")
    print("  ✅ AI 호출을 애매한 상황(0.4-0.7)에서만 제한")
    print("  ✅ 뉴스분석/시장심리분석 완전 제거")
    print("  ✅ GPT-3.5-turbo로 비용 절약")
    print("  ✅ 월 AI 비용 5천원 이하로 최적화")
    print("  ✅ 순수 기술적 분석 중심 전략")
    
    print("\n🎯 화목 하이브리드 + 최소 AI로 월 14% 달성!")

if __name__ == "__main__":
    Config.DATA_DIR.mkdir(exist_ok=True)
    asyncio.run(main())
