"""
🇯🇵 일본 주식 전략 모듈 - 최고퀸트프로젝트 (순수 기술분석 + 파라미터 최적화)
===========================================================================

일본 주식 시장 특화 전략:
- 일목균형표 (Ichimoku Kinko Hyo) 분석
- 모멘텀 돌파 (Momentum Breakout) 전략
- 일본 주요 기업 추적 (닛케이225 중심)
- 기술적 분석 통합
- 거래량 기반 신호 생성
- 순수 기술분석 (뉴스 제거)
- 파라미터 최적화 (신호 활성화)

Author: 최고퀸트팀
Version: 1.2.0 (파라미터 최적화)
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
import ta  # ta 라이브러리 사용 (talib 대신)

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class JPStockSignal:
    """일본 주식 시그널 데이터 클래스"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    strategy_source: str  # 'ichimoku', 'momentum_breakout', 'technical_analysis'
    ichimoku_signal: str  # 'bullish', 'bearish', 'neutral'
    momentum_score: float
    volume_ratio: float
    rsi: float
    sector: str
    reasoning: str
    target_price: float
    timestamp: datetime
    additional_data: Optional[Dict] = None

class JPStrategy:
    """🇯🇵 고급 일본 주식 전략 클래스 (순수 기술분석 + 파라미터 최적화)"""
    
    def __init__(self, config_path: str = "configs/settings.yaml"):
        """전략 초기화"""
        self.config = self._load_config(config_path)
        self.jp_config = self.config.get('jp_strategy', {})
        
        # settings.yaml에서 설정값 읽기
        self.enabled = self.jp_config.get('enabled', True)
        self.use_ichimoku = self.jp_config.get('ichimoku', True)
        self.use_momentum_breakout = self.jp_config.get('momentum_breakout', True)
        
        # 일목균형표 파라미터 (최적화: 더 민감하게)
        self.tenkan_period = self.jp_config.get('tenkan_period', 7)    # 9 → 7 (더 민감)
        self.kijun_period = self.jp_config.get('kijun_period', 20)     # 26 → 20 (더 빠른 반응)
        self.senkou_period = self.jp_config.get('senkou_period', 44)   # 52 → 44
        
        # 모멘텀 돌파 파라미터 (최적화: 신호 활성화)
        self.breakout_period = self.jp_config.get('breakout_period', 15)  # 20 → 15 (더 쉬운 돌파)
        self.volume_threshold = self.jp_config.get('volume_threshold', 1.2)  # 1.5 → 1.2 (완화)
        self.rsi_period = self.jp_config.get('rsi_period', 10)         # 14 → 10 (더 민감한 RSI)
        
        # 신뢰도 임계값 (최적화: 대폭 완화)
        self.confidence_threshold = self.jp_config.get('confidence_threshold', 0.60)  # 80% → 60%
        
        # 배당 보너스 설정 (일본 시장 특화)
        self.dividend_bonus_threshold = self.jp_config.get('dividend_bonus_threshold', 4.0)  # 4% 이상
        self.dividend_bonus_score = self.jp_config.get('dividend_bonus_score', 0.1)  # 10% 보너스
        
        # 추적할 일본 주식 (settings.yaml에서 로드)
        self.symbols = self.jp_config.get('symbols', {
            'TECH': ['7203.T', '6758.T', '9984.T', '6861.T', '4689.T'],
            'FINANCE': ['8306.T', '8316.T', '8411.T', '8355.T'],
            'CONSUMER': ['9983.T', '2914.T', '4568.T', '7974.T'],
            'INDUSTRIAL': ['6954.T', '6902.T', '7733.T', '6098.T']
        })
        
        # 모든 심볼을 플랫 리스트로 (.T는 도쿄증권거래소 접미사)
        self.all_symbols = [symbol for sector_symbols in self.symbols.values() 
                           for symbol in sector_symbols]
        
        if self.enabled:
            logger.info(f"🇯🇵 일본 주식 전략 초기화 완료 (최적화) - 추적 종목: {len(self.all_symbols)}개")
            logger.info(f"📊 일목균형표: {self.use_ichimoku} ({self.tenkan_period}/{self.kijun_period}), 모멘텀돌파: {self.use_momentum_breakout}")
            logger.info(f"🎯 신뢰도 임계값: {self.confidence_threshold:.0%} (완화), RSI: {self.rsi_period}일")
            logger.info(f"🔧 순수 기술분석 모드 (뉴스 분석 제거)")
        else:
            logger.info("🇯🇵 일본 주식 전략이 비활성화되어 있습니다")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return {}

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """심볼에 해당하는 섹터 찾기"""
        for sector, symbols in self.symbols.items():
            if symbol in symbols:
                return sector
        return 'UNKNOWN'

    async def _get_stock_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """주식 데이터 수집"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                logger.error(f"데이터 없음: {symbol}")
                return pd.DataFrame()
                
            return data
            
        except Exception as e:
            logger.error(f"주식 데이터 수집 실패 {symbol}: {e}")
            return pd.DataFrame()

    async def _get_dividend_yield(self, symbol: str) -> float:
        """배당 수익률 조회 (일본 시장 특화)"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield:
                return dividend_yield * 100  # 퍼센트로 변환
            return 0.0
        except Exception as e:
            logger.warning(f"배당 정보 조회 실패 {symbol}: {e}")
            return 0.0

    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict:
        """일목균형표 계산 (최적화된 파라미터)"""
        try:
            if len(data) < self.senkou_period:
                return {}
                
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # 전환선 (Tenkan-sen): 7일 중간값 (9일에서 단축)
            tenkan_high = high.rolling(window=self.tenkan_period).max()
            tenkan_low = low.rolling(window=self.tenkan_period).min()
            tenkan = (tenkan_high + tenkan_low) / 2
            
            # 기준선 (Kijun-sen): 20일 중간값 (26일에서 단축)
            kijun_high = high.rolling(window=self.kijun_period).max()
            kijun_low = low.rolling(window=self.kijun_period).min()
            kijun = (kijun_high + kijun_low) / 2
            
            # 선행스팬 A (Senkou Span A): (전환선 + 기준선) / 2
            senkou_a = ((tenkan + kijun) / 2).shift(self.kijun_period)
            
            # 선행스팬 B (Senkou Span B): 44일 중간값 (52일에서 단축)
            senkou_b_high = high.rolling(window=self.senkou_period).max()
            senkou_b_low = low.rolling(window=self.senkou_period).min()
            senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(self.kijun_period)
            
            # 후행스팬 (Chikou Span): 현재 종가를 20일 뒤로 (26일에서 단축)
            chikou = close.shift(-self.kijun_period)
            
            # 최신 값들
            latest_idx = -1
            current_price = close.iloc[latest_idx]
            current_tenkan = tenkan.iloc[latest_idx]
            current_kijun = kijun.iloc[latest_idx]
            current_senkou_a = senkou_a.iloc[latest_idx] if not pd.isna(senkou_a.iloc[latest_idx]) else 0
            current_senkou_b = senkou_b.iloc[latest_idx] if not pd.isna(senkou_b.iloc[latest_idx]) else 0
            
            return {
                'tenkan': current_tenkan,
                'kijun': current_kijun,
                'senkou_a': current_senkou_a,
                'senkou_b': current_senkou_b,
                'current_price': current_price,
                'tenkan_series': tenkan,
                'kijun_series': kijun
            }
            
        except Exception as e:
            logger.error(f"일목균형표 계산 실패: {e}")
            return {}

    def _analyze_ichimoku_signal(self, ichimoku_data: Dict) -> Tuple[str, float, str]:
        """일목균형표 신호 분석 (완화된 기준)"""
        if not ichimoku_data:
            return 'neutral', 0.0, "일목균형표 데이터 없음"
            
        try:
            price = ichimoku_data['current_price']
            tenkan = ichimoku_data['tenkan']
            kijun = ichimoku_data['kijun']
            senkou_a = ichimoku_data['senkou_a']
            senkou_b = ichimoku_data['senkou_b']
            
            signal_score = 0.0
            reasons = []
            
            # 1. 전환선과 기준선 관계 (가중치 증가)
            if tenkan > kijun:
                signal_score += 0.35  # 0.3 → 0.35
                reasons.append("전환선>기준선")
            elif tenkan < kijun:
                signal_score -= 0.35
                reasons.append("전환선<기준선")
                
            # 2. 가격과 구름(일목균형표) 관계 (가중치 증가)
            cloud_top = max(senkou_a, senkou_b) if senkou_a > 0 and senkou_b > 0 else 0
            cloud_bottom = min(senkou_a, senkou_b) if senkou_a > 0 and senkou_b > 0 else 0
            
            if cloud_top > 0:
                if price > cloud_top:
                    signal_score += 0.45  # 0.4 → 0.45
                    reasons.append("구름위")
                elif price < cloud_bottom:
                    signal_score -= 0.35  # -0.4 → -0.35 (매도 기준 완화)
                    reasons.append("구름아래")
                else:
                    reasons.append("구름속")
                    
            # 3. 구름의 색깔 (두께)
            if senkou_a > senkou_b:
                signal_score += 0.15  # 0.2 → 0.15
                reasons.append("상승구름")
            elif senkou_a < senkou_b:
                signal_score -= 0.15
                reasons.append("하락구름")
                
            # 4. 가격과 기준선 관계
            if price > kijun:
                signal_score += 0.05  # 0.1 → 0.05
                reasons.append("기준선위")
            else:
                signal_score -= 0.05
                reasons.append("기준선아래")
                
            # 신호 결정 (기준 완화)
            if signal_score >= 0.4:  # 0.6 → 0.4 (완화)
                signal = 'bullish'
            elif signal_score <= -0.3:  # -0.6 → -0.3 (완화)
                signal = 'bearish'
            else:
                signal = 'neutral'
                
            confidence = min(abs(signal_score), 1.0)
            reasoning = "일목: " + " | ".join(reasons)
            
            return signal, confidence, reasoning
            
        except Exception as e:
            logger.error(f"일목균형표 신호 분석 실패: {e}")
            return 'neutral', 0.0, f"일목 분석 실패: {str(e)}"

    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict:
        """모멘텀 지표 계산 (최적화된 파라미터)"""
        try:
            if len(data) < self.breakout_period:
                return {}
                
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # RSI 계산 (10일로 단축 - 더 민감)
            rsi = ta.momentum.RSIIndicator(close, window=self.rsi_period).rsi()
            
            # 볼린저 밴드 (15일로 단축)
            bb = ta.volatility.BollingerBands(close, window=15)
            bb_upper = bb.bollinger_hband()
            bb_middle = bb.bollinger_mavg() 
            bb_lower = bb.bollinger_lband()
            
            # MACD (더 민감한 설정)
            macd = ta.trend.MACD(close, window_fast=8, window_slow=21, window_sign=5)
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            
            # 거래량 비율 (최근 3일 평균 대비) - 더 짧은 기간
            recent_volume = volume.tail(3).mean()
            avg_volume = volume.tail(15).head(12).mean()  # 15일 중 최근 3일 제외
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 가격 돌파 체크 (15일 최고가 돌파) - 더 쉬운 돌파
            breakout_high = high.tail(self.breakout_period).head(self.breakout_period-1).max()
            current_price = close.iloc[-1]
            price_breakout = current_price > breakout_high
            
            return {
                'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
                'bb_upper': bb_upper.iloc[-1],
                'bb_middle': bb_middle.iloc[-1], 
                'bb_lower': bb_lower.iloc[-1],
                'macd': macd_line.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'volume_ratio': volume_ratio,
                'price_breakout': price_breakout,
                'breakout_high': breakout_high,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"모멘텀 지표 계산 실패: {e}")
            return {}

    def _analyze_momentum_breakout(self, momentum_data: Dict) -> Tuple[str, float, str]:
        """모멘텀 돌파 신호 분석 (완화된 기준)"""
        if not momentum_data:
            return 'neutral', 0.0, "모멘텀 데이터 없음"
            
        try:
            rsi = momentum_data.get('rsi', 50)
            volume_ratio = momentum_data.get('volume_ratio', 1.0)
            price_breakout = momentum_data.get('price_breakout', False)
            macd = momentum_data.get('macd', 0)
            macd_signal = momentum_data.get('macd_signal', 0)
            current_price = momentum_data.get('current_price', 0)
            bb_upper = momentum_data.get('bb_upper', 0)
            bb_lower = momentum_data.get('bb_lower', 0)
            
            signal_score = 0.0
            reasons = []
            
            # 1. 가격 돌파 체크 (가중치 증가)
            if price_breakout:
                signal_score += 0.45  # 0.4 → 0.45
                reasons.append("가격돌파")
                
            # 2. 거래량 증가 체크 (기준 완화)
            if volume_ratio >= self.volume_threshold:  # 1.2배 이상
                signal_score += 0.35  # 0.3 → 0.35
                reasons.append(f"거래량증가({volume_ratio:.1f}배)")
            elif volume_ratio < 0.7:  # 0.8 → 0.7 (더 관대)
                signal_score -= 0.15  # -0.2 → -0.15
                reasons.append("거래량감소")
                
            # 3. RSI 체크 (범위 확대)
            if 25 <= rsi <= 75:  # 30-70 → 25-75 (범위 확대)
                signal_score += 0.25  # 0.2 → 0.25
                reasons.append(f"RSI정상({rsi:.0f})")
            elif rsi > 85:  # 80 → 85 (더 관대)
                signal_score -= 0.25  # -0.3 → -0.25
                reasons.append(f"RSI과매수({rsi:.0f})")
            elif rsi < 15:  # 20 → 15 (더 관대)
                signal_score += 0.15  # 0.1 → 0.15
                reasons.append(f"RSI과매도({rsi:.0f})")
                
            # 4. MACD 신호 (가중치 증가)
            if macd > macd_signal:
                signal_score += 0.15  # 0.1 → 0.15
                reasons.append("MACD상승")
            else:
                signal_score -= 0.1
                reasons.append("MACD하락")
                
            # 5. 볼린저 밴드 위치
            if current_price > bb_upper:
                signal_score += 0.15  # 0.1 → 0.15
                reasons.append("밴드상단돌파")
            elif current_price < bb_lower:
                signal_score -= 0.15  # -0.2 → -0.15
                reasons.append("밴드하단이탈")
                
            # 신호 결정 (기준 대폭 완화)
            if signal_score >= 0.5:  # 0.7 → 0.5 (대폭 완화)
                signal = 'bullish'
            elif signal_score <= -0.3:  # -0.5 → -0.3 (완화)
                signal = 'bearish'
            else:
                signal = 'neutral'
                
            confidence = min(abs(signal_score), 1.0)
            reasoning = "모멘텀: " + " | ".join(reasons)
            
            return signal, confidence, reasoning
            
        except Exception as e:
            logger.error(f"모멘텀 돌파 분석 실패: {e}")
            return 'neutral', 0.0, f"모멘텀 분석 실패: {str(e)}"

    def _calculate_position_size(self, price: float, confidence: float, account_balance: float = 10000000) -> int:
        """포지션 크기 계산 (일본 주식용 - 엔화 기준)"""
        try:
            # 신뢰도에 따른 포지션 사이징
            base_position_pct = 0.025  # 기본 2.5% (0.02에서 증가)
            confidence_multiplier = confidence  # 신뢰도가 높을수록 큰 포지션
            
            position_pct = base_position_pct * confidence_multiplier
            position_pct = min(position_pct, 0.1)  # 일본 주식은 최대 10%로 제한 (8%에서 증가)
            
            position_value = account_balance * position_pct
            shares = int(position_value / price) if price > 0 else 0
            
            # 일본 주식은 100주 단위로 거래 (단원주)
            shares = (shares // 100) * 100
            
            return shares
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 실패: {e}")
            return 0

    def _calculate_target_price(self, current_price: float, confidence: float, signal: str) -> float:
        """목표주가 계산"""
        if current_price == 0:
            return 0
            
        # 신호에 따른 기대수익률 (일본 시장 특성 반영)
        if signal == 'buy':
            expected_return = confidence * 0.12  # 최대 12% 수익 기대 (15%에서 조정)
            return current_price * (1 + expected_return)
        elif signal == 'sell':
            expected_return = confidence * 0.08  # 8% 하락 예상 (10%에서 조정)
            return current_price * (1 - expected_return)
        else:
            return current_price

    async def analyze_symbol(self, symbol: str) -> JPStockSignal:
        """개별 일본 주식 분석 (최적화된 파라미터)"""
        if not self.enabled:
            logger.warning("일본 주식 전략이 비활성화되어 있습니다")
            return JPStockSignal(
                symbol=symbol, action='hold', confidence=0.0, price=0.0,
                strategy_source='disabled', ichimoku_signal='neutral', 
                momentum_score=0.0, volume_ratio=0.0, rsi=50.0,
                sector='UNKNOWN', reasoning="전략 비활성화", 
                target_price=0.0, timestamp=datetime.now()
            )
            
        try:
            # 주식 데이터 수집
            data = await self._get_stock_data(symbol)
            if data.empty:
                raise ValueError(f"주식 데이터 없음: {symbol}")

            current_price = data['Close'].iloc[-1]
            
            # 배당 수익률 조회 (일본 시장 특화)
            dividend_yield = await self._get_dividend_yield(symbol)
            
            # 1. 일목균형표 분석
            ichimoku_signal = 'neutral'
            ichimoku_confidence = 0.0
            ichimoku_reasoning = ""
            
            if self.use_ichimoku:
                ichimoku_data = self._calculate_ichimoku(data)
                ichimoku_signal, ichimoku_confidence, ichimoku_reasoning = self._analyze_ichimoku_signal(ichimoku_data)
                
            # 2. 모멘텀 돌파 분석
            momentum_signal = 'neutral'
            momentum_confidence = 0.0
            momentum_reasoning = ""
            volume_ratio = 1.0
            rsi = 50.0
            
            if self.use_momentum_breakout:
                momentum_data = self._calculate_momentum_indicators(data)
                momentum_signal, momentum_confidence, momentum_reasoning = self._analyze_momentum_breakout(momentum_data)
                volume_ratio = momentum_data.get('volume_ratio', 1.0)
                rsi = momentum_data.get('rsi', 50.0)
            
            # 3. 기술적 분석 종합 점수 (100% 기술분석)
            technical_score = 0.0
            strategy_source = 'neutral'
            
            if self.use_ichimoku and self.use_momentum_breakout:
                if ichimoku_signal == 'bullish':
                    technical_score += ichimoku_confidence * 0.6
                elif ichimoku_signal == 'bearish':
                    technical_score -= ichimoku_confidence * 0.6
                    
                if momentum_signal == 'bullish':
                    technical_score += momentum_confidence * 0.4
                elif momentum_signal == 'bearish':
                    technical_score -= momentum_confidence * 0.4
                    
                strategy_source = 'technical_analysis'
            elif self.use_ichimoku:
                if ichimoku_signal == 'bullish':
                    technical_score = ichimoku_confidence
                elif ichimoku_signal == 'bearish':
                    technical_score = -ichimoku_confidence
                strategy_source = 'ichimoku'
            elif self.use_momentum_breakout:
                if momentum_signal == 'bullish':
                    technical_score = momentum_confidence
                elif momentum_signal == 'bearish':
                    technical_score = -momentum_confidence
                strategy_source = 'momentum_breakout'
            
            # 4. 배당 보너스 (일본 시장 특화)
            dividend_bonus = 0.0
            if dividend_yield >= self.dividend_bonus_threshold:
                dividend_bonus = self.dividend_bonus_score
                technical_score += dividend_bonus
            
            # 5. 최종 점수 = 기술분석 점수 + 배당 보너스 (100% 기술분석)
            final_score = technical_score
            
            # 6. 최종 액션 결정 (대폭 완화된 기준)
            if final_score >= 0.5:  # 0.6 → 0.5 (완화)
                final_action = 'buy'
                confidence = min(final_score, 0.95)
            elif final_score <= -0.35:  # -0.5 → -0.35 (완화)
                final_action = 'sell'
                confidence = min(abs(final_score), 0.95)
            else:
                final_action = 'hold'
                confidence = 0.5
                
            # 7. 목표주가 및 포지션 크기 계산
            target_price = self._calculate_target_price(current_price, confidence, final_action)
            position_size = self._calculate_position_size(current_price, confidence)
            
            # 8. 기술분석 reasoning (뉴스 제거, 배당 포함)
            technical_reasoning = f"{ichimoku_reasoning} | {momentum_reasoning}"
            if dividend_bonus > 0:
                technical_reasoning += f" | 배당보너스: {dividend_yield:.1f}%"
            
            return JPStockSignal(
                symbol=symbol,
                action=final_action,
                confidence=confidence,
                price=current_price,
                strategy_source=strategy_source,
                ichimoku_signal=ichimoku_signal,
                momentum_score=final_score,
                volume_ratio=volume_ratio,
                rsi=rsi,
                sector=self._get_sector_for_symbol(symbol),
                reasoning=technical_reasoning,
                target_price=target_price,
                timestamp=datetime.now(),
                additional_data={
                    'technical_score': technical_score,
                    'final_score': final_score,
                    'dividend_yield': dividend_yield,
                    'dividend_bonus': dividend_bonus,
                    'position_size': position_size,
                    'ichimoku_confidence': ichimoku_confidence
