"""
🇯🇵 일본 주식 전략 모듈 - 최고퀸트프로젝트
==========================================

일본 주식 시장 특화 전략:
- 일목균형표 (Ichimoku Kinko Hyo) 분석
- 모멘텀 돌파 (Momentum Breakout) 전략
- 일본 주요 기업 추적 (닛케이225 중심)
- 기술적 분석 통합
- 거래량 기반 신호 생성

Author: 최고퀸트팀
Version: 1.0.0
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
import talib

# 뉴스 분석 모듈 import (있을 때만)
try:
    from news_analyzer import get_news_sentiment
    NEWS_ANALYZER_AVAILABLE = True
except ImportError:
    NEWS_ANALYZER_AVAILABLE = False

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class JPStockSignal:
    """일본 주식 시그널 데이터 클래스"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    strategy_source: str  # 'ichimoku', 'momentum_breakout', 'integrated_analysis'
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
    """🇯🇵 고급 일본 주식 전략 클래스"""
    
    def __init__(self, config_path: str = "configs/settings.yaml"):
        """전략 초기화"""
        self.config = self._load_config(config_path)
        self.jp_config = self.config.get('jp_strategy', {})
        
        # settings.yaml에서 설정값 읽기
        self.enabled = self.jp_config.get('enabled', True)
        self.use_ichimoku = self.jp_config.get('ichimoku', True)
        self.use_momentum_breakout = self.jp_config.get('momentum_breakout', True)
        
        # 일목균형표 파라미터 (settings.yaml 적용)
        self.tenkan_period = self.jp_config.get('tenkan_period', 9)
        self.kijun_period = self.jp_config.get('kijun_period', 26)
        self.senkou_period = self.jp_config.get('senkou_period', 52)
        
        # 모멘텀 돌파 파라미터 (settings.yaml 적용)
        self.breakout_period = self.jp_config.get('breakout_period', 20)
        self.volume_threshold = self.jp_config.get('volume_threshold', 1.5)
        self.rsi_period = self.jp_config.get('rsi_period', 14)
        
        # 뉴스 분석 통합 설정
        self.news_weight = self.jp_config.get('news_weight', 0.4)  # 뉴스 40%
        self.technical_weight = self.jp_config.get('technical_weight', 0.6)  # 기술분석 60%
        
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
            logger.info(f"🇯🇵 일본 주식 전략 초기화 완료 - 추적 종목: {len(self.all_symbols)}개")
            logger.info(f"📊 일목균형표: {self.use_ichimoku}, 모멘텀돌파: {self.use_momentum_breakout}")
            logger.info(f"🔗 뉴스 통합: {self.news_weight*100:.0f}% + 기술분석: {self.technical_weight*100:.0f}%")
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

    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict:
        """일목균형표 계산"""
        try:
            if len(data) < self.senkou_period:
                return {}
                
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # 전환선 (Tenkan-sen): 9일 중간값
            tenkan_high = high.rolling(window=self.tenkan_period).max()
            tenkan_low = low.rolling(window=self.tenkan_period).min()
            tenkan = (tenkan_high + tenkan_low) / 2
            
            # 기준선 (Kijun-sen): 26일 중간값
            kijun_high = high.rolling(window=self.kijun_period).max()
            kijun_low = low.rolling(window=self.kijun_period).min()
            kijun = (kijun_high + kijun_low) / 2
            
            # 선행스팬 A (Senkou Span A): (전환선 + 기준선) / 2
            senkou_a = ((tenkan + kijun) / 2).shift(self.kijun_period)
            
            # 선행스팬 B (Senkou Span B): 52일 중간값
            senkou_b_high = high.rolling(window=self.senkou_period).max()
            senkou_b_low = low.rolling(window=self.senkou_period).min()
            senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(self.kijun_period)
            
            # 후행스팬 (Chikou Span): 현재 종가를 26일 뒤로
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
        """일목균형표 신호 분석"""
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
            
            # 1. 전환선과 기준선 관계
            if tenkan > kijun:
                signal_score += 0.3
                reasons.append("전환선>기준선")
            elif tenkan < kijun:
                signal_score -= 0.3
                reasons.append("전환선<기준선")
                
            # 2. 가격과 구름(일목균형표) 관계
            cloud_top = max(senkou_a, senkou_b) if senkou_a > 0 and senkou_b > 0 else 0
            cloud_bottom = min(senkou_a, senkou_b) if senkou_a > 0 and senkou_b > 0 else 0
            
            if cloud_top > 0:
                if price > cloud_top:
                    signal_score += 0.4
                    reasons.append("구름위")
                elif price < cloud_bottom:
                    signal_score -= 0.4
                    reasons.append("구름아래")
                else:
                    reasons.append("구름속")
                    
            # 3. 구름의 색깔 (두께)
            if senkou_a > senkou_b:
                signal_score += 0.2
                reasons.append("상승구름")
            elif senkou_a < senkou_b:
                signal_score -= 0.2
                reasons.append("하락구름")
                
            # 4. 가격과 기준선 관계
            if price > kijun:
                signal_score += 0.1
                reasons.append("기준선위")
            else:
                signal_score -= 0.1
                reasons.append("기준선아래")
                
            # 신호 결정
            if signal_score >= 0.6:
                signal = 'bullish'
            elif signal_score <= -0.6:
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
        """모멘텀 지표 계산"""
        try:
            if len(data) < self.breakout_period:
                return {}
                
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values
            
            # RSI 계산
            rsi = talib.RSI(close, timeperiod=self.rsi_period)
            
            # 볼린저 밴드
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            
            # 거래량 비율 (최근 5일 평균 대비)
            recent_volume = np.mean(volume[-5:])
            avg_volume = np.mean(volume[-20:-5])
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 가격 돌파 체크 (20일 최고가 돌파)
            breakout_high = np.max(high[-self.breakout_period:-1])
            current_price = close[-1]
            price_breakout = current_price > breakout_high
            
            return {
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'bb_upper': bb_upper[-1],
                'bb_middle': bb_middle[-1], 
                'bb_lower': bb_lower[-1],
                'macd': macd[-1],
                'macd_signal': macd_signal[-1],
                'volume_ratio': volume_ratio,
                'price_breakout': price_breakout,
                'breakout_high': breakout_high,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"모멘텀 지표 계산 실패: {e}")
            return {}

    def _analyze_momentum_breakout(self, momentum_data: Dict) -> Tuple[str, float, str]:
        """모멘텀 돌파 신호 분석"""
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
            
            # 1. 가격 돌파 체크
            if price_breakout:
                signal_score += 0.4
                reasons.append("가격돌파")
                
            # 2. 거래량 증가 체크
            if volume_ratio >= self.volume_threshold:
                signal_score += 0.3
                reasons.append(f"거래량증가({volume_ratio:.1f}배)")
            elif volume_ratio < 0.8:
                signal_score -= 0.2
                reasons.append("거래량감소")
                
            # 3. RSI 체크
            if 30 <= rsi <= 70:  # 과매수/과매도 아닌 정상 범위
                signal_score += 0.2
                reasons.append(f"RSI정상({rsi:.0f})")
            elif rsi > 80:
                signal_score -= 0.3
                reasons.append(f"RSI과매수({rsi:.0f})")
            elif rsi < 20:
                signal_score += 0.1  # 과매도에서 반등 기대
                reasons.append(f"RSI과매도({rsi:.0f})")
                
            # 4. MACD 신호
            if macd > macd_signal:
                signal_score += 0.1
                reasons.append("MACD상승")
            else:
                signal_score -= 0.1
                reasons.append("MACD하락")
                
            # 5. 볼린저 밴드 위치
            if current_price > bb_upper:
                signal_score += 0.1
                reasons.append("밴드상단돌파")
            elif current_price < bb_lower:
                signal_score -= 0.2
                reasons.append("밴드하단이탈")
                
            # 신호 결정
            if signal_score >= 0.7:
                signal = 'bullish'
            elif signal_score <= -0.5:
                signal = 'bearish'
            else:
                signal = 'neutral'
                
            confidence = min(abs(signal_score), 1.0)
            reasoning = "모멘텀: " + " | ".join(reasons)
            
            return signal, confidence, reasoning
            
        except Exception as e:
            logger.error(f"모멘텀 돌파 분석 실패: {e}")
            return 'neutral', 0.0, f"모멘텀 분석 실패: {str(e)}"

    async def _get_news_sentiment(self, symbol: str) -> Tuple[float, str]:
        """뉴스 센티먼트 분석"""
        if not NEWS_ANALYZER_AVAILABLE:
            return 0.5, "뉴스 분석 모듈 없음"
            
        try:
            # news_analyzer.py의 get_news_sentiment 함수 호출
            news_result = await get_news_sentiment(symbol)
            
            if news_result and 'sentiment_score' in news_result:
                score = news_result['sentiment_score']  # 0.0 ~ 1.0
                summary = news_result.get('summary', 'No news summary')
                
                # 점수를 -1 ~ 1 범위로 변환 (0.5 = 중립)
                normalized_score = (score - 0.5) * 2
                
                return normalized_score, f"뉴스: {summary[:50]}"
            else:
                return 0.0, "뉴스 데이터 없음"
                
        except Exception as e:
            logger.error(f"뉴스 센티먼트 분석 실패 {symbol}: {e}")
            return 0.0, f"뉴스 분석 오류: {str(e)}"

    def _calculate_position_size(self, price: float, confidence: float, account_balance: float = 10000000) -> int:
        """포지션 크기 계산 (일본 주식용 - 엔화 기준)"""
        try:
            # 신뢰도에 따른 포지션 사이징
            base_position_pct = 0.02  # 기본 2%
            confidence_multiplier = confidence  # 신뢰도가 높을수록 큰 포지션
            
            position_pct = base_position_pct * confidence_multiplier
            position_pct = min(position_pct, 0.08)  # 일본 주식은 최대 8%로 제한
            
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
            
        # 신호에 따른 기대수익률
        if signal == 'buy':
            expected_return = confidence * 0.15  # 최대 15% 수익 기대
            return current_price * (1 + expected_return)
        elif signal == 'sell':
            expected_return = confidence * 0.10  # 10% 하락 예상
            return current_price * (1 - expected_return)
        else:
            return current_price

    async def analyze_symbol(self, symbol: str) -> JPStockSignal:
        """개별 일본 주식 분석 (뉴스 분석 통합)"""
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
            
            # 1. 기술적 분석
            ichimoku_signal = 'neutral'
            ichimoku_confidence = 0.0
            ichimoku_reasoning = ""
            
            if self.use_ichimoku:
                ichimoku_data = self._calculate_ichimoku(data)
                ichimoku_signal, ichimoku_confidence, ichimoku_reasoning = self._analyze_ichimoku_signal(ichimoku_data)
                
            # 모멘텀 돌파 분석
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
            
            # 기술적 분석 종합 점수
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
                    
                strategy_source = 'integrated_analysis'
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
            
            # 2. 뉴스 센티먼트 분석
            news_score, news_reasoning = await self._get_news_sentiment(symbol)
            
            # 3. 최종 통합 점수 (기술분석 60% + 뉴스 40%)
            final_score = (technical_score * self.technical_weight) + (news_score * self.news_weight)
            
            # 4. 최종 액션 결정
            if final_score >= 0.6:
                final_action = 'buy'
                confidence = min(final_score, 0.95)
            elif final_score <= -0.5:
                final_action = 'sell'
                confidence = min(abs(final_score), 0.95)
            else:
                final_action = 'hold'
                confidence = 0.5
                
            # 5. 목표주가 및 포지션 크기 계산
            target_price = self._calculate_target_price(current_price, confidence, final_action)
            position_size = self._calculate_position_size(current_price, confidence)
            
            # 6. 종합 reasoning
            combined_reasoning = f"{ichimoku_reasoning} | {momentum_reasoning} | {news_reasoning}"
            
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
                reasoning=combined_reasoning,
                target_price=target_price,
                timestamp=datetime.now(),
                additional_data={
                    'technical_score': technical_score,
                    'news_score': news_score,
                    'final_score': final_score,
                    'position_size': position_size,
                    'ichimoku_confidence': ichimoku_confidence,
                    'momentum_confidence': momentum_confidence
                }
            )

        except Exception as e:
            logger.error(f"일본 주식 분석 실패 {symbol}: {e}")
            return JPStockSignal(
                symbol=symbol,
                action='hold',
                confidence=0.0,
                price=0.0,
                strategy_source='error',
                ichimoku_signal='neutral',
                momentum_score=0.0,
                volume_ratio=0.0,
                rsi=50.0,
                sector='UNKNOWN',
                reasoning=f"분석 실패: {str(e)}",
                target_price=0.0,
                timestamp=datetime.now()
            )

    async def scan_by_sector(self, sector: str) -> List[JPStockSignal]:
        """섹터별 스캔"""
        if not self.enabled:
            logger.warning("일본 주식 전략이 비활성화되어 있습니다")
            return []
            
        if sector not in self.symbols:
            logger.error(f"알 수 없는 섹터: {sector}")
            return []
            
        logger.info(f"🔍 {sector} 섹터 (일본) 스캔 시작...")
        symbols = self.symbols[sector]
        
        signals = []
        for symbol in symbols:
            try:
                signal = await self.analyze_symbol(symbol)
                signals.append(signal)
                logger.info(f"✅ {symbol}: {signal.action} (신뢰도: {signal.confidence:.2f})")
                
                # API 호출 제한 고려
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 분석 실패: {e}")

        return signals

    async def scan_all_symbols(self) -> List[JPStockSignal]:
        """전체 심볼 스캔"""
        if not self.enabled:
            logger.warning("일본 주식 전략이 비활성화되어 있습니다")
            return []
            
        logger.info(f"🔍 {len(self.all_symbols)}개 일본 주식 스캔 시작...")
        
        all_signals = []
        for sector in self.symbols.keys():
            sector_signals = await self.scan_by_sector(sector)
            all_signals.extend(sector_signals)
            
            # 섹터간 대기
            await asyncio.sleep(2)

        logger.info(f"🎯 스캔 완료 - 매수:{len([s for s in all_signals if s.action=='buy'])}개, "
                   f"매도:{len([s for s in all_signals if s.action=='sell'])}개, "
                   f"보유:{len([s for s in all_signals if s.action=='hold'])}개")

        return all_signals

    async def get_top_picks(self, strategy: str = 'all', limit: int = 5) -> List[JPStockSignal]:
        """상위 종목 추천"""
        all_signals = await self.scan_all_symbols()
        
        # 전략별 필터링
        if strategy == 'ichimoku':
            filtered = [s for s in all_signals if 'ichimoku' in s.strategy_source and s.action == 'buy']
        elif strategy == 'momentum':
            filtered = [s for s in all_signals if 'momentum' in s.strategy_source and s.action == 'buy']
        else:
            filtered = [s for s in all_signals if s.action == 'buy']
        
        # 신뢰도 순 정렬
        sorted_signals = sorted(filtered, key=lambda x: x.confidence, reverse=True)
        
        return sorted_signals[:limit]

# 편의 함수들 (core.py에서 호출용)
async def analyze_jp(symbol: str) -> Dict:
    """단일 일본 주식 분석 (기존 인터페이스 호환)"""
    strategy = JPStrategy()
    signal = await strategy.analyze_symbol(symbol)
    
    result = {
        'decision': signal.action,
        'confidence_score': signal.confidence * 100,
        'reasoning': signal.reasoning,
        'target_price': signal.target_price,
        'ichimoku_signal': signal.ichimoku_signal,
        'rsi': signal.rsi,
        'volume_ratio': signal.volume_ratio,
        'price': signal.price,
        'sector': signal.sector
    }
    
    # 추가 데이터가 있으면 포함
    if signal.additional_data:
        result['additional_data'] = signal.additional_data
        
    return result

async def get_ichimoku_picks() -> List[Dict]:
    """일목균형표 기반 추천 종목"""
    strategy = JPStrategy()
    signals = await strategy.get_top_picks('ichimoku', limit=10)
    
    picks = []
    for signal in signals:
        picks.append({
            'symbol': signal.symbol,
            'sector': signal.sector,
            'confidence': signal.confidence,
            'ichimoku_signal': signal.ichimoku_signal,
            'reasoning': signal.reasoning,
            'target_price': signal.target_price,
            'current_price': signal.price
        })
    
    return picks

async def get_momentum_picks() -> List[Dict]:
    """모멘텀 돌파 기반 추천 종목"""
    strategy = JPStrategy()
    signals = await strategy.get_top_picks('momentum', limit=10)
    
    picks = []
    for signal in signals:
        picks.append({
            'symbol': signal.symbol,
            'sector': signal.sector,
            'confidence': signal.confidence,
            'momentum_score': signal.momentum_score,
            'volume_ratio': signal.volume_ratio,
            'rsi': signal.rsi,
            'target_price': signal.target_price,
            'current_price': signal.price
        })
    
    return picks

async def scan_jp_market() -> Dict:
    """일본 시장 전체 스캔"""
    strategy = JPStrategy()
    signals = await strategy.scan_all_symbols()
    
    buy_signals = [s for s in signals if s.action == 'buy']
    sell_signals = [s for s in signals if s.action == 'sell']
    
    return {
        'total_analyzed': len(signals),
        'buy_count': len(buy_signals),
        'sell_count': len(sell_signals),
        'top_buys': sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5],
        'top_sells': sorted(sell_signals, key=lambda x: x.confidence, reverse=True)[:5]
    }

if __name__ == "__main__":
    async def main():
        print("🇯🇵 최고퀸트프로젝트 - 일본 주식 전략 테스트 시작...")
        
        # 단일 주식 테스트 (토요타)
        print("\n📊 토요타(7203.T) 개별 분석 (뉴스 통합):")
        toyota_result = await analyze_jp('7203.T')
        print(f"토요타: {toyota_result}")
        
        # 상세 분석 결과 출력
        if 'additional_data' in toyota_result:
            additional = toyota_result['additional_data']
            print(f"  기술분석: {additional.get('technical_score', 0):.2f}")
            print(f"  뉴스점수: {additional.get('news_score', 0):.2f}")
            print(f"  최종점수: {additional.get('final_score', 0):.2f}")
            print(f"  포지션크기: {additional.get('position_size', 0)}주 (100주 단위)")
            print(f"  일목신뢰도: {additional.get('ichimoku_confidence', 0):.2f}")
            print(f"  모멘텀신뢰도: {additional.get('momentum_confidence', 0):.2f}")
        
        # 일목균형표 추천
        print("\n📈 일목균형표 기반 추천:")
        ichimoku_picks = await get_ichimoku_picks()
        for pick in ichimoku_picks[:3]:
            print(f"{pick['symbol']}: 신뢰도 {pick['confidence']:.2f}, 일목신호 {pick['ichimoku_signal']}")
        
        # 모멘텀 돌파 추천  
        print("\n🚀 모멘텀 돌파 기반 추천:")
        momentum_picks = await get_momentum_picks()
        for pick in momentum_picks[:3]:
            print(f"{pick['symbol']}: 신뢰도 {pick['confidence']:.2f}, RSI {pick['rsi']:.1f}")
    
    asyncio.run(main())
