"""
🪙 암호화폐 전략 모듈 - 최고퀸트프로젝트
==========================================

고급 암호화폐 트레이딩 전략:
- 거래량 급증 감지 (Volume Spike Detection)
- 공포탐욕지수 통합 분석
- 변동성 기반 포지션 조정
- 다중 시간프레임 분석
- 실시간 뉴스 센티먼트 통합

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
import requests
import pyupbit

# 뉴스 분석 모듈 import (있을 때만)
try:
    from news_analyzer import get_news_sentiment
    NEWS_ANALYZER_AVAILABLE = True
except ImportError:
    NEWS_ANALYZER_AVAILABLE = False

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class CoinSignal:
    """암호화폐 시그널 데이터 클래스"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    strategy_source: str  # 'volume_spike', 'fear_greed', 'integrated_analysis'
    volume_spike_ratio: float
    price_change_24h: float
    fear_greed_score: int
    volatility: float
    rsi: float
    sector: str
    reasoning: str
    target_price: float
    timestamp: datetime
    additional_data: Optional[Dict] = None

class CoinStrategy:
    """🪙 고급 암호화폐 전략 클래스"""
    
    def __init__(self, config_path: str = "configs/settings.yaml"):
        """전략 초기화"""
        self.config = self._load_config(config_path)
        self.coin_config = self.config.get('coin_strategy', {})
        
        # settings.yaml에서 설정값 읽기
        self.enabled = self.coin_config.get('enabled', True)
        
        # 거래량 분석 설정
        self.volume_spike_threshold = self.coin_config.get('volume_spike_threshold', 2.0)
        self.volume_analysis_period = self.coin_config.get('volume_analysis_period', 24)
        
        # 가격 움직임 설정
        self.price_change_threshold = self.coin_config.get('price_change_threshold', 0.05)
        self.volatility_window = self.coin_config.get('volatility_window', 20)
        
        # 뉴스 분석 통합 설정
        self.news_weight = self.coin_config.get('news_weight', 0.5)  # 뉴스 50%
        self.technical_weight = self.coin_config.get('technical_weight', 0.5)  # 기술분석 50%
        
        # 추적할 암호화폐 (settings.yaml에서 로드)
        self.symbols = self.coin_config.get('symbols', {
            'MAJOR': ['BTC-KRW', 'ETH-KRW', 'XRP-KRW', 'ADA-KRW'],
            'DEFI': ['UNI-KRW', 'LINK-KRW', 'AAVE-KRW'],
            'ALTCOIN': ['SOL-KRW', 'MATIC-KRW', 'DOT-KRW']
        })
        
        # 모든 심볼을 플랫 리스트로
        self.all_symbols = [symbol for sector_symbols in self.symbols.values() 
                           for symbol in sector_symbols]
        
        # 기술적 분석 파라미터
        self.rsi_period = 14
        self.volatility_limit = 0.2
        self.fear_greed_weight = 0.3
        
        if self.enabled:
            logger.info(f"🪙 암호화폐 전략 초기화 완료 - 추적 종목: {len(self.all_symbols)}개")
            logger.info(f"📊 거래량 임계값: {self.volume_spike_threshold}배, 변동성 한계: {self.volatility_limit}")
            logger.info(f"🔗 뉴스 통합: {self.news_weight*100:.0f}% + 기술분석: {self.technical_weight*100:.0f}%")
        else:
            logger.info("🪙 암호화폐 전략이 비활성화되어 있습니다")

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

    async def _get_news_sentiment(self, symbol: str) -> Tuple[float, str]:
        """뉴스 센티먼트 분석 (암호화폐 특화)"""
        if not NEWS_ANALYZER_AVAILABLE:
            return 0.5, "뉴스 분석 모듈 없음"
            
        try:
            # 심볼에서 코인명만 추출 (BTC-KRW -> BTC)
            coin_name = symbol.split('-')[0]
            
            # news_analyzer.py의 get_news_sentiment 함수 호출
            news_result = await get_news_sentiment(coin_name)
            
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

    async def get_fear_greed_index(self) -> int:
        """공포탐욕지수 조회"""
        try:
            response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return int(data["data"][0]["value"])
            return 50  # 중립값
        except Exception as e:
            logger.error(f"공포탐욕지수 조회 실패: {e}")
            return 50

    def _calculate_volume_spike(self, current_volume: float, avg_volume: float) -> float:
        """거래량 급증률 계산"""
        if avg_volume == 0:
            return 1.0
        return current_volume / avg_volume

    def _calculate_volatility(self, prices: List[float]) -> float:
        """변동성 계산 (표준편차)"""
        if len(prices) < 2:
            return 0.0
        return np.std(prices) / np.mean(prices)

    def _get_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_position_size(self, price: float, confidence: float, account_balance: float = 50000000) -> float:
        """포지션 크기 계산 (암호화폐용 - 원화 기준)"""
        try:
            # 신뢰도에 따른 포지션 사이징
            base_position_pct = 0.03  # 기본 3% (암호화폐는 변동성이 높아 보수적)
            confidence_multiplier = confidence  # 신뢰도가 높을수록 큰 포지션
            
            position_pct = base_position_pct * confidence_multiplier
            position_pct = min(position_pct, 0.10)  # 최대 10%로 제한
            
            position_value = account_balance * position_pct
            coin_amount = position_value / price if price > 0 else 0
            
            return coin_amount
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 실패: {e}")
            return 0

    def _calculate_target_price(self, current_price: float, confidence: float, action: str) -> float:
        """목표가격 계산"""
        if current_price == 0:
            return 0
            
        if action == 'buy':
            # 암호화폐는 변동성이 크므로 더 높은 수익 기대
            expected_return = confidence * 0.25  # 최대 25% 수익 기대
            return current_price * (1 + expected_return)
        elif action == 'sell':
            expected_loss = confidence * 0.15  # 15% 손실 예상
            return current_price * (1 - expected_loss)
        else:
            return current_price

    async def analyze_symbol(self, symbol: str) -> CoinSignal:
        """개별 암호화폐 분석 (뉴스 분석 통합)"""
        if not self.enabled:
            logger.warning("암호화폐 전략이 비활성화되어 있습니다")
            return CoinSignal(
                symbol=symbol, action='hold', confidence=0.0, price=0.0,
                strategy_source='disabled', volume_spike_ratio=0.0, price_change_24h=0.0,
                fear_greed_score=50, volatility=0.0, rsi=50.0, sector='UNKNOWN',
                reasoning="전략 비활성화", target_price=0.0, timestamp=datetime.now()
            )
            
        try:
            # 1. 현재 가격 및 기본 데이터
            current_price = pyupbit.get_current_price(symbol)
            if not current_price:
                raise ValueError(f"가격 데이터 없음: {symbol}")

            # 2. OHLCV 데이터 수집
            ohlcv_4h = pyupbit.get_ohlcv(symbol, interval="minute240", count=self.volume_analysis_period)
            ohlcv_1d = pyupbit.get_ohlcv(symbol, interval="day", count=30)
            
            if ohlcv_4h is None or ohlcv_1d is None or len(ohlcv_4h) == 0 or len(ohlcv_1d) == 0:
                raise ValueError(f"OHLCV 데이터 없음: {symbol}")

            # 3. 기술적 분석
            # 거래량 급증 감지
            current_volume = ohlcv_4h.iloc[-1]['volume']
            avg_volume = ohlcv_4h['volume'].mean()
            volume_spike = self._calculate_volume_spike(current_volume, avg_volume)

            # 24시간 가격 변동률
            price_24h_ago = ohlcv_1d.iloc[-2]['close'] if len(ohlcv_1d) >= 2 else current_price
            price_change_24h = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0

            # 변동성 계산
            prices_24h = ohlcv_4h['close'].tolist()
            volatility = self._calculate_volatility(prices_24h)

            # RSI 계산
            rsi = self._get_rsi(ohlcv_1d['close'].tolist(), self.rsi_period)
            
            # 공포탐욕지수
            fear_greed = await self.get_fear_greed_index()

            # 4. 기술적 분석 점수 계산
            technical_score = self._calculate_technical_score(
                volume_spike, price_change_24h, volatility, rsi, fear_greed
            )

            # 5. 뉴스 센티먼트 분석
            news_score, news_reasoning = await self._get_news_sentiment(symbol)

            # 6. 최종 통합 점수 (기술분석 50% + 뉴스 50%)
            final_score = (technical_score * self.technical_weight) + (news_score * self.news_weight)

            # 7. 최종 액션 결정
            if final_score >= 0.6:
                final_action = 'buy'
                confidence = min(final_score, 0.95)
                strategy_source = 'integrated_analysis'
            elif final_score <= -0.5:
                final_action = 'sell'
                confidence = min(abs(final_score), 0.95)
                strategy_source = 'risk_management'
            else:
                final_action = 'hold'
                confidence = 0.5
                strategy_source = 'neutral'

            # 8. 목표가격 및 포지션 크기 계산
            target_price = self._calculate_target_price(current_price, confidence, final_action)
            position_size = self._calculate_position_size(current_price, confidence)

            # 9. 종합 reasoning 생성
            technical_reasoning = self._generate_technical_reasoning(volume_spike, rsi, fear_greed, volatility)
            combined_reasoning = f"{technical_reasoning} | {news_reasoning}"

            return CoinSignal(
                symbol=symbol,
                action=final_action,
                confidence=confidence,
                price=current_price,
                strategy_source=strategy_source,
                volume_spike_ratio=volume_spike,
                price_change_24h=price_change_24h,
                fear_greed_score=fear_greed,
                volatility=volatility,
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
                    'avg_volume_24h': avg_volume,
                    'current_volume': current_volume
                }
            )

        except Exception as e:
            logger.error(f"암호화폐 분석 실패 {symbol}: {e}")
            return CoinSignal(
                symbol=symbol,
                action='hold',
                confidence=0.0,
                price=0.0,
                strategy_source='error',
                volume_spike_ratio=0.0,
                price_change_24h=0.0,
                fear_greed_score=50,
                volatility=0.0,
                rsi=50.0,
                sector='UNKNOWN',
                reasoning=f"분석 실패: {str(e)}",
                target_price=0.0,
                timestamp=datetime.now()
            )

    def _calculate_technical_score(self, volume_spike: float, price_change_24h: float, 
                                 volatility: float, rsi: float, fear_greed: int) -> float:
        """기술적 분석 점수 계산"""
        score = 0.0
        
        # 1. 거래량 급증 체크 (40% 가중치)
        if volume_spike >= self.volume_spike_threshold:
            score += 0.4 * min(volume_spike / self.volume_spike_threshold / 2, 1.0)
        
        # 2. RSI 기반 판단 (25% 가중치)
        if rsi < 30:  # 과매도
            score += 0.25
        elif rsi > 70:  # 과매수
            score -= 0.25

        # 3. 공포탐욕지수 반영 (20% 가중치)
        if fear_greed < 25:  # 극도의 공포 = 매수 기회
            score += 0.2
        elif fear_greed > 75:  # 극도의 탐욕 = 매도 시점
            score -= 0.2

        # 4. 24시간 가격 변동 (15% 가중치)
        if abs(price_change_24h) > self.price_change_threshold:
            if price_change_24h > 0:
                score += 0.15  # 상승 추세
            else:
                score -= 0.15  # 하락 추세

        # 5. 변동성 체크 (리스크 조정)
        if volatility > self.volatility_limit:
            score *= 0.7  # 변동성 높으면 보수적

        return max(-1.0, min(1.0, score))  # -1 ~ 1 범위로 제한

    def _generate_technical_reasoning(self, volume_spike: float, rsi: float, 
                                    fear_greed: int, volatility: float) -> str:
        """기술적 분석 reasoning 생성"""
        reasons = []
        
        if volume_spike >= self.volume_spike_threshold:
            reasons.append(f"거래량급증({volume_spike:.1f}배)")
            
        if rsi < 30:
            reasons.append(f"RSI과매도({rsi:.1f})")
        elif rsi > 70:
            reasons.append(f"RSI과매수({rsi:.1f})")
            
        if fear_greed < 25:
            reasons.append("극도공포")
        elif fear_greed > 75:
            reasons.append("극도탐욕")
        elif 25 <= fear_greed <= 75:
            reasons.append(f"공포탐욕지수({fear_greed})")
            
        if volatility > self.volatility_limit:
            reasons.append("고변동성")
            
        return "기술분석: " + " | ".join(reasons) if reasons else "기술분석: 보통"

    async def scan_by_sector(self, sector: str) -> List[CoinSignal]:
        """섹터별 스캔"""
        if not self.enabled:
            logger.warning("암호화폐 전략이 비활성화되어 있습니다")
            return []
            
        if sector not in self.symbols:
            logger.error(f"알 수 없는 섹터: {sector}")
            return []
            
        logger.info(f"🔍 {sector} 섹터 (암호화폐) 스캔 시작...")
        symbols = self.symbols[sector]
        
        signals = []
        for symbol in symbols:
            try:
                signal = await self.analyze_symbol(symbol)
                signals.append(signal)
                logger.info(f"✅ {symbol}: {signal.action} (신뢰도: {signal.confidence:.2f})")
                
                # API 호출 제한 고려 (업비트 API)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 분석 실패: {e}")

        return signals

    async def scan_all_symbols(self) -> List[CoinSignal]:
        """전체 심볼 스캔"""
        if not self.enabled:
            logger.warning("암호화폐 전략이 비활성화되어 있습니다")
            return []
            
        logger.info(f"🔍 {len(self.all_symbols)}개 암호화폐 스캔 시작...")
        
        all_signals = []
        for sector in self.symbols.keys():
            sector_signals = await self.scan_by_sector(sector)
            all_signals.extend(sector_signals)
            
            # 섹터간 대기
            await asyncio.sleep(0.5)

        logger.info(f"🎯 스캔 완료 - 매수:{len([s for s in all_signals if s.action=='buy'])}개, "
                   f"매도:{len([s for s in all_signals if s.action=='sell'])}개, "
                   f"보유:{len([s for s in all_signals if s.action=='hold'])}개")

        return all_signals

    async def get_top_picks(self, strategy: str = 'all', limit: int = 5) -> List[CoinSignal]:
        """상위 종목 추천"""
        all_signals = await self.scan_all_symbols()
        
        # 전략별 필터링
        if strategy == 'volume_spike':
            filtered = [s for s in all_signals if s.volume_spike_ratio >= self.volume_spike_threshold and s.action == 'buy']
        elif strategy == 'fear_greed':
            filtered = [s for s in all_signals if (s.fear_greed_score < 25 or s.fear_greed_score > 75) and s.action == 'buy']
        else:
            filtered = [s for s in all_signals if s.action == 'buy']
        
        # 신뢰도 순 정렬
        sorted_signals = sorted(filtered, key=lambda x: x.confidence, reverse=True)
        
        return sorted_signals[:limit]

# 편의 함수들 (core.py에서 호출용)
async def analyze_coin(symbol: str) -> Dict:
    """단일 암호화폐 분석 (기존 인터페이스 호환)"""
    strategy = CoinStrategy()
    signal = await strategy.analyze_symbol(symbol)
    
    result = {
        'decision': signal.action,
        'confidence_score': signal.confidence * 100,
        'reasoning': signal.reasoning,
        'target_price': signal.target_price,
        'volume_spike': signal.volume_spike_ratio,
        'fear_greed': signal.fear_greed_score,
        'rsi': signal.rsi,
        'price': signal.price,
        'sector': signal.sector,
        'price_change_24h': signal.price_change_24h
    }
    
    # 추가 데이터가 있으면 포함
    if signal.additional_data:
        result['additional_data'] = signal.additional_data
        
    return result

async def get_volume_spike_picks() -> List[Dict]:
    """거래량 급증 기반 추천 종목"""
    strategy = CoinStrategy()
    signals = await strategy.get_top_picks('volume_spike', limit=10)
    
    picks = []
    for signal in signals:
        picks.append({
            'symbol': signal.symbol,
            'sector': signal.sector,
            'confidence': signal.confidence,
            'volume_spike': signal.volume_spike_ratio,
            'reasoning': signal.reasoning,
            'target_price': signal.target_price,
            'current_price': signal.price
        })
    
    return picks

async def get_fear_greed_picks() -> List[Dict]:
    """공포탐욕지수 기반 추천 종목"""
    strategy = CoinStrategy()
    signals = await strategy.get_top_picks('fear_greed', limit=10)
    
    picks = []
    for signal in signals:
        picks.append({
            'symbol': signal.symbol,
            'sector': signal.sector,
            'confidence': signal.confidence,
            'fear_greed_score': signal.fear_greed_score,
            'reasoning': signal.reasoning,
            'target_price': signal.target_price,
            'current_price': signal.price
        })
    
    return picks

async def scan_crypto_market() -> Dict:
    """암호화폐 시장 전체 스캔"""
    strategy = CoinStrategy()
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
        print("🪙 최고퀸트프로젝트 - 암호화폐 전략 테스트 시작...")
        
        # 단일 코인 테스트
        print("\n📊 BTC-KRW 개별 분석 (뉴스 통합):")
        btc_result = await analyze_coin('BTC-KRW')
        print(f"BTC: {btc_result}")
        
        # 상세 분석 결과 출력
        if 'additional_data' in btc_result:
            additional = btc_result['additional_data']
            print(f"  기술분석: {additional.get('technical_score', 0):.2f}")
            print(f"  뉴스점수: {additional.get('news_score', 0):.2f}")
            print(f"  최종점수: {additional.get('final_score', 0):.2f}")
            print(f"  포지션크기: {additional.get('position_size', 0):.2f} 코인")
        
        # 거래량 급증 추천
        print("\n📈 거래량 급증 기반 추천:")
        volume_picks = await get_volume_spike_picks()
        for pick in volume_picks[:3]:
            print(f"{pick['symbol']}: 신뢰도 {pick['confidence']:.2f}, 거래량 {pick['volume_spike']:.1f}배")
        
        # 공포탐욕지수 추천  
        print("\n😱 공포탐욕지수 기반 추천:")
        fear_picks = await get_fear_greed_picks()
        for pick in fear_picks[:3]:
            print(f"{pick['symbol']}: 신뢰도 {pick['confidence']:.2f}, 공포탐욕 {pick['fear_greed_score']}")
    
    asyncio.run(main())
