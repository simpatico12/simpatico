"""
🪙 암호화폐 전략 모듈 - 최고퀸트프로젝트 (순수 기술분석 + 상위 10개 자동)
=================================================================================

고급 암호화폐 트레이딩 전략:
- 업비트 시가총액 상위 10개 자동 선택
- 거래량 급증 감지 (Volume Spike Detection)
- 공포탐욕지수 통합 분석
- 변동성 기반 포지션 조정
- 다중 시간프레임 분석
- 순수 기술적 분석 (뉴스 제거)

Author: 최고퀸트팀
Version: 1.2.0 (상위 10개 자동 + 뉴스 제거)
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

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class CoinSignal:
    """암호화폐 시그널 데이터 클래스"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    strategy_source: str  # 'volume_spike', 'fear_greed', 'technical_analysis'
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
    """🪙 고급 암호화폐 전략 클래스 (순수 기술분석 + 상위 10개 자동)"""
    
    def __init__(self, config_path: str = "settings.yaml"):
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
        
        # 업비트 상위 10개 코인 자동 선택
        self.top_10_symbols = []
        self.symbols = {}
        
        # 초기화 시 상위 10개 코인 로드
        if self.enabled:
            self._load_top_10_coins()
        
        # 기술적 분석 파라미터
        self.rsi_period = 14
        self.volatility_limit = 0.2
        self.fear_greed_weight = 0.3
        
        if self.enabled:
            logger.info(f"🪙 암호화폐 전략 초기화 완료 - 상위 {len(self.top_10_symbols)}개 코인 추적")
            logger.info(f"📊 추적 종목: {', '.join(self.top_10_symbols)}")
            logger.info(f"📊 거래량 임계값: {self.volume_spike_threshold}배, 변동성 한계: {self.volatility_limit}")
            logger.info(f"🔧 순수 기술분석 모드 (뉴스 분석 제거)")
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

    def _load_top_10_coins(self):
        """업비트 시가총액 상위 10개 코인 로드"""
        try:
            logger.info("🔍 업비트 상위 10개 코인 검색 중...")
            
            # 업비트 KRW 마켓 전체 티커 조회
            all_tickers = pyupbit.get_tickers(fiat="KRW")
            
            if not all_tickers:
                logger.error("업비트 티커 조회 실패")
                self._set_default_coins()
                return
            
            # 각 코인의 현재 가격과 24시간 거래량 조회
            coin_data = []
            
            # 배치로 가격 조회 (업비트 API 제한 고려)
            batch_size = 10
            for i in range(0, min(len(all_tickers), 50), batch_size):  # 상위 50개만 체크
                batch_tickers = all_tickers[i:i+batch_size]
                try:
                    prices = pyupbit.get_current_price(batch_tickers)
                    if prices:
                        for ticker in batch_tickers:
                            if ticker in prices and prices[ticker]:
                                # 24시간 거래량 데이터 조회
                                ohlcv = pyupbit.get_ohlcv(ticker, interval="day", count=1)
                                if ohlcv is not None and len(ohlcv) > 0:
                                    volume_krw = ohlcv.iloc[-1]['volume'] * prices[ticker]
                                    coin_data.append({
                                        'ticker': ticker,
                                        'price': prices[ticker],
                                        'volume_krw': volume_krw
                                    })
                    
                    # API 호출 제한 고려
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.warning(f"배치 {i} 처리 중 오류: {e}")
                    continue
            
            if not coin_data:
                logger.error("코인 데이터 수집 실패")
                self._set_default_coins()
                return
            
            # 거래량 기준 상위 10개 선택
            coin_data.sort(key=lambda x: x['volume_krw'], reverse=True)
            top_10 = coin_data[:10]
            
            self.top_10_symbols = [coin['ticker'] for coin in top_10]
            
            # 섹터별 분류 (간단히 분류)
            self.symbols = {
                'MAJOR': [],
                'ALTCOIN': [],
                'OTHERS': []
            }
            
            for ticker in self.top_10_symbols:
                if ticker in ['KRW-BTC', 'KRW-ETH']:
                    self.symbols['MAJOR'].append(ticker)
                elif len(self.symbols['ALTCOIN']) < 5:
                    self.symbols['ALTCOIN'].append(ticker)
                else:
                    self.symbols['OTHERS'].append(ticker)
            
            logger.info(f"✅ 상위 10개 코인 로드 완료:")
            for i, coin in enumerate(top_10, 1):
                logger.info(f"  {i}. {coin['ticker']}: {coin['volume_krw']/1e8:.1f}억원 거래량")
                
        except Exception as e:
            logger.error(f"상위 10개 코인 로드 실패: {e}")
            self._set_default_coins()

    def _set_default_coins(self):
        """기본 코인 설정 (API 실패 시)"""
        logger.info("기본 코인 목록으로 설정...")
        self.top_10_symbols = [
            'KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-AVAX',
            'KRW-DOGE', 'KRW-MATIC', 'KRW-ATOM', 'KRW-NEAR', 'KRW-HBAR'
        ]
        self.symbols = {
            'MAJOR': ['KRW-BTC', 'KRW-ETH', 'KRW-XRP'],
            'ALTCOIN': ['KRW-ADA', 'KRW-AVAX', 'KRW-DOGE', 'KRW-MATIC'],
            'OTHERS': ['KRW-ATOM', 'KRW-NEAR', 'KRW-HBAR']
        }

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """심볼에 해당하는 섹터 찾기"""
        for sector, symbols in self.symbols.items():
            if symbol in symbols:
                return sector
        return 'UNKNOWN'

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
        """개별 암호화폐 분석 (순수 기술분석)"""
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

            # 4. 순수 기술적 분석 점수 계산
            technical_score = self._calculate_technical_score(
                volume_spike, price_change_24h, volatility, rsi, fear_greed
            )

            # 5. 최종 점수 = 기술적 분석 점수 (100%)
            final_score = technical_score

            # 6. 최종 액션 결정
            if final_score >= 0.6:
                final_action = 'buy'
                confidence = min(final_score, 0.95)
                strategy_source = 'technical_analysis'
            elif final_score <= -0.5:
                final_action = 'sell'
                confidence = min(abs(final_score), 0.95)
                strategy_source = 'risk_management'
            else:
                final_action = 'hold'
                confidence = 0.5
                strategy_source = 'neutral'

            # 7. 목표가격 및 포지션 크기 계산
            target_price = self._calculate_target_price(current_price, confidence, final_action)
            position_size = self._calculate_position_size(current_price, confidence)

            # 8. 기술적 분석 reasoning 생성
            technical_reasoning = self._generate_technical_reasoning(volume_spike, rsi, fear_greed, volatility)

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
                reasoning=technical_reasoning,
                target_price=target_price,
                timestamp=datetime.now(),
                additional_data={
                    'technical_score': technical_score,
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

    async def scan_all_symbols(self) -> List[CoinSignal]:
        """상위 10개 심볼 스캔"""
        if not self.enabled:
            logger.warning("암호화폐 전략이 비활성화되어 있습니다")
            return []
            
        logger.info(f"🔍 상위 {len(self.top_10_symbols)}개 암호화폐 스캔 시작...")
        
        all_signals = []
        for symbol in self.top_10_symbols:
            try:
                signal = await self.analyze_symbol(symbol)
                all_signals.append(signal)
                
                action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                logger.info(f"{action_emoji} {symbol}: {signal.action} (신뢰도: {signal.confidence:.2f})")
                
                # API 호출 제한 고려
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 분석 실패: {e}")

        buy_count = len([s for s in all_signals if s.action == 'buy'])
        sell_count = len([s for s in all_signals if s.action == 'sell']) 
        hold_count = len([s for s in all_signals if s.action == 'hold'])

        logger.info(f"🎯 스캔 완료 - 매수:{buy_count}개, 매도:{sell_count}개, 보유:{hold_count}개")

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
    """암호화폐 시장 전체 스캔 (상위 10개)"""
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
        
        # 상위 10개 코인 스캔
        print("\n📊 업비트 상위 10개 코인 분석 (순수 기술분석):")
        strategy = CoinStrategy()
        
        if strategy.top_10_symbols:
            # 첫 번째 코인 상세 분석
            first_coin = strategy.top_10_symbols[0]
            result = await analyze_coin(first_coin)
            print(f"\n{first_coin} 상세 분석:")
            print(f"  액션: {result['decision']}")
            print(f"  신뢰도: {result['confidence_score']:.1f}%")
            print(f"  현재가: {result['price']:,.0f}원")
            print(f"  목표가: {result['target_price']:,.0f}원")
            print(f"  이유: {result['reasoning']}")
            
            # 전체 시장 스캔
            print(f"\n📈 상위 10개 코인 전체 스캔:")
            market_scan = await scan_crypto_market()
            print(f"  분석 완료: {market_scan['total_analyzed']}개")
            print(f"  매수 신호: {market_scan['buy_count']}개")
            print(f"  매도 신호: {market_scan['sell_count']}개")
            
            # 매수 추천 종목
            if market_scan['top_buys']:
                print(f"\n🎯 매수 추천 종목:")
                for i, buy_signal in enumerate(market_scan['top_buys'], 1):
                    print(f"  {i}. {buy_signal.symbol}: 신뢰도 {buy_signal.confidence:.2f}")
        else:
            print("❌ 상위 10개 코인 로드 실패")
    
    asyncio.run(main())
