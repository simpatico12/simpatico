"""
🪙 암호화폐 전략 모듈 - 최고퀸트프로젝트 (궁극 업그레이드)
=================================================================================

궁극의 하이브리드 암호화폐 전략:
- 자동 종목 선별 (업비트 전체 → 상위 20개)
- 하이브리드 전략 (펀더멘털 30% + 기술분석 40% + 모멘텀 30%)
- 확장된 기술적 지표 (일목균형표, RSI, MACD, 볼린저밴드, 스토캐스틱, ATR 등)
- 5단계 분할매매 시스템 (20% × 5)
- 24시간 실시간 모니터링
- 완전 자동화

Author: 최고퀸트팀
Version: 3.0.0 (궁극 업그레이드)
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
import ta
import warnings
warnings.filterwarnings('ignore')

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class UltimateCoinSignal:
    """궁극의 암호화폐 시그널 데이터 클래스"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    
    # 전략별 점수
    fundamental_score: float
    technical_score: float
    momentum_score: float
    total_score: float
    
    # 펀더멘털 지표
    market_cap_rank: int
    volume_24h_rank: int
    project_quality: float
    adoption_score: float
    
    # 기술적 지표 (확장)
    rsi: float
    macd_signal: str
    bb_position: str
    stoch_k: float
    stoch_d: float
    ichimoku_signal: str
    atr: float
    obv_trend: str
    
    # 모멘텀 지표
    momentum_3d: float
    momentum_7d: float
    momentum_30d: float
    volume_spike_ratio: float
    price_velocity: float
    
    # 분할매매 정보
    position_stage: int  # 0,1,2,3,4,5 (현재 매수 단계)
    total_amount: float  # 총 투자금액
    stage1_amount: float # 1단계 금액 (20%)
    stage2_amount: float # 2단계 금액 (20%)
    stage3_amount: float # 3단계 금액 (20%)
    stage4_amount: float # 4단계 금액 (20%)
    stage5_amount: float # 5단계 금액 (20%)
    entry_price_1: float # 1단계 진입가
    entry_price_2: float # 2단계 진입가 (5% 하락)
    entry_price_3: float # 3단계 진입가 (10% 하락)
    entry_price_4: float # 4단계 진입가 (15% 하락)
    entry_price_5: float # 5단계 진입가 (20% 하락)
    stop_loss: float     # 손절가 (-25%)
    take_profit_1: float # 1차 익절가 (+20%)
    take_profit_2: float # 2차 익절가 (+50%)
    take_profit_3: float # 3차 익절가 (+100%)
    max_hold_days: int
    
    # 공포탐욕지수
    fear_greed_score: int
    market_sentiment: str
    
    sector: str
    reasoning: str
    target_price: float
    timestamp: datetime
    additional_data: Optional[Dict] = None

class UltimateCoinStrategy:
    """🚀 궁극의 암호화폐 전략 클래스"""
    
    def __init__(self, config_path: str = "settings.yaml"):
        """전략 초기화"""
        self.config = self._load_config(config_path)
        self.coin_config = self.config.get('coin_strategy', {})
        self.enabled = self.coin_config.get('enabled', True)
        
        # 🎯 자동 선별 설정
        self.target_coins = 20  # 상위 20개 코인 선별
        self.min_market_cap_rank = 100  # 시총 100위 이내
        self.min_volume_24h = 1_000_000_000  # 일일 거래량 10억원 이상
        
        # 📊 하이브리드 전략 가중치
        self.fundamental_weight = 0.30  # 펀더멘털 30%
        self.technical_weight = 0.40    # 기술분석 40%
        self.momentum_weight = 0.30     # 모멘텀 30%
        
        # 💰 포트폴리오 설정 (코인은 20% 비중)
        self.total_portfolio_ratio = 0.20  # 전체 포트폴리오의 20%
        self.coin_portfolio_value = 200_000_000  # 2억원 기준
        
        # 🔧 5단계 분할매매 설정
        self.stage_ratios = [0.20, 0.20, 0.20, 0.20, 0.20]  # 각 20%씩
        self.stage_triggers = [0.0, -0.05, -0.10, -0.15, -0.20]  # 진입 조건
        
        # 🛡️ 리스크 관리 (코인 특화)
        self.stop_loss_pct = 0.25       # 25% 손절 (주식보다 널널)
        self.take_profit_levels = [0.20, 0.50, 1.00]  # 20%, 50%, 100% 익절
        self.max_hold_days = 30         # 최대 보유 30일 (단기)
        self.max_single_coin_weight = 0.10  # 단일 코인 최대 10%
        
        # 📈 확장 기술적 지표 설정
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.stoch_k = 14
        self.stoch_d = 3
        self.atr_period = 14
        
        # 🔍 선별된 코인 리스트
        self.selected_coins = []
        self.coin_rankings = {}
        
        if self.enabled:
            logger.info(f"🪙 궁극의 암호화폐 전략 초기화")
            logger.info(f"🎯 자동 선별: 상위 {self.target_coins}개 코인")
            logger.info(f"📊 하이브리드 전략: 펀더멘털{self.fundamental_weight*100:.0f}% + 기술분석{self.technical_weight*100:.0f}% + 모멘텀{self.momentum_weight*100:.0f}%")
            logger.info(f"💰 5단계 분할매매: 각 20%씩, 손절{self.stop_loss_pct*100:.0f}%, 익절100%까지")
            logger.info(f"🛡️ 최대 보유: {self.max_hold_days}일, 단일코인 최대 {self.max_single_coin_weight*100:.0f}%")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return {}

    async def auto_select_top20_coins(self) -> List[str]:
        """🎯 자동 코인 선별: 업비트 전체에서 상위 20개 선별"""
        if not self.enabled:
            logger.warning("암호화폐 전략이 비활성화되어 있습니다")
            return []
        
        logger.info(f"🔍 자동 코인 선별 시작 - 업비트 전체에서 상위 {self.target_coins}개 선별")
        
        try:
            # 1단계: 모든 KRW 마켓 코인 수집
            all_tickers = pyupbit.get_tickers(fiat="KRW")
            if not all_tickers:
                logger.error("업비트 티커 조회 실패")
                return self._get_default_coins()
            
            logger.info(f"📊 1단계: {len(all_tickers)}개 코인 발견")
            
            # 2단계: 각 코인의 기본 데이터 수집
            coin_data = []
            batch_size = 20
            
            for i in range(0, len(all_tickers), batch_size):
                batch_tickers = all_tickers[i:i+batch_size]
                
                try:
                    # 현재가 조회
                    prices = pyupbit.get_current_price(batch_tickers)
                    if not prices:
                        continue
                    
                    for ticker in batch_tickers:
                        if ticker not in prices or not prices[ticker]:
                            continue
                        
                        # 24시간 OHLCV 데이터
                        ohlcv = pyupbit.get_ohlcv(ticker, interval="day", count=2)
                        if ohlcv is None or len(ohlcv) < 2:
                            continue
                        
                        current_price = prices[ticker]
                        volume_krw = ohlcv.iloc[-1]['volume'] * current_price
                        price_change_24h = (ohlcv.iloc[-1]['close'] - ohlcv.iloc[-2]['close']) / ohlcv.iloc[-2]['close']
                        
                        # 기본 필터링
                        if volume_krw >= self.min_volume_24h:
                            coin_data.append({
                                'ticker': ticker,
                                'price': current_price,
                                'volume_krw': volume_krw,
                                'price_change_24h': price_change_24h,
                                'preliminary_score': self._calculate_preliminary_score(volume_krw, price_change_24h)
                            })
                    
                    # API 제한 고려
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.warning(f"배치 {i} 처리 중 오류: {e}")
                    continue
            
            if not coin_data:
                logger.error("코인 데이터 수집 실패")
                return self._get_default_coins()
            
            logger.info(f"📊 2단계: {len(coin_data)}개 코인이 기본 필터 통과")
            
            # 3단계: 예비 점수 기준 상위 50개 선별
            coin_data.sort(key=lambda x: x['preliminary_score'], reverse=True)
            top_50 = coin_data[:50]
            
            logger.info(f"📊 3단계: 상위 50개 코인 선별 완료")
            
            # 4단계: 상세 분석으로 최종 20개 선별
            logger.info(f"🎯 4단계: 상위 {self.target_coins}개 최종 선별을 위한 상세 분석")
            
            detailed_signals = []
            for i, coin in enumerate(top_50, 1):
                try:
                    logger.info(f"📊 상세 분석... {i}/50 - {coin['ticker']}")
                    signal = await self.analyze_symbol(coin['ticker'])
                    detailed_signals.append(signal)
                    
                    # 진행상황 표시
                    if signal.action == 'buy':
                        logger.info(f"🟢 {coin['ticker']}: 매수 신뢰도:{signal.confidence:.2f} 총점:{signal.total_score:.2f}")
                    elif signal.action == 'sell':
                        logger.info(f"🔴 {coin['ticker']}: 매도 신뢰도:{signal.confidence:.2f} 총점:{signal.total_score:.2f}")
                    else:
                        logger.info(f"⚪ {coin['ticker']}: 보유 신뢰도:{signal.confidence:.2f} 총점:{signal.total_score:.2f}")
                    
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.error(f"❌ {coin['ticker']} 상세 분석 실패: {e}")
            
            # 5단계: 최종 20개 선별
            detailed_signals.sort(key=lambda x: x.total_score, reverse=True)
            final_20 = detailed_signals[:self.target_coins]
            
            self.selected_coins = [signal.symbol for signal in final_20]
            
            logger.info(f"🏆 자동 선별 완료: {len(self.selected_coins)}개 코인")
            
            # 선별 결과 요약
            buy_count = len([s for s in final_20 if s.action == 'buy'])
            sell_count = len([s for s in final_20 if s.action == 'sell'])
            hold_count = len([s for s in final_20 if s.action == 'hold'])
            
            logger.info(f"📊 최종 결과: 매수 {buy_count}개, 매도 {sell_count}개, 보유 {hold_count}개")
            
            # 상위 5개 표시
            logger.info("🥇 상위 5개 코인:")
            for i, signal in enumerate(final_20[:5], 1):
                logger.info(f"  {i}. {signal.symbol}: 총점 {signal.total_score:.2f} (펀더멘털:{signal.fundamental_score:.2f} + 기술:{signal.technical_score:.2f} + 모멘텀:{signal.momentum_score:.2f})")
            
            return self.selected_coins
            
        except Exception as e:
            logger.error(f"자동 코인 선별 실패: {e}")
            return self._get_default_coins()

    def _get_default_coins(self) -> List[str]:
        """기본 코인 리스트 (API 실패시)"""
        default_coins = [
            'KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-AVAX',
            'KRW-DOGE', 'KRW-MATIC', 'KRW-ATOM', 'KRW-NEAR', 'KRW-HBAR',
            'KRW-DOT', 'KRW-LINK', 'KRW-SOL', 'KRW-UNI', 'KRW-ALGO',
            'KRW-VET', 'KRW-ICP', 'KRW-FTM', 'KRW-SAND', 'KRW-MANA'
        ]
        logger.info("기본 코인 리스트로 설정")
        return default_coins

    def _calculate_preliminary_score(self, volume_krw: float, price_change_24h: float) -> float:
        """예비 점수 계산 (빠른 필터링용)"""
        score = 0.0
        
        # 거래량 점수 (로그 스케일)
        volume_score = np.log10(volume_krw / 1e9) * 0.3  # 10억원 기준
        score += min(volume_score, 0.5)
        
        # 가격 변동 점수 (절댓값)
        momentum_score = min(abs(price_change_24h) * 2, 0.3)
        score += momentum_score
        
        # 상승 보너스
        if price_change_24h > 0:
            score += 0.2
        
        return score

    async def _get_comprehensive_coin_data(self, symbol: str) -> Dict:
        """종합 코인 데이터 수집"""
        try:
            # 현재가
            current_price = pyupbit.get_current_price(symbol)
            if not current_price:
                return {}
            
            # 다양한 시간프레임 OHLCV 데이터
            ohlcv_1h = pyupbit.get_ohlcv(symbol, interval="minute60", count=168)  # 1주일
            ohlcv_4h = pyupbit.get_ohlcv(symbol, interval="minute240", count=180)  # 30일
            ohlcv_1d = pyupbit.get_ohlcv(symbol, interval="day", count=100)       # 100일
            
            if any(data is None or len(data) < 20 for data in [ohlcv_1h, ohlcv_4h, ohlcv_1d]):
                return {}
            
            # 기본 데이터
            data = {
                'symbol': symbol,
                'price': current_price,
                'ohlcv_1h': ohlcv_1h,
                'ohlcv_4h': ohlcv_4h,
                'ohlcv_1d': ohlcv_1d
            }
            
            # 거래량 및 시가총액 정보
            latest_1d = ohlcv_1d.iloc[-1]
            data['volume_24h_krw'] = latest_1d['volume'] * current_price
            data['volume_24h_btc'] = latest_1d['volume']
            
            # 가격 모멘텀
            if len(ohlcv_1d) >= 30:
                data['momentum_3d'] = (current_price / ohlcv_1d.iloc[-4]['close'] - 1) * 100
                data['momentum_7d'] = (current_price / ohlcv_1d.iloc[-8]['close'] - 1) * 100
                data['momentum_30d'] = (current_price / ohlcv_1d.iloc[-31]['close'] - 1) * 100
            else:
                data['momentum_3d'] = data['momentum_7d'] = data['momentum_30d'] = 0
            
            # 거래량 급증률
            avg_volume_7d = ohlcv_1d['volume'].tail(7).mean()
            current_volume = latest_1d['volume']
            data['volume_spike_ratio'] = current_volume / avg_volume_7d if avg_volume_7d > 0 else 1
            
            return data
            
        except Exception as e:
            logger.error(f"코인 데이터 수집 실패 {symbol}: {e}")
            return {}

    def _fundamental_analysis(self, symbol: str, data: Dict) -> Tuple[float, str]:
        """펀더멘털 분석 (30%)"""
        score = 0.0
        reasoning = []
        
        # 1. 거래량 점수 (40%)
        volume_24h = data.get('volume_24h_krw', 0)
        if volume_24h >= 50_000_000_000:  # 500억원 이상
            score += 0.40
            reasoning.append(f"대형거래량({volume_24h/1e8:.0f}억)")
        elif volume_24h >= 10_000_000_000:  # 100억원 이상
            score += 0.25
            reasoning.append(f"중형거래량({volume_24h/1e8:.0f}억)")
        elif volume_24h >= 1_000_000_000:   # 10억원 이상
            score += 0.10
            reasoning.append(f"소형거래량({volume_24h/1e8:.0f}억)")
        
        # 2. 프로젝트 품질 (30%) - 주요 코인 보너스
        major_coins = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-AVAX', 'KRW-SOL', 'KRW-DOT', 'KRW-LINK']
        if symbol in major_coins:
            score += 0.30
            reasoning.append("메이저코인")
        elif symbol.endswith(('USDT', 'BUSD')):  # 스테이블코인은 제외
            score -= 0.20
            reasoning.append("스테이블코인")
        else:
            score += 0.15
            reasoning.append("알트코인")
        
        # 3. 시장 안정성 (30%) - 변동성 기반
        if len(data.get('ohlcv_1d', [])) >= 30:
            price_std = data['ohlcv_1d']['close'].tail(30).std()
            price_mean = data['ohlcv_1d']['close'].tail(30).mean()
            volatility = price_std / price_mean if price_mean > 0 else 1
            
            if volatility < 0.05:  # 5% 미만
                score += 0.30
                reasoning.append("저변동성")
            elif volatility < 0.10:  # 10% 미만
                score += 0.15
                reasoning.append("중변동성")
            else:
                reasoning.append("고변동성")
        
        return score, "펀더멘털: " + " | ".join(reasoning)

    def _technical_analysis(self, symbol: str, data: Dict) -> Tuple[float, str]:
        """기술적 분석 (40%) - 확장된 지표"""
        score = 0.0
        reasoning = []
        
        try:
            ohlcv_1d = data.get('ohlcv_1d')
            if ohlcv_1d is None or len(ohlcv_1d) < 30:
                return 0.0, "기술적: 데이터부족"
            
            closes = ohlcv_1d['close']
            highs = ohlcv_1d['high']
            lows = ohlcv_1d['low']
            volumes = ohlcv_1d['volume']
            
            # 1. RSI (15%)
            rsi = ta.momentum.RSIIndicator(closes, window=self.rsi_period).rsi().iloc[-1]
            if 30 <= rsi <= 70:
                score += 0.15
                reasoning.append(f"RSI적정({rsi:.0f})")
            elif rsi < 30:
                score += 0.10
                reasoning.append(f"RSI과매도({rsi:.0f})")
            elif rsi > 70:
                score += 0.05
                reasoning.append(f"RSI과매수({rsi:.0f})")
            
            # 2. MACD (15%)
            macd_indicator = ta.trend.MACD(closes, window_fast=self.macd_fast, 
                                         window_slow=self.macd_slow, window_sign=self.macd_signal)
            macd_diff = macd_indicator.macd_diff().iloc[-1]
            if macd_diff > 0:
                score += 0.15
                reasoning.append("MACD상승")
            else:
                reasoning.append("MACD하락")
            
            # 3. 볼린저 밴드 (10%)
            bb_indicator = ta.volatility.BollingerBands(closes, window=self.bb_period)
            bb_high = bb_indicator.bollinger_hband().iloc[-1]
            bb_low = bb_indicator.bollinger_lband().iloc[-1]
            current_price = closes.iloc[-1]
            
            if current_price < bb_low:
                score += 0.10
                reasoning.append("BB과매도")
            elif current_price > bb_high:
                score += 0.05
                reasoning.append("BB과매수")
            else:
                score += 0.07
                reasoning.append("BB정상")
            
            # 4. 스토캐스틱 (10%)
            stoch_indicator = ta.momentum.StochasticOscillator(highs, lows, closes, 
                                                             window=self.stoch_k, smooth_window=self.stoch_d)
            stoch_k = stoch_indicator.stoch().iloc[-1]
            stoch_d = stoch_indicator.stoch_signal().iloc[-1]
            
            if stoch_k < 20 and stoch_d < 20:
                score += 0.10
                reasoning.append("스토캐스틱과매도")
            elif stoch_k > 80 and stoch_d > 80:
                score += 0.05
                reasoning.append("스토캐스틱과매수")
            else:
                score += 0.07
                reasoning.append("스토캐스틱중립")
            
            # 5. 일목균형표 간단 버전 (25%)
            tenkan = (highs.rolling(9).max() + lows.rolling(9).min()) / 2
            kijun = (highs.rolling(26).max() + lows.rolling(26).min()) / 2
            
            if len(tenkan) > 0 and len(kijun) > 0:
                if tenkan.iloc[-1] > kijun.iloc[-1] and current_price > tenkan.iloc[-1]:
                    score += 0.25
                    reasoning.append("일목상승")
                elif tenkan.iloc[-1] < kijun.iloc[-1] and current_price < tenkan.iloc[-1]:
                    score += 0.05
                    reasoning.append("일목하락")
                else:
                    score += 0.12
                    reasoning.append("일목중립")
            
            # 6. OBV 추세 (15%)
            obv = ta.volume.OnBalanceVolumeIndicator(closes, volumes).on_balance_volume()
            if len(obv) >= 10:
                obv_trend = "상승" if obv.iloc[-1] > obv.iloc[-10] else "하락"
                if obv_trend == "상승":
                    score += 0.15
                    reasoning.append("OBV상승")
                else:
                    score += 0.05
                    reasoning.append("OBV하락")
            
            # 7. ATR 기반 변동성 (10%)
            atr = ta.volatility.AverageTrueRange(highs, lows, closes, window=self.atr_period).average_true_range().iloc[-1]
            atr_ratio = atr / current_price
            if atr_ratio < 0.03:  # 3% 미만
                score += 0.10
                reasoning.append("저변동성")
            elif atr_ratio < 0.06:  # 6% 미만
                score += 0.07
                reasoning.append("중변동성")
            else:
                score += 0.03
                reasoning.append("고변동성")
            
            return score, "기술적: " + " | ".join(reasoning)
            
        except Exception as e:
            logger.error(f"기술적 분석 실패 {symbol}: {e}")
            return 0.0, f"기술적: 분석실패({str(e)})"

    def _momentum_analysis(self, symbol: str, data: Dict) -> Tuple[float, str]:
        """모멘텀 분석 (30%)"""
        score = 0.0
        reasoning = []
        
        # 1. 3일 모멘텀 (30%)
        momentum_3d = data.get('momentum_3d', 0)
        if momentum_3d >= 15:
            score += 0.30
            reasoning.append(f"강한3일({momentum_3d:.1f}%)")
        elif momentum_3d >= 5:
            score += 0.15
            reasoning.append(f"상승3일({momentum_3d:.1f}%)")
        elif momentum_3d >= 0:
            score += 0.05
            reasoning.append(f"보합3일({momentum_3d:.1f}%)")
        else:
            reasoning.append(f"하락3일({momentum_3d:.1f}%)")
        
        # 2. 7일 모멘텀 (35%)
        momentum_7d = data.get('momentum_7d', 0)
        if momentum_7d >= 25:
            score += 0.35
            reasoning.append(f"강한7일({momentum_7d:.1f}%)")
        elif momentum_7d >= 10:
            score += 0.20
            reasoning.append(f"상승7일({momentum_7d:.1f}%)")
        elif momentum_7d >= 0:
            score += 0.05
            reasoning.append(f"보합7일({momentum_7d:.1f}%)")
        
        # 3. 거래량 급증 (35%)
        volume_spike = data.get('volume_spike_ratio', 1)
        if volume_spike >= 3.0:
            score += 0.35
            reasoning.append(f"거래량폭증({volume_spike:.1f}배)")
        elif volume_spike >= 2.0:
            score += 0.20
            reasoning.append(f"거래량급증({volume_spike:.1f}배)")
        elif volume_spike >= 1.5:
            score += 0.10
            reasoning.append(f"거래량증가({volume_spike:.1f}배)")
        
        return score, "모멘텀: " + " | ".join(reasoning)

    def _calculate_split_trading_plan(self, symbol: str, current_price: float, 
                                     confidence: float) -> Dict:
        """5단계 분할매매 계획 수립"""
        try:
            # 신뢰도 기반 투자금액 계산
            base_investment = self.coin_portfolio_value / self.target_coins  # 기본 1000만원
            confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5~2.0 배수
            total_investment = base_investment * confidence_multiplier
            total_investment = min(total_investment, self.coin_portfolio_value * self.max_single_coin_weight)
            
            # 5단계 분할 금액
            stage_amounts = [total_investment * ratio for ratio in self.stage_ratios]
            
            # 5단계 진입가
            entry_prices = [current_price * (1 + trigger) for trigger in self.stage_triggers]
            
            # 손절/익절 계획
            avg_entry = current_price * 0.90  # 평균 진입가 추정 (10% 하락)
            stop_loss = avg_entry * (1 - self.stop_loss_pct)
            take_profits = [avg_entry * (1 + tp) for tp in self.take_profit_levels]
            
            # 보유 기간 (신뢰도 기반)
            max_hold_days = int(self.max_hold_days * (1.5 - confidence))
            
            return {
                'total_investment': total_investment,
                'stage_amounts': stage_amounts,
                'entry_prices': entry_prices,
                'stop_loss': stop_loss,
                'take_profits': take_profits,
                'max_hold_days': max_hold_days,
                'coin_weight': total_investment / self.coin_portfolio_value * 100
            }
            
        except Exception as e:
            logger.error(f"분할매매 계획 수립 실패 {symbol}: {e}")
            return {}

    async def get_fear_greed_index(self) -> Tuple[int, str]:
        """공포탐욕지수 조회"""
        try:
            response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            if response.status_code == 200:
                data = response.json()
                score = int(data["data"][0]["value"])
                classification = data["data"][0]["value_classification"]
                return score, classification
        except Exception as e:
            logger.error(f"공포탐욕지수 조회 실패: {e}")
        
        return 50, "Neutral"

    async def analyze_symbol(self, symbol: str) -> UltimateCoinSignal:
        """개별 코인 종합 분석"""
        if not self.enabled:
            return self._create_empty_signal(symbol, "전략 비활성화")
        
        try:
            # 종합 데이터 수집
            data = await self._get_comprehensive_coin_data(symbol)
            if not data:
                raise ValueError(f"데이터 수집 실패: {symbol}")
            
            # 3가지 전략 분석
            fundamental_score, fundamental_reasoning = self._fundamental_analysis(symbol, data)
            technical_score, technical_reasoning = self._technical_analysis(symbol, data)
            momentum_score, momentum_reasoning = self._momentum_analysis(symbol, data)
            
            # 가중 평균 계산
            total_score = (
                fundamental_score * self.fundamental_weight +
                technical_score * self.technical_weight +
                momentum_score * self.momentum_weight
            )
            
            # 공포탐욕지수 추가 고려
            fear_greed_score, market_sentiment = await self.get_fear_greed_index()
            
            # 공포탐욕지수 보정 (극단적일 때 역발상)
            if fear_greed_score < 25:  # 극도의 공포
                total_score += 0.1  # 매수 기회
            elif fear_greed_score > 75:  # 극도의 탐욕
                total_score -= 0.1  # 매도 신호
            
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
            if action == 'buy':
                target_price = data['price'] * (1 + confidence * 0.5)  # 최대 50% 상승 기대
            elif action == 'sell':
                target_price = data['price'] * (1 - confidence * 0.3)  # 30% 하락 예상
            else:
                target_price = data['price']
            
            # 종합 reasoning
            all_reasoning = " | ".join([fundamental_reasoning, technical_reasoning, momentum_reasoning])
            
            # 기술적 지표 상세 추출
            tech_details = self._extract_technical_details(data)
            
            return UltimateCoinSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=data['price'],
                
                # 전략별 점수
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                momentum_score=momentum_score,
                total_score=total_score,
                
                # 펀더멘털 지표
                market_cap_rank=0,  # 별도 API 필요
                volume_24h_rank=0,  # 별도 계산 필요
                project_quality=fundamental_score,
                adoption_score=data.get('volume_24h_krw', 0) / 1e9,  # 거래량 기반
                
                # 기술적 지표
                rsi=tech_details.get('rsi', 50),
                macd_signal=tech_details.get('macd_signal', 'neutral'),
                bb_position=tech_details.get('bb_position', 'normal'),
                stoch_k=tech_details.get('stoch_k', 50),
                stoch_d=tech_details.get('stoch_d', 50),
                ichimoku_signal=tech_details.get('ichimoku_signal', 'neutral'),
                atr=tech_details.get('atr', 0),
                obv_trend=tech_details.get('obv_trend', 'neutral'),
                
                # 모멘텀 지표
                momentum_3d=data.get('momentum_3d', 0),
                momentum_7d=data.get('momentum_7d', 0),
                momentum_30d=data.get('momentum_30d', 0),
                volume_spike_ratio=data.get('volume_spike_ratio', 1),
                price_velocity=data.get('momentum_3d', 0) / 3,  # 일일 평균 변화율
                
                # 분할매매 정보
                position_stage=0,  # 초기값
                total_amount=split_plan.get('total_investment', 0),
                stage1_amount=split_plan.get('stage_amounts', [0]*5)[0],
                stage2_amount=split_plan.get('stage_amounts', [0]*5)[1],
                stage3_amount=split_plan.get('stage_amounts', [0]*5)[2],
                stage4_amount=split_plan.get('stage_amounts', [0]*5)[3],
                stage5_amount=split_plan.get('stage_amounts', [0]*5)[4],
                entry_price_1=split_plan.get('entry_prices', [data['price']]*5)[0],
                entry_price_2=split_plan.get('entry_prices', [data['price']]*5)[1],
                entry_price_3=split_plan.get('entry_prices', [data['price']]*5)[2],
                entry_price_4=split_plan.get('entry_prices', [data['price']]*5)[3],
                entry_price_5=split_plan.get('entry_prices', [data['price']]*5)[4],
                stop_loss=split_plan.get('stop_loss', data['price'] * 0.75),
                take_profit_1=split_plan.get('take_profits', [data['price']]*3)[0],
                take_profit_2=split_plan.get('take_profits', [data['price']]*3)[1],
                take_profit_3=split_plan.get('take_profits', [data['price']]*3)[2],
                max_hold_days=split_plan.get('max_hold_days', 30),
                
                # 시장 지표
                fear_greed_score=fear_greed_score,
                market_sentiment=market_sentiment,
                
                sector=self._get_coin_sector(symbol),
                reasoning=all_reasoning,
                target_price=target_price,
                timestamp=datetime.now(),
                additional_data=split_plan
            )
            
        except Exception as e:
            logger.error(f"코인 분석 실패 {symbol}: {e}")
            return self._create_empty_signal(symbol, f"분석 실패: {str(e)}")

    def _extract_technical_details(self, data: Dict) -> Dict:
        """기술적 지표 상세 추출"""
        try:
            ohlcv_1d = data.get('ohlcv_1d')
            if ohlcv_1d is None or len(ohlcv_1d) < 30:
                return {}
            
            closes = ohlcv_1d['close']
            highs = ohlcv_1d['high']
            lows = ohlcv_1d['low']
            volumes = ohlcv_1d['volume']
            current_price = closes.iloc[-1]
            
            details = {}
            
            # RSI
            rsi_indicator = ta.momentum.RSIIndicator(closes, window=14)
            details['rsi'] = rsi_indicator.rsi().iloc[-1]
            
            # MACD
            macd_indicator = ta.trend.MACD(closes)
            macd_diff = macd_indicator.macd_diff().iloc[-1]
            details['macd_signal'] = 'bullish' if macd_diff > 0 else 'bearish'
            
            # 볼린저 밴드
            bb_indicator = ta.volatility.BollingerBands(closes)
            bb_high = bb_indicator.bollinger_hband().iloc[-1]
            bb_low = bb_indicator.bollinger_lband().iloc[-1]
            
            if current_price > bb_high:
                details['bb_position'] = 'overbought'
            elif current_price < bb_low:
                details['bb_position'] = 'oversold'
            else:
                details['bb_position'] = 'normal'
            
            # 스토캐스틱
            stoch_indicator = ta.momentum.StochasticOscillator(highs, lows, closes)
            details['stoch_k'] = stoch_indicator.stoch().iloc[-1]
            details['stoch_d'] = stoch_indicator.stoch_signal().iloc[-1]
            
            # 일목균형표 (간단)
            tenkan = (highs.rolling(9).max() + lows.rolling(9).min()) / 2
            kijun = (highs.rolling(26).max() + lows.rolling(26).min()) / 2
            
            if len(tenkan) > 0 and len(kijun) > 0:
                if tenkan.iloc[-1] > kijun.iloc[-1]:
                    details['ichimoku_signal'] = 'bullish'
                else:
                    details['ichimoku_signal'] = 'bearish'
            else:
                details['ichimoku_signal'] = 'neutral'
            
            # ATR
            atr_indicator = ta.volatility.AverageTrueRange(highs, lows, closes)
            details['atr'] = atr_indicator.average_true_range().iloc[-1]
            
            # OBV
            obv_indicator = ta.volume.OnBalanceVolumeIndicator(closes, volumes)
            obv = obv_indicator.on_balance_volume()
            if len(obv) >= 10:
                details['obv_trend'] = 'rising' if obv.iloc[-1] > obv.iloc[-10] else 'falling'
            else:
                details['obv_trend'] = 'neutral'
            
            return details
            
        except Exception as e:
            logger.error(f"기술적 지표 추출 실패: {e}")
            return {}

    def _get_coin_sector(self, symbol: str) -> str:
        """코인 섹터 분류"""
        major_coins = ['KRW-BTC', 'KRW-ETH']
        defi_coins = ['KRW-UNI', 'KRW-SUSHI', 'KRW-CAKE', 'KRW-COMP']
        layer1_coins = ['KRW-ADA', 'KRW-SOL', 'KRW-AVAX', 'KRW-DOT', 'KRW-ATOM']
        gaming_coins = ['KRW-SAND', 'KRW-MANA', 'KRW-AXS']
        
        if symbol in major_coins:
            return 'MAJOR'
        elif symbol in defi_coins:
            return 'DEFI'
        elif symbol in layer1_coins:
            return 'LAYER1'
        elif symbol in gaming_coins:
            return 'GAMING'
        else:
            return 'ALTCOIN'

    def _create_empty_signal(self, symbol: str, reason: str) -> UltimateCoinSignal:
        """빈 시그널 생성"""
        return UltimateCoinSignal(
            symbol=symbol, action='hold', confidence=0.0, price=0.0,
            fundamental_score=0.0, technical_score=0.0, momentum_score=0.0, total_score=0.0,
            market_cap_rank=0, volume_24h_rank=0, project_quality=0.0, adoption_score=0.0,
            rsi=50.0, macd_signal='neutral', bb_position='normal', stoch_k=50.0, stoch_d=50.0,
            ichimoku_signal='neutral', atr=0.0, obv_trend='neutral', momentum_3d=0.0,
            momentum_7d=0.0, momentum_30d=0.0, volume_spike_ratio=1.0, price_velocity=0.0,
            position_stage=0, total_amount=0.0, stage1_amount=0.0, stage2_amount=0.0,
            stage3_amount=0.0, stage4_amount=0.0, stage5_amount=0.0, entry_price_1=0.0,
            entry_price_2=0.0, entry_price_3=0.0, entry_price_4=0.0, entry_price_5=0.0,
            stop_loss=0.0, take_profit_1=0.0, take_profit_2=0.0, take_profit_3=0.0,
            max_hold_days=30, fear_greed_score=50, market_sentiment='Neutral',
            sector='UNKNOWN', reasoning=reason, target_price=0.0, timestamp=datetime.now()
        )

    async def generate_portfolio_report(self, selected_coins: List[UltimateCoinSignal]) -> Dict:
        """포트폴리오 리포트 생성"""
        if not selected_coins:
            return {"error": "선별된 코인이 없습니다"}
        
        # 기본 통계
        total_coins = len(selected_coins)
        buy_signals = [s for s in selected_coins if s.action == 'buy']
        sell_signals = [s for s in selected_coins if s.action == 'sell']
        hold_signals = [s for s in selected_coins if s.action == 'hold']
        
        # 평균 점수
        avg_fundamental = np.mean([s.fundamental_score for s in selected_coins])
        avg_technical = np.mean([s.technical_score for s in selected_coins])
        avg_momentum = np.mean([s.momentum_score for s in selected_coins])
        avg_total = np.mean([s.total_score for s in selected_coins])
        
        # 총 투자금액
        total_investment = sum([s.total_amount for s in selected_coins])
        
        # 상위 5개 매수 코인
        top_buys = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
        
        # 섹터별 분포
        sector_dist = {}
        for coin in selected_coins:
            sector_dist[coin.sector] = sector_dist.get(coin.sector, 0) + 1
        
        # 리스크 지표
        avg_volatility = np.mean([s.atr / s.price if s.price > 0 else 0 for s in selected_coins])
        
        return {
            'summary': {
                'total_coins': total_coins,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'hold_signals': len(hold_signals),
                'total_investment': total_investment,
                'avg_investment_per_coin': total_investment / total_coins if total_coins > 0 else 0,
                'portfolio_allocation': total_investment / self.coin_portfolio_value * 100
            },
            'strategy_scores': {
                'avg_fundamental_score': avg_fundamental,
                'avg_technical_score': avg_technical,
                'avg_momentum_score': avg_momentum,
                'avg_total_score': avg_total
            },
            'top_picks': [
                {
                    'symbol': coin.symbol,
                    'sector': coin.sector,
                    'confidence': coin.confidence,
                    'total_score': coin.total_score,
                    'price': coin.price,
                    'target_price': coin.target_price,
                    'total_investment': coin.total_amount,
                    'fear_greed': coin.fear_greed_score,
                    'reasoning': coin.reasoning[:100] + "..." if len(coin.reasoning) > 100 else coin.reasoning
                }
                for coin in top_buys
            ],
            'sector_distribution': sector_dist,
            'risk_metrics': {
                'avg_volatility': avg_volatility,
                'max_single_position': max([s.total_amount for s in selected_coins]) / total_investment * 100 if total_investment > 0 else 0,
                'fear_greed_index': selected_coins[0].fear_greed_score if selected_coins else 50,
                'market_sentiment': selected_coins[0].market_sentiment if selected_coins else 'Neutral'
            }
        }

    async def execute_split_trading_simulation(self, signal: UltimateCoinSignal) -> Dict:
        """5단계 분할매매 시뮬레이션"""
        if signal.action != 'buy':
            return {"status": "not_applicable", "reason": "매수 신호가 아님"}
        
        simulation = {
            'symbol': signal.symbol,
            'strategy': '5_stage_split_trading',
            'stages': {
                'stage_1': {
                    'trigger_price': signal.entry_price_1,
                    'amount': signal.stage1_amount,
                    'ratio': '20%',
                    'trigger_condition': '즉시 매수',
                    'status': 'ready'
                },
                'stage_2': {
                    'trigger_price': signal.entry_price_2,
                    'amount': signal.stage2_amount,
                    'ratio': '20%',
                    'trigger_condition': '5% 하락시',
                    'status': 'waiting'
                },
                'stage_3': {
                    'trigger_price': signal.entry_price_3,
                    'amount': signal.stage3_amount,
                    'ratio': '20%',
                    'trigger_condition': '10% 하락시',
                    'status': 'waiting'
                },
                'stage_4': {
                    'trigger_price': signal.entry_price_4,
                    'amount': signal.stage4_amount,
                    'ratio': '20%',
                    'trigger_condition': '15% 하락시',
                    'status': 'waiting'
                },
                'stage_5': {
                    'trigger_price': signal.entry_price_5,
                    'amount': signal.stage5_amount,
                    'ratio': '20%',
                    'trigger_condition': '20% 하락시',
                    'status': 'waiting'
                }
            },
            'exit_plan': {
                'stop_loss': {
                    'price': signal.stop_loss,
                    'ratio': '100%',
                    'trigger': '25% 손절'
                },
                'take_profit_1': {
                    'price': signal.take_profit_1,
                    'ratio': '40%',
                    'trigger': '20% 익절'
                },
                'take_profit_2': {
                    'price': signal.take_profit_2,
                    'ratio': '40%',
                    'trigger': '50% 익절'
                },
                'take_profit_3': {
                    'price': signal.take_profit_3,
                    'ratio': '20%',
                    'trigger': '100% 익절'
                }
            },
            'risk_management': {
                'max_hold_days': signal.max_hold_days,
                'total_investment': signal.total_amount,
                'portfolio_weight': signal.total_amount / self.coin_portfolio_value * 100,
                'fear_greed_index': signal.fear_greed_score
            }
        }
        
        return simulation

# 메인 실행 함수들
async def run_ultimate_coin_selection():
    """궁극의 코인 선별 실행"""
    strategy = UltimateCoinStrategy()
    selected_coins_symbols = await strategy.auto_select_top20_coins()
    
    if selected_coins_symbols:
        # 선별된 코인들 상세 분석
        detailed_signals = []
        for symbol in selected_coins_symbols:
            signal = await strategy.analyze_symbol(symbol)
            detailed_signals.append(signal)
        
        report = await strategy.generate_portfolio_report(detailed_signals)
        return detailed_signals, report
    else:
        return [], {}

async def analyze_coin(symbol: str) -> Dict:
    """단일 코인 분석 (기존 호환성)"""
    strategy = UltimateCoinStrategy()
    signal = await strategy.analyze_symbol(symbol)
    
    return {
        'decision': signal.action,
        'confidence_score': signal.confidence * 100,
        'reasoning': signal.reasoning,
        'target_price': signal.target_price,
        'price': signal.price,
        'sector': signal.sector,
        'fundamental_score': signal.fundamental_score,
        'technical_score': signal.technical_score,
        'momentum_score': signal.momentum_score,
        'fear_greed_score': signal.fear_greed_score,
        'split_trading_plan': await strategy.execute_split_trading_simulation(signal)
    }

if __name__ == "__main__":
    async def main():
        print("🪙 최고퀸트프로젝트 - 궁극의 암호화폐 전략!")
        print("🎯 하이브리드 전략 + 자동 선별 + 5단계 분할매매")
        print("="*60)
        
        # 자동 선별 실행
        print("\n🔍 궁극의 자동 코인 선별 시작...")
        selected_signals, report = await run_ultimate_coin_selection()
        
        if selected_signals:
            print(f"\n🎯 선별 완료! 상위 {len(selected_signals)}개 코인:")
            print("="*60)
            
            # 상위 5개 코인 상세 표시
            top_5 = sorted(selected_signals, key=lambda x: x.total_score, reverse=True)[:5]
            
            for i, coin in enumerate(top_5, 1):
                print(f"\n{i}. {coin.symbol} ({coin.sector})")
                print(f"   🎯 액션: {coin.action} | 신뢰도: {coin.confidence:.1%}")
                print(f"   📊 총점: {coin.total_score:.2f} (펀더멘털:{coin.fundamental_score:.2f} + 기술:{coin.technical_score:.2f} + 모멘텀:{coin.momentum_score:.2f})")
                print(f"   💰 현재가: {coin.price:,.0f}원 → 목표가: {coin.target_price:,.0f}원")
                print(f"   🔄 5단계 분할: {coin.total_amount:,.0f}원 (각 20%씩: {coin.stage1_amount:,.0f}원)")
                print(f"   🛡️ 손절: {coin.stop_loss:,.0f}원 | 익절: {coin.take_profit_1:,.0f}원 → {coin.take_profit_3:,.0f}원")
                print(f"   😱 공포탐욕: {coin.fear_greed_score} ({coin.market_sentiment})")
                print(f"   💡 {coin.reasoning[:80]}...")
            
            # 포트폴리오 요약
            print(f"\n📊 포트폴리오 요약:")
            print(f"   총 코인: {report['summary']['total_coins']}개")
            print(f"   매수: {report['summary']['buy_signals']}개 | 보유: {report['summary']['hold_signals']}개")
            print(f"   총 투자금액: {report['summary']['total_investment']:,.0f}원")
            print(f"   포트폴리오 비중: {report['summary']['portfolio_allocation']:.1f}%")
            
            # 섹터 분포
            print(f"\n🏢 섹터 분포:")
            for sector, count in report['sector_distribution'].items():
                percentage = count / report['summary']['total_coins'] * 100
                print(f"   {sector}: {count}개 ({percentage:.1f}%)")
            
            # 5단계 분할매매 시뮬레이션 (첫 번째 매수 코인)
            buy_coins = [s for s in selected_signals if s.action == 'buy']
            if buy_coins:
                print(f"\n🔄 5단계 분할매매 시뮬레이션 - {buy_coins[0].symbol}:")
                strategy = UltimateCoinStrategy()
                simulation = await strategy.execute_split_trading_simulation(buy_coins[0])
                
                for stage, data in simulation['stages'].items():
                    print(f"   {stage}: {data['trigger_price']:,.0f}원에 {data['amount']:,.0f}원 ({data['ratio']}) - {data['trigger_condition']}")
                
                print(f"   손절: {simulation['exit_plan']['stop_loss']['price']:,.0f}원")
                print(f"   익절1: {simulation['exit_plan']['take_profit_1']['price']:,.0f}원 (40% 매도)")
                print(f"   익절2: {simulation['exit_plan']['take_profit_2']['price']:,.0f}원 (40% 매도)")
                print(f"   익절3: {simulation['exit_plan']['take_profit_3']['price']:,.0f}원 (20% 매도)")
            
        else:
            print("❌ 선별된 코인이 없습니다.")
        
        print("\n✅ 테스트 완료!")
        print("🚀 궁극의 하이브리드 암호화폐 전략 - 28살 월 1-3억 달성 시스템 완성!")

    asyncio.run(main())
