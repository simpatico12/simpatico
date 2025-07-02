#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🪙 최고퀸트프로젝트 - 궁극의 암호화폐 전략 (완전체 V5.0)
================================================================================
혼자 유지보수 가능한 완전체 시스템:
- 🔧 4개 설정파일 완벽 연동 (.env.example, .gitignore, requirements.txt, settings.yaml)
- 🤖 AI 품질평가 + 시장사이클 감지 + 6단계 정밀 필터링
- 📊 고급 기술지표 (Williams %R, ADX, 일목균형표, CCI, MFI 등)
- 💰 정밀한 5단계 분할매매 + 상관관계 포트폴리오 최적화
- 🛠️ 간소화된 구조 + 상세한 로깅으로 디버깅 용이

Author: 최고퀸트팀 | Version: 5.0 (완전체)
================================================================================
"""

import asyncio
import logging
import os
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests
import pyupbit
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# ============================================================================
# 📁 4개 파일 연동 설정 로더
# ============================================================================
class ConfigManager:
    """설정 관리자 - 4개 파일 통합 관리"""
    
    def __init__(self):
        load_dotenv()  # .env 파일 로드
        self.config = self._load_yaml_config()
        self._setup_logging()
        
    def _load_yaml_config(self) -> Dict:
        """settings.yaml 로드"""
        try:
            with open('settings.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ settings.yaml 로드 성공")
            return config
        except Exception as e:
            print(f"⚠️ settings.yaml 로드 실패: {e}")
            return self._get_default_config()
    
    def _setup_logging(self):
        """로깅 설정"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/coin_strategy.log', encoding='utf-8')
            ]
        )
        os.makedirs('logs', exist_ok=True)
    
    def get_coin_config(self) -> Dict:
        """코인 전략 설정 반환"""
        return self.config.get('coin_strategy', {})
    
    def get_api_key(self, key_name: str) -> str:
        """환경변수에서 API 키 조회"""
        return os.getenv(key_name, '')
    
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            'coin_strategy': {
                'enabled': True,
                'target_coins': 20,
                'min_volume_24h': 500000000,
                'portfolio_value': 200000000
            }

# ============================================================================
# 🎯 메인 실행 함수들
# ============================================================================

async def main():
    """메인 실행 함수"""
    print("🪙 최고퀸트프로젝트 - 궁극의 암호화폐 전략 (완전체 V5.0)")
    print("🔧 6단계정밀필터링 + 고급기술지표 + 5단계분할매매 + 상관관계최적화")
    print("="*80)
    
    # 환경 확인
    if not os.path.exists('settings.yaml'):
        print("⚠️ settings.yaml 파일이 없습니다. 기본 설정으로 진행합니다.")
    
    # 전략 정보 출력
    info = get_ultimate_strategy_info()
    print(f"\n📋 궁극의 전략 정보:")
    print(f"   이름: {info['name']}")
    print(f"   버전: {info['version']}")
    print(f"   주요 기능: {len(info['features'])}개")
    print(f"   기술지표: 기본 {len(info['technical_indicators']['basic'])}개 + "
          f"고급 {len(info['technical_indicators']['advanced'])}개")
    print(f"   필터링: {len(info['filtering_stages'])}단계")
    print(f"   분할매매: {info['split_trading']['stages']}단계")
    
    # 완전체 테스트 실행
    await test_ultimate_strategy()

def run_single_coin_analysis(symbol: str):
    """단일 코인 분석 실행 (동기 버전)"""
    return asyncio.run(analyze_single_coin(symbol))

def run_full_strategy():
    """전체 전략 실행 (동기 버전)"""
    return asyncio.run(run_ultimate_analysis())

def run_test():
    """테스트 실행 (동기 버전)"""
    return asyncio.run(test_ultimate_strategy())

# ============================================================================
# 📱 사용 예시 함수들
# ============================================================================

def example_usage():
    """사용 예시"""
    print("\n📖 사용 예시:")
    print("="*50)
    
    print("\n1️⃣ 단일 코인 분석:")
    print("python -c \"from coin_strategy import run_single_coin_analysis; "
          "print(run_single_coin_analysis('KRW-BTC'))\"")
    
    print("\n2️⃣ 전체 전략 실행:")
    print("python -c \"from coin_strategy import run_full_strategy; "
          "signals, report = run_full_strategy(); "
          "print(f'매수 신호: {len([s for s in signals if s.action == \\\"buy\\\"])}개')\"")
    
    print("\n3️⃣ 테스트 실행:")
    print("python -c \"from coin_strategy import run_test; run_test()\"")
    
    print("\n4️⃣ 메인 실행:")
    print("python coin_strategy.py")

if __name__ == "__main__":
    # 로그 폴더 생성
    os.makedirs('logs', exist_ok=True)
    
    # 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 궁극의 전략을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n📖 다음에도 최고퀸트프로젝트를 이용해 주세요!")
        example_usage()
        }

# ============================================================================
# 📊 확장된 데이터 클래스
# ============================================================================
@dataclass
class UltimateCoinSignal:
    """완전체 코인 시그널"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    total_score: float
    
    # 핵심 지표들
    fundamental_score: float
    technical_score: float
    momentum_score: float
    quality_score: float
    
    # 시장 정보
    market_cycle: str
    cycle_confidence: float
    sector: str
    
    # 🆕 고급 기술지표
    rsi: float
    williams_r: float
    cci: float
    mfi: float
    adx: float
    macd_signal: str
    bb_position: str
    ichimoku_signal: str
    
    # 🆕 모멘텀 지표
    momentum_3d: float
    momentum_7d: float
    momentum_30d: float
    volume_spike_ratio: float
    
    # 🆕 상관관계 분석
    btc_correlation: float
    portfolio_fit_score: float
    diversification_benefit: float
    
    # 🆕 정밀한 5단계 분할매매
    position_stage: int
    total_investment: float
    stage1_amount: float  # 20%
    stage2_amount: float  # 20%
    stage3_amount: float  # 20%
    stage4_amount: float  # 20%
    stage5_amount: float  # 20%
    entry_price_1: float  # 현재가
    entry_price_2: float  # -5%
    entry_price_3: float  # -10%
    entry_price_4: float  # -15%
    entry_price_5: float  # -20%
    stop_loss: float      # -25%
    take_profit_1: float  # +20%
    take_profit_2: float  # +50%
    take_profit_3: float  # +100%
    
    reasoning: str
    timestamp: datetime

# ============================================================================
# 🤖 AI 품질 평가기 (강화)
# ============================================================================
class EnhancedProjectQualityAnalyzer:
    """AI 기반 프로젝트 품질 평가 (강화 버전)"""
    
    def __init__(self):
        self.tier_database = {
            'tier_1': {'coins': ['BTC', 'ETH'], 'base_score': 0.95},
            'tier_2': {'coins': ['BNB', 'ADA', 'SOL', 'AVAX', 'DOT', 'MATIC'], 'base_score': 0.85},
            'tier_3': {'coins': ['ATOM', 'NEAR', 'LINK', 'UNI', 'AAVE'], 'base_score': 0.75},
            'tier_4': {'coins': ['SAND', 'MANA', 'AXS', 'VET', 'ALGO'], 'base_score': 0.65},
            'tier_5': {'coins': ['DOGE', 'SHIB'], 'base_score': 0.45}
        }
    
    def analyze_quality(self, symbol: str, market_data: Dict) -> Dict:
        """프로젝트 품질 종합 분석"""
        coin_name = symbol.replace('KRW-', '').upper()
        tier, base_score = self._get_coin_tier(coin_name)
        
        # 생태계 건전성
        ecosystem_score = self._analyze_ecosystem(market_data)
        
        # 혁신성 평가
        innovation_score = self._get_innovation_score(coin_name)
        
        # 채택도 분석
        adoption_score = self._analyze_adoption(market_data)
        
        # 최종 품질 점수
        total_quality = (base_score * 0.5 + ecosystem_score * 0.25 + 
                        innovation_score * 0.15 + adoption_score * 0.10)
        
        return {
            'quality_score': total_quality,
            'tier': tier,
            'ecosystem_score': ecosystem_score,
            'innovation_score': innovation_score,
            'adoption_score': adoption_score,
            'category': self._get_category(coin_name)
        }
    
    def _get_coin_tier(self, coin_name: str) -> Tuple[str, float]:
        """코인 등급 확인"""
        for tier, data in self.tier_database.items():
            if coin_name in data['coins']:
                return tier, data['base_score']
        return 'tier_unknown', 0.50
    
    def _analyze_ecosystem(self, market_data: Dict) -> float:
        """생태계 건전성 분석"""
        volume_24h = market_data.get('volume_24h_krw', 0)
        if volume_24h >= 100_000_000_000: return 0.9
        elif volume_24h >= 50_000_000_000: return 0.8
        elif volume_24h >= 10_000_000_000: return 0.7
        else: return 0.5
    
    def _get_innovation_score(self, coin_name: str) -> float:
        """혁신성 점수"""
        innovation_map = {
            'ETH': 0.95, 'ADA': 0.90, 'SOL': 0.88, 'AVAX': 0.85,
            'DOT': 0.85, 'ATOM': 0.80, 'LINK': 0.90, 'UNI': 0.85
        }
        return innovation_map.get(coin_name, 0.60)
    
    def _analyze_adoption(self, market_data: Dict) -> float:
        """채택도 분석"""
        volume_24h = market_data.get('volume_24h_krw', 0)
        if volume_24h >= 200_000_000_000: return 0.95
        elif volume_24h >= 100_000_000_000: return 0.85
        elif volume_24h >= 50_000_000_000: return 0.75
        else: return 0.60
    
    def _get_category(self, coin_name: str) -> str:
        """코인 카테고리 분류"""
        categories = {
            'L1_Blockchain': ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'ATOM'],
            'DeFi': ['UNI', 'AAVE', 'LINK', 'MKR', 'CRV'],
            'Gaming': ['SAND', 'MANA', 'AXS', 'ENJ'],
            'Meme': ['DOGE', 'SHIB'],
            'Infrastructure': ['LINK', 'VET', 'FIL']
        }
        for category, coins in categories.items():
            if coin_name in coins:
                return category
        return 'Other'

# ============================================================================
# 🔄 시장 사이클 감지기 (강화)
# ============================================================================
class EnhancedMarketCycleDetector:
    """시장 사이클 자동 감지 (강화 버전)"""
    
    async def detect_cycle(self) -> Dict:
        """현재 시장 사이클 감지"""
        try:
            # BTC 데이터 수집
            btc_data = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=60)
            if btc_data is                 # 5단계 분할매매 금액
                stage1_amount=split_plan.get('stage_amounts', [0]*5)[0],
                stage2_amount=split_plan.get('stage_amounts', [0]*5)[1],
                stage3_amount=split_plan.get('stage_amounts', [0]*5)[2],
                stage4_amount=split_plan.get('stage_amounts', [0]*5)[3],
                stage5_amount=split_plan.get('stage_amounts', [0]*5)[4],
                
                # 5단계 진입가격
                entry_price_1=split_plan.get('entry_prices', [data['price']]*5)[0],
                entry_price_2=split_plan.get('entry_prices', [data['price']]*5)[1],
                entry_price_3=split_plan.get('entry_prices', [data['price']]*5)[2],
                entry_price_4=split_plan.get('entry_prices', [data['price']]*5)[3],
                entry_price_5=split_plan.get('entry_prices', [data['price']]*5)[4],
                
                # 손절/익절 가격
                stop_loss=split_plan.get('stop_loss', data['price'] * 0.75),
                take_profit_1=split_plan.get('take_profits', [data['price']*1.2]*3)[0],
                take_profit_2=split_plan.get('take_profits', [data['price']*1.2]*3)[1],
                take_profit_3=split_plan.get('take_profits', [data['price']*1.2]*3)[2],
                reasoning=reasoning,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            self.logger.error(f"완전체 분석 실패 {symbol}: {e}")
            return self._create_empty_signal(symbol, f"분석 실패: {str(e)}")
    
    async def scan_all_coins(self) -> List[UltimateCoinSignal]:
        """전체 코인 완전체 스캔"""
        if not self.enabled:
            return []
        
        self.logger.info("🔍 궁극의 완전체 스캔 시작")
        
        try:
            # 시장 사이클 업데이트
            cycle_info = await self.cycle_detector.detect_cycle()
            self.current_cycle = cycle_info['cycle']
            self.cycle_confidence = cycle_info['confidence']
            
            self.logger.info(f"🔄 시장 사이클: {self.current_cycle} (신뢰도: {self.cycle_confidence:.2f})")
            
            # 6단계 정밀 필터링으로 코인 선별
            selected_symbols = await self._ultimate_auto_select_coins()
            if not selected_symbols:
                self.logger.error("6단계 정밀 필터링 실패")
                return []
            
            # 선별된 코인들 완전체 분석
            all_signals = []
            for i, symbol in enumerate(selected_symbols, 1):
                try:
                    self.logger.info(f"완전체 분석 중... {i}/{len(selected_symbols)} - {symbol}")
                    signal = await self.analyze_symbol(symbol)
                    all_signals.append(signal)
                    
                    # 진행 상황 로그
                    action_emoji = "🟢" if signal.action == "buy" else "🔴" if signal.action == "sell" else "⚪"
                    self.logger.info(f"{action_emoji} {symbol}: {signal.action} "
                                   f"신뢰도:{signal.confidence:.2f} 총점:{signal.total_score:.2f}")
                    
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    self.logger.error(f"개별 완전체 분석 실패 {symbol}: {e}")
                    continue
            
            # 결과 요약
            buy_count = len([s for s in all_signals if s.action == 'buy'])
            sell_count = len([s for s in all_signals if s.action == 'sell'])
            hold_count = len([s for s in all_signals if s.action == 'hold'])
            
            self.logger.info(f"✅ 완전체 스캔 완료!")
            self.logger.info(f"📊 결과: 매수 {buy_count}, 매도 {sell_count}, 보유 {hold_count}")
            
            return all_signals
        
        except Exception as e:
            self.logger.error(f"완전체 스캔 실패: {e}")
            return []
    
    async def _ultimate_auto_select_coins(self) -> List[str]:
        """6단계 정밀 필터링으로 코인 자동선별"""
        if self._is_cache_valid():
            self.logger.info("캐시된 선별 결과 사용")
            return [coin['symbol'] for coin in self.selected_coins]
        
        try:
            self.logger.info("6단계 정밀 필터링 자동선별 시작")
            start_time = time.time()
            
            # 모든 KRW 마켓 조회
            all_tickers = pyupbit.get_tickers(fiat="KRW")
            if not all_tickers:
                return self._get_default_coins()
            
            self.logger.info(f"📊 전체 코인: {len(all_tickers)}개")
            
            # 6단계 정밀 필터링 실행
            qualified_coins = await self.filtering_system.execute_six_stage_filtering(all_tickers)
            
            # 최종 선별
            final_selection = qualified_coins[:self.target_coins]
            selected_symbols = [coin['symbol'] for coin in final_selection]
            
            # 캐시 업데이트
            self.selected_coins = final_selection
            self.last_selection_time = datetime.now()
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ 6단계 정밀 필터링 완료: {len(selected_symbols)}개 코인 ({elapsed_time:.1f}초)")
            
            return selected_symbols
        
        except Exception as e:
            self.logger.error(f"6단계 정밀 필터링 실패: {e}")
            return self._get_default_coins()
    
    async def _get_comprehensive_coin_data(self, symbol: str) -> Optional[Dict]:
        """종합 코인 데이터 수집"""
        try:
            current_price = pyupbit.get_current_price(symbol)
            if not current_price:
                return None
            
            ohlcv = pyupbit.get_ohlcv(symbol, interval="day", count=60)
            if ohlcv is None or len(ohlcv) < 30:
                return None
            
            # 거래량 정보
            latest_volume = ohlcv.iloc[-1]['volume']
            avg_volume_7d = ohlcv['volume'].tail(7).mean()
            volume_spike = latest_volume / avg_volume_7d if avg_volume_7d > 0 else 1
            
            # 모멘텀 계산
            momentum_3d = (current_price / ohlcv.iloc[-4]['close'] - 1) * 100 if len(ohlcv) >= 4 else 0
            momentum_7d = (current_price / ohlcv.iloc[-8]['close'] - 1) * 100 if len(ohlcv) >= 8 else 0
            momentum_30d = (current_price / ohlcv.iloc[-31]['close'] - 1) * 100 if len(ohlcv) >= 31 else 0
            
            return {
                'symbol': symbol,
                'price': current_price,
                'ohlcv': ohlcv,
                'volume_24h_krw': latest_volume * current_price,
                'volume_spike_ratio': volume_spike,
                'momentum_3d': momentum_3d,
                'momentum_7d': momentum_7d,
                'momentum_30d': momentum_30d
            }
        
        except Exception as e:
            self.logger.error(f"종합 데이터 수집 실패 {symbol}: {e}")
            return None
    
    def _analyze_enhanced_fundamental(self, symbol: str, data: Dict, quality_analysis: Dict) -> float:
        """강화된 펀더멘털 분석"""
        try:
            score = 0.0
            
            # AI 품질 점수 (50%)
            quality_score = quality_analysis['quality_score']
            score += quality_score * 0.50
            
            # 거래량 점수 (30%)
            volume_krw = data.get('volume_24h_krw', 0)
            if volume_krw >= 200_000_000_000:
                score += 0.30
            elif volume_krw >= 100_000_000_000:
                score += 0.25
            elif volume_krw >= 50_000_000_000:
                score += 0.20
            elif volume_krw >= 10_000_000_000:
                score += 0.15
            else:
                score += 0.05
            
            # 생태계 건전성 (20%)
            ecosystem_score = quality_analysis.get('ecosystem_score', 0.5)
            score += ecosystem_score * 0.20
            
            return min(score, 1.0)
        except:
            return 0.5
    
    def _analyze_enhanced_momentum(self, data: Dict) -> float:
        """강화된 모멘텀 분석"""
        try:
            score = 0.0
            
            # 3일 모멘텀 (30%)
            momentum_3d = data.get('momentum_3d', 0)
            if momentum_3d >= 15:
                score += 0.30
            elif momentum_3d >= 8:
                score += 0.22
            elif momentum_3d >= 3:
                score += 0.15
            elif momentum_3d >= 0:
                score += 0.08
            
            # 7일 모멘텀 (30%)
            momentum_7d = data.get('momentum_7d', 0)
            if momentum_7d >= 25:
                score += 0.30
            elif momentum_7d >= 15:
                score += 0.22
            elif momentum_7d >= 5:
                score += 0.15
            elif momentum_7d >= 0:
                score += 0.08
            
            # 30일 모멘텀 (25%)
            momentum_30d = data.get('momentum_30d', 0)
            if momentum_30d >= 40:
                score += 0.25
            elif momentum_30d >= 25:
                score += 0.18
            elif momentum_30d >= 10:
                score += 0.12
            elif momentum_30d >= 0:
                score += 0.05
            
            # 거래량 급증 (15%)
            volume_spike = data.get('volume_spike_ratio', 1)
            if volume_spike >= 3.0:
                score += 0.15
            elif volume_spike >= 2.0:
                score += 0.12
            elif volume_spike >= 1.5:
                score += 0.08
            elif volume_spike >= 1.2:
                score += 0.05
            
            return min(score, 1.0)
        except:
            return 0.5
    
    def _get_cycle_based_weights(self) -> Dict:
        """시장 사이클별 가중치 (강화)"""
        if self.current_cycle == 'uptrend':
            return {'fundamental': 0.25, 'technical': 0.30, 'momentum': 0.45}
        elif self.current_cycle == 'downtrend':
            return {'fundamental': 0.55, 'technical': 0.30, 'momentum': 0.15}
        elif self.current_cycle == 'accumulation':
            return {'fundamental': 0.50, 'technical': 0.35, 'momentum': 0.15}
        elif self.current_cycle == 'distribution':
            return {'fundamental': 0.30, 'technical': 0.45, 'momentum': 0.25}
        else:  # sideways
            return {'fundamental': 0.35, 'technical': 0.35, 'momentum': 0.30}
    
    def _generate_enhanced_reasoning(self, fund_score: float, tech_score: float, 
                                   momentum_score: float, quality_analysis: Dict, 
                                   tech_details: Dict, market_cycle: str) -> str:
        """강화된 추론 생성"""
        reasoning_parts = []
        
        # 펀더멘털
        if fund_score >= 0.8:
            reasoning_parts.append("최고급펀더멘털")
        elif fund_score >= 0.6:
            reasoning_parts.append("우수펀더멘털")
        elif fund_score >= 0.4:
            reasoning_parts.append("보통펀더멘털")
        
        # 기술적 분석
        if tech_score >= 0.8:
            reasoning_parts.append("강한기술적매수")
        elif tech_score >= 0.6:
            reasoning_parts.append("기술적매수")
        elif tech_score >= 0.4:
            reasoning_parts.append("기술적중립")
        
        # 모멘텀
        if momentum_score >= 0.8:
            reasoning_parts.append("폭발적모멘텀")
        elif momentum_score >= 0.6:
            reasoning_parts.append("강한모멘텀")
        elif momentum_score >= 0.4:
            reasoning_parts.append("양호모멘텀")
        
        # AI 품질
        quality_score = quality_analysis.get('quality_score', 0.5)
        tier = quality_analysis.get('tier', 'unknown')
        if quality_score >= 0.9:
            reasoning_parts.append(f"최고품질({tier})")
        elif quality_score >= 0.7:
            reasoning_parts.append(f"우수품질({tier})")
        
        # 시장 사이클
        reasoning_parts.append(f"사이클:{market_cycle}")
        
        # 핵심 기술지표
        rsi = tech_details.get('rsi', 50)
        if rsi < 25:
            reasoning_parts.append("RSI극도과매도")
        elif rsi < 35:
            reasoning_parts.append("RSI과매도")
        elif rsi > 75:
            reasoning_parts.append("RSI과매수")
        
        williams_r = tech_details.get('williams_r', -50)
        if williams_r <= -85:
            reasoning_parts.append("WR극도과매도")
        
        adx = tech_details.get('adx', 25)
        if adx >= 30:
            reasoning_parts.append("강한트렌드")
        
        return " | ".join(reasoning_parts) if reasoning_parts else "중립신호"
    
    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if not self.last_selection_time or not self.selected_coins:
            return False
        
        time_diff = datetime.now() - self.last_selection_time
        return time_diff.total_seconds() < (self.cache_hours * 3600)
    
    def _get_default_coins(self) -> List[str]:
        """기본 코인 목록 (고품질 위주)"""
        return [
            'KRW-BTC', 'KRW-ETH', 'KRW-BNB', 'KRW-ADA', 'KRW-SOL',
            'KRW-AVAX', 'KRW-DOT', 'KRW-MATIC', 'KRW-ATOM', 'KRW-NEAR',
            'KRW-LINK', 'KRW-UNI', 'KRW-AAVE', 'KRW-SAND', 'KRW-MANA',
            'KRW-VET', 'KRW-ALGO', 'KRW-XLM', 'KRW-DOGE', 'KRW-XRP'
        ]
    
    def _create_empty_signal(self, symbol: str, reason: str) -> UltimateCoinSignal:
        """빈 시그널 생성"""
        return UltimateCoinSignal(
            symbol=symbol, action='hold', confidence=0.0, price=0.0, 
            total_score=0.0, fundamental_score=0.0, technical_score=0.0, 
            momentum_score=0.0, quality_score=0.0,
            market_cycle=self.current_cycle, 
            cycle_confidence=self.cycle_confidence,
            sector='Unknown', rsi=50.0, williams_r=-50.0, cci=0.0, 
            mfi=50.0, adx=25.0, macd_signal='neutral', 
            bb_position='normal', ichimoku_signal='neutral',
            momentum_3d=0.0, momentum_7d=0.0, momentum_30d=0.0, 
            volume_spike_ratio=1.0, btc_correlation=0.5, 
            portfolio_fit_score=0.5, diversification_benefit=0.5,
            position_stage=0, total_investment=0.0, 
            stage1_amount=0.0, stage2_amount=0.0, stage3_amount=0.0, 
            stage4_amount=0.0, stage5_amount=0.0,
            entry_price_1=0.0, entry_price_2=0.0, entry_price_3=0.0, 
            entry_price_4=0.0, entry_price_5=0.0,
            stop_loss=0.0, take_profit_1=0.0, take_profit_2=0.0, 
            take_profit_3=0.0, reasoning=reason, timestamp=datetime.now()
        )
    
    async def generate_ultimate_portfolio_report(self, signals: List[UltimateCoinSignal]) -> Dict:
        """궁극의 포트폴리오 리포트"""
        if not signals:
            return {"error": "분석된 코인이 없습니다"}
        
        # 기본 통계
        total_coins = len(signals)
        buy_signals = [s for s in signals if s.action == 'buy']
        sell_signals = [s for s in signals if s.action == 'sell']
        hold_signals = [s for s in signals if s.action == 'hold']
        
        # 평균 점수
        avg_scores = {
            'total': np.mean([s.total_score for s in signals]),
            'fundamental': np.mean([s.fundamental_score for s in signals]),
            'technical': np.mean([s.technical_score for s in signals]),
            'momentum': np.mean([s.momentum_score for s in signals]),
            'quality': np.mean([s.quality_score for s in signals])
        }
        
        # 고급 기술지표 평균
        avg_indicators = {
            'rsi': np.mean([s.rsi for s in signals]),
            'williams_r': np.mean([s.williams_r for s in signals]),
            'adx': np.mean([s.adx for s in signals]),
            'btc_correlation': np.mean([s.btc_correlation for s in signals])
        }
        
        # 섹터 분포
        sector_dist = {}
        for signal in signals:
            sector_dist[signal.sector] = sector_dist.get(signal.sector, 0) + 1
        
        # 총 투자금액 및 5단계 분할매매 정보
        total_investment = sum([s.total_investment for s in buy_signals])
        avg_stages = np.mean([s.position_stage for s in signals])
        
        # 상위 매수 추천 (신뢰도순)
        top_buys = sorted(buy_signals, key=lambda x: x.confidence, reverse=True)[:5]
        
        # 리스크 분석
        risk_analysis = {
            'avg_confidence': np.mean([s.confidence for s in buy_signals]) if buy_signals else 0,
            'max_single_investment': max([s.total_investment for s in signals]) if signals else 0,
            'portfolio_concentration': (max([s.total_investment for s in signals]) / 
                                      total_investment * 100) if total_investment > 0 else 0,
            'correlation_risk': self._assess_correlation_risk([s.btc_correlation for s in signals])
        }
        
        return {
            'summary': {
                'total_coins': total_coins,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'hold_signals': len(hold_signals),
                'total_investment': total_investment,
                'portfolio_allocation': (total_investment / self.portfolio_value) * 100
            },
            'scores': avg_scores,
            'advanced_indicators': avg_indicators,
            'market_analysis': {
                'current_cycle': self.current_cycle,
                'cycle_confidence': self.cycle_confidence,
                'cycle_optimized_coins': len([s for s in signals 
                                            if s.market_cycle == self.current_cycle])
            },
            'diversification': {
                'sector_distribution': sector_dist,
                'sector_count': len(sector_dist),
                'diversification_score': len(sector_dist) / max(total_coins, 1)
            },
            'split_trading_analysis': {
                'total_plans': len([s for s in buy_signals if s.total_investment > 0]),
                'avg_investment_per_coin': (total_investment / len(buy_signals) 
                                          if buy_signals else 0),
                'stage_distribution': avg_stages
            },
            'top_recommendations': [{
                'symbol': s.symbol,
                'confidence': round(s.confidence * 100, 1),
                'sector': s.sector,
                'price': s.price,
                'total_score': round(s.total_score, 3),
                'quality_score': round(s.quality_score, 3),
                'total_investment': s.total_investment,
                'btc_correlation': round(s.btc_correlation, 3),
                'stage1_entry': s.entry_price_1,
                'stop_loss': s.stop_loss,
                'take_profit_1': s.take_profit_1,
                'reasoning': s.reasoning
            } for s in top_buys],
            'risk_metrics': risk_analysis
        }
    
    def _assess_correlation_risk(self, correlations: List[float]) -> str:
        """상관관계 리스크 평가"""
        try:
            if not correlations:
                return 'UNKNOWN'
            
            avg_correlation = np.mean([abs(c) for c in correlations])
            
            if avg_correlation > 0.8:
                return 'HIGH'
            elif avg_correlation > 0.6:
                return 'MEDIUM'
            else:
                return 'LOW'
        except:
            return 'MEDIUM'

# ============================================================================
# 🛠️ 유틸리티 및 실행 함수들
# ============================================================================

async def analyze_single_coin(symbol: str) -> Dict:
    """단일 코인 완전체 분석"""
    strategy = UltimateCoinStrategy()
    signal = await strategy.analyze_symbol(symbol)
    
    return {
        'symbol': signal.symbol,
        'decision': signal.action,
        'confidence_percent': round(signal.confidence * 100, 1),
        'total_score': round(signal.total_score, 3),
        'scores': {
            'fundamental': round(signal.fundamental_score, 3),
            'technical': round(signal.technical_score, 3),
            'momentum': round(signal.momentum_score, 3),
            'quality': round(signal.quality_score, 3)
        },
        'advanced_indicators': {
            'rsi': round(signal.rsi, 1),
            'williams_r': round(signal.williams_r, 1),
            'cci': round(signal.cci, 1),
            'mfi': round(signal.mfi, 1),
            'adx': round(signal.adx, 1),
            'macd_signal': signal.macd_signal,
            'ichimoku_signal': signal.ichimoku_signal
        },
        'momentum_analysis': {
            'momentum_3d': round(signal.momentum_3d, 1),
            'momentum_7d': round(signal.momentum_7d, 1),
            'momentum_30d': round(signal.momentum_30d, 1),
            'volume_spike_ratio': round(signal.volume_spike_ratio, 2)
        },
        'market_info': {
            'cycle': signal.market_cycle,
            'cycle_confidence': round(signal.cycle_confidence, 2),
            'sector': signal.sector,
            'btc_correlation': round(signal.btc_correlation, 3)
        },
        'split_trading_plan': {
            'total_investment': signal.total_investment,
            'stage_amounts': [signal.stage1_amount, signal.stage2_amount, 
                             signal.stage3_amount, signal.stage4_amount, 
                             signal.stage5_amount],
            'entry_prices': [signal.entry_price_1, signal.entry_price_2, 
                           signal.entry_price_3, signal.entry_price_4, 
                           signal.entry_price_5],
            'stop_loss': signal.stop_loss,
            'take_profits': [signal.take_profit_1, signal.take_profit_2, 
                           signal.take_profit_3]
        },
        'reasoning': signal.reasoning
    }

async def run_ultimate_analysis() -> Tuple[List[UltimateCoinSignal], Dict]:
    """궁극의 완전체 분석 실행"""
    strategy = UltimateCoinStrategy()
    signals = await strategy.scan_all_coins()
    report = await strategy.generate_ultimate_portfolio_report(signals)
    return signals, report

def export_ultimate_csv(signals: List[UltimateCoinSignal], filename: str = None) -> str:
    """완전체 분석 결과 CSV 내보내기"""
    import csv
    
    if filename is None:
        filename = f"ultimate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더
        writer.writerow([
            'Symbol', 'Action', 'Confidence', 'Total_Score', 'Price',
            'Fundamental', 'Technical', 'Momentum', 'Quality',
            'RSI', 'Williams_R', 'CCI', 'MFI', 'ADX',
            'MACD_Signal', 'BB_Position', 'Ichimoku_Signal',
            'Momentum_3D', 'Momentum_7D', 'Momentum_30D', 'Volume_Spike',
            'BTC_Correlation', 'Sector', 'Market_Cycle',
            'Total_Investment', 'Entry1', 'Entry2', 'Entry3', 'Entry4', 'Entry5',
            'Stop_Loss', 'Take_Profit1', 'Take_Profit2', 'Take_Profit3',
            'Reasoning'
        ])
        
        # 데이터
        for s in signals:
            writer.writerow([
                s.symbol, s.action, f"{s.confidence:.3f}", f"{s.total_score:.3f}", s.price,
                f"{s.fundamental_score:.3f}", f"{s.technical_score:.3f}", 
                f"{s.momentum_score:.3f}", f"{s.quality_score:.3f}",
                f"{s.rsi:.1f}", f"{s.williams_r:.1f}", f"{s.cci:.1f}", 
                f"{s.mfi:.1f}", f"{s.adx:.1f}",
                s.macd_signal, s.bb_position, s.ichimoku_signal,
                f"{s.momentum_3d:.1f}", f"{s.momentum_7d:.1f}", 
                f"{s.momentum_30d:.1f}", f"{s.volume_spike_ratio:.2f}",
                f"{s.btc_correlation:.3f}", s.sector, s.market_cycle,
                s.total_investment, s.entry_price_1, s.entry_price_2, 
                s.entry_price_3, s.entry_price_4, s.entry_price_5, 
                s.stop_loss, s.take_profit_1, s.take_profit_2, 
                s.take_profit_3, s.reasoning
            ])
    
    return filename

def format_currency(amount: float) -> str:
    """통화 포맷팅"""
    if amount >= 1e12:
        return f"{amount/1e12:.1f}조원"
    elif amount >= 1e8:
        return f"{amount/1e8:.1f}억원"
    elif amount >= 1e4:
        return f"{amount/1e4:.1f}만원"
    else:
        return f"{amount:,.0f}원"

def print_ultimate_summary(signals: List[UltimateCoinSignal], report: Dict):
    """궁극의 분석 결과 요약 출력"""
    print("\n" + "="*80)
    print("🪙 궁극의 암호화폐 완전체 분석 결과")
    print("="*80)
    
    if not signals:
        print("❌ 분석된 코인이 없습니다.")
        return
    
    # 기본 통계
    summary = report['summary']
    print(f"\n📊 6단계 정밀 필터링 결과:")
    print(f"   총 분석: {summary['total_coins']}개")
    print(f"   매수 신호: {summary['buy_signals']}개")
    print(f"   매도 신호: {summary['sell_signals']}개")
    print(f"   보유 신호: {summary['hold_signals']}개")
    
    # 시장 분석
    market = report['market_analysis']
    print(f"\n🔄 시장 사이클 분석:") or len(btc_data) < 60:
                return self._default_cycle()
            
            # 다중 지표 분석
            trend_analysis = self._analyze_trend(btc_data)
            volume_analysis = self._analyze_volume(btc_data)
            momentum_analysis = self._analyze_momentum(btc_data)
            
            # 종합 판정
            cycle_score = (trend_analysis['score'] * 0.5 + 
                          volume_analysis['score'] * 0.3 + 
                          momentum_analysis['score'] * 0.2)
            
            cycle, confidence = self._determine_cycle(cycle_score)
            
            return {
                'cycle': cycle,
                'confidence': confidence,
                'trend': trend_analysis,
                'volume': volume_analysis,
                'momentum': momentum_analysis
            }
        
        except Exception as e:
            logging.error(f"시장 사이클 감지 실패: {e}")
            return self._default_cycle()
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """트렌드 분석"""
        closes = data['close']
        current = closes.iloc[-1]
        ma7 = closes.tail(7).mean()
        ma30 = closes.tail(30).mean()
        ma60 = closes.tail(60).mean()
        
        if current > ma7 > ma30 > ma60:
            return {'signal': 'strong_bullish', 'score': 0.8}
        elif current > ma7 > ma30:
            return {'signal': 'bullish', 'score': 0.6}
        elif current < ma7 < ma30 < ma60:
            return {'signal': 'strong_bearish', 'score': -0.8}
        elif current < ma7 < ma30:
            return {'signal': 'bearish', 'score': -0.6}
        else:
            return {'signal': 'neutral', 'score': 0.0}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """거래량 분석"""
        volumes = data['volume']
        recent_vol = volumes.tail(7).mean()
        avg_vol = volumes.tail(30).mean()
        
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        
        if vol_ratio >= 1.5:
            return {'signal': 'high_activity', 'score': 0.3}
        elif vol_ratio <= 0.7:
            return {'signal': 'low_activity', 'score': -0.3}
        else:
            return {'signal': 'normal_activity', 'score': 0.0}
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """모멘텀 분석"""
        closes = data['close']
        momentum_7d = (closes.iloc[-1] / closes.iloc[-8] - 1) * 100
        momentum_30d = (closes.iloc[-1] / closes.iloc[-31] - 1) * 100
        
        avg_momentum = (momentum_7d + momentum_30d) / 2
        
        if avg_momentum >= 15:
            return {'signal': 'strong_momentum', 'score': 0.4}
        elif avg_momentum >= 5:
            return {'signal': 'positive_momentum', 'score': 0.2}
        elif avg_momentum <= -15:
            return {'signal': 'negative_momentum', 'score': -0.4}
        elif avg_momentum <= -5:
            return {'signal': 'weak_momentum', 'score': -0.2}
        else:
            return {'signal': 'neutral_momentum', 'score': 0.0}
    
    def _determine_cycle(self, score: float) -> Tuple[str, float]:
        """사이클 결정"""
        if score >= 0.6:
            return 'uptrend', min(score, 0.95)
        elif score <= -0.6:
            return 'downtrend', min(abs(score), 0.95)
        elif 0.3 <= score < 0.6:
            return 'accumulation', score + 0.2
        elif -0.6 < score <= -0.3:
            return 'distribution', abs(score) + 0.2
        else:
            return 'sideways', 0.5
    
    def _default_cycle(self) -> Dict:
        """기본 사이클 반환"""
        return {
            'cycle': 'sideways',
            'confidence': 0.5,
            'trend': {'signal': 'neutral', 'score': 0.0},
            'volume': {'signal': 'normal_activity', 'score': 0.0},
            'momentum': {'signal': 'neutral_momentum', 'score': 0.0}
        }

# ============================================================================
# 📈 고급 기술적 분석기 (완전체)
# ============================================================================
class AdvancedTechnicalAnalyzer:
    """고급 기술적 지표 분석 (완전체)"""
    
    @staticmethod
    def analyze_all_indicators(ohlcv_data: pd.DataFrame) -> Tuple[float, Dict]:
        """모든 고급 기술지표 분석"""
        try:
            if len(ohlcv_data) < 50:
                return 0.5, {}
            
            score = 0.0
            details = {}
            
            # RSI (15%)
            rsi = AdvancedTechnicalAnalyzer._calculate_rsi(ohlcv_data['close'])
            if 30 <= rsi <= 70:
                score += 0.15
            elif rsi < 30:
                score += 0.12  # 과매도
            details['rsi'] = rsi
            
            # Williams %R (15%)
            williams_r = AdvancedTechnicalAnalyzer._calculate_williams_r(ohlcv_data)
            if williams_r <= -80:
                score += 0.15  # 과매도
            elif williams_r >= -20:
                score += 0.08  # 과매수
            else:
                score += 0.10
            details['williams_r'] = williams_r
            
            # CCI (15%)
            cci = AdvancedTechnicalAnalyzer._calculate_cci(ohlcv_data)
            if cci <= -100:
                score += 0.15  # 과매도
            elif cci >= 100:
                score += 0.08  # 과매수
            else:
                score += 0.10
            details['cci'] = cci
            
            # MFI (10%)
            mfi = AdvancedTechnicalAnalyzer._calculate_mfi(ohlcv_data)
            if mfi <= 20:
                score += 0.10  # 과매도
            elif mfi >= 80:
                score += 0.05  # 과매수
            else:
                score += 0.07
            details['mfi'] = mfi
            
            # ADX (10%)
            adx = AdvancedTechnicalAnalyzer._calculate_adx(ohlcv_data)
            if adx >= 25:
                score += 0.10  # 강한 트렌드
            elif adx >= 20:
                score += 0.07
            else:
                score += 0.03
            details['adx'] = adx
            
            # MACD (15%)
            macd_signal = AdvancedTechnicalAnalyzer._calculate_macd(ohlcv_data['close'])
            if macd_signal == 'bullish':
                score += 0.15
            elif macd_signal == 'neutral':
                score += 0.07
            details['macd_signal'] = macd_signal
            
            # 볼린저밴드 (10%)
            bb_position = AdvancedTechnicalAnalyzer._calculate_bb_signal(ohlcv_data['close'])
            if bb_position == 'oversold':
                score += 0.10
            elif bb_position == 'normal':
                score += 0.07
            details['bb_position'] = bb_position
            
            # 일목균형표 (10%)
            ichimoku_signal = AdvancedTechnicalAnalyzer._calculate_ichimoku(ohlcv_data)
            if ichimoku_signal == 'bullish':
                score += 0.10
            elif ichimoku_signal == 'neutral':
                score += 0.05
            details['ichimoku_signal'] = ichimoku_signal
            
            return min(score, 1.0), details
        
        except Exception as e:
            logging.error(f"기술적 분석 실패: {e}")
            return 0.5, {}
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    @staticmethod
    def _calculate_williams_r(data: pd.DataFrame, period: int = 14) -> float:
        """Williams %R 계산"""
        try:
            high_n = data['high'].rolling(window=period).max()
            low_n = data['low'].rolling(window=period).min()
            williams_r = -100 * ((high_n - data['close']) / (high_n - low_n))
            return williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50.0
        except:
            return -50.0
    
    @staticmethod
    def _calculate_cci(data: pd.DataFrame, period: int = 20) -> float:
        """Commodity Channel Index 계산"""
        try:
            tp = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - sma_tp) / (0.015 * mad)
            return cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def _calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
        """Money Flow Index 계산"""
        try:
            if 'volume' not in data.columns:
                return 50.0
            
            tp = (data['high'] + data['low'] + data['close']) / 3
            raw_money_flow = tp * data['volume']
            
            money_flow_positive, money_flow_negative = [], []
            for i in range(1, len(data)):
                if tp.iloc[i] > tp.iloc[i-1]:
                    money_flow_positive.append(raw_money_flow.iloc[i])
                    money_flow_negative.append(0)
                elif tp.iloc[i] < tp.iloc[i-1]:
                    money_flow_positive.append(0)
                    money_flow_negative.append(raw_money_flow.iloc[i])
                else:
                    money_flow_positive.append(0)
                    money_flow_negative.append(0)
            
            if len(money_flow_positive) < period - 1:
                return 50.0
            
            mf_positive = pd.Series(money_flow_positive).rolling(window=period-1).sum()
            mf_negative = pd.Series(money_flow_negative).rolling(window=period-1).sum()
            mf_negative = mf_negative.replace(0, 0.001)
            
            mfi = 100 - (100 / (1 + (mf_positive / mf_negative)))
            return mfi.iloc[-1] if len(mfi) > 0 and not pd.isna(mfi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    @staticmethod
    def _calculate_adx(data: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index 계산"""
        try:
            high, low, close = data['high'], data['low'], data['close']
            
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
            minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0
        except:
            return 25.0
    
    @staticmethod
    def _calculate_macd(prices: pd.Series) -> str:
        """MACD 시그널 계산"""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            
            if macd_line.iloc[-1] > signal_line.iloc[-1]:
                return 'bullish'
            elif macd_line.iloc[-1] < signal_line.iloc[-1]:
                return 'bearish'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    @staticmethod
    def _calculate_bb_signal(prices: pd.Series, period: int = 20) -> str:
        """볼린저밴드 시그널"""
        try:
            ma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = ma + (std * 2)
            lower = ma - (std * 2)
            current_price = prices.iloc[-1]
            
            if current_price < lower.iloc[-1]:
                return 'oversold'
            elif current_price > upper.iloc[-1]:
                return 'overbought'
            else:
                return 'normal'
        except:
            return 'normal'
    
    @staticmethod
    def _calculate_ichimoku(data: pd.DataFrame) -> str:
        """일목균형표 시그널"""
        try:
            high, low, close = data['high'], data['low'], data['close']
            
            # 전환선 (9일)
            tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
            # 기준선 (26일)
            kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
            # 선행스팬A
            senkou_a = ((tenkan + kijun) / 2).shift(26)
            # 선행스팬B
            senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
            
            current_price = close.iloc[-1]
            
            if len(tenkan) > 0 and len(kijun) > 0:
                if (tenkan.iloc[-1] > kijun.iloc[-1] and 
                    current_price > tenkan.iloc[-1]):
                    return 'bullish'
                elif (tenkan.iloc[-1] < kijun.iloc[-1] and 
                      current_price < tenkan.iloc[-1]):
                    return 'bearish'
            
            return 'neutral'
        except:
            return 'neutral'

# ============================================================================
# 🔗 상관관계 포트폴리오 최적화기
# ============================================================================
class CorrelationPortfolioOptimizer:
    """상관관계 기반 포트폴리오 최적화"""
    
    def __init__(self):
        self.correlation_threshold = 0.7
        self.max_same_sector = 3
    
    async def calculate_btc_correlation(self, symbol: str) -> float:
        """BTC와의 상관관계 계산"""
        try:
            if symbol == 'KRW-BTC':
                return 1.0
            
            btc_data = pyupbit.get_ohlcv('KRW-BTC', interval="day", count=30)
            coin_data = pyupbit.get_ohlcv(symbol, interval="day", count=30)
            
            if btc_data is None or coin_data is None or len(btc_data) < 30 or len(coin_data) < 30:
                return 0.5
            
            btc_returns = btc_data['close'].pct_change().dropna()
            coin_returns = coin_data['close'].pct_change().dropna()
            
            min_len = min(len(btc_returns), len(coin_returns))
            btc_returns = btc_returns.tail(min_len)
            coin_returns = coin_returns.tail(min_len)
            
            correlation = btc_returns.corr(coin_returns)
            return correlation if not pd.isna(correlation) else 0.5
        except:
            return 0.5
    
    def calculate_diversification_benefit(self, symbol: str, selected_symbols: List[str], 
                                        symbol_sectors: Dict[str, str]) -> float:
        """다양성 혜택 점수 계산"""
        try:
            if not selected_symbols:
                return 1.0
            
            current_sector = symbol_sectors.get(symbol, 'Other')
            selected_sectors = [symbol_sectors.get(s, 'Other') for s in selected_symbols]
            
            # 같은 섹터 개수 확인
            same_sector_count = selected_sectors.count(current_sector)
            
            if same_sector_count >= self.max_same_sector:
                return 0.3  # 페널티
            elif same_sector_count == 0:
                return 1.0  # 보너스
            else:
                return max(0.5, 1.0 - (same_sector_count * 0.2))
        except:
            return 0.7
    
    def optimize_portfolio_selection(self, candidates: List[Dict], target_count: int) -> List[Dict]:
        """포트폴리오 최적화 선별"""
        try:
            if len(candidates) <= target_count:
                return candidates
            
            # 점수순 정렬
            sorted_candidates = sorted(candidates, key=lambda x: x.get('selection_score', 0), reverse=True)
            
            selected = []
            selected_sectors = []
            sector_counts = {}
            
            for candidate in sorted_candidates:
                if len(selected) >= target_count:
                    break
                
                symbol = candidate['symbol']
                sector = candidate.get('sector', 'Other')
                
                # 섹터 다양성 체크
                current_sector_count = sector_counts.get(sector, 0)
                
                if current_sector_count < self.max_same_sector:
                    selected.append(candidate)
                    selected_sectors.append(sector)
                    sector_counts[sector] = current_sector_count + 1
            
            # 부족한 자리는 점수 순으로 채우기
            remaining_slots = target_count - len(selected)
            if remaining_slots > 0:
                remaining = [c for c in sorted_candidates if c not in selected]
                selected.extend(remaining[:remaining_slots])
            
            return selected[:target_count]
        except:
            return candidates[:target_count]

# ============================================================================
# 🎯 6단계 정밀 필터링 시스템
# ============================================================================
class SixStageFilteringSystem:
    """6단계 정밀 필터링 시스템"""
    
    def __init__(self, min_volume_24h: int, quality_analyzer, portfolio_optimizer):
        self.min_volume_24h = min_volume_24h
        self.quality_analyzer = quality_analyzer
        self.portfolio_optimizer = portfolio_optimizer
        self.logger = logging.getLogger(__name__)
    
    async def execute_six_stage_filtering(self, all_tickers: List[str]) -> List[Dict]:
        """6단계 정밀 필터링 실행"""
        self.logger.info("🔍 6단계 정밀 필터링 시작")
        
        # 1단계: 기본 필터링
        stage1_passed = await self._stage1_basic_filtering(all_tickers)
        self.logger.info(f"1단계 통과: {len(stage1_passed)}개 (기본 필터)")
        
        # 2단계: 거래량 및 가격 안정성
        stage2_passed = await self._stage2_volume_stability(stage1_passed)
        self.logger.info(f"2단계 통과: {len(stage2_passed)}개 (거래량 안정성)")
        
        # 3단계: AI 품질 1차 스크리닝
        stage3_passed = await self._stage3_quality_screening(stage2_passed)
        self.logger.info(f"3단계 통과: {len(stage3_passed)}개 (AI 품질 스크리닝)")
        
        # 4단계: 기술적 지표 2차 스크리닝
        stage4_passed = await self._stage4_technical_screening(stage3_passed)
        self.logger.info(f"4단계 통과: {len(stage4_passed)}개 (기술적 스크리닝)")
        
        # 5단계: 모멘텀 3차 스크리닝
        stage5_passed = await self._stage5_momentum_screening(stage4_passed)
        self.logger.info(f"5단계 통과: {len(stage5_passed)}개 (모멘텀 스크리닝)")
        
        # 6단계: 포트폴리오 최적화 및 상관관계 분석
        stage6_passed = await self._stage6_portfolio_optimization(stage5_passed)
        self.logger.info(f"6단계 통과: {len(stage6_passed)}개 (포트폴리오 최적화)")
        
        return stage6_passed
    
    async def _stage1_basic_filtering(self, all_tickers: List[str]) -> List[Dict]:
        """1단계: 기본 필터링"""
        qualified_coins = []
        batch_size = 15
        
        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._basic_filter_coin, ticker) for ticker in batch]
                
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        if result:
                            qualified_coins.append(result)
                    except:
                        continue
            
            await asyncio.sleep(0.3)
        
        return qualified_coins[:100]  # 상위 100개
    
    def _basic_filter_coin(self, symbol: str) -> Optional[Dict]:
        """단일 코인 기본 필터링"""
        try:
            current_price = pyupbit.get_current_price(symbol)
            if not current_price or current_price <= 0:
                return None
            
            # 30일 데이터 확인
            ohlcv = pyupbit.get_ohlcv(symbol, interval="day", count=30)
            if ohlcv is None or len(ohlcv) < 30:
                return None
            
            # 거래량 확인
            latest_volume = ohlcv.iloc[-1]['volume']
            volume_24h_krw = latest_volume * current_price
            
            if volume_24h_krw < self.min_volume_24h:
                return None
            
            # 최소 가격 필터
            if current_price < 10:
                return None
            
            return {
                'symbol': symbol,
                'price': current_price,
                'volume_24h_krw': volume_24h_krw,
                'ohlcv': ohlcv,
                'stage1_passed': True
            }
        except:
            return None
    
    async def _stage2_volume_stability(self, stage1_coins: List[Dict]) -> List[Dict]:
        """2단계: 거래량 및 가격 안정성"""
        qualified_coins = []
        
        for coin_data in stage1_coins:
            try:
                ohlcv = coin_data['ohlcv']
                
                # 거래량 안정성 체크
                volumes = ohlcv['volume'].tail(7)
                volume_std = volumes.std()
                volume_mean = volumes.mean()
                volume_cv = volume_std / volume_mean if volume_mean > 0 else 999
                
                if volume_cv > 2.0:  # 너무 불안정한 거래량 제외
                    continue
                
                # 가격 안정성 체크
                prices = ohlcv['close'].tail(7)
                price_std = prices.std()
                price_mean = prices.mean()
                price_cv = price_std / price_mean if price_mean > 0 else 999
                
                if price_cv > 0.3:  # 너무 변동성 큰 코인 제외
                    continue
                
                coin_data['volume_stability'] = 1 / (1 + volume_cv)
                coin_data['price_stability'] = 1 / (1 + price_cv)
                qualified_coins.append(coin_data)
                
            except:
                continue
        
        return qualified_coins[:80]  # 상위 80개
    
    async def _stage3_quality_screening(self, stage2_coins: List[Dict]) -> List[Dict]:
        """3단계: AI 품질 1차 스크리닝"""
        qualified_coins = []
        
        for coin_data in stage2_coins:
            try:
                symbol = coin_data['symbol']
                
                quality_analysis = self.quality_analyzer.analyze_quality(symbol, coin_data)
                
                # 최소 품질 기준
                if quality_analysis['quality_score'] < 0.4:
                    continue
                
                # Tier 5 (밈코인) 제한
                if quality_analysis['tier'] == 'tier_5':
                    continue
                
                coin_data['quality_analysis'] = quality_analysis
                qualified_coins.append(coin_data)
                
            except:
                continue
        
        return qualified_coins[:60]  # 상위 60개
    
    async def _stage4_technical_screening(self, stage3_coins: List[Dict]) -> List[Dict]:
        """4단계: 기술적 지표 2차 스크리닝"""
        qualified_coins = []
        
        for coin_data in stage3_coins:
            try:
                ohlcv = coin_data['ohlcv']
                
                technical_score, tech_details = AdvancedTechnicalAnalyzer.analyze_all_indicators(ohlcv)
                
                # 기술적 최소 기준
                if technical_score < 0.3:
                    continue
                
                # 극단적 지표 제외
                rsi = tech_details.get('rsi', 50)
                if rsi > 85 or rsi < 10:
                    continue
                
                coin_data['technical_score'] = technical_score
                coin_data['technical_details'] = tech_details
                qualified_coins.append(coin_data)
                
            except:
                continue
        
        return qualified_coins[:45]  # 상위 45개
    
    async def _stage5_momentum_screening(self, stage4_coins: List[Dict]) -> List[Dict]:
        """5단계: 모멘텀 3차 스크리닝"""
        qualified_coins = []
        
        for coin_data in stage4_coins:
            try:
                ohlcv = coin_data['ohlcv']
                current_price = coin_data['price']
                
                # 모멘텀 계산
                momentum_data = self._calculate_momentum_indicators(ohlcv, current_price)
                momentum_score = self._analyze_momentum_score(momentum_data)
                
                # 모멘텀 최소 기준
                if momentum_score < 0.25:
                    continue
                
                # 극단적 하락 제외
                if momentum_data.get('momentum_7d', 0) < -25:
                    continue
                
                coin_data['momentum_score'] = momentum_score
                coin_data['momentum_data'] = momentum_data
                qualified_coins.append(coin_data)
                
            except:
                continue
        
        return qualified_coins[:35]  # 상위 35개
    
    async def _stage6_portfolio_optimization(self, stage5_coins: List[Dict]) -> List[Dict]:
        """6단계: 포트폴리오 최적화"""
        try:
            # 각 코인의 섹터 정보 추가
            for coin_data in stage5_coins:
                symbol = coin_data['symbol']
                quality_analysis = coin_data.get('quality_analysis', {})
                coin_data['sector'] = quality_analysis.get('category', 'Other')
            
            # 종합 점수 계산
            for coin_data in stage5_coins:
                quality_score = coin_data.get('quality_analysis', {}).get('quality_score', 0.5)
                technical_score = coin_data.get('technical_score', 0.5)
                momentum_score = coin_data.get('momentum_score', 0.5)
                
                # 가중 평균
                total_score = (quality_score * 0.4 + technical_score * 0.35 + momentum_score * 0.25)
                coin_data['selection_score'] = total_score
            
            # 포트폴리오 최적화 적용
            symbol_sectors = {coin['symbol']: coin['sector'] for coin in stage5_coins}
            
            optimized_selection = self.portfolio_optimizer.optimize_portfolio_selection(
                stage5_coins, min(25, len(stage5_coins))
            )
            
            return optimized_selection
            
        except Exception as e:
            self.logger.error(f"6단계 포트폴리오 최적화 실패: {e}")
            return stage5_coins[:25]
    
    def _calculate_momentum_indicators(self, ohlcv: pd.DataFrame, current_price: float) -> Dict:
        """모멘텀 지표 계산"""
        try:
            data = {}
            
            if len(ohlcv) >= 30:
                data['momentum_3d'] = (current_price / ohlcv.iloc[-4]['close'] - 1) * 100
                data['momentum_7d'] = (current_price / ohlcv.iloc[-8]['close'] - 1) * 100
                data['momentum_30d'] = (current_price / ohlcv.iloc[-31]['close'] - 1) * 100
            else:
                data['momentum_3d'] = data['momentum_7d'] = data['momentum_30d'] = 0
            
            # 거래량 급증률
            if len(ohlcv) >= 7:
                avg_volume = ohlcv['volume'].tail(7).mean()
                current_volume = ohlcv.iloc[-1]['volume']
                data['volume_spike_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
            else:
                data['volume_spike_ratio'] = 1
            
            return data
        except:
            return {'momentum_3d': 0, 'momentum_7d': 0, 'momentum_30d': 0, 'volume_spike_ratio': 1}
    
    def _analyze_momentum_score(self, momentum_data: Dict) -> float:
        """모멘텀 점수 계산"""
        try:
            score = 0.0
            
            # 3일 모멘텀 (30%)
            momentum_3d = momentum_data.get('momentum_3d', 0)
            if momentum_3d >= 15:
                score += 0.30
            elif momentum_3d >= 5:
                score += 0.20
            elif momentum_3d >= 0:
                score += 0.10
            
            # 7일 모멘텀 (30%)
            momentum_7d = momentum_data.get('momentum_7d', 0)
            if momentum_7d >= 20:
                score += 0.30
            elif momentum_7d >= 10:
                score += 0.20
            elif momentum_7d >= 0:
                score += 0.10
            
            # 30일 모멘텀 (25%)
            momentum_30d = momentum_data.get('momentum_30d', 0)
            if momentum_30d >= 30:
                score += 0.25
            elif momentum_30d >= 15:
                score += 0.15
            elif momentum_30d >= 0:
                score += 0.05
            
            # 거래량 급증 (15%)
            volume_spike = momentum_data.get('volume_spike_ratio', 1)
            if volume_spike >= 2.5:
                score += 0.15
            elif volume_spike >= 1.8:
                score += 0.10
            elif volume_spike >= 1.3:
                score += 0.05
            
            return min(score, 1.0)
        except:
            return 0.5

# ============================================================================
# 💰 정밀한 5단계 분할매매 시스템
# ============================================================================
class FiveStageSpiltTradingSystem:
    """정밀한 5단계 분할매매 시스템"""
    
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.stage_ratios = [0.20, 0.20, 0.20, 0.20, 0.20]  # 각 단계 20%
        self.stage_triggers = [0.0, -0.05, -0.10, -0.15, -0.20]  # 진입 트리거
    
    def calculate_split_trading_plan(self, symbol: str, current_price: float, 
                                   confidence: float, market_cycle: str, 
                                   target_coins: int) -> Dict:
        """5단계 분할매매 계획 수립"""
        try:
            # 기본 투자금액 계산
            base_amount = self.portfolio_value / target_coins
            confidence_multiplier = 0.5 + (confidence * 1.5)
            total_investment = base_amount * confidence_multiplier
            
            # 시장 사이클별 조정
            cycle_multipliers = {
                'uptrend': 1.2,
                'accumulation': 1.1,
                'sideways': 1.0,
                'distribution': 0.8,
                'downtrend': 0.6
            }
            
            cycle_multiplier = cycle_multipliers.get(market_cycle, 1.0)
            total_investment *= cycle_multiplier
            
            # 최대 투자 한도 (포트폴리오의 8%)
            max_investment = self.portfolio_value * 0.08
            total_investment = min(total_investment, max_investment)
            
            # 각 단계별 투자금액
            stage_amounts = [total_investment * ratio for ratio in self.stage_ratios]
            
            # 시장 사이클별 진입 트리거 조정
            triggers = self._adjust_triggers_for_cycle(market_cycle)
            
            # 각 단계별 진입가격
            entry_prices = [current_price * (1 + trigger) for trigger in triggers]
            
            # 손절/익절가 계산
            avg_entry_price = current_price * 0.88  # 평균 진입가 추정
            stop_loss = avg_entry_price * 0.75  # -25% 손절
            
            # 시장 사이클별 익절 목표 조정
            take_profit_multipliers = self._get_take_profit_multipliers(market_cycle, confidence)
            take_profits = [avg_entry_price * (1 + tp) for tp in take_profit_multipliers]
            
            return {
                'symbol': symbol,
                'total_investment': total_investment,
                'stage_amounts': stage_amounts,
                'entry_prices': entry_prices,
                'stage_triggers': triggers,
                'stop_loss': stop_loss,
                'take_profits': take_profits,
                'market_cycle': market_cycle,
                'confidence': confidence,
                'portfolio_weight': (total_investment / self.portfolio_value) * 100,
                'risk_level': self._calculate_risk_level(confidence, market_cycle)
            }
        
        except Exception as e:
            logging.error(f"분할매매 계획 수립 실패 {symbol}: {e}")
            return {}
    
    def _adjust_triggers_for_cycle(self, market_cycle: str) -> List[float]:
        """시장 사이클별 진입 트리거 조정"""
        if market_cycle == 'uptrend':
            return [0.0, -0.03, -0.06, -0.10, -0.15]  # 빠른 진입
        elif market_cycle == 'downtrend':
            return [0.0, -0.08, -0.15, -0.22, -0.30]  # 신중한 진입
        elif market_cycle == 'accumulation':
            return [0.0, -0.04, -0.08, -0.12, -0.18]  # 점진적 진입
        elif market_cycle == 'distribution':
            return [0.0, -0.06, -0.12, -0.18, -0.25]  # 보수적 진입
        else:  # sideways
            return self.stage_triggers
    
    def _get_take_profit_multipliers(self, market_cycle: str, confidence: float) -> List[float]:
        """시장 사이클별 익절 목표"""
        base_multipliers = {
            'uptrend': [0.25, 0.60, 1.20],
            'accumulation': [0.15, 0.35, 0.70],
            'sideways': [0.20, 0.45, 0.90],
            'distribution': [0.12, 0.25, 0.50],
            'downtrend': [0.08, 0.18, 0.35]
        }
        
        multipliers = base_multipliers.get(market_cycle, [0.20, 0.45, 0.90])
        
        # 신뢰도 기반 조정
        confidence_factor = 0.7 + (confidence * 0.6)
        adjusted_multipliers = [m * confidence_factor for m in multipliers]
        
        return adjusted_multipliers
    
    def _calculate_risk_level(self, confidence: float, market_cycle: str) -> str:
        """리스크 레벨 계산"""
        risk_score = 0
        
        if confidence >= 0.8:
            risk_score -= 2
        elif confidence >= 0.6:
            risk_score -= 1
        elif confidence <= 0.4:
            risk_score += 2
        
        cycle_risk = {
            'uptrend': -1,
            'accumulation': 0,
            'sideways': 0,
            'distribution': 1,
            'downtrend': 2
        }
        
        risk_score += cycle_risk.get(market_cycle, 0)
        
        if risk_score <= -2:
            return 'VERY_LOW'
        elif risk_score <= 0:
            return 'LOW'
        elif risk_score <= 2:
            return 'MEDIUM'
        elif risk_score <= 4:
            return 'HIGH'
        else:
            return 'VERY_HIGH'

# ============================================================================
# 🚀 메인 전략 클래스 (완전체)
# ============================================================================
class UltimateCoinStrategy:
    """궁극의 암호화폐 전략 (완전체 V5.0)"""
    
    def __init__(self):
        # 설정 관리
        self.config_manager = ConfigManager()
        self.coin_config = self.config_manager.get_coin_config()
        self.enabled = self.coin_config.get('enabled', True)
        
        # 핵심 파라미터
        self.target_coins = self.coin_config.get('target_coins', 20)
        self.min_volume_24h = self.coin_config.get('min_volume_24h', 500_000_000)
        self.portfolio_value = self.coin_config.get('portfolio_value', 200_000_000)
        
        # 분석 시스템들
        self.quality_analyzer = EnhancedProjectQualityAnalyzer()
        self.cycle_detector = EnhancedMarketCycleDetector()
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.portfolio_optimizer = CorrelationPortfolioOptimizer()
        self.filtering_system = SixStageFilteringSystem(
            self.min_volume_24h, self.quality_analyzer, self.portfolio_optimizer
        )
        self.split_trading_system = FiveStageSpiltTradingSystem(self.portfolio_value)
        
        # 가중치 설정
        self.fundamental_weight = 0.35
        self.technical_weight = 0.35
        self.momentum_weight = 0.30
        
        # 캐시 시스템
        self.selected_coins = []
        self.last_selection_time = None
        self.cache_hours = 12
        
        # 현재 시장 상태
        self.current_cycle = 'sideways'
        self.cycle_confidence = 0.5
        
        self.logger = logging.getLogger(__name__)
        if self.enabled:
            self.logger.info("🪙 궁극의 암호화폐 전략 (완전체) 초기화 완료")
    
    async def analyze_symbol(self, symbol: str) -> UltimateCoinSignal:
        """개별 코인 완전체 분석"""
        if not self.enabled:
            return self._create_empty_signal(symbol, "전략 비활성화")
        
        try:
            # 기본 데이터 수집
            data = await self._get_comprehensive_coin_data(symbol)
            if not data:
                return self._create_empty_signal(symbol, "데이터 수집 실패")
            
            # AI 품질 분석
            quality_analysis = self.quality_analyzer.analyze_quality(symbol, data)
            
            # 고급 기술적 분석
            technical_score, tech_details = self.technical_analyzer.analyze_all_indicators(data['ohlcv'])
            
            # 강화된 펀더멘털 분석
            fundamental_score = self._analyze_enhanced_fundamental(symbol, data, quality_analysis)
            
            # 모멘텀 분석
            momentum_score = self._analyze_enhanced_momentum(data)
            
            # BTC 상관관계 계산
            btc_correlation = await self.portfolio_optimizer.calculate_btc_correlation(symbol)
            
            # 시장 사이클 기반 가중치
            weights = self._get_cycle_based_weights()
            
            # 총점 계산
            total_score = (
                fundamental_score * weights['fundamental'] +
                technical_score * weights['technical'] +
                momentum_score * weights['momentum']
            )
            
            # 품질 점수 반영
            total_score = (total_score * 0.8) + (quality_analysis['quality_score'] * 0.2)
            
            # 다양성 혜택 계산
            symbol_sectors = {symbol: quality_analysis['category']}
            diversification_benefit = self.portfolio_optimizer.calculate_diversification_benefit(
                symbol, [], symbol_sectors
            )
            portfolio_fit_score = total_score * diversification_benefit
            
            # 액션 결정
            if total_score >= 0.75:
                action, confidence = 'buy', min(total_score, 0.95)
            elif total_score <= 0.25:
                action, confidence = 'sell', min(1 - total_score, 0.95)
            else:
                action, confidence = 'hold', 0.5
            
            # 5단계 분할매매 계획
            split_plan = self.split_trading_system.calculate_split_trading_plan(
                symbol, data['price'], confidence, self.current_cycle, self.target_coins
            )
            
            # 추론 생성
            reasoning = self._generate_enhanced_reasoning(
                fundamental_score, technical_score, momentum_score,
                quality_analysis, tech_details, self.current_cycle
            )
            
            return UltimateCoinSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=data['price'],
                total_score=total_score,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                momentum_score=momentum_score,
                quality_score=quality_analysis['quality_score'],
                market_cycle=self.current_cycle,
                cycle_confidence=self.cycle_confidence,
                sector=quality_analysis['category'],
                rsi=tech_details.get('rsi', 50),
                williams_r=tech_details.get('williams_r', -50),
                cci=tech_details.get('cci', 0),
                mfi=tech_details.get('mfi', 50),
                adx=tech_details.get('adx', 25),
                macd_signal=tech_details.get('macd_signal', 'neutral'),
                bb_position=tech_details.get('bb_position', 'normal'),
                ichimoku_signal=tech_details.get('ichimoku_signal', 'neutral'),
                momentum_3d=data.get('momentum_3d', 0),
                momentum_7d=data.get('momentum_7d', 0),
                momentum_30d=data.get('momentum_30d', 0),
                volume_spike_ratio=data.get('volume_spike_ratio', 1),
                btc_correlation=btc_correlation,
                portfolio_fit_score=portfolio_fit_score,
                diversification_benefit=diversification_benefit,
                position_stage=0,
                total_investment=split_plan.get('total_investment', 0),
                }
    }

# ============================================================================
# 🎯 메인 실행 함수들
# ============================================================================

async def main():
    """메인 실행 함수"""
    print("🪙 최고퀸트프로젝트 - 궁극의 암호화폐 전략 (완전체 V5.0)")
    print("🔧 6단계정밀필터링 + 고급기술지표 + 5단계분할매매 + 상관관계최적화")
    print("="*80)
    
    # 환경 확인
    if not os.path.exists('settings.yaml'):
        print("⚠️ settings.yaml 파일이 없습니다. 기본 설정으로 진행합니다.")
    
    # 전략 정보 출력
    info = get_ultimate_strategy_info()
    print(f"\n📋 궁극의 전략 정보:")
    print(f"   이름: {info['name']}")
    print(f"   버전: {info['version']}")
    print(f"   주요 기능: {len(info['features'])}개")
    print(f"   기술지표: 기본 {len(info['technical_indicators']['basic'])}개 + "
          f"고급 {len(info['technical_indicators']['advanced'])}개")
    print(f"   필터링: {len(info['filtering_stages'])}단계")
    print(f"   분할매매: {info['split_trading']['stages']}단계")
    
    # 완전체 테스트 실행
    await test_ultimate_strategy()

def run_single_coin_analysis(symbol: str):
    """단일 코인 분석 실행 (동기 버전)"""
    return asyncio.run(analyze_single_coin(symbol))

def run_full_strategy():
    """전체 전략 실행 (동기 버전)"""
    return asyncio.run(run_ultimate_analysis())

def run_test():
    """테스트 실행 (동기 버전)"""
    return asyncio.run(test_ultimate_strategy())

# ============================================================================
# 📱 사용 예시 함수들
# ============================================================================

def example_usage():
    """사용 예시"""
    print("\n📖 사용 예시:")
    print("="*50)
    
    print("\n1️⃣ 단일 코인 분석:")
    print("python -c \"from coin_strategy import run_single_coin_analysis; "
          "print(run_single_coin_analysis('KRW-BTC'))\"")
    
    print("\n2️⃣ 전체 전략 실행:")
    print("python -c \"from coin_strategy import run_full_strategy; "
          "signals, report = run_full_strategy(); "
          "print(f'매수 신호: {len([s for s in signals if s.action == \\\"buy\\\"])}개')\"")
    
    print("\n3️⃣ 테스트 실행:")
    print("python -c \"from coin_strategy import run_test; run_test()\"")
    
    print("\n4️⃣ 메인 실행:")
    print("python coin_strategy.py")

if __name__ == "__main__":
    # 로그 폴더 생성
    os.makedirs('logs', exist_ok=True)
    
    # 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 궁극의 전략을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n📖 다음에도 최고퀸트프로젝트를 이용해 주세요!")
        example_usage()
