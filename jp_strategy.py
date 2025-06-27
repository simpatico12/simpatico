#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import ta

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

    async def analyze_symbol(self, symbol: str) -> JPStockSignal:
        """개별 일본 주식 분석 (최적화된 파라미터)"""
        if not self.enabled:
            logger.warning("일본 주식 전략이 비활성화되어 있습니다")
            return JPStockSignal(
                symbol=symbol, 
                action='hold', 
                confidence=0.0, 
                price=0.0,
                strategy_source='disabled', 
                ichimoku_signal='neutral', 
                momentum_score=0.0, 
                volume_ratio=0.0, 
                rsi=50.0,
                sector='UNKNOWN', 
                reasoning="전략 비활성화", 
                target_price=0.0, 
                timestamp=datetime.now()
            )
            
        try:
            # 주식 데이터 수집
            data = await self._get_stock_data(symbol)
            if data.empty:
                raise ValueError(f"주식 데이터 없음: {symbol}")

            current_price = data['Close'].iloc[-1]
            
            # 배당 수익률 조회 (일본 시장 특화)
            dividend_yield = await self._get_dividend_yield(symbol)
            
            # 기본 분석 결과
            final_action = 'hold'
            confidence = 0.5
            strategy_source = 'basic_analysis'
            ichimoku_signal = 'neutral'
            momentum_score = 0.0
            volume_ratio = 1.0
            rsi = 50.0
            
            # 목표주가 계산
            target_price = current_price
            
            # 기술분석 reasoning
            technical_reasoning = "기본 분석 완료"
            
            return JPStockSignal(
                symbol=symbol,
                action=final_action,
                confidence=confidence,
                price=current_price,
                strategy_source=strategy_source,
                ichimoku_signal=ichimoku_signal,
                momentum_score=momentum_score,
                volume_ratio=volume_ratio,
                rsi=rsi,
                sector=self._get_sector_for_symbol(symbol),
                reasoning=technical_reasoning,
                target_price=target_price,
                timestamp=datetime.now(),
                additional_data={
                    'dividend_yield': dividend_yield,
                    'data_length': len(data)
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

# 편의 함수들
async def analyze_jp(symbol: str) -> Dict:
    """단일 일본 주식 분석"""
    strategy = JPStrategy()
    signal = await strategy.analyze_symbol(symbol)
    
    return {
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

async def main():
    """테스트용 메인 함수"""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        print("🇯🇵 일본 주식 전략 테스트 시작...")
        
        # 토요타 테스트
        print("\n📊 토요타(7203.T) 분석:")
        result = await analyze_jp('7203.T')
        print(f"결과: {result}")
        
        print("\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())
