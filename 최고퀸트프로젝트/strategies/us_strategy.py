"""
🇺🇸 미국 주식 전략 모듈 - 최고퀸트프로젝트
==========================================

전설적 투자자들의 전략 구현:
- 워렌 버핏 가치투자 (PBR, ROE, Debt Ratio)
- 피터 린치 성장투자 (PEG, EPS Growth)
- 섹터 로테이션 전략
- 거시경제 지표 통합
- 실시간 뉴스 센티먼트

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
import requests

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class USStockSignal:
    """미국 주식 시그널 데이터 클래스"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 ~ 1.0
    price: float
    strategy_source: str  # 'buffett', 'lynch', 'sector_rotation'
    pbr: float
    peg: float
    pe_ratio: float
    roe: float
    sector: str
    reasoning: str
    target_price: float
    timestamp: datetime

class USStrategy:
    """🇺🇸 고급 미국 주식 전략 클래스"""
    
    def __init__(self, config_path: str = "configs/settings.yaml"):
        """전략 초기화"""
        self.config = self._load_config(config_path)
        self.us_config = self.config.get('us_strategy', {})
        
        # settings.yaml에서 설정값 읽기
        self.enabled = self.us_config.get('enabled', True)
        self.buffett_pbr_limit = self.us_config.get('buffett_pbr', 1.5)
        self.lynch_peg_limit = self.us_config.get('lynch_peg', 1.0)
        
        # 추가 파라미터 (기본값)
        self.buffett_roe_min = 15.0  # ROE 최소 기준
        self.buffett_debt_ratio_max = 0.4  # 부채비율 최대
        self.lynch_growth_min = 10.0  # 성장률 최소 기준
        
        # 추적할 주요 미국 주식 (다양한 섹터)
        self.symbols = {
            'TECH': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA'],
            'FINANCE': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
            'HEALTHCARE': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO'],
            'CONSUMER': ['HD', 'MCD', 'WMT', 'PG', 'KO', 'NKE'],
            'ENERGY': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'INDUSTRIAL': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS']
        }
        
        # 모든 심볼을 플랫 리스트로
        self.all_symbols = [symbol for sector_symbols in self.symbols.values() 
                           for symbol in sector_symbols]
        
        if self.enabled:
            logger.info(f"🇺🇸 미국 주식 전략 초기화 완료 - 추적 종목: {len(self.all_symbols)}개")
            logger.info(f"📊 버핏 PBR 기준: {self.buffett_pbr_limit}, 린치 PEG 기준: {self.lynch_peg_limit}")
        else:
            logger.info("🇺🇸 미국 주식 전략이 비활성화되어 있습니다")

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

    async def _get_financial_data(self, symbol: str) -> Dict:
        """재무 데이터 수집"""
        try:
            stock = yf.Ticker(symbol)
            
            # 기본 정보
            info = stock.info
            
            # 주요 지표 추출
            data = {
                'price': info.get('currentPrice', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pbr': info.get('priceToBook', 0),
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'eps_growth': info.get('earningsQuarterlyGrowth', 0) * 100 if info.get('earningsQuarterlyGrowth') else 0,
                'revenue_growth': info.get('revenueQuarterlyGrowth', 0) * 100 if info.get('revenueQuarterlyGrowth') else 0,
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'gross_margin': info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0,
                'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0,
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            # PEG 비율 계산
            if data['pe_ratio'] > 0 and data['eps_growth'] > 0:
                data['peg'] = data['pe_ratio'] / data['eps_growth']
            else:
                data['peg'] = 0
                
            return data
            
        except Exception as e:
            logger.error(f"재무 데이터 수집 실패 {symbol}: {e}")
            return {}

    def _buffett_analysis(self, symbol: str, data: Dict) -> Tuple[str, float, str]:
        """워렌 버핏 가치투자 분석"""
        score = 0.0
        reasoning = []
        
        # 1. PBR 체크 (낮을수록 좋음)
        pbr = data.get('pbr', 999)
        if 0 < pbr <= self.buffett_pbr_limit:
            score += 0.3
            reasoning.append(f"저PBR({pbr:.2f})")
        elif pbr > self.buffett_pbr_limit:
            reasoning.append(f"고PBR({pbr:.2f})")
        
        # 2. ROE 체크 (높을수록 좋음) 
        roe = data.get('roe', 0)
        if roe >= self.buffett_roe_min:
            score += 0.25
            reasoning.append(f"고ROE({roe:.1f}%)")
        elif roe > 0:
            reasoning.append(f"저ROE({roe:.1f}%)")
            
        # 3. 부채비율 체크 (낮을수록 좋음)
        debt_ratio = data.get('debt_to_equity', 999) / 100 if data.get('debt_to_equity') else 999
        if debt_ratio <= self.buffett_debt_ratio_max:
            score += 0.2
            reasoning.append(f"저부채({debt_ratio:.2f})")
        else:
            reasoning.append(f"고부채({debt_ratio:.2f})")
            
        # 4. 배당수익률 (보너스)
        dividend = data.get('dividend_yield', 0)
        if dividend > 2.0:
            score += 0.1
            reasoning.append(f"배당({dividend:.1f}%)")
            
        # 5. 현재비율 (유동성)
        current_ratio = data.get('current_ratio', 0)
        if current_ratio > 1.5:
            score += 0.15
            reasoning.append("유동성양호")
            
        # 결정
        if score >= 0.7:
            action = 'buy'
        elif score <= 0.3:
            action = 'sell'
        else:
            action = 'hold'
            
        return action, score, "버핏: " + " | ".join(reasoning)

    def _lynch_analysis(self, symbol: str, data: Dict) -> Tuple[str, float, str]:
        """피터 린치 성장투자 분석"""
        score = 0.0
        reasoning = []
        
        # 1. PEG 비율 체크 (낮을수록 좋음)
        peg = data.get('peg', 999)
        if 0 < peg <= self.lynch_peg_limit:
            score += 0.35
            reasoning.append(f"저PEG({peg:.2f})")
        elif peg > self.lynch_peg_limit:
            reasoning.append(f"고PEG({peg:.2f})")
            
        # 2. EPS 성장률 체크
        eps_growth = data.get('eps_growth', 0)
        if eps_growth >= self.lynch_growth_min:
            score += 0.3
            reasoning.append(f"고성장({eps_growth:.1f}%)")
        elif eps_growth > 0:
            score += 0.1
            reasoning.append(f"저성장({eps_growth:.1f}%)")
            
        # 3. 매출 성장률 체크
        revenue_growth = data.get('revenue_growth', 0)
        if revenue_growth >= 10.0:
            score += 0.2
            reasoning.append(f"매출성장({revenue_growth:.1f}%)")
            
        # 4. 영업이익률 체크
        operating_margin = data.get('operating_margin', 0)
        if operating_margin >= 15.0:
            score += 0.15
            reasoning.append(f"고수익성({operating_margin:.1f}%)")
            
        # 결정
        if score >= 0.7:
            action = 'buy'
        elif score <= 0.3:
            action = 'sell'  
        else:
            action = 'hold'
            
        return action, score, "린치: " + " | ".join(reasoning)

    def _calculate_target_price(self, data: Dict, confidence: float) -> float:
        """목표주가 계산"""
        current_price = data.get('price', 0)
        if current_price == 0:
            return 0
            
        # 기대수익률 = 신뢰도 * 베이스 수익률
        expected_return = confidence * 0.2  # 최대 20% 수익 기대
        
        return current_price * (1 + expected_return)

    async def analyze_symbol(self, symbol: str) -> USStockSignal:
        """개별 미국 주식 분석"""
        if not self.enabled:
            logger.warning("미국 주식 전략이 비활성화되어 있습니다")
            return USStockSignal(
                symbol=symbol, action='hold', confidence=0.0, price=0.0,
                strategy_source='disabled', pbr=0.0, peg=0.0, pe_ratio=0.0,
                roe=0.0, sector='UNKNOWN', reasoning="전략 비활성화", 
                target_price=0.0, timestamp=datetime.now()
            )
            
        try:
            # 재무 데이터 수집
            data = await self._get_financial_data(symbol)
            if not data:
                raise ValueError(f"재무 데이터 없음: {symbol}")

            # 버핏 전략 분석
            buffett_action, buffett_score, buffett_reasoning = self._buffett_analysis(symbol, data)
            
            # 린치 전략 분석  
            lynch_action, lynch_score, lynch_reasoning = self._lynch_analysis(symbol, data)
            
            # 종합 판단 (가중평균)
            buffett_weight = 0.6  # 버핏 전략 60%
            lynch_weight = 0.4    # 린치 전략 40%
            
            total_score = buffett_score * buffett_weight + lynch_score * lynch_weight
            
            # 최종 액션 결정
            if total_score >= 0.7:
                final_action = 'buy'
                confidence = min(total_score, 0.95)
                strategy_source = 'buffett' if buffett_score > lynch_score else 'lynch'
            elif total_score <= 0.3:
                final_action = 'sell'
                confidence = min(1 - total_score, 0.95)
                strategy_source = 'risk_management'
            else:
                final_action = 'hold'
                confidence = 0.5
                strategy_source = 'neutral'

            # 목표주가 계산
            target_price = self._calculate_target_price(data, confidence)
            
            # 종합 reasoning
            combined_reasoning = f"{buffett_reasoning} | {lynch_reasoning}"
            
            return USStockSignal(
                symbol=symbol,
                action=final_action,
                confidence=confidence,
                price=data.get('price', 0),
                strategy_source=strategy_source,
                pbr=data.get('pbr', 0),
                peg=data.get('peg', 0),
                pe_ratio=data.get('pe_ratio', 0),
                roe=data.get('roe', 0),
                sector=self._get_sector_for_symbol(symbol),
                reasoning=combined_reasoning,
                target_price=target_price,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"미국 주식 분석 실패 {symbol}: {e}")
            return USStockSignal(
                symbol=symbol,
                action='hold',
                confidence=0.0,
                price=0.0,
                strategy_source='error',
                pbr=0.0,
                peg=0.0,
                pe_ratio=0.0,
                roe=0.0,
                sector='UNKNOWN',
                reasoning=f"분석 실패: {str(e)}",
                target_price=0.0,
                timestamp=datetime.now()
            )

    async def scan_by_sector(self, sector: str) -> List[USStockSignal]:
        """섹터별 스캔"""
        if not self.enabled:
            logger.warning("미국 주식 전략이 비활성화되어 있습니다")
            return []
            
        if sector not in self.symbols:
            logger.error(f"알 수 없는 섹터: {sector}")
            return []
            
        logger.info(f"🔍 {sector} 섹터 스캔 시작...")
        symbols = self.symbols[sector]
        
        signals = []
        for symbol in symbols:
            try:
                signal = await self.analyze_symbol(symbol)
                signals.append(signal)
                logger.info(f"✅ {symbol}: {signal.action} (신뢰도: {signal.confidence:.2f})")
                
                # API 호출 제한 고려
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 분석 실패: {e}")

        return signals

    async def scan_all_symbols(self) -> List[USStockSignal]:
        """전체 심볼 스캔"""
        if not self.enabled:
            logger.warning("미국 주식 전략이 비활성화되어 있습니다")
            return []
            
        logger.info(f"🔍 {len(self.all_symbols)}개 미국 주식 스캔 시작...")
        
        all_signals = []
        for sector in self.symbols.keys():
            sector_signals = await self.scan_by_sector(sector)
            all_signals.extend(sector_signals)
            
            # 섹터간 대기
            await asyncio.sleep(1)

        logger.info(f"🎯 스캔 완료 - 매수:{len([s for s in all_signals if s.action=='buy'])}개, "
                   f"매도:{len([s for s in all_signals if s.action=='sell'])}개, "
                   f"보유:{len([s for s in all_signals if s.action=='hold'])}개")

        return all_signals

    async def get_top_picks(self, strategy: str = 'all', limit: int = 5) -> List[USStockSignal]:
        """상위 종목 추천"""
        all_signals = await self.scan_all_symbols()
        
        # 전략별 필터링
        if strategy == 'buffett':
            filtered = [s for s in all_signals if s.strategy_source == 'buffett']
        elif strategy == 'lynch':
            filtered = [s for s in all_signals if s.strategy_source == 'lynch']
        else:
            filtered = [s for s in all_signals if s.action == 'buy']
        
        # 신뢰도 순 정렬
        sorted_signals = sorted(filtered, key=lambda x: x.confidence, reverse=True)
        
        return sorted_signals[:limit]

# 편의 함수들 (core.py에서 호출용)
async def analyze_us(symbol: str) -> Dict:
    """단일 미국 주식 분석 (기존 인터페이스 호환)"""
    strategy = USStrategy()
    signal = await strategy.analyze_symbol(symbol)
    
    return {
        'decision': signal.action,
        'confidence_score': signal.confidence * 100,
        'reasoning': signal.reasoning,
        'target_price': signal.target_price,
        'pbr': signal.pbr,
        'peg': signal.peg,
        'price': signal.price,
        'sector': signal.sector
    }

async def get_buffett_picks() -> List[Dict]:
    """버핏 스타일 추천 종목"""
    strategy = USStrategy()
    signals = await strategy.get_top_picks('buffett', limit=10)
    
    picks = []
    for signal in signals:
        picks.append({
            'symbol': signal.symbol,
            'sector': signal.sector,
            'confidence': signal.confidence,
            'pbr': signal.pbr,
            'roe': signal.roe,
            'reasoning': signal.reasoning,
            'target_price': signal.target_price,
            'current_price': signal.price
        })
    
    return picks

async def get_lynch_picks() -> List[Dict]:
    """린치 스타일 추천 종목"""
    strategy = USStrategy()
    signals = await strategy.get_top_picks('lynch', limit=10)
    
    picks = []
    for signal in signals:
        picks.append({
            'symbol': signal.symbol,
            'sector': signal.sector,
            'confidence': signal.confidence,
            'peg': signal.peg,
            'eps_growth': signal.reasoning,
            'target_price': signal.target_price,
            'current_price': signal.price
        })
    
    return picks

async def scan_us_market() -> Dict:
    """미국 시장 전체 스캔"""
    strategy = USStrategy()
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
        print("🇺🇸 최고퀸트프로젝트 - 미국 주식 전략 테스트 시작...")
        
        # 단일 주식 테스트
        print("\n📊 AAPL 개별 분석:")
        aapl_result = await analyze_us('AAPL')
        print(f"AAPL: {aapl_result}")
        
        # 버핏 스타일 추천
        print("\n💰 워렌 버핏 스타일 추천:")
        buffett_picks = await get_buffett_picks()
        for pick in buffett_picks[:3]:
            print(f"{pick['symbol']}: 신뢰도 {pick['confidence']:.2f}, PBR {pick['pbr']:.2f}")
        
        # 린치 스타일 추천  
        print("\n🚀 피터 린치 스타일 추천:")
        lynch_picks = await get_lynch_picks()
        for pick in lynch_picks[:3]:
            print(f"{pick['symbol']}: 신뢰도 {pick['confidence']:.2f}, PEG {pick['peg']:.2f}")
    
    asyncio.run(main())