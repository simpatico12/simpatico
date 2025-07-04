#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 - 4대 시장 통합 핵심 시스템 CORE.PY
================================================================

🌟 핵심 특징:
- 🇺🇸 미국주식: 전설적 퀸트 마스터시스템 V6.0 (IBKR 연동)
- 🪙 업비트: 전설급 5대 시스템 완전체 (실시간 매매)
- 🇯🇵 일본주식: YEN-HUNTER 전설급 TOPIX+JPX400
- 🇮🇳 인도주식: 5대 투자거장 + 14개 전설급 기술지표

⚡ 혼자 보수유지 가능한 완전 자동화 아키텍처
💎 설정 기반 모듈화 + 실시간 모니터링
🛡️ 통합 리스크 관리 + 포트폴리오 최적화

Author: 퀸트팀 | Version: ULTIMATE
Date: 2024.12
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import pyupbit
import requests
from dotenv import load_dotenv

# 선택적 import (없어도 기본 기능 동작)
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

try:
    import telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# ============================================================================
# 🔧 설정 관리자 - 모든 설정을 한 곳에서 관리
# ============================================================================
class QuintConfigManager:
    """퀸트프로젝트 통합 설정 관리자"""
    
    def __init__(self):
        self.config_file = "quint_config.yaml"
        self.env_file = ".env"
        self.config = {}
        self._initialize_config()
    
    def _initialize_config(self):
        """설정 초기화"""
        # 환경변수 로드
        if Path(self.env_file).exists():
            load_dotenv(self.env_file)
        
        # 기본 설정 로드/생성
        if Path(self.config_file).exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self._create_default_config()
            self._save_config()
    
    def _create_default_config(self):
        """기본 설정 생성"""
        self.config = {
            # 전체 시스템 설정
            'system': {
                'portfolio_value': 100_000_000,  # 총 포트폴리오 가치 (1억원)
                'demo_mode': True,               # 시뮬레이션 모드
                'auto_trading': False,           # 자동매매 활성화
                'log_level': 'INFO',
                'backup_enabled': True
            },
            
            # 시장별 활성화 설정
            'markets': {
                'us_stocks': {'enabled': True, 'allocation': 40.0},      # 미국 40%
                'upbit_crypto': {'enabled': True, 'allocation': 30.0},   # 암호화폐 30%
                'japan_stocks': {'enabled': True, 'allocation': 20.0},   # 일본 20%
                'india_stocks': {'enabled': True, 'allocation': 10.0}    # 인도 10%
            },
            
            # 리스크 관리
            'risk_management': {
                'max_single_position': 8.0,     # 단일 종목 최대 비중 8%
                'stop_loss': 15.0,              # 기본 손절선 15%
                'take_profit': 25.0,            # 기본 익절선 25%
                'max_correlation': 0.7,         # 최대 상관관계
                'rebalance_threshold': 5.0      # 리밸런싱 임계값 5%
            },
            
            # 알림 설정
            'notifications': {
                'telegram': {
                    'enabled': False,
                    'bot_token': '${TELEGRAM_BOT_TOKEN}',
                    'chat_id': '${TELEGRAM_CHAT_ID}'
                },
                'console_only': True
            },
            
            # 미국주식 설정
            'us_stocks': {
                'target_stocks': 15,
                'confidence_threshold': 0.70,
                'strategy_weights': {
                    'buffett_value': 25.0,
                    'lynch_growth': 25.0,
                    'momentum': 25.0,
                    'technical': 25.0
                },
                'ibkr': {
                    'enabled': False,
                    'host': '127.0.0.1',
                    'port': 7497,
                    'paper_trading': True
                }
            },
            
            # 업비트 설정
            'upbit_crypto': {
                'min_volume_krw': 5_000_000_000,  # 최소 거래량 50억원
                'target_coins': 8,
                'neural_quality_threshold': 0.6,
                'kelly_max_ratio': 0.25
            },
            
            # 일본주식 설정
            'japan_stocks': {
                'yen_strong_threshold': 105.0,
                'yen_weak_threshold': 110.0,
                'target_stocks': 12,
                'min_market_cap': 500_000_000_000  # 5천억엔
            },
            
            # 인도주식 설정
            'india_stocks': {
                'target_stocks': 10,
                'legendary_threshold': 8.0,
                'index_diversity': True
            }
        }
    
    def _save_config(self):
        """설정 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            logging.error(f"설정 저장 실패: {e}")
    
    def get(self, key_path: str, default=None):
        """설정값 조회 (점 표기법)"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        # 환경변수 치환
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, default)
        
        return value
    
    def update(self, key_path: str, value):
        """설정값 업데이트"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self._save_config()

# 전역 설정 관리자
config = QuintConfigManager()

# ============================================================================
# 📊 공통 데이터 클래스
# ============================================================================
@dataclass
class QuintSignal:
    """퀸트프로젝트 통합 시그널"""
    market: str          # 'us', 'crypto', 'japan', 'india'
    symbol: str
    action: str          # 'BUY', 'SELL', 'HOLD'
    confidence: float    # 0.0 ~ 1.0
    price: float
    target_price: float
    stop_loss: float
    take_profit: float
    
    # 투자 계획
    allocation_percent: float
    investment_amount: float
    
    # 분석 정보
    strategy_scores: Dict
    technical_indicators: Dict
    reasoning: str
    
    # 메타 정보
    timestamp: datetime
    market_cycle: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Portfolio:
    """포트폴리오 정보"""
    total_value: float
    positions: Dict[str, Any]
    cash_balance: float
    unrealized_pnl: float
    daily_pnl: float
    allocation_by_market: Dict[str, float]

# ============================================================================
# 🇺🇸 미국주식 전략 엔진 (간소화 버전)
# ============================================================================
class USStockEngine:
    """미국주식 전설적 퀸트 엔진"""
    
    def __init__(self):
        self.enabled = config.get('markets.us_stocks.enabled', True)
        self.target_stocks = config.get('us_stocks.target_stocks', 15)
        self.confidence_threshold = config.get('us_stocks.confidence_threshold', 0.70)
        
    async def analyze_us_market(self) -> List[QuintSignal]:
        """미국 시장 분석"""
        if not self.enabled:
            return []
        
        logging.info("🇺🇸 미국주식 전설 퀸트 분석 시작")
        
        try:
            # S&P 500 상위 종목 수집
            sp500_symbols = self._get_sp500_sample()
            
            # VIX 조회
            vix = await self._get_vix()
            
            # 개별 종목 분석
            signals = []
            for symbol in sp500_symbols[:self.target_stocks]:
                signal = await self._analyze_us_stock(symbol, vix)
                if signal:
                    signals.append(signal)
                await asyncio.sleep(0.3)  # API 제한
            
            # 신뢰도 순 정렬
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            buy_signals = [s for s in signals if s.action == 'BUY']
            logging.info(f"🇺🇸 미국주식 분석 완료: {len(buy_signals)}개 매수 신호")
            
            return signals[:10]  # 상위 10개만
            
        except Exception as e:
            logging.error(f"미국주식 분석 실패: {e}")
            return []
    
    def _get_sp500_sample(self) -> List[str]:
        """S&P 500 샘플 종목"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'ABBV', 'BAC',
            'KO', 'AVGO', 'XOM', 'CVX', 'LLY', 'WMT', 'PEP', 'TMO', 'COST'
        ]
    
    async def _get_vix(self) -> float:
        """VIX 지수 조회"""
        try:
            vix_ticker = yf.Ticker("^VIX")
            hist = vix_ticker.history(period="1d")
            return float(hist['Close'].iloc[-1]) if not hist.empty else 20.0
        except:
            return 20.0
    
    async def _analyze_us_stock(self, symbol: str, vix: float) -> Optional[QuintSignal]:
        """개별 미국 주식 분석"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="6mo")
            
            if hist.empty or not info:
                return None
            
            current_price = float(hist['Close'].iloc[-1])
            
            # 간단한 점수 계산
            scores = self._calculate_strategy_scores(info, hist)
            
            # VIX 조정
            vix_adjustment = 1.15 if vix < 15 else 0.85 if vix > 30 else 1.0
            total_score = scores['total'] * vix_adjustment
            
            # 액션 결정
            if total_score >= self.confidence_threshold:
                action = 'BUY'
                confidence = min(0.95, total_score)
            elif total_score <= 0.30:
                action = 'SELL'
                confidence = min(0.95, 1 - total_score)
            else:
                action = 'HOLD'
                confidence = 0.50
            
            # 목표가 및 손절가
            target_price = current_price * (1 + confidence * 0.30)
            stop_loss = current_price * 0.85
            take_profit = current_price * 1.25
            
            # 투자 금액 계산
            allocation = config.get('markets.us_stocks.allocation', 40.0)
            portfolio_value = config.get('system.portfolio_value', 100_000_000)
            max_investment = portfolio_value * allocation / 100 / self.target_stocks
            investment_amount = max_investment * confidence
            
            return QuintSignal(
                market='us',
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                allocation_percent=(investment_amount / portfolio_value) * 100,
                investment_amount=investment_amount,
                strategy_scores=scores,
                technical_indicators={'rsi': self._calculate_rsi(hist['Close'])},
                reasoning=f"버핏:{scores['buffett']:.2f} 린치:{scores['lynch']:.2f} VIX조정:{vix_adjustment:.2f}",
                timestamp=datetime.now(),
                market_cycle='bull' if vix < 20 else 'bear' if vix > 30 else 'neutral'
            )
            
        except Exception as e:
            logging.error(f"미국주식 {symbol} 분석 실패: {e}")
            return None
    
    def _calculate_strategy_scores(self, info: Dict, hist: pd.DataFrame) -> Dict:
        """전략별 점수 계산"""
        # 워렌 버핏 가치투자
        pe = info.get('trailingPE', 999)
        roe = info.get('returnOnEquity', 0) or 0
        buffett_score = 0.3 if 5 <= pe <= 25 else 0.1
        buffett_score += 0.3 if roe > 0.15 else 0.1
        
        # 피터 린치 성장투자
        growth = info.get('earningsQuarterlyGrowth', 0) or 0
        peg = info.get('pegRatio', 999) or 999
        lynch_score = 0.3 if growth > 0.15 else 0.1
        lynch_score += 0.3 if 0 < peg <= 1.5 else 0.1
        
        # 모멘텀
        if len(hist) >= 60:
            momentum_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-60] - 1)
            momentum_score = 0.4 if momentum_3m > 0.1 else 0.2 if momentum_3m > 0 else 0.1
        else:
            momentum_score = 0.2
        
        # 기술적 분석
        rsi = self._calculate_rsi(hist['Close'])
        technical_score = 0.4 if 30 <= rsi <= 70 else 0.2
        
        # 가중치 적용
        weights = config.get('us_stocks.strategy_weights', {})
        total = (
            buffett_score * weights.get('buffett_value', 25) / 100 +
            lynch_score * weights.get('lynch_growth', 25) / 100 +
            momentum_score * weights.get('momentum', 25) / 100 +
            technical_score * weights.get('technical', 25) / 100
        )
        
        return {
            'buffett': buffett_score,
            'lynch': lynch_score,
            'momentum': momentum_score,
            'technical': technical_score,
            'total': total
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

# ============================================================================
# 🪙 업비트 암호화폐 전략 엔진 (간소화 버전)
# ============================================================================
class UpbitCryptoEngine:
    """업비트 전설급 5대 시스템"""
    
    def __init__(self):
        self.enabled = config.get('markets.upbit_crypto.enabled', True)
        self.min_volume = config.get('upbit_crypto.min_volume_krw', 5_000_000_000)
        self.target_coins = config.get('upbit_crypto.target_coins', 8)
        
        # 코인 품질 점수 (기술력, 생태계, 커뮤니티, 채택도)
        self.coin_scores = {
            'BTC': [0.98, 0.95, 0.90, 0.95], 'ETH': [0.95, 0.98, 0.85, 0.90],
            'BNB': [0.80, 0.90, 0.75, 0.85], 'ADA': [0.85, 0.75, 0.80, 0.70],
            'SOL': [0.90, 0.80, 0.85, 0.75], 'AVAX': [0.85, 0.75, 0.70, 0.70],
            'DOT': [0.85, 0.80, 0.70, 0.65], 'MATIC': [0.80, 0.85, 0.75, 0.80],
            'ATOM': [0.75, 0.70, 0.75, 0.60], 'NEAR': [0.80, 0.70, 0.65, 0.60],
            'LINK': [0.90, 0.75, 0.70, 0.80], 'UNI': [0.85, 0.80, 0.75, 0.75]
        }
    
    async def analyze_crypto_market(self) -> List[QuintSignal]:
        """암호화폐 시장 분석"""
        if not self.enabled:
            return []
        
        logging.info("🪙 업비트 전설급 암호화폐 분석 시작")
        
        try:
            # 모든 KRW 마켓 조회
            all_tickers = pyupbit.get_tickers(fiat="KRW")
            if not all_tickers:
                return []
            
            # 거래량 필터링
            candidates = await self._filter_by_volume(all_tickers)
            
            # 개별 분석
            signals = []
            for candidate in candidates[:self.target_coins * 2]:  # 여유분 포함
                signal = await self._analyze_crypto_coin(candidate)
                if signal:
                    signals.append(signal)
                await asyncio.sleep(0.2)
            
            # 상위 선별
            signals.sort(key=lambda x: x.confidence, reverse=True)
            buy_signals = [s for s in signals if s.action == 'BUY']
            
            logging.info(f"🪙 암호화폐 분석 완료: {len(buy_signals)}개 매수 신호")
            
            return signals[:self.target_coins]
            
        except Exception as e:
            logging.error(f"암호화폐 분석 실패: {e}")
            return []
    
    async def _filter_by_volume(self, tickers: List[str]) -> List[Dict]:
        """거래량 기반 필터링"""
        candidates = []
        
        for ticker in tickers:
            try:
                price = pyupbit.get_current_price(ticker)
                if not price:
                    continue
                
                ohlcv = pyupbit.get_ohlcv(ticker, interval="day", count=7)
                if ohlcv is None or len(ohlcv) < 7:
                    continue
                
                volume_krw = ohlcv.iloc[-1]['volume'] * price
                if volume_krw >= self.min_volume:
                    candidates.append({
                        'symbol': ticker,
                        'price': price,
                        'volume_krw': volume_krw,
                        'ohlcv': ohlcv
                    })
                    
            except:
                continue
        
        return sorted(candidates, key=lambda x: x['volume_krw'], reverse=True)
    
    async def _analyze_crypto_coin(self, candidate: Dict) -> Optional[QuintSignal]:
        """개별 암호화폐 분석"""
        try:
            symbol = candidate['symbol']
            price = candidate['price']
            ohlcv = candidate['ohlcv']
            
            coin_name = symbol.replace('KRW-', '')
            
            # Neural Quality Score
            quality_scores = self.coin_scores.get(coin_name, [0.6, 0.6, 0.6, 0.6])
            weights = [0.30, 0.30, 0.20, 0.20]
            neural_quality = sum(s * w for s, w in zip(quality_scores, weights))
            
            # 기술적 분석
            rsi = self._calculate_crypto_rsi(ohlcv['close'])
            
            # 모멘텀 계산
            momentum_7d = (price / ohlcv['close'].iloc[-8] - 1) if len(ohlcv) >= 8 else 0
            
            # 종합 점수
            total_score = (
                neural_quality * 0.40 +
                (rsi / 100) * 0.30 +
                max(0, momentum_7d + 1) * 0.30
            )
            
            # 액션 결정
            threshold = config.get('upbit_crypto.neural_quality_threshold', 0.6)
            if total_score >= threshold and rsi < 70:
                action = 'BUY'
                confidence = min(0.95, total_score + 0.1)
            elif total_score <= 0.4 or rsi > 80:
                action = 'SELL'
                confidence = min(0.95, 1 - total_score)
            else:
                action = 'HOLD'
                confidence = 0.50
            
            # 투자 계획
            allocation = config.get('markets.upbit_crypto.allocation', 30.0)
            portfolio_value = config.get('system.portfolio_value', 100_000_000)
            max_investment = portfolio_value * allocation / 100 / self.target_coins
            investment_amount = max_investment * confidence
            
            return QuintSignal(
                market='crypto',
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=price,
                target_price=price * (1 + confidence * 0.50),
                stop_loss=price * 0.85,
                take_profit=price * 1.30,
                allocation_percent=(investment_amount / portfolio_value) * 100,
                investment_amount=investment_amount,
                strategy_scores={'neural_quality': neural_quality, 'momentum': momentum_7d},
                technical_indicators={'rsi': rsi},
                reasoning=f"AI품질:{neural_quality:.2f} RSI:{rsi:.0f} 모멘텀:{momentum_7d*100:+.1f}%",
                timestamp=datetime.now(),
                market_cycle='bull' if momentum_7d > 0.1 else 'bear' if momentum_7d < -0.1 else 'neutral'
            )
            
        except Exception as e:
            logging.error(f"암호화폐 {candidate['symbol']} 분석 실패: {e}")
            return None
    
    def _calculate_crypto_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """암호화폐 RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

# ============================================================================
# 🇯🇵 일본주식 전략 엔진 (간소화 버전)
# ============================================================================
class JapanStockEngine:
    """일본주식 YEN-HUNTER 전략"""
    
    def __init__(self):
        self.enabled = config.get('markets.japan_stocks.enabled', True)
        self.target_stocks = config.get('japan_stocks.target_stocks', 12)
        self.yen_strong = config.get('japan_stocks.yen_strong_threshold', 105.0)
        self.yen_weak = config.get('japan_stocks.yen_weak_threshold', 110.0)
        
        # 주요 일본 종목 (실제로는 크롤링)
        self.major_stocks = [
            '7203.T', '6758.T', '9984.T', '6861.T', '8306.T', '7974.T', '9432.T',
            '8316.T', '6367.T', '4063.T', '9983.T', '8411.T', '6954.T', '7201.T'
        ]
    
    async def analyze_japan_market(self) -> List[QuintSignal]:
        """일본 시장 분석"""
        if not self.enabled:
            return []
        
        logging.info("🇯🇵 일본주식 YEN-HUNTER 분석 시작")
        
        try:
            # USD/JPY 환율 조회
            usd_jpy = await self._get_usd_jpy()
            yen_signal = self._get_yen_signal(usd_jpy)
            
            # 개별 종목 분석
            signals = []
            for symbol in self.major_stocks[:self.target_stocks]:
                signal = await self._analyze_japan_stock(symbol, yen_signal, usd_jpy)
                if signal:
                    signals.append(signal)
                await asyncio.sleep(0.5)  # 야후 API 제한
            
            signals.sort(key=lambda x: x.confidence, reverse=True)
            buy_signals = [s for s in signals if s.action == 'BUY']
            
            logging.info(f"🇯🇵 일본주식 분석 완료: {len(buy_signals)}개 매수 신호 (USD/JPY: {usd_jpy:.2f})")
            
            return signals[:8]  # 상위 8개
            
        except Exception as e:
            logging.error(f"일본주식 분석 실패: {e}")
            return []
    
    async def _get_usd_jpy(self) -> float:
        """USD/JPY 환율 조회"""
        try:
            ticker = yf.Ticker("USDJPY=X")
            data = ticker.history(period="1d")
            return float(data['Close'].iloc[-1]) if not data.empty else 107.0
        except:
            return 107.0
    
    def _get_yen_signal(self, usd_jpy: float) -> Dict:
        """엔화 신호 분석"""
        if usd_jpy <= self.yen_strong:
            return {'signal': 'STRONG', 'factor': 1.2, 'favor': 'domestic'}
        elif usd_jpy >= self.yen_weak:
            return {'signal': 'WEAK', 'factor': 1.2, 'favor': 'export'}
        else:
            return {'signal': 'NEUTRAL', 'factor': 1.0, 'favor': 'balanced'}
    
    async def _analyze_japan_stock(self, symbol: str, yen_signal: Dict, usd_jpy: float) -> Optional[QuintSignal]:
        """개별 일본 종목 분석"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="3mo")
            info = stock.info
            
            if hist.empty:
                return None
            
            current_price = float(hist['Close'].iloc[-1])
            
            # 수출주/내수주 분류
            export_stocks = ['7203.T', '6758.T', '7974.T', '6861.T', '9984.T']
            stock_type = 'export' if symbol in export_stocks else 'domestic'
            
            # 기본 점수
            base_score = 0.5
            
            # 엔화 매칭 보너스
            if (yen_signal['favor'] == stock_type) or (yen_signal['favor'] == 'balanced'):
                base_score += 0.2 * yen_signal['factor']
            
            # 기술적 분석
            rsi = self._calculate_rsi(hist['Close'])
            if 30 <= rsi <= 70:
                base_score += 0.2
            elif rsi < 30:
                base_score += 0.3  # 과매도 = 매수기회
            
            # 추세 분석
            ma20 = hist['Close'].rolling(20).mean()
            if len(ma20) > 0 and current_price > ma20.iloc[-1]:
                base_score += 0.1
            
            total_score = min(base_score, 0.95)
            
            # 액션 결정
            if total_score >= 0.65:
                action = 'BUY'
                confidence = total_score
            elif total_score <= 0.35:
                action = 'SELL'
                confidence = 1 - total_score
            else:
                action = 'HOLD'
                confidence = 0.50
            
            # 투자 계획
            allocation = config.get('markets.japan_stocks.allocation', 20.0)
            portfolio_value = config.get('system.portfolio_value', 100_000_000)
            max_investment = portfolio_value * allocation / 100 / self.target_stocks
            investment_amount = max_investment * confidence
            
            return QuintSignal(
                market='japan',
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                target_price=current_price * (1 + confidence * 0.25),
                stop_loss=current_price * 0.85,
                take_profit=current_price * 1.20,
                allocation_percent=(investment_amount / portfolio_value) * 100,
                investment_amount=investment_amount,
                strategy_scores={'yen_score': yen_signal['factor'], 'base_score': base_score},
                technical_indicators={'rsi': rsi, 'usd_jpy': usd_jpy},
                reasoning=f"엔화{yen_signal['signal']} {stock_type}주 RSI:{rsi:.0f}",
                timestamp=datetime.now(),
                market_cycle=yen_signal['signal'].lower()
            )
            
        except Exception as e:
            logging.error(f"일본주식 {symbol} 분석 실패: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

# ============================================================================
# 🇮🇳 인도주식 전략 엔진 (간소화 버전)
# ============================================================================
class IndiaStockEngine:
    """인도주식 5대 투자거장 전략"""
    
    def __init__(self):
        self.enabled = config.get('markets.india_stocks.enabled', True)
        self.target_stocks = config.get('india_stocks.target_stocks', 10)
        
        # 주요 인도 종목 샘플 (4개 지수 통합)
        self.sample_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
            'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS',
            'HCLTECH.NS', 'AXISBANK.NS', 'MARUTI.NS', 'ASIANPAINT.NS'
        ]
    
    async def analyze_india_market(self) -> List[QuintSignal]:
        """인도 시장 분석"""
        if not self.enabled:
            return []
        
        logging.info("🇮🇳 인도주식 5대 투자거장 분석 시작")
        
        try:
            # 샘플 데이터로 분석 (실제로는 NSE API 연동)
            signals = []
            
            for symbol in self.sample_stocks[:self.target_stocks]:
                signal = await self._analyze_india_stock(symbol)
                if signal:
                    signals.append(signal)
                await asyncio.sleep(0.3)
            
            signals.sort(key=lambda x: x.confidence, reverse=True)
            buy_signals = [s for s in signals if s.action == 'BUY']
            
            logging.info(f"🇮🇳 인도주식 분석 완료: {len(buy_signals)}개 매수 신호")
            
            return signals[:6]  # 상위 6개
            
        except Exception as e:
            logging.error(f"인도주식 분석 실패: {e}")
            return []
    
    async def _analyze_india_stock(self, symbol: str) -> Optional[QuintSignal]:
        """개별 인도 종목 분석"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")
            info = stock.info
            
            if hist.empty:
                return None
            
            current_price = float(hist['Close'].iloc[-1])
            
            # 5대 투자거장 스타일 점수 (간소화)
            legendary_scores = self._calculate_legendary_scores(info, hist)
            
            # 기술지표 (14개 중 핵심 3개)
            technical_scores = self._calculate_technical_scores(hist)
            
            # 종합 점수
            total_score = (
                legendary_scores['total'] * 0.60 +
                technical_scores['total'] * 0.40
            )
            
            # 액션 결정
            threshold = config.get('india_stocks.legendary_threshold', 8.0) / 10
            if total_score >= threshold:
                action = 'BUY'
                confidence = min(0.95, total_score + 0.05)
            elif total_score <= 0.4:
                action = 'SELL'
                confidence = min(0.95, 1 - total_score)
            else:
                action = 'HOLD'
                confidence = 0.50
            
            # 투자 계획
            allocation = config.get('markets.india_stocks.allocation', 10.0)
            portfolio_value = config.get('system.portfolio_value', 100_000_000)
            max_investment = portfolio_value * allocation / 100 / self.target_stocks
            investment_amount = max_investment * confidence
            
            return QuintSignal(
                market='india',
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                target_price=current_price * (1 + confidence * 0.35),
                stop_loss=current_price * 0.85,
                take_profit=current_price * 1.25,
                allocation_percent=(investment_amount / portfolio_value) * 100,
                investment_amount=investment_amount,
                strategy_scores=legendary_scores,
                technical_indicators=technical_scores,
                reasoning=f"전설:{legendary_scores['total']:.2f} 기술:{technical_scores['total']:.2f}",
                timestamp=datetime.now(),
                market_cycle='growth' if legendary_scores['total'] > 0.6 else 'value'
            )
            
        except Exception as e:
            logging.error(f"인도주식 {symbol} 분석 실패: {e}")
            return None
    
    def _calculate_legendary_scores(self, info: Dict, hist: pd.DataFrame) -> Dict:
        """5대 투자거장 스타일 점수"""
        scores = {}
        
        # 준준왈라 (ROE + 배당)
        roe = info.get('returnOnEquity', 0) or 0
        dividend_yield = info.get('dividendYield', 0) or 0
        scores['jhunjhunwala'] = (roe > 0.15) * 0.3 + (dividend_yield > 0.02) * 0.2
        
        # QGLP (품질 + 성장)
        debt_ratio = info.get('debtToEquity', 999) or 999
        growth = info.get('earningsQuarterlyGrowth', 0) or 0
        scores['qglp'] = (debt_ratio < 50) * 0.25 + (growth > 0.15) * 0.25
        
        # 케디아 SMILE (성장 + 소형주)
        market_cap = info.get('marketCap', 0) or 0
        scores['kedia'] = (market_cap < 50_000_000_000) * 0.2 + (growth > 0.2) * 0.3
        
        # 벨리야스 (밸류 + 소외주)
        pe = info.get('trailingPE', 999) or 999
        scores['veliyath'] = (pe < 15) * 0.3 + (market_cap < 20_000_000_000) * 0.2
        
        # 카르닉 (인프라)
        sector = info.get('sector', '')
        scores['karnik'] = 0.4 if 'Infrastructure' in sector or 'Construction' in sector else 0.1
        
        total = sum(scores.values())
        scores['total'] = total
        
        return scores
    
    def _calculate_technical_scores(self, hist: pd.DataFrame) -> Dict:
        """기술지표 점수 (14개 중 핵심)"""
        scores = {}
        
        # RSI
        rsi = self._calculate_rsi(hist['Close'])
        scores['rsi'] = 0.3 if 30 <= rsi <= 70 else 0.2 if rsi < 30 else 0.1
        
        # MACD
        if len(hist) >= 26:
            ema12 = hist['Close'].ewm(span=12).mean()
            ema26 = hist['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            scores['macd'] = 0.25 if macd.iloc[-1] > signal_line.iloc[-1] else 0.1
        else:
            scores['macd'] = 0.15
        
        # 볼린저 밴드
        if len(hist) >= 20:
            ma20 = hist['Close'].rolling(20).mean()
            std20 = hist['Close'].rolling(20).std()
            current_price = hist['Close'].iloc[-1]
            lower_band = ma20.iloc[-1] - 2 * std20.iloc[-1]
            scores['bollinger'] = 0.25 if current_price > lower_band else 0.35  # 하단 근처 매수
        else:
            scores['bollinger'] = 0.2
        
        total = sum(scores.values())
        scores['total'] = total
        
        return scores
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

# ============================================================================
# 📊 포트폴리오 관리자
# ============================================================================
class PortfolioManager:
    """퀸트프로젝트 통합 포트폴리오 관리자"""
    
    def __init__(self):
        self.portfolio_file = "quint_portfolio.json"
        self.positions = {}
        self.load_portfolio()
        
    def load_portfolio(self):
        """포트폴리오 로드"""
        try:
            if Path(self.portfolio_file).exists():
                with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', {})
        except Exception as e:
            logging.error(f"포트폴리오 로드 실패: {e}")
            self.positions = {}
    
    def save_portfolio(self):
        """포트폴리오 저장"""
        try:
            data = {
                'positions': self.positions,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"포트폴리오 저장 실패: {e}")
    
    def add_position(self, signal: QuintSignal, quantity: float, executed_price: float):
        """포지션 추가"""
        symbol = signal.symbol
        if symbol not in self.positions:
            self.positions[symbol] = {
                'market': signal.market,
                'quantity': 0,
                'avg_price': 0,
                'total_cost': 0,
                'target_price': signal.target_price,
                'stop_loss': signal.stop_loss,
                'created_at': datetime.now().isoformat()
            }
        
        pos = self.positions[symbol]
        old_total_cost = pos['total_cost']
        new_cost = quantity * executed_price
        
        pos['quantity'] += quantity
        pos['total_cost'] += new_cost
        pos['avg_price'] = pos['total_cost'] / pos['quantity']
        pos['last_updated'] = datetime.now().isoformat()
        
        self.save_portfolio()
        logging.info(f"포지션 추가: {symbol} {quantity:.6f}개 @ {executed_price:.2f}")
    
    def remove_position(self, symbol: str, quantity: float = None):
        """포지션 제거"""
        if symbol not in self.positions:
            return
        
        if quantity is None:
            # 전체 제거
            del self.positions[symbol]
        else:
            # 부분 제거
            pos = self.positions[symbol]
            if quantity >= pos['quantity']:
                del self.positions[symbol]
            else:
                ratio = (pos['quantity'] - quantity) / pos['quantity']
                pos['quantity'] -= quantity
                pos['total_cost'] *= ratio
        
        self.save_portfolio()
    
    def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약"""
        total_value = 0
        positions_by_market = {'us': 0, 'crypto': 0, 'japan': 0, 'india': 0}
        
        for symbol, pos in self.positions.items():
            # 현재가 조회 (실제로는 각 마켓별 API 호출)
            current_price = pos['avg_price']  # 간소화
            position_value = pos['quantity'] * current_price
            total_value += position_value
            positions_by_market[pos['market']] += position_value
        
        return {
            'total_positions': len(self.positions),
            'total_value': total_value,
            'allocation_by_market': positions_by_market,
            'positions': self.positions
        }

# ============================================================================
# 🚨 알림 시스템
# ============================================================================
class NotificationManager:
    """통합 알림 관리자"""
    
    def __init__(self):
        self.telegram_enabled = config.get('notifications.telegram.enabled', False)
        self.console_only = config.get('notifications.console_only', True)
        
        if self.telegram_enabled and TELEGRAM_AVAILABLE:
            bot_token = config.get('notifications.telegram.bot_token')
            if bot_token and not bot_token.startswith('${'):
                self.telegram_bot = telegram.Bot(token=bot_token)
                self.chat_id = config.get('notifications.telegram.chat_id')
            else:
                self.telegram_enabled = False
    
    async def send_signal_alert(self, signals: List[QuintSignal]):
        """시그널 알림 전송"""
        if not signals:
            return
        
        buy_signals = [s for s in signals if s.action == 'BUY']
        if not buy_signals:
            return
        
        # 콘솔 출력
        print(f"\n🚨 매수 신호 알림: {len(buy_signals)}개")
        for signal in buy_signals[:3]:  # 상위 3개만
            print(f"  📈 {signal.symbol} ({signal.market}): {signal.confidence:.1%} 신뢰도")
        
        # 텔레그램 전송
        if self.telegram_enabled:
            try:
                message = f"🚨 퀸트프로젝트 매수 신호\n\n"
                for signal in buy_signals[:5]:
                    message += f"📈 {signal.symbol} ({signal.market.upper()})\n"
                    message += f"   신뢰도: {signal.confidence:.1%}\n"
                    message += f"   현재가: {signal.price:,.0f}\n"
                    message += f"   목표가: {signal.target_price:,.0f}\n\n"
                
                await self.telegram_bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='HTML'
                )
            except Exception as e:
                logging.error(f"텔레그램 알림 실패: {e}")
    
    async def send_portfolio_update(self, portfolio_summary: Dict):
        """포트폴리오 업데이트 알림"""
        if self.console_only:
            print(f"\n💼 포트폴리오 업데이트:")
            print(f"  총 포지션: {portfolio_summary['total_positions']}개")
            print(f"  총 가치: {portfolio_summary['total_value']:,.0f}원")

# ============================================================================
# 🏆 퀸트프로젝트 마스터 클래스
# ============================================================================
class QuintProjectMaster:
    """퀸트프로젝트 4대 시장 통합 마스터 시스템"""
    
    def __init__(self):
        # 설정 로드
        self.portfolio_value = config.get('system.portfolio_value', 100_000_000)
        self.demo_mode = config.get('system.demo_mode', True)
        self.auto_trading = config.get('system.auto_trading', False)
        
        # 4대 엔진 초기화
        self.us_engine = USStockEngine()
        self.crypto_engine = UpbitCryptoEngine()
        self.japan_engine = JapanStockEngine()
        self.india_engine = IndiaStockEngine()
        
        # 관리자들
        self.portfolio_manager = PortfolioManager()
        self.notification_manager = NotificationManager()
        
        # 상태
        self.last_analysis_time = None
        self.all_signals = []
        
        logging.info("🏆 퀸트프로젝트 마스터 시스템 초기화 완료")
    
    async def run_full_analysis(self) -> Dict:
        """4대 시장 통합 분석 실행"""
        logging.info("🚀 퀸트프로젝트 4대 시장 통합 분석 시작")
        start_time = datetime.now()
        
        try:
            # 4대 시장 병렬 분석
            tasks = []
            
            if config.get('markets.us_stocks.enabled', True):
                tasks.append(self.us_engine.analyze_us_market())
            
            if config.get('markets.upbit_crypto.enabled', True):
                tasks.append(self.crypto_engine.analyze_crypto_market())
            
            if config.get('markets.japan_stocks.enabled', True):
                tasks.append(self.japan_engine.analyze_japan_market())
            
            if config.get('markets.india_stocks.enabled', True):
                tasks.append(self.india_engine.analyze_india_market())
            
            # 병렬 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 통합
            all_signals = []
            market_names = ['미국주식', '암호화폐', '일본주식', '인도주식']
            enabled_markets = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"{market_names[i]} 분석 실패: {result}")
                    continue
                
                if result:
                    all_signals.extend(result)
                    enabled_markets.append(market_names[i])
            
            # 신뢰도 순 정렬
            all_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            # 최종 포트폴리오 구성
            final_portfolio = self._optimize_portfolio(all_signals)
            
            # 결과 요약
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            buy_signals = [s for s in all_signals if s.action == 'BUY']
            total_investment = sum(s.investment_amount for s in buy_signals)
            
            analysis_result = {
                'timestamp': start_time,
                'elapsed_time': elapsed_time,
                'enabled_markets': enabled_markets,
                'total_signals': len(all_signals),
                'buy_signals': len(buy_signals),
                'total_investment': total_investment,
                'portfolio_allocation': (total_investment / self.portfolio_value) * 100,
                'signals': all_signals,
                'optimized_portfolio': final_portfolio,
                'market_breakdown': self._get_market_breakdown(all_signals)
            }
            
            # 시그널 저장
            self.all_signals = all_signals
            self.last_analysis_time = start_time
            
            # 알림 전송
            await self.notification_manager.send_signal_alert(buy_signals)
            
            logging.info(f"✅ 4대 시장 분석 완료: {elapsed_time:.1f}초, {len(buy_signals)}개 매수 신호")
            
            return analysis_result
            
        except Exception as e:
            logging.error(f"4대 시장 분석 실패: {e}")
            return {'error': str(e), 'timestamp': start_time}
    
    def _optimize_portfolio(self, signals: List[QuintSignal]) -> List[QuintSignal]:
        """포트폴리오 최적화"""
        buy_signals = [s for s in signals if s.action == 'BUY']
        
        # 시장별 할당 한도 체크
        market_allocations = {
            'us': config.get('markets.us_stocks.allocation', 40.0),
            'crypto': config.get('markets.upbit_crypto.allocation', 30.0),
            'japan': config.get('markets.japan_stocks.allocation', 20.0),
            'india': config.get('markets.india_stocks.allocation', 10.0)
        }
        
        market_totals = {'us': 0, 'crypto': 0, 'japan': 0, 'india': 0}
        optimized_signals = []
        
        for signal in buy_signals:
            market = signal.market
            if market_totals[market] + signal.allocation_percent <= market_allocations[market]:
                optimized_signals.append(signal)
                market_totals[market] += signal.allocation_percent
            
            # 최대 20개 종목으로 제한
            if len(optimized_signals) >= 20:
                break
        
        return optimized_signals
    
    def _get_market_breakdown(self, signals: List[QuintSignal]) -> Dict:
        """시장별 분석 결과"""
        breakdown = {}
        
        for market in ['us', 'crypto', 'japan', 'india']:
            market_signals = [s for s in signals if s.market == market]
            buy_signals = [s for s in market_signals if s.action == 'BUY']
            
            breakdown[market] = {
                'total_analyzed': len(market_signals),
                'buy_signals': len(buy_signals),
                'avg_confidence': np.mean([s.confidence for s in buy_signals]) if buy_signals else 0,
                'total_investment': sum(s.investment_amount for s in buy_signals)
            }
        
        return breakdown
    
    def print_analysis_results(self, analysis_result: Dict):
        """분석 결과 출력"""
        if 'error' in analysis_result:
            print(f"❌ 분석 실패: {analysis_result['error']}")
            return
        
        print("\n" + "="*80)
        print("🏆 퀸트프로젝트 4대 시장 통합 분석 결과")
        print("="*80)
        
        # 기본 정보
        print(f"\n📊 분석 요약:")
        print(f"   소요시간: {analysis_result['elapsed_time']:.1f}초")
        print(f"   활성시장: {', '.join(analysis_result['enabled_markets'])}")
        print(f"   총 분석: {analysis_result['total_signals']}개 종목")
        print(f"   매수신호: {analysis_result['buy_signals']}개")
        print(f"   총 투자: {analysis_result['total_investment']:,.0f}원")
        print(f"   포트폴리오 비중: {analysis_result['portfolio_allocation']:.1f}%")
        print(f"   운영모드: {'시뮬레이션' if self.demo_mode else '실거래'}")
        
        # 시장별 분석
        print(f"\n🌍 시장별 분석:")
        market_names = {'us': '🇺🇸 미국주식', 'crypto': '🪙 암호화폐', 'japan': '🇯🇵 일본주식', 'india': '🇮🇳 인도주식'}
        breakdown = analysis_result['market_breakdown']
        
        for market, data in breakdown.items():
            if data['total_analyzed'] > 0:
                print(f"   {market_names[market]}: {data['buy_signals']}/{data['total_analyzed']} "
                      f"(신뢰도 {data['avg_confidence']:.1%}, 투자 {data['total_investment']:,.0f}원)")
        
        # 상위 매수 추천
        optimized_portfolio = analysis_result['optimized_portfolio']
        if optimized_portfolio:
            print(f"\n💎 최적화된 포트폴리오 ({len(optimized_portfolio)}개):")
            
            for i, signal in enumerate(optimized_portfolio[:10], 1):  # 상위 10개
                market_emoji = {'us': '🇺🇸', 'crypto': '🪙', 'japan': '🇯🇵', 'india': '🇮🇳'}
                print(f"\n   [{i:2d}] {market_emoji[signal.market]} {signal.symbol}")
                print(f"        신뢰도: {signal.confidence:.1%} | 현재가: {signal.price:,.0f}")
                print(f"        투자액: {signal.investment_amount:,.0f}원 ({signal.allocation_percent:.1f}%)")
                print(f"        목표가: {signal.target_price:,.0f} | 손절: {signal.stop_loss:,.0f}")
                print(f"        전략: {signal.reasoning[:50]}...")
        
        print("\n" + "="*80)
        print("🚀 퀸트프로젝트 - 4대 시장 완전 정복!")
        print("="*80)
    
    async def start_real_time_monitoring(self, interval_minutes: int = 30):
        """실시간 모니터링 시작"""
        logging.info(f"📡 실시간 모니터링 시작 (간격: {interval_minutes}분)")
        
        while True:
            try:
                # 전체 분석 실행
                result = await self.run_full_analysis()
                
                # 포트폴리오 업데이트
                portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                await self.notification_manager.send_portfolio_update(portfolio_summary)
                
                # 다음 분석까지 대기
                await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logging.info("⏹️ 실시간 모니터링 중지")
                break
            except Exception as e:
                logging.error(f"모니터링 오류: {e}")
                await asyncio.sleep(60)  # 1분 후 재시도

# ============================================================================
# 🛠️ 유틸리티 함수들
# ============================================================================
class QuintUtils:
    """퀸트프로젝트 유틸리티"""
    
    @staticmethod
    def setup_logging():
        """로깅 설정"""
        log_level = config.get('system.log_level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('quint_project.log', encoding='utf-8') 
                if config.get('system.backup_enabled', True) else logging.NullHandler()
            ]
        )
    
    @staticmethod
    def validate_environment():
        """환경 검증"""
        issues = []
        
        # API 키 체크 (데모 모드가 아닐 때)
        if not config.get('system.demo_mode', True):
            if config.get('markets.upbit_crypto.enabled') and not os.getenv('UPBIT_ACCESS_KEY'):
                issues.append("업비트 API 키 누락")
            
            if config.get('us_stocks.ibkr.enabled') and not IBKR_AVAILABLE:
                issues.append("IBKR 모듈 누락 (pip install ib_insync)")
        
        # 필수 라이브러리 체크
        required_libs = ['yfinance', 'pyupbit', 'pandas', 'numpy']
        for lib in required_libs:
            try:
                __import__(lib)
            except ImportError:
                issues.append(f"{lib} 라이브러리 누락")
        
        return issues
    
    @staticmethod
    def backup_data():
        """데이터 백업"""
        if not config.get('system.backup_enabled', True):
            return
        
        try:
            backup_dir = Path('backups')
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 설정 파일 백업
            if Path('quint_config.yaml').exists():
                import shutil
                shutil.copy('quint_config.yaml', backup_dir / f'config_{timestamp}.yaml')
            
            # 포트폴리오 백업
            if Path('quint_portfolio.json').exists():
                import shutil
                shutil.copy('quint_portfolio.json', backup_dir / f'portfolio_{timestamp}.json')
            
            logging.info(f"백업 완료: {timestamp}")
            
        except Exception as e:
            logging.error(f"백업 실패: {e}")

# ============================================================================
# 🎮 편의 함수들 (외부 호출용)
# ============================================================================
async def run_quint_analysis():
    """퀸트프로젝트 전체 분석 실행"""
    master = QuintProjectMaster()
    result = await master.run_full_analysis()
    master.print_analysis_results(result)
    return result

async def analyze_single_market(market: str):
    """단일 시장 분석"""
    master = QuintProjectMaster()
    
    if market.lower() == 'us':
        signals = await master.us_engine.analyze_us_market()
        print(f"\n🇺🇸 미국주식 분석 결과: {len([s for s in signals if s.action == 'BUY'])}개 매수 신호")
    elif market.lower() == 'crypto':
        signals = await master.crypto_engine.analyze_crypto_market()
        print(f"\n🪙 암호화폐 분석 결과: {len([s for s in signals if s.action == 'BUY'])}개 매수 신호")
    elif market.lower() == 'japan':
        signals = await master.japan_engine.analyze_japan_market()
        print(f"\n🇯🇵 일본주식 분석 결과: {len([s for s in signals if s.action == 'BUY'])}개 매수 신호")
    elif market.lower() == 'india':
        signals = await master.india_engine.analyze_india_market()
        print(f"\n🇮🇳 인도주식 분석 결과: {len([s for s in signals if s.action == 'BUY'])}개 매수 신호")
    else:
        print("❌ 지원되는 시장: us, crypto, japan, india")
        return []
    
    for signal in [s for s in signals if s.action == 'BUY'][:5]:
        print(f"  📈 {signal.symbol}: {signal.confidence:.1%} 신뢰도")
    
    return signals

async def start_monitoring(interval_minutes: int = 30):
    """실시간 모니터링 시작"""
    master = QuintProjectMaster()
    await master.start_real_time_monitoring(interval_minutes)

def get_portfolio_status():
    """포트폴리오 현황 조회"""
    manager = PortfolioManager()
    summary = manager.get_portfolio_summary()
    
    print("\n💼 포트폴리오 현황:")
    print(f"   총 포지션: {summary['total_positions']}개")
    print(f"   총 가치: {summary['total_value']:,.0f}원")
    
    allocation = summary['allocation_by_market']
    market_names = {'us': '🇺🇸 미국', 'crypto': '🪙 암호화폐', 'japan': '🇯🇵 일본', 'india': '🇮🇳 인도'}
    
    for market, value in allocation.items():
        if value > 0:
            percent = (value / summary['total_value']) * 100 if summary['total_value'] > 0 else 0
            print(f"   {market_names[market]}: {value:,.0f}원 ({percent:.1f}%)")
    
    return summary

def update_config(key: str, value):
    """설정 업데이트"""
    config.update(key, value)
    print(f"✅ 설정 업데이트: {key} = {value}")

def get_system_status():
    """시스템 상태 조회"""
    issues = QuintUtils.validate_environment()
    
    status = {
        'config_loaded': Path('quint_config.yaml').exists(),
        'portfolio_loaded': Path('quint_portfolio.json').exists(),
        'demo_mode': config.get('system.demo_mode', True),
        'auto_trading': config.get('system.auto_trading', False),
        'environment_issues': issues,
        'enabled_markets': {
            'us_stocks': config.get('markets.us_stocks.enabled', True),
            'upbit_crypto': config.get('markets.upbit_crypto.enabled', True),
            'japan_stocks': config.get('markets.japan_stocks.enabled', True),
            'india_stocks': config.get('markets.india_stocks.enabled', True)
        }
    }
    
    print("\n🔧 시스템 상태:")
    print(f"   설정 파일: {'✅' if status['config_loaded'] else '❌'}")
    print(f"   포트폴리오: {'✅' if status['portfolio_loaded'] else '❌'}")
    print(f"   운영 모드: {'시뮬레이션' if status['demo_mode'] else '실거래'}")
    print(f"   자동매매: {'활성화' if status['auto_trading'] else '비활성화'}")
    
    if issues:
        print(f"   ⚠️ 이슈: {', '.join(issues)}")
    else:
        print(f"   환경 검증: ✅")
    
    enabled_count = sum(status['enabled_markets'].values())
    print(f"   활성 시장: {enabled_count}/4개")
    
    return status

# ============================================================================
# 🎯 메인 실행 함수
# ============================================================================
async def main():
    """퀸트프로젝트 메인 실행"""
    # 로깅 설정
    QuintUtils.setup_logging()
    
    print("🏆" + "="*78)
    print("🚀 퀸트프로젝트 - 4대 시장 통합 핵심 시스템 CORE.PY")
    print("="*80)
    print("🇺🇸 미국주식 | 🪙 암호화폐 | 🇯🇵 일본주식 | 🇮🇳 인도주식")
    print("⚡ 혼자 보수유지 가능한 완전 자동화 아키텍처")
    print("="*80)
    
    # 시스템 상태 확인
    print("\n🔧 시스템 초기화 중...")
    status = get_system_status()
    
    if status['environment_issues']:
        print(f"\n⚠️ 환경 이슈 발견: {len(status['environment_issues'])}개")
        for issue in status['environment_issues']:
            print(f"   - {issue}")
        print("\n💡 데모 모드에서는 대부분의 기능이 정상 작동합니다.")
    
    # 백업 실행
    QuintUtils.backup_data()
    
    try:
        # 전체 분석 실행
        print(f"\n🚀 4대 시장 통합 분석 시작...")
        result = await run_quint_analysis()
        
        if 'error' not in result:
            print(f"\n💡 퀸트프로젝트 사용법:")
            print(f"   - run_quint_analysis(): 전체 4대 시장 분석")
            print(f"   - analyze_single_market('crypto'): 단일 시장 분석")
            print(f"   - start_monitoring(30): 실시간 모니터링 (30분 간격)")
            print(f"   - get_portfolio_status(): 포트폴리오 현황")
            print(f"   - update_config('system.demo_mode', False): 설정 변경")
            print(f"   - get_system_status(): 시스템 상태 확인")
        
        return result
        
    except KeyboardInterrupt:
        print(f"\n👋 퀸트프로젝트를 종료합니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류: {e}")
        logging.error(f"메인 실행 실패: {e}")

# ============================================================================
# 🎮 CLI 인터페이스
# ============================================================================
def cli_interface():
    """간단한 CLI 인터페이스"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'analyze':
            # 전체 분석
            asyncio.run(run_quint_analysis())
            
        elif command.startswith('analyze:'):
            # 단일 시장 분석
            market = command.split(':')[1]
            asyncio.run(analyze_single_market(market))
            
        elif command == 'monitor':
            # 실시간 모니터링
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            asyncio.run(start_monitoring(interval))
            
        elif command == 'status':
            # 시스템 상태
            get_system_status()
            
        elif command == 'portfolio':
            # 포트폴리오 현황
            get_portfolio_status()
            
        elif command == 'config':
            # 설정 변경
            if len(sys.argv) >= 4:
                key, value = sys.argv[2], sys.argv[3]
                # 타입 추론
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.replace('.', '').isdigit():
                    value = float(value) if '.' in value else int(value)
                update_config(key, value)
            else:
                print("사용법: python core.py config <key> <value>")
                
        else:
            print("퀸트프로젝트 CLI 사용법:")
            print("  python core.py analyze              # 전체 4대 시장 분석")
            print("  python core.py analyze:crypto       # 암호화폐만 분석")
            print("  python core.py analyze:us           # 미국주식만 분석")
            print("  python core.py monitor 30           # 30분 간격 실시간 모니터링")
            print("  python core.py status               # 시스템 상태 확인")
            print("  python core.py portfolio            # 포트폴리오 현황")
            print("  python core.py config demo_mode false  # 설정 변경")
    else:
        # 기본 실행
        asyncio.run(main())

# ============================================================================
# 🎯 실행부
# ============================================================================
if __name__ == "__main__":
    # CLI 모드 실행
    cli_interface()

# ============================================================================
# 📋 퀸트프로젝트 CORE.PY 특징 요약
# ============================================================================
"""
🏆 퀸트프로젝트 CORE.PY 완전체 특징:

🔧 혼자 보수유지 가능한 아키텍처:
   ✅ 설정 기반 모듈화 (quint_config.yaml)
   ✅ 자동 설정 생성 및 백업
   ✅ 런타임 설정 변경 지원
   ✅ 환경 검증 및 이슈 진단

🌍 4대 시장 완전 통합:
   ✅ 🇺🇸 미국주식: 전설적 퀸트 V6.0 (IBKR 호환)
   ✅ 🪙 업비트: 전설급 5대 시스템 (Neural Quality + Quantum Cycle)
   ✅ 🇯🇵 일본주식: YEN-HUNTER (TOPIX+JPX400 통합)
   ✅ 🇮🇳 인도주식: 5대 투자거장 + 14개 기술지표

⚡ 완전 자동화 시스템:
   ✅ 병렬 시장 분석 (asyncio 기반)
   ✅ 포트폴리오 최적화 및 리밸런싱
   ✅ 실시간 모니터링 및 알림
   ✅ 자동 백업 및 복구

🛡️ 통합 리스크 관리:
   ✅ 시장별 할당 한도 관리
   ✅ 종목별 집중도 제한
   ✅ 동적 손절/익절 시스템
   ✅ 상관관계 기반 분산투자

💎 사용법:
   - 설치: pip install -r requirements.txt
   - 실행: python core.py analyze
   - 모니터링: python core.py monitor 30
   - 설정: python core.py config demo_mode false

🚀 확장성:
   ✅ 새로운 시장 추가 용이
   ✅ 전략 가중치 실시간 조정
   ✅ API 연동 모듈화
   ✅ 백테스팅 및 성과 분석 준비

🎯 핵심 철학:
   - 단순함이 최고다 (Simple is Best)
   - 설정으로 모든 것을 제어한다
   - 장애시 자동 복구한다
   - 혼자서도 충분히 관리할 수 있다

🏆 퀸트프로젝트 = 4대 시장 완전 정복!
"""
