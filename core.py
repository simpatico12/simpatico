#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 통합 코어 시스템 (core.py)
=================================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 (4대 전략 통합)

✨ 핵심 기능:
- 4대 전략 통합 관리 시스템
- IBKR 자동 환전 기능 (달러 ↔ 엔/루피)
- OpenAI GPT-4 기반 AI 매매 분석
- 네트워크 모니터링 + 끊김 시 전량 매도
- 통합 포지션 관리 + 리스크 제어
- 실시간 모니터링 + 알림 시스템
- 성과 추적 + 자동 백업
- 🚨 응급 오류 감지 시스템

Author: 퀸트마스터팀
Version: 1.2.0 (OpenAI 연동 + AI 자동매매)
"""

import asyncio
import logging
import sys
import os
import json
import time
import psutil
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
import requests
import aiohttp
import sqlite3
import shutil
import subprocess
import signal
import yfinance as yf
import pandas as pd
import numpy as np

# OpenAI 연동
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI 모듈 없음 - pip install openai 필요")

# 전략 모듈 임포트
try:
    from us_strategy import LegendaryQuantStrategy as USStrategy
    US_AVAILABLE = True
except ImportError:
    US_AVAILABLE = False
    print("⚠️ 미국 전략 모듈 없음")

try:
    from jp_strategy import YenHunter as JapanStrategy
    JAPAN_AVAILABLE = True
except ImportError:
    JAPAN_AVAILABLE = False
    print("⚠️ 일본 전략 모듈 없음")

try:
    from inda_strategy import LegendaryIndiaStrategy as IndiaStrategy
    INDIA_AVAILABLE = True
except ImportError:
    INDIA_AVAILABLE = False
    print("⚠️ 인도 전략 모듈 없음")

try:
    from coin_strategy import LegendaryQuantMaster as CryptoStrategy
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("⚠️ 암호화폐 전략 모듈 없음")

# 업비트 연동
try:
    import pyupbit
    UPBIT_AVAILABLE = True
except ImportError:
    UPBIT_AVAILABLE = False
    print("⚠️ 업비트 모듈 없음")

# IBKR 연동
try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("⚠️ IBKR 모듈 없음")

# ============================================================================
# 🎯 통합 설정 관리자
# ============================================================================
class CoreConfig:
    """통합 설정 관리"""
    
    def __init__(self):
        load_dotenv()
        
        # 시스템 설정
        self.TOTAL_PORTFOLIO_VALUE = float(os.getenv('TOTAL_PORTFOLIO_VALUE', 1000000000))
        self.MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', 0.05))
        
        # 전략별 활성화
        self.US_ENABLED = os.getenv('US_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.JAPAN_ENABLED = os.getenv('JAPAN_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.INDIA_ENABLED = os.getenv('INDIA_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.CRYPTO_ENABLED = os.getenv('CRYPTO_STRATEGY_ENABLED', 'true').lower() == 'true'
        
        # 전략별 자원 배분
        self.US_ALLOCATION = float(os.getenv('US_STRATEGY_ALLOCATION', 0.40))
        self.JAPAN_ALLOCATION = float(os.getenv('JAPAN_STRATEGY_ALLOCATION', 0.25))
        self.CRYPTO_ALLOCATION = float(os.getenv('CRYPTO_STRATEGY_ALLOCATION', 0.20))
        self.INDIA_ALLOCATION = float(os.getenv('INDIA_STRATEGY_ALLOCATION', 0.15))
        
        # OpenAI 설정
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        self.OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
        self.AI_ANALYSIS_ENABLED = os.getenv('AI_ANALYSIS_ENABLED', 'true').lower() == 'true'
        self.AI_AUTO_TRADE = os.getenv('AI_AUTO_TRADE', 'false').lower() == 'true'
        
        # IBKR 설정
        self.IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
        self.IBKR_PORT = int(os.getenv('IBKR_PORT', 7497))
        self.IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', 1))
        
        # 네트워크 모니터링
        self.NETWORK_MONITORING = os.getenv('NETWORK_MONITORING_ENABLED', 'true').lower() == 'true'
        self.NETWORK_CHECK_INTERVAL = int(os.getenv('NETWORK_CHECK_INTERVAL', 30))
        self.NETWORK_DISCONNECT_SELL = os.getenv('NETWORK_DISCONNECT_SELL_ALL', 'false').lower() == 'true'
        
        # 응급 오류 감지
        self.EMERGENCY_SELL_ON_ERROR = os.getenv('EMERGENCY_SELL_ON_ERROR', 'true').lower() == 'true'
        self.EMERGENCY_ERROR_COUNT = int(os.getenv('EMERGENCY_ERROR_COUNT', 5))
        self.EMERGENCY_MEMORY_THRESHOLD = int(os.getenv('EMERGENCY_MEMORY_THRESHOLD', 95))
        self.EMERGENCY_CPU_THRESHOLD = int(os.getenv('EMERGENCY_CPU_THRESHOLD', 90))
        self.EMERGENCY_DISK_THRESHOLD = int(os.getenv('EMERGENCY_DISK_THRESHOLD', 5))
        
        # 업비트 설정
        self.UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
        self.UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')
        self.UPBIT_DEMO_MODE = os.getenv('CRYPTO_DEMO_MODE', 'true').lower() == 'true'
        
        # 알림 설정
        self.TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # 데이터베이스
        self.DB_PATH = os.getenv('DATABASE_PATH', './data/quant_core.db')
        self.BACKUP_PATH = os.getenv('BACKUP_PATH', './backups/')

# ============================================================================
# 🚨 응급 오류 감지 시스템
# ============================================================================
class EmergencyErrorDetector:
    """응급 오류 감지 및 대응 시스템"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.error_counts = {}
        self.last_check_time = time.time()
        
        self.logger = logging.getLogger('EmergencyDetector')
    
    def record_error(self, error_type: str, error_message: str, critical: bool = False) -> bool:
        """오류 기록 및 응급 상황 판단"""
        current_time = time.time()
        
        if error_type not in self.error_counts:
            self.error_counts[error_type] = []
        
        self.error_counts[error_type].append({
            'timestamp': current_time,
            'message': error_message,
            'critical': critical
        })
        
        # 1시간 이전 오류 제거
        cutoff_time = current_time - 3600
        self.error_counts[error_type] = [
            error for error in self.error_counts[error_type] 
            if error['timestamp'] > cutoff_time
        ]
        
        # 치명적 오류 즉시 응급 처리
        if critical:
            self.logger.critical(f"🚨 치명적 오류 감지: {error_type} - {error_message}")
            return True
        
        # 일반 오류 누적 체크
        recent_errors = len(self.error_counts[error_type])
        if recent_errors >= self.config.EMERGENCY_ERROR_COUNT:
            self.logger.critical(f"🚨 오류 임계치 초과: {error_type} ({recent_errors}회)")
            return True
        
        return False
    
    def check_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태 체크"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_free_percent = (disk.free / disk.total) * 100
            
            # 응급 상황 판단
            emergency_needed = (
                cpu_percent > self.config.EMERGENCY_CPU_THRESHOLD or
                memory_percent > self.config.EMERGENCY_MEMORY_THRESHOLD or
                disk_free_percent < self.config.EMERGENCY_DISK_THRESHOLD
            )
            
            health_status = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_free_percent': disk_free_percent,
                'emergency_needed': emergency_needed,
                'timestamp': datetime.now().isoformat()
            }
            
            if emergency_needed:
                self.logger.critical(
                    f"🚨 시스템 리소스 위험: CPU={cpu_percent}%, "
                    f"메모리={memory_percent}%, 디스크여유={disk_free_percent}%"
                )
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"시스템 건강 체크 실패: {e}")
            return {'emergency_needed': False, 'error': str(e)}

# ============================================================================
# 🏦 IBKR 통합 관리자
# ============================================================================
class IBKRManager:
    """IBKR 연결 및 거래 관리"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.ib = None
        self.connected = False
        self.account_info = {}
        self.positions = {}
        
        self.logger = logging.getLogger('IBKRManager')
    
    async def connect(self):
        """IBKR 연결"""
        if not IBKR_AVAILABLE:
            self.logger.warning("IBKR 모듈 없음 - 암호화폐 전용 모드")
            return False
        
        try:
            self.ib = IB()
            await self.ib.connectAsync(
                host=self.config.IBKR_HOST,
                port=self.config.IBKR_PORT,
                clientId=self.config.IBKR_CLIENT_ID
            )
            
            self.connected = True
            self.logger.info("✅ IBKR 연결 성공")
            
            # 계좌 정보 업데이트
            await self._update_account_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"IBKR 연결 실패: {e}")
            self.connected = False
            return False
    
    async def _update_account_info(self):
        """계좌 정보 업데이트"""
        if not self.connected:
            return
        
        try:
            # 계좌 요약 정보
            account_summary = self.ib.accountSummary()
            self.account_info = {item.tag: item.value for item in account_summary}
            
            # 포지션 정보
            positions = self.ib.positions()
            self.positions = {}
            
            for position in positions:
                symbol = position.contract.symbol
                self.positions[symbol] = {
                    'position': position.position,
                    'avgCost': position.avgCost,
                    'marketPrice': position.marketPrice,
                    'marketValue': position.marketValue,
                    'currency': position.contract.currency,
                    'unrealizedPNL': position.unrealizedPNL
                }
            
            self.logger.debug(f"계좌 정보 업데이트: {len(self.positions)}개 포지션")
            
        except Exception as e:
            self.logger.error(f"계좌 정보 업데이트 실패: {e}")
    
    async def auto_currency_exchange(self, target_currency: str, amount: float):
        """자동 환전"""
        if not self.connected:
            return
        
        try:
            base_currency = 'USD'
            
            if target_currency == base_currency:
                return  # 같은 통화면 환전 불필요
            
            # 환율 확인
            forex_contract = Forex(f"{base_currency}{target_currency}")
            ticker = self.ib.reqMktData(forex_contract)
            
            await asyncio.sleep(2)  # 데이터 수신 대기
            
            if ticker.last:
                exchange_rate = ticker.last
                target_amount = amount / exchange_rate
                
                # 환전 주문
                order = MarketOrder('BUY', target_amount)
                trade = self.ib.placeOrder(forex_contract, order)
                
                self.logger.info(f"💱 환전 주문: {amount} {base_currency} → {target_amount:.2f} {target_currency}")
                
                return trade
            
        except Exception as e:
            self.logger.error(f"환전 실패 {target_currency}: {e}")
    
    async def emergency_sell_all(self) -> Dict[str, Any]:
        """응급 전량 매도"""
        if not self.connected:
            return {'error': 'IBKR 미연결'}
        
        try:
            self.logger.critical("🚨 응급 전량 매도 시작!")
            
            await self._update_account_info()
            sell_results = {}
            
            for symbol, position_info in self.positions.items():
                try:
                    quantity = abs(position_info['position'])
                    
                    if quantity > 0:
                        # 계약 생성 (통화별)
                        currency = position_info['currency']
                        
                        if currency == 'USD':
                            contract = Stock(symbol, 'SMART', 'USD')
                        elif currency == 'JPY':
                            contract = Stock(symbol, 'TSE', 'JPY')
                        elif currency == 'INR':
                            contract = Stock(symbol, 'NSE', 'INR')
                        else:
                            contract = Stock(symbol, 'SMART', currency)
                        
                        # 시장가 매도 주문
                        order = MarketOrder('SELL', quantity)
                        trade = self.ib.placeOrder(contract, order)
                        
                        sell_results[symbol] = {
                            'quantity': quantity,
                            'trade_id': trade.order.orderId
                        }
                        
                        self.logger.info(f"응급 매도: {symbol} {quantity}주")
                
                except Exception as e:
                    self.logger.error(f"응급 매도 실패 {symbol}: {e}")
                    sell_results[symbol] = {'error': str(e)}
            
            self.logger.critical(f"🚨 응급 전량 매도 완료: {len(sell_results)}개 종목")
            return sell_results
            
        except Exception as e:
            self.logger.error(f"응급 전량 매도 실패: {e}")
            return {'error': str(e)}

# ============================================================================
# 🤖 OpenAI AI 분석 엔진
# ============================================================================
class AIAnalysisEngine:
    """OpenAI GPT-4 기반 AI 매매 분석 엔진"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.client = None
        
        if OPENAI_AVAILABLE and config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
            self.client = openai
        
        self.logger = logging.getLogger('AIEngine')
        
        # AI 분석 프롬프트 템플릿
        self.analysis_prompts = {
            'market_analysis': """
당신은 세계적인 퀀트 전문가입니다. 다음 시장 데이터를 분석하고 매매 방향을 제시해주세요.

=== 시장 데이터 ===
{market_data}

=== 현재 포지션 ===
{current_positions}

=== 분석 요청 ===
1. 시장 트렌드 분석 (강세/약세/횡보)
2. 주요 리스크 요인
3. 매매 추천 (BUY/SELL/HOLD)
4. 목표가 및 손절가
5. 포지션 크기 추천
6. 신뢰도 점수 (1-100)

응답은 JSON 형식으로 해주세요:
{
    "trend_analysis": "분석 내용",
    "risk_factors": ["리스크1", "리스크2"],
    "recommendation": "BUY/SELL/HOLD",
    "target_price": 목표가,
    "stop_loss": 손절가,
    "position_size": 포지션_크기_퍼센트,
    "confidence": 신뢰도_점수,
    "reasoning": "판단 근거"
}
            """,
            
            'portfolio_optimization': """
당신은 포트폴리오 최적화 전문가입니다. 현재 포트폴리오를 분석하고 개선방안을 제시해주세요.

=== 현재 포트폴리오 ===
{portfolio_data}

=== 시장 상황 ===
{market_conditions}

=== 분석 요청 ===
1. 포트폴리오 균형도 평가
2. 리스크 분산 분석
3. 리밸런싱 필요 여부
4. 매도 추천 종목
5. 매수 추천 종목
6. 전체적인 포트폴리오 점수

응답은 JSON 형식으로 해주세요.
            """,
            
            'risk_assessment': """
당신은 리스크 관리 전문가입니다. 현재 상황의 리스크를 평가해주세요.

=== 포지션 데이터 ===
{position_data}

=== 시장 변동성 ===
{volatility_data}

위험도를 1-10으로 평가하고 대응방안을 제시해주세요.
            """
        }
    
    async def analyze_market_trend(self, symbol: str, market: str = 'US') -> Dict[str, Any]:
        """시장 트렌드 AI 분석"""
        if not self.client or not self.config.AI_ANALYSIS_ENABLED:
            return {'error': 'AI 분석 비활성화'}
        
        try:
            # 시장 데이터 수집
            market_data = await self._collect_market_data(symbol, market)
            
            # 현재 포지션 정보
            current_positions = await self._get_position_info(symbol)
            
            # AI 분석 요청
            prompt = self.analysis_prompts['market_analysis'].format(
                market_data=json.dumps(market_data, indent=2),
                current_positions=json.dumps(current_positions, indent=2)
            )
            
            response = await self._call_openai_api(prompt)
            
            # JSON 파싱
            try:
                analysis_result = json.loads(response)
                analysis_result['timestamp'] = datetime.now().isoformat()
                analysis_result['symbol'] = symbol
                analysis_result['market'] = market
                
                self.logger.info(f"🤖 AI 분석 완료: {symbol} - {analysis_result.get('recommendation', 'N/A')}")
                return analysis_result
                
            except json.JSONDecodeError:
                self.logger.error(f"AI 응답 JSON 파싱 실패: {response}")
                return {'error': 'JSON 파싱 실패', 'raw_response': response}
                
        except Exception as e:
            self.logger.error(f"AI 시장 분석 실패: {e}")
            return {'error': str(e)}
    
    async def analyze_portfolio_optimization(self, portfolio_data: Dict) -> Dict[str, Any]:
        """포트폴리오 최적화 AI 분석"""
        if not self.client or not self.config.AI_ANALYSIS_ENABLED:
            return {'error': 'AI 분석 비활성화'}
        
        try:
            # 시장 상황 수집
            market_conditions = await self._collect_market_conditions()
            
            prompt = self.analysis_prompts['portfolio_optimization'].format(
                portfolio_data=json.dumps(portfolio_data, indent=2),
                market_conditions=json.dumps(market_conditions, indent=2)
            )
            
            response = await self._call_openai_api(prompt)
            
            try:
                result = json.loads(response)
                result['timestamp'] = datetime.now().isoformat()
                
                self.logger.info("🤖 포트폴리오 최적화 분석 완료")
                return result
                
            except json.JSONDecodeError:
                return {'error': 'JSON 파싱 실패', 'raw_response': response}
                
        except Exception as e:
            self.logger.error(f"포트폴리오 최적화 분석 실패: {e}")
            return {'error': str(e)}
    
    async def assess_risk(self, position_data: Dict, volatility_data: Dict) -> Dict[str, Any]:
        """리스크 평가 AI 분석"""
        if not self.client or not self.config.AI_ANALYSIS_ENABLED:
            return {'error': 'AI 분석 비활성화'}
        
        try:
            prompt = self.analysis_prompts['risk_assessment'].format(
                position_data=json.dumps(position_data, indent=2),
                volatility_data=json.dumps(volatility_data, indent=2)
            )
            
            response = await self._call_openai_api(prompt)
            
            try:
                result = json.loads(response)
                result['timestamp'] = datetime.now().isoformat()
                
                self.logger.info("🤖 리스크 평가 완료")
                return result
                
            except json.JSONDecodeError:
                return {'error': 'JSON 파싱 실패', 'raw_response': response}
                
        except Exception as e:
            self.logger.error(f"리스크 평가 실패: {e}")
            return {'error': str(e)}
    
    async def _call_openai_api(self, prompt: str) -> str:
        """OpenAI API 호출"""
        try:
            response = await asyncio.to_thread(
                self.client.ChatCompletion.create,
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 세계 최고의 퀀트 투자 전문가입니다. 정확하고 실용적인 분석을 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API 호출 실패: {e}")
            raise
    
    async def _collect_market_data(self, symbol: str, market: str) -> Dict[str, Any]:
        """시장 데이터 수집"""
        try:
            # 야후 파이낸스에서 데이터 수집
            if market == 'US':
                ticker = symbol
            elif market == 'JAPAN':
                ticker = f"{symbol}.T"
            elif market == 'INDIA':
                ticker = f"{symbol}.NS"
            else:
                ticker = symbol
            
            stock = yf.Ticker(ticker)
            
            # 기본 정보
            info = stock.info
            
            # 가격 데이터 (최근 30일)
            hist = stock.history(period="1mo")
            
            # 기술적 지표 계산
            current_price = hist['Close'].iloc[-1]
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
            
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)  # 연간 변동성
            
            # RSI 계산
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            market_data = {
                'symbol': symbol,
                'current_price': float(current_price),
                'sma_20': float(sma_20) if not pd.isna(sma_20) else None,
                'sma_50': float(sma_50) if sma_50 and not pd.isna(sma_50) else None,
                'rsi': float(rsi) if not pd.isna(rsi) else None,
                'volatility': float(volatility),
                'volume_avg': float(hist['Volume'].mean()),
                'price_change_1d': float((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100),
                'price_change_7d': float((current_price - hist['Close'].iloc[-7]) / hist['Close'].iloc[-7] * 100) if len(hist) >= 7 else None,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 실패 {symbol}: {e}")
            return {'error': str(e)}
    
    async def _get_position_info(self, symbol: str) -> Dict[str, Any]:
        """현재 포지션 정보 조회"""
        # 실제 구현에서는 포지션 매니저에서 정보를 가져옴
        return {
            'symbol': symbol,
            'quantity': 0,
            'avg_cost': 0,
            'current_value': 0,
            'unrealized_pnl': 0
        }
    
    async def _collect_market_conditions(self) -> Dict[str, Any]:
        """전체 시장 상황 수집"""
        try:
            # 주요 지수들
            indices = {
                'SPY': '^GSPC',    # S&P 500
                'QQQ': '^IXIC',    # NASDAQ
                'VTI': '^GSPC',    # Total Stock Market
                'NIKKEI': '^N225', # 니케이
                'SENSEX': '^BSESN' # 인도 센섹스
            }
            
            market_conditions = {}
            
            for name, ticker in indices.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="5d")
                    
                    if len(hist) > 0:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        
                        market_conditions[name] = {
                            'price': float(current),
                            'change_pct': float((current - prev) / prev * 100)
                        }
                except:
                    continue
            
            # VIX (공포지수)
            try:
                vix = yf.Ticker('^VIX')
                vix_hist = vix.history(period="1d")
                if len(vix_hist) > 0:
                    market_conditions['VIX'] = float(vix_hist['Close'].iloc[-1])
            except:
                pass
            
            return market_conditions
            
        except Exception as e:
            self.logger.error(f"시장 상황 수집 실패: {e}")
            return {}

# ============================================================================
# 🎯 AI 기반 자동매매 실행기
# ============================================================================
class AITradingExecutor:
    """AI 분석 기반 자동매매 실행"""
    
    def __init__(self, config: CoreConfig, ai_engine: AIAnalysisEngine, ibkr_manager):
        self.config = config
        self.ai_engine = ai_engine
        self.ibkr_manager = ibkr_manager
        
        self.logger = logging.getLogger('AITrader')
        
        # 매매 실행 설정
        self.min_confidence = 70  # 최소 신뢰도
        self.max_position_size = 0.1  # 최대 포지션 크기 (10%)
        self.stop_loss_pct = 0.05  # 손절 비율 (5%)
        
    async def execute_ai_trading(self, symbol: str, market: str = 'US') -> Dict[str, Any]:
        """AI 분석 기반 자동매매 실행"""
        if not self.config.AI_AUTO_TRADE:
            return {'message': 'AI 자동매매 비활성화'}
        
        try:
            self.logger.info(f"🤖 AI 자동매매 시작: {symbol}")
            
            # AI 분석 실행
            analysis = await self.ai_engine.analyze_market_trend(symbol, market)
            
            if 'error' in analysis:
                return {'error': f'AI 분석 실패: {analysis["error"]}'}
            
            # 신뢰도 체크
            confidence = analysis.get('confidence', 0)
            if confidence < self.min_confidence:
                self.logger.info(f"신뢰도 부족 ({confidence}% < {self.min_confidence}%), 거래 건너뜀")
                return {'message': f'신뢰도 부족: {confidence}%'}
            
            # 매매 실행
            recommendation = analysis.get('recommendation', 'HOLD')
            
            if recommendation == 'BUY':
                result = await self._execute_buy_order(symbol, analysis, market)
            elif recommendation == 'SELL':
                result = await self._execute_sell_order(symbol, analysis, market)
            else:
                result = {'message': 'HOLD - 거래 없음'}
            
            # 결과 로깅
            self.logger.info(f"🤖 AI 매매 완료: {symbol} - {recommendation} (신뢰도: {confidence}%)")
            
            return {
                'symbol': symbol,
                'analysis': analysis,
                'execution_result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"AI 자동매매 실패: {e}")
            return {'error': str(e)}
    
    async def _execute_buy_order(self, symbol: str, analysis: Dict, market: str) -> Dict[str, Any]:
        """매수 주문 실행"""
        try:
            if not self.ibkr_manager.connected:
                return {'error': 'IBKR 미연결'}
            
            # 포지션 크기 계산
            recommended_size = min(analysis.get('position_size', 5), self.max_position_size * 100)
            portfolio_value = self.config.TOTAL_PORTFOLIO_VALUE
            
            if market == 'US':
                allocation = self.config.US_ALLOCATION
                currency = 'USD'
            elif market == 'JAPAN':
                allocation = self.config.JAPAN_ALLOCATION
                currency = 'JPY'
            elif market == 'INDIA':
                allocation = self.config.INDIA_ALLOCATION
                currency = 'INR'
            else:
                allocation = 0.1
                currency = 'USD'
            
            max_investment = portfolio_value * allocation * (recommended_size / 100)
            
            # 환전 실행 (필요시)
            if currency != 'USD':
                await self.ibkr_manager.auto_currency_exchange(currency, max_investment)
            
            # 현재가 조회
            current_price = analysis.get('target_price', 0)
            if current_price <= 0:
                return {'error': '유효하지 않은 가격'}
            
            # 매수 수량 계산
            quantity = int(max_investment / current_price)
            
            if quantity <= 0:
                return {'error': '매수 수량 부족'}
            
            # IBKR 주문 실행
            try:
                # 계약 생성
                if market == 'US':
                    contract = Stock(symbol, 'SMART', 'USD')
                elif market == 'JAPAN':
                    contract = Stock(symbol, 'TSE', 'JPY')
                elif market == 'INDIA':
                    contract = Stock(symbol, 'NSE', 'INR')
                else:
                    contract = Stock(symbol, 'SMART', 'USD')
                
                # 시장가 매수 주문
                order = MarketOrder('BUY', quantity)
                trade = self.ibkr_manager.ib.placeOrder(contract, order)
                
                # 주문 완료 대기
                for _ in range(30):
                    await asyncio.sleep(1)
                    if trade.isDone():
                        break
                
                if trade.isDone() and trade.orderStatus.status == 'Filled':
                    self.logger.info(f"✅ AI 매수 완료: {symbol} {quantity}주")
                    
                    # 손절 주문 설정
                    stop_loss_price = current_price * (1 - self.stop_loss_pct)
                    stop_order = StopOrder('SELL', quantity, stop_loss_price)
                    self.ibkr_manager.ib.placeOrder(contract, stop_order)
                    
                    return {
                        'status': 'success',
                        'action': 'BUY',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': trade.orderStatus.avgFillPrice,
                        'total_cost': quantity * trade.orderStatus.avgFillPrice,
                        'stop_loss': stop_loss_price
                    }
                else:
                    return {'error': f'주문 실패: {trade.orderStatus.status}'}
                    
            except Exception as e:
                return {'error': f'IBKR 주문 실패: {e}'}
            
        except Exception as e:
            self.logger.error(f"매수 주문 실행 실패: {e}")
            return {'error': str(e)}
    
    async def _execute_sell_order(self, symbol: str, analysis: Dict, market: str) -> Dict[str, Any]:
        """매도 주문 실행"""
        try:
            if not self.ibkr_manager.connected:
                return {'error': 'IBKR 미연결'}
            
            # 현재 포지션 확인
            await self.ibkr_manager._update_account_info()
            
            if symbol not in self.ibkr_manager.positions:
                return {'error': '보유 포지션 없음'}
            
            position_info = self.ibkr_manager.positions[symbol]
            quantity = int(abs(position_info['position']))
            
            if quantity <= 0:
                return {'error': '매도할 수량 없음'}
            
            try:
                # 계약 생성
                if market == 'US':
                    contract = Stock(symbol, 'SMART', 'USD')
                elif market == 'JAPAN':
                    contract = Stock(symbol, 'TSE', 'JPY')
                elif market == 'INDIA':
                    contract = Stock(symbol, 'NSE', 'INR')
                else:
                    contract = Stock(symbol, 'SMART', 'USD')
                
                # 시장가 매도 주문
                order = MarketOrder('SELL', quantity)
                trade = self.ibkr_manager.ib.placeOrder(contract, order)
                
                # 주문 완료 대기
                for _ in range(30):
                    await asyncio.sleep(1)
                    if trade.isDone():
                        break
                
                if trade.isDone() and trade.orderStatus.status == 'Filled':
                    # 손익 계산
                    avg_cost = position_info['avgCost']
                    sell_price = trade.orderStatus.avgFillPrice
                    profit_loss = (sell_price - avg_cost) * quantity
                    profit_pct = (sell_price - avg_cost) / avg_cost * 100
                    
                    self.logger.info(f"✅ AI 매도 완료: {symbol} {quantity}주 (손익: {profit_loss:+.2f})")
                    
                    return {
                        'status': 'success',
                        'action': 'SELL',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': sell_price,
                        'total_revenue': quantity * sell_price,
                        'profit_loss': profit_loss,
                        'profit_pct': profit_pct
                    }
                else:
                    return {'error': f'주문 실패: {trade.orderStatus.status}'}
                    
            except Exception as e:
                return {'error': f'IBKR 주문 실패: {e}'}
            
        except Exception as e:
            self.logger.error(f"매도 주문 실행 실패: {e}")
            return {'error': str(e)}

# ============================================================================
# 📊 통합 포지션 관리자
# ============================================================================
@dataclass
class UnifiedPosition:
    """통합 포지션 정보"""
    strategy: str
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    currency: str
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_date: datetime
    last_updated: datetime

class UnifiedPositionManager:
    """통합 포지션 관리"""
    
    def __init__(self, config: CoreConfig, ibkr_manager: IBKRManager):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.positions: Dict[str, UnifiedPosition] = {}
        
        self.logger = logging.getLogger('PositionManager')
    
    async def update_all_positions(self):
        """모든 포지션 업데이트"""
        try:
            if self.ibkr_manager.connected:
                await self.ibkr_manager._update_account_info()
                
                # IBKR 포지션을 통합 포지션으로 변환
                for symbol, pos_info in self.ibkr_manager.positions.items():
                    
                    # 전략 추정 (심볼 기반)
                    strategy = self._estimate_strategy(symbol, pos_info['currency'])
                    
                    unified_pos = UnifiedPosition(
                        strategy=strategy,
                        symbol=symbol,
                        quantity=pos_info['position'],
                        avg_cost=pos_info['avgCost'],
                        current_price=pos_info['marketPrice'],
                        currency=pos_info['currency'],
                        unrealized_pnl=pos_info['unrealizedPNL'],
                        unrealized_pnl_pct=(pos_info['marketPrice'] - pos_info['avgCost']) / pos_info['avgCost'] * 100,
                        entry_date=datetime.now(),  # 실제로는 DB에서 로드
                        last_updated=datetime.now()
                    )
                    
                    self.positions[f"{strategy}_{symbol}"] = unified_pos
                
                self.logger.info(f"📊 포지션 업데이트: {len(self.positions)}개")
                
        except Exception as e:
            self.logger.error(f"포지션 업데이트 실패: {e}")
    
    def _estimate_strategy(self, symbol: str, currency: str) -> str:
        """심볼과 통화로 전략 추정"""
        if currency == 'USD':
            return 'US'
        elif currency == 'JPY':
            return 'JAPAN'
        elif currency == 'INR':
            return 'INDIA'
        elif currency == 'KRW':
            return 'CRYPTO'
        else:
            return 'UNKNOWN'
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """포트폴리오 요약"""
        summary = {
            'total_positions': len(self.positions),
            'by_strategy': {},
            'by_currency': {},
            'total_unrealized_pnl': 0,
            'profitable_positions': 0,
            'losing_positions': 0
        }
        
        for pos in self.positions.values():
            # 전략별 집계
            if pos.strategy not in summary['by_strategy']:
                summary['by_strategy'][pos.strategy] = {'count': 0, 'pnl': 0}
            summary['by_strategy'][pos.strategy]['count'] += 1
            summary['by_strategy'][pos.strategy]['pnl'] += pos.unrealized_pnl
            
            # 통화별 집계
            if pos.currency not in summary['by_currency']:
                summary['by_currency'][pos.currency] = {'count': 0, 'pnl': 0}
            summary['by_currency'][pos.currency]['count'] += 1
            summary['by_currency'][pos.currency]['pnl'] += pos.unrealized_pnl
            
            # 전체 집계
            summary['total_unrealized_pnl'] += pos.unrealized_pnl
            
            if pos.unrealized_pnl > 0:
                summary['profitable_positions'] += 1
            else:
                summary['losing_positions'] += 1
        
        return summary

# ============================================================================
# 🌐 네트워크 모니터링 시스템
# ============================================================================
class NetworkMonitor:
    """네트워크 연결 모니터링"""
    
    def __init__(self, config: CoreConfig, ibkr_manager: IBKRManager):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.monitoring = False
        self.connection_failures = 0
        self.last_check_time = None
        
        self.logger = logging.getLogger('NetworkMonitor')
    
    async def start_monitoring(self):
        """네트워크 모니터링 시작"""
        if not self.config.NETWORK_MONITORING:
            return
        
        self.monitoring = True
        self.logger.info("🌐 네트워크 모니터링 시작")
        
        while self.monitoring:
            try:
                await self._check_connections()
                await asyncio.sleep(self.config.NETWORK_CHECK_INTERVAL)
            except Exception as e:
                self.logger.error(f"네트워크 모니터링 오류: {e}")
                await asyncio.sleep(60)
    
    async def _check_connections(self):
        """연결 상태 체크"""
        current_time = time.time()
        
        # 인터넷 연결 체크
        internet_ok = await self._check_internet()
        
        # IBKR 연결 체크
        ibkr_ok = self.ibkr_manager.connected and self.ibkr_manager.ib.isConnected()
        
        # API 서버 체크
        api_ok = await self._check_api_servers()
        
        if not internet_ok or not ibkr_ok or not api_ok:
            self.connection_failures += 1
            
            # IBKR 없이 운영시 더 관대한 기준 적용
            if not self.ibkr_manager.connected and api_ok and internet_ok:
                # IBKR 없어도 API와 인터넷이 되면 경고만
                if self.connection_failures == 1:  # 첫 번째만 로그
                    self.logger.info(f"ℹ️ IBKR 미연결 상태로 운영 중 (암호화폐 전략만 사용)")
                self.connection_failures = 0  # 실패 카운트 리셋
                return
                
            self.logger.warning(f"⚠️ 연결 실패 {self.connection_failures}회: 인터넷={internet_ok}, IBKR={ibkr_ok}, API={api_ok}")
            
            # 연속 실패시 응급 조치 (더 엄격한 기준)
            if self.connection_failures >= 5 and self.config.NETWORK_DISCONNECT_SELL:
                self.logger.critical("🚨 네트워크 연결 실패로 응급 매도 실행!")
                await self.ibkr_manager.emergency_sell_all()
                self.monitoring = False
        else:
            if self.connection_failures > 0:
                self.logger.info("✅ 네트워크 연결 복구")
            self.connection_failures = 0
        
        self.last_check_time = current_time
    
    async def _check_internet(self) -> bool:
        """인터넷 연결 체크"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://www.google.com', timeout=10) as response:
                    return response.status == 200
        except:
            return False
    
    async def _check_api_servers(self) -> bool:
        """API 서버 연결 체크"""
        try:
            servers_to_check = [
                'https://api.upbit.com/v1/market/all',  # 업비트 API
                'https://query1.finance.yahoo.com',     # 야후 파이낸스
            ]
            
            success_count = 0
            for server in servers_to_check:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(server, timeout=5) as response:
                            if response.status == 200:
                                success_count += 1
                except:
                    continue
            
            # 최소 1개 서버라도 연결되면 OK
            return success_count > 0
        except:
            return False

# ============================================================================
# 🔔 통합 알림 시스템
# ============================================================================
class NotificationManager:
    """통합 알림 관리"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.telegram_session = None
        
        self.logger = logging.getLogger('NotificationManager')
    
    async def send_notification(self, message: str, priority: str = 'normal'):
        """통합 알림 전송"""
        try:
            # 우선순위별 이모지
            priority_emojis = {
                'emergency': '🚨',
                'warning': '⚠️',
                'info': 'ℹ️',
                'success': '✅',
                'normal': '📊'
            }
            
            emoji = priority_emojis.get(priority, '📊')
            formatted_message = f"{emoji} {message}"
            
            # 텔레그램 알림
            if self.config.TELEGRAM_ENABLED:
                await self._send_telegram(formatted_message, priority)
            
            # 로그에도 기록
            if priority == 'emergency':
                self.logger.critical(formatted_message)
            elif priority == 'warning':
                self.logger.warning(formatted_message)
            else:
                self.logger.info(formatted_message)
                
        except Exception as e:
            self.logger.error(f"알림 전송 실패: {e}")
    
    async def _send_telegram(self, message: str, priority: str):
        """텔레그램 메시지 전송"""
        if not self.config.TELEGRAM_BOT_TOKEN or not self.config.TELEGRAM_CHAT_ID:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            # 응급 상황시 알림음 설정
            disable_notification = priority not in ['emergency', 'warning']
            
            data = {
                'chat_id': self.config.TELEGRAM_CHAT_ID,
                'text': f"🏆 퀸트프로젝트 알림\n\n{message}",
                'disable_notification': disable_notification,
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.logger.debug("텔레그램 알림 전송 성공")
                    else:
                        self.logger.error(f"텔레그램 알림 실패: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"텔레그램 전송 오류: {e}")

# ============================================================================
# 📈 성과 추적 시스템
# ============================================================================
class PerformanceTracker:
    """통합 성과 추적"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.db_path = config.DB_PATH
        self._init_database()
        
        self.logger = logging.getLogger('PerformanceTracker')
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 거래 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT,
                    symbol TEXT,
                    action TEXT,
                    quantity REAL,
                    price REAL,
                    currency TEXT,
                    timestamp DATETIME,
                    profit_loss REAL,
                    profit_percent REAL,
                    fees REAL,
                    ai_confidence REAL,
                    ai_reasoning TEXT
                )
            ''')
            
            # AI 분석 결과 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    market TEXT,
                    timestamp DATETIME,
                    recommendation TEXT,
                    confidence REAL,
                    target_price REAL,
                    stop_loss REAL,
                    reasoning TEXT,
                    executed BOOLEAN
                )
            ''')
            
            # 일일 성과 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date DATE PRIMARY KEY,
                    strategy TEXT,
                    total_value REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    daily_return REAL,
                    positions_count INTEGER
                )
            ''')
            
            # 시스템 로그 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    level TEXT,
                    component TEXT,
                    message TEXT,
                    details TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def record_ai_analysis(self, symbol: str, market: str, analysis: Dict):
        """AI 분석 결과 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_analysis 
                (symbol, market, timestamp, recommendation, confidence, target_price, stop_loss, reasoning, executed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, market, datetime.now().isoformat(),
                analysis.get('recommendation', 'HOLD'),
                analysis.get('confidence', 0),
                analysis.get('target_price', 0),
                analysis.get('stop_loss', 0),
                analysis.get('reasoning', ''),
                False
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"AI 분석 기록 실패: {e}")
    
    def record_trade(self, strategy: str, symbol: str, action: str, quantity: float, 
                    price: float, currency: str, profit_loss: float = 0, fees: float = 0,
                    ai_confidence: float = 0, ai_reasoning: str = ''):
        """거래 기록 (AI 정보 포함)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            profit_percent = 0
            if action == 'SELL' and profit_loss != 0:
                profit_percent = (profit_loss / (quantity * price - profit_loss)) * 100
            
            cursor.execute('''
                INSERT INTO trades 
                (strategy, symbol, action, quantity, price, currency, timestamp, profit_loss, profit_percent, fees, ai_confidence, ai_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (strategy, symbol, action, quantity, price, currency, 
                  datetime.now().isoformat(), profit_loss, profit_percent, fees, ai_confidence, ai_reasoning))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"거래 기록: {strategy} {symbol} {action} {quantity} (AI신뢰도: {ai_confidence}%)")
            
        except Exception as e:
            self.logger.error(f"거래 기록 실패: {e}")
    
    def record_daily_performance(self, strategy: str, total_value: float, 
                               unrealized_pnl: float, realized_pnl: float, positions_count: int):
        """일일 성과 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            daily_return = (unrealized_pnl + realized_pnl) / total_value * 100 if total_value > 0 else 0
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_performance 
                (date, strategy, total_value, unrealized_pnl, realized_pnl, daily_return, positions_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (today.isoformat(), strategy, total_value, unrealized_pnl, 
                  realized_pnl, daily_return, positions_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"일일 성과 기록 실패: {e}")
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """성과 요약 조회 (AI 성과 포함)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).date()
            
            # 전략별 성과
            cursor.execute('''
                SELECT strategy, SUM(profit_loss) as total_profit, COUNT(*) as trade_count,
                       AVG(profit_percent) as avg_profit_pct, 
                       SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                       AVG(ai_confidence) as avg_ai_confidence
                FROM trades 
                WHERE date(timestamp) >= ? AND action = 'SELL'
                GROUP BY strategy
            ''', (start_date.isoformat(),))
            
            strategy_performance = {}
            for row in cursor.fetchall():
                strategy, total_profit, trade_count, avg_profit_pct, winning_trades, avg_ai_confidence = row
                win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0
                
                strategy_performance[strategy] = {
                    'total_profit': total_profit or 0,
                    'trade_count': trade_count or 0,
                    'avg_profit_pct': avg_profit_pct or 0,
                    'win_rate': win_rate,
                    'winning_trades': winning_trades or 0,
                    'avg_ai_confidence': avg_ai_confidence or 0
                }
            
            # AI 분석 성과
            cursor.execute('''
                SELECT recommendation, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM ai_analysis 
                WHERE date(timestamp) >= ?
                GROUP BY recommendation
            ''', (start_date.isoformat(),))
            
            ai_performance = {}
            for row in cursor.fetchall():
                recommendation, count, avg_confidence = row
                ai_performance[recommendation] = {
                    'count': count,
                    'avg_confidence': avg_confidence or 0
                }
            
            conn.close()
            
            return {
                'strategy_performance': strategy_performance,
                'ai_performance': ai_performance
            }
            
        except Exception as e:
            self.logger.error(f"성과 요약 조회 실패: {e}")
            return {}

# ============================================================================
# 🔄 자동 백업 시스템
# ============================================================================
class BackupManager:
    """자동 백업 관리"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.backup_path = Path(config.BACKUP_PATH)
        self.backup_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('BackupManager')
    
    async def perform_backup(self):
        """백업 실행"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.backup_path / f"backup_{timestamp}"
            backup_dir.mkdir(exist_ok=True)
            
            # 데이터베이스 백업
            if os.path.exists(self.config.DB_PATH):
                shutil.copy2(self.config.DB_PATH, backup_dir / "quant_core.db")
            
            # 전략별 데이터베이스 백업
            strategy_dbs = [
                './data/us_performance.db',
                './data/japan_performance.db',
                './data/india_performance.db',
                './data/crypto_performance.db'
            ]
            
            for db_path in strategy_dbs:
                if os.path.exists(db_path):
                    shutil.copy2(db_path, backup_dir / os.path.basename(db_path))
            
            # 설정 파일 백업
            config_files = ['.env', 'settings.yaml', 'positions.json']
            for config_file in config_files:
                if os.path.exists(config_file):
                    shutil.copy2(config_file, backup_dir / config_file)
            
            # 백업 압축
            shutil.make_archive(str(backup_dir), 'zip', str(backup_dir))
            shutil.rmtree(backup_dir)  # 원본 폴더 삭제
            
            self.logger.info(f"✅ 백업 완료: {backup_dir}.zip")
            
            # 오래된 백업 정리
            await self._cleanup_old_backups()
            
        except Exception as e:
            self.logger.error(f"백업 실패: {e}")
    
    async def _cleanup_old_backups(self):
        """오래된 백업 정리"""
        try:
            backup_files = list(self.backup_path.glob("backup_*.zip"))
            
            # 날짜순 정렬
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 최근 30개만 유지
            for old_backup in backup_files[30:]:
                old_backup.unlink()
                self.logger.info(f"오래된 백업 삭제: {old_backup.name}")
                
        except Exception as e:
            self.logger.error(f"백업 정리 실패: {e}")

# ============================================================================
# 🏆 퀸트프로젝트 통합 코어 시스템
# ============================================================================
class QuantProjectCore:
    """퀸트프로젝트 통합 코어 시스템 (AI 기반)"""
    
    def __init__(self):
        # 설정 로드
        self.config = CoreConfig()
        
        # 로깅 설정 (가장 먼저)
        self._setup_logging()
        
        # 로거 초기화 (로깅 설정 직후)
        self.logger = logging.getLogger('QuantCore')
        
        # 핵심 컴포넌트 초기화
        self.emergency_detector = EmergencyErrorDetector(self.config)
        self.ibkr_manager = IBKRManager(self.config)
        self.ai_engine = AIAnalysisEngine(self.config)
        self.ai_trader = AITradingExecutor(self.config, self.ai_engine, self.ibkr_manager)
        self.position_manager = UnifiedPositionManager(self.config, self.ibkr_manager)
        self.network_monitor = NetworkMonitor(self.config, self.ibkr_manager)
        self.notification_manager = NotificationManager(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        self.backup_manager = BackupManager(self.config)
        
        # 전략 인스턴스
        self.strategies = {}
        self._init_strategies()
        
        # 시스템 상태
        self.running = False
        self.start_time = None
        
        # AI 매매 대상 종목 리스트
        self.ai_watchlist = {
            'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
            'JAPAN': ['7203', '6098', '9984', '6758', '8058'],  # 도요타, 렌고, 소프트뱅크 등
            'INDIA': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR']
        }
    
    def _setup_logging(self):
        """로깅 설정"""
        log_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # 파일 핸들러
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'quant_core.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    def _init_strategies(self):
        """전략 초기화"""
        try:
            # 미국 전략
            if self.config.US_ENABLED and US_AVAILABLE:
                self.strategies['US'] = USStrategy()
                self.logger.info("✅ 미국 전략 초기화 완료")
            
            # 일본 전략
            if self.config.JAPAN_ENABLED and JAPAN_AVAILABLE:
                self.strategies['JAPAN'] = JapanStrategy()
                self.logger.info("✅ 일본 전략 초기화 완료")
            
            # 인도 전략
            if self.config.INDIA_ENABLED and INDIA_AVAILABLE:
                self.strategies['INDIA'] = IndiaStrategy()
                self.logger.info("✅ 인도 전략 초기화 완료")
            
            # 암호화폐 전략
            if self.config.CRYPTO_ENABLED and CRYPTO_AVAILABLE:
                self.strategies['CRYPTO'] = CryptoStrategy()
                self.logger.info("✅ 암호화폐 전략 초기화 완료")
            
            if not self.strategies:
                self.logger.warning("⚠️ 활성화된 전략이 없습니다")
                
        except Exception as e:
            self.logger.error(f"전략 초기화 실패: {e}")
    
    async def start_system(self):
        """시스템 시작"""
        try:
            self.logger.info("🏆 퀸트프로젝트 통합 시스템 시작! (AI 기반)")
            self.start_time = datetime.now()
            self.running = True
            
            # IBKR 연결
            if IBKR_AVAILABLE:
                await self.ibkr_manager.connect()
            
            # OpenAI 연결 확인
            ai_status = "✅" if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY else "❌"
            
            # 시작 알림
            await self.notification_manager.send_notification(
                f"🚀 퀸트프로젝트 AI 시스템 시작\n"
                f"활성 전략: {', '.join(self.strategies.keys())}\n"
                f"IBKR 연결: {'✅' if self.ibkr_manager.connected else '❌'}\n"
                f"AI 분석: {ai_status}\n"
                f"자동매매: {'✅' if self.config.AI_AUTO_TRADE else '❌'}\n"
                f"포트폴리오: {self.config.TOTAL_PORTFOLIO_VALUE:,.0f}원",
                'success'
            )
            
            # 백그라운드 태스크 시작
            tasks = [
                asyncio.create_task(self._main_trading_loop()),
                asyncio.create_task(self._ai_analysis_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self.network_monitor.start_monitoring()),
                asyncio.create_task(self._backup_loop())
            ]
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"시스템 시작 실패: {e}")
            await self.emergency_shutdown(f"시스템 시작 실패: {e}")
    
    async def _main_trading_loop(self):
        """메인 거래 루프"""
        while self.running:
            try:
                # 시스템 건강 상태 체크
                health_status = self.emergency_detector.check_system_health()
                
                if health_status['emergency_needed']:
                    await self.emergency_shutdown("시스템 건강 상태 위험")
                    break
                
                # 각 전략 실행 (요일별)
                current_weekday = datetime.now().weekday()
                
                for strategy_name, strategy_instance in self.strategies.items():
                    try:
                        if self._should_run_strategy(strategy_name, current_weekday):
                            await self._execute_strategy(strategy_name, strategy_instance)
                    except Exception as e:
                        error_critical = self.emergency_detector.record_error(
                            f"{strategy_name}_error", str(e), critical=True
                        )
                        if error_critical:
                            await self.emergency_shutdown(f"{strategy_name} 전략 치명적 오류")
                            break
                
                # 포지션 업데이트
                await self.position_manager.update_all_positions()
                
                # 1시간 대기
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"메인 루프 오류: {e}")
                await asyncio.sleep(60)
    
    async def _ai_analysis_loop(self):
        """AI 분석 루프 (독립적으로 실행)"""
        while self.running:
            try:
                if not self.config.AI_ANALYSIS_ENABLED:
                    await asyncio.sleep(1800)  # AI 비활성화시 30분 대기
                    continue
                
                # 각 시장별 AI 분석 실행
                for market, symbols in self.ai_watchlist.items():
                    if market in self.strategies:  # 해당 전략이 활성화된 경우만
                        for symbol in symbols:
                            try:
                                # AI 분석 실행
                                analysis = await self.ai_engine.analyze_market_trend(symbol, market)
                                
                                if 'error' not in analysis:
                                    # 분석 결과 저장
                                    self.performance_tracker.record_ai_analysis(symbol, market, analysis)
                                    
                                    # 자동매매 실행 (설정된 경우)
                                    if self.config.AI_AUTO_TRADE:
                                        trade_result = await self.ai_trader.execute_ai_trading(symbol, market)
                                        
                                        if 'error' not in trade_result:
                                            # 성공적인 거래 알림
                                            await self.notification_manager.send_notification(
                                                f"🤖 AI 자동매매 실행\n"
                                                f"종목: {symbol} ({market})\n"
                                                f"액션: {trade_result.get('execution_result', {}).get('action', 'N/A')}\n"
                                                f"신뢰도: {analysis.get('confidence', 0)}%",
                                                'info'
                                            )
                                
                                # 과부하 방지를 위한 대기
                                await asyncio.sleep(10)
                                
                            except Exception as e:
                                self.logger.error(f"AI 분석 실패 {symbol}: {e}")
                                continue
                
                # AI 분석은 30분마다 실행
                await asyncio.sleep(1800)
                
            except Exception as e:
                self.logger.error(f"AI 분석 루프 오류: {e}")
                await asyncio.sleep(300)
    
    def _should_run_strategy(self, strategy_name: str, weekday: int) -> bool:
        """전략 실행 여부 판단"""
        # 월요일(0), 화요일(1), 수요일(2), 목요일(3), 금요일(4)
        
        if strategy_name == 'US':
            return weekday in [1, 3]  # 화목
        elif strategy_name == 'JAPAN':
            return weekday in [1, 3]  # 화목
        elif strategy_name == 'INDIA':
            return weekday == 2  # 수요일
        elif strategy_name == 'CRYPTO':
            return weekday in [0, 4]  # 월금
        
        return False
    
    async def _execute_strategy(self, strategy_name: str, strategy_instance):
        """개별 전략 실행"""
        try:
            self.logger.info(f"🎯 {strategy_name} 전략 실행 시작")
            
            # 전략별 필요 통화 환전
            if strategy_name == 'JAPAN':
                await self.ibkr_manager.auto_currency_exchange('JPY', 10000000)  # 1천만엔
            elif strategy_name == 'INDIA':
                await self.ibkr_manager.auto_currency_exchange('INR', 7500000)   # 750만루피
            
            # 전략 실행 (각 전략의 메인 함수 호출)
            if hasattr(strategy_instance, 'run_strategy'):
                result = await strategy_instance.run_strategy()
            elif hasattr(strategy_instance, 'execute_legendary_strategy'):
                result = await strategy_instance.execute_legendary_strategy()
            else:
                self.logger.warning(f"{strategy_name} 전략 실행 메서드를 찾을 수 없음")
                return
            
            self.logger.info(f"✅ {strategy_name} 전략 실행 완료")
            
            # 성과 기록
            if result:
                # 결과에 따른 성과 기록 로직
                pass
                
        except Exception as e:
            self.logger.error(f"{strategy_name} 전략 실행 실패: {e}")
            raise
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.running:
            try:
                # 포지션 모니터링
                portfolio_summary = self.position_manager.get_portfolio_summary()
                
                # 위험 상황 체크
                total_loss_pct = (portfolio_summary['total_unrealized_pnl'] / 
                                 self.config.TOTAL_PORTFOLIO_VALUE * 100)
                
                if total_loss_pct < -self.config.MAX_PORTFOLIO_RISK * 100:
                    await self.notification_manager.send_notification(
                        f"🚨 포트폴리오 손실 한계 초과!\n"
                        f"현재 손실: {total_loss_pct:.2f}%\n"
                        f"한계: {self.config.MAX_PORTFOLIO_RISK * 100:.1f}%",
                        'emergency'
                    )
                
                # 주기적 상태 보고 (6시간마다)
                if datetime.now().hour % 6 == 0 and datetime.now().minute < 5:
                    await self._send_status_report(portfolio_summary)
                
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(60)
    
    async def _backup_loop(self):
        """백업 루프"""
        while self.running:
            try:
                # 매일 새벽 3시에 백업
                now = datetime.now()
                if now.hour == 3 and now.minute < 10:
                    await self.backup_manager.perform_backup()
                    await asyncio.sleep(600)  # 10분 대기
                
                await asyncio.sleep(300)  # 5분마다 체크
                
            except Exception as e:
                self.logger.error(f"백업 루프 오류: {e}")
                await asyncio.sleep(3600)  # 1시간 대기
    
    async def _send_status_report(self, portfolio_summary: Dict):
        """상태 보고서 전송 (AI 정보 포함)"""
        try:
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            # 최근 AI 성과 조회
            performance_data = self.performance_tracker.get_performance_summary(7)  # 최근 7일
            ai_performance = performance_data.get('ai_performance', {})
            
            report = (
                f"📊 퀸트프로젝트 AI 상태 보고\n\n"
                f"🕐 가동시간: {uptime}\n"
                f"💼 총 포지션: {portfolio_summary['total_positions']}개\n"
                f"💰 미실현 손익: {portfolio_summary['total_unrealized_pnl']:+,.0f}원\n"
                f"📈 수익 포지션: {portfolio_summary['profitable_positions']}개\n"
                f"📉 손실 포지션: {portfolio_summary['losing_positions']}개\n\n"
                f"전략별 현황:\n"
            )
            
            for strategy, data in portfolio_summary['by_strategy'].items():
                report += f"  {strategy}: {data['count']}개 ({data['pnl']:+,.0f}원)\n"
            
            # AI 성과 추가
            if ai_performance:
                report += f"\n🤖 AI 분석 현황 (최근 7일):\n"
                for recommendation, data in ai_performance.items():
                    report += f"  {recommendation}: {data['count']}회 (평균신뢰도: {data['avg_confidence']:.1f}%)\n"
            
            await self.notification_manager.send_notification(report, 'info')
            
        except Exception as e:
            self.logger.error(f"상태 보고서 전송 실패: {e}")
    
    async def emergency_shutdown(self, reason: str):
        """응급 종료"""
        try:
            self.logger.critical(f"🚨 응급 종료: {reason}")
            
            # 응급 알림
            await self.notification_manager.send_notification(
                f"🚨 시스템 응급 종료\n"
                f"사유: {reason}\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                'emergency'
            )
            
            # 응급 매도 (설정된 경우)
            if self.config.EMERGENCY_SELL_ON_ERROR:
                if self.ibkr_manager.connected:
                    await self.ibkr_manager.emergency_sell_all()
            
            # 응급 백업
            await self.backup_manager.perform_backup()
            
            self.running = False
            
        except Exception as e:
            self.logger.error(f"응급 종료 실패: {e}")
    
    async def graceful_shutdown(self):
        """정상 종료"""
        try:
            self.logger.info("🛑 시스템 정상 종료 시작")
            
            # 종료 알림
            await self.notification_manager.send_notification(
                f"🛑 시스템 정상 종료\n"
                f"가동시간: {datetime.now() - self.start_time if self.start_time else '알수없음'}",
                'info'
            )
            
            # 네트워크 모니터링 중지
            self.network_monitor.monitoring = False
            
            # 최종 백업
            await self.backup_manager.perform_backup()
            
            # IBKR 연결 해제
            if self.ibkr_manager.connected:
                await self.ibkr_manager.ib.disconnectAsync()
            
            self.running = False
            self.logger.info("✅ 시스템 정상 종료 완료")
            
        except Exception as e:
            self.logger.error(f"정상 종료 실패: {e}")

# ============================================================================
# 🎮 편의 함수들
# ============================================================================
async def get_system_status():
    """시스템 상태 조회"""
    core = QuantProjectCore()
    await core.ibkr_manager.connect()
    await core.position_manager.update_all_positions()
    
    summary = core.position_manager.get_portfolio_summary()
    
    return {
        'strategies': list(core.strategies.keys()),
        'ibkr_connected': core.ibkr_manager.connected,
        'ai_enabled': core.config.AI_ANALYSIS_ENABLED,
        'auto_trade': core.config.AI_AUTO_TRADE,
        'total_positions': summary['total_positions'],
        'total_unrealized_pnl': summary['total_unrealized_pnl'],
        'by_strategy': summary['by_strategy']
    }

async def emergency_sell_all():
    """응급 전량 매도"""
    core = QuantProjectCore()
    await core.ibkr_manager.connect()
    
    if core.ibkr_manager.connected:
        results = await core.ibkr_manager.emergency_sell_all()
        return results
    else:
        return {}

async def run_ai_analysis(symbol: str, market: str = 'US'):
    """단일 종목 AI 분석 실행"""
    core = QuantProjectCore()
    analysis = await core.ai_engine.analyze_market_trend(symbol, market)
    return analysis

async def execute_ai_trade(symbol: str, market: str = 'US'):
    """단일 종목 AI 자동매매 실행"""
    core = QuantProjectCore()
    await core.ibkr_manager.connect()
    
    if core.ibkr_manager.connected:
        result = await core.ai_trader.execute_ai_trading(symbol, market)
        return result
    else:
        return {'error': 'IBKR 연결 실패'}

# ============================================================================
# 🏁 메인 실행부
# ============================================================================
async def main():
    """메인 실행 함수"""
    # 신호 핸들러 설정
    def signal_handler(signum, frame):
        print("\n🛑 종료 신호 수신, 정상 종료 중...")
        asyncio.create_task(core.graceful_shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 코어 시스템 생성
    core = QuantProjectCore()
    
    try:
        print("🏆" + "="*70)
        print("🏆 퀸트프로젝트 통합 코어 시스템 v1.2.0 (AI 기반)")
        print("🏆" + "="*70)
        print("✨ 4대 전략 통합 관리")
        print("✨ IBKR 자동 환전")
        print("✨ OpenAI GPT-4 AI 분석")
        print("✨ AI 기반 자동매매")
        print("✨ 네트워크 모니터링")
        print("✨ 응급 오류 감지")
        print("✨ 통합 포지션 관리")
        print("✨ 실시간 알림")
        print("✨ 자동 백업")
        print("🏆" + "="*70)
        
        # OpenAI 상태 확인
        if OPENAI_AVAILABLE and core.config.OPENAI_API_KEY:
            print("🤖 OpenAI 연동: ✅")
            print(f"🤖 AI 분석: {'✅' if core.config.AI_ANALYSIS_ENABLED else '❌'}")
            print(f"🤖 자동매매: {'✅' if core.config.AI_AUTO_TRADE else '❌'}")
        else:
            print("🤖 OpenAI 연동: ❌ (API 키 확인 필요)")
        
        print("🏆" + "="*70)
        
        # 시스템 시작
        await core.start_system()
        
    except KeyboardInterrupt:
        print("\n👋 사용자 중단")
        await core.graceful_shutdown()
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")
        await core.emergency_shutdown(f"시스템 오류: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 퀸트프로젝트 종료")
        sys.exit(0)
