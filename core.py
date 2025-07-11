#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 통합 코어 시스템 (core.py) - 최적화 버전
=============================================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 (4대 전략 통합)
✨ 핵심 기능:
- 4대 전략 통합 관리 시스템
- IBKR 자동 환전 기능 (달러 ↔ 엔/루피)
- OpenAI GPT-4 기반 기술적 분석 확신도 체크 (최적화)
- 네트워크 모니터링 + 끊김 시 전량 매도
- 통합 포지션 관리 + 리스크 제어
- 실시간 모니터링 + 알림 시스템
- 성과 추적 + 자동 백업
- 🚨 응급 오류 감지 시스템

Author: 퀸트마스터팀
Version: 1.3.0 (AI 최적화 + 기술적 분석 전용)
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

# OpenAI 연동 (기술적 분석 전용)
try:
    from openai import AsyncOpenAI
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
        
        # OpenAI 설정 (최적화)
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        self.OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')  # 저렴한 모델 사용
        self.AI_TECHNICAL_CHECK_ENABLED = os.getenv('AI_TECHNICAL_CHECK_ENABLED', 'true').lower() == 'true'
        self.AI_CONFIDENCE_THRESHOLD_LOW = float(os.getenv('AI_CONFIDENCE_THRESHOLD_LOW', 0.4))
        self.AI_CONFIDENCE_THRESHOLD_HIGH = float(os.getenv('AI_CONFIDENCE_THRESHOLD_HIGH', 0.7))
        self.AI_MONTHLY_TOKEN_LIMIT = int(os.getenv('AI_MONTHLY_TOKEN_LIMIT', 100000))  # 월 토큰 제한
        
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
# 🤖 AI 기술적 분석 확신도 체커 (최적화)
# ============================================================================
class AITechnicalConfidenceChecker:
    """AI 기반 기술적 분석 확신도 체크 (토큰 최적화)"""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.client = None
        self.monthly_token_usage = 0
        self.current_month = datetime.now().month
        
        if OPENAI_AVAILABLE and config.OPENAI_API_KEY:
            self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        
        self.logger = logging.getLogger('AIConfidenceChecker')
        
        # 월별 토큰 사용량 추적
        self._load_token_usage()
    
    def _load_token_usage(self):
        """월별 토큰 사용량 로드"""
        try:
            usage_file = './data/ai_token_usage.json'
            if os.path.exists(usage_file):
                with open(usage_file, 'r') as f:
                    usage_data = json.load(f)
                    current_month_key = f"{datetime.now().year}-{datetime.now().month}"
                    self.monthly_token_usage = usage_data.get(current_month_key, 0)
        except Exception as e:
            self.logger.error(f"토큰 사용량 로드 실패: {e}")
            self.monthly_token_usage = 0
    
    def _save_token_usage(self):
        """월별 토큰 사용량 저장"""
        try:
            usage_file = './data/ai_token_usage.json'
            os.makedirs(os.path.dirname(usage_file), exist_ok=True)
            
            usage_data = {}
            if os.path.exists(usage_file):
                with open(usage_file, 'r') as f:
                    usage_data = json.load(f)
            
            current_month_key = f"{datetime.now().year}-{datetime.now().month}"
            usage_data[current_month_key] = self.monthly_token_usage
            
            with open(usage_file, 'w') as f:
                json.dump(usage_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"토큰 사용량 저장 실패: {e}")
    
    def should_check_confidence(self, strategy_confidence: float) -> bool:
        """AI 확신도 체크 필요 여부 판단"""
        # 월 토큰 제한 체크
        if self.monthly_token_usage >= self.config.AI_MONTHLY_TOKEN_LIMIT:
            return False
        
        # 애매한 신뢰도 구간에서만 AI 호출
        return (self.config.AI_CONFIDENCE_THRESHOLD_LOW <= strategy_confidence <= 
                self.config.AI_CONFIDENCE_THRESHOLD_HIGH)
    
    async def check_technical_confidence(self, symbol: str, market: str, 
                                       strategy_signal: Dict, market_data: Dict) -> Dict[str, Any]:
        """기술적 분석 기반 확신도 체크 (최적화된 프롬프트)"""
        if not self.client or not self.config.AI_TECHNICAL_CHECK_ENABLED:
            return {'confidence_adjustment': 0, 'reasoning': 'AI 비활성화'}
        
        # 토큰 제한 체크
        if not self.should_check_confidence(strategy_signal.get('confidence', 0.5)):
            return {'confidence_adjustment': 0, 'reasoning': '신뢰도 범위 외 또는 토큰 제한'}
        
        try:
            # 핵심 기술적 지표만 사용한 최적화된 프롬프트
            technical_prompt = f"""기술지표 확신도 체크:
종목: {symbol}
신호: {strategy_signal.get('action', 'HOLD')}
현재가: {market_data.get('current_price', 0)}
RSI: {market_data.get('rsi', 50)}
MA20: {market_data.get('sma_20', 0)}
변동성: {market_data.get('volatility', 0):.3f}

기술적 관점에서 이 신호의 확신도를 -0.2~+0.2 범위로 조정하세요.
응답형식: {{"adjustment": 숫자, "reason": "간단한이유"}}"""
            
            # 토큰 수 추정 (대략 150-200 토큰)
            estimated_tokens = 200
            
            if self.monthly_token_usage + estimated_tokens > self.config.AI_MONTHLY_TOKEN_LIMIT:
                return {'confidence_adjustment': 0, 'reasoning': '월 토큰 제한 초과'}
            
            response = await self._call_openai_api(technical_prompt, max_tokens=100)
            
            # 토큰 사용량 업데이트
            self.monthly_token_usage += estimated_tokens
            self._save_token_usage()
            
            try:
                result = json.loads(response)
                adjustment = float(result.get('adjustment', 0))
                
                # 조정값 범위 제한
                adjustment = max(-0.2, min(0.2, adjustment))
                
                self.logger.info(f"🤖 AI 확신도 체크: {symbol} 조정={adjustment:+.2f}")
                
                return {
                    'confidence_adjustment': adjustment,
                    'reasoning': result.get('reason', ''),
                    'tokens_used': estimated_tokens,
                    'monthly_usage': self.monthly_token_usage
                }
                
            except json.JSONDecodeError:
                # JSON 파싱 실패시 간단한 규칙 기반 조정
                return self._fallback_confidence_check(market_data)
                
        except Exception as e:
            self.logger.error(f"AI 확신도 체크 실패: {e}")
            return self._fallback_confidence_check(market_data)
    
    def _fallback_confidence_check(self, market_data: Dict) -> Dict[str, Any]:
        """AI 실패시 간단한 기술적 분석 기반 조정"""
        adjustment = 0
        
        try:
            rsi = market_data.get('rsi', 50)
            volatility = market_data.get('volatility', 0)
            
            # RSI 기반 조정
            if rsi > 70:  # 과매수
                adjustment -= 0.1
            elif rsi < 30:  # 과매도
                adjustment += 0.1
            
            # 변동성 기반 조정
            if volatility > 0.3:  # 고변동성
                adjustment -= 0.05
            
            return {
                'confidence_adjustment': adjustment,
                'reasoning': f'Fallback: RSI={rsi}, Vol={volatility:.3f}',
                'tokens_used': 0
            }
            
        except Exception as e:
            return {'confidence_adjustment': 0, 'reasoning': f'Fallback 실패: {e}'}
    
    async def _call_openai_api(self, prompt: str, max_tokens: int = 100) -> str:
        """OpenAI API 호출 (최적화)"""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,  # gpt-4o-mini 사용
                messages=[
                    {"role": "system", "content": "기술적 분석 전문가. 간결한 JSON 응답만."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1  # 일관성을 위해 낮은 온도
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API 호출 실패: {e}")
            raise
    
    def get_monthly_usage_summary(self) -> Dict[str, Any]:
        """월별 사용량 요약"""
        return {
            'current_usage': self.monthly_token_usage,
            'limit': self.config.AI_MONTHLY_TOKEN_LIMIT,
            'remaining': self.config.AI_MONTHLY_TOKEN_LIMIT - self.monthly_token_usage,
            'usage_percentage': (self.monthly_token_usage / self.config.AI_MONTHLY_TOKEN_LIMIT * 100),
            'month': f"{datetime.now().year}-{datetime.now().month}"
        }

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
                    strategy_confidence REAL,
                    ai_confidence_adjustment REAL,
                    final_confidence REAL,
                    ai_reasoning TEXT
                )
            ''')
            
            # AI 확신도 체크 결과 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_confidence_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    market TEXT,
                    timestamp DATETIME,
                    strategy_confidence REAL,
                    ai_adjustment REAL,
                    final_confidence REAL,
                    reasoning TEXT,
                    tokens_used INTEGER,
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
            
            # AI 토큰 사용량 추적 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    tokens_used INTEGER,
                    api_calls INTEGER,
                    cost_estimate REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def record_ai_confidence_check(self, symbol: str, market: str, strategy_confidence: float, 
                                 ai_result: Dict, executed: bool = False):
        """AI 확신도 체크 결과 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_confidence_checks 
                (symbol, market, timestamp, strategy_confidence, ai_adjustment, final_confidence, 
                 reasoning, tokens_used, executed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, market, datetime.now().isoformat(),
                strategy_confidence,
                ai_result.get('confidence_adjustment', 0),
                strategy_confidence + ai_result.get('confidence_adjustment', 0),
                ai_result.get('reasoning', ''),
                ai_result.get('tokens_used', 0),
                executed
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"AI 확신도 체크 기록 실패: {e}")
    
    def record_trade(self, strategy: str, symbol: str, action: str, quantity: float, 
                    price: float, currency: str, profit_loss: float = 0, fees: float = 0,
                    strategy_confidence: float = 0, ai_adjustment: float = 0, 
                    ai_reasoning: str = ''):
        """거래 기록 (AI 확신도 정보 포함)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            profit_percent = 0
            if action == 'SELL' and profit_loss != 0:
                profit_percent = (profit_loss / (quantity * price - profit_loss)) * 100
            
            final_confidence = strategy_confidence + ai_adjustment
            
            cursor.execute('''
                INSERT INTO trades 
                (strategy, symbol, action, quantity, price, currency, timestamp, profit_loss, 
                 profit_percent, fees, strategy_confidence, ai_confidence_adjustment, 
                 final_confidence, ai_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (strategy, symbol, action, quantity, price, currency, 
                  datetime.now().isoformat(), profit_loss, profit_percent, fees, 
                  strategy_confidence, ai_adjustment, final_confidence, ai_reasoning))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"거래 기록: {strategy} {symbol} {action} {quantity} (최종신뢰도: {final_confidence:.2f})")
            
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
                       AVG(final_confidence) as avg_final_confidence
                FROM trades 
                WHERE date(timestamp) >= ? AND action = 'SELL'
                GROUP BY strategy
            ''', (start_date.isoformat(),))
            
            strategy_performance = {}
            for row in cursor.fetchall():
                strategy, total_profit, trade_count, avg_profit_pct, winning_trades, avg_final_confidence = row
                win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0
                
                strategy_performance[strategy] = {
                    'total_profit': total_profit or 0,
                    'trade_count': trade_count or 0,
                    'avg_profit_pct': avg_profit_pct or 0,
                    'win_rate': win_rate,
                    'winning_trades': winning_trades or 0,
                    'avg_final_confidence': avg_final_confidence or 0
                }
            
            # AI 확신도 체크 성과
            cursor.execute('''
                SELECT AVG(ai_adjustment) as avg_adjustment, 
                       COUNT(*) as total_checks,
                       SUM(tokens_used) as total_tokens,
                       AVG(CASE WHEN executed THEN 1.0 ELSE 0.0 END) as execution_rate
                FROM ai_confidence_checks 
                WHERE date(timestamp) >= ?
            ''', (start_date.isoformat(),))
            
            ai_stats = cursor.fetchone()
            ai_performance = {
                'avg_adjustment': ai_stats[0] or 0,
                'total_checks': ai_stats[1] or 0,
                'total_tokens': ai_stats[2] or 0,
                'execution_rate': (ai_stats[3] or 0) * 100
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
            
            # AI 토큰 사용량 파일 백업
            ai_usage_file = './data/ai_token_usage.json'
            if os.path.exists(ai_usage_file):
                shutil.copy2(ai_usage_file, backup_dir / 'ai_token_usage.json')
            
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
# 💰 실제 자동매매 실행 엔진
# ============================================================================
class AutoTradingEngine:
    """실제 자동매매 실행 엔진"""
    
    def __init__(self, config: CoreConfig, ibkr_manager: IBKRManager, 
                 performance_tracker: PerformanceTracker, notification_manager):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.performance_tracker = performance_tracker
        self.notification_manager = notification_manager
        
        # 안전장치 설정
        self.MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', 20))
        self.MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.05))  # 5%
        self.STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', 0.03))  # 3%
        self.MIN_CONFIDENCE_FOR_TRADE = float(os.getenv('MIN_CONFIDENCE_FOR_TRADE', 0.7))
        
        # 일일 거래 추적
        self.daily_trades = {}
        self.active_orders = {}
        
        self.logger = logging.getLogger('AutoTradingEngine')
        
        # 업비트 클라이언트 (암호화폐용)
        self.upbit = None
        if UPBIT_AVAILABLE and config.UPBIT_ACCESS_KEY:
            self.upbit = pyupbit.Upbit(config.UPBIT_ACCESS_KEY, config.UPBIT_SECRET_KEY)
    
    async def execute_signal(self, strategy_name: str, signal: Dict) -> Dict[str, Any]:
        """신호 실행 (실제 매매)"""
        try:
            symbol = signal.get('symbol', '')
            action = signal.get('action', 'HOLD')
            final_confidence = signal.get('final_confidence', 0)
            
            if action == 'HOLD':
                return {'message': 'HOLD 신호 - 매매 없음'}
            
            # 안전장치 체크
            safety_check = await self._safety_check(strategy_name, signal)
            if not safety_check['safe']:
                return {'error': f'안전장치 차단: {safety_check["reason"]}'}
            
            # 확신도 체크
            if final_confidence < self.MIN_CONFIDENCE_FOR_TRADE:
                return {'message': f'낮은 확신도 ({final_confidence:.2f}) - 매매 건너뜀'}
            
            # 전략별 매매 실행
            if strategy_name == 'CRYPTO':
                result = await self._execute_crypto_trade(symbol, action, signal)
            else:  # US, JAPAN, INDIA
                result = await self._execute_stock_trade(strategy_name, symbol, action, signal)
            
            # 성공시 일일 거래 카운트 증가
            if result.get('status') == 'success':
                today = datetime.now().date().isoformat()
                if today not in self.daily_trades:
                    self.daily_trades[today] = 0
                self.daily_trades[today] += 1
                
                # 알림 전송
                await self.notification_manager.send_notification(
                    f"💰 자동매매 실행!\n"
                    f"전략: {strategy_name}\n"
                    f"종목: {symbol}\n"
                    f"액션: {action}\n"
                    f"확신도: {final_confidence:.2f}\n"
                    f"수량: {result.get('quantity', 0)}\n"
                    f"가격: {result.get('price', 0)}",
                    'success'
                )
                
                # 거래 기록
                self.performance_tracker.record_trade(
                    strategy=strategy_name,
                    symbol=symbol,
                    action=action,
                    quantity=result.get('quantity', 0),
                    price=result.get('price', 0),
                    currency=self._get_currency_for_strategy(strategy_name),
                    strategy_confidence=signal.get('original_confidence', 0),
                    ai_adjustment=signal.get('ai_adjustment', 0),
                    ai_reasoning=signal.get('ai_reasoning', '')
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"매매 실행 실패: {e}")
            return {'error': str(e)}
    
    async def _safety_check(self, strategy_name: str, signal: Dict) -> Dict[str, Any]:
        """안전장치 체크"""
        try:
            # 일일 거래 한도 체크
            today = datetime.now().date().isoformat()
            daily_count = self.daily_trades.get(today, 0)
            
            if daily_count >= self.MAX_DAILY_TRADES:
                return {'safe': False, 'reason': f'일일 거래 한도 초과 ({daily_count}/{self.MAX_DAILY_TRADES})'}
            
            # 시장 시간 체크
            if not self._is_market_open(strategy_name):
                return {'safe': False, 'reason': f'{strategy_name} 시장 휴장'}
            
            # 포지션 크기 체크
            position_size = signal.get('position_size_pct', 5) / 100
            if position_size > self.MAX_POSITION_SIZE:
                return {'safe': False, 'reason': f'포지션 크기 초과 ({position_size:.1%} > {self.MAX_POSITION_SIZE:.1%})'}
            
            # 계좌 잔고 체크
            if strategy_name != 'CRYPTO' and not self.ibkr_manager.connected:
                return {'safe': False, 'reason': 'IBKR 연결 끊김'}
            
            return {'safe': True, 'reason': '안전'}
            
        except Exception as e:
            return {'safe': False, 'reason': f'안전장치 체크 실패: {e}'}
    
    def _is_market_open(self, strategy_name: str) -> bool:
        """시장 개장 시간 체크"""
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        
        # 주말 체크
        if weekday >= 5:  # 토요일(5), 일요일(6)
            return strategy_name == 'CRYPTO'  # 암호화폐만 24/7
        
        # 전략별 시장 시간
        if strategy_name == 'US':
            # 미국 시장: 23:30 ~ 06:00 (한국시간)
            return hour >= 23 or hour < 6
        elif strategy_name == 'JAPAN':
            # 일본 시장: 09:00 ~ 15:00 (한국시간)
            return 9 <= hour < 15
        elif strategy_name == 'INDIA':
            # 인도 시장: 12:45 ~ 19:15 (한국시간)
            return 12 <= hour < 20
        elif strategy_name == 'CRYPTO':
            # 암호화폐: 24시간
            return True
        
        return False
    
    async def _execute_stock_trade(self, strategy_name: str, symbol: str, action: str, signal: Dict) -> Dict[str, Any]:
        """주식 매매 실행 (IBKR)"""
        if not self.ibkr_manager.connected:
            return {'error': 'IBKR 미연결'}
        
        try:
            # 환전 먼저 실행
            if strategy_name == 'JAPAN':
                await self.ibkr_manager.auto_currency_exchange('JPY', 10000000)
            elif strategy_name == 'INDIA':
                await self.ibkr_manager.auto_currency_exchange('INR', 7500000)
            
            # 계약 생성
            contract = self._create_contract(strategy_name, symbol)
            
            # 현재가 조회
            ticker = self.ibkr_manager.ib.reqMktData(contract)
            await asyncio.sleep(2)  # 데이터 수신 대기
            
            current_price = ticker.last or ticker.close
            if not current_price:
                return {'error': f'{symbol} 현재가 조회 실패'}
            
            # 수량 계산
            position_size_pct = signal.get('position_size_pct', 3) / 100
            allocation = self._get_allocation_for_strategy(strategy_name)
            max_investment = self.config.TOTAL_PORTFOLIO_VALUE * allocation * position_size_pct
            
            if action == 'BUY':
                quantity = int(max_investment / current_price)
                if quantity <= 0:
                    return {'error': '매수 수량 부족'}
                
                # 매수 주문
                order = MarketOrder('BUY', quantity)
                trade = self.ibkr_manager.ib.placeOrder(contract, order)
                
                # 주문 완료 대기
                for _ in range(30):
                    await asyncio.sleep(1)
                    if trade.isDone():
                        break
                
                if trade.isDone() and trade.orderStatus.status == 'Filled':
                    # 손절 주문 자동 설정
                    stop_price = current_price * (1 - self.STOP_LOSS_PCT)
                    stop_order = StopOrder('SELL', quantity, stop_price)
                    self.ibkr_manager.ib.placeOrder(contract, stop_order)
                    
                    return {
                        'status': 'success',
                        'action': 'BUY',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': trade.orderStatus.avgFillPrice or current_price,
                        'total_cost': quantity * (trade.orderStatus.avgFillPrice or current_price),
                        'stop_loss': stop_price
                    }
                else:
                    return {'error': f'매수 주문 실패: {trade.orderStatus.status}'}
            
            elif action == 'SELL':
                # 현재 포지션 확인
                await self.ibkr_manager._update_account_info()
                
                if symbol not in self.ibkr_manager.positions:
                    return {'error': '보유 포지션 없음'}
                
                position_info = self.ibkr_manager.positions[symbol]
                quantity = int(abs(position_info['position']))
                
                if quantity <= 0:
                    return {'error': '매도할 수량 없음'}
                
                # 매도 주문
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
                    sell_price = trade.orderStatus.avgFillPrice or current_price
                    profit_loss = (sell_price - avg_cost) * quantity
                    
                    return {
                        'status': 'success',
                        'action': 'SELL',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': sell_price,
                        'total_revenue': quantity * sell_price,
                        'profit_loss': profit_loss,
                        'profit_pct': (sell_price - avg_cost) / avg_cost * 100
                    }
                else:
                    return {'error': f'매도 주문 실패: {trade.orderStatus.status}'}
            
        except Exception as e:
            self.logger.error(f"{strategy_name} 주식 매매 실행 실패: {e}")
            return {'error': str(e)}
    
    async def _execute_crypto_trade(self, symbol: str, action: str, signal: Dict) -> Dict[str, Any]:
        """암호화폐 매매 실행 (업비트)"""
        if not self.upbit:
            return {'error': '업비트 미연결'}
        
        try:
            # 업비트 심볼 변환 (예: BTC -> KRW-BTC)
            upbit_symbol = f"KRW-{symbol}" if not symbol.startswith('KRW-') else symbol
            
            # 현재가 조회
            ticker = pyupbit.get_current_price(upbit_symbol)
            if not ticker:
                return {'error': f'{symbol} 현재가 조회 실패'}
            
            # 포지션 크기 계산
            position_size_pct = signal.get('position_size_pct', 5) / 100
            allocation = self.config.CRYPTO_ALLOCATION
            max_investment = self.config.TOTAL_PORTFOLIO_VALUE * allocation * position_size_pct
            
            if action == 'BUY':
                # 매수 가능 금액 확인
                balances = self.upbit.get_balances()
                krw_balance = 0
                
                for balance in balances:
                    if balance['currency'] == 'KRW':
                        krw_balance = float(balance['balance'])
                        break
                
                if krw_balance < max_investment:
                    max_investment = krw_balance * 0.99  # 수수료 고려
                
                if max_investment < 5000:  # 업비트 최소 주문 금액
                    return {'error': '매수 금액 부족 (최소 5,000원)'}
                
                # 매수 주문
                if self.config.UPBIT_DEMO_MODE:
                    # 데모 모드
                    quantity = max_investment / ticker
                    result = {
                        'status': 'success',
                        'action': 'BUY',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': ticker,
                        'total_cost': max_investment,
                        'demo_mode': True
                    }
                else:
                    # 실제 매수
                    order_result = self.upbit.buy_market_order(upbit_symbol, max_investment)
                    
                    if order_result:
                        result = {
                            'status': 'success',
                            'action': 'BUY',
                            'symbol': symbol,
                            'quantity': float(order_result.get('executed_volume', 0)),
                            'price': ticker,
                            'total_cost': max_investment,
                            'order_id': order_result.get('uuid')
                        }
                    else:
                        return {'error': '업비트 매수 주문 실패'}
                
                return result
            
            elif action == 'SELL':
                # 보유 수량 확인
                balances = self.upbit.get_balances()
                coin_balance = 0
                
                for balance in balances:
                    if balance['currency'] == symbol:
                        coin_balance = float(balance['balance'])
                        break
                
                if coin_balance <= 0:
                    return {'error': '보유 수량 없음'}
                
                # 매도 주문
                if self.config.UPBIT_DEMO_MODE:
                    # 데모 모드
                    result = {
                        'status': 'success',
                        'action': 'SELL',
                        'symbol': symbol,
                        'quantity': coin_balance,
                        'price': ticker,
                        'total_revenue': coin_balance * ticker,
                        'demo_mode': True
                    }
                else:
                    # 실제 매도
                    order_result = self.upbit.sell_market_order(upbit_symbol, coin_balance)
                    
                    if order_result:
                        result = {
                            'status': 'success',
                            'action': 'SELL',
                            'symbol': symbol,
                            'quantity': coin_balance,
                            'price': ticker,
                            'total_revenue': coin_balance * ticker,
                            'order_id': order_result.get('uuid')
                        }
                    else:
                        return {'error': '업비트 매도 주문 실패'}
                
                return result
            
        except Exception as e:
            self.logger.error(f"암호화폐 매매 실행 실패: {e}")
            return {'error': str(e)}
    
    def _create_contract(self, strategy_name: str, symbol: str):
        """IBKR 계약 생성"""
        if strategy_name == 'US':
            return Stock(symbol, 'SMART', 'USD')
        elif strategy_name == 'JAPAN':
            return Stock(symbol, 'TSE', 'JPY')
        elif strategy_name == 'INDIA':
            return Stock(symbol, 'NSE', 'INR')
        else:
            return Stock(symbol, 'SMART', 'USD')
    
    def _get_allocation_for_strategy(self, strategy_name: str) -> float:
        """전략별 자산 배분 반환"""
        allocation_map = {
            'US': self.config.US_ALLOCATION,
            'JAPAN': self.config.JAPAN_ALLOCATION,
            'INDIA': self.config.INDIA_ALLOCATION,
            'CRYPTO': self.config.CRYPTO_ALLOCATION
        }
        return allocation_map.get(strategy_name, 0.1)
    
    def _get_currency_for_strategy(self, strategy_name: str) -> str:
        """전략별 통화 반환"""
        currency_map = {
            'US': 'USD',
            'JAPAN': 'JPY',
            'INDIA': 'INR',
            'CRYPTO': 'KRW'
        }
        return currency_map.get(strategy_name, 'USD')

# ============================================================================
# 🎯 전략 래퍼 (AI 확신도 체크 + 자동매매 통합)
# ============================================================================
class StrategyWrapper:
    """전략 실행을 위한 래퍼 (AI 확신도 체크 + 자동매매 통합)"""
    
    def __init__(self, strategy_instance, strategy_name: str, 
                 ai_checker: AITechnicalConfidenceChecker, 
                 performance_tracker: PerformanceTracker,
                 auto_trading_engine: AutoTradingEngine):
        self.strategy = strategy_instance
        self.strategy_name = strategy_name
        self.ai_checker = ai_checker
        self.performance_tracker = performance_tracker
        self.auto_trading_engine = auto_trading_engine
        
        self.logger = logging.getLogger(f'StrategyWrapper-{strategy_name}')
    
    async def execute_with_ai_and_trading(self) -> Dict[str, Any]:
        """AI 확신도 체크 + 실제 자동매매 실행"""
        try:
            self.logger.info(f"🎯 {self.strategy_name} 전략 실행 (AI + 자동매매)")
            
            # 1단계: 전략 실행
            if hasattr(self.strategy, 'run_strategy'):
                strategy_result = await self.strategy.run_strategy()
            elif hasattr(self.strategy, 'execute_legendary_strategy'):
                strategy_result = await self.strategy.execute_legendary_strategy()
            else:
                return {'error': f'{self.strategy_name} 전략 실행 메서드 없음'}
            
            if not strategy_result or 'signals' not in strategy_result:
                return {'message': f'{self.strategy_name} 전략 신호 없음'}
            
            # 2단계: 각 매매 신호에 대해 AI 확신도 체크 + 실제 매매
            enhanced_signals = []
            trade_results = []
            
            for signal in strategy_result['signals']:
                try:
                    symbol = signal.get('symbol', '')
                    market = self._get_market_for_strategy()
                    strategy_confidence = signal.get('confidence', 0.5)
                    
                    # 시장 데이터 수집
                    market_data = await self._collect_market_data(symbol, market)
                    
                    # AI 확신도 체크 (필요한 경우만)
                    ai_result = await self.ai_checker.check_technical_confidence(
                        symbol, market, signal, market_data
                    )
                    
                    # 최종 확신도 계산
                    final_confidence = strategy_confidence + ai_result.get('confidence_adjustment', 0)
                    final_confidence = max(0, min(1, final_confidence))  # 0-1 범위로 제한
                    
                    # 신호 업데이트
                    enhanced_signal = signal.copy()
                    enhanced_signal.update({
                        'original_confidence': strategy_confidence,
                        'ai_adjustment': ai_result.get('confidence_adjustment', 0),
                        'final_confidence': final_confidence,
                        'ai_reasoning': ai_result.get('reasoning', ''),
                        'tokens_used': ai_result.get('tokens_used', 0)
                    })
                    
                    enhanced_signals.append(enhanced_signal)
                    
                    # AI 체크 결과 기록
                    executed = final_confidence >= self.auto_trading_engine.MIN_CONFIDENCE_FOR_TRADE
                    self.performance_tracker.record_ai_confidence_check(
                        symbol, market, strategy_confidence, ai_result, executed
                    )
                    
                    # 3단계: 실제 자동매매 실행
                    if executed:
                        trade_result = await self.auto_trading_engine.execute_signal(
                            self.strategy_name, enhanced_signal
                        )
                        trade_results.append(trade_result)
                        
                        self.logger.info(
                            f"🚀 자동매매 실행: {symbol} {enhanced_signal.get('action')} "
                            f"확신도={final_confidence:.2f} "
                            f"결과={'성공' if trade_result.get('status') == 'success' else '실패'}"
                        )
                    else:
                        self.logger.info(
                            f"⏸️ 자동매매 건너뜀: {symbol} 확신도={final_confidence:.2f} "
                            f"(임계값 {self.auto_trading_engine.MIN_CONFIDENCE_FOR_TRADE:.2f} 미만)"
                        )
                    
                except Exception as e:
                    self.logger.error(f"신호 처리 실패 {signal.get('symbol', 'Unknown')}: {e}")
                    # 실패한 신호도 기록에 포함
                    enhanced_signals.append(signal)
            
            return {
                'strategy': self.strategy_name,
                'enhanced_signals': enhanced_signals,
                'trade_results': trade_results,
                'ai_usage': self.ai_checker.get_monthly_usage_summary(),
                'executed_trades': len([r for r in trade_results if r.get('status') == 'success']),
                'total_signals': len(enhanced_signals),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"{self.strategy_name} 전략 실행 실패: {e}")
            return {'error': str(e)}
    
    def _get_market_for_strategy(self) -> str:
        """전략에 따른 마켓 결정"""
        market_map = {
            'US': 'US',
            'JAPAN': 'JAPAN', 
            'INDIA': 'INDIA',
            'CRYPTO': 'CRYPTO'
        }
        return market_map.get(self.strategy_name, 'US')
    
    async def _collect_market_data(self, symbol: str, market: str) -> Dict[str, Any]:
        """시장 데이터 수집 (기술적 분석용)"""
        try:
            # 야후 파이낸스에서 데이터 수집
            if market == 'US':
                ticker = symbol
            elif market == 'JAPAN':
                ticker = f"{symbol}.T"
            elif market == 'INDIA':
                ticker = f"{symbol}.NS"
            elif market == 'CRYPTO':
                # 암호화폐는 업비트 API 사용
                return await self._collect_crypto_data(symbol)
            else:
                ticker = symbol
            
            stock = yf.Ticker(ticker)
            
            # 가격 데이터 (최근 30일)
            hist = stock.history(period="1mo")
            
            if len(hist) == 0:
                return {'symbol': symbol, 'current_price': 0, 'rsi': 50, 'volatility': 0}
            
            # 기술적 지표 계산
            current_price = hist['Close'].iloc[-1]
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
            
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)  # 연간 변동성
            
            # RSI 계산
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(hist) >= 14 else 50
            
            market_data = {
                'symbol': symbol,
                'current_price': float(current_price),
                'sma_20': float(sma_20) if not pd.isna(sma_20) else float(current_price),
                'rsi': float(rsi) if not pd.isna(rsi) else 50,
                'volatility': float(volatility) if not pd.isna(volatility) else 0
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 실패 {symbol}: {e}")
            return {'symbol': symbol, 'current_price': 0, 'rsi': 50, 'volatility': 0}
    
    async def _collect_crypto_data(self, symbol: str) -> Dict[str, Any]:
        """암호화폐 데이터 수집 (업비트)"""
        try:
            upbit_symbol = f"KRW-{symbol}" if not symbol.startswith('KRW-') else symbol
            
            # 현재가
            current_price = pyupbit.get_current_price(upbit_symbol)
            if not current_price:
                return {'symbol': symbol, 'current_price': 0, 'rsi': 50, 'volatility': 0}
            
            # OHLCV 데이터 (최근 30일)
            df = pyupbit.get_ohlcv(upbit_symbol, count=30)
            if df is None or len(df) == 0:
                return {'symbol': symbol, 'current_price': current_price, 'rsi': 50, 'volatility': 0}
            
            # 기술적 지표 계산
            sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price
            
            # 변동성 계산
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(365) if len(returns) > 0 else 0
            
            # RSI 계산
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(df) >= 14 else 50
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'sma_20': float(sma_20) if not pd.isna(sma_20) else current_price,
                'rsi': float(rsi) if not pd.isna(rsi) else 50,
                'volatility': float(volatility)
            }
            
        except Exception as e:
            self.logger.error(f"암호화폐 데이터 수집 실패 {symbol}: {e}")
            return {'symbol': symbol, 'current_price': 0, 'rsi': 50, 'volatility': 0}
# ============================================================================
# 🏆 퀸트프로젝트 통합 코어 시스템 (완전 자동매매)
# ============================================================================
class QuantProjectCore:
    """퀸트프로젝트 통합 코어 시스템 (완전 자동매매)"""
    
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
        self.ai_checker = AITechnicalConfidenceChecker(self.config)
        self.position_manager = UnifiedPositionManager(self.config, self.ibkr_manager)
        self.network_monitor = NetworkMonitor(self.config, self.ibkr_manager)
        self.notification_manager = NotificationManager(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        self.backup_manager = BackupManager(self.config)
        
        # 🚀 자동매매 엔진 추가
        self.auto_trading_engine = AutoTradingEngine(
            self.config, self.ibkr_manager, self.performance_tracker, self.notification_manager
        )
        
        # 전략 인스턴스와 래퍼 (자동매매 포함)
        self.strategies = {}
        self.strategy_wrappers = {}
        self._init_strategies()
        
        # 시스템 상태
        self.running = False
        self.start_time = None
    
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
        """전략 초기화 및 자동매매 래퍼 생성"""
        try:
            # 미국 전략
            if self.config.US_ENABLED and US_AVAILABLE:
                us_strategy = USStrategy()
                self.strategies['US'] = us_strategy
                self.strategy_wrappers['US'] = StrategyWrapper(
                    us_strategy, 'US', self.ai_checker, self.performance_tracker, self.auto_trading_engine
                )
                self.logger.info("✅ 미국 전략 초기화 완료 (자동매매 포함)")
            
            # 일본 전략
            if self.config.JAPAN_ENABLED and JAPAN_AVAILABLE:
                japan_strategy = JapanStrategy()
                self.strategies['JAPAN'] = japan_strategy
                self.strategy_wrappers['JAPAN'] = StrategyWrapper(
                    japan_strategy, 'JAPAN', self.ai_checker, self.performance_tracker, self.auto_trading_engine
                )
                self.logger.info("✅ 일본 전략 초기화 완료 (자동매매 포함)")
            
            # 인도 전략
            if self.config.INDIA_ENABLED and INDIA_AVAILABLE:
                india_strategy = IndiaStrategy()
                self.strategies['INDIA'] = india_strategy
                self.strategy_wrappers['INDIA'] = StrategyWrapper(
                    india_strategy, 'INDIA', self.ai_checker, self.performance_tracker, self.auto_trading_engine
                )
                self.logger.info("✅ 인도 전략 초기화 완료 (자동매매 포함)")
            
            # 암호화폐 전략
            if self.config.CRYPTO_ENABLED and CRYPTO_AVAILABLE:
                crypto_strategy = CryptoStrategy()
                self.strategies['CRYPTO'] = crypto_strategy
                self.strategy_wrappers['CRYPTO'] = StrategyWrapper(
                    crypto_strategy, 'CRYPTO', self.ai_checker, self.performance_tracker, self.auto_trading_engine
                )
                self.logger.info("✅ 암호화폐 전략 초기화 완료 (자동매매 포함)")
            
            if not self.strategies:
                self.logger.warning("⚠️ 활성화된 전략이 없습니다")
                
        except Exception as e:
            self.logger.error(f"전략 초기화 실패: {e}")
    
    async def start_system(self):
        """시스템 시작"""
        try:
            self.logger.info("🏆 퀸트프로젝트 완전 자동매매 시스템 시작! 💰")
            self.start_time = datetime.now()
            self.running = True
            
            # IBKR 연결
            if IBKR_AVAILABLE:
                await self.ibkr_manager.connect()
            
            # AI 체커 상태 확인
            ai_status = "✅" if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY else "❌"
            ai_usage = self.ai_checker.get_monthly_usage_summary()
            
            # 시작 알림
            await self.notification_manager.send_notification(
                f"🚀 퀸트프로젝트 완전 자동매매 시스템 시작! 💰\n"
                f"활성 전략: {', '.join(self.strategies.keys())}\n"
                f"IBKR 연결: {'✅' if self.ibkr_manager.connected else '❌'}\n"
                f"업비트 연결: {'✅' if self.auto_trading_engine.upbit else '❌'}\n"
                f"AI 기술적 분석: {ai_status}\n"
                f"AI 토큰 사용량: {ai_usage['current_usage']}/{ai_usage['limit']}\n"
                f"자동매매 임계값: {self.auto_trading_engine.MIN_CONFIDENCE_FOR_TRADE:.1f}\n"
                f"일일 최대 거래: {self.auto_trading_engine.MAX_DAILY_TRADES}회\n"
                f"포트폴리오: {self.config.TOTAL_PORTFOLIO_VALUE:,.0f}원",
                'success'
            )
            
            # 백그라운드 태스크 시작
            tasks = [
                asyncio.create_task(self._main_auto_trading_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self.network_monitor.start_monitoring()),
                asyncio.create_task(self._backup_loop()),
                asyncio.create_task(self._ai_usage_monitoring_loop()),
                asyncio.create_task(self._daily_reset_loop())
            ]
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"시스템 시작 실패: {e}")
            await self.emergency_shutdown(f"시스템 시작 실패: {e}")
    
    async def _main_auto_trading_loop(self):
        """메인 자동매매 루프"""
        while self.running:
            try:
                # 시스템 건강 상태 체크
                health_status = self.emergency_detector.check_system_health()
                
                if health_status['emergency_needed']:
                    await self.emergency_shutdown("시스템 건강 상태 위험")
                    break
                
                # 각 전략 실행 (요일별 + 시장 시간 체크)
                current_weekday = datetime.now().weekday()
                
                executed_any = False
                for strategy_name, strategy_wrapper in self.strategy_wrappers.items():
                    try:
                        if self._should_run_strategy(strategy_name, current_weekday):
                            self.logger.info(f"🎯 {strategy_name} 전략 자동매매 시작")
                            
                            result = await strategy_wrapper.execute_with_ai_and_trading()
                            
                            if 'error' in result:
                                error_critical = self.emergency_detector.record_error(
                                    f"{strategy_name}_error", result['error'], critical=True
                                )
                                if error_critical:
                                    await self.emergency_shutdown(f"{strategy_name} 전략 치명적 오류")
                                    break
                            else:
                                executed_any = True
                                await self._process_auto_trading_results(strategy_name, result)
                                
                    except Exception as e:
                        error_critical = self.emergency_detector.record_error(
                            f"{strategy_name}_error", str(e), critical=True
                        )
                        if error_critical:
                            await self.emergency_shutdown(f"{strategy_name} 전략 치명적 오류")
                            break
                
                # 포지션 업데이트
                await self.position_manager.update_all_positions()
                
                # 실행된 전략이 있으면 30분 후 재실행, 없으면 1시간 후
                wait_time = 1800 if executed_any else 3600
                self.logger.info(f"⏰ 다음 자동매매 실행까지 {wait_time//60}분 대기")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"메인 자동매매 루프 오류: {e}")
                await asyncio.sleep(60)
    
    async def _process_auto_trading_results(self, strategy_name: str, result: Dict):
        """자동매매 결과 처리"""
        try:
            trade_results = result.get('trade_results', [])
            executed_trades = result.get('executed_trades', 0)
            total_signals = result.get('total_signals', 0)
            
            # 성공한 거래가 있으면 상세 알림
            if executed_trades > 0:
                successful_trades = [r for r in trade_results if r.get('status') == 'success']
                
                trade_summary = []
                total_cost = 0
                
                for trade in successful_trades:
                    symbol = trade.get('symbol', 'Unknown')
                    action = trade.get('action', 'Unknown')
                    quantity = trade.get('quantity', 0)
                    price = trade.get('price', 0)
                    cost = trade.get('total_cost', trade.get('total_revenue', 0))
                    
                    trade_summary.append(f"{symbol} {action} {quantity} @ {price}")
                    total_cost += cost
                
                await self.notification_manager.send_notification(
                    f"🎯 {strategy_name} 자동매매 완료!\n"
                    f"실행된 거래: {executed_trades}/{total_signals}\n"
                    f"총 거래금액: {total_cost:,.0f}원\n"
                    f"거래 내역:\n" + "\n".join(trade_summary),
                    'success'
                )
            
            # AI 사용량 체크
            ai_usage = result.get('ai_usage', {})
            if ai_usage.get('usage_percentage', 0) >= 90:
                await self.notification_manager.send_notification(
                    f"⚠️ AI 토큰 사용량 경고\n"
                    f"현재 사용량: {ai_usage['current_usage']}/{ai_usage['limit']} "
                    f"({ai_usage['usage_percentage']:.1f}%)",
                    'warning'
                )
            
        except Exception as e:
            self.logger.error(f"자동매매 결과 처리 실패: {e}")
    
    async def _daily_reset_loop(self):
        """일일 리셋 루프"""
        while self.running:
            try:
                now = datetime.now()
                
                # 매일 자정에 일일 거래 카운트 리셋
                if now.hour == 0 and now.minute == 0:
                    self.auto_trading_engine.daily_trades = {}
                    
                    await self.notification_manager.send_notification(
                        "🔄 일일 거래 카운트 리셋 완료",
                        'info'
                    )
                    
                    await asyncio.sleep(60)  # 1분 대기 (중복 리셋 방지)
                
                await asyncio.sleep(60)  # 1분마다 체크
                
            except Exception as e:
                self.logger.error(f"일일 리셋 루프 오류: {e}")
                await asyncio.sleep(3600)
    
    def _should_run_strategy(self, strategy_name: str, weekday: int) -> bool:
        """전략 실행 여부 판단 (시장 시간 포함)"""
        # 기본 요일 체크
        weekday_check = False
        if strategy_name == 'US':
            weekday_check = weekday in [1, 3]  # 화목
        elif strategy_name == 'JAPAN':
            weekday_check = weekday in [1, 3]  # 화목
        elif strategy_name == 'INDIA':
            weekday_check = weekday == 2  # 수요일
        elif strategy_name == 'CRYPTO':
            weekday_check = weekday in [0, 4]  # 월금
        
        if not weekday_check:
            return False
        
        # 시장 시간 체크
        return self.auto_trading_engine._is_market_open(strategy_name)
    
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
                        f"한계: {self.config.MAX_PORTFOLIO_RISK * 100:.1f}%\n"
                        f"응급 매도를 고려하세요!",
                        'emergency'
                    )
                
                # 일일 거래 현황 체크
                today = datetime.now().date().isoformat()
                daily_trades = self.auto_trading_engine.daily_trades.get(today, 0)
                
                if daily_trades >= self.auto_trading_engine.MAX_DAILY_TRADES * 0.8:
                    await self.notification_manager.send_notification(
                        f"⚠️ 일일 거래 한도 임박\n"
                        f"현재: {daily_trades}/{self.auto_trading_engine.MAX_DAILY_TRADES}회",
                        'warning'
                    )
                
                # 주기적 상태 보고 (6시간마다)
                if datetime.now().hour % 6 == 0 and datetime.now().minute < 5:
                    await self._send_status_report(portfolio_summary)
                
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(60)
    
    async def _ai_usage_monitoring_loop(self):
        """AI 사용량 모니터링 루프"""
        while self.running:
            try:
                # 매시간 AI 사용량 체크
                ai_usage = self.ai_checker.get_monthly_usage_summary()
                
                # 월 한도 90% 이상 사용시 경고
                if ai_usage['usage_percentage'] >= 90:
                    await self.notification_manager.send_notification(
                        f"🚨 AI 토큰 사용량 위험!\n"
                        f"사용량: {ai_usage['current_usage']}/{ai_usage['limit']} "
                        f"({ai_usage['usage_percentage']:.1f}%)\n"
                        f"남은 토큰: {ai_usage['remaining']}",
                        'warning'
                    )
                
                # 매월 1일 사용량 리셋 체크
                if datetime.now().day == 1 and datetime.now().hour == 0:
                    if datetime.now().month != self.ai_checker.current_month:
                        self.ai_checker.current_month = datetime.now().month
                        self.ai_checker.monthly_token_usage = 0
                        self.ai_checker._save_token_usage()
                        
                        await self.notification_manager.send_notification(
                            "🔄 AI 토큰 사용량 월별 리셋 완료",
                            'info'
                        )
                
                await asyncio.sleep(3600)  # 1시간마다
                
            except Exception as e:
                self.logger.error(f"AI 사용량 모니터링 오류: {e}")
                await asyncio.sleep(3600)
    
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
        """상태 보고서 전송 (자동매매 정보 포함)"""
        try:
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            # 최근 AI 성과 조회
            performance_data = self.performance_tracker.get_performance_summary(7)  # 최근 7일
            ai_performance = performance_data.get('ai_performance', {})
            ai_usage = self.ai_checker.get_monthly_usage_summary()
            
            # 일일 거래 현황
            today = datetime.now().date().isoformat()
            daily_trades = self.auto_trading_engine.daily_trades.get(today, 0)
            
            report = (
                f"📊 퀸트프로젝트 완전 자동매매 상태 보고\n\n"
                f"🕐 가동시간: {uptime}\n"
                f"💼 총 포지션: {portfolio_summary['total_positions']}개\n"
                f"💰 미실현 손익: {portfolio_summary['total_unrealized_pnl']:+,.0f}원\n"
                f"📈 수익 포지션: {portfolio_summary['profitable_positions']}개\n"
                f"📉 손실 포지션: {portfolio_summary['losing_positions']}개\n"
                f"🤖 오늘 자동거래: {daily_trades}/{self.auto_trading_engine.MAX_DAILY_TRADES}회\n\n"
                f"전략별 현황:\n"
            )
            
            for strategy, data in portfolio_summary['by_strategy'].items():
                report += f"  {strategy}: {data['count']}개 ({data['pnl']:+,.0f}원)\n"
            
            # AI 사용량 및 성과 추가
            if ai_performance:
                report += f"\n🤖 AI 기술적 분석 현황 (최근 7일):\n"
                report += f"  확신도 체크: {ai_performance.get('total_checks', 0)}회\n"
                report += f"  평균 조정값: {ai_performance.get('avg_adjustment', 0):+.3f}\n"
                report += f"  실행률: {ai_performance.get('execution_rate', 0):.1f}%\n"
                report += f"  사용 토큰: {ai_performance.get('total_tokens', 0)}개\n"
            
            report += f"\n💾 월별 AI 토큰 사용량:\n"
            report += f"  현재: {ai_usage['current_usage']}/{ai_usage['limit']} ({ai_usage['usage_percentage']:.1f}%)\n"
            report += f"  남은 토큰: {ai_usage['remaining']}개\n\n"
            report += f"💡 자동매매 설정:\n"
            report += f"  최소 확신도: {self.auto_trading_engine.MIN_CONFIDENCE_FOR_TRADE:.1f}\n"
            report += f"  최대 포지션: {self.auto_trading_engine.MAX_POSITION_SIZE:.1%}\n"
            report += f"  손절 설정: {self.auto_trading_engine.STOP_LOSS_PCT:.1%}"
            
            await self.notification_manager.send_notification(report, 'info')
            
        except Exception as e:
            self.logger.error(f"상태 보고서 전송 실패: {e}")
    
    async def emergency_shutdown(self, reason: str):
        """응급 종료"""
        try:
            self.logger.critical(f"🚨 응급 종료: {reason}")
            
            # 응급 알림
            await self.notification_manager.send_notification(
                f"🚨 자동매매 시스템 응급 종료\n"
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
            self.logger.info("🛑 자동매매 시스템 정상 종료 시작")
            
            # 종료 알림
            ai_usage = self.ai_checker.get_monthly_usage_summary()
            today = datetime.now().date().isoformat()
            daily_trades = self.auto_trading_engine.daily_trades.get(today, 0)
            
            await self.notification_manager.send_notification(
                f"🛑 자동매매 시스템 정상 종료\n"
                f"가동시간: {datetime.now() - self.start_time if self.start_time else '알수없음'}\n"
                f"오늘 거래: {daily_trades}회\n"
                f"AI 토큰 사용: {ai_usage['current_usage']}/{ai_usage['limit']}",
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
            self.logger.info("✅ 자동매매 시스템 정상 종료 완료")
            
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
    ai_usage = core.ai_checker.get_monthly_usage_summary()
    
    return {
        'strategies': list(core.strategies.keys()),
        'ibkr_connected': core.ibkr_manager.connected,
        'upbit_connected': core.auto_trading_engine.upbit is not None,
        'ai_enabled': core.config.AI_TECHNICAL_CHECK_ENABLED,
        'ai_usage': ai_usage,
        'auto_trading_enabled': True,
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

async def check_ai_confidence(symbol: str, market: str = 'US', strategy_confidence: float = 0.5):
    """단일 종목 AI 확신도 체크"""
    core = QuantProjectCore()
    
    # 더미 신호와 시장 데이터로 테스트
    dummy_signal = {'symbol': symbol, 'confidence': strategy_confidence, 'action': 'BUY'}
    dummy_market_data = {'symbol': symbol, 'current_price': 100, 'rsi': 50, 'volatility': 0.2}
    
    result = await core.ai_checker.check_technical_confidence(
        symbol, market, dummy_signal, dummy_market_data
    )
    
    return result

async def get_ai_usage_summary():
    """AI 사용량 요약 조회"""
    core = QuantProjectCore()
    return core.ai_checker.get_monthly_usage_summary()

async def manual_trade_test(strategy_name: str, symbol: str, action: str):
    """수동 거래 테스트"""
    core = QuantProjectCore()
    await core.ibkr_manager.connect()
    
    # 테스트 신호 생성
    test_signal = {
        'symbol': symbol,
        'action': action,
        'final_confidence': 0.8,
        'position_size_pct': 2  # 2%만 테스트
    }
    
    result = await core.auto_trading_engine.execute_signal(strategy_name, test_signal)
    return result

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
        print("🏆 퀸트프로젝트 완전 자동매매 시스템 v1.3.0 💰")
        print("🏆" + "="*70)
        print("✨ 4대 전략 통합 관리")
        print("✨ IBKR 자동 환전")
        print("✨ AI 기술적 분석 확신도 체크 (최적화)")
        print("✨ 완전 자동매매 실행")
        print("✨ 월 토큰 사용량 제한 관리")
        print("✨ 네트워크 모니터링")
        print("✨ 응급 오류 감지")
        print("✨ 통합 포지션 관리")
        print("✨ 실시간 알림")
        print("✨ 자동 백업")
        print("🏆" + "="*70)
        
        # AI 상태 확인
        if OPENAI_AVAILABLE and core.config.OPENAI_API_KEY:
            ai_usage = core.ai_checker.get_monthly_usage_summary()
            print("🤖 AI 기술적 분석: ✅")
            print(f"🤖 모델: {core.config.OPENAI_MODEL}")
            print(f"🤖 월 토큰 사용량: {ai_usage['current_usage']}/{ai_usage['limit']} ({ai_usage['usage_percentage']:.1f}%)")
            print(f"🤖 확신도 체크 범위: {core.config.AI_CONFIDENCE_THRESHOLD_LOW:.1f}-{core.config.AI_CONFIDENCE_THRESHOLD_HIGH:.1f}")
        else:
            print("🤖 AI 기술적 분석: ❌ (API 키 확인 필요)")
        
        # 자동매매 설정 표시
        print("💰" + "="*70)
        print("💰 자동매매 설정:")
        print(f"💰 최소 확신도: {core.auto_trading_engine.MIN_CONFIDENCE_FOR_TRADE:.1f}")
        print(f"💰 일일 최대 거래: {core.auto_trading_engine.MAX_DAILY_TRADES}회")
        print(f"💰 최대 포지션 크기: {core.auto_trading_engine.MAX_POSITION_SIZE:.1%}")
        print(f"💰 자동 손절: {core.auto_trading_engine.STOP_LOSS_PCT:.1%}")
        print(f"💰 IBKR 연결: {'✅' if IBKR_AVAILABLE else '❌'}")
        print(f"💰 업비트 연결: {'✅' if UPBIT_AVAILABLE else '❌'}")
        print(f"💰 데모 모드: {'✅' if core.config.UPBIT_DEMO_MODE else '❌'}")
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
        print("\n👋 퀸트프로젝트 자동매매 시스템 종료")
        sys.exit(0)
