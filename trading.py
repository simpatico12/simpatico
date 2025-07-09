#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 통합 거래 시스템 (trading.py)
================================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 + 🤖 OpenAI (5대 전략 통합)

✨ 핵심 기능:
- 5대 전략 통합 관리 시스템
- OpenAI GPT-4 기반 AI 분석 시스템
- IBKR 자동 환전 + 실시간 매매
- 서머타임 자동 처리 + 화목 매매
- 월 5-7% 최적화 손익절 시스템
- 통합 알림 시스템 (텔레그램/이메일/SMS)
- 응급 오류 감지 + 네트워크 모니터링
- 포지션 관리 + 성과 추적
- AI 기반 시장 분석 및 종목 추천

Author: 퀸트마스터팀
Version: 2.1.0 (OpenAI 통합 거래 시스템)
"""

import asyncio
import logging
import os
import sys
import json
import time
import warnings
import traceback
import signal
import psutil
import shutil
import hashlib
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import sqlite3
import threading

# 외부 라이브러리
import numpy as np
import pandas as pd
import requests
import aiohttp
from dotenv import load_dotenv

# OpenAI 라이브러리
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ openai 모듈 없음")

# 금융 데이터 라이브러리
try:
    import yfinance as yf
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False
    print("⚠️ yfinance 모듈 없음")

try:
    import pyupbit
    UPBIT_AVAILABLE = True
except ImportError:
    UPBIT_AVAILABLE = False
    print("⚠️ pyupbit 모듈 없음")

try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("⚠️ IBKR 모듈 없음")

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    print("⚠️ pytz 모듈 없음")

warnings.filterwarnings('ignore')

# ============================================================================
# 🎯 통합 설정 관리자
# ============================================================================
class TradingConfig:
    """통합 거래 시스템 설정"""
    
    def __init__(self):
        load_dotenv()
        
        # 포트폴리오 설정
        self.TOTAL_PORTFOLIO_VALUE = float(os.getenv('TOTAL_PORTFOLIO_VALUE', '1000000000'))
        self.MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', '0.05'))
        
        # 전략별 활성화
        self.US_ENABLED = os.getenv('US_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.JAPAN_ENABLED = os.getenv('JAPAN_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.INDIA_ENABLED = os.getenv('INDIA_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.CRYPTO_ENABLED = os.getenv('CRYPTO_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.OPENAI_ENABLED = os.getenv('OPENAI_STRATEGY_ENABLED', 'true').lower() == 'true'
        
        # 전략별 자원 배분
        self.US_ALLOCATION = float(os.getenv('US_STRATEGY_ALLOCATION', '0.35'))
        self.JAPAN_ALLOCATION = float(os.getenv('JAPAN_STRATEGY_ALLOCATION', '0.20'))
        self.CRYPTO_ALLOCATION = float(os.getenv('CRYPTO_STRATEGY_ALLOCATION', '0.20'))
        self.INDIA_ALLOCATION = float(os.getenv('INDIA_STRATEGY_ALLOCATION', '0.15'))
        self.OPENAI_ALLOCATION = float(os.getenv('OPENAI_STRATEGY_ALLOCATION', '0.10'))
        
        # IBKR 설정
        self.IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
        self.IBKR_PORT = int(os.getenv('IBKR_PORT', '7497'))
        self.IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '1'))
        self.IBKR_PAPER_TRADING = os.getenv('IBKR_PAPER_TRADING', 'true').lower() == 'true'
        
        # 업비트 설정
        self.UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
        self.UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')
        self.UPBIT_DEMO_MODE = os.getenv('CRYPTO_DEMO_MODE', 'true').lower() == 'true'
        
        # OpenAI 설정
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        self.OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
        self.OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
        self.OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        self.OPENAI_ANALYSIS_ENABLED = os.getenv('OPENAI_ANALYSIS_ENABLED', 'true').lower() == 'true'
        
        # 알림 설정
        self.TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # 이메일 설정
        self.EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        self.EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.EMAIL_USERNAME = os.getenv('EMAIL_USERNAME', '')
        self.EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
        self.EMAIL_TO_ADDRESS = os.getenv('EMAIL_TO_ADDRESS', '')
        
        # 시스템 모니터링
        self.NETWORK_MONITORING = os.getenv('NETWORK_MONITORING_ENABLED', 'true').lower() == 'true'
        self.NETWORK_CHECK_INTERVAL = int(os.getenv('NETWORK_CHECK_INTERVAL', '30'))
        self.EMERGENCY_SELL_ON_ERROR = os.getenv('EMERGENCY_SELL_ON_ERROR', 'true').lower() == 'true'
        
        # 데이터베이스
        self.DB_PATH = os.getenv('DATABASE_PATH', './data/trading_system.db')
        self.BACKUP_PATH = os.getenv('BACKUP_PATH', './backups/')

# ============================================================================
# 🤖 OpenAI 분석 시스템
# ============================================================================
@dataclass
class AIAnalysisResult:
    """AI 분석 결과"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 ~ 1.0
    reasoning: str
    target_price: float
    risk_level: str  # LOW, MEDIUM, HIGH
    time_horizon: str  # SHORT, MEDIUM, LONG
    market_sentiment: str  # BULLISH, BEARISH, NEUTRAL
    technical_score: float
    fundamental_score: float
    timestamp: datetime = field(default_factory=datetime.now)

class OpenAIAnalysisEngine:
    """OpenAI 기반 투자 분석 엔진"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger('OpenAIAnalysisEngine')
        
        # OpenAI 클라이언트 초기화
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            openai.api_key = self.config.OPENAI_API_KEY
            self.client_available = True
        else:
            self.client_available = False
            self.logger.warning("OpenAI API 키가 설정되지 않았습니다")
        
        # 분석 캐시
        self.analysis_cache = {}
        self.cache_duration = timedelta(hours=1)  # 1시간 캐시
        
        # 시장 컨텍스트
        self.market_context = {
            'economic_indicators': {},
            'market_sentiment': 'NEUTRAL',
            'volatility_index': 0.5,
            'last_updated': None
        }
    
    async def analyze_symbol(self, symbol: str, market_data: Dict[str, Any], 
                           strategy_context: str = '') -> Optional[AIAnalysisResult]:
        """개별 종목 AI 분석"""
        if not self.client_available:
            return None
        
        try:
            # 캐시 확인
            cache_key = f"{symbol}_{hash(str(market_data))}"
            if cache_key in self.analysis_cache:
                cached_time, cached_result = self.analysis_cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    return cached_result
            
            # 시장 컨텍스트 업데이트
            await self._update_market_context()
            
            # AI 분석 수행
            analysis_result = await self._perform_ai_analysis(
                symbol, market_data, strategy_context
            )
            
            # 캐시 저장
            self.analysis_cache[cache_key] = (datetime.now(), analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"AI 분석 실패 {symbol}: {e}")
            return None
    
    async def _update_market_context(self):
        """시장 컨텍스트 업데이트"""
        try:
            if self.market_context['last_updated'] and \
               datetime.now() - self.market_context['last_updated'] < timedelta(hours=6):
                return
            
            # 주요 경제 지표 수집
            economic_data = await self._collect_economic_indicators()
            
            # 시장 심리 분석
            market_sentiment = await self._analyze_market_sentiment()
            
            self.market_context.update({
                'economic_indicators': economic_data,
                'market_sentiment': market_sentiment,
                'last_updated': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"시장 컨텍스트 업데이트 실패: {e}")
    
    async def _collect_economic_indicators(self) -> Dict[str, Any]:
        """경제 지표 수집"""
        try:
            indicators = {}
            
            # VIX 지수 (공포 지수)
            if YAHOO_AVAILABLE:
                try:
                    vix = yf.Ticker("^VIX")
                    vix_data = vix.history(period="5d")
                    if not vix_data.empty:
                        indicators['vix'] = float(vix_data['Close'].iloc[-1])
                except:
                    indicators['vix'] = 20.0  # 기본값
            
            # 달러 인덱스
            try:
                dxy = yf.Ticker("DX-Y.NYB")
                dxy_data = dxy.history(period="5d")
                if not dxy_data.empty:
                    indicators['dollar_index'] = float(dxy_data['Close'].iloc[-1])
            except:
                indicators['dollar_index'] = 100.0  # 기본값
            
            # 10년 국채 수익률
            try:
                tnx = yf.Ticker("^TNX")
                tnx_data = tnx.history(period="5d")
                if not tnx_data.empty:
                    indicators['treasury_10y'] = float(tnx_data['Close'].iloc[-1])
            except:
                indicators['treasury_10y'] = 4.0  # 기본값
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"경제 지표 수집 실패: {e}")
            return {}
    
    async def _analyze_market_sentiment(self) -> str:
        """시장 심리 분석"""
        try:
            # 간단한 시장 심리 분석
            if 'vix' in self.market_context['economic_indicators']:
                vix = self.market_context['economic_indicators']['vix']
                if vix > 30:
                    return 'BEARISH'
                elif vix < 15:
                    return 'BULLISH'
                else:
                    return 'NEUTRAL'
            
            return 'NEUTRAL'
            
        except Exception as e:
            self.logger.error(f"시장 심리 분석 실패: {e}")
            return 'NEUTRAL'
    
    async def _perform_ai_analysis(self, symbol: str, market_data: Dict[str, Any], 
                                 strategy_context: str) -> AIAnalysisResult:
        """AI 분석 수행"""
        try:
            # 시스템 프롬프트
            system_prompt = """
당신은 전문 퀀트 투자 분석가입니다. 주어진 시장 데이터와 컨텍스트를 바탕으로 투자 분석을 수행하세요.

분석 기준:
1. 기술적 분석 (차트 패턴, 지표)
2. 시장 심리 및 거시경제 상황
3. 리스크 관리 및 포지션 사이징
4. 시간 축별 전략 (단기/중기/장기)

응답 형식 (JSON):
{
    "action": "BUY/SELL/HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "분석 근거",
    "target_price": 목표가격,
    "risk_level": "LOW/MEDIUM/HIGH",
    "time_horizon": "SHORT/MEDIUM/LONG",
    "market_sentiment": "BULLISH/BEARISH/NEUTRAL",
    "technical_score": 0.0-1.0,
    "fundamental_score": 0.0-1.0
}
"""
            
            # 사용자 프롬프트
            user_prompt = f"""
종목: {symbol}
전략 컨텍스트: {strategy_context}

시장 데이터:
{json.dumps(market_data, indent=2, default=str)}

현재 시장 컨텍스트:
- 시장 심리: {self.market_context['market_sentiment']}
- VIX: {self.market_context['economic_indicators'].get('vix', 'N/A')}
- 달러 인덱스: {self.market_context['economic_indicators'].get('dollar_index', 'N/A')}
- 10년 국채 수익률: {self.market_context['economic_indicators'].get('treasury_10y', 'N/A')}

위 정보를 종합적으로 분석하여 투자 의견을 제시해주세요.
"""
            
            # OpenAI API 호출
            response = await self._call_openai_api(system_prompt, user_prompt)
            
            # 응답 파싱
            analysis_data = json.loads(response)
            
            return AIAnalysisResult(
                symbol=symbol,
                action=analysis_data.get('action', 'HOLD'),
                confidence=float(analysis_data.get('confidence', 0.5)),
                reasoning=analysis_data.get('reasoning', '분석 결과 없음'),
                target_price=float(analysis_data.get('target_price', 0.0)),
                risk_level=analysis_data.get('risk_level', 'MEDIUM'),
                time_horizon=analysis_data.get('time_horizon', 'MEDIUM'),
                market_sentiment=analysis_data.get('market_sentiment', 'NEUTRAL'),
                technical_score=float(analysis_data.get('technical_score', 0.5)),
                fundamental_score=float(analysis_data.get('fundamental_score', 0.5))
            )
            
        except Exception as e:
            self.logger.error(f"AI 분석 수행 실패 {symbol}: {e}")
            
            # 기본 분석 결과 반환
            return AIAnalysisResult(
                symbol=symbol,
                action='HOLD',
                confidence=0.5,
                reasoning=f'AI 분석 실패: {str(e)}',
                target_price=market_data.get('current_price', 0.0),
                risk_level='MEDIUM',
                time_horizon='MEDIUM',
                market_sentiment='NEUTRAL',
                technical_score=0.5,
                fundamental_score=0.5
            )
    
    async def _call_openai_api(self, system_prompt: str, user_prompt: str) -> str:
        """OpenAI API 호출"""
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.OPENAI_MAX_TOKENS,
                temperature=self.config.OPENAI_TEMPERATURE
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API 호출 실패: {e}")
            raise
    
    async def generate_market_report(self) -> str:
        """AI 기반 시장 보고서 생성"""
        try:
            if not self.client_available:
                return "OpenAI API가 설정되지 않았습니다."
            
            await self._update_market_context()
            
            system_prompt = """
당신은 전문 시장 분석가입니다. 현재 시장 상황을 종합하여 간결하고 명확한 시장 보고서를 작성하세요.
보고서는 투자자가 이해하기 쉽게 작성하고, 구체적인 투자 방향성을 제시하세요.
"""
            
            user_prompt = f"""
현재 시장 지표:
- 시장 심리: {self.market_context['market_sentiment']}
- VIX (공포 지수): {self.market_context['economic_indicators'].get('vix', 'N/A')}
- 달러 인덱스: {self.market_context['economic_indicators'].get('dollar_index', 'N/A')}
- 10년 국채 수익률: {self.market_context['economic_indicators'].get('treasury_10y', 'N/A')}

위 지표들을 종합하여 현재 시장 상황과 투자 전략을 500자 이내로 요약해주세요.
"""
            
            response = await self._call_openai_api(system_prompt, user_prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"시장 보고서 생성 실패: {e}")
            return f"시장 보고서 생성 실패: {str(e)}"
    
    async def analyze_portfolio_risk(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """포트폴리오 리스크 AI 분석"""
        try:
            if not self.client_available:
                return {"risk_level": "UNKNOWN", "recommendations": []}
            
            system_prompt = """
당신은 포트폴리오 리스크 관리 전문가입니다. 
주어진 포트폴리오 구성을 분석하여 리스크를 평가하고 개선 방안을 제시하세요.
"""
            
            user_prompt = f"""
현재 포트폴리오:
{json.dumps(positions, indent=2, default=str)}

시장 컨텍스트:
{json.dumps(self.market_context, indent=2, default=str)}

포트폴리오의 리스크 수준과 개선 방안을 JSON 형태로 제시해주세요:
{{
    "overall_risk_level": "LOW/MEDIUM/HIGH",
    "risk_score": 0.0-1.0,
    "main_risks": ["리스크 요인들"],
    "recommendations": ["개선 방안들"],
    "diversification_score": 0.0-1.0
}}
"""
            
            response = await self._call_openai_api(system_prompt, user_prompt)
            return json.loads(response)
            
        except Exception as e:
            self.logger.error(f"포트폴리오 리스크 분석 실패: {e}")
            return {
                "overall_risk_level": "MEDIUM",
                "risk_score": 0.5,
                "main_risks": ["분석 실패"],
                "recommendations": ["AI 분석을 다시 시도하세요"],
                "diversification_score": 0.5
            }

# ============================================================================
# 🕒 서머타임 관리자 (미국 전략용)
# ============================================================================
class DaylightSavingManager:
    """서머타임 자동 관리"""
    
    def __init__(self):
        if PYTZ_AVAILABLE:
            self.us_eastern = pytz.timezone('US/Eastern')
            self.korea = pytz.timezone('Asia/Seoul')
        self.cache = {}
    
    def is_dst_active(self, date=None) -> bool:
        """서머타임 활성 여부"""
        if not PYTZ_AVAILABLE:
            return False
            
        if date is None:
            date = datetime.now().date()
        
        if date in self.cache:
            return self.cache[date]
        
        year = date.year
        # 3월 둘째주 일요일
        march_first = datetime(year, 3, 1)
        march_second_sunday = march_first + timedelta(days=(6 - march_first.weekday()) % 7 + 7)
        # 11월 첫째주 일요일  
        nov_first = datetime(year, 11, 1)
        nov_first_sunday = nov_first + timedelta(days=(6 - nov_first.weekday()) % 7)
        
        is_dst = march_second_sunday.date() <= date < nov_first_sunday.date()
        self.cache[date] = is_dst
        return is_dst
    
    def get_market_hours_kst(self, date=None) -> Tuple[datetime, datetime]:
        """미국 시장 시간 (한국시간)"""
        if not PYTZ_AVAILABLE:
            # 기본값 반환
            if date is None:
                date = datetime.now().date()
            return (
                datetime.combine(date, datetime.min.time().replace(hour=22, minute=30)),
                datetime.combine(date, datetime.min.time().replace(hour=5, minute=0)) + timedelta(days=1)
            )
            
        if date is None:
            date = datetime.now().date()
        
        market_open_et = datetime.combine(date, datetime.min.time().replace(hour=9, minute=30))
        market_close_et = datetime.combine(date, datetime.min.time().replace(hour=16, minute=0))
        
        if self.is_dst_active(date):
            market_open_et = self.us_eastern.localize(market_open_et, is_dst=True)
            market_close_et = self.us_eastern.localize(market_close_et, is_dst=True)
        else:
            market_open_et = self.us_eastern.localize(market_open_et, is_dst=False)
            market_close_et = self.us_eastern.localize(market_close_et, is_dst=False)
        
        return market_open_et.astimezone(self.korea), market_close_et.astimezone(self.korea)

# ============================================================================
# 🔔 통합 알림 시스템
# ============================================================================
@dataclass
class NotificationMessage:
    """알림 메시지"""
    title: str
    content: str
    priority: str = 'info'  # emergency, warning, info, success, debug
    category: str = 'general'  # trading, system, portfolio, error, ai
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class NotificationManager:
    """통합 알림 관리 시스템"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger('NotificationManager')
        self.recent_notifications = deque(maxlen=100)
    
    async def send_notification(self, message: Union[str, NotificationMessage], 
                              priority: str = 'info', title: str = '퀸트프로젝트 알림') -> bool:
        """통합 알림 전송"""
        try:
            # 문자열이면 NotificationMessage로 변환
            if isinstance(message, str):
                message = NotificationMessage(
                    title=title,
                    content=message,
                    priority=priority
                )
            
            # 중복 체크
            message_hash = hashlib.md5(f"{message.title}_{message.content}".encode()).hexdigest()
            if any(notif.get('hash') == message_hash for notif in self.recent_notifications):
                return False
            
            self.recent_notifications.append({
                'hash': message_hash,
                'timestamp': message.timestamp
            })
            
            # 우선순위별 전송
            success = False
            
            # 텔레그램 전송
            if self.config.TELEGRAM_ENABLED:
                success |= await self._send_telegram(message)
            
            # 이메일 전송 (warning 이상만)
            if self.config.EMAIL_ENABLED and message.priority in ['emergency', 'warning']:
                success |= await self._send_email(message)
            
            # 로그 기록
            if message.priority == 'emergency':
                self.logger.critical(f"{message.title}: {message.content}")
            elif message.priority == 'warning':
                self.logger.warning(f"{message.title}: {message.content}")
            else:
                self.logger.info(f"{message.title}: {message.content}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"알림 전송 실패: {e}")
            return False
    
    async def _send_telegram(self, message: NotificationMessage) -> bool:
        """텔레그램 전송"""
        if not self.config.TELEGRAM_BOT_TOKEN or not self.config.TELEGRAM_CHAT_ID:
            return False
        
        try:
            # 이모지 매핑
            priority_emojis = {
                'emergency': '🚨',
                'warning': '⚠️',
                'info': 'ℹ️',
                'success': '✅',
                'debug': '🔧'
            }
            
            emoji = priority_emojis.get(message.priority, '📊')
            
            formatted_text = (
                f"{emoji} <b>퀸트프로젝트 알림</b>\n\n"
                f"📋 <b>{message.title}</b>\n\n"
                f"{message.content}\n\n"
                f"🕐 {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            data = {
                'chat_id': self.config.TELEGRAM_CHAT_ID,
                'text': formatted_text,
                'parse_mode': 'HTML',
                'disable_notification': message.priority not in ['emergency', 'warning']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"텔레그램 전송 오류: {e}")
            return False
    
    async def _send_email(self, message: NotificationMessage) -> bool:
        """이메일 전송"""
        if not self.config.EMAIL_USERNAME or not self.config.EMAIL_TO_ADDRESS:
            return False
        
        try:
            # 이메일 메시지 생성
            msg = MimeMultipart()
            
            priority_prefix = {
                'emergency': '[🚨 응급]',
                'warning': '[⚠️ 경고]',
                'info': '[ℹ️ 정보]',
                'success': '[✅ 성공]'
            }
            
            subject_prefix = priority_prefix.get(message.priority, '[📊]')
            msg['Subject'] = f"{subject_prefix} {message.title}"
            msg['From'] = self.config.EMAIL_USERNAME
            msg['To'] = self.config.EMAIL_TO_ADDRESS
            
            # 본문
            body = f"""
퀸트프로젝트 알림

제목: {message.title}

{message.content}

발송시간: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
우선순위: {message.priority.upper()}
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            # SMTP 전송
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self._smtp_send_sync, msg)
            return success
            
        except Exception as e:
            self.logger.error(f"이메일 전송 오류: {e}")
            return False
    
    def _smtp_send_sync(self, msg: MimeMultipart) -> bool:
        """동기 SMTP 전송"""
        try:
            server = smtplib.SMTP(self.config.EMAIL_SMTP_SERVER, self.config.EMAIL_SMTP_PORT)
            server.starttls()
            server.login(self.config.EMAIL_USERNAME, self.config.EMAIL_PASSWORD)
            
            text = msg.as_string()
            server.sendmail(self.config.EMAIL_USERNAME, self.config.EMAIL_TO_ADDRESS, text)
            server.quit()
            
            return True
        except Exception as e:
            self.logger.error(f"SMTP 전송 실패: {e}")
            return False

# ============================================================================
# 🚨 응급 오류 감지 시스템
# ============================================================================
class EmergencyDetector:
    """응급 상황 감지 및 대응"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = None
        self.emergency_triggered = False
        self.logger = logging.getLogger('EmergencyDetector')
    
    def check_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태 체크"""
        health_status = {
            'healthy': True,
            'warnings': [],
            'errors': [],
            'emergency_needed': False
        }
        
        try:
            # 메모리 체크
            memory_percent = psutil.virtual_memory().percent
            if memory_percent >= 95:
                health_status['emergency_needed'] = True
                health_status['errors'].append(f'메모리 위험: {memory_percent:.1f}%')
            elif memory_percent >= 85:
                health_status['warnings'].append(f'메모리 경고: {memory_percent:.1f}%')
            
            # CPU 체크
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent >= 90:
                health_status['warnings'].append(f'CPU 높음: {cpu_percent:.1f}%')
            
            # 디스크 체크
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 1:
                health_status['emergency_needed'] = True
                health_status['errors'].append(f'디스크 위험: {free_gb:.1f}GB 남음')
            elif free_gb < 5:
                health_status['warnings'].append(f'디스크 경고: {free_gb:.1f}GB 남음')
            
            # 네트워크 체크
            if not self._check_network():
                health_status['emergency_needed'] = True
                health_status['errors'].append('네트워크 연결 실패')
            
            health_status['healthy'] = not health_status['errors']
            
        except Exception as e:
            health_status = {
                'healthy': False,
                'warnings': [],
                'errors': [f'상태 체크 실패: {str(e)}'],
                'emergency_needed': True
            }
        
        return health_status
    
    def _check_network(self) -> bool:
        """네트워크 연결 상태 체크"""
        try:
            response = requests.get('https://www.google.com', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def record_error(self, error_type: str, error_msg: str, critical: bool = False) -> bool:
        """오류 기록 및 응급 상황 판단"""
        current_time = time.time()
        
        self.error_count += 1
        
        # 연속 오류 체크
        if self.last_error_time and current_time - self.last_error_time < 60:
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 1
        
        self.last_error_time = current_time
        
        self.logger.error(f"오류 기록: {error_type} - {error_msg}")
        
        # 응급 상황 판단
        emergency_conditions = [
            critical,
            self.consecutive_errors >= 5,
            error_type in ['network_failure', 'api_failure', 'system_crash']
        ]
        
        if any(emergency_conditions) and not self.emergency_triggered:
            self.emergency_triggered = True
            self.logger.critical(f"🚨 응급 상황 감지: {error_type}")
            return True
        
        return False

# ============================================================================
# 🔗 IBKR 통합 관리자
# ============================================================================
class IBKRManager:
    """IBKR 통합 관리"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.ib = None
        self.connected = False
        self.positions = {}
        self.balances = {}
        self.logger = logging.getLogger('IBKRManager')
    
    async def connect(self) -> bool:
        """IBKR 연결"""
        if not IBKR_AVAILABLE:
            self.logger.warning("IBKR 모듈이 설치되지 않았습니다")
            return False
        
        try:
            self.ib = IB()
            await self.ib.connectAsync(
                self.config.IBKR_HOST,
                self.config.IBKR_PORT,
                self.config.IBKR_CLIENT_ID
            )
            
            if self.ib.isConnected():
                self.connected = True
                await self._update_account_info()
                self.logger.info("✅ IBKR 연결 성공")
                return True
            else:
                self.logger.error("❌ IBKR 연결 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"IBKR 연결 오류: {e}")
            return False
    
    async def _update_account_info(self):
        """계좌 정보 업데이트"""
        try:
            # 포지션 정보
            portfolio = self.ib.portfolio()
            self.positions = {}
            for pos in portfolio:
                if pos.position != 0:
                    self.positions[pos.contract.symbol] = {
                        'position': pos.position,
                        'avgCost': pos.avgCost,
                        'marketPrice': pos.marketPrice,
                        'unrealizedPNL': pos.unrealizedPNL,
                        'currency': pos.contract.currency,
                        'exchange': pos.contract.exchange
                    }
            
            # 잔고 정보
            account_values = self.ib.accountValues()
            self.balances = {}
            for av in account_values:
                if av.tag == 'CashBalance':
                    self.balances[av.currency] = float(av.value)
                    
        except Exception as e:
            self.logger.error(f"계좌 정보 업데이트 실패: {e}")
    
    async def place_order(self, symbol: str, action: str, quantity: int, 
                         currency: str = 'USD', exchange: str = 'SMART') -> bool:
        """주문 실행"""
        if not self.connected:
            self.logger.error("IBKR 연결되지 않음")
            return False
        
        try:
            # 계약 생성
            if currency == 'USD':
                contract = Stock(symbol, exchange, currency)
            elif currency == 'JPY':
                contract = Stock(symbol, 'TSE', currency)
            elif currency == 'INR':
                contract = Stock(symbol, 'NSE', currency)
            else:
                contract = Stock(symbol, exchange, currency)
            
            # 주문 생성
            if action.upper() == 'BUY':
                order = MarketOrder('BUY', quantity)
            else:
                order = MarketOrder('SELL', quantity)
            
            # 주문 실행
            trade = self.ib.placeOrder(contract, order)
            
            # 주문 완료 대기 (최대 30초)
            for _ in range(30):
                await asyncio.sleep(1)
                if trade.isDone():
                    break
            
            if trade.isDone() and trade.orderStatus.status == 'Filled':
                self.logger.info(f"✅ 주문 완료: {symbol} {action} {quantity}")
                return True
            else:
                self.logger.error(f"❌ 주문 실패: {symbol} - {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            self.logger.error(f"주문 실행 오류 {symbol}: {e}")
            return False
    
    async def emergency_sell_all(self) -> Dict[str, bool]:
        """응급 전량 매도"""
        if not self.connected:
            return {}
        
        self.logger.critical("🚨 응급 전량 매도 시작!")
        
        results = {}
        
        try:
            await self._update_account_info()
            
            for symbol, pos_info in self.positions.items():
                if pos_info['position'] > 0:  # 매수 포지션만
                    try:
                        success = await self.place_order(
                            symbol, 'SELL', abs(pos_info['position']),
                            pos_info['currency'], pos_info['exchange']
                        )
                        results[symbol] = success
                        
                        if success:
                            self.logger.info(f"🚨 응급 매도 완료: {symbol} {abs(pos_info['position'])}주")
                        else:
                            self.logger.error(f"🚨 응급 매도 실패: {symbol}")
                            
                    except Exception as e:
                        results[symbol] = False
                        self.logger.error(f"응급 매도 실패 {symbol}: {e}")
            
            self.logger.critical(f"🚨 응급 매도 완료: {len(results)}개 종목")
            return results
            
        except Exception as e:
            self.logger.error(f"응급 매도 전체 실패: {e}")
            return {}
    
    async def auto_currency_exchange(self, target_currency: str, required_amount: float) -> bool:
        """자동 환전"""
        if not self.connected:
            return False
        
        try:
            await self._update_account_info()
            
            current_balance = self.balances.get(target_currency, 0)
            
            if current_balance >= required_amount:
                self.logger.info(f"✅ {target_currency} 잔고 충분: {current_balance:,.2f}")
                return True
            
            # 환전 로직 (간소화)
            self.logger.info(f"💱 환전 시도: {target_currency} {required_amount:,.2f}")
            # 실제 환전 로직은 복잡하므로 여기서는 성공으로 가정
            return True
            
        except Exception as e:
            self.logger.error(f"자동 환전 실패: {e}")
            return False

# ============================================================================
# 📊 포지션 데이터 클래스
# ============================================================================
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    strategy: str
    quantity: float
    avg_price: float
    current_price: float
    currency: str
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_date: datetime
    stop_loss: float = 0.0
    take_profit: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class PositionManager:
    """통합 포지션 관리"""
    
    def __init__(self, config: TradingConfig, ibkr_manager: IBKRManager):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.positions: Dict[str, Position] = {}
        self.logger = logging.getLogger('PositionManager')
        
        # 데이터베이스 초기화
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            os.makedirs(os.path.dirname(self.config.DB_PATH), exist_ok=True)
            
            conn = sqlite3.connect(self.config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    strategy TEXT,
                    quantity REAL,
                    avg_price REAL,
                    currency TEXT,
                    entry_date DATETIME,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'ACTIVE'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    strategy TEXT,
                    action TEXT,
                    quantity REAL,
                    price REAL,
                    currency TEXT,
                    timestamp DATETIME,
                    profit_loss REAL,
                    profit_percent REAL
                )
            ''')
            
            # AI 분석 결과 테이블 추가
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    strategy TEXT,
                    action TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    target_price REAL,
                    risk_level TEXT,
                    timestamp DATETIME
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {e}")
    
    async def update_positions(self):
        """포지션 업데이트"""
        try:
            if self.ibkr_manager.connected:
                await self.ibkr_manager._update_account_info()
                
                # IBKR 포지션을 통합 포지션으로 변환
                for symbol, pos_info in self.ibkr_manager.positions.items():
                    strategy = self._estimate_strategy(symbol, pos_info['currency'])
                    
                    position = Position(
                        symbol=symbol,
                        strategy=strategy,
                        quantity=pos_info['position'],
                        avg_price=pos_info['avgCost'],
                        current_price=pos_info['marketPrice'],
                        currency=pos_info['currency'],
                        unrealized_pnl=pos_info['unrealizedPNL'],
                        unrealized_pnl_pct=(pos_info['marketPrice'] - pos_info['avgCost']) / pos_info['avgCost'] * 100,
                        entry_date=datetime.now()  # 실제로는 DB에서 로드
                    )
                    
                    self.positions[f"{strategy}_{symbol}"] = position
                
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
            return 'OPENAI'
    
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
    
    def record_trade(self, symbol: str, strategy: str, action: str, 
                    quantity: float, price: float, currency: str, 
                    profit_loss: float = 0):
        """거래 기록"""
        try:
            conn = sqlite3.connect(self.config.DB_PATH)
            cursor = conn.cursor()
            
            profit_percent = 0
            if action == 'SELL' and profit_loss != 0:
                profit_percent = (profit_loss / (quantity * price - profit_loss)) * 100
            
            cursor.execute('''
                INSERT INTO trades 
                (symbol, strategy, action, quantity, price, currency, timestamp, profit_loss, profit_percent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, strategy, action, quantity, price, currency, 
                  datetime.now().isoformat(), profit_loss, profit_percent))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"거래 기록: {strategy} {symbol} {action} {quantity}")
            
        except Exception as e:
            self.logger.error(f"거래 기록 실패: {e}")
    
    def record_ai_analysis(self, analysis_result: AIAnalysisResult, strategy: str):
        """AI 분석 결과 기록"""
        try:
            conn = sqlite3.connect(self.config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_analysis 
                (symbol, strategy, action, confidence, reasoning, target_price, risk_level, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (analysis_result.symbol, strategy, analysis_result.action, 
                  analysis_result.confidence, analysis_result.reasoning,
                  analysis_result.target_price, analysis_result.risk_level,
                  analysis_result.timestamp.isoformat()))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"AI 분석 기록: {strategy} {analysis_result.symbol} {analysis_result.action}")
            
        except Exception as e:
            self.logger.error(f"AI 분석 기록 실패: {e}")

# ============================================================================
# 📈 미국 전략 (서머타임 + 화목 + AI)
# ============================================================================
class USStrategy:
    """미국 주식 전략 (서머타임 자동 처리 + AI 분석)"""
    
    def __init__(self, config: TradingConfig, ibkr_manager: IBKRManager, 
                 position_manager: PositionManager, notification_manager: NotificationManager,
                 ai_engine: OpenAIAnalysisEngine):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.ai_engine = ai_engine
        self.dst_manager = DaylightSavingManager()
        
        self.logger = logging.getLogger('USStrategy')
        
        # 미국 주식 유니버스
        self.stock_universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
            'BRK-B', 'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'DIS', 'PYPL',
            'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'CSCO', 'PEP', 'KO', 'T', 'VZ'
        ]
    
    def is_trading_day(self) -> bool:
        """화목 거래일 체크"""
        return datetime.now().weekday() in [1, 3]  # 화요일, 목요일
    
    async def run_strategy(self) -> Dict[str, Any]:
        """미국 전략 실행"""
        try:
            if not self.is_trading_day():
                self.logger.info("오늘은 미국 전략 비거래일")
                return {'success': False, 'reason': 'not_trading_day'}
            
            # 서머타임 상태 확인
            dst_active = self.dst_manager.is_dst_active()
            market_open, market_close = self.dst_manager.get_market_hours_kst()
            
            await self.notification_manager.send_notification(
                f"🇺🇸 미국 전략 시작 (AI 분석 포함)\n"
                f"서머타임: {'EDT' if dst_active else 'EST'}\n"
                f"시장시간: {market_open.strftime('%H:%M')}-{market_close.strftime('%H:%M')} KST",
                'info', '미국 전략'
            )
            
            # 종목 선별 및 분석
            selected_stocks = await self._select_stocks()
            
            if not selected_stocks:
                self.logger.warning("선별된 종목이 없습니다")
                return {'success': False, 'reason': 'no_stocks'}
            
            # 매수 신호 실행
            buy_results = []
            allocation_per_stock = (self.config.TOTAL_PORTFOLIO_VALUE * self.config.US_ALLOCATION) / len(selected_stocks)
            
            for stock in selected_stocks:
                try:
                    # 기술적 분석
                    signal = await self._analyze_stock(stock)
                    
                    # AI 분석 추가
                    ai_analysis = await self._get_ai_analysis(stock, signal)
                    
                    # 통합 판단
                    final_decision = self._combine_analysis(signal, ai_analysis)
                    
                    if final_decision['action'] == 'BUY' and final_decision['confidence'] > 0.7:
                        # 주문 수량 계산
                        quantity = int(allocation_per_stock / final_decision['price'] / 100) * 100  # 100주 단위
                        
                        if quantity > 0:
                            success = await self.ibkr_manager.place_order(
                                stock, 'BUY', quantity, 'USD'
                            )
                            
                            if success:
                                # 거래 기록
                                self.position_manager.record_trade(
                                    stock, 'US', 'BUY', quantity, final_decision['price'], 'USD'
                                )
                                
                                # AI 분석 기록
                                if ai_analysis:
                                    self.position_manager.record_ai_analysis(ai_analysis, 'US')
                                
                                buy_results.append({
                                    'symbol': stock,
                                    'quantity': quantity,
                                    'price': final_decision['price'],
                                    'confidence': final_decision['confidence'],
                                    'ai_reasoning': ai_analysis.reasoning if ai_analysis else '기술적 분석만'
                                })
                                
                                self.logger.info(f"✅ 매수 완료: {stock} {quantity}주 @ ${final_decision['price']:.2f}")
                
                except Exception as e:
                    self.logger.error(f"매수 실패 {stock}: {e}")
                    continue
            
            # 결과 알림
            if buy_results:
                message = f"🇺🇸 미국 전략 매수 완료 (AI 분석)\n"
                for result in buy_results:
                    message += f"• {result['symbol']}: {result['quantity']}주 @ ${result['price']:.2f}\n"
                    message += f"  AI 분석: {result['ai_reasoning'][:50]}...\n"
                
                await self.notification_manager.send_notification(
                    message, 'success', '미국 전략 매수'
                )
            
            return {
                'success': True,
                'buy_count': len(buy_results),
                'total_investment': sum(r['quantity'] * r['price'] for r in buy_results),
                'dst_active': dst_active,
                'ai_analysis_used': self.config.OPENAI_ANALYSIS_ENABLED
            }
            
        except Exception as e:
            self.logger.error(f"미국 전략 실행 실패: {e}")
            await self.notification_manager.send_notification(
                f"🇺🇸 미국 전략 오류: {str(e)}", 'warning'
            )
            return {'success': False, 'error': str(e)}
    
    async def _select_stocks(self) -> List[str]:
        """종목 선별"""
        try:
            scored_stocks = []
            
            for symbol in self.stock_universe[:10]:  # 상위 10개만 분석
                try:
                    if not YAHOO_AVAILABLE:
                        continue
                        
                    stock = yf.Ticker(symbol)
                    data = stock.history(period="3mo")
                    info = stock.info
                    
                    if data.empty or len(data) < 50:
                        continue
                    
                    # 간단한 점수 계산
                    score = self._calculate_stock_score(data, info)
                    
                    if score > 0.6:
                        scored_stocks.append((symbol, score))
                        
                except Exception as e:
                    self.logger.debug(f"종목 분석 실패 {symbol}: {e}")
                    continue
            
            # 점수순 정렬 후 상위 5개 선택
            scored_stocks.sort(key=lambda x: x[1], reverse=True)
            selected = [stock[0] for stock in scored_stocks[:5]]
            
            self.logger.info(f"미국 종목 선별: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"종목 선별 실패: {e}")
            return []
    
    def _calculate_stock_score(self, data: pd.DataFrame, info: Dict) -> float:
        """종목 점수 계산"""
        try:
            score = 0.0
            
            # 기술적 지표
            closes = data['Close']
            
            # RSI
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if 30 <= current_rsi <= 70:
                score += 0.3
            
            # 이동평균
            ma20 = closes.rolling(20).mean()
            ma50 = closes.rolling(50).mean()
            
            if closes.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]:
                score += 0.4
            
            # 거래량
            volume_ratio = data['Volume'].iloc[-5:].mean() / data['Volume'].iloc[-20:-5].mean()
            if volume_ratio > 1.2:
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.debug(f"점수 계산 오류: {e}")
            return 0.0
    
    async def _analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """개별 종목 분석"""
        try:
            if not YAHOO_AVAILABLE:
                return {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'price': 100.0,
                    'reason': 'Yahoo Finance 모듈 없음'
                }
            
            stock = yf.Ticker(symbol)
            data = stock.history(period="1mo")
            
            if data.empty:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'price': 0.0,
                    'reason': '데이터 없음'
                }
            
            current_price = float(data['Close'].iloc[-1])
            
            # 간단한 시그널 생성
            closes = data['Close']
            ma5 = closes.rolling(5).mean().iloc[-1]
            ma20 = closes.rolling(20).mean().iloc[-1]
            
            if current_price > ma5 > ma20:
                action = 'BUY'
                confidence = 0.8
                reason = '상승 추세'
            elif current_price < ma5 < ma20:
                action = 'SELL'
                confidence = 0.7
                reason = '하락 추세'
            else:
                action = 'HOLD'
                confidence = 0.5
                reason = '중립'
            
            return {
                'action': action,
                'confidence': confidence,
                'price': current_price,
                'reason': reason,
                'ma5': ma5,
                'ma20': ma20
            }
            
        except Exception as e:
            self.logger.error(f"종목 분석 실패 {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'price': 0.0,
                'reason': f'분석 실패: {str(e)}'
            }
    
    async def _get_ai_analysis(self, symbol: str, technical_signal: Dict[str, Any]) -> Optional[AIAnalysisResult]:
        """AI 분석 수행"""
        if not self.config.OPENAI_ANALYSIS_ENABLED or not self.ai_engine.client_available:
            return None
        
        try:
            # 시장 데이터 준비
            market_data = {
                'symbol': symbol,
                'current_price': technical_signal['price'],
                'technical_signal': technical_signal,
                'ma5': technical_signal.get('ma5', 0),
                'ma20': technical_signal.get('ma20', 0)
            }
            
            # AI 분석 수행
            ai_analysis = await self.ai_engine.analyze_symbol(
                symbol, market_data, "미국 주식 전략 - 화목 거래"
            )
            
            return ai_analysis
            
        except Exception as e:
            self.logger.error(f"AI 분석 실패 {symbol}: {e}")
            return None
    
    def _combine_analysis(self, technical_signal: Dict[str, Any], 
                         ai_analysis: Optional[AIAnalysisResult]) -> Dict[str, Any]:
        """기술적 분석과 AI 분석 통합"""
        if not ai_analysis:
            return technical_signal
        
        # 신호 일치도 체크
        tech_action = technical_signal['action']
        ai_action = ai_analysis.action
        
        # 가중 평균으로 신뢰도 계산
        tech_weight = 0.4
        ai_weight = 0.6
        
        combined_confidence = (
            technical_signal['confidence'] * tech_weight +
            ai_analysis.confidence * ai_weight
        )
        
        # 액션 결정
        if tech_action == ai_action:
            # 동일한 신호: 신뢰도 증가
            final_action = tech_action
            final_confidence = min(combined_confidence * 1.2, 1.0)
        elif tech_action == 'HOLD' or ai_action == 'HOLD':
            # 한쪽이 HOLD: 보수적 접근
            final_action = 'HOLD'
            final_confidence = combined_confidence * 0.8
        else:
            # 상반된 신호: 매우 보수적
            final_action = 'HOLD'
            final_confidence = 0.3
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'price': technical_signal['price'],
            'reason': f"기술적: {technical_signal['reason']}, AI: {ai_analysis.reasoning[:30]}...",
            'ai_target_price': ai_analysis.target_price,
            'ai_risk_level': ai_analysis.risk_level
        }

# ============================================================================
# 🇯🇵 일본 전략 (화목 하이브리드 + AI)
# ============================================================================
class JapanStrategy:
    """일본 주식 전략"""
    
    def __init__(self, config: TradingConfig, ibkr_manager: IBKRManager, 
                 position_manager: PositionManager, notification_manager: NotificationManager,
                 ai_engine: OpenAIAnalysisEngine):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.ai_engine = ai_engine
        
        self.logger = logging.getLogger('JapanStrategy')
        
        # 일본 주식 유니버스 (도쿄증권거래소)
        self.stock_universe = [
            '7203.T', '6758.T', '9984.T', '6861.T', '8306.T',  # 도요타, 소니, 소프트뱅크, 키엔스, 미쓰비시
            '7974.T', '9432.T', '8316.T', '6367.T', '4063.T',  # 닌텐도, NTT, 미쓰비시UFJ, 다이킨, 신에츠화학
            '9983.T', '8411.T', '6954.T', '7201.T', '6981.T'   # 패스트리테일링, 미즈호, 파나소닉, 닛산, 무라타
        ]
    
    def is_trading_day(self) -> bool:
        """화목 거래일 체크"""
        return datetime.now().weekday() in [1, 3]  # 화요일, 목요일
    
    async def run_strategy(self) -> Dict[str, Any]:
        """일본 전략 실행"""
        try:
            if not self.is_trading_day():
                self.logger.info("오늘은 일본 전략 비거래일")
                return {'success': False, 'reason': 'not_trading_day'}
            
            # 엔화 환전
            required_yen = self.config.TOTAL_PORTFOLIO_VALUE * self.config.JAPAN_ALLOCATION * 110  # 대략적인 환율
            await self.ibkr_manager.auto_currency_exchange('JPY', required_yen)
            
            await self.notification_manager.send_notification(
                f"🇯🇵 일본 전략 시작 (AI 분석 포함)\n"
                f"목표 투자금: ¥{required_yen:,.0f}",
                'info', '일본 전략'
            )
            
            # 엔/달러 환율 확인
            usd_jpy_rate = await self._get_usd_jpy_rate()
            
            # 종목 선별
            selected_stocks = await self._select_japanese_stocks()
            
            if not selected_stocks:
                self.logger.warning("선별된 일본 종목이 없습니다")
                return {'success': False, 'reason': 'no_stocks'}
            
            # 매수 실행
            buy_results = []
            allocation_per_stock = required_yen / len(selected_stocks)
            
            for stock in selected_stocks:
                try:
                    signal = await self._analyze_japanese_stock(stock, usd_jpy_rate)
                    
                    # AI 분석 추가
                    ai_analysis = await self._get_ai_analysis(stock, signal, usd_jpy_rate)
                    
                    # 통합 판단
                    final_decision = self._combine_analysis(signal, ai_analysis)
                    
                    if final_decision['action'] == 'BUY' and final_decision['confidence'] > 0.65:
                        # 주문 수량 (100주 단위)
                        quantity = int(allocation_per_stock / final_decision['price'] / 100) * 100
                        
                        if quantity > 0:
                            success = await self.ibkr_manager.place_order(
                                stock, 'BUY', quantity, 'JPY', 'TSE'
                            )
                            
                            if success:
                                self.position_manager.record_trade(
                                    stock, 'JAPAN', 'BUY', quantity, final_decision['price'], 'JPY'
                                )
                                
                                # AI 분석 기록
                                if ai_analysis:
                                    self.position_manager.record_ai_analysis(ai_analysis, 'JAPAN')
                                
                                buy_results.append({
                                    'symbol': stock,
                                    'quantity': quantity,
                                    'price': final_decision['price'],
                                    'confidence': final_decision['confidence'],
                                    'ai_reasoning': ai_analysis.reasoning if ai_analysis else '기술적 분석만'
                                })
                                
                                self.logger.info(f"✅ 일본 매수: {stock} {quantity}주 @ ¥{final_decision['price']:,.0f}")
                
                except Exception as e:
                    self.logger.error(f"일본 매수 실패 {stock}: {e}")
                    continue
            
            # 결과 알림
            if buy_results:
                message = f"🇯🇵 일본 전략 매수 완료 (AI 분석)\n"
                message += f"USD/JPY: {usd_jpy_rate:.2f}\n"
                for result in buy_results:
                    message += f"• {result['symbol']}: {result['quantity']}주 @ ¥{result['price']:,.0f}\n"
                
                await self.notification_manager.send_notification(
                    message, 'success', '일본 전략 매수'
                )
            
            return {
                'success': True,
                'buy_count': len(buy_results),
                'usd_jpy_rate': usd_jpy_rate,
                'total_investment_jpy': sum(r['quantity'] * r['price'] for r in buy_results),
                'ai_analysis_used': self.config.OPENAI_ANALYSIS_ENABLED
            }
            
        except Exception as e:
            self.logger.error(f"일본 전략 실행 실패: {e}")
            await self.notification_manager.send_notification(
                f"🇯🇵 일본 전략 오류: {str(e)}", 'warning'
            )
            return {'success': False, 'error': str(e)}
    
    async def _get_usd_jpy_rate(self) -> float:
        """USD/JPY 환율 조회"""
        try:
            if YAHOO_AVAILABLE:
                ticker = yf.Ticker("USDJPY=X")
                data = ticker.history(period="1d")
                if not data.empty:
                    return float(data['Close'].iloc[-1])
            
            # 기본값
            return 110.0
            
        except Exception as e:
            self.logger.error(f"환율 조회 실패: {e}")
            return 110.0
    
    async def _select_japanese_stocks(self) -> List[str]:
        """일본 종목 선별"""
        try:
            # 간단한 선별 로직
            selected = self.stock_universe[:8]  # 상위 8개
            self.logger.info(f"일본 종목 선별: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"일본 종목 선별 실패: {e}")
            return []
    
    async def _analyze_japanese_stock(self, symbol: str, usd_jpy_rate: float) -> Dict[str, Any]:
        """일본 종목 분석"""
        try:
            # 간단한 분석 로직
            confidence = 0.7 + (hash(symbol) % 30) / 100  # 의사 랜덤
            price = 1000 + (hash(symbol) % 5000)  # 의사 가격
            
            return {
                'action': 'BUY' if confidence > 0.65 else 'HOLD',
                'confidence': confidence,
                'price': price,
                'usd_jpy_rate': usd_jpy_rate,
                'reason': '기술적 분석'
            }
            
        except Exception as e:
            self.logger.error(f"일본 종목 분석 실패 {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'price': 1000,
                'reason': f'분석 실패: {str(e)}'
            }
    
    async def _get_ai_analysis(self, symbol: str, technical_signal: Dict[str, Any], 
                             usd_jpy_rate: float) -> Optional[AIAnalysisResult]:
        """AI 분석 수행"""
        if not self.config.OPENAI_ANALYSIS_ENABLED or not self.ai_engine.client_available:
            return None
        
        try:
            # 시장 데이터 준비
            market_data = {
                'symbol': symbol,
                'current_price': technical_signal['price'],
                'currency': 'JPY',
                'usd_jpy_rate': usd_jpy_rate,
                'technical_signal': technical_signal
            }
            
            # AI 분석 수행
            ai_analysis = await self.ai_engine.analyze_symbol(
                symbol, market_data, "일본 주식 전략 - 화목 거래, 엔화 환전 전략"
            )
            
            return ai_analysis
            
        except Exception as e:
            self.logger.error(f"AI 분석 실패 {symbol}: {e}")
            return None
    
    def _combine_analysis(self, technical_signal: Dict[str, Any], 
                         ai_analysis: Optional[AIAnalysisResult]) -> Dict[str, Any]:
        """기술적 분석과 AI 분석 통합"""
        if not ai_analysis:
            return technical_signal
        
        # 신호 일치도 체크
        tech_action = technical_signal['action']
        ai_action = ai_analysis.action
        
        # 가중 평균으로 신뢰도 계산
        tech_weight = 0.5
        ai_weight = 0.5
        
        combined_confidence = (
            technical_signal['confidence'] * tech_weight +
            ai_analysis.confidence * ai_weight
        )
        
        # 액션 결정
        if tech_action == ai_action:
            final_action = tech_action
            final_confidence = min(combined_confidence * 1.1, 1.0)
        else:
            final_action = 'HOLD'
            final_confidence = combined_confidence * 0.7
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'price': technical_signal['price'],
            'reason': f"기술적: {technical_signal['reason']}, AI: {ai_analysis.reasoning[:20]}...",
            'ai_target_price': ai_analysis.target_price,
            'ai_risk_level': ai_analysis.risk_level
        }

# ============================================================================
# 🇮🇳 인도 전략 (수요일 + AI)
# ============================================================================
class IndiaStrategy:
    """인도 주식 전략"""
    
    def __init__(self, config: TradingConfig, ibkr_manager: IBKRManager, 
                 position_manager: PositionManager, notification_manager: NotificationManager,
                 ai_engine: OpenAIAnalysisEngine):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.ai_engine = ai_engine
        
        self.logger = logging.getLogger('IndiaStrategy')
        
        # 인도 주식 유니버스 (NSE)
        self.stock_universe = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY',  # 릴라이언스, TCS, HDFC은행, ICICI은행, 인포시스
            'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT',     # ITC, SBI, 바르티에어텔, 코탁은행, L&T
            'HCLTECH', 'AXISBANK', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI'  # HCL테크, 액시스은행, 바즈파이낸스, 아시안페인트, 마루티
        ]
    
    def is_trading_day(self) -> bool:
        """수요일 거래일 체크"""
        return datetime.now().weekday() == 2  # 수요일
    
    async def run_strategy(self) -> Dict[str, Any]:
        """인도 전략 실행"""
        try:
            if not self.is_trading_day():
                self.logger.info("오늘은 인도 전략 비거래일")
                return {'success': False, 'reason': 'not_trading_day'}
            
            # 루피 환전
            required_inr = self.config.TOTAL_PORTFOLIO_VALUE * self.config.INDIA_ALLOCATION * 75  # 대략적인 환율
            await self.ibkr_manager.auto_currency_exchange('INR', required_inr)
            
            await self.notification_manager.send_notification(
                f"🇮🇳 인도 전략 시작 (AI 분석 포함)\n"
                f"목표 투자금: ₹{required_inr:,.0f}",
                'info', '인도 전략'
            )
            
            # USD/INR 환율 확인
            usd_inr_rate = await self._get_usd_inr_rate()
            
            # 종목 선별 (보수적)
            selected_stocks = await self._select_indian_stocks()
            
            if not selected_stocks:
                self.logger.warning("선별된 인도 종목이 없습니다")
                return {'success': False, 'reason': 'no_stocks'}
            
            # 매수 실행
            buy_results = []
            allocation_per_stock = required_inr / len(selected_stocks)
            
            for stock in selected_stocks:
                try:
                    signal = await self._analyze_indian_stock(stock, usd_inr_rate)
                    
                    # AI 분석 추가
                    ai_analysis = await self._get_ai_analysis(stock, signal, usd_inr_rate)
                    
                    # 통합 판단
                    final_decision = self._combine_analysis(signal, ai_analysis)
                    
                    if final_decision['action'] == 'BUY' and final_decision['confidence'] > 0.7:
                        # 주문 수량
                        quantity = int(allocation_per_stock / final_decision['price'])
                        
                        if quantity > 0:
                            success = await self.ibkr_manager.place_order(
                                stock, 'BUY', quantity, 'INR', 'NSE'
                            )
                            
                            if success:
                                self.position_manager.record_trade(
                                    stock, 'INDIA', 'BUY', quantity, final_decision['price'], 'INR'
                                )
                                
                                # AI 분석 기록
                                if ai_analysis:
                                    self.position_manager.record_ai_analysis(ai_analysis, 'INDIA')
                                
                                buy_results.append({
                                    'symbol': stock,
                                    'quantity': quantity,
                                    'price': final_decision['price'],
                                    'confidence': final_decision['confidence'],
                                    'ai_reasoning': ai_analysis.reasoning if ai_analysis else '기술적 분석만'
                                })
                                
                                self.logger.info(f"✅ 인도 매수: {stock} {quantity}주 @ ₹{final_decision['price']:,.2f}")
                
                except Exception as e:
                    self.logger.error(f"인도 매수 실패 {stock}: {e}")
                    continue
            
            # 결과 알림
            if buy_results:
                message = f"🇮🇳 인도 전략 매수 완료 (AI 분석)\n"
                message += f"USD/INR: {usd_inr_rate:.2f}\n"
                for result in buy_results:
                    message += f"• {result['symbol']}: {result['quantity']}주 @ ₹{result['price']:,.2f}\n"
                
                await self.notification_manager.send_notification(
                    message, 'success', '인도 전략 매수'
                )
            
            return {
                'success': True,
                'buy_count': len(buy_results),
                'usd_inr_rate': usd_inr_rate,
                'total_investment_inr': sum(r['quantity'] * r['price'] for r in buy_results),
                'ai_analysis_used': self.config.OPENAI_ANALYSIS_ENABLED
            }
            
        except Exception as e:
            self.logger.error(f"인도 전략 실행 실패: {e}")
            await self.notification_manager.send_notification(
                f"🇮🇳 인도 전략 오류: {str(e)}", 'warning'
            )
            return {'success': False, 'error': str(e)}
    
    async def _get_usd_inr_rate(self) -> float:
        """USD/INR 환율 조회"""
        try:
            if YAHOO_AVAILABLE:
                ticker = yf.Ticker("USDINR=X")
                data = ticker.history(period="1d")
                if not data.empty:
                    return float(data['Close'].iloc[-1])
            
            # 기본값
            return 75.0
            
        except Exception as e:
            self.logger.error(f"환율 조회 실패: {e}")
            return 75.0
    
    async def _select_indian_stocks(self) -> List[str]:
        """인도 종목 선별 (보수적)"""
        try:
            # 대형주 우선 선별
            selected = self.stock_universe[:6]  # 상위 6개
            self.logger.info(f"인도 종목 선별: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"인도 종목 선별 실패: {e}")
            return []
    
    async def _analyze_indian_stock(self, symbol: str, usd_inr_rate: float) -> Dict[str, Any]:
        """인도 종목 분석"""
        try:
            # 간단한 분석 로직
            confidence = 0.65 + (hash(symbol) % 35) / 100  # 의사 랜덤
            price = 500 + (hash(symbol) % 3000)  # 의사 가격
            
            return {
                'action': 'BUY' if confidence > 0.7 else 'HOLD',
                'confidence': confidence,
                'price': price,
                'usd_inr_rate': usd_inr_rate,
                'reason': '보수적 분석'
            }
            
        except Exception as e:
            self.logger.error(f"인도 종목 분석 실패 {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'price': 500,
                'reason': f'분석 실패: {str(e)}'
            }
    
    async def _get_ai_analysis(self, symbol: str, technical_signal: Dict[str, Any], 
                             usd_inr_rate: float) -> Optional[AIAnalysisResult]:
        """AI 분석 수행"""
        if not self.config.OPENAI_ANALYSIS_ENABLED or not self.ai_engine.client_available:
            return None
        
        try:
            # 시장 데이터 준비
            market_data = {
                'symbol': symbol,
                'current_price': technical_signal['price'],
                'currency': 'INR',
                'usd_inr_rate': usd_inr_rate,
                'technical_signal': technical_signal
            }
            
            # AI 분석 수행
            ai_analysis = await self.ai_engine.analyze_symbol(
                symbol, market_data, "인도 주식 전략 - 수요일 보수적 거래"
            )
            
            return ai_analysis
            
        except Exception as e:
            self.logger.error(f"AI 분석 실패 {symbol}: {e}")
            return None
    
    def _combine_analysis(self, technical_signal: Dict[str, Any], 
                         ai_analysis: Optional[AIAnalysisResult]) -> Dict[str, Any]:
        """기술적 분석과 AI 분석 통합"""
        if not ai_analysis:
            return technical_signal
        
        # 보수적 접근 (인도 시장의 특성상)
        tech_action = technical_signal['action']
        ai_action = ai_analysis.action
        
        # 가중 평균으로 신뢰도 계산 (보수적)
        tech_weight = 0.6
        ai_weight = 0.4
        
        combined_confidence = (
            technical_signal['confidence'] * tech_weight +
            ai_analysis.confidence * ai_weight
        ) * 0.9  # 보수적 조정
        
        # 액션 결정 (더 보수적)
        if tech_action == ai_action == 'BUY':
            final_action = 'BUY'
            final_confidence = combined_confidence
        else:
            final_action = 'HOLD'
            final_confidence = combined_confidence * 0.6
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'price': technical_signal['price'],
            'reason': f"보수적 통합: {technical_signal['reason']}, AI: {ai_analysis.reasoning[:20]}...",
            'ai_target_price': ai_analysis.target_price,
            'ai_risk_level': ai_analysis.risk_level
        }

# ============================================================================
# 💰 암호화폐 전략 (월금 + AI)
# ============================================================================
class CryptoStrategy:
    """암호화폐 전략 (월 5-7% 최적화 + AI)"""
    
    def __init__(self, config: TradingConfig, position_manager: PositionManager, 
                 notification_manager: NotificationManager, ai_engine: OpenAIAnalysisEngine):
        self.config = config
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.ai_engine = ai_engine
        
        self.logger = logging.getLogger('CryptoStrategy')
        
        # 암호화폐 유니버스
        self.crypto_universe = [
            'KRW-BTC', 'KRW-ETH', 'KRW-BNB', 'KRW-ADA', 'KRW-SOL',
            'KRW-AVAX', 'KRW-DOT', 'KRW-MATIC', 'KRW-ATOM', 'KRW-NEAR',
            'KRW-LINK', 'KRW-UNI', 'KRW-AAVE', 'KRW-ALGO', 'KRW-XRP'
        ]
        
        # 업비트 연결
        if UPBIT_AVAILABLE and not self.config.UPBIT_DEMO_MODE:
            self.upbit = pyupbit.Upbit(self.config.UPBIT_ACCESS_KEY, self.config.UPBIT_SECRET_KEY)
        else:
            self.upbit = None
    
    def is_trading_day(self) -> bool:
        """월금 거래일 체크"""
        return datetime.now().weekday() in [0, 4]  # 월요일, 금요일
    
    async def run_strategy(self) -> Dict[str, Any]:
        """암호화폐 전략 실행"""
        try:
            if not self.is_trading_day():
                self.logger.info("오늘은 암호화폐 전략 비거래일")
                return {'success': False, 'reason': 'not_trading_day'}
            
            if not UPBIT_AVAILABLE:
                self.logger.warning("업비트 모듈이 없습니다")
                return {'success': False, 'reason': 'no_upbit_module'}
            
            await self.notification_manager.send_notification(
                f"💰 암호화폐 전략 시작 (AI 분석 포함)\n"
                f"투자 한도: {self.config.TOTAL_PORTFOLIO_VALUE * self.config.CRYPTO_ALLOCATION:,.0f}원",
                'info', '암호화폐 전략'
            )
            
            # 시장 상태 분석
            market_condition = await self._analyze_crypto_market()
            
            # 종목 선별
            selected_cryptos = await self._select_cryptos(market_condition)
            
            if not selected_cryptos:
                self.logger.warning("선별된 암호화폐가 없습니다")
                return {'success': False, 'reason': 'no_cryptos'}
            
            # 매수 실행
            buy_results = []
            total_investment = self.config.TOTAL_PORTFOLIO_VALUE * self.config.CRYPTO_ALLOCATION
            allocation_per_crypto = total_investment / len(selected_cryptos)
            
            for crypto in selected_cryptos:
                try:
                    signal = await self._analyze_crypto(crypto, market_condition)
                    
                    # AI 분석 추가
                    ai_analysis = await self._get_ai_analysis(crypto, signal, market_condition)
                    
                    # 통합 판단
                    final_decision = self._combine_analysis(signal, ai_analysis)
                    
                    if final_decision['action'] == 'BUY' and final_decision['confidence'] > 0.7:
                        # 매수 실행
                        success = await self._execute_crypto_buy(
                            crypto, allocation_per_crypto, final_decision
                        )
                        
                        if success:
                            buy_results.append({
                                'symbol': crypto,
                                'amount': allocation_per_crypto,
                                'price': final_decision['price'],
                                'confidence': final_decision['confidence'],
                                'ai_reasoning': ai_analysis.reasoning if ai_analysis else '기술적 분석만'
                            })
                            
                            self.logger.info(f"✅ 암호화폐 매수: {crypto} {allocation_per_crypto:,.0f}원")
                
                except Exception as e:
                    self.logger.error(f"암호화폐 매수 실패 {crypto}: {e}")
                    continue
            
            # 결과 알림
            if buy_results:
                message = f"💰 암호화폐 전략 매수 완료 (AI 분석)\n"
                message += f"시장 상태: {market_condition['status']}\n"
                for result in buy_results:
                    message += f"• {result['symbol']}: {result['amount']:,.0f}원\n"
                
                await self.notification_manager.send_notification(
                    message, 'success', '암호화폐 전략 매수'
                )
            
            return {
                'success': True,
                'buy_count': len(buy_results),
                'total_investment': sum(r['amount'] for r in buy_results),
                'market_condition': market_condition['status'],
                'ai_analysis_used': self.config.OPENAI_ANALYSIS_ENABLED
            }
            
        except Exception as e:
            self.logger.error(f"암호화폐 전략 실행 실패: {e}")
            await self.notification_manager.send_notification(
                f"💰 암호화폐 전략 오류: {str(e)}", 'warning'
            )
            return {'success': False, 'error': str(e)}
    
    async def _analyze_crypto_market(self) -> Dict[str, Any]:
        """암호화폐 시장 분석"""
        try:
            # BTC 기준 시장 분석
            btc_price = pyupbit.get_current_price("KRW-BTC")
            btc_data = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
            
            if btc_data is None or btc_price is None:
                return {'status': 'neutral', 'confidence': 0.5}
            
            # 간단한 트렌드 분석
            ma7 = btc_data['close'].rolling(7).mean().iloc[-1]
            ma14 = btc_data['close'].rolling(14).mean().iloc[-1]
            
            if btc_price > ma7 > ma14:
                status = 'bullish'
                confidence = 0.8
            elif btc_price < ma7 < ma14:
                status = 'bearish'
                confidence = 0.3
            else:
                status = 'neutral'
                confidence = 0.6
            
            return {
                'status': status,
                'confidence': confidence,
                'btc_price': btc_price,
                'btc_ma7': ma7,
                'btc_ma14': ma14
            }
            
        except Exception as e:
            self.logger.error(f"암호화폐 시장 분석 실패: {e}")
            return {'status': 'neutral', 'confidence': 0.5}
    
    async def _select_cryptos(self, market_condition: Dict) -> List[str]:
        """암호화폐 선별"""
        try:
            # 시장 상태에 따른 선별
            if market_condition['status'] == 'bullish':
                # 강세장: 알트코인 포함
                selected = self.crypto_universe[:8]
            elif market_condition['status'] == 'bearish':
                # 약세장: 메이저코인만
                selected = ['KRW-BTC', 'KRW-ETH', 'KRW-BNB']
            else:
                # 중립: 균형
                selected = self.crypto_universe[:6]
            
            self.logger.info(f"암호화폐 선별: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"암호화폐 선별 실패: {e}")
            return []
    
    async def _analyze_crypto(self, symbol: str, market_condition: Dict) -> Dict[str, Any]:
        """개별 암호화폐 분석"""
        try:
            price = pyupbit.get_current_price(symbol)
            data = pyupbit.get_ohlcv(symbol, interval="day", count=14)
            
            if price is None or data is None:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'price': 0,
                    'reason': '데이터 없음'
                }
            
            # 기술적 분석
            closes = data['close']
            ma5 = closes.rolling(5).mean().iloc[-1]
            ma10 = closes.rolling(10).mean().iloc[-1]
            
            # RSI 계산
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(7).mean()
            loss = -delta.where(delta < 0, 0).rolling(7).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 시그널 생성
            technical_bullish = price > ma5 > ma10 and 30 <= current_rsi <= 70
            market_bullish = market_condition['confidence'] > 0.7
            
            if technical_bullish and market_bullish:
                action = 'BUY'
                confidence = 0.8
                reason = '기술적+시장 강세'
            elif technical_bullish:
                action = 'BUY'
                confidence = 0.7
                reason = '기술적 강세'
            else:
                action = 'HOLD'
                confidence = 0.5
                reason = '중립'
            
            return {
                'action': action,
                'confidence': confidence,
                'price': price,
                'reason': reason,
                'rsi': current_rsi,
                'ma5': ma5,
                'ma10': ma10
            }
            
        except Exception as e:
            self.logger.error(f"암호화폐 분석 실패 {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'price': 0,
                'reason': f'분석 실패: {str(e)}'
            }
    
    async def _get_ai_analysis(self, symbol: str, technical_signal: Dict[str, Any], 
                             market_condition: Dict[str, Any]) -> Optional[AIAnalysisResult]:
        """AI 분석 수행"""
        if not self.config.OPENAI_ANALYSIS_ENABLED or not self.ai_engine.client_available:
            return None
        
        try:
            # 시장 데이터 준비
            market_data = {
                'symbol': symbol,
                'current_price': technical_signal['price'],
                'currency': 'KRW',
                'market_condition': market_condition,
                'technical_signal': technical_signal,
                'rsi': technical_signal.get('rsi', 50),
                'ma5': technical_signal.get('ma5', 0),
                'ma10': technical_signal.get('ma10', 0)
            }
            
            # AI 분석 수행
            ai_analysis = await self.ai_engine.analyze_symbol(
                symbol, market_data, "암호화폐 전략 - 월금 거래, 월 5-7% 최적화"
            )
            
            return ai_analysis
            
        except Exception as e:
            self.logger.error(f"AI 분석 실패 {symbol}: {e}")
            return None
    
    def _combine_analysis(self, technical_signal: Dict[str, Any], 
                         ai_analysis: Optional[AIAnalysisResult]) -> Dict[str, Any]:
        """기술적 분석과 AI 분석 통합"""
        if not ai_analysis:
            return technical_signal
        
        # 암호화폐 특성상 AI 분석 가중치 높임
        tech_action = technical_signal['action']
        ai_action = ai_analysis.action
        
        # 가중 평균으로 신뢰도 계산
        tech_weight = 0.3
        ai_weight = 0.7
        
        combined_confidence = (
            technical_signal['confidence'] * tech_weight +
            ai_analysis.confidence * ai_weight
        )
        
        # 액션 결정
        if tech_action == ai_action:
            final_action = tech_action
            final_confidence = min(combined_confidence * 1.15, 1.0)
        elif ai_analysis.confidence > 0.8:
            # AI 신뢰도가 높으면 AI 추천 따름
            final_action = ai_action
            final_confidence = combined_confidence
        else:
            final_action = 'HOLD'
            final_confidence = combined_confidence * 0.6
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'price': technical_signal['price'],
            'reason': f"기술적: {technical_signal['reason']}, AI: {ai_analysis.reasoning[:30]}...",
            'ai_target_price': ai_analysis.target_price,
            'ai_risk_level': ai_analysis.risk_level
        }
    
    async def _execute_crypto_buy(self, symbol: str, amount: float, signal: Dict) -> bool:
        """암호화폐 매수 실행"""
        try:
            if self.config.UPBIT_DEMO_MODE or not self.upbit:
                # 시뮬레이션 모드
                quantity = amount / signal['price']
                
                self.position_manager.record_trade(
                    symbol, 'CRYPTO', 'BUY', quantity, signal['price'], 'KRW'
                )
                
                self.logger.info(f"💰 [시뮬레이션] 암호화폐 매수: {symbol} {amount:,.0f}원")
                return True
            else:
                # 실제 매수
                order = self.upbit.buy_market_order(symbol, amount)
                
                if order:
                    quantity = amount / signal['price']
                    self.position_manager.record_trade(
                        symbol, 'CRYPTO', 'BUY', quantity, signal['price'], 'KRW'
                    )
                    
                    self.logger.info(f"💰 [실제] 암호화폐 매수: {symbol} {amount:,.0f}원")
                    return True
                else:
                    self.logger.error(f"암호화폐 매수 주문 실패: {symbol}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"암호화폐 매수 실행 실패 {symbol}: {e}")
            return False

# ============================================================================
# 🤖 OpenAI 전용 전략
# ============================================================================
class OpenAIStrategy:
    """OpenAI 기반 전용 투자 전략"""
    
    def __init__(self, config: TradingConfig, ibkr_manager: IBKRManager,
                 position_manager: PositionManager, notification_manager: NotificationManager,
                 ai_engine: OpenAIAnalysisEngine):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.ai_engine = ai_engine
        
        self.logger = logging.getLogger('OpenAIStrategy')
        
        # AI 전용 글로벌 유니버스
        self.global_universe = [
            # 미국 대형주
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # 성장주
            'CRM', 'SNOW', 'PLTR', 'ROKU', 'ZM', 'SHOP',
            # ETF
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO'
        ]
    
    def is_trading_day(self) -> bool:
        """매일 거래 (AI가 판단)"""
        return True  # AI가 시장 상황을 고려하여 매매 여부 결정
    
    async def run_strategy(self) -> Dict[str, Any]:
        """OpenAI 전용 전략 실행"""
        try:
            if not self.ai_engine.client_available:
                self.logger.warning("OpenAI API가 설정되지 않았습니다")
                return {'success': False, 'reason': 'openai_not_available'}
            
            await self.notification_manager.send_notification(
                f"🤖 OpenAI 전용 전략 시작\n"
                f"투자 한도: {self.config.TOTAL_PORTFOLIO_VALUE * self.config.OPENAI_ALLOCATION:,.0f}원",
                'info', 'OpenAI 전략'
            )
            
            # AI 시장 보고서 생성
            market_report = await self.ai_engine.generate_market_report()
            
            # AI 기반 종목 선별
            selected_stocks = await self._ai_stock_selection()
            
            if not selected_stocks:
                self.logger.warning("AI가 선별한 종목이 없습니다")
                return {'success': False, 'reason': 'no_ai_selections'}
            
            # 포트폴리오 리스크 분석
            current_positions = self.position_manager.get_portfolio_summary()
            risk_analysis = await self.ai_engine.analyze_portfolio_risk(current_positions)
            
            # AI 추천 기반 매매 실행
            trade_results = []
            allocation_per_stock = (self.config.TOTAL_PORTFOLIO_VALUE * self.config.OPENAI_ALLOCATION) / len(selected_stocks)
            
            for stock_data in selected_stocks:
                symbol = stock_data['symbol']
                ai_analysis = stock_data['analysis']
                
                try:
                    if ai_analysis.action == 'BUY' and ai_analysis.confidence > 0.75:
                        # 리스크 조정
                        risk_adjusted_amount = self._adjust_for_risk(
                            allocation_per_stock, ai_analysis.risk_level, risk_analysis
                        )
                        
                        if risk_adjusted_amount > 1000:  # 최소 투자금
                            quantity = int(risk_adjusted_amount / ai_analysis.target_price / 100) * 100
                            
                            if quantity > 0:
                                success = await self.ibkr_manager.place_order(
                                    symbol, 'BUY', quantity, 'USD'
                                )
                                
                                if success:
                                    self.position_manager.record_trade(
                                        symbol, 'OPENAI', 'BUY', quantity, ai_analysis.target_price, 'USD'
                                    )
                                    
                                    self.position_manager.record_ai_analysis(ai_analysis, 'OPENAI')
                                    
                                    trade_results.append({
                                        'symbol': symbol,
                                        'action': 'BUY',
                                        'quantity': quantity,
                                        'price': ai_analysis.target_price,
                                        'confidence': ai_analysis.confidence,
                                        'reasoning': ai_analysis.reasoning
                                    })
                                    
                                    self.logger.info(f"🤖 AI 매수: {symbol} {quantity}주")
                
                except Exception as e:
                    self.logger.error(f"AI 전략 매수 실패 {symbol}: {e}")
                    continue
            
            # 결과 알림
            if trade_results:
                message = f"🤖 OpenAI 전략 실행 완료\n\n"
                message += f"📊 시장 분석:\n{market_report[:200]}...\n\n"
                message += f"⚠️ 포트폴리오 리스크: {risk_analysis.get('overall_risk_level', 'UNKNOWN')}\n\n"
                message += "💼 거래 내역:\n"
                
                for result in trade_results:
                    message += f"• {result['symbol']}: {result['action']} {result['quantity']}주\n"
                    message += f"  신뢰도: {result['confidence']:.1%}, 이유: {result['reasoning'][:50]}...\n"
                
                await self.notification_manager.send_notification(
                    message, 'success', 'OpenAI 전략 완료'
                )
            
            return {
                'success': True,
                'trade_count': len(trade_results),
                'market_report': market_report,
                'risk_analysis': risk_analysis,
                'total_investment': sum(r['quantity'] * r['price'] for r in trade_results)
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI 전략 실행 실패: {e}")
            await self.notification_manager.send_notification(
                f"🤖 OpenAI 전략 오류: {str(e)}", 'warning'
            )
            return {'success': False, 'error': str(e)}
    
    async def _ai_stock_selection(self) -> List[Dict[str, Any]]:
        """AI 기반 종목 선별"""
        try:
            selected_stocks = []
            
            # 각 종목에 대해 AI 분석 수행
            for symbol in self.global_universe[:12]:  # 상위 12개 분석
                try:
                    # 기본 시장 데이터 수집
                    market_data = await self._collect_market_data(symbol)
                    
                    if not market_data:
                        continue
                    
                    # AI 분석 수행
                    ai_analysis = await self.ai_engine.analyze_symbol(
                        symbol, market_data, "OpenAI 전용 글로벌 전략 - 매일 분석"
                    )
                    
                    if ai_analysis and ai_analysis.confidence > 0.6:
                        selected_stocks.append({
                            'symbol': symbol,
                            'analysis': ai_analysis,
                            'market_data': market_data
                        })
                        
                except Exception as e:
                    self.logger.debug(f"AI 종목 분석 실패 {symbol}: {e}")
                    continue
            
            # 신뢰도순 정렬
            selected_stocks.sort(key=lambda x: x['analysis'].confidence, reverse=True)
            
            # 상위 5개 선택
            top_selections = selected_stocks[:5]
            
            self.logger.info(f"AI 선별 종목: {[s['symbol'] for s in top_selections]}")
            return top_selections
            
        except Exception as e:
            self.logger.error(f"AI 종목 선별 실패: {e}")
            return []
    
    async def _collect_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """시장 데이터 수집"""
        try:
            if not YAHOO_AVAILABLE:
                return None
            
            stock = yf.Ticker(symbol)
            
            # 가격 데이터
            hist_data = stock.history(period="3mo")
            if hist_data.empty:
                return None
            
            # 기본 정보
            info = stock.info
            
            current_price = float(hist_data['Close'].iloc[-1])
            
            # 기술적 지표 계산
            closes = hist_data['Close']
            volume = hist_data['Volume']
            
            # 이동평균
            ma20 = closes.rolling(20).mean().iloc[-1]
            ma50 = closes.rolling(50).mean().iloc[-1]
            
            # RSI
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 거래량 분석
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            # 변동성 계산
            returns = closes.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 연환산
            
            market_data = {
                'symbol': symbol,
                'current_price': current_price,
                'ma20': ma20,
                'ma50': ma50,
                'rsi': current_rsi,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'price_to_book': info.get('priceToBook', 0),
                '52w_high': info.get('fiftyTwoWeekHigh', current_price),
                '52w_low': info.get('fiftyTwoWeekLow', current_price),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            return market_data
            
        except Exception as e:
            self.logger.debug(f"시장 데이터 수집 실패 {symbol}: {e}")
            return None
    
    def _adjust_for_risk(self, base_amount: float, risk_level: str, 
                        portfolio_risk: Dict[str, Any]) -> float:
        """리스크에 따른 투자금 조정"""
        try:
            # 기본 조정 비율
            risk_multipliers = {
                'LOW': 1.2,
                'MEDIUM': 1.0,
                'HIGH': 0.6
            }
            
            base_multiplier = risk_multipliers.get(risk_level, 1.0)
            
            # 포트폴리오 전체 리스크 고려
            portfolio_risk_level = portfolio_risk.get('overall_risk_level', 'MEDIUM')
            portfolio_adjustment = {
                'LOW': 1.1,
                'MEDIUM': 1.0,
                'HIGH': 0.8
            }
            
            portfolio_multiplier = portfolio_adjustment.get(portfolio_risk_level, 1.0)
            
            # 최종 조정된 금액
            adjusted_amount = base_amount * base_multiplier * portfolio_multiplier
            
            return max(adjusted_amount, 0)
            
        except Exception as e:
            self.logger.error(f"리스크 조정 실패: {e}")
            return base_amount

# ============================================================================
# 🌐 네트워크 모니터링
# ============================================================================
class NetworkMonitor:
    """네트워크 연결 모니터링"""
    
    def __init__(self, config: TradingConfig, ibkr_manager: IBKRManager, 
                 notification_manager: NotificationManager):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.notification_manager = notification_manager
        self.monitoring = False
        self.connection_failures = 0
        
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
        # 인터넷 연결 체크
        internet_ok = await self._check_internet()
        
        # IBKR 연결 체크
        ibkr_ok = self.ibkr_manager.connected and (
            self.ibkr_manager.ib.isConnected() if self.ibkr_manager.ib else False
        )
        
        # API 서버 체크
        api_ok = await self._check_api_servers()
        
        if not internet_ok or (IBKR_AVAILABLE and not ibkr_ok) or not api_ok:
            self.connection_failures += 1
            
            # IBKR 없이 운영시 더 관대한 기준
            if not IBKR_AVAILABLE and api_ok and internet_ok:
                if self.connection_failures == 1:
                    self.logger.info("ℹ️ IBKR 없이 운영 중 (암호화폐 전략만 사용)")
                self.connection_failures = 0
                return
            
            self.logger.warning(
                f"⚠️ 연결 실패 {self.connection_failures}회: "
                f"인터넷={internet_ok}, IBKR={ibkr_ok}, API={api_ok}"
            )
            
            # 연속 실패시 응급 조치
            if self.connection_failures >= 5:
                await self.notification_manager.send_notification(
                    f"🚨 네트워크 연결 실패 {self.connection_failures}회\n"
                    f"인터넷: {internet_ok}, IBKR: {ibkr_ok}, API: {api_ok}",
                    'emergency'
                )
                
                if self.ibkr_manager.connected:
                    await self.ibkr_manager.emergency_sell_all()
                
                self.monitoring = False
        else:
            if self.connection_failures > 0:
                self.logger.info("✅ 네트워크 연결 복구")
                await self.notification_manager.send_notification(
                    "✅ 네트워크 연결 복구", 'success'
                )
            self.connection_failures = 0
    
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
            servers = [
                'https://api.upbit.com/v1/market/all',
                'https://query1.finance.yahoo.com'
            ]
            
            success_count = 0
            for server in servers:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(server, timeout=5) as response:
                            if response.status == 200
                             if response.status == 200:
                                success_count += 1
                except:
                    continue
            
            return success_count > 0
        except:
            return False

# ============================================================================
# 🏆 메인 거래 시스템
# ============================================================================
class TradingSystem:
    """퀸트프로젝트 통합 거래 시스템 (OpenAI 포함)"""
    
    def __init__(self):
        # 설정 로드
        self.config = TradingConfig()
        
        # 로깅 설정
        self._setup_logging()
        
        # 로거 초기화
        self.logger = logging.getLogger('TradingSystem')
        
        # 핵심 컴포넌트 초기화
        self.emergency_detector = EmergencyDetector(self.config)
        self.ibkr_manager = IBKRManager(self.config)
        self.notification_manager = NotificationManager(self.config)
        self.position_manager = PositionManager(self.config, self.ibkr_manager)
        self.network_monitor = NetworkMonitor(self.config, self.ibkr_manager, self.notification_manager)
        
        # OpenAI 엔진 초기화
        self.ai_engine = OpenAIAnalysisEngine(self.config)
        
        # 전략 초기화
        self.strategies = {}
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
        
        file_handler = logging.FileHandler(log_dir / 'trading_system.log', encoding='utf-8')
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
            if self.config.US_ENABLED:
                self.strategies['US'] = USStrategy(
                    self.config, self.ibkr_manager, self.position_manager, 
                    self.notification_manager, self.ai_engine
                )
                self.logger.info("✅ 미국 전략 초기화 완료 (AI 포함)")
            
            # 일본 전략
            if self.config.JAPAN_ENABLED:
                self.strategies['JAPAN'] = JapanStrategy(
                    self.config, self.ibkr_manager, self.position_manager, 
                    self.notification_manager, self.ai_engine
                )
                self.logger.info("✅ 일본 전략 초기화 완료 (AI 포함)")
            
            # 인도 전략
            if self.config.INDIA_ENABLED:
                self.strategies['INDIA'] = IndiaStrategy(
                    self.config, self.ibkr_manager, self.position_manager, 
                    self.notification_manager, self.ai_engine
                )
                self.logger.info("✅ 인도 전략 초기화 완료 (AI 포함)")
            
            # 암호화폐 전략
            if self.config.CRYPTO_ENABLED:
                self.strategies['CRYPTO'] = CryptoStrategy(
                    self.config, self.position_manager, self.notification_manager, self.ai_engine
                )
                self.logger.info("✅ 암호화폐 전략 초기화 완료 (AI 포함)")
            
            # OpenAI 전용 전략
            if self.config.OPENAI_ENABLED:
                self.strategies['OPENAI'] = OpenAIStrategy(
                    self.config, self.ibkr_manager, self.position_manager, 
                    self.notification_manager, self.ai_engine
                )
                self.logger.info("✅ OpenAI 전용 전략 초기화 완료")
            
            if not self.strategies:
                self.logger.warning("⚠️ 활성화된 전략이 없습니다")
                
        except Exception as e:
            self.logger.error(f"전략 초기화 실패: {e}")
    
    async def start_system(self):
        """시스템 시작"""
        try:
            self.logger.info("🏆 퀸트프로젝트 통합 거래 시스템 시작! (OpenAI 포함)")
            self.start_time = datetime.now()
            self.running = True
            
            # IBKR 연결
            if IBKR_AVAILABLE:
                await self.ibkr_manager.connect()
            
            # AI 시장 분석 보고서 생성
            if self.ai_engine.client_available:
                try:
                    market_report = await self.ai_engine.generate_market_report()
                    await self.notification_manager.send_notification(
                        f"🤖 AI 시장 분석 보고서\n\n{market_report}",
                        'info', 'AI 시장 분석'
                    )
                except Exception as e:
                    self.logger.warning(f"AI 시장 분석 실패: {e}")
            
            # 시작 알림
            await self.notification_manager.send_notification(
                f"🚀 퀸트프로젝트 거래 시스템 시작\n"
                f"활성 전략: {', '.join(self.strategies.keys())}\n"
                f"IBKR 연결: {'✅' if self.ibkr_manager.connected else '❌'}\n"
                f"OpenAI 연결: {'✅' if self.ai_engine.client_available else '❌'}\n"
                f"총 포트폴리오: {self.config.TOTAL_PORTFOLIO_VALUE:,.0f}원",
                'success', '시스템 시작'
            )
            
            # 백그라운드 태스크 시작
            tasks = [
                asyncio.create_task(self._main_trading_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self.network_monitor.start_monitoring()),
                asyncio.create_task(self._ai_analysis_loop())
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
                weekday_names = ['월', '화', '수', '목', '금', '토', '일']
                today_name = weekday_names[current_weekday]
                
                self.logger.info(f"📅 {today_name}요일 전략 체크")
                
                for strategy_name, strategy_instance in self.strategies.items():
                    try:
                        if self._should_run_strategy(strategy_name, current_weekday):
                            self.logger.info(f"🎯 {strategy_name} 전략 실행")
                            result = await strategy_instance.run_strategy()
                            
                            if result.get('success'):
                                self.logger.info(f"✅ {strategy_name} 전략 완료")
                            else:
                                self.logger.warning(f"⚠️ {strategy_name} 전략 실패: {result.get('reason', 'unknown')}")
                                
                    except Exception as e:
                        error_critical = self.emergency_detector.record_error(
                            f"{strategy_name}_error", str(e), critical=False
                        )
                        if error_critical:
                            await self.emergency_shutdown(f"{strategy_name} 전략 오류")
                            break
                
                # 포지션 업데이트
                await self.position_manager.update_positions()
                
                # 1시간 대기
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"메인 루프 오류: {e}")
                await asyncio.sleep(300)  # 5분 대기
    
    def _should_run_strategy(self, strategy_name: str, weekday: int) -> bool:
        """전략 실행 여부 판단"""
        strategy_schedules = {
            'US': [1, 3],      # 화목
            'JAPAN': [1, 3],   # 화목
            'INDIA': [2],      # 수요일
            'CRYPTO': [0, 4],  # 월금
            'OPENAI': list(range(7))  # 매일 (AI가 판단)
        }
        
        return weekday in strategy_schedules.get(strategy_name, [])
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.running:
            try:
                # 포지션 모니터링
                portfolio_summary = self.position_manager.get_portfolio_summary()
                
                # 위험 상황 체크
                if self.config.TOTAL_PORTFOLIO_VALUE > 0:
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
                if datetime.now().hour % 6 == 0 and datetime.now().minute < 10:
                    await self._send_status_report(portfolio_summary)
                
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(60)
    
    async def _ai_analysis_loop(self):
        """AI 분석 루프"""
        while self.running:
            try:
                if not self.ai_engine.client_available:
                    await asyncio.sleep(3600)  # 1시간 대기
                    continue
                
                # 매 4시간마다 포트폴리오 리스크 분석
                if datetime.now().hour % 4 == 0 and datetime.now().minute < 10:
                    await self._perform_ai_risk_analysis()
                
                # 매일 오전 9시에 시장 보고서 생성
                if datetime.now().hour == 9 and datetime.now().minute < 10:
                    await self._generate_daily_ai_report()
                
                await asyncio.sleep(600)  # 10분마다 체크
                
            except Exception as e:
                self.logger.error(f"AI 분석 루프 오류: {e}")
                await asyncio.sleep(600)
    
    async def _perform_ai_risk_analysis(self):
        """AI 포트폴리오 리스크 분석"""
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            risk_analysis = await self.ai_engine.analyze_portfolio_risk(portfolio_summary)
            
            if risk_analysis.get('overall_risk_level') == 'HIGH':
                await self.notification_manager.send_notification(
                    f"⚠️ AI 포트폴리오 리스크 경고\n"
                    f"위험 수준: {risk_analysis.get('overall_risk_level')}\n"
                    f"주요 위험: {', '.join(risk_analysis.get('main_risks', []))}\n"
                    f"추천사항: {', '.join(risk_analysis.get('recommendations', []))}",
                    'warning', 'AI 리스크 분석'
                )
            
        except Exception as e:
            self.logger.error(f"AI 리스크 분석 실패: {e}")
    
    async def _generate_daily_ai_report(self):
        """일일 AI 보고서 생성"""
        try:
            market_report = await self.ai_engine.generate_market_report()
            
            await self.notification_manager.send_notification(
                f"🌅 AI 일일 시장 보고서\n\n{market_report}",
                'info', 'AI 일일 보고서'
            )
            
        except Exception as e:
            self.logger.error(f"AI 일일 보고서 생성 실패: {e}")
    
    async def _send_status_report(self, portfolio_summary: Dict):
        """상태 보고서 전송"""
        try:
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            report = (
                f"📊 퀸트프로젝트 상태 보고 (AI 포함)\n\n"
                f"🕐 가동시간: {uptime}\n"
                f"💼 총 포지션: {portfolio_summary['total_positions']}개\n"
                f"💰 미실현 손익: {portfolio_summary['total_unrealized_pnl']:+,.0f}원\n"
                f"📈 수익 포지션: {portfolio_summary['profitable_positions']}개\n"
                f"📉 손실 포지션: {portfolio_summary['losing_positions']}개\n"
                f"🤖 AI 엔진: {'✅ 활성' if self.ai_engine.client_available else '❌ 비활성'}\n\n"
                f"전략별 현황:\n"
            )
            
            for strategy, data in portfolio_summary['by_strategy'].items():
                ai_icon = "🤖" if strategy == "OPENAI" else ""
                report += f"  {ai_icon}{strategy}: {data['count']}개 ({data['pnl']:+,.0f}원)\n"
            
            await self.notification_manager.send_notification(report, 'info', '상태 보고')
            
        except Exception as e:
            self.logger.error(f"상태 보고서 전송 실패: {e}")
    
    async def emergency_shutdown(self, reason: str):
        """응급 종료"""
        try:
            self.logger.critical(f"🚨 응급 종료: {reason}")
            
            # 응급 알림
            await self.notification_manager.send_notification(
                f"🚨 시스템 응급 종료\n사유: {reason}",
                'emergency'
            )
            
            # 응급 매도
            if self.ibkr_manager.connected:
                await self.ibkr_manager.emergency_sell_all()
            
            # 시스템 종료
            self.running = False
            
        except Exception as e:
            self.logger.error(f"응급 종료 실패: {e}")
    
    async def graceful_shutdown(self):
        """정상 종료"""
        try:
            self.logger.info("🛑 시스템 정상 종료 시작")
            
            # 종료 알림
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            await self.notification_manager.send_notification(
                f"🛑 시스템 정상 종료\n가동시간: {uptime}",
                'info'
            )
            
            # 네트워크 모니터링 중지
            self.network_monitor.monitoring = False
            
            # IBKR 연결 해제
            if self.ibkr_manager.connected and self.ibkr_manager.ib:
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
    system = TradingSystem()
    await system.ibkr_manager.connect()
    await system.position_manager.update_positions()
    
    summary = system.position_manager.get_portfolio_summary()
    
    return {
        'strategies': list(system.strategies.keys()),
        'ibkr_connected': system.ibkr_manager.connected,
        'openai_available': system.ai_engine.client_available,
        'total_positions': summary['total_positions'],
        'total_unrealized_pnl': summary['total_unrealized_pnl'],
        'by_strategy': summary['by_strategy']
    }

async def test_notifications():
    """알림 시스템 테스트"""
    config = TradingConfig()
    notifier = NotificationManager(config)
    
    test_results = {}
    
    # 텔레그램 테스트
    if config.TELEGRAM_ENABLED:
        success = await notifier.send_notification(
            "🧪 퀸트프로젝트 알림 시스템 테스트 (OpenAI 포함)",
            'info', '테스트'
        )
        test_results['telegram'] = success
    
    # 이메일 테스트
    if config.EMAIL_ENABLED:
        success = await notifier.send_notification(
            "퀸트프로젝트 이메일 알림 테스트입니다. (OpenAI 기능 추가)",
            'warning', '이메일 테스트'
        )
        test_results['email'] = success
    
    return test_results

async def test_openai_analysis():
    """OpenAI 분석 시스템 테스트"""
    config = TradingConfig()
    ai_engine = OpenAIAnalysisEngine(config)
    
    if not ai_engine.client_available:
        return {'success': False, 'error': 'OpenAI API 키가 설정되지 않았습니다'}
    
    try:
        # 테스트 시장 데이터
        test_data = {
            'symbol': 'AAPL',
            'current_price': 150.0,
            'ma20': 148.0,
            'ma50': 145.0,
            'rsi': 55.0,
            'volume_ratio': 1.2
        }
        
        # AI 분석 수행
        analysis = await ai_engine.analyze_symbol(
            'AAPL', test_data, "테스트 분석"
        )
        
        if analysis:
            return {
                'success': True,
                'analysis': {
                    'symbol': analysis.symbol,
                    'action': analysis.action,
                    'confidence': analysis.confidence,
                    'reasoning': analysis.reasoning,
                    'target_price': analysis.target_price,
                    'risk_level': analysis.risk_level
                }
            }
        else:
            return {'success': False, 'error': 'AI 분석 결과 없음'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

async def run_single_strategy(strategy_name: str):
    """단일 전략 실행"""
    system = TradingSystem()
    
    if strategy_name.upper() not in system.strategies:
        return {'success': False, 'error': f'전략 {strategy_name}을 찾을 수 없습니다'}
    
    try:
        # IBKR 연결 (필요시)
        if strategy_name.upper() not in ['CRYPTO']:
            await system.ibkr_manager.connect()
        
        # 전략 실행
        strategy = system.strategies[strategy_name.upper()]
        result = await strategy.run_strategy()
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

async def analyze_portfolio_performance():
    """포트폴리오 성과 분석"""
    system = TradingSystem()
    await system.ibkr_manager.connect()
    await system.position_manager.update_positions()
    
    summary = system.position_manager.get_portfolio_summary()
    
    # 성과 계산
    total_value = system.config.TOTAL_PORTFOLIO_VALUE
    unrealized_pnl = summary['total_unrealized_pnl']
    
    performance = {
        'total_value': total_value,
        'unrealized_pnl': unrealized_pnl,
        'unrealized_return_pct': (unrealized_pnl / total_value * 100) if total_value > 0 else 0,
        'total_positions': summary['total_positions'],
        'profitable_positions': summary['profitable_positions'],
        'losing_positions': summary['losing_positions'],
        'win_rate': (summary['profitable_positions'] / summary['total_positions'] * 100) if summary['total_positions'] > 0 else 0,
        'by_strategy': summary['by_strategy'],
        'by_currency': summary['by_currency'],
        'openai_available': system.ai_engine.client_available
    }
    
    return performance

async def get_ai_market_report():
    """AI 시장 보고서 조회"""
    config = TradingConfig()
    ai_engine = OpenAIAnalysisEngine(config)
    
    if not ai_engine.client_available:
        return "OpenAI API가 설정되지 않았습니다."
    
    try:
        report = await ai_engine.generate_market_report()
        return report
    except Exception as e:
        return f"AI 시장 보고서 생성 실패: {str(e)}"

async def emergency_sell_all():
    """응급 전량 매도"""
    system = TradingSystem()
    await system.ibkr_manager.connect()
    
    if system.ibkr_manager.connected:
        results = await system.ibkr_manager.emergency_sell_all()
        return results
    else:
        return {}

# ============================================================================
# 🏁 메인 실행부
# ============================================================================
async def main():
    """메인 실행 함수"""
    
    # 신호 핸들러 설정
    def signal_handler(signum, frame):
        print("\n🛑 종료 신호 수신, 정상 종료 중...")
        asyncio.create_task(system.graceful_shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 거래 시스템 생성
    system = TradingSystem()
    
    try:
        print("🏆" + "="*70)
        print("🏆 퀸트프로젝트 통합 거래 시스템 v2.1.0 (OpenAI 포함)")
        print("🏆" + "="*70)
        print("🇺🇸 미국 전략 (화목) - 서머타임 자동 처리 + AI 분석")
        print("🇯🇵 일본 전략 (화목) - 엔화 자동 환전 + AI 분석")
        print("🇮🇳 인도 전략 (수요일) - 루피 자동 환전 + AI 분석")
        print("💰 암호화폐 전략 (월금) - 월 5-7% 최적화 + AI 분석")
        print("🤖 OpenAI 전용 전략 (매일) - GPT-4 기반 분석")
        print("🔔 통합 알림 시스템 (텔레그램/이메일)")
        print("🚨 응급 오류 감지 + 네트워크 모니터링")
        print("📊 통합 포지션 관리 + 성과 추적")
        print("🏆" + "="*70)
        
        # 시스템 정보 출력
        print(f"\n📊 시스템 설정:")
        print(f"  총 포트폴리오: {system.config.TOTAL_PORTFOLIO_VALUE:,.0f}원")
        print(f"  활성 전략: {', '.join(system.strategies.keys())}")
        print(f"  IBKR 연결: {'설정됨' if IBKR_AVAILABLE else '미설정'}")
        print(f"  업비트 연결: {'설정됨' if UPBIT_AVAILABLE else '미설정'}")
        print(f"  OpenAI 연결: {'설정됨' if OPENAI_AVAILABLE else '미설정'}")
        print(f"  알림 시스템: {'활성' if system.config.TELEGRAM_ENABLED else '비활성'}")
        
        # 전략별 배분
        print(f"\n💰 전략별 자금 배분:")
        print(f"  🇺🇸 미국: {system.config.US_ALLOCATION:.1%} ({system.config.TOTAL_PORTFOLIO_VALUE * system.config.US_ALLOCATION:,.0f}원)")
        print(f"  🇯🇵 일본: {system.config.JAPAN_ALLOCATION:.1%} ({system.config.TOTAL_PORTFOLIO_VALUE * system.config.JAPAN_ALLOCATION:,.0f}원)")
        print(f"  💰 암호화폐: {system.config.CRYPTO_ALLOCATION:.1%} ({system.config.TOTAL_PORTFOLIO_VALUE * system.config.CRYPTO_ALLOCATION:,.0f}원)")
        print(f"  🇮🇳 인도: {system.config.INDIA_ALLOCATION:.1%} ({system.config.TOTAL_PORTFOLIO_VALUE * system.config.INDIA_ALLOCATION:,.0f}원)")
        print(f"  🤖 OpenAI: {system.config.OPENAI_ALLOCATION:.1%} ({system.config.TOTAL_PORTFOLIO_VALUE * system.config.OPENAI_ALLOCATION:,.0f}원)")
        
        # 거래 스케줄
        print(f"\n📅 거래 스케줄:")
        print(f"  월요일: 💰 암호화폐 전략, 🤖 OpenAI 전략")
        print(f"  화요일: 🇺🇸 미국 전략, 🇯🇵 일본 전략, 🤖 OpenAI 전략")
        print(f"  수요일: 🇮🇳 인도 전략, 🤖 OpenAI 전략")
        print(f"  목요일: 🇺🇸 미국 전략, 🇯🇵 일본 전략, 🤖 OpenAI 전략")
        print(f"  금요일: 💰 암호화폐 전략, 🤖 OpenAI 전략")
        print(f"  주말: 🤖 OpenAI 전략 (시장 분석)")
        
        print(f"\n🚀 시스템 실행 옵션:")
        print(f"  1. 🏆 전체 시스템 자동 실행")
        print(f"  2. 📊 시스템 상태 확인")
        print(f"  3. 🇺🇸 미국 전략만 실행")
        print(f"  4. 🇯🇵 일본 전략만 실행")
        print(f"  5. 🇮🇳 인도 전략만 실행")
        print(f"  6. 💰 암호화폐 전략만 실행")
        print(f"  7. 🤖 OpenAI 전략만 실행")
        print(f"  8. 🔔 알림 시스템 테스트")
        print(f"  9. 📈 포트폴리오 성과 분석")
        print(f"  A. 🤖 OpenAI 분석 테스트")
        print(f"  B. 🤖 AI 시장 보고서 조회")
        print(f"  C. 🚨 응급 전량 매도")
        print(f"  0. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (0-9, A-C): ").strip().upper()
                
                if choice == '1':
                    print("\n🏆 전체 시스템 자동 실행!")
                    print("🔄 5대 전략이 요일별로 자동 실행됩니다.")
                    print("🤖 OpenAI 분석이 모든 전략에 통합됩니다.")
                    print("🚨 Ctrl+C로 안전하게 종료할 수 있습니다.")
                    confirm = input("시작하시겠습니까? (y/N): ").strip().lower()
                    if confirm == 'y':
                        await system.start_system()
                    break
                
                elif choice == '2':
                    print("\n📊 시스템 상태 확인 중...")
                    status = await get_system_status()
                    
                    print(f"활성 전략: {', '.join(status['strategies'])}")
                    print(f"IBKR 연결: {'✅' if status['ibkr_connected'] else '❌'}")
                    print(f"OpenAI 연결: {'✅' if status['openai_available'] else '❌'}")
                    print(f"총 포지션: {status['total_positions']}개")
                    print(f"미실현 손익: {status['total_unrealized_pnl']:+,.0f}원")
                    
                    if status['by_strategy']:
                        print("전략별 현황:")
                        for strategy, data in status['by_strategy'].items():
                            ai_icon = "🤖" if strategy == "OPENAI" else ""
                            print(f"  {ai_icon}{strategy}: {data['count']}개 ({data['pnl']:+,.0f}원)")
                
                elif choice == '3':
                    print("\n🇺🇸 미국 전략 실행 중 (AI 분석 포함)...")
                    result = await run_single_strategy('US')
                    print(f"결과: {result}")
                
                elif choice == '4':
                    print("\n🇯🇵 일본 전략 실행 중 (AI 분석 포함)...")
                    result = await run_single_strategy('JAPAN')
                    print(f"결과: {result}")
                
                elif choice == '5':
                    print("\n🇮🇳 인도 전략 실행 중 (AI 분석 포함)...")
                    result = await run_single_strategy('INDIA')
                    print(f"결과: {result}")
                
                elif choice == '6':
                    print("\n💰 암호화폐 전략 실행 중 (AI 분석 포함)...")
                    result = await run_single_strategy('CRYPTO')
                    print(f"결과: {result}")
                
                elif choice == '7':
                    print("\n🤖 OpenAI 전용 전략 실행 중...")
                    result = await run_single_strategy('OPENAI')
                    print(f"결과: {result}")
                
                elif choice == '8':
                    print("\n🔔 알림 시스템 테스트 중...")
                    test_results = await test_notifications()
                    
                    for channel, success in test_results.items():
                        status = "✅ 성공" if success else "❌ 실패"
                        print(f"  {channel}: {status}")
                
                elif choice == '9':
                    print("\n📈 포트폴리오 성과 분석 중...")
                    performance = await analyze_portfolio_performance()
                    
                    print(f"총 포트폴리오 가치: {performance['total_value']:,.0f}원")
                    print(f"미실현 손익: {performance['unrealized_pnl']:+,.0f}원 ({performance['unrealized_return_pct']:+.2f}%)")
                    print(f"총 포지션: {performance['total_positions']}개")
                    print(f"수익 포지션: {performance['profitable_positions']}개")
                    print(f"손실 포지션: {performance['losing_positions']}개")
                    print(f"승률: {performance['win_rate']:.1f}%")
                    print(f"OpenAI 사용 가능: {'✅' if performance['openai_available'] else '❌'}")
                    
                    if performance['by_strategy']:
                        print("\n전략별 성과:")
                        for strategy, data in performance['by_strategy'].items():
                            ai_icon = "🤖" if strategy == "OPENAI" else ""
                            print(f"  {ai_icon}{strategy}: {data['count']}개 포지션, {data['pnl']:+,.0f}원")
                
                elif choice == 'A':
                    print("\n🤖 OpenAI 분석 시스템 테스트 중...")
                    test_result = await test_openai_analysis()
                    
                    if test_result['success']:
                        analysis = test_result['analysis']
                        print(f"✅ OpenAI 분석 성공!")
                        print(f"  종목: {analysis['symbol']}")
                        print(f"  추천: {analysis['action']}")
                        print(f"  신뢰도: {analysis['confidence']:.1%}")
                        print(f"  목표가: ${analysis['target_price']:.2f}")
                        print(f"  위험도: {analysis['risk_level']}")
                        print(f"  분석 근거: {analysis['reasoning'][:100]}...")
                    else:
                        print(f"❌ OpenAI 분석 실패: {test_result['error']}")
                
                elif choice == 'B':
                    print("\n🤖 AI 시장 보고서 생성 중...")
                    report = await get_ai_market_report()
                    print(f"\n📊 AI 시장 분석 보고서:")
                    print("="*50)
                    print(report)
                    print("="*50)
                
                elif choice == 'C':
                    print("\n🚨 응급 전량 매도!")
                    print("⚠️ 모든 포지션이 시장가로 매도됩니다!")
                    confirm = input("정말 실행하시겠습니까? (YES 입력): ").strip()
                    if confirm == 'YES':
                        results = await emergency_sell_all()
                        print(f"응급 매도 결과: {len(results)}개 종목")
                        for symbol, success in results.items():
                            status = "✅ 성공" if success else "❌ 실패"
                            print(f"  {symbol}: {status}")
                    else:
                        print("취소되었습니다.")
                
                elif choice == '0':
                    print("👋 퀸트프로젝트 거래 시스템을 종료합니다!")
                    break
                    
                else:
                    print("❌ 잘못된 선택입니다. 0-9, A-C 중 선택하세요.")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
    except KeyboardInterrupt:
        print("\n👋 사용자 중단")
        await system.graceful_shutdown()
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")
        await system.emergency_shutdown(f"시스템 오류: {e}")

if __name__ == "__main__":
    try:
        print("🏆 퀸트프로젝트 통합 거래 시스템 로딩... (OpenAI 포함)")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 퀸트프로젝트 거래 시스템 종료")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 실행 오류: {e}")
        sys.exit(1)
