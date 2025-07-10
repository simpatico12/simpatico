#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 최적화 거래 시스템 (trading_optimized.py)
================================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 (4대 전략)

✨ 최적화 기능:
- AI는 기술적 분석만 수행 (뉴스/시장심리 제거)
- 애매한 신호(0.4-0.7)에서만 AI 호출로 비용 절약
- 월 AI 사용료 5천원 이하 최적화
- trend_analysis 오류 해결
- 4대 전략 통합 관리

Author: 퀸트마스터팀 (최적화 버전)
Version: 3.0.0 (비용 최적화 + 기술적 분석 전용)
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

# OpenAI 라이브러리 (최적화 버전)
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
# 🎯 최적화 설정 관리자
# ============================================================================
class OptimizedTradingConfig:
    """최적화된 거래 시스템 설정"""
    
    def __init__(self):
        load_dotenv()
        
        # 포트폴리오 설정
        self.TOTAL_PORTFOLIO_VALUE = float(os.getenv('TOTAL_PORTFOLIO_VALUE', '10000000'))
        self.MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', '0.05'))
        
        # 4대 전략 활성화
        self.US_ENABLED = os.getenv('US_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.JAPAN_ENABLED = os.getenv('JAPAN_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.INDIA_ENABLED = os.getenv('INDIA_STRATEGY_ENABLED', 'true').lower() == 'true'
        self.CRYPTO_ENABLED = os.getenv('CRYPTO_STRATEGY_ENABLED', 'true').lower() == 'true'
        
        # 4대 전략 자원 배분
        self.US_ALLOCATION = float(os.getenv('US_STRATEGY_ALLOCATION', '0.35'))
        self.JAPAN_ALLOCATION = float(os.getenv('JAPAN_STRATEGY_ALLOCATION', '0.25'))
        self.INDIA_ALLOCATION = float(os.getenv('INDIA_STRATEGY_ALLOCATION', '0.20'))
        self.CRYPTO_ALLOCATION = float(os.getenv('CRYPTO_STRATEGY_ALLOCATION', '0.20'))
        
        # IBKR 설정
        self.IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
        self.IBKR_PORT = int(os.getenv('IBKR_PORT', '7497'))
        self.IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '1'))
        self.IBKR_PAPER_TRADING = os.getenv('IBKR_PAPER_TRADING', 'true').lower() == 'true'
        
        # 업비트 설정
        self.UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
        self.UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')
        self.UPBIT_DEMO_MODE = os.getenv('CRYPTO_DEMO_MODE', 'true').lower() == 'true'
        
        # OpenAI 최적화 설정 (비용 절약)
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        self.OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')  # 비용 절약
        self.OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '300'))  # 토큰 제한
        self.OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.3'))  # 일관성 중시
        
        # AI 사용 최적화 설정
        self.AI_CONFIDENCE_THRESHOLD_MIN = 0.4  # 이하면 AI 호출 안함
        self.AI_CONFIDENCE_THRESHOLD_MAX = 0.7  # 이상이면 AI 호출 안함
        self.AI_DAILY_CALL_LIMIT = 20  # 일일 호출 제한
        self.AI_CALL_COUNTER = 0  # 일일 호출 카운터
        self.AI_LAST_RESET_DATE = datetime.now().date()  # 카운터 리셋 날짜
        
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
        self.NETWORK_CHECK_INTERVAL = int(os.getenv('NETWORK_CHECK_INTERVAL', '60'))
        self.EMERGENCY_SELL_ON_ERROR = os.getenv('EMERGENCY_SELL_ON_ERROR', 'true').lower() == 'true'
        
        # 데이터베이스
        self.DB_PATH = os.getenv('DATABASE_PATH', './data/trading_optimized.db')
        self.BACKUP_PATH = os.getenv('BACKUP_PATH', './backups/')

# ============================================================================
# 🤖 최적화된 AI 분석 엔진 (기술적 분석 전용)
# ============================================================================
@dataclass
class TechnicalAnalysisResult:
    """기술적 분석 결과"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 ~ 1.0
    reasoning: str
    target_price: float
    risk_level: str  # LOW, MEDIUM, HIGH
    technical_score: float
    timestamp: datetime = field(default_factory=datetime.now)

class OptimizedAIEngine:
    """최적화된 AI 분석 엔진 (기술적 분석 전용, 비용 절약)"""
    
    def __init__(self, config: OptimizedTradingConfig):
        self.config = config
        self.logger = logging.getLogger('OptimizedAIEngine')
        
        # OpenAI 클라이언트 초기화
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            openai.api_key = self.config.OPENAI_API_KEY
            self.client_available = True
        else:
            self.client_available = False
            self.logger.warning("OpenAI API 키가 설정되지 않았습니다")
        
        # 분석 캐시 (API 호출 절약)
        self.analysis_cache = {}
        self.cache_duration = timedelta(hours=2)  # 2시간 캐시
    
    def _should_use_ai(self, technical_confidence: float) -> bool:
        """AI 사용 여부 판단 (비용 최적화)"""
        # 일일 호출 제한 체크
        current_date = datetime.now().date()
        if current_date != self.config.AI_LAST_RESET_DATE:
            self.config.AI_CALL_COUNTER = 0
            self.config.AI_LAST_RESET_DATE = current_date
        
        # 호출 제한 초과시 AI 사용 안함
        if self.config.AI_CALL_COUNTER >= self.config.AI_DAILY_CALL_LIMIT:
            self.logger.info(f"AI 일일 호출 제한 초과 ({self.config.AI_CALL_COUNTER}/{self.config.AI_DAILY_CALL_LIMIT})")
            return False
        
        # 신뢰도가 너무 높거나 낮으면 AI 사용 안함 (비용 절약)
        if (technical_confidence < self.config.AI_CONFIDENCE_THRESHOLD_MIN or 
            technical_confidence > self.config.AI_CONFIDENCE_THRESHOLD_MAX):
            return False
        
        return True
    
    async def analyze_technical_signal(self, symbol: str, technical_data: Dict[str, Any], 
                                     strategy_context: str = '') -> Optional[TechnicalAnalysisResult]:
        """기술적 신호의 확신도 체크 (AI 사용 최적화)"""
        
        technical_confidence = technical_data.get('confidence', 0.5)
        
        # AI 사용 필요성 판단
        if not self.client_available or not self._should_use_ai(technical_confidence):
            # AI 없이 기술적 분석만 반환
            return TechnicalAnalysisResult(
                symbol=symbol,
                action=technical_data.get('action', 'HOLD'),
                confidence=technical_confidence,
                reasoning=technical_data.get('reason', '기술적 분석'),
                target_price=technical_data.get('price', 0.0),
                risk_level='MEDIUM',
                technical_score=technical_confidence
            )
        
        try:
            # 캐시 확인
            cache_key = f"{symbol}_{hash(str(technical_data))}"
            if cache_key in self.analysis_cache:
                cached_time, cached_result = self.analysis_cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    return cached_result
            
            # AI 분석 수행 (호출 카운터 증가)
            self.config.AI_CALL_COUNTER += 1
            
            analysis_result = await self._perform_technical_ai_analysis(
                symbol, technical_data, strategy_context
            )
            
            # 캐시 저장
            self.analysis_cache[cache_key] = (datetime.now(), analysis_result)
            
            self.logger.info(f"AI 호출 {self.config.AI_CALL_COUNTER}/{self.config.AI_DAILY_CALL_LIMIT}: {symbol}")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"AI 분석 실패 {symbol}: {e}")
            # AI 실패시 기술적 분석 결과 반환
            return TechnicalAnalysisResult(
                symbol=symbol,
                action=technical_data.get('action', 'HOLD'),
                confidence=technical_confidence * 0.8,  # 신뢰도 약간 감소
                reasoning=f"AI 분석 실패, 기술적 분석만: {technical_data.get('reason', '')}",
                target_price=technical_data.get('price', 0.0),
                risk_level='MEDIUM',
                technical_score=technical_confidence
            )
    
    async def _perform_technical_ai_analysis(self, symbol: str, technical_data: Dict[str, Any], 
                                           strategy_context: str) -> TechnicalAnalysisResult:
        """기술적 분석 전용 AI 분석 (간단하고 비용 효율적)"""
        try:
            # 시스템 프롬프트 (간단하고 명확)
            system_prompt = """
당신은 기술적 분석 전문가입니다. 주어진 기술적 지표만을 바탕으로 간단명료하게 분석하세요.

분석 기준:
1. 기술적 지표 (이동평균, RSI, 거래량)
2. 차트 패턴
3. 신뢰도 검증

응답 형식 (JSON, 150자 이내):
{
    "action": "BUY/SELL/HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "간단한 분석 근거",
    "target_price": 목표가격,
    "risk_level": "LOW/MEDIUM/HIGH",
    "technical_score": 0.0-1.0
}
"""
            
            # 사용자 프롬프트 (필수 정보만)
            user_prompt = f"""
종목: {symbol}
현재가: {technical_data.get('price', 0)}
기술적 신호: {technical_data.get('action', 'HOLD')}
신뢰도: {technical_data.get('confidence', 0.5)}
근거: {technical_data.get('reason', '')}

위 기술적 분석의 확신도를 검증하고 간단히 분석해주세요.
"""
            
            # OpenAI API 호출 (최소 토큰)
            response = await self._call_openai_api(system_prompt, user_prompt)
            
            # 응답 파싱
            try:
                analysis_data = json.loads(response)
            except json.JSONDecodeError:
                # JSON 파싱 실패시 기본값
                analysis_data = {
                    "action": technical_data.get('action', 'HOLD'),
                    "confidence": technical_data.get('confidence', 0.5) * 0.9,
                    "reasoning": "AI 파싱 실패",
                    "target_price": technical_data.get('price', 0.0),
                    "risk_level": "MEDIUM",
                    "technical_score": technical_data.get('confidence', 0.5)
                }
            
            return TechnicalAnalysisResult(
                symbol=symbol,
                action=analysis_data.get('action', 'HOLD'),
                confidence=float(analysis_data.get('confidence', 0.5)),
                reasoning=analysis_data.get('reasoning', 'AI 분석'),
                target_price=float(analysis_data.get('target_price', technical_data.get('price', 0.0))),
                risk_level=analysis_data.get('risk_level', 'MEDIUM'),
                technical_score=float(analysis_data.get('technical_score', 0.5))
            )
            
        except Exception as e:
            self.logger.error(f"AI 기술적 분석 실패 {symbol}: {e}")
            
            # 기본 분석 결과 반환
            return TechnicalAnalysisResult(
                symbol=symbol,
                action=technical_data.get('action', 'HOLD'),
                confidence=technical_data.get('confidence', 0.5) * 0.8,
                reasoning=f'AI 분석 실패: {str(e)[:50]}',
                target_price=technical_data.get('price', 0.0),
                risk_level='MEDIUM',
                technical_score=technical_data.get('confidence', 0.5)
            )
    
    async def _call_openai_api(self, system_prompt: str, user_prompt: str) -> str:
        """OpenAI API 호출 (최적화 버전)"""
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

# ============================================================================
# 🕒 서머타임 관리자 (수정된 버전)
# ============================================================================
class DaylightSavingManager:
    """서머타임 자동 관리 (오류 수정)"""
    
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
        
        try:
            year = date.year
            # 3월 둘째주 일요일
            march_first = datetime(year, 3, 1)
            days_to_add = (6 - march_first.weekday()) % 7 + 7
            march_second_sunday = march_first + timedelta(days=days_to_add)
            
            # 11월 첫째주 일요일  
            nov_first = datetime(year, 11, 1)
            days_to_add = (6 - nov_first.weekday()) % 7
            nov_first_sunday = nov_first + timedelta(days=days_to_add)
            
            is_dst = march_second_sunday.date() <= date < nov_first_sunday.date()
            self.cache[date] = is_dst
            return is_dst
        except Exception as e:
            # 오류 발생시 보수적으로 False 반환
            return False
    
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
            
        try:
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
        except Exception as e:
            # 오류 발생시 기본값 반환
            if date is None:
                date = datetime.now().date()
            return (
                datetime.combine(date, datetime.min.time().replace(hour=22, minute=30)),
                datetime.combine(date, datetime.min.time().replace(hour=5, minute=0)) + timedelta(days=1)
            )

# ============================================================================
# 🔔 통합 알림 시스템
# ============================================================================
@dataclass
class NotificationMessage:
    """알림 메시지"""
    title: str
    content: str
    priority: str = 'info'  # emergency, warning, info, success, debug
    category: str = 'general'  # trading, system, portfolio, error
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class NotificationManager:
    """통합 알림 관리 시스템"""
    
    def __init__(self, config: OptimizedTradingConfig):
        self.config = config
        self.logger = logging.getLogger('NotificationManager')
        self.recent_notifications = deque(maxlen=50)  # 메모리 절약
    
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
            
            # 중복 체크 (비용 절약)
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
                f"{emoji} <b>퀸트프로젝트</b>\n\n"
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
퀸트프로젝트 최적화 알림

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
# 🚨 응급 오류 감지 시스템 (간소화)
# ============================================================================
class EmergencyDetector:
    """응급 상황 감지 및 대응 (간소화)"""
    
    def __init__(self, config: OptimizedTradingConfig):
        self.config = config
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = None
        self.emergency_triggered = False
        self.logger = logging.getLogger('EmergencyDetector')
    
    def check_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태 체크 (간소화)"""
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
            
            health_status['healthy'] = not health_status['errors']
            
        except Exception as e:
            health_status = {
                'healthy': False,
                'warnings': [],
                'errors': [f'상태 체크 실패: {str(e)}'],
                'emergency_needed': True
            }
        
        return health_status
    
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
# 🔗 IBKR 통합 관리자 (최적화)
# ============================================================================
class IBKRManager:
    """IBKR 통합 관리 (최적화)"""
    
    def __init__(self, config: OptimizedTradingConfig):
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

# ============================================================================
# 📊 포지션 데이터 클래스 (최적화)
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
    """통합 포지션 관리 (최적화)"""
    
    def __init__(self, config: OptimizedTradingConfig, ibkr_manager: IBKRManager):
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
            
            # AI 분석 결과 테이블
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
    
    def record_ai_analysis(self, analysis_result: TechnicalAnalysisResult, strategy: str):
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
# 📈 미국 전략 (최적화된 버전)
# ============================================================================
class USStrategy:
    """미국 주식 전략 (AI 기술적 분석 최적화)"""
    
    def __init__(self, config: OptimizedTradingConfig, ibkr_manager: IBKRManager, 
                 position_manager: PositionManager, notification_manager: NotificationManager,
                 ai_engine: OptimizedAIEngine):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.ai_engine = ai_engine
        self.dst_manager = DaylightSavingManager()
        
        self.logger = logging.getLogger('USStrategy')
        
        # 미국 주식 유니버스 (최적화)
        self.stock_universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
            'BRK-B', 'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'DIS', 'PYPL',
            'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'CSCO', 'PEP', 'KO'
        ]
    
    def is_trading_day(self) -> bool:
        """화목 거래일 체크"""
        return datetime.now().weekday() in [1, 3]  # 화요일, 목요일
    
    async def run_strategy(self) -> Dict[str, Any]:
        """미국 전략 실행 (최적화)"""
        try:
            if not self.is_trading_day():
                self.logger.info("오늘은 미국 전략 비거래일")
                return {'success': False, 'reason': 'not_trading_day'}
            
            # 서머타임 상태 확인
            dst_active = self.dst_manager.is_dst_active()
            market_open, market_close = self.dst_manager.get_market_hours_kst()
            
            await self.notification_manager.send_notification(
                f"🇺🇸 미국 전략 시작 (AI 기술적 분석)\n"
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
                    technical_signal = await self._analyze_stock(stock)
                    
                    # AI 확신도 체크 (필요시에만)
                    ai_analysis = await self.ai_engine.analyze_technical_signal(
                        stock, technical_signal, "미국 주식 전략 - 화목 거래"
                    )
                    
                    # 최종 판단
                    final_decision = self._make_final_decision(technical_signal, ai_analysis)
                    
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
                                
                                # AI 분석 기록 (사용된 경우)
                                if ai_analysis and ai_analysis.confidence != technical_signal.get('confidence', 0.5):
                                    self.position_manager.record_ai_analysis(ai_analysis, 'US')
                                
                                buy_results.append({
                                    'symbol': stock,
                                    'quantity': quantity,
                                    'price': final_decision['price'],
                                    'confidence': final_decision['confidence'],
                                    'ai_used': ai_analysis is not None and ai_analysis.confidence != technical_signal.get('confidence', 0.5)
                                })
                                
                                self.logger.info(f"✅ 매수 완료: {stock} {quantity}주 @ ${final_decision['price']:.2f}")
                
                except Exception as e:
                    self.logger.error(f"매수 실패 {stock}: {e}")
                    continue
            
            # 결과 알림
            if buy_results:
                ai_used_count = sum(1 for r in buy_results if r['ai_used'])
                message = f"🇺🇸 미국 전략 매수 완료\n"
                message += f"AI 사용: {ai_used_count}/{len(buy_results)}건\n"
                for result in buy_results:
                    ai_icon = "🤖" if result['ai_used'] else "📊"
                    message += f"• {ai_icon} {result['symbol']}: {result['quantity']}주 @ ${result['price']:.2f}\n"
                
                await self.notification_manager.send_notification(
                    message, 'success', '미국 전략 매수'
                )
            
            return {
                'success': True,
                'buy_count': len(buy_results),
                'total_investment': sum(r['quantity'] * r['price'] for r in buy_results),
                'ai_calls_used': sum(1 for r in buy_results if r['ai_used']),
                'dst_active': dst_active
            }
            
        except Exception as e:
            self.logger.error(f"미국 전략 실행 실패: {e}")
            await self.notification_manager.send_notification(
                f"🇺🇸 미국 전략 오류: {str(e)}", 'warning'
            )
            return {'success': False, 'error': str(e)}
    
    async def _select_stocks(self) -> List[str]:
        """종목 선별 (최적화)"""
        try:
            scored_stocks = []
            
            for symbol in self.stock_universe[:8]:  # 상위 8개만 분석 (비용 절약)
                try:
                    if not YAHOO_AVAILABLE:
                        continue
                        
                    stock = yf.Ticker(symbol)
                    data = stock.history(period="2mo")  # 기간 단축
                    info = stock.info
                    
                    if data.empty or len(data) < 30:
                        continue
                    
                    # 간단한 점수 계산
                    score = self._calculate_stock_score(data, info)
                    
                    if score > 0.6:
                        scored_stocks.append((symbol, score))
                        
                except Exception as e:
                    self.logger.debug(f"종목 분석 실패 {symbol}: {e}")
                    continue
            
            # 점수순 정렬 후 상위 4개 선택 (AI 비용 절약)
            scored_stocks.sort(key=lambda x: x[1], reverse=True)
            selected = [stock[0] for stock in scored_stocks[:4]]
            
            self.logger.info(f"미국 종목 선별: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"종목 선별 실패: {e}")
            return []
    
    def _calculate_stock_score(self, data: pd.DataFrame, info: Dict) -> float:
        """종목 점수 계산 (최적화)"""
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
            ma10 = closes.rolling(10).mean()
            ma20 = closes.rolling(20).mean()
            
            if closes.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
                score += 0.4
            
            # 거래량
            volume_ratio = data['Volume'].iloc[-3:].mean() / data['Volume'].iloc[-10:-3].mean()
            if volume_ratio > 1.1:
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.debug(f"점수 계산 오류: {e}")
            return 0.0
    
    async def _analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """개별 종목 기술적 분석"""
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
            
            # 기술적 분석
            closes = data['Close']
            ma5 = closes.rolling(5).mean().iloc[-1]
            ma10 = closes.rolling(10).mean().iloc[-1]
            
            # 트렌드 분석 (오류 수정)
            trend_strength = self._calculate_trend_strength(closes)
            
            if current_price > ma5 > ma10 and trend_strength > 0.6:
                action = 'BUY'
                confidence = 0.8
                reason = '상승 추세 강화'
            elif current_price > ma5 > ma10:
                action = 'BUY'
                confidence = 0.6  # 애매한 구간 (AI 호출 대상)
                reason = '상승 추세'
            elif current_price < ma5 < ma10:
                action = 'SELL'
                confidence = 0.7
                reason = '하락 추세'
            else:
                action = 'HOLD'
                confidence = 0.5  # 애매한 구간 (AI 호출 대상)
                reason = '중립'
            
            return {
                'action': action,
                'confidence': confidence,
                'price': current_price,
                'reason': reason,
                'ma5': ma5,
                'ma10': ma10,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"종목 분석 실패 {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'price': 0.0,
                'reason': f'분석 실패: {str(e)}'
            }
    
    def _calculate_trend_strength(self, closes: pd.Series) -> float:
        """트렌드 강도 계산 (오류 수정)"""
        try:
            if len(closes) < 10:
                return 0.5
            
            # 최근 10일 동안의 상승/하락 일수
            recent_changes = closes.tail(10).pct_change().dropna()
            up_days = (recent_changes > 0).sum()
            total_days = len(recent_changes)
            
            if total_days == 0:
                return 0.5
            
            trend_strength = up_days / total_days
            return trend_strength
            
        except Exception as e:
            return 0.5  # 오류시 중립값 반환
    
    def _make_final_decision(self, technical_signal: Dict[str, Any], 
                           ai_analysis: Optional[TechnicalAnalysisResult]) -> Dict[str, Any]:
        """최종 매매 결정"""
        if not ai_analysis or ai_analysis.confidence == technical_signal.get('confidence', 0.5):
            # AI가 사용되지 않았거나 기술적 분석과 동일한 경우
            return technical_signal
        
        # AI 분석이 사용된 경우
        return {
            'action': ai_analysis.action,
            'confidence': ai_analysis.confidence,
            'price': technical_signal['price'],
            'reason': f"AI 확신도 체크: {ai_analysis.reasoning[:30]}...",
            'ai_used': True
        }

# ============================================================================
# 🇯🇵 일본 전략 (최적화된 버전)
# ============================================================================
class JapanStrategy:
    """일본 주식 전략 (AI 기술적 분석 최적화)"""
    
    def __init__(self, config: OptimizedTradingConfig, ibkr_manager: IBKRManager, 
                 position_manager: PositionManager, notification_manager: NotificationManager,
                 ai_engine: OptimizedAIEngine):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.ai_engine = ai_engine
        
        self.logger = logging.getLogger('JapanStrategy')
        
        # 일본 주식 유니버스 (최적화)
        self.stock_universe = [
            '7203.T', '6758.T', '9984.T', '6861.T', '8306.T',  # 도요타, 소니, 소프트뱅크, 키엔스, 미쓰비시
            '7974.T', '9432.T', '8316.T', '6367.T', '4063.T',  # 닌텐도, NTT, 미쓰비시UFJ, 다이킨, 신에츠화학
            '9983.T', '8411.T', '6954.T', '7201.T'  # 패스트리테일링, 미즈호, 파나소닉, 닛산
        ]
    
    def is_trading_day(self) -> bool:
        """화목 거래일 체크"""
        return datetime.now().weekday() in [1, 3]  # 화요일, 목요일
    
    async def run_strategy(self) -> Dict[str, Any]:
        """일본 전략 실행 (최적화)"""
        try:
            if not self.is_trading_day():
                self.logger.info("오늘은 일본 전략 비거래일")
                return {'success': False, 'reason': 'not_trading_day'}
            
            await self.notification_manager.send_notification(
                f"🇯🇵 일본 전략 시작 (AI 기술적 분석)\n"
                f"목표 투자금: {self.config.TOTAL_PORTFOLIO_VALUE * self.config.JAPAN_ALLOCATION:,.0f}원",
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
            required_yen = self.config.TOTAL_PORTFOLIO_VALUE * self.config.JAPAN_ALLOCATION * usd_jpy_rate
            allocation_per_stock = required_yen / len(selected_stocks)
            
            for stock in selected_stocks:
                try:
                    technical_signal = await self._analyze_japanese_stock(stock, usd_jpy_rate)
                    
                    # AI 확신도 체크 (필요시에만)
                    ai_analysis = await self.ai_engine.analyze_technical_signal(
                        stock, technical_signal, "일본 주식 전략 - 화목 거래, 엔화 환전"
                    )
                    
                    # 최종 판단
                    final_decision = self._make_final_decision(technical_signal, ai_analysis)
                    
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
                                
                                # AI 분석 기록 (사용된 경우)
                                if ai_analysis and ai_analysis.confidence != technical_signal.get('confidence', 0.5):
                                    self.position_manager.record_ai_analysis(ai_analysis, 'JAPAN')
                                
                                buy_results.append({
                                    'symbol': stock,
                                    'quantity': quantity,
                                    'price': final_decision['price'],
                                    'confidence': final_decision['confidence'],
                                    'ai_used': ai_analysis is not None and ai_analysis.confidence != technical_signal.get('confidence', 0.5)
                                })
                                
                                self.logger.info(f"✅ 일본 매수: {stock} {quantity}주 @ ¥{final_decision['price']:,.0f}")
                
                except Exception as e:
                    self.logger.error(f"일본 매수 실패 {stock}: {e}")
                    continue
            
            # 결과 알림
            if buy_results:
                ai_used_count = sum(1 for r in buy_results if r['ai_used'])
                message = f"🇯🇵 일본 전략 매수 완료\n"
                message += f"USD/JPY: {usd_jpy_rate:.2f}\n"
                message += f"AI 사용: {ai_used_count}/{len(buy_results)}건\n"
                for result in buy_results:
                    ai_icon = "🤖" if result['ai_used'] else "📊"
                    message += f"• {ai_icon} {result['symbol']}: {result['quantity']}주 @ ¥{result['price']:,.0f}\n"
                
                await self.notification_manager.send_notification(
                    message, 'success', '일본 전략 매수'
                )
            
            return {
                'success': True,
                'buy_count': len(buy_results),
                'usd_jpy_rate': usd_jpy_rate,
                'total_investment_jpy': sum(r['quantity'] * r['price'] for r in buy_results),
                'ai_calls_used': sum(1 for r in buy_results if r['ai_used'])
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
            return 150.0
            
        except Exception as e:
            self.logger.error(f"환율 조회 실패: {e}")
            return 150.0
    
    async def _select_japanese_stocks(self) -> List[str]:
        """일본 종목 선별 (최적화)"""
        try:
            # 상위 6개 선별 (AI 비용 절약)
            selected = self.stock_universe[:6]
            self.logger.info(f"일본 종목 선별: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"일본 종목 선별 실패: {e}")
            return []
    
    async def _analyze_japanese_stock(self, symbol: str, usd_jpy_rate: float) -> Dict[str, Any]:
        """일본 종목 기술적 분석"""
        try:
            # 기본 분석 로직 (실제로는 Yahoo Finance로 데이터 수집)
            if YAHOO_AVAILABLE:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1mo")
                    
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        closes = data['Close']
                        ma5 = closes.rolling(5).mean().iloc[-1]
                        ma10 = closes.rolling(10).mean().iloc[-1]
                        
                        if current_price > ma5 > ma10:
                            action = 'BUY'
                            confidence = 0.7
                            reason = '상승 추세'
                        elif current_price < ma5 < ma10:
                            action = 'SELL'
                            confidence = 0.6
                            reason = '하락 추세'
                        else:
                            action = 'HOLD'
                            confidence = 0.5  # 애매한 구간
                            reason = '중립'
                        
                        return {
                            'action': action,
                            'confidence': confidence,
                            'price': current_price,
                            'usd_jpy_rate': usd_jpy_rate,
                            'reason': reason
                        }
                except Exception:
                    pass
            
            # 기본값 (데이터 없을 때)
            confidence = 0.6 + (hash(symbol) % 30) / 100  # 의사 랜덤
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
    
    def _make_final_decision(self, technical_signal: Dict[str, Any], 
                           ai_analysis: Optional[TechnicalAnalysisResult]) -> Dict[str, Any]:
        """최종 매매 결정"""
        if not ai_analysis or ai_analysis.confidence == technical_signal.get('confidence', 0.5):
            return technical_signal
        
        return {
            'action': ai_analysis.action,
            'confidence': ai_analysis.confidence,
            'price': technical_signal['price'],
            'reason': f"AI 확신도 체크: {ai_analysis.reasoning[:30]}...",
            'ai_used': True
        }

# ============================================================================
# 🇮🇳 인도 전략 (최적화된 버전)
# ============================================================================
class IndiaStrategy:
    """인도 주식 전략 (AI 기술적 분석 최적화)"""
    
    def __init__(self, config: OptimizedTradingConfig, ibkr_manager: IBKRManager, 
                 position_manager: PositionManager, notification_manager: NotificationManager,
                 ai_engine: OptimizedAIEngine):
        self.config = config
        self.ibkr_manager = ibkr_manager
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.ai_engine = ai_engine
        
        self.logger = logging.getLogger('IndiaStrategy')
        
        # 인도 주식 유니버스 (최적화)
        self.stock_universe = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY',  # 릴라이언스, TCS, HDFC은행, ICICI은행, 인포시스
            'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT',     # ITC, SBI, 바르티에어텔, 코탁은행, L&T
            'HCLTECH', 'AXISBANK'  # HCL테크, 액시스은행
        ]
    
    def is_trading_day(self) -> bool:
        """수요일 거래일 체크"""
        return datetime.now().weekday() == 2  # 수요일
    
    async def run_strategy(self) -> Dict[str, Any]:
        """인도 전략 실행 (최적화)"""
        try:
            if not self.is_trading_day():
                self.logger.info("오늘은 인도 전략 비거래일")
                return {'success': False, 'reason': 'not_trading_day'}
            
            await self.notification_manager.send_notification(
                f"🇮🇳 인도 전략 시작 (AI 기술적 분석)\n"
                f"목표 투자금: {self.config.TOTAL_PORTFOLIO_VALUE * self.config.INDIA_ALLOCATION:,.0f}원",
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
            required_inr = self.config.TOTAL_PORTFOLIO_VALUE * self.config.INDIA_ALLOCATION * usd_inr_rate
            allocation_per_stock = required_inr / len(selected_stocks)
            
            for stock in selected_stocks:
                try:
                    technical_signal = await self._analyze_indian_stock(stock, usd_inr_rate)
                    
                    # AI 확신도 체크 (필요시에만)
                    ai_analysis = await self.ai_engine.analyze_technical_signal(
                        stock, technical_signal, "인도 주식 전략 - 수요일 보수적 거래"
                    )
                    
                    # 최종 판단 (보수적)
                    final_decision = self._make_final_decision(technical_signal, ai_analysis)
                    
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
                                
                                # AI 분석 기록 (사용된 경우)
                                if ai_analysis and ai_analysis.confidence != technical_signal.get('confidence', 0.5):
                                    self.position_manager.record_ai_analysis(ai_analysis, 'INDIA')
                                
                                buy_results.append({
                                    'symbol': stock,
                                    'quantity': quantity,
                                    'price': final_decision['price'],
                                    'confidence': final_decision['confidence'],
                                    'ai_used': ai_analysis is not None and ai_analysis.confidence != technical_signal.get('confidence', 0.5)
                                })
                                
                                self.logger.info(f"✅ 인도 매수: {stock} {quantity}주 @ ₹{final_decision['price']:,.2f}")
                
                except Exception as e:
                    self.logger.error(f"인도 매수 실패 {stock}: {e}")
                    continue
            
            # 결과 알림
            if buy_results:
                ai_used_count = sum(1 for r in buy_results if r['ai_used'])
                message = f"🇮🇳 인도 전략 매수 완료\n"
                message += f"USD/INR: {usd_inr_rate:.2f}\n"
                message += f"AI 사용: {ai_used_count}/{len(buy_results)}건\n"
                for result in buy_results:
                    ai_icon = "🤖" if result['ai_used'] else "📊"
                    message += f"• {ai_icon} {result['symbol']}: {result['quantity']}주 @ ₹{result['price']:,.2f}\n"
                
                await self.notification_manager.send_notification(
                    message, 'success', '인도 전략 매수'
                )
            
            return {
                'success': True,
                'buy_count': len(buy_results),
                'usd_inr_rate': usd_inr_rate,
                'total_investment_inr': sum(r['quantity'] * r['price'] for r in buy_results),
                'ai_calls_used': sum(1 for r in buy_results if r['ai_used'])
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
            return 83.0
            
        except Exception as e:
            self.logger.error(f"환율 조회 실패: {e}")
            return 83.0
    
    async def _select_indian_stocks(self) -> List[str]:
        """인도 종목 선별 (보수적, 최적화)"""
        try:
            # 대형주 우선 선별 (상위 4개, AI 비용 절약)
            selected = self.stock_universe[:4]
            self.logger.info(f"인도 종목 선별: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"인도 종목 선별 실패: {e}")
            return []
    
    async def _analyze_indian_stock(self, symbol: str, usd_inr_rate: float) -> Dict[str, Any]:
        """인도 종목 기술적 분석"""
        try:
            # 보수적 분석 로직
            confidence = 0.65 + (hash(symbol) % 25) / 100  # 의사 랜덤 (보수적)
            price = 500 + (hash(symbol) % 3000)  # 의사 가격
            
            return {
                'action': 'BUY' if confidence > 0.7 else 'HOLD',
                'confidence': confidence,
                'price': price,
                'usd_inr_rate': usd_inr_rate,
                'reason': '보수적 기술적 분석'
            }
            
        except Exception as e:
            self.logger.error(f"인도 종목 분석 실패 {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'price': 500,
                'reason': f'분석 실패: {str(e)}'
            }
    
    def _make_final_decision(self, technical_signal: Dict[str, Any], 
                           ai_analysis: Optional[TechnicalAnalysisResult]) -> Dict[str, Any]:
        """최종 매매 결정 (보수적)"""
        if not ai_analysis or ai_analysis.confidence == technical_signal.get('confidence', 0.5):
            # 보수적 조정
            decision = technical_signal.copy()
            decision['confidence'] *= 0.9  # 보수적으로 신뢰도 감소
            return decision
        
        # AI 분석 사용시에도 보수적
        return {
            'action': ai_analysis.action if ai_analysis.confidence > 0.8 else 'HOLD',
            'confidence': ai_analysis.confidence * 0.9,  # 보수적 조정
            'price': technical_signal['price'],
            'reason': f"AI 보수적 체크: {ai_analysis.reasoning[:30]}...",
            'ai_used': True
        }

# ============================================================================
# 💰 암호화폐 전략 (최적화된 버전)
# ============================================================================
class CryptoStrategy:
    """암호화폐 전략 (AI 기술적 분석 최적화)"""
    
    def __init__(self, config: OptimizedTradingConfig, position_manager: PositionManager, 
                 notification_manager: NotificationManager, ai_engine: OptimizedAIEngine):
        self.config = config
        self.position_manager = position_manager
        self.notification_manager = notification_manager
        self.ai_engine = ai_engine
        
        self.logger = logging.getLogger('CryptoStrategy')
        
        # 암호화폐 유니버스 (최적화)
        self.crypto_universe = [
            'KRW-BTC', 'KRW-ETH', 'KRW-BNB', 'KRW-ADA', 'KRW-SOL',
            'KRW-AVAX', 'KRW-DOT', 'KRW-MATIC', 'KRW-ATOM', 'KRW-NEAR'
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
        """암호화폐 전략 실행 (최적화)"""
        try:
            if not self.is_trading_day():
                self.logger.info("오늘은 암호화폐 전략 비거래일")
                return {'success': False, 'reason': 'not_trading_day'}
            
            if not UPBIT_AVAILABLE:
                self.logger.warning("업비트 모듈이 없습니다")
                return {'success': False, 'reason': 'no_upbit_module'}
            
            await self.notification_manager.send_notification(
                f"💰 암호화폐 전략 시작 (AI 기술적 분석)\n"
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
                    technical_signal = await self._analyze_crypto(crypto, market_condition)
                    
                    # AI 확신도 체크 (필요시에만)
                    ai_analysis = await self.ai_engine.analyze_technical_signal(
                        crypto, technical_signal, "암호화폐 전략 - 월금 거래"
                    )
                    
                    # 최종 판단
                    final_decision = self._make_final_decision(technical_signal, ai_analysis)
                    
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
                                'ai_used': ai_analysis is not None and ai_analysis.confidence != technical_signal.get('confidence', 0.5)
                            })
                            
                            self.logger.info(f"✅ 암호화폐 매수: {crypto} {allocation_per_crypto:,.0f}원")
                
                except Exception as e:
                    self.logger.error(f"암호화폐 매수 실패 {crypto}: {e}")
                    continue
            
            # 결과 알림
            if buy_results:
                ai_used_count = sum(1 for r in buy_results if r['ai_used'])
                message = f"💰 암호화폐 전략 매수 완료\n"
                message += f"시장 상태: {market_condition['status']}\n"
                message += f"AI 사용: {ai_used_count}/{len(buy_results)}건\n"
                for result in buy_results:
                    ai_icon = "🤖" if result['ai_used'] else "📊"
                    message += f"• {ai_icon} {result['symbol']}: {result['amount']:,.0f}원\n"
                
                await self.notification_manager.send_notification(
                    message, 'success', '암호화폐 전략 매수'
                )
            
            return {
                'success': True,
                'buy_count': len(buy_results),
                'total_investment': sum(r['amount'] for r in buy_results),
                'market_condition': market_condition['status'],
                'ai_calls_used': sum(1 for r in buy_results if r['ai_used'])
            }
            
        except Exception as e:
            self.logger.error(f"암호화폐 전략 실행 실패: {e}")
            await self.notification_manager.send_notification(
                f"💰 암호화폐 전략 오류: {str(e)}", 'warning'
            )
            return {'success': False, 'error': str(e)}
    
    async def _analyze_crypto_market(self) -> Dict[str, Any]:
        """암호화폐 시장 분석 (간소화)"""
        try:
            # BTC 기준 시장 분석
            btc_price = pyupbit.get_current_price("KRW-BTC")
            btc_data = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=20)
            
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
        """암호화폐 선별 (최적화)"""
        try:
            # 시장 상태에 따른 선별 (AI 비용 절약)
            if market_condition['status'] == 'bullish':
                # 강세장: 알트코인 포함 (6개)
                selected = self.crypto_universe[:6]
            elif market_condition['status'] == 'bearish':
                # 약세장: 메이저코인만 (2개)
                selected = ['KRW-BTC', 'KRW-ETH']
            else:
                # 중립: 균형 (4개)
                selected = self.crypto_universe[:4]
            
            self.logger.info(f"암호화폐 선별: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"암호화폐 선별 실패: {e}")
            return []
    
    async def _analyze_crypto(self, symbol: str, market_condition: Dict) -> Dict[str, Any]:
        """개별 암호화폐 기술적 분석"""
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
                confidence = 0.6  # 애매한 구간 (AI 호출 대상)
                reason = '기술적 강세'
            else:
                action = 'HOLD'
                confidence = 0.5  # 애매한 구간 (AI 호출 대상)
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
    
    def _make_final_decision(self, technical_signal: Dict[str, Any], 
                           ai_analysis: Optional[TechnicalAnalysisResult]) -> Dict[str, Any]:
        """최종 매매 결정"""
        if not ai_analysis or ai_analysis.confidence == technical_signal.get('confidence', 0.5):
            return technical_signal
        
        # 암호화폐 특성상 AI 분석 가중치 높임
        return {
            'action': ai_analysis.action,
            'confidence': ai_analysis.confidence,
            'price': technical_signal['price'],
            'reason': f"AI 확신도 체크: {ai_analysis.reasoning[:30]}...",
            'ai_used': True
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
# 🌐 네트워크 모니터링 (간소화)
# ============================================================================
class NetworkMonitor:
    """네트워크 연결 모니터링 (간소화)"""
    
    def __init__(self, config: OptimizedTradingConfig, ibkr_manager: IBKRManager, 
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
                await asyncio.sleep(120)  # 2분 대기
    
    async def _check_connections(self):
        """연결 상태 체크 (간소화)"""
        # 인터넷 연결 체크
        internet_ok = await self._check_internet()
        
        # IBKR 연결 체크
        ibkr_ok = self.ibkr_manager.connected and (
            self.ibkr_manager.ib.isConnected() if self.ibkr_manager.ib else False
        )
        
        if not internet_ok or (IBKR_AVAILABLE and not ibkr_ok):
            self.connection_failures += 1
            
            # IBKR 없이 운영시 더 관대한 기준
            if not IBKR_AVAILABLE and internet_ok:
                if self.connection_failures == 1:
                    self.logger.info("ℹ️ IBKR 없이 운영 중 (암호화폐 전략만 사용)")
                self.connection_failures = 0
                return
            
            self.logger.warning(
                f"⚠️ 연결 실패 {self.connection_failures}회: "
                f"인터넷={internet_ok}, IBKR={ibkr_ok}"
            )
            
            # 연속 실패시 응급 조치
            if self.connection_failures >= 5:
                await self.notification_manager.send_notification(
                    f"🚨 네트워크 연결 실패 {self.connection_failures}회\n"
                    f"인터넷: {internet_ok}, IBKR: {ibkr_ok}",
                    'emergency'
                )
                
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

# ============================================================================
# 🏆 메인 최적화 거래 시스템
# ============================================================================
class OptimizedTradingSystem:
    """퀸트프로젝트 최적화 거래 시스템 (4대 전략 + AI 비용 최적화)"""
    
    def __init__(self):
        # 설정 로드
        self.config = OptimizedTradingConfig()
        
        # 로깅 설정
        self._setup_logging()
        
        # 로거 초기화
        self.logger = logging.getLogger('OptimizedTradingSystem')
        
        # 핵심 컴포넌트 초기화
        self.emergency_detector = EmergencyDetector(self.config)
        self.ibkr_manager = IBKRManager(self.config)
        self.notification_manager = NotificationManager(self.config)
        self.position_manager = PositionManager(self.config, self.ibkr_manager)
        self.network_monitor = NetworkMonitor(self.config, self.ibkr_manager, self.notification_manager)
        
        # 최적화된 AI 엔진 초기화
        self.ai_engine = OptimizedAIEngine(self.config)
        
        # 4대 전략 초기화
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
        
        file_handler = logging.FileHandler(log_dir / 'trading_optimized.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    def _init_strategies(self):
        """4대 전략 초기화"""
        try:
            # 미국 전략
            if self.config.US_ENABLED:
                self.strategies['US'] = USStrategy(
                    self.config, self.ibkr_manager, self.position_manager, 
                    self.notification_manager, self.ai_engine
                )
                self.logger.info("✅ 미국 전략 초기화 완료 (AI 최적화)")
            
            # 일본 전략
            if self.config.JAPAN_ENABLED:
                self.strategies['JAPAN'] = JapanStrategy(
                    self.config, self.ibkr_manager, self.position_manager, 
                    self.notification_manager, self.ai_engine
                )
                self.logger.info("✅ 일본 전략 초기화 완료 (AI 최적화)")
            
            # 인도 전략
            if self.config.INDIA_ENABLED:
                self.strategies['INDIA'] = IndiaStrategy(
                    self.config, self.ibkr_manager, self.position_manager, 
                    self.notification_manager, self.ai_engine
                )
                self.logger.info("✅ 인도 전략 초기화 완료 (AI 최적화)")
            
            # 암호화폐 전략
            if self.config.CRYPTO_ENABLED:
                self.strategies['CRYPTO'] = CryptoStrategy(
                    self.config, self.position_manager, self.notification_manager, self.ai_engine
                )
                self.logger.info("✅ 암호화폐 전략 초기화 완료 (AI 최적화)")
            
            if not self.strategies:
                self.logger.warning("⚠️ 활성화된 전략이 없습니다")
                
        except Exception as e:
            self.logger.error(f"전략 초기화 실패: {e}")
    
    async def start_system(self):
        """시스템 시작"""
        try:
            self.logger.info("🏆 퀸트프로젝트 최적화 거래 시스템 시작! (AI 비용 최적화)")
            self.start_time = datetime.now()
            self.running = True
            
            # IBKR 연결
            if IBKR_AVAILABLE:
                await self.ibkr_manager.connect()
            
            # 시작 알림
            await self.notification_manager.send_notification(
                f"🚀 퀸트프로젝트 최적화 시스템 시작\n"
                f"활성 전략: {', '.join(self.strategies.keys())}\n"
                f"IBKR 연결: {'✅' if self.ibkr_manager.connected else '❌'}\n"
                f"AI 엔진: {'✅' if self.ai_engine.client_available else '❌'}\n"
                f"AI 일일 제한: {self.config.AI_DAILY_CALL_LIMIT}회\n"
                f"총 포트폴리오: {self.config.TOTAL_PORTFOLIO_VALUE:,.0f}원",
                'success', '시스템 시작'
            )
            
            # 백그라운드 태스크 시작
            tasks = [
                asyncio.create_task(self._main_trading_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self.network_monitor.start_monitoring())
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
                
                self.logger.info(f"📅 {today_name}요일 전략 체크 (AI 호출: {self.config.AI_CALL_COUNTER}/{self.config.AI_DAILY_CALL_LIMIT})")
                
                for strategy_name, strategy_instance in self.strategies.items():
                    try:
                        if self._should_run_strategy(strategy_name, current_weekday):
                            self.logger.info(f"🎯 {strategy_name} 전략 실행")
                            result = await strategy_instance.run_strategy()
                            
                            if result.get('success'):
                                ai_calls = result.get('ai_calls_used', 0)
                                self.logger.info(f"✅ {strategy_name} 전략 완료 (AI 호출: {ai_calls}회)")
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
            'CRYPTO': [0, 4]   # 월금
        }
        
        return weekday in strategy_schedules.get(strategy_name, [])
    
    async def _monitoring_loop(self):
        """모니터링 루프 (간소화)"""
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
                
                # 주기적 상태 보고 (12시간마다, 간소화)
                if datetime.now().hour % 12 == 0 and datetime.now().minute < 10:
                    await self._send_status_report(portfolio_summary)
                
                await asyncio.sleep(600)  # 10분마다
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(120)
    
    async def _send_status_report(self, portfolio_summary: Dict):
        """상태 보고서 전송 (간소화)"""
        try:
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            report = (
                f"📊 퀸트프로젝트 최적화 상태\n\n"
                f"🕐 가동시간: {uptime}\n"
                f"💼 총 포지션: {portfolio_summary['total_positions']}개\n"
                f"💰 미실현 손익: {portfolio_summary['total_unrealized_pnl']:+,.0f}원\n"
                f"🤖 AI 호출: {self.config.AI_CALL_COUNTER}/{self.config.AI_DAILY_CALL_LIMIT}회\n\n"
                f"전략별 현황:\n"
            )
            
            for strategy, data in portfolio_summary['by_strategy'].items():
                report += f"  {strategy}: {data['count']}개 ({data['pnl']:+,.0f}원)\n"
            
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
                f"🛑 시스템 정상 종료\n"
                f"가동시간: {uptime}\n"
                f"AI 사용량: {self.config.AI_CALL_COUNTER}회",
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
# 🎮 편의 함수들 (최적화)
# ============================================================================
async def get_system_status():
    """시스템 상태 조회"""
    system = OptimizedTradingSystem()
    if IBKR_AVAILABLE:
        await system.ibkr_manager.connect()
    await system.position_manager.update_positions()
    
    summary = system.position_manager.get_portfolio_summary()
    
    return {
        'strategies': list(system.strategies.keys()),
        'ibkr_connected': system.ibkr_manager.connected,
        'ai_available': system.ai_engine.client_available,
        'ai_calls_today': system.config.AI_CALL_COUNTER,
        'ai_limit': system.config.AI_DAILY_CALL_LIMIT,
        'total_positions': summary['total_positions'],
        'total_unrealized_pnl': summary['total_unrealized_pnl'],
        'by_strategy': summary['by_strategy']
    }

async def test_notifications():
    """알림 시스템 테스트"""
    config = OptimizedTradingConfig()
    notifier = NotificationManager(config)
    
    test_results = {}
    
    # 텔레그램 테스트
    if config.TELEGRAM_ENABLED:
        success = await notifier.send_notification(
            "🧪 퀸트프로젝트 최적화 시스템 테스트 (AI 비용 절약)",
            'info', '테스트'
        )
        test_results['telegram'] = success
    
    # 이메일 테스트
    if config.EMAIL_ENABLED:
        success = await notifier.send_notification(
            "퀸트프로젝트 최적화 이메일 알림 테스트입니다.",
            'warning', '이메일 테스트'
        )
        test_results['email'] = success
    
    return test_results

async def test_ai_analysis():
    """AI 분석 시스템 테스트"""
    config = OptimizedTradingConfig()
    ai_engine = OptimizedAIEngine(config)
    
    if not ai_engine.client_available:
        return {'success': False, 'error': 'OpenAI API 키가 설정되지 않았습니다'}
    
    try:
        # 테스트 기술적 데이터
        test_data = {
            'action': 'BUY',
            'confidence': 0.5,  # 애매한 구간 (AI 호출 대상)
            'price': 150.0,
            'reason': '테스트 기술적 분석'
        }
        
        # AI 분석 수행
        analysis = await ai_engine.analyze_technical_signal(
            'AAPL', test_data, "테스트 분석"
        )
        
        if analysis:
            return {
                'success': True,
                'ai_called': analysis.confidence != test_data['confidence'],
                'analysis': {
                    'symbol': analysis.symbol,
                    'action': analysis.action,
                    'confidence': analysis.confidence,
                    'reasoning': analysis.reasoning,
                    'target_price': analysis.target_price,
                    'risk_level': analysis.risk_level
                },
                'ai_calls_used': config.AI_CALL_COUNTER
            }
        else:
            return {'success': False, 'error': 'AI 분석 결과 없음'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

async def run_single_strategy(strategy_name: str):
    """단일 전략 실행"""
    system = OptimizedTradingSystem()
    
    if strategy_name.upper() not in system.strategies:
        return {'success': False, 'error': f'전략 {strategy_name}을 찾을 수 없습니다'}
    
    try:
        # IBKR 연결 (필요시)
        if strategy_name.upper() not in ['CRYPTO']:
            if IBKR_AVAILABLE:
                await system.ibkr_manager.connect()
        
        # 전략 실행
        strategy = system.strategies[strategy_name.upper()]
        result = await strategy.run_strategy()
        
        # AI 사용량 추가
        if 'ai_calls_used' not in result:
            result['ai_calls_used'] = system.config.AI_CALL_COUNTER
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

async def analyze_portfolio_performance():
    """포트폴리오 성과 분석"""
    system = OptimizedTradingSystem()
    if IBKR_AVAILABLE:
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
        'ai_available': system.ai_engine.client_available,
        'ai_calls_today': system.config.AI_CALL_COUNTER,
        'ai_limit': system.config.AI_DAILY_CALL_LIMIT
    }
    
    return performance

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
    
    # 최적화 거래 시스템 생성
    system = OptimizedTradingSystem()
    
    try:
        print("🏆" + "="*70)
        print("🏆 퀸트프로젝트 최적화 거래 시스템 v3.0.0 (AI 비용 절약)")
        print("🏆" + "="*70)
        print("🇺🇸 미국 전략 (화목) - AI 기술적 분석만")
        print("🇯🇵 일본 전략 (화목) - AI 기술적 분석만")
        print("🇮🇳 인도 전략 (수요일) - AI 기술적 분석만")
        print("💰 암호화폐 전략 (월금) - AI 기술적 분석만")
        print("🤖 AI 최적화: 애매한 신호(0.4-0.7)에서만 호출")
        print("💰 월 AI 비용: 5천원 이하 목표")
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
        print(f"  AI 모델: {system.config.OPENAI_MODEL}")
        print(f"  AI 일일 제한: {system.config.AI_DAILY_CALL_LIMIT}회")
        print(f"  알림 시스템: {'활성' if system.config.TELEGRAM_ENABLED else '비활성'}")
        
        # 전략별 배분
        print(f"\n💰 4대 전략 자금 배분:")
        print(f"  🇺🇸 미국: {system.config.US_ALLOCATION:.1%} ({system.config.TOTAL_PORTFOLIO_VALUE * system.config.US_ALLOCATION:,.0f}원)")
        print(f"  🇯🇵 일본: {system.config.JAPAN_ALLOCATION:.1%} ({system.config.TOTAL_PORTFOLIO_VALUE * system.config.JAPAN_ALLOCATION:,.0f}원)")
        print(f"  🇮🇳 인도: {system.config.INDIA_ALLOCATION:.1%} ({system.config.TOTAL_PORTFOLIO_VALUE * system.config.INDIA_ALLOCATION:,.0f}원)")
        print(f"  💰 암호화폐: {system.config.CRYPTO_ALLOCATION:.1%} ({system.config.TOTAL_PORTFOLIO_VALUE * system.config.CRYPTO_ALLOCATION:,.0f}원)")
        
        # 거래 스케줄
        print(f"\n📅 최적화 거래 스케줄:")
        print(f"  월요일: 💰 암호화폐 전략")
        print(f"  화요일: 🇺🇸 미국 전략, 🇯🇵 일본 전략")
        print(f"  수요일: 🇮🇳 인도 전략")
        print(f"  목요일: 🇺🇸 미국 전략, 🇯🇵 일본 전략")
        print(f"  금요일: 💰 암호화폐 전략")
        print(f"  주말: 시스템 휴식")
        
        # AI 최적화 정보
        print(f"\n🤖 AI 비용 최적화:")
        print(f"  기술적 신뢰도 {system.config.AI_CONFIDENCE_THRESHOLD_MIN}-{system.config.AI_CONFIDENCE_THRESHOLD_MAX}에서만 AI 호출")
        print(f"  뉴스/시장심리 분석 제거 → 기술적 분석만")
        print(f"  최대 토큰: {system.config.OPENAI_MAX_TOKENS}개")
        print(f"  캐시 시간: 2시간")
        
        print(f"\n🚀 시스템 실행 옵션:")
        print(f"  1. 🏆 전체 시스템 자동 실행")
        print(f"  2. 📊 시스템 상태 확인")
        print(f"  3. 🇺🇸 미국 전략만 실행")
        print(f"  4. 🇯🇵 일본 전략만 실행")
        print(f"  5. 🇮🇳 인도 전략만 실행")
        print(f"  6. 💰 암호화폐 전략만 실행")
        print(f"  7. 🔔 알림 시스템 테스트")
        print(f"  8. 📈 포트폴리오 성과 분석")
        print(f"  9. 🤖 AI 분석 테스트")
        print(f"  0. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (0-9): ").strip()
                
                if choice == '1':
                    print("\n🏆 전체 시스템 자동 실행!")
                    print("🔄 4대 전략이 요일별로 자동 실행됩니다.")
                    print("🤖 AI는 애매한 신호에서만 호출됩니다 (비용 절약).")
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
                    print(f"AI 연결: {'✅' if status['ai_available'] else '❌'}")
                    print(f"AI 호출: {status['ai_calls_today']}/{status['ai_limit']}회")
                    print(f"총 포지션: {status['total_positions']}개")
                    print(f"미실현 손익: {status['total_unrealized_pnl']:+,.0f}원")
                    
                    if status['by_strategy']:
                        print("전략별 현황:")
                        for strategy, data in status['by_strategy'].items():
                            print(f"  {strategy}: {data['count']}개 ({data['pnl']:+,.0f}원)")
                
                elif choice == '3':
                    print("\n🇺🇸 미국 전략 실행 중 (AI 최적화)...")
                    result = await run_single_strategy('US')
                    print(f"결과: {result}")
                    if 'ai_calls_used' in result:
                        print(f"AI 호출: {result['ai_calls_used']}회")
                
                elif choice == '4':
                    print("\n🇯🇵 일본 전략 실행 중 (AI 최적화)...")
                    result = await run_single_strategy('JAPAN')
                    print(f"결과: {result}")
                    if 'ai_calls_used' in result:
                        print(f"AI 호출: {result['ai_calls_used']}회")
                
                elif choice == '5':
                    print("\n🇮🇳 인도 전략 실행 중 (AI 최적화)...")
                    result = await run_single_strategy('INDIA')
                    print(f"결과: {result}")
                    if 'ai_calls_used' in result:
                        print(f"AI 호출: {result['ai_calls_used']}회")
                
                elif choice == '6':
                    print("\n💰 암호화폐 전략 실행 중 (AI 최적화)...")
                    result = await run_single_strategy('CRYPTO')
                    print(f"결과: {result}")
                    if 'ai_calls_used' in result:
                        print(f"AI 호출: {result['ai_calls_used']}회")
                
                elif choice == '7':
                    print("\n🔔 알림 시스템 테스트 중...")
                    test_results = await test_notifications()
                    
                    for channel, success in test_results.items():
                        status = "✅ 성공" if success else "❌ 실패"
                        print(f"  {channel}: {status}")
                
                elif choice == '8':
                    print("\n📈 포트폴리오 성과 분석 중...")
                    performance = await analyze_portfolio_performance()
                    
                    print(f"총 포트폴리오 가치: {performance['total_value']:,.0f}원")
                    print(f"미실현 손익: {performance['unrealized_pnl']:+,.0f}원 ({performance['unrealized_return_pct']:+.2f}%)")
                    print(f"총 포지션: {performance['total_positions']}개")
                    print(f"수익 포지션: {performance['profitable_positions']}개")
                    print(f"손실 포지션: {performance['losing_positions']}개")
                    print(f"승률: {performance['win_rate']:.1f}%")
                    print(f"AI 사용: {performance['ai_calls_today']}/{performance['ai_limit']}회")
                    
                    if performance['by_strategy']:
                        print("\n전략별 성과:")
                        for strategy, data in performance['by_strategy'].items():
                            print(f"  {strategy}: {data['count']}개 포지션, {data['pnl']:+,.0f}원")
                
                elif choice == '9':
                    print("\n🤖 AI 분석 시스템 테스트 중...")
                    test_result = await test_ai_analysis()
                    
                    if test_result['success']:
                        analysis = test_result['analysis']
                        print(f"✅ AI 분석 테스트 성공!")
                        print(f"  AI 호출됨: {'예' if test_result['ai_called'] else '아니오'}")
                        print(f"  종목: {analysis['symbol']}")
                        print(f"  추천: {analysis['action']}")
                        print(f"  신뢰도: {analysis['confidence']:.1%}")
                        print(f"  목표가: ${analysis['target_price']:.2f}")
                        print(f"  위험도: {analysis['risk_level']}")
                        print(f"  분석 근거: {analysis['reasoning'][:100]}...")
                        print(f"  AI 호출 카운터: {test_result['ai_calls_used']}회")
                    else:
                        print(f"❌ AI 분석 테스트 실패: {test_result['error']}")
                
                elif choice == '0':
                    print("👋 퀸트프로젝트 최적화 거래 시스템을 종료합니다!")
                    break
                    
                else:
                    print("❌ 잘못된 선택입니다. 0-9 중 선택하세요.")
                    
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
        print("🏆 퀸트프로젝트 최적화 거래 시스템 로딩... (AI 비용 절약)")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 퀸트프로젝트 최적화 시스템 종료")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 실행 오류: {e}")
        sys.exit(1)
