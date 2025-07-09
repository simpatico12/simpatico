#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 통합 스케줄러 시스템 (scheduler.py)
=======================================================
🕐 시간 기반 자동 거래 스케줄링 + 📊 전략 실행 관리 + 🤖 AI 분석

✨ 핵심 기능:
- 시간대별 전략 자동 실행
- 거래 시간 체크 및 관리
- 시장 휴무일 처리
- 백그라운드 모니터링
- 조건부 실행 (시장 상황)
- 스케줄 설정 관리
- 실행 결과 추적
- 🚨 스케줄 실패 감지
- 🤖 OpenAI 기반 시장 분석
- 🧠 AI 트레이딩 전략 추천
- 📈 포트폴리오 AI 최적화

Author: 퀸트마스터팀
Version: 1.0.0 (시간 기반 자동 스케줄링 + AI)
"""

import asyncio
import logging
import sys
import os
import json
import time
import threading
import traceback
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from dotenv import load_dotenv
import requests
import aiohttp
import sqlite3
import shutil
import subprocess
import pytz
from enum import Enum
import yaml

# OpenAI 통합
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI 모듈 없음")

# 전략 모듈 임포트
try:
    from core import QuantProjectCore, CoreConfig
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("⚠️ 코어 시스템 모듈 없음")

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

# 추가 라이브러리
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    print("⚠️ holidays 모듈 없음")

# ============================================================================
# 🕐 스케줄 관련 열거형 및 데이터 클래스
# ============================================================================
class ScheduleStatus(Enum):
    """스케줄 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class MarketType(Enum):
    """시장 타입"""
    US = "US"
    JAPAN = "JAPAN"
    INDIA = "INDIA"
    CRYPTO = "CRYPTO"
    KOREA = "KOREA"

class AIAnalysisType(Enum):
    """AI 분석 타입"""
    MARKET_SENTIMENT = "market_sentiment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    RISK_ASSESSMENT = "risk_assessment"
    TRADING_STRATEGY = "trading_strategy"
    NEWS_ANALYSIS = "news_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"

@dataclass
class ScheduleJob:
    """스케줄 작업 정의"""
    id: str
    name: str
    strategy: str
    function: str
    schedule_type: str  # daily, weekly, monthly, cron
    schedule_value: str  # 시간, 요일, cron 표현식
    market_type: MarketType
    enabled: bool = True
    timezone: str = "Asia/Seoul"
    max_runtime: int = 3600  # 최대 실행 시간 (초)
    retry_count: int = 3
    retry_delay: int = 300  # 재시도 지연 (초)
    conditions: Dict[str, Any] = field(default_factory=dict)
    last_run: Optional[datetime] = None
    last_status: ScheduleStatus = ScheduleStatus.PENDING
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    ai_enabled: bool = False  # AI 분석 활성화 여부

@dataclass
class ExecutionResult:
    """실행 결과"""
    job_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ScheduleStatus = ScheduleStatus.RUNNING
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    ai_analysis: Optional[Dict[str, Any]] = None

@dataclass
class AIAnalysisResult:
    """AI 분석 결과"""
    analysis_type: AIAnalysisType
    timestamp: datetime
    model_used: str
    input_data: Dict[str, Any]
    analysis_result: Dict[str, Any]
    confidence_score: float
    recommendations: List[str]
    risk_level: str
    execution_time: float

# ============================================================================
# 🤖 OpenAI 통합 관리자
# ============================================================================
class OpenAIManager:
    """OpenAI 통합 관리"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('OpenAIManager')
        self.client = None
        self.model = "gpt-4"
        
        self._init_openai()
    
    def _init_openai(self):
        """OpenAI 클라이언트 초기화"""
        try:
            if not OPENAI_AVAILABLE:
                self.logger.warning("OpenAI 라이브러리가 설치되지 않음")
                return
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.logger.warning("OpenAI API 키가 설정되지 않음")
                return
            
            self.client = AsyncOpenAI(api_key=api_key)
            self.logger.info("✅ OpenAI 클라이언트 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"OpenAI 초기화 실패: {e}")
    
    async def analyze_market_sentiment(self, market_data: Dict[str, Any]) -> AIAnalysisResult:
        """시장 센티먼트 분석"""
        try:
            start_time = time.time()
            
            prompt = f"""
            다음 시장 데이터를 분석하고 시장 센티먼트를 평가해주세요:

            시장 데이터:
            {json.dumps(market_data, indent=2, ensure_ascii=False)}

            다음 형태로 응답해주세요:
            1. 전체 시장 센티먼트 (매우 부정적/부정적/중립/긍정적/매우 긍정적)
            2. 주요 영향 요인 3가지
            3. 단기 전망 (1-3일)
            4. 위험 수준 (낮음/보통/높음/매우 높음)
            5. 투자 추천사항 3가지

            JSON 형태로 응답해주세요.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 전문적인 금융 시장 분석가입니다. 데이터를 기반으로 정확하고 실용적인 분석을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis_text = response.choices[0].message.content
            
            # JSON 파싱 시도
            try:
                analysis_json = json.loads(analysis_text)
            except json.JSONDecodeError:
                # JSON 파싱 실패시 텍스트로 저장
                analysis_json = {"raw_analysis": analysis_text}
            
            execution_time = time.time() - start_time
            
            return AIAnalysisResult(
                analysis_type=AIAnalysisType.MARKET_SENTIMENT,
                timestamp=datetime.now(),
                model_used=self.model,
                input_data=market_data,
                analysis_result=analysis_json,
                confidence_score=0.85,  # 기본값
                recommendations=analysis_json.get("투자추천사항", []),
                risk_level=analysis_json.get("위험수준", "보통"),
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"시장 센티먼트 분석 실패: {e}")
            raise
    
    async def optimize_portfolio(self, portfolio_data: Dict[str, Any]) -> AIAnalysisResult:
        """포트폴리오 최적화 분석"""
        try:
            start_time = time.time()
            
            prompt = f"""
            다음 포트폴리오 데이터를 분석하고 최적화 방안을 제시해주세요:

            포트폴리오 데이터:
            {json.dumps(portfolio_data, indent=2, ensure_ascii=False)}

            다음 항목들을 분석해주세요:
            1. 현재 포트폴리오의 위험도 평가
            2. 자산 배분 최적화 제안
            3. 리밸런싱 필요성
            4. 개선 가능한 포지션들
            5. 예상 수익률 향상 방안

            JSON 형태로 응답해주세요.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 포트폴리오 최적화 전문가입니다. 위험 관리와 수익 극대화의 균형을 추구합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            
            try:
                analysis_json = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis_json = {"raw_analysis": analysis_text}
            
            execution_time = time.time() - start_time
            
            return AIAnalysisResult(
                analysis_type=AIAnalysisType.PORTFOLIO_OPTIMIZATION,
                timestamp=datetime.now(),
                model_used=self.model,
                input_data=portfolio_data,
                analysis_result=analysis_json,
                confidence_score=0.8,
                recommendations=analysis_json.get("개선방안", []),
                risk_level=analysis_json.get("위험도", "보통"),
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"포트폴리오 최적화 분석 실패: {e}")
            raise
    
    async def assess_trading_risk(self, trading_data: Dict[str, Any]) -> AIAnalysisResult:
        """트레이딩 위험 평가"""
        try:
            start_time = time.time()
            
            prompt = f"""
            다음 트레이딩 데이터를 분석하고 위험을 평가해주세요:

            트레이딩 데이터:
            {json.dumps(trading_data, indent=2, ensure_ascii=False)}

            다음을 평가해주세요:
            1. 전체 위험 수준 (1-10점)
            2. 주요 위험 요인들
            3. 위험 완화 방안
            4. 포지션 크기 조정 제안
            5. 스톱로스 레벨 제안

            JSON 형태로 응답해주세요.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 리스크 관리 전문가입니다. 보수적이고 신중한 접근을 선호합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            analysis_text = response.choices[0].message.content
            
            try:
                analysis_json = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis_json = {"raw_analysis": analysis_text}
            
            execution_time = time.time() - start_time
            
            return AIAnalysisResult(
                analysis_type=AIAnalysisType.RISK_ASSESSMENT,
                timestamp=datetime.now(),
                model_used=self.model,
                input_data=trading_data,
                analysis_result=analysis_json,
                confidence_score=0.9,
                recommendations=analysis_json.get("위험완화방안", []),
                risk_level=analysis_json.get("위험수준", "높음"),
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"트레이딩 위험 평가 실패: {e}")
            raise
    
    async def generate_trading_strategy(self, market_context: Dict[str, Any]) -> AIAnalysisResult:
        """트레이딩 전략 생성"""
        try:
            start_time = time.time()
            
            prompt = f"""
            다음 시장 상황을 바탕으로 트레이딩 전략을 생성해주세요:

            시장 상황:
            {json.dumps(market_context, indent=2, ensure_ascii=False)}

            다음을 포함한 전략을 제시해주세요:
            1. 전략 개요
            2. 진입 조건
            3. 종료 조건
            4. 리스크 관리 방법
            5. 예상 수익률 및 위험도
            6. 타임프레임
            7. 자금 배분 제안

            JSON 형태로 응답해주세요.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 경험이 풍부한 퀀트 트레이더입니다. 데이터 기반의 체계적인 전략을 수립합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2500
            )
            
            analysis_text = response.choices[0].message.content
            
            try:
                analysis_json = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis_json = {"raw_analysis": analysis_text}
            
            execution_time = time.time() - start_time
            
            return AIAnalysisResult(
                analysis_type=AIAnalysisType.TRADING_STRATEGY,
                timestamp=datetime.now(),
                model_used=self.model,
                input_data=market_context,
                analysis_result=analysis_json,
                confidence_score=0.75,
                recommendations=analysis_json.get("전략제안", []),
                risk_level=analysis_json.get("위험도", "보통"),
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"트레이딩 전략 생성 실패: {e}")
            raise
    
    async def analyze_news_impact(self, news_data: List[Dict[str, Any]]) -> AIAnalysisResult:
        """뉴스 영향도 분석"""
        try:
            start_time = time.time()
            
            # 뉴스 데이터를 텍스트로 변환
            news_text = "\n".join([f"제목: {news.get('title', '')}\n내용: {news.get('content', '')[:500]}\n---" 
                                  for news in news_data[:10]])  # 최대 10개 뉴스만
            
            prompt = f"""
            다음 뉴스들이 금융 시장에 미칠 영향을 분석해주세요:

            뉴스 데이터:
            {news_text}

            다음을 분석해주세요:
            1. 각 뉴스의 시장 영향도 (1-10점)
            2. 영향받을 섹터/자산
            3. 단기/중기 영향 전망
            4. 투자 기회 또는 위험
            5. 대응 전략

            JSON 형태로 응답해주세요.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 뉴스와 시장의 상관관계를 분석하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            
            try:
                analysis_json = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis_json = {"raw_analysis": analysis_text}
            
            execution_time = time.time() - start_time
            
            return AIAnalysisResult(
                analysis_type=AIAnalysisType.NEWS_ANALYSIS,
                timestamp=datetime.now(),
                model_used=self.model,
                input_data={"news_count": len(news_data)},
                analysis_result=analysis_json,
                confidence_score=0.7,
                recommendations=analysis_json.get("대응전략", []),
                risk_level=analysis_json.get("위험수준", "보통"),
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"뉴스 영향도 분석 실패: {e}")
            raise

# ============================================================================
# 🎯 스케줄러 설정 관리자
# ============================================================================
class SchedulerConfig:
    """스케줄러 설정 관리"""
    
    def __init__(self):
        load_dotenv()
        
        # 기본 설정
        self.TIMEZONE = os.getenv('SCHEDULER_TIMEZONE', 'Asia/Seoul')
        self.MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', 3))
        self.JOB_TIMEOUT = int(os.getenv('JOB_TIMEOUT', 3600))
        self.RETRY_ENABLED = os.getenv('RETRY_ENABLED', 'true').lower() == 'true'
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
        
        # AI 설정
        self.AI_ENABLED = os.getenv('AI_ENABLED', 'true').lower() == 'true'
        self.AI_MODEL = os.getenv('AI_MODEL', 'gpt-4')
        self.AI_ANALYSIS_INTERVAL = int(os.getenv('AI_ANALYSIS_INTERVAL', 6))  # 시간
        self.AI_MAX_TOKENS = int(os.getenv('AI_MAX_TOKENS', 2000))
        
        # 시장 시간 설정 (각 국가의 현지 시간 기준)
        self.US_MARKET_OPEN = dt_time(9, 30)  # 미국 동부 서머타임 9:30 EDT (한국시간 23:30)
        self.US_MARKET_CLOSE = dt_time(16, 0)  # 미국 동부 서머타임 16:00 EDT (한국시간 06:00)
        self.JAPAN_MARKET_OPEN = dt_time(9, 0)  # 일본 시간 9:00 (한국시간 9:00)
        self.JAPAN_MARKET_CLOSE = dt_time(15, 0)  # 일본 시간 15:00 (한국시간 15:00)
        self.INDIA_MARKET_OPEN = dt_time(9, 15)  # 인도 시간 9:15 (한국시간 12:45)
        self.INDIA_MARKET_CLOSE = dt_time(15, 30)  # 인도 시간 15:30 (한국시간 19:00)
        
        # 알림 설정
        self.SCHEDULE_NOTIFICATIONS = os.getenv('SCHEDULE_NOTIFICATIONS', 'true').lower() == 'true'
        self.FAILURE_NOTIFICATIONS = os.getenv('FAILURE_NOTIFICATIONS', 'true').lower() == 'true'
        self.SUCCESS_NOTIFICATIONS = os.getenv('SUCCESS_NOTIFICATIONS', 'false').lower() == 'true'
        self.AI_NOTIFICATIONS = os.getenv('AI_NOTIFICATIONS', 'true').lower() == 'true'
        
        # 데이터베이스
        self.SCHEDULER_DB_PATH = os.getenv('SCHEDULER_DB_PATH', './data/scheduler.db')
        self.SCHEDULE_CONFIG_PATH = os.getenv('SCHEDULE_CONFIG_PATH', './config/schedules.yaml')
        
        # 로그 설정
        self.LOG_LEVEL = os.getenv('SCHEDULER_LOG_LEVEL', 'INFO')
        self.LOG_PATH = os.getenv('SCHEDULER_LOG_PATH', './logs/scheduler.log')

# ============================================================================
# 🗓️ 시장 시간 관리자
# ============================================================================
class MarketTimeManager:
    """시장 시간 및 휴무일 관리"""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.timezone_map = {
            MarketType.US: pytz.timezone('US/Eastern'),
            MarketType.JAPAN: pytz.timezone('Asia/Tokyo'),
            MarketType.INDIA: pytz.timezone('Asia/Kolkata'),
            MarketType.CRYPTO: pytz.timezone('UTC'),
            MarketType.KOREA: pytz.timezone('Asia/Seoul')
        }
        
        self.market_hours = {
            MarketType.US: (self.config.US_MARKET_OPEN, self.config.US_MARKET_CLOSE),
            MarketType.JAPAN: (self.config.JAPAN_MARKET_OPEN, self.config.JAPAN_MARKET_CLOSE),
            MarketType.INDIA: (self.config.INDIA_MARKET_OPEN, self.config.INDIA_MARKET_CLOSE),
            MarketType.CRYPTO: (dt_time(0, 0), dt_time(23, 59)),  # 24시간
            MarketType.KOREA: (dt_time(9, 0), dt_time(15, 30))
        }
        
        self.logger = logging.getLogger('MarketTimeManager')
        
        # 휴무일 데이터 로드
        self._load_holidays()
    
    def _load_holidays(self):
        """휴무일 데이터 로드"""
        self.holidays_data = {}
        
        if HOLIDAYS_AVAILABLE:
            try:
                # 각 국가별 휴무일
                self.holidays_data[MarketType.US] = holidays.UnitedStates(years=range(2020, 2030))
                self.holidays_data[MarketType.JAPAN] = holidays.Japan(years=range(2020, 2030))
                self.holidays_data[MarketType.INDIA] = holidays.India(years=range(2020, 2030))
                self.holidays_data[MarketType.KOREA] = holidays.SouthKorea(years=range(2020, 2030))
                self.holidays_data[MarketType.CRYPTO] = {}  # 암호화폐는 휴무일 없음
            except Exception as e:
                self.logger.warning(f"휴무일 데이터 로드 실패: {e}")
                self.holidays_data = {}
    
    def is_market_open(self, market_type: MarketType, target_time: datetime = None) -> bool:
        """시장 개장 여부 확인"""
        if target_time is None:
            target_time = datetime.now()
        
        try:
            # 암호화폐는 항상 개장
            if market_type == MarketType.CRYPTO:
                return True
            
            # 시장 시간대로 변환
            market_tz = self.timezone_map.get(market_type, pytz.timezone('Asia/Seoul'))
            market_time = target_time.astimezone(market_tz)
            
            # 주말 체크
            if market_time.weekday() >= 5:  # 토요일(5), 일요일(6)
                return False
            
            # 휴무일 체크
            if self._is_holiday(market_type, market_time.date()):
                return False
            
            # 시장 시간 체크 (서머타임 자동 처리)
            if market_type == MarketType.US:
                # 미국 시장은 현지 시간 기준으로 체크 (서머타임 자동 적용)
                us_time = target_time.astimezone(pytz.timezone('US/Eastern'))
                market_open = dt_time(9, 30)  # EDT/EST 모두 9:30
                market_close = dt_time(16, 0)  # EDT/EST 모두 16:00
                current_time = us_time.time()
                return market_open <= current_time <= market_close
            else:
                # 다른 시장들은 기존 방식 유지
                market_open, market_close = self.market_hours.get(market_type, (dt_time(0, 0), dt_time(23, 59)))
                current_time = market_time.time()
                return market_open <= current_time <= market_close
            
        except Exception as e:
            self.logger.error(f"시장 개장 여부 확인 실패: {e}")
            return False
    
    def _is_holiday(self, market_type: MarketType, target_date) -> bool:
        """휴무일 여부 확인"""
        holiday_list = self.holidays_data.get(market_type, {})
        return target_date in holiday_list
    
    def get_market_status_summary(self) -> Dict[str, bool]:
        """모든 시장의 현재 상태 요약"""
        status = {}
        for market_type in MarketType:
            status[market_type.value] = self.is_market_open(market_type)
        return status

# ============================================================================
# 📋 스케줄 작업 관리자
# ============================================================================
class JobManager:
    """스케줄 작업 관리"""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.jobs: Dict[str, ScheduleJob] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.execution_history: List[ExecutionResult] = []
        
        self.logger = logging.getLogger('JobManager')
        self._init_database()
        self._load_job_configs()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            os.makedirs(os.path.dirname(self.config.SCHEDULER_DB_PATH), exist_ok=True)
            
            conn = sqlite3.connect(self.config.SCHEDULER_DB_PATH)
            cursor = conn.cursor()
            
            # 작업 실행 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    status TEXT,
                    result_data TEXT,
                    error_message TEXT,
                    execution_time REAL,
                    memory_usage REAL,
                    cpu_usage REAL,
                    ai_analysis TEXT
                )
            ''')
            
            # AI 분석 결과 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type TEXT,
                    timestamp DATETIME,
                    model_used TEXT,
                    input_data TEXT,
                    analysis_result TEXT,
                    confidence_score REAL,
                    recommendations TEXT,
                    risk_level TEXT,
                    execution_time REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def _load_job_configs(self):
        """작업 설정 로드"""
        try:
            # 기본 작업 설정 생성
            self._create_default_jobs()
                
        except Exception as e:
            self.logger.error(f"작업 설정 로드 실패: {e}")
            self._create_default_jobs()
    
    def _create_default_jobs(self):
        """기본 작업 설정 생성"""
        default_jobs = [
            # 미국 전략 - 화목 밤 23:30 (미국 동부 서머타임 09:30 EDT에 맞춤)
            ScheduleJob(
                id="us_strategy_tue_thu",
                name="미국 전략 - 화목",
                strategy="US",
                function="execute_legendary_strategy",
                schedule_type="weekly",
                schedule_value="TUE,THU 23:30",
                market_type=MarketType.US,
                conditions={"market_open_required": True},
                ai_enabled=True
            ),
            
            # 일본 전략 - 화목 오전 8:45 (일본 시장 9시 개장 직전)
            ScheduleJob(
                id="japan_strategy_tue_thu",
                name="일본 전략 - 화목",
                strategy="JAPAN",
                function="execute_legendary_strategy",
                schedule_type="weekly",
                schedule_value="TUE,THU 08:45",
                market_type=MarketType.JAPAN,
                conditions={"market_open_required": True},
                ai_enabled=True
            ),
            
            # 인도 전략 - 수요일 오후 12:45 (인도시간 9:15 개장에 맞춤)
            ScheduleJob(
                id="india_strategy_wed",
                name="인도 전략 - 수요일",
                strategy="INDIA",
                function="execute_legendary_strategy",
                schedule_type="weekly",
                schedule_value="WED 12:45",
                market_type=MarketType.INDIA,
                conditions={"market_open_required": True},
                ai_enabled=True
            ),
            
            # 암호화폐 전략 - 월금 오전 9시 (24시간 시장이므로 한국 시간 기준)
            ScheduleJob(
                id="crypto_strategy_mon_fri",
                name="암호화폐 전략 - 월금",
                strategy="CRYPTO",
                function="execute_legendary_strategy",
                schedule_type="weekly",
                schedule_value="MON,FRI 09:00",
                market_type=MarketType.CRYPTO,
                conditions={},
                ai_enabled=True
            ),
            
            # 포트폴리오 모니터링 - 매일 오후 6시
            ScheduleJob(
                id="portfolio_monitoring_daily",
                name="포트폴리오 모니터링",
                strategy="CORE",
                function="update_portfolio_status",
                schedule_type="daily",
                schedule_value="18:00",
                market_type=MarketType.CRYPTO,  # 24시간 가능
                conditions={},
                ai_enabled=True
            ),
            
            # AI 시장 분석 - 6시간마다
            ScheduleJob(
                id="ai_market_analysis",
                name="AI 시장 분석",
                strategy="AI",
                function="analyze_market_sentiment",
                schedule_type="cron",
                schedule_value="0 */6 * * *",  # 6시간마다
                market_type=MarketType.CRYPTO,
                conditions={},
                ai_enabled=True
            ),
            
            # AI 포트폴리오 최적화 - 매일 자정
            ScheduleJob(
                id="ai_portfolio_optimization",
                name="AI 포트폴리오 최적화",
                strategy="AI",
                function="optimize_portfolio",
                schedule_type="daily",
                schedule_value="00:00",
                market_type=MarketType.CRYPTO,
                conditions={},
                ai_enabled=True
            ),
            
            # AI 위험 평가 - 매일 오전 6시, 오후 6시
            ScheduleJob(
                id="ai_risk_assessment",
                name="AI 위험 평가",
                strategy="AI",
                function="assess_trading_risk",
                schedule_type="daily",
                schedule_value="06:00,18:00",
                market_type=MarketType.CRYPTO,
                conditions={},
                ai_enabled=True
            ),
            
            # AI 뉴스 분석 - 3시간마다
            ScheduleJob(
                id="ai_news_analysis",
                name="AI 뉴스 분석",
                strategy="AI",
                function="analyze_news_impact",
                schedule_type="cron",
                schedule_value="0 */3 * * *",  # 3시간마다
                market_type=MarketType.CRYPTO,
                conditions={},
                ai_enabled=True
            ),
            
            # 시스템 상태 체크 - 매시간
            ScheduleJob(
                id="system_health_check",
                name="시스템 상태 체크",
                strategy="CORE",
                function="check_system_health",
                schedule_type="cron",
                schedule_value="0 * * * *",  # 매시간 정각
                market_type=MarketType.CRYPTO,
                conditions={}
            ),
            
            # 백업 작업 - 매일 새벽 3시
            ScheduleJob(
                id="daily_backup",
                name="일일 백업",
                strategy="CORE",
                function="perform_backup",
                schedule_type="daily",
                schedule_value="03:00",
                market_type=MarketType.CRYPTO,
                conditions={}
            )
        ]
        
        for job in default_jobs:
            self.jobs[job.id] = job
        
        self.logger.info(f"✅ {len(default_jobs)}개 기본 작업 생성 (AI 통합 포함)")
    
    def get_all_jobs_status(self) -> List[Dict[str, Any]]:
        """모든 작업 상태 조회"""
        results = []
        for job in self.jobs.values():
            is_running = job.id in self.running_jobs
            results.append({
                'id': job.id,
                'name': job.name,
                'enabled': job.enabled,
                'status': job.last_status.value,
                'is_running': is_running,
                'last_run': job.last_run,
                'next_run': job.next_run,
                'run_count': job.run_count,
                'success_count': job.success_count,
                'failure_count': job.failure_count,
                'success_rate': (job.success_count / job.run_count * 100) if job.run_count > 0 else 0,
                'ai_enabled': job.ai_enabled
            })
        return results

# ============================================================================
# ⚡ 작업 실행 엔진
# ============================================================================
class JobExecutor:
    """작업 실행 엔진"""
    
    def __init__(self, config: SchedulerConfig, job_manager: JobManager, market_manager: MarketTimeManager):
        self.config = config
        self.job_manager = job_manager
        self.market_manager = market_manager
        
        # 전략 인스턴스
        self.strategy_instances = {}
        self._init_strategies()
        
        # 코어 시스템
        self.core_system = None
        self._init_core_system()
        
        # AI 관리자
        self.ai_manager = None
        self._init_ai_manager()
        
        self.logger = logging.getLogger('JobExecutor')
    
    def _init_strategies(self):
        """전략 인스턴스 초기화"""
        try:
            if US_AVAILABLE:
                self.strategy_instances['US'] = USStrategy()
                self.logger.info("✅ 미국 전략 인스턴스 생성")
            
            if JAPAN_AVAILABLE:
                self.strategy_instances['JAPAN'] = JapanStrategy()
                self.logger.info("✅ 일본 전략 인스턴스 생성")
            
            if INDIA_AVAILABLE:
                self.strategy_instances['INDIA'] = IndiaStrategy()
                self.logger.info("✅ 인도 전략 인스턴스 생성")
            
            if CRYPTO_AVAILABLE:
                self.strategy_instances['CRYPTO'] = CryptoStrategy()
                self.logger.info("✅ 암호화폐 전략 인스턴스 생성")
                
        except Exception as e:
            self.logger.error(f"전략 인스턴스 초기화 실패: {e}")
    
    def _init_core_system(self):
        """코어 시스템 초기화"""
        try:
            if CORE_AVAILABLE:
                self.core_system = QuantProjectCore()
                self.logger.info("✅ 코어 시스템 인스턴스 생성")
        except Exception as e:
            self.logger.error(f"코어 시스템 초기화 실패: {e}")
    
    def _init_ai_manager(self):
        """AI 관리자 초기화"""
        try:
            if self.config.AI_ENABLED and OPENAI_AVAILABLE:
                self.ai_manager = OpenAIManager(self.config)
                self.logger.info("✅ AI 관리자 인스턴스 생성")
        except Exception as e:
            self.logger.error(f"AI 관리자 초기화 실패: {e}")
    
    async def execute_job(self, job: ScheduleJob) -> ExecutionResult:
        """작업 실행"""
        execution_result = ExecutionResult(
            job_id=job.id,
            start_time=datetime.now(),
            status=ScheduleStatus.RUNNING
        )
        
        try:
            self.logger.info(f"🚀 작업 실행 시작: {job.name}")
            
            # 실행 조건 체크
            if not self._check_execution_conditions(job):
                execution_result.status = ScheduleStatus.SKIPPED
                execution_result.error_message = "실행 조건 불만족"
                self.logger.info(f"⏭️ 작업 건너뛰기: {job.name} - 실행 조건 불만족")
                return execution_result
            
            # 타임아웃 설정
            timeout_task = asyncio.create_task(
                asyncio.wait_for(self._execute_job_function(job), timeout=job.max_runtime)
            )
            
            # 작업 실행
            result_data = await timeout_task
            
            # AI 분석 수행 (AI 활성화된 경우)
            if job.ai_enabled and self.ai_manager and job.strategy != 'AI':
                try:
                    ai_analysis = await self._perform_ai_analysis(job, result_data)
                    execution_result.ai_analysis = ai_analysis
                    result_data['ai_analysis'] = ai_analysis
                except Exception as e:
                    self.logger.warning(f"AI 분석 실패 (작업은 성공): {e}")
            
            execution_result.status = ScheduleStatus.COMPLETED
            execution_result.result_data = result_data
            execution_result.end_time = datetime.now()
            execution_result.execution_time = (execution_result.end_time - execution_result.start_time).total_seconds()
            
            # 성공 업데이트
            job.last_run = execution_result.start_time
            job.last_status = ScheduleStatus.COMPLETED
            job.run_count += 1
            job.success_count += 1
            
            self.logger.info(f"✅ 작업 완료: {job.name} ({execution_result.execution_time:.1f}초)")
            
        except asyncio.TimeoutError:
            execution_result.status = ScheduleStatus.FAILED
            execution_result.error_message = f"타임아웃 ({job.max_runtime}초)"
            execution_result.end_time = datetime.now()
            
            job.last_status = ScheduleStatus.FAILED
            job.run_count += 1
            job.failure_count += 1
            
            self.logger.error(f"⏰ 작업 타임아웃: {job.name}")
            
        except Exception as e:
            execution_result.status = ScheduleStatus.FAILED
            execution_result.error_message = str(e)
            execution_result.end_time = datetime.now()
            
            job.last_status = ScheduleStatus.FAILED
            job.run_count += 1
            job.failure_count += 1
            
            self.logger.error(f"❌ 작업 실행 실패: {job.name} - {e}")
            
        finally:
            # 실행 기록 저장
            self._save_execution_result(execution_result)
            
        return execution_result
    
    def _check_execution_conditions(self, job: ScheduleJob) -> bool:
        """실행 조건 체크"""
        try:
            conditions = job.conditions
            
            # 시장 개장 필요 조건
            if conditions.get('market_open_required', False):
                if not self.market_manager.is_market_open(job.market_type):
                    return False
            
            # 최소 대기 시간 조건
            min_interval = conditions.get('min_interval_hours', 0)
            if min_interval > 0 and job.last_run:
                time_since_last = datetime.now() - job.last_run
                if time_since_last < timedelta(hours=min_interval):
                    return False
            
            # 시스템 리소스 조건
            max_cpu = conditions.get('max_cpu_usage', 90)
            max_memory = conditions.get('max_memory_usage', 90)
            
            if self._get_system_cpu_usage() > max_cpu:
                return False
            
            if self._get_system_memory_usage() > max_memory:
                return False
            
            # 동시 실행 작업 수 제한
            if len(self.job_manager.running_jobs) >= self.config.MAX_CONCURRENT_JOBS:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"실행 조건 체크 실패: {e}")
            return False
    
    async def _execute_job_function(self, job: ScheduleJob) -> Dict[str, Any]:
        """작업 함수 실행"""
        try:
            if job.strategy == 'CORE':
                # 코어 시스템 함수 실행
                return await self._execute_core_function(job.function)
            elif job.strategy == 'AI':
                # AI 함수 실행
                return await self._execute_ai_function(job.function)
            else:
                # 전략 함수 실행
                return await self._execute_strategy_function(job.strategy, job.function)
                
        except Exception as e:
            self.logger.error(f"작업 함수 실행 실패: {e}")
            raise
    
    async def _execute_core_function(self, function_name: str) -> Dict[str, Any]:
        """코어 시스템 함수 실행"""
        if not self.core_system:
            raise Exception("코어 시스템이 초기화되지 않음")
        
        if function_name == 'update_portfolio_status':
            await self.core_system.position_manager.update_all_positions()
            summary = self.core_system.position_manager.get_portfolio_summary()
            return {'portfolio_summary': summary}
            
        elif function_name == 'check_system_health':
            health_status = self.core_system.emergency_detector.check_system_health()
            return {'health_status': health_status}
            
        elif function_name == 'perform_backup':
            await self.core_system.backup_manager.perform_backup()
            return {'backup_completed': True}
            
        else:
            raise Exception(f"알 수 없는 코어 함수: {function_name}")
    
    async def _execute_ai_function(self, function_name: str) -> Dict[str, Any]:
        """AI 함수 실행"""
        if not self.ai_manager:
            raise Exception("AI 관리자가 초기화되지 않음")
        
        if function_name == 'analyze_market_sentiment':
            # 시장 데이터 수집
            market_data = await self._collect_market_data()
            analysis = await self.ai_manager.analyze_market_sentiment(market_data)
            self._save_ai_analysis(analysis)
            return {'ai_analysis': analysis.__dict__}
            
        elif function_name == 'optimize_portfolio':
            # 포트폴리오 데이터 수집
            portfolio_data = await self._collect_portfolio_data()
            analysis = await self.ai_manager.optimize_portfolio(portfolio_data)
            self._save_ai_analysis(analysis)
            return {'ai_analysis': analysis.__dict__}
            
        elif function_name == 'assess_trading_risk':
            # 거래 데이터 수집
            trading_data = await self._collect_trading_data()
            analysis = await self.ai_manager.assess_trading_risk(trading_data)
            self._save_ai_analysis(analysis)
            return {'ai_analysis': analysis.__dict__}
            
        elif function_name == 'analyze_news_impact':
            # 뉴스 데이터 수집
            news_data = await self._collect_news_data()
            analysis = await self.ai_manager.analyze_news_impact(news_data)
            self._save_ai_analysis(analysis)
            return {'ai_analysis': analysis.__dict__}
            
        else:
            raise Exception(f"알 수 없는 AI 함수: {function_name}")
    
    async def _execute_strategy_function(self, strategy_name: str, function_name: str) -> Dict[str, Any]:
        """전략 함수 실행"""
        strategy_instance = self.strategy_instances.get(strategy_name)
        if not strategy_instance:
            raise Exception(f"전략 인스턴스가 없습니다: {strategy_name}")
        
        if function_name == 'execute_legendary_strategy':
            if hasattr(strategy_instance, 'run_strategy'):
                result = await strategy_instance.run_strategy()
            elif hasattr(strategy_instance, 'execute_legendary_strategy'):
                result = await strategy_instance.execute_legendary_strategy()
            else:
                raise Exception(f"전략 실행 메서드를 찾을 수 없습니다: {strategy_name}")
            
            return {'strategy_result': result}
        else:
            raise Exception(f"알 수 없는 전략 함수: {function_name}")
    
    async def _perform_ai_analysis(self, job: ScheduleJob, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI 분석 수행"""
        try:
            if job.strategy == 'US':
                # 미국 전략 결과에 대한 AI 분석
                analysis = await self.ai_manager.assess_trading_risk(result_data)
            elif job.strategy in ['JAPAN', 'INDIA', 'CRYPTO']:
                # 기타 전략 결과에 대한 AI 분석
                analysis = await self.ai_manager.assess_trading_risk(result_data)
            elif job.strategy == 'CORE' and job.function == 'update_portfolio_status':
                # 포트폴리오 모니터링에 대한 AI 분석
                analysis = await self.ai_manager.optimize_portfolio(result_data)
            else:
                return None
            
            self._save_ai_analysis(analysis)
            return analysis.__dict__
            
        except Exception as e:
            self.logger.error(f"AI 분석 실행 실패: {e}")
            return None
    
    async def _collect_market_data(self) -> Dict[str, Any]:
        """시장 데이터 수집"""
        try:
            # 시장 상태 정보
            market_status = self.market_manager.get_market_status_summary()
            
            # 포트폴리오 요약 (있는 경우)
            portfolio_summary = {}
            if self.core_system:
                try:
                    portfolio_summary = self.core_system.position_manager.get_portfolio_summary()
                except:
                    pass
            
            # 최근 실행 결과들
            recent_results = self.job_manager.execution_history[-10:] if self.job_manager.execution_history else []
            
            return {
                'timestamp': datetime.now().isoformat(),
                'market_status': market_status,
                'portfolio_summary': portfolio_summary,
                'recent_executions': len(recent_results),
                'system_health': {
                    'cpu_usage': self._get_system_cpu_usage(),
                    'memory_usage': self._get_system_memory_usage()
                }
            }
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 실패: {e}")
            return {'error': str(e)}
    
    async def _collect_portfolio_data(self) -> Dict[str, Any]:
        """포트폴리오 데이터 수집"""
        try:
            if not self.core_system:
                return {'error': '코어 시스템 없음'}
            
            portfolio_summary = self.core_system.position_manager.get_portfolio_summary()
            
            # 추가 포트폴리오 메트릭
            portfolio_data = {
                'summary': portfolio_summary,
                'timestamp': datetime.now().isoformat(),
                'market_status': self.market_manager.get_market_status_summary()
            }
            
            return portfolio_data
            
        except Exception as e:
            self.logger.error(f"포트폴리오 데이터 수집 실패: {e}")
            return {'error': str(e)}
    
    async def _collect_trading_data(self) -> Dict[str, Any]:
        """거래 데이터 수집"""
        try:
            # 최근 거래 실행 결과들
            recent_results = []
            for result in self.job_manager.execution_history[-5:]:
                if result.status == ScheduleStatus.COMPLETED:
                    recent_results.append({
                        'job_id': result.job_id,
                        'execution_time': result.execution_time,
                        'timestamp': result.start_time.isoformat()
                    })
            
            return {
                'recent_trades': recent_results,
                'timestamp': datetime.now().isoformat(),
                'market_status': self.market_manager.get_market_status_summary()
            }
            
        except Exception as e:
            self.logger.error(f"거래 데이터 수집 실패: {e}")
            return {'error': str(e)}
    
    async def _collect_news_data(self) -> List[Dict[str, Any]]:
        """뉴스 데이터 수집"""
        try:
            # 기본 뉴스 데이터 (실제 구현에서는 뉴스 API 사용)
            mock_news = [
                {
                    'title': 'Fed 금리 결정 발표',
                    'content': '연방준비제도가 기준금리 동결을 발표했습니다.',
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Financial News'
                },
                {
                    'title': '암호화폐 시장 동향',
                    'content': '비트코인이 새로운 저항선을 돌파했습니다.',
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Crypto News'
                }
            ]
            
            return mock_news
            
        except Exception as e:
            self.logger.error(f"뉴스 데이터 수집 실패: {e}")
            return []
    
    def _save_ai_analysis(self, analysis: AIAnalysisResult):
        """AI 분석 결과 저장"""
        try:
            conn = sqlite3.connect(self.config.SCHEDULER_DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_analyses 
                (analysis_type, timestamp, model_used, input_data, analysis_result, 
                 confidence_score, recommendations, risk_level, execution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.analysis_type.value,
                analysis.timestamp.isoformat(),
                analysis.model_used,
                json.dumps(analysis.input_data),
                json.dumps(analysis.analysis_result),
                analysis.confidence_score,
                json.dumps(analysis.recommendations),
                analysis.risk_level,
                analysis.execution_time
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"AI 분석 결과 저장 실패: {e}")
    
    def _get_system_cpu_usage(self) -> float:
        """시스템 CPU 사용률 조회"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
    
    def _get_system_memory_usage(self) -> float:
        """시스템 메모리 사용률 조회"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def _save_execution_result(self, result: ExecutionResult):
        """실행 결과 저장"""
        try:
            conn = sqlite3.connect(self.config.SCHEDULER_DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO job_executions 
                (job_id, start_time, end_time, status, result_data, error_message, 
                 execution_time, memory_usage, cpu_usage, ai_analysis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.job_id,
                result.start_time.isoformat(),
                result.end_time.isoformat() if result.end_time else None,
                result.status.value,
                json.dumps(result.result_data),
                result.error_message,
                result.execution_time,
                result.memory_usage,
                result.cpu_usage,
                json.dumps(result.ai_analysis) if result.ai_analysis else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"실행 결과 저장 실패: {e}")

# ============================================================================
# 📅 스케줄 계산기
# ============================================================================
class ScheduleCalculator:
    """스케줄 계산 및 다음 실행 시간 결정"""
    
    def __init__(self, market_manager: MarketTimeManager):
        self.market_manager = market_manager
        self.logger = logging.getLogger('ScheduleCalculator')
    
    def calculate_next_run(self, job: ScheduleJob) -> datetime:
        """다음 실행 시간 계산"""
        try:
            current_time = datetime.now(pytz.timezone(job.timezone))
            
            if job.schedule_type == 'daily':
                return self._calculate_daily_next_run(job, current_time)
            elif job.schedule_type == 'weekly':
                return self._calculate_weekly_next_run(job, current_time)
            elif job.schedule_type == 'monthly':
                return self._calculate_monthly_next_run(job, current_time)
            elif job.schedule_type == 'cron':
                return self._calculate_cron_next_run(job, current_time)
            else:
                # 기본값: 1시간 후
                return current_time + timedelta(hours=1)
                
        except Exception as e:
            self.logger.error(f"다음 실행 시간 계산 실패: {e}")
            return datetime.now() + timedelta(hours=1)
    
    def _calculate_daily_next_run(self, job: ScheduleJob, current_time: datetime) -> datetime:
        """일일 스케줄 계산"""
        try:
            # schedule_value 형식: "HH:MM" 또는 "HH:MM,HH:MM" (여러 시간)
            time_strings = job.schedule_value.split(',')
            
            next_runs = []
            for time_str in time_strings:
                time_str = time_str.strip()
                hour, minute = map(int, time_str.split(':'))
                
                # 오늘 시간이 지났으면 내일
                if current_time >= today_run:
                    next_run = today_run + timedelta(days=1)
                else:
                    next_run = today_run
                
                next_runs.append(next_run)
            
            # 가장 빠른 시간 반환
            return min(next_runs)
            
        except Exception as e:
            self.logger.error(f"일일 스케줄 계산 실패: {e}")
            return current_time + timedelta(days=1)
    
    def _calculate_weekly_next_run(self, job: ScheduleJob, current_time: datetime) -> datetime:
        """주간 스케줄 계산"""
        try:
            # schedule_value 형식: "MON,WED 09:30" 또는 "TUE 14:00"
            parts = job.schedule_value.split(' ')
            weekdays_str = parts[0]
            time_str = parts[1]
            
            hour, minute = map(int, time_str.split(':'))
            
            # 요일 변환
            weekday_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
            target_weekdays = [weekday_map[day.strip()] for day in weekdays_str.split(',')]
            
            # 미국 전략의 경우 서머타임 자동 조정
            if job.strategy == 'US':
                return self._calculate_us_schedule_with_dst(current_time, target_weekdays, hour, minute)
            
            # 다음 실행 가능한 날짜 찾기
            for i in range(7):
                check_date = current_time + timedelta(days=i)
                if check_date.weekday() in target_weekdays:
                    run_time = check_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # 오늘이면서 시간이 지나지 않았거나, 미래 날짜면
                    if (i == 0 and current_time < run_time) or i > 0:
                        return run_time
            
            # 기본값: 다음 주
            return current_time + timedelta(weeks=1)
            
        except Exception as e:
            self.logger.error(f"주간 스케줄 계산 실패: {e}")
            return current_time + timedelta(days=1)
    
    def _calculate_us_schedule_with_dst(self, current_time: datetime, target_weekdays: List[int], hour: int, minute: int) -> datetime:
        """미국 스케줄 서머타임 자동 조정"""
        try:
            # 현재 시간을 한국 시간대로 설정
            kst = pytz.timezone('Asia/Seoul')
            if current_time.tzinfo is None:
                current_time = kst.localize(current_time)
            
            # 미국 동부 시간대
            us_eastern = pytz.timezone('US/Eastern')
            
            # 다음 실행 가능한 날짜 찾기
            for i in range(7):
                check_date = current_time + timedelta(days=i)
                
                if check_date.weekday() in target_weekdays:
                    # 미국 동부시간으로 목표 시간 생성
                    us_date = check_date.astimezone(us_eastern)
                    us_target = us_date.replace(hour=9, minute=30, second=0, microsecond=0)  # 미국 개장 시간
                    
                    # 한국 시간으로 변환 (서머타임 자동 적용)
                    kst_target = us_target.astimezone(kst)
                    
                    # 오늘이면서 시간이 지나지 않았거나, 미래 날짜면
                    if (i == 0 and current_time < kst_target) or i > 0:
                        self.logger.info(f"🇺🇸 미국 스케줄: 미국시간 {us_target.strftime('%H:%M')} → 한국시간 {kst_target.strftime('%H:%M')}")
                        return kst_target.replace(tzinfo=None)  # naive datetime 반환
            
            # 기본값: 다음 주
            return current_time.replace(tzinfo=None) + timedelta(weeks=1)
            
        except Exception as e:
            self.logger.error(f"미국 스케줄 서머타임 계산 실패: {e}")
            # 실패시 기본 시간 (23:30) 사용
            for i in range(7):
                check_date = current_time + timedelta(days=i)
                if check_date.weekday() in target_weekdays:
                    run_time = check_date.replace(hour=23, minute=30, second=0, microsecond=0)
                    if (i == 0 and current_time < run_time) or i > 0:
                        return run_time
            return current_time + timedelta(weeks=1)
    
    def _calculate_monthly_next_run(self, job: ScheduleJob, current_time: datetime) -> datetime:
        """월간 스케줄 계산"""
        try:
            # schedule_value 형식: "1 09:00" (매월 1일 9시)
            parts = job.schedule_value.split(' ')
            day = int(parts[0])
            time_str = parts[1]
            
            hour, minute = map(int, time_str.split(':'))
            
            # 이번 달 실행 시간
            try:
                this_month_run = current_time.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
            except ValueError:
                # 해당 일이 이번 달에 없으면 다음 달
                this_month_run = current_time.replace(day=1, hour=hour, minute=minute, second=0, microsecond=0)
                this_month_run += timedelta(days=32)
                this_month_run = this_month_run.replace(day=day)
            
            # 시간이 지났으면 다음 달
            if current_time >= this_month_run:
                next_month = this_month_run.replace(day=1) + timedelta(days=32)
                next_run = next_month.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
            else:
                next_run = this_month_run
            
            return next_run
            
        except Exception as e:
            self.logger.error(f"월간 스케줄 계산 실패: {e}")
            return current_time + timedelta(days=30)
    
    def _calculate_cron_next_run(self, job: ScheduleJob, current_time: datetime) -> datetime:
        """크론 표현식 스케줄 계산"""
        try:
            # 간단한 크론 파서 (분 시 일 월 요일)
            cron_parts = job.schedule_value.split()
            
            if len(cron_parts) != 5:
                raise ValueError("잘못된 크론 표현식")
            
            minute, hour, day, month, weekday = cron_parts
            
            # 다음 실행 시간 계산 (단순화된 버전)
            next_run = current_time + timedelta(hours=1)
            
            # 정확한 크론 계산은 croniter 라이브러리 사용 권장
            if minute != '*':
                next_run = next_run.replace(minute=int(minute))
            if hour != '*':
                next_run = next_run.replace(hour=int(hour))
            
            return next_run
            
        except Exception as e:
            self.logger.error(f"크론 스케줄 계산 실패: {e}")
            return current_time + timedelta(hours=1)

# ============================================================================
# 🔔 스케줄러 알림 시스템
# ============================================================================
class SchedulerNotificationManager:
    """스케줄러 전용 알림 관리"""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.logger = logging.getLogger('SchedulerNotification')
    
    async def send_job_start_notification(self, job: ScheduleJob):
        """작업 시작 알림"""
        if not self.config.SCHEDULE_NOTIFICATIONS:
            return
        
        ai_status = "🤖 AI 분석 포함" if job.ai_enabled else ""
        message = f"🚀 작업 시작: {job.name}\n전략: {job.strategy}\n시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{ai_status}"
        await self._send_notification(message, 'info')
    
    async def send_job_success_notification(self, job: ScheduleJob, execution_result: ExecutionResult):
        """작업 성공 알림"""
        if not self.config.SUCCESS_NOTIFICATIONS:
            return
        
        ai_info = ""
        if execution_result.ai_analysis:
            ai_info = f"\n🤖 AI 분석: 완료"
        
        message = (
            f"✅ 작업 완료: {job.name}\n"
            f"실행 시간: {execution_result.execution_time:.1f}초\n"
            f"상태: 성공{ai_info}"
        )
        await self._send_notification(message, 'success')
    
    async def send_job_failure_notification(self, job: ScheduleJob, execution_result: ExecutionResult):
        """작업 실패 알림"""
        if not self.config.FAILURE_NOTIFICATIONS:
            return
        
        message = (
            f"❌ 작업 실패: {job.name}\n"
            f"오류: {execution_result.error_message}\n"
            f"실패 횟수: {job.failure_count}/{job.run_count}"
        )
        await self._send_notification(message, 'warning')
    
    async def send_ai_analysis_notification(self, analysis: AIAnalysisResult):
        """AI 분석 결과 알림"""
        if not self.config.AI_NOTIFICATIONS:
            return
        
        message = (
            f"🤖 AI 분석 완료\n"
            f"타입: {analysis.analysis_type.value}\n"
            f"위험도: {analysis.risk_level}\n"
            f"신뢰도: {analysis.confidence_score:.2f}\n"
            f"추천사항: {len(analysis.recommendations)}개"
        )
        await self._send_notification(message, 'info')
    
    async def send_schedule_summary(self, jobs_status: List[Dict[str, Any]]):
        """스케줄 요약 알림"""
        try:
            total_jobs = len(jobs_status)
            enabled_jobs = sum(1 for job in jobs_status if job['enabled'])
            running_jobs = sum(1 for job in jobs_status if job['is_running'])
            ai_enabled_jobs = sum(1 for job in jobs_status if job.get('ai_enabled', False))
            
            success_rate = 0
            if total_jobs > 0:
                total_runs = sum(job['run_count'] for job in jobs_status)
                total_success = sum(job['success_count'] for job in jobs_status)
                success_rate = (total_success / total_runs * 100) if total_runs > 0 else 0
            
            message = (
                f"📊 스케줄러 요약\n\n"
                f"📋 총 작업: {total_jobs}개\n"
                f"✅ 활성화: {enabled_jobs}개\n"
                f"🔄 실행 중: {running_jobs}개\n"
                f"🤖 AI 활성화: {ai_enabled_jobs}개\n"
                f"📈 성공률: {success_rate:.1f}%\n"
            )
            
            await self._send_notification(message, 'info')
            
        except Exception as e:
            self.logger.error(f"스케줄 요약 알림 실패: {e}")
    
    async def _send_notification(self, message: str, priority: str):
        """알림 전송"""
        try:
            # 텔레그램 알림
            telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
            telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if telegram_token and telegram_chat_id:
                url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                
                priority_emojis = {
                    'info': 'ℹ️',
                    'success': '✅',
                    'warning': '⚠️',
                    'error': '❌'
                }
                
                emoji = priority_emojis.get(priority, 'ℹ️')
                formatted_message = f"{emoji} 퀸트 스케줄러 AI\n\n{message}"
                
                data = {
                    'chat_id': telegram_chat_id,
                    'text': formatted_message,
                    'parse_mode': 'HTML'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        if response.status == 200:
                            self.logger.debug("알림 전송 성공")
                        else:
                            self.logger.error(f"알림 전송 실패: {response.status}")
            
        except Exception as e:
            self.logger.error(f"알림 전송 오류: {e}")

# ============================================================================
# 🏆 퀸트프로젝트 통합 스케줄러 시스템
# ============================================================================
class QuantProjectScheduler:
    """퀸트프로젝트 통합 스케줄러 시스템"""
    
    def __init__(self):
        # 설정 로드
        self.config = SchedulerConfig()
        
        # 로깅 설정
        self._setup_logging()
        self.logger = logging.getLogger('QuantScheduler')
        
        # 핵심 컴포넌트 초기화
        self.market_manager = MarketTimeManager(self.config)
        self.job_manager = JobManager(self.config)
        self.job_executor = JobExecutor(self.config, self.job_manager, self.market_manager)
        self.schedule_calculator = ScheduleCalculator(self.market_manager)
        self.notification_manager = SchedulerNotificationManager(self.config)
        
        # 시스템 상태
        self.running = False
        self.start_time = None
        self.scheduler_task = None
    
    def _setup_logging(self):
        """로깅 설정"""
        log_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # 파일 핸들러
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'scheduler.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # 스케줄러 로거 설정
        scheduler_logger = logging.getLogger('QuantScheduler')
        scheduler_logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        scheduler_logger.addHandler(console_handler)
        scheduler_logger.addHandler(file_handler)
        
        # 다른 로거들도 설정
        for logger_name in ['JobManager', 'JobExecutor', 'MarketTimeManager', 'ScheduleCalculator', 'OpenAIManager']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
    
    async def start_scheduler(self):
        """스케줄러 시작"""
        try:
            self.logger.info("🏆 퀸트프로젝트 AI 스케줄러 시작!")
            self.start_time = datetime.now()
            self.running = True
            
            # 모든 작업의 다음 실행 시간 계산
            await self._update_all_next_run_times()
            
            # 시작 알림
            market_status = self.market_manager.get_market_status_summary()
            jobs_status = self.job_manager.get_all_jobs_status()
            ai_enabled_count = sum(1 for job in jobs_status if job.get('ai_enabled', False))
            
            await self.notification_manager._send_notification(
                f"🚀 AI 스케줄러 시작\n"
                f"총 작업: {len(jobs_status)}개\n"
                f"활성화: {sum(1 for job in jobs_status if job['enabled'])}개\n"
                f"🤖 AI 활성화: {ai_enabled_count}개\n"
                f"시장 상태: {market_status}",
                'info'
            )
            
            # 메인 스케줄러 루프 시작
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            # 백그라운드 태스크들
            tasks = [
                self.scheduler_task,
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self._cleanup_loop()),
                asyncio.create_task(self._ai_monitoring_loop())
            ]
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"스케줄러 시작 실패: {e}")
            await self.shutdown()
    
    async def _scheduler_loop(self):
        """메인 스케줄러 루프"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # 실행 대상 작업 확인
                ready_jobs = []
                for job in self.job_manager.jobs.values():
                    if (job.enabled and 
                        job.next_run and 
                        current_time >= job.next_run and
                        job.id not in self.job_manager.running_jobs):
                        ready_jobs.append(job)
                
                # 작업 실행
                for job in ready_jobs:
                    if len(self.job_manager.running_jobs) < self.config.MAX_CONCURRENT_JOBS:
                        await self._execute_job_async(job)
                    else:
                        self.logger.warning(f"⚠️ 동시 실행 한계로 작업 지연: {job.name}")
                
                # 1분 대기
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"스케줄러 루프 오류: {e}")
                await asyncio.sleep(60)
    
    async def _execute_job_async(self, job: ScheduleJob):
        """비동기 작업 실행"""
        try:
            # 실행 중인 작업에 추가
            task = asyncio.create_task(self._run_job_with_retry(job))
            self.job_manager.running_jobs[job.id] = task
            
            ai_indicator = "🤖" if job.ai_enabled else ""
            self.logger.info(f"🔄 작업 스케줄: {job.name} {ai_indicator}")
            
        except Exception as e:
            self.logger.error(f"비동기 작업 실행 오류: {e}")
    
    async def _run_job_with_retry(self, job: ScheduleJob):
        """재시도 기능이 있는 작업 실행"""
        try:
            await self.notification_manager.send_job_start_notification(job)
            
            execution_result = None
            
            # 재시도 루프
            for attempt in range(job.retry_count + 1):
                try:
                    execution_result = await self.job_executor.execute_job(job)
                    
                    if execution_result.status == ScheduleStatus.COMPLETED:
                        await self.notification_manager.send_job_success_notification(job, execution_result)
                        
                        # AI 분석 결과 알림
                        if execution_result.ai_analysis and self.config.AI_NOTIFICATIONS:
                            try:
                                analysis_obj = AIAnalysisResult(**execution_result.ai_analysis)
                                await self.notification_manager.send_ai_analysis_notification(analysis_obj)
                            except:
                                pass
                        
                        break
                    elif execution_result.status == ScheduleStatus.SKIPPED:
                        break
                    else:
                        # 실패한 경우 재시도
                        if attempt < job.retry_count:
                            self.logger.warning(f"🔄 작업 재시도 {attempt + 1}/{job.retry_count}: {job.name}")
                            await asyncio.sleep(job.retry_delay)
                        else:
                            await self.notification_manager.send_job_failure_notification(job, execution_result)
                            
                except Exception as e:
                    self.logger.error(f"작업 실행 중 오류: {e}")
                    if attempt == job.retry_count:
                        execution_result = ExecutionResult(
                            job_id=job.id,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            status=ScheduleStatus.FAILED,
                            error_message=str(e)
                        )
                        await self.notification_manager.send_job_failure_notification(job, execution_result)
            
            # 다음 실행 시간 계산
            job.next_run = self.schedule_calculator.calculate_next_run(job)
            
        except Exception as e:
            self.logger.error(f"작업 재시도 실행 오류: {e}")
        finally:
            # 실행 중인 작업에서 제거
            if job.id in self.job_manager.running_jobs:
                del self.job_manager.running_jobs[job.id]
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.running:
            try:
                # 6시간마다 상태 요약 전송
                if datetime.now().hour % 6 == 0 and datetime.now().minute < 5:
                    jobs_status = self.job_manager.get_all_jobs_status()
                    await self.notification_manager.send_schedule_summary(jobs_status)
                
                # 정지된 작업 정리
                await self._cleanup_finished_jobs()
                
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(300)
    
    async def _ai_monitoring_loop(self):
        """AI 모니터링 루프"""
        while self.running:
            try:
                # AI 관련 상태 모니터링
                if self.config.AI_ENABLED and self.job_executor.ai_manager:
                    # AI 시스템 상태 체크
                    await self._check_ai_system_health()
                
                await asyncio.sleep(1800)  # 30분마다
                
            except Exception as e:
                self.logger.error(f"AI 모니터링 루프 오류: {e}")
                await asyncio.sleep(1800)
    
    async def _check_ai_system_health(self):
        """AI 시스템 상태 체크"""
        try:
            if not self.job_executor.ai_manager.client:
                self.logger.warning("⚠️ AI 클라이언트가 초기화되지 않음")
                return
            
            # 간단한 AI 상태 테스트
            test_data = {"test": "system_health_check"}
            start_time = time.time()
            
            try:
                response = await self.job_executor.ai_manager.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                response_time = time.time() - start_time
                
                if response_time < 10:  # 10초 이내 응답
                    self.logger.debug(f"✅ AI 시스템 정상 (응답시간: {response_time:.1f}초)")
                else:
                    self.logger.warning(f"⚠️ AI 시스템 응답 지연 ({response_time:.1f}초)")
                    
            except Exception as e:
                self.logger.error(f"❌ AI 시스템 상태 체크 실패: {e}")
                
        except Exception as e:
            self.logger.error(f"AI 시스템 상태 체크 오류: {e}")
    
    async def _cleanup_loop(self):
        """정리 루프"""
        while self.running:
            try:
                # 매일 새벽 2시에 오래된 실행 기록 정리
                now = datetime.now()
                if now.hour == 2 and now.minute < 10:
                    await self._cleanup_old_execution_records()
                    await self._cleanup_old_ai_analyses()
                    await asyncio.sleep(600)  # 10분 대기
                
                await asyncio.sleep(300)  # 5분마다 체크
                
            except Exception as e:
                self.logger.error(f"정리 루프 오류: {e}")
                await asyncio.sleep(3600)  # 1시간 대기
    
    async def _cleanup_finished_jobs(self):
        """완료된 작업 정리"""
        try:
            finished_job_ids = []
            
            for job_id, task in self.job_manager.running_jobs.items():
                if task.done():
                    finished_job_ids.append(job_id)
            
            for job_id in finished_job_ids:
                del self.job_manager.running_jobs[job_id]
                
        except Exception as e:
            self.logger.error(f"완료된 작업 정리 실패: {e}")
    
    async def _cleanup_old_execution_records(self):
        """오래된 실행 기록 정리"""
        try:
            conn = sqlite3.connect(self.config.SCHEDULER_DB_PATH)
            cursor = conn.cursor()
            
            # 30일 이전 기록 삭제
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            
            cursor.execute('''
                DELETE FROM job_executions 
                WHERE start_time < ?
            ''', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                self.logger.info(f"🧹 오래된 실행 기록 {deleted_count}개 삭제")
                
        except Exception as e:
            self.logger.error(f"오래된 실행 기록 정리 실패: {e}")
    
    async def _cleanup_old_ai_analyses(self):
        """오래된 AI 분석 기록 정리"""
        try:
            conn = sqlite3.connect(self.config.SCHEDULER_DB_PATH)
            cursor = conn.cursor()
            
            # 60일 이전 AI 분석 기록 삭제
            cutoff_date = (datetime.now() - timedelta(days=60)).isoformat()
            
            cursor.execute('''
                DELETE FROM ai_analyses 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                self.logger.info(f"🧹 오래된 AI 분석 기록 {deleted_count}개 삭제")
                
        except Exception as e:
            self.logger.error(f"오래된 AI 분석 기록 정리 실패: {e}")
    
    async def _update_all_next_run_times(self):
        """모든 작업의 다음 실행 시간 업데이트"""
        try:
            for job in self.job_manager.jobs.values():
                if job.enabled:
                    job.next_run = self.schedule_calculator.calculate_next_run(job)
                    ai_indicator = "🤖" if job.ai_enabled else ""
                    self.logger.info(f"📅 다음 실행: {job.name} {ai_indicator} -> {job.next_run}")
            
        except Exception as e:
            self.logger.error(f"다음 실행 시간 업데이트 실패: {e}")
    
    async def shutdown(self):
        """스케줄러 종료"""
        try:
            self.logger.info("🛑 AI 스케줄러 종료 시작")
            
            # 실행 중인 모든 작업 종료 대기
            if self.job_manager.running_jobs:
                self.logger.info(f"⏳ 실행 중인 작업 {len(self.job_manager.running_jobs)}개 종료 대기...")
                
                for job_id, task in self.job_manager.running_jobs.items():
                    if not task.done():
                        task.cancel()
                
                # 최대 30초 대기
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.job_manager.running_jobs.values(), return_exceptions=True),
                        timeout=30
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("⚠️ 일부 작업이 시간 내에 종료되지 않음")
            
            # 스케줄러 루프 종료
            self.running = False
            
            if self.scheduler_task and not self.scheduler_task.done():
                self.scheduler_task.cancel()
            
            # 종료 알림
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            await self.notification_manager._send_notification(
                f"🛑 AI 스케줄러 종료\n"
                f"가동시간: {uptime}\n"
                f"🤖 AI 기능 포함",
                'info'
            )
            
            self.logger.info("✅ AI 스케줄러 종료 완료")
            
        except Exception as e:
            self.logger.error(f"스케줄러 종료 실패: {e}")
    
    # ========================================================================
    # 🎮 편의 메서드들
    # ========================================================================
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """스케줄러 상태 조회"""
        try:
            jobs_status = self.job_manager.get_all_jobs_status()
            market_status = self.market_manager.get_market_status_summary()
            ai_enabled_jobs = sum(1 for job in jobs_status if job.get('ai_enabled', False))
            
            return {
                'running': self.running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime': str(datetime.now() - self.start_time) if self.start_time else None,
                'total_jobs': len(self.job_manager.jobs),
                'enabled_jobs': sum(1 for job in jobs_status if job['enabled']),
                'running_jobs': len(self.job_manager.running_jobs),
                'ai_enabled_jobs': ai_enabled_jobs,
                'ai_available': OPENAI_AVAILABLE and self.config.AI_ENABLED,
                'market_status': market_status,
                'jobs': jobs_status
            }
            
        except Exception as e:
            self.logger.error(f"스케줄러 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    def get_ai_analysis_summary(self) -> Dict[str, Any]:
        """AI 분석 요약 조회"""
        try:
            conn = sqlite3.connect(self.config.SCHEDULER_DB_PATH)
            cursor = conn.cursor()
            
            # 최근 24시간 AI 분석 통계
            since_time = (datetime.now() - timedelta(hours=24)).isoformat()
            
            cursor.execute('''
                SELECT analysis_type, COUNT(*), AVG(confidence_score), AVG(execution_time)
                FROM ai_analyses 
                WHERE timestamp > ?
                GROUP BY analysis_type
            ''', (since_time,))
            
            analysis_stats = {}
            for row in cursor.fetchall():
                analysis_type, count, avg_confidence, avg_time = row
                analysis_stats[analysis_type] = {
                    'count': count,
                    'avg_confidence': round(avg_confidence, 2) if avg_confidence else 0,
                    'avg_execution_time': round(avg_time, 2) if avg_time else 0
                }
            
            # 최근 분석 결과
            cursor.execute('''
                SELECT analysis_type, timestamp, confidence_score, risk_level
                FROM ai_analyses 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (since_time,))
            
            recent_analyses = []
            for row in cursor.fetchall():
                analysis_type, timestamp, confidence, risk_level = row
                recent_analyses.append({
                    'type': analysis_type,
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'risk_level': risk_level
                })
            
            conn.close()
            
            return {
                'ai_enabled': self.config.AI_ENABLED,
                'ai_available': OPENAI_AVAILABLE,
                'analysis_stats_24h': analysis_stats,
                'recent_analyses': recent_analyses,
                'total_analyses_24h': sum(stats['count'] for stats in analysis_stats.values())
            }
            
        except Exception as e:
            self.logger.error(f"AI 분석 요약 조회 실패: {e}")
            return {'error': str(e)}

# ============================================================================
# 🔧 편의 함수들
# ============================================================================

async def get_scheduler_status():
    """스케줄러 상태 조회 (편의 함수)"""
    scheduler = QuantProjectScheduler()
    return scheduler.get_scheduler_status()

async def get_ai_analysis_summary():
    """AI 분석 요약 조회 (편의 함수)"""
    scheduler = QuantProjectScheduler()
    return scheduler.get_ai_analysis_summary()

def show_current_schedule():
    """현재 스케줄 출력"""
    try:
        config = SchedulerConfig()
        job_manager = JobManager(config)
        market_manager = MarketTimeManager(config)
        calculator = ScheduleCalculator(market_manager)
        
        print("📋 현재 등록된 스케줄:")
        print("=" * 80)
        
        for job in job_manager.jobs.values():
            if job.enabled:
                next_run = calculator.calculate_next_run(job)
                status = "🟢" if job.enabled else "🔴"
                ai_indicator = "🤖" if job.ai_enabled else "⚙️"
                
                print(f"{status} {ai_indicator} {job.name}")
                print(f"   📅 다음 실행: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   🎯 전략: {job.strategy}")
                print(f"   🌍 시장: {job.market_type.value}")
                print(f"   📊 실행 횟수: {job.run_count} (성공: {job.success_count})")
                if job.ai_enabled:
                    print(f"   🤖 AI 분석: 활성화")
                print()
        
    except Exception as e:
        print(f"❌ 스케줄 조회 실패: {e}")

def show_ai_status():
    """AI 상태 출력"""
    try:
        config = SchedulerConfig()
        
        print("🤖 AI 시스템 상태:")
        print("=" * 50)
        print(f"AI 활성화: {'✅' if config.AI_ENABLED else '❌'}")
        print(f"OpenAI 라이브러리: {'✅' if OPENAI_AVAILABLE else '❌'}")
        print(f"API 키 설정: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
        print(f"사용 모델: {config.AI_MODEL}")
        print(f"분석 간격: {config.AI_ANALYSIS_INTERVAL}시간")
        print(f"최대 토큰: {config.AI_MAX_TOKENS}")
        print(f"AI 알림: {'✅' if config.AI_NOTIFICATIONS else '❌'}")
        
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            print("\n🧠 지원되는 AI 분석:")
            print("   • 시장 센티먼트 분석")
            print("   • 포트폴리오 최적화")
            print("   • 위험도 평가")
            print("   • 트레이딩 전략 생성")
            print("   • 뉴스 영향도 분석")
        
    except Exception as e:
        print(f"❌ AI 상태 조회 실패: {e}")

def show_help():
    """도움말 출력"""
    help_text = """
🏆 퀸트프로젝트 AI 스케줄러 시스템 v1.0.0

📋 사용법:
  python scheduler.py                    # AI 스케줄러 시작
  python scheduler.py --show-schedule    # 현재 스케줄 보기
  python scheduler.py --show-ai          # AI 상태 보기
  python scheduler.py --help             # 도움말

🔧 주요 기능:
  • 4대 전략 자동 스케줄링
  • 시장 시간 자동 체크
  • 실패 시 자동 재시도
  • 텔레그램 실시간 알림
  • 실행 기록 추적
  • 🤖 OpenAI 기반 AI 분석
  • 🧠 자동 시장 센티먼트 분석
  • 📈 포트폴리오 AI 최적화
  • ⚠️ AI 위험도 평가
  • 📰 뉴스 영향도 분석

📊 스케줄:
  • 미국 전략: 화목 23:30 (서머타임 적용) + AI 분석
  • 일본 전략: 화목 08:45 + AI 분석
  • 인도 전략: 수요일 12:45 + AI 분석
  • 암호화폐: 월금 09:00 + AI 분석
  • AI 시장 분석: 6시간마다
  • AI 포트폴리오 최적화: 매일 자정
  • AI 위험 평가: 매일 오전 6시, 오후 6시
  • AI 뉴스 분석: 3시간마다

🤖 AI 설정:
  환경변수에 다음을 설정하세요:
  OPENAI_API_KEY=your_api_key
  AI_ENABLED=true
  AI_MODEL=gpt-4
  AI_NOTIFICATIONS=true

🔗 문의: 퀸트마스터팀
"""
    print(help_text)

# ============================================================================
# 🏁 메인 실행부
# ============================================================================

async def main():
    """메인 실행 함수"""
    import signal
    
    # 신호 핸들러 설정
    def signal_handler(signum, frame):
        print("\n🛑 종료 신호 수신, AI 스케줄러 종료 중...")
        asyncio.create_task(scheduler.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 스케줄러 생성
    scheduler = QuantProjectScheduler()
    
    try:
        print("🏆" + "="*70)
        print("🏆 퀸트프로젝트 AI 통합 스케줄러 시스템 v1.0.0")
        print("🏆" + "="*70)
        print("⏰ 시간 기반 자동 거래 스케줄링")
        print("📊 4대 전략 통합 관리")
        print("🌐 시장 시간 체크")
        print("🔄 자동 재시도")
        print("📋 실행 기록 추적")
        print("🔔 실시간 알림")
        print("📈 성과 모니터링")
        print("🤖 OpenAI 기반 AI 분석")
        print("🧠 자동 시장 센티먼트 분석")
        print("📈 포트폴리오 AI 최적화")
        print("⚠️ AI 위험도 평가")
        print("📰 뉴스 영향도 분석")
        print("🏆" + "="*70)
        
        # AI 상태 출력
        if OPENAI_AVAILABLE and scheduler.config.AI_ENABLED:
            print("\n🤖 AI 시스템:")
            print(f"  OpenAI 모델: {scheduler.config.AI_MODEL}")
            print(f"  API 키: {'✅ 설정됨' if os.getenv('OPENAI_API_KEY') else '❌ 미설정'}")
            print(f"  분석 간격: {scheduler.config.AI_ANALYSIS_INTERVAL}시간")
        else:
            print("\n⚠️ AI 시스템: 비활성화")
        
        # 현재 시장 상태 출력
        market_status = scheduler.market_manager.get_market_status_summary()
        print("\n📊 현재 시장 상태:")
        for market, is_open in market_status.items():
            status = "🟢 개장" if is_open else "🔴 폐장"
            print(f"  {market}: {status}")
        
        # 등록된 작업 출력
        jobs_status = scheduler.job_manager.get_all_jobs_status()
        ai_enabled_jobs = sum(1 for job in jobs_status if job.get('ai_enabled', False))
        print(f"\n📋 등록된 작업: {len(jobs_status)}개 (🤖 AI 활성화: {ai_enabled_jobs}개)")
        for job in jobs_status:
            status_icon = "✅" if job['enabled'] else "⏸️"
            ai_icon = "🤖" if job.get('ai_enabled', False) else "⚙️"
            next_run = job['next_run'].strftime('%m-%d %H:%M') if job['next_run'] else "미정"
            print(f"  {status_icon} {ai_icon} {job['name']}: {next_run}")
        
        print("\n🚀 AI 스케줄러 시작...\n")
        
        # 스케줄러 시작
        await scheduler.start_scheduler()
        
    except KeyboardInterrupt:
        print("\n👋 사용자 중단")
        await scheduler.shutdown()
    except Exception as e:
        print(f"\n❌ AI 스케줄러 오류: {e}")
        await scheduler.shutdown()

if __name__ == "__main__":
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--show-schedule":
            show_current_schedule()
        elif arg == "--show-ai":
            show_ai_status()
        elif arg == "--help" or arg == "-h":
            show_help()
        else:
            print(f"❌ 알 수 없는 옵션: {arg}")
            print("사용법: python scheduler.py --help")
    else:
        # 기본 실행: AI 스케줄러 시작
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 퀸트프로젝트 AI 스케줄러 종료")
            sys.exit(0)
        except Exception as e:
            print(f"\n💥 예상치 못한 오류: {e}")
            sys.exit(1) 실행 시간
                today_run = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # 오늘 실행 시간
                today_run = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # 오늘 시간이 지났으면 내일
                if current_time >= today_run:
                    next_run = today_run + timedelta(days=1)
                else:
                    next_run = today_run
                
                next_runs.append(next_run)
            
            # 가장 빠른 시간 반환
            return min(next_runs)
