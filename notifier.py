#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔔 퀸트프로젝트 통합 알림 시스템 (notifier.py)
=================================================
🏆 텔레그램 + 이메일 + SMS + 디스코드 + 슬랙 + OpenAI 통합 알림

✨ 핵심 기능:
- 다중 채널 통합 알림 시스템
- 우선순위별 알림 라우팅
- 템플릿 기반 메시지 포맷팅
- 알림 히스토리 관리
- 실패 시 백업 채널 자동 전환
- 스팸 방지 및 중복 제거
- 개인화된 알림 설정
- 알림 성능 모니터링
- OpenAI 스마트 메시지 생성

Author: 퀸트마스터팀
Version: 1.2.0 (멀티채널 + 스마트 라우팅 + OpenAI)
"""

import asyncio
import logging
import os
import json
import time
import smtplib
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.image import MimeImage
from dotenv import load_dotenv
import aiohttp
import openai
import sqlite3
import requests
from collections import defaultdict, deque

# ============================================================================
# 🎯 알림 설정 관리자
# ============================================================================
class NotifierConfig:
    """알림 시스템 설정 관리"""
    
    def __init__(self):
        load_dotenv()
        
        # 텔레그램 설정
        self.TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
        self.TELEGRAM_BACKUP_CHAT_ID = os.getenv('TELEGRAM_BACKUP_CHAT_ID', '')
        
        # 이메일 설정
        self.EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        self.EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', 587))
        self.EMAIL_USERNAME = os.getenv('EMAIL_USERNAME', '')
        self.EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
        self.EMAIL_TO_ADDRESS = os.getenv('EMAIL_TO_ADDRESS', '')
        self.EMAIL_FROM_NAME = os.getenv('EMAIL_FROM_NAME', '퀸트프로젝트')
        
        # 디스코드 설정
        self.DISCORD_ENABLED = os.getenv('DISCORD_ENABLED', 'false').lower() == 'true'
        self.DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
        
        # 슬랙 설정
        self.SLACK_ENABLED = os.getenv('SLACK_ENABLED', 'false').lower() == 'true'
        self.SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')
        self.SLACK_TOKEN = os.getenv('SLACK_TOKEN', '')
        self.SLACK_CHANNEL = os.getenv('SLACK_CHANNEL', '#quant-alerts')
        
        # SMS 설정 (Twilio)
        self.SMS_ENABLED = os.getenv('SMS_ENABLED', 'false').lower() == 'true'
        self.TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
        self.TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
        self.TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER', '')
        self.SMS_TO_NUMBER = os.getenv('SMS_TO_NUMBER', '')
        
        # 카카오톡 설정
        self.KAKAO_ENABLED = os.getenv('KAKAO_ENABLED', 'false').lower() == 'true'
        self.KAKAO_REST_API_KEY = os.getenv('KAKAO_REST_API_KEY', '')
        self.KAKAO_ACCESS_TOKEN = os.getenv('KAKAO_ACCESS_TOKEN', '')
        
        # OpenAI 설정
        self.OPENAI_ENABLED = os.getenv('OPENAI_ENABLED', 'false').lower() == 'true'
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        self.OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', 150))
        self.OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', 0.7))
        
        # 알림 제어 설정
        self.NOTIFICATION_COOLDOWN = int(os.getenv('NOTIFICATION_COOLDOWN', 60))  # 중복 방지 시간
        self.MAX_NOTIFICATIONS_PER_HOUR = int(os.getenv('MAX_NOTIFICATIONS_PER_HOUR', 50))
        self.EMERGENCY_BYPASS = os.getenv('EMERGENCY_BYPASS', 'true').lower() == 'true'
        
        # 우선순위별 채널 설정
        self.PRIORITY_CHANNELS = {
            'emergency': ['telegram', 'sms', 'email', 'openai'],
            'warning': ['telegram', 'discord', 'openai'],
            'info': ['telegram', 'openai'],
            'success': ['telegram'],
            'debug': ['discord']
        }
        
        # 데이터베이스
        self.DB_PATH = os.getenv('NOTIFIER_DB_PATH', './data/notifications.db')

# ============================================================================
# 📨 알림 메시지 데이터 클래스
# ============================================================================
@dataclass
class NotificationMessage:
    """알림 메시지 구조"""
    title: str
    content: str
    priority: str = 'info'  # emergency, warning, info, success, debug
    category: str = 'general'  # trading, system, portfolio, error
    timestamp: datetime = field(default_factory=datetime.now)
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    channels: List[str] = field(default_factory=list)  # 특정 채널 지정
    retry_count: int = 0
    hash_id: str = field(init=False)
    ai_enhanced: bool = False  # AI로 향상된 메시지인지 여부
    
    def __post_init__(self):
        # 메시지 해시 생성 (중복 방지용)
        message_data = f"{self.title}_{self.content}_{self.category}"
        self.hash_id = hashlib.md5(message_data.encode()).hexdigest()

# ============================================================================
# 📊 알림 히스토리 관리자
# ============================================================================
class NotificationHistory:
    """알림 기록 및 통계 관리"""
    
    def __init__(self, config: NotifierConfig):
        self.config = config
        self.db_path = config.DB_PATH
        self._init_database()
        
        # 메모리 캐시
        self.recent_notifications = deque(maxlen=1000)
        self.hourly_counts = defaultdict(int)
        self.failed_notifications = deque(maxlen=100)
        
        self.logger = logging.getLogger('NotificationHistory')
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 알림 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash_id TEXT,
                    title TEXT,
                    content TEXT,
                    priority TEXT,
                    category TEXT,
                    channels TEXT,
                    status TEXT,
                    timestamp DATETIME,
                    delivery_time REAL,
                    error_message TEXT,
                    ai_enhanced INTEGER DEFAULT 0
                )
            ''')
            
            # 채널별 성능 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS channel_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT,
                    date DATE,
                    sent_count INTEGER,
                    success_count INTEGER,
                    avg_delivery_time REAL,
                    error_count INTEGER
                )
            ''')
            
            # 설정 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_settings (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT,
                    created_at DATETIME,
                    updated_at DATETIME
                )
            ''')
            
            # AI 응답 캐시 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_response_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_hash TEXT UNIQUE,
                    input_text TEXT,
                    response_text TEXT,
                    model TEXT,
                    created_at DATETIME,
                    usage_count INTEGER DEFAULT 1
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def record_notification(self, message: NotificationMessage, channels: List[str], 
                           status: str, delivery_time: float = 0, error_msg: str = ''):
        """알림 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO notification_logs 
                (hash_id, title, content, priority, category, channels, status, 
                 timestamp, delivery_time, error_message, ai_enhanced)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.hash_id, message.title, message.content[:500], 
                message.priority, message.category, ','.join(channels),
                status, message.timestamp.isoformat(), delivery_time, error_msg,
                1 if message.ai_enhanced else 0
            ))
            
            conn.commit()
            conn.close()
            
            # 메모리 캐시 업데이트
            self.recent_notifications.append({
                'hash_id': message.hash_id,
                'timestamp': message.timestamp,
                'status': status
            })
            
        except Exception as e:
            self.logger.error(f"알림 기록 실패: {e}")
    
    def is_duplicate(self, message: NotificationMessage, window_minutes: int = 5) -> bool:
        """중복 알림 체크"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        # 메모리 캐시에서 빠른 체크
        for recent in self.recent_notifications:
            if (recent['hash_id'] == message.hash_id and 
                recent['timestamp'] > cutoff_time and
                recent['status'] == 'success'):
                return True
        
        return False
    
    def check_rate_limit(self) -> bool:
        """시간당 알림 제한 체크"""
        current_hour = datetime.now().hour
        current_count = self.hourly_counts[current_hour]
        
        return current_count < self.config.MAX_NOTIFICATIONS_PER_HOUR
    
    def increment_hourly_count(self):
        """시간당 카운트 증가"""
        current_hour = datetime.now().hour
        self.hourly_counts[current_hour] += 1
        
        # 이전 시간 데이터 정리
        for hour in list(self.hourly_counts.keys()):
            if hour != current_hour:
                del self.hourly_counts[hour]
    
    def cache_ai_response(self, input_text: str, response_text: str, model: str):
        """AI 응답 캐시 저장"""
        try:
            input_hash = hashlib.md5(input_text.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO ai_response_cache 
                (input_hash, input_text, response_text, model, created_at, usage_count)
                VALUES (?, ?, ?, ?, ?, 
                    COALESCE((SELECT usage_count + 1 FROM ai_response_cache WHERE input_hash = ?), 1))
            ''', (input_hash, input_text, response_text, model, datetime.now().isoformat(), input_hash))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"AI 응답 캐시 저장 실패: {e}")
    
    def get_cached_ai_response(self, input_text: str) -> Optional[str]:
        """AI 응답 캐시 조회"""
        try:
            input_hash = hashlib.md5(input_text.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT response_text FROM ai_response_cache 
                WHERE input_hash = ? AND created_at > ?
            ''', (input_hash, (datetime.now() - timedelta(hours=24)).isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"AI 응답 캐시 조회 실패: {e}")
            return None
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """알림 통계 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).date()
            
            # 기간별 통계
            cursor.execute('''
                SELECT priority, COUNT(*) as count, 
                       AVG(delivery_time) as avg_time,
                       SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                       SUM(ai_enhanced) as ai_enhanced_count
                FROM notification_logs 
                WHERE date(timestamp) >= ?
                GROUP BY priority
            ''', (start_date.isoformat(),))
            
            priority_stats = {}
            for row in cursor.fetchall():
                priority, count, avg_time, success_count, ai_enhanced_count = row
                success_rate = (success_count / count * 100) if count > 0 else 0
                ai_usage_rate = (ai_enhanced_count / count * 100) if count > 0 else 0
                
                priority_stats[priority] = {
                    'count': count,
                    'avg_delivery_time': avg_time or 0,
                    'success_rate': success_rate,
                    'ai_usage_rate': ai_usage_rate
                }
            
            # 채널별 통계
            cursor.execute('''
                SELECT channels, COUNT(*) as count,
                       AVG(delivery_time) as avg_time,
                       SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count
                FROM notification_logs 
                WHERE date(timestamp) >= ?
                GROUP BY channels
            ''', (start_date.isoformat(),))
            
            channel_stats = {}
            for row in cursor.fetchall():
                channels, count, avg_time, success_count = row
                success_rate = (success_count / count * 100) if count > 0 else 0
                
                channel_stats[channels] = {
                    'count': count,
                    'avg_delivery_time': avg_time or 0,
                    'success_rate': success_rate
                }
            
            conn.close()
            
            return {
                'period_days': days,
                'priority_stats': priority_stats,
                'channel_stats': channel_stats,
                'total_recent': len(self.recent_notifications)
            }
            
        except Exception as e:
            self.logger.error(f"통계 조회 실패: {e}")
            return {}

# ============================================================================
# 🤖 OpenAI 채널
# ============================================================================
class OpenAIChannel:
    """OpenAI 스마트 메시지 생성 및 개선 채널"""
    
    def __init__(self, config: NotifierConfig):
        self.config = config
        self.session = None
        self.logger = logging.getLogger('OpenAIChannel')
        
        # OpenAI 클라이언트 설정
        self.headers = {
            'Authorization': f'Bearer {self.config.OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # 메시지 유형별 프롬프트 템플릿
        self.prompts = {
            'enhance': """다음 알림 메시지를 더 명확하고 유용하게 개선해주세요. 
금융/투자 전문용어는 유지하되, 읽기 쉽고 액션 아이템이 명확하도록 작성하세요.

우선순위: {priority}
카테고리: {category}
원본 메시지: {content}

개선된 메시지로 답변해주세요 (한국어):""",
            
            'summarize': """다음 상세한 정보를 간결한 알림 메시지로 요약해주세요.
핵심 포인트만 포함하여 읽기 쉽게 작성하세요.

정보: {content}

요약된 알림 메시지 (한국어):""",
            
            'translate': """다음 메시지를 한국어로 번역하되, 금융/투자 용어는 적절히 현지화해주세요.

원본: {content}

번역된 메시지:""",
            
            'priority_analysis': """다음 알림 메시지의 우선순위를 분석하고 적절한 수준을 추천해주세요.
emergency, warning, info, success, debug 중 하나를 선택하고 이유를 설명하세요.

메시지: {content}

분석 결과 (JSON 형식):
{{"priority": "추천_우선순위", "reason": "선택_이유"}}""",
            
            'smart_notification': """퀸트프로젝트 투자 시스템을 위한 스마트 알림을 생성해주세요.
다음 정보를 바탕으로 유용하고 액션 가능한 알림 메시지를 작성하세요.

상황: {situation}
데이터: {data}
타겟 사용자: 개인투자자/퀀트 트레이더

알림 메시지:"""
        }
    
    async def send_message(self, message: NotificationMessage) -> Tuple[bool, str]:
        """OpenAI로 메시지 개선 (실제 전송은 하지 않음)"""
        if not self.config.OPENAI_ENABLED or not self.config.OPENAI_API_KEY:
            return False, "OpenAI 설정 없음"
        
        try:
            # 메시지 개선
            enhanced_content = await self.enhance_message(message)
            
            if enhanced_content:
                # 원본 메시지 업데이트
                message.content = enhanced_content
                message.ai_enhanced = True
                return True, "메시지 AI 개선 완료"
            else:
                return False, "메시지 개선 실패"
                
        except Exception as e:
            error_msg = f"OpenAI 처리 오류: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def enhance_message(self, message: NotificationMessage) -> Optional[str]:
        """메시지 개선"""
        try:
            # 캐시 확인
            cache_key = f"{message.priority}_{message.category}_{message.content}"
            cached_response = await self._get_cached_response(cache_key)
            
            if cached_response:
                self.logger.info("AI 응답 캐시에서 조회")
                return cached_response
            
            # OpenAI API 호출
            prompt = self.prompts['enhance'].format(
                priority=message.priority,
                category=message.category,
                content=message.content
            )
            
            enhanced_content = await self._call_openai_api(prompt)
            
            # 캐시 저장
            if enhanced_content:
                await self._cache_response(cache_key, enhanced_content)
            
            return enhanced_content
            
        except Exception as e:
            self.logger.error(f"메시지 개선 실패: {e}")
            return None
    
    async def analyze_priority(self, content: str) -> Dict[str, str]:
        """메시지 우선순위 AI 분석"""
        try:
            prompt = self.prompts['priority_analysis'].format(content=content)
            response = await self._call_openai_api(prompt)
            
            if response:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # JSON 파싱 실패시 기본값
                    return {"priority": "info", "reason": "분석 실패"}
            
            return {"priority": "info", "reason": "AI 응답 없음"}
            
        except Exception as e:
            self.logger.error(f"우선순위 분석 실패: {e}")
            return {"priority": "info", "reason": f"오류: {e}"}
    
    async def generate_smart_notification(self, situation: str, data: Dict[str, Any]) -> Optional[str]:
        """상황 기반 스마트 알림 생성"""
        try:
            data_str = json.dumps(data, ensure_ascii=False, indent=2)
            
            prompt = self.prompts['smart_notification'].format(
                situation=situation,
                data=data_str
            )
            
            return await self._call_openai_api(prompt)
            
        except Exception as e:
            self.logger.error(f"스마트 알림 생성 실패: {e}")
            return None
    
    async def translate_message(self, content: str) -> Optional[str]:
        """메시지 번역"""
        try:
            prompt = self.prompts['translate'].format(content=content)
            return await self._call_openai_api(prompt)
            
        except Exception as e:
            self.logger.error(f"메시지 번역 실패: {e}")
            return None
    
    async def summarize_content(self, content: str) -> Optional[str]:
        """내용 요약"""
        try:
            prompt = self.prompts['summarize'].format(content=content)
            return await self._call_openai_api(prompt)
            
        except Exception as e:
            self.logger.error(f"내용 요약 실패: {e}")
            return None
    
    async def _call_openai_api(self, prompt: str) -> Optional[str]:
        """OpenAI API 호출"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            payload = {
                "model": self.config.OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "당신은 퀸트프로젝트의 전문 투자 알림 어시스턴트입니다. 명확하고 유용한 메시지를 작성해주세요."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.OPENAI_MAX_TOKENS,
                "temperature": self.config.OPENAI_TEMPERATURE
            }
            
            client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
response = client.chat.completions.create(
    model=payload["model"],
    messages=payload["messages"],
    max_tokens=payload["max_tokens"],
    temperature=payload["temperature"]
)
content = response.choices[0].message.content.strip()
return content
                    
        except Exception as e:
            self.logger.error(f"OpenAI API 호출 실패: {e}")
            return None
    
    async def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """캐시된 응답 조회"""
        try:
            # 간단한 메모리 캐시 (실제로는 데이터베이스 사용)
            return None  # 캐시 미구현
        except Exception:
            return None
    
    async def _cache_response(self, cache_key: str, response: str):
        """응답 캐시 저장"""
        try:
            # 캐시 저장 로직 (실제로는 데이터베이스 사용)
            pass
        except Exception:
            pass
    
    async def close(self):
        """세션 종료"""
        if self.session:
            await self.session.close()

# ============================================================================
# 📡 텔레그램 채널
# ============================================================================
class TelegramChannel:
    """텔레그램 알림 채널"""
    
    def __init__(self, config: NotifierConfig):
        self.config = config
        self.session = None
        self.logger = logging.getLogger('TelegramChannel')
    
    async def send_message(self, message: NotificationMessage) -> Tuple[bool, str]:
        """텔레그램 메시지 전송"""
        if not self.config.TELEGRAM_ENABLED or not self.config.TELEGRAM_BOT_TOKEN:
            return False, "텔레그램 설정 없음"
        
        try:
            # 메시지 포맷팅
            formatted_message = await self._format_message(message)
            
            # 우선순위별 알림음 설정
            disable_notification = message.priority not in ['emergency', 'warning']
            
            # 메인 채널로 전송
            success = await self._send_to_chat(
                self.config.TELEGRAM_CHAT_ID, 
                formatted_message, 
                disable_notification
            )
            
            # 응급상황시 백업 채널도 사용
            if message.priority == 'emergency' and self.config.TELEGRAM_BACKUP_CHAT_ID:
                await self._send_to_chat(
                    self.config.TELEGRAM_BACKUP_CHAT_ID,
                    f"🚨 백업 알림\n\n{formatted_message}",
                    False
                )
            
            return success, "전송 완료" if success else "전송 실패"
            
        except Exception as e:
            error_msg = f"텔레그램 전송 오류: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def _send_to_chat(self, chat_id: str, message: str, disable_notification: bool) -> bool:
        """특정 채팅방으로 메시지 전송"""
        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            data = {
                'chat_id': chat_id,
                'text': message,
                'disable_notification': disable_notification,
                'parse_mode': 'HTML'
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(url, json=data, timeout=10) as response:
                if response.status == 200:
                    return True
                else:
                    self.logger.error(f"텔레그램 API 오류: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"텔레그램 전송 실패: {e}")
            return False
    
    async def _format_message(self, message: NotificationMessage) -> str:
        """텔레그램용 메시지 포맷팅"""
        # 우선순위별 이모지
        priority_emojis = {
            'emergency': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅',
            'debug': '🔧'
        }
        
        # 카테고리별 이모지
        category_emojis = {
            'trading': '📈',
            'system': '🖥️',
            'portfolio': '💼',
            'error': '❌',
            'general': '📊'
        }
        
        emoji = priority_emojis.get(message.priority, '📊')
        cat_emoji = category_emojis.get(message.category, '📊')
        
        formatted = f"{emoji} <b>퀸트프로젝트 알림</b> {cat_emoji}\n\n"
        formatted += f"📋 <b>{message.title}</b>\n\n"
        formatted += f"{message.content}\n\n"
        
        # AI 개선 표시
        if message.ai_enhanced:
            formatted += "🤖 <i>AI로 개선된 메시지</i>\n\n"
        
        # 메타데이터 추가
        if message.metadata:
            formatted += "📄 <b>추가 정보:</b>\n"
            for key, value in message.metadata.items():
                formatted += f"  • {key}: {value}\n"
            formatted += "\n"
        
        # 타임스탬프
        formatted += f"🕐 {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return formatted
    
    async def close(self):
        """세션 종료"""
        if self.session:
            await self.session.close()

# ============================================================================
# 📧 이메일 채널
# ============================================================================
class EmailChannel:
    """이메일 알림 채널"""
    
    def __init__(self, config: NotifierConfig):
        self.config = config
        self.logger = logging.getLogger('EmailChannel')
    
    async def send_message(self, message: NotificationMessage) -> Tuple[bool, str]:
        """이메일 메시지 전송"""
        if not self.config.EMAIL_ENABLED or not self.config.EMAIL_USERNAME:
            return False, "이메일 설정 없음"
        
        try:
            # 메시지 생성
            msg = await self._create_email(message)
            
            # SMTP 전송
            success = await self._send_smtp(msg)
            
            return success, "전송 완료" if success else "전송 실패"
            
        except Exception as e:
            error_msg = f"이메일 전송 오류: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def _create_email(self, message: NotificationMessage) -> MimeMultipart:
        """이메일 메시지 생성"""
        msg = MimeMultipart('alternative')
        
        # 제목 설정
        priority_prefix = {
            'emergency': '[🚨 응급]',
            'warning': '[⚠️ 경고]',
            'info': '[ℹ️ 정보]',
            'success': '[✅ 성공]',
            'debug': '[🔧 디버그]'
        }
        
        subject_prefix = priority_prefix.get(message.priority, '[📊]')
        ai_suffix = " (AI 개선)" if message.ai_enhanced else ""
        msg['Subject'] = f"{subject_prefix} {message.title}{ai_suffix}"
        msg['From'] = f"{self.config.EMAIL_FROM_NAME} <{self.config.EMAIL_USERNAME}>"
        msg['To'] = self.config.EMAIL_TO_ADDRESS
        
        # HTML 본문 생성
        html_body = await self._create_html_body(message)
        msg.attach(MimeText(html_body, 'html', 'utf-8'))
        
        # 텍스트 본문 생성
        text_body = await self._create_text_body(message)
        msg.attach(MimeText(text_body, 'plain', 'utf-8'))
        
        return msg
    
    async def _create_html_body(self, message: NotificationMessage) -> str:
        """HTML 이메일 본문 생성"""
        # 우선순위별 색상
        priority_colors = {
            'emergency': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'success': '#28a745',
            'debug': '#6c757d'
        }
        
        color = priority_colors.get(message.priority, '#17a2b8')
        ai_badge = '<span style="background-color: #6f42c1; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">🤖 AI 개선</span>' if message.ai_enhanced else ''
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: #f8f9fa; padding: 20px; }}
                .metadata {{ background-color: #e9ecef; padding: 15px; margin-top: 15px; }}
                .footer {{ text-align: center; padding: 10px; color: #6c757d; font-size: 12px; }}
                .ai-badge {{ margin-top: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏆 퀸트프로젝트 알림</h1>
                    <h2>{message.title}</h2>
                    <div class="ai-badge">{ai_badge}</div>
                </div>
                <div class="content">
                    <p>{message.content.replace(chr(10), '<br>')}</p>
        """
        
        # 메타데이터 추가
        if message.metadata:
            html += '<div class="metadata"><h3>추가 정보</h3><ul>'
            for key, value in message.metadata.items():
                html += f'<li><strong>{key}:</strong> {value}</li>'
            html += '</ul></div>'
        
        html += f"""
                </div>
                <div class="footer">
                    <p>발송시간: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>우선순위: {message.priority.upper()} | 카테고리: {message.category}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    async def _create_text_body(self, message: NotificationMessage) -> str:
        """텍스트 이메일 본문 생성"""
        text = f"🏆 퀸트프로젝트 알림\n\n"
        text += f"제목: {message.title}\n\n"
        text += f"{message.content}\n\n"
        
        if message.ai_enhanced:
            text += "🤖 AI로 개선된 메시지\n\n"
        
        if message.metadata:
            text += "추가 정보:\n"
            for key, value in message.metadata.items():
                text += f"  - {key}: {value}\n"
            text += "\n"
        
        text += f"발송시간: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"우선순위: {message.priority.upper()} | 카테고리: {message.category}\n"
        
        return text
    
    async def _send_smtp(self, msg: MimeMultipart) -> bool:
        """SMTP로 이메일 전송"""
        try:
            # asyncio에서 동기 코드 실행
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self._smtp_send_sync, msg)
            return success
            
        except Exception as e:
            self.logger.error(f"SMTP 전송 실패: {e}")
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
            self.logger.error(f"SMTP 동기 전송 실패: {e}")
            return False

# ============================================================================
# 🎮 디스코드 채널
# ============================================================================
class DiscordChannel:
    """디스코드 알림 채널"""
    
    def __init__(self, config: NotifierConfig):
        self.config = config
        self.session = None
        self.logger = logging.getLogger('DiscordChannel')
    
    async def send_message(self, message: NotificationMessage) -> Tuple[bool, str]:
        """디스코드 메시지 전송"""
        if not self.config.DISCORD_ENABLED or not self.config.DISCORD_WEBHOOK_URL:
            return False, "디스코드 설정 없음"
        
        try:
            # 임베드 메시지 생성
            embed = await self._create_embed(message)
            
            # 웹훅으로 전송
            success = await self._send_webhook(embed)
            
            return success, "전송 완료" if success else "전송 실패"
            
        except Exception as e:
            error_msg = f"디스코드 전송 오류: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def _create_embed(self, message: NotificationMessage) -> Dict[str, Any]:
        """디스코드 임베드 메시지 생성"""
        # 우선순위별 색상
        priority_colors = {
            'emergency': 0xff0000,  # 빨강
            'warning': 0xffa500,    # 주황
            'info': 0x0099ff,       # 파랑
            'success': 0x00ff00,    # 초록
            'debug': 0x808080       # 회색
        }
        
        color = priority_colors.get(message.priority, 0x0099ff)
        
        title = f"🏆 {message.title}"
        if message.ai_enhanced:
            title += " 🤖"
        
        embed = {
            "title": title,
            "description": message.content,
            "color": color,
            "timestamp": message.timestamp.isoformat(),
            "footer": {
                "text": f"우선순위: {message.priority.upper()} | 카테고리: {message.category}" + (" | AI 개선" if message.ai_enhanced else "")
            }
        }
        
        # 메타데이터를 필드로 추가
        if message.metadata:
            embed["fields"] = []
            for key, value in message.metadata.items():
                embed["fields"].append({
                    "name": key,
                    "value": str(value),
                    "inline": True
                })
        
        return embed
    
    async def _send_webhook(self, embed: Dict[str, Any]) -> bool:
        """웹훅으로 메시지 전송"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            data = {
                "username": "퀸트프로젝트",
                "embeds": [embed]
            }
            
            async with self.session.post(
                self.config.DISCORD_WEBHOOK_URL, 
                json=data, 
                timeout=10
            ) as response:
                return response.status == 204
                
        except Exception as e:
            self.logger.error(f"디스코드 웹훅 전송 실패: {e}")
            return False
    
    async def close(self):
        """세션 종료"""
        if self.session:
            await self.session.close()

# ============================================================================
# 📱 슬랙 채널
# ============================================================================
class SlackChannel:
    """슬랙 알림 채널"""
    
    def __init__(self, config: NotifierConfig):
        self.config = config
        self.session = None
        self.logger = logging.getLogger('SlackChannel')
    
    async def send_message(self, message: NotificationMessage) -> Tuple[bool, str]:
        """슬랙 메시지 전송"""
        if not self.config.SLACK_ENABLED or not self.config.SLACK_WEBHOOK_URL:
            return False, "슬랙 설정 없음"
        
        try:
            # 슬랙 메시지 생성
            slack_message = await self._create_slack_message(message)
            
            # 웹훅으로 전송
            success = await self._send_webhook(slack_message)
            
            return success, "전송 완료" if success else "전송 실패"
            
        except Exception as e:
            error_msg = f"슬랙 전송 오류: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def _create_slack_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """슬랙 메시지 생성"""
        # 우선순위별 색상
        priority_colors = {
            'emergency': 'danger',
            'warning': 'warning',
            'info': 'good',
            'success': 'good',
            'debug': '#808080'
        }
        
        color = priority_colors.get(message.priority, 'good')
        
        title = f"🏆 {message.title}"
        if message.ai_enhanced:
            title += " 🤖"
        
        slack_message = {
            "channel": self.config.SLACK_CHANNEL,
            "username": "퀸트프로젝트",
            "icon_emoji": ":chart_with_upwards_trend:",
            "attachments": [{
                "color": color,
                "title": title,
                "text": message.content,
                "footer": f"우선순위: {message.priority.upper()} | 카테고리: {message.category}" + (" | AI 개선" if message.ai_enhanced else ""),
                "ts": int(message.timestamp.timestamp())
            }]
        }
        
        # 메타데이터를 필드로 추가
        if message.metadata:
            slack_message["attachments"][0]["fields"] = []
            for key, value in message.metadata.items():
                slack_message["attachments"][0]["fields"].append({
                    "title": key,
                    "value": str(value),
                    "short": True
                })
        
        return slack_message
    
    async def _send_webhook(self, slack_message: Dict[str, Any]) -> bool:
        """웹훅으로 메시지 전송"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                self.config.SLACK_WEBHOOK_URL, 
                json=slack_message, 
                timeout=10
            ) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"슬랙 웹훅 전송 실패: {e}")
            return False
    
    async def close(self):
        """세션 종료"""
        if self.session:
            await self.session.close()

# ============================================================================
# 📱 SMS 채널 (Twilio)
# ============================================================================
class SMSChannel:
    """SMS 알림 채널"""
    
    def __init__(self, config: NotifierConfig):
        self.config = config
        self.logger = logging.getLogger('SMSChannel')
    
    async def send_message(self, message: NotificationMessage) -> Tuple[bool, str]:
        """SMS 메시지 전송"""
        if not self.config.SMS_ENABLED or not self.config.TWILIO_ACCOUNT_SID:
            return False, "SMS 설정 없음"
        
        try:
            # SMS용 짧은 메시지 생성
            sms_text = await self._create_sms_text(message)
            
            # Twilio API로 전송
            success = await self._send_twilio_sms(sms_text)
            
            return success, "전송 완료" if success else "전송 실패"
            
        except Exception as e:
            error_msg = f"SMS 전송 오류: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def _create_sms_text(self, message: NotificationMessage) -> str:
        """SMS용 텍스트 생성 (160자 제한)"""
        priority_emojis = {
            'emergency': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅',
            'debug': '🔧'
        }
        
        emoji = priority_emojis.get(message.priority, '📊')
        
        # 짧은 메시지 구성
        sms_text = f"{emoji} {message.title[:30]}"
        
        # AI 표시
        if message.ai_enhanced:
            sms_text += " 🤖"
        
        # 내용 요약 (100자 이내)
        content_summary = message.content[:90]
        if len(message.content) > 90:
            content_summary += "..."
        
        sms_text += f"\n{content_summary}"
        
        # 시간 추가
        time_str = message.timestamp.strftime('%H:%M')
        sms_text += f"\n{time_str}"
        
        return sms_text
    
    async def _send_twilio_sms(self, text: str) -> bool:
        """Twilio API로 SMS 전송"""
        try:
            # Twilio 라이브러리가 없으면 HTTP 요청으로 대체
            try:
                from twilio.rest import Client
                
                client = Client(self.config.TWILIO_ACCOUNT_SID, self.config.TWILIO_AUTH_TOKEN)
                
                # asyncio에서 동기 코드 실행
                loop = asyncio.get_event_loop()
                message = await loop.run_in_executor(
                    None, 
                    lambda: client.messages.create(
                        body=text,
                        from_=self.config.TWILIO_FROM_NUMBER,
                        to=self.config.SMS_TO_NUMBER
                    )
                )
                
                return message.sid is not None
                
            except ImportError:
                # HTTP 요청으로 직접 전송
                return await self._send_twilio_http(text)
                
        except Exception as e:
            self.logger.error(f"Twilio SMS 전송 실패: {e}")
            return False
    
    async def _send_twilio_http(self, text: str) -> bool:
        """HTTP 요청으로 Twilio SMS 전송"""
        try:
            import base64
            
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.config.TWILIO_ACCOUNT_SID}/Messages.json"
            
            # 인증 헤더
            credentials = f"{self.config.TWILIO_ACCOUNT_SID}:{self.config.TWILIO_AUTH_TOKEN}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {
                'From': self.config.TWILIO_FROM_NUMBER,
                'To': self.config.SMS_TO_NUMBER,
                'Body': text
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=data, timeout=10) as response:
                    return response.status == 201
                    
        except Exception as e:
            self.logger.error(f"Twilio HTTP 전송 실패: {e}")
            return False

# ============================================================================
# 💬 카카오톡 채널
# ============================================================================
class KakaoChannel:
    """카카오톡 알림 채널"""
    
    def __init__(self, config: NotifierConfig):
        self.config = config
        self.session = None
        self.logger = logging.getLogger('KakaoChannel')
    
    async def send_message(self, message: NotificationMessage) -> Tuple[bool, str]:
        """카카오톡 메시지 전송"""
        if not self.config.KAKAO_ENABLED or not self.config.KAKAO_ACCESS_TOKEN:
            return False, "카카오톡 설정 없음"
        
        try:
            # 카카오톡 메시지 생성
            kakao_message = await self._create_kakao_message(message)
            
            # 카카오톡 API로 전송
            success = await self._send_kakao_api(kakao_message)
            
            return success, "전송 완료" if success else "전송 실패"
            
        except Exception as e:
            error_msg = f"카카오톡 전송 오류: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def _create_kakao_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """카카오톡 메시지 생성"""
        priority_emojis = {
            'emergency': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅',
            'debug': '🔧'
        }
        
        emoji = priority_emojis.get(message.priority, '📊')
        
        # 텍스트 메시지 구성
        text = f"{emoji} 퀸트프로젝트 알림\n\n"
        text += f"📋 {message.title}\n\n"
        text += f"{message.content}\n\n"
        
        if message.ai_enhanced:
            text += "🤖 AI로 개선된 메시지\n\n"
        
        text += f"🕐 {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        kakao_message = {
            "object_type": "text",
            "text": text,
            "link": {
                "web_url": "https://github.com/your-repo",
                "mobile_web_url": "https://github.com/your-repo"
            }
        }
        
        return kakao_message
    
    async def _send_kakao_api(self, kakao_message: Dict[str, Any]) -> bool:
        """카카오톡 API로 메시지 전송"""
        try:
            url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
            
            headers = {
                'Authorization': f'Bearer {self.config.KAKAO_ACCESS_TOKEN}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {
                'template_object': json.dumps(kakao_message, ensure_ascii=False)
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(url, headers=headers, data=data, timeout=10) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"카카오톡 API 전송 실패: {e}")
            return False
    
    async def close(self):
        """세션 종료"""
        if self.session:
            await self.session.close()

# ============================================================================
# 🎯 메시지 템플릿 관리자
# ============================================================================
class MessageTemplateManager:
    """메시지 템플릿 관리"""
    
    def __init__(self):
        self.templates = {
            'trading_signal': {
                'title': '🎯 거래 신호',
                'format': '{strategy} 전략에서 {action} 신호가 발생했습니다.\n\n종목: {symbol}\n가격: {price}\n수량: {quantity}'
            },
            'portfolio_alert': {
                'title': '💼 포트폴리오 알림',
                'format': '포트폴리오 {alert_type}이 발생했습니다.\n\n현재 손익: {pnl}\n총 가치: {total_value}\n위험도: {risk_level}'
            },
            'system_error': {
                'title': '❌ 시스템 오류',
                'format': '{component}에서 오류가 발생했습니다.\n\n오류 내용: {error_message}\n발생 시간: {timestamp}'
            },
            'network_status': {
                'title': '🌐 네트워크 상태',
                'format': '네트워크 연결 상태가 {status}로 변경되었습니다.\n\n상세 정보: {details}'
            },
            'performance_report': {
                'title': '📈 성과 보고서',
                'format': '{period} 성과 요약\n\n총 수익률: {return_rate}\n거래 횟수: {trade_count}\n승률: {win_rate}'
            },
            'ai_analysis': {
                'title': '🤖 AI 분석 결과',
                'format': '{analysis_type} 분석이 완료되었습니다.\n\n주요 발견사항: {findings}\n추천 행동: {recommendations}'
            }
        }
    
    def create_message_from_template(self, template_name: str, **kwargs) -> NotificationMessage:
        """템플릿으로 메시지 생성"""
        if template_name not in self.templates:
            raise ValueError(f"템플릿 '{template_name}'을 찾을 수 없습니다")
        
        template = self.templates[template_name]
        
        # 템플릿 포맷팅
        title = template['title']
        content = template['format'].format(**kwargs)
        
        # 우선순위 자동 결정
        priority = 'info'
        if 'error' in template_name or 'alert' in template_name:
            priority = 'warning'
        elif 'emergency' in kwargs.get('alert_type', ''):
            priority = 'emergency'
        elif 'signal' in template_name:
            priority = 'success'
        
        # 카테고리 자동 결정
        category = 'general'
        if 'trading' in template_name:
            category = 'trading'
        elif 'portfolio' in template_name:
            category = 'portfolio'
        elif 'system' in template_name or 'network' in template_name:
            category = 'system'
        elif 'error' in template_name:
            category = 'error'
        elif 'ai' in template_name:
            category = 'ai'
        
        return NotificationMessage(
            title=title,
            content=content,
            priority=priority,
            category=category,
            metadata=kwargs
        )

# ============================================================================
# 🏆 통합 알림 관리자
# ============================================================================
class UnifiedNotificationManager:
    """통합 알림 관리 시스템"""
    
    def __init__(self, config: NotifierConfig = None):
        self.config = config or NotifierConfig()
        
        # 로깅 설정
        self.logger = logging.getLogger('UnifiedNotificationManager')
        
        # 컴포넌트 초기화
        self.history = NotificationHistory(self.config)
        self.template_manager = MessageTemplateManager()
        
        # 채널 인스턴스
        self.channels = {
            'telegram': TelegramChannel(self.config),
            'email': EmailChannel(self.config),
            'discord': DiscordChannel(self.config),
            'slack': SlackChannel(self.config),
            'sms': SMSChannel(self.config),
            'kakao': KakaoChannel(self.config),
            'openai': OpenAIChannel(self.config)
        }
        
        # 활성화된 채널 확인
        self.active_channels = self._get_active_channels()
        
        self.logger.info(f"✅ 통합 알림 시스템 초기화 완료 (활성 채널: {', '.join(self.active_channels)})")
    
    def _get_active_channels(self) -> List[str]:
        """활성화된 채널 목록"""
        active = []
        
        if self.config.TELEGRAM_ENABLED and self.config.TELEGRAM_BOT_TOKEN:
            active.append('telegram')
        if self.config.EMAIL_ENABLED and self.config.EMAIL_USERNAME:
            active.append('email')
        if self.config.DISCORD_ENABLED and self.config.DISCORD_WEBHOOK_URL:
            active.append('discord')
        if self.config.SLACK_ENABLED and self.config.SLACK_WEBHOOK_URL:
            active.append('slack')
        if self.config.SMS_ENABLED and self.config.TWILIO_ACCOUNT_SID:
            active.append('sms')
        if self.config.KAKAO_ENABLED and self.config.KAKAO_ACCESS_TOKEN:
            active.append('kakao')
        if self.config.OPENAI_ENABLED and self.config.OPENAI_API_KEY:
            active.append('openai')
        
        return active
    
    async def send_notification(self, message: Union[NotificationMessage, str], 
                              priority: str = 'info', category: str = 'general',
                              title: str = '알림', metadata: Dict[str, Any] = None,
                              channels: List[str] = None, use_ai: bool = True) -> Dict[str, bool]:
        """통합 알림 전송"""
        
        # 문자열이면 NotificationMessage로 변환
        if isinstance(message, str):
            message = NotificationMessage(
                title=title,
                content=message,
                priority=priority,
                category=category,
                metadata=metadata or {}
            )
        
        # AI 개선 적용 (OpenAI 활성화 시)
        if use_ai and 'openai' in self.active_channels and not message.ai_enhanced:
            try:
                openai_channel = self.channels['openai']
                success, _ = await openai_channel.send_message(message)
                if success:
                    self.logger.info("메시지가 AI로 개선되었습니다")
            except Exception as e:
                self.logger.warning(f"AI 개선 실패, 원본 메시지 사용: {e}")
        
        # 중복 체크
        if self.history.is_duplicate(message):
            self.logger.debug(f"중복 알림 무시: {message.hash_id}")
            return {}
        
        # 속도 제한 체크 (응급상황 제외)
        if message.priority != 'emergency' and not self.history.check_rate_limit():
            self.logger.warning("시간당 알림 제한 초과")
            return {}
        
        # 전송할 채널 결정
        target_channels = channels or message.channels or self.config.PRIORITY_CHANNELS.get(message.priority, ['telegram'])
        # OpenAI는 실제 전송 채널이 아니므로 제외
        target_channels = [ch for ch in target_channels if ch in self.active_channels and ch != 'openai']
        
        if not target_channels:
            self.logger.warning("전송 가능한 채널이 없습니다")
            return {}
        
        # 채널별 전송
        results = {}
        start_time = time.time()
        
        for channel_name in target_channels:
            try:
                channel = self.channels[channel_name]
                success, error_msg = await channel.send_message(message)
                results[channel_name] = success
                
                if not success:
                    self.logger.error(f"{channel_name} 전송 실패: {error_msg}")
                
            except Exception as e:
                results[channel_name] = False
                self.logger.error(f"{channel_name} 전송 예외: {e}")
        
        # 전송 결과 기록
        delivery_time = time.time() - start_time
        success_count = sum(results.values())
        
        if success_count > 0:
            self.history.record_notification(
                message, target_channels, 'success', delivery_time
            )
            self.history.increment_hourly_count()
        else:
            self.history.record_notification(
                message, target_channels, 'failed', delivery_time, 
                f"모든 채널 실패: {results}"
            )
        
        self.logger.info(f"알림 전송 완료: {success_count}/{len(target_channels)} 성공")
        return results
    
    async def send_template_notification(self, template_name: str, use_ai: bool = True, **kwargs) -> Dict[str, bool]:
        """템플릿 기반 알림 전송"""
        try:
            message = self.template_manager.create_message_from_template(template_name, **kwargs)
            return await self.send_notification(message, use_ai=use_ai)
        except Exception as e:
            self.logger.error(f"템플릿 알림 전송 실패: {e}")
            return {}
    
    async def send_emergency_notification(self, title: str, content: str, 
                                        metadata: Dict[str, Any] = None) -> Dict[str, bool]:
        """응급 알림 전송 (모든 채널)"""
        # OpenAI 제외한 모든 활성 채널
        emergency_channels = [ch for ch in self.active_channels if ch != 'openai']
        
        message = NotificationMessage(
            title=title,
            content=content,
            priority='emergency',
            category='system',
            metadata=metadata or {},
            channels=emergency_channels
        )
        
        return await self.send_notification(message)
    
    async def send_ai_enhanced_notification(self, content: str, priority: str = 'info',
                                          title: str = 'AI 분석 알림') -> Dict[str, bool]:
        """AI로 개선된 알림 전송"""
        if 'openai' not in self.active_channels:
            self.logger.warning("OpenAI가 비활성화되어 일반 알림으로 전송")
            return await self.send_notification(content, priority=priority, title=title, use_ai=False)
        
        # AI로 우선순위 분석
        openai_channel = self.channels['openai']
        priority_analysis = await openai_channel.analyze_priority(content)
        analyzed_priority = priority_analysis.get('priority', priority)
        
        message = NotificationMessage(
            title=title,
            content=content,
            priority=analyzed_priority,
            category='ai',
            metadata={'ai_analysis': priority_analysis}
        )
        
        return await self.send_notification(message, use_ai=True)
    
    async def generate_smart_notification(self, situation: str, data: Dict[str, Any]) -> Dict[str, bool]:
        """상황 기반 스마트 알림 생성 및 전송"""
        if 'openai' not in self.active_channels:
            self.logger.error("OpenAI가 비활성화되어 스마트 알림을 생성할 수 없습니다")
            return {}
        
        try:
            openai_channel = self.channels['openai']
            smart_content = await openai_channel.generate_smart_notification(situation, data)
            
            if smart_content:
                message = NotificationMessage(
                    title=f"🤖 스마트 알림: {situation}",
                    content=smart_content,
                    priority='info',
                    category='ai',
                    metadata=data,
                    ai_enhanced=True
                )
                
                return await self.send_notification(message, use_ai=False)  # 이미 AI로 생성됨
            else:
                self.logger.error("스마트 알림 생성 실패")
                return {}
                
        except Exception as e:
            self.logger.error(f"스마트 알림 생성 오류: {e}")
            return {}
    
    async def test_all_channels(self) -> Dict[str, bool]:
        """모든 채널 테스트"""
        test_message = NotificationMessage(
            title="🧪 알림 시스템 테스트",
            content="이것은 퀸트프로젝트 알림 시스템의 테스트 메시지입니다.",
            priority='debug',
            category='system'
        )
        
        results = {}
        # OpenAI 제외한 실제 전송 채널만 테스트
        test_channels = [ch for ch in self.active_channels if ch != 'openai']
        
        for channel_name in test_channels:
            try:
                channel = self.channels[channel_name]
                success, error_msg = await channel.send_message(test_message)
                results[channel_name] = success
                
                if success:
                    self.logger.info(f"✅ {channel_name} 테스트 성공")
                else:
                    self.logger.error(f"❌ {channel_name} 테스트 실패: {error_msg}")
                    
            except Exception as e:
                results[channel_name] = False
                self.logger.error(f"❌ {channel_name} 테스트 예외: {e}")
        
        # OpenAI 기능 테스트
        if 'openai' in self.active_channels:
            try:
                openai_channel = self.channels['openai']
                test_content = await openai_channel.enhance_message(test_message)
                results['openai'] = test_content is not None
                
                if results['openai']:
                    self.logger.info("✅ OpenAI 테스트 성공")
                else:
                    self.logger.error("❌ OpenAI 테스트 실패")
                    
            except Exception as e:
                results['openai'] = False
                self.logger.error(f"❌ OpenAI 테스트 예외: {e}")
        
        return results
    
    async def get_ai_statistics(self) -> Dict[str, Any]:
        """AI 사용 통계 조회"""
        if 'openai' not in self.active_channels:
            return {"error": "OpenAI 비활성화"}
        
        try:
            stats = self.get_statistics(7)
            
            # AI 관련 통계 추가
            ai_stats = {
                "ai_enabled": True,
                "ai_model": self.config.OPENAI_MODEL,
                "ai_usage_by_priority": {},
                "total_ai_enhanced": 0
            }
            
            # 우선순위별 AI 사용률 계산
            for priority, data in stats.get('priority_stats', {}).items():
                ai_stats["ai_usage_by_priority"][priority] = data.get('ai_usage_rate', 0)
                ai_stats["total_ai_enhanced"] += data.get('count', 0) * data.get('ai_usage_rate', 0) / 100
            
            stats.update(ai_stats)
            return stats
            
        except Exception as e:
            self.logger.error(f"AI 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """알림 통계 조회"""
        return self.history.get_statistics(days)
    
    async def close(self):
        """리소스 정리"""
        for channel in self.channels.values():
            if hasattr(channel, 'close'):
                await channel.close()
        
        self.logger.info("✅ 알림 시스템 종료 완료")

# ============================================================================
# 🎮 편의 함수들
# ============================================================================
# 전역 알림 관리자 인스턴스
_global_notifier = None

def get_notifier() -> UnifiedNotificationManager:
    """전역 알림 관리자 인스턴스 반환"""
    global _global_notifier
    if _global_notifier is None:
        _global_notifier = UnifiedNotificationManager()
    return _global_notifier

async def send_quick_notification(message: str, priority: str = 'info', 
                                title: str = '퀸트프로젝트 알림', use_ai: bool = True) -> Dict[str, bool]:
    """빠른 알림 전송"""
    notifier = get_notifier()
    return await notifier.send_notification(message, priority=priority, title=title, use_ai=use_ai)

async def send_trading_signal(strategy: str, action: str, symbol: str, 
                            price: float, quantity: int, use_ai: bool = True) -> Dict[str, bool]:
    """거래 신호 알림"""
    notifier = get_notifier()
    return await notifier.send_template_notification(
        'trading_signal',
        use_ai=use_ai,
        strategy=strategy,
        action=action,
        symbol=symbol,
        price=price,
        quantity=quantity
    )

async def send_portfolio_alert(alert_type: str, pnl: float, total_value: float, 
                             risk_level: str, use_ai: bool = True) -> Dict[str, bool]:
    """포트폴리오 알림"""
    notifier = get_notifier()
    return await notifier.send_template_notification(
        'portfolio_alert',
        use_ai=use_ai,
        alert_type=alert_type,
        pnl=pnl,
        total_value=total_value,
        risk_level=risk_level
    )

async def send_system_error(component: str, error_message: str, use_ai: bool = True) -> Dict[str, bool]:
    """시스템 오류 알림"""
    notifier = get_notifier()
    return await notifier.send_template_notification(
        'system_error',
        use_ai=use_ai,
        component=component,
        error_message=error_message,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

async def send_emergency_alert(title: str, content: str) -> Dict[str, bool]:
    """응급 알림"""
    notifier = get_notifier()
    return await notifier.send_emergency_notification(title, content)

async def send_ai_analysis_result(analysis_type: str, findings: str, 
                                recommendations: str) -> Dict[str, bool]:
    """AI 분석 결과 알림"""
    notifier = get_notifier()
    return await notifier.send_template_notification(
        'ai_analysis',
        use_ai=True,
        analysis_type=analysis_type,
        findings=findings,
        recommendations=recommendations
    )

async def generate_smart_alert(situation: str, **data) -> Dict[str, bool]:
    """스마트 알림 생성"""
    notifier = get_notifier()
    return await notifier.generate_smart_notification(situation, data)

# ============================================================================
# 🏁 메인 실행부 (테스트용)
# ============================================================================
async def main():
    """메인 테스트 함수"""
    print("🔔" + "="*70)
    print("🔔 퀸트프로젝트 통합 알림 시스템 v1.2.0")
    print("🔔" + "="*70)
    print("✨ 다중 채널 통합 알림")
    print("✨ 우선순위별 라우팅")
    print("✨ 템플릿 기반 메시지")
    print("✨ 알림 히스토리 관리")
    print("✨ 스팸 방지 시스템")
    print("✨ 성능 모니터링")
    print("🤖 OpenAI 스마트 메시지 생성")
    print("🔔" + "="*70)
    
    # 알림 관리자 생성
    notifier = UnifiedNotificationManager()
    
    try:
        # 채널 테스트
        print("\n🧪 채널 테스트 시작...")
        test_results = await notifier.test_all_channels()
        
        for channel, success in test_results.items():
            status = "✅ 성공" if success else "❌ 실패"
            print(f"  {channel}: {status}")
        
        # 다양한 알림 테스트
        print("\n📨 알림 테스트 시작...")
        
        # 일반 알림 (AI 개선 포함)
        await send_quick_notification("시스템이 정상적으로 시작되었습니다.", 'success', use_ai=True)
        
        # 거래 신호 알림 (AI 개선 포함)
        await send_trading_signal('미국전략', 'BUY', 'AAPL', 150.25, 100, use_ai=True)
        
        # 포트폴리오 알림
        await send_portfolio_alert('수익 달성', 1500000, 50000000, '낮음', use_ai=True)
        
        # AI 분석 결과 알림
        await send_ai_analysis_result(
            '시장 동향 분석',
            '기술주 섹터의 상승 모멘텀이 지속되고 있습니다.',
            'AAPL, MSFT 등 대형 기술주 비중 확대를 권장합니다.'
        )
        
        # 스마트 알림 생성 테스트
        if 'openai' in notifier.active_channels:
            print("\n🤖 스마트 알림 생성 테스트...")
            await generate_smart_alert(
                '포트폴리오 리밸런싱 필요',
                current_allocation={'TECH': 60, 'FINANCE': 25, 'ENERGY': 15},
                target_allocation={'TECH': 50, 'FINANCE': 30, 'ENERGY': 20},
                total_value=100000000
            )
        
        # AI 통계 조회
        print("\n📊 AI 알림 통계:")
        ai_stats = await notifier.get_ai_statistics()
        print(json.dumps(ai_stats, indent=2, ensure_ascii=False))
        
        # 일반 통계 조회
        print("\n📈 전체 알림 통계:")
        stats = notifier.get_statistics(7)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("\n👋 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        await notifier.close()
        print("\n✅ 알림 시스템 종료")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 알림 시스템 테스트 종료")
        import sys
        sys.exit(0)
