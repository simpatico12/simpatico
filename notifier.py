#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 통합 알림 시스템 (notifier.py)
=================================================
텔레그램 + 이메일 + 슬랙 + 디스코드 + SMS 통합 알림

✨ 핵심 기능:
- 다중 채널 알림 (텔레그램, 이메일, 슬랙, 디스코드, SMS)
- 알림 레벨별 자동 라우팅 (INFO, WARNING, CRITICAL)
- 중복 알림 방지 시스템
- 실패 시 대안 채널 자동 전환
- 알림 통계 및 성공률 추적
- 템플릿 기반 메시지 포맷팅
- 첨부파일 지원 (이미지, 문서)

Author: 퀸트마스터팀
Version: 1.0.0
"""

import asyncio
import aiohttp
import smtplib
import logging
import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.image import MimeImage
from email.mime.base import MimeBase
from email import encoders
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import requests
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('notifier.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 📱 알림 레벨 및 데이터 클래스
# ============================================================================

class NotificationLevel(Enum):
    """알림 레벨"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class NotificationChannel(Enum):
    """알림 채널"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    SMS = "sms"
    WEBHOOK = "webhook"

@dataclass
class NotificationConfig:
    """알림 설정"""
    # 텔레그램
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # 이메일
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # 슬랙
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = ""
    slack_bot_token: str = ""
    
    # 디스코드
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    
    # SMS (Twilio)
    sms_enabled: bool = False
    sms_account_sid: str = ""
    sms_auth_token: str = ""
    sms_from_number: str = ""
    sms_to_numbers: List[str] = field(default_factory=list)
    
    # 일반 웹훅
    webhook_enabled: bool = False
    webhook_urls: List[str] = field(default_factory=list)
    
    # 레벨별 채널 설정
    level_channels: Dict[str, List[str]] = field(default_factory=dict)
    
    # 중복 방지 설정
    duplicate_prevention: bool = True
    duplicate_window_minutes: int = 5
    
    # 재시도 설정
    max_retries: int = 3
    retry_delay_seconds: int = 5

@dataclass
class NotificationMessage:
    """알림 메시지"""
    title: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    channels: Optional[List[NotificationChannel]] = None
    attachments: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None

@dataclass
class NotificationResult:
    """알림 결과"""
    channel: NotificationChannel
    success: bool
    message: str
    timestamp: datetime
    retry_count: int = 0
    response_data: Optional[Dict] = None

# ============================================================================
# 🔧 알림 설정 관리자
# ============================================================================

class NotificationConfigManager:
    """알림 설정 관리"""
    
    def __init__(self, config_file: str = "notification_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> NotificationConfig:
        """설정 파일 로드"""
        try:
            # 파일에서 로드
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return NotificationConfig(**data)
            
            # 환경변수에서 로드
            return self._load_from_env()
            
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            return NotificationConfig()
    
    def _load_from_env(self) -> NotificationConfig:
        """환경변수에서 설정 로드"""
        config = NotificationConfig(
            # 텔레그램
            telegram_enabled=os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true',
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID', ''),
            
            # 이메일
            email_enabled=os.getenv('EMAIL_ENABLED', 'false').lower() == 'true',
            email_smtp_server=os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
            email_smtp_port=int(os.getenv('EMAIL_SMTP_PORT', '587')),
            email_username=os.getenv('EMAIL_USERNAME', ''),
            email_password=os.getenv('EMAIL_PASSWORD', ''),
            email_to=os.getenv('EMAIL_TO', '').split(',') if os.getenv('EMAIL_TO') else [],
            
            # 슬랙
            slack_enabled=os.getenv('SLACK_ENABLED', 'false').lower() == 'true',
            slack_webhook_url=os.getenv('SLACK_WEBHOOK_URL', ''),
            slack_channel=os.getenv('SLACK_CHANNEL', '#general'),
            slack_bot_token=os.getenv('SLACK_BOT_TOKEN', ''),
            
            # 디스코드
            discord_enabled=os.getenv('DISCORD_ENABLED', 'false').lower() == 'true',
            discord_webhook_url=os.getenv('DISCORD_WEBHOOK_URL', ''),
            
            # SMS
            sms_enabled=os.getenv('SMS_ENABLED', 'false').lower() == 'true',
            sms_account_sid=os.getenv('SMS_ACCOUNT_SID', ''),
            sms_auth_token=os.getenv('SMS_AUTH_TOKEN', ''),
            sms_from_number=os.getenv('SMS_FROM_NUMBER', ''),
            sms_to_numbers=os.getenv('SMS_TO_NUMBERS', '').split(',') if os.getenv('SMS_TO_NUMBERS') else [],
            
            # 웹훅
            webhook_enabled=os.getenv('WEBHOOK_ENABLED', 'false').lower() == 'true',
            webhook_urls=os.getenv('WEBHOOK_URLS', '').split(',') if os.getenv('WEBHOOK_URLS') else [],
            
            # 레벨별 채널 설정
            level_channels={
                'debug': ['telegram'],
                'info': ['telegram', 'slack'],
                'warning': ['telegram', 'email', 'slack'],
                'critical': ['telegram', 'email', 'slack', 'discord'],
                'emergency': ['telegram', 'email', 'slack', 'discord', 'sms']
            }
        )
        
        # 설정 파일 저장
        self.save_config(config)
        return config
    
    def save_config(self, config: NotificationConfig):
        """설정 파일 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config.__dict__, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"✅ 설정 저장: {self.config_file}")
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")
    
    def update_config(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config(self.config)

# ============================================================================
# 📊 알림 통계 관리자
# ============================================================================

class NotificationStatsManager:
    """알림 통계 관리"""
    
    def __init__(self, db_path: str = "notification_stats.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 알림 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT,
                    title TEXT,
                    level TEXT,
                    channel TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    timestamp DATETIME,
                    response_time_ms INTEGER
                )
            ''')
            
            # 중복 방지 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS duplicate_prevention (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_hash TEXT UNIQUE,
                    first_sent DATETIME,
                    count INTEGER DEFAULT 1
                )
            ''')
            
            # 채널별 통계 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS channel_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT,
                    date DATE,
                    total_sent INTEGER DEFAULT 0,
                    successful_sent INTEGER DEFAULT 0,
                    failed_sent INTEGER DEFAULT 0,
                    avg_response_time_ms REAL DEFAULT 0,
                    UNIQUE(channel, date)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 알림 통계 DB 초기화 완료")
            
        except Exception as e:
            logger.error(f"통계 DB 초기화 실패: {e}")
    
    def log_notification(self, message: NotificationMessage, result: NotificationResult, response_time_ms: int):
        """알림 로그 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO notification_logs 
                (message_id, title, level, channel, success, error_message, retry_count, timestamp, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.message_id, message.title, message.level.value,
                result.channel.value, result.success, result.message,
                result.retry_count, result.timestamp.isoformat(), response_time_ms
            ))
            
            # 일일 통계 업데이트
            today = datetime.now().date()
            cursor.execute('''
                INSERT OR IGNORE INTO channel_stats (channel, date) VALUES (?, ?)
            ''', (result.channel.value, today))
            
            cursor.execute('''
                UPDATE channel_stats SET 
                    total_sent = total_sent + 1,
                    successful_sent = successful_sent + ?,
                    failed_sent = failed_sent + ?,
                    avg_response_time_ms = (avg_response_time_ms * (total_sent - 1) + ?) / total_sent
                WHERE channel = ? AND date = ?
            ''', (
                1 if result.success else 0,
                0 if result.success else 1,
                response_time_ms,
                result.channel.value, today
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"알림 로그 기록 실패: {e}")
    
    def check_duplicate(self, message: NotificationMessage, window_minutes: int = 5) -> bool:
        """중복 메시지 체크"""
        try:
            # 메시지 해시 생성
            message_content = f"{message.title}:{message.message}:{message.level.value}"
            message_hash = hashlib.md5(message_content.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 최근 중복 체크
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            cursor.execute('''
                SELECT count, first_sent FROM duplicate_prevention 
                WHERE message_hash = ? AND first_sent > ?
            ''', (message_hash, cutoff_time.isoformat()))
            
            result = cursor.fetchone()
            
            if result:
                # 중복 발견 - 카운트 증가
                cursor.execute('''
                    UPDATE duplicate_prevention SET count = count + 1 
                    WHERE message_hash = ?
                ''', (message_hash,))
                conn.commit()
                conn.close()
                return True
            else:
                # 새 메시지 - 기록
                cursor.execute('''
                    INSERT OR REPLACE INTO duplicate_prevention (message_hash, first_sent, count)
                    VALUES (?, ?, 1)
                ''', (message_hash, datetime.now().isoformat()))
                conn.commit()
                conn.close()
                return False
                
        except Exception as e:
            logger.error(f"중복 체크 실패: {e}")
            return False
    
    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """통계 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now().date() - timedelta(days=days)
            
            # 채널별 통계
            cursor.execute('''
                SELECT channel, 
                       SUM(total_sent) as total,
                       SUM(successful_sent) as success,
                       SUM(failed_sent) as failed,
                       AVG(avg_response_time_ms) as avg_response
                FROM channel_stats 
                WHERE date > ?
                GROUP BY channel
            ''', (cutoff_date,))
            
            channel_stats = {}
            for row in cursor.fetchall():
                channel_stats[row[0]] = {
                    'total_sent': row[1],
                    'successful_sent': row[2],
                    'failed_sent': row[3],
                    'success_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0,
                    'avg_response_time_ms': row[4] or 0
                }
            
            # 레벨별 통계
            cursor.execute('''
                SELECT level, COUNT(*) as count
                FROM notification_logs 
                WHERE date(timestamp) > ?
                GROUP BY level
            ''', (cutoff_date,))
            
            level_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                'period_days': days,
                'channel_stats': channel_stats,
                'level_stats': level_stats,
                'total_notifications': sum(level_stats.values()),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}

# ============================================================================
# 📤 개별 채널 핸들러
# ============================================================================

class TelegramHandler:
    """텔레그램 알림 핸들러"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send_message(self, message: NotificationMessage) -> NotificationResult:
        """텔레그램 메시지 전송"""
        start_time = time.time()
        
        try:
            # 메시지 포맷팅
            level_emoji = {
                NotificationLevel.DEBUG: '🔍',
                NotificationLevel.INFO: '💡',
                NotificationLevel.WARNING: '⚠️',
                NotificationLevel.CRITICAL: '🚨',
                NotificationLevel.EMERGENCY: '🆘'
            }
            
            emoji = level_emoji.get(message.level, '📢')
            formatted_message = f"{emoji} <b>{message.title}</b>\n\n{message.message}\n\n⏰ {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            data = {
                'chat_id': self.chat_id,
                'text': formatted_message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/sendMessage", json=data, timeout=30) as response:
                    response_data = await response.json()
                    
                    if response.status == 200 and response_data.get('ok'):
                        return NotificationResult(
                            channel=NotificationChannel.TELEGRAM,
                            success=True,
                            message="전송 성공",
                            timestamp=datetime.now(),
                            response_data=response_data
                        )
                    else:
                        return NotificationResult(
                            channel=NotificationChannel.TELEGRAM,
                            success=False,
                            message=f"API 오류: {response_data.get('description', 'Unknown error')}",
                            timestamp=datetime.now(),
                            response_data=response_data
                        )
            
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.TELEGRAM,
                success=False,
                message=f"전송 실패: {str(e)}",
                timestamp=datetime.now()
            )

class EmailHandler:
    """이메일 알림 핸들러"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, to_addresses: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.to_addresses = to_addresses
    
    async def send_message(self, message: NotificationMessage) -> NotificationResult:
        """이메일 메시지 전송"""
        try:
            # 메시지 구성
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.to_addresses)
            msg['Subject'] = f"[{message.level.value.upper()}] {message.title}"
            
            # 본문 작성
            body = f"""
{message.title}

{message.message}

---
알림 레벨: {message.level.value.upper()}
발생 시간: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
시스템: 퀸트프로젝트 알림 시스템
"""
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            # 첨부파일 처리
            if message.attachments:
                for attachment_path in message.attachments:
                    await self._add_attachment(msg, attachment_path)
            
            # 비동기 전송
            await asyncio.get_event_loop().run_in_executor(None, self._send_email_sync, msg)
            
            return NotificationResult(
                channel=NotificationChannel.EMAIL,
                success=True,
                message="이메일 전송 성공",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.EMAIL,
                success=False,
                message=f"이메일 전송 실패: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _send_email_sync(self, msg):
        """동기 이메일 전송"""
        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        server.starttls()
        server.login(self.username, self.password)
        server.send_message(msg)
        server.quit()
    
    async def _add_attachment(self, msg, file_path: str):
        """첨부파일 추가"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return
            
            with open(file_path, 'rb') as f:
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                    # 이미지 첨부
                    img_data = f.read()
                    image = MimeImage(img_data)
                    image.add_header('Content-Disposition', f'attachment; filename={file_path.name}')
                    msg.attach(image)
                else:
                    # 일반 파일 첨부
                    part = MimeBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename={file_path.name}')
                    msg.attach(part)
                    
        except Exception as e:
            logger.error(f"첨부파일 추가 실패 {file_path}: {e}")

class SlackHandler:
    """슬랙 알림 핸들러"""
    
    def __init__(self, webhook_url: str, channel: str = "#general"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_message(self, message: NotificationMessage) -> NotificationResult:
        """슬랙 메시지 전송"""
        try:
            # 레벨별 색상
            level_colors = {
                NotificationLevel.DEBUG: "#36a64f",     # 녹색
                NotificationLevel.INFO: "#36a64f",      # 녹색
                NotificationLevel.WARNING: "#ff9500",   # 주황색
                NotificationLevel.CRITICAL: "#ff0000",  # 빨간색
                NotificationLevel.EMERGENCY: "#8B0000"  # 진한 빨간색
            }
            
            color = level_colors.get(message.level, "#36a64f")
            
            payload = {
                "channel": self.channel,
                "username": "QuintProject Bot",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"[{message.level.value.upper()}] {message.title}",
                        "text": message.message,
                        "footer": "퀸트프로젝트 알림 시스템",
                        "ts": int(message.timestamp.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        return NotificationResult(
                            channel=NotificationChannel.SLACK,
                            success=True,
                            message="슬랙 전송 성공",
                            timestamp=datetime.now()
                        )
                    else:
                        error_text = await response.text()
                        return NotificationResult(
                            channel=NotificationChannel.SLACK,
                            success=False,
                            message=f"슬랙 전송 실패: {error_text}",
                            timestamp=datetime.now()
                        )
            
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.SLACK,
                success=False,
                message=f"슬랙 전송 오류: {str(e)}",
                timestamp=datetime.now()
            )

class DiscordHandler:
    """디스코드 알림 핸들러"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_message(self, message: NotificationMessage) -> NotificationResult:
        """디스코드 메시지 전송"""
        try:
            # 레벨별 색상 (10진수)
            level_colors = {
                NotificationLevel.DEBUG: 3066993,      # 녹색
                NotificationLevel.INFO: 3066993,       # 녹색
                NotificationLevel.WARNING: 16753920,   # 주황색
                NotificationLevel.CRITICAL: 16711680,  # 빨간색
                NotificationLevel.EMERGENCY: 9109504   # 진한 빨간색
            }
            
            color = level_colors.get(message.level, 3066993)
            
            payload = {
                "username": "QuintProject Bot",
                "avatar_url": "https://i.imgur.com/4M34hi2.png",
                "embeds": [
                    {
                        "title": f"[{message.level.value.upper()}] {message.title}",
                        "description": message.message,
                        "color": color,
                        "footer": {
                            "text": "퀸트프로젝트 알림 시스템"
                        },
                        "timestamp": message.timestamp.isoformat()
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=30) as response:
                    if response.status == 204:  # Discord는 204 No Content 반환
                        return NotificationResult(
                            channel=NotificationChannel.DISCORD,
                            success=True,
                            message="디스코드 전송 성공",
                            timestamp=datetime.now()
                        )
                    else:
                        error_text = await response.text()
                        return NotificationResult(
                            channel=NotificationChannel.DISCORD,
                            success=False,
                            message=f"디스코드 전송 실패: {error_text}",
                            timestamp=datetime.now()
                        )
            
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.DISCORD,
                success=False,
                message=f"디스코드 전송 오류: {str(e)}",
                timestamp=datetime.now()
            )

class SMSHandler:
    """SMS 알림 핸들러 (Twilio)"""
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str, to_numbers: List[str]):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers
    
    async def send_message(self, message: NotificationMessage) -> NotificationResult:
        """SMS 메시지 전송"""
        try:
            # SMS는 짧게 요약
            sms_text = f"[{message.level.value.upper()}] {message.title}\n{message.message[:100]}{'...' if len(message.message) > 100 else ''}"
            
            success_count = 0
            errors = []
            
            for to_number in self.to_numbers:
                try:
                    await self._send_single_sms(to_number, sms_text)
                    success_count += 1
                except Exception as e:
                    errors.append(f"{to_number}: {str(e)}")
            
            if success_count > 0:
                return NotificationResult(
                    channel=NotificationChannel.SMS,
                    success=True,
                    message=f"SMS 전송 성공: {success_count}/{len(self.to_numbers)}",
                    timestamp=datetime.now()
                )
            else:
                return NotificationResult(
                    channel=NotificationChannel.SMS,
                    success=False,
                    message=f"SMS 전송 실패: {'; '.join(errors)}",
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.SMS,
                success=False,
                message=f"SMS 전송 오류: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _send_single_sms(self, to_number: str, message_text: str):
        """개별 SMS 전송"""
        try:
            from twilio.rest import Client
            
            client = Client(self.account_sid, self.auth_token)
            
            # 비동기로 Twilio API 호출
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    body=message_text,
                    from_=self.from_number,
                    to=to_number
                )
            )
            
        except Exception as e:
            raise Exception(f"Twilio SMS 전송 실패: {str(e)}")

class WebhookHandler:
    """일반 웹훅 알림 핸들러"""
    
    def __init__(self, webhook_urls: List[str]):
        self.webhook_urls = webhook_urls
    
    async def send_message(self, message: NotificationMessage) -> NotificationResult:
        """웹훅 메시지 전송"""
        try:
            payload = {
                "title": message.title,
                "message": message.message,
                "level": message.level.value,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata or {}
            }
            
            success_count = 0
            errors = []
            
            for webhook_url in self.webhook_urls:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(webhook_url, json=payload, timeout=30) as response:
                            if response.status == 200:
                                success_count += 1
                            else:
                                errors.append(f"{webhook_url}: HTTP {response.status}")
                except Exception as e:
                    errors.append(f"{webhook_url}: {str(e)}")
            
            if success_count > 0:
                return NotificationResult(
                    channel=NotificationChannel.WEBHOOK,
                    success=True,
                    message=f"웹훅 전송 성공: {success_count}/{len(self.webhook_urls)}",
                    timestamp=datetime.now()
                )
            else:
                return NotificationResult(
                    channel=NotificationChannel.WEBHOOK,
                    success=False,
                    message=f"웹훅 전송 실패: {'; '.join(errors)}",
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.WEBHOOK,
                success=False,
                message=f"웹훅 전송 오류: {str(e)}",
                timestamp=datetime.now()
            )

# ============================================================================
# 🏆 통합 알림 관리자
# ============================================================================

class QuintNotificationManager:
    """퀸트프로젝트 통합 알림 관리자"""
    
    def __init__(self, config_file: str = "notification_config.json"):
        self.config_manager = NotificationConfigManager(config_file)
        self.stats_manager = NotificationStatsManager()
        self.handlers = {}
        self._init_handlers()
        
        logger.info("🏆 퀸트프로젝트 통합 알림 시스템 초기화 완료")
    
    def _init_handlers(self):
        """핸들러 초기화"""
        config = self.config_manager.config
        
        # 텔레그램
        if config.telegram_enabled and config.telegram_bot_token and config.telegram_chat_id:
            self.handlers[NotificationChannel.TELEGRAM] = TelegramHandler(
                config.telegram_bot_token, config.telegram_chat_id
            )
            logger.info("✅ 텔레그램 핸들러 초기화")
        
        # 이메일
        if config.email_enabled and config.email_username and config.email_password and config.email_to:
            self.handlers[NotificationChannel.EMAIL] = EmailHandler(
                config.email_smtp_server, config.email_smtp_port,
                config.email_username, config.email_password, config.email_to
            )
            logger.info(f"✅ 이메일 핸들러 초기화 ({len(config.email_to)}개 주소)")
        
        # 슬랙
        if config.slack_enabled and config.slack_webhook_url:
            self.handlers[NotificationChannel.SLACK] = SlackHandler(
                config.slack_webhook_url, config.slack_channel
            )
            logger.info("✅ 슬랙 핸들러 초기화")
        
        # 디스코드
        if config.discord_enabled and config.discord_webhook_url:
            self.handlers[NotificationChannel.DISCORD] = DiscordHandler(
                config.discord_webhook_url
            )
            logger.info("✅ 디스코드 핸들러 초기화")
        
        # SMS
        if config.sms_enabled and config.sms_account_sid and config.sms_auth_token:
            self.handlers[NotificationChannel.SMS] = SMSHandler(
                config.sms_account_sid, config.sms_auth_token,
                config.sms_from_number, config.sms_to_numbers
            )
            logger.info(f"✅ SMS 핸들러 초기화 ({len(config.sms_to_numbers)}개 번호)")
        
        # 웹훅
        if config.webhook_enabled and config.webhook_urls:
            self.handlers[NotificationChannel.WEBHOOK] = WebhookHandler(
                config.webhook_urls
            )
            logger.info(f"✅ 웹훅 핸들러 초기화 ({len(config.webhook_urls)}개 URL)")
        
        logger.info(f"🎯 활성화된 채널: {list(self.handlers.keys())}")
    
    async def send_notification(self, title: str, message: str, 
                              level: NotificationLevel = NotificationLevel.INFO,
                              channels: Optional[List[NotificationChannel]] = None,
                              attachments: Optional[List[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[NotificationChannel, NotificationResult]:
        """통합 알림 전송"""
        
        # 메시지 객체 생성
        notification = NotificationMessage(
            title=title,
            message=message,
            level=level,
            channels=channels,
            attachments=attachments,
            metadata=metadata,
            message_id=self._generate_message_id(title, message, level)
        )
        
        # 중복 체크
        if self.config_manager.config.duplicate_prevention:
            if self.stats_manager.check_duplicate(notification, 
                                                 self.config_manager.config.duplicate_window_minutes):
                logger.info(f"⚠️ 중복 알림 감지, 스킵: {title}")
                return {}
        
        # 채널 결정
        target_channels = self._determine_channels(notification)
        
        # 알림 전송
        results = {}
        for channel in target_channels:
            if channel in self.handlers:
                result = await self._send_with_retry(notification, channel)
                results[channel] = result
                
                # 통계 기록
                response_time = int((result.timestamp - notification.timestamp).total_seconds() * 1000)
                self.stats_manager.log_notification(notification, result, response_time)
        
        # 결과 로깅
        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        
        if success_count == total_count:
            logger.info(f"✅ 알림 전송 완료: {title} ({success_count}/{total_count} 성공)")
        elif success_count > 0:
            logger.warning(f"⚠️ 부분 전송 완료: {title} ({success_count}/{total_count} 성공)")
        else:
            logger.error(f"❌ 알림 전송 실패: {title} (모든 채널 실패)")
        
        return results
    
    def _generate_message_id(self, title: str, message: str, level: NotificationLevel) -> str:
        """메시지 ID 생성"""
        content = f"{title}:{message}:{level.value}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _determine_channels(self, notification: NotificationMessage) -> List[NotificationChannel]:
        """대상 채널 결정"""
        # 명시적 채널 지정이 있으면 사용
        if notification.channels:
            return [ch for ch in notification.channels if ch in self.handlers]
        
        # 레벨별 기본 채널 사용
        level_channels = self.config_manager.config.level_channels.get(notification.level.value, [])
        channels = []
        
        for channel_name in level_channels:
            try:
                channel = NotificationChannel(channel_name)
                if channel in self.handlers:
                    channels.append(channel)
            except ValueError:
                continue
        
        return channels
    
    async def _send_with_retry(self, notification: NotificationMessage, 
                              channel: NotificationChannel) -> NotificationResult:
        """재시도 로직이 포함된 전송"""
        handler = self.handlers[channel]
        max_retries = self.config_manager.config.max_retries
        retry_delay = self.config_manager.config.retry_delay_seconds
        
        for attempt in range(max_retries + 1):
            try:
                result = await handler.send_message(notification)
                result.retry_count = attempt
                
                if result.success:
                    return result
                
                # 실패 시 재시도 (마지막 시도가 아닌 경우)
                if attempt < max_retries:
                    logger.warning(f"🔄 {channel.value} 재시도 {attempt + 1}/{max_retries}: {result.message}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"❌ {channel.value} 최종 실패: {result.message}")
                    return result
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"🔄 {channel.value} 예외 재시도 {attempt + 1}/{max_retries}: {str(e)}")
                    await asyncio.sleep(retry_delay)
                else:
                    return NotificationResult(
                        channel=channel,
                        success=False,
                        message=f"최종 실패: {str(e)}",
                        timestamp=datetime.now(),
                        retry_count=attempt
                    )
        
        # 이론적으로 도달하지 않음
        return NotificationResult(
            channel=channel,
            success=False,
            message="알 수 없는 오류",
            timestamp=datetime.now(),
            retry_count=max_retries
        )
    
    # ========================================================================
    # 🎯 편의 메서드들
    # ========================================================================
    
    async def send_info(self, title: str, message: str, **kwargs):
        """정보 알림 전송"""
        return await self.send_notification(title, message, NotificationLevel.INFO, **kwargs)
    
    async def send_warning(self, title: str, message: str, **kwargs):
        """경고 알림 전송"""
        return await self.send_notification(title, message, NotificationLevel.WARNING, **kwargs)
    
    async def send_critical(self, title: str, message: str, **kwargs):
        """중요 알림 전송"""
        return await self.send_notification(title, message, NotificationLevel.CRITICAL, **kwargs)
    
    async def send_emergency(self, title: str, message: str, **kwargs):
        """긴급 알림 전송"""
        return await self.send_notification(title, message, NotificationLevel.EMERGENCY, **kwargs)
    
    async def send_trade_alert(self, symbol: str, action: str, price: float, quantity: float, 
                              strategy: str, profit_loss: Optional[float] = None):
        """거래 알림 전송"""
        emoji = "📈" if action.upper() == "BUY" else "📉"
        
        message = f"""
{emoji} 거래 실행

종목: {symbol}
액션: {action.upper()}
가격: {price:,.2f}
수량: {quantity:,.2f}
전략: {strategy}
"""
        
        if profit_loss is not None:
            pnl_emoji = "💰" if profit_loss > 0 else "💸"
            message += f"{pnl_emoji} 손익: {profit_loss:+,.2f}\n"
        
        message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_info(f"거래 실행: {symbol}", message)
    
    async def send_portfolio_summary(self, total_value: float, total_pnl: float, 
                                   total_return_pct: float, positions_count: int):
        """포트폴리오 요약 알림"""
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        
        message = f"""
💼 포트폴리오 현황

총 가치: ${total_value:,.2f}
{pnl_emoji} 손익: ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)
포지션 수: {positions_count}개

업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        level = NotificationLevel.INFO if total_pnl >= 0 else NotificationLevel.WARNING
        return await self.send_notification("포트폴리오 요약", message, level)
    
    async def send_system_alert(self, system_name: str, status: str, details: str = ""):
        """시스템 상태 알림"""
        if status.upper() == "UP":
            emoji = "✅"
            level = NotificationLevel.INFO
        elif status.upper() == "WARNING":
            emoji = "⚠️"
            level = NotificationLevel.WARNING
        else:
            emoji = "🚨"
            level = NotificationLevel.CRITICAL
        
        message = f"""
{emoji} 시스템 상태 변경

시스템: {system_name}
상태: {status.upper()}
"""
        
        if details:
            message += f"상세: {details}\n"
        
        message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_notification(f"시스템 알림: {system_name}", message, level)
    
    # ========================================================================
    # 📊 관리 및 유틸리티 메서드
    # ========================================================================
    
    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """알림 통계 조회"""
        return self.stats_manager.get_stats(days)
    
    def get_config(self) -> NotificationConfig:
        """현재 설정 조회"""
        return self.config_manager.config
    
    def update_config(self, **kwargs):
        """설정 업데이트"""
        self.config_manager.update_config(**kwargs)
        self._init_handlers()  # 핸들러 재초기화
        logger.info("⚙️ 설정 업데이트 및 핸들러 재초기화 완료")
    
    def test_channels(self) -> Dict[NotificationChannel, bool]:
        """채널 연결 테스트"""
        async def _test_all():
            test_message = NotificationMessage(
                title="알림 시스템 테스트",
                message="이것은 알림 시스템 연결 테스트 메시지입니다.",
                level=NotificationLevel.DEBUG
            )
            
            results = {}
            for channel, handler in self.handlers.items():
                try:
                    result = await handler.send_message(test_message)
                    results[channel] = result.success
                except Exception as e:
                    logger.error(f"채널 테스트 실패 {channel}: {e}")
                    results[channel] = False
            
            return results
        
        return asyncio.run(_test_all())
    
    def get_active_channels(self) -> List[NotificationChannel]:
        """활성화된 채널 목록"""
        return list(self.handlers.keys())
    
    def is_channel_active(self, channel: NotificationChannel) -> bool:
        """특정 채널 활성화 여부"""
        return channel in self.handlers

# ============================================================================
# 🧪 테스트 및 예제 함수들
# ============================================================================

async def test_notification_system():
    """알림 시스템 테스트"""
    print("🧪 퀸트프로젝트 알림 시스템 테스트 시작")
    
    # 알림 관리자 초기화
    notifier = QuintNotificationManager()
    
    # 채널 테스트
    print("\n📡 채널 연결 테스트")
    test_results = notifier.test_channels()
    for channel, success in test_results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {channel.value}: {status}")
    
    # 다양한 레벨 테스트
    print("\n📢 알림 레벨 테스트")
    
    await notifier.send_info("정보 알림 테스트", "이것은 일반 정보 알림입니다.")
    await asyncio.sleep(1)
    
    await notifier.send_warning("경고 알림 테스트", "주의가 필요한 상황입니다.")
    await asyncio.sleep(1)
    
    await notifier.send_critical("중요 알림 테스트", "즉시 확인이 필요한 상황입니다.")
    await asyncio.sleep(1)
    
    # 거래 알림 테스트
    print("\n💰 거래 알림 테스트")
    await notifier.send_trade_alert("AAPL", "BUY", 150.25, 100, "미국전략", 1250.50)
    await asyncio.sleep(1)
    
    # 포트폴리오 요약 테스트
    print("\n📊 포트폴리오 요약 테스트")
    await notifier.send_portfolio_summary(100000, 5000, 5.0, 15)
    await asyncio.sleep(1)
    
    # 시스템 알림 테스트
    print("\n🖥️ 시스템 알림 테스트")
    await notifier.send_system_alert("거래시스템", "UP", "모든 시스템이 정상 작동 중입니다.")
    
    # 통계 출력
    print("\n📈 알림 통계")
    stats = notifier.get_stats(1)  # 최근 1일
    print(f"총 알림 수: {stats.get('total_notifications', 0)}")
    
    for channel, data in stats.get('channel_stats', {}).items():
        print(f"  {channel}: {data['successful_sent']}/{data['total_sent']} 성공 ({data['success_rate']:.1f}%)")
    
    print("\n✅ 알림 시스템 테스트 완료")

async def example_usage():
    """사용 예제"""
    print("📚 퀸트프로젝트 알림 시스템 사용 예제")
    
    # 1. 기본 초기화
    notifier = QuintNotificationManager()
    
    # 2. 간단한 정보 알림
    await notifier.send_info("시스템 시작", "퀸트프로젝트가 성공적으로 시작되었습니다.")
    
    # 3. 특정 채널만 사용
    await notifier.send_warning(
        "네트워크 지연", 
        "네트워크 응답이 느려지고 있습니다.",
        channels=[NotificationChannel.TELEGRAM, NotificationChannel.SLACK]
    )
    
    # 4. 메타데이터와 함께
    await notifier.send_critical(
        "포지션 위험",
        "일부 포지션에서 큰 손실이 발생했습니다.",
        metadata={
            "strategy": "미국전략",
            "symbol": "TSLA",
            "loss_amount": -5000
        }
    )
    
    # 5. 거래 관련 편의 메서드
    await notifier.send_trade_alert("BTC", "SELL", 45000, 0.1, "암호화폐전략", 500)
    
    # 6. 설정 동적 변경
    notifier.update_config(
        telegram_enabled=True,
        telegram_bot_token="새토큰",
        duplicate_window_minutes=10
    )
    
    print("✅ 사용 예제 완료")

def create_default_config():
    """기본 설정 파일 생성"""
    config = {
        "telegram_enabled": False,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "email_enabled": False,
        "email_smtp_server": "smtp.gmail.com",
        "email_smtp_port": 587,
        "email_username": "",
        "email_password": "",
        "email_to": [],
        "slack_enabled": False,
        "slack_webhook_url": "",
        "slack_channel": "#general",
        "discord_enabled": False,
        "discord_webhook_url": "",
        "sms_enabled": False,
        "sms_account_sid": "",
        "sms_auth_token": "",
        "sms_from_number": "",
        "sms_to_numbers": [],
        "webhook_enabled": False,
        "webhook_urls": [],
        "level_channels": {
            "debug": ["telegram"],
            "info": ["telegram", "slack"],
            "warning": ["telegram", "email", "slack"],
            "critical": ["telegram", "email", "slack", "discord"],
            "emergency": ["telegram", "email", "slack", "discord", "sms"]
        },
        "duplicate_prevention": True,
        "duplicate_window_minutes": 5,
        "max_retries": 3,
        "retry_delay_seconds": 5
    }
    
    with open("notification_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ 기본 설정 파일 생성: notification_config.json")
    print("📝 설정을 수정한 후 알림 시스템을 사용하세요.")

# ============================================================================
# 🎮 CLI 인터페이스
# ============================================================================

async def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            await test_notification_system()
        elif command == "example":
            await example_usage()
        elif command == "config":
            create_default_config()
        elif command == "stats":
            notifier = QuintNotificationManager()
            stats = notifier.get_stats(7)
            print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
    else:
        print("""
🏆 퀸트프로젝트 통합 알림 시스템

사용법:
  python notifier.py test      # 알림 시스템 테스트
  python notifier.py example   # 사용 예제 실행
  python notifier.py config    # 기본 설정 파일 생성
  python notifier.py stats     # 알림 통계 조회

라이브러리로 사용:
  from notifier import QuintNotificationManager
  notifier = QuintNotificationManager()
  await notifier.send_info("제목", "메시지")
""")

if __name__ == "__main__":
    asyncio.run(main())
