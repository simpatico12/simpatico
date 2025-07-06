#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔔 전설적 퀸트프로젝트 NOTIFIER 시스템 - 완전통합판
================================================================

실시간 알림 + 성과 모니터링 + 다채널 알림 + 스마트 필터링
- 📱 텔레그램 실시간 알림 (매수/매도/포트폴리오)
- 📧 이메일 일일/주간 리포트
- 💬 디스코드 웹훅 연동
- 🔊 음성 알림 (Windows/Mac)
- 📊 대시보드 웹 인터페이스
- 🎯 스마트 알림 필터링 시스템
- 📈 성과 추적 및 자동 리포트

Author: 전설적퀸트팀 | Version: NOTIFIER v1.0
"""

import asyncio
import logging
import os
import sys
import json
import time
import smtplib
import ssl
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.image import MimeImage
import requests
import aiohttp
from jinja2 import Template
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import seaborn as sns
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import uuid
import hashlib
from dotenv import load_dotenv

# 선택적 임포트
try:
    import plyer
    DESKTOP_NOTIFICATIONS = True
except ImportError:
    DESKTOP_NOTIFICATIONS = False
    logging.warning("⚠️ plyer 없음 - 데스크톱 알림 비활성화")

try:
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logging.warning("⚠️ pyttsx3 없음 - 음성 알림 비활성화")

try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning("⚠️ Flask 없음 - 웹 대시보드 비활성화")

load_dotenv()

# ========================================================================================
# 📊 알림 데이터 클래스들
# ========================================================================================

@dataclass
class NotificationConfig:
    """알림 설정"""
    telegram_enabled: bool = True
    email_enabled: bool = False
    discord_enabled: bool = False
    voice_enabled: bool = False
    desktop_enabled: bool = True
    web_dashboard: bool = True
    
    # 필터링 설정
    min_confidence: float = 0.6
    min_signal_count: int = 2
    max_notifications_per_hour: int = 10
    quiet_hours: List[int] = field(default_factory=lambda: [22, 23, 0, 1, 2, 3, 4, 5, 6])
    
    # 채널별 설정
    telegram_signals: bool = True
    telegram_portfolio: bool = True
    telegram_performance: bool = True
    email_daily_report: bool = True
    email_weekly_report: bool = True

@dataclass
class NotificationMessage:
    """알림 메시지"""
    id: str
    type: str  # signal, portfolio, performance, system
    priority: str  # low, normal, high, critical
    title: str
    content: str
    timestamp: datetime
    channels: List[str]  # telegram, email, discord, voice, desktop
    metadata: Dict = field(default_factory=dict)
    sent_channels: List[str] = field(default_factory=list)
    retry_count: int = 0

@dataclass
class PerformanceSnapshot:
    """성과 스냅샷"""
    timestamp: datetime
    total_value: float
    total_pnl: float
    pnl_percentage: float
    strategy_breakdown: Dict
    top_performers: List[Dict]
    worst_performers: List[Dict]
    daily_change: float = 0.0
    weekly_change: float = 0.0

# ========================================================================================
# 🎨 메시지 템플릿 관리자
# ========================================================================================

class MessageTemplates:
    """메시지 템플릿 관리"""
    
    @staticmethod
    def signal_alert(signals: List, strategy: str) -> str:
        """신호 알림 템플릿"""
        strategy_emoji = {
            'us': '🇺🇸', 'japan': '🇯🇵', 
            'india': '🇮🇳', 'crypto': '🪙'
        }
        emoji = strategy_emoji.get(strategy, '📈')
        
        buy_signals = [s for s in signals if hasattr(s, 'action') and s.action == 'buy']
        
        if not buy_signals:
            return f"{emoji} {strategy.upper()} 전략: 매수 신호 없음"
        
        message = f"{emoji} {strategy.upper()} 전략 - 매수 신호 {len(buy_signals)}개 🎯\n\n"
        
        for i, signal in enumerate(buy_signals[:5], 1):
            symbol = getattr(signal, 'symbol', 'N/A')
            confidence = getattr(signal, 'confidence', 0) * 100
            price = getattr(signal, 'price', 0)
            target = getattr(signal, 'target_price', 0)
            reasoning = getattr(signal, 'reasoning', '')
            
            message += f"{i}. {symbol}\n"
            message += f"   💪 신뢰도: {confidence:.1f}%\n"
            message += f"   💰 현재가: {price:,.0f}\n"
            message += f"   🎯 목표가: {target:,.0f}\n"
            message += f"   📝 근거: {reasoning}\n\n"
        
        if len(buy_signals) > 5:
            message += f"... 외 {len(buy_signals) - 5}개 더"
        
        return message
    
    @staticmethod
    def portfolio_summary(summary: Dict) -> str:
        """포트폴리오 요약 템플릿"""
        total_value = summary.get('total_value', 0)
        total_pnl = summary.get('total_pnl', 0)
        pnl_pct = summary.get('pnl_percentage', 0)
        positions = summary.get('total_positions', 0)
        
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_text = f"+{total_pnl:,.0f}" if total_pnl >= 0 else f"{total_pnl:,.0f}"
        
        message = f"📊 퀸트 포트폴리오 현황\n"
        message += f"{'='*30}\n\n"
        message += f"💰 총 자산: {total_value:,.0f}원\n"
        message += f"{pnl_emoji} 손익: {pnl_text}원 ({pnl_pct:+.2f}%)\n"
        message += f"📋 포지션: {positions}개\n\n"
        
        # 전략별 현황
        strategy_breakdown = summary.get('strategy_breakdown', {})
        if strategy_breakdown:
            message += "📈 전략별 현황:\n"
            for strategy, data in strategy_breakdown.items():
                strategy_emoji = {
                    'us': '🇺🇸', 'japan': '🇯🇵', 
                    'india': '🇮🇳', 'crypto': '🪙'
                }
                emoji = strategy_emoji.get(strategy, '📊')
                
                count = data.get('count', 0)
                pnl = data.get('pnl', 0)
                pnl_symbol = "+" if pnl >= 0 else ""
                
                message += f"{emoji} {strategy.upper()}: {count}개 ({pnl_symbol}{pnl:,.0f}원)\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return message
    
    @staticmethod
    def performance_report(performance: PerformanceSnapshot) -> str:
        """성과 리포트 템플릿"""
        pnl_emoji = "📈" if performance.total_pnl >= 0 else "📉"
        
        message = f"📊 퀸트프로젝트 성과 리포트\n"
        message += f"{'='*35}\n\n"
        message += f"📅 일시: {performance.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
        
        message += f"💰 총 자산: {performance.total_value:,.0f}원\n"
        message += f"{pnl_emoji} 총 손익: {performance.total_pnl:+,.0f}원\n"
        message += f"📈 수익률: {performance.pnl_percentage:+.2f}%\n\n"
        
        if performance.daily_change != 0:
            daily_emoji = "⬆️" if performance.daily_change > 0 else "⬇️"
            message += f"{daily_emoji} 일간 변동: {performance.daily_change:+.2f}%\n"
        
        if performance.weekly_change != 0:
            weekly_emoji = "📈" if performance.weekly_change > 0 else "📉"
            message += f"{weekly_emoji} 주간 변동: {performance.weekly_change:+.2f}%\n"
        
        # 최고 수익 종목
        if performance.top_performers:
            message += f"\n🏆 최고 수익 종목:\n"
            for i, item in enumerate(performance.top_performers[:3], 1):
                symbol = item.get('symbol', 'N/A')
                pnl_pct = item.get('pnl_percentage', 0)
                message += f"{i}. {symbol}: +{pnl_pct:.1f}%\n"
        
        # 최저 수익 종목
        if performance.worst_performers:
            message += f"\n📉 주의 종목:\n"
            for i, item in enumerate(performance.worst_performers[:3], 1):
                symbol = item.get('symbol', 'N/A')
                pnl_pct = item.get('pnl_percentage', 0)
                message += f"{i}. {symbol}: {pnl_pct:.1f}%\n"
        
        return message
    
    @staticmethod
    def system_alert(title: str, message: str, level: str = "info") -> str:
        """시스템 알림 템플릿"""
        level_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨',
            'success': '✅'
        }
        emoji = level_emoji.get(level, 'ℹ️')
        
        return f"{emoji} {title}\n\n{message}\n\n⏰ {datetime.now().strftime('%H:%M:%S')}"

# ========================================================================================
# 📱 텔레그램 알림 관리자
# ========================================================================================

class TelegramNotifier:
    """텔레그램 알림 관리"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.session = None
        
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """메시지 전송"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message[:4000],  # 텔레그램 제한
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    logging.debug("텔레그램 메시지 전송 성공")
                    return True
                else:
                    logging.error(f"텔레그램 전송 실패: {response.status}")
                    return False
                    
        except Exception as e:
            logging.error(f"텔레그램 전송 오류: {e}")
            return False
    
    async def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """사진 전송"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                data.add_field('photo', photo, filename='chart.png')
                if caption:
                    data.add_field('caption', caption)
                
                async with self.session.post(url, data=data) as response:
                    return response.status == 200
                    
        except Exception as e:
            logging.error(f"텔레그램 사진 전송 오류: {e}")
            return False
    
    async def close(self):
        """세션 종료"""
        if self.session:
            await self.session.close()

# ========================================================================================
# 📧 이메일 알림 관리자
# ========================================================================================

class EmailNotifier:
    """이메일 알림 관리"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, from_email: str, to_email: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_email = to_email
    
    async def send_email(self, subject: str, content: str, html_content: str = None, attachments: List[str] = None) -> bool:
        """이메일 전송"""
        try:
            message = MimeMultipart('alternative')
            message['Subject'] = subject
            message['From'] = self.from_email
            message['To'] = self.to_email
            
            # 텍스트 내용
            text_part = MimeText(content, 'plain', 'utf-8')
            message.attach(text_part)
            
            # HTML 내용
            if html_content:
                html_part = MimeText(html_content, 'html', 'utf-8')
                message.attach(html_part)
            
            # 첨부파일
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            img_data = f.read()
                            img = MimeImage(img_data)
                            img.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
                            message.attach(img)
            
            # 전송
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(message)
            
            logging.debug("이메일 전송 성공")
            return True
            
        except Exception as e:
            logging.error(f"이메일 전송 실패: {e}")
            return False
    
    def generate_html_report(self, performance: PerformanceSnapshot, chart_path: str = None) -> str:
        """HTML 리포트 생성"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>퀸트프로젝트 성과 리포트</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; color: #333; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; text-align: center; }
                .summary { display: flex; justify-content: space-around; margin: 20px 0; }
                .metric { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }
                .metric h3 { margin: 0; color: #495057; }
                .metric .value { font-size: 24px; font-weight: bold; color: #007bff; }
                .positive { color: #28a745; }
                .negative { color: #dc3545; }
                .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
                .table th { background-color: #e9ecef; }
                .footer { text-align: center; margin-top: 30px; color: #6c757d; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏆 QuintProject 성과 리포트</h1>
                <p>{{ timestamp.strftime('%Y년 %m월 %d일 %H시 %M분') }}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>총 자산</h3>
                    <div class="value">{{ "{:,.0f}".format(total_value) }}원</div>
                </div>
                <div class="metric">
                    <h3>총 손익</h3>
                    <div class="value {{ 'positive' if total_pnl >= 0 else 'negative' }}">
                        {{ "{:+,.0f}".format(total_pnl) }}원
                    </div>
                </div>
                <div class="metric">
                    <h3>수익률</h3>
                    <div class="value {{ 'positive' if pnl_percentage >= 0 else 'negative' }}">
                        {{ "{:+.2f}".format(pnl_percentage) }}%
                    </div>
                </div>
            </div>
            
            {% if strategy_breakdown %}
            <h2>📊 전략별 현황</h2>
            <table class="table">
                <thead>
                    <tr><th>전략</th><th>포지션 수</th><th>손익</th></tr>
                </thead>
                <tbody>
                    {% for strategy, data in strategy_breakdown.items() %}
                    <tr>
                        <td>{{ strategy.upper() }}</td>
                        <td>{{ data.count }}개</td>
                        <td class="{{ 'positive' if data.pnl >= 0 else 'negative' }}">
                            {{ "{:+,.0f}".format(data.pnl) }}원
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
            
            {% if top_performers %}
            <h2>🏆 최고 수익 종목</h2>
            <table class="table">
                <thead>
                    <tr><th>종목</th><th>수익률</th></tr>
                </thead>
                <tbody>
                    {% for item in top_performers[:5] %}
                    <tr>
                        <td>{{ item.symbol }}</td>
                        <td class="positive">+{{ "{:.1f}".format(item.pnl_percentage) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
            
            <div class="footer">
                <p>🤖 QuintProject 자동 리포트 시스템</p>
                <p>이 리포트는 자동으로 생성되었습니다.</p>
            </div>
        </body>
        </html>
        """
        
        try:
            template = Template(html_template)
            return template.render(
                timestamp=performance.timestamp,
                total_value=performance.total_value,
                total_pnl=performance.total_pnl,
                pnl_percentage=performance.pnl_percentage,
                strategy_breakdown=performance.strategy_breakdown,
                top_performers=performance.top_performers
            )
        except Exception as e:
            logging.error(f"HTML 리포트 생성 실패: {e}")
            return ""

# ========================================================================================
# 🎨 차트 생성기
# ========================================================================================

class ChartGenerator:
    """차트 생성기"""
    
    def __init__(self):
        # 한글 폰트 설정
        self._setup_korean_font()
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _setup_korean_font(self):
        """한글 폰트 설정"""
        try:
            # 시스템에 따른 한글 폰트 설정
            if sys.platform.startswith('win'):
                plt.rcParams['font.family'] = 'Malgun Gothic'
            elif sys.platform.startswith('darwin'):
                plt.rcParams['font.family'] = 'AppleGothic'
            else:
                plt.rcParams['font.family'] = 'DejaVu Sans'
            
            plt.rcParams['axes.unicode_minus'] = False
        except:
            logging.warning("한글 폰트 설정 실패")
    
    def generate_portfolio_chart(self, performance_data: List[PerformanceSnapshot], save_path: str) -> bool:
        """포트폴리오 차트 생성"""
        try:
            if not performance_data:
                return False
            
            # 데이터 준비
            dates = [p.timestamp for p in performance_data]
            values = [p.total_value for p in performance_data]
            pnl_percentages = [p.pnl_percentage for p in performance_data]
            
            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 자산 가치 차트
            ax1.plot(dates, values, linewidth=2, color='#007bff', marker='o', markersize=4)
            ax1.fill_between(dates, values, alpha=0.3, color='#007bff')
            ax1.set_title('📊 포트폴리오 자산 가치 추이', fontsize=16, fontweight='bold')
            ax1.set_ylabel('자산 가치 (원)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e8:.1f}억'))
            
            # 수익률 차트
            colors = ['#28a745' if x >= 0 else '#dc3545' for x in pnl_percentages]
            ax2.bar(dates, pnl_percentages, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title('📈 수익률 추이', fontsize=16, fontweight='bold')
            ax2.set_ylabel('수익률 (%)', fontsize=12)
            ax2.set_xlabel('날짜', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 날짜 형식 설정
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return True
            
        except Exception as e:
            logging.error(f"포트폴리오 차트 생성 실패: {e}")
            return False
    
    def generate_strategy_pie_chart(self, strategy_breakdown: Dict, save_path: str) -> bool:
        """전략별 분배 파이 차트"""
        try:
            if not strategy_breakdown:
                return False
            
            # 데이터 준비
            labels = []
            values = []
            colors = []
            
            strategy_colors = {
                'us': '#1f77b4',
                'japan': '#ff7f0e',
                'india': '#2ca02c',
                'crypto': '#d62728'
            }
            
            for strategy, data in strategy_breakdown.items():
                labels.append(f"{strategy.upper()}\n({data['count']}개)")
                values.append(abs(data['value']))
                colors.append(strategy_colors.get(strategy, '#9467bd'))
            
            # 차트 생성
            fig, ax = plt.subplots(figsize=(10, 8))
            
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10}
            )
            
            ax.set_title('📊 전략별 자산 분배', fontsize=16, fontweight='bold', pad=20)
            
            # 범례 추가
            ax.legend(wedges, [f"{label.split('\\n')[0]}: {value:,.0f}원" 
                              for label, value in zip(labels, values)],
                     title="전략별 자산", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return True
            
        except Exception as e:
            logging.error(f"전략별 차트 생성 실패: {e}")
            return False

# ========================================================================================
# 🔔 통합 알림 관리자
# ========================================================================================

class NotificationManager:
    """통합 알림 관리자"""
    
    def __init__(self):
        self.config = NotificationConfig()
        self._load_config()
        
        # 알림 제공자들
        self.telegram = None
        self.email = None
        self.chart_generator = ChartGenerator()
        
        # 알림 큐 및 제한
        self.notification_queue = Queue()
        self.sent_notifications = []
        self.notification_history = []
        
        # 성과 데이터 저장
        self.performance_history: List[PerformanceSnapshot] = []
        self._load_performance_history()
        
        # 백그라운드 워커
        self.worker_thread = None
        self.running = False
        
        self._initialize_notifiers()
    
    def _load_config(self):
        """설정 로드"""
        try:
            # 환경변수에서 설정 로드
            self.config.telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'
            self.config.email_enabled = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
            self.config.voice_enabled = os.getenv('VOICE_ENABLED', 'false').lower() == 'true'
            
            # 필터링 설정
            self.config.min_confidence = float(os.getenv('MIN_CONFIDENCE', '0.6'))
            self.config.max_notifications_per_hour = int(os.getenv('MAX_NOTIFICATIONS_PER_HOUR', '10'))
            
        except Exception as e:
            logging.error(f"알림 설정 로드 실패: {e}")
    
    def _initialize_notifiers(self):
        """알림 제공자 초기화"""
        try:
            # 텔레그램
            if self.config.telegram_enabled:
                bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
                chat_id = os.getenv('TELEGRAM_CHAT_ID')
                if bot_token and chat_id:
                    self.telegram = TelegramNotifier(bot_token, chat_id)
                else:
                    logging.warning("텔레그램 설정 누락")
                    self.config.telegram_enabled = False
            
            # 이메일
            if self.config.email_enabled:
                smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
                smtp_port = int(os.getenv('SMTP_PORT', '587'))
                username = os.getenv('EMAIL_USERNAME')
                password = os.getenv('EMAIL_PASSWORD')
                from_email = os.getenv('FROM_EMAIL')
                to_email = os.getenv('TO_EMAIL')
                
                if all([username, password, from_email, to_email]):
                    self.email = EmailNotifier(smtp_server, smtp_port, username, password, from_email, to_email)
                else:
                    logging.warning("이메일 설정 누락")
                    self.config.email_enabled = False
            
            logging.info("🔔 알림 시스템 초기화 완료")
            
        except Exception as e:
            logging.error(f"알림 제공자 초기화 실패: {e}")
    
    def _load_performance_history(self):
        """성과 히스토리 로드"""
        try:
            history_file = "performance_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    snapshot = PerformanceSnapshot(
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        total_value=item['total_value'],
                        total_pnl=item['total_pnl'],
                        pnl_percentage=item['pnl_percentage'],
                        strategy_breakdown=item['strategy_breakdown'],
                        top_performers=item['top_performers'],
                        worst_performers=item['worst_performers'],
                        daily_change=item.get('daily_change', 0.0),
                        weekly_change=item.get('weekly_change', 0.0)
                    )
                    self.performance_history.append(snapshot)
                
                logging.info(f"📊 성과 히스토리 로드: {len(self.performance_history)}개")
        except Exception as e:
            logging.error(f"성과 히스토리 로드 실패: {e}")
    
    def _save_performance_history(self):
        """성과 히스토리 저장"""
        try:
            history_file = "performance_history.json"
            data = []
            
            for snapshot in self.performance_history[-100:]:  # 최근 100개만 저장
                data.append({
                    'timestamp': snapshot.timestamp.isoformat(),
                    'total_value': snapshot.total_value,
                    'total_pnl': snapshot.total_pnl,
                    'pnl_percentage': snapshot.pnl_percentage,
                    'strategy_breakdown': snapshot.strategy_breakdown,
                    'top_performers': snapshot.top_performers,
                    'worst_performers': snapshot.worst_performers,
                    'daily_change': snapshot.daily_change,
                    'weekly_change': snapshot.weekly_change
                })
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"성과 히스토리 저장 실패: {e}")
    
    def start_worker(self):
        """백그라운드 워커 시작"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logging.info("🔄 알림 워커 시작")
    
    def stop_worker(self):
        """워커 중지"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logging.info("⏹️ 알림 워커 중지")
    
    def _worker_loop(self):
        """워커 루프"""
        while self.running:
            try:
                # 알림 큐 처리
                if not self.notification_queue.empty():
                    message = self.notification_queue.get_nowait()
                    asyncio.run(self._process_notification(message))
                
                # 시간당 알림 제한 체크
                self._cleanup_sent_notifications()
                
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"워커 루프 오류: {e}")
                time.sleep(5)
    
    def _cleanup_sent_notifications(self):
        """시간당 알림 제한을 위한 정리"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.sent_notifications = [
            notif for notif in self.sent_notifications 
            if notif['timestamp'] > cutoff_time
        ]
    
    def _is_quiet_hours(self) -> bool:
        """조용한 시간 체크"""
        current_hour = datetime.now().hour
        return current_hour in self.config.quiet_hours
    
    def _should_filter_notification(self, message: NotificationMessage) -> bool:
        """알림 필터링 체크"""
        # 조용한 시간
        if self._is_quiet_hours() and message.priority not in ['critical']:
            return True
        
        # 시간당 제한
        if len(self.sent_notifications) >= self.config.max_notifications_per_hour:
            return True
        
        # 신뢰도 체크
        if message.type == 'signal':
            confidence = message.metadata.get('confidence', 0)
            if confidence < self.config.min_confidence:
                return True
        
        return False
    
    async def _process_notification(self, message: NotificationMessage):
        """알림 처리"""
        try:
            # 필터링 체크
            if self._should_filter_notification(message):
                logging.debug(f"알림 필터링됨: {message.title}")
                return
            
            success_channels = []
            
            # 채널별 전송
            for channel in message.channels:
                try:
                    if channel == 'telegram' and self.telegram and self.config.telegram_enabled:
                        success = await self.telegram.send_message(message.content)
                        if success:
                            success_channels.append(channel)
                    
                    elif channel == 'email' and self.email and self.config.email_enabled:
                        html_content = None
                        if message.type == 'performance':
                            performance = message.metadata.get('performance_snapshot')
                            if performance:
                                html_content = self.email.generate_html_report(performance)
                        
                        success = await self.email.send_email(
                            subject=message.title,
                            content=message.content,
                            html_content=html_content
                        )
                        if success:
                            success_channels.append(channel)
                    
                    elif channel == 'desktop' and DESKTOP_NOTIFICATIONS and self.config.desktop_enabled:
                        plyer.notification.notify(
                            title=message.title,
                            message=message.content[:200],
                            timeout=10
                        )
                        success_channels.append(channel)
                    
                    elif channel == 'voice' and VOICE_AVAILABLE and self.config.voice_enabled:
                        self._speak_notification(message.title)
                        success_channels.append(channel)
                
                except Exception as e:
                    logging.error(f"채널 {channel} 전송 실패: {e}")
            
            # 전송 기록
            message.sent_channels = success_channels
            self.sent_notifications.append({
                'id': message.id,
                'timestamp': datetime.now(),
                'channels': success_channels
            })
            
            # 히스토리 추가
            self.notification_history.append(message)
            if len(self.notification_history) > 1000:
                self.notification_history = self.notification_history[-500:]
            
            logging.info(f"✅ 알림 전송 완료: {message.title} -> {success_channels}")
            
        except Exception as e:
            logging.error(f"알림 처리 실패: {e}")
    
    def _speak_notification(self, text: str):
        """음성 알림"""
        try:
            if not VOICE_AVAILABLE:
                return
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(f"퀸트프로젝트 알림: {text}")
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logging.error(f"음성 알림 실패: {e}")
    
    # ===== 공개 메서드들 =====
    
    async def send_signal_alert(self, signals: List, strategy: str):
        """신호 알림 전송"""
        try:
            if not signals:
                return
            
            # 매수 신호만 필터링
            buy_signals = [s for s in signals if hasattr(s, 'action') and s.action == 'buy']
            if not buy_signals:
                return
            
            # 신뢰도 높은 순으로 정렬
            buy_signals.sort(key=lambda x: getattr(x, 'confidence', 0), reverse=True)
            
            # 메시지 생성
            content = MessageTemplates.signal_alert(buy_signals, strategy)
            
            message = NotificationMessage(
                id=str(uuid.uuid4()),
                type='signal',
                priority='high' if len(buy_signals) >= 3 else 'normal',
                title=f"🎯 {strategy.upper()} 매수 신호 {len(buy_signals)}개",
                content=content,
                timestamp=datetime.now(),
                channels=['telegram', 'desktop'],
                metadata={
                    'strategy': strategy,
                    'signal_count': len(buy_signals),
                    'confidence': sum(getattr(s, 'confidence', 0) for s in buy_signals) / len(buy_signals)
                }
            )
            
            self.notification_queue.put(message)
            logging.info(f"📤 신호 알림 큐에 추가: {strategy} {len(buy_signals)}개")
            
        except Exception as e:
            logging.error(f"신호 알림 전송 실패: {e}")
    
    async def send_portfolio_summary(self, portfolio_summary: Dict):
        """포트폴리오 요약 알림"""
        try:
            content = MessageTemplates.portfolio_summary(portfolio_summary)
            
            # 중요도 결정
            pnl_pct = portfolio_summary.get('pnl_percentage', 0)
            if abs(pnl_pct) >= 5:
                priority = 'high'
            elif abs(pnl_pct) >= 2:
                priority = 'normal'
            else:
                priority = 'low'
            
            message = NotificationMessage(
                id=str(uuid.uuid4()),
                type='portfolio',
                priority=priority,
                title="📊 포트폴리오 현황",
                content=content,
                timestamp=datetime.now(),
                channels=['telegram'],
                metadata=portfolio_summary
            )
            
            self.notification_queue.put(message)
            
        except Exception as e:
            logging.error(f"포트폴리오 알림 전송 실패: {e}")
    
    async def send_performance_report(self, performance: PerformanceSnapshot, include_chart: bool = True):
        """성과 리포트 전송"""
        try:
            # 성과 히스토리에 추가
            self.performance_history.append(performance)
            self._save_performance_history()
            
            # 일간/주간 변동 계산
            if len(self.performance_history) >= 2:
                prev_performance = self.performance_history[-2]
                performance.daily_change = performance.pnl_percentage - prev_performance.pnl_percentage
            
            if len(self.performance_history) >= 7:
                week_ago = self.performance_history[-7]
                performance.weekly_change = performance.pnl_percentage - week_ago.pnl_percentage
            
            content = MessageTemplates.performance_report(performance)
            
            channels = ['telegram']
            if self.config.email_enabled:
                channels.append('email')
            
            # 차트 생성
            chart_path = None
            if include_chart and len(self.performance_history) >= 2:
                chart_path = f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                if self.chart_generator.generate_portfolio_chart(self.performance_history[-30:], chart_path):
                    # 텔레그램에 차트 전송
                    if self.telegram:
                        await self.telegram.send_photo(chart_path, "📊 포트폴리오 성과 차트")
            
            message = NotificationMessage(
                id=str(uuid.uuid4()),
                type='performance',
                priority='normal',
                title="📈 성과 리포트",
                content=content,
                timestamp=datetime.now(),
                channels=channels,
                metadata={
                    'performance_snapshot': performance,
                    'chart_path': chart_path
                }
            )
            
            self.notification_queue.put(message)
            
            # 차트 파일 정리
            if chart_path and os.path.exists(chart_path):
                threading.Timer(300, lambda: os.remove(chart_path) if os.path.exists(chart_path) else None).start()
            
        except Exception as e:
            logging.error(f"성과 리포트 전송 실패: {e}")
    
    async def send_system_alert(self, title: str, message: str, level: str = "info"):
        """시스템 알림 전송"""
        try:
            content = MessageTemplates.system_alert(title, message, level)
            
            priority_map = {
                'info': 'low',
                'warning': 'normal',
                'error': 'high',
                'critical': 'critical'
            }
            
            channels = ['telegram', 'desktop']
            if level in ['error', 'critical']:
                channels.append('voice')
            
            notification = NotificationMessage(
                id=str(uuid.uuid4()),
                type='system',
                priority=priority_map.get(level, 'normal'),
                title=f"🚨 시스템 {title}",
                content=content,
                timestamp=datetime.now(),
                channels=channels,
                metadata={'level': level}
            )
            
            self.notification_queue.put(notification)
            
        except Exception as e:
            logging.error(f"시스템 알림 전송 실패: {e}")
    
    def get_notification_stats(self) -> Dict:
        """알림 통계 조회"""
        try:
            total_sent = len(self.sent_notifications)
            
            # 채널별 통계
            channel_stats = {}
            for notif in self.sent_notifications:
                for channel in notif['channels']:
                    channel_stats[channel] = channel_stats.get(channel, 0) + 1
            
            # 타입별 통계
            type_stats = {}
            for message in self.notification_history[-100:]:
                type_stats[message.type] = type_stats.get(message.type, 0) + 1
            
            return {
                'total_sent_last_hour': total_sent,
                'channel_stats': channel_stats,
                'type_stats': type_stats,
                'total_history': len(self.notification_history),
                'last_notification': self.notification_history[-1].timestamp.isoformat() if self.notification_history else None,
                'config': {
                    'telegram_enabled': self.config.telegram_enabled,
                    'email_enabled': self.config.email_enabled,
                    'voice_enabled': self.config.voice_enabled,
                    'desktop_enabled': self.config.desktop_enabled
                }
            }
            
        except Exception as e:
            logging.error(f"알림 통계 조회 실패: {e}")
            return {}

# ========================================================================================
# 🌐 웹 대시보드 (선택사항)
# ========================================================================================

class WebDashboard:
    """웹 대시보드"""
    
    def __init__(self, notification_manager: NotificationManager):
        self.notification_manager = notification_manager
        self.app = None
        
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self._setup_routes()
    
    def _setup_routes(self):
        """라우트 설정"""
        if not self.app:
            return
        
        @self.app.route('/')
        def dashboard():
            stats = self.notification_manager.get_notification_stats()
            
            dashboard_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>QuintProject 알림 대시보드</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
                    .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
                    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                    .stat { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 10px; }
                    .stat-value { font-size: 24px; font-weight: bold; color: #007bff; }
                    .stat-label { color: #666; margin-top: 5px; }
                    .status-good { color: #28a745; }
                    .status-bad { color: #dc3545; }
                    .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; 
                                  border-radius: 5px; cursor: pointer; margin: 10px; }
                    .refresh-btn:hover { background: #0056b3; }
                </style>
                <script>
                    function refreshPage() { location.reload(); }
                    setInterval(refreshPage, 30000); // 30초마다 새로고침
                </script>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🔔 QuintProject 알림 시스템</h1>
                        <p>실시간 알림 모니터링 대시보드</p>
                        <button class="refresh-btn" onclick="refreshPage()">🔄 새로고침</button>
                    </div>
                    
                    <div class="grid">
                        <div class="card">
                            <h2>📊 알림 통계</h2>
                            <div class="stat">
                                <div class="stat-value">{{ stats.get('total_sent_last_hour', 0) }}</div>
                                <div class="stat-label">지난 시간 전송</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value">{{ stats.get('total_history', 0) }}</div>
                                <div class="stat-label">총 알림 기록</div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h2>📱 채널별 현황</h2>
                            {% for channel, count in stats.get('channel_stats', {}).items() %}
                            <div class="stat">
                                <div class="stat-value">{{ count }}</div>
                                <div class="stat-label">{{ channel }}</div>
                            </div>
                            {% endfor %}
                        </div>
                        
                        <div class="card">
                            <h2>🎯 알림 유형</h2>
                            {% for type, count in stats.get('type_stats', {}).items() %}
                            <div class="stat">
                                <div class="stat-value">{{ count }}</div>
                                <div class="stat-label">{{ type }}</div>
                            </div>
                            {% endfor %}
                        </div>
                        
                        <div class="card">
                            <h2>⚙️ 시스템 상태</h2>
                            {% set config = stats.get('config', {}) %}
                            <p>📱 텔레그램: <span class="{{ 'status-good' if config.get('telegram_enabled') else 'status-bad' }}">
                                {{ '활성화' if config.get('telegram_enabled') else '비활성화' }}</span></p>
                            <p>📧 이메일: <span class="{{ 'status-good' if config.get('email_enabled') else 'status-bad' }}">
                                {{ '활성화' if config.get('email_enabled') else '비활성화' }}</span></p>
                            <p>🔊 음성: <span class="{{ 'status-good' if config.get('voice_enabled') else 'status-bad' }}">
                                {{ '활성화' if config.get('voice_enabled') else '비활성화' }}</span></p>
                            <p>🖥️ 데스크톱: <span class="{{ 'status-good' if config.get('desktop_enabled') else 'status-bad' }}">
                                {{ '활성화' if config.get('desktop_enabled') else '비활성화' }}</span></p>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>📈 최근 성과 차트</h2>
                        {% if stats.get('last_notification') %}
                        <p>마지막 알림: {{ stats.get('last_notification') }}</p>
                        {% else %}
                        <p>알림 기록이 없습니다.</p>
                        {% endif %}
                    </div>
                </div>
            </body>
            </html>
            """
            
            template = Template(dashboard_html)
            return template.render(stats=stats)
        
        @self.app.route('/api/stats')
        def api_stats():
            return jsonify(self.notification_manager.get_notification_stats())
        
        @self.app.route('/api/test/<channel>')
        def test_notification(channel):
            try:
                message = NotificationMessage(
                    id=str(uuid.uuid4()),
                    type='system',
                    priority='low',
                    title="테스트 알림",
                    content=f"{channel} 채널 테스트 메시지입니다.",
                    timestamp=datetime.now(),
                    channels=[channel],
                    metadata={'test': True}
                )
                
                self.notification_manager.notification_queue.put(message)
                return jsonify({'status': 'success', 'message': f'{channel} 테스트 알림 전송'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
    
    def run(self, host='127.0.0.1', port=8080, debug=False):
        """웹 서버 실행"""
        if self.app:
            logging.info(f"🌐 웹 대시보드 시작: http://{host}:{port}")
            self.app.run(host=host, port=port, debug=debug)
        else:
            logging.error("Flask가 설치되지 않아 웹 대시보드를 실행할 수 없습니다")

# ========================================================================================
# 🎮 편의 함수들
# ========================================================================================

async def quick_notification_test():
    """빠른 알림 테스트"""
    print("🔔 퀸트 알림 시스템 테스트")
    print("=" * 50)
    
    # 알림 관리자 초기화
    notifier = NotificationManager()
    notifier.start_worker()
    
    try:
        # 테스트 신호 생성
        test_signals = [
            type('Signal', (), {
                'symbol': 'AAPL',
                'action': 'buy',
                'confidence': 0.85,
                'price': 150.0,
                'target_price': 165.0,
                'reasoning': '테스트 신호'
            })()
        ]
        
        # 신호 알림 테스트
        print("📱 신호 알림 테스트...")
        await notifier.send_signal_alert(test_signals, 'us')
        
        # 포트폴리오 알림 테스트
        print("📊 포트폴리오 알림 테스트...")
        test_portfolio = {
            'total_value': 1000000000,
            'total_pnl': 50000000,
            'pnl_percentage': 5.0,
            'total_positions': 8,
            'strategy_breakdown': {
                'us': {'count': 3, 'pnl': 20000000},
                'japan': {'count': 2, 'pnl': 15000000},
                'crypto': {'count': 3, 'pnl': 15000000}
            }
        }
        await notifier.send_portfolio_summary(test_portfolio)
        
        # 시스템 알림 테스트
        print("🚨 시스템 알림 테스트...")
        await notifier.send_system_alert("테스트", "알림 시스템이 정상 작동합니다", "info")
        
        # 잠시 대기
        await asyncio.sleep(3)
        
        # 통계 출력
        stats = notifier.get_notification_stats()
        print(f"\n📈 알림 통계:")
        print(f"  - 전송됨: {stats.get('total_sent_last_hour', 0)}개")
        print(f"  - 히스토리: {stats.get('total_history', 0)}개")
        print(f"  - 텔레그램: {'✅' if stats.get('config', {}).get('telegram_enabled') else '❌'}")
        print(f"  - 이메일: {'✅' if stats.get('config', {}).get('email_enabled') else '❌'}")
        
        print("\n✅ 알림 테스트 완료!")
        
    finally:
        notifier.stop_worker()

def create_sample_env_for_notifications():
    """알림용 환경변수 샘플 생성"""
    env_content = """# QuintProject 알림 시스템 설정

# 텔레그램 설정
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# 이메일 설정 (선택사항)
EMAIL_ENABLED=false
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
TO_EMAIL=your_email@gmail.com

# 알림 필터링 설정
MIN_CONFIDENCE=0.6
MAX_NOTIFICATIONS_PER_HOUR=10

# 추가 알림 채널 (선택사항)
VOICE_ENABLED=false
DESKTOP_ENABLED=true
"""
    
    try:
        env_file = ".env.notifications"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print(f"✅ 알림 환경변수 샘플 생성: {env_file}")
        print("\n📋 다음 단계:")
        print("1. 텔레그램 봇 생성 및 토큰 발급")
        print("2. 채팅 ID 확인")
        print("3. .env.notifications 파일 편집")
        print("4. python notifier.py test 실행")
    except Exception as e:
        print(f"❌ 환경변수 파일 생성 실패: {e}")

def print_notifier_help():
    """알림 시스템 도움말"""
    help_text = """
🔔 QuintProject 알림 시스템 v1.0
=====================================

📋 주요 기능:
  - 📱 텔레그램 실시간 알림
  - 📧 이메일 리포트
  - 🔊 음성 알림 (선택사항)
  - 🖥️ 데스크톱 알림
  - 📊 웹 대시보드

🚀 명령어:
  python notifier.py test          # 알림 테스트
  python notifier.py dashboard     # 웹 대시보드 실행
  python notifier.py setup         # 초기 설정
  python notifier.py stats         # 통계 조회

🔧 텔레그램 봇 설정:
  1. @BotFather에게 /newbot 명령
  2. 봇 이름과 사용자명 설정
  3. 발급받은 토큰을 .env에 설정
  4. 봇과 대화 시작 후 /start
  5. @userinfobot으로 채팅 ID 확인

📧 이메일 설정:
  - Gmail 사용시: 앱 비밀번호 발급 필요
  - 2단계 인증 활성화 후 앱 비밀번호 생성
  - SMTP 서버: smtp.gmail.com:587

🎯 알림 유형:
  - signal: 매수/매도 신호
  - portfolio: 포트폴리오 현황
  - performance: 성과 리포트
  - system: 시스템 상태

📊 필터링 옵션:
  - 신뢰도 임계값 설정
  - 시간당 알림 제한
  - 조용한 시간 설정
  - 우선순위별 채널 분배
"""
    print(help_text)

# ========================================================================================
# 🏁 메인 실행부
# ========================================================================================

async def main():
    """메인 실행 함수"""
    print("🔔" + "=" * 70)
    print("🔥 전설적 퀸트프로젝트 NOTIFIER 시스템")
    print("📱 실시간 알림 + 성과 모니터링 + 다채널 통합")
    print("=" * 72)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            print("🧪 알림 시스템 테스트 시작...")
            await quick_notification_test()
            
        elif command == 'dashboard':
            print("🌐 웹 대시보드 시작...")
            notifier = NotificationManager()
            notifier.start_worker()
            
            try:
                dashboard = WebDashboard(notifier)
                dashboard.run(host='0.0.0.0', port=8080, debug=False)
            except KeyboardInterrupt:
                print("\n⏹️ 대시보드 중지")
            finally:
                notifier.stop_worker()
                
        elif command == 'setup':
            create_sample_env_for_notifications()
            
        elif command == 'stats':
            notifier = NotificationManager()
            stats = notifier.get_notification_stats()
            
            print("\n📊 알림 시스템 통계")
            print("=" * 30)
            print(f"지난 시간 전송: {stats.get('total_sent_last_hour', 0)}개")
            print(f"총 히스토리: {stats.get('total_history', 0)}개")
            
            print("\n📱 채널별 통계:")
            for channel, count in stats.get('channel_stats', {}).items():
                print(f"  {channel}: {count}개")
            
            print("\n🎯 알림 유형:")
            for msg_type, count in stats.get('type_stats', {}).items():
                print(f"  {msg_type}: {count}개")
                
        elif command == 'help' or command == '--help':
            print_notifier_help()
            
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print("사용법: python notifier.py [test|dashboard|setup|stats|help]")
    
    else:
        # 인터랙티브 모드
        print("\n🚀 알림 시스템 옵션:")
        print("  1. 알림 테스트")
        print("  2. 웹 대시보드 실행")
        print("  3. 실시간 모니터링")
        print("  4. 통계 조회")
        print("  5. 초기 설정")
        print("  6. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (1-6): ").strip()
                
                if choice == '1':
                    print("\n🧪 알림 테스트 실행...")
                    await quick_notification_test()
                
                elif choice == '2':
                    print("\n🌐 웹 대시보드 시작...")
                    notifier = NotificationManager()
                    notifier.start_worker()
                    
                    try:
                        dashboard = WebDashboard(notifier)
                        print("대시보드 주소: http://localhost:8080")
                        print("Ctrl+C로 중지할 수 있습니다.")
                        dashboard.run(host='127.0.0.1', port=8080, debug=False)
                    except KeyboardInterrupt:
                        print("\n⏹️ 대시보드 중지")
                    finally:
                        notifier.stop_worker()
                
                elif choice == '3':
                    print("\n🔄 실시간 모니터링 시작...")
                    notifier = NotificationManager()
                    notifier.start_worker()
                    
                    try:
                        print("알림 시스템이 백그라운드에서 실행중입니다.")
                        print("Ctrl+C로 중지할 수 있습니다.")
                        
                        # 주기적으로 상태 출력
                        while True:
                            await asyncio.sleep(30)
                            stats = notifier.get_notification_stats()
                            current_time = datetime.now().strftime('%H:%M:%S')
                            print(f"[{current_time}] 💌 전송: {stats.get('total_sent_last_hour', 0)}개, "
                                  f"히스토리: {stats.get('total_history', 0)}개")
                            
                    except KeyboardInterrupt:
                        print("\n⏹️ 모니터링 중지")
                    finally:
                        notifier.stop_worker()
                
                elif choice == '4':
                    print("\n📊 통계 조회 중...")
                    notifier = NotificationManager()
                    stats = notifier.get_notification_stats()
                    
                    print("\n📈 알림 시스템 현황")
                    print("=" * 40)
                    print(f"📤 지난 시간 전송: {stats.get('total_sent_last_hour', 0)}개")
                    print(f"📚 총 히스토리: {stats.get('total_history', 0)}개")
                    
                    if stats.get('channel_stats'):
                        print("\n📱 채널별 현황:")
                        for channel, count in stats.get('channel_stats', {}).items():
                            emoji = {'telegram': '📱', 'email': '📧', 'desktop': '🖥️', 'voice': '🔊'}.get(channel, '📢')
                            print(f"  {emoji} {channel}: {count}개")
                    
                    config = stats.get('config', {})
                    print("\n⚙️ 시스템 상태:")
                    print(f"  📱 텔레그램: {'✅ 활성화' if config.get('telegram_enabled') else '❌ 비활성화'}")
                    print(f"  📧 이메일: {'✅ 활성화' if config.get('email_enabled') else '❌ 비활성화'}")
                    print(f"  🔊 음성: {'✅ 활성화' if config.get('voice_enabled') else '❌ 비활성화'}")
                    print(f"  🖥️ 데스크톱: {'✅ 활성화' if config.get('desktop_enabled') else '❌ 비활성화'}")
                
                elif choice == '5':
                    print("\n🔧 초기 설정 시작...")
                    create_sample_env_for_notifications()
                
                elif choice == '6':
                    print("👋 알림 시스템을 종료합니다!")
                    break
                
                else:
                    print("❌ 잘못된 선택입니다. 1-6 중 선택하세요.")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

# ========================================================================================
# 🚀 Core 시스템과의 통합 인터페이스
# ========================================================================================

class CoreNotificationInterface:
    """Core 시스템과의 알림 인터페이스"""
    
    def __init__(self):
        self.notification_manager = NotificationManager()
        self.notification_manager.start_worker()
        
        # 로깅 핸들러 추가
        self._setup_logging_handler()
    
    def _setup_logging_handler(self):
        """로깅 핸들러 설정"""
        class NotificationHandler(logging.Handler):
            def __init__(self, notifier):
                super().__init__()
                self.notifier = notifier
            
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    asyncio.create_task(
                        self.notifier.send_system_alert(
                            "시스템 오류",
                            record.getMessage(),
                            "error"
                        )
                    )
        
        handler = NotificationHandler(self.notification_manager)
        logging.getLogger().addHandler(handler)
    
    async def notify_signals(self, signals_dict: Dict[str, List]):
        """Core에서 생성된 신호 알림"""
        for strategy, signals in signals_dict.items():
            if isinstance(signals, list) and signals:
                await self.notification_manager.send_signal_alert(signals, strategy)
    
    async def notify_portfolio(self, portfolio_summary: Dict):
        """포트폴리오 현황 알림"""
        await self.notification_manager.send_portfolio_summary(portfolio_summary)
    
    async def notify_performance(self, performance_data: Dict):
        """성과 데이터 알림"""
        # 딕셔너리를 PerformanceSnapshot으로 변환
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            total_value=performance_data.get('total_value', 0),
            total_pnl=performance_data.get('total_pnl', 0),
            pnl_percentage=performance_data.get('pnl_percentage', 0),
            strategy_breakdown=performance_data.get('strategy_breakdown', {}),
            top_performers=performance_data.get('top_performers', []),
            worst_performers=performance_data.get('worst_performers', [])
        )
        
        await self.notification_manager.send_performance_report(snapshot)
    
    async def notify_system_event(self, title: str, message: str, level: str = "info"):
        """시스템 이벤트 알림"""
        await self.notification_manager.send_system_alert(title, message, level)
    
    def cleanup(self):
        """정리"""
        if self.notification_manager:
            self.notification_manager.stop_worker()

# ========================================================================================
# 📦 유틸리티 함수들
# ========================================================================================

def install_requirements():
    """필요한 패키지 설치 가이드"""
    requirements = [
        "aiohttp",
        "jinja2", 
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "python-dotenv",
        "requests",
        # 선택사항
        "flask",  # 웹 대시보드
        "plyer",  # 데스크톱 알림
        "pyttsx3"  # 음성 알림
    ]
    
    print("📦 Notifier 시스템 필요 패키지:")
    print("=" * 50)
    
    for package in requirements:
        print(f"  - {package}")
    
    print(f"\n설치 명령어:")
    print(f"pip install {' '.join(requirements)}")
    
    print(f"\n필수 패키지만 설치:")
    essential = requirements[:7]
    print(f"pip install {' '.join(essential)}")

def check_notification_dependencies():
    """알림 시스템 의존성 체크"""
    print("🔍 알림 시스템 의존성 체크...")
    
    required = {
        'aiohttp': '비동기 HTTP 클라이언트',
        'jinja2': '템플릿 엔진',
        'matplotlib': '차트 생성',
        'pandas': '데이터 처리',
        'python-dotenv': '환경변수 로드'
    }
    
    optional = {
        'flask': '웹 대시보드',
        'plyer': '데스크톱 알림',
        'pyttsx3': '음성 알림',
        'seaborn': '고급 차트'
    }
    
    print("\n📋 필수 패키지:")
    missing_required = []
    for package, description in required.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ❌ {package} - {description}")
            missing_required.append(package)
    
    print("\n📦 선택적 패키지:")
    for package, description in optional.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ⚠️ {package} - {description} (선택사항)")
    
    if missing_required:
        print(f"\n❌ 누락된 필수 패키지: {', '.join(missing_required)}")
        print(f"설치 명령: pip install {' '.join(missing_required)}")
        return False
    else:
        print("\n✅ 모든 필수 패키지가 설치되어 있습니다!")
        return True

# ========================================================================================
# 🏁 최종 실행부
# ========================================================================================

if __name__ == "__main__":
    try:
        # 의존성 체크
        if not check_notification_dependencies():
            print("\n⚠️ 필수 패키지를 먼저 설치해주세요.")
            sys.exit(1)
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('notifier.log', encoding='utf-8')
            ]
        )
        
        # 메인 실행
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 알림 시스템이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        logging.error(f"실행 오류: {e}")

# ========================================================================================
# 🏁 최종 익스포트
# ========================================================================================

__all__ = [
    'NotificationManager',
    'NotificationConfig',
    'NotificationMessage', 
    'PerformanceSnapshot',
    'MessageTemplates',
    'TelegramNotifier',
    'EmailNotifier',
    'ChartGenerator',
    'WebDashboard',
    'CoreNotificationInterface',
    'quick_notification_test'
]

"""
🔔 QuintProject Notifier 사용 예시:

# 1. 기본 사용
from notifier import NotificationManager
notifier = NotificationManager()
notifier.start_worker()

# 2. 신호 알림
await notifier.send_signal_alert(signals, 'us')

# 3. 포트폴리오 알림
portfolio_summary = {...}
await notifier.send_portfolio_summary(portfolio_summary)

# 4. 성과 리포트
performance = PerformanceSnapshot(...)
await notifier.send_performance_report(performance)

# 5. Core 시스템 통합
from notifier import CoreNotificationInterface
interface = CoreNotificationInterface()
await interface.notify_signals(signals_dict)
await interface.notify_portfolio(portfolio_summary)

# 6. 웹 대시보드
from notifier import WebDashboard
dashboard = WebDashboard(notifier)
dashboard.run(host='0.0.0.0', port=8080)

# 7. 빠른 테스트
python notifier.py test

# 8. 웹 대시보드 실행
python notifier.py dashboard
"""
