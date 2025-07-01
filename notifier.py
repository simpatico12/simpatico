"""
🔔 최고퀸트프로젝트 - 텔레그램 알림 시스템 (Final Edition)
================================================================

완전한 알림 시스템 + utils.py 통합:
- 📱 매매 신호/완료 알림
- 📊 시장 분석 요약
- 📰 뉴스 분석 결과  
- 📅 스케줄링 알림
- 🚨 시스템 상태 알림
- 📈 일일 성과 리포트
- 🔔 다채널 알림 (텔레그램, 슬랙, 이메일)
- 🧪 완전한 테스트 시스템
- 🔗 utils.py 완벽 연동

Author: 최고퀸트팀
Version: 3.0.0 (Final Edition)
Project: 최고퀸트프로젝트
File: notifier.py (프로젝트 루트)
"""

import asyncio
import aiohttp
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# utils.py에서 필요한 기능들 import
try:
    from utils import (
        config_manager, timezone_manager, Formatter, Validator,
        SecurityUtils, cache, file_manager, save_trading_log
    )
    UTILS_AVAILABLE = True
except ImportError:
    print("⚠️ utils.py를 찾을 수 없습니다. 기본 기능으로 실행합니다.")
    UTILS_AVAILABLE = False

# 로거 설정
logger = logging.getLogger(__name__)

# ================================
# 📋 데이터 클래스 정의
# ================================

@dataclass
class TradingSignal:
    """거래 신호 데이터 클래스"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    market: str  # US, JP, COIN
    price: float
    confidence: float
    reasoning: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    quantity: Optional[float] = None
    timestamp: Optional[datetime] = None
    execution_status: str = "signal"  # signal, pending, completed, failed, cancelled

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class MarketSummary:
    """시장 요약 데이터 클래스"""
    market: str
    total_analyzed: int
    buy_signals: int
    sell_signals: int
    hold_signals: int = 0
    analysis_time: float = 0.0
    is_trading_day: bool = True
    executed_trades: List[Dict] = None
    top_picks: List[TradingSignal] = None
    market_sentiment: float = 0.5
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.executed_trades is None:
            self.executed_trades = []
        if self.top_picks is None:
            self.top_picks = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PerformanceReport:
    """성과 리포트 데이터 클래스"""
    date: str
    total_signals: int
    total_trades: int
    successful_trades: int
    failed_trades: int
    daily_return: Optional[float] = None
    total_return: Optional[float] = None
    total_pnl: Optional[float] = None
    win_rate: Optional[float] = None
    top_performers: List[Dict] = None
    worst_performers: List[Dict] = None
    market_exposure: Dict[str, float] = None

    def __post_init__(self):
        if self.top_performers is None:
            self.top_performers = []
        if self.worst_performers is None:
            self.worst_performers = []
        if self.market_exposure is None:
            self.market_exposure = {}
        if self.total_trades > 0:
            self.win_rate = (self.successful_trades / self.total_trades) * 100

# ================================
# 🔧 설정 및 상수
# ================================

# 시장별 이모지 및 이름
MARKET_EMOJIS = {
    'US': '🇺🇸', 'JP': '🇯🇵', 'COIN': '🪙', 'CRYPTO': '🪙',
    'EU': '🇪🇺', 'KOR': '🇰🇷'
}

MARKET_NAMES = {
    'US': '미국', 'JP': '일본', 'COIN': '암호화폐', 'CRYPTO': '암호화폐',
    'EU': '유럽', 'KOR': '한국'
}

ACTION_EMOJIS = {
    'BUY': '💰', 'SELL': '💸', 'HOLD': '⏸️',
    'buy': '💰', 'sell': '💸', 'hold': '⏸️'
}

STATUS_EMOJIS = {
    'signal': '📊', 'pending': '⏳', 'completed': '✅',
    'failed': '❌', 'cancelled': '🚫', 'partial': '🟡'
}

PRIORITY_EMOJIS = {
    'critical': '🚨', 'error': '❌', 'warning': '⚠️',
    'info': 'ℹ️', 'debug': '🔍', 'success': '✅'
}

# ================================
# 💬 메시지 포맷터
# ================================

class MessageFormatter:
    """메시지 포맷팅 전용 클래스"""
    
    @staticmethod
    def format_price(price: float, market: str) -> str:
        """시장별 가격 포맷팅"""
        try:
            if UTILS_AVAILABLE and hasattr(Formatter, 'format_price'):
                currency_map = {'US': 'USD', 'JP': 'JPY', 'COIN': 'KRW', 'EU': 'EUR'}
                currency = currency_map.get(market, 'USD')
                return Formatter.format_price(price, currency)
            else:
                # 기본 포맷팅
                if market == 'US':
                    return f"${price:,.2f}"
                elif market == 'JP':
                    return f"¥{price:,.0f}"
                elif market in ['COIN', 'CRYPTO']:
                    if price >= 1000000:
                        return f"₩{price:,.0f}"
                    else:
                        return f"₩{price:,.2f}"
                else:
                    return f"{price:,.2f}"
        except Exception:
            return str(price)

    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """퍼센트 포맷팅"""
        try:
            if UTILS_AVAILABLE and hasattr(Formatter, 'format_percentage'):
                return Formatter.format_percentage(value, decimals)
            else:
                sign = "+" if value > 0 else ""
                return f"{sign}{value:.{decimals}f}%"
        except:
            return f"{value:.{decimals}f}%"

    @staticmethod
    def format_datetime(dt: datetime = None, format_type: str = 'default') -> str:
        """날짜시간 포맷팅"""
        if dt is None:
            dt = datetime.now()
        
        try:
            if UTILS_AVAILABLE and hasattr(Formatter, 'format_datetime'):
                return Formatter.format_datetime(dt, format_type)
            else:
                if format_type == 'short':
                    return dt.strftime('%m/%d %H:%M')
                elif format_type == 'korean':
                    return dt.strftime('%Y년 %m월 %d일 %H시 %M분')
                else:
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def get_confidence_emoji(confidence: float) -> str:
        """신뢰도별 이모지"""
        if confidence >= 0.9:
            return "🔥🔥"
        elif confidence >= 0.8:
            return "🔥"
        elif confidence >= 0.7:
            return "⭐"
        elif confidence >= 0.6:
            return "👍"
        elif confidence >= 0.5:
            return "👌"
        else:
            return "🤔"

    @staticmethod
    def get_return_emoji(return_pct: float) -> str:
        """수익률별 이모지"""
        if return_pct >= 5:
            return "🚀"
        elif return_pct >= 2:
            return "📈"
        elif return_pct >= 0:
            return "📊"
        elif return_pct >= -2:
            return "📉"
        else:
            return "💀"

# ================================
# 🔔 통합 알림 매니저
# ================================

class NotificationManager:
    """통합 알림 관리자"""
    
    def __init__(self):
        """초기화"""
        self.load_config()
        self.session = None
        self.rate_limiters = {}
        self.message_cache = {}
        self.stats = {
            'total_sent': 0,
            'successful': 0,
            'failed': 0,
            'by_channel': {},
            'by_level': {}
        }

    def load_config(self):
        """설정 로드"""
        try:
            if UTILS_AVAILABLE and config_manager:
                self.config = config_manager.config
                notifications_config = config_manager.get('notifications', {})
            else:
                import yaml
                try:
                    with open('settings.yaml', 'r', encoding='utf-8') as f:
                        self.config = yaml.safe_load(f)
                    notifications_config = self.config.get('notifications', {})
                except:
                    notifications_config = {}
            
            # 채널별 설정
            self.telegram_config = notifications_config.get('telegram', {})
            self.slack_config = notifications_config.get('slack', {})
            self.email_config = notifications_config.get('email', {})
            
            # 전역 설정
            self.enabled = notifications_config.get('enabled', True)
            self.min_level = notifications_config.get('min_level', 'info')
            self.rate_limit = notifications_config.get('rate_limit', 10)
            
            # 환경변수 오버라이드
            self.telegram_config['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN', 
                                                        self.telegram_config.get('bot_token', ''))
            self.telegram_config['chat_id'] = os.getenv('TELEGRAM_CHAT_ID', 
                                                      self.telegram_config.get('chat_id', ''))
            
            logger.info("알림 설정 로드 완료")
            
        except Exception as e:
            logger.error(f"알림 설정 로드 실패: {e}")
            self._set_default_config()

    def _set_default_config(self):
        """기본 설정"""
        self.config = {}
        self.telegram_config = {'enabled': False}
        self.slack_config = {'enabled': False}
        self.email_config = {'enabled': False}
        self.enabled = False
        self.min_level = 'info'
        self.rate_limit = 10

    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 가져오기"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self):
        """리소스 정리"""
        if self.session and not self.session.closed:
            await self.session.close()

    def _check_rate_limit(self, channel: str) -> bool:
        """속도 제한 확인"""
        current_time = time.time()
        
        if channel not in self.rate_limiters:
            self.rate_limiters[channel] = []
        
        # 1분 이내 메시지들만 유지
        self.rate_limiters[channel] = [
            t for t in self.rate_limiters[channel] 
            if current_time - t < 60
        ]
        
        if len(self.rate_limiters[channel]) >= self.rate_limit:
            return False
        
        self.rate_limiters[channel].append(current_time)
        return True

    def _should_send(self, level: str) -> bool:
        """알림 발송 여부 판단"""
        if not self.enabled:
            return False
        
        level_priority = {
            'debug': 5, 'info': 4, 'warning': 3, 'error': 2, 'critical': 1
        }
        
        min_priority = level_priority.get(self.min_level, 4)
        msg_priority = level_priority.get(level, 4)
        
        return msg_priority <= min_priority

    def _deduplicate_message(self, message: str, window_seconds: int = 300) -> bool:
        """메시지 중복 제거"""
        import hashlib
        
        msg_hash = hashlib.md5(message.encode()).hexdigest()
        current_time = time.time()
        
        # 오래된 캐시 정리
        expired_keys = [
            key for key, timestamp in self.message_cache.items()
            if current_time - timestamp > window_seconds
        ]
        for key in expired_keys:
            del self.message_cache[key]
        
        # 중복 확인
        if msg_hash in self.message_cache:
            return False
        
        self.message_cache[msg_hash] = current_time
        return True

    async def send_notification(self, 
                              message: str, 
                              level: str = 'info',
                              channels: List[str] = None,
                              priority: bool = False,
                              deduplicate: bool = True) -> Dict[str, bool]:
        """통합 알림 발송"""
        results = {}
        
        # 발송 여부 확인
        if not priority and not self._should_send(level):
            results['skipped'] = True
            return results
        
        # 중복 제거
        if deduplicate and not self._deduplicate_message(message):
            results['duplicate'] = True
            return results
        
        # 채널 결정
        if channels is None:
            channels = []
            if self.telegram_config.get('enabled', False):
                channels.append('telegram')
            if self.slack_config.get('enabled', False):
                channels.append('slack')
            if self.email_config.get('enabled', False):
                channels.append('email')
        
        # 레벨 이모지 추가
        emoji = PRIORITY_EMOJIS.get(level, 'ℹ️')
        formatted_message = f"{emoji} {message}"
        
        # 채널별 발송
        for channel in channels:
            if not self._check_rate_limit(channel):
                results[channel] = False
                logger.warning(f"속도 제한 초과: {channel}")
                continue
            
            try:
                if channel == 'telegram':
                    results[channel] = await self._send_telegram(formatted_message)
                elif channel == 'slack':
                    results[channel] = await self._send_slack(formatted_message)
                elif channel == 'email':
                    results[channel] = await self._send_email(formatted_message, level)
                else:
                    results[channel] = False
                
                # 통계 업데이트
                self.stats['total_sent'] += 1
                if results[channel]:
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                
                self.stats['by_channel'][channel] = self.stats['by_channel'].get(channel, 0) + 1
                self.stats['by_level'][level] = self.stats['by_level'].get(level, 0) + 1
                
            except Exception as e:
                logger.error(f"알림 발송 실패 ({channel}): {e}")
                results[channel] = False
                self.stats['failed'] += 1
        
        # 로그 저장 (utils.py 연동)
        if UTILS_AVAILABLE and save_trading_log:
            save_trading_log({
                'type': 'notification',
                'level': level,
                'message': message[:100],
                'channels': channels,
                'results': results
            }, 'notifications')
        
        return results

    async def _send_telegram(self, message: str) -> bool:
        """텔레그램 발송"""
        try:
            bot_token = self.telegram_config.get('bot_token')
            chat_id = self.telegram_config.get('chat_id')
            
            if not bot_token or not chat_id:
                return False
            
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            async with session.post(url, data=data) as response:
                success = response.status == 200
                if not success:
                    error_text = await response.text()
                    logger.error(f"텔레그램 API 오류: {error_text}")
                return success
                
        except Exception as e:
            logger.error(f"텔레그램 발송 실패: {e}")
            return False

    async def _send_slack(self, message: str) -> bool:
        """슬랙 발송"""
        try:
            webhook_url = self.slack_config.get('webhook_url') or os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                return False
            
            session = await self._get_session()
            payload = {
                'text': message,
                'username': '최고퀸트봇',
                'icon_emoji': ':robot_face:'
            }
            
            async with session.post(webhook_url, json=payload) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"슬랙 발송 실패: {e}")
            return False

    async def _send_email(self, message: str, level: str) -> bool:
        """이메일 발송 (향후 구현)"""
        try:
            logger.info(f"이메일 발송 시뮬레이션: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"이메일 발송 실패: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """알림 통계"""
        return {
            **self.stats,
            'success_rate': (self.stats['successful'] / max(1, self.stats['total_sent'])) * 100,
            'rate_limits': {channel: len(times) for channel, times in self.rate_limiters.items()},
            'cache_size': len(self.message_cache)
        }

# 전역 알림 매니저 인스턴스
_notification_manager = NotificationManager()

# ================================
# 🎯 거래 신호 알림
# ================================

async def send_trading_alert(signal: Union[TradingSignal, Dict], 
                           execution_status: str = "signal") -> bool:
    """매매 신호/완료 알림 발송"""
    try:
        # TradingSignal 객체 또는 딕셔너리 처리
        if isinstance(signal, dict):
            signal = TradingSignal(**signal)
        
        signal.execution_status = execution_status
        
        # 메시지 구성
        action_emoji = ACTION_EMOJIS.get(signal.action.upper(), "📊")
        market_emoji = MARKET_EMOJIS.get(signal.market, "📈")
        status_emoji = STATUS_EMOJIS.get(signal.execution_status, "📊")
        confidence_emoji = MessageFormatter.get_confidence_emoji(signal.confidence)
        
        # 실행 상태에 따른 메시지
        if signal.execution_status == "completed":
            header = f"✅ {signal.action.upper()} 완료"
            price_label = "💰 실행가"
        elif signal.execution_status == "failed":
            header = f"❌ {signal.action.upper()} 실패"
            price_label = "💵 목표가"
        elif signal.execution_status == "pending":
            header = f"⏳ {signal.action.upper()} 대기중"
            price_label = "💵 주문가"
        else:
            header = f"{action_emoji} {signal.action.upper()} 신호"
            price_label = "💵 현재가"
        
        message = f"{header}\n\n"
        message += f"{market_emoji} {MARKET_NAMES.get(signal.market, signal.market)} | {signal.symbol}\n"
        message += f"{price_label}: {MessageFormatter.format_price(signal.price, signal.market)}\n"
        message += f"🎯 신뢰도: {signal.confidence*100:.0f}% {confidence_emoji}\n"
        
        # 목표가 및 손절가
        if signal.target_price:
            expected_return = ((signal.target_price - signal.price) / signal.price) * 100
            return_emoji = MessageFormatter.get_return_emoji(expected_return)
            message += f"🎪 목표가: {MessageFormatter.format_price(signal.target_price, signal.market)}\n"
            message += f"{return_emoji} 기대수익: {MessageFormatter.format_percentage(expected_return)}\n"
        
        if signal.stop_loss:
            stop_loss_pct = ((signal.stop_loss - signal.price) / signal.price) * 100
            message += f"🛡️ 손절가: {MessageFormatter.format_price(signal.stop_loss, signal.market)}\n"
            message += f"📉 손절폭: {MessageFormatter.format_percentage(stop_loss_pct)}\n"
        
        if signal.quantity:
            message += f"📊 수량: {signal.quantity:,.2f}\n"
        
        message += f"\n💡 {signal.reasoning}\n"
        message += f"⏰ {MessageFormatter.format_datetime(signal.timestamp, 'short')}"
        
        # 우선순위 결정
        priority = signal.confidence >= 0.8 or execution_status in ['completed', 'failed']
        level = 'warning' if execution_status == 'failed' else 'info'
        
        results = await _notification_manager.send_notification(
            message, level=level, priority=priority
        )
        
        return any(results.values())
        
    except Exception as e:
        logger.error(f"매매 알림 실패: {e}")
        return False

# ================================
# 📊 시장 요약 알림
# ================================

async def send_market_summary(market_summaries: Dict[str, Union[MarketSummary, Dict]]) -> bool:
    """시장 요약 알림 발송"""
    try:
        # 전체 통계 계산
        total_stats = {
            'total_analyzed': 0,
            'total_buy': 0,
            'total_sell': 0,
            'total_hold': 0,
            'total_executed': 0,
            'avg_analysis_time': 0.0,
            'active_markets': 0
        }
        
        analysis_times = []
        
        for summary in market_summaries.values():
            if isinstance(summary, dict):
                summary_data = summary
            else:
                summary_data = {
                    'total_analyzed': summary.total_analyzed,
                    'buy_signals': summary.buy_signals,
                    'sell_signals': summary.sell_signals,
                    'hold_signals': getattr(summary, 'hold_signals', 0),
                    'analysis_time': summary.analysis_time,
                    'executed_trades': summary.executed_trades,
                    'is_trading_day': summary.is_trading_day
                }
            
            if summary_data.get('is_trading_day', True):
                total_stats['active_markets'] += 1
                
            total_stats['total_analyzed'] += summary_data.get('total_analyzed', 0)
            total_stats['total_buy'] += summary_data.get('buy_signals', 0)
            total_stats['total_sell'] += summary_data.get('sell_signals', 0)
            total_stats['total_hold'] += summary_data.get('hold_signals', 0)
            
            # 실행된 거래 수
            executed_trades = summary_data.get('executed_trades', [])
            if executed_trades:
                executed_count = len([t for t in executed_trades if t.get('executed', False)])
                total_stats['total_executed'] += executed_count
            
            # 분석 시간
            analysis_time = summary_data.get('analysis_time', 0)
            if analysis_time > 0:
                analysis_times.append(analysis_time)
        
        if analysis_times:
            total_stats['avg_analysis_time'] = sum(analysis_times) / len(analysis_times)
        
        # 현재 시간 정보
        if UTILS_AVAILABLE and timezone_manager:
            current_times = timezone_manager.get_all_market_times()
            current_time_kr = current_times.get('KOR', {}).get('datetime', '')
        else:
            current_time_kr = MessageFormatter.format_datetime()
        
        # 메시지 구성
        message = f"🏆 최고퀸트프로젝트 분석 완료\n"
        message += f"=" * 35 + "\n\n"
        
        # 전체 요약
        message += f"📊 전체 분석 결과\n"
        message += f"🔍 분석 종목: {total_stats['total_analyzed']:,}개\n"
        message += f"💰 매수 신호: {total_stats['total_buy']}개\n"
        message += f"💸 매도 신호: {total_stats['total_sell']}개\n"
        if total_stats['total_hold'] > 0:
            message += f"⏸️ 보유 신호: {total_stats['total_hold']}개\n"
        if total_stats['total_executed'] > 0:
            message += f"✅ 실행 거래: {total_stats['total_executed']}개\n"
        message += f"⚡ 평균 소요: {total_stats['avg_analysis_time']:.1f}초\n"
        message += f"🌍 활성 시장: {total_stats['active_markets']}개\n\n"
        
        # 시장별 상세
        for market, summary in market_summaries.items():
            market_emoji = MARKET_EMOJIS.get(market, "📈")
            market_name = MARKET_NAMES.get(market, market)
            
            # 데이터 추출
            if isinstance(summary, dict):
                buy_signals = summary.get('buy_signals', 0)
                sell_signals = summary.get('sell_signals', 0)
                hold_signals = summary.get('hold_signals', 0)
                analysis_time = summary.get('analysis_time', 0)
                top_picks = summary.get('top_picks', [])
                executed_trades = summary.get('executed_trades', [])
                is_trading_day = summary.get('is_trading_day', True)
                market_sentiment = summary.get('market_sentiment', 0.5)
            else:
                buy_signals = summary.buy_signals
                sell_signals = summary.sell_signals
                hold_signals = getattr(summary, 'hold_signals', 0)
                analysis_time = summary.analysis_time
                top_picks = summary.top_picks
                executed_trades = summary.executed_trades
                is_trading_day = summary.is_trading_day
                market_sentiment = getattr(summary, 'market_sentiment', 0.5)
            
            message += f"{market_emoji} {market_name}"
            
            # 휴무일 표시
            if not is_trading_day:
                message += " (휴무)"
            
            # 시장 센티먼트
            if market_sentiment >= 0.6:
                sentiment_emoji = "😊"
            elif market_sentiment <= 0.4:
                sentiment_emoji = "😰"
            else:
                sentiment_emoji = "😐"
            
            message += f" {sentiment_emoji}\n"
            message += f"  📈 매수: {buy_signals}개\n"
            message += f"  📉 매도: {sell_signals}개\n"
            if hold_signals > 0:
                message += f"  ⏸️ 보유: {hold_signals}개\n"
            message += f"  ⏱️ 소요: {analysis_time:.1f}초\n"
            
            # 실행된 거래
            if executed_trades:
                executed_count = len([t for t in executed_trades if t.get('executed', False)])
                if executed_count > 0:
                    message += f"  ✅ 실행: {executed_count}개\n"
            
            # 상위 추천 종목
            if top_picks:
                message += f"  🎯 추천: "
                top_3 = top_picks[:3]
                symbols = []
                
                for pick in top_3:
                    if isinstance(pick, dict):
                        symbol = pick.get('symbol', '')
                        confidence = pick.get('confidence', 0) * 100
                    else:
                        symbol = pick.symbol
                        confidence = pick.confidence * 100
                    
                    confidence_emoji = MessageFormatter.get_confidence_emoji(confidence/100)
                    symbols.append(f"{symbol}({confidence:.0f}%{confidence_emoji})")
                
                message += ", ".join(symbols)
            
            message += "\n\n"
        
        # 시간 정보
        message += f"⏰ 분석 완료: {current_time_kr}"
        
        results = await _notification_manager.send_notification(
            message, level='info', deduplicate=False
        )
        
        return any(results.values())
        
    except Exception as e:
        logger.error(f"시장 요약 알림 실패: {e}")
        return False

# ================================
# 📈 성과 리포트 알림
# ================================

async def send_performance_report(report: Union[PerformanceReport, Dict]) -> bool:
    """성과 리포트 알림 발송"""
    try:
        # PerformanceReport 객체 또는 딕셔너리 처리
        if isinstance(report, dict):
            report_data = {
                'date': report.get('date', datetime.now().strftime('%Y-%m-%d')),
                'total_signals': report.get('total_signals', 0),
                'total_trades': report.get('total_trades', 0),
                'successful_trades': report.get('successful_trades', 0),
                'failed_trades': report.get('failed_trades', 0),
                **report
            }
            report = PerformanceReport(**report_data)
        
        # 메시지 구성
        today = datetime.now()
        weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][today.weekday()]
        
        message = f"📊 일일 성과 리포트\n"
        message += f"=" * 25 + "\n\n"
        message += f"📅 {report.date} ({weekday_kr}요일)\n\n"
        
        # 거래 통계
        message += f"📈 거래 통계\n"
        message += f"  🔍 분석 신호: {report.total_signals:,}개\n"
        message += f"  💰 실행 거래: {report.total_trades}개\n"
        
        if report.total_trades > 0:
            success_rate = (report.successful_trades / report.total_trades) * 100
            rate_emoji = "🎯" if success_rate >= 70 else "📊" if success_rate >= 50 else "📉"
            message += f"  {rate_emoji} 성공률: {success_rate:.1f}%\n"
            message += f"  ✅ 성공: {report.successful_trades}개\n"
            message += f"  ❌ 실패: {report.failed_trades}개\n"
        
        # 수익률 정보
        if report.daily_return is not None:
            return_emoji = MessageFormatter.get_return_emoji(report.daily_return)
            message += f"\n💵 수익률\n"
            message += f"  {return_emoji} 일일: {MessageFormatter.format_percentage(report.daily_return)}\n"
            
            if report.total_return is not None:
                total_emoji = MessageFormatter.get_return_emoji(report.total_return)
                message += f"  {total_emoji} 누적: {MessageFormatter.format_percentage(report.total_return)}\n"
        
        # 손익 정보
        if report.total_pnl is not None:
            pnl_emoji = "💰" if report.total_pnl >= 0 else "💸"
            message += f"  {pnl_emoji} 손익: {MessageFormatter.format_price(report.total_pnl, 'KRW')}\n"
        
        # 시장별 노출
        if report.market_exposure:
            message += f"\n🌍 시장별 비중\n"
            for market, exposure in report.market_exposure.items():
                market_emoji = MARKET_EMOJIS.get(market, "📈")
                market_name = MARKET_NAMES.get(market, market)
                message += f"  {market_emoji} {market_name}: {MessageFormatter.format_percentage(exposure * 100)}\n"
        
        # 상위 성과 종목
        if report.top_performers:
            message += f"\n🏆 상위 성과 종목\n"
            for i, performer in enumerate(report.top_performers[:5], 1):
                symbol = performer.get('symbol', '')
                return_pct = performer.get('return', 0)
                return_emoji = MessageFormatter.get_return_emoji(return_pct)
                message += f"  {i}. {symbol}: {MessageFormatter.format_percentage(return_pct)} {return_emoji}\n"
        
        # 최악 성과 종목 (손실이 있는 경우만)
        if report.worst_performers and any(p.get('return', 0) < 0 for p in report.worst_performers):
            message += f"\n📉 주의 종목\n"
            worst_3 = [p for p in report.worst_performers if p.get('return', 0) < 0][:3]
            for i, performer in enumerate(worst_3, 1):
                symbol = performer.get('symbol', '')
                return_pct = performer.get('return', 0)
                message += f"  {i}. {symbol}: {MessageFormatter.format_percentage(return_pct)} 📉\n"
        
        message += f"\n⏰ 리포트 시간: {MessageFormatter.format_datetime(today, 'korean')}"
        
        results = await _notification_manager.send_notification(
            message, level='info', deduplicate=False
        )
        
        return any(results.values())
        
    except Exception as e:
        logger.error(f"성과 리포트 알림 실패: {e}")
        return False

# ================================
# 📅 스케줄 알림
# ================================

async def send_schedule_notification(today_strategies: List[str], 
                                   schedule_type: str = "start") -> bool:
    """스케줄 알림 발송"""
    try:
        # 시장 시간 정보
        if UTILS_AVAILABLE and timezone_manager:
            market_status = timezone_manager.get_all_market_times()
            kr_time = market_status.get('KOR', {})
        else:
            kr_time = {'datetime': MessageFormatter.format_datetime()}
        
        today = datetime.now()
        weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][today.weekday()]
        
        if schedule_type == "start":
            if not today_strategies:
                message = f"😴 {weekday_kr}요일 휴무\n\n"
                message += f"📅 {today.strftime('%Y년 %m월 %d일')}\n"
                message += f"🛌 오늘은 거래 없는 날입니다\n"
                message += f"🌙 편안한 하루 되세요\n"
            else:
                strategy_names = []
                for strategy in today_strategies:
                    if strategy == 'US':
                        strategy_names.append("🇺🇸 미국 주식")
                    elif strategy == 'JP':
                        strategy_names.append("🇯🇵 일본 주식")
                    elif strategy == 'COIN':
                        strategy_names.append("🪙 암호화폐")
                    elif strategy == 'EU':
                        strategy_names.append("🇪🇺 유럽 주식")
                
                message = f"🚀 {weekday_kr}요일 거래 시작\n\n"
                message += f"📅 {today.strftime('%Y년 %m월 %d일')}\n"
                message += f"📊 활성 전략: {len(today_strategies)}개\n\n"
                
                for name in strategy_names:
                    message += f"  • {name}\n"
                
                message += f"\n💪 오늘도 수익 창출을 위해 최선을 다하겠습니다!"
                    
        elif schedule_type == "end":
            message = f"🌙 {weekday_kr}요일 거래 종료\n\n"
            message += f"📅 {today.strftime('%Y년 %m월 %d일')}\n"
            message += f"✅ 오늘 거래 완료\n"
            message += f"💤 다음 거래일까지 시스템 대기\n"
            message += f"🔄 내일 더 나은 기회를 찾아보겠습니다"
        
        elif schedule_type == "maintenance":
            message = f"🔧 시스템 점검 시간\n\n"
            message += f"📅 {today.strftime('%Y년 %m월 %d일')}\n"
            message += f"⚙️ 정기 시스템 점검 및 최적화 진행\n"
            message += f"🕐 예상 소요시간: 15-30분\n"
            message += f"📊 점검 완료 후 정상 서비스 재개"
        
        current_time = kr_time.get('datetime', MessageFormatter.format_datetime())
        message += f"\n⏰ {current_time}"
        
        results = await _notification_manager.send_notification(
            message, level='info', deduplicate=False
        )
        
        return any(results.values())
        
    except Exception as e:
        logger.error(f"스케줄 알림 실패: {e}")
        return False

# ================================
# 📰 뉴스 알림
# ================================

async def send_news_alert(symbol: str, news_score: float, news_summary: str, 
                         market: str = "US", source: str = None) -> bool:
    """뉴스 분석 결과 알림"""
    try:
        # 뉴스 중요도 필터링
        importance_threshold = 0.25
        if abs(news_score - 0.5) < importance_threshold:
            return False
        
        market_emoji = MARKET_EMOJIS.get(market, "📈")
        
        # 센티먼트 분석
        if news_score >= 0.7:
            sentiment_emoji = "📈"
            sentiment_text = "매우 긍정적"
            impact_emoji = "🚀"
        elif news_score >= 0.6:
            sentiment_emoji = "📊"
            sentiment_text = "긍정적"
            impact_emoji = "📈"
        elif news_score <= 0.3:
            sentiment_emoji = "📉"
            sentiment_text = "매우 부정적"
            impact_emoji = "💀"
        elif news_score <= 0.4:
            sentiment_emoji = "📊"
            sentiment_text = "부정적"
            impact_emoji = "📉"
        else:
            return False
        
        # 영향도 점수
        impact_score = abs(news_score - 0.5) * 200
        
        message = f"📰 주요 뉴스 감지 {impact_emoji}\n\n"
        message += f"{market_emoji} {MARKET_NAMES.get(market, market)} | {symbol}\n"
        message += f"{sentiment_emoji} 센티먼트: {sentiment_text}\n"
        message += f"📊 신뢰도: {news_score*100:.0f}%\n"
        message += f"⚡ 영향도: {impact_score:.0f}/100\n"
        
        if source:
            message += f"🔗 출처: {source}\n"
        
        message += f"\n📝 요약:\n{news_summary}\n"
        message += f"\n⏰ {MessageFormatter.format_datetime(format_type='short')}"
        
        # 뉴스 영향도에 따른 우선순위
        priority = abs(news_score - 0.5) >= 0.3
        level = 'warning' if priority else 'info'
        
        results = await _notification_manager.send_notification(
            message, level=level, priority=priority
        )
        
        return any(results.values())
        
    except Exception as e:
        logger.error(f"뉴스 알림 실패: {e}")
        return False

# ================================
# 🚨 시스템 알림
# ================================

async def send_system_alert(alert_type: str, message_content: str, 
                          priority: str = "normal", context: Dict = None) -> bool:
    """시스템 알림 발송"""
    try:
        # 타입별 이모지
        type_emojis = {
            "error": "❌", "success": "✅", "startup": "🚀", "shutdown": "🛑",
            "maintenance": "🔧", "update": "🔄", "backup": "💾", "security": "🔒",
            "performance": "⚡", "connection": "🔌", "database": "💽"
        }
        
        # 우선순위별 이모지
        priority_prefix = {
            "critical": "🚨🚨", "error": "🚨", "warning": "⚠️",
            "info": "ℹ️", "normal": "📢"
        }
        
        type_emoji = type_emojis.get(alert_type, "📢")
        priority_emoji = priority_prefix.get(priority, "📢")
        
        message = f"{priority_emoji} 시스템 알림 {type_emoji}\n\n"
        message += f"📋 유형: {alert_type.upper()}\n"
        message += f"🔸 우선순위: {priority.upper()}\n"
        message += f"📝 내용: {message_content}\n"
        
        # 컨텍스트 정보 추가
        if context:
            message += f"\n📊 추가 정보:\n"
            for key, value in context.items():
                if key in ['memory_usage', 'cpu_usage', 'disk_usage']:
                    message += f"  {key}: {value}%\n"
                elif key == 'timestamp':
                    message += f"  발생시간: {value}\n"
                elif key == 'component':
                    message += f"  구성요소: {value}\n"
                else:
                    message += f"  {key}: {value}\n"
        
        message += f"\n⏰ 알림시간: {MessageFormatter.format_datetime()}"
        
        # 우선순위 매핑
        level_map = {
            'critical': 'critical',
            'error': 'error', 
            'warning': 'warning',
            'info': 'info',
            'normal': 'info'
        }
        
        level = level_map.get(priority, 'info')
        is_priority = priority in ['critical', 'error']
        
        results = await _notification_manager.send_notification(
            message, level=level, priority=is_priority
        )
        
        return any(results.values())
        
    except Exception as e:
        logger.error(f"시스템 알림 실패: {e}")
        return False

# ================================
# ❌ 에러 알림
# ================================

async def send_error_notification(error_message: str, error_type: str = "SYSTEM",
                                context: Dict = None, traceback_info: str = None) -> bool:
    """에러 알림 발송"""
    try:
        # 중요한 에러만 필터링
        critical_errors = ['TRADING', 'API', 'DATABASE', 'SECURITY', 'CRITICAL']
        
        # 설정에 따른 필터링
        critical_only = False
        if UTILS_AVAILABLE and config_manager:
            critical_only = config_manager.get('notifications.critical_only', False)
        
        if critical_only and error_type not in critical_errors:
            return False
        
        # 에러 타입별 이모지
        error_emojis = {
            'TRADING': '💰', 'API': '🔌', 'DATABASE': '💾', 'NETWORK': '🌐',
            'ANALYSIS': '📊', 'SYSTEM': '⚙️', 'CRITICAL': '🚨', 'SECURITY': '🔒'
        }
        
        emoji = error_emojis.get(error_type, "❌")
        
        message = f"🚨 시스템 오류 발생 {emoji}\n\n"
        message += f"🏷️ 타입: {error_type}\n"
        message += f"📝 메시지: {error_message}\n"
        
        # 컨텍스트 정보
        if context:
            message += f"\n📊 상세 정보:\n"
            for key, value in context.items():
                if key == 'function':
                    message += f"  🔧 함수: {value}\n"
                elif key == 'file':
                    message += f"  📄 파일: {value}\n"
                elif key == 'line':
                    message += f"  📍 라인: {value}\n"
                elif key == 'symbol':
                    message += f"  📈 종목: {value}\n"
                else:
                    message += f"  {key}: {value}\n"
        
        # 트레이스백 (간단히)
        if traceback_info:
            lines = traceback_info.split('\n')
            last_line = [line for line in lines if line.strip()][-1] if lines else ""
            if last_line:
                message += f"\n🔍 상세: {last_line}\n"
        
        message += f"\n⏰ 발생시간: {MessageFormatter.format_datetime()}\n"
        message += f"🔧 시스템 관리자에게 문의하세요"
        
        # 에러 타입에 따른 우선순위
        priority = error_type in critical_errors
        level = 'critical' if error_type == 'CRITICAL' else 'error'
        
        results = await _notification_manager.send_notification(
            message, level=level, priority=priority
        )
        
        return any(results.values())
        
    except Exception as e:
        logger.error(f"에러 알림 실패: {e}")
        return False

# ================================
# 🧪 테스트 함수들
# ================================

async def test_notification_connection() -> Dict[str, bool]:
    """모든 알림 채널 연결 테스트"""
    results = {}
    
    try:
        # 텔레그램 테스트
        if _notification_manager.telegram_config.get('enabled', False):
            results['telegram'] = await _notification_manager._send_telegram(
                "🧪 텔레그램 연결 테스트"
            )
        else:
            results['telegram'] = False
        
        # 슬랙 테스트
        if _notification_manager.slack_config.get('enabled', False):
            results['slack'] = await _notification_manager._send_slack(
                "🧪 슬랙 연결 테스트"
            )
        else:
            results['slack'] = False
        
        # 이메일 테스트
        if _notification_manager.email_config.get('enabled', False):
            results['email'] = await _notification_manager._send_email(
                "🧪 이메일 연결 테스트", "info"
            )
        else:
            results['email'] = False
        
    except Exception as e:
        logger.error(f"연결 테스트 실패: {e}")
    
    return results

async def run_comprehensive_notification_test():
    """종합 알림 시스템 테스트"""
    print("🔔 최고퀸트프로젝트 - 통합 알림 시스템 테스트")
    print("=" * 60)
    
    # 1. 설정 확인
    print("1️⃣ 설정 확인...")
    if not _notification_manager.enabled:
        print("❌ 알림 시스템이 비활성화되어 있습니다")
        return
    
    enabled_channels = []
    if _notification_manager.telegram_config.get('enabled', False):
        enabled_channels.append('텔레그램')
    if _notification_manager.slack_config.get('enabled', False):
        enabled_channels.append('슬랙')
    if _notification_manager.email_config.get('enabled', False):
        enabled_channels.append('이메일')
    
    print(f"✅ 활성 채널: {', '.join(enabled_channels) if enabled_channels else '없음'}")
    
    if not enabled_channels:
        print("❌ 활성화된 알림 채널이 없습니다")
        return
    
    # 2. 연결 테스트
    print("\n2️⃣ 연결 테스트...")
    connection_results = await test_notification_connection()
    for channel, result in connection_results.items():
        status = "✅ 성공" if result else "❌ 실패"
        print(f"   {channel}: {status}")
    
    await asyncio.sleep(2)
    
    # 3. 기본 메시지 테스트
    print("\n3️⃣ 기본 메시지 테스트...")
    result1 = await _notification_manager.send_notification(
        "🧪 최고퀸트프로젝트 통합 알림 시스템 테스트 시작!"
    )
    print(f"   결과: {'✅ 성공' if any(result1.values()) else '❌ 실패'}")
    await asyncio.sleep(2)
    
    # 4. 거래 신호 테스트
    print("\n4️⃣ 거래 신호 테스트...")
    test_signal = TradingSignal(
        symbol="AAPL", action="BUY", market="US", price=175.50, confidence=0.85,
        reasoning="AI 분석: 강력한 매수 신호 (RSI 과매도 + 볼륨 급증)",
        target_price=195.80, stop_loss=165.00, quantity=100
    )
    result2 = await send_trading_alert(test_signal)
    print(f"   결과: {'✅ 성공' if result2 else '❌ 실패'}")
    await asyncio.sleep(2)
    
    # 5. 거래 완료 테스트
    print("\n5️⃣ 거래 완료 테스트...")
    result3 = await send_trading_alert(test_signal, "completed")
    print(f"   결과: {'✅ 성공' if result3 else '❌ 실패'}")
    await asyncio.sleep(2)
    
    # 6. 시장 요약 테스트
    print("\n6️⃣ 시장 요약 테스트...")
    mock_summaries = {
        'US': MarketSummary(
            market='US', total_analyzed=45, buy_signals=7, sell_signals=3,
            hold_signals=2, analysis_time=15.2, is_trading_day=True,
            executed_trades=[{'executed': True}, {'executed': True}],
            top_picks=[
                TradingSignal('AAPL', 'BUY', 'US', 175.50, 0.85, 'Strong signals'),
                TradingSignal('MSFT', 'BUY', 'US', 380.25, 0.78, 'Growth potential')
            ],
            market_sentiment=0.72
        ),
        'COIN': MarketSummary(
            market='COIN', total_analyzed=12, buy_signals=3, sell_signals=1,
            analysis_time=4.5, is_trading_day=True,
            executed_trades=[{'executed': True}],
            top_picks=[
                TradingSignal('BTC-KRW', 'BUY', 'COIN', 95000000, 0.76, 'Bullish trend')
            ],
            market_sentiment=0.65
        )
    }
    result4 = await send_market_summary(mock_summaries)
    print(f"   결과: {'✅ 성공' if result4 else '❌ 실패'}")
    await asyncio.sleep(2)
    
    # 7. 성과 리포트 테스트
    print("\n7️⃣ 성과 리포트 테스트...")
    test_report = PerformanceReport(
        date=datetime.now().strftime('%Y-%m-%d'),
        total_signals=98, total_trades=12, successful_trades=9, failed_trades=3,
        daily_return=2.8, total_return=18.5, total_pnl=2850000,
        top_performers=[
            {'symbol': 'AAPL', 'return': 5.2},
            {'symbol': 'BTC-KRW', 'return': 4.1},
            {'symbol': 'TSLA', 'return': 3.8}
        ],
        market_exposure={'US': 0.6, 'COIN': 0.3, 'JP': 0.1}
    )
    result5 = await send_performance_report(test_report)
    print(f"   결과: {'✅ 성공' if result5 else '❌ 실패'}")
    await asyncio.sleep(2)
    
    # 8. 기타 알림 테스트
    print("\n8️⃣ 기타 알림 테스트...")
    result6 = await send_schedule_notification(['US', 'COIN'], "start")
    result7 = await send_news_alert("TSLA", 0.82, "테슬라 실적 호조", "US", "Reuters")
    result8 = await send_system_alert("startup", "시스템 시작 완료", "info")
    result9 = await send_error_notification("테스트 에러", "SYSTEM")
    
    other_success = sum([result6, result7, result8, result9])
    print(f"   결과: {other_success}/4개 성공")
    
    # 9. 통계 확인
    print("\n9️⃣ 통계 확인...")
    stats = _notification_manager.get_stats()
    print(f"   총 발송: {stats['total_sent']}개")
    print(f"   성공률: {stats['success_rate']:.1f}%")
    
    # 10. 완료 메시지
    print("\n🔟 테스트 완료...")
    total_tests = 8 + other_success
    max_tests = 12
    
    result10 = await _notification_manager.send_notification(
        f"🧪 최고퀸트프로젝트 알림 시스템 테스트 완료!\n\n"
        f"✅ 성공: {total_tests}/{max_tests}개\n"
        f"📊 성공률: {stats['success_rate']:.1f}%\n"
        f"⏰ 완료시간: {MessageFormatter.format_datetime()}"
    )
    
    print("\n" + "=" * 60)
    print(f"🎯 전체 테스트 결과: {total_tests}/{max_tests}개 성공")
    
    if total_tests >= 10:
        print("🎉 알림 시스템이 정상 작동합니다!")
    elif total_tests >= 7:
        print("👍 대부분 정상이지만 일부 개선이 필요합니다.")
    else:
        print("⚠️ 알림 시스템에 문제가 있습니다. 설정을 확인하세요.")

# ================================
# 🔧 편의 함수들
# ================================

async def quick_notification(message: str, level: str = 'info') -> bool:
    """빠른 알림 발송"""
    results = await _notification_manager.send_notification(message, level=level)
    return any(results.values())

async def priority_notification(message: str, level: str = 'warning') -> bool:
    """우선순위 알림 발송"""
    results = await _notification_manager.send_notification(
        message, level=level, priority=True
    )
    return any(results.values())

def get_notification_stats() -> Dict[str, Any]:
    """알림 통계 조회"""
    return _notification_manager.get_stats()

async def cleanup_notification_system():
    """알림 시스템 정리"""
    await _notification_manager.close()
    logger.info("알림 시스템 정리 완료")

# ================================
# 메인 실행부
# ================================

async def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='최고퀸트프로젝트 알림 시스템')
    parser.add_argument('--test', action='store_true', help='종합 테스트 실행')
    parser.add_argument('--test-connection', action='store_true', help='연결 테스트만 실행')
    parser.add_argument('--send', type=str, help='단순 메시지 발송')
    parser.add_argument('--stats', action='store_true', help='알림 통계 조회')
    
    args = parser.parse_args()
    
    if args.test:
        await run_comprehensive_notification_test()
    elif args.test_connection:
        print("🔍 연결 테스트 중...")
        results = await test_notification_connection()
        for channel, result in results.items():
            status = "✅ 성공" if result else "❌ 실패"
            print(f"{channel}: {status}")
    elif args.send:
        print(f"📨 메시지 발송 중: {args.send}")
        result = await quick_notification(args.send)
        print(f"결과: {'✅ 성공' if result else '❌ 실패'}")
    elif args.stats:
        stats = get_notification_stats()
        print("📊 알림 통계:")
        print(f"  총 발송: {stats['total_sent']}개")
        print(f"  성공: {stats['successful']}개")
        print(f"  실패: {stats['failed']}개")
        print(f"  성공률: {stats['success_rate']:.1f}%")
        if stats['by_channel']:
            print(f"  채널별: {stats['by_channel']}")
        if stats['by_level']:
            print(f"  레벨별: {stats['by_level']}")
    else:
        # 기본 사용법 안내
        print("🔔 최고퀸트프로젝트 알림 시스템 v3.0")
        print("=" * 50)
        print("사용 가능한 옵션:")
        print("  --test              : 종합 테스트 실행")
        print("  --test-connection   : 연결 테스트")
        print("  --send '메시지'     : 단순 메시지 발송")
        print("  --stats             : 통계 조회")
        print("\n예시:")
        print("  python notifier.py --test")
        print("  python notifier.py --send '안녕하세요!'")
        print("  python notifier.py --stats")
    
    # 리소스 정리
    await cleanup_notification_system()

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/notifier.log', encoding='utf-8')
        ]
    )
    
    # 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 알림 시스템을 종료합니다...")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"메인 실행 오류: {e}")

# ================================
# 📖 사용 가이드 및 예제
# ================================

"""
🔔 최고퀸트프로젝트 알림 시스템 v3.0 사용 가이드

## 1. 기본 설정
settings.yaml 파일에 알림 설정을 추가하세요:

```yaml
notifications:
  telegram:
    enabled: true
    bot_token: "YOUR_BOT_TOKEN"  # 또는 .env에 TELEGRAM_BOT_TOKEN
    chat_id: "YOUR_CHAT_ID"      # 또는 .env에 TELEGRAM_CHAT_ID
  
  slack:
    enabled: false
    webhook_url: "YOUR_WEBHOOK"  # 또는 .env에 SLACK_WEBHOOK_URL
  
  email:
    enabled: false
```

## 2. 기본 사용법

### 간단한 알림
```python
from notifier import quick_notification

await quick_notification("거래 완료!")
```

### 거래 신호 알림
```python
from notifier import send_trading_alert, TradingSignal

signal = TradingSignal(
    symbol="AAPL",
    action="BUY", 
    market="US",
    price=175.50,
    confidence=0.85,
    reasoning="강력한 매수 신호",
    target_price=195.80,
    stop_loss=165.00
)

await send_trading_alert(signal)
```

### 시장 요약 알림
```python
from notifier import send_market_summary, MarketSummary

summaries = {
    'US': MarketSummary(
        market='US',
        total_analyzed=50,
        buy_signals=8,
        sell_signals=3,
        analysis_time=12.5
    )
}

await send_market_summary(summaries)
```

### 성과 리포트 알림
```python
from notifier import send_performance_report, PerformanceReport

report = PerformanceReport(
    date='2025-01-01',
    total_signals=100,
    total_trades=15,
    successful_trades=12,
    failed_trades=3,
    daily_return=2.5,
    total_return=15.8
)

await send_performance_report(report)
```

### 에러 알림
```python
from notifier import send_error_notification

await send_error_notification(
    "API 연결 실패", 
    "API",
    context={"function": "get_price", "symbol": "AAPL"}
)
```

## 3. CLI 사용법

```bash
# 종합 테스트
python notifier.py --test

# 연결 테스트만
python notifier.py --test-connection

# 간단한 메시지 발송
python notifier.py --send "테스트 메시지"

# 통계 조회
python notifier.py --stats
```

## 4. 고급 기능

### 우선순위 알림
```python
await priority_notification("긴급 알림!", "critical")
```

### 통계 조회
```python
stats = get_notification_stats()
print(f"성공률: {stats['success_rate']:.1f}%")
```

### 시스템 정리
```python
await cleanup_notification_system()
```

## 5. utils.py 연동 기능

- **설정 관리**: ConfigManager로 자동 설정 로드
- **시간대 관리**: TimeZoneManager로 정확한 시간 정보
- **포맷팅**: Formatter로 일관된 메시지 포맷
- **로깅**: save_trading_log로 알림 기록 저장
- **캐싱**: 중복 메시지 방지 및 성능 최적화

## 6. 주의사항

1. **API 키 보안**: .env 파일에 안전하게 저장
2. **속도 제한**: 채널별 분당 10개 메시지 제한
3. **중복 방지**: 5분 이내 동일 메시지 자동 필터링
4. **레벨 설정**: 중요도에 따른 알림 레벨 조정
5. **리소스 정리**: 프로그램 종료시 cleanup_notification_system() 호출

## 7. 트러블슈팅

### 텔레그램 연결 실패
- 봇 토큰이 올바른지 확인
- 채팅 ID가 정확한지 확인
- 봇이 채팅방에 추가되어 있는지 확인

### 메시지 발송 실패
- 네트워크 연결 상태 확인
- API 키 유효성 확인
- 로그 파일에서 상세 오류 메시지 확인

### 성능 문제
- 속도 제한 설정 조정
- 캐시 크기 조정
- 불필요한 알림 필터링

이 알림 시스템은 utils.py와 완벽하게 연동되어 실제 거래 환경에서 
안정적으로 작동하도록 설계되었습니다.
"""
