"""
🔔 최고퀸트프로젝트 - 텔레그램 알림 시스템
=======================================

완전한 알림 시스템:
- 📱 매매 신호/완료 알림
- 📊 시장 분석 요약
- 📰 뉴스 분석 결과  
- 📅 스케줄링 알림
- 🚨 시스템 상태 알림
- 📈 일일 성과 리포트
- 🧪 완전한 테스트 시스템

Author: 최고퀸트팀
Version: 1.0.0
Project: 최고퀸트프로젝트
"""

import asyncio
import aiohttp
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

# 설정 파일 로드
def load_config() -> Dict:
    """설정 파일 로드"""
    try:
        with open('configs/settings.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"설정 파일 로드 실패: {e}")
        return {}

# 상수 정의
MARKET_EMOJIS = {
    'US': '🇺🇸',
    'JP': '🇯🇵', 
    'COIN': '🪙',
    'CRYPTO': '🪙'
}

MARKET_NAMES = {
    'US': '미국',
    'JP': '일본',
    'COIN': '암호화폐',
    'CRYPTO': '암호화폐'
}

ACTION_EMOJIS = {
    'buy': '💰',
    'sell': '💸',
    'hold': '⏸️'
}

def format_price(price: float, market: str) -> str:
    """시장별 가격 포맷팅"""
    try:
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
    except:
        return str(price)

class TelegramNotifier:
    """텔레그램 알림 클래스"""
    
    def __init__(self):
        self.config = load_config()
        self.telegram_config = self.config.get('notifications', {}).get('telegram', {})
        self.bot_token = self.telegram_config.get('bot_token', '')
        self.chat_id = self.telegram_config.get('chat_id', '')
        self.enabled = self.telegram_config.get('enabled', False)
        
        # API 기본 설정
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.session = None
        
    async def _get_session(self):
        """HTTP 세션 가져오기"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self.session
    
    async def _send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """텔레그램 메시지 전송 (내부 함수)"""
        if not self.enabled or not self.bot_token or not self.chat_id:
            return False
        
        try:
            session = await self._get_session()
            url = f"{self.base_url}/sendMessage"
            
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    return True
                else:
                    error_text = await response.text()
                    logging.error(f"텔레그램 API 오류 ({response.status}): {error_text}")
                    return False
                    
        except asyncio.TimeoutError:
            logging.error("텔레그램 메시지 전송 타임아웃")
            return False
        except Exception as e:
            logging.error(f"텔레그램 메시지 전송 실패: {e}")
            return False
    
    async def close(self):
        """세션 종료"""
        if self.session and not self.session.closed:
            await self.session.close()

# 전역 notifier 인스턴스
_notifier = TelegramNotifier()

async def send_telegram_message(message: str) -> bool:
    """기본 텔레그램 메시지 발송"""
    try:
        return await _notifier._send_message(message)
    except Exception as e:
        logging.error(f"메시지 발송 실패: {e}")
        return False

async def send_trading_alert(market: str, symbol: str, action: str, price: float, 
                          confidence: float, reasoning: str, target_price: float = None,
                          execution_status: str = "signal") -> bool:
    """매매 신호/완료 알림 발송 (완전체 버전)"""
    try:
        config = load_config()
        telegram_config = config.get('notifications', {}).get('telegram', {})
        
        if not telegram_config.get('enabled', False):
            return False
        
        # 액션에 따른 이모지 매핑
        action_emoji = ACTION_EMOJIS.get(action.lower(), "📊")
        market_emoji = MARKET_EMOJIS.get(market, "📈")
        
        # 신뢰도에 따른 강조 표시
        if confidence >= 0.8:
            confidence_emoji = "🔥"
        elif confidence >= 0.6:
            confidence_emoji = "⭐"
        else:
            confidence_emoji = ""
        
        # 실행 상태에 따른 메시지 차별화
        if execution_status == "completed":
            # 매매 완료 알림
            message = f"✅ {action.upper()} 완료\n\n"
            message += f"{market_emoji} {market} | {symbol}\n"
            message += f"💰 실행가: {format_price(price, market)}\n"
            message += f"🎯 신뢰도: {confidence*100:.0f}% {confidence_emoji}\n"
            
            if target_price:
                expected_return = ((target_price - price) / price) * 100
                message += f"🎪 목표가: {format_price(target_price, market)}\n"
                message += f"📈 기대수익: {expected_return:+.1f}%\n"
            
            message += f"\n💡 {reasoning}\n"
            message += f"⏰ 실행시간: {datetime.now().strftime('%H:%M:%S')}"
            
        elif execution_status == "failed":
            # 매매 실패 알림
            message = f"❌ {action.upper()} 실패\n\n"
            message += f"{market_emoji} {market} | {symbol}\n"
            message += f"💵 목표가: {format_price(price, market)}\n"
            message += f"🎯 신뢰도: {confidence*100:.0f}% {confidence_emoji}\n"
            message += f"\n💡 사유: {reasoning}\n"
            message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            
        else:
            # 기본 신호 알림
            message = f"{action_emoji} {action.upper()} 신호\n\n"
            message += f"{market_emoji} {market} | {symbol}\n"
            message += f"💵 현재가: {format_price(price, market)}\n"
            message += f"🎯 신뢰도: {confidence*100:.0f}% {confidence_emoji}\n"
            
            if target_price:
                expected_return = ((target_price - price) / price) * 100
                message += f"🎪 목표가: {format_price(target_price, market)}\n"
                message += f"📈 기대수익: {expected_return:+.1f}%\n"
            
            message += f"\n💡 {reasoning}\n"
            message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        return await send_telegram_message(message)
        
    except Exception as e:
        logging.error(f"매매 알림 실패: {e}")
        return False

async def send_market_summary(market_summaries: Dict) -> bool:
    """시장 요약 알림 발송 (완전체 버전)"""
    try:
        config = load_config()
        telegram_config = config.get('notifications', {}).get('telegram', {})
        
        if not telegram_config.get('enabled', False):
            return False
        
        # 요약 통계 계산
        total_analyzed = 0
        total_buy = 0
        total_sell = 0
        total_executed = 0
        
        for summary in market_summaries.values():
            if hasattr(summary, 'total_analyzed'):
                total_analyzed += summary.total_analyzed
                total_buy += summary.buy_signals
                total_sell += summary.sell_signals
                total_executed += len([t for t in summary.executed_trades if t.executed])
            else:
                # 딕셔너리 형태인 경우 (호환성)
                total_analyzed += summary.get('total_analyzed', 0)
                total_buy += summary.get('buy_signals', 0)
                total_sell += summary.get('sell_signals', 0)
        
        # 메시지 구성
        message = f"🏆 최고퀸트프로젝트 분석 완료\n"
        message += f"=" * 30 + "\n\n"
        message += f"📊 전체 분석 결과\n"
        message += f"🔍 분석 종목: {total_analyzed}개\n"
        message += f"💰 매수 신호: {total_buy}개\n"
        message += f"💸 매도 신호: {total_sell}개\n"
        if total_executed > 0:
            message += f"✅ 실행 거래: {total_executed}개\n"
        message += f"\n"
        
        # 시장별 상세 정보
        for market, summary in market_summaries.items():
            market_emoji = MARKET_EMOJIS.get(market, "📈")
            market_name = MARKET_NAMES.get(market, market)
            
            # MarketSummary 객체 또는 딕셔너리 모두 지원
            if hasattr(summary, 'total_analyzed'):
                buy_signals = summary.buy_signals
                sell_signals = summary.sell_signals
                analysis_time = summary.analysis_time
                top_picks = summary.top_picks
                executed_trades = summary.executed_trades
                is_trading_day = summary.is_trading_day
            else:
                buy_signals = summary.get('buy_signals', 0)
                sell_signals = summary.get('sell_signals', 0)
                analysis_time = summary.get('analysis_time', 0)
                top_picks = summary.get('top_picks', [])
                executed_trades = summary.get('executed_trades', [])
                is_trading_day = summary.get('is_trading_day', True)
            
            message += f"{market_emoji} {market_name}"
            if not is_trading_day:
                message += " (오늘 휴무)"
            message += f"\n"
            message += f"  📈 매수: {buy_signals}개\n"
            message += f"  📉 매도: {sell_signals}개\n"
            message += f"  ⏱️ 소요: {analysis_time:.1f}초\n"
            
            # 실행된 거래
            if executed_trades:
                executed_count = len([t for t in executed_trades if t.executed])
                if executed_count > 0:
                    message += f"  ✅ 실행: {executed_count}개\n"
            
            # 상위 추천 종목
            if top_picks:
                message += f"  🎯 추천: "
                top_3 = top_picks[:3]
                if hasattr(top_picks[0], 'symbol'):
                    # TradingSignal 객체
                    symbols = [f"{pick.symbol}({pick.confidence*100:.0f}%)" for pick in top_3]
                else:
                    # 딕셔너리
                    symbols = [f"{pick['symbol']}({pick['confidence']*100:.0f}%)" for pick in top_3]
                message += ", ".join(symbols)
            message += "\n\n"
        
        message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await send_telegram_message(message)
        
    except Exception as e:
        logging.error(f"시장 요약 알림 실패: {e}")
        return False

async def send_schedule_notification(today_strategies: List[str], schedule_type: str = "start") -> bool:
    """스케줄 기반 알림 발송 (신규)"""
    try:
        config = load_config()
        telegram_config = config.get('notifications', {}).get('telegram', {})
        
        if not telegram_config.get('enabled', False):
            return False
        
        today = datetime.now()
        weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][today.weekday()]
        
        if schedule_type == "start":
            if not today_strategies:
                message = f"😴 {weekday_kr}요일 휴무\n\n"
                message += f"📅 {today.strftime('%Y-%m-%d')}\n"
                message += f"🛌 오늘은 거래 없는 날입니다\n"
                message += f"⏰ {today.strftime('%H:%M:%S')}"
            else:
                strategy_names = []
                for strategy in today_strategies:
                    if strategy == 'US':
                        strategy_names.append("🇺🇸 미국 주식")
                    elif strategy == 'JP':
                        strategy_names.append("🇯🇵 일본 주식")
                    elif strategy == 'COIN':
                        strategy_names.append("🪙 암호화폐")
                
                message = f"🚀 {weekday_kr}요일 거래 시작\n\n"
                message += f"📅 {today.strftime('%Y-%m-%d')}\n"
                message += f"📊 활성 전략: {len(today_strategies)}개\n"
                message += "\n".join([f"  • {name}" for name in strategy_names])
                message += f"\n\n⏰ {today.strftime('%H:%M:%S')}"
                
        elif schedule_type == "end":
            message = f"🌙 {weekday_kr}요일 거래 종료\n\n"
            message += f"📅 {today.strftime('%Y-%m-%d')}\n"
            message += f"✅ 오늘 거래 완료\n"
            message += f"💤 다음 거래일까지 대기\n"
            message += f"⏰ {today.strftime('%H:%M:%S')}"
        
        return await send_telegram_message(message)
        
    except Exception as e:
        logging.error(f"스케줄 알림 실패: {e}")
        return False

async def send_news_alert(symbol: str, news_score: float, news_summary: str, market: str = "US") -> bool:
    """뉴스 분석 결과 알림 (신규)"""
    try:
        config = load_config()
        telegram_config = config.get('notifications', {}).get('telegram', {})
        
        if not telegram_config.get('enabled', False):
            return False
        
        # 뉴스 중요도 체크 (높은 스코어만 알림)
        if abs(news_score - 0.5) < 0.2:  # 중립적인 뉴스는 제외
            return False
        
        market_emoji = MARKET_EMOJIS.get(market, "📈")
        
        # 뉴스 센티먼트에 따른 이모지
        if news_score >= 0.7:
            sentiment_emoji = "📈"
            sentiment_text = "긍정적"
        elif news_score <= 0.3:
            sentiment_emoji = "📉"
            sentiment_text = "부정적"
        else:
            return False  # 중립적 뉴스는 알림 안함
        
        message = f"📰 주요 뉴스 감지\n\n"
        message += f"{market_emoji} {market} | {symbol}\n"
        message += f"{sentiment_emoji} 센티먼트: {sentiment_text} ({news_score*100:.0f}%)\n\n"
        message += f"📝 요약: {news_summary}\n\n"
        message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        return await send_telegram_message(message)
        
    except Exception as e:
        logging.error(f"뉴스 알림 실패: {e}")
        return False

async def send_system_alert(alert_type: str, message_content: str, priority: str = "normal") -> bool:
    """시스템 상태 알림 (신규)"""
    try:
        config = load_config()
        telegram_config = config.get('notifications', {}).get('telegram', {})
        
        if not telegram_config.get('enabled', False):
            return False
        
        # 우선순위에 따른 이모지
        priority_emojis = {
            "critical": "🚨",
            "warning": "⚠️", 
            "info": "ℹ️",
            "normal": "📢"
        }
        
        # 알림 타입별 이모지
        type_emojis = {
            "error": "❌",
            "success": "✅",
            "startup": "🚀",
            "shutdown": "🛑",
            "maintenance": "🔧"
        }
        
        emoji = type_emojis.get(alert_type, priority_emojis.get(priority, "📢"))
        
        message = f"{emoji} 시스템 알림\n\n"
        message += f"📋 타입: {alert_type.upper()}\n"
        message += f"📝 내용: {message_content}\n"
        message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await send_telegram_message(message)
        
    except Exception as e:
        logging.error(f"시스템 알림 실패: {e}")
        return False

async def send_daily_report(performance_data: Dict) -> bool:
    """일일 성과 리포트 알림 (신규)"""
    try:
        config = load_config()
        telegram_config = config.get('notifications', {}).get('telegram', {})
        
        if not telegram_config.get('enabled', False):
            return False
        
        today = datetime.now()
        
        message = f"📊 일일 성과 리포트\n"
        message += f"=" * 20 + "\n\n"
        message += f"📅 {today.strftime('%Y년 %m월 %d일 (%A)')}\n\n"
        
        # 거래 통계
        total_signals = performance_data.get('total_signals', 0)
        total_trades = performance_data.get('total_trades', 0)
        successful_trades = performance_data.get('successful_trades', 0)
        
        message += f"📈 거래 통계\n"
        message += f"  🔍 분석 신호: {total_signals}개\n"
        message += f"  💰 실행 거래: {total_trades}개\n"
        if total_trades > 0:
            success_rate = (successful_trades / total_trades) * 100
            message += f"  ✅ 성공률: {success_rate:.1f}%\n"
        message += "\n"
        
        # 수익률 (있는 경우)
        daily_return = performance_data.get('daily_return')
        if daily_return is not None:
            return_emoji = "📈" if daily_return >= 0 else "📉"
            message += f"💵 수익률\n"
            message += f"  {return_emoji} 일일: {daily_return:+.2f}%\n"
            
            total_return = performance_data.get('total_return')
            if total_return is not None:
                message += f"  🎯 누적: {total_return:+.2f}%\n"
            message += "\n"
        
        # 상위 성과 종목
        top_performers = performance_data.get('top_performers', [])
        if top_performers:
            message += f"🏆 상위 성과 종목\n"
            for i, performer in enumerate(top_performers[:3], 1):
                symbol = performer.get('symbol', '')
                return_pct = performer.get('return', 0)
                message += f"  {i}. {symbol}: {return_pct:+.1f}%\n"
            message += "\n"
        
        message += f"⏰ 리포트 시간: {today.strftime('%H:%M:%S')}"
        
        return await send_telegram_message(message)
        
    except Exception as e:
        logging.error(f"일일 리포트 알림 실패: {e}")
        return False

async def send_error_notification(error_message: str, error_type: str = "SYSTEM") -> bool:
    """에러 알림 발송 (업데이트)"""
    try:
        config = load_config()
        telegram_config = config.get('notifications', {}).get('telegram', {})
        
        if not telegram_config.get('enabled', False):
            return False
        
        # 중요한 에러만 알림 (설정에 따라)
        critical_only = config.get('notifications', {}).get('critical_only', False)
        critical_errors = ['TRADING', 'API', 'DATABASE', 'CRITICAL']
        
        if critical_only and error_type not in critical_errors:
            return False
        
        # 에러 타입별 이모지
        error_emojis = {
            'TRADING': '💰',
            'API': '🔌',
            'DATABASE': '💾',
            'NETWORK': '🌐',
            'ANALYSIS': '📊',
            'SYSTEM': '⚙️',
            'CRITICAL': '🚨'
        }
        
        emoji = error_emojis.get(error_type, "❌")
        
        message = f"{emoji} 시스템 오류\n\n"
        message += f"🏷️ 타입: {error_type}\n"
        message += f"📝 메시지: {error_message}\n"
        message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        message += f"🔧 시스템 관리자에게 문의하세요"
        
        return await send_telegram_message(message)
        
    except Exception as e:
        logging.error(f"에러 알림 실패: {e}")
        return False

async def test_telegram_connection() -> bool:
    """텔레그램 연결 테스트"""
    try:
        config = load_config()
        telegram_config = config.get('notifications', {}).get('telegram', {})
        
        if not telegram_config.get('enabled', False):
            print("❌ 텔레그램 알림이 비활성화되어 있습니다")
            return False
        
        if not telegram_config.get('bot_token') or not telegram_config.get('chat_id'):
            print("❌ 텔레그램 설정이 incomplete합니다")
            print("📋 bot_token과 chat_id를 configs/settings.yaml에 설정하세요")
            return False
        
        # 간단한 연결 테스트
        result = await send_telegram_message("🧪 텔레그램 연결 테스트")
        
        if result:
            print("✅ 텔레그램 연결 성공")
            return True
        else:
            print("❌ 텔레그램 연결 실패")
            return False
            
    except Exception as e:
        print(f"❌ 텔레그램 연결 테스트 실패: {e}")
        return False

async def test_all_notifications():
    """🧪 전체 알림 시스템 테스트 (완전체 버전)"""
    print("🔔 최고퀸트프로젝트 알림 시스템 테스트")
    print("=" * 50)
    
    # 설정 확인
    config = load_config()
    telegram_config = config.get('notifications', {}).get('telegram', {})
    
    if not telegram_config.get('enabled', False):
        print("❌ 텔레그램 알림이 비활성화되어 있습니다")
        print("📋 configs/settings.yaml에서 telegram.enabled를 true로 설정하세요")
        return
    
    if not telegram_config.get('bot_token') or not telegram_config.get('chat_id'):
        print("❌ 텔레그램 설정이 incomplete합니다")
        print("📋 bot_token과 chat_id를 설정하세요")
        return
    
    print("✅ 텔레그램 설정 확인 완료")
    print()
    
    # 1. 기본 메시지 테스트
    print("1️⃣ 기본 메시지 테스트...")
    result1 = await send_telegram_message("🧪 최고퀸트프로젝트 알림 시스템 테스트 시작!")
    print(f"   결과: {'✅ 성공' if result1 else '❌ 실패'}")
    
    await asyncio.sleep(2)
    
    # 2. 매매 신호 알림 테스트
    print("2️⃣ 매매 신호 알림 테스트...")
    result2 = await send_trading_alert(
        market="US", symbol="AAPL", action="buy", price=175.50, 
        confidence=0.85, reasoning="버핏: 저PBR(1.2) | 린치: 저PEG(0.8) | 뉴스: 긍정적 실적 발표",
        target_price=195.80
    )
    print(f"   결과: {'✅ 성공' if result2 else '❌ 실패'}")
    
    await asyncio.sleep(2)
    
    # 3. 매매 완료 알림 테스트
    print("3️⃣ 매매 완료 알림 테스트...")
    result3 = await send_trading_alert(
        market="COIN", symbol="BTC-KRW", action="buy", price=95000000,
        confidence=0.78, reasoning="거래량 급증(2.3배) | RSI 과매도(25.4) | 뉴스: ETF 승인 소식",
        target_price=105000000, execution_status="completed"
    )
    print(f"   결과: {'✅ 성공' if result3 else '❌ 실패'}")
    
    await asyncio.sleep(2)
    
    # 4. 시장 요약 알림 테스트
    print("4️⃣ 시장 요약 알림 테스트...")
    mock_summaries = {
        'US': {
            'total_analyzed': 40,
            'buy_signals': 5,
            'sell_signals': 2,
            'analysis_time': 12.3,
            'is_trading_day': True,
            'executed_trades': [{'executed': True}, {'executed': True}],
            'top_picks': [
                {'symbol': 'AAPL', 'confidence': 0.85},
                {'symbol': 'MSFT', 'confidence': 0.78},
                {'symbol': 'NVDA', 'confidence': 0.72}
            ]
        },
        'JP': {
            'total_analyzed': 24,
            'buy_signals': 3,
            'sell_signals': 1,
            'analysis_time': 8.7,
            'is_trading_day': True,
            'executed_trades': [{'executed': True}],
            'top_picks': [
                {'symbol': '7203.T', 'confidence': 0.81},
                {'symbol': '6758.T', 'confidence': 0.74}
            ]
        },
        'COIN': {
            'total_analyzed': 10,
            'buy_signals': 2,
            'sell_signals': 0,
            'analysis_time': 3.2,
            'is_trading_day': True,
            'executed_trades': [{'executed': True}],
            'top_picks': [
                {'symbol': 'BTC-KRW', 'confidence': 0.78}
            ]
        }
    }
    result4 = await send_market_summary(mock_summaries)
    print(f"   결과: {'✅ 성공' if result4 else '❌ 실패'}")
    
    await asyncio.sleep(2)
    
    # 5. 스케줄 알림 테스트
    print("5️⃣ 스케줄 알림 테스트...")
    result5 = await send_schedule_notification(['US', 'JP'], "start")
    print(f"   결과: {'✅ 성공' if result5 else '❌ 실패'}")
    
    await asyncio.sleep(2)
    
    # 6. 뉴스 알림 테스트
    print("6️⃣ 뉴스 알림 테스트...")
    result6 = await send_news_alert(
        symbol="TSLA", news_score=0.8, 
        news_summary="테슬라 Q3 실적 예상치 크게 상회, 전기차 판매량 급증",
        market="US"
    )
    print(f"   결과: {'✅ 성공' if result6 else '❌ 실패'}")
    
    await asyncio.sleep(2)
    
    # 7. 시스템 알림 테스트
    print("7️⃣ 시스템 알림 테스트...")
    result7 = await send_system_alert(
        "startup", "최고퀸트프로젝트 시스템이 성공적으로 시작되었습니다", "info"
    )
    print(f"   결과: {'✅ 성공' if result7 else '❌ 실패'}")
    
    await asyncio.sleep(2)
    
    # 8. 일일 리포트 테스트
    print("8️⃣ 일일 리포트 테스트...")
    mock_performance = {
        'total_signals': 74,
        'total_trades': 8,
        'successful_trades': 6,
        'daily_return': 2.3,
        'total_return': 15.7,
        'top_performers': [
            {'symbol': 'AAPL', 'return': 4.2},
            {'symbol': 'BTC-KRW', 'return': 3.1},
            {'symbol': '7203.T', 'return': 2.8}
        ]
    }
    result8 = await send_daily_report(mock_performance)
    print(f"   결과: {'✅ 성공' if result8 else '❌ 실패'}")
    
    await asyncio.sleep(2)
    
    # 9. 에러 알림 테스트
    print("9️⃣ 에러 알림 테스트...")
    result9 = await send_error_notification("테스트용 오류 메시지입니다", "SYSTEM")
    print(f"   결과: {'✅ 성공' if result9 else '❌ 실패'}")
    
    await asyncio.sleep(2)
    
    # 10. 마무리 메시지
    print("🔟 테스트 완료 메시지...")
    success_count = sum([result1, result2, result3, result4, result5, result6, result7, result8, result9])
    result10 = await send_telegram_message(
        f"🧪 최고퀸트프로젝트 알림 시스템 테스트 완료!\n\n"
        f"✅ 성공: {success_count}/9개\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"   결과: {'✅ 성공' if result10 else '❌ 실패'}")
    
    print()
    print(f"🎯 전체 테스트 결과: {success_count + result10}/10개 성공")
    if success_count + result10 == 10:
        print("🎉 모든 알림 기능이 정상 작동합니다!")
    else:
        print("⚠️ 일부 알림에 문제가 있습니다. 설정을 확인하세요.")
        
if __name__ == "__main__":
    print("🔔 최고퀸트프로젝트 텔레그램 알림 시스템")
    print("=" * 50)
    
    # 기본 설정 체크
    config = load_config()
    if config:
        print("✅ 설정 파일 로드 성공")
        
        # 간단한 연결 테스트
        asyncio.run(test_telegram_connection())
        print()
        
        # 전체 알림 테스트
        asyncio.run(test_all_notifications())
    else:
        print("❌ 설정 파일을 찾을 수 없습니다")
        print("📋 configs/settings.yaml 파일을 확인하세요")