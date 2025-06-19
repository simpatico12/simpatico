# notifier.py
"""
텔레그램 알림 모듈 - 퀸트프로젝트 수준
안정성과 확장성을 고려한 설계
"""
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from functools import wraps
import time

from telegram import Bot
from telegram.error import TelegramError, NetworkError, TimedOut
from logger import logger
from config import get_config


class TelegramNotifier:
    """텔레그램 알림 전송 클래스"""
    
    def __init__(self):
        self.cfg = get_config()['telegram']
        self.bot = Bot(token=self.cfg['token'])
        self.chat_id = self.cfg['chat_id']
        self.retry_count = self.cfg.get('retry_count', 3)
        self.retry_delay = self.cfg.get('retry_delay', 2)
        
    def retry_on_failure(func):
        """실패시 재시도 데코레이터"""
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            for attempt in range(self.retry_count):
                try:
                    return await func(self, *args, **kwargs)
                except (NetworkError, TimedOut) as e:
                    if attempt < self.retry_count - 1:
                        logger.warning(f"전송 실패 {attempt + 1}회: {e}")
                        await asyncio.sleep(self.retry_delay)
                    else:
                        raise
                except Exception as e:
                    logger.error(f"예상치 못한 오류: {e}")
                    raise
            return None
        return wrapper
    
    @retry_on_failure
    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """기본 메시지 전송"""
        try:
            # 메시지 길이 체크 (텔레그램 제한: 4096자)
            if len(message) > 4096:
                return await self._send_long_message(message, parse_mode)
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            logger.info(f"메시지 전송 성공: {message[:50]}...")
            return True
            
        except TelegramError as e:
            logger.error(f"텔레그램 전송 실패: {e}")
            return False
    
    async def _send_long_message(self, message: str, parse_mode: str) -> bool:
        """긴 메시지 분할 전송"""
        chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
        for i, chunk in enumerate(chunks):
            await self.send_message(f"[{i+1}/{len(chunks)}]\n{chunk}", parse_mode)
            await asyncio.sleep(0.5)  # 연속 전송 방지
        return True
    
    async def send_trading_alert(self, data: Dict[str, Any]) -> bool:
        """트레이딩 알림 전송"""
        message = self._format_trading_message(data)
        return await self.send_message(message)
    
    def _format_trading_message(self, data: Dict[str, Any]) -> str:
        """트레이딩 메시지 포맷팅"""
        emoji = "🟢" if data.get('type') == 'buy' else "🔴"
        
        return f"""
{emoji} <b>{data.get('type', '').upper()} 신호</b>
━━━━━━━━━━━━━━━
📊 종목: {data.get('symbol', 'N/A')}
💰 가격: {data.get('price', 0):,.2f}
📈 수량: {data.get('quantity', 0)}
⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━
{data.get('reason', '')}
"""

    async def send_error_alert(self, error: Exception, context: str = "") -> bool:
        """에러 알림 전송"""
        message = f"""
⚠️ <b>시스템 에러</b>
━━━━━━━━━━━━━━━
📍 위치: {context}
❌ 에러: {type(error).__name__}
💬 내용: {str(error)}
⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━
"""
        return await self.send_message(message)
    
    async def send_batch_messages(self, messages: List[str]) -> List[bool]:
        """여러 메시지 일괄 전송"""
        results = []
        for msg in messages:
            result = await self.send_message(msg)
            results.append(result)
            await asyncio.sleep(1)  # API 제한 회피
        return results


# 싱글톤 인스턴스
notifier = TelegramNotifier()


# 동기 함수 래퍼 (기존 코드 호환성)
def send(message: str) -> None:
    """기존 동기 함수 인터페이스 유지"""
    asyncio.run(notifier.send_message(message))


# 사용 예시
if __name__ == "__main__":
    # 기본 메시지
    send("시스템 시작")
    
    # 트레이딩 알림
    trade_data = {
        'type': 'buy',
        'symbol': 'AAPL',
        'price': 150