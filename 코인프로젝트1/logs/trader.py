# trader.py
"""
고급 트레이딩 실행 모듈 - 퀸트프로젝트 수준
안정적인 주문 실행과 정교한 리스크 관리
"""
import time
import asyncio
from datetime import date, datetime
from typing import Dict, Optional, Tuple
from decimal import Decimal
from enum import Enum

import pyupbit
from config import get_config
from notifier import notifier
from logger import logger
from core.risk import check_asset_ratio
from db import SessionLocal, TradeRecord
from utils import get_price, get_total_asset_value, get_cash_balance, log_trade, save_trade


class OrderType(Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"


class Decision(Enum):
    """매매 결정"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class AdvancedTrader:
    """고급 트레이딩 클래스"""
    
    def __init__(self):
        self.cfg = get_config()
        self.upbit = pyupbit.Upbit(
            self.cfg['api']['access_key'],
            self.cfg['api']['secret_key']
        )
        self.max_retry = self.cfg.get('max_retry', 3)
        self.order_delay = self.cfg.get('order_delay', 0.5)
        
    def has_traded_today(self, asset_type: str) -> bool:
        """오늘 거래 여부 확인"""
        with SessionLocal() as session:
            today = date.today().isoformat()
            count = session.query(TradeRecord)\
                .filter(TradeRecord.asset_type == asset_type)\
                .filter(TradeRecord.timestamp.startswith(today))\
                .count()
            return count > 0
    
    def calculate_decision(self, fg: float, sentiment: str, asset: str) -> Tuple[Decision, int]:
        """매매 결정 계산 (확장 가능한 구조)"""
        # 기본 로직
        if fg <= 30 and sentiment == 'positive':
            return Decision.BUY, 95
        elif fg <= 50 and sentiment != 'negative':
            return Decision.BUY, 85
        elif fg >= 80 and sentiment == 'negative':
            return Decision.SELL, 90
        elif sentiment == 'negative':
            return Decision.SELL, 80
        else:
            return Decision.HOLD, 60
    
    def calculate_position_size(self, asset_type: str, total_value: float) -> float:
        """포지션 크기 계산"""
        base_pct = self.cfg['trading'][asset_type]['percentage'] / 100
        
        # 켈리 공식 기반 동적 조정 (옵션)
        # win_rate = self.get_historical_win_rate(asset_type)
        # kelly_f = (win_rate * 2 - 1) / 1  # 간단한 켈리
        # position_pct = min(base_pct, kelly_f * 0.25)  # 보수적 적용
        
        return total_value * base_pct
    
    def execute_buy_order(self, ticker: str, amount: float) -> Optional[Dict]:
        """매수 주문 실행 (재시도 포함)"""
        for attempt in range(self.max_retry):
            try:
                # 주문 전 잔고 확인
                balance = self.upbit.get_balance("KRW")
                if balance < amount:
                    logger.warning(f"잔고 부족: {balance:.0f} < {amount:.0f}")
                    return None
                
                # 시장가 매수
                order = self.upbit.buy_market_order(ticker, amount)
                
                # 주문 확인
                if 'error' in order:
                    raise Exception(order['error']['message'])
                
                # 체결 대기
                time.sleep(2)
                
                # 체결 확인
                order_info = self.upbit.get_order(order['uuid'])
                if order_info['state'] == 'done':
                    return order_info
                    
            except Exception as e:
                logger.error(f"매수 실패 {attempt+1}/{self.max_retry}: {e}")
                if attempt < self.max_retry - 1:
                    time.sleep(self.order_delay * (attempt + 1))
                    
        return None
    
    def execute_sell_order(self, ticker: str, volume: float) -> Optional[Dict]:
        """매도 주문 실행 (재시도 포함)"""
        for attempt in range(self.max_retry):
            try:
                # 보유량 확인
                asset = ticker.split('-')[1]
                balance = self.upbit.get_balance(asset)
                
                if balance < volume * 0.99:  # 수수료 고려
                    logger.warning(f"보유량 부족: {balance:.8f} < {volume:.8f}")
                    return None
                
                # 시장가 매도
                order = self.upbit.sell_market_order(ticker, volume)
                
                if 'error' in order:
                    raise Exception(order['error']['message'])
                
                time.sleep(2)
                
                # 체결 확인
                order_info = self.upbit.get_order(order['uuid'])
                if order_info['state'] == 'done':
                    return order_info
                    
            except Exception as e:
                logger.error(f"매도 실패 {attempt+1}/{self.max_retry}: {e}")
                if attempt < self.max_retry - 1:
                    time.sleep(self.order_delay * (attempt + 1))
                    
        return None
    
    async def execute_trade(self, asset: str, asset_type: str, fg: float, 
                           sentiment: str, **kwargs) -> None:
        """메인 트레이딩 함수"""
        try:
            # 일일 거래 제한 확인
            if self.has_traded_today(asset_type):
                logger.info(f"{asset_type} 일일 거래 완료")
                return
            
            # 계좌 정보 조회
            total_value = get_total_asset_value(self.upbit)
            cash = get_cash_balance(self.upbit)
            current_price = get_price(asset, asset_type)
            
            # 매매 결정
            decision, confidence = self.calculate_decision(fg, sentiment, asset)
            
            # 알림
            emoji = {"buy": "🟢", "sell": "🔴", "hold": "⏸️"}[decision.value]
            await notifier.send_message(
                f"{emoji} <b>[{asset_type.upper()}] {asset}</b>\n"
                f"결정: {decision.value.upper()} | 신뢰도: {confidence}%\n"
                f"FG: {fg} | 감성: {sentiment}"
            )
            
            if decision == Decision.HOLD:
                return
            
            # 리스크 체크
            if not check_asset_ratio(asset, asset_type, current_price, total_value, cash):
                await notifier.send_message(f"⚠️ {asset} 리스크 한도 초과")
                return
            
            # 주문 실행
            ticker = f"KRW-{asset}"
            order_result = None
            
            if decision == Decision.BUY:
                amount = self.calculate_position_size(asset_type, total_value)
                order_result = self.execute_buy_order(ticker, amount)
                
            elif decision == Decision.SELL:
                volume = self.upbit.get_balance(asset)
                if volume > 0:
                    order_result = self.execute_sell_order(ticker, volume)
            
            # 결과 처리
            if order_result:
                await self._process_order_result(
                    asset, asset_type, decision, confidence,
                    order_result, total_value, cash, current_price
                )
            else:
                await notifier.send_error_alert(
                    Exception("주문 실행 실패"),
                    f"{asset} {decision.value}"
                )
                
        except Exception as e:
            logger.error(f"트레이딩 오류 {asset}: {e}")
            await notifier.send_error_alert(e, "execute_trade")
    
    async def _process_order_result(self, asset: str, asset_type: str, 
                                   decision: Decision, confidence: int,
                                   order: Dict, total_value: float, 
                                   cash: float, current_price: float):
        """주문 결과 처리"""
        # 체결 정보 파싱
        executed_volume = float(order.get('executed_volume', 0))
        avg_price = float(order.get('avg_price', current_price))
        fee = float(order.get('paid_fee', 0))
        
        # 잔고 업데이트
        new_balance = self.upbit.get_balance(asset)
        new_cash = get_cash_balance(self.upbit)
        
        # 수익률 계산 (매도시)
        profit_rate = 0
        if decision == Decision.SELL and avg_price > 0:
            # 평균 매수가 조회 (구현 필요)
            # avg_buy_price = self.get_avg_buy_price(asset)
            # profit_rate = (avg_price - avg_buy_price) / avg_buy_price * 100
            pass
        
        # 거래 기록
        trade_data = {
            'decision': decision.value,
            'confidence_score': confidence,
            'executed_volume': executed_volume,
            'avg_price': avg_price,
            'fee': fee,
            'profit_rate': profit_rate
        }
        
        balance_data = {
            'asset_balance': new_balance,
            'cash_balance': new_cash,
            'total_asset': total_value
        }
        
        save_trade(asset, asset_type, trade_data, balance_data, current_price)
        log_trade(asset, trade_data, balance_data, current_price)
        
        # 성공 알림
        if decision == Decision.BUY:
            msg = f"✅ <b>{asset} 매수 완료</b>\n"
            msg += f"수량: {executed_volume:.8f}\n"
            msg += f"평균가: {avg_price:,.0f}원\n"
            msg += f"사용금액: {executed_volume * avg_price:,.0f}원"
        else:
            msg = f"✅ <b>{asset} 매도 완료</b>\n"
            msg += f"수량: {executed_volume:.8f}\n"
            msg += f"평균가: {avg_price:,.0f}원\n"
            msg += f"매도금액: {executed_volume * avg_price:,.0f}원"
            if profit_rate != 0:
                msg += f"\n수익률: {profit_rate:+.2f}%"
        
        await notifier.send_message(msg)


# 싱글톤 인스턴스
trader = AdvancedTrader()

# 기존 인터페이스 호환
def execute_trade(asset: str, asset_type: str, fg: float, sentiment: str, **kwargs):
    """동기 함수 래퍼"""
    asyncio.run(trader.execute_trade(asset, asset_type, fg, sentiment, **kwargs))
