#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 통합 트레이딩 시스템 (trading.py)
=================================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 통합 관리

✨ 핵심 기능:
- 4대 전략 통합 실행 및 관리
- IBKR + 업비트 자동 거래
- 실시간 포지션 모니터링
- 리스크 관리 및 손익절 시스템
- 통합 알림 시스템
- 성과 추적 및 분석
- 응급 매도 시스템

Author: 퀸트마스터팀
Version: 1.0.0
"""

import asyncio
import logging
import os
import sys
import json
import time
import signal
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 📊 데이터 클래스 정의
# ============================================================================

@dataclass
class TradingConfig:
    """트레이딩 설정"""
    # 전략별 활성화 설정
    us_strategy_enabled: bool = True
    japan_strategy_enabled: bool = True
    india_strategy_enabled: bool = True
    crypto_strategy_enabled: bool = True
    
    # 거래 일정 설정
    us_trading_days: List[int] = field(default_factory=lambda: [0, 3])  # 월, 목
    japan_trading_days: List[int] = field(default_factory=lambda: [1, 3])  # 화, 목
    india_trading_days: List[int] = field(default_factory=lambda: [2])  # 수
    crypto_trading_days: List[int] = field(default_factory=lambda: [0, 4])  # 월, 금
    
    # 투자 설정
    total_capital: float = 10_000_000  # 1천만원
    max_portfolio_size: int = 20
    max_position_per_strategy: int = 8
    emergency_sell_enabled: bool = True
    
    # 리스크 설정
    max_daily_loss_pct: float = 2.0
    max_weekly_loss_pct: float = 5.0
    max_monthly_loss_pct: float = 8.0
    position_size_limit_pct: float = 15.0
    
    # 모니터링 설정
    monitoring_interval: int = 300  # 5분
    health_check_interval: int = 60  # 1분
    
    # 알림 설정
    notification_enabled: bool = True
    critical_alert_enabled: bool = True

@dataclass
class Position:
    """통합 포지션"""
    strategy: str
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    currency: str
    entry_date: datetime
    stop_loss: float
    take_profit: List[float]
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class TradeSignal:
    """거래 신호"""
    strategy: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    quantity: float
    stop_loss: float
    take_profit: List[float]
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StrategyPerformance:
    """전략별 성과"""
    strategy: str
    total_positions: int
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    avg_holding_days: float
    best_performer: str
    worst_performer: str
    last_updated: datetime = field(default_factory=datetime.now)

# ============================================================================
# 🎯 개별 전략 래퍼 클래스들
# ============================================================================

class USStrategyWrapper:
    """미국 주식 전략 래퍼"""
    
    def __init__(self):
        try:
            from us_strategy import LegendaryQuantStrategy
            self.strategy = LegendaryQuantStrategy()
            self.available = True
            logger.info("✅ 미국 전략 초기화 완료")
        except ImportError as e:
            logger.error(f"❌ 미국 전략 모듈 로드 실패: {e}")
            self.strategy = None
            self.available = False
    
    async def get_signals(self) -> List[TradeSignal]:
        """미국 주식 신호 생성"""
        if not self.available:
            return []
        
        try:
            signals = await self.strategy.scan_all_stocks()
            trade_signals = []
            
            for signal in signals:
                if signal.action == 'BUY':
                    trade_signal = TradeSignal(
                        strategy='us',
                        symbol=signal.symbol,
                        action='BUY',
                        confidence=signal.confidence,
                        price=signal.price,
                        quantity=100,  # 기본 수량
                        stop_loss=signal.stop_loss,
                        take_profit=[signal.target_price],
                        reason=signal.reasoning
                    )
                    trade_signals.append(trade_signal)
            
            return trade_signals
            
        except Exception as e:
            logger.error(f"미국 전략 신호 생성 실패: {e}")
            return []
    
    def should_trade_today(self) -> bool:
        """오늘 거래 가능 여부"""
        return datetime.now().weekday() in [0, 3]  # 월, 목

class JapanStrategyWrapper:
    """일본 주식 전략 래퍼"""
    
    def __init__(self):
        try:
            from jp_strategy import YenHunter
            self.strategy = YenHunter()
            self.available = True
            logger.info("✅ 일본 전략 초기화 완료")
        except ImportError as e:
            logger.error(f"❌ 일본 전략 모듈 로드 실패: {e}")
            self.strategy = None
            self.available = False
    
    async def get_signals(self) -> List[TradeSignal]:
        """일본 주식 신호 생성"""
        if not self.available:
            return []
        
        try:
            signals = await self.strategy.hunt_and_analyze()
            trade_signals = []
            
            for signal in signals:
                if signal.action == 'BUY':
                    trade_signal = TradeSignal(
                        strategy='japan',
                        symbol=signal.symbol,
                        action='BUY',
                        confidence=signal.confidence,
                        price=signal.price,
                        quantity=signal.position_size,
                        stop_loss=signal.stop_loss,
                        take_profit=[signal.take_profit1, signal.take_profit2, signal.take_profit3],
                        reason=signal.reason
                    )
                    trade_signals.append(trade_signal)
            
            return trade_signals
            
        except Exception as e:
            logger.error(f"일본 전략 신호 생성 실패: {e}")
            return []
    
    def should_trade_today(self) -> bool:
        """오늘 거래 가능 여부"""
        return datetime.now().weekday() in [1, 3]  # 화, 목

class IndiaStrategyWrapper:
    """인도 주식 전략 래퍼"""
    
    def __init__(self):
        try:
            from inda_strategy import LegendaryIndiaStrategy
            self.strategy = LegendaryIndiaStrategy()
            self.available = True
            logger.info("✅ 인도 전략 초기화 완료")
        except ImportError as e:
            logger.error(f"❌ 인도 전략 모듈 로드 실패: {e}")
            self.strategy = None
            self.available = False
    
    async def get_signals(self) -> List[TradeSignal]:
        """인도 주식 신호 생성"""
        if not self.available:
            return []
        
        try:
            # 샘플 데이터로 전략 실행
            sample_df = self.strategy.create_sample_data()
            results = self.strategy.run_strategy(sample_df, enable_trading=False)
            
            trade_signals = []
            selected_stocks = results.get('selected_stocks', pd.DataFrame())
            
            for _, stock in selected_stocks.head(5).iterrows():
                if stock.get('final_score', 0) > 15:
                    trade_signal = TradeSignal(
                        strategy='india',
                        symbol=stock['ticker'],
                        action='BUY',
                        confidence=stock['final_score'] / 30,
                        price=stock['close'],
                        quantity=100,
                        stop_loss=stock.get('conservative_stop_loss', stock['close'] * 0.95),
                        take_profit=[stock.get('conservative_take_profit', stock['close'] * 1.10)],
                        reason=f"스코어: {stock['final_score']:.1f}"
                    )
                    trade_signals.append(trade_signal)
            
            return trade_signals
            
        except Exception as e:
            logger.error(f"인도 전략 신호 생성 실패: {e}")
            return []
    
    def should_trade_today(self) -> bool:
        """오늘 거래 가능 여부"""
        return datetime.now().weekday() == 2  # 수

class CryptoStrategyWrapper:
    """암호화폐 전략 래퍼"""
    
    def __init__(self):
        try:
            from coin_strategy import LegendaryQuantMaster
            self.strategy = LegendaryQuantMaster(demo_mode=True)
            self.available = True
            logger.info("✅ 암호화폐 전략 초기화 완료")
        except ImportError as e:
            logger.error(f"❌ 암호화폐 전략 모듈 로드 실패: {e}")
            self.strategy = None
            self.available = False
    
    async def get_signals(self) -> List[TradeSignal]:
        """암호화폐 신호 생성"""
        if not self.available:
            return []
        
        try:
            signals = await self.strategy.execute_legendary_strategy()
            trade_signals = []
            
            for signal in signals:
                if signal.action == 'BUY':
                    trade_signal = TradeSignal(
                        strategy='crypto',
                        symbol=signal.symbol,
                        action='BUY',
                        confidence=signal.confidence,
                        price=signal.price,
                        quantity=signal.total_investment / signal.price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profits,
                        reason=signal.ai_explanation
                    )
                    trade_signals.append(trade_signal)
            
            return trade_signals
            
        except Exception as e:
            logger.error(f"암호화폐 전략 신호 생성 실패: {e}")
            return []
    
    def should_trade_today(self) -> bool:
        """오늘 거래 가능 여부"""
        return datetime.now().weekday() in [0, 4]  # 월, 금

# ============================================================================
# 🔗 거래소 연결 관리자
# ============================================================================

class ExchangeManager:
    """거래소 연결 관리"""
    
    def __init__(self):
        self.ibkr_connected = False
        self.upbit_connected = False
        self._init_connections()
    
    def _init_connections(self):
        """거래소 연결 초기화"""
        # IBKR 연결 확인
        try:
            from ib_insync import IB
            self.ibkr_available = True
            logger.info("✅ IBKR API 사용 가능")
        except ImportError:
            self.ibkr_available = False
            logger.warning("⚠️ IBKR API 없음 (시뮬레이션 모드)")
        
        # 업비트 연결 확인
        try:
            import pyupbit
            self.upbit_available = True
            logger.info("✅ 업비트 API 사용 가능")
        except ImportError:
            self.upbit_available = False
            logger.warning("⚠️ 업비트 API 없음 (시뮬레이션 모드)")
    
    async def connect_ibkr(self) -> bool:
        """IBKR 연결"""
        if not self.ibkr_available:
            return False
        
        try:
            from ib_insync import IB
            ib = IB()
            await ib.connectAsync('127.0.0.1', 7497, clientId=999)
            self.ibkr_connected = True
            logger.info("✅ IBKR 연결 성공")
            return True
        except Exception as e:
            logger.error(f"❌ IBKR 연결 실패: {e}")
            return False
    
    def connect_upbit(self) -> bool:
        """업비트 연결"""
        if not self.upbit_available:
            return False
        
        try:
            access_key = os.getenv('UPBIT_ACCESS_KEY')
            secret_key = os.getenv('UPBIT_SECRET_KEY')
            
            if access_key and secret_key:
                import pyupbit
                upbit = pyupbit.Upbit(access_key, secret_key)
                self.upbit_connected = True
                logger.info("✅ 업비트 연결 성공")
                return True
            else:
                logger.warning("⚠️ 업비트 API 키 없음")
                return False
        except Exception as e:
            logger.error(f"❌ 업비트 연결 실패: {e}")
            return False
    
    async def execute_trade(self, signal: TradeSignal, demo_mode: bool = True) -> bool:
        """거래 실행"""
        try:
            if demo_mode:
                logger.info(f"🎭 시뮬레이션 거래: {signal.action} {signal.symbol} {signal.quantity}")
                return True
            
            if signal.strategy in ['us', 'japan', 'india'] and self.ibkr_connected:
                return await self._execute_ibkr_trade(signal)
            elif signal.strategy == 'crypto' and self.upbit_connected:
                return self._execute_upbit_trade(signal)
            else:
                logger.warning(f"⚠️ 거래소 연결 없음: {signal.strategy}")
                return False
                
        except Exception as e:
            logger.error(f"거래 실행 실패: {e}")
            return False
    
    async def _execute_ibkr_trade(self, signal: TradeSignal) -> bool:
        """IBKR 거래 실행"""
        try:
            # IBKR 거래 로직 구현
            logger.info(f"📈 IBKR 거래: {signal.action} {signal.symbol}")
            return True
        except Exception as e:
            logger.error(f"IBKR 거래 실패: {e}")
            return False
    
    def _execute_upbit_trade(self, signal: TradeSignal) -> bool:
        """업비트 거래 실행"""
        try:
            # 업비트 거래 로직 구현
            logger.info(f"💰 업비트 거래: {signal.action} {signal.symbol}")
            return True
        except Exception as e:
            logger.error(f"업비트 거래 실패: {e}")
            return False

# ============================================================================
# 📊 포지션 관리자
# ============================================================================

class PositionManager:
    """통합 포지션 관리"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.position_file = "positions.json"
        self.load_positions()
    
    def load_positions(self):
        """포지션 로드"""
        try:
            if Path(self.position_file).exists():
                with open(self.position_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for key, pos_data in data.items():
                    self.positions[key] = Position(
                        strategy=pos_data['strategy'],
                        symbol=pos_data['symbol'],
                        quantity=pos_data['quantity'],
                        avg_price=pos_data['avg_price'],
                        current_price=pos_data['current_price'],
                        currency=pos_data['currency'],
                        entry_date=datetime.fromisoformat(pos_data['entry_date']),
                        stop_loss=pos_data['stop_loss'],
                        take_profit=pos_data['take_profit'],
                        unrealized_pnl=pos_data.get('unrealized_pnl', 0),
                        unrealized_pnl_pct=pos_data.get('unrealized_pnl_pct', 0),
                        last_updated=datetime.fromisoformat(pos_data.get('last_updated', datetime.now().isoformat()))
                    )
                    
                logger.info(f"📂 포지션 로드 완료: {len(self.positions)}개")
        except Exception as e:
            logger.error(f"포지션 로드 실패: {e}")
    
    def save_positions(self):
        """포지션 저장"""
        try:
            data = {}
            for key, position in self.positions.items():
                data[key] = {
                    'strategy': position.strategy,
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'current_price': position.current_price,
                    'currency': position.currency,
                    'entry_date': position.entry_date.isoformat(),
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': position.unrealized_pnl_pct,
                    'last_updated': position.last_updated.isoformat()
                }
            
            with open(self.position_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"포지션 저장 실패: {e}")
    
    def add_position(self, signal: TradeSignal):
        """포지션 추가"""
        try:
            key = f"{signal.strategy}_{signal.symbol}"
            
            position = Position(
                strategy=signal.strategy,
                symbol=signal.symbol,
                quantity=signal.quantity,
                avg_price=signal.price,
                current_price=signal.price,
                currency=self._get_currency(signal.strategy),
                entry_date=datetime.now(),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            self.positions[key] = position
            self.save_positions()
            
            logger.info(f"➕ 포지션 추가: {signal.strategy} {signal.symbol} {signal.quantity}")
            
        except Exception as e:
            logger.error(f"포지션 추가 실패: {e}")
    
    def remove_position(self, strategy: str, symbol: str):
        """포지션 제거"""
        try:
            key = f"{strategy}_{symbol}"
            if key in self.positions:
                del self.positions[key]
                self.save_positions()
                logger.info(f"➖ 포지션 제거: {strategy} {symbol}")
        except Exception as e:
            logger.error(f"포지션 제거 실패: {e}")
    
    def update_position_prices(self, price_data: Dict[str, float]):
        """포지션 현재가 업데이트"""
        try:
            for key, position in self.positions.items():
                if position.symbol in price_data:
                    old_price = position.current_price
                    new_price = price_data[position.symbol]
                    
                    position.current_price = new_price
                    position.unrealized_pnl = (new_price - position.avg_price) * position.quantity
                    position.unrealized_pnl_pct = ((new_price - position.avg_price) / position.avg_price) * 100
                    position.last_updated = datetime.now()
                    
                    # 큰 변동시 로그
                    price_change = abs((new_price - old_price) / old_price) * 100
                    if price_change > 5:
                        logger.info(f"💹 {position.symbol}: {price_change:+.1f}% @ {new_price}")
            
            self.save_positions()
            
        except Exception as e:
            logger.error(f"포지션 가격 업데이트 실패: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """포트폴리오 요약"""
        try:
            summary = {
                'total_positions': len(self.positions),
                'total_value': 0,
                'total_pnl': 0,
                'by_strategy': {},
                'top_performers': [],
                'worst_performers': []
            }
            
            positions_with_pnl = []
            
            for position in self.positions.values():
                value = position.current_price * position.quantity
                summary['total_value'] += value
                summary['total_pnl'] += position.unrealized_pnl
                
                # 전략별 집계
                if position.strategy not in summary['by_strategy']:
                    summary['by_strategy'][position.strategy] = {
                        'count': 0, 'value': 0, 'pnl': 0
                    }
                
                summary['by_strategy'][position.strategy]['count'] += 1
                summary['by_strategy'][position.strategy]['value'] += value
                summary['by_strategy'][position.strategy]['pnl'] += position.unrealized_pnl
                
                positions_with_pnl.append((position, position.unrealized_pnl_pct))
            
            # 수익률 정렬
            positions_with_pnl.sort(key=lambda x: x[1], reverse=True)
            
            summary['top_performers'] = [
                {'symbol': pos.symbol, 'strategy': pos.strategy, 'pnl_pct': pnl_pct}
                for pos, pnl_pct in positions_with_pnl[:3]
            ]
            
            summary['worst_performers'] = [
                {'symbol': pos.symbol, 'strategy': pos.strategy, 'pnl_pct': pnl_pct}
                for pos, pnl_pct in positions_with_pnl[-3:]
            ]
            
            # 총 수익률
            if summary['total_value'] > 0:
                summary['total_pnl_pct'] = (summary['total_pnl'] / (summary['total_value'] - summary['total_pnl'])) * 100
            else:
                summary['total_pnl_pct'] = 0
            
            return summary
            
        except Exception as e:
            logger.error(f"포트폴리오 요약 실패: {e}")
            return {}
    
    def _get_currency(self, strategy: str) -> str:
        """전략별 통화 반환"""
        currency_map = {
            'us': 'USD',
            'japan': 'JPY', 
            'india': 'INR',
            'crypto': 'KRW'
        }
        return currency_map.get(strategy, 'USD')

# ============================================================================
# 🚨 리스크 관리자
# ============================================================================

class RiskManager:
    """리스크 관리 시스템"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.monthly_pnl = 0
        self.risk_alerts = []
    
    def check_position_risk(self, position: Position) -> bool:
        """개별 포지션 리스크 체크"""
        try:
            # 손절선 체크
            if position.current_price <= position.stop_loss:
                self.risk_alerts.append(f"🚨 {position.symbol} 손절선 도달")
                return False
            
            # 최대 손실 체크
            if position.unrealized_pnl_pct < -self.config.max_daily_loss_pct:
                self.risk_alerts.append(f"⚠️ {position.symbol} 일일 손실 한도 초과")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"포지션 리스크 체크 실패: {e}")
            return True
    
    def check_portfolio_risk(self, portfolio_summary: Dict) -> bool:
        """포트폴리오 리스크 체크"""
        try:
            total_pnl_pct = portfolio_summary.get('total_pnl_pct', 0)
            
            # 일일 손실 한도
            if total_pnl_pct < -self.config.max_daily_loss_pct:
                self.risk_alerts.append(f"🚨 일일 손실 한도 초과: {total_pnl_pct:.2f}%")
                return False
            
            # 주간 손실 한도
            if total_pnl_pct < -self.config.max_weekly_loss_pct:
                self.risk_alerts.append(f"🚨 주간 손실 한도 초과: {total_pnl_pct:.2f}%")
                return False
            
            # 월간 손실 한도
            if total_pnl_pct < -self.config.max_monthly_loss_pct:
                self.risk_alerts.append(f"🚨 월간 손실 한도 초과: {total_pnl_pct:.2f}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"포트폴리오 리스크 체크 실패: {e}")
            return True
    
    def should_allow_new_position(self, strategy: str, portfolio_summary: Dict) -> bool:
        """신규 포지션 허용 여부"""
        try:
            # 전략별 포지션 수 제한
            strategy_positions = portfolio_summary.get('by_strategy', {}).get(strategy, {}).get('count', 0)
            if strategy_positions >= self.config.max_position_per_strategy:
                return False
            
            # 전체 포지션 수 제한
            total_positions = portfolio_summary.get('total_positions', 0)
            if total_positions >= self.config.max_portfolio_size:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"신규 포지션 허용 체크 실패: {e}")
            return False
    
    def get_risk_alerts(self) -> List[str]:
        """리스크 알림 조회"""
        alerts = self.risk_alerts.copy()
        self.risk_alerts.clear()
        return alerts

# ============================================================================
# 📱 알림 관리자 (간소화)
# ============================================================================

class NotificationManager:
    """간소화된 알림 관리자"""
    
    def __init__(self):
        self.enabled = os.getenv('NOTIFICATION_ENABLED', 'true').lower() == 'true'
        
        # 텔레그램 설정
        self.telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
    
    async def send_alert(self, title: str, message: str, level: str = 'info'):
        """알림 전송"""
        if not self.enabled:
            return
        
        try:
            formatted_message = f"🏆 퀸트프로젝트\n\n📌 {title}\n\n{message}\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            if self.telegram_enabled and self.telegram_bot_token and self.telegram_chat_id:
                await self._send_telegram(formatted_message)
                
            # 로그로도 출력
            log_level = getattr(logging, level.upper(), logging.INFO)
            logger.log(log_level, f"📢 {title}: {message}")
            
        except Exception as e:
            logger.error(f"알림 전송 실패: {e}")
    
    async def _send_telegram(self, message: str):
        """텔레그램 전송"""
        try:
            import aiohttp
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=10) as response:
                    if response.status == 200:
                        logger.debug("텔레그램 알림 전송 완료")
                    else:
                        logger.error(f"텔레그램 전송 실패: {response.status}")
                        
        except Exception as e:
            logger.error(f"텔레그램 전송 오류: {e}")

# ============================================================================
# 🏆 통합 트레이딩 시스템
# ============================================================================

class QuintTradingSystem:
    """퀸트프로젝트 통합 트레이딩 시스템"""
    
    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        
        # 핵심 컴포넌트 초기화
        self.position_manager = PositionManager()
        self.exchange_manager = ExchangeManager()
        self.risk_manager = RiskManager(self.config)
        self.notification_manager = NotificationManager()
        
        # 전략 래퍼 초기화
        self.strategies = {}
        self._init_strategies()
        
        # 상태 변수
        self.is_running = False
        self.emergency_mode = False
        self.last_health_check = datetime.now()
        
        # 성과 추적
        self.trade_count = 0
        self.total_pnl = 0
        
        logger.info("🏆 퀸트프로젝트 통합 트레이딩 시스템 초기화 완료")
    
    def _init_strategies(self):
        """전략 시스템 초기화"""
        if self.config.us_strategy_enabled:
            self.strategies['us'] = USStrategyWrapper()
        
        if self.config.japan_strategy_enabled:
            self.strategies['japan'] = JapanStrategyWrapper()
        
        if self.config.india_strategy_enabled:
            self.strategies['india'] = IndiaStrategyWrapper()
        
        if self.config.crypto_strategy_enabled:
            self.strategies['crypto'] = CryptoStrategyWrapper()
        
        active_strategies = [k for k, v in self.strategies.items() if v.available]
        logger.info(f"🎯 활성화된 전략: {active_strategies}")
    
    async def start(self):
        """시스템 시작"""
        logger.info("🚀 퀸트프로젝트 통합 트레이딩 시스템 시작")
        
        try:
            self.is_running = True
            
            # 거래소 연결
            await self.exchange_manager.connect_ibkr()
            self.exchange_manager.connect_upbit()
            
            # 시작 알림
            await self.notification_manager.send_alert(
                "🚀 시스템 시작",
                f"퀸트프로젝트 통합 트레이딩 시스템이 시작되었습니다.\n"
                f"활성화된 전략: {list(self.strategies.keys())}\n"
                f"총 자본: {self.config.total_capital:,.0f}원\n"
                f"응급매도: {'✅ 활성화' if self.config.emergency_sell_enabled else '❌ 비활성화'}"
            )
            
            # 메인 루프 시작
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"시스템 시작 실패: {e}")
            await self.shutdown()
    
    async def _main_loop(self):
        """메인 실행 루프"""
        logger.info("🔄 메인 루프 시작")
        
        while self.is_running:
            try:
                # 건강 상태 체크
                await self._health_check()
                
                # 거래 실행 (스케줄 기반)
                await self._execute_trading_cycle()
                
                # 포지션 모니터링
                await self._monitor_positions()
                
                # 리스크 관리
                await self._manage_risks()
                
                # 대기
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"메인 루프 오류: {e}")
                await asyncio.sleep(60)  # 1분 후 재시도
    
    async def _health_check(self):
        """시스템 건강 상태 체크"""
        try:
            current_time = datetime.now()
            
            # 메모리 및 시스템 리소스 체크
            import psutil
            memory_usage = psutil.virtual_memory().percent
            
            if memory_usage > 90:
                await self.notification_manager.send_alert(
                    "⚠️ 시스템 경고", 
                    f"메모리 사용량이 높습니다: {memory_usage:.1f}%",
                    "warning"
                )
            
            self.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"건강 상태 체크 실패: {e}")
    
    async def _execute_trading_cycle(self):
        """거래 실행 사이클"""
        try:
            current_time = datetime.now()
            current_weekday = current_time.weekday()
            current_hour = current_time.hour
            
            # 거래 시간 체크 (오전 9-11시)
            if not (9 <= current_hour <= 11):
                return
            
            for strategy_name, strategy_wrapper in self.strategies.items():
                if not strategy_wrapper.available:
                    continue
                
                # 전략별 거래 요일 체크
                if not strategy_wrapper.should_trade_today():
                    continue
                
                try:
                    await self._execute_strategy(strategy_name, strategy_wrapper)
                except Exception as e:
                    logger.error(f"전략 실행 실패 {strategy_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"거래 사이클 실행 실패: {e}")
    
    async def _execute_strategy(self, strategy_name: str, strategy_wrapper):
        """개별 전략 실행"""
        try:
            logger.info(f"🎯 {strategy_name} 전략 실행")
            
            # 신호 생성
            signals = await strategy_wrapper.get_signals()
            
            if not signals:
                logger.info(f"📭 {strategy_name} 신호 없음")
                return
            
            # 포트폴리오 현황 확인
            portfolio_summary = self.position_manager.get_portfolio_summary()
            
            executed_trades = 0
            for signal in signals[:3]:  # 상위 3개만
                try:
                    # 리스크 체크
                    if not self.risk_manager.should_allow_new_position(strategy_name, portfolio_summary):
                        logger.warning(f"⚠️ {strategy_name} 신규 포지션 제한")
                        break
                    
                    # 거래 실행
                    success = await self.exchange_manager.execute_trade(signal, demo_mode=True)
                    
                    if success:
                        # 포지션 추가
                        self.position_manager.add_position(signal)
                        executed_trades += 1
                        self.trade_count += 1
                        
                        # 거래 알림
                        await self.notification_manager.send_alert(
                            f"📈 거래 실행 ({strategy_name})",
                            f"종목: {signal.symbol}\n"
                            f"액션: {signal.action}\n"
                            f"가격: {signal.price:,.2f}\n"
                            f"수량: {signal.quantity:,.2f}\n"
                            f"신뢰도: {signal.confidence:.1%}\n"
                            f"이유: {signal.reason}"
                        )
                        
                        # 짧은 대기
                        await asyncio.sleep(2)
                
                except Exception as e:
                    logger.error(f"개별 거래 실행 실패: {e}")
                    continue
            
            if executed_trades > 0:
                logger.info(f"✅ {strategy_name} 전략 완료: {executed_trades}개 거래")
            
        except Exception as e:
            logger.error(f"전략 실행 오류 {strategy_name}: {e}")
    
    async def _monitor_positions(self):
        """포지션 모니터링"""
        try:
            if not self.position_manager.positions:
                return
            
            # 현재가 업데이트 (간소화된 버전)
            price_data = await self._fetch_current_prices()
            self.position_manager.update_position_prices(price_data)
            
            # 손익절 체크
            positions_to_close = []
            
            for key, position in self.position_manager.positions.items():
                # 손절 체크
                if position.current_price <= position.stop_loss:
                    positions_to_close.append((key, position, "STOP_LOSS"))
                    continue
                
                # 익절 체크
                for i, take_profit in enumerate(position.take_profit):
                    if position.current_price >= take_profit:
                        positions_to_close.append((key, position, f"TAKE_PROFIT_{i+1}"))
                        break
                
                # 장기 보유 체크 (2주 초과)
                holding_days = (datetime.now() - position.entry_date).days
                if holding_days > 14:
                    positions_to_close.append((key, position, "TIME_LIMIT"))
            
            # 포지션 정리 실행
            for key, position, reason in positions_to_close:
                await self._close_position(key, position, reason)
                
        except Exception as e:
            logger.error(f"포지션 모니터링 실패: {e}")
    
    async def _fetch_current_prices(self) -> Dict[str, float]:
        """현재가 조회 (간소화)"""
        try:
            price_data = {}
            
            # 실제로는 각 거래소별로 현재가를 조회해야 함
            # 여기서는 시뮬레이션용 랜덤 가격 변동
            import random
            
            for position in self.position_manager.positions.values():
                # ±2% 랜덤 변동
                change_pct = random.uniform(-0.02, 0.02)
                new_price = position.current_price * (1 + change_pct)
                price_data[position.symbol] = new_price
            
            return price_data
            
        except Exception as e:
            logger.error(f"현재가 조회 실패: {e}")
            return {}
    
    async def _close_position(self, key: str, position: Position, reason: str):
        """포지션 정리"""
        try:
            # 매도 신호 생성
            sell_signal = TradeSignal(
                strategy=position.strategy,
                symbol=position.symbol,
                action='SELL',
                confidence=1.0,
                price=position.current_price,
                quantity=position.quantity,
                stop_loss=0,
                take_profit=[],
                reason=reason
            )
            
            # 거래 실행
            success = await self.exchange_manager.execute_trade(sell_signal, demo_mode=True)
            
            if success:
                # 수익률 계산
                profit_loss = position.unrealized_pnl
                profit_pct = position.unrealized_pnl_pct
                
                # 포지션 제거
                self.position_manager.remove_position(position.strategy, position.symbol)
                
                # 통계 업데이트
                self.total_pnl += profit_loss
                
                # 알림 전송
                emoji = "💰" if profit_loss > 0 else "💸"
                await self.notification_manager.send_alert(
                    f"{emoji} 포지션 정리 ({position.strategy})",
                    f"종목: {position.symbol}\n"
                    f"사유: {reason}\n"
                    f"수익률: {profit_pct:+.2f}%\n"
                    f"손익: {profit_loss:+,.0f}원\n"
                    f"보유일: {(datetime.now() - position.entry_date).days}일"
                )
                
                logger.info(f"📉 포지션 정리: {position.symbol} ({reason}) {profit_pct:+.2f}%")
            
        except Exception as e:
            logger.error(f"포지션 정리 실패: {e}")
    
    async def _manage_risks(self):
        """리스크 관리"""
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            
            # 포트폴리오 리스크 체크
            if not self.risk_manager.check_portfolio_risk(portfolio_summary):
                if self.config.emergency_sell_enabled:
                    await self._emergency_sell_all("RISK_LIMIT_EXCEEDED")
            
            # 개별 포지션 리스크 체크
            for position in self.position_manager.positions.values():
                if not self.risk_manager.check_position_risk(position):
                    await self._close_position(
                        f"{position.strategy}_{position.symbol}",
                        position,
                        "RISK_MANAGEMENT"
                    )
            
            # 리스크 알림 처리
            risk_alerts = self.risk_manager.get_risk_alerts()
            for alert in risk_alerts:
                await self.notification_manager.send_alert("🚨 리스크 경고", alert, "warning")
                
        except Exception as e:
            logger.error(f"리스크 관리 실패: {e}")
    
    async def _emergency_sell_all(self, reason: str):
        """응급 전량 매도"""
        logger.critical(f"🚨 응급 전량 매도 실행: {reason}")
        
        try:
            self.emergency_mode = True
            
            positions_to_sell = list(self.position_manager.positions.items())
            
            for key, position in positions_to_sell:
                await self._close_position(key, position, f"EMERGENCY_{reason}")
                await asyncio.sleep(1)  # 1초 간격
            
            # 긴급 알림
            await self.notification_manager.send_alert(
                "🚨 응급 전량 매도 실행",
                f"사유: {reason}\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"매도 포지션: {len(positions_to_sell)}개",
                "critical"
            )
            
        except Exception as e:
            logger.error(f"응급 매도 실행 실패: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            
            return {
                'system': {
                    'is_running': self.is_running,
                    'emergency_mode': self.emergency_mode,
                    'last_health_check': self.last_health_check.isoformat(),
                    'trade_count': self.trade_count,
                    'total_pnl': self.total_pnl
                },
                'strategies': {
                    'available': [k for k, v in self.strategies.items() if v.available],
                    'total': len(self.strategies)
                },
                'portfolio': portfolio_summary,
                'exchange': {
                    'ibkr_connected': self.exchange_manager.ibkr_connected,
                    'upbit_connected': self.exchange_manager.upbit_connected
                },
                'config': {
                    'total_capital': self.config.total_capital,
                    'max_portfolio_size': self.config.max_portfolio_size,
                    'emergency_sell_enabled': self.config.emergency_sell_enabled
                }
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """시스템 종료"""
        logger.info("🛑 퀸트프로젝트 통합 트레이딩 시스템 종료")
        
        try:
            self.is_running = False
            
            # 포지션 저장
            self.position_manager.save_positions()
            
            # 종료 알림
            portfolio_summary = self.position_manager.get_portfolio_summary()
            await self.notification_manager.send_alert(
                "🛑 시스템 종료",
                f"퀸트프로젝트 시스템이 종료되었습니다.\n"
                f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"최종 포지션: {portfolio_summary.get('total_positions', 0)}개\n"
                f"총 거래: {self.trade_count}회\n"
                f"총 손익: {self.total_pnl:+,.0f}원"
            )
            
        except Exception as e:
            logger.error(f"시스템 종료 오류: {e}")

# ============================================================================
# 🎮 CLI 인터페이스
# ============================================================================

class TradingCLI:
    """트레이딩 시스템 CLI"""
    
    def __init__(self):
        self.trading_system = None
    
    def print_banner(self):
        """배너 출력"""
        banner = """
🏆════════════════════════════════════════════════════════════════🏆
   ██████╗ ██╗   ██╗██╗███╗   ██╗████████╗    ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
  ██╔═══██╗██║   ██║██║████╗  ██║╚══██╔══╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ 
  ██║   ██║██║   ██║██║██╔██╗ ██║   ██║█████╗    ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗
  ██║▄▄ ██║██║   ██║██║██║╚██╗██║   ██║╚════╝    ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
  ╚██████╔╝╚██████╔╝██║██║ ╚████║   ██║          ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
   ╚══▀▀═╝  ╚═════╝ ╚═╝╚═╝  ╚═══╝   ╚═╝          ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                                                                      
        퀸트프로젝트 통합 트레이딩 시스템 v1.0.0                      
        🇺🇸 미국 + 🇯🇵 일본 + 🇮🇳 인도 + 💰 암호화폐           
🏆════════════════════════════════════════════════════════════════🏆
        """
        print(banner)
    
    async def start_interactive_mode(self):
        """대화형 모드"""
        self.print_banner()
        
        while True:
            try:
                print("\n" + "="*60)
                print("🎮 퀸트프로젝트 통합 트레이딩 시스템")
                print("="*60)
                
                if self.trading_system is None:
                    print("1. 🚀 시스템 시작")
                    print("2. ⚙️  설정 확인")
                    print("3. 📊 시스템 상태 (읽기 전용)")
                    print("0. 🚪 종료")
                else:
                    print("1. 📊 실시간 상태")
                    print("2. 💼 포트폴리오 현황")
                    print("3. 🎯 전략 상태")
                    print("4. 📈 성과 분석")
                    print("5. 🚨 리스크 현황")
                    print("6. 🔧 설정 변경")
                    print("7. 🛑 시스템 종료")
                    print("8. 🚨 응급 매도")
                    print("0. 🚪 프로그램 종료")
                
                choice = input("\n선택하세요: ").strip()
                
                if self.trading_system is None:
                    await self._handle_startup_menu(choice)
                else:
                    await self._handle_running_menu(choice)
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                await asyncio.sleep(2)
    
    async def _handle_startup_menu(self, choice: str):
        """시작 메뉴 처리"""
        if choice == '1':
            print("🚀 트레이딩 시스템을 시작합니다...")
            
            # 설정 생성
            config = TradingConfig(
                total_capital=float(input("총 자본 입력 (기본 10,000,000): ") or "10000000"),
                emergency_sell_enabled=input("응급매도 활성화? (y/N): ").lower() == 'y'
            )
            
            self.trading_system = QuintTradingSystem(config)
            
            # 백그라운드에서 시작
            asyncio.create_task(self.trading_system.start())
            await asyncio.sleep(3)  # 시작 대기
            
        elif choice == '2':
            self._show_config()
            
        elif choice == '3':
            await self._show_readonly_status()
            
        elif choice == '0':
            exit(0)
    
    async def _handle_running_menu(self, choice: str):
        """실행 중 메뉴 처리"""
        if choice == '1':
            await self._show_realtime_status()
            
        elif choice == '2':
            await self._show_portfolio()
            
        elif choice == '3':
            await self._show_strategy_status()
            
        elif choice == '4':
            await self._show_performance()
            
        elif choice == '5':
            await self._show_risk_status()
            
        elif choice == '6':
            await self._change_settings()
            
        elif choice == '7':
            await self._shutdown_system()
            
        elif choice == '8':
            await self._emergency_sell()
            
        elif choice == '0':
            await self._shutdown_system()
            exit(0)
    
    def _show_config(self):
        """설정 표시"""
        print("\n⚙️ 시스템 설정")
        print("="*40)
        
        # 환경변수 체크
        env_vars = [
            'TELEGRAM_BOT_TOKEN', 'UPBIT_ACCESS_KEY', 'IBKR_HOST'
        ]
        
        for var in env_vars:
            value = os.getenv(var, '')
            status = "✅" if value else "❌"
            masked_value = f"{value[:8]}***" if len(value) > 8 else "없음"
            print(f"  {status} {var}: {masked_value}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_readonly_status(self):
        """읽기 전용 상태"""
        print("\n📊 시스템 상태 (읽기 전용)")
        print("="*40)
        
        # 모듈 가용성 체크
        modules = [
            ('미국 전략', 'us_strategy'),
            ('일본 전략', 'jp_strategy'),
            ('인도 전략', 'inda_strategy'),
            ('암호화폐 전략', 'coin_strategy')
        ]
        
        for name, module in modules:
            try:
                __import__(module)
                print(f"  ✅ {name}")
            except ImportError:
                print(f"  ❌ {name}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_realtime_status(self):
        """실시간 상태"""
        if not self.trading_system:
            return
        
        status = self.trading_system.get_system_status()
        
        print("\n📊 실시간 시스템 상태")
        print("="*50)
        
        # 시스템 상태
        system = status.get('system', {})
        print(f"🔄 실행 상태: {'✅ 실행 중' if system.get('is_running') else '❌ 중지'}")
        print(f"🚨 응급 모드: {'⚠️ 활성화' if system.get('emergency_mode') else '✅ 정상'}")
        print(f"📈 총 거래: {system.get('trade_count', 0)}회")
        print(f"💰 총 손익: {system.get('total_pnl', 0):+,.0f}원")
        
        # 포트폴리오 상태
        portfolio = status.get('portfolio', {})
        print(f"\n💼 포트폴리오:")
        print(f"  포지션 수: {portfolio.get('total_positions', 0)}개")
        print(f"  총 가치: {portfolio.get('total_value', 0):,.0f}원")
        print(f"  미실현 손익: {portfolio.get('total_pnl', 0):+,.0f}원 ({portfolio.get('total_pnl_pct', 0):+.2f}%)")
        
        # 거래소 연결
        exchange = status.get('exchange', {})
        print(f"\n🔗 거래소 연결:")
        print(f"  IBKR: {'✅ 연결됨' if exchange.get('ibkr_connected') else '❌ 끊김'}")
        print(f"  업비트: {'✅ 연결됨' if exchange.get('upbit_connected') else '❌ 끊김'}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_portfolio(self):
        """포트폴리오 현황"""
        if not self.trading_system:
            return
        
        portfolio = self.trading_system.position_manager.get_portfolio_summary()
        
        print("\n💼 포트폴리오 현황")
        print("="*50)
        
        if portfolio.get('total_positions', 0) == 0:
            print("📭 현재 보유 포지션이 없습니다.")
            else:
            print(f"총 포지션: {portfolio['total_positions']}개")
            print(f"총 가치: {portfolio['total_value']:,.0f}원")
            print(f"미실현 손익: {portfolio['total_pnl']:+,.0f}원 ({portfolio['total_pnl_pct']:+.2f}%)")
            
            # 전략별 현황
            print(f"\n📊 전략별 현황:")
            for strategy, data in portfolio.get('by_strategy', {}).items():
                emoji_map = {'us': '🇺🇸', 'japan': '🇯🇵', 'india': '🇮🇳', 'crypto': '💰'}
                emoji = emoji_map.get(strategy, '📈')
                print(f"  {emoji} {strategy}: {data['count']}개 포지션, {data['value']:,.0f}원 ({data['pnl']:+,.0f}원)")
            
            # 상위/하위 수익 종목
            if portfolio.get('top_performers'):
                print(f"\n🏆 상위 수익 종목:")
                for perf in portfolio['top_performers']:
                    print(f"  📈 {perf['symbol']} ({perf['strategy']}): {perf['pnl_pct']:+.2f}%")
            
            if portfolio.get('worst_performers'):
                print(f"\n📉 하위 수익 종목:")
                for perf in portfolio['worst_performers']:
                    print(f"  📉 {perf['symbol']} ({perf['strategy']}): {perf['pnl_pct']:+.2f}%")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_strategy_status(self):
        """전략 상태"""
        if not self.trading_system:
            return
        
        print("\n🎯 전략 상태")
        print("="*50)
        
        for strategy_name, strategy_wrapper in self.trading_system.strategies.items():
            emoji_map = {'us': '🇺🇸', 'japan': '🇯🇵', 'india': '🇮🇳', 'crypto': '💰'}
            emoji = emoji_map.get(strategy_name, '📈')
            
            status = "✅ 활성화" if strategy_wrapper.available else "❌ 비활성화"
            trading_day = "📅 거래일" if strategy_wrapper.should_trade_today() else "⏸️ 비거래일"
            
            print(f"{emoji} {strategy_name.upper()} 전략:")
            print(f"  상태: {status}")
            print(f"  오늘: {trading_day}")
            
            # 거래 요일 정보
            if strategy_name == 'us':
                print(f"  거래일: 화요일, 목요일 (23:30 한국시간)")
            elif strategy_name == 'japan':
                print(f"  거래일: 화요일, 목요일 (09:00-15:00)")
            elif strategy_name == 'india':
                print(f"  거래일: 수요일 (09:00-15:00)")
            elif strategy_name == 'crypto':
                print(f"  거래일: 월요일, 금요일 (24시간)")
            
            print()
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_performance(self):
        """성과 분석"""
        if not self.trading_system:
            return
        
        print("\n📈 성과 분석")
        print("="*50)
        
        # 전체 성과
        total_trades = self.trading_system.trade_count
        total_pnl = self.trading_system.total_pnl
        
        print(f"📊 전체 성과:")
        print(f"  총 거래 횟수: {total_trades}회")
        print(f"  총 손익: {total_pnl:+,.0f}원")
        
        if total_trades > 0:
            avg_pnl = total_pnl / total_trades
            print(f"  거래당 평균: {avg_pnl:+,.0f}원")
        
        # 포지션별 성과
        portfolio = self.trading_system.position_manager.get_portfolio_summary()
        
        if portfolio.get('total_positions', 0) > 0:
            print(f"\n💼 현재 포지션 성과:")
            print(f"  미실현 손익: {portfolio['total_pnl']:+,.0f}원")
            print(f"  수익률: {portfolio['total_pnl_pct']:+.2f}%")
            
            # 전략별 성과
            for strategy, data in portfolio.get('by_strategy', {}).items():
                if data['pnl'] != 0:
                    pnl_pct = (data['pnl'] / (data['value'] - data['pnl'])) * 100
                    print(f"  {strategy}: {data['pnl']:+,.0f}원 ({pnl_pct:+.2f}%)")
        
        # 일별/주별/월별 목표 대비
        print(f"\n🎯 목표 대비:")
        config = self.trading_system.config
        current_pnl_pct = portfolio.get('total_pnl_pct', 0)
        
        print(f"  일일 목표: {current_pnl_pct:+.2f}% / ±{config.max_daily_loss_pct:.1f}%")
        print(f"  주간 목표: {current_pnl_pct:+.2f}% / ±{config.max_weekly_loss_pct:.1f}%")
        print(f"  월간 목표: {current_pnl_pct:+.2f}% / ±{config.max_monthly_loss_pct:.1f}%")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _show_risk_status(self):
        """리스크 현황"""
        if not self.trading_system:
            return
        
        print("\n🚨 리스크 현황")
        print("="*50)
        
        portfolio = self.trading_system.position_manager.get_portfolio_summary()
        config = self.trading_system.config
        
        # 전체 리스크
        total_pnl_pct = portfolio.get('total_pnl_pct', 0)
        total_positions = portfolio.get('total_positions', 0)
        
        print(f"📊 전체 리스크:")
        print(f"  현재 수익률: {total_pnl_pct:+.2f}%")
        print(f"  포지션 수: {total_positions}/{config.max_portfolio_size}")
        
        # 한도 체크
        daily_risk = abs(total_pnl_pct) / config.max_daily_loss_pct * 100
        weekly_risk = abs(total_pnl_pct) / config.max_weekly_loss_pct * 100
        monthly_risk = abs(total_pnl_pct) / config.max_monthly_loss_pct * 100
        
        print(f"\n⚠️ 리스크 레벨:")
        print(f"  일일 리스크: {daily_risk:.1f}% {'🔴' if daily_risk > 80 else '🟡' if daily_risk > 50 else '🟢'}")
        print(f"  주간 리스크: {weekly_risk:.1f}% {'🔴' if weekly_risk > 80 else '🟡' if weekly_risk > 50 else '🟢'}")
        print(f"  월간 리스크: {monthly_risk:.1f}% {'🔴' if monthly_risk > 80 else '🟡' if monthly_risk > 50 else '🟢'}")
        
        # 개별 포지션 리스크
        high_risk_positions = []
        for position in self.trading_system.position_manager.positions.values():
            if position.unrealized_pnl_pct < -5:  # -5% 이하
                high_risk_positions.append(position)
        
        if high_risk_positions:
            print(f"\n🔴 고위험 포지션:")
            for pos in high_risk_positions[:5]:  # 상위 5개
                print(f"  📉 {pos.symbol} ({pos.strategy}): {pos.unrealized_pnl_pct:+.2f}%")
        else:
            print(f"\n✅ 고위험 포지션 없음")
        
        # 응급매도 설정
        print(f"\n🚨 응급매도 설정:")
        print(f"  활성화: {'✅ ON' if config.emergency_sell_enabled else '❌ OFF'}")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _change_settings(self):
        """설정 변경"""
        if not self.trading_system:
            return
        
        print("\n🔧 설정 변경")
        print("="*50)
        
        config = self.trading_system.config
        
        print("현재 설정:")
        print(f"1. 총 자본: {config.total_capital:,.0f}원")
        print(f"2. 최대 포지션 수: {config.max_portfolio_size}개")
        print(f"3. 응급매도: {'✅ 활성화' if config.emergency_sell_enabled else '❌ 비활성화'}")
        print(f"4. 일일 손실 한도: {config.max_daily_loss_pct}%")
        print(f"5. 모니터링 간격: {config.monitoring_interval}초")
        
        choice = input("\n변경할 설정 번호 (0: 취소): ").strip()
        
        if choice == '1':
            new_capital = input(f"새로운 총 자본 (현재: {config.total_capital:,.0f}): ")
            if new_capital:
                config.total_capital = float(new_capital)
                print("✅ 총 자본이 변경되었습니다.")
        
        elif choice == '2':
            new_max = input(f"새로운 최대 포지션 수 (현재: {config.max_portfolio_size}): ")
            if new_max:
                config.max_portfolio_size = int(new_max)
                print("✅ 최대 포지션 수가 변경되었습니다.")
        
        elif choice == '3':
            new_emergency = input("응급매도 활성화? (y/n): ").lower()
            if new_emergency in ['y', 'n']:
                config.emergency_sell_enabled = new_emergency == 'y'
                print("✅ 응급매도 설정이 변경되었습니다.")
        
        elif choice == '4':
            new_daily = input(f"새로운 일일 손실 한도 (현재: {config.max_daily_loss_pct}%): ")
            if new_daily:
                config.max_daily_loss_pct = float(new_daily)
                print("✅ 일일 손실 한도가 변경되었습니다.")
        
        elif choice == '5':
            new_interval = input(f"새로운 모니터링 간격 (현재: {config.monitoring_interval}초): ")
            if new_interval:
                config.monitoring_interval = int(new_interval)
                print("✅ 모니터링 간격이 변경되었습니다.")
        
        elif choice == '0':
            return
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _emergency_sell(self):
        """응급 매도"""
        if not self.trading_system:
            return
        
        print("\n🚨 응급 매도")
        print("="*50)
        
        portfolio = self.trading_system.position_manager.get_portfolio_summary()
        total_positions = portfolio.get('total_positions', 0)
        
        if total_positions == 0:
            print("📭 매도할 포지션이 없습니다.")
            input("\n계속하려면 Enter를 누르세요...")
            return
        
        print(f"⚠️ 경고: {total_positions}개 포지션을 모두 매도합니다.")
        print(f"현재 미실현 손익: {portfolio.get('total_pnl', 0):+,.0f}원")
        
        confirm = input("\n정말로 응급 매도를 실행하시겠습니까? (YES 입력): ").strip()
        
        if confirm == "YES":
            print("🚨 응급 매도를 실행합니다...")
            await self.trading_system._emergency_sell_all("USER_MANUAL_REQUEST")
            print("✅ 응급 매도가 완료되었습니다.")
        else:
            print("❌ 응급 매도가 취소되었습니다.")
        
        input("\n계속하려면 Enter를 누르세요...")
    
    async def _shutdown_system(self):
        """시스템 종료"""
        if not self.trading_system:
            return
        
        print("🛑 시스템을 종료합니다...")
        await self.trading_system.shutdown()
        self.trading_system = None
        print("✅ 시스템이 안전하게 종료되었습니다.")

# ============================================================================
# 🎮 메인 실행 함수들
# ============================================================================

async def run_trading_system():
    """트레이딩 시스템 직접 실행"""
    print("🚀 퀸트프로젝트 통합 트레이딩 시스템 시작")
    
    # 기본 설정으로 시스템 생성
    config = TradingConfig()
    trading_system = QuintTradingSystem(config)
    
    try:
        await trading_system.start()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        logger.error(f"시스템 실행 오류: {e}")
    finally:
        await trading_system.shutdown()

async def test_strategies():
    """전략 테스트"""
    print("🧪 전략 테스트 시작")
    
    strategies = {
        'us': USStrategyWrapper(),
        'japan': JapanStrategyWrapper(),
        'india': IndiaStrategyWrapper(),
        'crypto': CryptoStrategyWrapper()
    }
    
    for name, strategy in strategies.items():
        if not strategy.available:
            print(f"❌ {name} 전략 사용 불가")
            continue
        
        print(f"\n🎯 {name} 전략 테스트:")
        
        try:
            signals = await strategy.get_signals()
            print(f"  📊 생성된 신호: {len(signals)}개")
            
            for signal in signals[:3]:  # 상위 3개만
                print(f"  📈 {signal.symbol}: {signal.action} (신뢰도: {signal.confidence:.1%})")
                
        except Exception as e:
            print(f"  ❌ 테스트 실패: {e}")

def create_default_config():
    """기본 설정 파일 생성"""
    config = {
        'trading': {
            'total_capital': 10_000_000,
            'max_portfolio_size': 20,
            'emergency_sell_enabled': True,
            'us_strategy_enabled': True,
            'japan_strategy_enabled': True,
            'india_strategy_enabled': True,
            'crypto_strategy_enabled': True
        },
        'risk': {
            'max_daily_loss_pct': 2.0,
            'max_weekly_loss_pct': 5.0,
            'max_monthly_loss_pct': 8.0,
            'position_size_limit_pct': 15.0
        },
        'monitoring': {
            'monitoring_interval': 300,
            'health_check_interval': 60
        },
        'notification': {
            'notification_enabled': True,
            'critical_alert_enabled': True
        }
    }
    
    with open("trading_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ 기본 설정 파일 생성: trading_config.json")

async def quick_status():
    """빠른 상태 확인"""
    print("📊 퀸트프로젝트 트레이딩 시스템 빠른 상태 확인")
    print("="*60)
    
    # 모듈 가용성
    modules = [
        ('🇺🇸 미국 전략', 'us_strategy'),
        ('🇯🇵 일본 전략', 'jp_strategy'),
        ('🇮🇳 인도 전략', 'inda_strategy'),
        ('💰 암호화폐 전략', 'coin_strategy')
    ]
    
    print("📦 모듈 가용성:")
    for name, module in modules:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
    
    # 환경변수 체크
    print(f"\n🔑 환경변수:")
    env_vars = ['TELEGRAM_BOT_TOKEN', 'UPBIT_ACCESS_KEY', 'IBKR_HOST']
    for var in env_vars:
        value = os.getenv(var, '')
        status = "✅" if value else "❌"
        print(f"  {status} {var}")
    
    # 포지션 파일 체크
    print(f"\n📁 파일 시스템:")
    files = ['positions.json', 'trading.log', 'trading_config.json']
    for file in files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
    
    # 현재 시간 및 거래 가능 여부
    now = datetime.now()
    weekday = now.weekday()
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    
    print(f"\n📅 현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')} ({weekday_names[weekday]})")
    
    trading_status = {
        0: "💰 암호화폐 거래일",
        1: "🇺🇸 미국, 🇯🇵 일본 거래일",
        2: "🇮🇳 인도 거래일", 
        3: "🇺🇸 미국, 🇯🇵 일본 거래일",
        4: "💰 암호화폐 거래일",
        5: "📴 주말",
        6: "📴 주말"
    }
    
    print(f"🎯 오늘: {trading_status.get(weekday, '알 수 없음')}")

# ============================================================================
# 🎮 CLI 메인 함수
# ============================================================================

async def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "run":
            await run_trading_system()
        elif command == "test":
            await test_strategies()
        elif command == "config":
            create_default_config()
        elif command == "status":
            await quick_status()
        elif command == "cli":
            cli = TradingCLI()
            await cli.start_interactive_mode()
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print_help()
    else:
        # 대화형 CLI 시작
        cli = TradingCLI()
        await cli.start_interactive_mode()

def print_help():
    """도움말 출력"""
    help_text = """
🏆 퀸트프로젝트 통합 트레이딩 시스템

사용법:
  python trading.py           # 대화형 CLI 시작
  python trading.py run       # 트레이딩 시스템 직접 실행
  python trading.py test      # 전략 테스트
  python trading.py config    # 기본 설정 파일 생성
  python trading.py status    # 빠른 상태 확인
  python trading.py cli       # 대화형 CLI 시작

✨ 주요 기능:
  🇺🇸 미국주식 전략 (화, 목 23:30)
  🇯🇵 일본주식 전략 (화, 목 09:00)
  🇮🇳 인도주식 전략 (수 09:00)
  💰 암호화폐 전략 (월, 금 09:00)
  
  📊 실시간 포지션 모니터링
  🚨 리스크 관리 및 응급매도
  📱 통합 알림 시스템
  📈 성과 추적 및 분석

🔧 환경설정:
  TELEGRAM_BOT_TOKEN=your_token
  UPBIT_ACCESS_KEY=your_key
  IBKR_HOST=127.0.0.1
  NOTIFICATION_ENABLED=true
  EMERGENCY_SELL_ON_ERROR=true

📁 주요 파일:
  trading.log          # 거래 로그
  positions.json       # 포지션 데이터
  trading_config.json  # 설정 파일
"""
    print(help_text)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 종료되었습니다.")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        logger.critical(f"메인 실행 오류: {e}")
        import traceback
        traceback.print_exc()
