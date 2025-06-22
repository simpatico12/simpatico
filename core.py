"""
🏆 최고퀸트프로젝트 - 핵심 실행 엔진
=====================================

전 세계 시장 통합 매매 시스템:
- 🇺🇸 미국 주식 (버핏 + 린치 전략)
- 🇯🇵 일본 주식 (일목균형표 + 모멘텀)
- 🪙 암호화폐 (거래량 급증 + 기술분석)
- 📊 통합 리스크 관리
- 🔔 실시간 알림 시스템
- 📈 성과 추적 및 리포트

Author: 최고퀸트팀
Version: 1.0.0
Project: 최고퀸트프로젝트
"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml
import pandas as pd
from dataclasses import dataclass, asdict
import traceback

# 프로젝트 모듈 import
try:
    from strategies.us_strategy import USStrategy, analyze_us, get_buffett_picks, get_lynch_picks
    US_STRATEGY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 미국 주식 전략 로드 실패: {e}")
    US_STRATEGY_AVAILABLE = False

try:
    from strategies.jp_strategy import JPStrategy, analyze_jp, get_ichimoku_picks, get_momentum_picks
    JP_STRATEGY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 일본 주식 전략 로드 실패: {e}")
    JP_STRATEGY_AVAILABLE = False

try:
    from strategies.coin_strategy import CoinStrategy, analyze_coin, get_volume_spike_picks
    COIN_STRATEGY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 암호화폐 전략 로드 실패: {e}")
    COIN_STRATEGY_AVAILABLE = False

try:
    from notifier import send_telegram_message, send_trading_alert, send_market_summary
    NOTIFIER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 알림 모듈 로드 실패: {e}")
    NOTIFIER_AVAILABLE = False

try:
    from scheduler import get_today_strategies, is_trading_time
    SCHEDULER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 스케줄러 모듈 로드 실패: {e}")
    SCHEDULER_AVAILABLE = False

try:
    from trading import TradingExecutor, execute_trade_signal
    TRADING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 매매 모듈 로드 실패: {e}")
    TRADING_AVAILABLE = False

# 로깅 설정
def setup_logging():
    """로깅 시스템 설정"""
    # logs 폴더 생성
    os.makedirs('logs', exist_ok=True)
    
    # 로그 파일명 (날짜별)
    log_filename = f"logs/quant_{datetime.now().strftime('%Y%m%d')}.log"
    
    # 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🏆 최고퀸트프로젝트 로깅 시스템 초기화 완료")
    return logger

@dataclass
class TradingSignal:
    """통합 매매 신호 데이터 클래스"""
    market: str  # 'US', 'JP', 'COIN'
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    strategy: str
    reasoning: str
    target_price: float
    timestamp: datetime
    sector: Optional[str] = None
    position_size: Optional[float] = None  # 실제 매매용 포지션 크기
    additional_data: Optional[Dict] = None

@dataclass
class TradeExecution:
    """매매 실행 결과 데이터 클래스"""
    signal: TradingSignal
    executed: bool
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    quantity: Optional[float] = None
    order_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class MarketSummary:
    """시장별 요약 데이터"""
    market: str
    total_analyzed: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    top_picks: List[TradingSignal]
    executed_trades: List[TradeExecution]  # 실행된 거래
    analysis_time: float
    errors: List[str]
    is_trading_day: bool  # 오늘 해당 시장 거래일인지

class QuantTradingEngine:
    """🏆 최고퀸트프로젝트 메인 엔진"""
    
    def __init__(self, config_path: str = "configs/settings.yaml"):
        """엔진 초기화"""
        self.logger = setup_logging()
        self.config_path = config_path
        self.config = self._load_config()
        
        # 데이터 폴더 생성
        os.makedirs('data', exist_ok=True)
        
        # 전략 객체 초기화
        self.us_strategy = None
        self.jp_strategy = None
        self.coin_strategy = None
        
        # 오늘 실행할 전략 확인 (스케줄링)
        self.today_strategies = self._get_today_strategies()
        
        self._initialize_strategies()
        
        # 매매 실행 설정
        self.trading_config = self.config.get('trading', {})
        self.auto_execution = self.trading_config.get('auto_execution', False)
        self.paper_trading = self.trading_config.get('paper_trading', True)
        
        # 매매 실행기 초기화
        self.trading_executor = None
        if TRADING_AVAILABLE and self.auto_execution:
            try:
                self.trading_executor = TradingExecutor(config_path)
                self.logger.info(f"💰 매매 실행기 초기화 완료 (모의거래: {self.paper_trading})")
            except Exception as e:
                self.logger.error(f"❌ 매매 실행기 초기화 실패: {e}")
        
        # 리스크 관리 설정
        self.risk_config = self.config.get('risk_management', {})
        self.max_position_size = self.risk_config.get('max_position_size', 0.1)
        self.stop_loss = self.risk_config.get('stop_loss', -0.05)
        self.take_profit = self.risk_config.get('take_profit', 0.15)
        self.max_daily_trades = self.risk_config.get('max_daily_trades', 10)
        
        # 실행 통계
        self.daily_trades_count = 0
        self.total_signals_generated = 0
        self.session_start_time = datetime.now()
        
        self.logger.info("🚀 최고퀸트프로젝트 엔진 초기화 완료")
        self.logger.info(f"⚙️ 자동매매: {self.auto_execution}, 모의거래: {self.paper_trading}")
        self.logger.info(f"📊 오늘 활성 전략: {len(self.today_strategies)}개 - {self.today_strategies}")

    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"✅ 설정 파일 로드 성공: {self.config_path}")
                return config
        except Exception as e:
            self.logger.error(f"❌ 설정 파일 로드 실패: {e}")
            return {}

    def _get_today_strategies(self) -> List[str]:
        """오늘 실행할 전략 목록 조회"""
        if SCHEDULER_AVAILABLE:
            try:
                strategies = get_today_strategies(self.config)
                return strategies
            except Exception as e:
                self.logger.error(f"❌ 스케줄러 조회 실패: {e}")
        
        # 기본값: 모든 전략 활성화
        return ['US', 'JP', 'COIN']

    def _initialize_strategies(self):
        """전략 객체들 초기화 (스케줄링 고려)"""
        try:
            # 미국 주식 전략
            if US_STRATEGY_AVAILABLE and 'US' in self.today_strategies:
                us_config = self.config.get('us_strategy', {})
                if us_config.get('enabled', False):
                    self.us_strategy = USStrategy(self.config_path)
                    self.logger.info("🇺🇸 미국 주식 전략 활성화")
                else:
                    self.logger.info("🇺🇸 미국 주식 전략 설정에서 비활성화")
            elif 'US' not in self.today_strategies:
                self.logger.info("🇺🇸 미국 주식 전략 오늘 비활성화 (스케줄)")
            
            # 일본 주식 전략
            if JP_STRATEGY_AVAILABLE and 'JP' in self.today_strategies:
                jp_config = self.config.get('jp_strategy', {})
                if jp_config.get('enabled', False):
                    self.jp_strategy = JPStrategy(self.config_path)
                    self.logger.info("🇯🇵 일본 주식 전략 활성화")
                else:
                    self.logger.info("🇯🇵 일본 주식 전략 설정에서 비활성화")
            elif 'JP' not in self.today_strategies:
                self.logger.info("🇯🇵 일본 주식 전략 오늘 비활성화 (스케줄)")
            
            # 암호화폐 전략
            if COIN_STRATEGY_AVAILABLE and 'COIN' in self.today_strategies:
                coin_config = self.config.get('coin_strategy', {})
                if coin_config.get('enabled', False):
                    self.coin_strategy = CoinStrategy(self.config_path)
                    self.logger.info("🪙 암호화폐 전략 활성화")
                else:
                    self.logger.info("🪙 암호화폐 전략 설정에서 비활성화")
            elif 'COIN' not in self.today_strategies:
                self.logger.info("🪙 암호화폐 전략 오늘 비활성화 (스케줄)")
                    
        except Exception as e:
            self.logger.error(f"❌ 전략 초기화 실패: {e}")

    def _check_trading_time(self) -> bool:
        """현재 시간이 거래 시간인지 확인"""
        try:
            if SCHEDULER_AVAILABLE:
                return is_trading_time(self.config)
            else:
                # 기본값: 9시-16시만 거래
                current_hour = datetime.now().hour
                return 9 <= current_hour <= 16
                
        except Exception as e:
            self.logger.error(f"거래 시간 확인 실패: {e}")
            return True

    async def _execute_trades(self, signals: List[TradingSignal]) -> List[TradeExecution]:
        """매매 신호 실행"""
        executed_trades = []
        
        if not self.trading_executor or not self.auto_execution:
            self.logger.info("📊 매매 신호만 생성 (실행 비활성화)")
            # 신호만 생성하고 실행하지 않음
            for signal in signals:
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="자동 매매 비활성화"
                ))
            return executed_trades
        
        # 거래 시간 체크
        if not self._check_trading_time():
            self.logger.info("⏰ 거래 시간이 아님 - 신호만 생성")
            for signal in signals:
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="거래 시간 아님"
                ))
            return executed_trades
        
        # 일일 거래 한도 체크
        if self.daily_trades_count >= self.max_daily_trades:
            self.logger.warning(f"⚠️ 일일 거래 한도 도달: {self.daily_trades_count}/{self.max_daily_trades}")
            for signal in signals:
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="일일 거래 한도 초과"
                ))
            return executed_trades
        
        # 실제 매매 실행
        for signal in signals:
            if signal.action in ['buy', 'sell']:
                try:
                    self.logger.info(f"💰 {signal.action.upper()} 주문 실행: {signal.symbol}")
                    
                    execution_result = await execute_trade_signal(signal)
                    
                    if execution_result and execution_result.get('success', False):
                        executed_trades.append(TradeExecution(
                            signal=signal,
                            executed=True,
                            execution_price=execution_result.get('price'),
                            execution_time=datetime.now(),
                            quantity=execution_result.get('quantity'),
                            order_id=execution_result.get('order_id')
                        ))
                        
                        self.daily_trades_count += 1
                        
                        # 실행 알림 발송
                        if NOTIFIER_AVAILABLE:
                            await send_trading_alert(
                                signal.market, signal.symbol, signal.action,
                                execution_result.get('price', signal.price),
                                signal.confidence, 
                                f"✅ 매매 완료: {signal.reasoning}",
                                signal.target_price
                            )
                        
                        self.logger.info(f"✅ 매매 완료: {signal.symbol} {signal.action}")
                        
                    else:
                        error_msg = execution_result.get('error', '알 수 없는 오류') if execution_result else '실행 결과 없음'
                        executed_trades.append(TradeExecution(
                            signal=signal,
                            executed=False,
                            error_message=error_msg
                        ))
                        self.logger.error(f"❌ 매매 실패: {signal.symbol} - {error_msg}")
                        
                except Exception as e:
                    executed_trades.append(TradeExecution(
                        signal=signal,
                        executed=False,
                        error_message=str(e)
                    ))
                    self.logger.error(f"❌ 매매 실행 중 오류 {signal.symbol}: {e}")
                    
                # API 호출 제한 고려
                await asyncio.sleep(1)
            else:
                # hold 신호는 실행하지 않음
                executed_trades.append(TradeExecution(
                    signal=signal,
                    executed=False,
                    error_message="HOLD 신호"
                ))
        
        return executed_trades

    def _apply_risk_management(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """리스크 관리 적용"""
        filtered_signals = []
        
        # 일일 거래 제한 체크
        if self.daily_trades_count >= self.max_daily_trades:
            self.logger.warning(f"⚠️ 일일 거래 한도 도달: {self.daily_trades_count}/{self.max_daily_trades}")
            return filtered_signals
        
        # 신뢰도 기준 필터링
        for signal in signals:
            if signal.action == 'buy':
                # 매수 신호는 높은 신뢰도만
                if signal.confidence >= 0.7:
                    filtered_signals.append(signal)
                else:
                    self.logger.debug(f"낮은 신뢰도로 매수 신호 제외: {signal.symbol} ({signal.confidence:.2f})")
                    
            elif signal.action == 'sell':
                # 매도 신호는 중간 신뢰도 이상
                if signal.confidence >= 0.5:
                    filtered_signals.append(signal)
                else:
                    self.logger.debug(f"낮은 신뢰도로 매도 신호 제외: {signal.symbol} ({signal.confidence:.2f})")
        
        return filtered_signals

    async def analyze_us_market(self) -> MarketSummary:
        """🇺🇸 미국 시장 분석"""
        start_time = datetime.now()
        errors = []
        signals = []
        executed_trades = []
        
        if not self.us_strategy:
            return MarketSummary(
                market='US', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], analysis_time=0.0, 
                errors=['전략 비활성화'], is_trading_day='US' in self.today_strategies
            )
        
        try:
            self.logger.info("🔍 미국 시장 분석 시작...")
            
            # 전체 시장 스캔
            us_signals = await self.us_strategy.scan_all_symbols()
            
            # TradingSignal 형태로 변환
            for signal in us_signals:
                trading_signal = TradingSignal(
                    market='US',
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    price=signal.price,
                    strategy=signal.strategy_source,
                    reasoning=signal.reasoning,
                    target_price=signal.target_price,
                    timestamp=signal.timestamp,
                    sector=signal.sector,
                    position_size=signal.additional_data.get('position_size') if signal.additional_data else None,
                    additional_data=signal.additional_data
                )
                signals.append(trading_signal)
            
            # 리스크 관리 적용
            filtered_signals = self._apply_risk_management(signals)
            
            # 매매 실행 (매수/매도 신호만)
            trade_signals = [s for s in filtered_signals if s.action in ['buy', 'sell']]
            if trade_signals:
                executed_trades = await self._execute_trades(trade_signals)
            
            # 통계 계산
            buy_signals = len([s for s in signals if s.action == 'buy'])
            sell_signals = len([s for s in signals if s.action == 'sell'])
            hold_signals = len([s for s in signals if s.action == 'hold'])
            
            # 상위 종목 선정 (매수 신호 중 신뢰도 높은 순)
            top_picks = sorted([s for s in signals if s.action == 'buy'], 
                             key=lambda x: x.confidence, reverse=True)[:5]
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ 미국 시장 분석 완료 - 매수:{buy_signals}, 매도:{sell_signals}, 보유:{hold_signals}")
            if executed_trades:
                executed_count = len([t for t in executed_trades if t.executed])
                self.logger.info(f"💰 실행된 거래: {executed_count}개")
            
            return MarketSummary(
                market='US',
                total_analyzed=len(signals),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                top_picks=top_picks,
                executed_trades=executed_trades,
                analysis_time=analysis_time,
                errors=errors,
                is_trading_day='US' in self.today_strategies
            )
            
        except Exception as e:
            error_msg = f"미국 시장 분석 실패: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
            
            return MarketSummary(
                market='US', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], 
                analysis_time=(datetime.now() - start_time).total_seconds(),
                errors=errors, is_trading_day='US' in self.today_strategies
            )

    async def analyze_jp_market(self) -> MarketSummary:
        """🇯🇵 일본 시장 분석"""
        start_time = datetime.now()
        errors = []
        signals = []
        executed_trades = []
        
        if not self.jp_strategy:
            return MarketSummary(
                market='JP', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], analysis_time=0.0, 
                errors=['전략 비활성화'], is_trading_day='JP' in self.today_strategies
            )
        
        try:
            self.logger.info("🔍 일본 시장 분석 시작...")
            
            # 전체 시장 스캔
            jp_signals = await self.jp_strategy.scan_all_symbols()
            
            # TradingSignal 형태로 변환
            for signal in jp_signals:
                trading_signal = TradingSignal(
                    market='JP',
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    price=signal.price,
                    strategy=signal.strategy_source,
                    reasoning=signal.reasoning,
                    target_price=signal.target_price,
                    timestamp=signal.timestamp,
                    sector=signal.sector,
                    position_size=signal.additional_data.get('position_size') if signal.additional_data else None,
                    additional_data=signal.additional_data
                )
                signals.append(trading_signal)
            
            # 리스크 관리 적용
            filtered_signals = self._apply_risk_management(signals)
            
            # 매매 실행 (매수/매도 신호만)
            trade_signals = [s for s in filtered_signals if s.action in ['buy', 'sell']]
            if trade_signals:
                executed_trades = await self._execute_trades(trade_signals)
            
            # 통계 계산
            buy_signals = len([s for s in signals if s.action == 'buy'])
            sell_signals = len([s for s in signals if s.action == 'sell'])
            hold_signals = len([s for s in signals if s.action == 'hold'])
            
            # 상위 종목 선정
            top_picks = sorted([s for s in signals if s.action == 'buy'], 
                             key=lambda x: x.confidence, reverse=True)[:5]
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ 일본 시장 분석 완료 - 매수:{buy_signals}, 매도:{sell_signals}, 보유:{hold_signals}")
            if executed_trades:
                executed_count = len([t for t in executed_trades if t.executed])
                self.logger.info(f"💰 실행된 거래: {executed_count}개")
            
            return MarketSummary(
                market='JP',
                total_analyzed=len(signals),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                top_picks=top_picks,
                executed_trades=executed_trades,
                analysis_time=analysis_time,
                errors=errors,
                is_trading_day='JP' in self.today_strategies
            )
            
        except Exception as e:
            error_msg = f"일본 시장 분석 실패: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
            
            return MarketSummary(
                market='JP', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], 
                analysis_time=(datetime.now() - start_time).total_seconds(),
                errors=errors, is_trading_day='JP' in self.today_strategies
            )

    async def analyze_coin_market(self) -> MarketSummary:
        """🪙 암호화폐 시장 분석"""
        start_time = datetime.now()
        errors = []
        signals = []
        executed_trades = []
        
        if not self.coin_strategy:
            return MarketSummary(
                market='COIN', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], analysis_time=0.0, 
                errors=['전략 비활성화'], is_trading_day='COIN' in self.today_strategies
            )
        
        try:
            self.logger.info("🔍 암호화폐 시장 분석 시작...")
            
            # 전체 시장 스캔
            coin_signals = await self.coin_strategy.scan_all_symbols()
            
            # TradingSignal 형태로 변환
            for signal in coin_signals:
                trading_signal = TradingSignal(
                    market='COIN',
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    price=signal.price,
                    strategy=signal.strategy_source,
                    reasoning=signal.reasoning,
                    target_price=signal.target_price,
                    timestamp=signal.timestamp,
                    sector=signal.sector,
                    position_size=signal.additional_data.get('position_size') if signal.additional_data else None,
                    additional_data=signal.additional_data
                )
                signals.append(trading_signal)
            
            # 리스크 관리 적용
            filtered_signals = self._apply_risk_management(signals)
            
            # 매매 실행 (매수/매도 신호만)
            trade_signals = [s for s in filtered_signals if s.action in ['buy', 'sell']]
            if trade_signals:
                executed_trades = await self._execute_trades(trade_signals)
            
            # 통계 계산
            buy_signals = len([s for s in signals if s.action == 'buy'])
            sell_signals = len([s for s in signals if s.action == 'sell'])
            hold_signals = len([s for s in signals if s.action == 'hold'])
            
            # 상위 종목 선정 (매수 신호 중 신뢰도 높은 순)
            top_picks = sorted([s for s in signals if s.action == 'buy'], 
                             key=lambda x: x.confidence, reverse=True)[:5]
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ 암호화폐 시장 분석 완료 - 매수:{buy_signals}, 매도:{sell_signals}, 보유:{hold_signals}")
            if executed_trades:
                executed_count = len([t for t in executed_trades if t.executed])
                self.logger.info(f"💰 실행된 거래: {executed_count}개")
            
            return MarketSummary(
                market='COIN',
                total_analyzed=len(signals),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                top_picks=top_picks,
                executed_trades=executed_trades,
                analysis_time=analysis_time,
                errors=errors,
                is_trading_day='COIN' in self.today_strategies
            )
            
        except Exception as e:
            error_msg = f"암호화폐 시장 분석 실패: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
            
            return MarketSummary(
                market='COIN', total_analyzed=0, buy_signals=0, sell_signals=0,
                hold_signals=0, top_picks=[], executed_trades=[], 
                analysis_time=(datetime.now() - start_time).total_seconds(),
                errors=errors, is_trading_day='COIN' in self.today_strategies
            )

    async def run_full_analysis(self) -> Dict[str, MarketSummary]:
        """🌍 전체 시장 통합 분석"""
        self.logger.info("🚀 전체 시장 통합 분석 시작...")
        start_time = datetime.now()
        
        # 병렬로 모든 시장 분석
        tasks = []
        
        if self.us_strategy:
            tasks.append(self.analyze_us_market())
        if self.jp_strategy:
            tasks.append(self.analyze_jp_market())
        if self.coin_strategy:
            tasks.append(self.analyze_coin_market())
        
        if not tasks:
            self.logger.warning("⚠️ 활성화된 전략이 없습니다")
            return {}
        
        # 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 정리
        market_summaries = {}
        total_signals = 0
        total_buy_signals = 0
        
        for result in results:
            if isinstance(result, MarketSummary):
                market_summaries[result.market] = result
                total_signals += result.total_analyzed
                total_buy_signals += result.buy_signals
            elif isinstance(result, Exception):
                self.logger.error(f"❌ 시장 분석 중 오류: {result}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        self.total_signals_generated += total_signals
        
        self.logger.info(f"🎯 전체 분석 완료 - {len(market_summaries)}개 시장, "
                        f"총 {total_signals}개 신호, 매수 {total_buy_signals}개, "
                        f"소요시간: {total_time:.1f}초")
        
        # 결과 저장
        await self._save_analysis_results(market_summaries)
        
        # 알림 발송
        if NOTIFIER_AVAILABLE:
            await self._send_analysis_notification(market_summaries)
        
        return market_summaries

    async def _save_analysis_results(self, market_summaries: Dict[str, MarketSummary]):
        """분석 결과 저장"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/analysis_{timestamp}.json"
            
            # 직렬화 가능한 형태로 변환
            save_data = {
                'timestamp': timestamp,
                'session_info': {
                    'start_time': self.session_start_time.isoformat(),
                    'total_signals_generated': self.total_signals_generated,
                    'daily_trades_count': self.daily_trades_count,
                    'today_strategies': self.today_strategies
                },
                'market_summaries': {}
            }
            
            for market, summary in market_summaries.items():
                save_data['market_summaries'][market] = {
                    'market': summary.market,
                    'total_analyzed': summary.total_analyzed,
                    'buy_signals': summary.buy_signals,
                    'sell_signals': summary.sell_signals,
                    'hold_signals': summary.hold_signals,
                    'analysis_time': summary.analysis_time,
                    'errors': summary.errors,
                    'is_trading_day': summary.is_trading_day,
                    'top_picks': [asdict(signal) for signal in summary.top_picks],
                    'executed_trades_count': len([t for t in summary.executed_trades if t.executed])
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"📊 분석 결과 저장 완료: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ 분석 결과 저장 실패: {e}")

    async def _send_analysis_notification(self, market_summaries: Dict[str, MarketSummary]):
        """분석 결과 알림 발송"""
        try:
            # 알림 설정 확인
            notification_config = self.config.get('notifications', {})
            if not notification_config.get('telegram', {}).get('enabled', False):
                return
            
            # 통합 요약 알림 발송
            await send_market_summary(market_summaries)
            
            self.logger.info("📱 알림 발송 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 알림 발송 실패: {e}")

    async def get_quick_analysis(self, symbols: List[str]) -> List[TradingSignal]:
        """빠른 개별 종목 분석"""
        signals = []
        
        for symbol in symbols:
            try:
                # 시장 판별 (간단한 방식)
                if symbol.endswith('.T'):
                    # 일본 주식
                    if self.jp_strategy:
                        result = await analyze_jp(symbol)
                        signal = TradingSignal(
                            market='JP', symbol=symbol, action=result['decision'],
                            confidence=result['confidence_score']/100, price=result['price'],
                            strategy='jp_quick', reasoning=result['reasoning'],
                            target_price=result['target_price'], timestamp=datetime.now()
                        )
                        signals.append(signal)
                        
                elif '-' in symbol and 'KRW' in symbol:
                    # 암호화폐
                    if self.coin_strategy:
                        result = await analyze_coin(symbol)
                        signal = TradingSignal(
                            market='COIN', symbol=symbol, action=result['decision'],
                            confidence=result['confidence_score']/100, price=result['price'],
                            strategy='coin_quick', reasoning=result['reasoning'],
                            target_price=result['target_price'], timestamp=datetime.now()
                        )
                        signals.append(signal)
                        
                else:
                    # 미국 주식
                    if self.us_strategy:
                        result = await analyze_us(symbol)
                        signal = TradingSignal(
                            market='US', symbol=symbol, action=result['decision'],
                            confidence=result['confidence_score']/100, price=result['price'],
                            strategy='us_quick', reasoning=result['reasoning'],
                            target_price=result['target_price'], timestamp=datetime.now()
                        )
                        signals.append(signal)
                        
            except Exception as e:
                self.logger.error(f"❌ {symbol} 빠른 분석 실패: {e}")
        
        return signals

    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        uptime = (datetime.now() - self.session_start_time).total_seconds()
        
        return {
            'system_status': 'running',
            'uptime_seconds': uptime,
            'strategies_enabled': {
                'us_strategy': self.us_strategy is not None,
                'jp_strategy': self.jp_strategy is not None,
                'coin_strategy': self.coin_strategy is not None
            },
            'today_strategies': self.today_strategies,
            'daily_trades_count': self.daily_trades_count,
            'total_signals_generated': self.total_signals_generated,
            'max_daily_trades': self.max_daily_trades,
            'auto_execution': self.auto_execution,
            'paper_trading': self.paper_trading,
            'session_start_time': self.session_start_time.isoformat(),
            'last_config_load': self.config_path
        }

# 편의 함수들
async def run_single_analysis():
    """단일 분석 실행"""
    engine = QuantTradingEngine()
    results = await engine.run_full_analysis()
    return results

async def analyze_symbols(symbols: List[str]):
    """특정 종목들 분석"""
    engine = QuantTradingEngine()
    signals = await engine.get_quick_analysis(symbols)
    return signals

def get_engine_status():
    """엔진 상태 조회"""
    engine = QuantTradingEngine()
    return engine.get_system_status()

# 메인 실행 함수
async def main():
    """메인 실행 함수"""
    try:
        print("🏆 최고퀸트프로젝트 시작!")
        print("=" * 50)
        
        # 엔진 초기화
        engine = QuantTradingEngine()
        
        # 시스템 상태 출력
        status = engine.get_system_status()
        print(f"💻 시스템 상태: {status['system_status']}")
        print(f"📊 활성화된 전략: {sum(status['strategies_enabled'].values())}개")
        print(f"🔄 일일 거래 한도: {status['daily_trades_count']}/{status['max_daily_trades']}")
        print(f"📅 오늘 실행 전략: {status['today_strategies']}")
        print()
        
        # 전체 시장 분석 실행
        results = await engine.run_full_analysis()
        
        # 결과 요약 출력
        print("\n📈 분석 결과 요약:")
        print("-" * 30)
        
        total_buy = 0
        total_executed = 0
        for market, summary in results.items():
            market_name = {'US': '🇺🇸 미국', 'JP': '🇯🇵 일본', 'COIN': '🪙 코인'}.get(market, market)
            executed_count = len([t for t in summary.executed_trades if t.executed])
            print(f"{market_name}: 매수 {summary.buy_signals}개 / 전체 {summary.total_analyzed}개 "
                  f"/ 실행 {executed_count}개 ({summary.analysis_time:.1f}초)")
            total_buy += summary.buy_signals
            total_executed += executed_count
            
            # 상위 추천 종목
            if summary.top_picks:
                print(f"  상위 추천: ", end="")
                top_3 = summary.top_picks[:3]
                symbols = [f"{pick.symbol}({pick.confidence*100:.0f}%)" for pick in top_3]
                print(", ".join(symbols))
        
        print(f"\n🎯 총 매수 신호: {total_buy}개")
        if total_executed > 0:
            print(f"💰 실행된 거래: {total_executed}개")
        print(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다")
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())