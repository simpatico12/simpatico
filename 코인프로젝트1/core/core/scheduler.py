#!/usr/bin/env python3
"""
🏆 COMPLETE ELITE TRADING SYSTEM 🏆
완전히 독립적인 하나의 파일로 모든 기능 구현

사용법:
1. python complete_elite_system.py
2. 메뉴에서 원하는 기능 선택
3. 끝!
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import time
import warnings
warnings.filterwarnings('ignore')

# 선택적 패키지들
try:
    from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = None

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("elite_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 시간대 설정
seoul_tz = pytz.timezone('Asia/Seoul')
ny_tz = pytz.timezone('America/New_York')

def get_seoul_now():
    return datetime.now(seoul_tz)

# 데이터베이스 테이블 정의 (선택적)
if SQLALCHEMY_AVAILABLE:
    class TradeRecord(Base):
        __tablename__ = 'trades'
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.now)
        market = Column(String(20), nullable=False)
        symbol = Column(String(50), nullable=False)
        action = Column(String(10), nullable=False)
        quantity = Column(Float, nullable=False)
        price = Column(Float, nullable=False)
        confidence = Column(Float, nullable=False)
        strategy = Column(String(50), nullable=False)
        pnl = Column(Float, default=0.0)
        trade_data = Column(Text)  # metadata 대신 trade_data 사용

# 데이터 클래스들
@dataclass
class MarketData:
    """시장 데이터"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    rsi: float = 50.0
    macd: float = 0.0
    momentum: float = 0.0
    volatility: float = 0.02

@dataclass
class TradingSignal:
    """거래 시그널"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    quantity: float
    target_price: float
    stop_loss: Optional[float] = None
    strategy: str = "default"
    timestamp: datetime = field(default_factory=get_seoul_now)
    factors: Dict = field(default_factory=dict)

@dataclass
class ScheduledTask:
    """스케줄 작업"""
    market: str
    days: List[int]  # 0=월, 1=화, ..., 6=일
    time: str        # "HH:MM" 형식
    enabled: bool = True

class TradingAPIWrapper:
    """메인 API 래퍼"""
    
    def __init__(self):
        # 포트폴리오 설정
        self.initial_capital = 10000000  # 1천만원
        self.portfolio = {
            'coin': {'balance': self.initial_capital * 0.3, 'positions': {}},
            'japan': {'balance': self.initial_capital * 0.35, 'positions': {}},
            'us': {'balance': self.initial_capital * 0.35, 'positions': {}}
        }
        
        # 거래 기록
        self.trade_history = []
        self.performance_stats = {}
        
        # 데이터베이스 (선택적)
        self.db_session = None
        if SQLALCHEMY_AVAILABLE:
            self._init_database()
        
        logger.info("✅ TradingAPIWrapper 초기화 완료")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            engine = create_engine('sqlite:///elite_trading.db', echo=False)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            self.db_session = Session()
            logger.info("✅ 데이터베이스 연결 완료")
        except Exception as e:
            logger.warning(f"⚠️ 데이터베이스 초기화 실패: {e}")

    async def execute_trading(self, market: str):
        """🚀 메인 거래 실행"""
        logger.info(f"🚀 {market} 매매 시작")
        
        try:
            # ① 시장 데이터 수집
            market_data = await self.fetch_market_data(market)
            logger.info(f"📊 {market} 데이터 수집 완료: {len(market_data)}개 종목")
            
            # ② 시그널 생성
            signals = self.generate_signal(market_data)
            logger.info(f"🧠 {market} 시그널 생성: {len(signals)}개")
            
            # ③ 리스크 관리
            approved_signals = self.check_risk(signals)
            logger.info(f"🛡️ {market} 리스크 승인: {len(approved_signals)}개")
            
            # ④ 주문 실행
            results = await self.execute_order(market, approved_signals)
            logger.info(f"📝 {market} 주문 완료: {len(results)}개")
            
            # ⑤ 성과 기록
            await self.track_performance(market, results)
            
            logger.info(f"✅ {market} 매매 완료")
            
        except Exception as e:
            logger.error(f"❌ {market} 매매 실행 실패: {e}")

    async def fetch_market_data(self, market: str) -> Dict[str, MarketData]:
        """📊 시장 데이터 수집"""
        logger.info(f"📊 {market} 데이터 수집 중...")
        
        symbols = self._get_symbols(market)
        market_data = {}
        
        for symbol in symbols[:3]:  # 최대 3개 종목
            try:
                if market == "coin":
                    # 암호화폐 시뮬레이션 데이터
                    price = np.random.uniform(20000, 80000)
                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        price=price,
                        volume=np.random.uniform(1000, 10000),
                        timestamp=get_seoul_now(),
                        rsi=np.random.uniform(25, 75),
                        macd=np.random.uniform(-1000, 1000),
                        momentum=np.random.uniform(-0.05, 0.05),
                        volatility=np.random.uniform(0.02, 0.08)
                    )
                
                elif YFINANCE_AVAILABLE and market in ["japan", "us"]:
                    # 실제 주식 데이터
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="5d")
                        
                        if not hist.empty:
                            latest = hist.iloc[-1]
                            prices = hist['Close'].values
                            
                            market_data[symbol] = MarketData(
                                symbol=symbol,
                                price=float(latest['Close']),
                                volume=float(latest['Volume']),
                                timestamp=get_seoul_now(),
                                rsi=50 + np.random.uniform(-15, 15),
                                macd=np.random.uniform(-2, 2),
                                momentum=prices[-1]/prices[-5] - 1 if len(prices) >= 5 else 0,
                                volatility=np.std(np.diff(prices)/prices[:-1]) if len(prices) > 1 else 0.02
                            )
                        else:
                            raise Exception("No data")
                            
                    except:
                        # 실패시 시뮬레이션 데이터
                        market_data[symbol] = MarketData(
                            symbol=symbol,
                            price=np.random.uniform(50, 500),
                            volume=np.random.uniform(100000, 5000000),
                            timestamp=get_seoul_now(),
                            rsi=np.random.uniform(30, 70),
                            macd=np.random.uniform(-5, 5),
                            momentum=np.random.uniform(-0.03, 0.03),
                            volatility=np.random.uniform(0.01, 0.05)
                        )
                
                else:
                    # 시뮬레이션 데이터
                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        price=np.random.uniform(50, 500),
                        volume=np.random.uniform(100000, 5000000),
                        timestamp=get_seoul_now(),
                        rsi=np.random.uniform(30, 70),
                        macd=np.random.uniform(-5, 5),
                        momentum=np.random.uniform(-0.03, 0.03),
                        volatility=np.random.uniform(0.01, 0.05)
                    )
                    
            except Exception as e:
                logger.warning(f"⚠️ {symbol} 데이터 수집 실패: {e}")
        
        return market_data

    def _get_symbols(self, market: str) -> List[str]:
        """시장별 심볼 목록"""
        symbols_map = {
            'coin': ['BTC-KRW', 'ETH-KRW', 'XRP-KRW', 'ADA-KRW'],
            'japan': ['7203.T', '6758.T', '9984.T', '8306.T'],  # Toyota, Sony, SoftBank, MUFG
            'us': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        }
        return symbols_map.get(market, [])

    def generate_signal(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """📈 고급 시그널 생성"""
        logger.info("🧠 AI 시그널 생성 중...")
        
        signals = []
        
        for symbol, data in market_data.items():
            try:
                # 다중 팩터 분석
                factors = {}
                
                # RSI 팩터
                if data.rsi < 30:
                    rsi_signal = (30 - data.rsi) / 30  # 과매도
                elif data.rsi > 70:
                    rsi_signal = -(data.rsi - 70) / 30  # 과매수
                else:
                    rsi_signal = 0
                factors['rsi'] = rsi_signal
                
                # MACD 팩터
                factors['macd'] = np.tanh(data.macd / 1000)  # -1 to 1 정규화
                
                # 모멘텀 팩터
                factors['momentum'] = np.tanh(data.momentum * 20)
                
                # 변동성 팩터 (높으면 부정적)
                factors['volatility'] = -min(data.volatility * 25, 1.0)
                
                # 가격 트렌드 팩터 (간단한 추세)
                factors['trend'] = np.random.uniform(-0.5, 0.5)  # 시뮬레이션
                
                # 가중 평균으로 종합 시그널 계산
                weights = {
                    'rsi': 0.25,
                    'macd': 0.25, 
                    'momentum': 0.30,
                    'volatility': 0.10,
                    'trend': 0.10
                }
                
                composite_signal = sum(factors[k] * weights[k] for k in factors.keys())
                
                # 시그널 결정
                if composite_signal > 0.35:
                    action = "BUY"
                    confidence = min(0.95, 0.6 + abs(composite_signal))
                elif composite_signal < -0.35:
                    action = "SELL" 
                    confidence = min(0.95, 0.6 + abs(composite_signal))
                else:
                    action = "HOLD"
                    confidence = 0.5
                
                # 포지션 크기 계산
                max_position = 0.15  # 최대 15%
                position_size = max_position * confidence
                
                # 투자 금액 계산
                portfolio_balance = 1000000  # 기본 100만원
                investment_amount = portfolio_balance * position_size
                quantity = investment_amount / data.price
                
                # 스톱로스 계산
                atr = data.volatility * data.price  # Average True Range 근사치
                stop_loss = None
                if action == "BUY":
                    stop_loss = data.price - (2.0 * atr)
                elif action == "SELL":
                    stop_loss = data.price + (2.0 * atr)
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    quantity=quantity,
                    target_price=data.price,
                    stop_loss=stop_loss,
                    strategy="Elite_Multi_Factor",
                    timestamp=get_seoul_now(),
                    factors=factors
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 시그널 생성 실패: {e}")
        
        # 신뢰도 순으로 정렬
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals

    def check_risk(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """🛡️ 리스크 관리"""
        logger.info("🛡️ 리스크 체크 중...")
        
        approved = []
        
        for signal in signals:
            # 기본 리스크 체크
            if signal.confidence < 0.65:
                logger.warning(f"🚫 {signal.symbol} 신뢰도 부족: {signal.confidence:.2f}")
                continue
            
            # 포지션 크기 체크
            investment_value = signal.quantity * signal.target_price
            if investment_value > 500000:  # 50만원 초과
                logger.warning(f"🚫 {signal.symbol} 투자금액 초과: ₩{investment_value:,.0f}")
                continue
            
            # 변동성 체크
            volatility_factor = signal.factors.get('volatility', 0)
            if volatility_factor < -0.8:  # 너무 높은 변동성
                logger.warning(f"🚫 {signal.symbol} 변동성 위험: {volatility_factor:.2f}")
                continue
            
            # HOLD 시그널 제외
            if signal.action == "HOLD":
                continue
            
            approved.append(signal)
            
            # 최대 5개까지만
            if len(approved) >= 5:
                break
        
        return approved

    async def execute_order(self, market: str, signals: List[TradingSignal]) -> List[Dict]:
        """📝 주문 실행"""
        logger.info(f"📝 {market} 주문 실행 중...")
        
        results = []
        
        for signal in signals:
            try:
                # 주문 실행 시뮬레이션
                execution_result = {
                    'timestamp': get_seoul_now(),
                    'market': market,
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'quantity': signal.quantity,
                    'price': signal.target_price,
                    'total_value': signal.quantity * signal.target_price,
                    'confidence': signal.confidence,
                    'strategy': signal.strategy,
                    'status': 'EXECUTED',
                    'order_id': f"ORDER_{int(time.time())}_{len(results)}"
                }
                
                results.append(execution_result)
                
                # 거래 기록
                self.trade_history.append(execution_result)
                
                # 데이터베이스 저장
                if self.db_session and SQLALCHEMY_AVAILABLE:
                    try:
                        trade_record = TradeRecord(
                            timestamp=execution_result['timestamp'],
                            market=market,
                            symbol=signal.symbol,
                            action=signal.action,
                            quantity=signal.quantity,
                            price=signal.target_price,
                            confidence=signal.confidence,
                            strategy=signal.strategy,
                            trade_data=json.dumps(signal.factors)
                        )
                        self.db_session.add(trade_record)
                        self.db_session.commit()
                    except Exception as e:
                        logger.warning(f"⚠️ DB 저장 실패: {e}")
                
                logger.info(f"📝 {signal.symbol} {signal.action} 주문 완료: "
                          f"수량 {signal.quantity:.2f}, 금액 ₩{signal.quantity * signal.target_price:,.0f}")
                
                # 주문 간 딜레이
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ {signal.symbol} 주문 실행 실패: {e}")
        
        return results

    async def track_performance(self, market: str, results: List[Dict]):
        """📊 성과 추적"""
        if not results:
            return
        
        total_trades = len(results)
        total_value = sum(r['total_value'] for r in results)
        avg_confidence = sum(r['confidence'] for r in results) / total_trades
        
        buy_count = sum(1 for r in results if r['action'] == 'BUY')
        sell_count = sum(1 for r in results if r['action'] == 'SELL')
        
        self.performance_stats[market] = {
            'timestamp': get_seoul_now(),
            'total_trades': total_trades,
            'total_value': total_value,
            'avg_confidence': avg_confidence,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'avg_trade_size': total_value / total_trades if total_trades > 0 else 0
        }
        
        logger.info(f"📊 {market} 성과 요약:")
        logger.info(f"   총 거래: {total_trades}건")
        logger.info(f"   총 금액: ₩{total_value:,.0f}")
        logger.info(f"   평균 신뢰도: {avg_confidence:.2f}")
        logger.info(f"   매수/매도: {buy_count}/{sell_count}")

class TradingScheduler:
    """📅 거래 스케줄러"""
    
    def __init__(self, api_wrapper: TradingAPIWrapper):
        self.api = api_wrapper
        self.tasks = []
        self.running = False
        
        # 기본 스케줄 설정
        self.setup_tasks()
        
        logger.info("📅 스케줄러 초기화 완료")
    
    def setup_tasks(self):
        """기본 스케줄 설정"""
        self.tasks = [
            ScheduledTask(market="coin", days=[0, 4], time="08:30"),    # 월, 금 08:30
            ScheduledTask(market="japan", days=[1, 3], time="10:00"),   # 화, 목 10:00  
            ScheduledTask(market="us", days=[1, 3], time="22:30"),      # 화, 목 22:30
        ]
        logger.info(f"🔧 스케줄 설정 완료: {len(self.tasks)}개 작업")
    
    async def run(self):
        """🚀 스케줄러 실행"""
        logger.info("🚀 스케줄러 시작 (서울 시간 기준)")
        self.running = True
        
        while self.running:
            try:
                now = get_seoul_now()
                
                for task in self.tasks:
                    if not task.enabled:
                        continue
                    
                    # 요일 체크 (0=월요일, 6=일요일)
                    if now.weekday() in task.days:
                        # 시간 체크
                        target_time = datetime.strptime(task.time, "%H:%M").time()
                        current_time = now.time()
                        
                        # 정확한 시간에 실행 (30초 오차 허용)
                        if (current_time.hour == target_time.hour and 
                            current_time.minute == target_time.minute and
                            current_time.second < 30):
                            
                            logger.info(f"⏰ {task.market} 매매 시간! ({task.time})")
                            await self.api.execute_trading(task.market)
                
                # 1분마다 체크
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("⌨️ 사용자에 의한 스케줄러 중지")
                break
            except Exception as e:
                logger.error(f"❌ 스케줄러 오류: {e}")
                await asyncio.sleep(10)
        
        self.running = False
        logger.info("🛑 스케줄러 중지됨")
    
    def get_status(self) -> Dict:
        """스케줄러 상태"""
        return {
            'running': self.running,
            'tasks': [
                {
                    'market': task.market,
                    'days': task.days,
                    'time': task.time,
                    'enabled': task.enabled
                }
                for task in self.tasks
            ],
            'current_time': get_seoul_now().isoformat(),
            'next_executions': self._get_next_executions()
        }
    
    def _get_next_executions(self) -> List[Dict]:
        """다음 실행 시간들"""
        now = get_seoul_now()
        next_executions = []
        
        for task in self.tasks:
            if not task.enabled:
                continue
                
            # 다음 실행 시간 계산
            for day in task.days:
                days_ahead = (day - now.weekday()) % 7
                if days_ahead == 0:  # 오늘
                    target_time = datetime.strptime(task.time, "%H:%M").time()
                    if now.time() > target_time:
                        days_ahead = 7  # 다음 주
                
                next_exec = now + timedelta(days=days_ahead)
                next_exec = next_exec.replace(
                    hour=int(task.time.split(':')[0]),
                    minute=int(task.time.split(':')[1]),
                    second=0,
                    microsecond=0
                )
                
                next_executions.append({
                    'market': task.market,
                    'datetime': next_exec.isoformat(),
                    'days_from_now': days_ahead
                })
        
        return sorted(next_executions, key=lambda x: x['datetime'])

# 메인 실행 함수
async def main():
    """🚀 메인 실행"""
    try:
        print("🏆" + "="*60 + "🏆")
        print("        COMPLETE ELITE TRADING SYSTEM")
        print("🏆" + "="*60 + "🏆")
        
        # 시스템 초기화
        logger.info("🔧 시스템 초기화 중...")
        api = TradingAPIWrapper()
        scheduler = TradingScheduler(api)
        
        # 패키지 상태 체크
        print(f"\n📦 패키지 상태:")
        print(f"   SQLAlchemy: {'✅' if SQLALCHEMY_AVAILABLE else '❌'}")
        print(f"   yfinance: {'✅' if YFINANCE_AVAILABLE else '❌'}")
        print(f"   데이터베이스: {'✅' if api.db_session else '❌'}")
        
        while True:
            print("\n" + "="*50)
            print("📋 메뉴")
            print("="*50)
            print("1. 🪙 코인 매매 테스트")
            print("2. 🇯🇵 일본 주식 매매 테스트")
            print("3. 🇺🇸 미국 주식 매매 테스트")
            print("4. 🕐 자동 스케줄러 시작")
            print("5. 📊 시스템 상태 확인")
            print("6. 📈 성과 리포트")
            print("7. 🔧 스케줄 상태")
            print("0. 🚪 종료")
            print("="*50)
            
            choice = input("선택하세요 (0-7): ").strip()
            
            if choice == '1':
                await api.execute_trading("coin")
                
            elif choice == '2':
                await api.execute_trading("japan")
                
            elif choice == '3':
                await api.execute_trading("us")
                
            elif choice == '4':
                print("🕐 자동 스케줄러 시작됩니다...")
                print("   중지하려면 Ctrl+C를 누르세요")
                await scheduler.run()
                
            elif choice == '5':
                print("\n📊 시스템 상태:")
                print(f"   포트폴리오: {json.dumps(api.portfolio, indent=2, ensure_ascii=False)}")
                print(f"   총 거래 기록: {len(api.trade_history)}건")
                
            elif choice == '6':
                print("\n📈 성과 리포트:")
                if api.performance_stats:
                    for market, stats in api.performance_stats.items():
                        print(f"\n{market.upper()} 시장:")
                        print(f"   거래 수: {stats['total_trades']}건")
                        print(f"   총 금액: ₩{stats['total_value']:,.0f}")
                        print(f"   평균 신뢰도: {stats['avg_confidence']:.2f}")
                        print(f"   매수/매도: {stats['buy_count']}/{stats['sell_count']}")
                else:
                    print("   아직 거래 기록이 없습니다.")
                    
            elif choice == '7':
                status = scheduler.get_status()
                print(f"\n🕐 스케줄러 상태:")
                print(f"   실행 중: {status['running']}")
                print(f"   현재 시간: {status['current_time']}")
                print(f"\n📅 예정된 거래:")
                for exec_info in status['next_executions'][:3]:
                    print(f"   {exec_info['market']}: {exec_info['datetime']}")
                    
            elif choice == '0':
                logger.info("👋 시스템 종료")
                break
                
            else:
                print("올바른 번호를 선택하세요 (0-7)")
        
    except KeyboardInterrupt:
        logger.info("⌨️ 사용자에 의한 종료")
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
