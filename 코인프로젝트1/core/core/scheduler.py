"""
🔧 Metadata 에러 수정된 트레이딩 시스템
SQLAlchemy 예약어 충돌 해결
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
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("elite_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

seoul_tz = pytz.timezone('Asia/Seoul')

# 데이터베이스 설정
Base = declarative_base()

class TradeRecord(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    market = Column(String(20), nullable=False)
    symbol = Column(String(50), nullable=False)
    action = Column(String(10), nullable=False)  # BUY/SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=False)
    pnl = Column(Float, default=0.0)
    extra_data = Column(Text)  # 🔧 metadata → extra_data로 변경!

class PortfolioRecord(Base):
    __tablename__ = 'portfolio'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    market = Column(String(20), nullable=False)
    total_value = Column(Float, nullable=False)
    positions = Column(Text)  # JSON
    daily_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)

@dataclass
class MarketData:
    """시장 데이터 구조"""
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
    """트레이딩 시그널"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    quantity: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    strategy: str = "default"
    timestamp: datetime = field(default_factory=lambda: datetime.now(seoul_tz))
    signal_data: Dict = field(default_factory=dict)  # 🔧 metadata → signal_data로 변경!

class QuantAPIWrapper:
    """수정된 Quant API Wrapper"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 포트폴리오 상태
        self.initial_capital = self.config.get('initial_capital', 1000000)
        self.portfolio = {
            'coin': {'balance': self.initial_capital * 0.3, 'positions': {}},
            'japan': {'balance': self.initial_capital * 0.35, 'positions': {}},
            'us': {'balance': self.initial_capital * 0.35, 'positions': {}}
        }
        
        logger.info("✅ QuantAPIWrapper 초기화 완료")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            db_url = self.config.get('database_url', 'sqlite:///trading_system.db')
            self.engine = create_engine(db_url, echo=False)
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            logger.info("✅ 데이터베이스 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 데이터베이스 초기화 실패: {e}")
            self.session = None

    async def execute_trading(self, market: str):
        """🚀 메인 트레이딩 실행"""
        logger.info(f"🚀 {market} 매매 시작")
        
        try:
            # ① 데이터 수집
            market_data = await self.fetch_market_data(market)
            
            # ② 시그널 생성
            signals = self.generate_signals(market_data)
            
            # ③ 리스크 관리
            approved_signals = self.risk_management(signals)
            
            # ④ 주문 실행
            execution_results = await self.execute_orders(market, approved_signals)
            
            # ⑤ 성과 추적
            await self.track_performance(market, execution_results)
            
            logger.info(f"✅ {market} 매매 완료")
            
        except Exception as e:
            logger.error(f"❌ {market} 매매 실행 중 오류: {e}")

    async def fetch_market_data(self, market: str) -> Dict[str, MarketData]:
        """📊 시장 데이터 수집"""
        logger.info(f"📊 {market} 데이터 수집 중...")
        
        market_data = {}
        symbols = self._get_symbols(market)
        
        try:
            if market == "coin":
                # 암호화폐 시뮬레이션 데이터
                for symbol in symbols[:3]:
                    price = np.random.uniform(30000, 70000)  # BTC 가격대
                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        price=price,
                        volume=np.random.uniform(100, 1000),
                        timestamp=datetime.now(seoul_tz),
                        rsi=np.random.uniform(20, 80),
                        macd=np.random.uniform(-500, 500),
                        momentum=np.random.uniform(-0.05, 0.05),
                        volatility=np.random.uniform(0.02, 0.08)
                    )
            
            else:  # 주식 (japan, us)
                try:
                    import yfinance as yf
                    for symbol in symbols[:3]:
                        try:
                            ticker = yf.Ticker(symbol)
                            hist = ticker.history(period="5d")
                            
                            if not hist.empty:
                                latest = hist.iloc[-1]
                                prices = hist['Close'].values
                                
                                # 기술적 지표 계산
                                returns = np.diff(prices) / prices[:-1]
                                rsi = 50 + np.random.uniform(-20, 20)  # 간단한 RSI 시뮬레이션
                                
                                market_data[symbol] = MarketData(
                                    symbol=symbol,
                                    price=float(latest['Close']),
                                    volume=float(latest['Volume']),
                                    timestamp=datetime.now(seoul_tz),
                                    rsi=rsi,
                                    macd=np.random.uniform(-2, 2),
                                    momentum=returns[-1] if len(returns) > 0 else 0.0,
                                    volatility=np.std(returns) if len(returns) > 1 else 0.02
                                )
                                
                        except Exception as e:
                            logger.warning(f"⚠️ {symbol} 실제 데이터 로드 실패, 시뮬레이션 데이터 사용: {e}")
                            # 시뮬레이션 데이터
                            market_data[symbol] = MarketData(
                                symbol=symbol,
                                price=np.random.uniform(50, 300),
                                volume=np.random.uniform(1000000, 10000000),
                                timestamp=datetime.now(seoul_tz),
                                rsi=np.random.uniform(30, 70),
                                macd=np.random.uniform(-2, 2),
                                momentum=np.random.uniform(-0.03, 0.03),
                                volatility=np.random.uniform(0.01, 0.04)
                            )
                            
                except ImportError:
                    logger.warning("⚠️ yfinance 없음, 시뮬레이션 데이터 사용")
                    # 시뮬레이션 데이터만 사용
                    for symbol in symbols[:3]:
                        market_data[symbol] = MarketData(
                            symbol=symbol,
                            price=np.random.uniform(50, 300),
                            volume=np.random.uniform(1000000, 10000000),
                            timestamp=datetime.now(seoul_tz),
                            rsi=np.random.uniform(30, 70),
                            macd=np.random.uniform(-2, 2),
                            momentum=np.random.uniform(-0.03, 0.03),
                            volatility=np.random.uniform(0.01, 0.04)
                        )
            
            logger.info(f"✅ {market} 데이터 수집 완료: {len(market_data)}개 종목")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ {market} 데이터 수집 실패: {e}")
            return {}

    def _get_symbols(self, market: str) -> List[str]:
        """시장별 심볼 목록"""
        symbols_map = {
            'coin': ['BTC-KRW', 'ETH-KRW', 'XRP-KRW'],
            'japan': ['7203.T', '6758.T', '9984.T'],  # Toyota, Sony, SoftBank
            'us': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        }
        return symbols_map.get(market, [])

    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """📈 AI 기반 시그널 생성"""
        logger.info("🧠 AI 시그널 생성 중...")
        
        signals = []
        
        try:
            for symbol, data in market_data.items():
                # 다중 팩터 분석
                factors = {}
                
                # RSI 시그널
                if data.rsi < 30:
                    rsi_signal = (30 - data.rsi) / 30  # 과매도
                elif data.rsi > 70:
                    rsi_signal = -(data.rsi - 70) / 30  # 과매수
                else:
                    rsi_signal = 0
                
                factors['rsi'] = rsi_signal
                
                # MACD 시그널
                factors['macd'] = np.tanh(data.macd / 100)  # -1 to 1
                
                # 모멘텀 시그널
                factors['momentum'] = np.tanh(data.momentum * 20)
                
                # 변동성 시그널 (높은 변동성은 부정적)
                factors['volatility'] = -min(data.volatility * 20, 1.0)
                
                # 종합 시그널 계산
                weights = {'rsi': 0.3, 'macd': 0.3, 'momentum': 0.3, 'volatility': 0.1}
                total_signal = sum(factors[k] * weights[k] for k in factors.keys())
                
                # 시그널 결정
                if total_signal > 0.3:
                    action = "BUY"
                    confidence = min(0.9, 0.5 + abs(total_signal))
                elif total_signal < -0.3:
                    action = "SELL"
                    confidence = min(0.9, 0.5 + abs(total_signal))
                else:
                    action = "HOLD"
                    confidence = 0.5
                
                # 포지션 크기 계산
                max_position_size = 0.15  # 최대 15%
                position_size = max_position_size * confidence
                portfolio_value = 1000000  # 기본값
                
                quantity = (portfolio_value * position_size) / data.price
                
                # 스톱로스/익절 계산
                atr = data.volatility * data.price
                stop_loss = data.price - (2 * atr) if action == "BUY" else data.price + (2 * atr)
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    quantity=quantity,
                    target_price=data.price,
                    stop_loss=stop_loss if action != "HOLD" else None,
                    strategy="Multi_Factor_AI",
                    timestamp=datetime.now(seoul_tz),
                    signal_data=factors  # 🔧 metadata → signal_data
                )
                
                signals.append(signal)
            
            logger.info(f"✅ 시그널 생성 완료: {len(signals)}개")
            return signals
            
        except Exception as e:
            logger.error(f"❌ 시그널 생성 실패: {e}")
            return []

    def risk_management(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """🛡️ 리스크 관리"""
        logger.info("🛡️ 리스크 관리 중...")
        
        approved_signals = []
        
        try:
            for signal in signals:
                # 기본 리스크 체크
                if signal.confidence < 0.6:
                    logger.warning(f"🚫 {signal.symbol} 신뢰도 부족: {signal.confidence:.2f}")
                    continue
                
                # 포지션 크기 체크
                if signal.quantity * signal.target_price > 200000:  # 20만원 이상
                    logger.warning(f"🚫 {signal.symbol} 포지션 크기 초과")
                    continue
                
                # 변동성 체크 (signal_data에서 가져오기)
                volatility_signal = signal.signal_data.get('volatility', 0)
                if volatility_signal < -0.8:  # 너무 높은 변동성
                    logger.warning(f"🚫 {signal.symbol} 변동성 위험")
                    continue
                
                approved_signals.append(signal)
            
            logger.info(f"✅ 리스크 관리 완료: {len(approved_signals)}개 승인")
            return approved_signals
            
        except Exception as e:
            logger.error(f"❌ 리스크 관리 실패: {e}")
            return signals

    async def execute_orders(self, market: str, signals: List[TradingSignal]) -> List[Dict]:
        """📝 주문 실행"""
        logger.info(f"📝 {market} 주문 실행 중...")
        
        execution_results = []
        
        try:
            for signal in signals:
                if signal.action == "HOLD":
                    continue
                
                # 시뮬레이션 주문 실행
                result = {
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'quantity': signal.quantity,
                    'price': signal.target_price,
                    'total_value': signal.quantity * signal.target_price,
                    'timestamp': datetime.now(seoul_tz),
                    'status': 'SIMULATED'
                }
                
                execution_results.append(result)
                
                # 거래 기록 저장
                await self._save_trade_record(market, signal, result)
                
                logger.info(f"📝 {signal.symbol} {signal.action} 주문 완료: "
                          f"수량 {signal.quantity:.2f}, 가격 ₩{signal.target_price:,.0f}")
            
            logger.info(f"✅ {market} 주문 실행 완료: {len(execution_results)}개")
            return execution_results
            
        except Exception as e:
            logger.error(f"❌ {market} 주문 실행 실패: {e}")
            return []

    async def _save_trade_record(self, market: str, signal: TradingSignal, result: Dict):
        """거래 기록 저장"""
        try:
            if self.session:
                trade = TradeRecord(
                    timestamp=result['timestamp'],
                    market=market,
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=signal.quantity,
                    price=signal.target_price,
                    confidence=signal.confidence,
                    strategy=signal.strategy,
                    extra_data=json.dumps(signal.signal_data)  # 🔧 metadata → extra_data
                )
                
                self.session.add(trade)
                self.session.commit()
                
        except Exception as e:
            logger.error(f"❌ 거래 기록 저장 실패: {e}")
            if self.session:
                self.session.rollback()

    async def track_performance(self, market: str, execution_results: List[Dict]):
        """📊 성과 추적"""
        logger.info(f"📊 {market} 성과 추적 중...")
        
        try:
            if not execution_results:
                return
            
            total_trades = len(execution_results)
            total_value = sum(r['total_value'] for r in execution_results)
            
            logger.info(f"📊 {market} 성과 요약:")
            logger.info(f"   실행된 거래: {total_trades}개")
            logger.info(f"   총 거래금액: ₩{total_value:,.0f}")
            logger.info(f"   평균 거래금액: ₩{total_value/total_trades:,.0f}")
            
        except Exception as e:
            logger.error(f"❌ {market} 성과 추적 실패: {e}")

    def get_status(self) -> Dict:
        """시스템 상태 반환"""
        return {
            'portfolio': self.portfolio,
            'database_connected': self.session is not None,
            'timestamp': datetime.now(seoul_tz).isoformat()
        }

# 스케줄러 클래스도 수정
class InstitutionalTradingScheduler:
    """수정된 스케줄러"""
    
    def __init__(self, api_wrapper: QuantAPIWrapper, config: Dict = None):
        self.api = api_wrapper
        self.config = config or {}
        self.tasks = []
        self.running = False
        
        logger.info("📅 트레이딩 스케줄러 초기화 완료")
    
    def setup_default_tasks(self):
        """기본 작업 설정"""
        self.tasks = [
            {"market": "coin", "day": [0, 4], "time": "08:30"},   # 월, 금 08:30
            {"market": "japan", "day": [1, 3], "time": "10:00"},  # 화, 목 10:00
            {"market": "us", "day": [1, 3], "time": "22:30"},     # 화, 목 22:30
        ]
        logger.info(f"🔧 기본 작업 {len(self.tasks)}개 설정 완료")
    
    async def run(self):
        """스케줄러 실행"""
        if not self.tasks:
            self.setup_default_tasks()
        
        logger.info("🚀 스케줄러 시작 (서울 시간 기준)")
        self.running = True
        
        while self.running:
            try:
                now = datetime.now(seoul_tz)
                
                for task in self.tasks:
                    if now.weekday() in task["day"]:
                        target = datetime.strptime(task["time"], "%H:%M").time()
                        
                        if (now.time().hour == target.hour and 
                            now.time().minute == target.minute and
                            now.time().second < 30):
                            
                            logger.info(f"⏰ {task['market']} 매매 시간!")
                            await self.api.execute_trading(task["market"])
                
                await asyncio.sleep(60)  # 1분마다 체크
                
            except KeyboardInterrupt:
                logger.info("⌨️ 스케줄러 중지")
                break
            except Exception as e:
                logger.error(f"❌ 스케줄러 오류: {e}")
                await asyncio.sleep(10)
        
        self.running = False
    
    def get_status(self) -> Dict:
        """스케줄러 상태"""
        return {
            'running': self.running,
            'tasks': self.tasks,
            'current_time': datetime.now(seoul_tz).isoformat()
        }

# 메인 실행 함수
async def main():
    """메인 실행"""
    try:
        logger.info("🏆 Elite Trading System 시작")
        
        # API 래퍼 초기화
        config = {'initial_capital': 1000000}
        api = QuantAPIWrapper(config)
        
        # 상태 확인
        status = api.get_status()
        logger.info(f"📊 시스템 상태: {status}")
        
        # 스케줄러 초기화
        scheduler = InstitutionalTradingScheduler(api)
        
        # 테스트 실행
        print("\n" + "="*50)
        print("🏆 Elite Trading System 메뉴")
        print("="*50)
        print("1. 🪙 코인 매매 테스트")
        print("2. 🇯🇵 일본 주식 매매 테스트")
        print("3. 🇺🇸 미국 주식 매매 테스트")
        print("4. 🕐 자동 스케줄러 시작")
        print("5. 📊 시스템 상태 확인")
        print("0. 🚪 종료")
        
        while True:
            choice = input("\n선택하세요 (0-5): ").strip()
            
            if choice == '1':
                await api.execute_trading("coin")
            elif choice == '2':
                await api.execute_trading("japan")
            elif choice == '3':
                await api.execute_trading("us")
            elif choice == '4':
                logger.info("자동 스케줄러 시작 - Ctrl+C로 중지")
                await scheduler.run()
            elif choice == '5':
                status = api.get_status()
                print(f"📊 시스템 상태: {json.dumps(status, indent=2, ensure_ascii=False)}")
            elif choice == '0':
                logger.info("👋 시스템 종료")
                break
            else:
                print("올바른 번호를 선택하세요 (0-5)")
        
    except KeyboardInterrupt:
        logger.info("⌨️ 사용자에 의한 종료")
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")

if __name__ == "__main__":
    asyncio.run(main())
