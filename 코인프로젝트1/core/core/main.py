"""
퀀트 트레이딩 시스템 메인 실행 파일
- 시스템 초기화 및 구성요소 통합
- 실시간 모니터링 및 시그널 생성
- 포트폴리오 관리 및 리스크 모니터링
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List

# 로컬 모듈 import
try:
    from logger import get_logger, info, error, warning
    from api_wrapper import QuantAPIWrapper
    from scheduler import InstitutionalTradingScheduler
    
    logger = get_logger(__name__)
    info("📦 모든 모듈 import 성공!")
    
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("다음 명령어로 필요한 패키지를 설치하세요:")
    print("pip install ccxt pandas numpy ta aiohttp redis sqlalchemy psutil")
    sys.exit(1)

class QuantTradingSystem:
    """퀀트 트레이딩 시스템 메인 클래스"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.api_wrapper = None
        self.scheduler = None
        self.running = False
        
        info("🏗️ 퀀트 트레이딩 시스템 초기화 중...")
        
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT'],
            'timeframes': ['1h', '4h', '1d'],
            'sandbox': True,  # 프로덕션에서는 False로 변경
            'database_url': 'sqlite:///quant_data.db',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'monitoring_interval': 60,  # 1분
            'signal_confidence_threshold': 0.7,
            'max_position_size': 0.1,  # 포트폴리오의 10%
            'stop_loss_percentage': 0.05,  # 5% 손실 제한
            'take_profit_percentage': 0.15,  # 15% 수익 실현
            'portfolio': {
                'initial_balance': 10000,  # $10,000
                'positions': {}
            }
        }
    
    async def initialize(self):
        """시스템 구성요소 초기화"""
        try:
            info("🔧 시스템 구성요소 초기화 시작")
            
            # 1. API 래퍼 초기화
            self.api_wrapper = QuantAPIWrapper(self.config)
            info("✅ API 래퍼 초기화 완료")
            
            # 2. 시스템 상태 확인
            status = self.api_wrapper.get_status()
            info(f"📊 시스템 상태: {status}")
            
            # 3. 스케줄러 초기화
            self.scheduler = InstitutionalTradingScheduler(self.api_wrapper, self.config)
            info("✅ 스케줄러 초기화 완료")
            
            # 4. 초기 데이터 수집 테스트
            await self._test_data_collection()
            
            info("🎉 시스템 초기화 완료!")
            return True
            
        except Exception as e:
            error(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    async def _test_data_collection(self):
        """초기 데이터 수집 테스트"""
        try:
            info("🧪 데이터 수집 테스트 시작")
            
            test_symbol = self.config['symbols'][0]
            market_data = await self.api_wrapper.fetch_comprehensive_market_data(test_symbol)
            
            info(f"📈 {test_symbol} 테스트 데이터:")
            info(f"   가격: ${market_data.price:,.2f}")
            info(f"   RSI: {market_data.rsi:.2f}")
            info(f"   공포탐욕지수: {market_data.fear_greed_index}")
            
            # 시그널 생성 테스트
            signals = self.api_wrapper.generate_trading_signals(market_data)
            info(f"🎯 생성된 시그널 수: {len(signals)}")
            
            for signal in signals:
                if signal.confidence > 0.7:
                    info(f"   🔥 고신뢰도: {signal.signal_type} (신뢰도: {signal.confidence:.2f})")
            
        except Exception as e:
            warning(f"⚠️ 데이터 수집 테스트 중 에러: {e}")
    
    async def start_monitoring(self):
        """실시간 모니터링 시작"""
        if self.running:
            warning("⚠️ 모니터링이 이미 실행 중입니다")
            return
        
        self.running = True
        info("🚀 실시간 모니터링 시작")
        
        # 시그널 콜백 함수
        async def signal_callback(symbol, market_data, signals):
            await self._process_signals(symbol, market_data, signals)
        
        try:
            # 병렬로 모니터링과 스케줄러 실행
            await asyncio.gather(
                self.api_wrapper.run_live_monitoring(
                    self.config['symbols'], 
                    callback=signal_callback
                ),
                self.scheduler.start(),
                return_exceptions=True
            )
        except KeyboardInterrupt:
            info("⌨️ 사용자에 의한 중지")
        except Exception as e:
            error(f"❌ 모니터링 중 에러: {e}")
        finally:
            await self.stop()
    
    async def _process_signals(self, symbol: str, market_data, signals: List):
        """시그널 처리 및 거래 결정"""
        try:
            high_confidence_signals = [
                s for s in signals 
                if s.confidence >= self.config['signal_confidence_threshold']
            ]
            
            if not high_confidence_signals:
                return
            
            info(f"🔥 {symbol} 고신뢰도 시그널 {len(high_confidence_signals)}개 발견")
            
            for signal in high_confidence_signals:
                info(f"   📊 {signal.strategy_name}: {signal.signal_type}")
                info(f"      신뢰도: {signal.confidence:.2f}")
                info(f"      현재가: ${market_data.price:,.2f}")
                
                # 여기에 실제 거래 로직 추가
                await self._execute_trade_decision(symbol, signal, market_data)
                
        except Exception as e:
            error(f"❌ 시그널 처리 중 에러: {e}")
    
    async def _execute_trade_decision(self, symbol: str, signal, market_data):
        """거래 결정 실행 (시뮬레이션)"""
        try:
            # 현재는 시뮬레이션만 (실제 거래는 추가 구현 필요)
            position_size = self._calculate_position_size(symbol, signal.confidence)
            
            if signal.signal_type == 'BUY':
                info(f"💰 BUY 시그널 처리: {symbol}")
                info(f"   포지션 크기: {position_size:.4f}")
                info(f"   목표가: ${market_data.price * 1.15:,.2f}")
                info(f"   손절가: ${market_data.price * 0.95:,.2f}")
                
            elif signal.signal_type == 'SELL':
                info(f"💸 SELL 시그널 처리: {symbol}")
                info(f"   포지션 크기: {position_size:.4f}")
                
            # 포트폴리오 업데이트 (시뮬레이션)
            self._update_portfolio_simulation(symbol, signal, position_size)
            
        except Exception as e:
            error(f"❌ 거래 결정 실행 중 에러: {e}")
    
    def _calculate_position_size(self, symbol: str, confidence: float) -> float:
        """포지션 크기 계산"""
        base_size = self.config['max_position_size']
        
        # 신뢰도에 따른 포지션 크기 조정
        adjusted_size = base_size * confidence
        
        return min(adjusted_size, base_size)
    
    def _update_portfolio_simulation(self, symbol: str, signal, position_size: float):
        """포트폴리오 시뮬레이션 업데이트"""
        try:
            current_positions = self.config['portfolio']['positions']
            
            if signal.signal_type == 'BUY':
                current_positions[symbol] = current_positions.get(symbol, 0) + position_size
            elif signal.signal_type == 'SELL':
                current_positions[symbol] = max(0, current_positions.get(symbol, 0) - position_size)
            
            info(f"📊 포트폴리오 업데이트: {symbol} = {current_positions.get(symbol, 0):.4f}")
            
        except Exception as e:
            error(f"❌ 포트폴리오 업데이트 중 에러: {e}")
    
    async def stop(self):
        """시스템 중지"""
        self.running = False
        if self.scheduler:
            await self.scheduler.stop()
        info("🛑 시스템 중지 완료")
    
    def get_portfolio_status(self) -> Dict:
        """포트폴리오 상태 반환"""
        return {
            'positions': self.config['portfolio']['positions'],
            'initial_balance': self.config['portfolio']['initial_balance'],
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_backtest(self, start_date: str, end_date: str):
        """백테스팅 실행"""
        info(f"📈 백테스팅 시작: {start_date} ~ {end_date}")
        # 여기에 백테스팅 로직 추가
        warning("⚠️ 백테스팅 기능은 아직 구현되지 않았습니다")

async def main():
    """메인 실행 함수"""
    try:
        info("=" * 60)
        info("🚀 퀀트 트레이딩 시스템 시작")
        info("=" * 60)
        
        # 환경변수 확인
        required_env_vars = ['BINANCE_API_KEY', 'BINANCE_SECRET']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            warning(f"⚠️ 누락된 환경변수: {missing_vars}")
            warning("일부 기능이 제한될 수 있습니다.")
        
        # 시스템 초기화
        system = QuantTradingSystem()
        
        if not await system.initialize():
            error("❌ 시스템 초기화 실패")
            return
        
        # 메뉴 표시
        while True:
            print("\n" + "=" * 50)
            print("📊 퀀트 트레이딩 시스템 메뉴")
            print("=" * 50)
            print("1. 실시간 모니터링 시작")
            print("2. 포트폴리오 상태 확인")
            print("3. 시스템 상태 확인")
            print("4. 백테스팅 실행")
            print("5. 단일 시그널 테스트")
            print("0. 종료")
            print("=" * 50)
            
            try:
                choice = input("선택하세요 (0-5): ").strip()
                
                if choice == '1':
                    info("실시간 모니터링을 시작합니다...")
                    info("중지하려면 Ctrl+C를 누르세요")
                    await system.start_monitoring()
                    
                elif choice == '2':
                    portfolio = system.get_portfolio_status()
                    info("📊 현재 포트폴리오:")
                    for symbol, amount in portfolio['positions'].items():
                        info(f"   {symbol}: {amount:.4f}")
                    
                elif choice == '3':
                    if system.api_wrapper:
                        status = system.api_wrapper.get_status()
                        info("🔍 시스템 상태:")
                        info(f"   거래소 연결: {status['exchanges']}")
                        info(f"   데이터베이스: {status['database']}")
                        info(f"   캐시: {status['cache']}")
                    
                elif choice == '4':
                    start_date = input("시작 날짜 (YYYY-MM-DD): ")
                    end_date = input("종료 날짜 (YYYY-MM-DD): ")
                    await system.run_backtest(start_date, end_date)
                    
                elif choice == '5':
                    symbol = input("테스트할 심볼 (예: BTC/USDT): ") or 'BTC/USDT'
                    info(f"🧪 {symbol} 시그널 테스트 중...")
                    
                    market_data = await system.api_wrapper.fetch_comprehensive_market_data(symbol)
                    signals = system.api_wrapper.generate_trading_signals(market_data)
                    
                    info(f"📈 {symbol} 현재 상태:")
                    info(f"   가격: ${market_data.price:,.2f}")
                    info(f"   RSI: {market_data.rsi:.2f}")
                    info(f"   시그널 수: {len(signals)}")
                    
                    for signal in signals:
                        info(f"   📊 {signal.strategy_name}: {signal.signal_type} (신뢰도: {signal.confidence:.2f})")
                    
                elif choice == '0':
                    info("시스템을 종료합니다...")
                    await system.stop()
                    break
                    
                else:
                    warning("올바른 번호를 선택하세요 (0-5)")
                    
            except KeyboardInterrupt:
                info("⌨️ 메뉴로 돌아갑니다...")
                continue
            except Exception as e:
                error(f"❌ 실행 중 에러: {e}")
                continue
    
    except Exception as e:
        error(f"❌ 메인 실행 중 심각한 에러: {e}")
    finally:
        info("👋 프로그램을 종료합니다")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 사용자에 의한 종료")
    except Exception as e:
        print(f"❌ 프로그램 실행 중 에러: {e}")
        sys.exit(1)
