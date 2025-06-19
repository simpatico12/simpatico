# tests/conftest.py
"""
Pytest 설정 및 공통 픽스처
"""

import pytest
import asyncio
import os
from datetime import datetime
from unittest.mock import Mock

# 테스트 환경 설정
os.environ['TRADING_ENV'] = 'test'
os.environ['DATABASE_URL'] = 'sqlite:///test_quant.db'


@pytest.fixture(scope='session')
def event_loop():
    """비동기 테스트를 위한 이벤트 루프"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """테스트용 설정"""
    return {
        'api': {
            'access_key': 'test_access',
            'secret_key': 'test_secret'
        },
        'telegram': {
            'token': 'test_token',
            'chat_id': 'test_chat'
        },
        'trading': {
            'coin': {'percentage': 20},
            'stock': {'percentage': 30}
        }
    }


@pytest.fixture
def mock_upbit():
    """Upbit API 모킹"""
    mock = Mock()
    mock.get_balance.return_value = 1000000
    mock.get_balances.return_value = [
        {'currency': 'KRW', 'balance': '1000000'},
        {'currency': 'BTC', 'balance': '0.1', 'avg_buy_price': '50000000'}
    ]
    return mock


@pytest.fixture(autouse=True)
def cleanup_db():
    """테스트 후 DB 정리"""
    yield
    # 테스트 DB 정리
    if os.path.exists('test_quant.db'):
        os.remove('test_quant.db')


# pytest.ini
"""
[pytest]
minversion = 6.0
addopts = -ra -q --strict-markers
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    slow: 오래 걸리는 테스트
    integration: 통합 테스트
    performance: 성능 테스트
    backtest: 백테스트 관련
    consistency: 일관성 테스트

asyncio_mode = auto
"""

# tests/test_utils.py
"""테스트 유틸리티"""

import random
from datetime import datetime, timedelta


def generate_mock_trades(count=100):
    """모의 거래 데이터 생성"""
    trades = []
    
    for i in range(count):
        trade = {
            'timestamp': datetime.now() - timedelta(days=random.randint(1, 30)),
            'asset': random.choice(['BTC', 'ETH', 'AAPL', 'MSFT']),
            'asset_type': random.choice(['coin', 'stock']),
            'decision': random.choice(['buy', 'sell']),
            'confidence_score': random.randint(60, 95),
            'avg_price': random.uniform(1000, 100000),
            'profit_rate': random.uniform(-5, 10)
        }
        trades.append(trade)
    
    return trades


def generate_mock_market_data():
    """모의 시장 데이터 생성"""
    return {
        'price': random.uniform(10000, 100000),
        'volume': random.uniform(1000000, 10000000),
        'high': random.uniform(10500, 105000),
        'low': random.uniform(9500, 95000),
        'change_rate': random.uniform(-0.1, 0.1)
    }


# tests/test_performance.py
"""성능 벤치마크 테스트"""

import pytest
import time
import cProfile
import pstats
from io import StringIO


class TestPerformance:
    """성능 테스트 스위트"""
    
    @pytest.mark.performance
    def test_strategy_execution_time(self):
        """전략 실행 시간 벤치마크"""
        from strategy import analyze_coin
        
        iterations = 100
        start = time.time()
        
        for _ in range(iterations):
            analyze_coin("BTC")
        
        elapsed = time.time() - start
        avg_time = elapsed / iterations
        
        assert avg_time < 0.1, f"평균 실행 시간 초과: {avg_time:.3f}초"
        print(f"\n평균 실행 시간: {avg_time*1000:.1f}ms")
    
    @pytest.mark.performance
    def test_database_query_performance(self):
        """DB 쿼리 성능 테스트"""
        from db import db_manager
        from tests.test_utils import generate_mock_trades
        
        # 테스트 데이터 삽입
        trades = generate_mock_trades(1000)
        for trade in trades:
            db_manager.save_trade(trade)
        
        # 쿼리 성능 측정
        start = time.time()
        results = db_manager.get_recent_trades(days=30)
        query_time = time.time() - start
        
        assert query_time < 0.5, f"쿼리 시간 초과: {query_time:.3f}초"
        assert len(results) > 0
    
    @pytest.mark.performance
    def test_profiling(self):
        """코드 프로파일링"""
        from strategy import analyze_coin
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # 프로파일링할 코드
        for _ in range(10):
            analyze_coin("BTC")
        
        profiler.disable()
        
        # 결과 출력
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # 상위 10개
        
        print("\n=== 프로파일링 결과 ===")
        print(s.getvalue())