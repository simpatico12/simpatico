# tests/test_signals.py
"""
전략 시그널 테스트 - 퀸트프로젝트 수준
포괄적인 테스트와 성능 검증 포함
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
import numpy as np

from strategy import analyze_coin, analyze_japan, analyze_us
from exceptions import DataFetchError, TradingError


class TestStrategySignals:
    """전략 시그널 테스트 클래스"""
    
    @pytest.fixture
    def mock_market_data(self):
        """시장 데이터 모킹"""
        return {
            'price': 50000,
            'volume': 1000000,
            'change_rate': 0.05,
            'high': 52000,
            'low': 48000
        }
    
    @pytest.fixture
    def mock_news_data(self):
        """뉴스 데이터 모킹"""
        return [
            {'title': 'Bitcoin reaches new high', 'sentiment': 0.8},
            {'title': 'Market volatility concerns', 'sentiment': -0.3},
            {'title': 'Institutional adoption grows', 'sentiment': 0.6}
        ]
    
    # 기본 기능 테스트
    @pytest.mark.parametrize("coin,expected_keys", [
        ("BTC", {"decision", "confidence_score", "reason", "indicators"}),
        ("ETH", {"decision", "confidence_score", "reason", "indicators"}),
        ("XRP", {"decision", "confidence_score", "reason", "indicators"}),
    ])
    def test_analyze_coin_returns_complete_dict(self, coin, expected_keys):
        """코인 분석 결과 구조 검증"""
        with patch('strategy.get_market_data') as mock_market:
            mock_market.return_value = {'price': 50000, 'volume': 1000000}
            
            result = analyze_coin(coin)
            
            assert isinstance(result, dict)
            assert set(result.keys()) >= expected_keys
            assert result['decision'] in ['buy', 'sell', 'hold']
            assert 0 <= result['confidence_score'] <= 100
            assert isinstance(result['reason'], str)
            assert len(result['reason']) > 0
    
    @pytest.mark.parametrize("stock", ["7203.T", "6758.T", "9984.T"])
    def test_analyze_japan_with_market_conditions(self, stock, mock_market_data):
        """일본 주식 분석 - 시장 조건별"""
        test_scenarios = [
            {'fg_index': 10, 'sentiment': 'positive', 'expected': 'buy'},
            {'fg_index': 90, 'sentiment': 'negative', 'expected': 'sell'},
            {'fg_index': 50, 'sentiment': 'neutral', 'expected': 'hold'}
        ]
        
        for scenario in test_scenarios:
            with patch('strategy.get_fear_greed_index', return_value=scenario['fg_index']):
                with patch('strategy.get_news_sentiment', return_value=scenario['sentiment']):
                    result = analyze_japan(stock)
                    
                    # 기본 검증
                    assert isinstance(result, dict)
                    assert 'decision' in result
                    
                    # 시나리오별 검증 (전략에 따라 다를 수 있음)
                    if scenario['fg_index'] < 20 and scenario['sentiment'] == 'positive':
                        assert result['confidence_score'] >= 80
    
    @pytest.mark.parametrize("stock", ["AAPL", "MSFT", "GOOGL", "AMZN"])
    def test_analyze_us_edge_cases(self, stock):
        """미국 주식 분석 - 엣지 케이스"""
        edge_cases = [
            {'price': 0, 'volume': 0},  # 거래 정지
            {'price': 1000000, 'volume': 1},  # 비정상 가격
            {'price': None, 'volume': None},  # 데이터 없음
        ]
        
        for case in edge_cases:
            with patch('strategy.get_market_data', return_value=case):
                result = analyze_us(stock)
                
                # 에러 상황에서도 안전한 응답
                assert result['decision'] == 'hold'
                assert result['confidence_score'] < 50
                assert 'error' in result.get('reason', '').lower() or \
                       'insufficient' in result.get('reason', '').lower()
    
    # 성능 테스트
    @pytest.mark.performance
    def test_strategy_performance(self):
        """전략 실행 성능 테스트"""
        coins = ["BTC", "ETH", "XRP", "ADA", "DOT"]
        
        start_time = time.time()
        results = []
        
        for coin in coins:
            result = analyze_coin(coin)
            results.append(result)
        
        execution_time = time.time() - start_time
        
        # 성능 기준: 5개 코인 분석에 5초 이내
        assert execution_time < 5.0, f"성능 기준 초과: {execution_time:.2f}초"
        
        # 모든 결과가 유효한지 확인
        assert all(r['decision'] in ['buy', 'sell', 'hold'] for r in results)
    
    # 통합 테스트
    @pytest.mark.integration
    async def test_multi_asset_analysis(self):
        """다중 자산 동시 분석"""
        assets = {
            'coins': ['BTC', 'ETH'],
            'japan': ['7203.T', '6758.T'],
            'us': ['AAPL', 'MSFT']
        }
        
        async def analyze_async(analyze_func, asset):
            """비동기 래퍼"""
            return await asyncio.to_thread(analyze_func, asset)
        
        # 모든 자산 동시 분석
        tasks = []
        for coin in assets['coins']:
            tasks.append(analyze_async(analyze_coin, coin))
        for stock in assets['japan']:
            tasks.append(analyze_async(analyze_japan, stock))
        for stock in assets['us']:
            tasks.append(analyze_async(analyze_us, stock))
        
        results = await asyncio.gather(*tasks)
        
        # 모든 결과 검증
        assert len(results) == 6
        assert all(isinstance(r, dict) for r in results)
        assert all('decision' in r for r in results)
    
    # 에러 처리 테스트
    def test_handle_api_errors(self):
        """API 에러 처리"""
        with patch('strategy.get_market_data', side_effect=DataFetchError("API Error")):
            result = analyze_coin("BTC")
            
            assert result['decision'] == 'hold'
            assert result['confidence_score'] == 0
            assert 'error' in result['reason'].lower()
    
    def test_handle_calculation_errors(self):
        """계산 에러 처리"""
        with patch('strategy.calculate_indicators', side_effect=ValueError("Invalid data")):
            result = analyze_coin("BTC")
            
            assert result['decision'] == 'hold'
            assert result['confidence_score'] < 30
    
    # 일관성 테스트
    @pytest.mark.consistency
    def test_decision_consistency(self):
        """동일 조건에서 일관된 결정"""
        mock_data = {'price': 50000, 'volume': 1000000, 'fg_index': 30}
        
        with patch('strategy.get_market_data', return_value=mock_data):
            # 같은 조건에서 5번 실행
            results = [analyze_coin("BTC") for _ in range(5)]
            
            # 모든 결정이 동일해야 함
            decisions = [r['decision'] for r in results]
            assert len(set(decisions)) == 1, "동일 조건에서 다른 결정 발생"
            
            # 신뢰도 차이는 5% 이내
            confidences = [r['confidence_score'] for r in results]
            assert max(confidences) - min(confidences) <= 5
    
    # 백테스트 호환성
    @pytest.mark.backtest
    def test_strategy_backtest_compatibility(self):
        """백테스트 시스템과의 호환성"""
        # 백테스트에서 사용하는 형식으로 호출
        historical_data = {
            'asset': 'BTC',
            'fg': 25,
            'sentiment': 'positive',
            'price': 50000,
            'volume': 1000000
        }
        
        # 전략 함수가 백테스트 파라미터를 받을 수 있는지
        result = analyze_coin(**historical_data)
        
        assert isinstance(result, dict)
        assert all(key in result for key in ['decision', 'confidence_score'])


# tests/test_risk_management.py
"""리스크 관리 테스트"""

class TestRiskManagement:
    """리스크 관리 테스트"""
    
    @pytest.fixture
    def portfolio_state(self):
        """포트폴리오 상태"""
        return {
            'total_value': 10000000,
            'cash': 3000000,
            'positions': {
                'BTC': {'value': 3000000, 'ratio': 0.3},
                'ETH': {'value': 2000000, 'ratio': 0.2},
                'AAPL': {'value': 2000000, 'ratio': 0.2}
            }
        }
    
    def test_position_limit_check(self, portfolio_state):
        """포지션 한도 체크"""
        from core.risk import check_asset_ratio
        
        # 한도 내 포지션
        assert check_asset_ratio('XRP', 'coin', 1000, 
                               portfolio_state['total_value'],
                               portfolio_state['cash']) == True
        
        # 한도 초과 포지션
        assert check_asset_ratio('BTC', 'coin', 50000,
                               portfolio_state['total_value'],
                               1000000) == False
    
    def test_max_drawdown_calculation(self):
        """최대 낙폭 계산"""
        from backtest import BacktestEngine
        
        values = [100, 110, 105, 95, 90, 95, 100, 85, 90]
        engine = BacktestEngine()
        mdd = engine._calculate_max_drawdown(values)
        
        # 110에서 85로 하락 = 22.7% 낙폭
        assert 22 <= mdd <= 23
    
    @pytest.mark.parametrize("volatility,expected_action", [
        (0.1, "normal"),  # 낮은 변동성
        (0.3, "caution"),  # 중간 변동성
        (0.5, "reduce"),   # 높은 변동성
    ])
    def test_volatility_based_sizing(self, volatility, expected_action):
        """변동성 기반 포지션 크기 조정"""
        from core.risk import calculate_position_size
        
        base_size = 1000000
        adjusted_size = calculate_position_size(base_size, volatility)
        
        if expected_action == "normal":
            assert adjusted_size == base_size
        elif expected_action == "caution":
            assert adjusted_size < base_size
        elif expected_action == "reduce":
            assert adjusted_size < base_size * 0.5


# tests/test_integration.py
"""통합 테스트"""

class TestIntegration:
    """시스템 통합 테스트"""
    
    @pytest.mark.integration
    async def test_full_trading_cycle(self):
        """전체 거래 사이클 테스트"""
        from collector import collector
        from trader import trader
        
        # 1. 뉴스 수집
        await collector.collect(['BTC'], 'coin')
        
        # 2. 전략 실행
        result = analyze_coin('BTC')
        
        # 3. 거래 실행 (모의)
        with patch('trader.execute_buy_order', return_value={'status': 'success'}):
            await trader.execute_trade(
                'BTC', 'coin', 
                fg=30, 
                sentiment='positive'
            )
        
        # 4. 결과 검증
        assert result is not None
        assert 'decision' in result
    
    @pytest.mark.slow
    def test_database_operations(self):
        """데이터베이스 작업 테스트"""
        from db import db_manager, TradeRecord
        
        # 거래 저장
        trade_data = {
            'timestamp': datetime.now(),
            'asset_type': 'test',
            'asset': 'TEST',
            'decision': 'buy',
            'confidence_score': 85,
            'avg_price': 100
        }
        
        saved = db_manager.save_trade(trade_data)
        assert saved.id is not None
        
        # 조회
        trades = db_manager.get_recent_trades(days=1)
        assert len(trades) > 0
        
        # 정리 (테스트 데이터 삭제)
        with db_manager.get_db() as db:
            db.query(TradeRecord).filter_by(asset='TEST').delete()
            db.commit()


# Pytest 설정
if __name__ == "__main__":
    pytest.main([
        "-v",  # 상세 출력
        "--cov=.",  # 커버리지
        "--cov-report=html",  # HTML 리포트
        "-m", "not slow",  # slow 마크 제외
    ])