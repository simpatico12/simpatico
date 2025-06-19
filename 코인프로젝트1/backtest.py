# backtest.py
"""
고급 백테스트 모듈 - 퀸트프로젝트 수준
실전과 동일한 환경에서 전략 검증
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from logger import logger
from config import get_config


@dataclass
class Position:
    """포지션 정보"""
    asset: str
    quantity: float
    entry_price: float
    entry_time: datetime
    position_size: float
    
    def current_value(self, current_price: float) -> float:
        return self.quantity * current_price
    
    def profit_rate(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price * 100


@dataclass
class Trade:
    """거래 기록"""
    asset: str
    action: str  # 'buy' or 'sell'
    price: float
    quantity: float
    timestamp: datetime
    fee: float
    profit: float = 0.0
    profit_rate: float = 0.0


class BacktestEngine:
    """백테스트 엔진"""
    
    def __init__(self, initial_capital: float = 10_000_000):
        self.cfg = get_config()
        self.initial_capital = initial_capital
        self.reset()
        
    def reset(self):
        """엔진 초기화"""
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        
    def run_backtest(self, strategy_func: Callable, 
                    historical_data: pd.DataFrame,
                    asset_type: str = 'stock',
                    **kwargs) -> Dict:
        """
        백테스트 실행
        
        Parameters:
        - strategy_func: 전략 함수 (fg, sentiment 등을 받아 decision 반환)
        - historical_data: 과거 데이터 (columns: date, asset, price, volume, fg, sentiment)
        - asset_type: 자산 유형
        """
        self.reset()
        
        # 설정 로드
        fee_rate = self.cfg['backtest'].get('fee_rate', 0.0025)  # 0.25%
        slippage = self.cfg['backtest'].get('slippage', 0.001)   # 0.1%
        position_limit = self.cfg['trading'][asset_type]['percentage'] / 100
        
        # 데이터 정렬
        data = historical_data.sort_values('date').copy()
        
        # 일별 처리
        prev_value = self.initial_capital
        
        for date in data['date'].unique():
            daily_data = data[data['date'] == date]
            
            # 각 종목 분석
            for _, row in daily_data.iterrows():
                asset = row['asset']
                current_price = row['price']
                
                # 전략 실행
                try:
                    decision_data = strategy_func(
                        asset=asset,
                        fg=row.get('fg', 50),
                        sentiment=row.get('sentiment', 'neutral'),
                        price=current_price,
                        volume=row.get('volume', 0),
                        **kwargs
                    )
                except Exception as e:
                    logger.error(f"전략 실행 오류 {asset}: {e}")
                    continue
                
                decision = decision_data.get('decision', 'hold')
                confidence = decision_data.get('confidence', 50)
                
                # 매매 실행
                if decision == 'buy' and asset not in self.positions:
                    self._execute_buy(
                        asset, current_price, date, 
                        position_limit, fee_rate, slippage, confidence
                    )
                elif decision == 'sell' and asset in self.positions:
                    self._execute_sell(
                        asset, current_price, date,
                        fee_rate, slippage
                    )
            
            # 포트폴리오 가치 계산
            portfolio_value = self._calculate_portfolio_value(daily_data)
            self.portfolio_values.append((date, portfolio_value))
            
            # 일일 수익률
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
            prev_value = portfolio_value
        
        # 성과 지표 계산
        return self._calculate_performance_metrics()
    
    def _execute_buy(self, asset: str, price: float, timestamp: datetime,
                    position_limit: float, fee_rate: float, 
                    slippage: float, confidence: int):
        """매수 실행"""
        # 포지션 크기 계산 (신뢰도 반영)
        base_size = self.capital * position_limit
        position_size = base_size * (confidence / 100)
        
        # 슬리피지 적용
        actual_price = price * (1 + slippage)
        
        # 수량 계산
        quantity = position_size / actual_price
        
        # 수수료 계산
        fee = position_size * fee_rate
        total_cost = position_size + fee
        
        # 잔고 확인
        if total_cost > self.capital:
            return
        
        # 포지션 생성
        self.positions[asset] = Position(
            asset=asset,
            quantity=quantity,
            entry_price=actual_price,
            entry_time=timestamp,
            position_size=position_size
        )
        
        # 거래 기록
        self.trades.append(Trade(
            asset=asset,
            action='buy',
            price=actual_price,
            quantity=quantity,
            timestamp=timestamp,
            fee=fee
        ))
        
        # 잔고 차감
        self.capital -= total_cost
        
    def _execute_sell(self, asset: str, price: float, timestamp: datetime,
                     fee_rate: float, slippage: float):
        """매도 실행"""
        position = self.positions.get(asset)
        if not position:
            return
        
        # 슬리피지 적용
        actual_price = price * (1 - slippage)
        
        # 매도 금액
        sell_amount = position.quantity * actual_price
        fee = sell_amount * fee_rate
        net_amount = sell_amount - fee
        
        # 수익 계산
        profit = net_amount - position.position_size
        profit_rate = position.profit_rate(actual_price)
        
        # 거래 기록
        self.trades.append(Trade(
            asset=asset,
            action='sell',
            price=actual_price,
            quantity=position.quantity,
            timestamp=timestamp,
            fee=fee,
            profit=profit,
            profit_rate=profit_rate
        ))
        
        # 포지션 제거 및 잔고 추가
        del self.positions[asset]
        self.capital += net_amount
        
    def _calculate_portfolio_value(self, daily_data: pd.DataFrame) -> float:
        """포트폴리오 총 가치 계산"""
        total = self.capital
        
        # 보유 포지션 가치
        for asset, position in self.positions.items():
            current_price = daily_data[daily_data['asset'] == asset]['price'].iloc[0] \
                if asset in daily_data['asset'].values else position.entry_price
            total += position.current_value(current_price)
        
        return total
    
    def _calculate_performance_metrics(self) -> Dict:
        """성과 지표 계산"""
        if not self.portfolio_values:
            return {}
        
        # 기본 통계
        final_value = self.portfolio_values[-1][1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # 거래 통계
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.profit > 0]
        losing_trades = [t for t in self.trades if t.profit < 0]
        
        win_rate = len(winning_trades) / len([t for t in self.trades if t.action == 'sell']) * 100 \
            if any(t.action == 'sell' for t in self.trades) else 0
        
        # 수익률 통계
        returns_array = np.array(self.daily_returns)
        volatility = np.std(returns_array) * np.sqrt(252)  # 연환산
        
        # 샤프 비율 (무위험 이자율 2% 가정)
        risk_free_rate = 0.02 / 252
        excess_returns = returns_array - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) \
            if np.std(excess_returns) > 0 else 0
        
        # 최대 낙폭
        portfolio_values = [v[1] for v in self.portfolio_values]
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # 평균 보유 기간
        holding_periods = []
        for trade in self.trades:
            if trade.action == 'sell':
                # 매칭되는 매수 찾기
                buy_trade = next(
                    (t for t in reversed(self.trades) 
                     if t.asset == trade.asset and t.action == 'buy' 
                     and t.timestamp < trade.timestamp),
                    None
                )
                if buy_trade:
                    period = (trade.timestamp - buy_trade.timestamp).days
                    holding_periods.append(period)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        return {
            # 수익률
            'total_return': total_return,
            'annual_return': total_return / (len(self.portfolio_values) / 252) if len(self.portfolio_values) > 0 else 0,
            'volatility': volatility * 100,
            
            # 리스크
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            
            # 거래 통계
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': np.mean([t.profit_rate for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.profit_rate for t in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum(t.profit for t in winning_trades) / sum(t.profit for t in losing_trades)) \
                if losing_trades and sum(t.profit for t in losing_trades) != 0 else 0,
            
            # 기타
            'avg_holding_period': avg_holding_period,
            'final_capital': final_value,
            'total_fee': sum(t.fee for t in self.trades),
            
            # 상세 데이터
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'daily_returns': self.daily_returns
        }
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """최대 낙폭 계산"""
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def plot_results(self, metrics: Dict, save_path: str = None):
        """백테스트 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 포트폴리오 가치 추이
        dates = [v[0] for v in metrics['portfolio_values']]
        values = [v[1] for v in metrics['portfolio_values']]
        
        axes[0, 0].plot(dates, values, linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value (KRW)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 일일 수익률 분포
        axes[0, 1].hist(metrics['daily_returns'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Daily Returns Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. 누적 수익률
        cumulative_returns = np.cumprod(1 + np.array(metrics['daily_returns'])) - 1
        axes[1, 0].plot(dates[1:], cumulative_returns * 100, linewidth=2)
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 성과 요약
        summary_text = f"""
Total Return: {metrics['total_return']:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2f}%
Win Rate: {metrics['win_rate']:.2f}%
Total Trades: {metrics['total_trades']}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                       transform=axes[1, 1].transAxes, verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 글로벌 인스턴스
backtest_engine = BacktestEngine()

# 기존 인터페이스 호환
def run_backtest(strategy_func, historical_data, **kwargs):
    """백테스트 실행 (기존 인터페이스)"""
    return backtest_engine.run_backtest(strategy_func, historical_data, **kwargs)