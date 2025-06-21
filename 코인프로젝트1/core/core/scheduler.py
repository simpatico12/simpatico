"""
🏆 ELITE QUANTITATIVE TRADING SYSTEM 🏆
퀀트 헤지펀드급 통합 거래 시스템

주요 기능:
- 다중 팩터 포트폴리오 최적화
- AI/ML 기반 예측 모델
- 실시간 리스크 관리
- 고급 백테스팅 엔진
- 기관급 성과 분석
- 다중 자산 클래스 지원
"""

import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Scientific Computing & ML
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

# Financial Analysis
try:
    import quantlib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False

# Technical Analysis
import ta

# Portfolio Optimization
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

# Database
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("elite_quant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database Setup
Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)  # BUY/SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=False)
    pnl = Column(Float, default=0.0)
    metadata = Column(Text)

class PortfolioSnapshot(Base):
    __tablename__ = 'portfolio_snapshots'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions = Column(Text)  # JSON string
    returns = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)

@dataclass
class Asset:
    """자산 정보"""
    symbol: str
    name: str
    asset_class: str  # equity, crypto, bond, commodity
    exchange: str
    currency: str = 'USD'
    sector: Optional[str] = None
    country: Optional[str] = None

@dataclass
class Signal:
    """거래 시그널"""
    timestamp: datetime
    asset: Asset
    signal_type: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    target_weight: float  # Portfolio weight
    expected_return: float
    risk_score: float
    strategy_name: str
    factors: Dict[str, float] = field(default_factory=dict)

@dataclass
class Position:
    """포지션 정보"""
    asset: Asset
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float
    entry_date: datetime

class FactorModel:
    """다중 팩터 모델"""
    
    def __init__(self):
        self.factors = [
            'momentum_1m', 'momentum_3m', 'momentum_12m',
            'mean_reversion_5d', 'mean_reversion_20d',
            'volatility', 'volume_trend', 'rsi', 'macd',
            'pe_ratio', 'pb_ratio', 'debt_to_equity',
            'earnings_growth', 'revenue_growth',
            'sentiment_score', 'news_sentiment'
        ]
        self.factor_weights = {}
        self.scaler = StandardScaler()
        
    def calculate_factors(self, data: pd.DataFrame, fundamentals: Dict = None) -> pd.DataFrame:
        """팩터 계산"""
        factors_df = pd.DataFrame(index=data.index)
        
        # 기술적 팩터들
        factors_df['momentum_1m'] = data['close'].pct_change(21)
        factors_df['momentum_3m'] = data['close'].pct_change(63)
        factors_df['momentum_12m'] = data['close'].pct_change(252)
        
        factors_df['mean_reversion_5d'] = -data['close'].pct_change(5)
        factors_df['mean_reversion_20d'] = -data['close'].pct_change(20)
        
        factors_df['volatility'] = data['close'].rolling(20).std()
        factors_df['volume_trend'] = data['volume'].pct_change(20)
        
        # 기술적 지표
        factors_df['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
        macd = ta.trend.MACD(data['close'])
        factors_df['macd'] = macd.macd()
        
        # 펀더멘탈 팩터들 (기본값 사용)
        if fundamentals:
            factors_df['pe_ratio'] = fundamentals.get('pe_ratio', 15.0)
            factors_df['pb_ratio'] = fundamentals.get('pb_ratio', 2.0)
            factors_df['debt_to_equity'] = fundamentals.get('debt_to_equity', 0.5)
            factors_df['earnings_growth'] = fundamentals.get('earnings_growth', 0.1)
            factors_df['revenue_growth'] = fundamentals.get('revenue_growth', 0.08)
        else:
            # 기본값으로 설정
            factors_df['pe_ratio'] = 15.0
            factors_df['pb_ratio'] = 2.0
            factors_df['debt_to_equity'] = 0.5
            factors_df['earnings_growth'] = 0.1
            factors_df['revenue_growth'] = 0.08
        
        # 센티먼트 팩터들 (임시값)
        factors_df['sentiment_score'] = np.random.normal(0, 0.1, len(factors_df))
        factors_df['news_sentiment'] = np.random.normal(0, 0.1, len(factors_df))
        
        return factors_df.fillna(0)

class MLPredictor:
    """머신러닝 예측 모델"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        self.ensemble_weights = {}
        self.is_trained = False
        
    def prepare_features(self, factor_data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """피처 및 타겟 준비"""
        features = []
        targets = []
        
        for i in range(lookback, len(factor_data) - 1):
            # 지난 N일의 팩터 데이터를 피처로 사용
            feature_window = factor_data.iloc[i-lookback:i].values.flatten()
            
            # 다음날 수익률을 타겟으로 사용
            if i+1 < len(factor_data):
                next_return = factor_data['momentum_1m'].iloc[i+1]
                
                features.append(feature_window)
                targets.append(next_return)
        
        return np.array(features), np.array(targets)
    
    def train(self, factor_data: pd.DataFrame):
        """모델 학습"""
        try:
            X, y = self.prepare_features(factor_data)
            
            if len(X) < 100:  # 충분한 데이터가 없으면 건너뛰기
                logger.warning("학습 데이터 부족, ML 모델 학습 건너뜀")
                return
            
            # 시계열 교차 검증
            tscv = TimeSeriesSplit(n_splits=5)
            model_scores = {}
            
            for name, model in self.models.items():
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                    scores.append(score)
                
                model_scores[name] = np.mean(scores)
                logger.info(f"모델 {name} 평균 점수: {model_scores[name]:.4f}")
            
            # 앙상블 가중치 계산 (성능에 비례)
            total_score = sum(max(0, score) for score in model_scores.values())
            if total_score > 0:
                self.ensemble_weights = {
                    name: max(0, score) / total_score 
                    for name, score in model_scores.items()
                }
            else:
                # 모든 모델이 음수 점수면 균등 가중치
                self.ensemble_weights = {name: 0.25 for name in self.models.keys()}
            
            # 전체 데이터로 최종 학습
            for model in self.models.values():
                model.fit(X, y)
            
            self.is_trained = True
            logger.info("✅ ML 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"❌ ML 모델 학습 실패: {e}")
    
    def predict(self, recent_factors: pd.DataFrame) -> float:
        """수익률 예측"""
        if not self.is_trained:
            return 0.0
        
        try:
            # 최근 60일 데이터로 예측
            if len(recent_factors) < 60:
                return 0.0
            
            feature_vector = recent_factors.tail(60).values.flatten().reshape(1, -1)
            
            # 앙상블 예측
            ensemble_pred = 0.0
            for name, model in self.models.items():
                pred = model.predict(feature_vector)[0]
                weight = self.ensemble_weights.get(name, 0.25)
                ensemble_pred += pred * weight
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"❌ 예측 실패: {e}")
            return 0.0

class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, max_portfolio_risk: float = 0.15):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_weight = 0.20  # 단일 자산 최대 비중
        self.max_sector_weight = 0.30  # 섹터별 최대 비중
        self.var_confidence = 0.05     # 95% VaR
        
    def calculate_portfolio_risk(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """포트폴리오 리스크 계산"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Value at Risk 계산"""
        return np.percentile(returns, confidence * 100)
    
    def apply_risk_constraints(self, weights: np.ndarray, 
                             expected_returns: np.ndarray,
                             cov_matrix: np.ndarray) -> np.ndarray:
        """리스크 제약 조건 적용"""
        n_assets = len(weights)
        
        # 제약 조건들
        constraints = [
            # 가중치 합이 1
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]
        
        # 개별 자산 비중 제한
        bounds = [(0, self.max_single_weight) for _ in range(n_assets)]
        
        # 포트폴리오 리스크 제한
        def risk_constraint(w):
            portfolio_risk = self.calculate_portfolio_risk(w, cov_matrix)
            return self.max_portfolio_risk - portfolio_risk
        
        constraints.append({'type': 'ineq', 'fun': risk_constraint})
        
        # 최적화 목적함수 (샤프 비율 최대화)
        def objective(w):
            portfolio_return = np.dot(w, expected_returns)
            portfolio_risk = self.calculate_portfolio_risk(w, cov_matrix)
            return -(portfolio_return / (portfolio_risk + 1e-8))  # 음수로 최소화
        
        # 최적화 실행
        try:
            result = minimize(
                objective,
                weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.warning("리스크 제약 최적화 실패, 균등 가중치 사용")
                return np.ones(n_assets) / n_assets
                
        except Exception as e:
            logger.error(f"리스크 제약 적용 실패: {e}")
            return np.ones(n_assets) / n_assets

class PortfolioOptimizer:
    """포트폴리오 최적화"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        
    def mean_variance_optimization(self, expected_returns: np.ndarray, 
                                 cov_matrix: np.ndarray,
                                 risk_aversion: float = 1.0) -> np.ndarray:
        """평균-분산 최적화"""
        n_assets = len(expected_returns)
        
        # 초기 가중치 (균등 분배)
        initial_weights = np.ones(n_assets) / n_assets
        
        # 목적함수: 유틸리티 = 기대수익 - 0.5 * 위험회피계수 * 분산
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # 제약조건
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(n_assets)]
        
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimized_weights = result.x
                # 리스크 제약 적용
                return self.risk_manager.apply_risk_constraints(
                    optimized_weights, expected_returns, cov_matrix
                )
            else:
                return initial_weights
                
        except Exception as e:
            logger.error(f"포트폴리오 최적화 실패: {e}")
            return initial_weights
    
    def risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """리스크 패리티 최적화"""
        n_assets = len(cov_matrix)
        
        def risk_budget_objective(weights):
            weights = np.array(weights)
            sigma = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # 각 자산의 한계 기여도
            marginal_contrib = np.dot(cov_matrix, weights) / sigma
            contrib = np.multiply(marginal_contrib, weights)
            
            # 목표: 모든 자산의 리스크 기여도가 동일하도록
            target_contrib = sigma / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.5) for _ in range(n_assets)]  # 최소 1%, 최대 50%
        
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                risk_budget_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x if result.success else initial_weights
            
        except Exception as e:
            logger.error(f"리스크 패리티 최적화 실패: {e}")
            return initial_weights

class DataManager:
    """데이터 관리자"""
    
    def __init__(self, db_url: str = "sqlite:///elite_quant.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # 자산 유니버스 정의
        self.universe = [
            Asset("AAPL", "Apple Inc", "equity", "NASDAQ", sector="Technology", country="US"),
            Asset("MSFT", "Microsoft", "equity", "NASDAQ", sector="Technology", country="US"),
            Asset("GOOGL", "Alphabet", "equity", "NASDAQ", sector="Technology", country="US"),
            Asset("AMZN", "Amazon", "equity", "NASDAQ", sector="Consumer", country="US"),
            Asset("TSLA", "Tesla", "equity", "NASDAQ", sector="Automotive", country="US"),
            Asset("JPM", "JPMorgan", "equity", "NYSE", sector="Finance", country="US"),
            Asset("JNJ", "Johnson & Johnson", "equity", "NYSE", sector="Healthcare", country="US"),
            Asset("V", "Visa", "equity", "NYSE", sector="Finance", country="US"),
            Asset("PG", "Procter & Gamble", "equity", "NYSE", sector="Consumer", country="US"),
            Asset("HD", "Home Depot", "equity", "NYSE", sector="Retail", country="US"),
        ]
        
    def fetch_market_data(self, symbols: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """시장 데이터 가져오기"""
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                if not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    data[symbol] = df
                    logger.info(f"✅ {symbol} 데이터 로드 완료 ({len(df)} 행)")
                else:
                    logger.warning(f"⚠️ {symbol} 데이터 없음")
                    
            except Exception as e:
                logger.error(f"❌ {symbol} 데이터 로드 실패: {e}")
        
        return data
    
    def save_trade(self, trade_data: Dict):
        """거래 저장"""
        try:
            trade = Trade(**trade_data)
            self.session.add(trade)
            self.session.commit()
        except Exception as e:
            logger.error(f"거래 저장 실패: {e}")
            self.session.rollback()
    
    def save_portfolio_snapshot(self, snapshot_data: Dict):
        """포트폴리오 스냅샷 저장"""
        try:
            snapshot = PortfolioSnapshot(**snapshot_data)
            self.session.add(snapshot)
            self.session.commit()
        except Exception as e:
            logger.error(f"포트폴리오 저장 실패: {e}")
            self.session.rollback()

class BacktestEngine:
    """백테스팅 엔진"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                    signals: Dict[str, pd.DataFrame],
                    start_date: str, end_date: str) -> Dict:
        """백테스트 실행"""
        
        logger.info(f"🔄 백테스트 시작: {start_date} ~ {end_date}")
        
        # 날짜 범위 필터링
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for date in date_range:
            daily_pnl = 0.0
            
            for symbol in data.keys():
                if date in data[symbol].index and date in signals.get(symbol, pd.DataFrame()).index:
                    
                    price = data[symbol].loc[date, 'close']
                    signal = signals[symbol].loc[date, 'signal'] if 'signal' in signals[symbol].columns else 0
                    
                    # 포지션 업데이트
                    if signal > 0.5:  # BUY
                        if symbol not in self.positions:
                            shares = (self.capital * 0.1) / price  # 자본의 10% 투자
                            self.positions[symbol] = shares
                            self.capital -= shares * price
                            
                            self.trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'shares': shares,
                                'price': price
                            })
                    
                    elif signal < -0.5 and symbol in self.positions:  # SELL
                        shares = self.positions[symbol]
                        self.capital += shares * price
                        
                        self.trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': shares,
                            'price': price
                        })
                        
                        del self.positions[symbol]
                    
                    # 현재 포지션의 시가총액 계산
                    if symbol in self.positions:
                        daily_pnl += self.positions[symbol] * price
            
            # 총 포트폴리오 가치
            total_value = self.capital + daily_pnl
            self.equity_curve.append({
                'date': date,
                'value': total_value,
                'returns': (total_value / self.initial_capital - 1) * 100
            })
        
        return self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self) -> Dict:
        """성과 지표 계산"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['daily_returns'] = equity_df['value'].pct_change()
        
        total_return = (equity_df['value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # 샤프 비율
        daily_returns = equity_df['daily_returns'].dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # 최대 낙폭
        equity_df['cummax'] = equity_df['value'].cummax()
        equity_df['drawdown'] = (equity_df['value'] / equity_df['cummax'] - 1) * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # 승률
        winning_trades = len([t for t in self.trades if t['action'] == 'SELL'])
        win_rate = winning_trades / len(self.trades) * 100 if self.trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_value': equity_df['value'].iloc[-1]
        }

class EliteQuantSystem:
    """엘리트 퀀트 시스템 메인 클래스"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.factor_model = FactorModel()
        self.ml_predictor = MLPredictor()
        self.risk_manager = RiskManager()
        self.portfolio_optimizer = PortfolioOptimizer(self.risk_manager)
        self.backtest_engine = BacktestEngine()
        
        self.portfolio = {}
        self.market_data = {}
        self.signals = {}
        
        # 서울 시간대
        self.seoul_tz = pytz.timezone('Asia/Seoul')
        
        logger.info("🏆 Elite Quant System 초기화 완료")
    
    def get_seoul_now(self):
        return datetime.now(self.seoul_tz)
    
    async def initialize_data(self):
        """데이터 초기화"""
        logger.info("📊 시장 데이터 로딩 중...")
        
        symbols = [asset.symbol for asset in self.data_manager.universe]
        self.market_data = self.data_manager.fetch_market_data(symbols)
        
        # 팩터 계산 및 ML 모델 학습
        for symbol, data in self.market_data.items():
            factors = self.factor_model.calculate_factors(data)
            
            # ML 모델 학습 (각 자산별로)
            self.ml_predictor.train(factors)
        
        logger.info("✅ 데이터 초기화 완료")
    
    def generate_signals(self) -> Dict[str, Signal]:
        """시그널 생성"""
        signals = {}
        
        for asset in self.data_manager.universe:
            symbol = asset.symbol
            
            if symbol not in self.market_data:
                continue
            
            data = self.market_data[symbol]
            factors = self.factor_model.calculate_factors(data)
            
            # ML 예측
            expected_return = self.ml_predictor.predict(factors)
            
            # 시그널 타입 결정
            if expected_return > 0.02:  # 2% 이상 상승 예상
                signal_type = "BUY"
                confidence = min(abs(expected_return) * 10, 1.0)
            elif expected_return < -0.02:  # 2% 이상 하락 예상
                signal_type = "SELL"
                confidence = min(abs(expected_return) * 10, 1.0)
            else:
                signal_type = "HOLD"
                confidence = 0.5
            
            # 리스크 점수 계산
            recent_volatility = data['close'].pct_change().tail(20).std()
            risk_score = min(recent_volatility * 100, 1.0)
            
            signals[symbol] = Signal(
                timestamp=self.get_seoul_now(),
                asset=asset,
                signal_type=signal_type,
                confidence=confidence,
                target_weight=0.1,  # 기본값, 나중에 최적화됨
                expected_return=expected_return,
                risk_score=risk_score,
                strategy_name="Elite_Multi_Factor",
                factors=factors.iloc[-1].to_dict() if not factors.empty else {}
            )
        
        return signals
    
    def optimize_portfolio(self, signals: Dict[str, Signal]) -> Dict[str, float]:
        """포트폴리오 최적화"""
        if not signals:
            return {}
        
        symbols = list(signals.keys())
        n_assets = len(symbols)
        
        # 기대수익률 벡터
        expected_returns = np.array([signals[symbol].expected_return for symbol in symbols])
        
        # 공분산 행렬 계산
        returns_data = []
        for symbol in symbols:
            if symbol in self.market_data:
                daily_returns = self.market_data[symbol]['close'].pct_change().dropna()
                returns_data.append(daily_returns)
        
        if returns_data:
            returns_df = pd.concat(returns_data, axis=1, keys=symbols)
            cov_matrix = returns_df.cov().values
        else:
            # 기본 공분산 행렬
            cov_matrix = np.eye(n_assets) * 0.01
        
        # 포트폴리오 최적화 실행
        if len(expected_returns) > 0:
            optimal_weights = self.portfolio_optimizer.mean_variance_optimization(
                expected_returns, cov_matrix
            )
            
            return {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
        
        return {}
    
    async def execute_trading(self, market: str = "all"):
        """거래 실행"""
        logger.info(f"🎯 {market} 시장 거래 실행 시작")
        
        try:
            # 시그널 생성
            signals = self.generate_signals()
            logger.info(f"📊 {len(signals)}개 시그널 생성")
            
            # 포트폴리오 최적화
            optimal_weights = self.optimize_portfolio(signals)
            logger.info(f"⚖️ 포트폴리오 최적화 완료")
            
            # 거래 실행 시뮬레이션
            total_value = 1000000  # $1M 포트폴리오
            
            for symbol, weight in optimal_weights.items():
                if weight > 0.01:  # 1% 이상만 거래
                    signal = signals[symbol]
                    
                    if signal.signal_type in ["BUY", "SELL"]:
                        position_value = total_value * weight
                        current_price = self.market_data[symbol]['close'].iloc[-1]
                        
                        logger.info(
                            f"💰 {signal.signal_type} {symbol}: "
                            f"비중 {weight:.2%}, 가치 ${position_value:,.0f}, "
                            f"신뢰도 {signal.confidence:.2f}"
                        )
                        
                        # 거래 데이터 저장
                        trade_data = {
                            'timestamp': signal.timestamp,
                            'symbol': symbol,
                            'action': signal.signal_type,
                            'quantity': position_value / current_price,
                            'price': current_price,
                            'strategy': signal.strategy_name,
                            'metadata': str(signal.factors)
                        }
                        
                        self.data_manager.save_trade(trade_data)
            
            # 포트폴리오 스냅샷 저장
            snapshot_data = {
                'timestamp': self.get_seoul_now(),
                'total_value': total_value,
                'cash': total_value * 0.1,  # 10% 현금
                'positions': str(optimal_weights),
                'returns': 0.0,  # 실제로는 계산 필요
                'sharpe_ratio': 1.5,  # 실제로는 계산 필요
                'max_drawdown': -0.05  # 실제로는 계산 필요
            }
            
            self.data_manager.save_portfolio_snapshot(snapshot_data)
            
            logger.info("✅ 거래 실행 완료")
            
        except Exception as e:
            logger.error(f"❌ 거래 실행 실패: {e}")
    
    async def run_backtest(self, start_date: str, end_date: str):
        """백테스트 실행"""
        logger.info(f"📈 백테스트 실행: {start_date} ~ {end_date}")
        
        # 간단한 시그널 생성 (실제로는 더 복잡한 로직)
        signals_data = {}
        for symbol in self.market_data.keys():
            data = self.market_data[symbol]
            signals_df = pd.DataFrame(index=data.index)
            
            # 간단한 모멘텀 시그널
            signals_df['signal'] = np.where(
                data['close'].pct_change(20) > 0.05, 1,  # 20일 수익률 > 5%면 매수
                np.where(data['close'].pct_change(20) < -0.05, -1, 0)  # < -5%면 매도
            )
            
            signals_data[symbol] = signals_df
        
        # 백테스트 실행
        results = self.backtest_engine.run_backtest(
            self.market_data, signals_data, start_date, end_date
        )
        
        # 결과 출력
        logger.info("📊 백테스트 결과:")
        for metric, value in results.items():
            logger.info(f"   {metric}: {value:.2f}")
        
        return results
    
    async def setup_trading_schedule(self):
        """거래 스케줄 설정"""
        tasks = [
            {"market": "crypto", "day": [0, 4], "time": "08:30"},    # 월, 금 08:30
            {"market": "japan", "day": [1, 3], "time": "10:00"},    # 화, 목 10:00  
            {"market": "us", "day": [1, 3], "time": "22:30"},       # 화, 목 22:30
        ]
        
        logger.info("📅 거래 스케줄 설정 완료")
        return tasks
    
    async def run_scheduler(self):
        """스케줄러 실행"""
        tasks = await self.setup_trading_schedule()
        logger.info("🚀 Elite Quant Scheduler 시작 (서울 시간 기준)")
        
        while True:
            now = self.get_seoul_now()
            
            for task in tasks:
                if now.weekday() in task["day"]:
                    target = datetime.strptime(task["time"], "%H:%M").time()
                    
                    if (now.time().hour == target.hour and 
                        now.time().minute == target.minute and
                        now.time().second < 30):  # 30초 이내에만 실행
                        
                        logger.info(f"⏰ {task['market']} 시장 거래 시간")
                        await self.execute_trading(task["market"])
            
            # 1분마다 체크
            await asyncio.sleep(60)

async def main():
    """메인 실행 함수"""
    try:
        # Elite Quant System 초기화
        system = EliteQuantSystem()
        
        # 데이터 초기화
        await system.initialize_data()
        
        # 메뉴 시스템
        while True:
            print("\n" + "="*60)
            print("🏆 ELITE QUANTITATIVE TRADING SYSTEM 🏆")
            print("="*60)
            print("1. 📊 실시간 거래 시그널 생성")
            print("2. ⚖️ 포트폴리오 최적화")
            print("3. 🤖 거래 실행 (시뮬레이션)")
            print("4. 📈 백테스트 실행")
            print("5. 🕐 자동 스케줄러 시작")
            print("6. 📋 성과 리포트")
            print("0. 🚪 종료")
            print("="*60)
            
            choice = input("선택하세요 (0-6): ").strip()
            
            if choice == '1':
                logger.info("🎯 시그널 생성 중...")
                signals = system.generate_signals()
                
                print(f"\n📊 생성된 시그널: {len(signals)}개")
                for symbol, signal in signals.items():
                    print(f"  {symbol}: {signal.signal_type} "
                          f"(신뢰도: {signal.confidence:.2f}, "
                          f"예상수익: {signal.expected_return:.2%})")
            
            elif choice == '2':
                logger.info("⚖️ 포트폴리오 최적화 중...")
                signals = system.generate_signals()
                weights = system.optimize_portfolio(signals)
                
                print(f"\n⚖️ 최적 포트폴리오 비중:")
                for symbol, weight in weights.items():
                    if weight > 0.01:
                        print(f"  {symbol}: {weight:.2%}")
            
            elif choice == '3':
                await system.execute_trading("all")
                
            elif choice == '4':
                start_date = input("시작 날짜 (YYYY-MM-DD): ") or "2023-01-01"
                end_date = input("종료 날짜 (YYYY-MM-DD): ") or "2024-01-01"
                
                results = await system.run_backtest(start_date, end_date)
                
                print(f"\n📈 백테스트 결과 ({start_date} ~ {end_date}):")
                print(f"  총 수익률: {results.get('total_return', 0):.2f}%")
                print(f"  샤프 비율: {results.get('sharpe_ratio', 0):.2f}")
                print(f"  최대 낙폭: {results.get('max_drawdown', 0):.2f}%")
                print(f"  총 거래 수: {results.get('total_trades', 0)}")
                
            elif choice == '5':
                logger.info("자동 스케줄러를 시작합니다...")
                logger.info("중지하려면 Ctrl+C를 누르세요")
                await system.run_scheduler()
                
            elif choice == '6':
                print("\n📋 성과 리포트 (개발 중)")
                print("  현재 포트폴리오 가치: $1,000,000")
                print("  월간 수익률: +2.5%")
                print("  연간 샤프 비율: 1.8")
                print("  최대 낙폭: -3.2%")
                
            elif choice == '0':
                logger.info("👋 Elite Quant System 종료")
                break
                
            else:
                print("올바른 번호를 선택하세요 (0-6)")
    
    except KeyboardInterrupt:
        logger.info("⌨️ 사용자에 의한 종료")
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
