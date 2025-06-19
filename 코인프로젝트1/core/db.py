# db.py
"""
고급 데이터베이스 모듈 - 퀸트프로젝트 수준
트레이딩 데이터 관리 및 분석 기능 통합
"""
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Boolean, Index, desc, func, and_, or_
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# DB 설정
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///quant.db')

# SQLite 최적화 설정
if DATABASE_URL.startswith('sqlite'):
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        connect_args={
            'check_same_thread': False,
            'timeout': 30
        },
        poolclass=StaticPool
    )
else:
    engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TradeRecord(Base):
    """거래 기록 테이블"""
    __tablename__ = 'trades'
    
    # 기본 정보
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    asset_type = Column(String, nullable=False, index=True)
    asset = Column(String, nullable=False, index=True)
    
    # 거래 정보
    decision = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    executed_volume = Column(Float)
    fee = Column(Float, default=0)
    
    # 잔고 정보
    asset_balance = Column(Float)
    cash_balance = Column(Float)
    total_asset = Column(Float)
    
    # 가격 정보
    avg_price = Column(Float)
    now_price = Column(Float)
    profit_rate = Column(Float)
    
    # 추가 메타데이터
    fg_index = Column(Float)
    sentiment = Column(String)
    is_success = Column(Boolean, default=True)
    
    # 복합 인덱스
    __table_args__ = (
        Index('idx_asset_timestamp', 'asset', 'timestamp'),
        Index('idx_type_timestamp', 'asset_type', 'timestamp'),
    )


class MarketData(Base):
    """시장 데이터 테이블"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    asset_type = Column(String, nullable=False)
    asset = Column(String, nullable=False)
    
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    
    fg_index = Column(Float)
    sentiment = Column(String)
    
    __table_args__ = (
        Index('idx_market_asset_time', 'asset', 'timestamp'),
    )


class Performance(Base):
    """성과 분석 테이블"""
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)
    
    total_value = Column(Float, nullable=False)
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    win_rate = Column(Float)
    
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    
    best_asset = Column(String)
    worst_asset = Column(String)


# 테이블 생성
Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """DB 세션 컨텍스트 매니저"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    @staticmethod
    def save_trade(trade_data: Dict) -> TradeRecord:
        """거래 저장"""
        with get_db() as db:
            trade = TradeRecord(**trade_data)
            db.add(trade)
            db.commit()
            db.refresh(trade)
            return trade
    
    @staticmethod
    def get_recent_trades(asset_type: Optional[str] = None, 
                         days: int = 7) -> List[TradeRecord]:
        """최근 거래 조회"""
        with get_db() as db:
            query = db.query(TradeRecord)
            
            if asset_type:
                query = query.filter(TradeRecord.asset_type == asset_type)
            
            start_date = datetime.now() - timedelta(days=days)
            query = query.filter(TradeRecord.timestamp >= start_date)
            
            return query.order_by(desc(TradeRecord.timestamp)).all()
    
    @staticmethod
    def get_asset_performance(asset: str, days: int = 30) -> Dict:
        """종목별 성과 분석"""
        with get_db() as db:
            start_date = datetime.now() - timedelta(days=days)
            
            trades = db.query(TradeRecord).filter(
                and_(
                    TradeRecord.asset == asset,
                    TradeRecord.timestamp >= start_date
                )
            ).all()
            
            if not trades:
                return {}
            
            # 성과 계산
            total_trades = len(trades)
            win_trades = len([t for t in trades if t.profit_rate > 0])
            
            avg_profit = db.query(func.avg(TradeRecord.profit_rate)).filter(
                and_(
                    TradeRecord.asset == asset,
                    TradeRecord.timestamp >= start_date,
                    TradeRecord.profit_rate > 0
                )
            ).scalar() or 0
            
            avg_loss = db.query(func.avg(TradeRecord.profit_rate)).filter(
                and_(
                    TradeRecord.asset == asset,
                    TradeRecord.timestamp >= start_date,
                    TradeRecord.profit_rate < 0
                )
            ).scalar() or 0
            
            return {
                'asset': asset,
                'total_trades': total_trades,
                'win_rate': win_trades / total_trades * 100 if total_trades > 0 else 0,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_return': sum(t.profit_rate for t in trades if t.profit_rate)
            }
    
    @staticmethod
    def get_daily_summary(date: datetime = None) -> Dict:
        """일일 거래 요약"""
        if not date:
            date = datetime.now().date()
        
        with get_db() as db:
            start = datetime.combine(date, datetime.min.time())
            end = datetime.combine(date, datetime.max.time())
            
            trades = db.query(TradeRecord).filter(
                and_(
                    TradeRecord.timestamp >= start,
                    TradeRecord.timestamp <= end
                )
            ).all()
            
            if not trades:
                return {'date': date, 'trades': 0}
            
            return {
                'date': date,
                'trades': len(trades),
                'buy_count': len([t for t in trades if t.decision == 'buy']),
                'sell_count': len([t for t in trades if t.decision == 'sell']),
                'total_volume': sum(t.executed_volume or 0 for t in trades),
                'total_fee': sum(t.fee or 0 for t in trades),
                'assets': list(set(t.asset for t in trades))
            }
    
    @staticmethod
    def calculate_portfolio_metrics() -> Dict:
        """포트폴리오 지표 계산"""
        with get_db() as db:
            # 최근 30일 데이터
            start_date = datetime.now() - timedelta(days=30)
            
            records = db.query(TradeRecord).filter(
                TradeRecord.timestamp >= start_date
            ).order_by(TradeRecord.timestamp).all()
            
            if not records:
                return {}
            
            # 일별 수익률 계산
            daily_returns = []
            prev_value = records[0].total_asset
            
            for record in records[1:]:
                if record.total_asset and prev_value:
                    daily_return = (record.total_asset - prev_value) / prev_value
                    daily_returns.append(daily_return)
                    prev_value = record.total_asset
            
            if not daily_returns:
                return {}
            
            # 지표 계산
            import numpy as np
            returns_array = np.array(daily_returns)
            
            return {
                'total_return': (records[-1].total_asset - records[0].total_asset) / records[0].total_asset * 100,
                'volatility': np.std(returns_array) * np.sqrt(252) * 100,  # 연환산
                'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown([r.total_asset for r in records if r.total_asset]),
                'win_rate': len([r for r in records if r.profit_rate > 0]) / len(records) * 100
            }
    
    @staticmethod
    def _calculate_max_drawdown(values: List[float]) -> float:
        """최대 낙폭 계산"""
        if not values:
            return 0
        
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    @staticmethod
    def cleanup_old_records(days: int = 365):
        """오래된 레코드 정리"""
        with get_db() as db:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            deleted = db.query(TradeRecord).filter(
                TradeRecord.timestamp < cutoff_date
            ).delete()
            
            db.commit()
            return deleted


# 글로벌 인스턴스
db_manager = DatabaseManager()