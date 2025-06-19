"""
Advanced Database Management System for Quantitative Trading
===========================================================

퀀트 트레이딩을 위한 고급 데이터베이스 관리 시스템
거래 기록, 성과 분석, 백테스팅 데이터 등 모든 데이터를 체계적으로 관리

Features:
- 다중 데이터베이스 지원 (SQLite, PostgreSQL, MySQL)
- 고급 ORM 모델 및 관계 설정
- 실시간 성과 분석 및 집계
- 데이터 마이그레이션 및 백업 시스템
- 비동기 처리 및 커넥션 풀링
- 데이터 압축 및 파티셔닝
- 자동 인덱싱 및 쿼리 최적화

Author: Your Name
Version: 3.0.0
Created: 2025-06-18
"""

import os
import json
import asyncio
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import logging
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import gzip
import pickle

# Core 패키지 import
try:
    from core import BaseComponent, config, logger, ValidationError
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# SQLAlchemy imports
try:
    from sqlalchemy import (
        create_engine, Column, Integer, String, Float, DateTime, 
        Text, Boolean, Index, ForeignKey, func, and_, or_, desc, asc,
        event, BigInteger, Numeric, LargeBinary
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session, relationship, scoped_session
    from sqlalchemy.pool import StaticPool, QueuePool
    from sqlalchemy.exc import SQLAlchemyError, IntegrityError
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy.dialects.sqlite import INTEGER
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.error("SQLAlchemy가 설치되지 않았습니다. pip install sqlalchemy")

# 비동기 SQLAlchemy
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    ASYNC_SQLALCHEMY_AVAILABLE = True
except ImportError:
    ASYNC_SQLALCHEMY_AVAILABLE = False
    logger.warning("비동기 SQLAlchemy를 사용할 수 없습니다. pip install asyncpg")

# 기존 호환성을 위한 import
try:
    from db import TradeRecord as LegacyTradeRecord, Base as LegacyBase, engine as legacy_engine
    LEGACY_DB_AVAILABLE = True
except ImportError:
    LEGACY_DB_AVAILABLE = False
    logger.warning("기존 db.py를 찾을 수 없습니다. 독립 실행 모드")

# =============================================================================
# 상수 및 설정
# =============================================================================

class DatabaseType(Enum):
    """데이터베이스 유형"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"

class BackupFormat(Enum):
    """백업 형식"""
    SQL = "sql"
    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED = "compressed"

# 기본 설정
DEFAULT_DB_CONFIG = {
    'database_url': 'sqlite:///trading_data.db',
    'pool_size': 10,
    'max_overflow': 20,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'echo': False,  # SQL 로깅
    'backup_enabled': True,
    'backup_interval_hours': 24,
    'compression_enabled': True,
    'auto_vacuum': True,
    'performance_monitoring': True
}

# =============================================================================
# 고급 ORM 모델들
# =============================================================================

# 새로운 Base 선언
Base = declarative_base()

class TimestampMixin:
    """타임스탬프 믹스인"""
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class TradeRecord(Base, TimestampMixin):
    """고급 거래 기록 테이블"""
    __tablename__ = 'trade_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 기본 거래 정보
    timestamp = Column(DateTime, nullable=False, index=True)
    asset = Column(String(20), nullable=False, index=True)
    asset_type = Column(String(20), nullable=False, index=True)
    decision = Column(String(20), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    
    # 잔고 및 가격 정보
    asset_balance = Column(Numeric(20, 8), nullable=False)
    cash_balance = Column(Numeric(15, 2), nullable=False)
    avg_price = Column(Numeric(15, 2), nullable=False)
    now_price = Column(Numeric(15, 2), nullable=False)
    total_asset = Column(Numeric(15, 2), nullable=False)
    
    # 추가 고급 정보
    strategy_name = Column(String(50))
    market_conditions = Column(Text)  # JSON 형태
    risk_score = Column(Float)
    expected_return = Column(Float)
    stop_loss = Column(Numeric(15, 2))
    take_profit = Column(Numeric(15, 2))
    
    # 실행 정보
    execution_time_ms = Column(Float)  # 실행 시간 (밀리초)
    exchange = Column(String(20))
    order_id = Column(String(100))
    fees = Column(Numeric(10, 4))
    
    # 메타데이터
    session_id = Column(String(50))  # 거래 세션 ID
    notes = Column(Text)
    is_backtest = Column(Boolean, default=False)
    
    # 인덱스 정의
    __table_args__ = (
        Index('idx_asset_timestamp', 'asset', 'timestamp'),
        Index('idx_decision_confidence', 'decision', 'confidence_score'),
        Index('idx_total_asset_timestamp', 'total_asset', 'timestamp'),
    )

class PortfolioSnapshot(Base, TimestampMixin):
    """포트폴리오 스냅샷"""
    __tablename__ = 'portfolio_snapshots'
    
    id = Column(Integer, primary_key=True)
    snapshot_date = Column(DateTime, nullable=False, index=True)
    
    # 포트폴리오 가치
    total_value = Column(Numeric(15, 2), nullable=False)
    cash_balance = Column(Numeric(15, 2), nullable=False)
    invested_amount = Column(Numeric(15, 2), nullable=False)
    
    # 수익률 정보
    daily_return = Column(Float)
    daily_return_pct = Column(Float)
    total_return = Column(Float)
    total_return_pct = Column(Float)
    
    # 위험 지표
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    volatility = Column(Float)
    
    # 포트폴리오 구성
    positions_count = Column(Integer)
    diversification_score = Column(Float)
    concentration_ratio = Column(Float)  # 최대 보유 비중
    
    # 거래 통계
    trades_count_daily = Column(Integer)
    win_rate = Column(Float)
    avg_trade_return = Column(Float)
    
    # 메타데이터
    market_regime = Column(String(20))  # bull, bear, sideways
    economic_indicators = Column(Text)  # JSON 형태

class AssetPrice(Base, TimestampMixin):
    """자산 가격 이력"""
    __tablename__ = 'asset_prices'
    
    id = Column(Integer, primary_key=True)
    asset = Column(String(20), nullable=False, index=True)
    asset_type = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # OHLCV 데이터
    open_price = Column(Numeric(15, 2))
    high_price = Column(Numeric(15, 2))
    low_price = Column(Numeric(15, 2))
    close_price = Column(Numeric(15, 2), nullable=False)
    volume = Column(Numeric(20, 8))
    
    # 추가 지표
    market_cap = Column(Numeric(20, 2))
    change_24h = Column(Float)
    change_pct_24h = Column(Float)
    
    # 기술적 지표 (선택적)
    rsi = Column(Float)
    macd = Column(Float)
    ma_20 = Column(Numeric(15, 2))
    ma_50 = Column(Numeric(15, 2))
    bollinger_upper = Column(Numeric(15, 2))
    bollinger_lower = Column(Numeric(15, 2))
    
    # 메타데이터
    source = Column(String(20), default='api')
    quality_score = Column(Float, default=1.0)  # 데이터 품질 점수
    
    __table_args__ = (
        Index('idx_asset_timestamp_unique', 'asset', 'timestamp', unique=True),
        Index('idx_close_price_timestamp', 'close_price', 'timestamp'),
    )

class NewsRecord(Base, TimestampMixin):
    """뉴스 기록"""
    __tablename__ = 'news_records'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # 뉴스 정보
    title = Column(String(500), nullable=False)
    content = Column(Text)
    url = Column(String(1000))
    source = Column(String(50), nullable=False)
    
    # 관련 자산
    related_assets = Column(String(200))  # 쉼표로 구분된 자산 목록
    
    # 감성 분석
    sentiment_score = Column(Float)  # -1.0 ~ 1.0
    sentiment_label = Column(String(20))  # positive, negative, neutral
    confidence = Column(Float)
    keywords = Column(Text)  # JSON 형태
    
    # 영향도
    market_impact_score = Column(Float)
    relevance_score = Column(Float)
    
    # 언어 및 지역
    language = Column(String(5), default='ko')
    region = Column(String(5), default='KR')
    
    __table_args__ = (
        Index('idx_sentiment_timestamp', 'sentiment_score', 'timestamp'),
        Index('idx_related_assets', 'related_assets'),
    )

class BacktestResult(Base, TimestampMixin):
    """백테스트 결과"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    test_name = Column(String(100), nullable=False)
    strategy_name = Column(String(50), nullable=False)
    
    # 테스트 기간
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # 초기 설정
    initial_capital = Column(Numeric(15, 2), nullable=False)
    
    # 최종 결과
    final_value = Column(Numeric(15, 2), nullable=False)
    total_return = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=False)
    
    # 위험 지표
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_duration = Column(Integer)  # 일수
    volatility = Column(Float)
    
    # 거래 통계
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    avg_winning_trade = Column(Float)
    avg_losing_trade = Column(Float)
    largest_win = Column(Float)
    largest_loss = Column(Float)
    
    # 기간별 성과
    best_day = Column(Float)
    worst_day = Column(Float)
    best_month = Column(Float)
    worst_month = Column(Float)
    
    # 메타데이터
    parameters = Column(Text)  # JSON 형태 전략 파라미터
    market_conditions = Column(Text)  # JSON 형태
    notes = Column(Text)

class SystemMetrics(Base, TimestampMixin):
    """시스템 성능 지표"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # 데이터베이스 성능
    db_connection_count = Column(Integer)
    avg_query_time_ms = Column(Float)
    slow_queries_count = Column(Integer)
    db_size_mb = Column(Float)
    
    # 시스템 리소스
    cpu_usage_pct = Column(Float)
    memory_usage_mb = Column(Float)
    disk_usage_pct = Column(Float)
    
    # 거래 시스템 지표
    api_response_time_ms = Column(Float)
    order_success_rate = Column(Float)
    data_freshness_score = Column(Float)
    
    # 오류 및 경고
    error_count = Column(Integer)
    warning_count = Column(Integer)
    critical_events = Column(Text)  # JSON 형태

# =============================================================================
# 데이터베이스 관리자
# =============================================================================

class DatabaseManager(BaseComponent):
    """고급 데이터베이스 관리자"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        super().__init__("DatabaseManager")
        self.config_dict = config_dict or DEFAULT_DB_CONFIG
        
        # 데이터베이스 연결
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        
        # 백업 관리
        self.backup_thread = None
        self.backup_enabled = self.config_dict.get('backup_enabled', True)
        
        # 성능 모니터링
        self.performance_metrics = {}
        self.query_stats = {}
        
        # 캐시
        self.result_cache = {}
        self.cache_ttl = 300  # 5분
    
    def _do_initialize(self):
        """데이터베이스 관리자 초기화"""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy가 필요합니다")
        
        # 엔진 생성
        self._create_engines()
        
        # 세션 팩토리 생성
        self._create_session_factories()
        
        # 테이블 생성
        self._create_tables()
        
        # 기존 데이터 마이그레이션
        if LEGACY_DB_AVAILABLE:
            self._migrate_legacy_data()
        
        # 백업 스케줄러 시작
        if self.backup_enabled:
            self._start_backup_scheduler()
        
        # 성능 모니터링 시작
        self._setup_performance_monitoring()
        
        self.logger.info("고급 데이터베이스 관리자 초기화 완료")
    
    def _create_engines(self):
        """데이터베이스 엔진 생성"""
        database_url = self.config_dict.get('database_url', DEFAULT_DB_CONFIG['database_url'])
        
        # 동기 엔진
        if database_url.startswith('sqlite'):
            # SQLite 최적화 설정
            self.engine = create_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={
                    'check_same_thread': False,
                    'timeout': 20,
                    'isolation_level': None
                },
                echo=self.config_dict.get('echo', False)
            )
            
            # SQLite 성능 최적화
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                cursor.close()
        
        else:
            # PostgreSQL/MySQL 설정
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.config_dict.get('pool_size', 10),
                max_overflow=self.config_dict.get('max_overflow', 20),
                pool_timeout=self.config_dict.get('pool_timeout', 30),
                pool_recycle=self.config_dict.get('pool_recycle', 3600),
                echo=self.config_dict.get('echo', False)
            )
        
        # 비동기 엔진 (가능한 경우)
        if ASYNC_SQLALCHEMY_AVAILABLE:
            async_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
            if async_url != database_url:  # PostgreSQL인 경우만
                try:
                    self.async_engine = create_async_engine(async_url)
                except Exception as e:
                    self.logger.warning(f"비동기 엔진 생성 실패: {e}")
    
    def _create_session_factories(self):
        """세션 팩토리 생성"""
        self.SessionLocal = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        )
        
        if self.async_engine:
            self.AsyncSessionLocal = async_sessionmaker(
                self.async_engine, class_=AsyncSession, expire_on_commit=False
            )
    
    def _create_tables(self):
        """테이블 생성"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("데이터베이스 테이블 생성/업데이트 완료")
        except Exception as e:
            self.logger.error(f"테이블 생성 실패: {e}")
            raise
    
    def _migrate_legacy_data(self):
        """기존 데이터 마이그레이션"""
        try:
            if not LEGACY_DB_AVAILABLE:
                return
            
            # 기존 데이터가 있는지 확인
            legacy_session = sessionmaker(bind=legacy_engine)()
            legacy_records = legacy_session.query(LegacyTradeRecord).all()
            
            if not legacy_records:
                legacy_session.close()
                return
            
            # 새로운 형식으로 마이그레이션
            migrated_count = 0
            with self.get_session() as session:
                for legacy_record in legacy_records:
                    # 이미 마이그레이션된 데이터인지 확인
                    existing = session.query(TradeRecord).filter(
                        TradeRecord.timestamp == legacy_record.timestamp,
                        TradeRecord.asset == legacy_record.asset
                    ).first()
                    
                    if existing:
                        continue
                    
                    # 새로운 레코드 생성
                    new_record = TradeRecord(
                        timestamp=legacy_record.timestamp,
                        asset=legacy_record.asset,
                        asset_type=legacy_record.asset_type,
                        decision=legacy_record.decision,
                        confidence_score=legacy_record.confidence_score,
                        asset_balance=Decimal(str(legacy_record.asset_balance)),
                        cash_balance=Decimal(str(legacy_record.cash_balance)),
                        avg_price=Decimal(str(legacy_record.avg_price)),
                        now_price=Decimal(str(legacy_record.now_price)),
                        total_asset=Decimal(str(legacy_record.total_asset)),
                        session_id="migrated",
                        notes="Migrated from legacy database"
                    )
                    
                    session.add(new_record)
                    migrated_count += 1
                
                session.commit()
            
            legacy_session.close()
            
            if migrated_count > 0:
                self.logger.info(f"기존 데이터 마이그레이션 완료: {migrated_count}건")
        
        except Exception as e:
            self.logger.error(f"데이터 마이그레이션 실패: {e}")
    
    def _start_backup_scheduler(self):
        """백업 스케줄러 시작"""
        def backup_worker():
            while self.backup_enabled:
                try:
                    self.create_backup()
                    interval_hours = self.config_dict.get('backup_interval_hours', 24)
                    threading.Event().wait(interval_hours * 3600)  # 시간을 초로 변환
                except Exception as e:
                    self.logger.error(f"백업 스케줄러 오류: {e}")
                    threading.Event().wait(3600)  # 1시간 후 재시도
        
        self.backup_thread = threading.Thread(target=backup_worker, daemon=True)
        self.backup_thread.start()
        self.logger.info("백업 스케줄러 시작")
    
    def _setup_performance_monitoring(self):
        """성능 모니터링 설정"""
        if not self.config_dict.get('performance_monitoring', True):
            return
        
        # 쿼리 실행 시간 모니터링
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = datetime.now()
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if hasattr(context, '_query_start_time'):
                total_time = (datetime.now() - context._query_start_time).total_seconds() * 1000
                
                # 통계 업데이트
                if 'total_queries' not in self.query_stats:
                    self.query_stats['total_queries'] = 0
                    self.query_stats['total_time'] = 0
                    self.query_stats['slow_queries'] = 0
                
                self.query_stats['total_queries'] += 1
                self.query_stats['total_time'] += total_time
                
                if total_time > 1000:  # 1초 이상은 느린 쿼리
                    self.query_stats['slow_queries'] += 1
                    self.logger.warning(f"느린 쿼리 감지: {total_time:.2f}ms")
    
    # =============================================================================
    # 컨텍스트 관리자
    # =============================================================================
    
    @contextmanager
    def get_session(self):
        """세션 컨텍스트 매니저"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"데이터베이스 세션 오류: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self):
        """비동기 세션 컨텍스트 매니저"""
        if not self.AsyncSessionLocal:
            raise RuntimeError("비동기 세션을 사용할 수 없습니다")
        
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self.logger.error(f"비동기 데이터베이스 세션 오류: {e}")
                raise
    
    # =============================================================================
    # 거래 기록 관리 (기존 호환성 + 고급 기능)
    # =============================================================================
    
    def save_trade(self, asset: str, asset_type: str, decision: str, confidence: float,
                   asset_balance: float, cash_balance: float, avg_price: float,
                   now_price: float, total_asset: float, **kwargs) -> int:
        """
        거래 기록 저장 (기존 호환성 + 고급 기능)
        
        Returns:
            int: 저장된 레코드 ID
        """
        try:
            with self.get_session() as session:
                record = TradeRecord(
                    timestamp=kwargs.get('timestamp', datetime.now()),
                    asset=asset,
                    asset_type=asset_type,
                    decision=decision,
                    confidence_score=confidence,
                    asset_balance=Decimal(str(asset_balance)),
                    cash_balance=Decimal(str(cash_balance)),
                    avg_price=Decimal(str(avg_price)),
                    now_price=Decimal(str(now_price)),
                    total_asset=Decimal(str(total_asset)),
                    
                    # 고급 필드들
                    strategy_name=kwargs.get('strategy_name'),
                    market_conditions=json.dumps(kwargs.get('market_conditions', {})),
                    risk_score=kwargs.get('risk_score'),
                    expected_return=kwargs.get('expected_return'),
                    stop_loss=Decimal(str(kwargs.get('stop_loss', 0))) if kwargs.get('stop_loss') else None,
                    take_profit=Decimal(str(kwargs.get('take_profit', 0))) if kwargs.get('take_profit') else None,
                    execution_time_ms=kwargs.get('execution_time_ms'),
                    exchange=kwargs.get('exchange'),
                    order_id=kwargs.get('order_id'),
                    fees=Decimal(str(kwargs.get('fees', 0))) if kwargs.get('fees') else None,
                    session_id=kwargs.get('session_id'),
                    notes=kwargs.get('notes'),
                    is_backtest=kwargs.get('is_backtest', False)
                )
                
                session.add(record)
                session.flush()  # ID를 얻기 위해
                record_id = record.id
                
                self.logger.debug(f"거래 기록 저장: {asset} {decision} (ID: {record_id})")
                return record_id
                
        except Exception as e:
            self.logger.error(f"거래 기록 저장 실패: {e}")
            raise
    
    def get_all_trades(self, limit: Optional[int] = None, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      asset: Optional[str] = None) -> List[TradeRecord]:
        """
        거래 기록 조회 (기존 호환성 + 필터링)
        """
        try:
            with self.get_session() as session:
                query = session.query(TradeRecord)
                
                # 필터 적용
                if start_date:
                    query = query.filter(TradeRecord.timestamp >= start_date)
                if end_date:
                    query = query.filter(TradeRecord.timestamp <= end_date)
                if asset:
                    query = query.filter(TradeRecord.asset == asset)
                
                # 정렬 및 제한
                query = query.order_by(desc(TradeRecord.timestamp))
                if limit:
                    query = query.limit(limit)
                
                return query.all()
                
        except Exception as e:
            self.logger.error(f"거래 기록 조회 실패: {e}")
            return []
    
    def get_trade_statistics(self, days: int = 30, asset: Optional[str] = None) -> Dict[str, Any]:
        """거래 통계 조회"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                query = session.query(TradeRecord).filter(TradeRecord.timestamp >= start_date)
                
                if asset:
                    query = query.filter(TradeRecord.asset == asset)
                
                trades = query.all()
                
                if not trades:
                    return {}
                
                # 기본 통계
                total_trades = len(trades)
                buy_trades = len([t for t in trades if 'buy' in t.decision.lower()])
                sell_trades = len([t for t in trades if 'sell' in t.decision.lower()])
                hold_trades = total_trades - buy_trades - sell_trades
                
                  # 신뢰도 통계
                confidences = [t.confidence_score for t in trades]
                avg_confidence = sum(confidences) / len(confidences)
                max_confidence = max(confidences)
                min_confidence = min(confidences)
                
                # 자산 가치 변화
                first_trade = min(trades, key=lambda x: x.timestamp)
                last_trade = max(trades, key=lambda x: x.timestamp)
                
                total_return = float(last_trade.total_asset - first_trade.total_asset)
                total_return_pct = (total_return / float(first_trade.total_asset)) * 100 if first_trade.total_asset > 0 else 0
                
                # 거래 빈도
                days_with_trades = len(set(t.timestamp.date() for t in trades))
                avg_trades_per_day = total_trades / days_with_trades if days_with_trades > 0 else 0
                
                return {
                    'period_days': days,
                    'total_trades': total_trades,
                    'buy_trades': buy_trades,
                    'sell_trades': sell_trades,
                    'hold_trades': hold_trades,
                    'avg_confidence': avg_confidence,
                    'max_confidence': max_confidence,
                    'min_confidence': min_confidence,
                    'initial_value': float(first_trade.total_asset),
                    'final_value': float(last_trade.total_asset),
                    'total_return': total_return,
                    'total_return_pct': total_return_pct,
                    'days_with_trades': days_with_trades,
                    'avg_trades_per_day': avg_trades_per_day,
                    'first_trade_date': first_trade.timestamp.isoformat(),
                    'last_trade_date': last_trade.timestamp.isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"거래 통계 조회 실패: {e}")
            return {}
    
    # =============================================================================
    # 포트폴리오 스냅샷 관리
    # =============================================================================
    
    def save_portfolio_snapshot(self, portfolio_data: Dict[str, Any]) -> int:
        """포트폴리오 스냅샷 저장"""
        try:
            with self.get_session() as session:
                snapshot = PortfolioSnapshot(
                    snapshot_date=portfolio_data.get('snapshot_date', datetime.now()),
                    total_value=Decimal(str(portfolio_data['total_value'])),
                    cash_balance=Decimal(str(portfolio_data.get('cash_balance', 0))),
                    invested_amount=Decimal(str(portfolio_data.get('invested_amount', 0))),
                    daily_return=portfolio_data.get('daily_return'),
                    daily_return_pct=portfolio_data.get('daily_return_pct'),
                    total_return=portfolio_data.get('total_return'),
                    total_return_pct=portfolio_data.get('total_return_pct'),
                    sharpe_ratio=portfolio_data.get('sharpe_ratio'),
                    max_drawdown=portfolio_data.get('max_drawdown'),
                    volatility=portfolio_data.get('volatility'),
                    positions_count=portfolio_data.get('positions_count', 0),
                    diversification_score=portfolio_data.get('diversification_score'),
                    concentration_ratio=portfolio_data.get('concentration_ratio'),
                    trades_count_daily=portfolio_data.get('trades_count_daily', 0),
                    win_rate=portfolio_data.get('win_rate'),
                    avg_trade_return=portfolio_data.get('avg_trade_return'),
                    market_regime=portfolio_data.get('market_regime'),
                    economic_indicators=json.dumps(portfolio_data.get('economic_indicators', {}))
                )
                
                session.add(snapshot)
                session.flush()
                snapshot_id = snapshot.id
                
                self.logger.debug(f"포트폴리오 스냅샷 저장: {snapshot_id}")
                return snapshot_id
                
        except Exception as e:
            self.logger.error(f"포트폴리오 스냅샷 저장 실패: {e}")
            raise
    
    def get_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        """포트폴리오 성과 분석"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                snapshots = session.query(PortfolioSnapshot)\
                    .filter(PortfolioSnapshot.snapshot_date >= start_date)\
                    .order_by(PortfolioSnapshot.snapshot_date).all()
                
                if not snapshots:
                    return {}
                
                # 기본 성과 지표
                first_snapshot = snapshots[0]
                last_snapshot = snapshots[-1]
                
                total_return = float(last_snapshot.total_value - first_snapshot.total_value)
                total_return_pct = (total_return / float(first_snapshot.total_value)) * 100 if first_snapshot.total_value > 0 else 0
                
                # 일일 수익률 계산
                daily_returns = []
                for i in range(1, len(snapshots)):
                    prev_value = float(snapshots[i-1].total_value)
                    curr_value = float(snapshots[i].total_value)
                    if prev_value > 0:
                        daily_return = (curr_value - prev_value) / prev_value
                        daily_returns.append(daily_return)
                
                # 통계 계산
                if daily_returns:
                    avg_daily_return = sum(daily_returns) / len(daily_returns)
                    volatility = (sum((r - avg_daily_return) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
                    
                    # 샤프 비율 (무위험 수익률 0% 가정)
                    sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
                    
                    # 최대 낙폭
                    max_value = float(first_snapshot.total_value)
                    max_drawdown = 0
                    for snapshot in snapshots:
                        current_value = float(snapshot.total_value)
                        if current_value > max_value:
                            max_value = current_value
                        else:
                            drawdown = (max_value - current_value) / max_value
                            max_drawdown = max(max_drawdown, drawdown)
                else:
                    avg_daily_return = 0
                    volatility = 0
                    sharpe_ratio = 0
                    max_drawdown = 0
                
                # 승률 계산
                winning_days = len([r for r in daily_returns if r > 0])
                win_rate = winning_days / len(daily_returns) if daily_returns else 0
                
                return {
                    'period_days': days,
                    'snapshots_count': len(snapshots),
                    'initial_value': float(first_snapshot.total_value),
                    'final_value': float(last_snapshot.total_value),
                    'total_return': total_return,
                    'total_return_pct': total_return_pct,
                    'avg_daily_return': avg_daily_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'avg_positions': sum(s.positions_count or 0 for s in snapshots) / len(snapshots),
                    'avg_diversification': sum(s.diversification_score or 0 for s in snapshots) / len(snapshots),
                    'first_snapshot_date': first_snapshot.snapshot_date.isoformat(),
                    'last_snapshot_date': last_snapshot.snapshot_date.isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"포트폴리오 성과 분석 실패: {e}")
            return {}
    
    # =============================================================================
    # 자산 가격 데이터 관리
    # =============================================================================
    
    def save_asset_price(self, asset: str, asset_type: str, price_data: Dict[str, Any]) -> int:
        """자산 가격 데이터 저장"""
        try:
            with self.get_session() as session:
                # 중복 확인
                existing = session.query(AssetPrice).filter(
                    AssetPrice.asset == asset,
                    AssetPrice.timestamp == price_data.get('timestamp', datetime.now())
                ).first()
                
                if existing:
                    # 업데이트
                    for key, value in price_data.items():
                        if hasattr(existing, key) and value is not None:
                            if key in ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'market_cap']:
                                setattr(existing, key, Decimal(str(value)))
                            else:
                                setattr(existing, key, value)
                    
                    existing.updated_at = datetime.now()
                    price_id = existing.id
                else:
                    # 새로 생성
                    price_record = AssetPrice(
                        asset=asset,
                        asset_type=asset_type,
                        timestamp=price_data.get('timestamp', datetime.now()),
                        open_price=Decimal(str(price_data.get('open_price', 0))) if price_data.get('open_price') else None,
                        high_price=Decimal(str(price_data.get('high_price', 0))) if price_data.get('high_price') else None,
                        low_price=Decimal(str(price_data.get('low_price', 0))) if price_data.get('low_price') else None,
                        close_price=Decimal(str(price_data['close_price'])),
                        volume=Decimal(str(price_data.get('volume', 0))) if price_data.get('volume') else None,
                        market_cap=Decimal(str(price_data.get('market_cap', 0))) if price_data.get('market_cap') else None,
                        change_24h=price_data.get('change_24h'),
                        change_pct_24h=price_data.get('change_pct_24h'),
                        rsi=price_data.get('rsi'),
                        macd=price_data.get('macd'),
                        ma_20=Decimal(str(price_data.get('ma_20', 0))) if price_data.get('ma_20') else None,
                        ma_50=Decimal(str(price_data.get('ma_50', 0))) if price_data.get('ma_50') else None,
                        bollinger_upper=Decimal(str(price_data.get('bollinger_upper', 0))) if price_data.get('bollinger_upper') else None,
                        bollinger_lower=Decimal(str(price_data.get('bollinger_lower', 0))) if price_data.get('bollinger_lower') else None,
                        source=price_data.get('source', 'api'),
                        quality_score=price_data.get('quality_score', 1.0)
                    )
                    
                    session.add(price_record)
                    session.flush()
                    price_id = price_record.id
                
                self.logger.debug(f"자산 가격 저장: {asset} {price_data.get('close_price')}")
                return price_id
                
        except Exception as e:
            self.logger.error(f"자산 가격 저장 실패: {e}")
            raise
    
    def get_asset_price_history(self, asset: str, days: int = 30) -> List[AssetPrice]:
        """자산 가격 이력 조회"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                prices = session.query(AssetPrice)\
                    .filter(AssetPrice.asset == asset)\
                    .filter(AssetPrice.timestamp >= start_date)\
                    .order_by(AssetPrice.timestamp).all()
                
                return prices
                
        except Exception as e:
            self.logger.error(f"자산 가격 이력 조회 실패: {e}")
            return []
    
    def calculate_technical_indicators(self, asset: str, days: int = 50) -> Dict[str, Any]:
        """기술적 지표 계산"""
        try:
            prices = self.get_asset_price_history(asset, days)
            
            if len(prices) < 20:
                return {}
            
            close_prices = [float(p.close_price) for p in prices]
            
            # 이동평균 계산
            ma_20 = sum(close_prices[-20:]) / 20 if len(close_prices) >= 20 else None
            ma_50 = sum(close_prices[-50:]) / 50 if len(close_prices) >= 50 else None
            
            # RSI 계산 (간단 버전)
            if len(close_prices) >= 14:
                gains = []
                losses = []
                for i in range(1, len(close_prices)):
                    change = close_prices[i] - close_prices[i-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(-change)
                
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
            else:
                rsi = None
            
            # 볼린저 밴드 계산
            if len(close_prices) >= 20:
                recent_prices = close_prices[-20:]
                mean = sum(recent_prices) / len(recent_prices)
                variance = sum((p - mean) ** 2 for p in recent_prices) / len(recent_prices)
                std_dev = variance ** 0.5
                
                bollinger_upper = mean + (std_dev * 2)
                bollinger_lower = mean - (std_dev * 2)
            else:
                bollinger_upper = None
                bollinger_lower = None
            
            return {
                'asset': asset,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'rsi': rsi,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'current_price': close_prices[-1] if close_prices else None,
                'price_change_pct': ((close_prices[-1] - close_prices[0]) / close_prices[0] * 100) if len(close_prices) > 1 else 0,
                'calculation_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {e}")
            return {}
    
    # =============================================================================
    # 뉴스 데이터 관리
    # =============================================================================
    
    def save_news_record(self, news_data: Dict[str, Any]) -> int:
        """뉴스 기록 저장"""
        try:
            with self.get_session() as session:
                # 중복 확인 (URL 기준)
                existing = session.query(NewsRecord).filter(
                    NewsRecord.url == news_data.get('url')
                ).first()
                
                if existing:
                    return existing.id
                
                news_record = NewsRecord(
                    timestamp=news_data.get('timestamp', datetime.now()),
                    title=news_data['title'][:500],  # 길이 제한
                    content=news_data.get('content', ''),
                    url=news_data.get('url', ''),
                    source=news_data.get('source', 'unknown'),
                    related_assets=','.join(news_data.get('related_assets', [])) if news_data.get('related_assets') else None,
                    sentiment_score=news_data.get('sentiment_score'),
                    sentiment_label=news_data.get('sentiment_label'),
                    confidence=news_data.get('confidence'),
                    keywords=json.dumps(news_data.get('keywords', [])),
                    market_impact_score=news_data.get('market_impact_score'),
                    relevance_score=news_data.get('relevance_score'),
                    language=news_data.get('language', 'ko'),
                    region=news_data.get('region', 'KR')
                )
                
                session.add(news_record)
                session.flush()
                news_id = news_record.id
                
                self.logger.debug(f"뉴스 기록 저장: {news_id}")
                return news_id
                
        except Exception as e:
            self.logger.error(f"뉴스 기록 저장 실패: {e}")
            raise
    
    def get_news_sentiment_summary(self, asset: str = None, days: int = 7) -> Dict[str, Any]:
        """뉴스 감성 요약"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                query = session.query(NewsRecord).filter(NewsRecord.timestamp >= start_date)
                
                if asset:
                    query = query.filter(NewsRecord.related_assets.contains(asset))
                
                news_records = query.all()
                
                if not news_records:
                    return {}
                
                # 감성 분석
                sentiment_scores = [n.sentiment_score for n in news_records if n.sentiment_score is not None]
                
                positive_count = len([n for n in news_records if n.sentiment_label == 'positive'])
                negative_count = len([n for n in news_records if n.sentiment_label == 'negative'])
                neutral_count = len(news_records) - positive_count - negative_count
                
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                
                # 소스별 분석
                source_counts = {}
                for news in news_records:
                    source_counts[news.source] = source_counts.get(news.source, 0) + 1
                
                return {
                    'period_days': days,
                    'total_news': len(news_records),
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'avg_sentiment_score': avg_sentiment,
                    'sentiment_distribution': {
                        'positive_pct': (positive_count / len(news_records)) * 100,
                        'negative_pct': (negative_count / len(news_records)) * 100,
                        'neutral_pct': (neutral_count / len(news_records)) * 100
                    },
                    'source_counts': source_counts,
                    'recent_news_titles': [n.title for n in news_records[-5:]]  # 최근 5개
                }
                
        except Exception as e:
            self.logger.error(f"뉴스 감성 요약 실패: {e}")
            return {}
    
    # =============================================================================
    # 백테스트 결과 관리
    # =============================================================================
    
    def save_backtest_result(self, backtest_data: Dict[str, Any]) -> int:
        """백테스트 결과 저장"""
        try:
            with self.get_session() as session:
                backtest = BacktestResult(
                    test_name=backtest_data['test_name'],
                    strategy_name=backtest_data['strategy_name'],
                    start_date=backtest_data['start_date'],
                    end_date=backtest_data['end_date'],
                    initial_capital=Decimal(str(backtest_data['initial_capital'])),
                    final_value=Decimal(str(backtest_data['final_value'])),
                    total_return=backtest_data['total_return'],
                    total_return_pct=backtest_data['total_return_pct'],
                    sharpe_ratio=backtest_data.get('sharpe_ratio'),
                    sortino_ratio=backtest_data.get('sortino_ratio'),
                    max_drawdown=backtest_data.get('max_drawdown'),
                    max_drawdown_duration=backtest_data.get('max_drawdown_duration'),
                    volatility=backtest_data.get('volatility'),
                    total_trades=backtest_data.get('total_trades', 0),
                    winning_trades=backtest_data.get('winning_trades', 0),
                    losing_trades=backtest_data.get('losing_trades', 0),
                    win_rate=backtest_data.get('win_rate'),
                    avg_winning_trade=backtest_data.get('avg_winning_trade'),
                    avg_losing_trade=backtest_data.get('avg_losing_trade'),
                    largest_win=backtest_data.get('largest_win'),
                    largest_loss=backtest_data.get('largest_loss'),
                    best_day=backtest_data.get('best_day'),
                    worst_day=backtest_data.get('worst_day'),
                    best_month=backtest_data.get('best_month'),
                    worst_month=backtest_data.get('worst_month'),
                    parameters=json.dumps(backtest_data.get('parameters', {})),
                    market_conditions=json.dumps(backtest_data.get('market_conditions', {})),
                    notes=backtest_data.get('notes')
                )
                
                session.add(backtest)
                session.flush()
                backtest_id = backtest.id
                
                self.logger.info(f"백테스트 결과 저장: {backtest_data['test_name']} (ID: {backtest_id})")
                return backtest_id
                
        except Exception as e:
            self.logger.error(f"백테스트 결과 저장 실패: {e}")
            raise
    
    def get_backtest_comparison(self, strategy_names: List[str] = None) -> Dict[str, Any]:
        """백테스트 결과 비교"""
        try:
            with self.get_session() as session:
                query = session.query(BacktestResult)
                
                if strategy_names:
                    query = query.filter(BacktestResult.strategy_name.in_(strategy_names))
                
                backtests = query.order_by(desc(BacktestResult.created_at)).all()
                
                if not backtests:
                    return {}
                
                # 전략별 최고 성과
                strategy_performance = {}
                for backtest in backtests:
                    strategy = backtest.strategy_name
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = []
                    
                    strategy_performance[strategy].append({
                        'test_name': backtest.test_name,
                        'total_return_pct': backtest.total_return_pct,
                        'sharpe_ratio': backtest.sharpe_ratio,
                        'max_drawdown': backtest.max_drawdown,
                        'win_rate': backtest.win_rate,
                        'total_trades': backtest.total_trades,
                        'test_date': backtest.created_at.isoformat()
                    })
                
                # 전략별 평균 성과
                strategy_averages = {}
                for strategy, results in strategy_performance.items():
                    if results:
                        strategy_averages[strategy] = {
                            'avg_return_pct': sum(r['total_return_pct'] for r in results) / len(results),
                            'avg_sharpe_ratio': sum(r['sharpe_ratio'] for r in results if r['sharpe_ratio']) / len([r for r in results if r['sharpe_ratio']]) if any(r['sharpe_ratio'] for r in results) else 0,
                            'avg_max_drawdown': sum(r['max_drawdown'] for r in results if r['max_drawdown']) / len([r for r in results if r['max_drawdown']]) if any(r['max_drawdown'] for r in results) else 0,
                            'avg_win_rate': sum(r['win_rate'] for r in results if r['win_rate']) / len([r for r in results if r['win_rate']]) if any(r['win_rate'] for r in results) else 0,
                            'total_tests': len(results),
                            'best_performance': max(results, key=lambda x: x['total_return_pct'])
                        }
                
                return {
                    'total_backtests': len(backtests),
                    'strategies_tested': len(strategy_performance),
                    'strategy_performance': strategy_performance,
                    'strategy_averages': strategy_averages,
                    'best_overall': max(backtests, key=lambda x: x.total_return_pct),
                    'most_recent': max(backtests, key=lambda x: x.created_at)
                }
                
        except Exception as e:
            self.logger.error(f"백테스트 비교 실패: {e}")
            return {}
    
    # =============================================================================
    # 시스템 성능 모니터링
    # =============================================================================
    
    def save_system_metrics(self, metrics_data: Dict[str, Any]) -> int:
        """시스템 성능 지표 저장"""
        try:
            with self.get_session() as session:
                metrics = SystemMetrics(
                    timestamp=metrics_data.get('timestamp', datetime.now()),
                    db_connection_count=metrics_data.get('db_connection_count'),
                    avg_query_time_ms=metrics_data.get('avg_query_time_ms'),
                    slow_queries_count=metrics_data.get('slow_queries_count'),
                    db_size_mb=metrics_data.get('db_size_mb'),
                    cpu_usage_pct=metrics_data.get('cpu_usage_pct'),
                    memory_usage_mb=metrics_data.get('memory_usage_mb'),
                    disk_usage_pct=metrics_data.get('disk_usage_pct'),
                    api_response_time_ms=metrics_data.get('api_response_time_ms'),
                    order_success_rate=metrics_data.get('order_success_rate'),
                    data_freshness_score=metrics_data.get('data_freshness_score'),
                    error_count=metrics_data.get('error_count', 0),
                    warning_count=metrics_data.get('warning_count', 0),
                    critical_events=json.dumps(metrics_data.get('critical_events', []))
                )
                
                session.add(metrics)
                session.flush()
                metrics_id = metrics.id
                
                return metrics_id
                
        except Exception as e:
            self.logger.error(f"시스템 지표 저장 실패: {e}")
            raise
    
    def get_system_health_report(self, hours: int = 24) -> Dict[str, Any]:
        """시스템 건강도 리포트"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            with self.get_session() as session:
                metrics = session.query(SystemMetrics)\
                    .filter(SystemMetrics.timestamp >= start_time)\
                    .order_by(SystemMetrics.timestamp).all()
                
                if not metrics:
                    return {}
                
                # 성능 지표 평균
                avg_cpu = sum(m.cpu_usage_pct for m in metrics if m.cpu_usage_pct) / len([m for m in metrics if m.cpu_usage_pct]) if any(m.cpu_usage_pct for m in metrics) else 0
                avg_memory = sum(m.memory_usage_mb for m in metrics if m.memory_usage_mb) / len([m for m in metrics if m.memory_usage_mb]) if any(m.memory_usage_mb for m in metrics) else 0
                avg_query_time = sum(m.avg_query_time_ms for m in metrics if m.avg_query_time_ms) / len([m for m in metrics if m.avg_query_time_ms]) if any(m.avg_query_time_ms for m in metrics) else 0
                
                # 오류 통계
                total_errors = sum(m.error_count for m in metrics if m.error_count)
                total_warnings = sum(m.warning_count for m in metrics if m.warning_count)
                
                # 최신 상태
                latest_metrics = metrics[-1] if metrics else None
                
                # 건강도 점수 계산 (0-100)
                health_score = 100
                if avg_cpu > 80:
                    health_score -= 20
                elif avg_cpu > 60:
                    health_score -= 10
                
                if avg_query_time > 1000:  # 1초 이상
                    health_score -= 15
                elif avg_query_time > 500:  # 0.5초 이상
                    health_score -= 5
                
                if total_errors > 0:
                    health_score -= min(total_errors * 5, 30)
                
                if total_warnings > 10:
                    health_score -= 10
                
                health_score = max(0, health_score)
                
                return {
                    'period_hours': hours,
                    'metrics_count': len(metrics),
                    'health_score': health_score,
                    'health_status': 'good' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical',
                    'performance': {
                        'avg_cpu_usage_pct': avg_cpu,
                        'avg_memory_usage_mb': avg_memory,
                        'avg_query_time_ms': avg_query_time,
                        'total_errors': total_errors,
                        'total_warnings': total_warnings
                    },
                    'latest_status': {
                        'timestamp': latest_metrics.timestamp."""