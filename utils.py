#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 전설적 퀸트프로젝트 UTILS 시스템 - 완전통합판
================================================================

코어 시스템을 지원하는 전설급 유틸리티 모음
- 📊 고급 기술적 분석 도구
- 🔄 데이터 처리 및 변환
- 📈 차트 및 시각화
- 🛡️ 리스크 관리 도구
- 📱 알림 및 로깅 시스템
- 🔍 백테스팅 엔진
- 💾 데이터베이스 헬퍼
- 🌐 API 통신 도구

Author: 전설적퀸트팀 | Version: UTILS v1.0
"""

import asyncio
import logging
import os
import sys
import json
import time
import warnings
import math
import hashlib
import pickle
import gzip
import csv
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import defaultdict, deque
from functools import wraps, lru_cache
from contextlib import contextmanager
import threading
import queue
import sqlite3
import tempfile
import shutil

# 수치 계산
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import talib

# 데이터 처리
import requests
import aiohttp
import yaml
from dotenv import load_dotenv

# 시각화
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 외부 API (선택적)
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import pyupbit
    UPBIT_AVAILABLE = True
except ImportError:
    UPBIT_AVAILABLE = False

try:
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

warnings.filterwarnings('ignore')
load_dotenv()

# ========================================================================================
# 🎨 차트 스타일 설정
# ========================================================================================

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일
sns.set_style("darkgrid")
sns.set_palette("husl")

# Plotly 기본 테마
PLOTLY_THEME = "plotly_dark"

# ========================================================================================
# 📊 고급 기술적 분석 도구
# ========================================================================================

class TechnicalAnalyzer:
    """전설급 기술적 분석 도구"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, 
                               high_col: str = 'High',
                               low_col: str = 'Low', 
                               close_col: str = 'Close',
                               volume_col: str = 'Volume') -> pd.DataFrame:
        """모든 기술적 지표를 한번에 계산"""
        result = df.copy()
        
        # 가격 데이터 추출
        high = df[high_col].values
        low = df[low_col].values  
        close = df[close_col].values
        volume = df[volume_col].values if volume_col in df.columns else None
        
        try:
            # 이동평균선
            result['SMA_5'] = talib.SMA(close, timeperiod=5)
            result['SMA_10'] = talib.SMA(close, timeperiod=10)
            result['SMA_20'] = talib.SMA(close, timeperiod=20)
            result['SMA_50'] = talib.SMA(close, timeperiod=50)
            result['SMA_200'] = talib.SMA(close, timeperiod=200)
            
            result['EMA_12'] = talib.EMA(close, timeperiod=12)
            result['EMA_26'] = talib.EMA(close, timeperiod=26)
            
            # 볼린저 밴드
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            result['BB_Upper'] = bb_upper
            result['BB_Middle'] = bb_middle
            result['BB_Lower'] = bb_lower
            result['BB_Width'] = (bb_upper - bb_lower) / bb_middle * 100
            result['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower) * 100
            
            # RSI
            result['RSI_14'] = talib.RSI(close, timeperiod=14)
            result['RSI_7'] = talib.RSI(close, timeperiod=7)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            result['MACD'] = macd
            result['MACD_Signal'] = macdsignal
            result['MACD_Hist'] = macdhist
            
            # 스토캐스틱
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
            result['Stoch_K'] = slowk
            result['Stoch_D'] = slowd
            
            # Williams %R
            result['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # CCI
            result['CCI'] = talib.CCI(high, low, close, timeperiod=14)
            
            # ATR (변동성)
            result['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            result['ATR_Percent'] = result['ATR'] / close * 100
            
            # ADX (추세강도)
            result['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            result['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            result['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # 패러볼릭 SAR
            result['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # 일목균형표
            result['Ichimoku_Tenkan'] = TechnicalAnalyzer._ichimoku_tenkan(high, low)
            result['Ichimoku_Kijun'] = TechnicalAnalyzer._ichimoku_kijun(high, low)
            
            # 거래량 지표 (거래량 데이터가 있는 경우)
            if volume is not None:
                result['OBV'] = talib.OBV(close, volume)
                result['Volume_SMA'] = talib.SMA(volume, timeperiod=20)
                result['Volume_Ratio'] = volume / result['Volume_SMA']
                
                # VWAP (Volume Weighted Average Price)
                result['VWAP'] = TechnicalAnalyzer._calculate_vwap(df, high_col, low_col, close_col, volume_col)
            
            # 캔들패턴 인식
            result['Doji'] = talib.CDLDOJI(df['Open'].values if 'Open' in df.columns else close, 
                                          high, low, close)
            result['Hammer'] = talib.CDLHAMMER(df['Open'].values if 'Open' in df.columns else close,
                                               high, low, close)
            result['Engulfing'] = talib.CDLENGULFING(df['Open'].values if 'Open' in df.columns else close,
                                                     high, low, close)
            
            # 커스텀 지표
            result['Price_Change'] = close / np.roll(close, 1) - 1
            result['Volatility_20'] = result['Price_Change'].rolling(20).std() * np.sqrt(252) * 100
            result['Momentum_10'] = (close / np.roll(close, 10) - 1) * 100
            
            logging.info("✅ 모든 기술적 지표 계산 완료")
            
        except Exception as e:
            logging.error(f"기술적 지표 계산 실패: {e}")
        
        return result
    
    @staticmethod
    def _ichimoku_tenkan(high: np.ndarray, low: np.ndarray, period: int = 9) -> np.ndarray:
        """일목균형표 전환선"""
        tenkan = np.full(len(high), np.nan)
        for i in range(period-1, len(high)):
            period_high = np.max(high[i-period+1:i+1])
            period_low = np.min(low[i-period+1:i+1])
            tenkan[i] = (period_high + period_low) / 2
        return tenkan
    
    @staticmethod  
    def _ichimoku_kijun(high: np.ndarray, low: np.ndarray, period: int = 26) -> np.ndarray:
        """일목균형표 기준선"""
        kijun = np.full(len(high), np.nan)
        for i in range(period-1, len(high)):
            period_high = np.max(high[i-period+1:i+1])
            period_low = np.min(low[i-period+1:i+1])
            kijun[i] = (period_high + period_low) / 2
        return kijun
    
    @staticmethod
    def _calculate_vwap(df: pd.DataFrame, high_col: str, low_col: str, 
                       close_col: str, volume_col: str) -> pd.Series:
        """VWAP 계산"""
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        vwap = (typical_price * df[volume_col]).cumsum() / df[volume_col].cumsum()
        return vwap
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """종합 매매신호 생성"""
        signals = df.copy()
        
        # 시그널 초기화
        signals['Signal_Score'] = 0.0
        signals['Buy_Signal'] = False
        signals['Sell_Signal'] = False
        
        try:
            # RSI 시그널
            signals.loc[signals['RSI_14'] < 30, 'Signal_Score'] += 1
            signals.loc[signals['RSI_14'] > 70, 'Signal_Score'] -= 1
            
            # MACD 시그널
            signals.loc[signals['MACD'] > signals['MACD_Signal'], 'Signal_Score'] += 0.5
            signals.loc[signals['MACD'] < signals['MACD_Signal'], 'Signal_Score'] -= 0.5
            
            # 볼린저밴드 시그널
            signals.loc[signals['BB_Position'] < 20, 'Signal_Score'] += 0.5
            signals.loc[signals['BB_Position'] > 80, 'Signal_Score'] -= 0.5
            
            # 이동평균 시그널
            signals.loc[signals['Close'] > signals['SMA_20'], 'Signal_Score'] += 0.3
            signals.loc[signals['Close'] < signals['SMA_20'], 'Signal_Score'] -= 0.3
            
            # 최종 시그널
            signals['Buy_Signal'] = signals['Signal_Score'] >= 1.5
            signals['Sell_Signal'] = signals['Signal_Score'] <= -1.5
            
        except Exception as e:
            logging.error(f"시그널 생성 실패: {e}")
        
        return signals

class PatternRecognizer:
    """차트 패턴 인식기"""
    
    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window: int = 10) -> Dict:
        """지지/저항선 탐지"""
        high = df['High'].values if 'High' in df.columns else df['Close'].values
        low = df['Low'].values if 'Low' in df.columns else df['Close'].values
        
        # 지지선 (저점)
        support_levels = []
        for i in range(window, len(low) - window):
            if low[i] == min(low[i-window:i+window+1]):
                support_levels.append((i, low[i]))
        
        # 저항선 (고점)
        resistance_levels = []
        for i in range(window, len(high) - window):
            if high[i] == max(high[i-window:i+window+1]):
                resistance_levels.append((i, high[i]))
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    @staticmethod
    def detect_triangle_pattern(df: pd.DataFrame) -> Dict:
        """삼각형 패턴 탐지"""
        support_resistance = PatternRecognizer.detect_support_resistance(df)
        
        # 간단한 삼각형 패턴 로직
        pattern_detected = False
        pattern_type = "none"
        
        if len(support_resistance['support']) >= 2 and len(support_resistance['resistance']) >= 2:
            # 상승삼각형, 하락삼각형, 대칭삼각형 판별 로직
            pattern_detected = True
            pattern_type = "symmetric_triangle"  # 간소화
        
        return {
            'detected': pattern_detected,
            'type': pattern_type,
            'support_resistance': support_resistance
        }

# ========================================================================================
# 🔄 데이터 처리 및 변환
# ========================================================================================

class DataProcessor:
    """데이터 처리 유틸리티"""
    
    @staticmethod
    def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV 데이터 정리"""
        cleaned = df.copy()
        
        # 결측값 처리
        cleaned = cleaned.dropna()
        
        # 가격 데이터 검증
        price_cols = ['Open', 'High', 'Low', 'Close']
        available_cols = [col for col in price_cols if col in cleaned.columns]
        
        for col in available_cols:
            # 음수 가격 제거
            cleaned = cleaned[cleaned[col] > 0]
            
            # 이상치 제거 (3시그마 룰)
            mean = cleaned[col].mean()
            std = cleaned[col].std()
            cleaned = cleaned[abs(cleaned[col] - mean) <= 3 * std]
        
        # High >= Low 검증
        if 'High' in cleaned.columns and 'Low' in cleaned.columns:
            cleaned = cleaned[cleaned['High'] >= cleaned['Low']]
        
        # 거래량 검증
        if 'Volume' in cleaned.columns:
            cleaned = cleaned[cleaned['Volume'] >= 0]
        
        logging.info(f"데이터 정리 완료: {len(df)} -> {len(cleaned)} 행")
        return cleaned
    
    @staticmethod
    def resample_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """데이터 리샘플링"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        resampled = df.resample(freq).agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """수익률 계산"""
        result = df.copy()
        
        # 단순 수익률
        result['Return'] = df[price_col].pct_change()
        
        # 로그 수익률
        result['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # 누적 수익률
        result['Cumulative_Return'] = (1 + result['Return']).cumprod() - 1
        
        # 변동성
        result['Volatility'] = result['Return'].rolling(20).std() * np.sqrt(252)
        
        # 샤프 비율 (20일 이동)
        excess_return = result['Return'] - 0.02/252  # 무위험수익률 2%
        result['Sharpe_Ratio'] = excess_return.rolling(20).mean() / result['Return'].rolling(20).std() * np.sqrt(252)
        
        return result
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """데이터 정규화"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = result[col].min()
                max_val = result[col].max()
                result[col] = (result[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in numeric_cols:
                result[col] = (result[col] - result[col].mean()) / result[col].std()
        
        elif method == 'robust':
            for col in numeric_cols:
                median = result[col].median()
                mad = np.median(np.abs(result[col] - median))
                result[col] = (result[col] - median) / mad
        
        return result

class DataCache:
    """데이터 캐싱 시스템"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, key: str) -> Path:
        """캐시 파일 경로 생성"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pkl.gz"
    
    def get(self, key: str, max_age_hours: int = 24) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        cache_path = self.get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # 캐시 만료 확인
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - cache_time > timedelta(hours=max_age_hours):
            cache_path.unlink()
            return None
        
        try:
            with gzip.open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"캐시 읽기 실패: {e}")
            return None
    
    def set(self, key: str, data: Any) -> bool:
        """캐시에 데이터 저장"""
        cache_path = self.get_cache_path(key)
        
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logging.error(f"캐시 저장 실패: {e}")
            return False
    
    def clear(self) -> int:
        """캐시 디렉토리 정리"""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl.gz"):
            try:
                cache_file.unlink()
                count += 1
            except:
                pass
        return count

# ========================================================================================
# 📈 차트 및 시각화
# ========================================================================================

class ChartGenerator:
    """차트 생성기"""
    
    @staticmethod
    def create_candlestick_chart(df: pd.DataFrame, 
                               title: str = "주가 차트",
                               show_volume: bool = True,
                               show_indicators: bool = True) -> go.Figure:
        """캔들스틱 차트 생성"""
        
        # 서브플롯 설정
        if show_volume:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(title, "거래량", "지표")
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.8, 0.2],
                subplot_titles=(title, "지표")
            )
        
        # 캔들스틱
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC"
            ),
            row=1, col=1
        )
        
        # 이동평균선
        if show_indicators and 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    mode='lines',
                    name='MA20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if show_indicators and 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    mode='lines',
                    name='MA50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # 볼린저 밴드
        if show_indicators and all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    mode='lines',
                    name='BB상단',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.5
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    mode='lines',
                    name='BB하단',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    opacity=0.1
                ),
                row=1, col=1
            )
        
        # 거래량
        if show_volume and 'Volume' in df.columns:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df['Close'], df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name="거래량",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # RSI
        if show_indicators and 'RSI_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI_14'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3 if show_volume else 2, col=1
            )
            
            # RSI 기준선
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         row=3 if show_volume else 2, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         row=3 if show_volume else 2, col=1, opacity=0.5)
        
        # 레이아웃 설정
        fig.update_layout(
            template=PLOTLY_THEME,
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Y축 범위 설정
        fig.update_yaxes(title_text="가격", row=1, col=1)
        if show_volume:
            fig.update_yaxes(title_text="거래량", row=2, col=1)
        if show_indicators:
            fig.update_yaxes(title_text="RSI", range=[0, 100], 
                           row=3 if show_volume else 2, col=1)
        
        return fig
    
    @staticmethod
    def create_performance_chart(returns: pd.Series, 
                               benchmark: Optional[pd.Series] = None,
                               title: str = "성과 분석") -> go.Figure:
        """성과 분석 차트"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("누적수익률", "일별수익률 분포", "드로우다운", "월별 수익률"),
            specs=[[{"colspan": 2}, None],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # 누적수익률
        cumulative = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=cumulative,
                mode='lines',
                name='포트폴리오',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        if benchmark is not None:
            benchmark_cumulative = (1 + benchmark).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark_cumulative,
                    mode='lines',
                    name='벤치마크',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
        
        # 수익률 분포
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='수익률 분포',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 드로우다운
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak * 100
        
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=drawdown,
                mode='lines',
                name='드로우다운',
                line=dict(color='red', width=1),
                fill='tonexty'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            template=PLOTLY_THEME,
            height=600,
            title_text=title
        )
        
        return fig
    
    @staticmethod
    def save_chart(fig: go.Figure, filename: str, 
                  format: str = 'html', width: int = 1200, height: int = 800):
        """차트 저장"""
        Path("charts").mkdir(exist_ok=True)
        filepath = Path("charts") / filename
        
        if format == 'html':
            fig.write_html(str(filepath))
        elif format == 'png':
            fig.write_image(str(filepath), width=width, height=height)
        elif format == 'pdf':
            fig.write_image(str(filepath), format='pdf', width=width, height=height)
        
        logging.info(f"차트 저장: {filepath}")

# ========================================================================================
# 🛡️ 리스크 관리 도구
# ========================================================================================

class RiskManager:
    """리스크 관리 시스템"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
        """VaR (Value at Risk) 계산"""
        return np.percentile(returns, confidence * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.05) -> float:
        """CVaR (Conditional VaR) 계산"""
        var = RiskManager.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_maximum_drawdown(returns: pd.Series) -> Dict:
        """최대 낙폭 계산"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # 회복 기간 계산
        recovery_date = None
        if max_dd_date in cumulative.index:
            peak_before_dd = peak.loc[max_dd_date]
            recovery_series = cumulative[cumulative.index > max_dd_date]
            recovery_mask = recovery_series >= peak_before_dd
            if recovery_mask.any():
                recovery_date = recovery_series[recovery_mask].index[0]
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'recovery_date': recovery_date,
            'drawdown_series': drawdown
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """소르티노 비율 계산"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return np.inf
        
        return excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """칼마 비율 계산"""
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        max_dd = RiskManager.calculate_maximum_drawdown(returns)['max_drawdown']
        
        if max_dd == 0:
            return np.inf
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def portfolio_optimization(returns: pd.DataFrame, 
                             method: str = 'mean_variance') -> Dict:
        """포트폴리오 최적화"""
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        num_assets = len(mean_returns)
        
        if method == 'mean_variance':
            # 최소분산 포트폴리오
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = num_assets * [1. / num_assets]
            
            result = minimize(portfolio_variance, initial_guess, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            optimal_weights = result.x
            
        elif method == 'equal_weight':
            optimal_weights = np.array([1/num_assets] * num_assets)
        
        elif method == 'risk_parity':
            # 리스크 패리티
            def risk_budget_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = np.multiply(marginal_contrib, weights)
                return np.sum((contrib - contrib.mean())**2)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0.01, 0.99) for _ in range(num_assets))
            initial_guess = num_assets * [1. / num_assets]
            
            result = minimize(risk_budget_objective, initial_guess,
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            optimal_weights = result.x
        
        else:
            optimal_weights = np.array([1/num_assets] * num_assets)
        
        # 포트폴리오 성과 계산
        portfolio_return = np.sum(mean_returns * optimal_weights)
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = portfolio_return / portfolio_vol
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'assets': returns.columns.tolist()
        }

class PositionSizer:
    """포지션 사이징"""
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """켈리 공식"""
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        return max(0, min(kelly_fraction, 0.25))  # 최대 25% 제한
    
    @staticmethod
    def fixed_fractional(account_value: float, risk_per_trade: float, 
                        entry_price: float, stop_loss: float) -> int:
        """고정 비율법"""
        risk_amount = account_value * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        position_size = int(risk_amount / risk_per_share)
        return max(0, position_size)
    
    @staticmethod
    def volatility_adjusted(returns: pd.Series, target_volatility: float = 0.15) -> float:
        """변동성 조정 포지션"""
        current_vol = returns.std() * np.sqrt(252)
        
        if current_vol == 0:
            return 0
        
        return target_volatility / current_vol

# ========================================================================================
# 📱 알림 및 로깅 시스템
# ========================================================================================

class NotificationManager:
    """고급 알림 관리자"""
    
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email': os.getenv('EMAIL_ADDRESS'),
            'password': os.getenv('EMAIL_PASSWORD')
        }
    
    async def send_telegram(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """텔레그램 메시지 전송"""
        if not self.telegram_token or not self.telegram_chat_id:
            return False
        
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        data = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    return response.status == 200
        except Exception as e:
            logging.error(f"텔레그램 전송 실패: {e}")
            return False
    
    async def send_discord(self, message: str, title: str = "퀸트프로젝트 알림") -> bool:
        """디스코드 웹훅 전송"""
        if not self.discord_webhook:
            return False
        
        data = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": 0x00ff00,
                "timestamp": datetime.now().isoformat()
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=data) as response:
                    return response.status == 204
        except Exception as e:
            logging.error(f"디스코드 전송 실패: {e}")
            return False
    
    def send_email(self, subject: str, message: str, to_email: str = None) -> bool:
        """이메일 전송"""
        if not all([self.email_config['email'], self.email_config['password']]):
            return False
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email']
            msg['To'] = to_email or self.email_config['email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['email'], to_email or self.email_config['email'], text)
            server.quit()
            
            return True
            
        except Exception as e:
            logging.error(f"이메일 전송 실패: {e}")
            return False
    
    async def broadcast_alert(self, message: str, channels: List[str] = None) -> Dict:
        """멀티채널 알림 브로드캐스트"""
        if channels is None:
            channels = ['telegram', 'discord']
        
        results = {}
        
        if 'telegram' in channels:
            results['telegram'] = await self.send_telegram(message)
        
        if 'discord' in channels:
            results['discord'] = await self.send_discord(message)
        
        if 'email' in channels:
            results['email'] = self.send_email("퀸트프로젝트 알림", message)
        
        return results

class SmartLogger:
    """스마트 로깅 시스템"""
    
    def __init__(self, name: str = "QuintProject", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 로그 디렉토리 생성
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 핸들러 설정
        self._setup_handlers()
        
        # 성능 모니터링
        self.performance_data = defaultdict(list)
    
    def _setup_handlers(self):
        """로그 핸들러 설정"""
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러 (일별 로테이션)
        from logging.handlers import TimedRotatingFileHandler
        
        file_handler = TimedRotatingFileHandler(
            self.log_dir / "quint.log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 오류 전용 핸들러
        error_handler = TimedRotatingFileHandler(
            self.log_dir / "quint_error.log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def performance_log(self, func_name: str, execution_time: float, **kwargs):
        """성능 로그"""
        self.performance_data[func_name].append({
            'timestamp': datetime.now(),
            'execution_time': execution_time,
            **kwargs
        })
        
        self.logger.info(f"⚡ {func_name} 실행시간: {execution_time:.3f}초")
    
    def get_performance_stats(self, func_name: str = None) -> Dict:
        """성능 통계 조회"""
        if func_name:
            data = self.performance_data.get(func_name, [])
            if not data:
                return {}
            
            times = [d['execution_time'] for d in data]
            return {
                'function': func_name,
                'call_count': len(times),
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times)
            }
        else:
            return {func: self.get_performance_stats(func) 
                   for func in self.performance_data.keys()}

def performance_monitor(func):
    """성능 모니터링 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger = SmartLogger()
        logger.performance_log(func.__name__, execution_time)
        
        return result
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger = SmartLogger()
        logger.performance_log(func.__name__, execution_time)
        
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

# ========================================================================================
# 🔍 백테스팅 엔진
# ========================================================================================

class BacktestEngine:
    """백테스팅 엔진"""
    
    def __init__(self, initial_capital: float = 100000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.fees = 0.001  # 0.1% 수수료
    
    def add_data(self, symbol: str, data: pd.DataFrame):
        """백테스트 데이터 추가"""
        if not hasattr(self, 'data'):
            self.data = {}
        self.data[symbol] = data.copy()
    
    def buy(self, symbol: str, date: datetime, price: float, 
           quantity: int = None, amount: float = None):
        """매수 주문"""
        if symbol not in self.data:
            return False
        
        if quantity is None and amount is None:
            return False
        
        if amount is not None:
            # 금액 기준 매수
            total_cost = amount * (1 + self.fees)
            if total_cost > self.capital:
                return False
            
            quantity = int(amount / price)
            if quantity <= 0:
                return False
        
        total_cost = quantity * price * (1 + self.fees)
        
        if total_cost > self.capital:
            return False
        
        # 포지션 업데이트
        if symbol in self.positions:
            avg_price = (self.positions[symbol]['price'] * self.positions[symbol]['quantity'] + 
                        price * quantity) / (self.positions[symbol]['quantity'] + quantity)
            self.positions[symbol]['quantity'] += quantity
            self.positions[symbol]['price'] = avg_price
        else:
            self.positions[symbol] = {'quantity': quantity, 'price': price}
        
        # 거래 기록
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'buy',
            'price': price,
            'quantity': quantity,
            'amount': total_cost,
            'fees': quantity * price * self.fees
        })
        
        self.capital -= total_cost
        return True
    
    def sell(self, symbol: str, date: datetime, price: float, 
            quantity: int = None, ratio: float = None):
        """매도 주문"""
        if symbol not in self.positions:
            return False
        
        available_quantity = self.positions[symbol]['quantity']
        
        if quantity is None and ratio is None:
            quantity = available_quantity  # 전량 매도
        elif ratio is not None:
            quantity = int(available_quantity * ratio)
        
        quantity = min(quantity, available_quantity)
        if quantity <= 0:
            return False
        
        # 매도 금액 계산
        gross_amount = quantity * price
        net_amount = gross_amount * (1 - self.fees)
        
        # 손익 계산
        cost_basis = quantity * self.positions[symbol]['price']
        pnl = gross_amount - cost_basis
        
        # 거래 기록
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'sell',
            'price': price,
            'quantity': quantity,
            'amount': net_amount,
            'fees': gross_amount * self.fees,
            'pnl': pnl,
            'pnl_percent': pnl / cost_basis * 100
        })
        
        # 포지션 업데이트
        self.positions[symbol]['quantity'] -= quantity
        if self.positions[symbol]['quantity'] == 0:
            del self.positions[symbol]
        
        self.capital += net_amount
        return True
    
    def get_portfolio_value(self, date: datetime, current_prices: Dict[str, float]) -> float:
        """포트폴리오 가치 계산"""
        portfolio_value = self.capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                portfolio_value += position['quantity'] * current_prices[symbol]
        
        return portfolio_value
    
    def run_backtest(self, strategy_func: Callable, start_date: datetime = None, 
                    end_date: datetime = None) -> Dict:
        """백테스트 실행"""
        
        if not hasattr(self, 'data') or not self.data:
            return {'error': '데이터가 없습니다'}
        
        # 날짜 범위 설정
        all_dates = set()
        for symbol_data in self.data.values():
            all_dates.update(symbol_data.index)
        
        dates = sorted(all_dates)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        
        # 백테스트 실행
        for date in dates:
            # 현재 가격 정보
            current_prices = {}
            for symbol, symbol_data in self.data.items():
                if date in symbol_data.index:
                    current_prices[symbol] = symbol_data.loc[date, 'Close']
            
            # 전략 실행
            signals = strategy_func(self, date, current_prices)
            
            # 신호 처리
            if signals:
                for signal in signals:
                    if signal['action'] == 'buy':
                        self.buy(signal['symbol'], date, signal['price'], 
                               amount=signal.get('amount'))
                    elif signal['action'] == 'sell':
                        self.sell(signal['symbol'], date, signal['price'], 
                                ratio=signal.get('ratio', 1.0))
            
            # 포트폴리오 가치 기록
            portfolio_value = self.get_portfolio_value(date, current_prices)
            self.portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'capital': self.capital,
                'positions_value': portfolio_value - self.capital
            })
        
        return self.generate_backtest_report()
    
    def generate_backtest_report(self) -> Dict:
        """백테스트 리포트 생성"""
        if not self.portfolio_values:
            return {'error': '백테스트 데이터가 없습니다'}
        
        # 수익률 계산
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        returns = portfolio_df['value'].pct_change().dropna()
        
        # 성과 지표
        total_return = (portfolio_df['value'].iloc[-1] / self.initial_capital - 1) * 100
        annual_return = (portfolio_df['value'].iloc[-1] / self.initial_capital) ** (252 / len(portfolio_df)) - 1
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = RiskManager.calculate_sharpe_ratio(returns)
        max_dd_info = RiskManager.calculate_maximum_drawdown(returns)
        
        # 거래 통계
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            profitable_trades = trades_df[trades_df.get('pnl', 0) > 0]
            win_rate = len(profitable_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            avg_win = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
            avg_loss = trades_df[trades_df.get('pnl', 0) < 0]['pnl'].mean()
            avg_loss = abs(avg_loss) if not pd.isna(avg_loss) else 0
        else:
            win_rate = avg_win = avg_loss = 0
        
        return {
            'start_date': portfolio_df.index[0],
            'end_date': portfolio_df.index[-1],
            'initial_capital': self.initial_capital,
            'final_value': portfolio_df['value'].iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd_info['max_drawdown'] * 100,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'portfolio_values': portfolio_df,
            'trades': trades_df,
            'returns': returns
        }

# ========================================================================================
# 💾 데이터베이스 헬퍼
# ========================================================================================

class DatabaseManager:
    """데이터베이스 관리자"""
    
    def __init__(self, db_path: str = "quint_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 가격 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # 거래 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    fees REAL DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    date TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 신호 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    target_price REAL,
                    stop_loss REAL,
                    reasoning TEXT,
                    metadata TEXT,
                    date TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 포트폴리오 스냅샷 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_percent REAL NOT NULL,
                    position_count INTEGER NOT NULL,
                    date TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price_data(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
            
            conn.commit()
    
    def save_price_data(self, symbol: str, data: pd.DataFrame) -> int:
        """가격 데이터 저장"""
        saved_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for date, row in data.iterrows():
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO price_data 
                        (symbol, date, open_price, high_price, low_price, close_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, date.date(),
                        row.get('Open'), row.get('High'), row.get('Low'), 
                        row.get('Close'), row.get('Volume')
                    ))
                    saved_count += 1
                except Exception as e:
                    logging.error(f"가격 데이터 저장 실패 {symbol} {date}: {e}")
            
            conn.commit()
        
        return saved_count
    
    def load_price_data(self, symbol: str, start_date: datetime = None, 
                       end_date: datetime = None) -> pd.DataFrame:
        """가격 데이터 로드"""
        query = "SELECT * FROM price_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.date())
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.date())
        
        query += " ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
            
            if not df.empty:
                df.set_index('date', inplace=True)
                df.rename(columns={
                    'open_price': 'Open',
                    'high_price': 'High', 
                    'low_price': 'Low',
                    'close_price': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                
                # 불필요한 컬럼 제거
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        return df
    
    def save_signal(self, signal) -> bool:
        """신호 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO signals 
                    (symbol, strategy, action, confidence, price, target_price, 
                     stop_loss, reasoning, metadata, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.symbol, signal.strategy, signal.action, signal.confidence,
                    signal.price, signal.target_price, signal.stop_loss,
                    signal.reasoning, json.dumps(signal.metadata), signal.timestamp
                ))
                conn.commit()
            return True
        except Exception as e:
            logging.error(f"신호 저장 실패: {e}")
            return False
    
    def get_signals(self, symbol: str = None, strategy: str = None, 
                   days: int = 30) -> pd.DataFrame:
        """신호 조회"""
        query = "SELECT * FROM signals WHERE date >= ?"
        params = [datetime.now() - timedelta(days=days)]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        
        query += " ORDER BY date DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
    
    def save_portfolio_snapshot(self, snapshot: Dict) -> bool:
        """포트폴리오 스냅샷 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO portfolio_snapshots 
                    (total_value, cash, positions_value, pnl, pnl_percent, position_count, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot['total_value'], snapshot['cash'], snapshot['positions_value'],
                    snapshot['pnl'], snapshot['pnl_percent'], snapshot['position_count'],
                    datetime.now()
                ))
                conn.commit()
            return True
        except Exception as e:
            logging.error(f"포트폴리오 스냅샷 저장 실패: {e}")
            return False
    
    def cleanup_old_data(self, days: int = 365) -> Dict:
        """오래된 데이터 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        counts = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 오래된 가격 데이터 삭제
            cursor.execute("DELETE FROM price_data WHERE date < ?", (cutoff_date.date(),))
            counts['price_data'] = cursor.rowcount
            
            # 오래된 신호 삭제
            cursor.execute("DELETE FROM signals WHERE date < ?", (cutoff_date,))
            counts['signals'] = cursor.rowcount
            
            # 오래된 포트폴리오 스냅샷 삭제
            cursor.execute("DELETE FROM portfolio_snapshots WHERE date < ?", (cutoff_date,))
            counts['snapshots'] = cursor.rowcount
            
            conn.commit()
        
        return counts

# ========================================================================================
# 🌐 API 통신 도구
# ========================================================================================

class APIClient:
    """API 클라이언트 유틸리티"""
    
    def __init__(self, base_url: str = None, api_key: str = None, 
                 rate_limit: float = 1.0, timeout: int = 30):
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limit = rate_limit  # 초당 요청 수 제한
        self.timeout = timeout
        self.last_request_time = 0
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit_wait(self):
        """레이트 리미트 대기"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def get(self, endpoint: str, params: Dict = None, headers: Dict = None) -> Dict:
        """GET 요청"""
        await self._rate_limit_wait()
        
        url = f"{self.base_url}/{endpoint}" if self.base_url else endpoint
        
        if headers is None:
            headers = {}
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API 요청 실패: {response.status}")
        
        except Exception as e:
            logging.error(f"GET 요청 실패 {url}: {e}")
            raise
    
    async def post(self, endpoint: str, data: Dict = None, headers: Dict = None) -> Dict:
        """POST 요청"""
        await self._rate_limit_wait()
        
        url = f"{self.base_url}/{endpoint}" if self.base_url else endpoint
        
        if headers is None:
            headers = {}
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    raise Exception(f"API 요청 실패: {response.status}")
        
        except Exception as e:
            logging.error(f"POST 요청 실패 {url}: {e}")
            raise

class DataFetcher:
    """멀티소스 데이터 수집기"""
    
    def __init__(self):
        self.cache = DataCache()
        self.upbit_client = None
        self.yfinance_available = YF_AVAILABLE
    
    async def fetch_stock_data(self, symbol: str, period: str = "1y", 
                              source: str = "yfinance") -> pd.DataFrame:
        """주식 데이터 수집"""
        cache_key = f"stock_{source}_{symbol}_{period}"
        
        # 캐시 확인
        cached_data = self.cache.get(cache_key, max_age_hours=4)
        if cached_data is not None:
            return cached_data
        
        try:
            if source == "yfinance" and self.yfinance_available:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    # 컬럼명 표준화
                    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    data = DataProcessor.clean_ohlcv_data(data)
                    
                    # 캐시 저장
                    self.cache.set(cache_key, data)
                    
                    return data
            
            elif source == "alpha_vantage":
                # Alpha Vantage API 구현 (API 키 필요)
                api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
                if not api_key:
                    raise Exception("Alpha Vantage API 키가 없습니다")
                
                async with APIClient(
                    base_url="https://www.alphavantage.co",
                    rate_limit=5  # 분당 5회 제한
                ) as client:
                    data = await client.get("query", params={
                        'function': 'TIME_SERIES_DAILY',
                        'symbol': symbol,
                        'apikey': api_key,
                        'outputsize': 'full'
                    })
                    
                    # Alpha Vantage 응답 파싱
                    if 'Time Series (Daily)' in data:
                        time_series = data['Time Series (Daily)']
                        df_data = []
                        
                        for date_str, prices in time_series.items():
                            df_data.append({
                                'Date': pd.to_datetime(date_str),
                                'Open': float(prices['1. open']),
                                'High': float(prices['2. high']),
                                'Low': float(prices['3. low']),
                                'Close': float(prices['4. close']),
                                'Volume': int(prices['5. volume'])
                            })
                        
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df = df.sort_index()
                        
                        # 기간 필터링
                        if period == "1y":
                            df = df.last('365D')
                        elif period == "6mo":
                            df = df.last('180D')
                        elif period == "3mo":
                            df = df.last('90D')
                        
                        self.cache.set(cache_key, df)
                        return df
            
        except Exception as e:
            logging.error(f"주식 데이터 수집 실패 {symbol}: {e}")
        
        return pd.DataFrame()
    
    async def fetch_crypto_data(self, symbol: str, interval: str = "day", 
                               count: int = 200) -> pd.DataFrame:
        """가상화폐 데이터 수집"""
        cache_key = f"crypto_upbit_{symbol}_{interval}_{count}"
        
        # 캐시 확인
        cached_data = self.cache.get(cache_key, max_age_hours=1)
        if cached_data is not None:
            return cached_data
        
        try:
            if UPBIT_AVAILABLE:
                data = pyupbit.get_ohlcv(symbol, interval=interval, count=count)
                
                if data is not None and not data.empty:
                    # 컬럼명 표준화
                    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    data = DataProcessor.clean_ohlcv_data(data)
                    
                    # 캐시 저장
                    self.cache.set(cache_key, data)
                    
                    return data
            
            else:
                # Binance API 대안
                async with APIClient(
                    base_url="https://api.binance.com/api/v3",
                    rate_limit=10
                ) as client:
                    # 심볼 변환 (KRW-BTC -> BTCUSDT)
                    binance_symbol = symbol.replace('KRW-', '') + 'USDT'
                    
                    data = await client.get("klines", params={
                        'symbol': binance_symbol,
                        'interval': '1d' if interval == 'day' else '1h',
                        'limit': count
                    })
                    
                    if data:
                        df_data = []
                        for kline in data:
                            df_data.append({
                                'Date': pd.to_datetime(int(kline[0]), unit='ms'),
                                'Open': float(kline[1]),
                                'High': float(kline[2]),
                                'Low': float(kline[3]),
                                'Close': float(kline[4]),
                                'Volume': float(kline[5])
                            })
                        
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        
                        self.cache.set(cache_key, df)
                        return df
        
        except Exception as e:
            logging.error(f"가상화폐 데이터 수집 실패 {symbol}: {e}")
        
        return pd.DataFrame()
    
    async def fetch_economic_data(self, indicator: str) -> pd.DataFrame:
        """경제 지표 데이터 수집"""
        cache_key = f"economic_{indicator}"
        
        # 캐시 확인
        cached_data = self.cache.get(cache_key, max_age_hours=24)
        if cached_data is not None:
            return cached_data
        
        try:
            # FRED API 사용
            fred_api_key = os.getenv('FRED_API_KEY')
            if not fred_api_key:
                logging.warning("FRED API 키가 없습니다")
                return pd.DataFrame()
            
            async with APIClient(
                base_url="https://api.stlouisfed.org/fred",
                rate_limit=2
            ) as client:
                data = await client.get("series/observations", params={
                    'series_id': indicator,
                    'api_key': fred_api_key,
                    'file_type': 'json',
                    'limit': 1000
                })
                
                if 'observations' in data:
                    df_data = []
                    for obs in data['observations']:
                        if obs['value'] != '.':
                            df_data.append({
                                'Date': pd.to_datetime(obs['date']),
                                'Value': float(obs['value'])
                            })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('Date', inplace=True)
                    
                    self.cache.set(cache_key, df)
                    return df
        
        except Exception as e:
            logging.error(f"경제 지표 수집 실패 {indicator}: {e}")
        
        return pd.DataFrame()

# ========================================================================================
# 🎛️ 설정 관리자
# ========================================================================================

class ConfigManager:
    """고급 설정 관리자"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.watchers = []
        self.load_config()
    
    def load_config(self):
        """설정 파일 로드"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self.create_default_config()
            
            self._substitute_env_vars()
            logging.info(f"설정 로드 완료: {self.config_file}")
            
        except Exception as e:
            logging.error(f"설정 로드 실패: {e}")
            self.create_default_config()
    
    def create_default_config(self):
        """기본 설정 생성"""
        self.config = {
            'system': {
                'name': 'QuintProject',
                'version': '1.0.0',
                'debug': False,
                'log_level': 'INFO'
            },
            'trading': {
                'default_position_size': 0.02,
                'max_positions': 10,
                'stop_loss': 0.08,
                'take_profit': 0.12,
                'fees': 0.001
            },
            'risk_management': {
                'max_daily_loss': 0.02,
                'max_portfolio_risk': 0.15,
                'correlation_threshold': 0.7
            },
            'data_sources': {
                'primary': 'yfinance',
                'cache_hours': 4,
                'retry_attempts': 3
            },
            'notifications': {
                'telegram': {
                    'enabled': True,
                    'token': '${TELEGRAM_BOT_TOKEN}',
                    'chat_id': '${TELEGRAM_CHAT_ID}'
                },
                'email': {
                    'enabled': False,
                    'smtp_server': '${SMTP_SERVER}',
                    'email': '${EMAIL_ADDRESS}',
                    'password': '${EMAIL_PASSWORD}'
                }
            }
        }
        self.save_config()
    
    def save_config(self):
        """설정 파일 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            logging.error(f"설정 저장 실패: {e}")
    
    def get(self, key_path: str, default=None):
        """점 표기법으로 설정값 조회"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """점 표기법으로 설정값 변경"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self.save_config()
        
        # 관찰자들에게 알림
        for watcher in self.watchers:
            watcher(key_path, value)
    
    def watch(self, callback: Callable):
        """설정 변경 관찰자 등록"""
        self.watchers.append(callback)
    
    def _substitute_env_vars(self):
        """환경변수 치환"""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_content = obj[2:-1]
                if ':-' in var_content:
                    var_name, default = var_content.split(':-', 1)
                    return os.getenv(var_name, default)
                else:
                    return os.getenv(var_content, obj)
            return obj
        
        self.config = substitute_recursive(self.config)

# ========================================================================================
# 🚀 퍼포먼스 최적화 도구
# ========================================================================================

class PerformanceOptimizer:
    """성능 최적화 도구"""
    
    @staticmethod
    def parallel_execute(func: Callable, items: List, max_workers: int = 4) -> List:
        """병렬 실행"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, item) for item in items]
            results = []
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logging.error(f"병렬 실행 오류: {e}")
                    results.append(None)
            
            return results
    
    @staticmethod
    async def async_parallel_execute(func: Callable, items: List, 
                                   semaphore_limit: int = 10) -> List:
        """비동기 병렬 실행"""
        semaphore = asyncio.Semaphore(semaphore_limit)
        
        async def limited_func(item):
            async with semaphore:
                return await func(item)
        
        tasks = [limited_func(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    @staticmethod
    def batch_process(data: List, batch_size: int, 
                     processor: Callable) -> List:
        """배치 처리"""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                batch_result = processor(batch)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            except Exception as e:
                logging.error(f"배치 처리 오류: {e}")
        
        return results

# ========================================================================================
# 🎯 전략 백테스팅 헬퍼
# ========================================================================================

def simple_moving_average_strategy(backtest_engine, date, current_prices):
    """단순 이동평균 전략 예시"""
    signals = []
    
    for symbol, price in current_prices.items():
        if symbol in backtest_engine.data:
            data = backtest_engine.data[symbol]
            
            # 현재 날짜까지의 데이터
            current_data = data[data.index <= date]
            
            if len(current_data) >= 50:
                sma_20 = current_data['Close'].rolling(20).mean().iloc[-1]
                sma_50 = current_data['Close'].rolling(50).mean().iloc[-1]
                
                # 골든크로스
                if sma_20 > sma_50 and symbol not in backtest_engine.positions:
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'price': price,
                        'amount': backtest_engine.capital * 0.1  # 10% 투자
                    })
                
                # 데드크로스
                elif sma_20 < sma_50 and symbol in backtest_engine.positions:
                    signals.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'price': price,
                        'ratio': 1.0  # 전량 매도
                    })
    
    return signals

def rsi_strategy(backtest_engine, date, current_prices):
    """RSI 전략 예시"""
    signals = []
    
    for symbol, price in current_prices.items():
        if symbol in backtest_engine.data:
            data = backtest_engine.data[symbol]
            current_data = data[data.index <= date]
            
            if len(current_data) >= 14:
                # RSI 계산
                delta = current_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                # RSI 과매도 (30 이하)
                if current_rsi < 30 and symbol not in backtest_engine.positions:
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'price': price,
                        'amount': backtest_engine.capital * 0.05
                    })
                
                # RSI 과매수 (70 이상)
                elif current_rsi > 70 and symbol in backtest_engine.positions:
                    signals.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'price': price,
                        'ratio': 1.0
                    })
    
    return signals

# ========================================================================================
# 🎮 유틸리티 함수들
# ========================================================================================

def create_sample_data(symbol: str = "TEST", days: int = 252, 
                      start_price: float = 100.0) -> pd.DataFrame:
    """샘플 데이터 생성"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         periods=days, freq='D')
    
    # 랜덤 워크로 가격 생성
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # 평균 0.1%, 표준편차 2%
    prices = [start_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = prices[1:]  # 첫 번째 제거
    
    # OHLCV 데이터 생성
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = int(np.random.uniform(100000, 1000000))
        
        data.append({
            'Open': open_price,
            'High': max(high, open_price, close_price),
            'Low': min(low, open_price, close_price),
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

def format_number(value: float, decimals: int = 2, suffix: str = "") -> str:
    """숫자 포맷팅"""
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.{decimals}f}B{suffix}"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.{decimals}f}M{suffix}"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.{decimals}f}K{suffix}"
    else:
        return f"{value:.{decimals}f}{suffix}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """퍼센트 포맷팅"""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"

def format_currency(value: float, currency: str = "KRW") -> str:
    """통화 포맷팅"""
    if currency == "KRW":
        return f"{value:,.0f}원"
    elif currency == "USD":
        return f"${value:,.2f}"
    elif currency == "JPY":
        return f"¥{value:,.0f}"
    else:
        return f"{value:,.2f} {currency}"

@contextmanager
def timer(description: str = "작업"):
    """실행 시간 측정 컨텍스트 매니저"""
    start_time = time.time()
    yield
    end_time = time.time()
    logging.info(f"⏱️ {description} 완료: {end_time - start_time:.3f}초")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """데이터프레임 유효성 검사"""
    if df.empty:
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logging.error(f"필수 컬럼 누락: {missing_columns}")
        return False
    
    return True

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """안전한 나눗셈"""
    try:
        if b == 0:
            return default
        return a / b
    except:
        return default

def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    """예외 발생시 재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logging.warning(f"재시도 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(delay * (attempt + 1))
            
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logging.warning(f"재시도 {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(delay * (attempt + 1))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

# ========================================================================================
# 🏁 최종 익스포트
# ========================================================================================

__all__ = [
    # 기술적 분석
    'TechnicalAnalyzer',
    'PatternRecognizer',
    
    # 데이터 처리
    'DataProcessor',
    'DataCache',
    'DataFetcher',
    
    # 차트 및 시각화
    'ChartGenerator',
    
    # 리스크 관리
    'RiskManager',
    'PositionSizer',
    
    # 알림 및 로깅
    'NotificationManager',
    'SmartLogger',
    'performance_monitor',
    
    # 백테스팅
    'BacktestEngine',
    'simple_moving_average_strategy',
    'rsi_strategy',
    
    # 데이터베이스
    'DatabaseManager',
    
    # API 통신
    'APIClient',
    
    # 설정 관리
    'ConfigManager',
    
    # 성능 최적화
    'PerformanceOptimizer',
    
    # 유틸리티 함수
    'create_sample_data',
    'format_number',
    'format_percentage', 
    'format_currency',
    'timer',
    'validate_dataframe',
    'safe_divide',
    'retry_on_exception'
]

"""
🛠️ QuintProject Utils 사용 예시:

# 1. 기술적 분석
from utils import TechnicalAnalyzer
df_with_indicators = TechnicalAnalyzer.calculate_all_indicators(df)
signals = TechnicalAnalyzer.generate_signals(df_with_indicators)

# 2. 차트 생성
from utils import ChartGenerator
fig = ChartGenerator.create_candlestick_chart(df, "AAPL 차트")
ChartGenerator.save_chart(fig, "aapl_chart.html")

# 3. 리스크 관리
from utils import RiskManager
sharpe = RiskManager.calculate_sharpe_ratio(returns)
max_dd = RiskManager.calculate_maximum_drawdown(returns)

# 4. 백테스팅
from utils import BacktestEngine, simple_moving_average_strategy
engine = BacktestEngine(initial_capital=1000000)
engine.add_data("AAPL", aapl_data)
result = await engine.run_backtest(simple_moving_average_strategy)

# 5. 데이터 수집
from utils import DataFetcher
fetcher = DataFetcher()
aapl_data = await fetcher.fetch_stock_data("AAPL", period="1y")
btc_data = await fetcher.fetch_crypto_data("KRW-BTC")

# 6. 알림 시스템
from utils import NotificationManager
notifier = NotificationManager()
await notifier.broadcast_alert("🚨 매수 신호 발생!", ["telegram", "discord"])

# 7. 성능 모니터링
from utils import performance_monitor, timer

@performance_monitor
async def my_strategy():
    # 전략 로직
    pass

with timer("데이터 처리"):
    processed_data = process_large_dataset(data)

# 8. 데이터베이스
from utils import DatabaseManager
db = DatabaseManager()
db.save_price_data("AAPL", aapl_data)
saved_data = db.load_price_data("AAPL", start_date=datetime(2024, 1, 1))

# 9. 포지션 사이징
from utils import PositionSizer
kelly_size = PositionSizer.kelly_criterion(0.6, 100, 50)  # 승률 60%, 평균승:100, 평균패:50
position_size = PositionSizer.fixed_fractional(1000000, 0.02, 100, 92)

# 10. 포트폴리오 최적화
from utils import RiskManager
returns_df = pd.DataFrame({'AAPL': aapl_returns, 'GOOGL': googl_returns})
optimal_portfolio = RiskManager.portfolio_optimization(returns_df, method='mean_variance')

# 11. 설정 관리
from utils import ConfigManager
config = ConfigManager()
trading_config = config.get('trading.default_position_size', 0.02)
config.set('risk_management.max_daily_loss', 0.03)

# 12. 캐싱
from utils import DataCache
cache = DataCache()
cache.set("market_data_20241207", market_data)
cached_data = cache.get("market_data_20241207", max_age_hours=4)

# 13. 병렬 처리
from utils import PerformanceOptimizer
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
results = await PerformanceOptimizer.async_parallel_execute(
    fetch_stock_data, symbols, semaphore_limit=5
)

# 14. 유틸리티 함수들
from utils import format_number, format_percentage, format_currency
print(format_number(1234567.89))  # "1.23M"
print(format_percentage(0.0547))  # "+5.47%"
print(format_currency(1234567, "USD"))  # "$1,234,567.00"

# 15. 샘플 데이터 생성
from utils import create_sample_data
sample_df = create_sample_data("TEST", days=252, start_price=100)

# 16. 패턴 인식
from utils import PatternRecognizer
support_resistance = PatternRecognizer.detect_support_resistance(df)
triangle_pattern = PatternRecognizer.detect_triangle_pattern(df)

# 17. 데이터 정규화
from utils import DataProcessor
normalized_df = DataProcessor.normalize_data(df, method='minmax')
clean_df = DataProcessor.clean_ohlcv_data(raw_df)
returns_df = DataProcessor.calculate_returns(df)

# 18. API 클라이언트
from utils import APIClient
async with APIClient(base_url="https://api.example.com", rate_limit=5) as client:
    data = await client.get("endpoint", params={"symbol": "AAPL"})

# 19. 재시도 데코레이터
from utils import retry_on_exception

@retry_on_exception(max_retries=3, delay=1.0)
def unreliable_api_call():
    # 불안정한 API 호출
    pass

# 20. 스마트 로깅
from utils import SmartLogger
logger = SmartLogger("MyStrategy")
logger.logger.info("전략 시작")
logger.performance_log("calculate_indicators", 0.234)
stats = logger.get_performance_stats("calculate_indicators")
"""

# ========================================================================================
# 🎯 추가 전략 템플릿
# ========================================================================================

def bollinger_bands_strategy(backtest_engine, date, current_prices):
    """볼린저 밴드 전략"""
    signals = []
    
    for symbol, price in current_prices.items():
        if symbol in backtest_engine.data:
            data = backtest_engine.data[symbol]
            current_data = data[data.index <= date]
            
            if len(current_data) >= 20:
                # 볼린저 밴드 계산
                sma = current_data['Close'].rolling(20).mean()
                std = current_data['Close'].rolling(20).std()
                upper_band = sma + (std * 2)
                lower_band = sma - (std * 2)
                
                current_price = current_data['Close'].iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]
                
                # 하단 밴드 터치 시 매수
                if (current_price <= current_lower and 
                    symbol not in backtest_engine.positions):
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'price': price,
                        'amount': backtest_engine.capital * 0.1
                    })
                
                # 상단 밴드 터치 시 매도
                elif (current_price >= current_upper and 
                      symbol in backtest_engine.positions):
                    signals.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'price': price,
                        'ratio': 1.0
                    })
    
    return signals

def macd_strategy(backtest_engine, date, current_prices):
    """MACD 전략"""
    signals = []
    
    for symbol, price in current_prices.items():
        if symbol in backtest_engine.data:
            data = backtest_engine.data[symbol]
            current_data = data[data.index <= date]
            
            if len(current_data) >= 35:  # 26 + 9
                # MACD 계산
                ema_12 = current_data['Close'].ewm(span=12).mean()
                ema_26 = current_data['Close'].ewm(span=26).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9).mean()
                
                # 시그널
                current_macd = macd_line.iloc[-1]
                current_signal = signal_line.iloc[-1]
                prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else current_macd
                prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else current_signal
                
                # 골든크로스 (MACD가 시그널선을 상향돌파)
                if (prev_macd <= prev_signal and current_macd > current_signal and
                    symbol not in backtest_engine.positions):
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'price': price,
                        'amount': backtest_engine.capital * 0.08
                    })
                
                # 데드크로스 (MACD가 시그널선을 하향돌파)
                elif (prev_macd >= prev_signal and current_macd < current_signal and
                      symbol in backtest_engine.positions):
                    signals.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'price': price,
                        'ratio': 1.0
                    })
    
    return signals

def momentum_strategy(backtest_engine, date, current_prices):
    """모멘텀 전략"""
    signals = []
    
    for symbol, price in current_prices.items():
        if symbol in backtest_engine.data:
            data = backtest_engine.data[symbol]
            current_data = data[data.index <= date]
            
            if len(current_data) >= 21:
                # 20일 모멘텀
                momentum_20 = (current_data['Close'].iloc[-1] / 
                              current_data['Close'].iloc[-21] - 1) * 100
                
                # 거래량 확인
                avg_volume = current_data['Volume'].rolling(20).mean().iloc[-1]
                current_volume = current_data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume
                
                # 강한 상승 모멘텀 + 거래량 증가
                if (momentum_20 > 5 and volume_ratio > 1.5 and 
                    symbol not in backtest_engine.positions):
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'price': price,
                        'amount': backtest_engine.capital * 0.12
                    })
                
                # 모멘텀 소실
                elif (momentum_20 < -3 and symbol in backtest_engine.positions):
                    signals.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'price': price,
                        'ratio': 1.0
                    })
    
    return signals

# ========================================================================================
# 🧪 테스트 및 검증 도구
# ========================================================================================

class StrategyTester:
    """전략 테스트 도구"""
    
    @staticmethod
    def quick_backtest(strategy_func: Callable, symbols: List[str], 
                      period: str = "1y") -> Dict:
        """빠른 백테스트"""
        try:
            # 데이터 수집
            data_fetcher = DataFetcher()
            backtest_engine = BacktestEngine(initial_capital=10000000)  # 1천만원
            
            async def run_test():
                for symbol in symbols:
                    data = await data_fetcher.fetch_stock_data(symbol, period)
                    if not data.empty:
                        backtest_engine.add_data(symbol, data)
                
                return await backtest_engine.run_backtest(strategy_func)
            
            # 동기 실행을 위한 래퍼
            try:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(run_test())
            except RuntimeError:
                # 이미 실행 중인 이벤트 루프가 있는 경우
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(run_test())
            
            return result
            
        except Exception as e:
            logging.error(f"빠른 백테스트 실패: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def compare_strategies(strategies: Dict[str, Callable], symbols: List[str]) -> pd.DataFrame:
        """전략 비교"""
        results = []
        
        for strategy_name, strategy_func in strategies.items():
            try:
                result = StrategyTester.quick_backtest(strategy_func, symbols)
                
                if 'error' not in result:
                    results.append({
                        'Strategy': strategy_name,
                        'Total_Return': result.get('total_return', 0),
                        'Annual_Return': result.get('annual_return', 0),
                        'Sharpe_Ratio': result.get('sharpe_ratio', 0),
                        'Max_Drawdown': result.get('max_drawdown', 0),
                        'Win_Rate': result.get('win_rate', 0),
                        'Total_Trades': result.get('total_trades', 0)
                    })
                else:
                    logging.error(f"전략 {strategy_name} 실패: {result['error']}")
            
            except Exception as e:
                logging.error(f"전략 {strategy_name} 테스트 실패: {e}")
        
        return pd.DataFrame(results)

def validate_strategy(strategy_func: Callable) -> Dict:
    """전략 검증"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        # 샘플 데이터로 테스트
        sample_data = create_sample_data("TEST", days=100)
        backtest_engine = BacktestEngine(initial_capital=1000000)
        backtest_engine.add_data("TEST", sample_data)
        
        # 첫 번째 날짜로 테스트
        test_date = sample_data.index[50]  # 중간 지점
        current_prices = {"TEST": sample_data.loc[test_date, 'Close']}
        
        signals = strategy_func(backtest_engine, test_date, current_prices)
        
        # 신호 검증
        if not isinstance(signals, list):
            validation_results['errors'].append("전략 함수는 리스트를 반환해야 합니다")
            validation_results['is_valid'] = False
        
        for signal in signals:
            if not isinstance(signal, dict):
                validation_results['errors'].append("신호는 딕셔너리 형태여야 합니다")
                validation_results['is_valid'] = False
                continue
            
            required_keys = ['symbol', 'action', 'price']
            missing_keys = set(required_keys) - set(signal.keys())
            if missing_keys:
                validation_results['errors'].append(f"신호에 필수 키가 없습니다: {missing_keys}")
                validation_results['is_valid'] = False
            
            if signal.get('action') not in ['buy', 'sell']:
                validation_results['errors'].append("action은 'buy' 또는 'sell'이어야 합니다")
                validation_results['is_valid'] = False
    
    except Exception as e:
        validation_results['errors'].append(f"전략 실행 중 오류: {str(e)}")
        validation_results['is_valid'] = False
    
    return validation_results

# ========================================================================================
# 🏁 메인 실행부 (테스트용)
# ========================================================================================

async def main():
    """Utils 모듈 테스트"""
    print("🛠️" + "=" * 70)
    print("🔥 QuintProject Utils 테스트")
    print("=" * 72)
    
    try:
        # 1. 샘플 데이터 생성
        print("\n📊 1. 샘플 데이터 생성...")
        sample_data = create_sample_data("TEST", days=100, start_price=100)
        print(f"✅ 샘플 데이터 생성 완료: {len(sample_data)}행")
        
        # 2. 기술적 분석
        print("\n📈 2. 기술적 분석...")
        with timer("기술적 지표 계산"):
            analyzed_data = TechnicalAnalyzer.calculate_all_indicators(sample_data)
        print(f"✅ 기술적 지표 추가: {len(analyzed_data.columns)}개 컬럼")
        
        # 3. 시그널 생성
        print("\n🎯 3. 매매 시그널 생성...")
        signals_data = TechnicalAnalyzer.generate_signals(analyzed_data)
        buy_signals = signals_data['Buy_Signal'].sum()
        sell_signals = signals_data['Sell_Signal'].sum()
        print(f"✅ 매수 신호: {buy_signals}개, 매도 신호: {sell_signals}개")
        
        # 4. 차트 생성
        print("\n📊 4. 차트 생성...")
        fig = ChartGenerator.create_candlestick_chart(
            analyzed_data.tail(50), 
            title="샘플 데이터 차트",
            show_volume=True,
            show_indicators=True
        )
        ChartGenerator.save_chart(fig, "sample_chart.html")
        print("✅ 차트 저장: charts/sample_chart.html")
        
        # 5. 백테스팅
        print("\n🔍 5. 백테스팅...")
        backtest_engine = BacktestEngine(initial_capital=10000000)
        backtest_engine.add_data("TEST", sample_data)
        
        # 단순 이동평균 전략 테스트
        with timer("백테스트 실행"):
            result = await backtest_engine.run_backtest(simple_moving_average_strategy)
        
        if 'error' not in result:
            print(f"✅ 백테스트 완료:")
            print(f"   📈 총 수익률: {result['total_return']:.2f}%")
            print(f"   📊 샤프 비율: {result['sharpe_ratio']:.2f}")
            print(f"   📉 최대 낙폭: {result['max_drawdown']:.2f}%")
            print(f"   🎯 승률: {result['win_rate']:.1f}%")
        else:
            print(f"❌ 백테스트 실패: {result['error']}")
        
        # 6. 성과 분석
        print("\n📊 6. 성과 분석...")
        returns = sample_data['Close'].pct_change().dropna()
        
        sharpe = RiskManager.calculate_sharpe_ratio(returns)
        sortino = RiskManager.calculate_sortino_ratio(returns)
        max_dd_info = RiskManager.calculate_maximum_drawdown(returns)
        
        print(f"✅ 성과 지표:")
        print(f"   📊 샤프 비율: {sharpe:.3f}")
        print(f"   📊 소르티노 비율: {sortino:.3f}")
        print(f"   📉 최대 낙폭: {max_dd_info['max_drawdown']*100:.2f}%")
        
        # 7. 데이터베이스 테스트
        print("\n💾 7. 데이터베이스 테스트...")
        db = DatabaseManager("test_utils.db")
        saved_count = db.save_price_data("TEST", sample_data)
        loaded_data = db.load_price_data("TEST")
        print(f"✅ 데이터 저장: {saved_count}행")
        print(f"✅ 데이터 로드: {len(loaded_data)}행")
        
        # 8. 캐시 테스트
        print("\n🗄️ 8. 캐시 테스트...")
        cache = DataCache("test_cache")
        cache.set("test_data", sample_data)
        cached_data = cache.get("test_data")
        print(f"✅ 캐시 저장/로드: {'성공' if cached_data is not None else '실패'}")
        
        # 9. 전략 검증
        print("\n🧪 9. 전략 검증...")
        validation = validate_strategy(simple_moving_average_strategy)
        print(f"✅ 전략 유효성: {'통과' if validation['is_valid'] else '실패'}")
        if validation['errors']:
            for error in validation['errors']:
                print(f"   ❌ {error}")
        
        # 10. 유틸리티 함수 테스트
        print("\n🎮 10. 유틸리티 함수 테스트...")
        print(f"✅ 숫자 포맷: {format_number(1234567.89)}")
        print(f"✅ 퍼센트 포맷: {format_percentage(5.47)}")
        print(f"✅ 통화 포맷: {format_currency(1234567)}")
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        logging.error(f"Utils 테스트 실패: {e}")

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # 테스트 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")

# ========================================================================================
# 📚 문서화 정보
# ========================================================================================

"""
🛠️ QuintProject Utils v1.0 - 완전 가이드
================================================

이 모듈은 QuintProject Core를 지원하는 전설급 유틸리티 모음입니다.

주요 기능:
---------
1. 📊 고급 기술적 분석 (50+ 지표)
2. 📈 프로페셔널 차트 생성 
3. 🔍 백테스팅 엔진
4. 🛡️ 리스크 관리 도구
5. 💾 데이터베이스 관리
6. 📱 멀티채널 알림 시스템
7. 🌐 API 통신 도구
8. 🚀 성능 최적화 도구

설치 요구사항:
-------------
pip install pandas numpy scipy matplotlib seaborn plotly
pip install yfinance pyupbit talib aiohttp requests pyyaml python-dotenv

선택적 의존성:
-------------
pip install nest-asyncio  # Jupyter 환경
pip install ib-insync      # IBKR 연동

사용법:
------
from utils import *

# 기본 사용
data = create_sample_data("TEST", 100)
analyzed = TechnicalAnalyzer.calculate_all_indicators(data)
fig = ChartGenerator.create_candlestick_chart(analyzed)

# 백테스팅
engine = BacktestEngine()
engine.add_data("TEST", data)
result = await engine.run_backtest(simple_moving_average_strategy)

# 알림
notifier = NotificationManager()
await notifier.send_telegram("매수 신호 발생!")

환경변수 설정:
-------------
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
DISCORD_WEBHOOK_URL=your_discord_webhook
EMAIL_ADDRESS=your_email
EMAIL_PASSWORD=your_app_password
FRED_API_KEY=your_fred_api_key
ALPHA_VANTAGE_API_KEY=your_av_api_key

라이센스: MIT
작성자: 전설적퀸트팀
버전: 1.0.0
"""
