#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 퀸트프로젝트 유틸리티 모듈 (utils.py)
=============================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 통합 유틸리티

✨ 핵심 기능:
- 📊 데이터 분석 유틸리티
- 💱 환율 변환 헬퍼
- 📈 기술적 지표 계산
- 🔔 알림 포맷팅
- 🌐 네트워크 유틸리티
- 📁 파일 관리
- 🛡️ 보안 헬퍼
- ⏰ 시간 관리
- 📊 성과 분석

Author: 퀸트마스터팀
Version: 1.0.0 (완전체)
"""

import os
import sys
import json
import time
import asyncio
import logging
import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import aiohttp
import requests
from functools import wraps
import yaml
import pickle
import gzip
import base64
import urllib.parse
import socket
import psutil
import platform

# ============================================================================
# 📊 데이터 클래스 유틸리티
# ============================================================================

@dataclass
class TradingSignal:
    """거래 신호"""
    symbol: str
    strategy: str
    action: str  # BUY, SELL, HOLD
    price: float
    confidence: float
    reason: str
    timestamp: datetime
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    currency: str = 'USD'

@dataclass
class MarketData:
    """시장 데이터"""
    symbol: str
    price: float
    volume: float
    change_pct: float
    high_24h: float
    low_24h: float
    market_cap: Optional[float] = None
    timestamp: datetime = None
    currency: str = 'USD'

@dataclass
class RiskMetrics:
    """리스크 지표"""
    portfolio_value: float
    var_1d: float  # 1일 VaR
    var_1w: float  # 1주 VaR
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    correlation_spy: float
    concentration_risk: float

@dataclass
class PerformanceMetrics:
    """성과 지표"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int

# ============================================================================
# 💱 환율 변환 유틸리티
# ============================================================================

class CurrencyUtils:
    """환율 변환 유틸리티"""
    
    # 환율 캐시
    _exchange_rates = {}
    _last_update = None
    _cache_duration = 300  # 5분
    
    @classmethod
    async def convert_currency(cls, amount: float, from_currency: str, to_currency: str) -> float:
        """통화 변환"""
        if from_currency == to_currency:
            return amount
        
        # 환율 정보 업데이트
        await cls._update_rates_if_needed()
        
        # KRW 기준 변환
        if from_currency == 'KRW':
            if to_currency in cls._exchange_rates:
                return amount / cls._exchange_rates[to_currency]
        elif to_currency == 'KRW':
            if from_currency in cls._exchange_rates:
                return amount * cls._exchange_rates[from_currency]
        else:
            # 교차 환율 (KRW 경유)
            if from_currency in cls._exchange_rates and to_currency in cls._exchange_rates:
                krw_amount = amount * cls._exchange_rates[from_currency]
                return krw_amount / cls._exchange_rates[to_currency]
        
        # 환율 정보 없으면 원본 반환
        return amount
    
    @classmethod
    async def _update_rates_if_needed(cls):
        """필요시 환율 업데이트"""
        now = datetime.now()
        
        if (cls._last_update is None or 
            (now - cls._last_update).seconds > cls._cache_duration):
            
            await cls._fetch_exchange_rates()
    
    @classmethod
    async def _fetch_exchange_rates(cls):
        """환율 정보 가져오기"""
        try:
            # 한국은행 API 사용
            url = "https://www.bok.or.kr/portal/singl/openapi/exchangeJSON.do"
            params = {'lang': 'ko', 'per': 'day', 'keytype': 'json'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data:
                            if item['CUR_UNIT'] == 'USD':
                                cls._exchange_rates['USD'] = float(item['DEAL_BAS_R'].replace(',', ''))
                            elif item['CUR_UNIT'] == 'JPY(100)':
                                cls._exchange_rates['JPY'] = float(item['DEAL_BAS_R'].replace(',', '')) / 100
                            elif item['CUR_UNIT'] == 'INR':
                                cls._exchange_rates['INR'] = float(item['DEAL_BAS_R'].replace(',', ''))
                        
                        cls._exchange_rates['KRW'] = 1.0
                        cls._last_update = datetime.now()
                        
        except Exception as e:
            logging.error(f"환율 정보 업데이트 실패: {e}")
            # 기본값 설정
            if not cls._exchange_rates:
                cls._exchange_rates = {'USD': 1300, 'JPY': 9.5, 'INR': 16, 'KRW': 1.0}
    
    @classmethod
    def format_currency(cls, amount: float, currency: str) -> str:
        """통화 포맷팅"""
        if currency == 'KRW':
            return f"₩{amount:,.0f}"
        elif currency == 'USD':
            return f"${amount:,.2f}"
        elif currency == 'JPY':
            return f"¥{amount:,.0f}"
        elif currency == 'INR':
            return f"₹{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"

# ============================================================================
# 📈 기술적 지표 계산 유틸리티
# ============================================================================

class TechnicalIndicators:
    """기술적 지표 계산"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """단순이동평균 (SMA)"""
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(avg)
        
        return sma_values
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """지수이동평균 (EMA)"""
        if len(prices) < period:
            return []
        
        alpha = 2 / (period + 1)
        ema_values = [prices[0]]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return []
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(delta, 0) for delta in deltas]
        losses = [-min(delta, 0) for delta in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        for i in range(period, len(deltas)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
            
            # 다음 계산을 위한 평균 업데이트
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return rsi_values
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, List[float]]:
        """볼린저 밴드"""
        if len(prices) < period:
            return {'upper': [], 'middle': [], 'lower': []}
        
        sma_values = TechnicalIndicators.sma(prices, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            std = np.std(window)
            
            middle = sma_values[i - period + 1]
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            upper_band.append(upper)
            lower_band.append(lower)
        
        return {
            'upper': upper_band,
            'middle': sma_values,
            'lower': lower_band
        }
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """MACD"""
        if len(prices) < slow:
            return {'macd': [], 'signal': [], 'histogram': []}
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        # MACD 라인
        macd_line = []
        start_idx = slow - fast
        for i in range(len(ema_slow)):
            macd = ema_fast[i + start_idx] - ema_slow[i]
            macd_line.append(macd)
        
        # 신호선
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        
        # 히스토그램
        histogram = []
        signal_start = len(macd_line) - len(signal_line)
        for i in range(len(signal_line)):
            hist = macd_line[i + signal_start] - signal_line[i]
            histogram.append(hist)
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

# ============================================================================
# 📊 성과 분석 유틸리티
# ============================================================================

class PerformanceAnalyzer:
    """성과 분석 도구"""
    
    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """수익률 계산"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        return returns
    
    @staticmethod
    def calculate_metrics(returns: List[float], benchmark_returns: List[float] = None) -> PerformanceMetrics:
        """성과 지표 계산"""
        if not returns:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        returns_array = np.array(returns)
        
        # 기본 지표
        total_return = np.prod(1 + returns_array) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # 샤프 비율 (무위험 이자율 3% 가정)
        risk_free_rate = 0.03
        excess_returns = returns_array - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        
        # 소르티노 비율
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # 최대 낙폭
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # 승률 및 손익비
        positive_returns = returns_array[returns_array > 0]
        negative_returns = returns_array[returns_array < 0]
        
        win_rate = len(positive_returns) / len(returns_array) if len(returns_array) > 0 else 0
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=len(returns)
        )
    
    @staticmethod
    def calculate_var(returns: List[float], confidence: float = 0.05) -> float:
        """VaR (Value at Risk) 계산"""
        if not returns:
            return 0
        
        returns_array = np.array(returns)
        return np.percentile(returns_array, confidence * 100)
    
    @staticmethod
    def calculate_beta(returns: List[float], market_returns: List[float]) -> float:
        """베타 계산"""
        if len(returns) != len(market_returns) or len(returns) < 2:
            return 1.0
        
        returns_array = np.array(returns)
        market_array = np.array(market_returns)
        
        covariance = np.cov(returns_array, market_array)[0][1]
        market_variance = np.var(market_array)
        
        return covariance / market_variance if market_variance > 0 else 1.0

# ============================================================================
# 🌐 네트워크 유틸리티
# ============================================================================

class NetworkUtils:
    """네트워크 관련 유틸리티"""
    
    @staticmethod
    async def check_internet_connection(hosts: List[Tuple[str, int]] = None, timeout: int = 5) -> Dict[str, Any]:
        """인터넷 연결 상태 체크"""
        if hosts is None:
            hosts = [
                ('8.8.8.8', 53),      # Google DNS
                ('1.1.1.1', 53),      # Cloudflare DNS
                ('yahoo.com', 80),    # Yahoo Finance
                ('upbit.com', 443)    # Upbit
            ]
        
        results = {
            'connected': False,
            'latency': float('inf'),
            'successful_hosts': 0,
            'total_hosts': len(hosts),
            'details': {}
        }
        
        start_time = time.time()
        
        for host, port in hosts:
            try:
                # 비동기 소켓 연결 테스트
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=timeout
                )
                writer.close()
                await writer.wait_closed()
                
                results['successful_hosts'] += 1
                results['details'][host] = True
                
            except Exception as e:
                results['details'][host] = False
        
        # 전체 레이턴시 계산
        results['latency'] = (time.time() - start_time) * 1000
        
        # 절반 이상 성공하면 연결된 것으로 판단
        results['connected'] = results['successful_hosts'] >= len(hosts) // 2
        
        return results
    
    @staticmethod
    def get_public_ip() -> Optional[str]:
        """공용 IP 주소 조회"""
        try:
            response = requests.get('https://ipapi.co/ip/', timeout=5)
            if response.status_code == 200:
                return response.text.strip()
        except:
            pass
        
        return None
    
    @staticmethod
    def get_network_info() -> Dict[str, Any]:
        """네트워크 정보 수집"""
        info = {
            'hostname': socket.gethostname(),
            'local_ip': socket.gethostbyname(socket.gethostname()),
            'public_ip': NetworkUtils.get_public_ip(),
            'network_interfaces': {}
        }
        
        try:
            # 네트워크 인터페이스 정보
            for interface, addresses in psutil.net_if_addrs().items():
                info['network_interfaces'][interface] = []
                for addr in addresses:
                    info['network_interfaces'][interface].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask
                    })
        except Exception as e:
            logging.error(f"네트워크 정보 수집 실패: {e}")
        
        return info

# ============================================================================
# 📁 파일 관리 유틸리티
# ============================================================================

class FileUtils:
    """파일 관리 유틸리티"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """디렉토리 생성 (없으면)"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], backup_dir: Union[str, Path] = None) -> Optional[Path]:
        """파일 백업"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None
            
            if backup_dir is None:
                backup_dir = file_path.parent / 'backups'
            
            backup_dir = Path(backup_dir)
            FileUtils.ensure_directory(backup_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name
            
            import shutil
            shutil.copy2(file_path, backup_path)
            
            return backup_path
            
        except Exception as e:
            logging.error(f"파일 백업 실패: {e}")
            return None
    
    @staticmethod
    def compress_file(file_path: Union[str, Path], delete_original: bool = False) -> Optional[Path]:
        """파일 압축 (gzip)"""
        try:
            file_path = Path(file_path)
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            if delete_original:
                file_path.unlink()
            
            return compressed_path
            
        except Exception as e:
            logging.error(f"파일 압축 실패: {e}")
            return None
    
    @staticmethod
    def cleanup_old_files(directory: Union[str, Path], days: int = 30, pattern: str = "*") -> int:
        """오래된 파일 정리"""
        try:
            directory = Path(directory)
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_count = 0
            
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            logging.error(f"파일 정리 실패: {e}")
            return 0
    
    @staticmethod
    def get_file_size_human(file_path: Union[str, Path]) -> str:
        """파일 크기를 사람이 읽기 쉬운 형태로"""
        try:
            size = Path(file_path).stat().st_size
            
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            
            return f"{size:.1f} PB"
            
        except Exception:
            return "Unknown"

# ============================================================================
# 🔔 알림 포맷팅 유틸리티
# ============================================================================

class NotificationFormatter:
    """알림 메시지 포맷팅"""
    
    @staticmethod
    def format_trading_signal(signal: TradingSignal) -> str:
        """거래 신호 알림 포맷"""
        emoji = "🟢" if signal.action == "BUY" else "🔴" if signal.action == "SELL" else "🟡"
        
        message = f"""
{emoji} {signal.action} 신호 발생!

📊 종목: {signal.symbol}
🎯 전략: {signal.strategy}
💰 가격: {CurrencyUtils.format_currency(signal.price, signal.currency)}
📈 신뢰도: {signal.confidence:.1%}
📝 사유: {signal.reason}
⏰ 시간: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if signal.target_price:
            message += f"🎯 목표가: {CurrencyUtils.format_currency(signal.target_price, signal.currency)}\n"
        
        if signal.stop_loss:
            message += f"🛑 손절가: {CurrencyUtils.format_currency(signal.stop_loss, signal.currency)}\n"
        
        return message.strip()
    
    @staticmethod
    def format_portfolio_summary(portfolio_data: Dict) -> str:
        """포트폴리오 요약 알림 포맷"""
        total_value = portfolio_data.get('total_krw_value', 0)
        total_pnl = portfolio_data.get('total_unrealized_pnl', 0)
        return_pct = portfolio_data.get('total_return_pct', 0)
        
        emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        message = f"""
💼 포트폴리오 현황

{emoji} 총 가치: {CurrencyUtils.format_currency(total_value, 'KRW')}
{emoji} 손익: {CurrencyUtils.format_currency(total_pnl, 'KRW')} ({return_pct:+.2f}%)
📊 포지션: {portfolio_data.get('total_positions', 0)}개

🎯 전략별 현황:
"""
        
        for strategy, data in portfolio_data.get('by_strategy', {}).items():
            strategy_emoji = "🇺🇸" if strategy == "us" else "🇯🇵" if strategy == "japan" else "🇮🇳" if strategy == "india" else "💰"
            message += f"{strategy_emoji} {strategy}: {data['count']}개, {CurrencyUtils.format_currency(data['krw_value'], 'KRW')}\n"
        
        return message.strip()
    
    @staticmethod
    def format_system_alert(title: str, level: str, details: Dict) -> str:
        """시스템 알림 포맷"""
        level_emojis = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        
        emoji = level_emojis.get(level, 'ℹ️')
        
        message = f"""
{emoji} {title}

📊 시스템 상태:
"""
        
        for key, value in details.items():
            if isinstance(value, bool):
                status = "✅" if value else "❌"
                message += f"{status} {key}\n"
            elif isinstance(value, (int, float)):
                message += f"📊 {key}: {value:,.2f}\n"
            else:
                message += f"📝 {key}: {value}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message

# ============================================================================
# 🛡️ 보안 유틸리티
# ============================================================================

class SecurityUtils:
    """보안 관련 유틸리티"""
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """API 키 생성"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
        """비밀번호 해시화"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return base64.b64encode(password_hash).decode(), salt
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """비밀번호 검증"""
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return base64.b64encode(password_hash).decode() == hashed
    
    @staticmethod
    def encrypt_sensitive_data(data: str, key: str = None) -> Tuple[str, str]:
        """민감한 데이터 암호화 (간단한 XOR)"""
        if key is None:
            key = secrets.token_hex(16)
        
        # 간단한 XOR 암호화 (실제 운영환경에서는 더 강력한 암호화 사용 권장)
        key_bytes = key.encode()
        data_bytes = data.encode()
        
        encrypted = []
        for i, byte in enumerate(data_bytes):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        
        encrypted_data = base64.b64encode(bytes(encrypted)).decode()
        return encrypted_data, key
    
    @staticmethod
    def decrypt_sensitive_data(encrypted_data: str, key: str) -> str:
        """민감한 데이터 복호화"""
        try:
            key_bytes = key.encode()
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            
            decrypted = []
            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
            
            return bytes(decrypted).decode()
            
        except Exception:
            return ""
    
    @staticmethod
    def mask_sensitive_info(text: str, mask_char: str = "*", visible_chars: int = 4) -> str:
        """민감한 정보 마스킹"""
        if len(text) <= visible_chars:
            return mask_char * len(text)
        
        if visible_chars == 0:
            return mask_char * len(text)
        
        visible_part = text[:visible_chars]
        masked_part = mask_char * (len(text) - visible_chars)
        
        return visible_part + masked_part

# ============================================================================
# ⏰ 시간 관리 유틸리티
# ============================================================================

class TimeUtils:
    """시간 관리 유틸리티"""
    
    # 주요 시장 시간대
    TIMEZONES = {
        'KST': 'Asia/Seoul',
        'EST': 'America/New_York',
        'JST': 'Asia/Tokyo',
        'IST': 'Asia/Kolkata',
        'UTC': 'UTC'
    }
    
    @staticmethod
    def get_market_time(market: str) -> datetime:
        """시장별 현재 시간"""
        try:
            import pytz
            
            timezone_map = {
                'KR': 'Asia/Seoul',
                'US': 'America/New_York',
                'JP': 'Asia/Tokyo',
                'IN': 'Asia/Kolkata'
            }
            
            tz_name = timezone_map.get(market.upper(), 'UTC')
            tz = pytz.timezone(tz_name)
            
            return datetime.now(tz)
            
        except ImportError:
            # pytz가 없으면 기본 시간 사용
            return datetime.now()
    
    @staticmethod
    def is_market_open(market: str, current_time: datetime = None) -> bool:
        """시장 오픈 여부 확인"""
        if current_time is None:
            current_time = TimeUtils.get_market_time(market)
        
        weekday = current_time.weekday()  # 0=월요일, 6=일요일
        hour = current_time.hour
        minute = current_time.minute
        
        # 주말은 모든 시장 휴장
        if weekday >= 5:  # 토요일, 일요일
            return False
        
        if market.upper() == 'KR':
            # 한국 시장: 09:00 - 15:30
            return (9 <= hour < 15) or (hour == 15 and minute <= 30)
        
        elif market.upper() == 'US':
            # 미국 시장: 09:30 - 16:00 (EST)
            return (9 <= hour < 16) or (hour == 9 and minute >= 30)
        
        elif market.upper() == 'JP':
            # 일본 시장: 09:00 - 11:30, 12:30 - 15:00
            return ((9 <= hour < 11) or (hour == 11 and minute <= 30) or 
                    (12 <= hour < 15) or (hour == 12 and minute >= 30))
        
        elif market.upper() == 'IN':
            # 인도 시장: 09:15 - 15:30
            return ((9 <= hour < 15) or (hour == 9 and minute >= 15) or 
                    (hour == 15 and minute <= 30))
        
        elif market.upper() == 'CRYPTO':
            # 암호화폐: 24시간
            return True
        
        return False
    
    @staticmethod
    def get_next_market_open(market: str) -> datetime:
        """다음 시장 개장 시간"""
        current_time = TimeUtils.get_market_time(market)
        
        # 이미 개장 중이면 다음 개장일
        if TimeUtils.is_market_open(market, current_time):
            current_time += timedelta(days=1)
        
        # 다음 평일 찾기
        while current_time.weekday() >= 5:  # 주말 스킵
            current_time += timedelta(days=1)
        
        # 시장별 개장 시간 설정
        if market.upper() == 'KR':
            open_time = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
        elif market.upper() == 'US':
            open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        elif market.upper() == 'JP':
            open_time = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
        elif market.upper() == 'IN':
            open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            open_time = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
        
        return open_time
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """시간 지속 시간을 사람이 읽기 쉬운 형태로"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            return f"{seconds/60:.1f}분"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}시간"
        else:
            return f"{seconds/86400:.1f}일"
    
    @staticmethod
    def get_trading_session_info(market: str) -> Dict[str, Any]:
        """거래 세션 정보"""
        current_time = TimeUtils.get_market_time(market)
        is_open = TimeUtils.is_market_open(market, current_time)
        next_open = TimeUtils.get_next_market_open(market)
        
        if is_open:
            # 마감까지 남은 시간 계산
            if market.upper() == 'KR':
                close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
            elif market.upper() == 'US':
                close_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            elif market.upper() == 'JP':
                if current_time.hour < 12:
                    close_time = current_time.replace(hour=11, minute=30, second=0, microsecond=0)
                else:
                    close_time = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
            elif market.upper() == 'IN':
                close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
            else:
                close_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            time_to_close = (close_time - current_time).total_seconds()
        else:
            time_to_close = 0
        
        time_to_open = (next_open - current_time).total_seconds()
        
        return {
            'market': market.upper(),
            'current_time': current_time,
            'is_open': is_open,
            'next_open': next_open,
            'time_to_open': time_to_open,
            'time_to_close': time_to_close,
            'time_to_open_formatted': TimeUtils.format_duration(time_to_open),
            'time_to_close_formatted': TimeUtils.format_duration(time_to_close) if time_to_close > 0 else "마감됨"
        }

# ============================================================================
# 💾 데이터 처리 유틸리티
# ============================================================================

class DataUtils:
    """데이터 처리 유틸리티"""
    
    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """안전한 float 변환"""
        try:
            if isinstance(value, str):
                # 쉼표 제거 및 공백 제거
                value = value.replace(',', '').strip()
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        """안전한 int 변환"""
        try:
            return int(DataUtils.safe_float(value, default))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def clean_symbol(symbol: str) -> str:
        """심볼 정리 (공백, 특수문자 제거)"""
        if not symbol:
            return ""
        
        # 기본 정리
        cleaned = symbol.upper().strip()
        
        # 특수 케이스 처리
        cleaned = cleaned.replace('.KS', '')  # 한국 주식
        cleaned = cleaned.replace('.KQ', '')  # 코스닥
        cleaned = cleaned.replace('-USD', '')  # 암호화폐
        cleaned = cleaned.replace('KRW-', '')  # 업비트
        
        return cleaned
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_percent: float, 
                              entry_price: float, stop_loss_price: float) -> float:
        """포지션 사이즈 계산 (리스크 기반)"""
        if stop_loss_price <= 0 or entry_price <= 0 or entry_price == stop_loss_price:
            return 0
        
        # 위험 금액 계산
        risk_amount = account_balance * (risk_percent / 100)
        
        # 주당 손실 금액
        loss_per_share = abs(entry_price - stop_loss_price)
        
        # 포지션 사이즈
        position_size = risk_amount / loss_per_share
        
        return position_size
    
    @staticmethod
    def normalize_data(data: List[float]) -> List[float]:
        """데이터 정규화 (0-1)"""
        if not data:
            return []
        
        min_val = min(data)
        max_val = max(data)
        
        if min_val == max_val:
            return [0.5] * len(data)
        
        return [(x - min_val) / (max_val - min_val) for x in data]
    
    @staticmethod
    def moving_average_crossover(fast_ma: List[float], slow_ma: List[float]) -> List[int]:
        """이동평균 교차 신호 (1: 골든크로스, -1: 데드크로스, 0: 신호없음)"""
        if len(fast_ma) != len(slow_ma) or len(fast_ma) < 2:
            return []
        
        signals = [0]  # 첫 번째는 항상 0
        
        for i in range(1, len(fast_ma)):
            prev_fast, prev_slow = fast_ma[i-1], slow_ma[i-1]
            curr_fast, curr_slow = fast_ma[i], slow_ma[i]
            
            # 골든크로스: 빠른 MA가 느린 MA를 상향 돌파
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                signals.append(1)
            # 데드크로스: 빠른 MA가 느린 MA를 하향 돌파
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                signals.append(-1)
            else:
                signals.append(0)
        
        return signals

# ============================================================================
# 🎨 시각화 유틸리티
# ============================================================================

class VisualizationUtils:
    """시각화 관련 유틸리티"""
    
    @staticmethod
    def create_price_chart_ascii(prices: List[float], width: int = 50, height: int = 10) -> str:
        """ASCII 가격 차트 생성"""
        if not prices or len(prices) < 2:
            return "데이터 부족"
        
        # 가격 정규화
        min_price = min(prices)
        max_price = max(prices)
        
        if min_price == max_price:
            return "가격 변동 없음"
        
        # 차트 생성
        chart_lines = []
        
        for row in range(height):
            line = ""
            threshold = max_price - (row / height) * (max_price - min_price)
            
            for i, price in enumerate(prices[-width:]):
                if price >= threshold:
                    line += "█"
                else:
                    line += " "
            
            chart_lines.append(line)
        
        # 정보 추가
        chart = f"Price Chart (Last {min(len(prices), width)} periods)\n"
        chart += f"High: {max_price:.2f} " + "█" * width + f" {max_price:.2f}\n"
        
        for line in chart_lines:
            chart += "     " + line + "\n"
        
        chart += f" Low: {min_price:.2f} " + "█" * width + f" {min_price:.2f}\n"
        chart += f"Latest: {prices[-1]:.2f} | Change: {((prices[-1] - prices[0]) / prices[0] * 100):+.2f}%"
        
        return chart
    
    @staticmethod
    def create_performance_table(metrics: PerformanceMetrics) -> str:
        """성과 지표 테이블 생성"""
        table = """
📊 Performance Metrics
═══════════════════════════════════════════
│ Metric              │ Value              │
├─────────────────────┼────────────────────┤"""
        
        metrics_data = [
            ("Total Return", f"{metrics.total_return:.2%}"),
            ("Annualized Return", f"{metrics.annualized_return:.2%}"),
            ("Volatility", f"{metrics.volatility:.2%}"),
            ("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}"),
            ("Sortino Ratio", f"{metrics.sortino_ratio:.2f}"),
            ("Max Drawdown", f"{metrics.max_drawdown:.2%}"),
            ("Win Rate", f"{metrics.win_rate:.2%}"),
            ("Profit Factor", f"{metrics.profit_factor:.2f}"),
            ("Average Win", f"{metrics.avg_win:.2%}"),
            ("Average Loss", f"{metrics.avg_loss:.2%}"),
            ("Total Trades", f"{metrics.total_trades:,}")
        ]
        
        for metric, value in metrics_data:
            table += f"\n│ {metric:<19} │ {value:>18} │"
        
        table += "\n═══════════════════════════════════════════"
        
        return table

# ============================================================================
# 🔧 시스템 유틸리티
# ============================================================================

class SystemUtils:
    """시스템 관련 유틸리티"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """시스템 정보 수집"""
        try:
            info = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'hostname': platform.node(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent,
                    'used': psutil.virtual_memory().used,
                    'free': psutil.virtual_memory().free
                },
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                },
                'boot_time': datetime.fromtimestamp(psutil.boot_time()),
                'uptime': datetime.now() - datetime.fromtimestamp(psutil.boot_time())
            }
            
            return info
            
        except Exception as e:
            logging.error(f"시스템 정보 수집 실패: {e}")
            return {}
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """의존성 모듈 체크"""
        dependencies = {
            'numpy': False,
            'pandas': False,
            'aiohttp': False,
            'requests': False,
            'psutil': False,
            'yaml': False,
            'sqlite3': False,
            'ib_insync': False,
            'pyupbit': False,
            'pytz': False
        }
        
        for module in dependencies:
            try:
                __import__(module)
                dependencies[module] = True
            except ImportError:
                dependencies[module] = False
        
        return dependencies
    
    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """바이트를 사람이 읽기 쉬운 형태로"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    @staticmethod
    def get_resource_usage() -> Dict[str, Any]:
        """리소스 사용량 조회"""
        try:
            # 현재 프로세스
            process = psutil.Process()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_info': process.memory_info(),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
                'create_time': datetime.fromtimestamp(process.create_time()),
                'runtime': datetime.now() - datetime.fromtimestamp(process.create_time())
            }
            
        except Exception as e:
            logging.error(f"리소스 사용량 조회 실패: {e}")
            return {}

# ============================================================================
# 🔄 재시도 및 회복력 유틸리티
# ============================================================================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logging.warning(f"함수 {func.__name__} 실패 (시도 {attempt + 1}/{max_retries + 1}), {wait_time}초 후 재시도: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"함수 {func.__name__} 최종 실패: {e}")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logging.warning(f"함수 {func.__name__} 실패 (시도 {attempt + 1}/{max_retries + 1}), {wait_time}초 후 재시도: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"함수 {func.__name__} 최종 실패: {e}")
            
            raise last_exception
        
        # 비동기 함수인지 확인
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# ============================================================================
# 🎯 전략 유틸리티
# ============================================================================

class StrategyUtils:
    """전략 관련 유틸리티"""
    
    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """켈리 기준 계산"""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # 안전을 위해 25% 이하로 제한
        return min(max(kelly_fraction, 0), 0.25)
    
    @staticmethod
    def calculate_compound_growth(initial_amount: float, monthly_return: float, months: int) -> Dict[str, float]:
        """복리 성장 계산"""
        final_amount = initial_amount * ((1 + monthly_return) ** months)
        total_return = final_amount - initial_amount
        annualized_return = ((final_amount / initial_amount) ** (12 / months)) - 1
        
        return {
            'initial_amount': initial_amount,
            'final_amount': final_amount,
            'total_return': total_return,
            'total_return_pct': (total_return / initial_amount) * 100,
            'annualized_return_pct': annualized_return * 100,
            'months': months
        }
    
    @staticmethod
    def calculate_diversification_score(positions: List[Dict]) -> float:
        """다각화 점수 계산 (0-1)"""
        if not positions:
            return 0
        
        total_value = sum(pos.get('value', 0) for pos in positions)
        if total_value <= 0:
            return 0
        
        # HHI (Herfindahl-Hirschman Index) 계산
        hhi = sum((pos.get('value', 0) / total_value) ** 2 for pos in positions)
        
        # 정규화된 다각화 점수 (1 - normalized_hhi)
        max_hhi = 1.0  # 모든 자금이 하나의 포지션에 집중된 경우
        min_hhi = 1.0 / len(positions)  # 완전히 균등 분산된 경우
        
        if max_hhi == min_hhi:
            return 1.0
        
        normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi)
        diversification_score = 1 - normalized_hhi
        
        return max(0, min(1, diversification_score))

# ============================================================================
# 📊 로깅 유틸리티
# ============================================================================

class LoggingUtils:
    """로깅 관련 유틸리티"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: int = logging.INFO, 
                    max_bytes: int = 10*1024*1024, backup_count: int = 5) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 이미 핸들러가 있으면 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 (로테이션)
        if log_file:
            from logging.handlers import RotatingFileHandler
            
            # 로그 디렉토리 생성
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_function_call(func):
        """함수 호출 로깅 데코레이터"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            start_time = time.time()
            logger.debug(f"함수 시작: {func.__name__}({args}, {kwargs})")
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"함수 완료: {func.__name__} ({duration:.3f}초)")
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"함수 실패: {func.__name__} ({duration:.3f}초): {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            start_time = time.time()
            logger.debug(f"함수 시작: {func.__name__}({args}, {kwargs})")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"함수 완료: {func.__name__} ({duration:.3f}초)")
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"함수 실패: {func.__name__} ({duration:.3f}초): {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    @staticmethod
    def create_trade_log_entry(signal: TradingSignal, execution_price: float, 
                              quantity: float, commission: float = 0) -> Dict[str, Any]:
        """거래 로그 엔트리 생성"""
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal.symbol,
            'strategy': signal.strategy,
            'action': signal.action,
            'signal_price': signal.price,
            'execution_price': execution_price,
            'quantity': quantity,
            'commission': commission,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'slippage': execution_price - signal.price,
            'slippage_pct': ((execution_price - signal.price) / signal.price) * 100 if signal.price > 0 else 0
        }

# ============================================================================
# 🌟 메인 유틸리티 클래스
# ============================================================================

class QuintUtils:
    """퀸트프로젝트 통합 유틸리티"""
    
    def __init__(self):
        self.currency = CurrencyUtils()
        self.technical = TechnicalIndicators()
        self.performance = PerformanceAnalyzer()
        self.network = NetworkUtils()
        self.file = FileUtils()
        self.notification = NotificationFormatter()
        self.security = SecurityUtils()
        self.time = TimeUtils()
        self.data = DataUtils()
        self.visualization = VisualizationUtils()
        self.system = SystemUtils()
        self.strategy = StrategyUtils()
        self.logging = LoggingUtils()
    
    def get_system_status(self) -> Dict[str, Any]:
        """전체 시스템 상태 조회"""
        return {
            'system_info': self.system.get_system_info(),
            'resource_usage': self.system.get_resource_usage(),
            'dependencies': self.system.check_dependencies(),
            'network_info': self.network.get_network_info(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """종합 헬스 체크"""
        health = {
            'overall_status': 'healthy',
            'checks': {}
        }
        
        try:
            # 네트워크 연결 체크
            network_result = await self.network.check_internet_connection()
            health['checks']['network'] = {
                'status': 'ok' if network_result['connected'] else 'error',
                'latency': network_result['latency'],
                'details': network_result
            }
            
            # 시스템 리소스 체크
            resource_usage = self.system.get_resource_usage()
            memory_ok = resource_usage.get('memory_percent', 0) < 80
            health['checks']['resources'] = {
                'status': 'ok' if memory_ok else 'warning',
                'memory_percent': resource_usage.get('memory_percent', 0),
                'details': resource_usage
            }
            
            # 의존성 체크
            dependencies = self.system.check_dependencies()
            critical_deps = ['numpy', 'pandas', 'aiohttp', 'requests']
            deps_ok = all(dependencies.get(dep, False) for dep in critical_deps)
            health['checks']['dependencies'] = {
                'status': 'ok' if deps_ok else 'error',
                'missing': [dep for dep in critical_deps if not dependencies.get(dep, False)],
                'details': dependencies
            }
            
            # 전체 상태 결정
            if not network_result['connected']:
                health['overall_status'] = 'critical'
            elif not deps_ok:
                health['overall_status'] = 'error'
            elif not memory_ok:
                health['overall_status'] = 'warning'
            
        except Exception as e:
            health['overall_status'] = 'error'
            health['error'] = str(e)
        
        return health

# ============================================================================
# 🧪 테스트 유틸리티
# ============================================================================

class TestUtils:
    """테스트 관련 유틸리티"""
    
    @staticmethod
    def generate_mock_price_data(days: int = 30, start_price: float = 100.0, 
                                volatility: float = 0.02) -> List[Dict[str, Any]]:
        """모의 가격 데이터 생성"""
        import random
        
        data = []
        current_price = start_price
        current_date = datetime.now() - timedelta(days=days)
        
        for _ in range(days):
            # 랜덤 워크
            change = random.gauss(0, volatility)
            current_price *= (1 + change)
            
            # 거래량도 랜덤 생성
            volume = random.randint(10000, 1000000)
            
            data.append({
                'date': current_date,
                'price': round(current_price, 2),
                'volume': volume,
                'change_pct': change * 100
            })
            
            current_date += timedelta(days=1)
        
        return data
    
    @staticmethod
    def generate_mock_signals(count: int = 10) -> List[TradingSignal]:
        """모의 거래 신호 생성"""
        import random
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        strategies = ['momentum', 'mean_reversion', 'breakout', 'rsi_divergence']
        actions = ['BUY', 'SELL', 'HOLD']
        
        signals = []
        
        for _ in range(count):
            signal = TradingSignal(
                symbol=random.choice(symbols),
                strategy=random.choice(strategies),
                action=random.choice(actions),
                price=round(random.uniform(50, 500), 2),
                confidence=round(random.uniform(0.6, 0.95), 2),
                reason=f"Technical indicator signal",
                timestamp=datetime.now() - timedelta(minutes=random.randint(1, 1440)),
                target_price=round(random.uniform(60, 600), 2) if random.random() > 0.5 else None,
                stop_loss=round(random.uniform(40, 450), 2) if random.random() > 0.5 else None,
                position_size=round(random.uniform(100, 10000), 2),
                currency='USD'
            )
            signals.append(signal)
        
        return signals
    
    @staticmethod
    def validate_trading_signal(signal: TradingSignal) -> Dict[str, Any]:
        """거래 신호 검증"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 필수 필드 체크
        if not signal.symbol:
            validation['errors'].append("Symbol is required")
        
        if signal.action not in ['BUY', 'SELL', 'HOLD']:
            validation['errors'].append("Invalid action")
        
        if signal.price <= 0:
            validation['errors'].append("Price must be positive")
        
        if not (0 <= signal.confidence <= 1):
            validation['errors'].append("Confidence must be between 0 and 1")
        
        # 경고 체크
        if signal.confidence < 0.7:
            validation['warnings'].append("Low confidence signal")
        
        if signal.target_price and signal.action == 'BUY' and signal.target_price <= signal.price:
            validation['warnings'].append("Target price should be higher than entry price for BUY signal")
        
        if signal.stop_loss and signal.action == 'BUY' and signal.stop_loss >= signal.price:
            validation['warnings'].append("Stop loss should be lower than entry price for BUY signal")
        
        validation['is_valid'] = len(validation['errors']) == 0
        
        return validation

# ============================================================================
# 🔍 디버깅 유틸리티
# ============================================================================

class DebugUtils:
    """디버깅 관련 유틸리티"""
    
    @staticmethod
    def profile_function(func):
        """함수 실행 시간 프로파일링 데코레이터"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import cProfile
            import pstats
            from io import StringIO
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                profiler.disable()
                
                # 결과 출력
                stream = StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats('cumulative').print_stats(10)
                
                print(f"\n=== Profile for {func.__name__} ===")
                print(stream.getvalue())
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import cProfile
            import pstats
            from io import StringIO
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.disable()
                
                # 결과 출력
                stream = StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats('cumulative').print_stats(10)
                
                print(f"\n=== Profile for {func.__name__} ===")
                print(stream.getvalue())
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    @staticmethod
    def memory_usage(func):
        """메모리 사용량 모니터링 데코레이터"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            process = psutil.Process()
            
            # 실행 전 메모리
            memory_before = process.memory_info().rss
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                # 실행 후 메모리
                memory_after = process.memory_info().rss
                memory_diff = memory_after - memory_before
                
                print(f"Memory usage for {func.__name__}:")
                print(f"  Before: {SystemUtils.format_bytes(memory_before)}")
                print(f"  After:  {SystemUtils.format_bytes(memory_after)}")
                print(f"  Diff:   {SystemUtils.format_bytes(memory_diff)}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            process = psutil.Process()
            
            # 실행 전 메모리
            memory_before = process.memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 실행 후 메모리
                memory_after = process.memory_info().rss
                memory_diff = memory_after - memory_before
                
                print(f"Memory usage for {func.__name__}:")
                print(f"  Before: {SystemUtils.format_bytes(memory_before)}")
                print(f"  After:  {SystemUtils.format_bytes(memory_after)}")
                print(f"  Diff:   {SystemUtils.format_bytes(memory_diff)}")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    @staticmethod
    def debug_print_vars(**kwargs):
        """변수 디버그 출력"""
        print("\n=== Debug Variables ===")
        for name, value in kwargs.items():
            print(f"{name}: {type(value).__name__} = {repr(value)}")
        print("=" * 25)
    
    @staticmethod
    def trace_calls(func):
        """함수 호출 추적 데코레이터"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            print(f"→ Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = await func(*args, **kwargs)
                print(f"← {func.__name__} returned: {type(result).__name__}")
                return result
            except Exception as e:
                print(f"← {func.__name__} raised: {type(e).__name__}: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            print(f"→ Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                print(f"← {func.__name__} returned: {type(result).__name__}")
                return result
            except Exception as e:
                print(f"← {func.__name__} raised: {type(e).__name__}: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

# ============================================================================
# 🌐 API 유틸리티
# ============================================================================

class APIUtils:
    """API 관련 유틸리티"""
    
    @staticmethod
    async def make_api_request(url: str, method: str = 'GET', headers: Dict = None, 
                              data: Dict = None, timeout: int = 30, 
                              retries: int = 3) -> Optional[Dict]:
        """안전한 API 요청"""
        headers = headers or {}
        
        for attempt in range(retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=method,
                        url=url,
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limit
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logging.error(f"API 요청 실패: {response.status} - {url}")
                            return None
            
            except asyncio.TimeoutError:
                logging.warning(f"API 타임아웃 (시도 {attempt + 1}/{retries + 1}): {url}")
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
            
            except Exception as e:
                logging.error(f"API 요청 오류 (시도 {attempt + 1}/{retries + 1}): {e}")
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    @staticmethod
    def create_api_signature(secret_key: str, message: str, algorithm: str = 'sha256') -> str:
        """API 서명 생성"""
        import hmac
        
        if algorithm == 'sha256':
            signature = hmac.new(
                secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return signature
    
    @staticmethod
    def rate_limit(calls_per_second: float):
        """Rate limiting 데코레이터"""
        min_interval = 1.0 / calls_per_second
        last_called = [0.0]
        
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                now = time.time()
                time_since_last = now - last_called[0]
                
                if time_since_last < min_interval:
                    await asyncio.sleep(min_interval - time_since_last)
                
                last_called[0] = time.time()
                return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                now = time.time()
                time_since_last = now - last_called[0]
                
                if time_since_last < min_interval:
                    time.sleep(min_interval - time_since_last)
                
                last_called[0] = time.time()
                return func(*args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator

# ============================================================================
# 📈 백테스팅 유틸리티
# ============================================================================

class BacktestUtils:
    """백테스팅 관련 유틸리티"""
    
    @staticmethod
    def run_simple_backtest(price_data: List[float], signals: List[int], 
                          initial_capital: float = 100000, 
                          commission: float = 0.001) -> Dict[str, Any]:
        """간단한 백테스트 실행"""
        if len(price_data) != len(signals):
            raise ValueError("Price data and signals must have same length")
        
        portfolio_value = [initial_capital]
        position = 0  # 현재 포지션 (주식 수)
        cash = initial_capital
        trades = []
        
        for i in range(len(signals)):
            current_price = price_data[i]
            signal = signals[i]
            
            if signal == 1 and position == 0:  # 매수 신호
                # 전액 매수
                shares_to_buy = cash // current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    commission_cost = cost * commission
                    
                    if cash >= cost + commission_cost:
                        position = shares_to_buy
                        cash -= (cost + commission_cost)
                        
                        trades.append({
                            'date_index': i,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'commission': commission_cost
                        })
            
            elif signal == -1 and position > 0:  # 매도 신호
                # 전량 매도
                proceeds = position * current_price
                commission_cost = proceeds * commission
                
                cash += (proceeds - commission_cost)
                
                trades.append({
                    'date_index': i,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'commission': commission_cost
                })
                
                position = 0
            
            # 포트폴리오 가치 계산
            current_value = cash + (position * current_price)
            portfolio_value.append(current_value)
        
        # 성과 계산
        final_value = portfolio_value[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # 수익률 시계열
        returns = []
        for i in range(1, len(portfolio_value)):
            ret = (portfolio_value[i] - portfolio_value[i-1]) / portfolio_value[i-1]
            returns.append(ret)
        
        # 성과 지표 계산
        metrics = PerformanceAnalyzer.calculate_metrics(returns)
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'portfolio_value': portfolio_value,
            'trades': trades,
            'total_trades': len(trades),
            'metrics': metrics,
            'commission_paid': sum(trade['commission'] for trade in trades)
        }
    
    @staticmethod
    def calculate_benchmark_comparison(strategy_returns: List[float], 
                                     benchmark_returns: List[float]) -> Dict[str, float]:
        """벤치마크 대비 성과 비교"""
        if len(strategy_returns) != len(benchmark_returns):
            raise ValueError("Strategy and benchmark returns must have same length")
        
        strategy_cumulative = np.prod([1 + r for r in strategy_returns]) - 1
        benchmark_cumulative = np.prod([1 + r for r in benchmark_returns]) - 1
        
        excess_return = strategy_cumulative - benchmark_cumulative
        
        # 베타 계산
        beta = PerformanceAnalyzer.calculate_beta(strategy_returns, benchmark_returns)
        
        # 알파 계산 (CAPM)
        risk_free_rate = 0.03  # 3% 가정
        alpha = strategy_cumulative - (risk_free_rate + beta * (benchmark_cumulative - risk_free_rate))
        
        # 상관계수
        correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
        
        return {
            'strategy_return': strategy_cumulative,
            'benchmark_return': benchmark_cumulative,
            'excess_return': excess_return,
            'alpha': alpha,
            'beta': beta,
            'correlation': correlation
        }

# ============================================================================
# 🎨 차트 생성 유틸리티
# ============================================================================

class ChartUtils:
    """차트 생성 유틸리티 (텍스트 기반)"""
    
    @staticmethod
    def create_sparkline(data: List[float], width: int = 20) -> str:
        """스파크라인 생성"""
        if not data:
            return ""
        
        # 데이터 정규화
        min_val = min(data)
        max_val = max(data)
        
        if min_val == max_val:
            return "─" * width
        
        # 8단계 막대 문자
        bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        # 데이터를 width 길이로 샘플링
        step = len(data) / width
        sampled_data = [data[int(i * step)] for i in range(width)]
        
        # 정규화 및 막대 변환
        sparkline = ""
        for value in sampled_data:
            normalized = (value - min_val) / (max_val - min_val)
            bar_index = min(int(normalized * len(bars)), len(bars) - 1)
            sparkline += bars[bar_index]
        
        return sparkline
    
    @staticmethod
    def create_trend_indicator(current: float, previous: float) -> str:
        """트렌드 표시기"""
        if current > previous:
            return "📈"
        elif current < previous:
            return "📉"
        else:
            return "➡️"
    
    @staticmethod
    def create_gauge(value: float, min_val: float = 0, max_val: float = 100, 
                    width: int = 20) -> str:
        """게이지 차트 생성"""
        if max_val <= min_val:
            return "Invalid range"
        
        # 정규화
        normalized = max(0, min(1, (value - min_val) / (max_val - min_val)))
        filled_length = int(normalized * width)
        
        # 게이지 생성
        gauge = "["
        gauge += "█" * filled_length
        gauge += "░" * (width - filled_length)
        gauge += f"] {value:.1f}"
        
        return gauge

# ============================================================================
# 🔧 설정 관리 유틸리티
# ============================================================================

class ConfigUtils:
    """설정 관리 유틸리티"""
    
    @staticmethod
    def load_config(config_path: str, default_config: Dict = None) -> Dict:
        """설정 파일 로드"""
        config_path = Path(config_path)
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                        return yaml.safe_load(f)
                    elif config_path.suffix.lower() == '.json':
                        return json.load(f)
                    else:
                        logging.warning(f"지원되지 않는 설정 파일 형식: {config_path.suffix}")
                        return default_config or {}
            except Exception as e:
                logging.error(f"설정 파일 로드 실패: {e}")
                return default_config or {}
        else:
            logging.warning(f"설정 파일 없음: {config_path}")
            if default_config:
                ConfigUtils.save_config(config_path, default_config)
            return default_config or {}
    
    @staticmethod
    def save_config(config_path: str, config: Dict):
        """설정 파일 저장"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    logging.warning(f"지원되지 않는 설정 파일 형식: {config_path.suffix}")
        except Exception as e:
            logging.error(f"설정 파일 저장 실패: {e}")
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """설정 병합 (재귀적)"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigUtils.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config(config: Dict, schema: Dict) -> Dict[str, List[str]]:
        """설정 검증 (간단한 스키마 기반)"""
        errors = []
        warnings = []
        
        def validate_recursive(cfg: Dict, sch: Dict, path: str = ""):
            for key, expected_type in sch.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in cfg:
                    if isinstance(expected_type, dict) and 'required' in expected_type and expected_type['required']:
                        errors.append(f"필수 설정 누락: {current_path}")
                    continue
                
                value = cfg[key]
                
                if isinstance(expected_type, dict):
                    if 'type' in expected_type:
                        expected_python_type = expected_type['type']
                        if not isinstance(value, expected_python_type):
                            errors.append(f"타입 오류 {current_path}: {type(value).__name__} (예상: {expected_python_type.__name__})")
                    
                    if 'min' in expected_type and isinstance(value, (int, float)) and value < expected_type['min']:
                        errors.append(f"값이 너무 작음 {current_path}: {value} < {expected_type['min']}")
                    
                    if 'max' in expected_type and isinstance(value, (int, float)) and value > expected_type['max']:
                        errors.append(f"값이 너무 큼 {current_path}: {value} > {expected_type['max']}")
                    
                    if 'choices' in expected_type and value not in expected_type['choices']:
                        errors.append(f"잘못된 선택 {current_path}: {value} (가능한 값: {expected_type['choices']})")
                
                elif isinstance(expected_type, type):
                    if not isinstance(value, expected_type):
                        errors.append(f"타입 오류 {current_path}: {type(value).__name__} (예상: {expected_type.__name__})")
                
                elif isinstance(expected_type, dict) and isinstance(value, dict):
                    validate_recursive(value, expected_type, current_path)
        
        validate_recursive(config, schema)
        
        return {'errors': errors, 'warnings': warnings}

# ============================================================================
# 🚀 실행 및 초기화
# ============================================================================

def initialize_quint_utils(log_level: int = logging.INFO) -> QuintUtils:
    """퀸트 유틸리티 초기화"""
    # 기본 로거 설정
    LoggingUtils.setup_logger(
        name='quint_utils',
        log_file='./logs/quint_utils.log',
        level=log_level
    )
    
    # 유틸리티 인스턴스 생성
    utils = QuintUtils()
    
    # 초기화 로그
    logger = logging.getLogger('quint_utils')
    logger.info("🏆 퀸트프로젝트 유틸리티 모듈 초기화 완료")
    
    return utils

# 전역 유틸리티 인스턴스 (편의용)
utils = None

def get_utils() -> QuintUtils:
    """전역 유틸리티 인스턴스 조회"""
    global utils
    if utils is None:
        utils = initialize_quint_utils()
    return utils

# ============================================================================
# 📝 사용 예시 및 테스트
# ============================================================================

if __name__ == "__main__":
    # 기본 사용법 예시
    async def main():
        print("🏆 퀸트프로젝트 유틸리티 모듈 테스트")
        print("=" * 50)
        
        # 유틸리티 초기화
        quint_utils = initialize_quint_utils(logging.INFO)
        
        # 시스템 상태 체크
        print("🔍 시스템 상태 체크...")
        system_status = quint_utils.get_system_status()
        print(f"  플랫폼: {system_status['system_info'].get('platform', 'Unknown')}")
        print(f"  메모리 사용률: {system_status['resource_usage'].get('memory_percent', 0):.1f}%")
        
        # 헬스 체크
        print("\n🏥 헬스 체크...")
        health = await quint_utils.health_check()
        print(f"  전체 상태: {health['overall_status']}")
        for check_name, check_result in health['checks'].items():
            print(f"  {check_name}: {check_result['status']}")
        
        # 기술적 지표 테스트
        print("\n📈 기술적 지표 테스트...")
        test_prices = [100, 102, 98, 105, 110, 108, 112, 115, 113, 118]
        sma = quint_utils.technical.sma(test_prices, 5)
        rsi = quint_utils.technical.rsi(test_prices)
        print(f"  SMA(5): {sma[-1] if sma else 'N/A':.2f}")
        print(f"  RSI: {rsi[-1] if rsi else 'N/A':.2f}")
        
        # 환율 변환 테스트
        print("\n💱 환율 변환 테스트...")
        await quint_utils.currency._update_rates_if_needed()
        krw_amount = await quint_utils.currency.convert_currency(100, 'USD', 'KRW')
        print(f"  $100 = ₩{krw_amount:,.0f}")
        
        # 모의 신호 생성 및 검증
        print("\n🎯 거래 신호 테스트...")
        test_signals = TestUtils.generate_mock_signals(3)
        for i, signal in enumerate(test_signals):
            validation = TestUtils.validate_trading_signal(signal)
            print(f"  신호 {i+1}: {signal.symbol} {signal.action} - {'✅' if validation['is_valid'] else '❌'}")
        
        # 성과 분석 테스트
        print("\n📊 성과 분석 테스트...")
        test_returns = [0.02, -0.01, 0.015, -0.005, 0.03, -0.02, 0.01, 0.025, -0.015, 0.02]
        metrics = quint_utils.performance.calculate_metrics(test_returns)
        print(f"  총 수익률: {metrics.total_return:.2%}")
        print(f"  샤프 비율: {metrics.sharpe_ratio:.2f}")
        print(f"  최대 낙폭: {metrics.max_drawdown:.2%}")
        
        # 백테스트 테스트
        print("\n🔄 백테스트 테스트...")
        test_prices = [100, 102, 98, 105, 110, 108, 112, 115, 113, 118, 120, 115, 122, 125, 120]
        test_signals = [0, 1, 0, 0, 0, -1, 1, 0, 0, 0, -1, 1, 0, 0, -1]
        backtest_result = BacktestUtils.run_simple_backtest(test_prices, test_signals, 100000, 0.001)
        print(f"  초기 자본: ₩{backtest_result['initial_capital']:,.0f}")
        print(f"  최종 가치: ₩{backtest_result['final_value']:,.0f}")
        print(f"  총 수익률: {backtest_result['total_return_pct']:+.2f}%")
        print(f"  총 거래: {backtest_result['total_trades']}회")
        
        # 시각화 테스트
        print("\n🎨 시각화 테스트...")
        sparkline = quint_utils.visualization.create_sparkline(test_prices, 20)
        gauge = quint_utils.visualization.create_gauge(75, 0, 100, 20)
        print(f"  가격 스파크라인: {sparkline}")
        print(f"  진행률 게이지: {gauge}")
        
        # 네트워크 테스트
        print("\n🌐 네트워크 테스트...")
        network_status = await quint_utils.network.check_internet_connection()
        print(f"  연결 상태: {'✅' if network_status['connected'] else '❌'}")
        print(f"  지연시간: {network_status['latency']:.1f}ms")
        print(f"  성공한 호스트: {network_status['successful_hosts']}/{network_status['total_hosts']}")
        
        # 시간 유틸리티 테스트
        print("\n⏰ 시간 유틸리티 테스트...")
        markets = ['KR', 'US', 'JP', 'IN', 'CRYPTO']
        for market in markets:
            session_info = quint_utils.time.get_trading_session_info(market)
            status = "🟢 열림" if session_info['is_open'] else "🔴 닫힘"
            print(f"  {market} 시장: {status}")
        
        # 보안 유틸리티 테스트
        print("\n🔐 보안 유틸리티 테스트...")
        api_key = quint_utils.security.generate_api_key(16)
        masked_key = quint_utils.security.mask_sensitive_info(api_key, '*', 4)
        print(f"  생성된 API 키: {masked_key}")
        
        password = "test_password_123"
        hashed, salt = quint_utils.security.hash_password(password)
        is_valid = quint_utils.security.verify_password(password, hashed, salt)
        print(f"  비밀번호 검증: {'✅' if is_valid else '❌'}")
        
        # 파일 유틸리티 테스트
        print("\n📁 파일 유틸리티 테스트...")
        test_dir = Path("./test_data")
        quint_utils.file.ensure_directory(test_dir)
        print(f"  테스트 디렉토리 생성: {test_dir.exists()}")
        
        # 데이터 유틸리티 테스트
        print("\n📊 데이터 유틸리티 테스트...")
        test_values = ["1,234.56", "invalid", "789", None, ""]
        safe_floats = [quint_utils.data.safe_float(v, 0.0) for v in test_values]
        print(f"  안전한 float 변환: {safe_floats}")
        
        normalized_data = quint_utils.data.normalize_data([10, 20, 15, 30, 25])
        print(f"  정규화된 데이터: {[f'{x:.2f}' for x in normalized_data]}")
        
        # 전략 유틸리티 테스트
        print("\n🎯 전략 유틸리티 테스트...")
        kelly_fraction = quint_utils.strategy.calculate_kelly_criterion(0.6, 0.15, 0.10)
        print(f"  켈리 기준: {kelly_fraction:.2%}")
        
        compound_result = quint_utils.strategy.calculate_compound_growth(1000000, 0.06, 12)
        print(f"  복리 성장 (월 6%, 1년): {compound_result['final_amount']:,.0f}원")
        
        # 알림 포맷팅 테스트
        print("\n📱 알림 포맷팅 테스트...")
        test_signal = TradingSignal(
            symbol="AAPL",
            strategy="momentum",
            action="BUY",
            price=150.25,
            confidence=0.85,
            reason="Golden cross detected",
            timestamp=datetime.now(),
            target_price=160.00,
            stop_loss=145.00,
            currency="USD"
        )
        
        formatted_signal = quint_utils.notification.format_trading_signal(test_signal)
        print("  거래 신호 알림:")
        print("  " + "\n  ".join(formatted_signal.split('\n')))
        
        print("\n✅ 모든 테스트 완료!")
        print("=" * 50)
    
    # 비동기 메인 함수 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 📚 문서화 및 도움말
# ============================================================================

def show_help():
    """도움말 표시"""
    help_text = """
🏆 퀸트프로젝트 유틸리티 모듈 (utils.py)
=============================================

📋 주요 클래스 및 기능:

1. 🏢 QuintUtils - 통합 유틸리티 클래스
   └── 모든 유틸리티 기능에 대한 통합 접근점

2. 💱 CurrencyUtils - 환율 변환
   ├── convert_currency() - 통화 변환
   ├── format_currency() - 통화 포맷팅
   └── _fetch_exchange_rates() - 환율 정보 가져오기

3. 📈 TechnicalIndicators - 기술적 지표
   ├── sma() - 단순이동평균
   ├── ema() - 지수이동평균
   ├── rsi() - RSI 지표
   ├── bollinger_bands() - 볼린저 밴드
   └── macd() - MACD 지표

4. 📊 PerformanceAnalyzer - 성과 분석
   ├── calculate_returns() - 수익률 계산
   ├── calculate_metrics() - 성과 지표 계산
   ├── calculate_var() - VaR 계산
   └── calculate_beta() - 베타 계산

5. 🌐 NetworkUtils - 네트워크 유틸리티
   ├── check_internet_connection() - 인터넷 연결 체크
   ├── get_public_ip() - 공용 IP 조회
   └── get_network_info() - 네트워크 정보 수집

6. 📁 FileUtils - 파일 관리
   ├── ensure_directory() - 디렉토리 생성
   ├── backup_file() - 파일 백업
   ├── compress_file() - 파일 압축
   └── cleanup_old_files() - 오래된 파일 정리

7. 🔔 NotificationFormatter - 알림 포맷팅
   ├── format_trading_signal() - 거래 신호 포맷
   ├── format_portfolio_summary() - 포트폴리오 요약 포맷
   └── format_system_alert() - 시스템 알림 포맷

8. 🛡️ SecurityUtils - 보안 유틸리티
   ├── generate_api_key() - API 키 생성
   ├── hash_password() - 비밀번호 해시화
   ├── encrypt_sensitive_data() - 데이터 암호화
   └── mask_sensitive_info() - 민감 정보 마스킹

9. ⏰ TimeUtils - 시간 관리
   ├── get_market_time() - 시장별 현재 시간
   ├── is_market_open() - 시장 개장 여부
   ├── get_next_market_open() - 다음 개장 시간
   └── get_trading_session_info() - 거래 세션 정보

10. 💾 DataUtils - 데이터 처리
    ├── safe_float() / safe_int() - 안전한 타입 변환
    ├── clean_symbol() - 심볼 정리
    ├── calculate_position_size() - 포지션 사이즈 계산
    └── normalize_data() - 데이터 정규화

11. 🎨 VisualizationUtils - 시각화
    ├── create_price_chart_ascii() - ASCII 가격 차트
    └── create_performance_table() - 성과 테이블

12. 🔧 SystemUtils - 시스템 유틸리티
    ├── get_system_info() - 시스템 정보
    ├── check_dependencies() - 의존성 체크
    └── get_resource_usage() - 리소스 사용량

📝 사용 예시:

```python
# 유틸리티 초기화
from utils import get_utils
utils = get_utils()

# 환율 변환
krw_amount = await utils.currency.convert_currency(100, 'USD', 'KRW')

# 기술적 지표 계산
prices = [100, 102, 98, 105, 110]
sma = utils.technical.sma(prices, 3)
rsi = utils.technical.rsi(prices)

# 성과 분석
returns = [0.02, -0.01, 0.015, -0.005, 0.03]
metrics = utils.performance.calculate_metrics(returns)

# 시장 시간 체크
is_open = utils.time.is_market_open('KR')
session_info = utils.time.get_trading_session_info('US')

# 네트워크 상태 체크
network_status = await utils.network.check_internet_connection()

# 파일 관리
utils.file.ensure_directory('./data')
backup_path = utils.file.backup_file('./important_file.txt')

# 보안 기능
api_key = utils.security.generate_api_key()
hashed, salt = utils.security.hash_password('my_password')

# 데이터 처리
safe_value = utils.data.safe_float('1,234.56', 0)
normalized = utils.data.normalize_data([1, 2, 3, 4, 5])
```

🔧 데코레이터:

- @retry_on_failure - 자동 재시도
- @LoggingUtils.log_function_call - 함수 호출 로깅
- @DebugUtils.profile_function - 성능 프로파일링
- @DebugUtils.memory_usage - 메모리 사용량 모니터링
- @APIUtils.rate_limit - API 호출 제한

🧪 테스트 및 디버깅:

- TestUtils.generate_mock_data() - 모의 데이터 생성
- TestUtils.validate_trading_signal() - 신호 검증
- DebugUtils.debug_print_vars() - 변수 디버그 출력
- BacktestUtils.run_simple_backtest() - 간단한 백테스트

🌟 특별 기능:

- 실시간 환율 자동 업데이트
- 다중 시장 시간대 지원
- 안전한 API 요청 처리
- 자동 파일 백업 및 압축
- 포괄적인 성과 분석
- ASCII 기반 시각화
- 통합 헬스 체크

💡 팁:
- 모든 비동기 함수는 await와 함께 사용
- 환율 정보는 자동으로 캐시됨 (5분)
- 로그는 자동으로 로테이션됨
- 민감한 정보는 자동으로 마스킹됨
- 모든 예외는 적절히 처리됨

📞 문의: 퀸트마스터팀
📅 버전: 1.0.0 (완전체)
"""
    print(help_text)

# 모듈 정보
__version__ = "1.0.0"
__author__ = "퀸트마스터팀"
__description__ = "퀸트프로젝트 통합 유틸리티 모듈"
__all__ = [
    'QuintUtils', 'CurrencyUtils', 'TechnicalIndicators', 'PerformanceAnalyzer',
    'NetworkUtils', 'FileUtils', 'NotificationFormatter', 'SecurityUtils',
    'TimeUtils', 'DataUtils', 'VisualizationUtils', 'SystemUtils',
    'StrategyUtils', 'LoggingUtils', 'TestUtils', 'DebugUtils',
    'APIUtils', 'BacktestUtils', 'ChartUtils', 'ConfigUtils',
    'TradingSignal', 'MarketData', 'RiskMetrics', 'PerformanceMetrics',
    'retry_on_failure', 'initialize_quint_utils', 'get_utils', 'show_help'
]
