#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 퀸트프로젝트 유틸리티 모듈 (utils.py)
==============================================
🇺🇸 미국주식 + 🇯🇵 일본주식 + 🇮🇳 인도주식 + 💰 암호화폐 공통 유틸리티

✨ 핵심 기능:
- 데이터 전처리 및 변환
- 기술지표 계산
- 리스크 관리 함수
- 시간 및 날짜 유틸리티
- 환율 및 화폐 변환
- 성과 분석 도구
- 로깅 및 알림 헬퍼
- 파일 I/O 유틸리티
- OpenAI API 통합 (GPT-4, 텍스트 분석, 투자 조언)

Author: 퀸트마스터팀
Version: 1.2.0 (OpenAI 통합 유틸리티)
"""

import asyncio
import logging
import json
import csv
import os
import sqlite3
import hashlib
import time
import math
import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import aiohttp
import pandas as pd
import numpy as np
import pytz
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# OpenAI 관련 임포트
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI 패키지가 설치되지 않았습니다. pip install openai로 설치해주세요.")

# ============================================================================
# 🎯 상수 및 설정
# ============================================================================
class Currency(Enum):
    """지원 통화"""
    USD = "USD"
    KRW = "KRW"
    JPY = "JPY"
    INR = "INR"
    BTC = "BTC"
    ETH = "ETH"

class Market(Enum):
    """지원 시장"""
    US_STOCK = "US_STOCK"
    KOREA_STOCK = "KOREA_STOCK"
    JAPAN_STOCK = "JAPAN_STOCK"
    INDIA_STOCK = "INDIA_STOCK"
    CRYPTO = "CRYPTO"

class TimeZones:
    """시간대 상수"""
    UTC = pytz.UTC
    SEOUL = pytz.timezone('Asia/Seoul')
    TOKYO = pytz.timezone('Asia/Tokyo')
    NEW_YORK = pytz.timezone('America/New_York')
    MUMBAI = pytz.timezone('Asia/Kolkata')

class OpenAIModel(Enum):
    """OpenAI 모델"""
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

class AnalysisType(Enum):
    """분석 유형"""
    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    SENTIMENT = "SENTIMENT"
    RISK = "RISK"
    PORTFOLIO = "PORTFOLIO"

# ============================================================================
# 📊 데이터 처리 유틸리티
# ============================================================================
class DataProcessor:
    """데이터 전처리 및 변환"""
    
    @staticmethod
    def clean_numeric_data(data: Union[str, int, float], default: float = 0.0) -> float:
        """숫자 데이터 정리"""
        try:
            if data is None or data == '':
                return default
            
            if isinstance(data, str):
                # 콤마, 공백 제거
                cleaned = data.replace(',', '').replace(' ', '')
                # 퍼센트 기호 제거
                if '%' in cleaned:
                    cleaned = cleaned.replace('%', '')
                    return float(cleaned) / 100
                
                return float(cleaned)
            
            return float(data)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """안전한 나눗셈"""
        try:
            if denominator == 0 or denominator is None:
                return default
            return numerator / denominator
        except (TypeError, ZeroDivisionError):
            return default
    
    @staticmethod
    def normalize_symbol(symbol: str, market: Market) -> str:
        """심볼명 정규화"""
        if not symbol:
            return ""
        
        symbol = symbol.upper().strip()
        
        if market == Market.KOREA_STOCK:
            # 한국 주식 코드 정규화
            return symbol.replace('A', '').zfill(6)
        elif market == Market.CRYPTO:
            # 암호화폐 심볼 정규화
            if '-' not in symbol and symbol != 'KRW':
                return f"KRW-{symbol}"
        
        return symbol
    
    @staticmethod
    def format_number(num: float, decimal_places: int = 2) -> str:
        """숫자 포맷팅"""
        try:
            if abs(num) >= 1_000_000_000:
                return f"{num/1_000_000_000:.1f}B"
            elif abs(num) >= 1_000_000:
                return f"{num/1_000_000:.1f}M"
            elif abs(num) >= 1_000:
                return f"{num/1_000:.1f}K"
            else:
                return f"{num:,.{decimal_places}f}"
        except:
            return "0"
    
    @staticmethod
    def calculate_percentage(value: float, total: float) -> float:
        """퍼센트 계산"""
        return DataProcessor.safe_divide(value * 100, total, 0.0)

# ============================================================================
# 🤖 OpenAI 통합 유틸리티
# ============================================================================
@dataclass
class OpenAIConfig:
    """OpenAI 설정"""
    api_key: str
    model: str = OpenAIModel.GPT_4.value
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30

@dataclass
class AIAnalysisResult:
    """AI 분석 결과"""
    analysis_type: str
    symbol: str
    analysis: str
    confidence: float
    recommendations: List[str]
    risk_level: str
    timestamp: datetime

class OpenAIHelper:
    """OpenAI API 통합 헬퍼"""
    
    def __init__(self, config: OpenAIConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI 패키지가 설치되지 않았습니다.")
        
        self.config = config
        openai.api_key = config.api_key
        self.logger = logging.getLogger(__name__)
    
    async def analyze_stock(self, symbol: str, market_data: Dict[str, Any], 
                           analysis_type: AnalysisType = AnalysisType.TECHNICAL) -> AIAnalysisResult:
        """주식 분석"""
        try:
            # 프롬프트 생성
            prompt = self._generate_analysis_prompt(symbol, market_data, analysis_type)
            
            # OpenAI API 호출
            response = await self._call_openai_api(prompt)
            
            # 결과 파싱
            analysis_result = self._parse_analysis_response(response, symbol, analysis_type)
            
            return analysis_result
        
        except Exception as e:
            self.logger.error(f"AI 주식 분석 실패: {e}")
            return AIAnalysisResult(
                analysis_type=analysis_type.value,
                symbol=symbol,
                analysis="분석 중 오류가 발생했습니다.",
                confidence=0.0,
                recommendations=["전문가와 상담하세요."],
                risk_level="HIGH",
                timestamp=datetime.now()
            )
    
    async def generate_trading_strategy(self, portfolio_data: Dict[str, Any], 
                                      risk_tolerance: str = "MEDIUM") -> str:
        """거래 전략 생성"""
        try:
            prompt = f"""
다음 포트폴리오 데이터를 바탕으로 거래 전략을 제안해주세요:

포트폴리오 정보:
{json.dumps(portfolio_data, indent=2, ensure_ascii=False)}

리스크 허용도: {risk_tolerance}

다음 형식으로 답변해주세요:
1. 현재 포트폴리오 분석
2. 리스크 평가
3. 추천 거래 전략
4. 구체적인 액션 아이템
5. 주의사항

전문적이고 실용적인 조언을 제공해주세요.
"""
            
            response = await self._call_openai_api(prompt)
            return response
        
        except Exception as e:
            self.logger.error(f"거래 전략 생성 실패: {e}")
            return "거래 전략 생성 중 오류가 발생했습니다. 전문가와 상담하세요."
    
    async def analyze_market_sentiment(self, news_articles: List[str], 
                                     symbols: List[str] = None) -> Dict[str, Any]:
        """시장 감성 분석"""
        try:
            # 뉴스 기사들을 하나의 텍스트로 결합
            combined_text = "\n\n".join(news_articles[:10])  # 최대 10개 기사
            
            symbol_filter = f"특히 {', '.join(symbols)} 종목에 대해 " if symbols else ""
            
            prompt = f"""
다음 뉴스 기사들을 분석하여 시장 감성을 평가해주세요:

{combined_text}

{symbol_filter}다음 항목들을 분석해주세요:
1. 전체적인 시장 감성 (POSITIVE/NEUTRAL/NEGATIVE)
2. 감성 점수 (0-100)
3. 주요 감성 키워드
4. 투자자들이 주의해야 할 포인트
5. 단기/중기 시장 전망

JSON 형식으로 구조화된 결과를 제공해주세요.
"""
            
            response = await self._call_openai_api(prompt)
            
            # JSON 파싱 시도
            try:
                sentiment_data = json.loads(response)
            except:
                # JSON 파싱 실패시 기본 구조 반환
                sentiment_data = {
                    "overall_sentiment": "NEUTRAL",
                    "sentiment_score": 50,
                    "key_keywords": ["시장", "분석"],
                    "analysis": response
                }
            
            return sentiment_data
        
        except Exception as e:
            self.logger.error(f"시장 감성 분석 실패: {e}")
            return {
                "overall_sentiment": "NEUTRAL",
                "sentiment_score": 50,
                "error": str(e)
            }
    
    async def get_investment_advice(self, user_profile: Dict[str, Any], 
                                  market_conditions: Dict[str, Any]) -> str:
        """개인 맞춤 투자 조언"""
        try:
            prompt = f"""
다음 투자자 프로필과 시장 상황을 바탕으로 개인 맞춤 투자 조언을 제공해주세요:

투자자 프로필:
{json.dumps(user_profile, indent=2, ensure_ascii=False)}

현재 시장 상황:
{json.dumps(market_conditions, indent=2, ensure_ascii=False)}

다음 내용을 포함해서 조언해주세요:
1. 현재 포트폴리오 평가
2. 리스크 관리 방안
3. 자산 배분 조정 제안
4. 투자 타이밍 조언
5. 장기 투자 전략

투자자의 위험 성향과 투자 목표에 맞는 구체적이고 실용적인 조언을 제공해주세요.
"""
            
            response = await self._call_openai_api(prompt)
            return response
        
        except Exception as e:
            self.logger.error(f"투자 조언 생성 실패: {e}")
            return "투자 조언 생성 중 오류가 발생했습니다. 전문가와 상담하세요."
    
    async def explain_financial_terms(self, terms: List[str], 
                                    language: str = "Korean") -> Dict[str, str]:
        """금융 용어 설명"""
        try:
            terms_text = ", ".join(terms)
            
            prompt = f"""
다음 금융 용어들을 {language}로 쉽고 명확하게 설명해주세요:
{terms_text}

각 용어에 대해:
1. 정의
2. 실제 사용 예시
3. 투자에서의 중요성

일반인도 이해할 수 있도록 쉬운 언어로 설명해주세요.
"""
            
            response = await self._call_openai_api(prompt)
            
            # 응답을 파싱하여 각 용어별로 분리
            explanations = {}
            lines = response.split('\n')
            current_term = None
            current_explanation = []
            
            for line in lines:
                line = line.strip()
                if any(term in line for term in terms):
                    if current_term:
                        explanations[current_term] = '\n'.join(current_explanation)
                    current_term = next((term for term in terms if term in line), None)
                    current_explanation = [line]
                elif current_term and line:
                    current_explanation.append(line)
            
            if current_term:
                explanations[current_term] = '\n'.join(current_explanation)
            
            return explanations
        
        except Exception as e:
            self.logger.error(f"금융 용어 설명 실패: {e}")
            return {term: "설명을 생성할 수 없습니다." for term in terms}
    
    async def generate_market_report(self, market_data: Dict[str, Any], 
                                   timeframe: str = "daily") -> str:
        """시장 보고서 생성"""
        try:
            prompt = f"""
다음 시장 데이터를 바탕으로 {timeframe} 시장 보고서를 작성해주세요:

{json.dumps(market_data, indent=2, ensure_ascii=False)}

보고서 구성:
1. 시장 개요 및 주요 동향
2. 섹터별 성과 분석
3. 주요 이슈 및 이벤트
4. 기술적 분석 요약
5. 투자자 행동 분석
6. 향후 전망 및 주목 포인트

전문적이면서도 이해하기 쉬운 보고서를 작성해주세요.
"""
            
            response = await self._call_openai_api(prompt)
            return response
        
        except Exception as e:
            self.logger.error(f"시장 보고서 생성 실패: {e}")
            return "시장 보고서 생성 중 오류가 발생했습니다."
    
    async def _call_openai_api(self, prompt: str, system_message: str = None) -> str:
        """OpenAI API 호출"""
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            else:
                messages.append({
                    "role": "system", 
                    "content": "당신은 전문적인 금융 분석가이자 투자 advisor입니다. 정확하고 실용적인 조언을 제공하되, 투자 위험에 대해서도 명확히 안내해주세요."
                })
            
            messages.append({"role": "user", "content": prompt})
            
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    openai.ChatCompletion.create,
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                ),
                timeout=self.config.timeout
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            self.logger.error(f"OpenAI API 호출 실패: {e}")
            raise
    
    def _generate_analysis_prompt(self, symbol: str, market_data: Dict[str, Any], 
                                analysis_type: AnalysisType) -> str:
        """분석 프롬프트 생성"""
        base_prompt = f"""
{symbol} 종목에 대한 {analysis_type.value} 분석을 수행해주세요.

시장 데이터:
{json.dumps(market_data, indent=2, ensure_ascii=False)}
"""
        
        if analysis_type == AnalysisType.TECHNICAL:
            base_prompt += """
기술적 분석 요소:
1. 가격 추세 분석
2. 거래량 분석
3. 지지/저항선
4. 기술적 지표 (RSI, MACD, 볼린저 밴드 등)
5. 차트 패턴
6. 매매 시점 제안
"""
        elif analysis_type == AnalysisType.FUNDAMENTAL:
            base_prompt += """
기본적 분석 요소:
1. 재무 건전성
2. 성장성 지표
3. 밸류에이션
4. 업종 전망
5. 경쟁력 분석
6. 투자 가치 평가
"""
        elif analysis_type == AnalysisType.SENTIMENT:
            base_prompt += """
감성 분석 요소:
1. 시장 심리
2. 뉴스 및 이슈 분석
3. 투자자 행동 패턴
4. 소셜 미디어 감성
5. 전문가 의견 종합
"""
        
        base_prompt += """
분석 결과를 다음 형식으로 제공해주세요:
- 종합 의견 (BUY/HOLD/SELL)
- 신뢰도 (0-100%)
- 주요 근거 3가지
- 리스크 요인
- 투자 제안사항
"""
        
        return base_prompt
    
    def _parse_analysis_response(self, response: str, symbol: str, 
                               analysis_type: AnalysisType) -> AIAnalysisResult:
        """분석 응답 파싱"""
        # 신뢰도 추출
        confidence = 50.0  # 기본값
        if "신뢰도" in response or "%" in response:
            import re
            confidence_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
            if confidence_match:
                confidence = float(confidence_match.group(1))
        
        # 추천사항 추출
        recommendations = []
        if "BUY" in response.upper() or "매수" in response:
            recommendations.append("매수 고려")
        elif "SELL" in response.upper() or "매도" in response:
            recommendations.append("매도 고려")
        elif "HOLD" in response.upper() or "보유" in response:
            recommendations.append("보유 유지")
        
        # 리스크 수준 결정
        risk_level = "MEDIUM"
        if "위험" in response or "리스크" in response or confidence < 30:
            risk_level = "HIGH"
        elif confidence > 80:
            risk_level = "LOW"
        
        return AIAnalysisResult(
            analysis_type=analysis_type.value,
            symbol=symbol,
            analysis=response,
            confidence=confidence,
            recommendations=recommendations,
            risk_level=risk_level,
            timestamp=datetime.now()
        )

# ============================================================================
# 📈 기술지표 계산
# ============================================================================
class TechnicalIndicators:
    """기술지표 계산 함수들"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """단순이동평균 (Simple Moving Average)"""
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(avg)
        
        return sma_values
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """지수이동평균 (Exponential Moving Average)"""
        if len(prices) < period:
            return []
        
        alpha = 2 / (period + 1)
        ema_values = []
        
        # 첫 번째 EMA는 SMA로 계산
        first_sma = sum(prices[:period]) / period
        ema_values.append(first_sma)
        
        # 나머지 EMA 계산
        for i in range(period, len(prices)):
            ema = alpha * prices[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return []
        
        gains = []
        losses = []
        
        # 가격 변화 계산
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        rsi_values = []
        
        # 첫 번째 RSI 계산
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        # 나머지 RSI 계산
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
        """볼린저 밴드"""
        if len(prices) < period:
            return [], [], []
        
        middle_band = TechnicalIndicators.sma(prices, period)
        upper_band = []
        lower_band = []
        
        for i in range(period - 1, len(prices)):
            subset = prices[i - period + 1:i + 1]
            std = statistics.stdev(subset)
            sma_val = middle_band[i - period + 1]
            
            upper_band.append(sma_val + (std_dev * std))
            lower_band.append(sma_val - (std_dev * std))
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return [], [], []
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        # MACD 라인 계산
        macd_line = []
        start_index = slow - fast
        
        for i in range(len(ema_slow)):
            macd_val = ema_fast[i + start_index] - ema_slow[i]
            macd_line.append(macd_val)
        
        # 시그널 라인 계산
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        
        # 히스토그램 계산
        histogram = []
        signal_start = signal - 1
        
        for i in range(len(signal_line)):
            hist_val = macd_line[i + signal_start] - signal_line[i]
            histogram.append(hist_val)
        
        return macd_line, signal_line, histogram

# ============================================================================
# 💰 리스크 관리 유틸리티
# ============================================================================
class RiskManager:
    """리스크 관리 함수들"""
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_per_trade: float, 
                              entry_price: float, stop_loss_price: float) -> float:
        """포지션 크기 계산"""
        try:
            risk_amount = account_balance * risk_per_trade
            price_diff = abs(entry_price - stop_loss_price)
            
            if price_diff == 0:
                return 0
            
            position_size = risk_amount / price_diff
            return max(0, position_size)
        except:
            return 0
    
    @staticmethod
    def calculate_max_drawdown(returns: List[float]) -> float:
        """최대 낙폭 계산"""
        if not returns:
            return 0
        
        cumulative_returns = [1]
        for ret in returns:
            cumulative_returns.append(cumulative_returns[-1] * (1 + ret))
        
        max_drawdown = 0
        peak = cumulative_returns[0]
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """샤프 지수 계산"""
        if len(returns) < 2:
            return 0
        
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0
        
        excess_return = mean_return - risk_free_rate / 252  # 일일 무위험 수익률
        sharpe = excess_return / std_return * math.sqrt(252)  # 연환산
        
        return sharpe
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
        """VaR (Value at Risk) 계산"""
        if not returns:
            return 0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0
    
    @staticmethod
    def is_position_size_valid(position_size: float, account_balance: float, 
                             max_position_pct: float = 0.1) -> bool:
        """포지션 크기 유효성 검사"""
        max_position_value = account_balance * max_position_pct
        return 0 < position_size <= max_position_value

# ============================================================================
# 🕐 시간 및 날짜 유틸리티
# ============================================================================
class TimeUtils:
    """시간 관련 유틸리티"""
    
    @staticmethod
    def get_current_time(timezone_name: str = 'Asia/Seoul') -> datetime:
        """현재 시간 조회"""
        tz = pytz.timezone(timezone_name)
        return datetime.now(tz)
    
    @staticmethod
    def is_market_open(market: Market, current_time: datetime = None) -> bool:
        """시장 개장 여부 확인"""
        if current_time is None:
            current_time = datetime.now(TimeZones.UTC)
        
        # 현지 시간으로 변환
        if market == Market.US_STOCK:
            local_time = current_time.astimezone(TimeZones.NEW_YORK)
            # 월-금, 9:30-16:00 (EST)
            return (local_time.weekday() < 5 and 
                   9.5 <= local_time.hour + local_time.minute/60 <= 16)
        
        elif market == Market.KOREA_STOCK:
            local_time = current_time.astimezone(TimeZones.SEOUL)
            # 월-금, 9:00-15:30 (KST)
            return (local_time.weekday() < 5 and 
                   9 <= local_time.hour + local_time.minute/60 <= 15.5)
        
        elif market == Market.JAPAN_STOCK:
            local_time = current_time.astimezone(TimeZones.TOKYO)
            # 월-금, 9:00-11:30, 12:30-15:00 (JST)
            time_decimal = local_time.hour + local_time.minute/60
            return (local_time.weekday() < 5 and 
                   ((9 <= time_decimal <= 11.5) or (12.5 <= time_decimal <= 15)))
        
        elif market == Market.INDIA_STOCK:
            local_time = current_time.astimezone(TimeZones.MUMBAI)
            # 월-금, 9:15-15:30 (IST)
            return (local_time.weekday() < 5 and 
                   9.25 <= local_time.hour + local_time.minute/60 <= 15.5)
        
        elif market == Market.CRYPTO:
            # 암호화폐는 24시간 거래
            return True
        
        return False
    
    @staticmethod
    def get_next_trading_day(market: Market, from_date: datetime = None) -> datetime:
        """다음 거래일 조회"""
        if from_date is None:
            from_date = datetime.now()
        
        next_day = from_date + timedelta(days=1)
        
        # 암호화폐는 매일 거래
        if market == Market.CRYPTO:
            return next_day
        
        # 주말 건너뛰기
        while next_day.weekday() >= 5:  # 토요일(5), 일요일(6)
            next_day += timedelta(days=1)
        
        return next_day
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
        """날짜 시간 포맷팅"""
        try:
            return dt.strftime(format_str)
        except:
            return ""
    
    @staticmethod
    def parse_datetime(date_str: str, format_str: str = '%Y-%m-%d %H:%M:%S') -> Optional[datetime]:
        """문자열을 datetime으로 변환"""
        try:
            return datetime.strptime(date_str, format_str)
        except:
            return None

# ============================================================================
# 💱 환율 및 화폐 유틸리티
# ============================================================================
class CurrencyConverter:
    """환율 변환 유틸리티"""
    
    def __init__(self):
        self.exchange_rates = {}
        self.last_update = None
        self.cache_duration = timedelta(hours=1)  # 1시간 캐시
    
    async def get_exchange_rate(self, from_currency: Currency, to_currency: Currency) -> float:
        """환율 조회"""
        if from_currency == to_currency:
            return 1.0
        
        # 캐시 확인
        rate_key = f"{from_currency.value}_{to_currency.value}"
        
        if (self.last_update and 
            datetime.now() - self.last_update < self.cache_duration and
            rate_key in self.exchange_rates):
            return self.exchange_rates[rate_key]
        
        # 환율 API 호출
        try:
            rate = await self._fetch_exchange_rate(from_currency.value, to_currency.value)
            self.exchange_rates[rate_key] = rate
            self.last_update = datetime.now()
            return rate
        except:
            # 기본값 반환
            return self._get_default_rate(from_currency, to_currency)
    
    async def _fetch_exchange_rate(self, from_curr: str, to_curr: str) -> float:
        """외부 API에서 환율 조회"""
        try:
            # 여러 API 시도
            apis = [
                f"https://api.exchangerate-api.com/v4/latest/{from_curr}",
                f"https://api.fixer.io/latest?base={from_curr}",
            ]
            
            for api_url in apis:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(api_url, timeout=5) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'rates' in data and to_curr in data['rates']:
                                    return float(data['rates'][to_curr])
                except:
                    continue
            
            raise Exception("모든 API 실패")
            
        except Exception as e:
            raise Exception(f"환율 조회 실패: {e}")
    
    def _get_default_rate(self, from_currency: Currency, to_currency: Currency) -> float:
        """기본 환율 (백업용)"""
        default_rates = {
            ('USD', 'KRW'): 1200.0,
            ('USD', 'JPY'): 110.0,
            ('USD', 'INR'): 75.0,
            ('KRW', 'USD'): 1/1200.0,
            ('JPY', 'USD'): 1/110.0,
            ('INR', 'USD'): 1/75.0,
        }
        
        key = (from_currency.value, to_currency.value)
        reverse_key = (to_currency.value, from_currency.value)
        
        if key in default_rates:
            return default_rates[key]
        elif reverse_key in default_rates:
            return 1 / default_rates[reverse_key]
        else:
            return 1.0
    
    async def convert_amount(self, amount: float, from_currency: Currency, 
                           to_currency: Currency) -> float:
        """금액 환전"""
        rate = await self.get_exchange_rate(from_currency, to_currency)
        return amount * rate

# ============================================================================
# 📊 성과 분석 도구
# ============================================================================
@dataclass
class PerformanceMetrics:
    """성과 지표 데이터 클래스"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int

class PerformanceAnalyzer:
    """성과 분석기"""
    
    @staticmethod
    def calculate_performance_metrics(trades: List[Dict]) -> PerformanceMetrics:
        """성과 지표 계산"""
        if not trades:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # 수익률 계산
        returns = []
        profits = []
        losses = []
        
        for trade in trades:
            if 'profit_loss' in trade and trade['profit_loss'] is not None:
                pnl = float(trade['profit_loss'])
                
                if 'entry_price' in trade and 'quantity' in trade:
                    entry_value = float(trade['entry_price']) * float(trade['quantity'])
                    if entry_value > 0:
                        ret = pnl / entry_value
                        returns.append(ret)
                
                if pnl > 0:
                    profits.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))
        
        # 기본 통계
        total_trades = len(trades)
        winning_trades = len(profits)
        losing_trades = len(losses)
        
        # 수익률 계산
        total_return = sum(returns) if returns else 0
        annual_return = total_return * 252 / len(returns) if returns else 0  # 연환산
        
        # 변동성
        volatility = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0
        
        # 샤프 지수
        sharpe_ratio = RiskManager.calculate_sharpe_ratio(returns)
        
        # 최대 낙폭
        max_drawdown = RiskManager.calculate_max_drawdown(returns)
        
        # 승률
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 프로핏 팩터
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 1  # 0으로 나누기 방지
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades
        )
    
    @staticmethod
    def generate_performance_report(metrics: PerformanceMetrics) -> str:
        """성과 보고서 생성"""
        report = f"""
📊 성과 분석 보고서
{'='*40}

📈 수익률 지표:
  • 총 수익률: {metrics.total_return:.2%}
  • 연환산 수익률: {metrics.annual_return:.2%}
  • 변동성: {metrics.volatility:.2%}

🎯 리스크 지표:
  • 샤프 지수: {metrics.sharpe_ratio:.2f}
  • 최대 낙폭: {metrics.max_drawdown:.2%}

📊 거래 통계:
  • 총 거래 수: {metrics.total_trades}회
  • 승률: {metrics.win_rate:.1f}%
  • 수익 거래: {metrics.winning_trades}회
  • 손실 거래: {metrics.losing_trades}회
  • 프로핏 팩터: {metrics.profit_factor:.2f}

📝 평가:
"""
        
        # 성과 평가
        if metrics.sharpe_ratio > 2:
            report += "  🌟 우수한 위험 대비 수익률\n"
        elif metrics.sharpe_ratio > 1:
            report += "  ✅ 양호한 위험 대비 수익률\n"
        else:
            report += "  ⚠️ 개선 필요한 위험 대비 수익률\n"
        
        if metrics.win_rate > 60:
            report += "  🎯 높은 승률\n"
        elif metrics.win_rate > 40:
            report += "  📊 적정 승률\n"
        else:
            report += "  📉 낮은 승률\n"
        
        if metrics.max_drawdown < 0.1:
            report += "  🛡️ 안정적인 리스크 관리\n"
        elif metrics.max_drawdown < 0.2:
            report += "  ⚖️ 적정한 리스크 수준\n"
        else:
            report += "  ⚠️ 높은 리스크 수준\n"
        
        return report

# ============================================================================
# 📝 로깅 및 알림 헬퍼
# ============================================================================
class LoggingHelper:
    """로깅 헬퍼 클래스"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 중복 핸들러 방지
        if logger.handlers:
            return logger
        
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_trade(logger: logging.Logger, strategy: str, symbol: str, action: str, 
                  quantity: float, price: float, reason: str = ""):
        """거래 로그"""
        message = f"🔄 {strategy} | {action} {symbol} | {quantity}주 @ {price:,.0f} | {reason}"
        logger.info(message)
    
    @staticmethod
    def log_error_with_context(logger: logging.Logger, error: Exception, context: Dict[str, Any]):
        """컨텍스트와 함께 오류 로그"""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        logger.error(f"❌ 오류 발생: {str(error)} | 컨텍스트: {context_str}")

def log_execution_time(func):
    """실행 시간 로그 데코레이터"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger = logging.getLogger(func.__name__)
            logger.info(f"⏱️ {func.__name__} 실행 완료: {execution_time:.2f}초")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger = logging.getLogger(func.__name__)
            logger.error(f"❌ {func.__name__} 실행 실패: {execution_time:.2f}초, 오류: {e}")
            
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger = logging.getLogger(func.__name__)
            logger.info(f"⏱️ {func.__name__} 실행 완료: {execution_time:.2f}초")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger = logging.getLogger(func.__name__)
            logger.error(f"❌ {func.__name__} 실행 실패: {execution_time:.2f}초, 오류: {e}")
            
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# ============================================================================
# 💾 파일 I/O 유틸리티
# ============================================================================
class FileManager:
    """파일 관리 유틸리티"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """디렉토리 생성 확인"""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path], 
                  indent: int = 2) -> bool:
        """JSON 파일 저장"""
        try:
            path_obj = Path(file_path)
            FileManager.ensure_directory(path_obj.parent)
            
            with open(path_obj, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
            return True
        except Exception as e:
            logging.error(f"JSON 저장 실패 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: Union[str, Path], default: Dict[str, Any] = None) -> Dict[str, Any]:
        """JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"JSON 로드 실패 {file_path}: {e}")
            return default or {}
    
    @staticmethod
    def save_csv(data: List[Dict[str, Any]], file_path: Union[str, Path], 
                 fieldnames: List[str] = None) -> bool:
        """CSV 파일 저장"""
        try:
            if not data:
                return False
            
            path_obj = Path(file_path)
            FileManager.ensure_directory(path_obj.parent)
            
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            with open(path_obj, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            return True
        except Exception as e:
            logging.error(f"CSV 저장 실패 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_csv(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """CSV 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logging.warning(f"CSV 로드 실패 {file_path}: {e}")
            return []
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], backup_dir: Union[str, Path] = None) -> Optional[Path]:
        """파일 백업"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return None
            
            if backup_dir is None:
                backup_dir = source_path.parent / 'backups'
            
            backup_path = Path(backup_dir)
            FileManager.ensure_directory(backup_path)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
            backup_file_path = backup_path / backup_filename
            
            import shutil
            shutil.copy2(source_path, backup_file_path)
            
            return backup_file_path
        except Exception as e:
            logging.error(f"파일 백업 실패 {file_path}: {e}")
            return None

# ============================================================================
# 🔐 보안 및 암호화 유틸리티
# ============================================================================
class SecurityUtils:
    """보안 관련 유틸리티"""
    
    @staticmethod
    def hash_string(text: str, algorithm: str = 'sha256') -> str:
        """문자열 해시"""
        try:
            hash_obj = hashlib.new(algorithm)
            hash_obj.update(text.encode('utf-8'))
            return hash_obj.hexdigest()
        except Exception as e:
            logging.error(f"해시 생성 실패: {e}")
            return ""
    
    @staticmethod
    def mask_sensitive_data(data: str, show_chars: int = 4) -> str:
        """민감한 데이터 마스킹"""
        if len(data) <= show_chars * 2:
            return '*' * len(data)
        
        return data[:show_chars] + '*' * (len(data) - show_chars * 2) + data[-show_chars:]
    
    @staticmethod
    def validate_api_key(api_key: str, min_length: int = 16) -> bool:
        """API 키 유효성 검사"""
        if not api_key or len(api_key) < min_length:
            return False
        
        # 기본적인 패턴 검사
        import re
        pattern = r'^[A-Za-z0-9\-_]+
        return bool(re.match(pattern, api_key))

# ============================================================================
# 🌐 네트워크 유틸리티
# ============================================================================
class NetworkUtils:
    """네트워크 관련 유틸리티"""
    
    @staticmethod
    async def check_internet_connection(timeout: int = 5) -> bool:
        """인터넷 연결 확인"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://www.google.com', timeout=timeout) as response:
                    return response.status == 200
        except:
            return False
    
    @staticmethod
    async def ping_server(url: str, timeout: int = 5) -> Tuple[bool, float]:
        """서버 핑 테스트"""
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    response_time = time.time() - start_time
                    return response.status == 200, response_time
        except:
            response_time = time.time() - start_time
            return False, response_time
    
    @staticmethod
    async def safe_api_request(url: str, method: str = 'GET', headers: Dict = None, 
                             data: Dict = None, timeout: int = 10, 
                             max_retries: int = 3) -> Optional[Dict]:
        """안전한 API 요청"""
        headers = headers or {}
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method, url, headers=headers, json=data, timeout=timeout
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limit
                            await asyncio.sleep(2 ** attempt)  # 지수 백오프
                            continue
                        else:
                            logging.warning(f"API 요청 실패: {response.status}")
                            return None
            except asyncio.TimeoutError:
                logging.warning(f"API 요청 타임아웃 (시도 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"API 요청 오류: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        return None

# ============================================================================
# 📊 데이터베이스 유틸리티
# ============================================================================
class DatabaseUtils:
    """데이터베이스 관련 유틸리티"""
    
    @staticmethod
    def create_connection(db_path: str) -> Optional[sqlite3.Connection]:
        """SQLite 연결 생성"""
        try:
            FileManager.ensure_directory(Path(db_path).parent)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
            return conn
        except Exception as e:
            logging.error(f"데이터베이스 연결 실패: {e}")
            return None
    
    @staticmethod
    def execute_query(conn: sqlite3.Connection, query: str, 
                     params: Tuple = None) -> Optional[List[sqlite3.Row]]:
        """쿼리 실행"""
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                conn.commit()
                return None
        except Exception as e:
            logging.error(f"쿼리 실행 실패: {e}")
            conn.rollback()
            return None
    
    @staticmethod
    def bulk_insert(conn: sqlite3.Connection, table: str, data: List[Dict[str, Any]]) -> bool:
        """대량 데이터 삽입"""
        if not data:
            return True
        
        try:
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            cursor = conn.cursor()
            values = [[row[col] for col in columns] for row in data]
            cursor.executemany(query, values)
            conn.commit()
            
            return True
        except Exception as e:
            logging.error(f"대량 삽입 실패: {e}")
            conn.rollback()
            return False
    
    @staticmethod
    def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        """테이블 존재 여부 확인"""
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            return cursor.fetchone() is not None
        except:
            return False

# ============================================================================
# 🔧 시스템 모니터링 유틸리티
# ============================================================================
class SystemMonitor:
    """시스템 모니터링 유틸리티"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """시스템 정보 조회"""
        try:
            import psutil
            
            # CPU 정보
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_total = memory.total / (1024**3)  # GB
            memory_available = memory.available / (1024**3)  # GB
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free = disk.free / (1024**3)  # GB
            
            # 네트워크 정보
            network = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'percent': memory_percent,
                    'total_gb': memory_total,
                    'available_gb': memory_available
                },
                'disk': {
                    'percent': disk_percent,
                    'free_gb': disk_free
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                }
            }
        except Exception as e:
            logging.error(f"시스템 정보 조회 실패: {e}")
            return {}
    
    @staticmethod
    def check_system_health() -> Dict[str, Any]:
        """시스템 건강 상태 확인"""
        info = SystemMonitor.get_system_info()
        
        health_status = {
            'healthy': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # CPU 체크
            cpu_percent = info.get('cpu', {}).get('percent', 0)
            if cpu_percent > 90:
                health_status['errors'].append(f'CPU 사용률 위험: {cpu_percent:.1f}%')
                health_status['healthy'] = False
            elif cpu_percent > 75:
                health_status['warnings'].append(f'CPU 사용률 높음: {cpu_percent:.1f}%')
            
            # 메모리 체크
            memory_percent = info.get('memory', {}).get('percent', 0)
            if memory_percent > 95:
                health_status['errors'].append(f'메모리 사용률 위험: {memory_percent:.1f}%')
                health_status['healthy'] = False
            elif memory_percent > 85:
                health_status['warnings'].append(f'메모리 사용률 높음: {memory_percent:.1f}%')
            
            # 디스크 체크
            disk_percent = info.get('disk', {}).get('percent', 0)
            disk_free = info.get('disk', {}).get('free_gb', 0)
            if disk_free < 1:
                health_status['errors'].append(f'디스크 공간 부족: {disk_free:.1f}GB')
                health_status['healthy'] = False
            elif disk_free < 5:
                health_status['warnings'].append(f'디스크 공간 경고: {disk_free:.1f}GB')
            
        except Exception as e:
            health_status['errors'].append(f'건강 상태 체크 실패: {str(e)}')
            health_status['healthy'] = False
        
        return health_status

# ============================================================================
# 🎨 데이터 시각화 헬퍼
# ============================================================================
class VisualizationHelper:
    """데이터 시각화 헬퍼"""
    
    @staticmethod
    def create_ascii_chart(values: List[float], width: int = 50, height: int = 10) -> str:
        """ASCII 차트 생성"""
        if not values:
            return "데이터 없음"
        
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            return "모든 값이 동일함"
        
        # 정규화
        normalized = [(val - min_val) / (max_val - min_val) for val in values]
        
        # 차트 생성
        chart_lines = []
        for y in range(height):
            line = ""
            threshold = 1 - (y / height)
            
            for x in range(min(width, len(normalized))):
                if normalized[x] >= threshold:
                    line += "█"
                else:
                    line += " "
            
            chart_lines.append(line)
        
        # 라벨 추가
        chart = f"최고: {max_val:.2f}\n"
        chart += "\n".join(chart_lines)
        chart += f"\n최저: {min_val:.2f}"
        
        return chart
    
    @staticmethod
    def format_table(data: List[Dict[str, Any]], headers: List[str] = None) -> str:
        """테이블 포맷팅"""
        if not data:
            return "데이터 없음"
        
        if headers is None:
            headers = list(data[0].keys())
        
        # 컬럼 너비 계산
        col_widths = {}
        for header in headers:
            col_widths[header] = len(str(header))
            for row in data:
                if header in row:
                    col_widths[header] = max(col_widths[header], len(str(row[header])))
        
        # 헤더 생성
        header_line = " | ".join(str(header).ljust(col_widths[header]) for header in headers)
        separator_line = "-+-".join("-" * col_widths[header] for header in headers)
        
        # 데이터 행 생성
        data_lines = []
        for row in data:
            line = " | ".join(str(row.get(header, "")).ljust(col_widths[header]) for header in headers)
            data_lines.append(line)
        
        return "\n".join([header_line, separator_line] + data_lines)

# ============================================================================
# 🔄 재시도 및 복구 유틸리티
# ============================================================================
def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                        logging.warning(f"{func.__name__} 실패 (시도 {attempt + 1}/{max_retries}), {wait_time}초 후 재시도: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"{func.__name__} 최종 실패: {e}")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                        logging.warning(f"{func.__name__} 실패 (시도 {attempt + 1}/{max_retries}), {wait_time}초 후 재시도: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"{func.__name__} 최종 실패: {e}")
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# ============================================================================
# 🤖 AI 기반 분석 통합 클래스
# ============================================================================
class AIAnalysisIntegrator:
    """AI 분석 통합 관리 클래스"""
    
    def __init__(self, openai_config: OpenAIConfig):
        if not OPENAI_AVAILABLE:
            logging.warning("OpenAI가 사용 불가능합니다. AI 분석 기능이 제한됩니다.")
            self.openai_helper = None
        else:
            self.openai_helper = OpenAIHelper(openai_config)
        
        self.logger = logging.getLogger(__name__)
    
    async def comprehensive_analysis(self, symbol: str, market_data: Dict[str, Any], 
                                   news_data: List[str] = None, 
                                   user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """종합 분석 (기술적 + 기본적 + 감성 분석)"""
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': {},
            'fundamental_analysis': {},
            'sentiment_analysis': {},
            'ai_recommendations': {},
            'risk_assessment': {},
            'overall_score': 0
        }
        
        if not self.openai_helper:
            results['error'] = "OpenAI가 사용 불가능합니다."
            return results
        
        try:
            # 1. 기술적 분석
            if 'prices' in market_data:
                prices = market_data['prices']
                results['technical_analysis'] = await self._technical_analysis(prices)
            
            # 2. AI 기반 분석
            if self.openai_helper:
                # 기술적 분석
                tech_analysis = await self.openai_helper.analyze_stock(
                    symbol, market_data, AnalysisType.TECHNICAL
                )
                results['ai_technical'] = asdict(tech_analysis)
                
                # 기본적 분석
                fund_analysis = await self.openai_helper.analyze_stock(
                    symbol, market_data, AnalysisType.FUNDAMENTAL
                )
                results['ai_fundamental'] = asdict(fund_analysis)
                
                # 감성 분석
                if news_data:
                    sentiment = await self.openai_helper.analyze_market_sentiment(
                        news_data, [symbol]
                    )
                    results['sentiment_analysis'] = sentiment
                
                # 개인 맞춤 조언
                if user_profile:
                    advice = await self.openai_helper.get_investment_advice(
                        user_profile, market_data
                    )
                    results['personalized_advice'] = advice
            
            # 3. 종합 점수 계산
            results['overall_score'] = self._calculate_overall_score(results)
            
        except Exception as e:
            self.logger.error(f"종합 분석 실패: {e}")
            results['error'] = str(e)
        
        return results
    
    async def _technical_analysis(self, prices: List[float]) -> Dict[str, Any]:
        """기술적 분석 수행"""
        analysis = {}
        
        try:
            if len(prices) >= 20:
                # 이동평균
                sma_20 = TechnicalIndicators.sma(prices, 20)
                sma_50 = TechnicalIndicators.sma(prices, 50) if len(prices) >= 50 else []
                
                analysis['sma_20'] = sma_20[-1] if sma_20 else None
                analysis['sma_50'] = sma_50[-1] if sma_50 else None
                
                # 현재가 vs 이동평균
                current_price = prices[-1]
                if sma_20:
                    analysis['price_vs_sma20'] = (current_price - sma_20[-1]) / sma_20[-1] * 100
                
                # RSI
                rsi_values = TechnicalIndicators.rsi(prices)
                if rsi_values:
                    analysis['rsi'] = rsi_values[-1]
                    analysis['rsi_signal'] = (
                        'OVERSOLD' if rsi_values[-1] < 30 else
                        'OVERBOUGHT' if rsi_values[-1] > 70 else
                        'NEUTRAL'
                    )
                
                # 볼린저 밴드
                upper, middle, lower = TechnicalIndicators.bollinger_bands(prices)
                if upper and middle and lower:
                    analysis['bollinger'] = {
                        'upper': upper[-1],
                        'middle': middle[-1],
                        'lower': lower[-1],
                        'position': (
                            'UPPER' if current_price > upper[-1] else
                            'LOWER' if current_price < lower[-1] else
                            'MIDDLE'
                        )
                    }
                
                # MACD
                macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)
                if macd_line and signal_line:
                    analysis['macd'] = {
                        'macd': macd_line[-1],
                        'signal': signal_line[-1],
                        'histogram': histogram[-1] if histogram else 0,
                        'trend': 'BULLISH' if macd_line[-1] > signal_line[-1] else 'BEARISH'
                    }
        
        except Exception as e:
            self.logger.error(f"기술적 분석 실패: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """종합 점수 계산 (0-100)"""
        score = 50  # 기본 점수
        
        try:
            # 기술적 분석 점수
            tech_score = 0
            if 'technical_analysis' in results:
                tech = results['technical_analysis']
                
                # RSI 점수
                if 'rsi' in tech:
                    rsi = tech['rsi']
                    if 30 <= rsi <= 70:
                        tech_score += 10
                    elif rsi < 30:
                        tech_score += 15  # 과매도 - 매수 기회
                    else:
                        tech_score -= 5   # 과매수 - 위험
                
                # 가격 vs 이동평균
                if 'price_vs_sma20' in tech:
                    if tech['price_vs_sma20'] > 0:
                        tech_score += 10
                    else:
                        tech_score -= 10
                
                # MACD 트렌드
                if 'macd' in tech and tech['macd'].get('trend') == 'BULLISH':
                    tech_score += 10
            
            # AI 분석 점수
            ai_score = 0
            if 'ai_technical' in results:
                confidence = results['ai_technical'].get('confidence', 50)
                ai_score += (confidence - 50) / 5  # 신뢰도를 점수로 변환
            
            # 감성 분석 점수
            sentiment_score = 0
            if 'sentiment_analysis' in results:
                sentiment = results['sentiment_analysis']
                if sentiment.get('overall_sentiment') == 'POSITIVE':
                    sentiment_score += 15
                elif sentiment.get('overall_sentiment') == 'NEGATIVE':
                    sentiment_score -= 15
            
            # 최종 점수 계산
            score = max(0, min(100, score + tech_score + ai_score + sentiment_score))
        
        except Exception as e:
            self.logger.error(f"점수 계산 실패: {e}")
        
        return score

# ============================================================================
# 🎯 편의 함수들
# ============================================================================

# 전역 인스턴스
_currency_converter = CurrencyConverter()
_performance_analyzer = PerformanceAnalyzer()

async def convert_currency(amount: float, from_curr: str, to_curr: str) -> float:
    """간편 환율 변환"""
    try:
        from_currency = Currency(from_curr.upper())
        to_currency = Currency(to_curr.upper())
        return await _currency_converter.convert_amount(amount, from_currency, to_currency)
    except:
        return amount

def format_currency(amount: float, currency: str = 'KRW') -> str:
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

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """변화율 계산"""
    return DataProcessor.safe_divide((new_value - old_value) * 100, old_value, 0.0)

def is_trading_time(market: Market) -> bool:
    """거래 시간 확인"""
    return TimeUtils.is_market_open(market)

def get_safe_filename(filename: str) -> str:
    """안전한 파일명 생성"""
    import re
    # 특수문자 제거
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 연속된 언더스코어 제거
    safe_name = re.sub(r'_+', '_', safe_name)
    return safe_name.strip('_')

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """문자열 자르기"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# OpenAI 관련 편의 함수들
async def quick_stock_analysis(symbol: str, prices: List[float], 
                             openai_api_key: str = None) -> Dict[str, Any]:
    """빠른 주식 분석"""
    if not openai_api_key or not OPENAI_AVAILABLE:
        # 기본 기술적 분석만 수행
        return {
            'symbol': symbol,
            'technical_only': True,
            'rsi': TechnicalIndicators.rsi(prices)[-1] if len(prices) > 14 else None,
            'sma_20': TechnicalIndicators.sma(prices, 20)[-1] if len(prices) >= 20 else None
        }
    
    try:
        config = OpenAIConfig(api_key=openai_api_key)
        integrator = AIAnalysisIntegrator(config)
        
        market_data = {'prices': prices, 'symbol': symbol}
        return await integrator.comprehensive_analysis(symbol, market_data)
    
    except Exception as e:
        logging.error(f"빠른 분석 실패: {e}")
        return {'error': str(e)}

async def ai_investment_advice(portfolio: Dict[str, Any], 
                             user_profile: Dict[str, Any],
                             openai_api_key: str) -> str:
    """AI 투자 조언"""
    if not openai_api_key or not OPENAI_AVAILABLE:
        return "OpenAI API 키가 필요합니다."
    
    try:
        config = OpenAIConfig(api_key=openai_api_key)
        helper = OpenAIHelper(config)
        
        return await helper.get_investment_advice(user_profile, portfolio)
    
    except Exception as e:
        logging.error(f"AI 투자 조언 실패: {e}")
        return f"조언 생성 실패: {str(e)}"

# ============================================================================
# 🎊 시스템 정보 출력
# ============================================================================
def print_system_banner():
    """시스템 배너 출력"""
    banner = f"""
🛠️ ═══════════════════════════════════════════════════════════════════════════ 🛠️
🛠️                        퀸트프로젝트 유틸리티 모듈 v1.2.0                         🛠️
🛠️                               🤖 OpenAI 통합 버전                               🛠️
🛠️ ═══════════════════════════════════════════════════════════════════════════ 🛠️

✨ 핵심 기능:
  📊 데이터 전처리 및 변환      🔢 기술지표 계산 (SMA, EMA, RSI, MACD)
  💰 리스크 관리 도구           🕐 시간 및 날짜 유틸리티
  💱 환율 및 화폐 변환          📈 성과 분석 도구
  📝 로깅 및 알림 헬퍼          💾 파일 I/O 유틸리티
  🔐 보안 및 암호화             🌐 네트워크 유틸리티
  📊 데이터베이스 도구          🔧 시스템 모니터링
  🎨 데이터 시각화              🔄 재시도 및 복구

🤖 AI 통합 기능:
  🎯 AI 기반 주식 분석          📊 시장 감성 분석
  💡 개인 맞춤 투자 조언        📰 뉴스 분석 및 요약
  📈 거래 전략 생성             📚 금융 용어 설명
  📋 시장 보고서 자동 생성      🔍 종합 분석 시스템

🎯 지원 시장: 🇺🇸 미국주식 | 🇰🇷 한국주식 | 🇯🇵 일본주식 | 🇮🇳 인도주식 | 💰 암호화폐

OpenAI 사용 가능: {'✅ 설치됨' if OPENAI_AVAILABLE else '❌ 설치 필요 (pip install openai)'}

🛠️ ═══════════════════════════════════════════════════════════════════════════ 🛠️
"""
    print(banner)

# 모듈 로드시 배너 출력
if __name__ == "__main__":
    print_system_banner()
    
    # 간단한 테스트
    print("🔍 유틸리티 테스트:")
    print(f"  • 숫자 정리: {DataProcessor.clean_numeric_data('1,234.56')}")
    print(f"  • 퍼센트 계산: {calculate_percentage_change(100, 120):.1f}%")
    print(f"  • 통화 포맷: {format_currency(1234567, 'KRW')}")
    print(f"  • 현재 시간: {TimeUtils.get_current_time()}")
    print(f"  • 시스템 정보: CPU {SystemMonitor.get_system_info().get('cpu', {}).get('percent', 0):.1f}%")
    
    if OPENAI_AVAILABLE:
        print("🤖 OpenAI 기능:")
        print("  • AI 주식 분석 준비 완료")
        print("  • 시장 감성 분석 준비 완료")
        print("  • 투자 조언 시스템 준비 완료")
    else:
        print("⚠️  OpenAI 설치 필요:")
        print("  • pip install openai")
    
    print("✅ 모든 유틸리티 정상 로드 완료!")
