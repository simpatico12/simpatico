#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
인도 전설 투자전략 완전판 - 레전드 에디션 + IBKR 연동 + OpenAI 분석
================================================================

5대 투자 거장 철학 + 고급 기술지표 + 자동선별 시스템 + IBKR 자동매매 + OpenAI 분석
- 실시간 자동 매매 신호 생성 + 손절/익절 시스템
- 백테스팅 + 포트폴리오 관리 + 리스크 제어
- OpenAI 기반 시장 분석 및 투자 의사결정 지원
- 혼자 운용 가능한 완전 자동화 전략 + IBKR API 연동

전설의 비밀 공식들과 숨겨진 지표들 모두 구현 + 실제 거래 + AI 분석
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import json
import logging
import threading
import os
import asyncio
warnings.filterwarnings('ignore')

# IBKR API 임포트 (선택사항)
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    print("✅ IBKR API 준비완료")
except ImportError:
    print("ℹ️ IBKR API 없음 (백테스팅만 가능)")
    EClient = None
    EWrapper = None

# OpenAI API 임포트 (새로 추가)
try:
    import openai
    print("✅ OpenAI API 준비완료")
    OPENAI_AVAILABLE = True
except ImportError:
    print("ℹ️ OpenAI API 없음 (pip install openai 필요)")
    OPENAI_AVAILABLE = False
    openai = None

# ================== OpenAI 연동 클래스 (새로 추가) ==================

class OpenAIAnalyzer:
    """OpenAI 기반 투자 분석 시스템"""
    
    def __init__(self, api_key=None):
        self.client = None
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.connected = False
        self.logger = self.setup_logging()
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                openai.api_key = self.api_key
                self.client = openai
                self.connected = True
                self.logger.info("✅ OpenAI 연결 성공!")
            except Exception as e:
                self.logger.error(f"❌ OpenAI 연결 실패: {e}")
        else:
            self.logger.warning("⚠️ OpenAI API 키가 없습니다")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def analyze_market_sentiment(self, stock_data):
        """AI 기반 시장 심리 분석"""
        if not self.connected:
            return {"sentiment": "neutral", "confidence": 0.5, "reasoning": "OpenAI 연결 없음"}
        
        try:
            # 주요 지표 요약
            indicators_summary = self._prepare_indicators_summary(stock_data)
            
            prompt = f"""
            당신은 인도 주식시장 전문 애널리스트입니다. 다음 기술지표 데이터를 분석하여 시장 심리를 평가해주세요:
            
            {indicators_summary}
            
            다음 형식으로 답변해주세요:
            1. 전체적인 시장 심리 (bullish/bearish/neutral)
            2. 신뢰도 (0-1)
            3. 핵심 근거 3가지
            4. 주의사항
            
            간결하고 명확하게 답변해주세요.
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 인도 주식시장 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            sentiment = self._extract_sentiment(analysis)
            
            return {
                "sentiment": sentiment,
                "confidence": 0.8,
                "analysis": analysis,
                "reasoning": "OpenAI GPT-4 분석"
            }
            
        except Exception as e:
            self.logger.error(f"❌ 시장 심리 분석 실패: {e}")
            return {"sentiment": "neutral", "confidence": 0.5, "reasoning": f"분석 오류: {str(e)}"}
    
    def generate_investment_insights(self, selected_stocks, market_data):
        """AI 기반 투자 인사이트 생성"""
        if not self.connected:
            return ["OpenAI 연결이 필요합니다"]
        
        try:
            stocks_summary = self._prepare_stocks_summary(selected_stocks)
            
            prompt = f"""
            인도 주식시장 투자 전문가로서 다음 선별된 종목들을 분석해주세요:
            
            {stocks_summary}
            
            다음 관점에서 분석해주세요:
            1. 각 종목의 투자 매력도
            2. 포트폴리오 구성 시 주의사항
            3. 현재 시장 상황에서의 리스크 요인
            4. 단기 투자 전략 권장사항
            
            실용적이고 구체적인 조언을 해주세요.
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 20년 경력의 인도 주식시장 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.4
            )
            
            insights = response.choices[0].message.content
            return insights.split('\n')
            
        except Exception as e:
            self.logger.error(f"❌ 투자 인사이트 생성 실패: {e}")
            return [f"인사이트 생성 오류: {str(e)}"]
    
    def risk_assessment_ai(self, portfolio, market_conditions):
        """AI 기반 리스크 평가"""
        if not self.connected:
            return {"risk_level": "medium", "warnings": ["OpenAI 연결 필요"]}
        
        try:
            portfolio_summary = self._prepare_portfolio_summary(portfolio)
            
            prompt = f"""
            다음 포트폴리오의 리스크를 전문적으로 평가해주세요:
            
            {portfolio_summary}
            
            평가 기준:
            1. 섹터 집중도 리스크
            2. 시장 변동성 리스크
            3. 개별 종목 리스크
            4. 유동성 리스크
            
            리스크 레벨 (low/medium/high)과 구체적인 경고사항을 제시해주세요.
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 리스크 관리 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.2
            )
            
            assessment = response.choices[0].message.content
            risk_level = self._extract_risk_level(assessment)
            warnings = self._extract_warnings(assessment)
            
            return {
                "risk_level": risk_level,
                "warnings": warnings,
                "full_assessment": assessment
            }
            
        except Exception as e:
            self.logger.error(f"❌ AI 리스크 평가 실패: {e}")
            return {"risk_level": "medium", "warnings": [f"평가 오류: {str(e)}"]}
    
    def optimize_entry_timing(self, stock_ticker, technical_data):
        """AI 기반 진입 타이밍 최적화"""
        if not self.connected:
            return {"timing": "wait", "confidence": 0.5}
        
        try:
            tech_summary = self._prepare_technical_summary(technical_data)
            
            prompt = f"""
            {stock_ticker} 종목의 기술적 분석 데이터를 보고 최적의 진입 타이밍을 판단해주세요:
            
            {tech_summary}
            
            결론:
            1. 진입 권장도 (buy_now/wait/avoid)
            2. 신뢰도 (0-1)
            3. 핵심 근거
            4. 예상 목표가
            
            간결하게 답변해주세요.
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 기술적 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            timing_analysis = response.choices[0].message.content
            timing = self._extract_timing_decision(timing_analysis)
            
            return {
                "timing": timing,
                "confidence": 0.75,
                "analysis": timing_analysis
            }
            
        except Exception as e:
            self.logger.error(f"❌ 진입 타이밍 분석 실패: {e}")
            return {"timing": "wait", "confidence": 0.5}
    
    def generate_daily_report(self, strategy_results):
        """AI 기반 일일 투자 리포트 생성"""
        if not self.connected:
            return "일일 리포트 생성을 위해 OpenAI 연결이 필요합니다."
        
        try:
            results_summary = self._prepare_results_summary(strategy_results)
            
            prompt = f"""
            오늘의 인도 주식 투자전략 실행 결과를 바탕으로 전문적인 일일 리포트를 작성해주세요:
            
            {results_summary}
            
            리포트 구성:
            1. 오늘의 주요 성과 요약
            2. 선별된 종목들의 투자 포인트
            3. 포트폴리오 현황 평가
            4. 내일의 투자 전략 방향
            5. 주의사항 및 리스크 경고
            
            전문적이면서도 이해하기 쉽게 작성해주세요.
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 인도 주식 투자 전문가이자 애널리스트입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.4
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"❌ 일일 리포트 생성 실패: {e}")
            return f"리포트 생성 중 오류 발생: {str(e)}"
    
    # 보조 메서드들
    def _prepare_indicators_summary(self, stock_data):
        """기술지표 요약 준비"""
        if stock_data.empty:
            return "데이터 없음"
        
        summary = f"""
        주요 기술지표 현황:
        - RSI: {stock_data.get('rsi', 'N/A').iloc[-1] if 'rsi' in stock_data.columns else 'N/A'}
        - MACD: {stock_data.get('macd_histogram', 'N/A').iloc[-1] if 'macd_histogram' in stock_data.columns else 'N/A'}
        - ADX: {stock_data.get('adx', 'N/A').iloc[-1] if 'adx' in stock_data.columns else 'N/A'}
        - 볼린저밴드 위치: {'상단' if stock_data.get('close', 0).iloc[-1] > stock_data.get('bb_upper', 0).iloc[-1] else '하단' if stock_data.get('close', 0).iloc[-1] < stock_data.get('bb_lower', 0).iloc[-1] else '중간' if 'close' in stock_data.columns and 'bb_upper' in stock_data.columns else 'N/A'}
        - 거래량 급증: {'예' if stock_data.get('volume_spike', False).iloc[-1] else '아니오' if 'volume_spike' in stock_data.columns else 'N/A'}
        """
        return summary
    
    def _prepare_stocks_summary(self, selected_stocks):
        """선별 종목 요약 준비"""
        if selected_stocks.empty:
            return "선별된 종목 없음"
        
        summary = "선별된 종목들:\n"
        for _, stock in selected_stocks.head(5).iterrows():
            summary += f"- {stock.get('ticker', 'N/A')}: 점수 {stock.get('final_score', 'N/A')}, 주가 ₹{stock.get('close', 'N/A')}\n"
        
        return summary
    
    def _prepare_portfolio_summary(self, portfolio):
        """포트폴리오 요약 준비"""
        if not portfolio:
            return "포트폴리오 없음"
        
        summary = "포트폴리오 구성:\n"
        for ticker, details in portfolio.items():
            summary += f"- {ticker}: ₹{details.get('allocation', 'N/A'):,.0f} ({details.get('weight_pct', 'N/A'):.1f}%)\n"
        
        return summary
    
    def _prepare_technical_summary(self, technical_data):
        """기술적 분석 요약 준비"""
        return f"""
        기술적 지표:
        - 추세: {'상승' if technical_data.get('trend', 'neutral') == 'bullish' else '하락' if technical_data.get('trend', 'neutral') == 'bearish' else '중립'}
        - 모멘텀: {technical_data.get('momentum', 'N/A')}
        - 지지/저항: {technical_data.get('support_resistance', 'N/A')}
        """
    
    def _prepare_results_summary(self, results):
        """전략 결과 요약 준비"""
        selected_count = len(results.get('selected_stocks', pd.DataFrame()))
        portfolio_count = len(results.get('portfolio', {}))
        alerts_count = len(results.get('alerts', []))
        
        return f"""
        투자전략 실행 결과:
        - 선별된 종목 수: {selected_count}개
        - 포트폴리오 구성: {portfolio_count}개 종목
        - 생성된 알림: {alerts_count}개
        - IBKR 연결: {'성공' if results.get('ibkr_connected') else '실패'}
        - 수요일 거래 가능: {'예' if results.get('wednesday_status', {}).get('is_wednesday') else '아니오'}
        """
    
    def _extract_sentiment(self, analysis):
        """분석에서 감정 추출"""
        analysis_lower = analysis.lower()
        if 'bullish' in analysis_lower or '상승' in analysis_lower:
            return 'bullish'
        elif 'bearish' in analysis_lower or '하락' in analysis_lower:
            return 'bearish'
        else:
            return 'neutral'
    
    def _extract_risk_level(self, assessment):
        """평가에서 리스크 레벨 추출"""
        assessment_lower = assessment.lower()
        if 'high' in assessment_lower or '높' in assessment_lower:
            return 'high'
        elif 'low' in assessment_lower or '낮' in assessment_lower:
            return 'low'
        else:
            return 'medium'
    
    def _extract_warnings(self, assessment):
        """평가에서 경고사항 추출"""
        lines = assessment.split('\n')
        warnings = []
        for line in lines:
            if '경고' in line or '주의' in line or 'warning' in line.lower():
                warnings.append(line.strip())
        return warnings[:3]  # 최대 3개
    
    def _extract_timing_decision(self, analysis):
        """분석에서 타이밍 결정 추출"""
        analysis_lower = analysis.lower()
        if 'buy_now' in analysis_lower or '즉시' in analysis_lower:
            return 'buy_now'
        elif 'avoid' in analysis_lower or '회피' in analysis_lower:
            return 'avoid'
        else:
            return 'wait'

# ================== IBKR 연동 클래스 (기존) ==================

class IBKRConnector:
    """간단한 IBKR 연결 클래스"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.positions = {}
        self.logger = self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def connect(self, host='127.0.0.1', port=7497, client_id=1):
        """IBKR 연결"""
        if not EClient:
            self.logger.error("❌ IBKR API가 설치되지 않았습니다")
            return False
            
        try:
            # 실제 연결 로직은 여기에 구현
            self.logger.info("🔗 IBKR 연결 시도중...")
            # self.client = IBKRClient()
            # self.client.connect(host, port, client_id)
            self.connected = True
            self.logger.info("✅ IBKR 연결 성공!")
            return True
        except Exception as e:
            self.logger.error(f"❌ IBKR 연결 실패: {e}")
            return False
    
    def create_contract(self, symbol):
        """인도 주식 계약 생성"""
        if not EClient:
            return None
            
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "NSE"
        contract.currency = "INR"
        return contract
    
    def place_buy_order(self, symbol, quantity, price=None):
        """매수 주문"""
        if not self.connected:
            self.logger.error("❌ IBKR 연결 필요")
            return False
            
        self.logger.info(f"📈 매수 주문: {symbol} {quantity}주 @₹{price or 'Market'}")
        # 실제 주문 로직
        return True
    
    def place_sell_order(self, symbol, quantity, price=None):
        """매도 주문"""
        if not self.connected:
            self.logger.error("❌ IBKR 연결 필요")
            return False
            
        self.logger.info(f"📉 매도 주문: {symbol} {quantity}주 @₹{price or 'Market'}")
        # 실제 주문 로직
        return True
    
    def get_positions(self):
        """포지션 조회"""
        # 샘플 포지션 데이터
        return {
            'RELIANCE': {'quantity': 100, 'avg_cost': 2500},
            'TCS': {'quantity': 50, 'avg_cost': 3200}
        }

class LegendaryIndiaStrategy:
    """인도 전설 투자자 5인방 통합 전략 (원본 + IBKR + OpenAI 연동)"""
    
    def __init__(self, openai_api_key=None):
        self.portfolio = {}
        self.signals = {}
        self.risk_metrics = {}
        
        # IBKR 연결 (기존)
        self.ibkr = IBKRConnector()
        
        # OpenAI 연결 (새로 추가)
        self.ai_analyzer = OpenAIAnalyzer(openai_api_key)
        
    # ================== 기본 + 전설급 기술지표 라이브러리 (기존) ==================
    
    def bollinger_bands(self, df, period=20, std_dev=2):
        """볼린저 밴드 + 스퀴즈 감지"""
        df['bb_middle'] = df['close'].rolling(period).mean()
        df['bb_std'] = df['close'].rolling(period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(50).quantile(0.1)
        return df
    
    def advanced_macd(self, df, fast=12, slow=26, signal=9):
        """MACD + 히스토그램 + 다이버전스"""
        df['ema_fast'] = df['close'].ewm(span=fast).mean()
        df['ema_slow'] = df['close'].ewm(span=slow).mean()
        df['macd_line'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd_line'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        df['macd_momentum'] = df['macd_histogram'].diff()
        return df
    
    def adx_system(self, df, period=14):
        """ADX + DI 시스템 (추세 강도 측정)"""
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > 
                                (df['low'].shift(1) - df['low']), 
                                df['high'] - df['high'].shift(1), 0)
        df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > 
                                 (df['high'] - df['high'].shift(1)), 
                                 df['low'].shift(1) - df['low'], 0)
        
        df['plus_di'] = 100 * (df['plus_dm'].rolling(period).mean() / 
                              df['true_range'].rolling(period).mean())
        df['minus_di'] = 100 * (df['minus_dm'].rolling(period).mean() / 
                               df['true_range'].rolling(period).mean())
        df['adx'] = 100 * abs(df['plus_di'] - df['minus_di']).rolling(period).mean() / \
                   (df['plus_di'] + df['minus_di'])
        return df
    
    def stochastic_slow(self, df, k_period=14, d_period=3):
        """스토캐스틱 슬로우 + 과매수/과매도"""
        df['lowest_low'] = df['low'].rolling(k_period).min()
        df['highest_high'] = df['high'].rolling(k_period).max()
        df['stoch_k'] = 100 * (df['close'] - df['lowest_low']) / \
                       (df['highest_high'] - df['lowest_low'])
        df['stoch_d'] = df['stoch_k'].rolling(d_period).mean()
        df['stoch_slow'] = df['stoch_d'].rolling(d_period).mean()
        return df
    
    def volume_profile(self, df, period=20):
        """거래량 프로파일 + 이상 급증 감지"""
        df['volume_sma'] = df['volume'].rolling(period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_spike'] = df['volume_ratio'] > 2.0
        df['volume_momentum'] = df['volume'].pct_change(5)
        return df
    
    def rsi_advanced(self, df, period=14):
        """RSI + 다이버전스 감지"""
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_oversold'] = df['rsi'] < 30
        df['rsi_overbought'] = df['rsi'] > 70
        return df
    
    def ichimoku_cloud(self, df, tenkan=9, kijun=26, senkou_b=52):
        """일목균형표 - 트렌드 + 지지저항 + 미래 예측"""
        df['tenkan_sen'] = (df['high'].rolling(tenkan).max() + df['low'].rolling(tenkan).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(kijun).max() + df['low'].rolling(kijun).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun)
        df['senkou_span_b'] = ((df['high'].rolling(senkou_b).max() + df['low'].rolling(senkou_b).min()) / 2).shift(kijun)
        df['chikou_span'] = df['close'].shift(-kijun)
        
        # 구름 두께 (변동성 지표)
        df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])
        
        # 구름 위/아래 신호
        df['above_cloud'] = (df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b'])
        df['below_cloud'] = (df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b'])
        
        # TK 크로스
        df['tk_bullish'] = (df['tenkan_sen'] > df['kijun_sen']) & (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1))
        df['tk_bearish'] = (df['tenkan_sen'] < df['kijun_sen']) & (df['tenkan_sen'].shift(1) >= df['kijun_sen'].shift(1))
        
        return df
    
    def elliott_wave_detector(self, df, lookback=50):
        """엘리어트 파동 감지 - 간소화 버전"""
        # 단순한 파동 강도 계산
        df['wave_strength'] = abs(df['close'].pct_change(lookback))
        
        # 피보나치 레벨 계산
        df['high_50'] = df['high'].rolling(lookback).max()
        df['low_50'] = df['low'].rolling(lookback).min()
        df['fib_236'] = df['low_50'] + (df['high_50'] - df['low_50']) * 0.236
        df['fib_382'] = df['low_50'] + (df['high_50'] - df['low_50']) * 0.382
        df['fib_618'] = df['low_50'] + (df['high_50'] - df['low_50']) * 0.618
        
        # 간단한 파동 완성 신호
        df['wave_5_complete'] = (df['close'] > df['fib_618']) & (df['rsi'] > 70)
        df['wave_c_complete'] = (df['close'] < df['fib_382']) & (df['rsi'] < 30)
        
        return df
    
    def vwap_advanced(self, df, period=20):
        """고급 VWAP - 거래량 가중 평균가 + 편차밴드"""
        # 기본 VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
        
        # VWAP 편차 계산
        df['vwap_deviation'] = df['close'] - df['vwap']
        df['vwap_std'] = df['vwap_deviation'].rolling(period).std()
        
        # VWAP 밴드
        df['vwap_upper'] = df['vwap'] + df['vwap_std'] * 2
        df['vwap_lower'] = df['vwap'] - df['vwap_std'] * 2
        
        # 기관 매매 신호 (대량거래 + VWAP 돌파)
        df['institutional_buying'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5) & (df['close'] > df['vwap'])
        df['institutional_selling'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5) & (df['close'] < df['vwap'])
        
        return df
    
    def detect_divergence(self, price, indicator, lookback=20):
        """가격-지표 다이버전스 감지 (안전한 버전)"""
        try:
            # 간단한 다이버전스 계산
            price_change = price.diff(lookback)
            indicator_change = indicator.diff(lookback)
            
            # 다이버전스: 가격과 지표가 반대 방향
            bullish_div = (price_change < 0) & (indicator_change > 0)
            bearish_div = (price_change > 0) & (indicator_change < 0)
            
            return (bullish_div.astype(int) - bearish_div.astype(int))
        except:
            # 에러 시 0 반환
            return pd.Series(0, index=price.index)
    
    def calculate_all_legendary_indicators(self, df):
        """모든 전설급 기술지표 계산"""
        print("🔥 전설급 기술지표 계산 시작...")
        
        # 기존 기본 지표들
        df = self.bollinger_bands(df)
        df = self.advanced_macd(df)
        df = self.adx_system(df)
        df = self.stochastic_slow(df)
        df = self.volume_profile(df)
        df = self.rsi_advanced(df)
        
        # 전설급 고급 지표들
        df = self.ichimoku_cloud(df)
        df = self.elliott_wave_detector(df)
        df = self.vwap_advanced(df)
        
        print("✅ 전설급 기술지표 계산 완료!")
        return df
    
    # ================== 전설 투자자 전략 구현 (기존) ==================
    
    def rakesh_jhunjhunwala_strategy(self, df):
        """라케시 준준왈라 - 워런 버핏 킬러 전략"""
        # 3-5-7 룰 구현
        df['roe_trend'] = (df['ROE'] > 15).astype(int) if 'ROE' in df.columns else 0
        df['profit_streak'] = (df['Operating_Profit'] > 0).astype(int) if 'Operating_Profit' in df.columns else 0
        df['dividend_streak'] = (df['Dividend_Yield'] > 1.0).astype(int) if 'Dividend_Yield' in df.columns else 0
        
        # 경영진 지분율 + 프로모터 pledge 체크
        if 'Promoter_Holding' in df.columns and 'Promoter_Pledge' in df.columns:
            df['promoter_strength'] = ((df['Promoter_Holding'] >= 30) & (df['Promoter_Pledge'] <= 15)).astype(int)
        else:
            df['promoter_strength'] = 0
        
        # 준준왈라 스코어
        df['jhunjhunwala_score'] = (
            df['roe_trend'] * 3 +
            df['profit_streak'] * 2 +
            df['dividend_streak'] * 1 +
            df['promoter_strength'] * 2
        )
        return df
    
    def raamdeo_agrawal_qglp(self, df):
        """라메데오 아그라왈 - QGLP 진화 전략"""
        # Quality (품질) - 복합 지표
        df['quality_score'] = 0
        if 'Debt_to_Equity' in df.columns:
            df['quality_score'] += (df['Debt_to_Equity'] < 0.5).astype(int) * 2
        if 'Current_Ratio' in df.columns:
            df['quality_score'] += (df['Current_Ratio'] > 1.5).astype(int) * 1
        
        # Growth (성장) - 단순화
        df['growth_score'] = 0
        if 'EPS_growth' in df.columns:
            df['growth_score'] += (df['EPS_growth'] > 0.20).astype(int) * 2
        
        # QGLP 종합 점수
        df['qglp_score'] = df['quality_score'] + df['growth_score']
        return df
    
    def vijay_kedia_smile(self, df):
        """비제이 케디아 - SMILE 투자법"""
        df['market_cap_score'] = 3  # 기본값
        if 'Market_Cap' in df.columns:
            df['market_cap_score'] = np.where(df['Market_Cap'] < 50000, 3,
                                     np.where(df['Market_Cap'] < 200000, 2, 1))
        
        df['smile_score'] = df['market_cap_score'] * 2
        return df
    
    def porinju_veliyath_contrarian(self, df):
        """포리뉴 벨리야스 - 콘트라리안 마스터"""
        df['underdog_score'] = 0
        if 'Analyst_Coverage' in df.columns:
            df['underdog_score'] += (df['Analyst_Coverage'] <= 2).astype(int) * 2
        if 'PBV' in df.columns:
            df['underdog_score'] += (df['PBV'] < 1.0).astype(int) * 2
        return df
    
    def nitin_karnik_infra(self, df):
        """니틴 카르닉 - 인프라 제왕 전략"""
        df['karnik_score'] = 2  # 기본 점수
        return df
    
    # ================== 자동 선별 시스템 (기존) ==================
    
    def calculate_all_indicators(self, df):
        """모든 전설급 기술지표 계산"""
        return self.calculate_all_legendary_indicators(df)
    
    def apply_all_strategies(self, df):
        """5대 전설 전략 모두 적용"""
        df = self.rakesh_jhunjhunwala_strategy(df)
        df = self.raamdeo_agrawal_qglp(df)
        df = self.vijay_kedia_smile(df)
        df = self.porinju_veliyath_contrarian(df)
        df = self.nitin_karnik_infra(df)
        return df
    
    def generate_master_score(self, df):
        """마스터 통합 점수 생성 - 전설급 기술지표 반영"""
        # 각 전략별 가중치
        weights = {
            'jhunjhunwala_score': 0.25,
            'qglp_score': 0.25,
            'smile_score': 0.20,
            'underdog_score': 0.15,
            'karnik_score': 0.15
        }
        
        df['master_score'] = 0
        for strategy, weight in weights.items():
            if strategy in df.columns:
                df['master_score'] += df[strategy] * weight
        
        # 전설급 기술적 지표 보정 (대폭 강화!)
        df['legendary_technical_bonus'] = (
            # 기본 모멘텀 지표
            (df['macd_histogram'] > 0).astype(int) * 1 +
            (df['adx'] > 25).astype(int) * 1 +
            (~df['rsi_overbought']).astype(int) * 1 +
            df['volume_spike'].astype(int) * 1 +
            df['bb_squeeze'].astype(int) * 2 +
            
            # 일목균형표 시스템
            df['above_cloud'].astype(int) * 3 +
            df['tk_bullish'].astype(int) * 2 +
            
            # 엘리어트 파동
            df['wave_5_complete'].astype(int) * 2 +
            (df['wave_strength'] > 0.1).astype(int) * 1 +
            
            # VWAP 시스템
            df['institutional_buying'].astype(int) * 2 +
            (df['close'] > df['vwap']).astype(int) * 1
        )
        
        df['final_score'] = df['master_score'] + df['legendary_technical_bonus']
        return df
    
    def auto_stock_selection(self, df, top_n=10):
        """자동 종목 선별"""
        # 기본 필터링
        basic_filter = (
            (df['Market_Cap'] > 1000) if 'Market_Cap' in df.columns else True
        )
        
        # 필터링된 데이터에서 상위 종목 선별
        if isinstance(basic_filter, bool):
            filtered_df = df.copy()
        else:
            filtered_df = df[basic_filter].copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        # 점수 순으로 정렬
        selected_stocks = filtered_df.nlargest(top_n, 'final_score')
        
        # 안전한 컬럼 반환
        return_columns = ['ticker', 'company_name', 'final_score', 'master_score', 'close']
        available_columns = [col for col in return_columns if col in selected_stocks.columns]
        
        return selected_stocks[available_columns] if available_columns else selected_stocks
    
    # ================== OpenAI 기반 AI 분석 기능 (새로 추가) ==================
    
    def ai_enhanced_stock_selection(self, df, top_n=10):
        """AI 강화 종목 선별"""
        print("🤖 OpenAI 기반 AI 종목 선별 시작...")
        
        # 기본 선별
        basic_selected = self.auto_stock_selection(df, top_n * 2)  # 더 많이 선별
        
        if basic_selected.empty or not self.ai_analyzer.connected:
            print("ℹ️ AI 분석 없이 기본 선별 결과 반환")
            return basic_selected.head(top_n)
        
        # AI 기반 시장 심리 분석
        market_sentiment = self.ai_analyzer.analyze_market_sentiment(df)
        print(f"🧠 AI 시장 심리: {market_sentiment['sentiment']} (신뢰도: {market_sentiment['confidence']:.1f})")
        
        # 각 종목별 AI 타이밍 분석
        ai_scores = []
        for _, stock in basic_selected.iterrows():
            ticker = stock['ticker']
            
            # 해당 종목의 기술적 데이터 준비
            stock_tech_data = {
                'trend': 'bullish' if stock.get('final_score', 0) > 15 else 'neutral',
                'momentum': stock.get('macd_histogram', 0),
                'support_resistance': f"지지: {stock.get('bb_lower', 0):.2f}, 저항: {stock.get('bb_upper', 0):.2f}"
            }
            
            # AI 타이밍 분석
            timing_result = self.ai_analyzer.optimize_entry_timing(ticker, stock_tech_data)
            
            # AI 점수 계산
            ai_bonus = 0
            if timing_result['timing'] == 'buy_now':
                ai_bonus = 5
            elif timing_result['timing'] == 'wait':
                ai_bonus = 0
            else:  # avoid
                ai_bonus = -10
            
            ai_bonus *= timing_result['confidence']
            
            ai_scores.append({
                'ticker': ticker,
                'ai_timing_score': ai_bonus,
                'ai_confidence': timing_result['confidence'],
                'ai_recommendation': timing_result['timing']
            })
        
        # AI 점수를 DataFrame에 추가
        ai_df = pd.DataFrame(ai_scores)
        enhanced_selection = basic_selected.merge(ai_df, on='ticker', how='left')
        
        # 최종 점수에 AI 점수 반영
        enhanced_selection['ai_enhanced_score'] = (
            enhanced_selection['final_score'] + 
            enhanced_selection['ai_timing_score'].fillna(0)
        )
        
        # AI 강화 점수로 재정렬
        final_selection = enhanced_selection.nlargest(top_n, 'ai_enhanced_score')
        
        print(f"✅ AI 강화 종목 선별 완료: {len(final_selection)}개 종목")
        return final_selection
    
    def ai_risk_assessment(self, portfolio, selected_stocks):
        """AI 기반 리스크 평가"""
        if not self.ai_analyzer.connected:
            return self.risk_management(pd.DataFrame())  # 기본 리스크 평가로 fallback
        
        print("🛡️ AI 기반 리스크 평가 시작...")
        
        # 시장 조건 준비
        market_conditions = {
            'volatility': 'medium',
            'sector_rotation': 'active',
            'liquidity': 'good'
        }
        
        # AI 리스크 평가
        ai_risk = self.ai_analyzer.risk_assessment_ai(portfolio, market_conditions)
        
        # 기존 리스크 메트릭과 결합
        traditional_risk = self.risk_management(pd.DataFrame())
        
        # AI 강화 리스크 메트릭
        enhanced_risk = {
            **traditional_risk,
            'ai_risk_level': ai_risk['risk_level'],
            'ai_warnings': ai_risk['warnings'],
            'ai_assessment': ai_risk.get('full_assessment', ''),
            'overall_risk_score': self._calculate_overall_risk_score(ai_risk, traditional_risk)
        }
        
        print(f"🛡️ AI 리스크 레벨: {ai_risk['risk_level']}")
        print(f"⚠️ AI 경고 수: {len(ai_risk['warnings'])}")
        
        return enhanced_risk
    
    def _calculate_overall_risk_score(self, ai_risk, traditional_risk):
        """전체 리스크 점수 계산"""
        risk_mapping = {'low': 1, 'medium': 2, 'high': 3}
        
        ai_score = risk_mapping.get(ai_risk['risk_level'], 2)
        traditional_score = 2  # 기본값
        
        # 가중 평균 (AI 60%, 전통적 40%)
        overall_score = (ai_score * 0.6) + (traditional_score * 0.4)
        
        if overall_score <= 1.5:
            return 'low'
        elif overall_score <= 2.5:
            return 'medium'
        else:
            return 'high'
    
    def generate_ai_insights(self, strategy_results):
        """AI 기반 투자 인사이트 생성"""
        if not self.ai_analyzer.connected:
            return ["AI 연결이 필요합니다"]
        
        print("🧠 AI 투자 인사이트 생성 중...")
        
        selected_stocks = strategy_results.get('selected_stocks', pd.DataFrame())
        portfolio = strategy_results.get('portfolio', {})
        
        # 시장 데이터 준비 (간단한 버전)
        market_data = {
            'trend': 'bullish',
            'volatility': 'medium',
            'volume': 'above_average'
        }
        
        # AI 인사이트 생성
        insights = self.ai_analyzer.generate_investment_insights(selected_stocks, market_data)
        
        print("✅ AI 투자 인사이트 생성 완료")
        return insights if isinstance(insights, list) else [insights]
    
    def create_ai_daily_report(self, strategy_results):
        """AI 기반 일일 투자 리포트"""
        if not self.ai_analyzer.connected:
            return "AI 일일 리포트 생성을 위해 OpenAI 연결이 필요합니다."
        
        print("📊 AI 일일 리포트 생성 중...")
        
        report = self.ai_analyzer.generate_daily_report(strategy_results)
        
        print("✅ AI 일일 리포트 생성 완료")
        return report
    
    # ================== 2주 스윙 손익절 시스템 (기존) ==================
    
    def calculate_swing_stops(self, df):
        """2주 스윙용 동적 손익절가 계산"""
        
        # 지수별 기본 손익절비
        stop_loss_pct = {
            'NIFTY50': 0.07,   # -7%
            'SENSEX': 0.07,    # -7%  
            'NEXT50': 0.09,    # -9%
            'SMALLCAP': 0.11   # -11%
        }
        
        take_profit_pct = {
            'NIFTY50': 0.14,   # +14%
            'SENSEX': 0.14,    # +14%
            'NEXT50': 0.18,    # +18%  
            'SMALLCAP': 0.22   # +22%
        }
        
        # 각 종목별 손익절가 계산
        df['stop_loss_price'] = 0
        df['take_profit_price'] = 0
        df['swing_stop_pct'] = 0
        df['swing_profit_pct'] = 0
        
        for idx, row in df.iterrows():
            index_cat = row.get('index_category', 'OTHER')
            current_price = row.get('close', row.get('Price', 0))
            
            # 지수별 손익절비 설정
            if 'NIFTY50' in str(index_cat):
                stop_pct = stop_loss_pct['NIFTY50']
                profit_pct = take_profit_pct['NIFTY50']
            elif 'SENSEX' in str(index_cat):
                stop_pct = stop_loss_pct['SENSEX']
                profit_pct = take_profit_pct['SENSEX']
            elif 'NEXT50' in str(index_cat):
                stop_pct = stop_loss_pct['NEXT50']
                profit_pct = take_profit_pct['NEXT50']
            elif 'SMALLCAP' in str(index_cat):
                stop_pct = stop_loss_pct['SMALLCAP']
                profit_pct = take_profit_pct['SMALLCAP']
            else:
                stop_pct = 0.08  # 기본값
                profit_pct = 0.16
            
            # 전설급 신호 강도에 따른 조정
            final_score = row.get('final_score', 0)
            if final_score > 20:  # 전설급 신호
                stop_pct *= 1.5  # 손절 여유있게
                profit_pct *= 1.8  # 익절 크게
            elif final_score > 15:  # 강한 신호
                stop_pct *= 1.2
                profit_pct *= 1.4
            
            # 손익절가 계산
            if current_price > 0:
                df.loc[idx, 'stop_loss_price'] = current_price * (1 - stop_pct)
                df.loc[idx, 'take_profit_price'] = current_price * (1 + profit_pct)
                df.loc[idx, 'swing_stop_pct'] = stop_pct * 100
                df.loc[idx, 'swing_profit_pct'] = profit_pct * 100
        
        return df
    
    def track_current_positions(self):
        """현재 포지션 추적 및 상태 출력"""
        from datetime import datetime, timedelta
        
        # 샘플 포지션 (실제로는 DB나 파일에서 로드)
        positions = {
            'RELIANCE': {
                'entry_date': '2024-12-20',
                'entry_price': 2450,
                'current_price': 2528,
                'stop_loss': 2278,
                'take_profit': 2793,
                'index_category': 'NIFTY50'
            },
            'TCS': {
                'entry_date': '2024-12-18', 
                'entry_price': 3200,
                'current_price': 3256,
                'stop_loss': 2976,
                'take_profit': 3648,
                'index_category': 'NIFTY50'
            },
            'HDFCBANK': {
                'entry_date': '2024-12-22',
                'entry_price': 1650,
                'current_price': 1623,
                'stop_loss': 1535,
                'take_profit': 1881,
                'index_category': 'SENSEX'
            }
        }
        
        position_status = []
        today = datetime.now()
        
        for ticker, pos in positions.items():
            # 경과일 계산
            entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
            days_held = (today - entry_date).days
            days_remaining = 14 - days_held
            
            # 손익률 계산
            pnl_pct = ((pos['current_price'] - pos['entry_price']) / pos['entry_price']) * 100
            
            # 손절/익절선까지 거리
            stop_distance = ((pos['current_price'] - pos['stop_loss']) / pos['current_price']) * 100
            profit_distance = ((pos['take_profit'] - pos['current_price']) / pos['current_price']) * 100
            
            # 상태 결정
            if pnl_pct >= 0:
                status = "🟢"
            elif pnl_pct > -3:
                status = "🟡"
            else:
                status = "🔴"
            
            position_status.append({
                'ticker': ticker,
                'days_held': days_held,
                'days_remaining': max(0, days_remaining),
                'pnl_pct': pnl_pct,
                'stop_distance': stop_distance,
                'profit_distance': profit_distance,
                'status': status,
                'current_price': pos['current_price'],
                'entry_price': pos['entry_price']
            })
        
        return position_status
    
    def essential_alerts(self):
        """핵심 알림 시스템"""
        alerts = []
        
        # 현재 포지션 상태 가져오기
        positions = self.track_current_positions()
        
        for pos in positions:
            ticker = pos['ticker']
            
            # 1. 손절선 80% 근접 경고
            if pos['stop_distance'] < 20:  # 손절선까지 20% 미만
                alerts.append(f"🚨 {ticker} 손절선 근접! 현재 거리: {pos['stop_distance']:.1f}%")
            
            # 2. 익절 달성
            if pos['pnl_pct'] >= 10:
                alerts.append(f"🎯 {ticker} 익절 기회! 수익률: +{pos['pnl_pct']:.1f}%")
            
            # 3. 2주 만료 임박 (2일 이하)
            if pos['days_remaining'] <= 2 and pos['days_remaining'] > 0:
                alerts.append(f"⏰ {ticker} 만료 {pos['days_remaining']}일 전 - 포지션 정리 검토")
            
            # 4. 2주 초과 홀딩
            if pos['days_remaining'] <= 0:
                alerts.append(f"🔄 {ticker} 2주 초과 홀딩 - 즉시 정리 권장")
        
        # 5. 신규 매수 기회 (샘플)
        new_opportunities = ['WIPRO', 'BAJFINANCE', 'MARUTI']
        for stock in new_opportunities[:1]:  # 1개만 샘플로
            alerts.append(f"💎 {stock} 새로운 전설급 매수 신호 감지")
        
        return alerts
    
    def apply_index_specific_strategy(self, df):
        """지수별 맞춤 전략 적용"""
        print("🎯 지수별 맞춤 전략 적용 중...")
        
        # 지수별 가중치 조정
        df['index_bonus'] = 0
        
        for idx, row in df.iterrows():
            index_cat = row.get('index_category', '')
            
            if 'NIFTY50' in str(index_cat):
                # 대형주 - 안정성 중심 (준준왈라 + QGLP 강화)
                df.loc[idx, 'index_bonus'] = (
                    row.get('jhunjhunwala_score', 0) * 0.4 +
                    row.get('qglp_score', 0) * 0.3 +
                    (row.get('ROE', 0) > 20) * 2
                )
                
            elif 'SENSEX' in str(index_cat):
                # 블루칩 - 품질 우선
                df.loc[idx, 'index_bonus'] = (
                    row.get('qglp_score', 0) * 0.5 +
                    row.get('jhunjhunwala_score', 0) * 0.3
                )
                
            elif 'NEXT50' in str(index_cat):
                # 성장주 - 성장성 중심
                df.loc[idx, 'index_bonus'] = (
                    row.get('smile_score', 0) * 0.4 +
                    row.get('legendary_technical_bonus', 0) * 0.3
                )
                
            elif 'SMALLCAP' in str(index_cat):
                # 소형주 - 밸류 발굴
                df.loc[idx, 'index_bonus'] = (
                    row.get('underdog_score', 0) * 0.4 +
                    row.get('karnik_score', 0) * 0.3
                )
        
        # 최종 점수에 지수 보너스 반영
        df['final_score_with_index'] = df['final_score'] + df['index_bonus']
        
        return df
    
    def select_by_index(self, df, top_per_index=5):
        """지수별 상위 종목 선별"""
        index_results = {}
        
        # 4개 지수별로 분리 선별
        for index_name in ['NIFTY50', 'SENSEX', 'NEXT50', 'SMALLCAP']:
            index_stocks = df[df.get('index_category', '').str.contains(index_name, na=False)].copy()
            
            if len(index_stocks) > 0:
                # 해당 지수 내에서 상위 종목 선별
                score_col = 'final_score_with_index' if 'final_score_with_index' in index_stocks.columns else 'final_score'
                top_stocks = index_stocks.nlargest(top_per_index, score_col)
                index_results[index_name] = top_stocks[[
                    'ticker', 'company_name', score_col, 'close'
                ]]
        
        return index_results
    
    def generate_legendary_buy_signals(self, df):
        """전설급 매수 신호 생성 - 다중 지표 컨센서스"""
        # 기본 매수 조건
        basic_conditions = (
            (df['final_score'] > df['final_score'].quantile(0.8)) &  # 상위 20%
            (df['macd_histogram'] > 0) &  # MACD 상승
            (df['adx'] > 20) &  # 추세 강도
            (df['rsi'] < 70) &  # 과매수 방지
            (df['close'] > df['bb_middle']) &  # 볼린저 중심선 상향
            (df['volume_spike'] == True)  # 거래량 급증
        )
        
        # 전설급 추가 조건들
        legendary_conditions = (
            # 일목균형표 강세 확인
            df['above_cloud'] & df['tk_bullish'] |
            
            # VWAP + 기관 매수 신호
            (df['close'] > df['vwap']) & df['institutional_buying'] |
            
            # 엘리어트 파동 + 피보나치 레벨
            (df['close'] > df['fib_618']) & (df['wave_strength'] > 0.1)
        )
        
        # 최종 매수 신호
        df['legendary_buy_signal'] = basic_conditions & legendary_conditions
        
        return df
    
    def generate_legendary_sell_signals(self, df):
        """전설급 매도 신호 생성 - 정교한 익절/손절"""
        # 진입가격이 없으면 현재가로 설정
        if 'entry_price' not in df.columns:
            df['entry_price'] = df['close']
        
        # 전설급 익절 조건
        legendary_take_profit = (
            # 기본 익절
            (df['close'] / df['entry_price'] > 1.25) |  # 25% 수익
            
            # 기술적 익절 신호
            (df['rsi'] > 80) & df['rsi_overbought'] |
            (df['close'] < df['bb_lower']) |
            
            # 엘리어트 5파 완성
            df['wave_5_complete'] & (df['rsi'] > 70)
        )
        
        # 전설급 손절 조건
        legendary_stop_loss = (
            # 기본 손절
            (df['close'] / df['entry_price'] < 0.90) |  # 10% 손실
            
            # 기술적 손절 신호
            (df['adx'] < 15) & (df['close'] < df['vwap']) |
            df['below_cloud'] & df['tk_bearish']
        )
        
        df['legendary_sell_signal'] = legendary_take_profit | legendary_stop_loss
        
        return df
    
    # ================== IBKR 자동매매 시스템 (기존) ==================
    
    def connect_ibkr(self):
        """IBKR 연결"""
        return self.ibkr.connect()
    
    def execute_auto_trading(self, selected_stocks, max_investment=1000000):
        """자동 거래 실행 - 2주 스윙 손익절 적용"""
        if not self.ibkr.connected:
            print("❌ IBKR 연결이 필요합니다")
            return
        
        print("\n🚀 자동 거래 시작 (2주 스윙 전략)...")
        
        # 매수 신호 종목들
        for _, stock in selected_stocks.head(5).iterrows():  # 상위 5개
            symbol = stock['ticker']
            price = stock['close']
            score = stock['final_score']
            
            # 투자금액 계산
            investment = min(max_investment / 5, 200000)  # 균등분할, 최대 20만
            quantity = int(investment / price)
            
            if quantity > 0 and score > 15:  # 최소 점수 조건
                success = self.ibkr.place_buy_order(symbol, quantity, price)
                if success:
                    # 2주 스윙 손익절가 설정 (브래킷 주문)
                    stop_loss_price = stock.get('stop_loss_price', price * 0.92)
                    take_profit_price = stock.get('take_profit_price', price * 1.18)
                    
                    print(f"✅ 매수 완료: {symbol} {quantity}주")
                    print(f"   💰 진입가: ₹{price:.2f}")
                    print(f"   🛑 손절가: ₹{stop_loss_price:.2f} ({stock.get('swing_stop_pct', 8):.1f}%)")
                    print(f"   🎯 익절가: ₹{take_profit_price:.2f} ({stock.get('swing_profit_pct', 18):.1f}%)")
                    time.sleep(1)
        
        # 기존 포지션 점검 (2주 스윙 기준)
        positions = self.track_current_positions()
        for pos in positions:
            symbol = pos['ticker']
            
            # 2주 만료 또는 손익절 조건 체크
            if pos['days_remaining'] <= 0:
                # 2주 만료 - 무조건 정리
                print(f"⏰ 2주 만료: {symbol} 포지션 정리")
                # self.ibkr.place_sell_order(symbol, quantity)
                
            elif pos['stop_distance'] < 5:
                # 손절선 임박
                print(f"🚨 손절 실행: {symbol} {pos['pnl_pct']:.1f}%")
                # self.ibkr.place_sell_order(symbol, quantity)
                
            elif pos['pnl_pct'] >= 15:
                # 익절 기회
                print(f"🎯 익절 실행: {symbol} +{pos['pnl_pct']:.1f}%")
                # self.ibkr.place_sell_order(symbol, quantity)
    
    def create_sample_data(self):
        """실제 테스트용 샘플 데이터 생성 - 4개 지수 통합"""
        print("📊 NSE 4대 지수 샘플 데이터 생성 중...")
        
        # 4개 지수별 종목들
        nifty_50 = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'ITC', 'SBIN', 'BHARTIARTL',
            'KOTAKBANK', 'LT', 'HCLTECH', 'AXISBANK', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI',
            'NESTLEIND', 'WIPRO', 'ULTRACEMCO', 'TITAN', 'SUNPHARMA'
        ]
        
        all_symbols = nifty_50[:10]  # 간단히 10개만
        sectors = ['IT', 'Banking', 'Pharma', 'Auto', 'FMCG']
        
        sample_data = []
        
        for i, symbol in enumerate(all_symbols):
            # 60일간 데이터 생성
            dates = pd.date_range(start='2024-11-01', periods=60, freq='D')
            
            # 가격 데이터 (트렌드 반영)
            base_price = np.random.uniform(1500, 3500)
            prices = []
            current_price = base_price
            
            for j in range(60):
                change = np.random.normal(0.002, 0.02)
                current_price *= (1 + change)
                prices.append(current_price)
            
            # DataFrame 생성
            df_sample = pd.DataFrame({
                'date': dates,
                'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
                'high': [p * np.random.uniform(1.00, 1.05) for p in prices],
                'low': [p * np.random.uniform(0.95, 1.00) for p in prices],
                'close': prices,
                'volume': [np.random.randint(500000, 5000000) for _ in range(60)],
            })
            
            # 기업 기본 정보
            df_sample['ticker'] = symbol
            df_sample['company_name'] = f"{symbol} Limited"
            df_sample['Sector'] = np.random.choice(sectors)
            df_sample['index_category'] = 'NIFTY50'  # 추가된 필드
            
            # 펀더멘털 데이터
            df_sample['ROE'] = np.random.uniform(15, 35)
            df_sample['ROCE'] = np.random.uniform(18, 30)
            df_sample['Debt_to_Equity'] = np.random.uniform(0.1, 1.0)
            df_sample['Promoter_Holding'] = np.random.uniform(40, 75)
            df_sample['Promoter_Pledge'] = np.random.uniform(0, 15)
            df_sample['Operating_Profit'] = np.random.uniform(5000, 50000)
            df_sample['Dividend_Yield'] = np.random.uniform(1, 5)
            df_sample['EPS_growth'] = np.random.uniform(10, 50)
            df_sample['Current_Ratio'] = np.random.uniform(1, 3)
            df_sample['Market_Cap'] = np.random.uniform(50000, 500000)
            df_sample['Analyst_Coverage'] = np.random.randint(1, 10)
            df_sample['PBV'] = np.random.uniform(0.5, 5)
            
            sample_data.append(df_sample)
        
        # 전체 데이터 합치기
        full_df = pd.concat(sample_data, ignore_index=True)
        print(f"✅ {len(all_symbols)}개 종목, {len(full_df)}개 데이터 포인트 생성 완료")
        
        return full_df
    
    # ================== 수요일 전용 월 5~7% 안정형 1주 스윙 시스템 ==================
    
    def calculate_conservative_weekly_stops(self, df):
        """수요일 전용 월 5~7% 안정형 1주 스윙 시스템"""
        
        # 안정형 손절비 (타이트하게)
        stop_loss_pct = {
            'NIFTY50': 0.03,   # -3% (대형주 안정)
            'SENSEX': 0.03,    # -3% (블루칩)
            'NEXT50': 0.04,    # -4% (중형주)
            'SMALLCAP': 0.05   # -5% (소형주, 비중 제한)
        }
        
        # 안정형 익절비 (욕심부리지 않고)
        take_profit_pct = {
            'NIFTY50': 0.06,   # +6% (안정 수익)
            'SENSEX': 0.06,    # +6% 
            'NEXT50': 0.08,    # +8% (성장주)
            'SMALLCAP': 0.10   # +10% (고위험고수익)
        }
        
        # 각 종목별 손익절가 계산
        df['conservative_stop_loss'] = 0
        df['conservative_take_profit'] = 0
        df['weekly_stop_pct'] = 0
        df['weekly_profit_pct'] = 0
        df['target_weekly_return'] = 0
        
        for idx, row in df.iterrows():
            index_cat = row.get('index_category', 'OTHER')
            current_price = row.get('close', row.get('Price', 0))
            final_score = row.get('final_score', 0)
            
            # 지수별 손익절비 설정
            if 'NIFTY50' in str(index_cat):
                stop_pct = stop_loss_pct['NIFTY50']
                profit_pct = take_profit_pct['NIFTY50']
            elif 'SENSEX' in str(index_cat):
                stop_pct = stop_loss_pct['SENSEX']
                profit_pct = take_profit_pct['SENSEX']
            elif 'NEXT50' in str(index_cat):
                stop_pct = stop_loss_pct['NEXT50']
                profit_pct = take_profit_pct['NEXT50']
            elif 'SMALLCAP' in str(index_cat):
                stop_pct = stop_loss_pct['SMALLCAP']
                profit_pct = take_profit_pct['SMALLCAP']
            else:
                stop_pct = 0.04  # 기본값
                profit_pct = 0.07
            
            # 신호 강도에 따른 미세 조정 (보수적으로)
            if final_score > 25:  # 매우 강한 신호
                profit_pct *= 1.2  # 익절만 약간 높이기
            elif final_score > 20:  # 강한 신호
                profit_pct *= 1.1
            elif final_score < 15:  # 약한 신호 - 진입 자체를 제한
                stop_pct = 0.02  # 매우 타이트한 손절
                profit_pct = 0.04
            
            # 손익절가 계산
            if current_price > 0:
                df.loc[idx, 'conservative_stop_loss'] = current_price * (1 - stop_pct)
                df.loc[idx, 'conservative_take_profit'] = current_price * (1 + profit_pct)
                df.loc[idx, 'weekly_stop_pct'] = stop_pct * 100
                df.loc[idx, 'weekly_profit_pct'] = profit_pct * 100
                df.loc[idx, 'target_weekly_return'] = profit_pct * 100
        
        return df
    
    def wednesday_only_filter(self):
        """수요일만 거래 허용 체크"""
        from datetime import datetime
        
        today = datetime.now()
        is_wednesday = today.weekday() == 2  # 수요일 = 2
        
        return {
            'is_wednesday': is_wednesday,
            'current_day': today.strftime('%A'),
            'next_wednesday': 'Next Wednesday' if not is_wednesday else 'Today!'
        }
    
    def conservative_stock_selection(self, df, max_stocks=4):
        """안정형 종목 선별 (엄격한 기준)"""
        
        # 안정성 우선 필터링
        stability_filter = (
            (df['final_score'] >= 20) &  # 고점수만 (엄격)
            (df['Market_Cap'] > 50000) &  # 대중형주 위주
            (df['adx'] > 25) &  # 강한 추세
            (df['above_cloud'] == True) &  # 일목균형표 구름 위
            (df['rsi'] < 65) &  # 과매수 방지
            (df['close'] > df['vwap']) &  # VWAP 상향
            (df['volume_spike'] == True) &  # 거래량 확인
            (df.get('Debt_to_Equity', 1) < 1.0)  # 재무 안정성
        )
        
        # 필터링된 데이터
        filtered_df = df[stability_filter].copy()
        
        if len(filtered_df) == 0:
            print("❌ 안정성 기준을 만족하는 종목이 없습니다")
            return pd.DataFrame()
        
        # 지수별 분산 (대형주 70%, 중소형주 30%)
        selected_stocks = []
        
        # 1. NIFTY50/SENSEX 우선 선택 (2~3개)
        large_cap = filtered_df[
            (filtered_df['index_category'].str.contains('NIFTY50|SENSEX', na=False))
        ].nlargest(3, 'final_score')
        
        # 2. NEXT50 중에서 1개 (성장주)
        if 'NEXT50' in str(filtered_df['index_category'].values):
            mid_cap = filtered_df[
                (filtered_df['index_category'].str.contains('NEXT50', na=False))
            ].nlargest(1, 'final_score')
        else:
            mid_cap = pd.DataFrame()
        
        # 3. 최종 조합
        if not mid_cap.empty:
            selected_stocks = pd.concat([large_cap, mid_cap], ignore_index=True)
        else:
            selected_stocks = large_cap
        selected_stocks = selected_stocks.head(max_stocks)
        
        # 추가 안전장치: 섹터 분산 체크
        if not selected_stocks.empty:
            sector_counts = selected_stocks['Sector'].value_counts()
            if sector_counts.max() > 2:  # 한 섹터에 2개 초과 금지
                print("⚠️ 섹터 집중도 경고 - 분산 조정")
        
        return selected_stocks
    
    def calculate_position_sizing_conservative(self, selected_stocks, total_capital=10000000):
        """안정형 포지션 사이징 (리스크 제한)"""
        
        n_stocks = len(selected_stocks)
        if n_stocks == 0:
            return {}
        
        # 보수적 자금 배분
        max_investment_per_stock = total_capital * 0.20  # 종목당 최대 20%
        risk_budget_per_trade = total_capital * 0.02  # 거래당 리스크 2%
        
        portfolio = {}
        
        for _, stock in selected_stocks.iterrows():
            ticker = stock['ticker']
            price = stock['close']
            score = stock['final_score']
            stop_loss_pct = stock.get('weekly_stop_pct', 4) / 100
            
            # 점수 기반 가중치 (보수적)
            score_weight = min(score / 30, 1.0)  # 최대 1.0
            base_allocation = total_capital / n_stocks
            
            # 리스크 기반 포지션 사이징
            risk_per_share = price * stop_loss_pct
            max_shares_by_risk = int(risk_budget_per_trade / risk_per_share) if risk_per_share > 0 else 0
            
            # 실제 배분
            allocation = min(base_allocation * score_weight, max_investment_per_stock)
            shares_by_capital = int(allocation / price) if price > 0 else 0
            
            # 최종 주식 수 (리스크 제한 적용)
            final_shares = min(shares_by_capital, max_shares_by_risk)
            final_allocation = final_shares * price
            
            portfolio[ticker] = {
                'allocation': final_allocation,
                'shares': final_shares,
                'score': score,
                'entry_price': price,
                'stop_loss': stock.get('conservative_stop_loss', price * 0.96),
                'take_profit': stock.get('conservative_take_profit', price * 1.06),
                'weekly_target': stock.get('target_weekly_return', 6),
                'risk_amount': final_shares * risk_per_share,
                'weight_pct': (final_allocation / total_capital) * 100 if total_capital > 0 else 0
            }
        
        return portfolio
    
    def weekly_position_tracker(self):
        """주간 포지션 추적 (수요일 기준)"""
        from datetime import datetime, timedelta
        
        # 샘플 포지션 (실제로는 DB에서 로드)
        positions = {
            'RELIANCE': {
                'entry_date': '2024-12-25',  # 지난 수요일
                'entry_price': 2450,
                'current_price': 2520,  # +2.9%
                'stop_loss': 2377,  # -3%
                'take_profit': 2597,  # +6%
                'shares': 40,
                'target_return': 6
            },
            'TCS': {
                'entry_date': '2024-12-25',
                'entry_price': 3200,
                'current_price': 3168,  # -1.0%
                'stop_loss': 3104,  # -3%
                'take_profit': 3392,  # +6%
                'shares': 30,
                'target_return': 6
            },
            'HDFCBANK': {
                'entry_date': '2024-12-25',
                'entry_price': 1650,
                'current_price': 1683,  # +2.0%
                'stop_loss': 1601,  # -3%
                'take_profit': 1749,  # +6%
                'shares': 60,
                'target_return': 6
            }
        }
        
        position_status = []
        today = datetime.now()
        
        for ticker, pos in positions.items():
            entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
            days_held = (today - entry_date).days
            days_until_wednesday = (9 - today.weekday()) % 7 if today.weekday() != 2 else 0
            
            # 손익률 계산
            pnl_pct = ((pos['current_price'] - pos['entry_price']) / pos['entry_price']) * 100
            pnl_amount = (pos['current_price'] - pos['entry_price']) * pos['shares']
            
            # 목표 달성률
            target_achievement = (pnl_pct / pos['target_return']) * 100
            
            # 상태 결정
            if pnl_pct >= pos['target_return']:
                status = "🎯 목표달성"
            elif pnl_pct >= pos['target_return'] * 0.7:
                status = "🟢 순조진행"
            elif pnl_pct >= 0:
                status = "🟡 관찰필요"
            elif pnl_pct >= -2:
                status = "🟠 주의경고"
            else:
                status = "🔴 손절위험"
            
            position_status.append({
                'ticker': ticker,
                'days_held': days_held,
                'days_until_exit': days_until_wednesday,
                'pnl_pct': pnl_pct,
                'pnl_amount': pnl_amount,
                'target_achievement': target_achievement,
                'status': status,
                'current_price': pos['current_price'],
                'entry_price': pos['entry_price'],
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit']
            })
        
        return position_status
    
    def conservative_alerts(self):
        """안정형 알림 시스템 (월 5~7% 목표)"""
        alerts = []
        
        # 수요일 체크
        wednesday_status = self.wednesday_only_filter()
        if wednesday_status['is_wednesday']:
            alerts.append("📅 오늘은 수요일 - 새로운 진입 검토 가능!")
        else:
            alerts.append(f"📅 오늘은 {wednesday_status['current_day']} - 포지션 관리만")
        
        # 포지션 상태 체크
        positions = self.weekly_position_tracker()
        
        total_pnl = sum([pos['pnl_amount'] for pos in positions])
        avg_performance = sum([pos['pnl_pct'] for pos in positions]) / len(positions) if positions else 0
        
        for pos in positions:
            ticker = pos['ticker']
            
            # 1. 목표 달성 (6%+)
            if pos['pnl_pct'] >= 6:
                alerts.append(f"🎯 {ticker} 목표 달성! 수익: +{pos['pnl_pct']:.1f}% (₹{pos['pnl_amount']:,.0f})")
            
            # 2. 손절 임박 (-2.5%+)
            elif pos['pnl_pct'] <= -2.5:
                alerts.append(f"🚨 {ticker} 손절 임박! 손실: {pos['pnl_pct']:.1f}% - 즉시 검토 필요")
            
            # 3. 순조진행 (3%+)
            elif pos['pnl_pct'] >= 3:
                alerts.append(f"🟢 {ticker} 순조진행: +{pos['pnl_pct']:.1f}% (목표 {pos['target_achievement']:.0f}%)")
        
        # 4. 전체 포트폴리오 상태
        if avg_performance >= 4:
            alerts.append(f"🏆 포트폴리오 우수! 평균 수익률: +{avg_performance:.1f}%")
        elif avg_performance >= 1:
            alerts.append(f"📊 포트폴리오 양호: 평균 수익률: +{avg_performance:.1f}%")
        else:
            alerts.append(f"⚠️ 포트폴리오 점검 필요: 평균 {avg_performance:.1f}%")
        
        # 5. 월간 목표 진행률
        monthly_target = 6  # 월 6% 목표
        weekly_progress = avg_performance
        monthly_projection = weekly_progress * 4
        
        if monthly_projection >= monthly_target:
            alerts.append(f"🎊 월간 목표 달성 가능! 예상: {monthly_projection:.1f}% (목표: {monthly_target}%)")
        else:
            alerts.append(f"📈 월간 목표까지: {monthly_target - monthly_projection:.1f}%p 더 필요")
        
        return alerts
    
    def execute_conservative_trading(self, selected_stocks, max_investment=2000000):
        """안정형 자동 거래 실행 (월 5~7% 목표)"""
        if not self.ibkr.connected:
            print("❌ IBKR 연결이 필요합니다")
            return
        
        print("\n🎯 안정형 자동 거래 시작 (주간 1~2% 목표)...")
        
        # 엄격한 진입 조건 재확인
        for _, stock in selected_stocks.iterrows():
            symbol = stock['ticker']
            price = stock['close']
            score = stock['final_score']
            
            # 재확인: 점수 20점 이상만
            if score < 20:
                print(f"⚠️ {symbol} 점수 부족 ({score:.1f}) - 패스")
                continue
            
            # 포지션 사이즈 계산 (보수적)
            investment = min(max_investment / len(selected_stocks), 500000)  # 최대 50만
            quantity = int(investment / price)
            
            if quantity > 0:
                success = self.ibkr.place_buy_order(symbol, quantity, price)
                if success:
                    stop_loss = stock.get('conservative_stop_loss', price * 0.97)
                    take_profit = stock.get('conservative_take_profit', price * 1.06)
                    
                    print(f"✅ 안정형 매수: {symbol} {quantity}주")
                    print(f"   💰 진입가: ₹{price:.2f}")
                    print(f"   🛑 손절가: ₹{stop_loss:.2f} (-{stock.get('weekly_stop_pct', 3):.1f}%)")
                    print(f"   🎯 익절가: ₹{take_profit:.2f} (+{stock.get('weekly_profit_pct', 6):.1f}%)")
                    print(f"   📊 신뢰도: {score:.1f}/30점")
                    time.sleep(1)
        
        print("📊 다음 수요일까지 포지션 유지 예정")
    
    def run_conservative_strategy(self, df, enable_trading=False):
        """안정형 월 5~7% 수요일 전용 전략 실행"""
        print("🎯 월 5~7% 안정형 수요일 전용 인도 투자전략 실행 중...")
        
        # 수요일 체크
        wednesday_status = self.wednesday_only_filter()
        print(f"📅 오늘: {wednesday_status['current_day']} | 거래가능: {wednesday_status['is_wednesday']}")
        
        # 1. 기술지표 계산
        df = self.calculate_all_indicators(df)
        print("✅ 전설급 기술지표 계산 완료")
        
        # 2. 전설 전략 적용
        df = self.apply_all_strategies(df)
        print("✅ 5대 전설 전략 적용 완료")
        
        # 3. 통합 점수 생성
        df = self.generate_master_score(df)
        print("✅ 마스터 점수 생성 완료")
        
        # 4. 지수별 맞춤 전략 적용
        df = self.apply_index_specific_strategy(df)
        print("✅ 지수별 맞춤 전략 적용 완료")
        
        # 5. 안정형 1주 스윙 손익절 계산
        df = self.calculate_conservative_weekly_stops(df)
        print("✅ 안정형 1주 스윙 손익절 시스템 적용 완료")
        
        # 6. AI 강화 종목 선별 (새로 추가!)
        if self.ai_analyzer.connected:
            selected_stocks = self.ai_enhanced_stock_selection(df, max_stocks=4)
            print("✅ AI 강화 종목 선별 완료")
        else:
            selected_stocks = self.conservative_stock_selection(df, max_stocks=4)
            print("✅ 안정형 종목 선별 완료 (AI 없이)")
        
        # 7. 수요일 IBKR 자동매매
        if enable_trading and wednesday_status['is_wednesday']:
            print("\n💰 수요일 안정형 자동매매 시작...")
            if self.connect_ibkr():
                self.execute_conservative_trading(selected_stocks)
                print("✅ 안정형 자동매매 완료")
            else:
                print("❌ IBKR 연결 실패 - 분석만 진행")
        elif enable_trading and not wednesday_status['is_wednesday']:
            print(f"📅 오늘은 {wednesday_status['current_day']} - 거래 없음 (수요일만 거래)")
        
        # 8. 안정형 포트폴리오 구성
        portfolio = self.calculate_position_sizing_conservative(selected_stocks)
        print("✅ 안정형 포트폴리오 구성 완료")
        
        # 9. AI 강화 리스크 평가 (새로 추가!)
        risk_metrics = self.ai_risk_assessment(portfolio, selected_stocks)
        print("✅ AI 강화 리스크 평가 완료")
        
        # 10. 주간 포지션 추적
        position_status = self.weekly_position_tracker()
        print("✅ 주간 포지션 추적 완료")
        
        # 11. 안정형 알림 생성
        alerts = self.conservative_alerts()
        print("✅ 안정형 알림 시스템 완료")
        
        # 12. AI 인사이트 생성 (새로 추가!)
        strategy_results = {
            'selected_stocks': selected_stocks,
            'portfolio': portfolio,
            'risk_metrics': risk_metrics,
            'position_status': position_status,
            'alerts': alerts,
            'wednesday_status': wednesday_status,
            'ibkr_connected': getattr(self.ibkr, 'connected', False)
        }
        
        ai_insights = self.generate_ai_insights(strategy_results)
        ai_daily_report = self.create_ai_daily_report(strategy_results)
        print("✅ AI 인사이트 및 일일 리포트 생성 완료")
        
        return {
            **strategy_results,
            'conservative_data': df[['ticker', 'close', 'weekly_stop_pct', 'weekly_profit_pct', 
                                   'conservative_stop_loss', 'conservative_take_profit']].head(10) if all(col in df.columns for col in ['ticker', 'close', 'weekly_stop_pct', 'weekly_profit_pct', 'conservative_stop_loss', 'conservative_take_profit']) else pd.DataFrame(),
            'ai_insights': ai_insights,
            'ai_daily_report': ai_daily_report,
            'ai_connected': self.ai_analyzer.connected
        }
    
    # ================== 메인 실행 함수 (OpenAI 추가) ==================

    async def run_strategy(self, df=None, enable_trading=False):
        """전체 전략 실행 - AI 강화 안정형 월 5~7% 시스템"""
        if df is None:
            df = self.create_sample_data()
        return self.run_conservative_strategy(df, enable_trading)
    
    # ================== 포트폴리오 관리 (기존) ==================
    
    def portfolio_management(self, selected_stocks, total_capital=10000000):
        """포트폴리오 관리 (1천만원 기준)"""
        n_stocks = len(selected_stocks)
        if n_stocks == 0:
            return {}
        
        # 균등 분할 + 점수 가중치
        base_allocation = total_capital / n_stocks
        
        portfolio = {}
        for _, stock in selected_stocks.iterrows():
            weight = stock['final_score'] / selected_stocks['final_score'].sum()
            allocation = base_allocation * (0.7 + 0.6 * weight)  # 70% 균등 + 30% 가중
            
            portfolio[stock['ticker']] = {
                'allocation': allocation,
                'shares': int(allocation / stock['close']),
                'score': stock['final_score'],
                'entry_price': stock['close']
            }
        
        return portfolio
    
    def risk_management(self, df):
        """리스크 관리"""
        # 간단한 리스크 메트릭
        risk_metrics = {
            'portfolio_beta': 1.2,
            'max_sector_concentration': 0.3,
            'diversification_score': 0.7,
            'avg_volatility': 0.25
        }
        
        return risk_metrics

# ================== 실제 실행 및 데모 (OpenAI 추가) ==================

async def main():
    """메인 실행 함수"""
    print("🇮🇳 인도 전설 투자전략 + IBKR 자동매매 + OpenAI 분석 시스템")
    print("=" * 80)
    print("⚡ 추가된 기능들:")
    print("🔥 실시간 자동매매 | 💰 스마트 손익절 | 📊 포지션 관리")
    print("🤖 OpenAI 시장분석 | 🧠 AI 투자인사이트 | 📈 AI 리스크평가")
    print("=" * 80)
    
    # OpenAI API 키 설정 안내
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key and OPENAI_AVAILABLE:
        print("\n💡 OpenAI 기능을 사용하려면 API 키를 설정하세요:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   또는 코드에서 직접 설정 가능")
        
        user_key = input("\nOpenAI API 키를 입력하세요 (Enter로 건너뛰기): ").strip()
        if user_key:
            openai_key = user_key
    
    # 전략 시스템 초기화 (OpenAI 키 포함)
    strategy = LegendaryIndiaStrategy(openai_api_key=openai_key)
    
    # 실행 모드 선택
    print("\n실행 모드를 선택하세요:")
    print("1. 📊 백테스팅만 (IBKR 없이)")
    print("2. 🚀 실제 거래 (IBKR 연동)")
    print("3. 📈 포지션 확인 (IBKR)")
    print("4. 🤖 AI 분석 테스트")
    
    choice = input("\n번호 입력 (기본값: 1): ").strip() or "1"
    
    if choice == "1":
        # 백테스팅 모드
        print("\n🔬 백테스팅 모드 시작...")
        sample_df = strategy.create_sample_data()
        results = await strategy.run_strategy(sample_df, enable_trading=False)
        
    elif choice == "2":
        # 실제 거래 모드
        print("\n🚀 실제 거래 모드 시작...")
        print("⚠️ 실제 자금이 사용됩니다. 신중하게 진행하세요!")
        confirm = input("계속하시겠습니까? (yes/no): ")
        
        if confirm.lower() == 'yes':
            sample_df = strategy.create_sample_data()
            results = await strategy.run_strategy(sample_df, enable_trading=True)
        else:
            print("❌ 취소되었습니다")
            return
            
    elif choice == "3":
        # 포지션 확인 모드
        print("\n📈 포지션 확인 모드...")
        if strategy.connect_ibkr():
            positions = strategy.ibkr.get_positions()
            print("\n현재 포지션:")
            for symbol, pos in positions.items():
                print(f"📊 {symbol}: {pos['quantity']}주 @₹{pos['avg_cost']}")
        else:
            print("❌ IBKR 연결 실패")
        return
        
    elif choice == "4":
        # AI 분석 테스트 모드
        print("\n🤖 AI 분석 테스트 모드...")
        if not strategy.ai_analyzer.connected:
            print("❌ OpenAI 연결이 필요합니다")
            return
            
        # 샘플 데이터로 AI 분석 테스트
        sample_df = strategy.create_sample_data()
        
        # 기본 분석 실행
        df = strategy.calculate_all_indicators(sample_df)
        df = strategy.apply_all_strategies(df)
        df = strategy.generate_master_score(df)
        
        # AI 분석 테스트
        print("\n🧠 AI 시장 심리 분석 테스트...")
        sentiment = strategy.ai_analyzer.analyze_market_sentiment(df)
        print(f"결과: {sentiment}")
        
        # AI 종목 선별 테스트
        print("\n🎯 AI 종목 선별 테스트...")
        ai_stocks = strategy.ai_enhanced_stock_selection(df, 3)
        print(f"선별된 종목: {len(ai_stocks)}개")
        
        return
    
    else:
        print("❌ 잘못된 선택 - 백테스팅 모드로 진행")
        sample_df = strategy.create_sample_data()
        results = await strategy.run_strategy(sample_df, enable_trading=False)
    
    # 결과 상세 출력 - AI 강화 안정형 월 5~7% 버전
    print("\n🎯 === AI 강화 안정형 월 5~7% 인도 투자전략 결과 ===")
    print("="*90)
    
    # AI 연결 상태
    if results.get('ai_connected'):
        print("🤖 OpenAI 분석: 활성화 ✅")
    else:
        print("🤖 OpenAI 분석: 비활성화 ❌")
    
    # 수요일 거래 가능 여부
    wednesday_status = results.get('wednesday_status', {})
    if wednesday_status.get('is_wednesday'):
        print("📅 오늘은 수요일 - 새로운 포지션 진입 가능! 🟢")
    else:
        day = wednesday_status.get('current_day', '알수없음')
        print(f"📅 오늘은 {day} - 포지션 관리만 (다음 수요일까지 대기) 🟡")
    
    selected = results['selected_stocks']
    if not selected.empty:
        print(f"\n📊 AI 강화 엄격한 기준으로 {len(selected)}개 안정형 종목 선별!")
        print("-" * 90)
        
        for idx, (_, stock) in enumerate(selected.iterrows(), 1):
            print(f"🥇 #{idx:2d} | {stock['ticker']:12} | {stock.get('company_name', 'N/A')[:20]:20}")
            print(f"    💰 주가: ₹{stock['close']:8.2f} | 🎯 점수: {stock.get('final_score', 0):6.1f}/30")
            
            # AI 추가 정보
            if 'ai_enhanced_score' in stock:
                ai_score = stock.get('ai_enhanced_score', 0)
                ai_rec = stock.get('ai_recommendation', 'N/A')
                ai_conf = stock.get('ai_confidence', 0)
                print(f"    🤖 AI점수: {ai_score:6.1f} | 추천: {ai_rec} | 신뢰도: {ai_conf:.1f}")
            
            # 안정형 손익절 정보
            stop_pct = stock.get('weekly_stop_pct', 3)
            profit_pct = stock.get('weekly_profit_pct', 6)
            stop_price = stock.get('conservative_stop_loss', 0)
            profit_price = stock.get('conservative_take_profit', 0)
            
            print(f"    🛑 손절: ₹{stop_price:7.2f} (-{stop_pct:3.1f}%) | 🎯 익절: ₹{profit_price:7.2f} (+{profit_pct:3.1f}%)")
            print("-" * 90)
    else:
        print("❌ 안정성 기준을 만족하는 종목이 없습니다")
        print("   (점수 20+ & 대형주 & 구름위 & 과매수아님 & AI 검증 등)")
    
    # AI 인사이트 출력
    if results.get('ai_connected') and results.get('ai_insights'):
        print("\n🧠 === AI 투자 인사이트 ===")
        print("="*90)
        
        insights = results.get('ai_insights', [])
        for i, insight in enumerate(insights[:5], 1):  # 최대 5개
            print(f"{i}. {insight}")
    
    # 주간 포지션 현황
    print("\n📊 === 현재 주간 포지션 현황 ===")
    print("="*90)
    
    positions = results.get('position_status', [])
    if positions:
        total_pnl = sum([pos['pnl_amount'] for pos in positions])
        avg_performance = sum([pos['pnl_pct'] for pos in positions]) / len(positions)
        
        print(f"📈 총 포지션: {len(positions)}개 | 평균 수익률: {avg_performance:+5.1f}% | 총 손익: ₹{total_pnl:,.0f}")
        print("-" * 90)
        
        for pos in positions:
            status_icon = pos['status']
            ticker = pos['ticker']
            pnl_pct = pos['pnl_pct']
            pnl_amount = pos['pnl_amount']
            days_held = pos['days_held']
            target_achieve = pos['target_achievement']
            
            print(f"{status_icon} {ticker:12} | {days_held}일차 | {pnl_pct:+5.1f}% | ₹{pnl_amount:+8,.0f}")
            print(f"    📊 목표달성률: {target_achieve:5.1f}% | 진입: ₹{pos['entry_price']:,.0f} → 현재: ₹{pos['current_price']:,.0f}")
            print("-" * 90)
        
        # 월간 수익률 예상
        monthly_projection = avg_performance * 4  # 주 4회
        print(f"📈 월간 예상 수익률: {monthly_projection:+5.1f}% (목표: 5~7%)")
        
        if monthly_projection >= 5:
            print("🎊 월간 목표 달성 가능! 훌륭합니다! 🎯")
        else:
            needed = 5 - monthly_projection
            print(f"📊 목표까지 {needed:+4.1f}%p 더 필요")
    else:
        print("📭 현재 보유 포지션 없음")
    
    # AI 강화 포트폴리오 구성
    print("\n💼 === AI 강화 안정형 포트폴리오 구성 ===")
    print("="*90)
    
    portfolio = results['portfolio']
    if portfolio:
        print("💎 AI 검증 + 리스크 제한 투자 배분:")
        print("-" * 90)
        
        total_investment = 0
        total_risk = 0
        
        for ticker, details in portfolio.items():
            investment = details['allocation']
            shares = details['shares']
            score = details['score']
            price = details['entry_price']
            weight = details['weight_pct']
            risk_amount = details['risk_amount']
            weekly_target = details['weekly_target']
            
            print(f"📈 {ticker:12} | ₹{investment:8,.0f} ({weight:4.1f}%) | {shares:4,}주 | 목표: +{weekly_target:2.0f}%")
            print(f"    💰 진입가: ₹{price:7.2f} | 🛡️ 리스크: ₹{risk_amount:6,.0f} | 점수: {score:4.1f}")
            
            total_investment += investment
            total_risk += risk_amount
        
        print("-" * 90)
        print(f"💰 총 투자금액: ₹{total_investment:10,.0f}")
        print(f"🛡️ 총 리스크:   ₹{total_risk:10,.0f} ({(total_risk/total_investment)*100:4.1f}%)")
        print(f"💵 현금 보유:   ₹{10000000 - total_investment:10,.0f}")
    
    # AI 강화 리스크 평가
    print("\n🛡️ === AI 강화 리스크 평가 ===")
    print("="*90)
    
    risk_metrics = results.get('risk_metrics', {})
    if risk_metrics:
        ai_risk_level = risk_metrics.get('ai_risk_level', 'N/A')
        overall_risk = risk_metrics.get('overall_risk_score', 'N/A')
        ai_warnings = risk_metrics.get('ai_warnings', [])
        
        print(f"🤖 AI 리스크 레벨: {ai_risk_level}")
        print(f"📊 종합 리스크 점수: {overall_risk}")
        print(f"⚠️ AI 경고사항: {len(ai_warnings)}개")
        
        if ai_warnings:
            for warning in ai_warnings[:3]:  # 최대 3개
                print(f"   • {warning}")
    
    # 핵심 알림
    print("\n🚨 === 핵심 알림 ===")
    print("="*80)
    
    alerts = results.get('alerts', [])
    for alert in alerts:
        print(f"• {alert}")
    
    # 연결 상태 종합
    print("\n🔗 === 시스템 연결 상태 ===")
    print("="*80)
    
    if results.get('ibkr_connected'):
        print("✅ IBKR 연결 성공 - 자동매매 활성화")
        if wednesday_status.get('is_wednesday'):
            print("💰 수요일 자동매매 실행됨")
        else:
            print("📅 수요일이 아니므로 거래 대기 중")
    else:
        print("❌ IBKR 연결 없음 - 분석만 진행")
        print("🔧 실제 거래를 원하면 IBKR API 설정 필요")
    
    if results.get('ai_connected'):
        print("✅ OpenAI 연결 성공 - AI 분석 활성화")
        print("🧠 시장심리, 인사이트, 리스크평가 AI 지원")
    else:
        print("❌ OpenAI 연결 없음 - 기본 분석만")
        print("🔧 AI 기능을 원하면 OpenAI API 키 설정 필요")
    
    # AI 일일 리포트
    if results.get('ai_connected') and results.get('ai_daily_report'):
        print("\n📋 === AI 일일 투자 리포트 ===")
        print("="*90)
        
        daily_report = results.get('ai_daily_report', '')
        # 리포트를 적절한 길이로 나누어 출력
        report_lines = daily_report.split('\n')
        for line in report_lines[:15]:  # 최대 15줄
            if line.strip():
                print(line)
    
    # 안정형 전략 가이드
    print("\n🎯 === AI 강화 안정형 월 5~7% 투자 가이드 ===")
    print("="*90)
    print("📅 수요일만 거래: 매주 수요일에만 신규 진입")
    print("🎯 목표 수익률: 주간 1~2% → 월간 5~7%")
    print("🛡️ 리스크 관리: 종목당 최대 -3~5% 손절")
    print("📊 엄격한 선별: 점수 20+ & 대형주 위주")
    print("🤖 AI 검증: OpenAI 기반 시장분석 및 타이밍 최적화")
    print("⚖️ 분산 투자: 최대 4종목, 섹터별 분산")
    print("💰 포지션 크기: 총 자산의 20% 이하/종목")
    print("📈 승률 목표: 80%+ (안정성 우선)")
    print("🧠 AI 지원: 실시간 시장심리, 리스크평가, 투자인사이트")
    
    print("\n🇮🇳 AI 강화 안정형 월 5~7% 인도 투자전략 완료! 🎯")
    print("💎 AI와 함께하는 스마트한 투자의 새로운 경험! 🤖💰")
    print("="*90)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
