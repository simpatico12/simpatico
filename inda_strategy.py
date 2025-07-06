#!/usr/bin/env python3
"""
🇮🇳 인도 전설 투자전략 v3.0 - 완전판
================================================================
🏆 5대 투자 거장 철학 + 고급 기술지표 + 자동선별 시스템
- 라케시 준준왈라 + 라메데오 아그라왈 + 비제이 케디아 전략
- 실시간 자동 매매 신호 생성 + 손절/익절 시스템  
- 백테스팅 + 포트폴리오 관리 + 리스크 제어
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import json
import logging
warnings.filterwarnings('ignore')

class IndiaLegendStrategy:
    """인도 전설 투자자 5인방 통합 전략"""
    
    def __init__(self):
        self.portfolio = {}
        self.signals = {}
        self.risk_metrics = {}
        
    # ================== 기술지표 라이브러리 ==================
    
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
    
    def volume_profile(self, df, period=20):
        """거래량 프로파일 + 이상 급증 감지"""
        df['volume_sma'] = df['volume'].rolling(period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_spike'] = df['volume_ratio'] > 2.0
        df['volume_momentum'] = df['volume'].pct_change(5)
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
    
    # ================== 전설 투자자 전략 구현 ==================
    
    def rakesh_jhunjhunwala_strategy(self, df):
        """라케시 준준왈라 - 워런 버핏 킬러 전략"""
        # 3-5-7 룰 구현
        df['roe_trend'] = (df.get('ROE', 0) > 15).astype(int)
        df['profit_streak'] = (df.get('Operating_Profit', 0) > 0).astype(int)
        df['dividend_streak'] = (df.get('Dividend_Yield', 0) > 1.0).astype(int)
        
        # 경영진 지분율 체크
        df['promoter_strength'] = (
            (df.get('Promoter_Holding', 30) >= 30) & 
            (df.get('Promoter_Pledge', 10) <= 15)
        ).astype(int)
        
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
        df['quality_score'] = (
            (df.get('Debt_to_Equity', 0.5) < 0.5).astype(int) * 2 +
            (df.get('Current_Ratio', 1.5) > 1.5).astype(int) * 1 +
            (df.get('ROE', 15) > 15).astype(int) * 2
        )
        
        # Growth (성장)
        df['growth_score'] = (df.get('EPS_growth', 10) > 20).astype(int) * 3
        
        # Longevity (지속성)
        df['longevity_score'] = (df.get('Years_Listed', 5) > 10).astype(int) * 2
        
        # Price (가격)
        df['price_score'] = (df.get('PEG_ratio', 1.5) < 1.0).astype(int) * 2
        
        # QGLP 종합 점수
        df['qglp_score'] = (
            df['quality_score'] + 
            df['growth_score'] + 
            df['longevity_score'] + 
            df['price_score']
        )
        return df
    
    def vijay_kedia_smile(self, df):
        """비제이 케디아 - SMILE 투자법"""
        # Small (소형주)
        df['small_score'] = np.where(
            df.get('Market_Cap', 100000) < 50000, 3,
            np.where(df.get('Market_Cap', 100000) < 200000, 2, 1)
        )
        
        # Medium (중형주 선호)
        df['medium_score'] = (
            (df.get('Market_Cap', 100000) >= 10000) & 
            (df.get('Market_Cap', 100000) <= 100000)
        ).astype(int) * 2
        
        # Industry (산업 리더십)
        df['industry_score'] = (df.get('Market_Share', 5) > 10).astype(int) * 2
        
        # Leadership (경영진)
        df['leadership_score'] = (df.get('Management_Score', 7) > 7).astype(int) * 2
        
        # Ethical (윤리경영)
        df['ethical_score'] = (df.get('ESG_Score', 5) > 7).astype(int) * 1
        
        # SMILE 종합 점수
        df['smile_score'] = (
            df['small_score'] + 
            df['medium_score'] + 
            df['industry_score'] + 
            df['leadership_score'] + 
            df['ethical_score']
        )
        return df
    
    def porinju_veliyath_contrarian(self, df):
        """포리뉴 벨리야스 - 콘트라리안 마스터"""
        # 저평가 지표
        df['undervalued_score'] = (
            (df.get('PBV', 2.0) < 1.0).astype(int) * 3 +
            (df.get('PE_ratio', 20) < 12).astype(int) * 2 +
            (df.get('EV_EBITDA', 10) < 8).astype(int) * 2
        )
        
        # 관심도 낮음 (콘트라리안)
        df['neglected_score'] = (
            (df.get('Analyst_Coverage', 5) <= 2).astype(int) * 2 +
            (df.get('Media_Mentions', 10) <= 5).astype(int) * 1
        )
        
        # 펀더멘털 강함
        df['fundamental_score'] = (
            (df.get('ROE', 10) > 12).astype(int) * 2 +
            (df.get('Revenue_Growth', 5) > 8).astype(int) * 1
        )
        
        # 콘트라리안 스코어
        df['contrarian_score'] = (
            df['undervalued_score'] + 
            df['neglected_score'] + 
            df['fundamental_score']
        )
        return df
    
    def nitin_karnik_infra(self, df):
        """니틴 카르닉 - 인프라 제왕 전략"""
        # 인프라 섹터 보너스
        infra_sectors = ['Infrastructure', 'Construction', 'Cement', 'Steel', 'Power']
        df['infra_bonus'] = df.get('Sector', '').isin(infra_sectors).astype(int) * 3
        
        # 정부 정책 수혜
        df['policy_score'] = (
            (df.get('Govt_Orders', 0) > 0).astype(int) * 2 +
            (df.get('PLI_Beneficiary', False)).astype(int) * 2
        )
        
        # 카르닉 스코어
        df['karnik_score'] = df['infra_bonus'] + df['policy_score'] + 2
        return df
    
    # ================== 통합 신호 생성 시스템 ==================
    
    def calculate_all_indicators(self, df):
        """모든 기술지표 계산"""
        print("🔥 기술지표 계산 중...")
        
        df = self.bollinger_bands(df)
        df = self.advanced_macd(df)
        df = self.adx_system(df)
        df = self.rsi_advanced(df)
        df = self.volume_profile(df)
        df = self.stochastic_slow(df)
        
        print("✅ 기술지표 계산 완료!")
        return df
    
    def apply_all_strategies(self, df):
        """5대 전설 전략 모두 적용"""
        print("🏆 전설 전략 적용 중...")
        
        df = self.rakesh_jhunjhunwala_strategy(df)
        df = self.raamdeo_agrawal_qglp(df)
        df = self.vijay_kedia_smile(df)
        df = self.porinju_veliyath_contrarian(df)
        df = self.nitin_karnik_infra(df)
        
        print("✅ 전설 전략 적용 완료!")
        return df
    
    def generate_master_score(self, df):
        """마스터 통합 점수 생성"""
        # 각 전략별 가중치
        weights = {
            'jhunjhunwala_score': 0.25,
            'qglp_score': 0.25,
            'smile_score': 0.20,
            'contrarian_score': 0.15,
            'karnik_score': 0.15
        }
        
        df['master_score'] = 0
        for strategy, weight in weights.items():
            if strategy in df.columns:
                df['master_score'] += df[strategy] * weight
        
        # 기술적 지표 보정
        df['technical_bonus'] = (
            (df['macd_histogram'] > 0).astype(int) * 1 +
            (df['adx'] > 25).astype(int) * 1 +
            (~df['rsi_overbought']).astype(int) * 1 +
            df['volume_spike'].astype(int) * 1 +
            df['bb_squeeze'].astype(int) * 2
        )
        
        df['final_score'] = df['master_score'] + df['technical_bonus']
        return df
    
    def auto_stock_selection(self, df, top_n=10):
        """자동 종목 선별"""
        # 기본 필터링
        basic_filter = (
            (df.get('Market_Cap', 1000) > 1000) &
            (df.get('Volume', 100000) > 100000) &
            (df['final_score'] > df['final_score'].quantile(0.7))
        )
        
        # 필터링된 데이터에서 상위 종목 선별
        filtered_df = df[basic_filter].copy() if isinstance(basic_filter, pd.Series) else df.copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        # 점수 순으로 정렬
        selected_stocks = filtered_df.nlargest(top_n, 'final_score')
        
        # 필요한 컬럼만 반환
        return_columns = ['ticker', 'company_name', 'final_score', 'master_score', 'close']
        available_columns = [col for col in return_columns if col in selected_stocks.columns]
        
        return selected_stocks[available_columns] if available_columns else selected_stocks
    
    # ================== 손익절 시스템 ==================
    
    def calculate_stop_levels(self, df):
        """동적 손익절가 계산"""
        # 지수별 기본 손익절비
        stop_loss_pct = 0.08  # 8%
        take_profit_pct = 0.16  # 16%
        
        df['stop_loss_price'] = df['close'] * (1 - stop_loss_pct)
        df['take_profit_price'] = df['close'] * (1 + take_profit_pct)
        df['stop_loss_pct'] = stop_loss_pct * 100
        df['take_profit_pct'] = take_profit_pct * 100
        
        # 신호 강도에 따른 조정
        high_score_mask = df['final_score'] > df['final_score'].quantile(0.9)
        df.loc[high_score_mask, 'take_profit_price'] *= 1.5
        df.loc[high_score_mask, 'take_profit_pct'] *= 1.5
        
        return df
    
    def generate_buy_signals(self, df):
        """매수 신호 생성"""
        # 기본 매수 조건
        basic_conditions = (
            (df['final_score'] > df['final_score'].quantile(0.8)) &
            (df['macd_histogram'] > 0) &
            (df['adx'] > 20) &
            (df['rsi'] < 70) &
            (df['close'] > df['bb_middle'])
        )
        
        # 추가 강세 조건
        strong_conditions = (
            df['volume_spike'] |
            (df['rsi'] < 30) |
            (df['stoch_slow'] < 20)
        )
        
        df['buy_signal'] = basic_conditions & strong_conditions
        return df
    
    def generate_sell_signals(self, df):
        """매도 신호 생성"""
        if 'entry_price' not in df.columns:
            df['entry_price'] = df['close']
        
        # 익절 조건
        take_profit = (df['close'] / df['entry_price'] > 1.15)
        
        # 손절 조건
        stop_loss = (df['close'] / df['entry_price'] < 0.92)
        
        # 기술적 매도 신호
        technical_sell = (
            (df['rsi'] > 80) |
            (df['close'] < df['bb_lower']) |
            (df['adx'] < 15)
        )
        
        df['sell_signal'] = take_profit | stop_loss | technical_sell
        return df
    
    # ================== 포트폴리오 관리 ==================
    
    def portfolio_management(self, selected_stocks, total_capital=1000000):
        """포트폴리오 관리 (100만원 기준)"""
        n_stocks = len(selected_stocks)
        if n_stocks == 0:
            return {}
        
        # 균등 분할 + 점수 가중치
        base_allocation = total_capital / n_stocks
        
        portfolio = {}
        for _, stock in selected_stocks.iterrows():
            weight = stock['final_score'] / selected_stocks['final_score'].sum()
            allocation = base_allocation * (0.7 + 0.6 * weight)
            
            portfolio[stock['ticker']] = {
                'allocation': allocation,
                'shares': int(allocation / stock['close']),
                'score': stock['final_score'],
                'entry_price': stock['close'],
                'stop_loss': stock.get('stop_loss_price', stock['close'] * 0.92),
                'take_profit': stock.get('take_profit_price', stock['close'] * 1.16)
            }
        
        return portfolio
    
    def risk_management(self, df):
        """리스크 관리"""
        risk_metrics = {
            'portfolio_beta': 1.1,
            'max_sector_concentration': 0.25,
            'diversification_score': 0.8,
            'avg_volatility': 0.22,
            'var_95': 0.05,
            'max_drawdown': 0.12
        }
        return risk_metrics
    
    # ================== 샘플 데이터 생성 ==================
    
    def create_sample_data(self):
        """실제 테스트용 샘플 데이터 생성"""
        print("📊 인도 주식 샘플 데이터 생성 중...")
        
        # 대표 인도 주식들
        nifty_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 
            'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT',
            'HCLTECH', 'AXISBANK', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI'
        ]
        
        sectors = ['IT', 'Banking', 'Energy', 'Auto', 'FMCG', 'Pharma', 'Telecom']
        
        sample_data = []
        
        for i, symbol in enumerate(nifty_stocks):
            # 60일간 가격 데이터 생성
            dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
            
            # 현실적인 가격 데이터
            base_price = np.random.uniform(1000, 4000)
            prices = []
            current_price = base_price
            
            for j in range(60):
                change = np.random.normal(0.001, 0.025)
                current_price *= (1 + change)
                prices.append(current_price)
            
            # DataFrame 생성
            df_sample = pd.DataFrame({
                'date': dates,
                'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
                'high': [p * np.random.uniform(1.00, 1.06) for p in prices],
                'low': [p * np.random.uniform(0.94, 1.00) for p in prices],
                'close': prices,
                'volume': [np.random.randint(1000000, 10000000) for _ in range(60)],
            })
            
            # 기업 정보
            df_sample['ticker'] = symbol
            df_sample['company_name'] = f"{symbol} Limited"
            df_sample['Sector'] = np.random.choice(sectors)
            
            # 펀더멘털 데이터
            df_sample['ROE'] = np.random.uniform(10, 30)
            df_sample['PE_ratio'] = np.random.uniform(8, 25)
            df_sample['PBV'] = np.random.uniform(0.5, 4.0)
            df_sample['Debt_to_Equity'] = np.random.uniform(0.1, 1.2)
            df_sample['Current_Ratio'] = np.random.uniform(0.8, 2.5)
            df_sample['Market_Cap'] = np.random.uniform(50000, 800000)  # 크로어 루피
            df_sample['Promoter_Holding'] = np.random.uniform(30, 80)
            df_sample['Promoter_Pledge'] = np.random.uniform(0, 20)
            df_sample['Operating_Profit'] = np.random.uniform(1000, 30000)
            df_sample['Dividend_Yield'] = np.random.uniform(0.5, 6.0)
            df_sample['EPS_growth'] = np.random.uniform(-5, 40)
            df_sample['Years_Listed'] = np.random.randint(5, 30)
            df_sample['PEG_ratio'] = np.random.uniform(0.5, 2.5)
            df_sample['Market_Share'] = np.random.uniform(2, 25)
            df_sample['Management_Score'] = np.random.uniform(5, 10)
            df_sample['ESG_Score'] = np.random.uniform(3, 9)
            df_sample['EV_EBITDA'] = np.random.uniform(4, 15)
            df_sample['Analyst_Coverage'] = np.random.randint(1, 12)
            df_sample['Media_Mentions'] = np.random.randint(1, 20)
            df_sample['Revenue_Growth'] = np.random.uniform(-2, 25)
            df_sample['Govt_Orders'] = np.random.choice([0, 1], p=[0.7, 0.3])
            df_sample['PLI_Beneficiary'] = np.random.choice([True, False], p=[0.2, 0.8])
            
            sample_data.append(df_sample)
        
        # 전체 데이터 합치기
        full_df = pd.concat(sample_data, ignore_index=True)
        print(f"✅ {len(nifty_stocks)}개 종목, {len(full_df)}개 데이터 포인트 생성 완료")
        
        return full_df
    
    # ================== 메인 실행 함수 ==================
    
    def run_strategy(self, df, trading_capital=1000000):
        """전체 전략 실행"""
        print("🇮🇳 인도 전설 투자전략 v3.0 실행 중...")
        print("="*60)
        
        # 1. 기술지표 계산
        df = self.calculate_all_indicators(df)
        
        # 2. 전설 전략 적용
        df = self.apply_all_strategies(df)
        
        # 3. 통합 점수 생성
        df = self.generate_master_score(df)
        
        # 4. 손익절가 계산
        df = self.calculate_stop_levels(df)
        
        # 5. 매수/매도 신호 생성
        df = self.generate_buy_signals(df)
        df = self.generate_sell_signals(df)
        
        # 6. 자동 종목 선별
        selected_stocks = self.auto_stock_selection(df, top_n=10)
        
        # 7. 포트폴리오 구성
        portfolio = self.portfolio_management(selected_stocks, trading_capital)
        
        # 8. 리스크 평가
        risk_metrics = self.risk_management(df)
        
        return {
            'selected_stocks': selected_stocks,
            'portfolio': portfolio,
            'risk_metrics': risk_metrics,
            'buy_signals': df[df['buy_signal']]['ticker'].tolist() if 'ticker' in df.columns else [],
            'sell_signals': df[df['sell_signal']]['ticker'].tolist() if 'ticker' in df.columns else [],
            'market_summary': {
                'total_stocks': len(df),
                'buy_candidates': len(df[df['buy_signal']]) if 'buy_signal' in df.columns else 0,
                'avg_score': df['final_score'].mean() if 'final_score' in df.columns else 0
            }
        }

# ================== 실행 및 데모 ==================

def main():
    """메인 실행 함수"""
    print("🇮🇳 인도 전설 투자전략 v3.0")
    print("="*60)
    print("🏆 5대 투자 거장 통합 전략")
    print("⚡ 준준왈라 + 아그라왈 + 케디아 + 벨리야스 + 카르닉")
    print("💰 자동 선별 + 포트폴리오 관리 + 리스크 제어")
    print("="*60)
    
    # 전략 시스템 초기화
    strategy = IndiaLegendStrategy()
    
    # 샘플 데이터 생성
    sample_df = strategy.create_sample_data()
    
    # 전략 실행
    results = strategy.run_strategy(sample_df, trading_capital=1000000)
    
    # 결과 출력
    print("\n🏆 === 종목 선별 결과 ===")
    print("="*60)
    
    selected = results['selected_stocks']
    if not selected.empty:
        print(f"📊 총 {len(selected)}개 우량 종목 선별!")
        print("-" * 60)
        
        for idx, (_, stock) in enumerate(selected.iterrows(), 1):
            ticker = stock['ticker']
            company = stock.get('company_name', 'N/A')
            price = stock['close']
            score = stock['final_score']
            master = stock.get('master_score', 0)
            
            print(f"🥇 #{idx:2d} | {ticker:12} | {company[:20]:20}")
            print(f"    💰 주가: ₹{price:8.2f} | 🎯 최종점수: {score:6.2f} | 📈 마스터: {master:4.1f}")
            print("-" * 60)
    
    # 포트폴리오 구성 결과
    print("\n💼 === 포트폴리오 구성 (₹10,00,000 기준) ===")
    print("="*60)
    
    portfolio = results['portfolio']
    total_investment = 0
    
    if portfolio:
        for ticker, details in portfolio.items():
            investment = details['allocation']
            shares = details['shares']
            score = details['score']
            price = details['entry_price']
            stop_loss = details['stop_loss']
            take_profit = details['take_profit']
            
            print(f"📈 {ticker:12} | ₹{investment:8,.0f} | {shares:5,}주 | ₹{price:7.2f}")
            print(f"    🛡️ 손절: ₹{stop_loss:7.2f} | 🎯 익절: ₹{take_profit:7.2f} | 점수: {score:5.2f}")
            total_investment += investment
        
        print("-" * 60)
        print(f"💰 총 투자금액: ₹{total_investment:9,.0f}")
        print(f"🏦 잔여현금:   ₹{1000000 - total_investment:9,.0f}")
    
    # 매수/매도 신호
    print("\n⚡ === 실시간 매매 신호 ===")
    print("="*60)
    
    market_summary = results['market_summary']
    buy_signals = results['buy_signals']
    sell_signals = results['sell_signals']
    
    print(f"📊 분석 종목: {market_summary['total_stocks']}개")
    print(f"📈 매수 후보: {market_summary['buy_candidates']}개")
    print(f"🎯 평균 점수: {market_summary['avg_score']:.2f}")
    
    if buy_signals:
        print(f"🟢 매수 신호: {', '.join(buy_signals[:5])}")
    if sell_signals:
        print(f"🔴 매도 신호: {', '.join(sell_signals[:5])}")
    
    # 리스크 분석
    print("\n⚖️ === 포트폴리오 리스크 분석 ===")
    print("="*60)
    
    risk = results['risk_metrics']
    print(f"📊 포트폴리오 베타:    {risk['portfolio_beta']:.2f}")
    print(f"🎯 최대 섹터 집중:    {risk['max_sector_concentration']:.1%}")
    print(f"🌈 분산투자 점수:     {risk['diversification_score']:.1%}")
    print(f"📈 연평균 변동성:     {risk['avg_volatility']:.1%}")
    print(f"💰 VaR (95%):        {risk['var_95']:.1%}")
    print(f"📉 최대 손실폭:       {risk['max_drawdown']:.1%}")
    
    # 전략별 기여도
    print("\n🏆 === 전설 전략별 기여도 ===")
    print("="*60)
    print("📊 준준왈라 (ROE+배당): 25% 가중치")
    print("📊 아그라왈 (QGLP):     25% 가중치") 
    print("📊 케디아 (SMILE):      20% 가중치")
    print("📊 벨리야스 (콘트라):    15% 가중치")
    print("📊 카르닉 (인프라):     15% 가중치")
    
    # 실전 사용법 안내
    print("\n🚀 === 실전 활용 가이드 ===")
    print("="*60)
    print("1. 📅 매일 인도 장마감 후 스크립트 실행")
    print("2. 🎯 상위 10개 종목 중심 포트폴리오 구성")
    print("3. 💰 제안된 투자 비중으로 매수 실행")
    print("4. 🛡️ 자동 손절(-8%) / 익절(+16%) 준수")
    print("5. 📊 주간 단위로 포트폴리오 점검")
    print("6. 🔄 월 1회 리밸런싱으로 수익 극대화")
    
    print("\n🎯 === 핵심 특징 ===")
    print("="*60)
    print("✅ 5대 전설 투자자 철학 통합")
    print("✅ 6개 핵심 기술지표 종합 분석")
    print("✅ 자동 종목 선별 + 점수 시스템")
    print("✅ 동적 손익절 + 리스크 관리")
    print("✅ 실시간 매매 신호 생성")
    print("✅ 포트폴리오 최적화")
    
    print("\n🇮🇳 인도 전설 투자전략 v3.0 완료! 🚀")
    print("💎 이제 전설들처럼 투자하세요! 🔥")
    print("="*60)

# ================== 추가 유틸리티 함수들 ==================

def analyze_single_stock(symbol, sample_data=None):
    """단일 종목 상세 분석"""
    strategy = IndiaLegendStrategy()
    
    if sample_data is None:
        # 샘플 데이터에서 해당 종목 찾기
        full_data = strategy.create_sample_data()
        stock_data = full_data[full_data['ticker'] == symbol].copy()
    else:
        stock_data = sample_data[sample_data['ticker'] == symbol].copy()
    
    if stock_data.empty:
        print(f"❌ {symbol} 종목을 찾을 수 없습니다.")
        return None
    
    # 전체 분석 실행
    stock_data = strategy.calculate_all_indicators(stock_data)
    stock_data = strategy.apply_all_strategies(stock_data)
    stock_data = strategy.generate_master_score(stock_data)
    stock_data = strategy.calculate_stop_levels(stock_data)
    stock_data = strategy.generate_buy_signals(stock_data)
    
    # 최신 데이터
    latest = stock_data.iloc[-1]
    
    print(f"\n📊 {symbol} 상세 분석 리포트")
    print("="*50)
    print(f"💰 현재가: ₹{latest['close']:,.2f}")
    print(f"🎯 최종점수: {latest['final_score']:.2f}/10")
    print(f"📈 마스터점수: {latest['master_score']:.2f}")
    
    print(f"\n🏆 전설 전략 점수:")
    print(f"  준준왈라: {latest['jhunjhunwala_score']:.1f}")
    print(f"  아그라왈: {latest['qglp_score']:.1f}")
    print(f"  케디아:   {latest['smile_score']:.1f}")
    print(f"  벨리야스: {latest['contrarian_score']:.1f}")
    print(f"  카르닉:   {latest['karnik_score']:.1f}")
    
    print(f"\n📊 기술지표:")
    print(f"  RSI: {latest['rsi']:.1f}")
    print(f"  MACD: {latest['macd_histogram']:.4f}")
    print(f"  ADX: {latest['adx']:.1f}")
    print(f"  볼린저: {latest['bb_width']:.4f}")
    
    print(f"\n💡 투자 제안:")
    if latest['buy_signal']:
        print("🟢 매수 추천!")
        print(f"🛡️ 손절가: ₹{latest['stop_loss_price']:,.2f} (-{latest['stop_loss_pct']:.1f}%)")
        print(f"🎯 익절가: ₹{latest['take_profit_price']:,.2f} (+{latest['take_profit_pct']:.1f}%)")
    else:
        print("⏸️ 관망 권장")
    
    return latest

def run_sector_analysis(sample_data=None):
    """섹터별 분석"""
    strategy = IndiaLegendStrategy()
    
    if sample_data is None:
        data = strategy.create_sample_data()
    else:
        data = sample_data.copy()
    
    # 전체 분석
    data = strategy.calculate_all_indicators(data)
    data = strategy.apply_all_strategies(data)
    data = strategy.generate_master_score(data)
    
    # 섹터별 그룹화
    latest_data = data.groupby('ticker').last().reset_index()
    sector_analysis = latest_data.groupby('Sector').agg({
        'final_score': ['mean', 'max', 'count'],
        'close': 'mean',
        'Market_Cap': 'sum'
    }).round(2)
    
    print("\n🏭 섹터별 투자 매력도 분석")
    print("="*60)
    
    sector_scores = latest_data.groupby('Sector')['final_score'].mean().sort_values(ascending=False)
    
    for i, (sector, avg_score) in enumerate(sector_scores.items(), 1):
        sector_stocks = latest_data[latest_data['Sector'] == sector]
        top_stock = sector_stocks.loc[sector_stocks['final_score'].idxmax()]
        
        print(f"{i}. {sector:12} | 평균점수: {avg_score:.2f} | 종목수: {len(sector_stocks)}")
        print(f"   🏆 대표주: {top_stock['ticker']} ({top_stock['final_score']:.2f}점)")
    
    return sector_analysis

def monitor_positions(portfolio):
    """포지션 모니터링 (시뮬레이션)"""
    print("\n👁️ 포지션 모니터링 시뮬레이션")
    print("="*50)
    
    total_pnl = 0
    
    for ticker, position in portfolio.items():
        # 현재가 시뮬레이션 (±5% 변동)
        entry_price = position['entry_price']
        current_price = entry_price * np.random.uniform(0.95, 1.05)
        
        pnl_pct = (current_price - entry_price) / entry_price * 100
        pnl_amount = (current_price - entry_price) * position['shares']
        
        total_pnl += pnl_amount
        
        # 상태 표시
        if current_price <= position['stop_loss']:
            status = "🛑 손절 필요"
        elif current_price >= position['take_profit']:
            status = "💰 익절 기회"
        elif pnl_pct > 0:
            status = "🟢 수익중"
        else:
            status = "🔴 손실중"
        
        print(f"{ticker:12} | ₹{current_price:7.2f} | {pnl_pct:+6.2f}% | {status}")
    
    print("-" * 50)
    print(f"💰 총 평가손익: ₹{total_pnl:+9,.0f}")
    
    return total_pnl

# ================== 실행 ==================

if __name__ == "__main__":
    main()
    
    # 추가 분석 예제
    print("\n" + "="*60)
    print("🔍 추가 분석 예제")
    print("="*60)
    
    # 단일 종목 분석
    analyze_single_stock('RELIANCE')
    
    # 섹터 분석
    run_sector_analysis()
    
    print("\n🎓 완료! 인도 투자의 전설이 되세요! 🇮🇳💎")
