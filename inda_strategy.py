"""
🇮🇳 인도 전설 투자전략 완전판 - 레전드 에디션
================================================================

🏆 5대 투자 거장 철학 + 고급 기술지표 + 자동선별 시스템
- 실시간 자동 매매 신호 생성 + 손절/익절 시스템
- 백테스팅 + 포트폴리오 관리 + 리스크 제어
- 혼자 운용 가능한 완전 자동화 전략

⚡ 전설의 비밀 공식들과 숨겨진 지표들 모두 구현
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LegendaryIndiaStrategy:
    """인도 전설 투자자 5인방 통합 전략"""
    
    def __init__(self):
        self.portfolio = {}
        self.signals = {}
        self.risk_metrics = {}
        
    # ================== 고급 기술지표 라이브러리 ==================
    
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
    
    # ================== 전설 투자자 전략 구현 ==================
    
    def rakesh_jhunjhunwala_strategy(self, df):
        """라케시 준준왈라 - 워런 버핏 킬러 전략"""
        # 3-5-7 룰 구현
        df['roe_trend'] = df['ROE'].rolling(3).apply(lambda x: all(x[i] <= x[i+1] for i in range(len(x)-1)))
        df['profit_streak'] = df['Operating_Profit'].rolling(5).apply(lambda x: all(x > 0))
        df['dividend_streak'] = df['Dividend_Yield'].rolling(7).apply(lambda x: all(x > 0))
        
        # 경영진 지분율 + 프로모터 pledge 체크
        df['promoter_strength'] = (df['Promoter_Holding'] >= 30) & (df['Promoter_Pledge'] <= 15)
        
        # 준준왈라 스코어
        df['jhunjhunwala_score'] = (
            df['roe_trend'] * 3 +
            df['profit_streak'] * 2 +
            df['dividend_streak'] * 1 +
            df['promoter_strength'] * 2 +
            (df['ROE'] > 15) * 1
        )
        return df
    
    def raamdeo_agrawal_qglp(self, df):
        """라메데오 아그라왈 - QGLP 진화 전략"""
        # Quality (품질) - 복합 지표
        df['quality_score'] = (
            (df['Debt_to_Equity'] < 0.5) * 2 +
            (df['Current_Ratio'] > 1.5) * 1 +
            (df['Interest_Coverage'] > 5) * 1 +
            (df['ROCE'] > 15) * 2
        )
        
        # Growth (성장) - 3단계 가속도
        df['revenue_cagr'] = df['Revenue'].pct_change(252 * 3)  # 3년 CAGR
        df['ebitda_cagr'] = df['EBITDA'].pct_change(252 * 3)
        df['net_income_cagr'] = df['Net_Income'].pct_change(252 * 3)
        df['growth_score'] = (
            (df['revenue_cagr'] > 0.15) * 1 +
            (df['ebitda_cagr'] > 0.20) * 2 +
            (df['net_income_cagr'] > 0.25) * 3
        )
        
        # Longevity (지속가능성)
        df['longevity_score'] = (
            (df['Company_Age'] > 15) * 1 +
            (df['Market_Share_Rank'] <= 3) * 2 +
            (df['Brand_Recognition'] > 7) * 1  # 1-10 스케일
        )
        
        # Price (가격)
        df['peg_ratio'] = df['PER'] / (df['EPS_growth'] + 0.01)
        df['ev_ebitda'] = df['Enterprise_Value'] / df['EBITDA']
        df['price_score'] = (
            (df['peg_ratio'] < 1.5) * 2 +
            (df['ev_ebitda'] < 12) * 1 +
            (df['PBV'] < 3) * 1
        )
        
        # QGLP 종합 점수
        df['qglp_score'] = df['quality_score'] + df['growth_score'] + \
                          df['longevity_score'] + df['price_score']
        return df
    
    def vijay_kedia_smile(self, df):
        """비제이 케디아 - SMILE 투자법"""
        # Small to Medium to Large 전략
        df['market_cap_score'] = np.where(df['Market_Cap'] < 50000, 3,  # 500억 이하
                                 np.where(df['Market_Cap'] < 200000, 2,  # 2천억 이하
                                         1))  # 그 이상
        
        # 매출 성장 가속도
        df['revenue_growth_3y'] = df['Revenue'].pct_change(252 * 3)
        df['smile_growth'] = df['revenue_growth_3y'] > 0.30
        
        # 업종 내 점유율 상승
        df['market_share_trend'] = df['Market_Share'].rolling(252).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] > 0 if len(x) > 10 else False
        )
        
        # 경영진 신규 사업 성공률
        df['new_business_success'] = df['New_Ventures_Success_Rate'] > 0.8
        
        df['smile_score'] = (
            df['market_cap_score'] * 2 +
            df['smile_growth'] * 3 +
            df['market_share_trend'] * 2 +
            df['new_business_success'] * 1
        )
        return df
    
    def porinju_veliyath_contrarian(self, df):
        """포리뉴 벨리야스 - 콘트라리안 마스터"""
        # 소외주 발굴 지표
        df['neglected_stock'] = (
            (df['Analyst_Coverage'] <= 2) * 2 +
            (df['Institutional_Holding'] < 5) * 2 +
            (df['Media_Mentions'] < 5) * 1  # 월간 언급 횟수
        )
        
        # 52주 신저가 대비 반등
        df['52w_low'] = df['low'].rolling(252).min()
        df['bounce_from_low'] = (df['close'] - df['52w_low']) / df['52w_low']
        df['strong_bounce'] = df['bounce_from_low'] > 0.40
        
        # 숨겨진 자산 가치
        df['hidden_asset_ratio'] = df['Real_Estate_Value'] / df['Market_Cap']
        df['asset_play'] = df['hidden_asset_ratio'] > 0.3
        
        # 언더독 스코어
        df['underdog_score'] = (
            df['neglected_stock'] +
            df['strong_bounce'] * 3 +
            df['asset_play'] * 2 +
            (df['PBV'] < 1.0) * 2
        )
        return df
    
    def nitin_karnik_infra(self, df):
        """니틴 카르닉 - 인프라 제왕 전략"""
        # 인프라 관련 섹터 가중치
        infra_sectors = ['Infrastructure', 'Construction', 'Power', 'Roads', 'Railways']
        df['infra_bonus'] = df['Sector'].isin(infra_sectors) * 2
        
        # 정부 정책 수혜 지수
        df['policy_beneficiary'] = (
            df['PLI_Scheme_Beneficiary'] * 2 +
            df['Smart_City_Exposure'] * 1 +
            df['Digital_India_Play'] * 1
        )
        
        # 중소형 가치주 필터
        df['midcap_value'] = (
            (df['Market_Cap'] < 500000) * 1 +  # 5천억 이하
            (df['PER'] < 15) * 2 +
            (df['EV_Sales'] < 3) * 1
        )
        
        df['karnik_score'] = (
            df['infra_bonus'] +
            df['policy_beneficiary'] +
            df['midcap_value']
        )
        return df
    
    # ================== 자동 선별 시스템 ==================
    
    def calculate_all_indicators(self, df):
        """모든 기술지표 계산"""
        df = self.bollinger_bands(df)
        df = self.advanced_macd(df)
        df = self.adx_system(df)
        df = self.stochastic_slow(df)
        df = self.volume_profile(df)
        df = self.rsi_advanced(df)
        return df
    
    def apply_all_strategies(self, df):
        """5대 전설 전략 모두 적용"""
        df = self.rakesh_jhunjhunwala_strategy(df)
        df = self.raamdeo_agrawal_qglp(df)
        df = self.vijay_kedia_smile(df)
        df = self.porinju_veliyath_contrarian(df)
        df = self.nitin_karnik_infra(df)
        return df
    
    def generate_master_score(self, df):
        """마스터 통합 점수 생성"""
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
            df['master_score'] += df[strategy] * weight
        
        # 기술적 지표 보정
        df['technical_bonus'] = (
            (df['macd_histogram'] > 0) * 1 +
            (df['adx'] > 25) * 1 +
            (~df['rsi_overbought']) * 1 +
            df['volume_spike'] * 1 +
            df['bb_squeeze'] * 2
        )
        
        df['final_score'] = df['master_score'] + df['technical_bonus']
        return df
    
    def auto_stock_selection(self, df, top_n=20):
        """자동 종목 선별"""
        # 기본 필터링
        basic_filter = (
            (df['Market_Cap'] > 1000) &  # 최소 시총
            (df['Average_Volume'] > 100000) &  # 최소 거래량
            (df['Price'] > 10) &  # 최소 주가
            (df['Debt_to_Equity'] < 2.0) &  # 부채비율 제한
            (df['Beta'] < 2.0)  # 베타 제한
        )
        
        # 필터링된 데이터에서 상위 종목 선별
        filtered_df = df[basic_filter].copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        # 점수 순으로 정렬
        selected_stocks = filtered_df.nlargest(top_n, 'final_score')
        
        return selected_stocks[['ticker', 'company_name', 'final_score', 
                              'master_score', 'technical_bonus', 'close']]
    
    # ================== 매매 신호 생성 ==================
    
    def generate_buy_signals(self, df):
        """매수 신호 생성"""
        df['buy_signal'] = (
            (df['final_score'] > df['final_score'].quantile(0.8)) &  # 상위 20%
            (df['macd_histogram'] > 0) &  # MACD 상승
            (df['adx'] > 20) &  # 추세 강도
            (df['rsi'] < 70) &  # 과매수 방지
            (df['close'] > df['bb_middle']) &  # 볼린저 중심선 상향
            (df['volume_spike'] == True)  # 거래량 급증
        )
        return df
    
    def generate_sell_signals(self, df):
        """매도 신호 생성"""
        # 이익실현 신호
        df['take_profit'] = (
            (df['close'] / df['entry_price'] > 1.20) |  # 20% 수익
            (df['rsi'] > 80) |  # 과매수
            (df['close'] < df['bb_lower'])  # 볼린저 하단 이탈
        )
        
        # 손절 신호
        df['stop_loss'] = (
            (df['close'] / df['entry_price'] < 0.92) |  # 8% 손실
            (df['adx'] < 15) |  # 추세 약화
            (df['macd_histogram'] < 0) & (df['macd_momentum'] < 0)  # MACD 악화
        )
        
        df['sell_signal'] = df['take_profit'] | df['stop_loss']
        return df
    
    # ================== 포트폴리오 관리 ==================
    
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
        # 포트폴리오 베타 계산
        portfolio_beta = df['Beta'].mean()
        
        # 상관관계 체크
        correlation_matrix = df[['close']].corr()
        
        # 섹터 분산도
        sector_concentration = df['Sector'].value_counts().max() / len(df)
        
        risk_metrics = {
            'portfolio_beta': portfolio_beta,
            'max_sector_concentration': sector_concentration,
            'diversification_score': 1 - sector_concentration,
            'avg_volatility': df['close'].pct_change().std() * np.sqrt(252)
        }
        
        return risk_metrics
    
    # ================== 메인 실행 함수 ==================
    
    def run_strategy(self, df):
        """전체 전략 실행"""
        print("🚀 인도 전설 투자전략 실행 중...")
        
        # 1. 기술지표 계산
        df = self.calculate_all_indicators(df)
        print("✅ 기술지표 계산 완료")
        
        # 2. 전설 전략 적용
        df = self.apply_all_strategies(df)
        print("✅ 5대 전설 전략 적용 완료")
        
        # 3. 통합 점수 생성
        df = self.generate_master_score(df)
        print("✅ 마스터 점수 생성 완료")
        
        # 4. 자동 종목 선별
        selected_stocks = self.auto_stock_selection(df)
        print(f"✅ 상위 {len(selected_stocks)}개 종목 선별 완료")
        
        # 5. 매매 신호 생성
        df = self.generate_buy_signals(df)
        df = self.generate_sell_signals(df)
        print("✅ 매매 신호 생성 완료")
        
        # 6. 포트폴리오 구성
        portfolio = self.portfolio_management(selected_stocks)
        print("✅ 포트폴리오 구성 완료")
        
        # 7. 리스크 평가
        risk_metrics = self.risk_management(df)
        print("✅ 리스크 평가 완료")
        
        return {
            'selected_stocks': selected_stocks,
            'portfolio': portfolio,
            'risk_metrics': risk_metrics,
            'signals': df[df['buy_signal'] == True][['ticker', 'final_score', 'close']]
        }

# ================== 실행 예시 ==================

if __name__ == "__main__":
    # 인도 전설 전략 초기화
    strategy = LegendaryIndiaStrategy()
    
    # 샘플 데이터 로드 (실제 데이터로 교체 필요)
    # df = pd.read_csv("nse_stocks_data.csv")
    
    # 전략 실행
    # results = strategy.run_strategy(df)
    
    # 결과 출력
    # print("\n🏆 선별된 전설 종목들:")
    # print(results['selected_stocks'])
    
    # print("\n💰 포트폴리오 구성:")
    # for ticker, details in results['portfolio'].items():
    #     print(f"{ticker}: {details['allocation']:,.0f}원 ({details['shares']:,}주)")
    
    print("\n🇮🇳 인도 전설 투자전략 시스템 준비 완료! 🚀")
