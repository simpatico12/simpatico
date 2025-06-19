# streamlit_app.py
"""
퀀트 트레이딩 실시간 대시보드 - 퀸트프로젝트 수준
고급 차트와 실시간 모니터링 기능 포함
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List
import time

# 프로젝트 모듈
from db import db_manager, TradeRecord
from config import get_config
import pyupbit

# 페이지 설정
st.set_page_config(
    page_title="퀀트 트레이딩 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .success-metric {
        color: #1f77b4;
    }
    .danger-metric {
        color: #d62728;
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """트레이딩 대시보드 클래스"""
    
    def __init__(self):
        self.cfg = get_config()
        self.upbit = self._init_exchange()
        
    def _init_exchange(self):
        """거래소 연결"""
        try:
            return pyupbit.Upbit(
                self.cfg['api']['access_key'],
                self.cfg['api']['secret_key']
            )
        except:
            return None
    
    def render_header(self):
        """헤더 렌더링"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("🚀 퀀트 트레이딩 대시보드")
        
        with col2:
            if st.button("🔄 새로고침"):
                st.rerun()
        
        with col3:
            st.write(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def render_realtime_metrics(self):
        """실시간 지표"""
        st.header("📊 실시간 포트폴리오")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # 계좌 정보
        if self.upbit:
            balances = self.upbit.get_balances()
            total_krw = sum(float(b['balance']) * float(b['avg_buy_price']) 
                          for b in balances if b['currency'] != 'KRW')
            cash = next((float(b['balance']) for b in balances 
                        if b['currency'] == 'KRW'), 0)
            total_value = total_krw + cash
        else:
            total_value = cash = total_krw = 0
        
        # 성과 지표
        metrics = db_manager.calculate_portfolio_metrics()
        
        with col1:
            st.metric(
                "총 자산",
                f"₩{total_value:,.0f}",
                f"{metrics.get('total_return', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                "보유 현금",
                f"₩{cash:,.0f}",
                f"{(cash/total_value*100):.1f}%" if total_value > 0 else "0%"
            )
        
        with col3:
            st.metric(
                "투자 금액",
                f"₩{total_krw:,.0f}",
                f"{(total_krw/total_value*100):.1f}%" if total_value > 0 else "0%"
            )
        
        with col4:
            win_rate = metrics.get('win_rate', 0)
            st.metric(
                "승률",
                f"{win_rate:.1f}%",
                "🟢" if win_rate > 50 else "🔴"
            )
        
        with col5:
            sharpe = metrics.get('sharpe_ratio', 0)
            st.metric(
                "샤프비율",
                f"{sharpe:.2f}",
                "⬆️" if sharpe > 1 else "⬇️"
            )
    
    def render_portfolio_chart(self):
        """포트폴리오 차트"""
        st.header("💼 포트폴리오 구성")
        
        if not self.upbit:
            st.warning("거래소 연결 필요")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 자산 구성 파이차트
            balances = self.upbit.get_balances()
            portfolio_data = []
            
            for b in balances:
                if b['currency'] != 'KRW' and float(b['balance']) > 0:
                    value = float(b['balance']) * float(b['avg_buy_price'])
                    portfolio_data.append({
                        'asset': b['currency'],
                        'value': value,
                        'balance': float(b['balance']),
                        'avg_price': float(b['avg_buy_price'])
                    })
            
            if portfolio_data:
                df_portfolio = pd.DataFrame(portfolio_data)
                fig = px.pie(
                    df_portfolio, 
                    values='value', 
                    names='asset',
                    title='자산별 비중'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("보유 자산 없음")
        
        with col2:
            # 자산별 수익률
            if portfolio_data:
                for asset in portfolio_data:
                    current_price = pyupbit.get_current_price(f"KRW-{asset['asset']}")
                    if current_price:
                        profit_rate = (current_price - asset['avg_price']) / asset['avg_price'] * 100
                        color = "green" if profit_rate > 0 else "red"
                        st.markdown(
                            f"**{asset['asset']}**: "
                            f"<span style='color:{color}'>{profit_rate:+.2f}%</span>",
                            unsafe_allow_html=True
                        )
    
    def render_trading_history(self):
        """거래 내역"""
        st.header("📈 거래 분석")
        
        # 필터 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days = st.selectbox("기간", [7, 30, 90, 365], index=1)
        
        with col2:
            asset_type = st.selectbox("자산 유형", ["전체", "stock", "coin"])
        
        with col3:
            decision = st.selectbox("거래 유형", ["전체", "buy", "sell"])
        
        # 데이터 조회
        trades = db_manager.get_recent_trades(
            asset_type=None if asset_type == "전체" else asset_type,
            days=days
        )
        
        if decision != "전체":
            trades = [t for t in trades if t.decision == decision]
        
        if not trades:
            st.info("거래 내역이 없습니다")
            return
        
        # DataFrame 변환
        df_trades = pd.DataFrame([{
            'timestamp': t.timestamp,
            'asset': t.asset,
            'type': t.asset_type,
            'decision': t.decision,
            'price': t.avg_price,
            'volume': t.executed_volume,
            'profit_rate': t.profit_rate or 0,
            'confidence': t.confidence_score
        } for t in trades])
        
        # 거래 통계
        col1, col2 = st.columns(2)
        
        with col1:
            # 일별 거래량 차트
            daily_trades = df_trades.groupby(df_trades['timestamp'].dt.date).size()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_trades.index,
                y=daily_trades.values,
                name='거래 수'
            ))
            fig.update_layout(title='일별 거래 활동', height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 수익률 히스토그램
            profit_trades = df_trades[df_trades['profit_rate'] != 0]
            if not profit_trades.empty:
                fig = px.histogram(
                    profit_trades,
                    x='profit_rate',
                    nbins=30,
                    title='수익률 분포',
                    color_discrete_sequence=['#1f77b4']
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_analysis(self):
        """성과 분석"""
        st.header("📊 성과 분석")
        
        # 누적 수익률 차트
        trades = db_manager.get_recent_trades(days=90)
        if not trades:
            st.info("분석할 데이터가 없습니다")
            return
        
        # 시계열 데이터 준비
        df_perf = pd.DataFrame([{
            'date': t.timestamp.date(),
            'total_value': t.total_asset,
            'decision': t.decision,
            'asset': t.asset
        } for t in trades])
        
        # 일별 자산 추이
        daily_value = df_perf.groupby('date')['total_value'].last()
        
        # 누적 수익률 계산
        initial_value = daily_value.iloc[0] if len(daily_value) > 0 else 1
        cumulative_returns = (daily_value / initial_value - 1) * 100
        
        # 차트 생성
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('누적 수익률 (%)', '일별 거래 수'),
            row_heights=[0.7, 0.3]
        )
        
        # 누적 수익률
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='누적 수익률',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # 0% 기준선
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1)
        
        # 일별 거래 수
        daily_trades = df_perf.groupby('date').size()
        fig.add_trace(
            go.Bar(
                x=daily_trades.index,
                y=daily_trades.values,
                name='거래 수'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 종목별 성과
        st.subheader("🏆 종목별 성과")
        
        asset_performance = []
        unique_assets = df_perf['asset'].unique()
        
        for asset in unique_assets:
            perf = db_manager.get_asset_performance(asset, days=90)
            if perf:
                asset_performance.append(perf)
        
        if asset_performance:
            df_asset_perf = pd.DataFrame(asset_performance)
            df_asset_perf = df_asset_perf.sort_values('total_return', ascending=False)
            
            # 상위/하위 5개
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🟢 상위 성과**")
                top5 = df_asset_perf.head(5)
                for _, row in top5.iterrows():
                    st.write(f"• {row['asset']}: {row['total_return']:.2f}% (승률 {row['win_rate']:.1f}%)")
            
            with col2:
                st.write("**🔴 하위 성과**")
                bottom5 = df_asset_perf.tail(5)
                for _, row in bottom5.iterrows():
                    st.write(f"• {row['asset']}: {row['total_return']:.2f}% (승률 {row['win_rate']:.1f}%)")
    
    def render_risk_analysis(self):
        """리스크 분석"""
        st.header("⚠️ 리스크 관리")
        
        metrics = db_manager.calculate_portfolio_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 최대 낙폭
            mdd = metrics.get('max_drawdown', 0)
            color = "red" if mdd > 15 else "orange" if mdd > 10 else "green"
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h3>최대 낙폭 (MDD)</h3>
                <h1 style='color: {color}'>{mdd:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 변동성
            volatility = metrics.get('volatility', 0)
            color = "red" if volatility > 30 else "orange" if volatility > 20 else "green"
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h3>변동성 (연환산)</h3>
                <h1 style='color: {color}'>{volatility:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # 평균 손실
            recent_trades = db_manager.get_recent_trades(days=30)
            losses = [t.profit_rate for t in recent_trades if t.profit_rate and t.profit_rate < 0]
            avg_loss = np.mean(losses) if losses else 0
            
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h3>평균 손실</h3>
                <h1 style='color: red'>{avg_loss:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # 리스크 경고
        warnings = []
        if mdd > 20:
            warnings.append("⚠️ 최대 낙폭이 20%를 초과했습니다. 포지션 크기 조정을 고려하세요.")
        if volatility > 30:
            warnings.append("⚠️ 높은 변동성이 감지되었습니다. 분산 투자를 강화하세요.")
        if len(losses) > len(recent_trades) * 0.6:
            warnings.append("⚠️ 최근 손실 거래가 많습니다. 전략 재검토가 필요합니다.")
        
        if warnings:
            st.warning("\n\n".join(warnings))
    
    def render_sidebar(self):
        """사이드바"""
        with st.sidebar:
            st.header("⚙️ 설정")
            
            # 자동 새로고침
            auto_refresh = st.checkbox("자동 새로고침 (10초)")
            if auto_refresh:
                time.sleep(10)
                st.rerun()
            
            # 테마 선택
            theme = st.selectbox("테마", ["Light", "Dark"])
            
            # 데이터 내보내기
            st.header("💾 데이터 내보내기")
            
            if st.button("거래 내역 CSV 다운로드"):
                trades = db_manager.get_recent_trades(days=365)
                if trades:
                    df = pd.DataFrame([t.__dict__ for t in trades])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 다운로드",
                        data=csv,
                        file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            # 시스템 정보
            st.header("ℹ️ 시스템 정보")
            st.write(f"버전: {self.cfg.get('version', '1.0.0')}")
            st.write(f"환경: {self.cfg.get('environment', 'production')}")
            
            # 최근 에러
            st.header("🚨 최근 에러")
            # 에러 로그 표시 (구현 필요)


def main():
    """메인 함수"""
    dashboard = TradingDashboard()
    
    # 헤더
    dashboard.render_header()
    
    # 실시간 지표
    dashboard.render_realtime_metrics()
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["포트폴리오", "거래 분석", "성과 분석", "리스크 관리"])
    
    with tab1:
        dashboard.render_portfolio_chart()
    
    with tab2:
        dashboard.render_trading_history()
    
    with tab3:
        dashboard.render_performance_analysis()
    
    with tab4:
        dashboard.render_risk_analysis()
    
    # 사이드바
    dashboard.render_sidebar()


if __name__ == "__main__":
    main()