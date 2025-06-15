import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime

# DB 연결
conn = sqlite3.connect("trading.db", check_same_thread=False)

# 거래 기록 로드
def load_trading_history(start=None, end=None):
    query = "SELECT * FROM trading_history"
    if start and end:
        query += f" WHERE DATE(timestamp) BETWEEN '{start}' AND '{end}'"
    query += " ORDER BY id DESC"
    return pd.read_sql_query(query, conn)

# 반성 기록 로드
def load_reflection_history():
    query = """
    SELECT r.*, t.asset_name, t.asset_type, t.timestamp
    FROM trading_reflection r
    JOIN trading_history t ON r.trading_id = t.id
    ORDER BY r.reflection_date DESC
    """
    return pd.read_sql_query(query, conn)

# Streamlit 시작
st.set_page_config(page_title="📈 AI 자동매매 통합 대시보드", layout="wide")
st.title("📈 AI 코인/미국/일본 자동매매 대시보드")
st.markdown("AI 기반 자동매매 기록과 반성 결과를 시각적으로 분석합니다.")

# 날짜 필터 + 자산 유형 필터
st.subheader("📅 거래 필터")
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("시작 날짜", value=datetime(2025, 1, 1))
with col2:
    end_date = st.date_input("종료 날짜", value=datetime.today())
with col3:
    asset_types = st.multiselect("자산 유형", ["coin", "us", "japan"], default=["coin", "us", "japan"])

# 데이터 로드
df = load_trading_history(start_date, end_date)
if not df.empty:
    df = df[df["asset_type"].isin(asset_types)]

# 거래 데이터
st.subheader("📋 거래 기록")
if df.empty:
    st.warning("❗ 선택한 조건에 맞는 거래 기록이 없습니다.")
else:
    st.dataframe(df, use_container_width=True)

    st.subheader("📊 요약 통계")
    col1, col2, col3 = st.columns(3)
    col1.metric("총 거래 수", len(df))
    col2.metric("평균 신뢰도", f"{df['confidence_score'].mean():.2f}")
    col3.metric("평균 비중 (%)", f"{df['percentage'].mean():.2f}")

    st.subheader("📈 신뢰도 분포")
    fig_conf = px.histogram(df, x="confidence_score", nbins=10, color="asset_type", title="신뢰도 분포")
    st.plotly_chart(fig_conf, use_container_width=True)

    st.subheader("📉 매수가 대비 현재 수익률")
    df["수익률(%)"] = ((df["current_price"] - df["avg_buy_price"]) / df["avg_buy_price"]) * 100
    fig_profit = px.line(df.sort_values("timestamp"),
                         x="timestamp",
                         y="수익률(%)",
                         color="asset_type",
                         line_group="asset_name",
                         title="실현 수익률 추이")
    st.plotly_chart(fig_profit, use_container_width=True)

# 반성 기록
st.subheader("🧠 반성 기록 요약")
reflection_df = load_reflection_history()

if reflection_df.empty:
    st.info("🙇 반성 기록이 없습니다.")
else:
    st.dataframe(reflection_df, use_container_width=True)

    st.subheader("🎯 반성 성공률 분포")
    fig_success = px.histogram(reflection_df, x="success_rate", nbins=10, color="asset_type", title="반성 성공률 분포")
    st.plotly_chart(fig_success, use_container_width=True)

st.caption("© 2025 AI 자동매매 시스템 - 코인/미국/일본 통합 분석")

