import streamlit as st
import pandas as pd
import sqlite3

DB_FILE = "trading.db"

def load_trading_history():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM trading_history ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def load_reflection():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT r.*, t.timestamp 
        FROM trading_reflection r 
        JOIN trading_history t ON r.trading_id = t.id 
        ORDER BY r.reflection_date DESC
    """, conn)
    conn.close()
    return df

st.title("📊 AI 자동매매 대시보드")

st.subheader("📈 거래 기록")
trades_df = load_trading_history()
st.dataframe(trades_df if not trades_df.empty else pd.DataFrame({"메시지": ["거래 기록 없음"]}))

st.subheader("💡 반성 기록")
reflection_df = load_reflection()
st.dataframe(reflection_df if not reflection_df.empty else pd.DataFrame({"메시지": ["반성 기록 없음"]}))

st.subheader("📅 기간 필터")
start_date = st.date_input("시작일")
end_date = st.date_input("종료일")

if not trades_df.empty:
    filtered = trades_df[
        (pd.to_datetime(trades_df["timestamp"]) >= pd.to_datetime(start_date)) &
        (pd.to_datetime(trades_df["timestamp"]) <= pd.to_datetime(end_date))
    ]
    st.dataframe(filtered)

