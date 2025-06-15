import streamlit as st
import pandas as pd
import sqlite3

DB_FILE = "trading.db"

def load_trading_history():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("SELECT * FROM trading_history ORDER BY timestamp DESC", conn)
        return df
    except Exception as e:
        st.error(f"❌ 거래 기록 불러오기 오류: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def load_reflection():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            """
            SELECT r.*, t.timestamp 
            FROM trading_reflection r 
            JOIN trading_history t ON r.trading_id = t.id 
            ORDER BY r.reflection_date DESC
            """, conn)
        return df
    except Exception as e:
        st.error(f"❌ 반성 기록 불러오기 오류: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

st.title("📊 AI 코인/미국/일본 자동매매 대시보드")

st.subheader("📈 거래 기록")
trades_df = load_trading_history()
if trades_df.empty:
    st.warning("거래 기록이 없습니다.")
else:
    st.dataframe(trades_df)

st.subheader("💡 반성 기록 요약")
reflection_df = load_reflection()
if reflection_df.empty:
    st.warning("반성 기록이 없습니다.")
else:
    st.dataframe(reflection_df)

st.subheader("📅 날짜별 필터")
start_date = st.date_input("시작 날짜")
end_date = st.date_input("종료 날짜")

if not trades_df.empty:
    filtered = trades_df[
        (pd.to_datetime(trades_df['timestamp']) >= pd.to_datetime(start_date)) &
        (pd.to_datetime(trades_df['timestamp']) <= pd.to_datetime(end_date))
    ]
    st.dataframe(filtered)

