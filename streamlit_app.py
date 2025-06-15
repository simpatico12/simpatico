import sqlite3
import pandas as pd
import streamlit as st

def get_db_columns(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    return [col[1] for col in columns]

def load_reflection_history(conn, trading_cols):
    # 실제 존재하는 컬럼으로 쿼리 작성
    select_cols = ["r.*"]
    for col in ["asset_name", "asset_type", "timestamp"]:
        if col in trading_cols:
            select_cols.append(f"t.{col}")
        else:
            st.warning(f"⚠️ trading_history 테이블에 '{col}' 컬럼 없음 → 쿼리에서 제외")

    select_clause = ", ".join(select_cols)
    query = f"""
        SELECT {select_clause}
        FROM trading_reflection r
        JOIN trading_history t ON r.trading_id = t.id
        ORDER BY r.reflection_date DESC
    """

    st.code(query, language="sql")
    
    df = pd.read_sql_query(query, conn)
    return df

def main():
    st.title("📈 AI 코인/미국/일본 자동매매 대시보드")

    conn = sqlite3.connect("trading.db")

    # 날짜 필터
    start_date = st.date_input("시작 날짜", pd.to_datetime("2025-01-01"))
    end_date = st.date_input("종료 날짜", pd.to_datetime("2025-06-15"))

    # 자산 유형 필터
    asset_filter = st.multiselect("자산 유형", ["coin", "us", "japan"], default=["coin", "us", "japan"])

    try:
        trading_cols = get_db_columns(conn, "trading_history")
        df = load_reflection_history(conn, trading_cols)

        # 필터링
        if not df.empty:
            df["reflection_date"] = pd.to_datetime(df["reflection_date"], errors="coerce")
            df = df[(df["reflection_date"] >= pd.to_datetime(start_date)) & (df["reflection_date"] <= pd.to_datetime(end_date))]
            if "asset_type" in df.columns:
                df = df[df["asset_type"].isin(asset_filter)]

            st.subheader("거래 기록")
            if df.empty:
                st.warning("선택한 조건에 맞는 거래 기록이 없습니다.")
            else:
                st.dataframe(df)

        else:
            st.warning("데이터가 없습니다.")

    except Exception as e:
        st.error(f"❌ 쿼리 실행 중 에러: {e}")

    conn.close()

if __name__ == "__main__":
    main()
