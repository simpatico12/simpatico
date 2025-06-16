import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "trades.db"

def load_trades():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    return df

def main():
    st.set_page_config(page_title="AI 자동매매 대시보드", layout="wide")
    st.title("🚀 AI 자동매매 거래 대시보드")

    df = load_trades()

    if df.empty:
        st.warning("거래 데이터가 없습니다.")
        return

    # 자산 종류 필터
    asset_types = df['asset_type'].unique().tolist()
    selected_type = st.selectbox("자산 종류 선택", ["전체"] + asset_types)
    if selected_type != "전체":
        df = df[df['asset_type'] == selected_type]

    # 날짜 필터
    min_date = pd.to_datetime(df['datetime']).min()
    max_date = pd.to_datetime(df['datetime']).max()
    date_range = st.date_input("날짜 범위 선택", [min_date, max_date])
    if len(date_range) == 2:
        df = df[(pd.to_datetime(df['datetime']) >= pd.to_datetime(date_range[0])) &
                (pd.to_datetime(df['datetime']) <= pd.to_datetime(date_range[1]))]

    # 거래 요약
    st.subheader("📌 거래 요약")
    st.write(f"총 거래 수: {len(df)}")
    st.write(f"최근 거래일: {df['datetime'].max()}")

    # 수익률 계산
    df["profit_rate"] = (df["now_price"] - df["avg_price"]) / df["avg_price"]
    df["profit_rate"] = df["profit_rate"].fillna(0)

    # FG/감성/투표 결과 컬럼 있으면 표시
    extra_cols = []
    if "fg_score" in df.columns:
        extra_cols.append("fg_score")
    if "sentiment" in df.columns:
        extra_cols.append("sentiment")
    if "strategy_votes" in df.columns:
        extra_cols.append("strategy_votes")

    if extra_cols:
        st.subheader("🌟 추가 정보")
        st.dataframe(df[["datetime", "asset", "decision", "confidence_score"] + extra_cols])

    # 수익률 히스토리
    st.subheader("💹 수익률 히스토리")
    st.line_chart(df.set_index('datetime')["profit_rate"])

    # 수익률 분포
    st.subheader("📈 수익률 분포")
    fig, ax = plt.subplots()
    ax.hist(df["profit_rate"], bins=20, alpha=0.7)
    ax.set_xlabel("수익률")
    ax.set_ylabel("거래 수")
    st.pyplot(fig)

    # 개별 거래
    st.subheader("📝 개별 거래 데이터")
    st.dataframe(df)

    # 반성 메모
    if "memo" in df.columns:
        st.subheader("🗒️ 반성 메모")
        for _, row in df.iterrows():
            if row["memo"]:
                st.markdown(f"- **{row['datetime']}** | {row['asset']} : {row['memo']}")

if __name__ == "__main__":
    main()

