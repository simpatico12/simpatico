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
    st.title("📊 AI 자동매매 거래 대시보드")

    df = load_trades()

    if df.empty:
        st.warning("거래 데이터가 없습니다.")
        return

    # asset_type 선택 필터
    asset_types = df['asset_type'].unique().tolist()
    selected_type = st.selectbox("자산 종류 선택", ["전체"] + asset_types)

    if selected_type != "전체":
        df = df[df['asset_type'] == selected_type]

    # 거래 요약
    st.subheader("📌 거래 요약")
    st.write(f"총 거래 수: {len(df)}")
    st.write(f"최근 거래일: {df['timestamp'].max()}")

    # 수익률 히스토리
    st.subheader("💹 수익률 히스토리")
    df["profit_rate"] = (df["current_price"] - df["avg_price"]) / df["avg_price"]
    df["profit_rate"] = df["profit_rate"].fillna(0)
    st.line_chart(df["profit_rate"])

    # 수익률 분포
    st.subheader("📈 수익률 분포")
    fig, ax = plt.subplots()
    ax.hist(df["profit_rate"], bins=20, alpha=0.7)
    ax.set_xlabel("수익률")
    ax.set_ylabel("거래 수")
    st.pyplot(fig)

    # 개별 거래 데이터
    st.subheader("📝 개별 거래 데이터")
    st.dataframe(df)

    # 반성 메모 (있을 경우)
    if "memo" in df.columns:
        st.subheader("🗒️ 반성 메모")
        for i, row in df.iterrows():
            if row["memo"]:
                st.markdown(f"- {row['timestamp']} | {row['asset']} : {row['memo']}")

if __name__ == "__main__":
    main()
