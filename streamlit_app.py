import sqlite3
import pandas as pd

def get_db_columns(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"\n📌 {table_name} 컬럼 정보:")
    for col in columns:
        print(f"- {col[1]} ({col[2]})")
    return [col[1] for col in columns]

def load_reflection_history(conn, trading_cols):
    # 존재하는 컬럼만 선택
    select_cols = ["r.*"]
    for col in ["asset_name", "asset_type", "timestamp"]:
        if col in trading_cols:
            select_cols.append(f"t.{col}")
        else:
            print(f"⚠️ trading_history 테이블에 {col} 컬럼 없음 → 쿼리에서 제외")

    # 쿼리 생성
    select_clause = ", ".join(select_cols)
    query = f"""
        SELECT {select_clause}
        FROM trading_reflection r
        JOIN trading_history t ON r.trading_id = t.id
        ORDER BY r.reflection_date DESC
    """
    print(f"\n✅ 최종 실행 쿼리:\n{query}")
    
    # 쿼리 실행
    df = pd.read_sql_query(query, conn)
    return df

if __name__ == "__main__":
    conn = sqlite3.connect("trading.db")
    
    # 컬럼 정보 확인
    trading_cols = get_db_columns(conn, "trading_history")
    _ = get_db_columns(conn, "trading_reflection")

    # 데이터 로드
    try:
        df = load_reflection_history(conn, trading_cols)
        print("\n📊 데이터 미리보기:")
        print(df.head())
    except Exception as e:
        print(f"\n❌ 쿼리 실행 중 에러: {e}")

    conn.close()

