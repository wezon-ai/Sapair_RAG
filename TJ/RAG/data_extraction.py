import psycopg2
import pandas as pd
import os

# PostgreSQL 데이터베이스 연결 정보
db_config = {
    'host': "sapair-rds.cp622qgow5ff.ap-northeast-2.rds.amazonaws.com",
    'port': 5432,
    'dbname': 'sapair_dev',
    'user': 'sapair_user',  # PostgreSQL 사용자 이름
    'password': "sapair!7033"  # PostgreSQL 비밀번호
}

# CSV 파일 저장 경로 설정
output_dir = "wezontest_csv"
os.makedirs(output_dir, exist_ok=True)

try:
    # PostgreSQL 데이터베이스 연결
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # wezontest 스키마의 모든 테이블 이름 가져오기
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'wezontest';
    """)
    tables = cursor.fetchall()

    # 각 테이블 데이터를 CSV로 저장
    for table_name in tables:
        table_name = table_name[0]
        print(f"Exporting table: {table_name}")

        # 테이블 데이터를 DataFrame으로 가져오기
        query = f"SELECT * FROM wezontest.{table_name};"
        df = pd.read_sql_query(query, conn)

        # 테이블 데이터를 CSV로 저장
        csv_file_path = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        print(f"Saved {table_name} to {csv_file_path}")

except Exception as e:
    print(f"Error: {e}")

finally:
    # 연결 닫기
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()
