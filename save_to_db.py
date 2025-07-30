# ~/my-mlops-project/save_to_db.py
import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from io import StringIO
import csv

# .env에서 DB 연결 정보 불러오기
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "my-mlops-db")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")

def save_csv_to_db_main_function(csv_path):
    print(f"\n--- 시청 로그 데이터를 DB에 저장 시작 ---")
    conn = None
    cur = None
    try:
        # 1. CSV 파일 존재 확인
        if not os.path.exists(csv_path):
            print(f"오류: CSV 파일 '{csv_path}'을(를) 찾을 수 없습니다.")
            return

        # 2. CSV 파일 읽기
        df = pd.read_csv(csv_path)
        print(f"'{csv_path}' 파일에서 {len(df)}개의 레코드를 읽었습니다.")

        # 3. PostgreSQL 연결
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        print("PostgreSQL 데이터베이스에 연결되었습니다.")

        # 4. 테이블 생성 (없으면 생성)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS watch_logs (
                user_id INTEGER,
                content_id INTEGER,
                watch_seconds INTEGER,
                rating REAL,
                popularity REAL,
                title TEXT,
                original_title TEXT,
                overview TEXT,
                release_date DATE,
                genre_ids TEXT,
                original_language TEXT,
                backdrop_path TEXT,
                poster_path TEXT,
                vote_count INTEGER,
                vote_average REAL,
                adult BOOLEAN,
                video BOOLEAN,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print("테이블 'watch_logs' 확인 또는 생성 완료.")

        # 5. 데이터 프레임을 CSV 형식으로 메모리 버퍼에 저장 (모든 필드에 따옴표)
        buffer = StringIO()
        df.to_csv(
            buffer,
            index=False,
            header=False,
            sep=',',
            quotechar='"',
            quoting=csv.QUOTE_ALL
        )
        buffer.seek(0)

        # 6. 컬럼 리스트 정의
        csv_columns = [
            'user_id', 'content_id', 'watch_seconds', 'rating', 'popularity',
            'title', 'original_title', 'overview', 'release_date', 'genre_ids',
            'original_language', 'backdrop_path', 'poster_path', 'vote_count',
            'vote_average', 'adult', 'video'
        ]

        # 7. CSV 형식으로 PostgreSQL에 COPY
        copy_sql = f"""
        COPY watch_logs ({', '.join(csv_columns)})
        FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER ',', QUOTE '"');
        """
        cur.copy_expert(copy_sql, buffer)

        conn.commit()
        print(f"총 {len(df)}개의 시청 로그 데이터가 'watch_logs' 테이블에 성공적으로 저장되었습니다.")

    except psycopg2.Error as db_err:
        print(f"데이터베이스 오류 발생: {db_err}")
        if conn:
            conn.rollback()
    except pd.errors.EmptyDataError:
        print(f"오류: '{csv_path}' 파일이 비어 있습니다.")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        print(f"--- 시청 로그 데이터를 DB에 저장 종료 ---")
