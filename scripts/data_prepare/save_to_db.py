import os
import psycopg2
import csv

import pandas as pd

from dotenv import load_dotenv
from datetime import datetime  
from io import StringIO
from datetime import datetime
from scripts.utils.utils import project_path


# --- 환경 변수 파일 경로 설정 ---
env_path = os.path.join(project_path(), '.env')
paths_env_path = os.path.join(project_path(), '.paths', 'paths.env')

# --- 환경 변수 로드 ---
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=paths_env_path)

# --- DB 설정값 읽기 ---
DB_HOST = os.getenv("DB_HOST", "my-mlops-db")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")


def save_csv_to_db_main_function(csv_path,logger):
    #로그 파일 경로 설정
    log_dir = os.path.join(project_path(), os.getenv("LOGS_SCRIPTS_DIR", "logs/scripts"))
    log_filename = datetime.now().strftime('db_ingestion_%Y%m%d_%H%M%S.log')
    log_file_path = os.path.join(log_dir, log_filename)
    
    logger = Logger(log_file_path, print_also = True) #콘솔에 출력 하도록

    conn = None
    cur = None
    try:
        logger.write(f"\n--- 시청 로그 데이터를 DB에 저장 시작 ---")

        if not os.path.exists(csv_path):
            logger.write(f"오류: CSV 파일 '{csv_path}'을(를) 찾을 수 없습니다.", print_error=True)
            return

        df = pd.read_csv(csv_path)
        logger.write(f"'{csv_path}' 파일에서 {len(df)}개의 레코드를 읽었습니다.")

        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        print("PostgreSQL 데이터베이스에 연결되었습니다.")

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
                release_date TEXT,
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
        logger.write("테이블 'watch_logs' 확인 또는 생성 완료.")

        buffer = StringIO()
        df.to_csv(
            buffer,
            index=False,
            header=False,
            sep=',',
            quotechar='"',
            quoting=csv.QUOTE_ALL,
        )
        buffer.seek(0)

        csv_columns = [
            'user_id', 'content_id', 'watch_seconds', 'rating', 'popularity',
            'title', 'original_title', 'overview', 'release_date', 'genre_ids',
            'original_language', 'backdrop_path', 'poster_path', 'vote_count',
            'vote_average', 'adult', 'video'
        ]

        copy_sql = f"""
        COPY watch_logs ({', '.join(csv_columns)})
        FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER ',', QUOTE '"', NULL '');
        """
        cur.copy_expert(copy_sql, buffer)

        conn.commit()
        logger.write(f"총 {len(df)}개의 시청 로그 데이터가 'watch_logs' 테이블에 성공적으로 저장되었습니다.")

    except psycopg2.Error as db_err:
        logger.write(f"데이터베이스 오류 발생: {db_err}")
        if conn:
            conn.rollback()
    except pd.errors.EmptyDataError:
        logger.write(f"오류: '{csv_path}' 파일이 비어 있습니다.")
    except Exception as e:
        logger.write(f"알 수 없는 오류 발생: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logger.write(f"--- 시청 로그 데이터를 DB에 저장 종료 ---")