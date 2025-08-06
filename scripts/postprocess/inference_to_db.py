import os
import psycopg2
import csv  
import sys 
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime  
from io import StringIO
from sqlalchemy import create_engine, text 

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
DB_PORT = os.environ.get('DB_PORT', '5432') 

# --- 엔진 전역 변수 (한 번만 생성) ---
_engine = None

def get_engine(db_name, logger):  
    global _engine
    if _engine is None:
        database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db_name}"
        try:
            _engine = create_engine(database_url)
            logger.write(f"데이터베이스 엔진 생성 완료: {db_name} on {DB_HOST}:{DB_PORT}")  
        except Exception as e:
            logger.write(f"[ERROR] 데이터베이스 엔진 생성 실패: {e}", print_error=True) 
            raise
    return _engine


def write_db(data: pd.DataFrame, db_name, table_name, logger): 
    engine = get_engine(db_name, logger) 
    conn = None  
    try:
        logger.write(f"테이블 '{table_name}'에 {len(data)}개의 행을 쓰는 중...")  
        with engine.connect() as conn:
            if table_name == "recommend":
                conn.execute(text(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'recommend'
                        ) THEN
                            CREATE TABLE recommend (
                                recommend_content_id INTEGER,
                                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );                             
                        ELSIF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'recommend' AND column_name = 'ingested_at'
                        ) THEN
                            ALTER TABLE recommend
                            ADD COLUMN ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                        END IF;
                    END
                    $$;
                """))
                conn.commit()
                logger.write(f"테이블 '{table_name}' 구조 확인 및 업데이트 완료.") 

        if "ingested_at" not in data.columns:
            data["ingested_at"] = datetime.now()
            logger.write("'ingested_at' 컬럼 추가 및 현재 시간으로 설정.") 

        data.to_sql(table_name, engine, if_exists="append", index=False)
        logger.write(f"총 {len(data)}개의 행이 '{db_name}' 데이터베이스의 '{table_name}' 테이블에 성공적으로 기록되었습니다.") 

    except psycopg2.Error as db_err:
        logger.write(f"[ERROR] 데이터베이스에 쓰는 중 오류 발생: {db_err}", print_error=True)  
        if conn:
            conn.rollback()
    except Exception as e:
        logger.write(f"[ERROR] 테이블 '{table_name}'에 쓰는 중 알 수 없는 오류 발생: {e}", print_error=True) 
    finally:
        if conn and not conn.closed:
            conn.close()
        logger.write(f"테이블 '{table_name}' 쓰기 작업 종료.")  


def read_db(db_name, table_name, k=10, logger=None):  
    engine = get_engine(db_name, logger)  
    content_ids = []
    try:
        logger.write(f"테이블 '{table_name}'에서 최근 {k}개의 추천 콘텐츠 ID를 읽는 중...")  
        query = text(f"""
            SELECT recommend_content_id
            FROM {table_name}
            ORDER BY ingested_at DESC
            LIMIT :k
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"k": k}).scalars().all()
            content_ids = [str(cid) for cid in result]
        logger.write(f"테이블 '{table_name}'에서 {len(content_ids)}개의 추천 콘텐츠 ID 읽기 완료.")  
    except Exception as e:
        logger.write(f"[ERROR] DB에서 최근 추천 목록을 읽는 중 오류 발생: {e}", print_error=True)  
    return content_ids


def get_movie_metadata_by_ids(db_name, content_ids, logger):  
    engine = get_engine(db_name, logger)  
    metadata = {}
    if not content_ids:
        logger.write("[WARN] 메타데이터 조회를 위한 콘텐츠 ID가 없습니다.", print_also=True)  
        return metadata

    try:
        logger.write(f"{len(content_ids)}개의 콘텐츠 ID에 대한 영화 메타데이터 조회 중...")  
        placeholders = ', '.join([f":id_{i}" for i in range(len(content_ids))])
        bind_params = {f"id_{i}": int(cid) for i, cid in enumerate(content_ids)}

        query = text(f"""
            SELECT content_id, title, poster_path, overview
            FROM watch_logs
            WHERE content_id IN ({placeholders})
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, bind_params).mappings().all()
            for row in result:
                metadata[row["content_id"]] = {
                    "title": row["title"],
                    "poster_url": f"https://image.tmdb.org/t/p/original{row['poster_path']}" if row['poster_path'] else None,
                    "overview": row["overview"]
                }
        logger.write(f"총 {len(metadata)}개의 영화 메타데이터 조회 완료.") 
    except Exception as e:
        logger.write(f"[ERROR] 영화 메타데이터 조회 중 오류 발생: {e}", print_error=True)  
    return metadata
