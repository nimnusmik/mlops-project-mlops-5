import os
import boto3

import pandas as pd

from botocore.exceptions import ClientError
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine
from scripts.utils.utils import project_path


# 환경 변수 경로 설정
env_path = os.path.join(project_path(), '.env')
paths_env_path = os.path.join(project_path(), '.paths', 'paths.env')

# 환경 변수 로드
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=paths_env_path)

# DB 설정
DB_HOST = os.getenv("DB_HOST", "my-mlops-db")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_PORT = os.environ.get('DB_PORT', '5432')

# S3 설정
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
RAW_DATA_TABLE_NAME = os.environ.get('RAW_DATA_TABLE_NAME', 'watch_logs')

# PostgreSQL 연결
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
try:
    engine = create_engine(DATABASE_URL)
    print(f"PostgreSQL engine created for {DB_NAME} on {DB_HOST}")
except Exception as e:
    print(f"Error creating PostgreSQL engine: {e}")

s3_client = boto3.client('s3')

def export_and_upload_data_to_s3():
    print(f"Starting data export from {RAW_DATA_TABLE_NAME}...")
    try:
        if not S3_BUCKET_NAME:
            print("Error: S3_BUCKET_NAME is not set. Please check your .env file.")
            return

        df = pd.read_sql_table(RAW_DATA_TABLE_NAME, engine)
        print(f"Successfully read {len(df)} rows from {RAW_DATA_TABLE_NAME}.")

        if df.empty:
            print("No data to export. Exiting.")
            return

        local_parquet_path = f"/tmp/{RAW_DATA_TABLE_NAME}_backup.parquet"
        df.to_parquet(local_parquet_path, index=False)
        print(f"Data saved locally to {local_parquet_path}")

        today = datetime.now()
        s3_key = f"raw_data/{today.strftime('%Y/%m/%d')}/{RAW_DATA_TABLE_NAME}_{today.strftime('%Y%m%d%H%M%S')}.parquet"

        s3_client.upload_file(local_parquet_path, S3_BUCKET_NAME, s3_key)
        print(f"Successfully uploaded {local_parquet_path} to s3://{S3_BUCKET_NAME}/{s3_key}")

        os.remove(local_parquet_path)
        print(f"Removed local temporary file: {local_parquet_path}")

    except ClientError as e:
        print(f"S3 Client Error: {e}")
    except Exception as e:
        print(f"An error occurred during export or upload: {e}")
    finally:
        engine.dispose()
