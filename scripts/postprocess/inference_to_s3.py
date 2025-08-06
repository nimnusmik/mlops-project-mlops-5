import os
import boto3
import pandas as pd
import sys 

from dotenv import load_dotenv
from datetime import datetime
from botocore.exceptions import ClientError
from scripts.utils.utils import project_path

load_dotenv(dotenv_path=os.path.join(project_path(), ".env"))
load_dotenv(dotenv_path=os.path.join(project_path(), ".paths", "paths.env"))

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client("s3")

def upload_inference_result_to_s3(df: pd.DataFrame, logger=None): 
    _logger = logger if logger else sys.stdout

    try:
        if not S3_BUCKET_NAME:
            _logger.write("[ERROR] S3_BUCKET_NAME이 설정되지 않았습니다. S3에 추론 결과를 업로드할 수 없습니다.", print_error=True) 
            return

        if df.empty:
            _logger.write("[ERROR] 추론 결과 데이터프레임이 비어 있습니다. 업로드할 내용이 없습니다.", print_also=True) 
            return

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        s3_dir = f"inference/{now.strftime('%Y/%m/%d')}"
        s3_key = f"{s3_dir}/recommend_{timestamp}.parquet"
        local_path = f"/tmp/recommend_{timestamp}.parquet"

        df.to_parquet(local_path, index=False)
        _logger.write(f"추론 결과 데이터가 로컬에 임시 저장되었습니다: {local_path}")  

        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        _logger.write(f"S3에 성공적으로 업로드되었습니다: s3://{S3_BUCKET_NAME}/{s3_key}") 

        os.remove(local_path)
        _logger.write("로컬 임시 파일이 제거되었습니다.") 

    except ClientError as e:
        _logger.write(f"[ERROR] S3 클라이언트 오류 발생: {e}", print_error=True)  
    except Exception as e:
        _logger.write(f"[ERROR] 추론 결과 업로드 중 알 수 없는 오류 발생: {e}", print_error=True)  
