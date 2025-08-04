import os
import boto3
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime
from botocore.exceptions import ClientError
from scripts.utils.utils import project_path

load_dotenv(dotenv_path=os.path.join(project_path(), ".env"))
load_dotenv(dotenv_path=os.path.join(project_path(), ".paths", "paths.env"))

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client("s3")

def upload_inference_result_to_s3(df: pd.DataFrame):
    try:
        if df.empty:
            print("Inference result empty. Uploaded none.")
            return

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        s3_dir = f"inference/{now.strftime('%Y/%m/%d')}"
        s3_key = f"{s3_dir}/recommend_{timestamp}.parquet"
        local_path = f"/tmp/recommend_{timestamp}.parquet"

        df.to_parquet(local_path, index=False)
        print(f"Data saved locally to {local_path}")

        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        print(f"Successfully uploaded to s3://{S3_BUCKET_NAME}/{s3_key}")

        os.remove(local_path)
        print("Removed local temporary file")

    except ClientError as e:
        print(f"S3 Client Error: {e}")
    except Exception as e:
        print(f"An error occurred during export or upload: {e}")
