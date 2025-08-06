import os
import datetime
import pandas as pd
import sys

from scripts.utils.utils import project_path


def save_inference_to_local(df, model_name="model", save_dir="data/processed", logger=None):
    _logger = logger if logger else sys.stdout

    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    filename = f"{model_name}_{timestamp}.csv"
    save_path = os.path.join(project_path(), save_dir, filename)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        df.to_csv(save_path, index=False)
        _logger.write(f"추론 결과가 로컬 경로에 저장되었습니다: {save_path}")  
        return save_path

    except Exception as e:
        _logger.write(f"[ERROR] 추론 결과를 로컬 경로에 저장 실패: {save_path} (오류: {e})", print_error=True)
        return None  