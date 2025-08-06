import os
import sys
import glob
import pickle

sys.path.append(
    os.path.dirname(
            os.path.dirname(
                    os.path.dirname(
                            os.path.abspath(__file__)
                        )
                )
        )
)

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from scripts.utils.utils import model_dir, calculate_hash, read_hash, save_hash  
from scripts.model.movie_predictor import MoviePredictor
from scripts.dataset.watch_log import WatchLogDataset, get_datasets
from scripts.dataset.data_loader import SimpleDataLoader
from scripts.evaluate.evaluate import evaluate
from scripts.postprocess.inference_to_db import write_db


def load_checkpoint(logger):  
    target_dir = model_dir(MoviePredictor.name)
    models_path = os.path.join(target_dir, "*.pkl")
        
    all_models = sorted(glob.glob(models_path))

    if not all_models:
        logger.write(f"[ERROR] '{target_dir}' 경로에서 모델을 찾을 수 없습니다.", print_error=True)  
        raise FileNotFoundError(f"모델을 찾을 수 없습니다: {target_dir}")
    
    latest_model = all_models[-1]
    logger.write(f"최신 모델 발견: {latest_model}") 

    if model_validation(latest_model, logger):  
        with open(latest_model, "rb") as f:
            checkpoint = pickle.load(f)
        logger.write(f"체크포인트 로드 성공: {latest_model}")  
        return checkpoint
    else:
        logger.write("[ERROR] 해시 불일치: 유효하지 않은 파일입니다.", print_error=True) 
        raise FileExistsError("해시 불일치: 유효하지 않은 파일입니다.")


def init_model(checkpoint, logger):  
    model = MoviePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint.get("scaler", None)
    label_encoder = checkpoint.get("label_encoder", None)
    logger.write("체크포인트로부터 모델 초기화 완료.")  
    return model, scaler, label_encoder


def model_validation(model_path, logger):  
    original_hash = read_hash(model_path,logger)
    current_hash = calculate_hash(model_path)
    if original_hash == current_hash:
        logger.write(f"모델 유효성 검사 성공: {model_path}")  
        return True
    else:
        logger.write(f"[ERROR] 모델 유효성 검사 실패: {model_path} (해시 불일치).", print_error=True)  
        return False
    

def make_inference_df(data):
    columns = "user_id content_id watch_seconds rating popularity".split()
    return pd.DataFrame(
        data=[data],
        columns=columns
    )


def inference(model, scaler, label_encoder, data, logger, batch_size=1): 
    logger.write("추론 프로세스 시작...") 
    if data.size > 0:
        df = make_inference_df(data)
        dataset = WatchLogDataset(df, scaler=scaler, label_encoder=label_encoder)
        logger.write("입력 데이터로 WatchLogDataset 생성 완료.") 
    else:
        _, _, dataset = get_datasets(scaler=scaler, label_encoder=label_encoder)
        logger.write("[ERROR] 추론 데이터가 제공되지 않았습니다. 테스트 데이터셋을 사용합니다.", print_also=True)  

    dataloader = SimpleDataLoader(
        dataset.features, dataset.labels, logger=logger, batch_size=batch_size, shuffle=False
    )
    
    loss, predictions, accuracy = evaluate(model, dataloader, logger) 
    logger.write(f"추론 완료. 손실: {loss:.4f}, 정확도: {accuracy:.4f}")  

    return [dataset.decode_content_id(idx) for idx in predictions]

 
def recommend_to_df(recommend):
    return pd.DataFrame(
        data=recommend,
        columns="recommend_content_id".split()
    )
