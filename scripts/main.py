
import os
import sys
import fire
import mlflow
import numpy as np
from datetime import datetime 

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
from mlflow import MlflowClient
from icecream import ic

from scripts.utils.utils import init_seed, project_path, auto_increment_run_suffix
from scripts.utils.logger import Logger  
from scripts.data_prepare.crawler import TMDBCrawler
from scripts.data_prepare.preprocessing import TMDBPreProcessor
from scripts.data_prepare.save_to_db import save_csv_to_db_main_function
from scripts.data_prepare.s3_exporter import export_and_upload_data_to_s3
from scripts.dataset.watch_log import get_datasets
from scripts.dataset.data_loader import SimpleDataLoader
from scripts.model.movie_predictor import MoviePredictor, model_save
from scripts.train.train import train
from scripts.evaluate.evaluate import evaluate
from scripts.utils.enums import ModelTypes
from scripts.inference.inference import (
    inference, recommend_to_df, load_checkpoint, init_model
)
from scripts.postprocess.inference_to_db import write_db
from scripts.postprocess.inference_to_local import save_inference_to_local
from scripts.postprocess.inference_to_s3 import upload_inference_result_to_s3
from scripts.utils.utils import save_hash, read_hash, model_dir, calculate_hash  


# .env 파일과 .paths/paths.env 파일을 모두 로드
env_path = os.path.join(project_path(), '.env')
paths_env_path = os.path.join(project_path(), '.paths', 'paths.env')
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=paths_env_path)

os.environ["DATA_RAW_DIR"] = os.path.join(project_path(), os.getenv("DATA_RAW_DIR", "data/raw"))
mlflow.set_tracking_uri(f"file:{os.path.join(project_path(), 'logs', 'mlflow')}")


def run_popular_movie_pipeline(logger):
    logger.write("\n--- TMDB 인기 영화 크롤링 시작 ---")  
    tmdb_crawler = TMDBCrawler(logger=logger)
    crawled_movies = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=10)

    if not crawled_movies:
        logger.write("[ERROR] 크롤링된 영화 데이터가 없습니다. 파이프라인을 종료합니다.", print_error=True)  
        return

    tmdb_crawler.save_movies_to_json_file(crawled_movies, "popular_movies_raw")
    logger.write(f"총 {len(crawled_movies)}개의 영화 데이터를 크롤링하여 저장했습니다.")  

    logger.write("\n--- 영화 데이터 전처리 및 시청 로그 생성 시작 ---") 
    tmdb_preprocessor = TMDBPreProcessor(movies=crawled_movies, logger=logger)
    tmdb_preprocessor.run()
    tmdb_preprocessor.save("watch_log")

    watch_log_csv_path = os.path.join(os.environ["DATA_RAW_DIR"], "watch_log.csv")
    save_csv_to_db_main_function(watch_log_csv_path, logger) 
    export_and_upload_data_to_s3(logger)
    logger.write("\n--- 모든 파이프라인 실행 완료 ---")  


def get_next_run_name(experiment_name, base_name="movie-predictor", pad=3):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return f"{base_name}-000"

    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    if not runs:
        return f"{base_name}-000"

    latest_run_name = runs[0].data.tags.get("mlflow.runName", f"{base_name}-000")
    return auto_increment_run_suffix(latest_run_name, pad=pad)

 
def run_train(model_name, batch_size=16, dim=256, num_epochs=500, logger=None):
    _logger = logger if logger else sys.stdout
    _logger.write(f"모델 학습 시작: {model_name} (배치 크기={batch_size}, 차원={dim}, 에포크={num_epochs})")  

    init_seed()
    ModelTypes.validation(model_name)
    model_class = ModelTypes[model_name.upper()].value

    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = SimpleDataLoader(train_dataset.features, train_dataset.labels, logger=_logger, batch_size=batch_size, shuffle=True)
    val_loader = SimpleDataLoader(val_dataset.features, val_dataset.labels, logger=_logger, batch_size=batch_size, shuffle=False)
    test_loader = SimpleDataLoader(test_dataset.features, test_dataset.labels, logger=_logger, batch_size=batch_size, shuffle=False)

    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": train_dataset.num_classes,
        "hidden_dim": dim
    }
    model = model_class(**model_params)

    experiment_name = model_name.replace("_", "-")
    mlflow.set_experiment(experiment_name)
    next_run_name = get_next_run_name(experiment_name, base_name=experiment_name)

    with mlflow.start_run(run_name=next_run_name):
        mlflow.log_params({
            "model_name": model_name,
            "batch_size": batch_size,
            "hidden_dim": dim,
            "num_epochs": num_epochs
        })
        _logger.write("MLflow 런 시작. 파라미터 로깅 중.") 

        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, _logger)  
            val_loss, _, val_acc = evaluate(model, val_loader, _logger) 
            test_loss, predictions, test_acc = evaluate(model, test_loader, _logger)  

            _logger.write(f"에포크 {epoch + 1}/{num_epochs}: 학습 손실={train_loss:.4f}, 검증 손실={val_loss:.4f}, 검증 정확도={val_acc:.4f}, 테스트 손실={test_loss:.4f}, 테스트 정확도={test_acc:.4f}") # 한글 메시지
            #ic(f"Epoch {epoch + 1}/{num_epochs}")
            #ic(train_loss, val_loss, val_acc, test_loss, test_acc)

            mlflow.log_metric("Loss/Train", train_loss, step=epoch)
            mlflow.log_metric("Loss/Valid", val_loss, step=epoch)
            mlflow.log_metric("Accuracy/Valid", val_acc, step=epoch)
            mlflow.log_metric("Loss/Test", test_loss, step=epoch)
            mlflow.log_metric("Accuracy/Test", test_acc, step=epoch)

        save_path = model_save(
            model=model,
            model_params=model_params,
            epoch=num_epochs,
            loss=train_loss,
            scaler=train_dataset.scaler,
            label_encoder=train_dataset.label_encoder,
            logger=_logger
        )
        mlflow.log_artifact(save_path)
        _logger.write(f"MLflow에 모델 아티팩트 로깅됨: {save_path}")  

        decoded_predictions = [train_dataset.decode_content_id(idx) for idx in predictions]
        _logger.write(f"디코딩된 예측 결과: {decoded_predictions[:5]}...")  
        #ic(decoded_predictions)

 
def run_inference(data=None, batch_size=16, logger=None):
    _logger = logger if logger else sys.stdout 
    _logger.write("추론 파이프라인 시작...")  

    checkpoint = load_checkpoint(_logger)
    model, scaler, label_encoder = init_model(checkpoint, _logger)

    data = np.array(data or [])
    recommend = inference(model, scaler, label_encoder, data, _logger, batch_size)
    _logger.write(f"추론 결과: {recommend}")  

    recommend_df = recommend_to_df(recommend)
    write_db(recommend_df, os.environ["DB_NAME"], "recommend", _logger) 
    save_inference_to_local(recommend_df, model_name="movie_predictor", logger=_logger)
    upload_inference_result_to_s3(recommend_df, logger=_logger)  
    _logger.write("추론 결과가 DB, 로컬, S3에 저장되었습니다.")  


def run_all_data_pipeline(model_name, batch_size=16, dim=256, num_epochs=500): 
    # 단계별 로그를 위한 폴더 생성 및 로거 초기화
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 데이터 준비(pipeline) 로그
    log_dir_pipeline = os.path.join(project_path(), os.getenv("LOGS_PIPELINE_DIR", "logs/scripts/pipeline"))
    os.makedirs(log_dir_pipeline, exist_ok=True)
    logger_pipeline = Logger(os.path.join(log_dir_pipeline, f'main_pipeline_{timestamp}.log'), print_also=True)
    
    logger_pipeline.write("--- 인기 영화 파이프라인 실행 중... ---")  
    run_popular_movie_pipeline(logger_pipeline)  
    logger_pipeline.close()
    
    # 2. 학습(train) 로그
    log_dir_train = os.path.join(project_path(), os.getenv("LOGS_TRAIN_DIR", "logs/scripts/train"))
    os.makedirs(log_dir_train, exist_ok=True)
    logger_train = Logger(os.path.join(log_dir_train, f'train_run_{timestamp}.log'), print_also=True)
    
    logger_train.write("--- 학습 파이프라인 실행 중... ---")  
    run_train(model_name, batch_size, dim, num_epochs, logger_train) 
    logger_train.close()
    
    # 3. 추론(inference) 로그
    log_dir_inference = os.path.join(project_path(), os.getenv("LOGS_INFERENCE_DIR", "logs/scripts/inference"))
    os.makedirs(log_dir_inference, exist_ok=True)
    logger_inference = Logger(os.path.join(log_dir_inference, f'inference_{timestamp}.log'), print_also=True)

    logger_inference.write("--- 추론 파이프라인 실행 중... ---")  
    run_inference(batch_size=batch_size, logger=logger_inference)
    logger_inference.close()


if __name__ == '__main__':
    # 메인 로거는 전체 실행 흐름만 기록
    log_dir = os.path.join(project_path(), os.getenv("LOGS_SCRIPTS_DIR", "logs/scripts"))
    log_filename = datetime.now().strftime('main_workflow_%Y%m%d_%H%M%S.log')
    log_file_path = os.path.join(log_dir, log_filename)

    logger = Logger(log_file_path, print_also=True)

    try:
        logger.start_redirect()  
        logger.write("\n--- 메인 파이프라인 실행 시작 ---") 
        
        fire.Fire({
            "prepare-data": lambda: run_popular_movie_pipeline(logger),
            "train": lambda model_name, batch_size=16, dim=256, num_epochs=500: \
                     run_train(model_name, batch_size, dim, num_epochs, logger),
            "inference": lambda data=None, batch_size=16: run_inference(data, batch_size, logger),
            "all": lambda model_name, batch_size=16, dim=256, num_epochs=500: \
                   run_all_data_pipeline(model_name, batch_size, dim, num_epochs)
        })
        
        logger.write("\n--- 메인 파이프라인 실행 완료 ---")  

    except Exception as e:
        logger.write(f"[ERROR] 메인 파이프라인 실행 중 오류 발생: {e}", print_error=True)  

    finally:
        logger.stop_redirect()  
        logger.close()