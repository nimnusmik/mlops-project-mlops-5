import os
import sys
import fire
import mlflow
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
from mlflow import MlflowClient
from icecream import ic

from scripts.utils.utils import init_seed, project_path, auto_increment_run_suffix
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

# 환경변수 로드
env_path = os.path.join(project_path(), '.env')
paths_env_path = os.path.join(project_path(), '.paths', 'paths.env')
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=paths_env_path)

# 데이터 및 mlflow 경로설정
os.environ["DATA_RAW_DIR"] = os.path.join(project_path(), os.getenv("DATA_RAW_DIR", "data/raw"))
mlflow.set_tracking_uri(f"file:{os.path.join(project_path(), 'logs', 'mlflow')}")


def run_popular_movie_pipeline():
    print("\n--- TMDB 인기 영화 크롤링 시작 ---")
    tmdb_crawler = TMDBCrawler()
    crawled_movies = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=10)

    if not crawled_movies:
        print("크롤링된 영화 데이터가 없습니다. 파이프라인을 종료합니다.")
        return

    tmdb_crawler.save_movies_to_json_file(crawled_movies, "popular_movies_raw")
    print(f"총 {len(crawled_movies)}개의 영화 데이터를 크롤링하여 저장했습니다.")

    print("\n--- 영화 데이터 전처리 및 시청 로그 생성 시작 ---")
    tmdb_preprocessor = TMDBPreProcessor(crawled_movies)
    tmdb_preprocessor.run()
    tmdb_preprocessor.save("watch_log")

    watch_log_csv_path = os.path.join(os.environ["DATA_RAW_DIR"], "watch_log.csv")
    save_csv_to_db_main_function(watch_log_csv_path)
    export_and_upload_data_to_s3()
    print("\n--- 모든 파이프라인 실행 완료 ---")


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


def run_train(model_name, batch_size=16, dim=256, num_epochs=100):
    init_seed()
    ModelTypes.validation(model_name)
    model_class = ModelTypes[model_name.upper()].value

    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = SimpleDataLoader(train_dataset.features, train_dataset.labels, batch_size=batch_size, shuffle=True)
    val_loader = SimpleDataLoader(val_dataset.features, val_dataset.labels, batch_size=batch_size, shuffle=False)
    test_loader = SimpleDataLoader(test_dataset.features, test_dataset.labels, batch_size=batch_size, shuffle=False)

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

        for epoch in range(num_epochs):
            train_loss = train(model, train_loader)
            val_loss, _, val_acc = evaluate(model, val_loader)
            test_loss, predictions, test_acc = evaluate(model, test_loader)

            ic(f"Epoch {epoch + 1}/{num_epochs}")
            ic(train_loss, val_loss, val_acc, test_loss, test_acc)

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
        )
        mlflow.log_artifact(save_path)

        decoded_predictions = [train_dataset.decode_content_id(idx) for idx in predictions]
        ic(decoded_predictions)


def run_inference(data=None, batch_size=16):
    checkpoint = load_checkpoint()
    model, scaler, label_encoder = init_model(checkpoint)

    data = np.array(data or [])
    recommend = inference(model, scaler, label_encoder, data, batch_size)
    print("\nInference Result:", recommend)

    recommend_df = recommend_to_df(recommend)
    write_db(recommend_df, os.environ["DB_NAME"], "recommend")
    save_inference_to_local(recommend_df, model_name="movie_predictor")
    upload_inference_result_to_s3(recommend_df)


def run_all_data_pipeline(model_name, batch_size=16, dim=256, num_epochs=100):
    run_popular_movie_pipeline()
    run_train(model_name, batch_size, dim, num_epochs)
    run_inference(batch_size=batch_size)


if __name__ == '__main__':
    fire.Fire({
        "prepare-data": run_popular_movie_pipeline,
        "train": run_train,
        "inference": run_inference,
        "all": run_all_data_pipeline
    })
