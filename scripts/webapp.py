import os
import sys
import uvicorn
import numpy as np
from datetime import datetime 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
from sqlalchemy import text, create_engine 

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from scripts.utils.logger import Logger 
from scripts.inference.inference import (
    load_checkpoint, init_model, inference, recommend_to_df
)
from scripts.postprocess.inference_to_db import write_db, read_db, get_movie_metadata_by_ids, get_engine 

load_dotenv()

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.getenv("LOGS_WEBAPP_DIR", "logs/webapp"))
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime('webapp_pipeline_%Y%m%d_%H%M%S.log')
log_file_path = os.path.join(log_dir, log_filename)
webapp_logger = Logger(log_file_path, print_also=True) 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

try:
    webapp_logger.write("모델 체크포인트 로드 중...")
    checkpoint = load_checkpoint(webapp_logger) 
    webapp_logger.write("모델 초기화 중...")
    model, scaler, label_encoder = init_model(checkpoint, webapp_logger) 
    webapp_logger.write("모델 로드 및 초기화 완료.")
except Exception as e:
    webapp_logger.write(f"[ERROR] 모델 로드 및 초기화 실패: {e}", print_error=True)
    raise RuntimeError(f"Failed to load model: {e}")

class InferenceInput(BaseModel):
    user_id: int
    content_id: int
    watch_seconds: int
    rating: float
    popularity: float

# POST /predict
@app.post("/predict")
async def predict(input_data: InferenceInput):
    webapp_logger.write(f"추론 요청 수신: User ID={input_data.user_id}, Content ID={input_data.content_id}")
    try:
        # 1. 추천 수행
        data = np.array([
            input_data.user_id,
            input_data.content_id,
            input_data.watch_seconds,
            input_data.rating,
            input_data.popularity
        ])
        result = inference(model, scaler, label_encoder, data, webapp_logger)
        webapp_logger.write(f"추론 완료. 추천 결과: {result}")

        # 2. 추천 결과 DB 저장
        df_to_save = recommend_to_df(result)
        write_db(df_to_save, os.getenv("DB_NAME"), "recommend", webapp_logger)
        webapp_logger.write("추론 결과 DB 저장 완료.")

        # 3. 메타데이터 포함한 결과 리턴
        metadata = get_movie_metadata_by_ids(os.getenv("DB_NAME"), result, webapp_logger)
        recommendations = [
            {
                "content_id": int(cid),
                "title": metadata.get(int(cid), {}).get("title", ""),
                "poster_url": metadata.get(int(cid), {}).get("poster_url", ""),
                "overview": metadata.get(int(cid), {}).get("overview", "")
            }
            for cid in result
        ]
        webapp_logger.write(f"User {input_data.user_id}에게 {len(recommendations)}개의 추천 결과 반환.")

        return {
            "user_id": input_data.user_id,
            "recommendations": recommendations
        }

    except Exception as e:
        webapp_logger.write(f"[ERROR] /predict 요청 처리 중 오류 발생: {e}", print_error=True)
        raise HTTPException(status_code=500, detail="추론 처리 중 오류가 발생했습니다.")


# GET /latest-recommendations
@app.get("/latest-recommendations")
async def latest_recommendations(k: int = 10):
    webapp_logger.write(f"최근 추천 목록 요청 수신: {k}개")
    try:
        content_ids = read_db(os.getenv("DB_NAME"), "recommend", k=k, logger=webapp_logger)
        unique_ids = list(dict.fromkeys(content_ids))
        
        if not unique_ids:
            webapp_logger.write("[ERROR] 최근 추천 목록이 비어 있습니다.", print_also=True)
            return {"recent_recommendations": []}

        metadata = get_movie_metadata_by_ids(os.getenv("DB_NAME"), [int(cid) for cid in unique_ids], webapp_logger)
        webapp_logger.write(f"DB에서 {len(unique_ids)}개의 최근 추천 콘텐츠 ID 메타데이터 조회 완료.")

        recommendations = [
            {
                "content_id": cid,
                "title": metadata.get(int(cid), {}).get("title", ""),
                "poster_url": metadata.get(int(cid), {}).get("poster_url", ""),
                "overview": metadata.get(int(cid), {}).get("overview", "")
            }
            for cid in unique_ids
            if int(cid) in metadata
        ]
        webapp_logger.write(f"{len(recommendations)}개의 최근 추천 목록 반환.")

        return {"recent_recommendations": recommendations}
    except Exception as e:
        webapp_logger.write(f"[ERROR] /latest-recommendations 요청 처리 중 오류 발생: {e}", print_error=True)
        raise HTTPException(status_code=500, detail="최근 추천 목록 조회 중 오류가 발생했습니다.")


# Get /available-content-ids
@app.get("/available-content-ids")
async def available_ids():
    webapp_logger.write("사용 가능한 콘텐츠 ID 목록 요청 수신.")
    try:
        if label_encoder is None:
            webapp_logger.write("[ERROR] LabelEncoder가 초기화되지 않았습니다.", print_error=True)
            raise HTTPException(status_code=500, detail="서버 오류: LabelEncoder가 초기화되지 않았습니다.")
            
        count = len(label_encoder.classes_)
        available_content_ids = label_encoder.classes_.tolist()
        webapp_logger.write(f"사용 가능한 콘텐츠 ID {count}개 반환.")
        return {
            "count": count,
            "available_content_ids": available_content_ids
        }
    except Exception as e:
        webapp_logger.write(f"[ERROR] /available-content-ids 요청 처리 중 오류 발생: {e}", print_error=True)
        raise HTTPException(status_code=500, detail="사용 가능한 콘텐츠 ID 조회 중 오류가 발생했습니다.")


# Get /available-contents
@app.get("/available-contents")
async def available_contents():
    webapp_logger.write("사용 가능한 콘텐츠 목록 요청 수신.")
    try:
        engine = get_engine(os.getenv("DB_NAME"), webapp_logger)
        query = text("""
            SELECT DISTINCT content_id, title, poster_path
            FROM watch_logs
            WHERE poster_path IS NOT NULL AND title IS NOT NULL
            ORDER BY title ASC
        """)
        with engine.connect() as conn:
            result = conn.execute(query).mappings().all() 
            contents = [
                {
                    "content_id": int(row["content_id"]),
                    "title": row["title"],
                    "poster_url": f"https://image.tmdb.org/t/p/original{row['poster_path']}"
                }
                for row in result
            ]
        webapp_logger.write(f"사용 가능한 콘텐츠 {len(contents)}개 반환.")
        return {"available_contents": contents}
    except Exception as e:
        webapp_logger.write(f"[ERROR] /available-contents 요청 처리 중 오류 발생: {e}", print_error=True)
        raise HTTPException(status_code=500, detail="사용 가능한 콘텐츠 조회 중 오류가 발생했습니다.")


# Get /health
@app.get("/health")
async def health_check():
    webapp_logger.write("헬스 체크 요청 수신.")
    try:
        engine = get_engine(os.getenv("DB_NAME"), webapp_logger)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        webapp_logger.write("헬스 체크 완료: DB 연결 정상.")
        return {"status": "ok", "db_connection": "ok"}
    except Exception as e:
        webapp_logger.write(f"[ERROR] 헬스 체크 실패: DB 연결 오류: {e}", print_error=True)
        return {"status": "error", "db_connection": "failed", "detail": str(e)}


# 서버 직접 실행 시
if __name__ == "__main__":
    webapp_logger.start_redirect()
    webapp_logger.write("웹 애플리케이션 서버 시작 중...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        webapp_logger.write(f"[ERROR] 웹 애플리케이션 서버 실행 중 치명적인 오류 발생: {e}", print_error=True)
    finally:
        webapp_logger.write("웹 애플리케이션 서버 종료.")
        webapp_logger.stop_redirect()
        webapp_logger.close()
