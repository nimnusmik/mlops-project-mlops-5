import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
from sqlalchemy import text

from scripts.inference.inference import (
    load_checkpoint, init_model, inference, recommend_to_df
)
from scripts.postprocess.inference_to_db import write_db, read_db, get_movie_metadata_by_ids, get_engine

# FastAPI 앱 정의 및 CORS 설정
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 환경변수 로드
load_dotenv()

# 모델 불러오기 (서버 시작 시 1회)
checkpoint = load_checkpoint()
model, scaler, label_encoder = init_model(checkpoint)

# 요청 데이터 스키마 정의
class InferenceInput(BaseModel):
    user_id: int
    content_id: int
    watch_seconds: int
    rating: float
    popularity: float

# POST /predict
@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        # 1. 추천 수행
        data = np.array([
            input_data.user_id,
            input_data.content_id,
            input_data.watch_seconds,
            input_data.rating,
            input_data.popularity
        ])
        result = inference(model, scaler, label_encoder, data)

        # 2. 추천 결과 DB 저장
        df_to_save = recommend_to_df(result)
        write_db(df_to_save, os.getenv("DB_NAME"), "recommend")

        # 3. 메타데이터 포함한 결과 리턴
        metadata = get_movie_metadata_by_ids(os.getenv("DB_NAME"), result)
        recommendations = [
            {
                "content_id": int(cid),
                "title": metadata.get(int(cid), {}).get("title", ""),
                "poster_url": metadata.get(int(cid), {}).get("poster_url", ""),
                "overview": metadata.get(int(cid), {}).get("overview", "")
            }
            for cid in result
        ]

        return {
            "user_id": input_data.user_id,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GET /latest-recommendations
@app.get("/latest-recommendations")
async def latest_recommendations(k: int = 10):
    try:
        content_ids = read_db(os.getenv("DB_NAME"), "recommend", k=k)
        unique_ids = list(dict.fromkeys(content_ids))
        if not unique_ids:
            return {"recent_recommendations": []}

        metadata = get_movie_metadata_by_ids("postgres", [int(cid) for cid in unique_ids])

        recommendations = [
            {
                "content_id": cid,
                "title": metadata.get(int(cid), {}).get("title", ""),
                "poster_url": metadata.get(int(cid), {}).get("poster_url", ""),
                "overview": metadata.get(int(cid), {}).get("overview", "")
            }
            for cid in unique_ids
            if cid in metadata 
        ]

        return {"recent_recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get /available-content-ids
@app.get("/available-content-ids")
async def available_ids():
    return {
        "count": len(label_encoder.classes_),
        "available_content_ids": label_encoder.classes_.tolist()
    }

# Get /available-contents
@app.get("/available-contents")
async def available_contents():
    try:
        engine = get_engine("postgres")
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
        return {"available_contents": contents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get /health
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# 서버 직접 실행 시
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
