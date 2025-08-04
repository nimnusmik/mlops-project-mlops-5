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

from scripts.inference.inference import (
    load_checkpoint, init_model, inference, recommend_to_df
)
from scripts.postprocess.inference_to_db import read_db

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

class InferenceBatchInput(BaseModel):
    batch: List[InferenceInput]


# POST /predict
@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        data = np.array([
            input_data.user_id,
            input_data.content_id,
            input_data.watch_seconds,
            input_data.rating,
            input_data.popularity
        ])
        result = inference(model, scaler, label_encoder, data)
        return {
            "user_id": input_data.user_id,
            "recommended_content_id": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# POST /predict/batch
@app.post("/predict/batch")
async def predict_batch(input_batch: InferenceBatchInput):
    try:
        input_list = input_batch.batch

        data_list = []
        user_ids = []

        for item in input_list:
            user_ids.append(item.user_id)
            data_list.append([
                item.user_id,
                item.content_id,
                item.watch_seconds,
                item.rating,
                item.popularity
            ])

        data = np.array(data_list)

        results = inference(model, scaler, label_encoder, data, batch_size=len(data))

        output = []
        for uid, reco in zip(user_ids, results):
            output.append({
                "user_id": uid,
                "recommended_content_id": reco
            })

        return {"batch_results": output}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

# GET /latest-recommendations
@app.get("/latest-recommendations")
async def latest_recommendations(k: int = 5):
    try:
        result = read_db("mlops", "recommend", k=k)
        return {"recent_recommend_content_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get /available-content-ids
@app.get("/available-content-ids")
async def available_ids():
    return {
        "count": len(label_encoder.classes_),
        "available_content_ids": label_encoder.classes_.tolist()
    }


# Get /health
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# 서버 직접 실행 시
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

