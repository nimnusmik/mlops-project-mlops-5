import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

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
    content_id: int
    watch_seconds: int
    rating: float
    popularity: float

# POST /predict
@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        data = np.array([
            input_data.content_id,
            input_data.watch_seconds,
            input_data.rating,
            input_data.popularity
        ])
        result = inference(model, scaler, label_encoder, data.reshape(1, -1))
        return {"recommended_content_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GET /latest-recommendations
@app.get("/latest-recommendations")
async def latest_recommendations(k: int = 5):
    try:
        result = read_db("mlops", "recommend", k=k)
        return {"recent_recommend_content_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 서버 직접 실행 시
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

