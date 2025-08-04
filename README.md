# 2025-08-04 update 
## Inference
* 업데이트 버젼: v1.2.5
* 역할
    1. TMDB 데이터 다운로드
    2. 가상유저데이터 생성
    3. 모델 학습 및 추론
        * 학습평가기준: Cross-entropy Loss, Accuracy
        * Contents ID 추천
    4. 추론 결과 저장 및 PostgreSQL, S3에 업로드 
* 사용방법
```bash
# 이미지 다운로드
docker pull jkim1209/mlops-project:1.2.5

# PostgreSQL 네트워크 생성 (처음 한 번만 필요)
docker network create mlops-net

# PostgreSQL 컨테이너 실행
docker run -d \
  --name my-mlops-db \
  --network mlops-net \
  -e POSTGRES_DB=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=root \
  -p 5432:5432 \
  postgres:13

# 현재 로컬 디렉토리에 .env파일 준비
# .env 파일은 슬랙으로 공유드렸습니다.

# 모델 추론 컨테이너 실행
docker run -it --rm \
  --name mlops-pipeline-runner \
  --network mlops-net \
  --env-file .env \
  jkim1209/mlops-project:1.2.5 \
  python scripts/main.py all movie_predictor

```

## API
* 업데이트 버젼: v1.0.2
* 역할: API 서버 실행
* 사용방법
```bash
# 주의: 반드시 위 추론 이미지를 한 번 이상 실행시킨 후여야 하며, my-mlops-db 컨테이너가 종료되지 않고 실행되고 있어야 합니다. 또한 위 컨테이너에서 이용했던 .env파일이 그대로 디렉토리에 존재해야 합니다.

# 이미지 다운로드
docker pull jkim1209/mlops-api:1.0.2

# API 서버 실행
docker run -d \
  --name mlops-api \
  -p 8000:8000 \
  --network mlops-net \
  jkim1209/mlops-api:1.0.2
```

* 주요 엔드포인트

| Method | Endpoint                      | 설명                                       |
| ------ | ----------------------------- | ------------------------------------------ |
| GET    | `http://localhost:8000/docs`  | Swagger UI 인터페이스                      |
| GET    | `/health`                     | 서버 상태 확인 (헬스체크)                  |
| GET    | `/available-content-ids`      | 추천 가능한(학습된) 콘텐츠 ID 목록 조회    |
| GET    | `/latest-recommendations?k=5` | 가장 최근 추천 결과 k개 조회 (기본 5개)    |
| POST   | `/predict`                    | 단일 사용자 입력에 대한 콘텐츠 추천        |
| POST   | `/predict/batch`              | 복수 사용자 입력(batch)에 대한 콘텐츠 추천 |

