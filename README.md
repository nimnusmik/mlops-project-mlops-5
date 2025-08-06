# 2025-08-06 update 

주요 업데이트
* 도커 컨테이너 및 네트워크 네이밍 통일
> PostgreSQL DB   :   `my-mlops-db`   
> 모델 학습 및 추론 :   `my-mlops-model`    
> FastAPI         :   `my-mlops-api`    
> React           :   `my-mlops-frontend`   
> 네트워크         :   `my-mlops-network`   
> → 앞으로 도커컴포즈파일 작성시 (특히 네트워크) 위 네이밍 사용하시면 됩니다.

* FastAPI 도커 엔드포인트 변경사항
> `/predict` 에 title, poster_url 추가    
> `/predict/batch` 삭제   
> `/latest-recommendations` 기본값 10으로 변경    
> `/available-contents` 추가    

* React 추가
> 초기화면은 `/latest-recommendations` 표시   
> `POST /predict` 됨에 따라 해당 영화가 추가 제공

---

## 모델 학습 및 추론
* 업데이트 버젼: v1.4.3
* 역할
> TMDB 데이터 다운로드      
> 가상유저데이터 생성       
> 모델 학습 및 추론 (학습평가기준: Cross-entropy Loss, Accuracy)    
> 추론 결과 저장 및 PostgreSQL, S3에 업로드     


## FastAPI 
* 업데이트 버젼: v1.3.3
* 역할: FastAPI 서버 실행 및 추천 결과 제공 (영화 제목 및 포스터)
* 주요 엔드포인트

| Method | Endpoint                       | 설명                                             |
| ------ | ------------------------------ | ------------------------------------------------ |
| GET    | `http://localhost:8000/docs`   | Swagger UI 인터페이스                            |
| GET    | `/health`                      | 서버 상태 확인 (헬스체크)                        |
| GET    | `/available-content-ids`       | 추천 가능한(학습된) 콘텐츠 ID 목록 조회          |
| GET    | `/available-contents`          | 추천 가능한(학습된) 콘텐츠 ID, 제목, 포스터 조회 |
| GET    | `/latest-recommendations?k=10` | 가장 최근 추천 결과 k개 조회 (기본 10개)         |
| POST   | `/predict`                     | 단일 사용자 입력에 대한 콘텐츠 추천              |


## React
* 업데이트 버젼: v1.1.1
* 역할: FastAPI 결과를 React서버를 이용하여 사용자에게 제공 (영화 타이틀 및 포스터)
> 기본적으로 첫 화면에 latest-recommendations 제공    
> `POST /predict` 됨에 따라 해당 영화가 추가 제공

---




* 파이프라인 실행 방법

```bash
# '모델 학습 및 추론', 'FastAPI', 'React'에 해당하는 이미지를 다운로드
docker pull jkim1209/mlops-model:1.4.3
docker pull jkim1209/mlops-api:1.3.3
docker pull jkim1209/mlops-frontend:1.1.1

# 현재 로컬 디렉토리에 .env파일 준비
# .env 파일은 슬랙으로 공유드렸습니다.

# PostgreSQL 네트워크 생성 (처음 한 번만 필요)
docker network create my-mlops-network

# PostgreSQL DB 생성
docker run -d \
  --name my-mlops-db \
  --network my-mlops-network \
  -e POSTGRES_DB=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=root \
  -p 5432:5432 \
  postgres:13

# 모델 학습 및 추론 컨테이너 실행 (실행 후 컨테이너는 중지)
docker run -it \
  --name my-mlops-model \
  --network my-mlops-network \
  --env-file .env \
  jkim1209/mlops-model:1.4.3 \
  python scripts/main.py all movie_predictor

# FastAPI 컨테이너 실행 (백그라운드)
docker run -it -d \
  --name my-mlops-api \
  --network my-mlops-network \
  -p 8000:8000 \
  --env-file .env \
  jkim1209/mlops-api:1.3.3

# React 컨테이너 실행 (백그라운드)
docker run -it -d \
  --name my-mlops-frontend \
  --network my-mlops-network \
  -p 3000:3000 \
  jkim1209/mlops-frontend:1.1.1
```

## 주요 포트
> PostgreSQL: `5432`    
> FastAPI : `8000`  
> React : `3000` 