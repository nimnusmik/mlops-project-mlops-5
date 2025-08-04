# 2025-08-04 update 
## Inference
* 업데이트 버젼: v1.2.3
* 이미지 다운로드: `docker pull jkim1209/mlops-project:1.2.3`
* 역할
    1. TMDB 데이터 다운로드
    2. 가상유저데이터 생성
    3. 모델 학습 및 추론
        * 학습평가기준: Cross-entropy Loss, Accuracy
        * Contents ID 추천
    4. 추론 결과 저장
        * `data/processed/` 폴더에 저장
        * PostgreSQL, S3 업로드 
* 사용방법
```bash
# 이미지 다운로드
docker pull jkim1209/mlops-project:1.2.3

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
  jkim1209/mlops-project:latest \
  python scripts/main.py run_all_data_pipeline

```

## API 이미지