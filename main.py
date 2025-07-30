# ~/my-mlops-project/main.py 내용 (수정될 부분)
import pandas as pd
from dotenv import load_dotenv

from crawler import TMDBCrawler
from preprocessing import TMDBPreProcessor
from save_to_db import save_csv_to_db_main_function # 이 줄을 추가합니다.

load_dotenv()

# CSV 파일 경로 정의 (main.py에서 사용)
CSV_FILE_PATH = "./result/watch_log.csv"

def run_popular_movie_pipeline():
    tmdb_crawler = TMDBCrawler()

    print("\n--- TMDB 인기 영화 크롤링 시작 ---")
    crawled_movies = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=10)

    if not crawled_movies:
        print("크롤링된 영화 데이터가 없습니다. 파이프라인을 종료합니다.")
        return

    tmdb_crawler.save_movies_to_json_file(crawled_movies, "./result", "popular_movies_raw")
    print(f"총 {len(crawled_movies)}개의 영화 데이터를 크롤링하여 저장했습니다.")

    print("\n--- 영화 데이터 전처리 및 시청 로그 생성 시작 ---")
    tmdb_preprocessor = TMDBPreProcessor(crawled_movies)
    tmdb_preprocessor.run()

    tmdb_preprocessor.save("watch_log") # 이 부분에서 watch_log.csv가 생성됩니다.
    print("시청 로그 생성이 완료되었습니다.")

    # --- DB 저장 로직 호출 추가 ---
    save_csv_to_db_main_function(CSV_FILE_PATH) # 이 함수를 호출합니다.


if __name__ == '__main__':
    run_popular_movie_pipeline()
