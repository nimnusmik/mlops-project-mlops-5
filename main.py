import pandas as pd 
from dotenv import load_dotenv

from crawler import TMDBCrawler
from preprocessing import TMDBPreProcessor

load_dotenv()

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

    tmdb_preprocessor.save("watch_log")
    print("시청 로그 생성이 완료되었습니다.")


if __name__ == '__main__':
    run_popular_movie_pipeline()

