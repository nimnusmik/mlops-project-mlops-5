import os
import json
import time

import requests
from dotenv import load_dotenv 

load_dotenv()


class TMDBCrawler:
    def __init__(
        self, region="KR",
        language="ko-KR",
        image_language="ko",
        request_interval_seconds=0.4
    ):
        self._base_url = os.environ.get("TMDB_BASE_URL", "https://api.themoviedb.org/3")
        self._api_key = os.environ.get("TMDB_API_KEY")
        self._region = region  
        self._language = language   
        self._image_language = image_language  
        self._request_interval_seconds = request_interval_seconds 

    def get_popular_movies(self, page):

        params = {
            "api_key": self._api_key,  
            "language": self._language, 
            "region": self._region,      
            "page": page                 
        }

        response = requests.get(f"{self._base_url}/movie/popular", params=params)

        if not response.status_code == 200:
            print(f"Error fetching popular movies: Status Code {response.status_code}, Response: {response.text}")
            return [] 

        return json.loads(response.text)["results"]

    def get_bulk_popular_movies(self, start_page, end_page):

        movies = []

        for page in range(start_page, end_page + 1):
            print(f"Fetching popular movies page {page}...")
            page_movies = self.get_popular_movies(page)  
            if page_movies: 
                movies.extend(page_movies) 
            time.sleep(self._request_interval_seconds)  

        return movies

    @staticmethod
    def save_movies_to_json_file(movies, dst="./result", filename="popular_movies"):
        os.makedirs(dst, exist_ok=True)
        data = {"movies": movies}  
        filepath = os.path.join(dst, f"{filename}.json") 

        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Movies saved to {filepath}")

