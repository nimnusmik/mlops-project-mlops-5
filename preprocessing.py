import random
import numpy as np
import pandas as pd
import os
import json

class TMDBPreProcessor:
    def __init__(self, movies: list, user_count=300, max_select_count=20):
        random.seed(0)  
        self._movies = movies
        self._features = pd.DataFrame()
        self._users = list(range(1, user_count+1))
        self._max_select_count = max_select_count
        self._max_runtime_seconds = 120 * 60  

    @staticmethod
    def augmentation(movie):
        
        movie_id = movie.get("id")
         
        rating = movie.get("vote_average", 0.0) 
        popularity = movie.get("popularity", 0.0)  

        # 추가할 영화 정보들
        title = movie.get("title", "")
        original_title = movie.get("original_title", "")
        overview = movie.get("overview", "")
        release_date = movie.get("release_date", "")
        genre_ids = movie.get("genre_ids", []) # 장르 ID 리스트
        original_language = movie.get("original_language", "")
        backdrop_path = movie.get("backdrop_path", "")
        poster_path = movie.get("poster_path", "")
        vote_count = movie.get("vote_count", 0)
        vote_average_val = movie.get("vote_average", 0.0)  
        adult = movie.get("adult", False)
        video = movie.get("video", False)

        if movie_id is None:
            print(f"Warning: Skipping augmentation for movie with no ID: {movie}")
            return []

        count = int(pow(2, max(1, rating)))
        
        data = {
            "content_id": str(movie_id),  
            "rating": rating, 
            "popularity": popularity,
            "title": title,
            "original_title": original_title,
            "overview": overview,
            "release_date": release_date,
            "genre_ids": json.dumps(genre_ids), 
            "original_language": original_language,
            "backdrop_path": backdrop_path,
            "poster_path": poster_path,
            "vote_count": vote_count,
            "vote_average": vote_average_val, # <-- 여기에 vote_average 값을 추가합니다.
            "adult": adult,
            "video": video
        }
        return [data] * count

    def generate_watch_second(self, rating):
        base = 1.1
        noise_level = 0.1
        
        # rating이 유효하지 않거나 0 이하면 최소 시청 시간으로 처리
        if not isinstance(rating, (int, float)) or rating <= 0:
            return 0 # 또는 특정 최소값으로 설정

        # 평점에 따른 기본 시청 시간 계산
        # 정규화된 평점 구간에 따른 시청 시간 곡선을 만들기 위함
        normalized_rating = (rating - 0) / (10 - 0) # 평점을 0-10 기준으로 정규화 (TMDB는 0-10)
        
        # 비선형적인 시청 시간 분포를 만들기 위해 pow 함수 사용
        # 평점이 높을수록 시청 시간이 기하급수적으로 늘어나는 경향을 모방
        base_time_factor = (base ** normalized_rating - base ** 0) / (base ** 1 - base ** 0)
        base_time = self._max_runtime_seconds * base_time_factor

        # 가우시안 노이즈 추가
        noise = np.random.normal(0, noise_level * base_time)
        watch_second = base_time + noise

        # 시청 시간을 정수로 변환하고, 최소 0, 최대 _max_runtime_seconds 범위로 클리핑
        watch_second = int(np.clip(watch_second, 0, self._max_runtime_seconds))
        # print(f"Rating: {rating}, Watch Seconds: {watch_second}") # 디버깅용 출력
        return watch_second

    def selection(self, user_id, features):
        """
        특정 사용자에게 가상의 시청 로그를 생성합니다.
        증강된 특징들 중에서 무작위로 일부를 선택합니다.
        """
        # 선택할 콘텐츠 개수를 무작위로 결정
        select_count = random.randint(0, self._max_select_count)
        # print(f"user [{user_id}] is select [{select_count}] contents") # 디버깅용 출력

        if select_count == 0:
            return []

        # features 리스트가 비어있을 경우 예외 처리
        if not features:
            # print(f"Warning: No features available for selection for user [{user_id}].") # 디버깅용 출력
            return []

        # 주어진 특징들 중에서 무작위로 select_count만큼 선택
        # k는 features 리스트의 길이보다 클 수 없으므로 min(k, len(features)) 사용
        k_to_select = min(select_count, len(features))
        if k_to_select == 0: # 선택할 수 있는 특징이 없는 경우
            return []

        selected_features = random.choices(features, k=k_to_select)

        result = [
            {
                "user_id": str(user_id),
                "content_id": str(feature.get("content_id")),
                "watch_seconds": self.generate_watch_second(feature.get("rating", 0.0)),
                "rating": feature.get("rating", 0.0), 
                "popularity": feature.get("popularity", 0.0),
                "title": feature.get("title", ""),
                "original_title": feature.get("original_title", ""),
                "overview": feature.get("overview", ""),
                "release_date": feature.get("release_date", ""),
                "genre_ids": feature.get("genre_ids", "[]"), 
                "original_language": feature.get("original_language", ""),
                "backdrop_path": feature.get("backdrop_path", ""),
                "poster_path": feature.get("poster_path", ""),
                "vote_count": feature.get("vote_count", 0),
                "vote_average": feature.get("vote_average", 0.0), # <-- 이 줄을 추가하여 별도 컬럼으로 만듭니다.
                "adult": feature.get("adult", False),
                "video": feature.get("video", False)
            } for feature in selected_features
        ]
        return result

    def run(self):
        features = []
         
        if not self._movies:
            print("Warning: No movies provided to TMDBPreProcessor. 'run' method will not generate features.")
            self._features = pd.DataFrame()  
            return

        print(f"Preprocessing {len(self._movies)} movies for feature augmentation...")
        for movie in self._movies:
            features.extend(self.augmentation(movie))

        if not features:  
            print("Warning: No features generated after augmentation. Skipping selection.")
            self._features = pd.DataFrame()  
            return

        selected_features = []
        print(f"Generating watch logs for {len(self._users)} users from {len(features)} augmented features...")
        for user_id in self._users:
            selected_features.extend(self.selection(user_id, features))

        if not selected_features:
            print("Warning: No watch logs generated after selection.")
            self._features = pd.DataFrame() 
            return

        df = pd.DataFrame.from_records(selected_features)
        self._features = df
        print(f"Successfully generated {len(self._features)} watch log entries.")


    def save(self, filename):
        if not self._features.empty:
            result_dir = "./result"
            os.makedirs(result_dir, exist_ok=True)
            
            file_path = os.path.join(result_dir, f"{filename}.csv")
            self._features.to_csv(file_path, header=True, index=False)
            print(f"Successfully saved preprocessed data to {file_path}")
        
        else:
            print(f"No features to save for {filename}.")

    @property
    def features(self):
        return self._features
