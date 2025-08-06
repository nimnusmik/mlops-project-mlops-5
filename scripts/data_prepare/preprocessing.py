import random
import numpy as np
import pandas as pd
import os
import json
import csv
import sys 

from scripts.utils.utils import project_path


class TMDBPreProcessor:
    def __init__(self, movies: list, logger, user_count=300, max_select_count=20):
        random.seed(0)
        self.logger = logger
        self._movies = movies
        self._features = pd.DataFrame()
        self._users = list(range(1, user_count + 1))
        self._max_select_count = max_select_count
        self._max_runtime_seconds = 120 * 60

    @staticmethod
    def augmentation(movie):
        movie_id = movie.get("id")
        rating = movie.get("vote_average", 0.0)
        popularity = movie.get("popularity", 0.0)
        title = movie.get("title", "")
        original_title = movie.get("original_title", "")
        overview = movie.get("overview", "")
        release_date = movie.get("release_date", "")
        genre_ids = movie.get("genre_ids", [])
        original_language = movie.get("original_language", "")
        backdrop_path = movie.get("backdrop_path", "")
        poster_path = movie.get("poster_path", "")
        vote_count = movie.get("vote_count", 0)
        vote_average_val = movie.get("vote_average", 0.0)
        adult = movie.get("adult", False)
        video = movie.get("video", False)
        if movie_id is None:
            print(f"[WARN] Skipping augmentation for movie with no ID: {movie}")
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
            "vote_average": vote_average_val,
            "adult": adult,
            "video": video
        }
        return [data] * count

    def generate_watch_second(self, rating):
        base = 1.1
        noise_level = 0.1
        if not isinstance(rating, (int, float)) or rating <= 0:
            return 0
        normalized_rating = (rating - 0) / (10 - 0)
        base_time_factor = (base ** normalized_rating - base ** 0) / (base ** 1 - base ** 0)
        base_time = self._max_runtime_seconds * base_time_factor
        noise = np.random.normal(0, noise_level * base_time)
        watch_second = base_time + noise
        watch_second = int(np.clip(watch_second, 0, self._max_runtime_seconds))
        return watch_second

    def selection(self, user_id, features):
        select_count = random.randint(0, self._max_select_count)
        if select_count == 0 or not features:
            return []
        k_to_select = min(select_count, len(features))
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
                "vote_average": feature.get("vote_average", 0.0),
                "adult": feature.get("adult", False),
                "video": feature.get("video", False)
            } for feature in selected_features
        ]
        return result

    def run(self):
        features = []
        if not self._movies:
            self.logger.write("[WARN] No movies provided to TMDBPreProcessor. 'run' method will not generate features.")
            self._features = pd.DataFrame()
            return
        self.logger.write(f"Preprocessing {len(self._movies)} movies for feature augmentation...")
        for movie in self._movies:
            features.extend(self.augmentation(movie))
        if not features:
            self.logger.write("[WARN] No features generated after augmentation. Skipping selection.")
            self._features = pd.DataFrame()
            return
        selected_features = []
        self.logger.write(f"Generating watch logs for {len(self._users)} users from {len(features)} augmented features...")
        for user_id in self._users:
            selected_features.extend(self.selection(user_id, features))
        if not selected_features:
            self.logger.write("[WARN] No watch logs generated after selection.")
            self._features = pd.DataFrame()
            return
        df = pd.DataFrame.from_records(selected_features)
        self._features = df
        self.logger.write(f"Successfully generated {len(self._features)} watch log entries.")

    def save(self, filename):
        if not self._features.empty:
            result_dir = os.getenv("DATA_RAW_DIR", os.path.join(project_path(), "data", "raw"))
            os.makedirs(result_dir, exist_ok=True)

            file_path = os.path.join(result_dir, f"{filename}.csv")
            self._features.to_csv(file_path, header=True, index=False, quoting=csv.QUOTE_ALL)
            self.logger.write(f"Successfully saved raw data to {file_path}")
        else:
            self.logger.write(f"[WARN] No features to save for {filename}.")

    @property
    def features(self):
        return self._features
