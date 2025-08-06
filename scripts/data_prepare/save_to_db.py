import os
import psycopg2
import csv
import sys
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime
from io import StringIO
from scripts.utils.utils import project_path


# --- Environment variable file paths ---
env_path = os.path.join(project_path(), '.env')
paths_env_path = os.path.join(project_path(), '.paths', 'paths.env')

# --- Load environment variables ---
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=paths_env_path)

# --- Read DB configuration values ---
DB_HOST = os.getenv("DB_HOST", "my-mlops-db")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")


def save_csv_to_db_main_function(csv_path, logger):
    
    conn = None
    cur = None
    try:
        logger.write(f"\n--- Starting to save watch log data to DB ---")

        if not os.path.exists(csv_path):
            logger.write(f"[ERROR] CSV file '{csv_path}' not found.", print_error=True)
            return

        df = pd.read_csv(csv_path)
        logger.write(f"Read {len(df)} records from '{csv_path}' file.")

        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        # This print will be captured by the main.py's logger
        print("Connected to PostgreSQL database.") 

        cur.execute("""
            CREATE TABLE IF NOT EXISTS watch_logs (
                user_id INTEGER,
                content_id INTEGER,
                watch_seconds INTEGER,
                rating REAL,
                popularity REAL,
                title TEXT,
                original_title TEXT,
                overview TEXT,
                release_date TEXT,
                genre_ids TEXT,
                original_language TEXT,
                backdrop_path TEXT,
                poster_path TEXT,
                vote_count INTEGER,
                vote_average REAL,
                adult BOOLEAN,
                video BOOLEAN,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        logger.write("Table 'watch_logs' checked or created successfully.")

        buffer = StringIO()
        df.to_csv(
            buffer,
            index=False,
            header=False,
            sep=',',
            quotechar='"',
            quoting=csv.QUOTE_ALL,
        )
        buffer.seek(0)

        csv_columns = [
            'user_id', 'content_id', 'watch_seconds', 'rating', 'popularity',
            'title', 'original_title', 'overview', 'release_date', 'genre_ids',
            'original_language', 'backdrop_path', 'poster_path', 'vote_count',
            'vote_average', 'adult', 'video'
        ]

        copy_sql = f"""
        COPY watch_logs ({', '.join(csv_columns)})
        FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER ',', QUOTE '"', NULL '');
        """
        cur.copy_expert(copy_sql, buffer)

        conn.commit()
        logger.write(f"Successfully saved {len(df)} watch log data entries to 'watch_logs' table.")

    except psycopg2.Error as db_err:
        logger.write(f"[ERROR] Database error occurred: {db_err}", print_error=True)
        if conn:
            conn.rollback()
    except pd.errors.EmptyDataError:
        logger.write(f"[ERROR] Error: '{csv_path}' file is empty.", print_error=True)
    except Exception as e:
        logger.write(f"[ERROR] An unknown error occurred: {e}", print_error=True)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logger.write(f"--- Finished saving watch log data to DB ---")
