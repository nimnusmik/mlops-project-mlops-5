import os
import pandas as pd

from datetime import datetime
from sqlalchemy import create_engine, text

def get_engine(db_name):
    engine = create_engine(
        f"postgresql://{os.environ.get('DB_USER')}:{os.environ.get('DB_PASSWORD')}"
        f"@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT')}/{db_name}"
    )
    return engine


def write_db(data: pd.DataFrame, db_name, table_name):
    engine = get_engine(db_name)

    with engine.connect() as conn:
        if table_name == "recommend":
            conn.execute(text(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table_name}'
                    ) THEN
                        CREATE TABLE {table_name} (
                            recommend_content_id INTEGER,
                            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    ELSIF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{table_name}' AND column_name = 'ingested_at'
                    ) THEN
                        ALTER TABLE {table_name}
                        ADD COLUMN ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                    END IF;
                END
                $$;
            """))
            conn.commit()

    if "ingested_at" not in data.columns:
        data["ingested_at"] = datetime.now()

    data.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"{len(data)} rows written to {table_name} in {db_name}")


def read_db(db_name, table_name, k=10):
    engine = get_engine(db_name)
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                SELECT recommend_content_id
                FROM {table_name}
                ORDER BY ingested_at DESC
                LIMIT :k
            """), {"k": k}
        )
        return [row[0] for row in result]


def get_movie_metadata_by_ids(db_name, content_ids):
    engine = get_engine(db_name)
    query = text("""
        SELECT DISTINCT content_id, title, poster_path, overview
        FROM watch_logs
        WHERE content_id = ANY(:content_ids)
        AND poster_path IS NOT NULL
        AND title IS NOT NULL
    """)

    content_ids = [int(cid) for cid in content_ids]
    
    with engine.connect() as conn:
        result = conn.execute(query, {"content_ids": content_ids}).mappings().all()
        return {
            int(row["content_id"]): {
                "title": row["title"],
                "poster_url": f"https://image.tmdb.org/t/p/original{row['poster_path']}",
                "overview": row["overview"]
            } for row in result
        }
