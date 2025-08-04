import os
import pandas as pd

from sqlalchemy import create_engine, text


def get_engine(db_name):
    engine = create_engine(
        f"postgresql://{os.environ.get('DB_USER')}:{os.environ.get('DB_PASSWORD')}"
        f"@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT')}/{db_name}"
    )
    return engine


def write_db(data: pd.DataFrame, db_name, table_name):
    engine = get_engine(db_name)
    data.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"{len(data)} rows written to {table_name} in {db_name}")


def read_db(db_name, table_name, k=10):
    engine = get_engine(db_name)
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                SELECT recommend_content_id
                FROM {table_name}
                ORDER BY index DESC
                LIMIT :k
            """), {"k": k}
        )
        return [row[0] for row in result]
