from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
)
from feast.data_format import ParquetFormat
from feast.types import Int64, Float64  # Field의 dtype에 사용
from feast.value_type import ValueType # Entity의 value_type에 사용
from datetime import timedelta

# 엔티티 정의
# value_type에 ValueType.INT64 사용
user = Entity(name="user_id", value_type=ValueType.INT64) 
content = Entity(name="content_id", value_type=ValueType.INT64)

# 오프라인 소스 정의
raw_watch_log_source = FileSource(
     path="s3://my-mlops-raw-data-backup-3xplusy/raw_data/*/watch_logs_*.parquet",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
)

# 피처 뷰 정의
watch_features = FeatureView(
    name="watch_features",
    entities=[user, content],
    ttl=timedelta(days=7),
    online=True,
    source=raw_watch_log_source,
    schema=[
        Field(name="rating", dtype=Float64),
        Field(name="popularity", dtype=Float64),
        Field(name="watch_seconds", dtype=Int64),
    ]
)