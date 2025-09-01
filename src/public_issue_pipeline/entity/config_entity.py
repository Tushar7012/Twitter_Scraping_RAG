from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """Data class for data ingestion configuration."""
    query: str
    max_results: int
    tweet_fields: list
    user_fields: list

@dataclass(frozen=True)
class DatabaseConfig:
    """Data class for database configuration."""
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: str