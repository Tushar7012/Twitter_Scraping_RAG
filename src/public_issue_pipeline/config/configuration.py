from public_issue_pipeline.constants import *
from public_issue_pipeline.entity.config_entity import DataIngestionConfig, DatabaseConfig

class ConfigurationManager:
    """Manages and provides configuration for the application."""

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Returns the data ingestion configuration."""
        return DataIngestionConfig(
            query=QUERY,
            max_results=MAX_RESULTS,
            tweet_fields=TWEET_FIELDS,
            user_fields=USER_FIELDS
        )

    def get_database_config(self) -> DatabaseConfig:
        """Returns the database configuration."""
        return DatabaseConfig(
            db_name=DB_NAME,
            db_user=DB_USER,
            db_password=DB_PASSWORD,
            db_host=DB_HOST,
            db_port=DB_PORT
        )