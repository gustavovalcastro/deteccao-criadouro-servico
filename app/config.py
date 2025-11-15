"""Configuration settings for the application."""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # GCP Cloud Storage
    GCP_STORAGE_BUCKET_NAME: str = os.getenv(
        "GCP_STORAGE_BUCKET_NAME", ""
    )
    GCP_PROJECT_ID: str = os.getenv(
        "GCP_PROJECT_ID", ""
    )
    GCP_CREDENTIALS_PATH: str = os.getenv(
        "GCP_CREDENTIALS_PATH", ""
    )
    
    # YOLO Model
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "/app/models/model.pt")
    
    # External API
    EXTERNAL_API_URL: str = os.getenv("EXTERNAL_API_URL", "")
    EXTERNAL_API_KEY: Optional[str] = os.getenv("EXTERNAL_API_KEY", None)
    
    # Queue settings
    QUEUE_MAX_WORKERS: int = int(os.getenv("QUEUE_MAX_WORKERS", "4"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


