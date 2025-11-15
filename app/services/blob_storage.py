from typing import Optional
from google.cloud import storage
from google.oauth2 import service_account
import os
from app.config import settings


class BlobStorageService:
    """Service for interacting with GCP Cloud Storage."""
    
    def __init__(self):
        """Initialize GCP Cloud Storage client."""
        if not settings.GCP_STORAGE_BUCKET_NAME:
            raise ValueError("GCP_STORAGE_BUCKET_NAME is required")
        if not settings.GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID is required")
        if not settings.GCP_CREDENTIALS_PATH:
            raise ValueError("GCP_CREDENTIALS_PATH is required")
        
        # Load credentials from file
        if not os.path.exists(settings.GCP_CREDENTIALS_PATH):
            raise FileNotFoundError(
                f"GCP credentials file not found: {settings.GCP_CREDENTIALS_PATH}"
            )
        
        credentials = service_account.Credentials.from_service_account_file(
            settings.GCP_CREDENTIALS_PATH
        )
        
        self.storage_client = storage.Client(
            project=settings.GCP_PROJECT_ID,
            credentials=credentials
        )
        self.bucket_name = settings.GCP_STORAGE_BUCKET_NAME
    
    def download_image(self, blob_url: str) -> bytes:
        """
        Download an image from GCP Cloud Storage.
        
        For public URLs (https://storage.googleapis.com/...), downloads directly via HTTP.
        For gs:// URLs or private objects, uses the GCP Storage client.
        
        Args:
            blob_url: Full URL of the blob (e.g., gs://bucket-name/path/to/blob.jpg or https://storage.googleapis.com/bucket-name/path/to/blob.jpg)
            
        Returns:
            Image bytes
        """
        from urllib.parse import urlparse
        import httpx
        
        parsed = urlparse(blob_url)
        
        # If it's a public HTTP/HTTPS URL, download directly without modification
        # The URL is used as-is since it's already a valid public URL
        if parsed.scheme in ("http", "https"):
            # Download directly via HTTP (for public objects)
            # Use the URL as-is since it's already valid
            response = httpx.get(blob_url, timeout=30.0)
            response.raise_for_status()
            return response.content
        
        # For gs:// URLs, use the GCP Storage client
        elif parsed.scheme == "gs":
            # Format: gs://bucket-name/path/to/blob.jpg
            bucket_name = parsed.netloc
            # Remove query parameters from path if present
            blob_name = parsed.path.lstrip("/").split("?")[0]
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.download_as_bytes()
        
        else:
            raise ValueError(f"Unsupported URL scheme: {blob_url}")
    
    def upload_image(
        self, 
        image_bytes: bytes, 
        blob_name: str,
        bucket_name: Optional[str] = None
    ) -> str:
        """
        Upload an image to GCP Cloud Storage.
        
        Args:
            image_bytes: Image data as bytes
            blob_name: Name for the blob (e.g., processed/processed_image_123.jpg)
            bucket_name: Bucket name (defaults to configured bucket)
            
        Returns:
            Public HTTPS URL of the uploaded blob
        """
        bucket = self.storage_client.bucket(bucket_name or self.bucket_name)
        
        # Ensure bucket exists (will raise exception if not found)
        if not bucket.exists():
            raise ValueError(f"Bucket does not exist: {bucket_name or self.bucket_name}")
        
        # Upload blob
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type="image/jpeg")
        
        # Return the public HTTPS URL
        return f"https://storage.googleapis.com/{bucket.name}/{blob_name}"


# Global instance (lazy initialization)
_blob_storage_service: Optional[BlobStorageService] = None


def get_blob_storage_service() -> BlobStorageService:
    """Get or create blob storage service instance."""
    global _blob_storage_service
    if _blob_storage_service is None:
        _blob_storage_service = BlobStorageService()
    return _blob_storage_service
