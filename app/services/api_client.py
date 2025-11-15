"""External API client for submitting detection results."""
import httpx
from typing import Dict, Any
from app.config import settings


class APIClient:
    """Client for calling external API."""
    
    def __init__(self):
        """Initialize API client."""
        self.api_url = settings.EXTERNAL_API_URL
        self.api_key = settings.EXTERNAL_API_KEY
    
    async def update_result(
        self,
        result_id: int,
        processed_image_url: str,
        status: str,
        object_count: int
    ) -> Dict[str, Any]:
        """
        Update result in external API.
        
        Args:
            result_id: ID of the result to update
            processed_image_url: URL of the processed image in GCP Cloud Storage
            status: Status of the processing ("finished" or "failed")
            object_count: Number of detected objects
            
        Returns:
            Response from API as dictionary
        """
        if not self.api_url:
            raise ValueError("EXTERNAL_API_URL is not configured")
        
        # Construct the full URL for the PUT endpoint
        update_url = f"{self.api_url.rstrip('/')}/results/updateResultImage"
        
        payload = {
            "id": result_id,
            "resultImage": processed_image_url,
            "status": status,
            "object_count": object_count
        }
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with httpx.AsyncClient() as client:
            response = await client.put(
                update_url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()


# Global instance
api_client = APIClient()


