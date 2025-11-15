"""Pydantic models for request/response schemas."""
from pydantic import BaseModel, HttpUrl


class ImageProcessRequest(BaseModel):
    """Request model for image processing."""
    image_url: HttpUrl
    resultId: int


class ImageProcessResponse(BaseModel):
    """Response model for image processing request."""
    message: str
    queued_image: str


class DetectionResult(BaseModel):
    """Model for detection results."""
    image_url: str
    processed_image_url: str
    detected_objects_count: int



