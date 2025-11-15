"""Queue processor for async image processing."""
import asyncio
import logging
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from app.services.blob_storage import get_blob_storage_service
from app.services.yolo_processor import get_yolo_processor
from app.services.api_client import api_client
from app.config import settings


logger = logging.getLogger(__name__)


class QueueProcessor:
    """Service for processing images in a queue."""
    
    def __init__(self):
        """Initialize queue processor."""
        self.executor = ThreadPoolExecutor(max_workers=settings.QUEUE_MAX_WORKERS)
        self.queue: List[Tuple[str, int]] = []  # List of (image_url, resultId) tuples
        self.processing = False
    
    def add_to_queue(self, image_url: str, result_id: int):
        """Add an image URL and resultId to the processing queue."""
        self.queue.append((image_url, result_id))
        logger.info(f"Added image to queue: {image_url} with resultId: {result_id}")
    
    async def process_queue(self):
        """Process all images in the queue."""
        if self.processing:
            return
        
        self.processing = True
        
        while self.queue:
            image_url, result_id = self.queue.pop(0)
            try:
                await self._process_image(image_url, result_id)
            except Exception as e:
                logger.error(f"Error processing image {image_url}: {str(e)}", exc_info=True)
                # Send failed status to API
                try:
                    await api_client.update_result(
                        result_id=result_id,
                        processed_image_url="",
                        status="failed",
                        object_count=0
                    )
                except Exception as api_error:
                    logger.error(f"Failed to send error status to API: {str(api_error)}", exc_info=True)
        
        self.processing = False
    
    async def _process_image(self, image_url: str, result_id: int):
        """
        Process a single image: download, detect, upload, and submit to API.
        
        Args:
            image_url: URL of the image in GCP Cloud Storage
            result_id: ID of the result to update
        """
        logger.info(f"Processing image: {image_url} with resultId: {result_id}")
        
        # Download image (blocking I/O in thread pool)
        loop = asyncio.get_event_loop()
        blob_service = get_blob_storage_service()
        image_bytes = await loop.run_in_executor(
            self.executor,
            blob_service.download_image,
            image_url
        )
        
        # Process with YOLO (blocking CPU work in thread pool)
        processed_bytes, detected_count = await loop.run_in_executor(
            self.executor,
            self._run_yolo_detection,
            image_bytes
        )
        
        # Generate blob name for processed image in the "processed" folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        blob_name = f"processed/processed_{timestamp}.jpg"
        
        # Upload processed image (blocking I/O in thread pool)
        processed_image_url = await loop.run_in_executor(
            self.executor,
            blob_service.upload_image,
            processed_bytes,
            blob_name
        )
        
        logger.info(
            f"Processed image uploaded: {processed_image_url}, "
            f"detected {detected_count} objects"
        )
        
        # Submit to external API (async HTTP)
        try:
            await api_client.update_result(
                result_id=result_id,
                processed_image_url=processed_image_url,
                status="finished",
                object_count=detected_count
            )
            logger.info(f"Successfully updated result {result_id} in API")
        except Exception as e:
            logger.error(f"Failed to update result in API: {str(e)}", exc_info=True)
            # Try to send failed status
            try:
                await api_client.update_result(
                    result_id=result_id,
                    processed_image_url="",
                    status="failed",
                    object_count=0
                )
            except Exception as api_error:
                logger.error(f"Failed to send error status to API: {str(api_error)}", exc_info=True)
    
    def _run_yolo_detection(self, image_bytes: bytes):
        """Run YOLO detection (must be called from thread pool)."""
        yolo_processor = get_yolo_processor()
        return yolo_processor.detect_objects(image_bytes)


# Global instance
queue_processor = QueueProcessor()

