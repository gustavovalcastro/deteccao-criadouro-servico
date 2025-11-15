"""FastAPI application main entry point."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.models import ImageProcessRequest, ImageProcessResponse
from app.services.queue_processor import queue_processor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Detecção de Criadouro Service",
    description="Service for detecting objects in images using YOLO model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Detecção de Criadouro Service API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/process-images", response_model=ImageProcessResponse)
async def process_images(
    request: ImageProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Process an image from GCP Cloud Storage URL.
    
    This endpoint:
    1. Accepts a single image URL
    2. Adds it to the processing queue
    3. Returns immediately with a success message
    
    Processing happens in the background:
    - Image is downloaded from GCP Cloud Storage
    - YOLO model detects objects and draws bounding boxes
    - Processed image is uploaded to GCP Cloud Storage
    - Result is submitted to external API
    """
    image_url = str(request.image_url)
    result_id = request.resultId
    
    # Add image URL and resultId to the processing queue
    queue_processor.add_to_queue(image_url, result_id)
    
    # Trigger background processing
    background_tasks.add_task(queue_processor.process_queue)
    
    logger.info(f"Queued image for processing: {image_url} with resultId: {result_id}")
    
    return ImageProcessResponse(
        message="Image has been queued for processing",
        queued_image=image_url
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


