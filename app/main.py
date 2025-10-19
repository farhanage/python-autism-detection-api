import os
import io
import logging
from typing import Dict
import torch
import torch.nn as nn
from PIL import Image
import timm
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from safetensors.torch import load_file
import torchvision.transforms as transforms
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security settings
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10 MB default
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8080))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application...")
    try:
        load_model()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}", exc_info=True)
        # Don't raise - allow app to start with lazy loading
    yield
    # Shutdown (optional cleanup)
    logger.info("Application shutdown")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Autism Detection API",
    description="API for autism detection using deep learning model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded"}
))

# Response models for better OpenAPI documentation
class PredictionResult(BaseModel):
    predicted_class: str
    predicted_class_id: int
    confidence: float
    class_probabilities: Dict[str, float]

class PredictionResponse(BaseModel):
    success: bool
    filename: str
    prediction: PredictionResult

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; img-src 'self' https://fastapi.tiangolo.com"
    return response

# Add CORS middleware to allow file uploads from web interfaces
cors_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Global variables for model and device
model = None
device = None
class_names = ["Autistic", "Non-Autistic"]  # Based on typical binary classification

# Model and preprocessing configuration
IMG_SIZE = 224
# Get the absolute path to the model file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.safetensors")

# Image preprocessing pipeline (matching training pipeline)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # normalize to [-1,1] as in training
])


def load_model():
    """Load the EfficientNetB5 model from safetensors file"""
    global model, device
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        logger.info(f"Model path: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}")
            logger.warning("Model will be loaded on first request (lazy loading)")
            return False
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        
        # Create the same model architecture as in training
        model = timm.create_model("tf_efficientnet_b5", pretrained=False, num_classes=2)
        
        # Load the trained weights
        tensors = load_file(MODEL_PATH)
        model.load_state_dict(tensors)
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return False


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess the input image for model inference"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing transformations
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(device)
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Failed to process image format")


def predict_image(image_tensor: torch.Tensor) -> Dict:
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence_score = confidence.item()
            
            # Get probabilities for all classes
            probs = probabilities[0].cpu().numpy()
            
            result = {
                "predicted_class": class_names[predicted_class],
                "predicted_class_id": predicted_class,
                "confidence": float(confidence_score),
                "class_probabilities": {
                    class_names[0]: float(probs[0]),
                    class_names[1]: float(probs[1])
                }
            }
            
            return result
            
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction service temporarily unavailable")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Autism Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for autism detection",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }

@app.post("/predict", 
          summary="Predict Autism from Image",
          description="Upload an image file to get autism detection prediction",
          response_description="Prediction results with confidence scores",
          response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict_autism(request, file: UploadFile = File(
    description="Image file (JPEG, PNG, etc.)"
)):
    """
    Predict autism from uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response with prediction results
    """
    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Validate file extension
        if file.filename:
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            file_ext = os.path.splitext(file.filename.lower())[1]
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format"
                )
        
        # Check file size
        image_data = await file.read()
        if len(image_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB"
            )
        
        # Validate image can be opened
        try:
            image = Image.open(io.BytesIO(image_data))
            # Verify it's actually a valid image by loading it
            image.verify()
            # Reopen since verify() closes the image
            image = Image.open(io.BytesIO(image_data))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file"
            )
        
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded when processing request")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable"
            )
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        prediction_result = predict_image(image_tensor)
        
        # Return results
        return {
            "success": True,
            "filename": file.filename,
            "prediction": prediction_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request"
        )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
