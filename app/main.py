import os
import io
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    yield
    # Shutdown (optional cleanup)

app = FastAPI(
    title="Autism Detection API",
    description="API for autism detection using deep learning model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

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

# Add CORS middleware to allow file uploads from web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Debug path information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
    
    # Create the same model architecture as in training
    model = timm.create_model("tf_efficientnet_b5", pretrained=False, num_classes=2)
    
    # Load the trained weights
    try:
        if os.path.exists(MODEL_PATH):
            tensors = load_file(MODEL_PATH)
            model.load_state_dict(tensors)
            model = model.to(device)
            model.eval()
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


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
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


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
async def predict_autism(
    file: UploadFile = File(
        description="Image file (JPEG, PNG, etc.)"
    )
):
    """
    Predict autism from uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response with prediction results
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    # Additional validation for file extension
    if file.filename:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File extension {file_ext} not supported. Use: {', '.join(allowed_extensions)}"
            )
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please check server status."
        )
    
    try:
        # Read and open image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
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
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
