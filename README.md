# Autism Detection API

A FastAPI-based web service for autism detection using a deep learning model (EfficientNetB5) trained with PyTorch.

## Features

- **Image Upload**: Upload images for autism detection
- **Real-time Prediction**: Get instant predictions with confidence scores
- **Health Check**: Monitor API status and model availability
- **Docker Support**: Easy deployment with Docker
- **Interactive Documentation**: Built-in Swagger UI at `/docs`

## Quick Start

### Method 1: Direct Python Execution

1. **Install Dependencies**
   ```bash
   cd deployment
   pip install -r requirements.txt
   ```

2. **Run the API**
   ```bash
   python app/main.py
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Method 2: Docker Deployment

1. **Build Docker Image**
   ```bash
   cd deployment
   docker build -t autism-detection-api .
   ```

2. **Run Container**
   ```bash
   docker run -p 8000:8000 autism-detection-api
   ```

## API Endpoints

### GET `/`
Returns basic API information and available endpoints.

### GET `/health`
Health check endpoint that returns:
- API status
- Model loading status
- Device information (CPU/GPU)

### POST `/predict`
Upload an image file for autism detection.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "success": true,
  "filename": "example.jpg",
  "prediction": {
    "predicted_class": "Autistic",
    "predicted_class_id": 1,
    "confidence": 0.8542,
    "class_probabilities": {
      "Non-Autistic": 0.1458,
      "Autistic": 0.8542
    }
  }
}
```

## Usage Examples

### Python Requests Example

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict autism from image
with open("test_image.jpg", "rb") as f:
    files = {"file": ("test_image.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

### cURL Example

```bash
# Health check
curl http://localhost:8000/health

# Upload and predict
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
```

### JavaScript/Fetch Example

```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Upload and predict
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Model Information

- **Architecture**: EfficientNetB5 (timm implementation)
- **Classes**: Binary classification (Non-Autistic, Autistic)
- **Input Size**: 224x224 pixels
- **Normalization**: [-1, 1] range
- **Format**: SafeTensors (.safetensors)

## File Structure

```
deployment/
├── app/
│   └── main.py          # FastAPI application
├── model/
│   └── best_model.safetensors  # Trained model weights
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
└── README.md           # This file
```

## Development

### Adding New Features

1. **Custom Preprocessing**: Modify the `preprocess_image()` function
2. **Multiple Models**: Extend the model loading logic
3. **Batch Prediction**: Add batch processing endpoints
4. **Model Metrics**: Add model performance monitoring

### Environment Variables

You can configure the API using environment variables:

```bash
export MODEL_PATH="/path/to/model/best_model.safetensors"
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `best_model.safetensors` exists in the `model/` directory
2. **CUDA errors**: The API automatically falls back to CPU if GPU is not available
3. **Image format errors**: Ensure uploaded images are in supported formats (JPEG, PNG, etc.)
4. **Memory issues**: For large images, consider adding image size validation

### Logs

The application logs important information including:
- Device selection (CPU/GPU)
- Model loading status
- Prediction errors

## Security Considerations

- **File Size Limits**: Consider adding file size limits for production
- **Rate Limiting**: Implement rate limiting for production deployments
- **Input Validation**: Additional image validation may be needed
- **HTTPS**: Use HTTPS in production environments

## Performance

- **Model Loading**: Model is loaded once at startup
- **Inference Speed**: Depends on device (GPU recommended for production)
- **Concurrent Requests**: FastAPI handles concurrent requests efficiently
- **Memory Usage**: ~2GB RAM recommended for EfficientNetB5

## License

This project is for educational and research purposes.