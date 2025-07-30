# 🚀 MLflow Model Serving - Complete Guide

This guide provides all the commands needed to set up and run the iris model serving API locally and with Docker.

## 🏗️ Quick Setup (Automated)

Run the automated setup script:

```bash
./setup.sh
```

## 📋 Manual Setup Commands

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python simple_train.py
```

### 3. Start the API Server

```bash
python app.py
```

The API will be available at:

- **Main API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### 4. Test the API

#### Test Welcome Endpoint

```bash
curl http://localhost:8000/
```

#### Test Prediction Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Expected response:

```json
{
  "prediction": 0,
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

## 🐳 Docker Deployment

### 1. Build Docker Image

```bash
docker build -t iris-model-api .
```

### 2. Run Docker Container

```bash
docker run -p 8000:8000 iris-model-api
```

### 3. Test Docker Container

```bash
# Test welcome endpoint
curl http://localhost:8000/

# Test prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.9, 3.0, 5.1, 1.8]}'
```

## 🌐 Cloud Deployment Options

### Deploy to Heroku

```bash
# Install Heroku CLI first
heroku create your-iris-api
git push heroku main
```

### Deploy to Railway

```bash
# Connect your GitHub repo to Railway
# Set PORT environment variable to 8000
```

### Deploy to Google Cloud Run

```bash
gcloud run deploy iris-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📊 API Endpoints

### GET `/`

Returns a welcome message.

**Response:**

```json
{
  "message": "Welcome to the Iris Model Prediction API"
}
```

### POST `/predict`

Makes predictions using the trained iris model.

**Request Body:**

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response:**

```json
{
  "prediction": 0,
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Iris Classes:**

- `0`: Iris Setosa
- `1`: Iris Versicolor
- `2`: Iris Virginica

## 🔧 Troubleshooting

### Model Not Found Error

If you get a "Model not loaded" error:

```bash
# Train a new model
python simple_train.py

# Restart the API
python app.py
```

### Port Already in Use

If port 8000 is busy:

```bash
# Kill existing processes
pkill -f "python app.py"

# Or use a different port
uvicorn app:app --host 0.0.0.0 --port 8001
```

### Docker Build Issues

Ensure you have:

- Docker installed and running
- All files in the current directory
- Model file exists in `models/iris_model.pkl`

## 📈 Performance Testing

### Load Testing with curl

```bash
# Simple load test
for i in {1..100}; do
  curl -s -X POST "http://localhost:8000/predict" \
       -H "Content-Type: application/json" \
       -d '{"features": [5.1, 3.5, 1.4, 0.2]}' &
done
wait
```

### Using Apache Bench

```bash
# Install apache2-utils first
ab -n 1000 -c 10 -T "application/json" \
   -p test_payload.json http://localhost:8000/predict
```

Create `test_payload.json`:

```json
{ "features": [5.1, 3.5, 1.4, 0.2] }
```

## 🔍 Monitoring

### Health Check Endpoint

Add this to your `app.py` for monitoring:

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
```

### View Logs

```bash
# Docker logs
docker logs <container_id>

# Local logs
tail -f /var/log/iris-api.log
```

## 🚀 Next Steps

1. **Add Authentication**: Implement API keys or JWT tokens
2. **Add Logging**: Use structured logging for better monitoring
3. **Add Caching**: Cache predictions for repeated requests
4. **Add Batch Processing**: Support multiple predictions in one request
5. **Add Model Versioning**: Support A/B testing with multiple models
6. **Add Metrics**: Implement Prometheus metrics for monitoring

## 📝 Example Integration

### Python Client

```python
import requests

def predict_iris(features):
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": features}
    )
    return response.json()

# Usage
result = predict_iris([5.1, 3.5, 1.4, 0.2])
print(f"Predicted class: {result['prediction']}")
```

### JavaScript Client

```javascript
async function predictIris(features) {
  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ features: features }),
  });
  return await response.json();
}

// Usage
predictIris([5.1, 3.5, 1.4, 0.2]).then((result) =>
  console.log("Prediction:", result.prediction)
);
```

## 🎯 Summary

You now have a complete model serving solution that:

- ✅ Loads a trained scikit-learn model
- ✅ Serves predictions via REST API
- ✅ Includes interactive documentation
- ✅ Can be containerized with Docker
- ✅ Ready for cloud deployment
- ✅ Includes comprehensive testing commands

The API is production-ready and can handle real-time predictions for the Iris dataset classification task.
