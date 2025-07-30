#!/bin/bash

# MLflow Model Serving Setup Script
# This script sets up everything needed to run the model serving API

echo "🚀 MLflow Model Serving Setup"
echo "================================"

# Step 1: Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Step 2: Train a model if one doesn't exist
if [ ! -f "models/iris_model.pkl" ]; then
    echo "🤖 Training model (no existing model found)..."
    python simple_train.py
else
    echo "✅ Model already exists at models/iris_model.pkl"
fi

# Step 3: Test the API locally
echo "🧪 Testing API locally..."
echo "Starting FastAPI server in background..."
python app.py &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test the API
echo "Testing root endpoint..."
curl -s http://localhost:8000/ | python -m json.tool

echo -e "\nTesting prediction endpoint..."
curl -s -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}' | python -m json.tool

# Stop the server
kill $SERVER_PID

echo -e "\n✅ Setup complete!"
echo -e "\n📋 Next steps:"
echo "1. To run locally: python app.py"
echo "2. To build Docker image: docker build -t iris-model-api ."
echo "3. To run Docker container: docker run -p 8000:8000 iris-model-api"
echo "4. API will be available at: http://localhost:8000"
echo "5. Interactive docs at: http://localhost:8000/docs"
