from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import uvicorn
from typing import List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Iris Model Prediction API",
    description="API for making predictions using the trained Iris model",
    version="1.0.0",
)


# Define input data model
class IrisData(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {"features": [5.1, 3.5, 1.4, 0.2]}  # Example of iris features
        }


# Load the model at startup
@app.on_event("startup")
async def load_model():
    global model

    # Try loading joblib model first (simple and direct)
    if os.path.exists("models/iris_model.pkl"):
        try:
            model = joblib.load("models/iris_model.pkl")
            print("Model loaded successfully from: models/iris_model.pkl")
            return
        except Exception as e:
            print(f"Failed to load joblib model: {str(e)}")

    # Fallback to MLflow approaches
    import mlflow

    model_approaches = [
        ("models:/IrisRandomForest/Staging", "Model Registry - Staging"),
        ("models:/IrisRandomForest/latest", "Model Registry - Latest"),
        ("mlruns/models", "Local models directory"),
        ("mlruns/0/85352b5f8d474b4f850f206501da8f7b/artifacts/model", "Run artifacts"),
    ]

    for model_uri, description in model_approaches:
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model loaded successfully from: {description} ({model_uri})")
            return
        except Exception as e:
            print(f"Failed to load model from {description}: {str(e)}")
            continue

    model = None
    print(
        "Warning: No model could be loaded. API will return errors for prediction requests."
    )
    print(
        "Please ensure you have trained a model first by running: python simple_train.py"
    )


@app.get("/")
async def root():
    return {"message": "Welcome to the Iris Model Prediction API"}


@app.post("/predict")
async def predict(data: IrisData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input features to numpy array
        features = np.array([data.features])
        # Make prediction
        prediction = model.predict(features)

        # Convert prediction to Python type for JSON serialization
        prediction = (
            prediction.tolist()[0] if isinstance(prediction, np.ndarray) else prediction
        )

        return {"prediction": prediction, "features": data.features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
