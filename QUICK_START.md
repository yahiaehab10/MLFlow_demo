# MLflow Model Serving - Essential Commands

## Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python simple_train.py

# 3. Start the API
python app.py
```

**API Available at:** http://localhost:8000
**Interactive Docs:** http://localhost:8000/docs

---

## Test Commands

```bash
# Test the API
python test_api.py

# Manual test with curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## Docker Commands

```bash
# Build image
docker build -t iris-model-api .

# Run container
docker run -p 8000:8000 iris-model-api

# Test Docker container
curl http://localhost:8000/predict \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## Project Files

- **`app.py`** - FastAPI application
- **`simple_train.py`** - Model training script
- **`test_api.py`** - API test suite
- **`Dockerfile`** - Docker configuration
- **`requirements.txt`** - Python dependencies
- **`setup.sh`** - Automated setup script
- **`SERVING_GUIDE.md`** - Complete documentation

---

## What This Gives You

**REST API** for iris classification
**Trained model** ready to serve
**Docker support** for containerization
**Interactive docs** at `/docs`
**Comprehensive tests** included
**Production ready** setup

---

## API Usage Examples

### Python

```python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(response.json())  # {"prediction": 0, "features": [...]}
```

### JavaScript

```javascript
fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ features: [5.1, 3.5, 1.4, 0.2] }),
})
  .then((r) => r.json())
  .then((data) => console.log(data));
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```
