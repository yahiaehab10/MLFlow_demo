FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.pipeline"]
