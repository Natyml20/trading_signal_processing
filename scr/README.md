# Trading Signal Processing System

A FastAPI-based REST API for processing trading signals and tracking portfolio performance.

## Features

- Process trading signals (BUY, SELL, HOLD)
- Track portfolio performance
- Health check endpoint
- Input validation using Pydantic
- Containerized with Docker

## API Endpoints

### Health Check
- `GET /health`
  - Returns the health status of the API

### Signal Processing
- `POST /signal`
  - Process a new trading signal
  - Request body should follow the Signal model structure

### Performance
- `GET /performance`
  - Get current portfolio performance metrics

## Running the Application

### Using Docker

1. Build the Docker image:
```bash
docker build -t trading-signal-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 trading-signal-api
```

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

## API Documentation

Once the application is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example Signal Request

```json
{
    "datetime": "2025-02-24T00:00:00",
    "close": 95439.81388,
    "mvrv_btc_momentum": -0.003825082,
    "spot_volume_daily_sum": 4592772971,
    "signal": 1,
    "summary": "Market analysis summary",
    "next_news_prediction": "Predicted market movement",
    "sentiment": "positive",
    "index": 0,
    "key_factors": "Key market factors",
    "dominant_emotions": "optimism",
    "dominant_sentiment": "positive",
    "intensity": 7,
    "psychology_explanation": "Market psychology analysis"
}
```
