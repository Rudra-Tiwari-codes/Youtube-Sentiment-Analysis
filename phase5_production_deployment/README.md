# Phase 5: Production Deployment

Production-ready REST API for sentiment analysis inference using the trained DistilBERT model.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- 4GB+ RAM

### Deploy with Docker

**Windows:**
```powershell
.\phase5_production_deployment\deploy.ps1 -DeployType cpu
```

**Linux/Mac:**
```bash
bash phase5_production_deployment/deploy.sh cpu
```

API available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Local Development

```bash
pip install -r requirements.txt
python phase5_production_deployment/01_fastapi_server.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check with metrics |
| `/predict` | POST | Single text prediction |
| `/predict/batch` | POST | Batch predictions (max 100) |
| `/model-info` | GET | Model metadata |
| `/docs` | GET | Interactive documentation |

## Usage Examples

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing news!", "return_confidence": true}'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible", "Okay"], "return_confidence": true}'
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "The economy is improving", "return_confidence": True}
)
print(response.json())
```

## Testing

```bash
# Test model loading
python phase5_production_deployment/test_model_loading.py

# Test API client
python phase5_production_deployment/02_api_client.py

# Load testing
python phase5_production_deployment/03_load_testing.py
```

## Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f sentiment-api

# Stop services
docker-compose down

# Restart
docker-compose restart
```

## Performance

### CPU Inference
- Throughput: 50+ requests/second
- Average latency: 48ms
- P95 latency: 95ms
- Memory: ~1.2GB per worker

### GPU Inference (Optional)

```bash
.\deploy.ps1 -DeployType gpu  # Windows
bash deploy.sh gpu            # Linux/Mac
```

5-10x faster inference with NVIDIA GPU and Docker runtime.

## Configuration

Edit `docker-compose.yml` to adjust:
- Number of workers
- Port mapping
- Resource limits
- Environment variables

## Monitoring

Health endpoint returns:
- Model status
- Device information
- Request statistics
- Inference time metrics
- System resource usage

## Troubleshooting

**Port in use:**
```bash
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Linux/Mac
```

**Check logs:**
```bash
docker-compose logs sentiment-api
```

**Verify model:**
```bash
python phase5_production_deployment/test_model_loading.py
```

## Files

- `01_fastapi_server.py` - FastAPI application
- `02_api_client.py` - Example client
- `03_load_testing.py` - Load testing suite
- `Dockerfile` - CPU container
- `Dockerfile.gpu` - GPU container
- `docker-compose.yml` - Orchestration
- `deploy.ps1` / `deploy.sh` - Deployment scripts
- `test_model_loading.py` - Model verification

## Production Checklist

- [ ] Configure CORS for your domain
- [ ] Add authentication (API keys)
- [ ] Enable rate limiting
- [ ] Set up HTTPS/TLS
- [ ] Configure log rotation
- [ ] Set up monitoring alerts
- [ ] Review resource limits

## Model Information

- Model: DistilBERT fine-tuned
- Parameters: 66.9M
- Accuracy: 96.47%
- F1-Score: 96.47%
- Training data: 131,608 YouTube comments
- Classes: Negative, Neutral, Positive

## Status

Production-ready. Last updated: December 11, 2025.
