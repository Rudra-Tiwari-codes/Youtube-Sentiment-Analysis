"""
Phase 5: Production FastAPI Server

High-performance REST API for sentiment analysis inference.
Serves DistilBERT model with proper error handling, logging, and monitoring.

Date: December 9, 2025
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
import logging
from pathlib import Path
import yaml
from datetime import datetime
import psutil
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready sentiment analysis for Hinglish social media comments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
MODEL = None
TOKENIZER = None
DEVICE = None
MODEL_METADATA = {}
REQUEST_COUNT = 0
TOTAL_INFERENCE_TIME = 0.0


# Request/Response Models
class SentimentRequest(BaseModel):
    """Request model for sentiment analysis"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    return_confidence: bool = Field(default=True, description="Include confidence scores")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    return_confidence: bool = Field(default=True, description="Include confidence scores")


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""
    text: str
    sentiment: str
    confidence: Optional[Dict[str, float]] = None
    inference_time_ms: float
    timestamp: str


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis"""
    results: List[SentimentResponse]
    total_inference_time_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    total_requests: int
    avg_inference_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    uptime_seconds: float


# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Model loading
def load_model():
    """Load DistilBERT model and tokenizer"""
    global MODEL, TOKENIZER, DEVICE, MODEL_METADATA
    
    logger.info("Loading DistilBERT model...")
    start_time = time.time()
    
    try:
        # Determine device
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {DEVICE}")
        
        # Load fine-tuned DistilBERT checkpoint
        model_path = Path(__file__).parent.parent / 'phase3_transformer_models' / 'checkpoints' / 'distilbert'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}\n"
                "Please ensure the trained checkpoint is available."
            )
        
        logger.info(f"Loading fine-tuned model from: {model_path}")
        TOKENIZER = AutoTokenizer.from_pretrained(str(model_path))
        MODEL = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        MODEL.to(DEVICE)
        MODEL.eval()
        
        # Store metadata
        MODEL_METADATA = {
            'model_name': 'distilbert-fine-tuned-sentiment',
            'model_path': str(model_path),
            'accuracy': '96.47%',  # From trained checkpoint
            'f1_macro': '95.63%',
            'f1_weighted': '96.47%',
            'num_parameters': sum(p.numel() for p in MODEL.parameters()),
            'device': str(DEVICE),
            'load_time_seconds': time.time() - start_time
        }
        
        logger.info(f" Model loaded successfully in {MODEL_METADATA['load_time_seconds']:.2f}s")
        logger.info(f"   Parameters: {MODEL_METADATA['num_parameters']:,}")
        logger.info(f"   Device: {DEVICE}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


# Inference function
def predict_sentiment(text: str, return_confidence: bool = True) -> Dict[str, Any]:
    """
    Predict sentiment for a single text
    
    Args:
        text: Input text
        return_confidence: Whether to return confidence scores
    
    Returns:
        Dictionary with sentiment and optionally confidence scores
    """
    global REQUEST_COUNT, TOTAL_INFERENCE_TIME
    
    start_time = time.time()
    
    try:
        # Tokenize
        inputs = TOKENIZER(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = MODEL(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        
        # Map to sentiment labels
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map[predicted_class]
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update metrics
        REQUEST_COUNT += 1
        TOTAL_INFERENCE_TIME += inference_time
        
        result = {
            'sentiment': sentiment,
            'inference_time_ms': round(inference_time, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        if return_confidence:
            result['confidence'] = {
                'Negative': round(probs[0][0].item(), 4),
                'Neutral': round(probs[0][1].item(), 4),
                'Positive': round(probs[0][2].item(), 4)
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global START_TIME
    START_TIME = time.time()
    load_model()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    process = psutil.Process(os.getpid())
    
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        device=str(DEVICE) if DEVICE else "unknown",
        total_requests=REQUEST_COUNT,
        avg_inference_time_ms=round(
            TOTAL_INFERENCE_TIME / REQUEST_COUNT if REQUEST_COUNT > 0 else 0.0,
            2
        ),
        memory_usage_mb=round(process.memory_info().rss / 1024 / 1024, 2),
        cpu_percent=process.cpu_percent(),
        uptime_seconds=round(time.time() - START_TIME, 2)
    )


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    """
    Predict sentiment for a single text
    
    Example:
    ```
    {
        "text": "This job market is terrible",
        "return_confidence": true
    }
    ```
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = predict_sentiment(request.text, request.return_confidence)
        return SentimentResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result.get('confidence'),
            inference_time_ms=result['inference_time_ms'],
            timestamp=result['timestamp']
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchSentimentRequest):
    """
    Predict sentiment for multiple texts (batch inference)
    
    Example:
    ```
    {
        "texts": [
            "Great job opportunities",
            "Unemployment is rising"
        ],
        "return_confidence": true
    }
    ```
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        batch_start = time.time()
        results = []
        
        for text in request.texts:
            result = predict_sentiment(text, request.return_confidence)
            results.append(SentimentResponse(
                text=text,
                sentiment=result['sentiment'],
                confidence=result.get('confidence'),
                inference_time_ms=result['inference_time_ms'],
                timestamp=result['timestamp']
            ))
        
        total_time = (time.time() - batch_start) * 1000
        
        return BatchSentimentResponse(
            results=results,
            total_inference_time_ms=round(total_time, 2),
            batch_size=len(request.texts)
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model metadata"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_metadata": MODEL_METADATA,
        "statistics": {
            "total_requests": REQUEST_COUNT,
            "avg_inference_time_ms": round(
                TOTAL_INFERENCE_TIME / REQUEST_COUNT if REQUEST_COUNT > 0 else 0.0,
                2
            )
        }
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run server
    uvicorn.run(
        "01_fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Increase for production
        log_level="info"
    )
