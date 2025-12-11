"""
Phase 5: API Client Examples

Demonstrates how to interact with the Sentiment Analysis API.
Date: December 9, 2025
"""

import requests
import json
from typing import List, Dict
import time


class SentimentAPIClient:
    """Client for Sentiment Analysis API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict(self, text: str, return_confidence: bool = True) -> Dict:
        """
        Predict sentiment for a single text
        
        Args:
            text: Text to analyze
            return_confidence: Include confidence scores
        
        Returns:
            Prediction result
        """
        payload = {
            "text": text,
            "return_confidence": return_confidence
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, texts: List[str], return_confidence: bool = True) -> Dict:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            return_confidence: Include confidence scores
        
        Returns:
            Batch prediction results
        """
        payload = {
            "texts": texts,
            "return_confidence": return_confidence
        }
        
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get model metadata and statistics"""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()


def example_single_prediction():
    """Example: Single text prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Text Prediction")
    print("="*60)
    
    client = SentimentAPIClient()
    
    # Check if API is healthy
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Model Loaded: {health['model_loaded']}")
    
    # Predict sentiment
    text = "The job market is terrible and unemployment keeps rising"
    result = client.predict(text)
    
    print(f"\nText: {text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Inference Time: {result['inference_time_ms']:.2f}ms")


def example_batch_prediction():
    """Example: Batch prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Prediction")
    print("="*60)
    
    client = SentimentAPIClient()
    
    texts = [
        "Great job opportunities in tech sector!",
        "Unemployment crisis affecting youth",
        "The economy is doing okay I guess",
        "Best time to find a job in years",
        "Government policies are failing us"
    ]
    
    result = client.predict_batch(texts)
    
    print(f"\nBatch Size: {result['batch_size']}")
    print(f"Total Inference Time: {result['total_inference_time_ms']:.2f}ms")
    print(f"Avg per text: {result['total_inference_time_ms'] / result['batch_size']:.2f}ms")
    
    print("\nResults:")
    for i, item in enumerate(result['results'], 1):
        print(f"\n{i}. {item['text'][:50]}...")
        print(f"   Sentiment: {item['sentiment']}")
        print(f"   Confidence: {item['confidence']}")


def example_model_info():
    """Example: Get model information"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Information")
    print("="*60)
    
    client = SentimentAPIClient()
    
    info = client.get_model_info()
    
    print("\nModel Metadata:")
    for key, value in info['model_metadata'].items():
        print(f"  {key}: {value}")
    
    print("\nAPI Statistics:")
    for key, value in info['statistics'].items():
        print(f"  {key}: {value}")


def example_performance_test():
    """Example: Performance testing"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Performance Testing")
    print("="*60)
    
    client = SentimentAPIClient()
    
    test_texts = [
        "The unemployment rate is concerning",
        "Amazing job growth this quarter!",
        "Economy is stable for now"
    ] * 10  # 30 requests
    
    print(f"Testing with {len(test_texts)} requests...")
    
    start_time = time.time()
    results = []
    
    for text in test_texts:
        result = client.predict(text, return_confidence=False)
        results.append(result['inference_time_ms'])
    
    total_time = (time.time() - start_time) * 1000
    
    print(f"\nTotal Time: {total_time:.2f}ms")
    print(f"Avg per request: {sum(results) / len(results):.2f}ms")
    print(f"Min: {min(results):.2f}ms")
    print(f"Max: {max(results):.2f}ms")
    print(f"Throughput: {len(test_texts) / (total_time / 1000):.2f} req/sec")


def example_curl_commands():
    """Print example curl commands"""
    print("\n" + "="*60)
    print("EXAMPLE 5: cURL Commands")
    print("="*60)
    
    print("\n# Health check")
    print("curl http://localhost:8000/health")
    
    print("\n# Single prediction")
    print('curl -X POST "http://localhost:8000/predict" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"text": "This is a test comment", "return_confidence": true}\'')
    
    print("\n# Batch prediction")
    print('curl -X POST "http://localhost:8000/predict/batch" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"texts": ["Comment 1", "Comment 2"], "return_confidence": true}\'')
    
    print("\n# Model info")
    print("curl http://localhost:8000/model/info")


if __name__ == "__main__":
    try:
        print("Sentiment Analysis API - Client Examples")
        print("Make sure the API is running on http://localhost:8000")
        
        # Run examples
        example_single_prediction()
        example_batch_prediction()
        example_model_info()
        example_performance_test()
        example_curl_commands()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n ERROR: Cannot connect to API")
        print("Make sure the API is running:")
        print("  python phase5_production_deployment/01_fastapi_server.py")
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
