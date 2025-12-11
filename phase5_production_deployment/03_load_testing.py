"""
Phase 5: Load Testing & Performance Benchmarks

Comprehensive load testing for the sentiment analysis API.
Tests throughput, latency, and concurrent request handling.

Date: December 9, 2025
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class LoadTester:
    """Load testing for sentiment API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def single_request(self, session: aiohttp.ClientSession, text: str) -> Dict:
        """Make a single API request"""
        start_time = time.time()
        
        async with session.post(
            f"{self.base_url}/predict",
            json={"text": text, "return_confidence": False}
        ) as response:
            result = await response.json()
            latency = (time.time() - start_time) * 1000  # ms
            
            return {
                'latency_ms': latency,
                'status': response.status,
                'sentiment': result.get('sentiment')
            }
    
    async def concurrent_requests(self, num_requests: int, texts: List[str]) -> List[Dict]:
        """Execute concurrent requests"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                text = texts[i % len(texts)]
                tasks.append(self.single_request(session, text))
            
            return await asyncio.gather(*tasks)
    
    def run_load_test(self, num_requests: int, texts: List[str]) -> Dict:
        """Run load test with specified parameters"""
        logger.info(f"Running load test: {num_requests} requests")
        
        start_time = time.time()
        results = asyncio.run(self.concurrent_requests(num_requests, texts))
        total_time = time.time() - start_time
        
        latencies = [r['latency_ms'] for r in results]
        successful = sum(1 for r in results if r['status'] == 200)
        
        metrics = {
            'num_requests': num_requests,
            'total_time_s': total_time,
            'successful_requests': successful,
            'failed_requests': num_requests - successful,
            'throughput_rps': num_requests / total_time,
            'avg_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'p95_latency_ms': self.percentile(latencies, 95),
            'p99_latency_ms': self.percentile(latencies, 99),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'latencies': latencies
        }
        
        return metrics
    
    @staticmethod
    def percentile(data: List[float], p: float) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int((p / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


def test_throughput_scaling():
    """Test API throughput with increasing load"""
    logger.info("="*60)
    logger.info("TEST 1: Throughput Scaling")
    logger.info("="*60)
    
    tester = LoadTester()
    
    test_texts = [
        "The unemployment rate is concerning",
        "Amazing job growth this quarter!",
        "Economy is stable for now",
        "Job market looks promising",
        "Economic crisis affecting everyone"
    ]
    
    test_sizes = [10, 50, 100, 200, 500]
    results = []
    
    for size in test_sizes:
        logger.info(f"\nTesting {size} concurrent requests...")
        metrics = tester.run_load_test(size, test_texts)
        results.append(metrics)
        
        logger.info(f"  Throughput: {metrics['throughput_rps']:.2f} req/s")
        logger.info(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        logger.info(f"  P95 Latency: {metrics['p95_latency_ms']:.2f}ms")
    
    # Create visualization
    output_dir = Path(__file__).parent / 'benchmarks'
    output_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Throughput plot
    axes[0].plot(df['num_requests'], df['throughput_rps'], marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Concurrent Requests', fontsize=12)
    axes[0].set_ylabel('Throughput (req/s)', fontsize=12)
    axes[0].set_title('API Throughput vs Load', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Latency plot
    axes[1].plot(df['num_requests'], df['avg_latency_ms'], marker='o', label='Avg', linewidth=2)
    axes[1].plot(df['num_requests'], df['p95_latency_ms'], marker='s', label='P95', linewidth=2)
    axes[1].plot(df['num_requests'], df['p99_latency_ms'], marker='^', label='P99', linewidth=2)
    axes[1].set_xlabel('Concurrent Requests', fontsize=12)
    axes[1].set_ylabel('Latency (ms)', fontsize=12)
    axes[1].set_title('API Latency vs Load', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_scaling.png', dpi=300, bbox_inches='tight')
    logger.info(f"\n Saved: {output_dir / 'throughput_scaling.png'}")
    
    # Save metrics
    df.to_csv(output_dir / 'throughput_metrics.csv', index=False)
    logger.info(f" Saved: {output_dir / 'throughput_metrics.csv'}")
    
    return results


def test_latency_distribution():
    """Test latency distribution under steady load"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Latency Distribution")
    logger.info("="*60)
    
    tester = LoadTester()
    
    test_texts = [
        "The unemployment rate is concerning",
        "Amazing job growth this quarter!",
        "Economy is stable for now"
    ]
    
    logger.info("\nRunning 200 concurrent requests...")
    metrics = tester.run_load_test(200, test_texts)
    
    logger.info(f"\nLatency Statistics:")
    logger.info(f"  Min: {metrics['min_latency_ms']:.2f}ms")
    logger.info(f"  Avg: {metrics['avg_latency_ms']:.2f}ms")
    logger.info(f"  Median: {metrics['median_latency_ms']:.2f}ms")
    logger.info(f"  P95: {metrics['p95_latency_ms']:.2f}ms")
    logger.info(f"  P99: {metrics['p99_latency_ms']:.2f}ms")
    logger.info(f"  Max: {metrics['max_latency_ms']:.2f}ms")
    
    # Create visualization
    output_dir = Path(__file__).parent / 'benchmarks'
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(metrics['latencies'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(metrics['avg_latency_ms'], color='red', linestyle='--', linewidth=2, label=f'Avg: {metrics["avg_latency_ms"]:.2f}ms')
    plt.axvline(metrics['p95_latency_ms'], color='orange', linestyle='--', linewidth=2, label=f'P95: {metrics["p95_latency_ms"]:.2f}ms')
    plt.xlabel('Latency (ms)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Latency Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(metrics['latencies'], vert=True)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title('Latency Box Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"\n Saved: {output_dir / 'latency_distribution.png'}")
    
    return metrics


def generate_benchmark_report(throughput_results: List[Dict], latency_metrics: Dict):
    """Generate comprehensive benchmark report"""
    logger.info("\n" + "="*60)
    logger.info("GENERATING BENCHMARK REPORT")
    logger.info("="*60)
    
    output_dir = Path(__file__).parent / 'benchmarks'
    report_file = output_dir / 'benchmark_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SENTIMENT ANALYSIS API PERFORMANCE BENCHMARK REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("TEST 1: THROUGHPUT SCALING\n")
        f.write("-"*70 + "\n")
        for result in throughput_results:
            f.write(f"\nLoad: {result['num_requests']} concurrent requests\n")
            f.write(f"  Total Time: {result['total_time_s']:.2f}s\n")
            f.write(f"  Throughput: {result['throughput_rps']:.2f} req/s\n")
            f.write(f"  Avg Latency: {result['avg_latency_ms']:.2f}ms\n")
            f.write(f"  P95 Latency: {result['p95_latency_ms']:.2f}ms\n")
            f.write(f"  P99 Latency: {result['p99_latency_ms']:.2f}ms\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("TEST 2: LATENCY DISTRIBUTION (200 requests)\n")
        f.write("-"*70 + "\n")
        f.write(f"Min Latency: {latency_metrics['min_latency_ms']:.2f}ms\n")
        f.write(f"Avg Latency: {latency_metrics['avg_latency_ms']:.2f}ms\n")
        f.write(f"Median Latency: {latency_metrics['median_latency_ms']:.2f}ms\n")
        f.write(f"P95 Latency: {latency_metrics['p95_latency_ms']:.2f}ms\n")
        f.write(f"P99 Latency: {latency_metrics['p99_latency_ms']:.2f}ms\n")
        f.write(f"Max Latency: {latency_metrics['max_latency_ms']:.2f}ms\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-"*70 + "\n")
        
        max_throughput = max(r['throughput_rps'] for r in throughput_results)
        f.write(f"Peak Throughput: {max_throughput:.2f} req/s\n")
        f.write(f"Average Latency: {latency_metrics['avg_latency_ms']:.2f}ms\n")
        f.write(f"P95 Latency: {latency_metrics['p95_latency_ms']:.2f}ms\n")
        f.write(f"Success Rate: {latency_metrics['successful_requests'] / latency_metrics['num_requests'] * 100:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-"*70 + "\n")
        f.write("1. For high-throughput scenarios: Consider horizontal scaling\n")
        f.write("2. For low-latency requirements: Use GPU inference\n")
        f.write("3. For cost optimization: Batch processing recommended\n")
        f.write("4. For production: Set up load balancer with 2-4 workers\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    logger.info(f" Saved: {report_file}")


if __name__ == "__main__":
    try:
        logger.info("Sentiment Analysis API Load Testing")
        logger.info("Make sure the API is running on http://localhost:8000\n")
        
        # Run tests
        throughput_results = test_throughput_scaling()
        latency_metrics = test_latency_distribution()
        
        # Generate report
        generate_benchmark_report(throughput_results, latency_metrics)
        
        logger.info("\n" + "="*60)
        logger.info(" All load tests completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f" Load testing failed: {str(e)}")
        raise
