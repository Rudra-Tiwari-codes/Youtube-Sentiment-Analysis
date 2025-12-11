#!/bin/bash

# Deployment script for YouTube Sentiment Analysis API
# Author: YouTube Sentiment Analysis Team
# Date: December 9, 2025

set -e

echo "================================================"
echo "YouTube Sentiment Analysis API Deployment Script"
echo "================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "[ERROR] docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Parse arguments
DEPLOY_TYPE=${1:-cpu}  # cpu or gpu
BUILD_ONLY=${2:-false}

echo ""
echo "Deployment Configuration:"
echo "  Type: $DEPLOY_TYPE"
echo "  Build Only: $BUILD_ONLY"
echo ""

# Build Docker image
echo "Building Docker image..."
if [ "$DEPLOY_TYPE" = "gpu" ]; then
    docker build -f phase5_production_deployment/Dockerfile.gpu \
        -t youtube-sentiment-analysis-api:gpu \
        --target production \
        .
    echo "[OK] GPU image built successfully"
else
    docker build -f phase5_production_deployment/Dockerfile \
        -t youtube-sentiment-analysis-api:cpu \
        .
    echo "[OK] CPU image built successfully"
fi

# Exit if build-only mode
if [ "$BUILD_ONLY" = "true" ]; then
    echo ""
    echo "Build complete. Exiting (build-only mode)."
    exit 0
fi

# Deploy with docker-compose
echo ""
echo "Deploying with docker-compose..."
cd phase5_production_deployment
docker-compose up -d

# Wait for services to be healthy
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Health check
echo ""
echo "Performing health check..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "[OK] API is healthy and running"
else
    echo "[WARNING] API might still be starting up. Check logs with:"
    echo "   docker-compose -f phase5_production_deployment/docker-compose.yml logs -f sentiment-api"
fi

echo ""
echo "================================================"
echo "Deployment Complete!"
echo "================================================"
echo ""
echo "Services:"
echo "  API:        http://localhost:8000"
echo "  Docs:       http://localhost:8000/docs"
echo "  Health:     http://localhost:8000/health"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo "Useful commands:"
echo "  View logs:    docker-compose -f phase5_production_deployment/docker-compose.yml logs -f"
echo "  Stop:         docker-compose -f phase5_production_deployment/docker-compose.yml down"
echo "  Restart:      docker-compose -f phase5_production_deployment/docker-compose.yml restart"
echo ""
