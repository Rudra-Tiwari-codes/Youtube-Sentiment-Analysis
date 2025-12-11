# Deployment script for YouTube Sentiment Analysis API (Windows)
# Author: YouTube Sentiment Analysis Team
# Date: December 9, 2025

param(
    [string]$DeployType = "cpu",  # cpu or gpu
    [switch]$BuildOnly = $false
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "YouTube Sentiment Analysis API Deployment Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if Docker is installed
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is installed
if (!(Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] docker-compose is not installed. Please install docker-compose first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Deployment Configuration:"
Write-Host "  Type: $DeployType"
Write-Host "  Build Only: $BuildOnly"
Write-Host ""

# Navigate to project root
$projectRoot = Split-Path -Parent $PSScriptRoot

# Build Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
if ($DeployType -eq "gpu") {
    docker build -f "$projectRoot\phase5_production_deployment\Dockerfile.gpu" `
        -t youtube-sentiment-analysis-api:gpu `
        $projectRoot
    Write-Host "[OK] GPU image built successfully" -ForegroundColor Green
} else {
    docker build -f "$projectRoot\phase5_production_deployment\Dockerfile" `
        -t youtube-sentiment-analysis-api:cpu `
        $projectRoot
    Write-Host "[OK] CPU image built successfully" -ForegroundColor Green
}

# Exit if build-only mode
if ($BuildOnly) {
    Write-Host ""
    Write-Host "Build complete. Exiting (build-only mode)." -ForegroundColor Green
    exit 0
}

# Deploy with docker-compose
Write-Host ""
Write-Host "Deploying with docker-compose..." -ForegroundColor Yellow
Push-Location "$projectRoot\phase5_production_deployment"
docker-compose up -d
Pop-Location

# Wait for services to be healthy
Write-Host ""
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Health check
Write-Host ""
Write-Host "Performing health check..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] API is healthy and running" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARNING] API might still be starting up. Check logs with:" -ForegroundColor Yellow
    Write-Host "   docker-compose -f phase5_production_deployment\docker-compose.yml logs -f sentiment-api"
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services:"
Write-Host "  API:        http://localhost:8000"
Write-Host "  Docs:       http://localhost:8000/docs"
Write-Host "  Health:     http://localhost:8000/health"
Write-Host "  Prometheus: http://localhost:9090"
Write-Host "  Grafana:    http://localhost:3000 (admin/admin)"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  View logs:    docker-compose -f phase5_production_deployment\docker-compose.yml logs -f"
Write-Host "  Stop:         docker-compose -f phase5_production_deployment\docker-compose.yml down"
Write-Host "  Restart:      docker-compose -f phase5_production_deployment\docker-compose.yml restart"
Write-Host ""
