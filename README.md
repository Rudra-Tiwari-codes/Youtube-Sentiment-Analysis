# YouTube Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sentiment analysis on 131K+ YouTube comments about Indian employment, with 96.47% accuracy using DistilBERT. Handles Hinglish (English-Hindi code-mixed) text.

## Quick Start

`ash
git clone https://github.com/Rudra-Tiwari-codes/Youtube-Sentiment-Analysis.git
cd Youtube-Sentiment-Analysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run API
cd phase5_production_deployment
python 01_fastapi_server.py
`

Visit http://localhost:8000/docs

## Project Structure

- phase1_data_engineering/ - Data cleaning and EDA
- phase2_baseline_models/ - Classical ML (SVM, LogReg)
- phase3_transformer_models/ - DistilBERT fine-tuning
- phase4_explainability/ - SHAP analysis and error analysis
- phase5_production_deployment/ - FastAPI + Docker

## Results

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| Logistic Regression | 87.25% | 84.05% |
| Linear SVM | 88.57% | 85.89% |
| DistilBERT | 96.47% | 95.63% |

## Tech Stack

ML: PyTorch, HuggingFace Transformers, Scikit-learn
Data: Pandas, NumPy
API: FastAPI, Uvicorn
Deployment: Docker, docker-compose

## License

MIT License
