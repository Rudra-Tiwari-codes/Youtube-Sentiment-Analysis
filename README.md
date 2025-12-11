# YouTube Sentiment Analysis: Indian Employment Discourse

A comprehensive sentiment analysis system for YouTube comments on Indian employment topics. Processes 131,608 comments using classical machine learning and transformer models, achieving 96.47% accuracy with DistilBERT.

## Research Overview

### Objective
Analyze public sentiment on Indian employment through YouTube comments, handling Hinglish (English-Hindi code-mixed) text with production-ready deployment.

### Dataset
- **Source**: YouTube comments on employment-related videos
- **Size**: 131,608 comments
- **Language**: Hinglish (English-Hindi code-mixed)
- **Classes**: Negative, Neutral, Positive
- **Split**: 80% train, 10% validation, 10% test

### Key Findings

**Model Performance Comparison**

| Model | Accuracy | F1-Macro | F1-Weighted | Parameters |
|-------|----------|----------|-------------|------------|
| Naive Bayes | 71.39% | 67.23% | 70.92% | Multinomial |
| Random Forest | 73.16% | 68.45% | 72.54% | 100 trees |
| Gradient Boosting | 75.43% | 71.25% | 74.89% | 100 estimators |
| Logistic Regression | 87.25% | 85.12% | 86.98% | 10K features |
| Linear SVM | 88.57% | 86.63% | 88.34% | 10K features |
| **DistilBERT** | **96.47%** | **95.63%** | **96.47%** | **66.9M** |

**Research Contributions**
1. **Performance Improvement**: 7.90% accuracy gain over best baseline (Linear SVM)
2. **Multilingual Handling**: Effective processing of code-mixed Hinglish text
3. **Error Analysis**: Identified confusion patterns between Neutral/Positive classes
4. **Production System**: Complete deployment pipeline with Docker and FastAPI
5. **Explainability**: SHAP analysis revealing key sentiment indicators

### Architecture

The system follows a five-phase pipeline:

1. **Data Engineering**: Collection, cleaning, language detection, sentiment labeling
2. **Baseline Models**: Classical ML with TF-IDF features
3. **Transformer Models**: DistilBERT fine-tuning
4. **Explainability**: SHAP analysis and error patterns
5. **Production Deployment**: REST API with monitoring

## Quick Start

### Installation

```bash
git clone https://github.com/Rudra-Tiwari-codes/Youtube-Sentiment-Analysis.git
cd Youtube-Sentiment-Analysis
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run API Server

```bash
cd phase5_production_deployment
python 01_fastapi_server.py
```

Visit http://localhost:8000/docs for interactive API documentation.

### Docker Deployment

```bash
# Windows
.\phase5_production_deployment\deploy.ps1 -DeployType cpu

# Linux/Mac
bash phase5_production_deployment/deploy.sh cpu
```

## Project Structure

```
Youtube-Sentiment-Analysis/
├── phase1_data_engineering/       # Data processing pipeline
│   ├── 01_export_data.py
│   ├── 02_clean_data.py
│   ├── 03_language_detection.py
│   ├── 04_sentiment_labeling.py
│   ├── 05_train_test_split.py
│   ├── 06_eda_analysis.py
│   └── figures/                   # 5 EDA visualizations
├── phase2_baseline_models/        # Classical ML models
│   ├── 01_tfidf_baseline.py
│   ├── 02_classical_models.py
│   ├── evaluation/
│   └── figures/                   # 11 performance plots
├── phase3_transformer_models/     # Deep learning models
│   ├── 01_data_preparation.py
│   ├── 05_distilbert_hf_trainer.py
│   ├── checkpoints/distilbert/    # Trained model (255MB)
│   └── figures/                   # 4 training visualizations
├── phase4_explainability/         # Model interpretation
│   ├── 00_generate_predictions.py
│   ├── 01_error_analysis.py
│   ├── 02_shap_analysis.py
│   ├── 03_model_comparison.py
│   ├── figures/                   # 7 SHAP plots
│   └── reports/                   # 4 analysis reports
└── phase5_production_deployment/  # Production API
    ├── 01_fastapi_server.py
    ├── 02_api_client.py
    ├── 03_load_testing.py
    ├── Dockerfile
    └── docker-compose.yml
```

## Research Methodology

### Phase 1: Data Engineering
- **Data Collection**: YouTube API scraping with keyword targeting
- **Preprocessing**: Text normalization, URL removal, special character handling
- **Language Detection**: Automated English/Hindi/Mixed classification
- **Labeling**: VADER sentiment analysis for initial labels
- **EDA**: Distribution analysis, temporal trends, length patterns

### Phase 2: Baseline Models
- **Feature Engineering**: TF-IDF with 10,000 features
- **Models Tested**: Naive Bayes, Random Forest, Gradient Boosting, Logistic Regression, Linear SVM
- **Best Baseline**: Linear SVM (88.57% accuracy)
- **Insights**: Traditional ML struggles with code-mixed text and nuanced sentiments

### Phase 3: Transformer Models
- **Model**: DistilBERT (distilbert-base-uncased)
- **Fine-tuning**: 3 epochs, learning rate 2e-5, batch size 16
- **Hardware**: GPU training (NVIDIA CUDA)
- **Training Time**: ~30 minutes on GPU
- **Result**: 96.47% accuracy, 95.63% macro F1

### Phase 4: Explainability
- **Error Analysis**: 465 misclassifications analyzed
- **Confusion Patterns**: Neutral-Positive confusion (1.85%), Negative-Positive confusion (5.70%)
- **SHAP Analysis**: Token-level importance for predictions
- **Model Comparison**: Cross-phase performance evaluation

### Phase 5: Production Deployment
- **Framework**: FastAPI with Uvicorn
- **Containerization**: Docker with CPU/GPU support
- **Performance**: 50+ req/sec, 48ms avg latency
- **Monitoring**: Health checks, metrics, logging

## Research Results

### Per-Class Performance (DistilBERT)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 92.40% | 92.53% | 92.46% | 2,088 |
| Neutral | 98.56% | 97.05% | 97.80% | 5,721 |
| Positive | 95.86% | 97.38% | 96.62% | 5,352 |

### Error Distribution
- Total misclassifications: 465 (3.53%)
- Negative errors: 156 (7.47% of class)
- Neutral errors: 169 (2.95% of class)
- Positive errors: 140 (2.62% of class)

### Key Insights
1. **Neutral class** achieves highest precision (98.56%)
2. **Positive class** has highest recall (97.38%)
3. **Negative class** shows most confusion with Positive (5.70%)
4. Model performs consistently across all sentiment categories
5. Code-mixed text handled effectively without language-specific preprocessing

## API Usage

### Single Prediction

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "The job market is improving significantly!",
        "return_confidence": True
    }
)

print(response.json())
# {
#   "sentiment": "Positive",
#   "confidence": {"Negative": 0.01, "Neutral": 0.04, "Positive": 0.95},
#   "inference_time_ms": 45.2
# }
```

### Batch Prediction

```python
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": [
            "Great employment opportunities!",
            "High unemployment is concerning",
            "The situation is stable"
        ]
    }
)
```

## Performance Metrics

### Model Comparison

**Accuracy Improvement Over Baseline**
- vs Naive Bayes: +25.08%
- vs Random Forest: +23.31%
- vs Gradient Boosting: +21.04%
- vs Logistic Regression: +9.22%
- vs Linear SVM: +7.90%

**F1-Score Improvement**
- Macro F1: +9.00% over Linear SVM
- Weighted F1: +8.13% over Linear SVM

### Inference Performance
- CPU: ~50ms per prediction
- GPU: ~10ms per prediction
- Batch (100): ~2.5s on CPU
- Throughput: 50+ requests/second

## Technical Stack

**Machine Learning**
- PyTorch 2.1+
- HuggingFace Transformers
- Scikit-learn
- SHAP

**Data Processing**
- Pandas
- NumPy
- NLTK
- langdetect

**API & Deployment**
- FastAPI
- Uvicorn
- Docker
- Docker Compose

**Visualization**
- Matplotlib
- Seaborn
- Plotly

## Reproducibility

All experiments are reproducible with provided scripts:

```bash
# Phase 1: Data processing
python phase1_data_engineering/01_export_data.py
python phase1_data_engineering/02_clean_data.py
python phase1_data_engineering/03_language_detection.py
python phase1_data_engineering/04_sentiment_labeling.py
python phase1_data_engineering/05_train_test_split.py
python phase1_data_engineering/06_eda_analysis.py

# Phase 2: Baseline models
python phase2_baseline_models/01_tfidf_baseline.py
python phase2_baseline_models/02_classical_models.py

# Phase 3: Transformer training (requires GPU)
python phase3_transformer_models/01_data_preparation.py
python phase3_transformer_models/05_distilbert_hf_trainer.py

# Phase 4: Analysis
python phase4_explainability/00_generate_predictions.py
python phase4_explainability/01_error_analysis.py
python phase4_explainability/02_shap_analysis.py
python phase4_explainability/03_model_comparison.py
```

## Future Work

1. **Multilingual Models**: Test XLM-RoBERTa, mBERT for better Hinglish support
2. **Active Learning**: Iterative labeling for hard examples
3. **Aspect-Based Analysis**: Extract specific employment topics (wages, opportunities, policies)
4. **Temporal Analysis**: Track sentiment trends over time
5. **Multi-modal**: Incorporate video metadata and engagement metrics
6. **Model Compression**: Quantization and distillation for faster inference
7. **Real-time Processing**: Stream processing for live comments

## Citation

If you use this work in your research, please cite:

```bibtex
@software{youtube_sentiment_analysis_2025,
  author = {Tiwari, Rudra},
  title = {YouTube Sentiment Analysis: Indian Employment Discourse},
  year = {2025},
  url = {https://github.com/Rudra-Tiwari-codes/Youtube-Sentiment-Analysis}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- HuggingFace for transformer models and infrastructure
- YouTube Data API for data access
- Open-source ML community for tools and libraries

## Contact

For questions or collaboration:
- GitHub: [@Rudra-Tiwari-codes](https://github.com/Rudra-Tiwari-codes)
- Repository: [Youtube-Sentiment-Analysis](https://github.com/Rudra-Tiwari-codes/Youtube-Sentiment-Analysis)

---

**Last Updated**: December 11, 2025  
**Status**: Production-Ready  
**Model Version**: 1.0.0
