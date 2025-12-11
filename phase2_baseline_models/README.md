# Phase 2: Baseline Models

Classical ML models for sentiment classification.

## Scripts

1. `01_tfidf_baseline.py` - TF-IDF + Logistic Regression
2. `02_classical_models.py` - Compare Logistic Regression, Random Forest, SVM, XGBoost

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 87.23% | 87.18% | 87.23% | 87.20% |
| Random Forest | 85.91% | 86.15% | 85.91% | 85.88% |
| Linear SVM | 88.57% | 88.54% | 88.57% | 88.55% |
| XGBoost | 86.45% | 86.52% | 86.45% | 86.42% |

Linear SVM is the best baseline model.

## Output

- `evaluation/` - Model metrics and reports
- `figures/` - Confusion matrices and performance plots
- `models/` - Saved models

## Usage

```bash
python 01_tfidf_baseline.py
python 02_classical_models.py
```
