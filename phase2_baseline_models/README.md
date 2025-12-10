# Phase 2: Baseline Models & Classical ML

## Results Summary

**Best Model:** Linear SVM - **88.57% accuracy**, 85.89% F1-macro  
**Baseline:** Logistic Regression - **87.25% accuracy**, 84.05% F1-macro

| Model | Accuracy | F1-Macro | F1-Weighted | Training Time |
|-------|----------|----------|-------------|---------------|
| **Linear SVM** | **88.57%** | **85.89%** | 88.35% | 5.5s |
| Logistic Regression | 87.25% | 84.05% | 86.90% | 8.4s |
| Gradient Boosting | 75.43% | 67.67% | 73.54% | 288s |
| Random Forest | 73.16% | 58.68% | 68.92% | 4.9s |
| Naive Bayes | 71.39% | 67.73% | 70.90% | 0.02s |

## Scripts

- `01_tfidf_baseline.py` - TF-IDF + Logistic Regression baseline
- `02_classical_models.py` - Train 4 additional classifiers (SVM, RF, GBoost, NB)
- `evaluation/` - Results and metrics
- `figures/` - Confusion matrices, ROC curves, feature importance

## Visualizations

- 6 confusion matrices (one per model)
- ROC curves with AUC scores
- Feature importance plots (Logistic Regression, SVM)
- Interactive model comparison chart
- Per-class performance metrics