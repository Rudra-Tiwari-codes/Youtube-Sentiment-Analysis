# Phase 4: Explainability

SHAP analysis and error analysis for model interpretability.

## Scripts

1. `00_generate_predictions.py` - Generate predictions on test set
2. `01_error_analysis.py` - Analyze misclassifications
3. `02_shap_analysis.py` - SHAP values for feature importance
4. `03_model_comparison.py` - Compare baseline vs transformer

## Output

- `figures/` - SHAP plots and error analysis visualizations
- `reports/` - Error analysis reports

## Usage

```bash
python 00_generate_predictions.py
python 01_error_analysis.py
python 02_shap_analysis.py
python 03_model_comparison.py
```
