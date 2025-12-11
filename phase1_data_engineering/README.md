# Phase 1: Data Engineering

Process 131,608 YouTube comments about Indian employment.

## Scripts

1. `01_export_data.py` - Export from SQLite database
2. `02_clean_data.py` - Clean and normalize text
3. `03_language_detection.py` - Detect language (English/Hindi/Mixed)
4. `04_sentiment_labeling.py` - Label sentiments using VADER
5. `05_train_test_split.py` - Split into train/val/test (80/10/10)
6. `06_eda_analysis.py` - Generate EDA visualizations

## Output

- `data/processed/comments_cleaned.csv`
- `data/processed/train.csv`, `val.csv`, `test.csv`
- `figures/` - EDA visualizations

## Usage

```bash
python 01_export_data.py
python 02_clean_data.py
python 03_language_detection.py
python 04_sentiment_labeling.py
python 05_train_test_split.py
python 06_eda_analysis.py
```
