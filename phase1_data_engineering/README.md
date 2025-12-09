# Phase 1: Data Engineering & Hinglish Preparation

## Objectives

1. Export raw data from SQLite to CSV format
2. Build production-grade cleaning pipeline
3. Implement language detection (English/Hindi/Code-Mixed)
4. Add sentiment labels for supervised learning
5. Create stratified train/val/test split (80/10/10)
6. Generate comprehensive EDA visualizations

## Data Quality Issues Identified

### Input Data Challenges
- **Emojis**: 43,263 comments contained emojis (removed)
- **HTML Entities**: 2,214 instances of `&#39;`, `&quot;`, etc. (decoded)
- **User Mentions**: 223 `@username` patterns (removed)
- **URLs**: 683 URL instances (removed)
- **Empty Records**: 501 records became empty after cleaning (removed)
- **Inconsistent Punctuation**: Standardized across 131,608 records

### Language Distribution (Post-Detection)
- **English**: 124,157 records (94.34%)
- **Hindi**: 4,499 records (3.42%)
- **Code-Mixed (Hinglish)**: 2,865 records (2.18%)
- **Unknown**: 87 records (0.07%)

### Sentiment Distribution (VADER Labels)
- **Neutral**: 57,210 records (43.47%)
- **Positive**: 53,520 records (40.67%)
- **Negative**: 20,878 records (15.86%)

## Pipeline Architecture

```
SQLite DB (132k records)
    ↓
1. Data Export → CSV
    ↓
2. Text Cleaning
    - Demojize emojis
    - Decode HTML entities
    - Remove user handles
    - Standardize punctuation
    - Lowercase normalization
    ↓
3. Language Detection
    - langdetect + fastText
    - Tag: English/Hindi/Code-Mixed
    ↓
4. Sentiment Labeling
    - VADER for initial labels
    - Manual validation sample
    ↓
5. Stratified Split
    - Preserve sentiment ratio
    - Preserve language ratio
    - 80/10/10 split
    ↓
6. EDA & Visualization
    - Length vs sentiment correlation
    - Language distribution
    - Temporal patterns
```



### Visualizations
- `phase1_data_engineering/figures/language_distribution.html` - Interactive Plotly chart
- `phase1_data_engineering/figures/sentiment_distribution.html` - Sentiment balance
- `phase1_data_engineering/figures/length_vs_sentiment.html` - Correlation analysis
- `phase1_data_engineering/figures/temporal_trends.html` - Comments over time

### Processing Performance (for nerds)
- **Total Time:** ~90 seconds for 131,608 records
- **Export:** 2.30s (SQLite → CSV)
- **Cleaning:** 22.71s (5,795 records/second)
- **Language Detection:** 12.20s (10,791 records/second)
- **Sentiment Labeling:** 36.30s (3,626 records/second)
- **Train/Val/Test Split:** 7.83s
- **EDA Generation:** ~10s (5 interactive visualizations)

### Data Quality Achieved
- **Final Dataset:** 131,608 records (99.62% retention)
- **Unique Authors:** 120,602
- **Date Range:** 6+ years (2019-10-06 to 2025-12-08)
- **Average Comment Length:** 101.5 characters
- **VADER-TextBlob Correlation:** 0.5271

### Train/Val/Test Distribution
- **Train:** 105,286 records (80.0%) - 43.65 MB
- **Validation:** 13,161 records (10.0%) - 5.45 MB
- **Test:** 13,161 records (10.0%) - 5.47 MB

**Sentiment preservation in all splits:**
- Positive: 40.67%
- Negative: 15.86%
- Neutral: 43.47%