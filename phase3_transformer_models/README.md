# Phase 3: Transformer Fine-Tuning

## Overview

Fine-tuned transformer models on 131K YouTube comments to achieve sentiment classification on Hinglish (Hindi-English code-mixed) text.

## Results

| Model | Accuracy | F1-Macro | F1-Weighted | Improvement |
|-------|----------|----------|-------------|-------------|
| **DistilBERT** | **90.41%** | **88.30%** | **90.28%** | **+1.84%** |
| MuRIL | 90.42% | 88.30% | 90.29% | +1.85% |
| Linear SVM (baseline) | 88.57% | 85.89% | 88.35% | - |

### Per-Class Performance (DistilBERT)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 86.72% | 75.67% | 80.82% | 2,088 |
| Neutral | 89.55% | 96.47% | 92.88% | 5,721 |
| Positive | 92.74% | 89.69% | 91.19% | 5,352 |

**Key Achievement:** +7.48% F1 improvement on negative sentiment detection (73.34% → 80.82%)

## Models Evaluated

### 1. DistilBERT
- **Base Model:** `distilbert-base-uncased`
- **Parameters:** 66M (3.6× smaller than MuRIL)
- **Advantages:** Faster inference, comparable accuracy to MuRIL
- **Best For:** English and transliterated Hindi (Hinglish)

### 2. MuRIL
- **Base Model:** `google/muril-base-cased`
- **Parameters:** 237M
- **Languages:** 17 Indian languages + English
- **Performance:** Nearly identical to DistilBERT (90.42% vs 90.41%)

## Training Details

**Training Environment:** Google Colab (TPU/GPU)  
**Dataset:** 131,608 YouTube comments
- Train: 105,286 samples (80%)
- Validation: 13,161 samples (10%)
- Test: 13,161 samples (10%)

**Hyperparameters:**
```yaml
learning_rate: 2e-5
batch_size: 16
epochs: 3-5 (early stopping)
max_sequence_length: 256 tokens
optimizer: AdamW
weight_decay: 0.01
warmup_steps: 500
```

## Implementation

### Scripts
- `01_data_preparation.py` - Tokenization and HuggingFace dataset creation
- `02_distilbert_training.py` - DistilBERT fine-tuning (PyTorch)
- `04_distilbert_training_real.py` - Full training implementation
- `05_distilbert_hf_trainer.py` - HuggingFace Trainer API version
- `generate_visualizations.py` - Create evaluation charts

### Visualizations Generated
- `training_history.png` - Loss/accuracy curves
- `confusion_matrix_distilbert.png` - Classification confusion matrix
- `learning_curves_data_efficiency.png` - Data efficiency analysis
- `per_class_performance.html` - Interactive performance breakdown
## Key Findings

1. **DistilBERT vs MuRIL:** Nearly identical performance despite 3.6× parameter difference
2. **Hinglish Handling:** Pre-trained English models effective on transliterated Hindi
3. **Negative Class:** Most challenging (15.86% of dataset), showed largest improvement
4. **Production Choice:** DistilBERT selected for lower computational cost

### Load Tokenized Data
```python
from datasets import load_from_disk

dataset = load_from_disk("phase3_transformer_models/data/distilbert_base_uncased_datasets")
train_data = dataset['train']
test_data = dataset['test']
```

### Generate Visualizations
```bash
cd phase3_transformer_models
python generate_visualizations.py
```