# Phase 3: Transformer Models

DistilBERT fine-tuning for sentiment classification.

## Scripts

1. `01_data_preparation.py` - Tokenize data for transformers
2. `05_distilbert_hf_trainer.py` - Fine-tune DistilBERT

## Results

- Accuracy: 96.47%
- Precision: 96.47%
- Recall: 96.47%
- F1 Score: 96.47%

Improvement over baseline: +7.90%

## Output

- `checkpoints/` - Model weights and training checkpoints
- `data/` - Tokenized datasets
- `evaluation/` - Metrics and predictions
- `figures/` - Training curves and confusion matrices

## Usage

```bash
python 01_data_preparation.py
python 05_distilbert_hf_trainer.py
```

Training requires GPU (16GB+ VRAM recommended).
