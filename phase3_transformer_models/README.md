# Phase 3: Transformer Models

DistilBERT fine-tuning for sentiment classification.

## Scripts

1. `01_data_preparation.py` - Tokenize data for transformers
2. `05_distilbert_hf_trainer.py` - Fine-tune DistilBERT
3. `06_distilbert_verification.py` - Multi-seed re-run with per-language, VADER-confidence and
   leakage analysis for the paper (run via `Phase3_DistilBERT_Verification_Colab.ipynb`)

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
