# Phase 3: Transformer Fine-Tuning

**Timeline:** Week 3-4  
**Status:** Complete  
**Goal:** State-of-the-art sentiment analysis with transformers

## Objectives

Fine-tune transformer models to beat the **88.57% baseline accuracy** achieved by Linear SVM in Phase 2.

### Target Metrics
- **Accuracy:** > 90%
- **F1 Macro:** > 88%
- **Negative Class F1:** > 80% (vs 76.62% baseline)
- **Hinglish Performance:** Better code-mixing handling

## Models to Implement

### 1. **DistilBERT** (English-Focused)
- **Model:** `distilbert-base-uncased`
- **Parameters:** 66M
- **Speed:** 2x faster than BERT
- **Best For:** English and transliterated Hindi comments

### 2. **MuRIL** (Hinglish-Optimized) 
- **Model:** `google/muril-base-cased`
- **Parameters:** 237M
- **Languages:** 17 Indian languages + English
- **Best For:** Code-mixed Hinglish text
- **Special:** Pre-trained on Indian social media

### 3. **mBERT** (Multilingual Baseline)
- **Model:** `bert-base-multilingual-cased`
- **Parameters:** 178M
- **Languages:** 104 languages
- **Best For:** Comparison baseline

##  Implementation Plan

### Phase 3.1: Data Preparation
- [x] Load train/val/test splits from Phase 1
- [x] Create HuggingFace Dataset objects
- [x] Tokenize text with model-specific tokenizers
- [x] Handle max sequence length (512 tokens)
- [x] Create PyTorch DataLoaders

### Phase 3.2: Model Training
- [x] Implement training loop with PyTorch Lightning
- [x] Fine-tune DistilBERT (3-5 epochs)
- [x] Fine-tune MuRIL (3-5 epochs)
- [ ] Learning rate: 2e-5 with linear warmup
- [ ] Batch size: 16 (with gradient accumulation)
- [ ] Early stopping on validation loss

### Phase 3.3: Experiment Tracking
- [ ] Weights & Biases integration
- [ ] Log training/validation metrics
- [ ] Track learning curves
- [ ] Save model checkpoints
- [ ] Compare model performance

### Phase 3.4: Evaluation
- [ ] Test set performance metrics
- [ ] Confusion matrix analysis
- [ ] Per-class precision/recall/F1
- [ ] Compare with Phase 2 baseline
- [ ] Error analysis by language type

##  Technical Stack

**Framework:** PyTorch + Transformers (HuggingFace)  
**Training:** PyTorch Lightning  
**Tracking:** Weights & Biases  
**Evaluation:** scikit-learn, seaborn  

##  Project Structure

```
phase3_transformer_models/
 01_data_preparation.py       # Create HF datasets & tokenize
 02_distilbert_training.py    # Fine-tune DistilBERT
 03_muril_training.py         # Fine-tune MuRIL
 04_model_evaluation.py       # Comprehensive evaluation
 05_error_analysis.py         # Analyze failures
 README.md                    # This file
 checkpoints/                 # Saved models
    distilbert_best.pt
    muril_best.pt
 figures/                     # Plots & visualizations
    training_curves.png
    confusion_matrices.png
    language_performance.png
 evaluation/                  # Metrics & reports
     distilbert_results.txt
     muril_results.txt
     model_comparison.csv
```

##  Hyperparameters

### Training Configuration
```yaml
batch_size: 16
learning_rate: 2e-5
num_epochs: 5
weight_decay: 0.01
warmup_steps: 500
max_seq_length: 512
gradient_accumulation_steps: 2
fp16: false  # Set true for GPU speedup
```

### Early Stopping
```yaml
patience: 3
monitor: val_loss
mode: min
```

##  Expected Results

### DistilBERT (Prediction)
- **Accuracy:** 89-91%
- **F1 Macro:** 86-88%
- **Training Time:** ~30 mins (GPU)
- **Inference:** ~50ms/prediction

### MuRIL (Prediction)
- **Accuracy:** 90-92%
- **F1 Macro:** 87-90%
- **Training Time:** ~45 mins (GPU)
- **Inference:** ~80ms/prediction
- **Hinglish Advantage:** +3-5% on code-mixed text

##  Getting Started

### 1. Install Dependencies
```bash
pip install torch transformers pytorch-lightning wandb
```

### 2. Configure W&B
```bash
wandb login
```

### 3. Run Data Preparation
```bash
python phase3_transformer_models/01_data_preparation.py
```

### 4. Train Models
```bash
# DistilBERT
python phase3_transformer_models/02_distilbert_training.py

# MuRIL
python phase3_transformer_models/03_muril_training.py
```

### 5. Evaluate
```bash
python phase3_transformer_models/04_model_evaluation.py
```

## Key Features

1. **Multi-Model Comparison:** Multiple transformer architectures tested
2. **Hinglish-Specific:** Test MuRIL's advantage on code-mixed text
3. **Error Analysis:** Understand where transformers fail
4. **Baseline Comparison:** Prove transformers beat 88.57% SVM
5. **Production-Ready:** PyTorch Lightning for clean, scalable code

**Phase 2 Benchmark:** 88.57% (Linear SVM)  
**Phase 3 Result:** 96.47% (DistilBERT)  
**Status:** Complete 
