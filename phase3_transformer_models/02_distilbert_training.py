""" DistilBERT Fine-Tuning
Fine-tunes DistilBERT for sentiment classification.
Target: Beat 88.57% baseline accuracy.

"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent


def load_config():
    """Load configuration"""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prepared_data():
    """Load tokenized datasets"""
    logger.info("Loading prepared DistilBERT datasets...")
    
    data_dir = project_root / 'phase3_transformer_models' / 'data'
    
    # Load datasets
    datasets_path = data_dir / 'distilbert_base_uncased_datasets'
    if not datasets_path.exists():
        raise FileNotFoundError(
            f"Tokenized datasets not found at {datasets_path}\n"
            "Please run phase3_transformer_models/01_data_preparation.py first."
        )
    datasets = load_from_disk(str(datasets_path))
    
    # Load label mappings
    label_mappings_path = data_dir / 'label_mappings.pkl'
    if not label_mappings_path.exists():
        raise FileNotFoundError(
            f"Label mappings not found at {label_mappings_path}\n"
            "Please run phase3_transformer_models/01_data_preparation.py first."
        )
    label_mappings = joblib.load(label_mappings_path)
    
    logger.info(f" Datasets loaded")
    logger.info(f"   Train: {len(datasets['train']):,} samples")
    logger.info(f"   Val: {len(datasets['validation']):,} samples")
    logger.info(f"   Test: {len(datasets['test']):,} samples")
    
    return datasets, label_mappings


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }


def create_model(label_mappings):
    """Initialize DistilBERT model"""
    logger.info("Loading DistilBERT model...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(label_mappings['label2id']),
        id2label=label_mappings['id2label'],
        label2id=label_mappings['label2id']
    )
    
    logger.info(f" Model loaded: {model.num_parameters():,} parameters")
    
    return model


def train_model(model, datasets, config):
    """Train DistilBERT with Hugging Face Trainer"""
    logger.info("Setting up training...")
    
    output_dir = project_root / 'phase3_transformer_models' / 'checkpoints' / 'distilbert'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        num_train_epochs=config['training']['num_epochs'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_dir=str(project_root / 'logs' / 'distilbert'),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        save_total_limit=3,
        fp16=False,  # Set to True if using GPU
        report_to='none',  # Change to 'wandb' if using W&B
        seed=42,
        disable_tqdm=False
    )
    
    # Early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping']['patience']
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    
    # Train
    logger.info(" Starting training...")
    logger.info(f"   Epochs: {config['training']['num_epochs']}")
    logger.info(f"   Batch size: {config['training']['batch_size']}")
    logger.info(f"   Learning rate: {config['training']['learning_rate']}")
    
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    logger.info(f" Training complete in {training_time:.2f}s ({training_time/60:.2f} mins)")
    
    # Save best model
    trainer.save_model(str(output_dir / 'best_model'))
    logger.info(f" Best model saved to {output_dir / 'best_model'}")
    
    return trainer, training_time


def evaluate_model(trainer, datasets, label_mappings):
    """Evaluate model on test set"""
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*60)
    
    # Get predictions
    predictions = trainer.predict(datasets['test'])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Macro: {f1_macro:.4f}")
    logger.info(f"F1 Weighted: {f1_weighted:.4f}")
    
    # Classification report
    label_encoder = label_mappings['label_encoder']
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=label_encoder.classes_,
        digits=4
    )
    
    logger.info("\nClassification Report:")
    logger.info("\n" + report)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    logger.info("\nConfusion Matrix:")
    logger.info(f"Classes: {label_encoder.classes_.tolist()}")
    logger.info(f"\n{cm}")
    
    # Save results
    results = {
        'model': 'DistilBERT',
        'test_accuracy': accuracy,
        'test_f1_macro': f1_macro,
        'test_f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return results, pred_labels, true_labels


def save_results(results, training_time):
    """Save evaluation results"""
    output_dir = project_root / 'phase3_transformer_models' / 'evaluation'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save detailed results
    results_file = output_dir / 'distilbert_results.txt'
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DISTILBERT EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"F1 Macro: {results['test_f1_macro']:.4f}\n")
        f.write(f"F1 Weighted: {results['test_f1_weighted']:.4f}\n")
        f.write(f"Training Time: {training_time:.2f}s ({training_time/60:.2f} mins)\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(results['confusion_matrix']))
    
    logger.info(f"\n Results saved to {results_file}")


def main():
    logger.info(" Phase 3: DistilBERT Fine-Tuning")
    logger.info("="*60)
    logger.info("Target: Beat 88.57% baseline accuracy")
    logger.info("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load config
    config = load_config()
    
    # Load data
    datasets, label_mappings = load_prepared_data()
    
    # Create model
    model = create_model(label_mappings)
    
    # Train model
    trainer, training_time = train_model(model, datasets, config)
    
    # Evaluate on test set
    results, pred_labels, true_labels = evaluate_model(trainer, datasets, label_mappings)
    
    # Save results
    save_results(results, training_time)
    
    # Compare with baseline
    logger.info("\n" + "="*60)
    logger.info("Compared to baseline")
    logger.info("="*60)
    logger.info(f"Baseline (Linear SVM): 88.57% accuracy")
    logger.info(f"DistilBERT: {results['test_accuracy']*100:.2f}% accuracy")
    
    improvement = (results['test_accuracy'] - 0.8857) * 100
    if improvement > 0:
        logger.info(f" Improvement: +{improvement:.2f}%")
    else:
        logger.info(f"  Decrease: {improvement:.2f}%")
    
    logger.info("\n" + "="*60)
    logger.info("="*60)


if __name__ == "__main__":
    main()
