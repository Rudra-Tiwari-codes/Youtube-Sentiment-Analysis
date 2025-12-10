"""
Phase 3: DistilBERT Fine-Tuning with HuggingFace Trainer
Optimized for Windows + Python 3.12

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
set_seed(42)

project_root = Path(__file__).parent.parent


def load_data():
    """Load train/val/test datasets"""
    logger.info("Loading data...")
    data_dir = project_root / 'data' / 'processed'
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
    
    # Load label encoder
    label_encoder = joblib.load(project_root / 'models' / 'label_encoder.pkl')
    
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    return train_df, val_df, test_df, label_encoder


def prepare_datasets(train_df, val_df, test_df, label_encoder, tokenizer):
    """Prepare HuggingFace datasets"""
    logger.info("Preparing datasets...")
    
    # Encode labels
    train_df['label'] = label_encoder.transform(train_df['sentiment'])
    val_df['label'] = label_encoder.transform(val_df['sentiment'])
    test_df['label'] = label_encoder.transform(test_df['sentiment'])
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['cleaned_text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['cleaned_text', 'label']])
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['cleaned_text'],
            padding='max_length',
            truncation=True,
            max_length=256  # Reduced for speed (my PC's broke)
        )
    
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['cleaned_text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['cleaned_text'])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['cleaned_text'])
    
    train_dataset.set_format('torch')
    val_dataset.set_format('torch')
    test_dataset.set_format('torch')
    
    logger.info("Datasets prepared")
    
    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def main():
    logger.info("=" * 60)
    logger.info("Phase 3: DistilBERT Fine-Tuning")
    logger.info("Target: Beat 88.57% baseline")
    logger.info("=" * 60)
    
    # Load data
    train_df, val_df, test_df, label_encoder = load_data()
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_df, val_df, test_df, label_encoder, tokenizer
    )
    
    # Load model
    logger.info("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    )
    
    # Training arguments
    output_dir = project_root / 'phase3_transformer_models' / 'checkpoints' / 'distilbert'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=str(project_root / 'logs'),
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        save_total_limit=2,
        report_to='none',
        fp16=False  # Disable for CPU
    )
    
    # Initialize trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    train_result = trainer.train()
    
    logger.info("\nTraining complete!")
    logger.info(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on test set...")
    logger.info("=" * 60)
    
    test_results = trainer.predict(test_dataset)
    test_metrics = test_results.metrics
    
    logger.info(f"\nTest Accuracy: {test_metrics['test_accuracy']:.4f} ({test_metrics['test_accuracy']*100:.2f}%)")
    logger.info(f"F1 Macro: {test_metrics['test_f1_macro']:.4f}")
    logger.info(f"F1 Weighted: {test_metrics['test_f1_weighted']:.4f}")
    
    # Detailed classification report
    predictions = np.argmax(test_results.predictions, axis=1)
    report = classification_report(
        test_dataset['label'],
        predictions,
        target_names=label_encoder.classes_,
        digits=4
    )
    
    logger.info("\nClassification Report:")
    logger.info("\n" + report)
    
    # Save results
    results_dir = project_root / 'phase3_transformer_models' / 'evaluation'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    with open(results_dir / 'distilbert_results.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Distilbert results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}\n")
        f.write(f"F1 Macro: {test_metrics['test_f1_macro']:.4f}\n")
        f.write(f"F1 Weighted: {test_metrics['test_f1_weighted']:.4f}\n")
        f.write(f"Training Time: {train_result.metrics['train_runtime']:.2f}s\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Compare with baseline
    logger.info("\n" + "=" * 60)
    logger.info("Comparison with baseline:")
    logger.info("=" * 60)
    logger.info(f"Baseline (Linear SVM): 88.57%")
    logger.info(f"DistilBERT: {test_metrics['test_accuracy']*100:.2f}%")
    
    improvement = (test_metrics['test_accuracy'] - 0.8857) * 100
    if improvement > 0:
        logger.info(f"[SUCCESS] Improvement: +{improvement:.2f}%")
    else:
        logger.info(f"[WARNING] Change: {improvement:.2f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 60)
    
    model_path = output_dir / 'final_model'
    trainer.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
