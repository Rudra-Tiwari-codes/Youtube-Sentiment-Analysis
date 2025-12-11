"""
DistilBERT Fine-Tuning
Proper transformer fine-tuning with PyTorch and HuggingFace Transformers.
This is a REAL implementation using DistilBERT for sentiment classification.
WARNING: This script requires a GPU for efficient training. Training on CPU may be very slow.
I strongly recommend using a machine with a CUDA-capable GPU OR GOOGLE'S TPU, which I used. 
A more reliable and faster training can be achieved with TPUss and I would recommend using the similar .ipynb notebook for TPU training.
which is present in the checkpoints folder. Use this model for inference and evaluation, and only use this if you have a really good PC.

"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import time
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent


class SentimentDataset(Dataset):
    """Custom Dataset for sentiment classification"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data():
    """Load train/val/test datasets"""
    logger.info("Loading data...")
    
    data_dir = project_root / 'data' / 'processed'
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    # Load label encoder
    label_encoder = joblib.load(project_root / 'models' / 'label_encoder.pkl')
    
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    return train_df, val_df, test_df, label_encoder


def create_data_loaders(train_df, val_df, test_df, label_encoder, tokenizer, batch_size=8):
    """Create PyTorch DataLoaders"""
    logger.info(f"Creating DataLoaders with batch_size={batch_size}...")
    
    # Encode labels
    train_labels = label_encoder.transform(train_df['sentiment'])
    val_labels = label_encoder.transform(val_df['sentiment'])
    test_labels = label_encoder.transform(test_df['sentiment'])
    
    # Create datasets
    train_dataset = SentimentDataset(
        train_df['cleaned_text'].values,
        train_labels,
        tokenizer
    )
    val_dataset = SentimentDataset(
        val_df['cleaned_text'].values,
        val_labels,
        tokenizer
    )
    test_dataset = SentimentDataset(
        test_df['cleaned_text'].values,
        test_labels,
        tokenizer
    )
    
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"[OK] DataLoaders created: {len(train_loader)} train batches")
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    avg_loss = total_loss / len(data_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': predictions,
        'true_labels': true_labels
    }


def train_model(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    """Full training loop"""
    logger.info("Starting training...")
    logger.info(f"Epochs: {epochs} | Learning Rate: {lr}")
    logger.info(f"Device: {device}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps
    )
    
    best_f1 = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val F1 Macro: {val_metrics['f1_macro']:.4f}")
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            logger.info(f"[BEST MODEL] New best! F1 Macro: {best_f1:.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"\n[COMPLETE] Training complete in {training_time:.2f}s ({training_time/60:.2f} mins)")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, training_time


def save_results(test_metrics, label_encoder, training_time):
    """Save evaluation results"""
    logger.info("\nSaving results...")
    
    output_dir = project_root / 'phase3_transformer_models' / 'evaluation'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Classification report
    report = classification_report(
        test_metrics['true_labels'],
        test_metrics['predictions'],
        target_names=label_encoder.classes_,
        digits=4
    )
    
    # Save detailed results
    with open(output_dir / 'distilbert_results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("Actual trained model results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"F1 Macro: {test_metrics['f1_macro']:.4f}\n")
        f.write(f"F1 Weighted: {test_metrics['f1_weighted']:.4f}\n")
        f.write(f"Training Time: {training_time:.2f}s ({training_time/60:.2f} mins)\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        cm = confusion_matrix(test_metrics['true_labels'], test_metrics['predictions'])
        f.write(str(cm))
    
    logger.info(f"[SUCCESS] Results saved to {output_dir}")
    
    return report


def main():
    logger.info("DistilBERT Fine-Tuning")
    logger.info("="*60)
    logger.info("Target: Beat 88.57% baseline accuracy")
    logger.info("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_df, val_df, test_df, label_encoder = load_data()
    
    # Load tokenizer
    logger.info("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create dataloaders (reduced batch size for CPU)
    batch_size = 4 if device.type == 'cpu' else 16
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, label_encoder, tokenizer, batch_size
    )
    
    # Load model
    logger.info("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(label_encoder.classes_)
    )
    model.to(device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model (reduced epochs for CPU)
    epochs = 2 if device.type == 'cpu' else 3
    model, training_time = train_model(
        model, train_loader, val_loader, device, epochs=epochs
    )
    
    # Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*60)
    test_metrics = evaluate(model, test_loader, device)
    
    logger.info(f"\nTest Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    logger.info(f"F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    
    report = save_results(test_metrics, label_encoder, training_time)
    logger.info("\nClassification Report:")
    logger.info("\n" + report)
    
    model_path = project_root / 'phase3_transformer_models' / 'checkpoints' / 'distilbert_best.pt'
    model_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"[SAVED] Model saved to {model_path}")
    logger.info("\n" + "="*60)
    logger.info("COMPARISON WITH BASELINE")
    logger.info("="*60)
    logger.info(f"Baseline (Linear SVM): 88.57% accuracy")
    logger.info(f"DistilBERT: {test_metrics['accuracy']*100:.2f}% accuracy")
    
    improvement = (test_metrics['accuracy'] - 0.8857) * 100
    if improvement > 0:
        logger.info(f"[SUCCESS] Improvement: +{improvement:.2f}%")
    else:
        logger.info(f"[WARNING] Change: {improvement:.2f}%")
    
    logger.info("\n" + "="*60)
    logger.info("[COMPLETE] DISTILBERT TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
