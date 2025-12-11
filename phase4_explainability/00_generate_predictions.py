"""
Generate Predictions for Error Analysis
Loads trained DistilBERT checkpoint and generates predictions on test set.
This is required for detailed error analysis with actual predictions.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent


class SentimentDataset(Dataset):
    """Simple dataset for inference"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def load_model_and_tokenizer():
    """Load trained DistilBERT checkpoint"""
    logger.info("Loading trained DistilBERT model...")
    
    checkpoint_dir = project_root / 'phase3_transformer_models' / 'checkpoints' / 'distilbert'
    
    # Check if checkpoint exists
    if not (checkpoint_dir / 'model.safetensors').exists() and not (checkpoint_dir / 'pytorch_model.bin').exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_dir}\n"
            "Please download the checkpoint from Google Colab first."
        )
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Device: {device}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device


def generate_predictions(model, tokenizer, device, test_df, batch_size=32):
    """Generate predictions on test set"""
    logger.info(f"Generating predictions for {len(test_df):,} samples...")
    
    # Create dataset and dataloader
    dataset = SentimentDataset(test_df['cleaned_text'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)


def save_predictions(test_df, predictions, probabilities, label_encoder):
    """Save predictions to file"""
    logger.info("Saving predictions...")
    
    # Create output dataframe
    output_df = test_df.copy()
    output_df['predicted_label'] = label_encoder.inverse_transform(predictions)
    output_df['predicted_class'] = predictions
    
    # Add probability columns
    for i, class_name in enumerate(label_encoder.classes_):
        output_df[f'prob_{class_name}'] = probabilities[:, i]
    
    # Add confidence and correctness
    output_df['confidence'] = probabilities.max(axis=1)
    output_df['correct'] = (predictions == output_df['sentiment_encoded'])
    
    # Save
    output_dir = project_root / 'phase4_explainability' / 'data'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = output_dir / 'test_predictions_distilbert.csv'
    output_df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved to: {output_path}")
    
    return output_df


def main():
    logger.info("="*70)
    logger.info("GENERATE DISTILBERT PREDICTIONS FOR ERROR ANALYSIS")
    logger.info("="*70)
    
    # Load test data
    logger.info("\nLoading test data...")
    data_dir = project_root / 'data' / 'processed'
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    # Load label encoder
    label_encoder = joblib.load(project_root / 'models' / 'label_encoder.pkl')
    test_df['sentiment_encoded'] = label_encoder.transform(test_df['sentiment'])
    
    logger.info(f"Test set: {len(test_df):,} samples")
    logger.info(f"Classes: {label_encoder.classes_}")
    
    # Load model
    try:
        model, tokenizer, device = load_model_and_tokenizer()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Generate predictions
    predictions, probabilities = generate_predictions(
        model, tokenizer, device, test_df, batch_size=32
    )
    
    # Calculate accuracy
    accuracy = (predictions == test_df['sentiment_encoded']).mean()
    logger.info(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Save predictions
    output_df = save_predictions(test_df, predictions, probabilities, label_encoder)
    
    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total samples: {len(output_df):,}")
    logger.info(f"Correct predictions: {output_df['correct'].sum():,}")
    logger.info(f"Incorrect predictions: {(~output_df['correct']).sum():,}")
    logger.info(f"Overall accuracy: {accuracy*100:.2f}%")
    logger.info(f"\nMean confidence: {output_df['confidence'].mean():.4f}")
    logger.info(f"Confidence (correct): {output_df[output_df['correct']]['confidence'].mean():.4f}")
    logger.info(f"Confidence (incorrect): {output_df[~output_df['correct']]['confidence'].mean():.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("PREDICTIONS SAVED - Ready for error analysis!")
    logger.info("="*70)
    logger.info(f"Output: phase4_explainability/data/test_predictions_distilbert.csv")


if __name__ == "__main__":
    main()
