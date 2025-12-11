"""
Data Preparation for Transformers
Prepares HuggingFace datasets and tokenizes text for transformer models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import joblib

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


def load_data():
    """Load train/val/test splits from Phase 1"""
    logger.info("Loading data from Phase 1...")
    
    data_dir = project_root / 'data' / 'processed'
    
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    logger.info(f"Train: {len(train_df):,} samples")
    logger.info(f"Val: {len(val_df):,} samples")
    logger.info(f"Test: {len(test_df):,} samples")
    
    return train_df, val_df, test_df


def prepare_label_mapping():
    """Create label mapping for transformers"""
    # Load label encoder from Phase 2
    model_dir = project_root / 'models'
    label_encoder_path = model_dir / 'label_encoder.pkl'
    
    if not label_encoder_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found at {label_encoder_path}\n"
            "Please run phase2_baseline_models/01_tfidf_baseline.py first to create it."
        )
    
    label_encoder = joblib.load(label_encoder_path)
    
    # Create label2id and id2label mappings
    label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    logger.info(f"Label mapping: {label2id}")
    
    return label2id, id2label, label_encoder


def create_hf_datasets(train_df, val_df, test_df, label_encoder):
    """Convert pandas DataFrames to HuggingFace Dataset objects"""
    logger.info("Creating HF Datasets from DataFrames ")
    
    # Prepare data dictionaries
    def prepare_split(df):
        return {
            'text': df['cleaned_text'].tolist(),
            'label': label_encoder.transform(df['sentiment']).tolist()
        }
    
    train_dict = prepare_split(train_df)
    val_dict = prepare_split(val_df)
    test_dict = prepare_split(test_df)
    
    # Create Dataset objects
    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)
    
    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    logger.info(" HuggingFace datasets created")
    logger.info(f"   Train: {len(train_dataset):,} samples")
    logger.info(f"   Val: {len(val_dataset):,} samples")
    logger.info(f"   Test: {len(test_dataset):,} samples")
    
    return dataset_dict


def tokenize_datasets(dataset_dict, model_name, max_length=512):
    """Tokenize datasets with model-specific tokenizer"""
    logger.info(f"Tokenizing with {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
    
    # Tokenize all splits
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing"
    )
    
    logger.info(f" Tokenization complete")
    
    return tokenized_datasets, tokenizer


def analyze_sequence_lengths(train_df):
    """Analyze text lengths to determine optimal max_length"""
    logger.info("Analyzing sequence lengths...")
    
    lengths = train_df['cleaned_text'].str.split().str.len()
    
    percentiles = [50, 75, 90, 95, 99, 100]
    for p in percentiles:
        val = np.percentile(lengths, p)
        logger.info(f"   {p}th percentile: {val:.0f} tokens")
    
    # Check how many exceed 512
    exceed_512 = (lengths > 512).sum()
    logger.info(f"   Comments > 512 tokens: {exceed_512:,} ({exceed_512/len(lengths)*100:.2f}%)")
    return lengths


def save_datasets(tokenized_datasets, tokenizer, model_name):
    """Save tokenized datasets and tokenizer"""
    output_dir = project_root / 'phase3_transformer_models' / 'data'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Clean model name for filename
    model_slug = model_name.replace('/', '_').replace('-', '_')
    
    # Save datasets
    dataset_path = output_dir / f'{model_slug}_datasets'
    tokenized_datasets.save_to_disk(str(dataset_path))
    logger.info(f" Datasets saved to {dataset_path}")
    
    # Save tokenizer
    tokenizer_path = output_dir / f'{model_slug}_tokenizer'
    tokenizer.save_pretrained(str(tokenizer_path))
    logger.info(f" Tokenizer saved to {tokenizer_path}")
    
    return dataset_path, tokenizer_path


def main():
    logger.info(" Data Preparation for Transformers")
    logger.info("="*60)
    
    # Load config
    config = load_config()
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Analyze sequence lengths
    analyze_sequence_lengths(train_df)
    
    # Prepare labels
    label2id, id2label, label_encoder = prepare_label_mapping()
    
    # Create HuggingFace datasets
    dataset_dict = create_hf_datasets(train_df, val_df, test_df, label_encoder)
    
    # Process for DistilBERT
    logger.info("\n" + "="*60)
    logger.info("Preparing DistilBERT Data")
    logger.info("="*60)
    distilbert_datasets, distilbert_tokenizer = tokenize_datasets(
        dataset_dict, 
        'distilbert-base-uncased',
        max_length=512
    )
    save_datasets(distilbert_datasets, distilbert_tokenizer, 'distilbert-base-uncased')
    
    # Process for MuRIL
    logger.info("\n" + "="*60)
    logger.info("PREPARING MURIL DATA")
    logger.info("="*60)
    muril_datasets, muril_tokenizer = tokenize_datasets(
        dataset_dict,
        'google/muril-base-cased',
        max_length=512
    )
    save_datasets(muril_datasets, muril_tokenizer, 'google/muril-base-cased')
    
    # Save label mappings (pickle format for label_encoder)
    label_mapping_path = project_root / 'phase3_transformer_models' / 'data' / 'label_mappings.pkl'
    joblib.dump({
        'label2id': label2id,
        'id2label': id2label,
        'label_encoder': label_encoder
    }, label_mapping_path)
    logger.info(f"\n Label mappings saved to {label_mapping_path}")
    
    logger.info("\n" + "="*60)
    logger.info(" Data prep done")
    logger.info("="*60)
    logger.info("Ready to train transformers")


if __name__ == "__main__":
    main()
