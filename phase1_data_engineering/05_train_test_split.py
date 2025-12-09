"""
Split data into training, validation, and test sets.

We split the data into 3 parts:
- 80% for training the model
- 10% for validation (tuning the model)
- 10% for final testing

Important: We make sure each split has the same mix of positive/negative/neutral
comments, so the model sees balanced examples
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Fixed seed for reproducibility
RANDOM_SEED = 42

def create_stratified_split(input_path: str, output_dir: str) -> dict:
    """
    Split data into train (80%), validation (10%), and test (10%) sets.
    """
    start_time = datetime.now()
    logger.info(f"Starting stratified split creation: {input_path}")
    
    # Load data
    logger.info(f"Loading labeled data: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count:,} records")
    
    # Stratify only by sentiment (simpler, more robust for small groups)
    # Language distribution will naturally be preserved with large sample size...we do this because my computer is slow and not that strong.
    logger.info(f"Sentiment distribution:")
    for sentiment, count in df['sentiment'].value_counts().items():
        logger.info(f"  {sentiment}: {count:,} records")
    
    # First split: 80% train, 20% temp (will become val + test)
    logger.info("Creating train/temp split...")
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df['sentiment']
    )
    
    # Second split: Split temp into 50/50 (giving us 10% val, 10% test of original)
    logger.info("Creating val/test split:")
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df['sentiment']
    )
    
    # No stratify_key to remove (we stratified only by sentiment)
    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(train_df):,} ({len(train_df)/initial_count*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df):,} ({len(val_df)/initial_count*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df):,} ({len(test_df)/initial_count*100:.1f}%)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_file = output_path / "train.csv"
    val_file = output_path / "val.csv"
    test_file = output_path / "test.csv"
    
    logger.info("Saving split files...")
    train_df.to_csv(train_file, index=False, encoding='utf-8')
    val_df.to_csv(val_file, index=False, encoding='utf-8')
    test_df.to_csv(test_file, index=False, encoding='utf-8')
    
    # Calculate distribution statistics
    def get_distribution_stats(df_split, split_name):
        """Calculate distribution statistics for a split"""
        sentiment_dist = df_split['sentiment'].value_counts(normalize=True) * 100
        language_dist = df_split['language'].value_counts(normalize=True) * 100
        
        return {
            'name': split_name,
            'count': len(df_split),
            'sentiment': sentiment_dist.to_dict(),
            'language': language_dist.to_dict()
        }
    
    train_stats = get_distribution_stats(train_df, 'Train')
    val_stats = get_distribution_stats(val_df, 'Validation')
    test_stats = get_distribution_stats(test_df, 'Test')
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    stats = {
        'total_records': initial_count,
        'train': train_stats,
        'val': val_stats,
        'test': test_stats,
        'train_file': str(train_file),
        'val_file': str(val_file),
        'test_file': str(test_file),
        'train_size_mb': train_file.stat().st_size / (1024 * 1024),
        'val_size_mb': val_file.stat().st_size / (1024 * 1024),
        'test_size_mb': test_file.stat().st_size / (1024 * 1024),
        'processing_duration_seconds': duration
    }
    
    return stats


def validate_split_quality(stats: dict):
    """
    Validate that stratification preserved distributions.
    Checks if train/val/test have similar distributions.
    """
    
    
    logger.info("Validating split quality by comparing distributions  : ")
    
    # Compare sentiment distributions
    sentiments = ['Positive', 'Negative', 'Neutral']
    logger.info("\nSentiment distribution comparison:")
    logger.info(f"{'Split':<12} {'Positive':>10} {'Negative':>10} {'Neutral':>10}")
    logger.info("-" * 50)
    
    for split_stats in [stats['train'], stats['val'], stats['test']]:
        name = split_stats['name']
        sent_dist = split_stats['sentiment']
        pos = sent_dist.get('Positive', 0)
        neg = sent_dist.get('Negative', 0)
        neu = sent_dist.get('Neutral', 0)
        logger.info(f"{name:<12} {pos:>9.2f}% {neg:>9.2f}% {neu:>9.2f}%")
    
    # Compare language distributions
    languages = ['English', 'Hindi', 'Code-Mixed', 'Unknown']
    logger.info("\nLanguage distribution comparison:")
    logger.info(f"{'Split':<12} {'English':>10} {'Hindi':>10} {'Code-Mixed':>12} {'Unknown':>10}")
    logger.info("-" * 60)
    
    for split_stats in [stats['train'], stats['val'], stats['test']]:
        name = split_stats['name']
        lang_dist = split_stats['language']
        eng = lang_dist.get('English', 0)
        hin = lang_dist.get('Hindi', 0)
        mix = lang_dist.get('Code-Mixed', 0)
        unk = lang_dist.get('Unknown', 0)
        logger.info(f"{name:<12} {eng:>9.2f}% {hin:>9.2f}% {mix:>11.2f}% {unk:>9.2f}%")
    
    logger.info("\n Split quality validation complete!")


def main():
    INPUT_PATH = "./data/processed/comments_labeled.csv"
    OUTPUT_DIR = "./data/processed"
    
    logger.info("="*80)
    logger.info("Training/Validation/Test Stratified Split") # STRATIFIEDDDDDDDDDDDD
    logger.info("="*80)
    
    # Verify input exists
    if not Path(INPUT_PATH).exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        return
    
    stats = create_stratified_split(INPUT_PATH, OUTPUT_DIR)
    
    # Validate split quality
    validate_split_quality(stats)
    
    logger.info("="*80)
    logger.info("STRATIFIED SPLIT COMPLETE ")
    logger.info("="*80)
    
    # Print summary
    '''
    print("\n" + "="*80)
    print(" STRATIFIED SPLIT SUMMARY")
    print("="*80)
    print(f" Total records: {stats['total_records']:,}")
    print(f" Processing time: {stats['processing_duration_seconds']:.2f}s")
    
    print(f"\n Train Set:")
    print(f"   Records: {stats['train']['count']:,}")
    print(f"   File: {stats['train_file']}")
    print(f"   Size: {stats['train_size_mb']:.2f} MB")
    
    print(f"\n Validation Set:")
    print(f"   Records: {stats['val']['count']:,}")
    print(f"   File: {stats['val_file']}")
    print(f"   Size: {stats['val_size_mb']:.2f} MB")
    
    print(f"\n Test Set:")
    print(f"   Records: {stats['test']['count']:,}")
    print(f"   File: {stats['test_file']}")
    print(f"   Size: {stats['test_size_mb']:.2f} MB")
    
    print("\n Stratification preserved sentiment and language distributions!")
    print("="*80 + "\n")

'''
if __name__ == "__main__":
    main()
