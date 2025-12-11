"""
Clean up messy YouTube comments.
We now clean:
- Weird HTML codes like &#39; (converts to regular quotes)
- Emojis (removes them)
- @mentions (removes them) 
- URLs (remove)
- Extra spaces and punctuation
The goal is to get nice, clean text that's ready for analysis.
"""

import pandas as pd
import re
import html
from pathlib import Path
from datetime import datetime
import logging
from emoji import demojize, replace_emoji
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Enable tqdm pandas integration
tqdm.pandas()
class TextCleaner:
    """
    Cleans up messy social media text.
    Removes emojis, mentions, URLs, and fixes HTML codes.
    Keeps track of what was cleaned for reporting.
    """
    
    def __init__(self):
        self.stats = {
            'html_entities_decoded': 0,
            'emojis_found': 0,
            'mentions_removed': 0,
            'urls_removed': 0,
            'empty_after_cleaning': 0
        }
    
    def decode_html_entities(self, text: str) -> str:
        """Fix HTML codes like &#39; back to normal characters"""
        if pd.isna(text):
            return text
        decoded = html.unescape(text)
        if decoded != text:
            self.stats['html_entities_decoded'] += 1
        return decoded
    
    def handle_emojis(self, text: str, mode: str = 'remove') -> str:
        """
        Remove emojis from text or convert them to words like :smile:
        """
        if pd.isna(text):
            return text
        
        # Count emojis for statistics
        original = text
        demojized = demojize(text)
        if demojized != original:
            self.stats['emojis_found'] += 1
        
        if mode == 'remove':
            return replace_emoji(text, replace='')
        elif mode == 'demojize':
            return demojized
        return text
    
    def remove_mentions(self, text: str) -> str:
        """Remove @username mentions"""
        if pd.isna(text):
            return text
        
        original = text
        # Pattern: @ followed by alphanumeric, underscore, or hyphen
        cleaned = re.sub(r'@[\w\-]+', '', text)
        
        if cleaned != original:
            self.stats['mentions_removed'] += 1
        
        return cleaned
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs and URL fragments"""
        if pd.isna(text):
            return text
        
        original = text
        # Pattern: http/https URLs
        cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Pattern: www URLs
        cleaned = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
        
        if cleaned != original:
            self.stats['urls_removed'] += 1
        
        return cleaned
    
    def standardize_punctuation(self, text: str) -> str:
        """Standardize punctuation and whitespace"""
        if pd.isna(text):
            return text
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Standardize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Apply full cleaning pipeline to text entry. 
        Pipeline: HTML → Emojis → Mentions → URLs → Punctuation
        """
        if pd.isna(text):
            return ""
        
        # Step 1: Decode HTML entities
        text = self.decode_html_entities(text)
        
        # Step 2: Handle emojis (remove for cleaner analysis)
        text = self.handle_emojis(text, mode='remove')
        
        # Step 3: Remove user mentions
        text = self.remove_mentions(text)
        
        # Step 4: Remove URLs
        text = self.remove_urls(text)
        
        # Step 5: Standardize punctuation
        text = self.standardize_punctuation(text)
        
        # Check if empty after cleaning
        if not text or text.isspace():
            self.stats['empty_after_cleaning'] += 1
            return ""
        
        return text
    
    def get_statistics(self) -> dict:
        """Return cleaning statistics"""
        return self.stats.copy()

def clean_dataset(input_path: str, output_path: str) -> dict:
    """Clean the entire dataset with progress tracking.
    Args:
        input_path: Path to raw CSV
        output_path: Path to save cleaned CSV.
    """
    start_time = datetime.now()
    logger.info(f"Starting data cleaning: {input_path}")
    
    # Load data
    logger.info("Loading raw data...")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count:,} records")
    
    # Create cleaner instance
    cleaner = TextCleaner()
    
    # Apply cleaning with progress bar
    logger.info("Applying cleaning pipeline...")
    df['cleaned_text'] = df['text'].progress_apply(cleaner.clean_text)
    
    # Calculate text length statistics
    df['original_length'] = df['text'].str.len()
    df['cleaned_length'] = df['cleaned_text'].str.len()
    df['length_reduction_pct'] = ((df['original_length'] - df['cleaned_length']) / df['original_length'] * 100).round(2)
    
    # Remove empty records
    empty_mask = df['cleaned_text'].str.strip() == ''
    empty_count = empty_mask.sum()
    
    if empty_count > 0:
        logger.warning(f"Removing {empty_count} empty records after cleaning")
        df = df[~empty_mask].copy()
    
    final_count = len(df)
    
    # Save cleaned data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Cleaned data saved to {output_file}")
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    cleaning_stats = cleaner.get_statistics()
    
    stats = {
        'initial_records': initial_count,
        'final_records': final_count,
        'removed_records': initial_count - final_count,
        'html_entities_decoded': cleaning_stats['html_entities_decoded'],
        'emojis_found': cleaning_stats['emojis_found'],
        'mentions_removed': cleaning_stats['mentions_removed'],
        'urls_removed': cleaning_stats['urls_removed'],
        'avg_length_reduction_pct': df['length_reduction_pct'].mean(),
        'processing_duration_seconds': duration,
        'records_per_second': final_count / duration,
        'output_file': str(output_file),
        'file_size_mb': output_file.stat().st_size / (1024 * 1024)
    }
    
    return stats


def main():
    INPUT_PATH = "./data/raw/youtube_comments_raw.csv"
    OUTPUT_PATH = "./data/processed/comments_cleaned.csv"
    
    logger.info("="*80)
    logger.info("PHASE 1 - STEP 2: TEXT CLEANING PIPELINE")
    logger.info("="*80)
    
    # Verify input
    if not Path(INPUT_PATH).exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        return

    stats = clean_dataset(INPUT_PATH, OUTPUT_PATH)
    
    logger.info("="*80)
    logger.info("Data Cleaned")
    logger.info("="*80)
    
    # Print summary (optional)
    '''
    print("\n" + "="*80)
    print("CLEANING SUMMARY")
    print("="*80)
    print(f" Processed {stats['initial_records']:,} records in {stats['processing_duration_seconds']:.2f}s")
    print(f" Speed: {stats['records_per_second']:.0f} records/second")
    print(f" HTML entities decoded: {stats['html_entities_decoded']:,}")
    print(f" Emojis found: {stats['emojis_found']:,}")
    print(f" User mentions removed: {stats['mentions_removed']:,}")
    print(f" URLs removed: {stats['urls_removed']:,}")
    print(f" Average text reduction: {stats['avg_length_reduction_pct']:.1f}%")
    print(f" Final dataset: {stats['final_records']:,} records ({stats['file_size_mb']:.2f} MB)")
    if stats['removed_records'] > 0:
        print(f"  Removed {stats['removed_records']:,} empty records")
    print(f" Output: {stats['output_file']}")
    print("="*80 + "\n")

'''


if __name__ == "__main__":
    main()
