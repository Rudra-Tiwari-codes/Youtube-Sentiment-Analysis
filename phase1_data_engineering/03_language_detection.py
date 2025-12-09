"""
Figure out what language each comment is in.
This script detects:
- Pure English comments
- Pure Hindi comments (Devanagari script)
- Mixed Hinglish comments (both languages)
"""

import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import logging
from langdetect import detect, DetectorFactory, LangDetectException
from tqdm import tqdm
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Make langdetect deterministic
DetectorFactory.seed = 42

# Enable tqdm pandas integration
tqdm.pandas()


class LanguageDetector:
    """
    Detects if text is English, Hindi, or mixed (Hinglish).
    """
    
    # Unicode ranges for Devanagari script (Hindi)
    DEVANAGARI_RANGE = r'[\u0900-\u097F]'
    
    # Unicode ranges for English/Latin script
    ENGLISH_RANGE = r'[a-zA-Z]'
    
    def __init__(self):
        self.stats = {
            'english': 0,
            'hindi': 0,
            'code_mixed': 0,
            'detection_failures': 0,
            'short_texts': 0
        }
    
    def has_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari (Hindi) characters"""
        if pd.isna(text):
            return False
        return bool(re.search(self.DEVANAGARI_RANGE, text))
    
    def has_english(self, text: str) -> bool:
        """Check if text contains English/Latin characters"""
        if pd.isna(text):
            return False
        return bool(re.search(self.ENGLISH_RANGE, text))
    
    def count_script_chars(self, text: str) -> dict:
        """Count characters by script type"""
        if pd.isna(text):
            return {'devanagari': 0, 'english': 0, 'other': 0}
        
        devanagari = len(re.findall(self.DEVANAGARI_RANGE, text))
        english = len(re.findall(self.ENGLISH_RANGE, text))
        total = len(text.replace(' ', ''))  # Exclude spaces
        other = total - devanagari - english
        
        return {
            'devanagari': devanagari,
            'english': english,
            'other': other,
            'total': total
        }
    
    def detect_language_basic(self, text: str) -> str:
        """
        Basic langdetect-based detection.
        Returns: 'en', 'hi', or 'unknown'
        """
        if pd.isna(text) or not text or len(text.strip()) < 3:
            return 'unknown'
        
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return 'unknown'
    
    def detect_language(self, text: str) -> str:
        """
        Advanced language detection with Hinglish support.
        Returns: 'English', 'Hindi', 'Code-Mixed', or 'Unknown'
        """
        if pd.isna(text) or not text or len(text.strip()) < 3:
            self.stats['short_texts'] += 1
            return 'Unknown'
        
        # Count characters by script
        script_counts = self.count_script_chars(text)
        total_chars = script_counts['total']
        
        if total_chars == 0:
            self.stats['short_texts'] += 1
            return 'Unknown'
        
        devanagari_pct = (script_counts['devanagari'] / total_chars) * 100
        english_pct = (script_counts['english'] / total_chars) * 100
        
        # Decision thresholds
        PURE_THRESHOLD = 80  # 80%+ of one script = pure language
        MIXED_THRESHOLD = 15  # 15%+ of each script = code-mixed
        
        # Detect code-mixing
        if devanagari_pct >= MIXED_THRESHOLD and english_pct >= MIXED_THRESHOLD:
            self.stats['code_mixed'] += 1
            return 'Code-Mixed'
        
        # Primarily Hindi
        if devanagari_pct >= PURE_THRESHOLD:
            self.stats['hindi'] += 1
            return 'Hindi'
        
        # Primarily English
        if english_pct >= PURE_THRESHOLD:
            self.stats['english'] += 1
            return 'English'
        
        # Use langdetect as fallback for transliterated text
        try:
            lang_code = self.detect_language_basic(text)
            
            if lang_code == 'hi':
                self.stats['hindi'] += 1
                return 'Hindi'
            elif lang_code == 'en':
                self.stats['english'] += 1
                return 'English'
            else:
                # If langdetect says it's another language, treat as code-mixe
                self.stats['code_mixed'] += 1
                return 'Code-Mixed'
        except:
            self.stats['detection_failures'] += 1
            return 'Unknown'
    
    def get_statistics(self) -> dict:
        """Return detection statistics"""
        return self.stats.copy()


def detect_languages(input_path: str, output_path: str) -> dict:
    """
    Detect languages for all comments in the dataset
    """
    start_time = datetime.now()
    logger.info(f"Starting language detection: {input_path}")
    
    # Load data
    logger.info("Loading cleaned data...")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count:,} records")
    
    # Create detector instance
    detector = LanguageDetector()
    
    # Apply language detection with progress bar
    logger.info("Detecting languages...")
    df['language'] = df['cleaned_text'].progress_apply(detector.detect_language)
    
    # Calculate language distribution
    language_dist = df['language'].value_counts()
    language_pct = (df['language'].value_counts(normalize=True) * 100).round(2)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Language-labeled data saved to {output_file}")
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    detection_stats = detector.get_statistics()
    
    stats = {
        'total_records': initial_count,
        'language_distribution': language_dist.to_dict(),
        'language_percentages': language_pct.to_dict(),
        'detection_stats': detection_stats,
        'processing_duration_seconds': duration,
        'records_per_second': initial_count / duration,
        'output_file': str(output_file),
        'file_size_mb': output_file.stat().st_size / (1024 * 1024)
    }
    
    return stats


def main():
    INPUT_PATH = "./data/processed/comments_cleaned.csv"
    OUTPUT_PATH = "./data/processed/comments_language_labeled.csv"
    
    logger.info("="*80)
    logger.info("Language Detection")
    logger.info("="*80)
    
    # Verify input exists
    if not Path(INPUT_PATH).exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        return
    
    # Detect languages
    stats = detect_languages(INPUT_PATH, OUTPUT_PATH)
    
    logger.info("="*80)
    logger.info("Language Detection Complete")
    logger.info("="*80)
    
    # Print summary (again optional)
    '''
    print("\n" + "="*80)
    print(" LANGUAGE DETECTION SUMMARY")
    print("="*80)
    print(f" Processed {stats['total_records']:,} records in {stats['processing_duration_seconds']:.2f}s")
    print(f" Speed: {stats['records_per_second']:.0f} records/second")
    print(f"\n Language Distribution:")
    for lang, count in stats['language_distribution'].items():
        pct = stats['language_percentages'][lang]
        print(f"   {lang:12} : {count:6,} records ({pct:5.2f}%)")
    
    print(f"\n Detection Quality:")
    print(f"   Short texts (< 3 chars): {stats['detection_stats']['short_texts']:,}")
    print(f"   Detection failures: {stats['detection_stats']['detection_failures']:,}")
    
    print(f"\n Output: {stats['output_file']} ({stats['file_size_mb']:.2f} MB)")
    print("="*80 + "\n")
    '''


if __name__ == "__main__":
    main()
