"""
Label each comment as Positive, Negative, or Neutral.
Uses VADER (a tool designed for social media) to automatically label comments.
This gives us training labels for our machine learning model.
Categories:
- Positive: Happy, optimistic comments
- Negative: Angry, frustrated comments
- Neutral: Factual or mixed sentiment
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Enable tqdm pandas integration
tqdm.pandas()


class SentimentLabeler:
    """
    Automatically label comments as positive, negative, or neutral. VADER does the heavy lifting.
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.stats = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total_processed': 0
        }
    
    def get_vader_sentiment(self, text: str) -> dict:
        if pd.isna(text) or not text:
            return {
                'neg': 0.0,
                'neu': 1.0,
                'pos': 0.0,
                'compound': 0.0
            }
        
        return self.vader.polarity_scores(text)
    
    def get_textblob_sentiment(self, text: str) -> dict:
        if pd.isna(text) or not text:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0
            }
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0
            }
    
    def classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on VADER compound score.
        
        Thresholds:
        - Positive: > 0.05
        - Negative: < -0.05
        - Neutral: -0.05 to 0.05
        """
        
        if compound_score >= 0.05:
            self.stats['positive'] += 1
            return 'Positive'
        elif compound_score <= -0.05:
            self.stats['negative'] += 1
            return 'Negative'
        else:
            self.stats['neutral'] += 1
            return 'Neutral'
    
    def label_sentiment(self, text: str) -> dict:
        """
        Generate comprehensive sentiment labels for a comment.
        Returns:
            dict with VADER scores, TextBlob scores, and final label
        """
        self.stats['total_processed'] += 1
        
        # Get VADER sentiment
        vader_scores = self.get_vader_sentiment(text)
        
        # Get TextBlob sentiment
        textblob_scores = self.get_textblob_sentiment(text)
        
        # Classify using VADER compound score
        sentiment_label = self.classify_sentiment(vader_scores['compound'])
        
        return {
            'sentiment': sentiment_label,
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_scores['polarity'],
            'textblob_subjectivity': textblob_scores['subjectivity']
        }
    
    def get_statistics(self) -> dict:
        """Return labeling statistics"""
        stats = self.stats.copy()
        total = stats['total_processed']
        if total > 0:
            stats['positive_pct'] = (stats['positive'] / total) * 100
            stats['negative_pct'] = (stats['negative'] / total) * 100
            stats['neutral_pct'] = (stats['neutral'] / total) * 100
        return stats


def label_sentiments(input_path: str, output_path: str) -> dict:
    """
    Label sentiments for all comments in the dataset.
    """
    start_time = datetime.now()
    logger.info(f"Starting sentiment labeling: {input_path}")
    
    # Load data
    logger.info(f"Loading language-labeled data: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count:,} records")
    
    # Create labeler instance
    labeler = SentimentLabeler()
    
    # Apply sentiment labeling with progress bar to make it look cool lol
    logger.info("Labeling sentiments with VADER + TextBlob...")
    
    # Process in batches for efficiency
    sentiment_results = []
    for text in tqdm(df['cleaned_text'], desc="Labeling"):
        sentiment_results.append(labeler.label_sentiment(text))
    
    # Convert results to DataFrame columns
    sentiment_df = pd.DataFrame(sentiment_results)
    
    # Merge with original data
    df = pd.concat([df, sentiment_df], axis=1)
    
    # Calculate correlation between VADER and TextBlob
    correlation = df['vader_compound'].corr(df['textblob_polarity'])
    
    # Sentiment distribution by language
    sentiment_by_language = df.groupby(['language', 'sentiment']).size().unstack(fill_value=0)
    
    # Save labeled data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Sentiment-labeled data saved to {output_file}")
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    labeling_stats = labeler.get_statistics()
    
    # NUMBERS are very important, Mr. Kennedy    
    stats = {
        'total_records': initial_count,
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
        'sentiment_percentages': (df['sentiment'].value_counts(normalize=True) * 100).round(2).to_dict(),
        'vader_textblob_correlation': round(correlation, 4),
        'sentiment_by_language': sentiment_by_language.to_dict(),
        'avg_vader_compound': round(df['vader_compound'].mean(), 4),
        'std_vader_compound': round(df['vader_compound'].std(), 4),
        'processing_duration_seconds': duration,
        'records_per_second': initial_count / duration,
        'output_file': str(output_file),
        'file_size_mb': output_file.stat().st_size / (1024 * 1024)
    }
    
    return stats

#Same like the prev 3 files
def main():
    INPUT_PATH = "./data/processed/comments_language_labeled.csv"
    OUTPUT_PATH = "./data/processed/comments_labeled.csv"
    
    logger.info("="*80)
    logger.info("Sentiment Labeling")
    logger.info("="*80)
    
    # Verify input exists
    if not Path(INPUT_PATH).exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        return
    
    # Label sentiments
    stats = label_sentiments(INPUT_PATH, OUTPUT_PATH)
    
    logger.info("="*80)
    logger.info("SENTIMENT LABELING COMPLETE ")
    logger.info("="*80)
    
    # Print summary 
    '''
    print("\n" + "="*80)
    print(" SENTIMENT LABELING SUMMARY")
    print("="*80)
    print(f" Processed {stats['total_records']:,} records in {stats['processing_duration_seconds']:.2f}s")
    print(f" Speed: {stats['records_per_second']:.0f} records/second")
    
    print(f"\n Sentiment Distribution:")
    for sentiment, count in stats['sentiment_distribution'].items():
        pct = stats['sentiment_percentages'][sentiment]
        print(f"   {sentiment:10} : {count:6,} records ({pct:5.2f}%)")
    
    print(f"\n Quality Metrics:")
    print(f"   VADER-TextBlob correlation: {stats['vader_textblob_correlation']:.4f}")
    print(f"   Average VADER compound: {stats['avg_vader_compound']:.4f}")
    print(f"   Std deviation: {stats['std_vader_compound']:.4f}")
    
    print(f"\n Output: {stats['output_file']} ({stats['file_size_mb']:.2f} MB)")
    print("="*80 + "\n")
'''

if __name__ == "__main__":
    main()
