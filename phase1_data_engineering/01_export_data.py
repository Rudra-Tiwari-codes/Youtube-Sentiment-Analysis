"""
Export YouTube comments from database to CSV file.
Takes all the comments we collected and saves them to a CSV file
so we can clean and analyze them in the next steps.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def export_raw_data(db_path: str, output_path: str) -> dict:
    """    Export comments from database to CSV file. """
    start_time = datetime.now()
    logger.info(f"Starting data export from {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Count total records
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_posts WHERE source = 'youtube'")
        total_records = cursor.fetchone()[0]
        logger.info(f"Found {total_records:,} YouTube comments to export")
        
        # Export data with optimized query
        query = """
            SELECT 
                id,
                post_id,
                text,
                author,
                created_date,
                url,
                likes,
                comments_count,
                collected_date
            FROM raw_posts
            WHERE source = 'youtube'
            ORDER BY created_date DESC
        """
        
        logger.info("Executing export query...")
        df = pd.read_sql_query(query, conn)
        
        # Data validation
        assert len(df) == total_records, "Record count mismatch!"
        assert df['text'].notna().all(), "Found NULL text values!"
        
        logger.info(f"Loaded {len(df):,} records into DataFrame")
        
        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Data exported to {output_file}")
        
        conn.close()
        
        # Calculate statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        stats = {
            'total_records': len(df),
            'unique_authors': df['author'].nunique(),
            'date_range': {
                'earliest': df['created_date'].min(),
                'latest': df['created_date'].max()
            },
            'null_counts': df.isnull().sum().to_dict(),
            'export_duration_seconds': duration,
            'output_file': str(output_file),
            'file_size_mb': output_file.stat().st_size / (1024 * 1024)
        }
        
        # Log statistics
        logger.info("Export Statistics:")
        logger.info(f"  Total Records: {stats['total_records']:,}")
        logger.info(f"  Unique Authors: {stats['unique_authors']:,}")
        logger.info(f"  Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        logger.info(f"  File Size: {stats['file_size_mb']:.2f} MB")
        logger.info(f"  Duration: {stats['export_duration_seconds']:.2f} seconds")
        
        return stats
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise


def main():
    """Main execution function"""
    DB_PATH = "./data/unemployment_sentiment.db"
    OUTPUT_PATH = "./data/raw/youtube_comments_raw.csv"
    logger.info("Starting data export...")
    
    # Verify database exists
    if not Path(DB_PATH).exists():
        logger.error(f"Database not found: {DB_PATH}")
        return
    
    # Export data
    stats = export_raw_data(DB_PATH, OUTPUT_PATH)
    
    logger.info("Export complete!")
    print(f"\nDone! Exported {stats['total_records']:,} comments from {stats['unique_authors']:,} authors")
    print(f"Saved to: {stats['output_file']} ({stats['file_size_mb']:.2f} MB)\n")


if __name__ == "__main__":
    main()
