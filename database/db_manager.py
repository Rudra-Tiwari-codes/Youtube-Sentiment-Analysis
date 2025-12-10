"""
Database Manager for Unemployment Sentiment Analysis
Handles SQLite database operations for storing collected data
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager:
    """Manage SQLite database for sentiment analysis data"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection"""
        if db_path is None:
            db_path = os.getenv('SQLITE_DB_PATH', './data/unemployment_sentiment.db')
        
        # Create directory
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.create_tables()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get or create database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            self.conn.execute('PRAGMA journal_mode=WAL')
            self.conn.execute('PRAGMA busy_timeout=30000')
        return self.conn
    
    def create_tables(self):
        """Create necessary database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source VARCHAR(50) NOT NULL,
                post_id VARCHAR(255) UNIQUE,
                text TEXT NOT NULL,
                author VARCHAR(255),
                created_date DATETIME,
                url TEXT,
                location VARCHAR(100),
                platform_metadata TEXT,
                collected_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Additional fields for different sources
                likes INTEGER DEFAULT 0,
                comments_count INTEGER DEFAULT 0,
                shares INTEGER DEFAULT 0,
                
                -- Processing status
                is_processed BOOLEAN DEFAULT 0,
                
                UNIQUE(source, post_id)
            )
        ''')
        
        # Processed posts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_post_id INTEGER NOT NULL,
                cleaned_text TEXT,
                language VARCHAR(10),
                
                -- Sentiment scores
                vader_positive FLOAT,
                vader_negative FLOAT,
                vader_neutral FLOAT,
                vader_compound FLOAT,
                
                textblob_polarity FLOAT,
                textblob_subjectivity FLOAT,
                
                -- ML model predictions
                ml_sentiment VARCHAR(50),
                ml_confidence FLOAT,
                
                -- Emotion classification
                primary_emotion VARCHAR(50),
                emotion_scores TEXT,
                
                -- Topic modeling
                topics TEXT,
                
                -- Metadata
                processed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (raw_post_id) REFERENCES raw_posts(id)
            )
        ''')
        
        # Collection statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source VARCHAR(50),
                collection_date DATE,
                posts_collected INTEGER,
                errors_count INTEGER,
                collection_duration_seconds FLOAT,
                notes TEXT
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_source ON raw_posts(source)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_date ON raw_posts(created_date)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_is_processed ON raw_posts(is_processed)
        ''')
        
        conn.commit()
        print(f"[OK] Database initialized at: {self.db_path}")
    
    def insert_post(self, source: str, post_data: Dict) -> Optional[int]:
        """
        Insert a post into the database
        Args:
            source: Data source (youtube, twitter, news, etc.)
            post_data: Dictionary containing post data
        Returns:
            ID of inserted post or None if duplicate
        """
        import time
        max_retries = 5
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO raw_posts (
                        source, post_id, text, author, created_date, url,
                        location, platform_metadata, likes, comments_count, shares
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    source,
                    post_data.get('post_id'),
                    post_data.get('text'),
                    post_data.get('author'),
                    post_data.get('created_date'),
                    post_data.get('url'),
                    post_data.get('location'),
                    json.dumps(post_data.get('metadata', {})),
                    post_data.get('likes', 0),
                    post_data.get('comments_count', 0),
                    post_data.get('shares', 0)
                ))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Duplicate post
                return None
            except sqlite3.OperationalError as e:
                if 'locked' in str(e) and attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    return None
            except Exception as e:
                print(f"Error inserting post: {e}")
                try:
                    conn.rollback()
                except:
                    pass
                return None
        return None
    
    def insert_batch(self, source: str, posts: List[Dict]) -> Tuple[int, int]:
        """
        Insert multiple posts at once
        Returns:
            Tuple of (successful_inserts, duplicates)
        """
        successful = 0
        duplicates = 0
        for post in posts:
            result = self.insert_post(source, post)
            if result:
                successful += 1
            else:
                duplicates += 1
        
        return successful, duplicates
    
    def get_unprocessed_posts(self, limit: int = 100) -> List[Dict]:
        """Get posts that haven't been processed yet"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM raw_posts
            WHERE is_processed = 0
            LIMIT ?
        ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def mark_as_processed(self, post_id: int):
        """Mark a post as processed"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE raw_posts
            SET is_processed = 1
            WHERE id = ?
        ''', (post_id,))
        
        conn.commit()
    
    def insert_processed_data(self, processed_data: Dict):
        """Insert processed sentiment analysis results"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO processed_posts (
                raw_post_id, cleaned_text, language,
                vader_positive, vader_negative, vader_neutral, vader_compound,
                textblob_polarity, textblob_subjectivity,
                ml_sentiment, ml_confidence,
                primary_emotion, emotion_scores, topics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            processed_data['raw_post_id'],
            processed_data.get('cleaned_text'),
            processed_data.get('language'),
            processed_data.get('vader_positive'),
            processed_data.get('vader_negative'),
            processed_data.get('vader_neutral'),
            processed_data.get('vader_compound'),
            processed_data.get('textblob_polarity'),
            processed_data.get('textblob_subjectivity'),
            processed_data.get('ml_sentiment'),
            processed_data.get('ml_confidence'),
            processed_data.get('primary_emotion'),
            json.dumps(processed_data.get('emotion_scores', {})),
            json.dumps(processed_data.get('topics', []))
        ))
        
        conn.commit()
    
    def log_collection_stats(self, source: str, posts_collected: int, 
                            errors_count: int, duration: float, notes: str = ""):
        """Log collection statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO collection_stats (
                source, collection_date, posts_collected, 
                errors_count, collection_duration_seconds, notes
            ) VALUES (?, DATE('now'), ?, ?, ?, ?)
        ''', (source, posts_collected, errors_count, duration, notes))
        
        conn.commit()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total posts by source
        cursor.execute('''
            SELECT source, COUNT(*) as count
            FROM raw_posts
            GROUP BY source
        ''')
        source_stats = {row['source']: row['count'] for row in cursor.fetchall()}
        
        # Processing stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_processed = 1 THEN 1 ELSE 0 END) as processed,
                SUM(CASE WHEN is_processed = 0 THEN 1 ELSE 0 END) as pending
            FROM raw_posts
        ''')
        processing_stats = dict(cursor.fetchone())
        
        return {
            'by_source': source_stats,
            'processing': processing_stats,
            'total_posts': processing_stats['total']
        }
    
    def get_source_count(self, source: str) -> int:
        """
        Get count of posts from a specific source
        
        Args:
            source: Data source (youtube, twitter, news, telegram, etc.)
            
        Returns:
            Number of posts from that source
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM raw_posts
            WHERE source = ?
        ''', (source,))
        
        result = cursor.fetchone()
        return result['count'] if result else 0
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


if __name__ == "__main__":
    # Test the database
    db = DatabaseManager()
    
    # Test insert
    test_post = {
        'post_id': 'test_123',
        'text': 'Testing unemployment sentiment analysis database',
        'author': 'test_user',
        'created_date': datetime.now().isoformat(),
        'url': 'http://example.com/test',
        'location': 'India',
        'metadata': {'test': True},
        'likes': 10
    }
    
    result = db.insert_post('test', test_post)
    print(f"Inserted post with ID: {result}")
    
    # Get stats
    stats = db.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"Total posts: {stats['total_posts']}")
    print(f"By source: {stats['by_source']}")
    print(f"Processing: {stats['processing']}")
    
    db.close()
    print("\n Database test completed successfully!")
