"""
HIGHLY OPTIMIZED YouTube Collector
- Sorts by viewCount (popular videos have more comments)
- Filters videos with 0 comments BEFORE fetching
- Smart keyword rotation
- Adaptive batch sizing
"""

import os
import sys
import time
import random
import json
import hashlib
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from utils.logger import get_logger

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
YOUTUBE_API_KEY_NEW = os.getenv('YOUTUBE_API_KEY_NEW')  # Backup API key

# 58 Unemployment keywords
UNEMPLOYMENT_KEYWORDS = [
    'unemployment india', 'jobless india', 'berozgaari', 'job crisis india',
    'layoffs india', 'job cuts india', 'retrenchment india', 'pink slip india',
    'IT layoffs india', 'tech layoffs india', 'startup layoffs india',
    'mass layoffs', 'unemployment rate india', 'youth unemployment india',
    'graduate unemployment', 'fresher unemployment', 'engineering unemployment',
    'MBA jobless', 'degree no job india', 'educated unemployed',
    'job loss india', 'firing india', 'termination india',
    'unemployment anxiety', 'jobless stress india', 'career crisis india',
    'no jobs india', 'job scarcity india', 'employment crisis india',
    'CMIE unemployment', 'rural unemployment', 'urban unemployment',
    'women unemployment india', 'skills gap india', 'placement crisis',
    'campus placement crisis', 'IIT unemployment', 'engineering jobless',
    'automation unemployment', 'AI job loss india', 'job market crash',
    'recession jobs india', 'economic crisis jobs', 'downsizing india',
    'cost cutting layoffs', 'restructuring layoffs', 'performance layoffs',
    'reverse migration india', 'brain drain unemployment', 'talent exodus',
    'skill india program', 'MGNREGA', 'employment policy india',
    'job schemes india', 'unemployment benefits', 'jobless youth',
    'service sector jobs crisis', 'manufacturing jobs crisis', 'retail layoffs india'
]

# 100 Employment keywords  
EMPLOYMENT_KEYWORDS = [
    'jobs in india', 'hiring india', 'recruitment india', 'careers india',
    'job opportunities india', 'job openings india', 'job vacancies india',
    'government jobs india', 'sarkari naukri', 'bank jobs india',
    'SSC jobs', 'UPSC jobs', 'railway jobs india', 'defense jobs india',
    'teaching jobs india', 'professor jobs', 'lecturer jobs india',
    'IT jobs india', 'software jobs india', 'developer jobs india',
    'data scientist jobs', 'AI jobs india', 'machine learning jobs',
    'cyber security jobs', 'cloud engineer jobs', 'devops jobs india',
    'digital marketing jobs', 'content writer jobs', 'SEO jobs india',
    'sales jobs india', 'marketing jobs india', 'HR jobs india',
    'finance jobs india', 'accounting jobs india', 'CA jobs india',
    'MBA jobs india', 'management jobs', 'business analyst jobs',
    'consultant jobs india', 'startup jobs india', 'product manager jobs',
    'project manager jobs', 'scrum master jobs', 'agile jobs india',
    'civil engineer jobs', 'mechanical engineer jobs', 'electrical engineer jobs',
    'chemical engineer jobs', 'production engineer jobs', 'quality engineer jobs',
    'design engineer jobs', 'automobile engineer jobs', 'aerospace engineer jobs',
    'pharmacy jobs india', 'medical jobs india', 'doctor jobs india',
    'nurse jobs india', 'healthcare jobs', 'hospital jobs india',
    'BPO jobs india', 'call center jobs', 'customer service jobs',
    'telecom jobs india', 'banking jobs india', 'insurance jobs india',
    'real estate jobs', 'construction jobs india', 'architecture jobs',
    'interior design jobs', 'graphic design jobs', 'animation jobs india',
    'media jobs india', 'journalism jobs', 'PR jobs india',
    'hotel jobs india', 'hospitality jobs', 'chef jobs india',
    'travel jobs india', 'aviation jobs india', 'logistics jobs india',
    'supply chain jobs', 'operations jobs india', 'warehouse jobs india',
    'retail jobs india', 'ecommerce jobs', 'amazon jobs india',
    'flipkart jobs', 'swiggy jobs', 'zomato jobs', 'uber jobs india',
    'ola jobs india', 'delivery jobs india', 'driver jobs india',
    'security guard jobs', 'housekeeping jobs', 'receptionist jobs india',
    'admin jobs india', 'office assistant jobs', 'data entry jobs',
    'fresher jobs india', 'internship india', 'part time jobs india',
    'work from home jobs', 'remote jobs india', 'freelance jobs india',
    'entry level jobs india', 'graduate jobs india', 'BE jobs india',
    'BTech jobs india', 'MCA jobs india', 'BCA jobs india',
    'job hunting india', 'job search india', 'job fairs india',
    'walk in interview', 'job notifications india', 'latest jobs india',
    'jobs in delhi', 'jobs in mumbai', 'jobs in bangalore', 'jobs in hyderabad',
    'jobs in chennai', 'jobs in pune', 'jobs in kolkata', 'jobs in ahmedabad',
    'naukri india', 'rozgaar india', 'employment india', 'career opportunities'
]


class SuperOptimizedCollector:
    def __init__(self, category='unemployment'):
        self.category = category
        self.keywords = UNEMPLOYMENT_KEYWORDS if category == 'unemployment' else EMPLOYMENT_KEYWORDS
        self.keyword_pool = self.keywords.copy()
        random.shuffle(self.keyword_pool)
        
        self.current_api_key = YOUTUBE_API_KEY
        self.backup_api_key = YOUTUBE_API_KEY_NEW
        self.youtube = build('youtube', 'v3', developerKey=self.current_api_key)
        self.db = DatabaseManager()
        self.logger = get_logger(f'youtube_{category}')
        
        # Load processed videos
        self.processed_videos = self.load_processed_videos()
    
    def switch_api_key(self):
        """Switch to backup API key when quota exhausted"""
        if self.current_api_key == YOUTUBE_API_KEY and self.backup_api_key:
            print("\n  Primary API quota exhausted!")
            print(" Switching to backup API key...")
            self.current_api_key = self.backup_api_key
            self.youtube = build('youtube', 'v3', developerKey=self.current_api_key)
            print(" Switched to backup API key successfully!")
            return True
        else:
            print("\n Both API keys exhausted!")
            return False
    
    def load_processed_videos(self):
        """Load set of already processed video IDs"""
        processed = set()
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT json_extract(platform_metadata, '$.video_id') as vid
                FROM raw_posts 
                WHERE source = 'youtube' AND platform_metadata IS NOT NULL
            """)
            for row in cursor.fetchall():
                if row[0]:
                    processed.add(row[0])
            self.logger.info(f"Loaded {len(processed)} processed video IDs")
        except Exception as e:
            self.logger.error(f"Error loading processed videos: {e}")
        return processed
    
    def search_and_filter_videos(self, keyword, max_results=50):
        """Search videos AND filter by comment count in ONE go"""
        try:
            # Step 1: Search for videos (sorted by viewCount)
            search_response = self.youtube.search().list(
                q=f"{keyword} india",
                part='id',
                maxResults=max_results,
                type='video',
                order='viewCount',  # Most viewed = more comments
                regionCode='IN',
                relevanceLanguage='en'
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response.get('items', []) 
                        if item['id']['kind'] == 'youtube#video' and item['id']['videoId'] not in self.processed_videos]
            
            if not video_ids:
                return []
            
            # Step 2: Get statistics for ALL videos in ONE API call
            stats_response = self.youtube.videos().list(
                part='statistics,snippet',
                id=','.join(video_ids)
            ).execute()
            
            # Step 3: Filter and sort
            videos = []
            for item in stats_response.get('items', []):
                stats = item.get('statistics', {})
                comment_count = int(stats.get('commentCount', 0))
                
                # ONLY include videos WITH comments
                if comment_count > 0:
                    videos.append({
                        'video_id': item['id'],
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle'],
                        'comment_count': comment_count,
                        'view_count': int(stats.get('viewCount', 0))
                    })
            
            # Sort by comment count DESC (most efficient)
            videos.sort(key=lambda x: x['comment_count'], reverse=True)
            return videos
            
        except HttpError as e:
            # Check if quota exceeded
            if e.resp.status == 403 and 'quotaExceeded' in str(e):
                print("\n  API Quota exceeded!")
                if self.switch_api_key():
                    # Retry with new key
                    return self.search_and_filter_videos(keyword, max_results)
                else:
                    raise Exception("All API keys exhausted")
            self.logger.error(f"Error searching {keyword}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error searching {keyword}: {e}")
            return []
    
    def get_video_comments(self, video_id, video_info, max_comments=100):
        """Get comments from video"""
        comments_data = []
        
        try:
            response = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                order='relevance',
                textFormat='plainText'
            ).execute()
            
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                
                if len(comment['textDisplay']) >= 20:
                    comments_data.append({
                        'post_id': f"youtube_{item['id']}",
                        'text': comment['textDisplay'][:1000],
                        'author': comment.get('authorDisplayName', 'Unknown'),
                        'created_date': comment['publishedAt'],
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'location': 'India',
                        'metadata': {
                            'video_title': video_info['title'],
                            'channel': video_info['channel'],
                            'video_id': video_id,
                            'category': self.category
                        },
                        'likes': comment.get('likeCount', 0),
                        'comments_count': 0,
                        'shares': 0
                    })
            
            if comments_data:
                self.processed_videos.add(video_id)
            
            return comments_data
            
        except HttpError as e:
            # Check if quota exceeded
            if e.resp.status == 403 and 'quotaExceeded' in str(e):
                print("\n  API Quota exceeded!")
                if self.switch_api_key():
                    # Retry with new key
                    return self.get_video_comments(video_id, video_info, max_comments)
                else:
                    raise Exception("All API keys exhausted")
            self.logger.error(f"Error getting comments for {video_id}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error getting comments for {video_id}: {e}")
            return []
    
    def collect_for_keyword(self, keyword, target_comments=250):
        """Collect comments for keyword"""
        print(f"\n {keyword}")
        
        videos = self.search_and_filter_videos(keyword, max_results=50)
        
        if not videos:
            print(f"   ⊘ No videos with comments")
            return 0
        
        # Show efficiency metrics
        top_counts = [v['comment_count'] for v in videos[:5]]
        print(f"    {len(videos)} videos | Top: {top_counts}")
        
        total_collected = 0
        idx = 0
        
        for idx, video in enumerate(videos, 1):
            if total_collected >= target_comments:
                break
            
            print(f"   [{idx}] {video['title'][:40]}... ({video['comment_count']}) ", end="", flush=True)
            
            comments = self.get_video_comments(video['video_id'], video, max_comments=100)
            
            if comments:
                successful, duplicates = self.db.insert_batch('youtube', comments)
                total_collected += successful
                if successful > 0:
                    print(f" +{successful}")
                else:
                    print(f"⊘")
            else:
                print(f"⊘")
            
            time.sleep(random.uniform(0.2, 0.4))  # Faster since we're smarter
        
        print(f"    {total_collected} comments from {idx} videos")
        return total_collected
    
    def get_count(self):
        """Get current count"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM raw_posts
            WHERE source = 'youtube' 
            AND json_extract(platform_metadata, '$.category') = ?
        """, (self.category,))
        return cursor.fetchone()[0]
    
    def run_collection(self, target=66000):
        """Run super optimized collection"""
        print(f"\n{'='*80}")
        print(f" SUPER OPTIMIZED {self.category.upper()} COLLECTOR")
        print(f"{'='*80}")
        print(f" Target: {target:,}")
        print(f" Keywords: {len(self.keywords)}")
        print(f" Optimizations:")
        print(f"   • Sort by viewCount (popular videos)")
        print(f"   • Filter zero-comment videos BEFORE fetching")
        print(f"   • Sort results by comment count (most first)")
        print(f"   • Adaptive batch sizing")
        print(f"   • Smart keyword rotation")
        print(f"{'='*80}\n")
        
        cycle = 0
        
        while True:
            cycle += 1
            current = self.get_count()
            remaining = target - current
            
            print(f"\n{'='*80}")
            print(f"[Cycle #{cycle}] {datetime.now().strftime('%H:%M:%S')}")
            print(f" Current: {current:,} | Remaining: {remaining:,} | Progress: {(current/target)*100:.1f}%")
            print(f"{'='*80}")
            
            if remaining <= 0:
                print(f"\n TARGET REACHED! {current:,} comments!")
                break
            
            # Refill keyword pool
            if not self.keyword_pool:
                self.keyword_pool = self.keywords.copy()
                random.shuffle(self.keyword_pool)
                print(f" Reshuffled {len(self.keyword_pool)} keywords")
            
            # Adaptive: More keywords when far from target
            batch_size = min(20, max(5, remaining // 200))
            batch = self.keyword_pool[:batch_size]
            self.keyword_pool = self.keyword_pool[batch_size:]
            
            print(f" Processing {len(batch)} keywords")
            
            cycle_total = 0
            for keyword in batch:
                collected = self.collect_for_keyword(keyword, target_comments=250)
                cycle_total += collected
                
                if collected > 0:
                    remaining -= collected
                    if remaining <= 0:
                        break
            
            print(f"\n Cycle #{cycle}: +{cycle_total} comments")
            print(f" Total: {self.get_count():,}/{target:,}")
            
            time.sleep(2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('category', choices=['unemployment', 'employment'])
    parser.add_argument('--target', type=int, default=66000)
    args = parser.parse_args()
    
    collector = SuperOptimizedCollector(args.category)
    collector.run_collection(args.target)


if __name__ == '__main__':
    main()
