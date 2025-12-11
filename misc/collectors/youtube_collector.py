"""
YOUTUBE DATA COLLECTOR
Focused collection of YouTube comments for unemployment sentiment analysis
Target: 300,000 YouTube posts
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from utils.logger import get_logger

# Load environment variables
load_dotenv()

# YouTube API Key from .env
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Comprehensive keyword list for unemployment in India
KEYWORDS = [
    # Core unemployment terms
    'unemployment india', 'jobless india', 'berozgaari', 'job crisis india',
    'no jobs india', 'employment crisis', 'youth unemployment india',
    
    # Layoffs and job loss
    'layoffs india', 'job loss india', 'IT layoffs india', 'tech layoffs india',
    'mass layoffs', 'retrenchment india', 'fired india', 'pink slip india',
    
    # Job search and placement
    'job market india', 'job search india', 'can\'t find job india',
    'placement crisis', 'campus placement india', 'fresher jobs india',
    
    # Education and unemployment
    'engineering unemployment', 'MBA jobless', 'graduate unemployment',
    'IIT unemployment', 'educated unemployed india', 'degree no job india',
    
    # Economic impact
    'salary cuts india', 'job insecurity india', 'gig economy india',
    'contract jobs india', 'temporary jobs india',
    
    # Mental health and social
    'unemployment depression', 'jobless stress india', 'unemployment anxiety',
    'career crisis india', 'hopeless job search',
    
    # Specific sectors
    'IT jobs crisis', 'startup layoffs india', 'manufacturing jobs india',
    'service sector jobs', 'retail jobs crisis',
    
    # Government and policy
    'unemployment rate india', 'CMIE unemployment', 'jobs data india',
    'employment policy india', 'rozgar yojana', 'skill india program',
    
    # Migration and demographics
    'brain drain india', 'migration for jobs', 'reverse migration india',
    'rural unemployment', 'urban unemployment india',
    
    # Future and automation
    'AI job loss india', 'automation unemployment', 'future of jobs india',
    'job displacement india', 'skills gap india'
]

class YouTubeCollector:
    def __init__(self):
        import httplib2
        # Create HTTP client with timeout
        http = httplib2.Http(timeout=10)
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, http=http)
        self.db = DatabaseManager()
        self.logger = get_logger('youtube_collector')
        self.used_keywords = []  # Track used keywords for diversity
        self.keyword_pool = KEYWORDS.copy()  # Copy of all keywords
        
    def search_videos(self, keyword, max_results=20, sort_order='relevance'):
        """Search for videos related to keyword"""
        try:
            # Vary time range to get different videos (30/60/90 days)
            days_back = random.choice([30, 60, 90])
            published_after = (datetime.now() - timedelta(days=days_back)).isoformat() + 'Z'
            
            search_response = self.youtube.search().list(
                q=keyword,
                part='id,snippet',
                type='video',
                maxResults=max_results,
                order=sort_order,  # Use variable sort order
                publishedAfter=published_after,
                regionCode='IN',
                relevanceLanguage='en'
            ).execute()
            
            videos = []
            for item in search_response.get('items', []):
                if item['id']['kind'] == 'youtube#video':
                    videos.append({
                        'video_id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle']
                    })
            
            return videos
            
        except HttpError as e:
            self.logger.error(f"YouTube API error searching videos: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error searching videos for '{keyword}': {e}")
            return []
    
    def get_video_comments(self, video_id, max_comments=100):
        """Get comments from a video"""
        comments_data = []
        
        try:
            # Get video details first (with timeout protection)
            try:
                video_response = self.youtube.videos().list(
                    part='snippet,statistics',
                    id=video_id
                ).execute()
            except Exception as e:
                self.logger.debug(f"Could not get video details for {video_id}: {e}")
                return []
            
            if not video_response.get('items'):
                return []
            
            video_info = video_response['items'][0]
            video_title = video_info['snippet']['title']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            channel_title = video_info['snippet']['channelTitle']
            
            # Get comments (with timeout protection)
            try:
                comments_response = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=min(100, max_comments),
                    order='relevance',
                    textFormat='plainText'
                ).execute()
            except Exception as e:
                self.logger.debug(f"Could not get comments for {video_id}: {e}")
                return []
            
            for item in comments_response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                
                # Only include substantial comments (>20 chars)
                if len(comment['textDisplay']) >= 20:
                    comments_data.append({
                        'post_id': f"youtube_{item['id']}",
                        'text': comment['textDisplay'][:1000],
                        'author': comment.get('authorDisplayName', 'Unknown'),
                        'created_date': comment['publishedAt'],
                        'url': video_url,
                        'location': 'India',
                        'metadata': {
                            'video_title': video_title,
                            'channel': channel_title,
                            'video_id': video_id
                        },
                        'likes': comment.get('likeCount', 0),
                        'comments_count': 0,
                        'shares': 0
                    })
            
            return comments_data
            
        except HttpError as e:
            if 'commentsDisabled' in str(e):
                self.logger.debug(f"Comments disabled for video {video_id}")
            else:
                self.logger.error(f"YouTube API error getting comments: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error getting comments for video {video_id}: {e}")
            return []
    
    def collect_for_keyword(self, keyword, target_comments=200):
        """Collect comments for a specific keyword"""
        print(f"\n Keyword: {keyword}")
        print(f"   Target: {target_comments} comments")
        
        total_collected = 0
        
        # Rotate through different sort orders to get diverse videos
        sort_orders = ['relevance', 'date', 'viewCount']
        sort_order = random.choice(sort_orders)
        
        # Search for videos with varied sorting
        videos = self.search_videos(keyword, max_results=20, sort_order=sort_order)
        
        if not videos:
            print(f"     No videos found")
            return 0
        
        print(f"    Found {len(videos)} videos (sorted by: {sort_order})")
        
        # Collect comments from each video
        for idx, video in enumerate(videos, 1):
            if total_collected >= target_comments:
                break
            
            print(f"   [{idx}/{len(videos)}] Processing: {video['title'][:40]}...", end=" ", flush=True)
            
            try:
                comments = self.get_video_comments(
                    video['video_id'], 
                    max_comments=min(100, target_comments - total_collected)
                )
            except Exception as e:
                self.logger.error(f"Error processing video {video['video_id']}: {e}")
                print(f" Error")
                continue
            
            if comments:
                try:
                    successful, failed = self.db.insert_batch('youtube', comments)
                    total_collected += successful
                    
                    print(f" +{successful}")
                except Exception as e:
                    self.logger.error(f"Database error: {e}")
                    print(f" DB Error")
            else:
                print(f"⊘ No comments")
            
            # Rate limiting
            time.sleep(random.uniform(0.5, 1.5))
        
        print(f"    Total collected: {total_collected}")
        return total_collected
    
    def run_collection(self, target_total=1000, keywords_per_cycle=5):
        """Run collection cycle with diverse keyword rotation"""
        cycle = 0
        
        print("="*80)
        print(" YOUTUBE DATA COLLECTOR")
        print("="*80)
        print(f" NEW Target: 150K total posts")
        print(f" YouTube Target: 60K posts (40% of 150K)")
        print(f" Total keywords: {len(KEYWORDS)}")
        print(f" Keywords per cycle: {keywords_per_cycle} (rotating for diversity)")
        print("="*80)
        
        # Get current stats
        stats = self.db.get_stats()
        youtube_count = stats.get('by_source', {}).get('youtube', 0)
        youtube_target = 60000
        print(f"\n Current YouTube count: {youtube_count:,}")
        print(f" Remaining to 60K target: {youtube_target - youtube_count:,}")
        
        while True:
            try:
                cycle += 1
                print(f"\n{'='*80}")
                print(f"[Cycle #{cycle}] {datetime.now().strftime('%H:%M:%S')}")
                print("="*80)
                
                # DIVERSE KEYWORD SELECTION - rotate through all keywords
                if len(self.keyword_pool) < keywords_per_cycle:
                    # Reset pool when exhausted, ensuring fresh rotation
                    print(" Keyword pool exhausted, resetting for new rotation...")
                    self.keyword_pool = KEYWORDS.copy()
                    random.shuffle(self.keyword_pool)
                
                # Select keywords from pool (ensures no repeats until all used)
                selected_keywords = []
                for _ in range(min(keywords_per_cycle, len(self.keyword_pool))):
                    keyword = self.keyword_pool.pop(0)
                    selected_keywords.append(keyword)
                
                print(f" Selected diverse keywords: {', '.join(selected_keywords)}")
                
                cycle_total = 0
                comments_per_keyword = target_total // keywords_per_cycle
                
                for keyword in selected_keywords:
                    collected = self.collect_for_keyword(keyword, target_comments=comments_per_keyword)
                    cycle_total += collected
                    
                    # Small delay between keywords
                    time.sleep(random.uniform(2, 4))
                
                # Get updated stats
                stats = self.db.get_stats()
                youtube_count = stats.get('by_source', {}).get('youtube', 0)
                total_count = stats.get('total_posts', 0)
                youtube_pct = (youtube_count / total_count * 100) if total_count > 0 else 0
                youtube_target = 60000
                progress_to_target = (youtube_count / youtube_target) * 100
                overall_progress = (total_count / 150000) * 100
                
                print("\n" + "="*80)
                print(f" Cycle #{cycle} Complete: +{cycle_total} new comments")
                print(f" Total YouTube: {youtube_count:,} ({youtube_pct:.1f}% of database)")
                print(f" Progress to 60K YouTube: {progress_to_target:.1f}%")
                print(f" Remaining to YouTube target: {youtube_target - youtube_count:,}")
                print(f" Overall progress to 150K: {overall_progress:.1f}%")
                print("="*80)
                
                # Check if we've reached 60K YouTube target
                if youtube_count >= youtube_target:
                    print(f"\n TARGET REACHED! {youtube_count:,} / {youtube_target:,} YouTube posts collected!")
                    print(" YouTube collection phase complete!")
                    break
                
                # NO WAIT - Continue immediately to next cycle
                print(f"\n Starting next cycle immediately... (Remaining: {youtube_target - youtube_count:,})")
                time.sleep(2)  # Just 2 seconds for API rate limiting
                
            except KeyboardInterrupt:
                print("\n\n Stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in cycle #{cycle}: {e}")
                print(f"\n Error in cycle: {str(e)}")
                print("⏰ Waiting 60s before retry...")
                time.sleep(60)
        
        # Final stats
        stats = self.db.get_stats()
        youtube_count = stats.get('by_source', {}).get('youtube', 0)
        total_count = stats.get('total_posts', 0)
        print("\n" + "="*80)
        print(" FINAL STATISTICS")
        print("="*80)
        print(f"YouTube comments: {youtube_count:,}")
        print(f"Progress to 60K YouTube: {(youtube_count/60000)*100:.1f}%")
        print(f"Total posts: {total_count:,}")
        print(f"Overall progress to 150K: {(total_count/150000)*100:.1f}%")
        print("="*80)

if __name__ == "__main__":
    collector = YouTubeCollector()
    # Run continuously until 60K target reached (no breaks!)
    collector.run_collection(target_total=1000, keywords_per_cycle=5)
