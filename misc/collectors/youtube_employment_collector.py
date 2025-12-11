"""
YOUTUBE EMPLOYMENT DATA COLLECTOR
Collects YouTube comments about EMPLOYMENT and JOB SUCCESS in India
Target: 30,000 employment-related posts (50% of 60K YouTube target)
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

# Employment-focused keywords for India
EMPLOYMENT_KEYWORDS = [
    # Job openings and hiring
    'jobs india', 'hiring india', 'job openings india', 'recruitment india',
    'job vacancies india', 'walk in interview india', 'job fair india',
    'career opportunities india', 'job portal india',
    
    # Success stories
    'got job india', 'placed india', 'job offer india', 'hired india',
    'employment success india', 'job success story', 'career growth india',
    'promotion india', 'salary hike india',
    
    # IT and Tech hiring
    'IT hiring india', 'tech jobs india', 'software jobs india',
    'developer jobs india', 'coding jobs india', 'tech hiring boom india',
    'startup jobs india', 'IT recruitment india',
    
    # Government and PSU jobs
    'government jobs india', 'sarkari naukri', 'PSU recruitment india',
    'bank jobs india', 'railway jobs india', 'SSC jobs india',
    'UPSC jobs', 'state government jobs', 'govt job vacancy',
    
    # Sectors actively hiring
    'manufacturing jobs india', 'pharma jobs india', 'healthcare jobs india',
    'banking jobs india', 'finance jobs india', 'consulting jobs india',
    'education jobs india', 'teaching jobs india',
    
    # Campus placement and freshers
    'campus placement india', 'fresher hiring india', 'graduate trainee india',
    'entry level jobs india', 'internship india', 'campus recruitment india',
    'IIT placements', 'engineering placements', 'MBA placements',
    
    # Skill development and training
    'upskilling india', 'reskilling india', 'job training india',
    'skill development india', 'vocational training india',
    'certification jobs india', 'learn and earn india',
    
    # Positive job market trends
    'job market recovery india', 'employment growth india', 'job creation india',
    'hiring boom india', 'job demand india', 'employment opportunities india',
    'career options india', 'best jobs india',
    
    # Work culture and satisfaction
    'work culture india', 'job satisfaction india', 'employee benefits india',
    'work life balance india', 'best companies india', 'dream job india',
    
    # Entrepreneurship and freelancing
    'startup india', 'entrepreneur india', 'freelancing india',
    'self employment india', 'business opportunities india',
    'side hustle india', 'gig economy success',
    
    # Remote and flexible work
    'remote jobs india', 'work from home india', 'flexible jobs india',
    'online jobs india', 'digital jobs india',
    
    # Career guidance
    'career advice india', 'job tips india', 'resume tips india',
    'interview preparation india', 'career counseling india',
    'job search tips india', 'LinkedIn jobs india'
]

class YouTubeEmploymentCollector:
    def __init__(self):
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.db = DatabaseManager()
        self.logger = get_logger('youtube_employment_collector')
        self.used_keywords = []  # Track used keywords for diversity
        self.keyword_pool = EMPLOYMENT_KEYWORDS.copy()  # Copy of all keywords
        
    def search_videos(self, keyword, max_results=20):
        """Search for videos related to keyword"""
        try:
            # Search for videos from last 60 days
            published_after = (datetime.now() - timedelta(days=60)).isoformat() + 'Z'
            
            request = self.youtube.search().list(
                part='id,snippet',
                q=keyword,
                type='video',
                maxResults=max_results,
                publishedAfter=published_after,
                relevanceLanguage='en',
                order='relevance'
            )
            response = request.execute()
            
            video_ids = []
            for item in response.get('items', []):
                video_id = item['id']['videoId']
                video_ids.append(video_id)
                
            return video_ids
            
        except HttpError as e:
            self.logger.error(f"HTTP error searching videos: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error searching videos: {e}")
            return []
    
    def get_video_comments(self, video_id, max_results=100):
        """Get comments from a video"""
        try:
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=max_results,
                order='relevance',
                textFormat='plainText'
            )
            response = request.execute()
            
            comments = []
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                
                # Filter: Only keep comments longer than 20 characters
                if len(comment['textDisplay']) > 20:
                    comments.append({
                        'text': comment['textDisplay'],
                        'author': comment['authorDisplayName'],
                        'likes': comment['likeCount'],
                        'published_at': comment['publishedAt'],
                        'video_id': video_id
                    })
            
            return comments
            
        except HttpError as e:
            if 'commentsDisabled' in str(e):
                return []
            self.logger.error(f"HTTP error getting comments: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error getting comments: {e}")
            return []
    
    def get_diverse_keywords(self, count=5):
        """Get diverse keywords ensuring no repeats until all are used"""
        # If we've used all keywords, reset the pool
        if len(self.keyword_pool) < count:
            self.logger.info(f"Resetting keyword pool (used all {len(EMPLOYMENT_KEYWORDS)} keywords)")
            self.keyword_pool = EMPLOYMENT_KEYWORDS.copy()
            self.used_keywords = []
        
        # Select random keywords from remaining pool
        selected = random.sample(self.keyword_pool, min(count, len(self.keyword_pool)))
        
        # Move selected keywords from pool to used
        for kw in selected:
            self.keyword_pool.remove(kw)
            self.used_keywords.append(kw)
        
        return selected
    
    def collect_for_keyword(self, keyword):
        """Collect comments for a single keyword"""
        print(f"\n Keyword: {keyword}")
        
        # Search for videos
        video_ids = self.search_videos(keyword, max_results=10)
        print(f"    Found {len(video_ids)} videos")
        
        total_comments = 0
        for video_id in video_ids:
            comments = self.get_video_comments(video_id, max_results=100)
            
            # Save to database with sentiment_category
            for comment in comments:
                try:
                    self.db.insert_post(
                        source='youtube_employment',
                        post_id=f"yt_emp_{video_id}_{comment['author']}_{comment['published_at']}",
                        text=comment['text'],
                        metadata={
                            'author': comment['author'],
                            'video_id': video_id,
                            'published_at': comment['published_at'],
                            'keyword': keyword,
                            'sentiment_category': 'employment'  # Tag as employment-related
                        },
                        likes=comment['likes']
                    )
                    total_comments += 1
                except Exception as e:
                    continue
            
            time.sleep(0.5)  # Rate limiting between videos
        
        print(f"    Collected {total_comments} comments")
        return total_comments
    
    def run_continuous_collection(self):
        """Run continuous collection until target is reached"""
        print("="*80)
        print(" YOUTUBE EMPLOYMENT COLLECTOR")
        print("="*80)
        print(f" Target: 30,000 employment posts (50% of YouTube target)")
        print(f" Total keywords: {len(EMPLOYMENT_KEYWORDS)}")
        print(f" Keywords per cycle: 5 (rotating for diversity)")
        print("="*80)
        
        cycle = 0
        TARGET = 30000  # 50% of 60K YouTube target
        
        while True:
            cycle += 1
            
            # Check current count
            stats = self.db.get_stats()
            youtube_employment_count = stats['by_source'].get('youtube_employment', 0)
            
            print(f"\n Current employment count: {youtube_employment_count}")
            print(f" Remaining to 30K target: {TARGET - youtube_employment_count}")
            
            if youtube_employment_count >= TARGET:
                print("\n" + "="*80)
                print(" TARGET REACHED!")
                print(f" Collected {youtube_employment_count} employment posts")
                print("="*80)
                break
            
            print("\n" + "="*80)
            print(f"[Cycle #{cycle}] {datetime.now().strftime('%H:%M:%S')}")
            print("="*80)
            
            # Get diverse keywords for this cycle
            keywords = self.get_diverse_keywords(count=5)
            print(f" Selected keywords: {', '.join(keywords)}")
            
            cycle_total = 0
            for keyword in keywords:
                count = self.collect_for_keyword(keyword)
                cycle_total += count
                time.sleep(2)  # Small delay between keywords
            
            print(f"\n Cycle #{cycle} Summary:")
            print(f"   Total collected: {cycle_total} comments")
            print(f"   Keywords used so far: {len(self.used_keywords)}/{len(EMPLOYMENT_KEYWORDS)}")
            print(f"   Keywords remaining in pool: {len(self.keyword_pool)}")
            
            # Check overall progress
            stats = self.db.get_stats()
            total_posts = stats['total_posts']
            youtube_employment_count = stats['by_source'].get('youtube_employment', 0)
            
            print(f"\n Overall Progress:")
            print(f"   Employment posts: {youtube_employment_count}/30,000 ({youtube_employment_count/300:.1f}%)")
            print(f"   Total database: {total_posts}/150,000 ({total_posts/1500:.1f}%)")
            
            # Short pause before next cycle
            print("\n⏸  Pausing 2 seconds before next cycle...")
            time.sleep(2)

if __name__ == "__main__":
    collector = YouTubeEmploymentCollector()
    collector.run_continuous_collection()
