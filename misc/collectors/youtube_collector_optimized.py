"""
OPTIMIZED YOUTUBE COLLECTOR
- Tracks unemployment vs employment separately  
- Avoids re-processing same videos
- Efficient API usage
- Fast collection with smart video selection
"""

import os
import sys
import time
import random
import hashlib
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from utils.logger import get_logger
from youtube_keywords import UNEMPLOYMENT_KEYWORDS, EMPLOYMENT_KEYWORDS

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

class OptimizedYouTubeCollector:
    def __init__(self, category='unemployment'):
        """
        Initialize collector for specific category
        Args:
            category: 'unemployment' or 'employment'
        """
        import httplib2
        http = httplib2.Http(timeout=10)
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, http=http)
        self.db = DatabaseManager()
        self.logger = get_logger(f'youtube_collector_{category}')
        
        # Category tracking
        self.category = category
        self.keywords = UNEMPLOYMENT_KEYWORDS if category == 'unemployment' else EMPLOYMENT_KEYWORDS
        self.keyword_pool = self.keywords.copy()
        random.shuffle(self.keyword_pool)
        
        # Track processed videos to avoid duplicates
        self.processed_videos = set()
        self.load_processed_videos()
        
        print(f"\n{'='*80}")
        print(f" OPTIMIZED YOUTUBE COLLECTOR - {category.upper()}")
        print(f"{'='*80}")
        print(f" Keywords loaded: {len(self.keywords)}")
        print(f" Previously processed videos: {len(self.processed_videos)}")
        print(f"{'='*80}\n")
    
    def load_processed_videos(self):
        """Load video IDs that have already been processed"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Get all unique video IDs from metadata
            cursor.execute("""
                SELECT DISTINCT platform_metadata 
                FROM raw_posts 
                WHERE source = 'youtube'
            """)
            
            for row in cursor.fetchall():
                try:
                    import json
                    metadata = json.loads(row[0]) if row[0] else {}
                    video_id = metadata.get('video_id')
                    if video_id:
                        self.processed_videos.add(video_id)
                except:
                    continue
                    
            self.logger.info(f"Loaded {len(self.processed_videos)} processed video IDs")
        except Exception as e:
            self.logger.error(f"Error loading processed videos: {e}")
    
    def search_videos(self, keyword, max_results=30):
        """Search for NEW videos (not already processed)"""
        try:
            # Vary parameters for diversity
            days_options = [30, 60, 90, 180]
            sort_options = ['relevance', 'date', 'viewCount', 'rating']
            
            days_back = random.choice(days_options)
            sort_order = random.choice(sort_options)
            published_after = (datetime.now() - timedelta(days=days_back)).isoformat() + 'Z'
            
            search_response = self.youtube.search().list(
                q=keyword,
                part='id,snippet',
                type='video',
                maxResults=max_results,
                order=sort_order,
                publishedAfter=published_after,
                regionCode='IN',
                relevanceLanguage='en'
            ).execute()
            
            # Filter out already processed videos
            new_videos = []
            for item in search_response.get('items', []):
                if item['id']['kind'] == 'youtube#video':
                    video_id = item['id']['videoId']
                    
                    # Skip if already processed
                    if video_id in self.processed_videos:
                        continue
                    
                    new_videos.append({
                        'video_id': video_id,
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle']
                    })
            
            return new_videos
            
        except HttpError as e:
            self.logger.error(f"YouTube API error: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error searching '{keyword}': {e}")
            return []
    
    def get_video_comments(self, video_id, max_comments=100):
        """Get comments from a video with category tag"""
        comments_data = []
        
        try:
            # Get video details
            try:
                video_response = self.youtube.videos().list(
                    part='snippet,statistics',
                    id=video_id
                ).execute()
            except:
                return []
            
            if not video_response.get('items'):
                return []
            
            video_info = video_response['items'][0]
            video_title = video_info['snippet']['title']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            channel_title = video_info['snippet']['channelTitle']
            
            # Get comments
            try:
                comments_response = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=min(100, max_comments),
                    order='relevance',
                    textFormat='plainText'
                ).execute()
            except:
                return []
            
            for item in comments_response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                
                # Only substantial comments
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
                            'video_id': video_id,
                            'category': self.category  # Tag with category!
                        },
                        'likes': comment.get('likeCount', 0),
                        'comments_count': 0,
                        'shares': 0
                    })
            
            # Mark video as processed
            if comments_data:
                self.processed_videos.add(video_id)
            
            return comments_data
            
        except Exception as e:
            self.logger.error(f"Error getting comments for {video_id}: {e}")
            return []
    
    def collect_for_keyword(self, keyword, target_comments=200):
        """Optimized comment collection"""
        print(f"\n {keyword}")
        
        total_collected = 0
        
        # Search for videos (already filtered & sorted)
        videos = self.search_videos(keyword, max_results=50)
        
        if not videos:
            print(f"   ⊘ No new videos with comments")
            return 0
        
        # Show top videos by comment count
        top_counts = [v['comment_count'] for v in videos[:5]]
        print(f"    {len(videos)} videos (top comments: {top_counts})")
        
        # Process videos (already sorted by comment count)
        videos_processed = 0
        for video in videos:
            if total_collected >= target_comments:
                break
            
            videos_processed += 1
            print(f"   [{videos_processed}] {video['title'][:40]}... ({video['comment_count']} cmts) ", end="", flush=True)
            
            comments = self.get_video_comments(video['video_id'], max_comments=100)
            
            if comments:
                successful, duplicates = self.db.insert_batch('youtube', comments)
                total_collected += successful
                if successful > 0:
                    print(f" +{successful}")
                else:
                    print(f"⊘ (all duplicates)")
            else:
                print(f"⊘")
            
            # Reduced delay since we're being smarter
            time.sleep(random.uniform(0.2, 0.5))
        
        print(f"    Collected: {total_collected} from {videos_processed} videos")
        return total_collected
    
    def get_category_count(self):
        """Get count for this category"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Count posts with this category in metadata
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM raw_posts
                WHERE source = 'youtube' 
                AND platform_metadata LIKE ?
            """, (f'%"category": "{self.category}"%',))
            
            result = cursor.fetchone()
            return result['count'] if result else 0
        except:
            return 0
    
    def run_collection(self, target=30000, keywords_per_cycle=10, comments_per_keyword=150):
        """
        Run optimized collection
        Args:
            target: Total comments to collect (30K unemployment or 30K employment)
            keywords_per_cycle: How many keywords per cycle (more = faster)
            comments_per_keyword: Comments to collect per keyword (less = more diversity)
        """
        
        print(f"\n TARGET: {target:,} {self.category} comments")
        
        cycle = 0
        
        while True:
            cycle += 1
            
            # Get current count
            current_count = self.get_category_count()
            remaining = target - current_count
            
            print(f"\n{'='*80}")
            print(f"[Cycle #{cycle}] {datetime.now().strftime('%H:%M:%S')}")
            print(f" Current: {current_count:,} | Remaining: {remaining:,} | {(current_count/target)*100:.1f}%")
            print(f"{'='*80}")
            
            if current_count >= target:
                print(f"\n TARGET REACHED! {current_count:,}/{target:,}")
                break
            
            # Refresh keyword pool if needed
            if len(self.keyword_pool) < keywords_per_cycle:
                self.keyword_pool = self.keywords.copy()
                random.shuffle(self.keyword_pool)
                print(" Keyword pool refreshed")
            
            # Select keywords
            selected = [self.keyword_pool.pop(0) for _ in range(min(keywords_per_cycle, len(self.keyword_pool)))]
            
            print(f" Keywords: {', '.join(selected[:3])}{'...' if len(selected) > 3 else ''}")
            
            cycle_total = 0
            
            for keyword in selected:
                collected = self.collect_for_keyword(keyword, target_comments=comments_per_keyword)
                cycle_total += collected
                time.sleep(1)  # Brief pause between keywords
            
            print(f"\n Cycle #{cycle} Complete: +{cycle_total} comments")
            print(f" Total {self.category}: {current_count + cycle_total:,}")
            
            time.sleep(2)  # API cooldown
        
        # Final stats
        final_count = self.get_category_count()
        print(f"\n{'='*80}")
        print(" COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"Category: {self.category.upper()}")
        print(f"Collected: {final_count:,}")
        print(f"Videos processed: {len(self.processed_videos):,}")
        print(f"{'='*80}\n")


def main():
    """Run collection based on command line argument"""
    import sys
    
    # Default to unemployment if no argument
    category = sys.argv[1] if len(sys.argv) > 1 else 'unemployment'
    
    if category not in ['unemployment', 'employment']:
        print("Usage: python youtube_collector_optimized.py [unemployment|employment]")
        sys.exit(1)
    
    collector = OptimizedYouTubeCollector(category=category)
    
    try:
        if category == 'unemployment':
            # Get remaining to 30K
            current = collector.get_category_count()
            remaining = 30000 - current
            print(f"\n Current unemployment comments: {current:,}")
            print(f" Need {remaining:,} more to reach 30K\n")
            
            if remaining > 0:
                collector.run_collection(
                    target=30000,
                    keywords_per_cycle=8,  # Faster collection
                    comments_per_keyword=120  # More diversity
                )
            else:
                print(" Already at target!")
        else:
            # Employment: collect 30K
            collector.run_collection(
                target=30000,
                keywords_per_cycle=10,  # Even faster with 100 keywords
                comments_per_keyword=100
            )
    
    except KeyboardInterrupt:
        print("\n\n Stopped by user")
        current = collector.get_category_count()
        print(f" Progress saved: {current:,} {category} comments")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
