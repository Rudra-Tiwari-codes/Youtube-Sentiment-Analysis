"""
STEP 2: Collect EMPLOYMENT comments (30K total)
100 diverse employment/hiring/jobs keywords
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from collectors.youtube_collector_optimized import OptimizedYouTubeCollector

if __name__ == '__main__':
    print("""

 EMPLOYMENT COMMENTS COLLECTION

 Goal: Collect 30,000 employment-related comments
 Keywords: 100 jobs/hiring/careers/sectors keywords
 Strategy: Fresh videos, diverse sectors, efficient collection
 Auto-save: Every comment saved immediately

    """)
    
    collector = OptimizedYouTubeCollector(category='employment')
    
    try:
        current = collector.get_category_count()
        
        print(f" Current employment comments: {current:,}")
        print(f" Target: 30,000\n")
        
        collector.run_collection(
            target=30000,
            keywords_per_cycle=10,  # Use 10 keywords per cycle (faster with 100 keywords)
            comments_per_keyword=100  # 100 comments per keyword for diversity
        )
            
    except KeyboardInterrupt:
        print("\n\n Stopped by user")
        current = collector.get_category_count()
        print(f" Progress saved: {current:,} employment comments")
