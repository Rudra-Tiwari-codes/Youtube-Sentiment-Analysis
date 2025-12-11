"""
STEP 1: Collect remaining UNEMPLOYMENT comments (to reach 30K)
Current: ~24K | Need: ~6K more
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collectors.youtube_collector_optimized import OptimizedYouTubeCollector

if __name__ == '__main__':
    print("""

 UNEMPLOYMENT COMMENTS COLLECTION

 Goal: Reach 30,000 unemployment-related comments
 Keywords: 72 unemployment/layoffs/crisis keywords
 Strategy: Skip already-processed videos, collect efficiently
 Auto-save: Every comment saved immediately

    """)
    
    collector = OptimizedYouTubeCollector(category='unemployment')
    
    try:
        current = collector.get_category_count()
        remaining = 30000 - current
        
        print(f" Current unemployment comments: {current:,}")
        print(f" Need {remaining:,} more to reach 30K\n")
        
        if remaining > 0:
            collector.run_collection(
                target=30000,
                keywords_per_cycle=8,
                comments_per_keyword=120
            )
        else:
            print(" Already at 30K target!")
            
    except KeyboardInterrupt:
        print("\n\n Stopped by user")
        current = collector.get_category_count()
        print(f" Progress saved: {current:,} unemployment comments")
