"""Run SUPER OPTIMIZED Employment Collector"""
import sys
import time
from datetime import datetime, timedelta
sys.path.insert(0, '.')
from collectors.youtube_super_optimized import SuperOptimizedCollector

print("""

 SUPER OPTIMIZED EMPLOYMENT COLLECTOR - 2 HOUR RUN

 Target: 30,000 employment comments
⏱  Max runtime: 2 hours (or until target/quota reached)
 Keywords: 100 diverse employment keywords
 OPTIMIZATIONS:
   • Videos sorted by VIEW COUNT (popular = more comments)
   • Filters videos with ZERO comments BEFORE fetching
   • Results sorted by COMMENT COUNT (most efficient first)
   • Adaptive batch sizing (faster when far from target)
   • Smart keyword rotation
   • Auto-switch to backup API key if quota exhausted
 Every comment saved immediately

""")

collector = SuperOptimizedCollector('employment')
start_time = datetime.now()
end_time = start_time + timedelta(hours=2)

print(f"⏰ Started at: {start_time.strftime('%H:%M:%S')}")
print(f"⏰ Will run until: {end_time.strftime('%H:%M:%S')} (or target reached)")
print("="*80 + "\n")

try:
    # Modify run_collection to check time
    target = 30000
    cycle = 0
    
    while True:
        cycle += 1
        current = collector.get_count()
        remaining = target - current
        
        # Check if target reached
        if remaining <= 0:
            print(f"\n TARGET REACHED! {current:,} comments!")
            break
        
        # Check if 2 hours elapsed
        if datetime.now() >= end_time:
            print(f"\n⏰ 2 HOUR LIMIT REACHED!")
            print(f" Collected: {current:,} / {target:,} ({(current/target)*100:.1f}%)")
            print(f" Remaining: {remaining:,}")
            break
        
        print(f"\n{'='*80}")
        print(f"[Cycle #{cycle}] {datetime.now().strftime('%H:%M:%S')}")
        print(f" Current: {current:,} | Remaining: {remaining:,} | Progress: {(current/target)*100:.1f}%")
        print(f"⏱  Time remaining: {str(end_time - datetime.now()).split('.')[0]}")
        print(f"{'='*80}")
        
        # Refill keyword pool
        if not collector.keyword_pool:
            collector.keyword_pool = collector.keywords.copy()
            import random
            random.shuffle(collector.keyword_pool)
            print(f" Reshuffled {len(collector.keyword_pool)} keywords")
        
        # Adaptive batch size
        batch_size = min(20, max(5, remaining // 200))
        batch = collector.keyword_pool[:batch_size]
        collector.keyword_pool = collector.keyword_pool[batch_size:]
        
        print(f" Processing {len(batch)} keywords")
        
        cycle_total = 0
        for keyword in batch:
            collected = collector.collect_for_keyword(keyword, target_comments=250)
            cycle_total += collected
            
            if collected > 0:
                remaining -= collected
                if remaining <= 0:
                    break
            
            # Check time every keyword
            if datetime.now() >= end_time:
                break
        
        print(f"\n Cycle #{cycle}: +{cycle_total} comments")
        print(f" Total: {collector.get_count():,}/{target:,}")
        
        time.sleep(2)
    
    # Final stats
    end_actual = datetime.now()
    duration = end_actual - start_time
    final_count = collector.get_count()
    
    print(f"\n{'='*80}")
    print(f" FINAL STATISTICS")
    print(f"{'='*80}")
    print(f" Employment comments: {final_count:,} / {target:,} ({(final_count/target)*100:.1f}%)")
    print(f"⏱  Duration: {str(duration).split('.')[0]}")
    print(f" Rate: {final_count / (duration.total_seconds() / 60):.1f} comments/minute")
    print(f" All data saved to database!")
    print(f"{'='*80}")
    
except KeyboardInterrupt:
    print("\n\n Stopped by user")
    print(f" Current count: {collector.get_count():,}")
    print(" All progress saved!")
except Exception as e:
    print(f"\n Error: {str(e)}")
    if "quota" in str(e).lower() or "exhausted" in str(e).lower():
        print("  All API keys exhausted!")
    print(f" Current count: {collector.get_count():,}")
    print(" All progress saved!")
