"""Check current database statistics"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager

db = DatabaseManager()
stats = db.get_stats()

print('='*60)
print('CURRENT DATABASE STATISTICS')
print('='*60)
print(f'Total posts: {stats["total_posts"]}')
print(f'\nBy source:')
for k, v in stats['by_source'].items():
    print(f'  {k}: {v}')
print(f'\nProcessing status:')
print(f'  Processed: {stats["processing"]["processed"]}')
print(f'  Pending: {stats["processing"]["pending"]}')
print('='*60)

# Target for research
target = 150_000  # 150K posts
current = stats["total_posts"]
remaining = target - current

print(f'\nTARGET: 150,000 posts')
print(f'Current: {current:,}')
print(f'Remaining: {remaining:,}')
print(f'Progress: {(current/target)*100:.2f}%')
print(f'\nDISTRIBUTION TARGET:')
print(f'  YouTube (40%): {int(target * 0.40):,} posts')
print(f'  Twitter (40%): {int(target * 0.40):,} posts')
print(f'  News (30%): {int(target * 0.30):,} posts')
print('='*60)
