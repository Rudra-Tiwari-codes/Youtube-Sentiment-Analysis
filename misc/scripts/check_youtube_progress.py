"""
Quick YouTube Collection Progress Checker
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import DatabaseManager

db = DatabaseManager()

print("\n" + "="*60)
print(" YOUTUBE COLLECTION PROGRESS")
print("="*60)

# Get total counts
conn = db.get_connection()
cursor = conn.cursor()

# Count by category
cursor.execute("""
    SELECT 
        json_extract(platform_metadata, '$.category') as category,
        COUNT(*) as count
    FROM raw_posts 
    WHERE source = 'youtube'
    GROUP BY category
""")

results = cursor.fetchall()

unemployment = 0
employment = 0
other = 0

for row in results:
    category = row[0]
    count = row[1]
    if category == 'unemployment':
        unemployment = count
    elif category == 'employment':
        employment = count
    else:
        other = count

total = unemployment + employment + other

print(f"\n Unemployment: {unemployment:,} / 30,000 ({unemployment/300:.1f}%)")
print(f" Employment: {employment:,} / 30,000 ({employment/300:.1f}%)")
if other > 0:
    print(f" Other: {other:,}")
print(f"\n Total YouTube: {total:,} / 60,000 ({total/600:.1f}%)")

remaining_unemployment = 30000 - unemployment
remaining_employment = 30000 - employment

print(f"\n Remaining:")
print(f"   Unemployment: {remaining_unemployment:,}")
print(f"   Employment: {remaining_employment:,}")

print("\n" + "="*60)
