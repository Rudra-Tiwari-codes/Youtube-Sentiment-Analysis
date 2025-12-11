import sys
sys.path.insert(0, '.')
from database.db_manager import DatabaseManager

db = DatabaseManager()
conn = db.get_connection()
cursor = conn.cursor()

print("\n" + "="*60)
print("DETAILED YOUTUBE CATEGORY BREAKDOWN")
print("="*60)

# Get all categories
cursor.execute("""
    SELECT 
        json_extract(platform_metadata, '$.category') as category,
        COUNT(*) as count
    FROM raw_posts 
    WHERE source = 'youtube'
    GROUP BY category
    ORDER BY count DESC
""")

results = cursor.fetchall()
total = 0

print("\nCategories found:")
for row in results:
    cat = row[0] if row[0] else 'None/NULL'
    count = row[1]
    total += count
    print(f"  {cat}: {count:,}")

print(f"\n TOTAL YouTube: {total:,}")

# Double check employment count
cursor.execute("""
    SELECT COUNT(*) FROM raw_posts
    WHERE source = 'youtube' 
    AND json_extract(platform_metadata, '$.category') = 'employment'
""")
emp_count = cursor.fetchone()[0]

print(f"\n Employment (verified): {emp_count:,}")

# Check unemployment count
cursor.execute("""
    SELECT COUNT(*) FROM raw_posts
    WHERE source = 'youtube' 
    AND json_extract(platform_metadata, '$.category') = 'unemployment'
""")
unemp_count = cursor.fetchone()[0]

print(f" Unemployment (verified): {unemp_count:,}")
print("="*60)
