"""
Update uncategorized YouTube comments to unemployment category
"""
import sys
sys.path.insert(0, '.')
from database.db_manager import DatabaseManager
import json

db = DatabaseManager()
conn = db.get_connection()
cursor = conn.cursor()

print("Updating uncategorized YouTube comments to 'unemployment' category...")

# Update all YouTube comments without category to unemployment
cursor.execute("""
    UPDATE raw_posts 
    SET platform_metadata = json_set(
        COALESCE(platform_metadata, '{}'), 
        '$.category', 
        'unemployment'
    )
    WHERE source = 'youtube' 
    AND (
        json_extract(platform_metadata, '$.category') IS NULL 
        OR json_extract(platform_metadata, '$.category') = ''
    )
""")

affected = cursor.rowcount
conn.commit()

print(f" Updated {affected:,} YouTube comments to unemployment category")

# Verify
cursor.execute("""
    SELECT 
        json_extract(platform_metadata, '$.category') as category,
        COUNT(*) as count
    FROM raw_posts 
    WHERE source = 'youtube'
    GROUP BY category
""")

print("\n Updated counts:")
for row in cursor.fetchall():
    category = row[0] or 'None'
    count = row[1]
    print(f"   {category}: {count:,}")

db.close()
