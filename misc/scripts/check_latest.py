import sqlite3
from datetime import datetime

db = "./data/unemployment_sentiment.db"
conn = sqlite3.connect(db)
cursor = conn.cursor()

print("\n" + "="*60)
print("LATEST ENTRIES BY SOURCE")
print("="*60)

cursor.execute("""
    SELECT source, COUNT(*) as count, MAX(created_date) as latest 
    FROM raw_posts 
    GROUP BY source
""")

for row in cursor.fetchall():
    source, count, latest = row
    print(f"{source:15} | {count:6} | {latest}")

print("\n" + "="*60)
print("ENTRIES IN LAST 5 MINUTES")
print("="*60)

from datetime import datetime, timedelta
five_min_ago = (datetime.now() - timedelta(minutes=5)).isoformat()

cursor.execute("""
    SELECT source, COUNT(*) as count
    FROM raw_posts 
    WHERE created_date >= ?
    GROUP BY source
""", (five_min_ago,))

results = cursor.fetchall()
if results:
    for row in results:
        print(f"{row[0]:15} | {row[1]} new entries")
else:
    print("  NO NEW ENTRIES IN LAST 5 MINUTES!")
    print("Collectors might be stuck or hitting rate limits")

conn.close()
