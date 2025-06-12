import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'memory.db')

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS traces (
        id INTEGER PRIMARY KEY,
        layer TEXT,
        vector BLOB,
        label INTEGER,
        epoch INTEGER,
        strength REAL
    )
''')
conn.commit()
conn.close()
print("✅ Base initialisée :", DB_PATH)

