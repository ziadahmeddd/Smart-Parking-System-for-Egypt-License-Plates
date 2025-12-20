import sqlite3
import os

DB_NAME = "parking_system.db"

def full_reset():
    if not os.path.exists(DB_NAME):
        print("❌ Database not found.")
        return

    # Connect with timeout to avoid "Database Locked" errors
    conn = sqlite3.connect(DB_NAME, timeout=10)
    cursor = conn.cursor()
    
    try:
        # 1. Delete all car data
        cursor.execute("DELETE FROM vehicle_logs")
        
        # 2. RESET THE ID COUNTER TO 1 (Crucial Step)
        # SQLite stores the auto-increment count in a hidden table called 'sqlite_sequence'
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='vehicle_logs'")
        
        conn.commit()
        print("✅ Database wiped. Next ID will be 1.")
        
    except sqlite3.OperationalError as e:
        print(f"❌ Error: {e}")
        print("💡 Hint: Close 'DB Browser' if it is open!")
    finally:
        conn.close()

if __name__ == "__main__":
    full_reset()