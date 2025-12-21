"""
Improved Database Module with Connection Pooling, Exit Tracking, and Error Handling
"""
import sqlite3
import datetime
import threading
from typing import Optional, List, Tuple, Dict
from contextlib import contextmanager
import logging
import config

# Setup logging
logger = logging.getLogger(__name__)

# Thread-local storage for connections
_thread_local = threading.local()

@contextmanager
def get_connection():
    """
    Context manager for database connections with proper error handling.
    Uses thread-local storage to avoid connection conflicts.
    """
    conn = None
    try:
        # Check if thread already has a connection
        if not hasattr(_thread_local, 'connection') or _thread_local.connection is None:
            conn = sqlite3.connect(
                config.DB_NAME,
                timeout=config.DB_TIMEOUT,
                check_same_thread=config.DB_CHECK_SAME_THREAD
            )
            # Enable Write-Ahead Logging for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row  # Access columns by name
            _thread_local.connection = conn
        else:
            conn = _thread_local.connection
            
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Unexpected error in database operation: {e}")
        raise

def initialize_db() -> None:
    """
    Creates the database and tables if they don't exist.
    Includes both entry and exit tracking.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Main vehicle logs table with entry and exit tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    image_path TEXT,
                    allocated_spot INTEGER,
                    destination_building TEXT,
                    parking_duration_seconds INTEGER,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_plate_active 
                ON vehicle_logs(plate_number, is_active)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_entry_time 
                ON vehicle_logs(entry_time DESC)
            ''')
            
            # System events log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            logger.info(f"✅ Database initialized: {config.DB_NAME}")
            print(f"✅ Database initialized: {config.DB_NAME}")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def save_entry(
    plate_text: str,
    image_path: str,
    allocated_spot: Optional[int] = None,
    destination: Optional[str] = None
) -> Optional[int]:
    """
    Saves a new vehicle entry to the database.
    
    Args:
        plate_text: License plate number
        image_path: Path to stored plate image
        allocated_spot: Parking spot allocated by AI
        destination: Building destination (A or B)
        
    Returns:
        Entry ID if successful, None otherwise
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.datetime.now()
            
            cursor.execute('''
                INSERT INTO vehicle_logs 
                (plate_number, entry_time, image_path, allocated_spot, destination_building, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
            ''', (plate_text, timestamp, image_path, allocated_spot, destination))
            
            entry_id = cursor.lastrowid
            logger.info(f"💾 Entry saved: {plate_text} | Spot {allocated_spot} | ID {entry_id}")
            print(f"💾 Saved to DB: {plate_text} | {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            return entry_id
            
    except sqlite3.IntegrityError as e:
        logger.error(f"Database integrity error saving entry for {plate_text}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error saving entry for {plate_text}: {e}")
        return None

def save_exit(plate_text: str) -> bool:
    """
    Records vehicle exit and calculates parking duration.
    
    Args:
        plate_text: License plate number
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            exit_time = datetime.datetime.now()
            
            # Find the most recent active entry for this plate
            cursor.execute('''
                SELECT id, entry_time 
                FROM vehicle_logs 
                WHERE plate_number = ? AND is_active = 1
                ORDER BY entry_time DESC 
                LIMIT 1
            ''', (plate_text,))
            
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No active entry found for plate: {plate_text}")
                return False
            
            entry_id = result['id']
            entry_time = datetime.datetime.fromisoformat(result['entry_time'])
            
            # Calculate duration
            duration = (exit_time - entry_time).total_seconds()
            
            # Update the record
            cursor.execute('''
                UPDATE vehicle_logs 
                SET exit_time = ?,
                    parking_duration_seconds = ?,
                    is_active = 0
                WHERE id = ?
            ''', (exit_time, int(duration), entry_id))
            
            logger.info(f"🚗 Exit logged: {plate_text} | Duration: {int(duration)}s")
            print(f"🚗 Exit logged: {plate_text} | Duration: {int(duration)}s")
            return True
            
    except Exception as e:
        logger.error(f"Error saving exit for {plate_text}: {e}")
        return False

def get_active_vehicles() -> List[Dict]:
    """
    Returns list of currently parked vehicles.
    
    Returns:
        List of dictionaries containing vehicle information
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    id,
                    plate_number,
                    entry_time,
                    allocated_spot,
                    destination_building
                FROM vehicle_logs 
                WHERE is_active = 1
                ORDER BY entry_time DESC
            ''')
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Error fetching active vehicles: {e}")
        return []

def is_vehicle_currently_parked(plate_text: str) -> bool:
    """
    Check if a vehicle is currently in the parking lot.
    
    Args:
        plate_text: License plate number
        
    Returns:
        True if vehicle is currently parked
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) as count
                FROM vehicle_logs 
                WHERE plate_number = ? AND is_active = 1
            ''', (plate_text,))
            
            result = cursor.fetchone()
            return result['count'] > 0
            
    except Exception as e:
        logger.error(f"Error checking vehicle status for {plate_text}: {e}")
        return False

def get_parking_statistics() -> Dict:
    """
    Returns parking lot statistics.
    
    Returns:
        Dictionary with statistics
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute("SELECT COUNT(*) as count FROM vehicle_logs")
            total_entries = cursor.fetchone()['count']
            
            # Currently active
            cursor.execute("SELECT COUNT(*) as count FROM vehicle_logs WHERE is_active = 1")
            active_count = cursor.fetchone()['count']
            
            # Average parking duration (completed stays only)
            cursor.execute('''
                SELECT AVG(parking_duration_seconds) as avg_duration 
                FROM vehicle_logs 
                WHERE parking_duration_seconds IS NOT NULL
            ''')
            avg_duration = cursor.fetchone()['avg_duration'] or 0
            
            return {
                'total_entries': total_entries,
                'currently_parked': active_count,
                'average_duration_seconds': int(avg_duration),
                'available_spots': len(config.PARKING_SPOTS) - active_count
            }
            
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        return {}

def fetch_all_logs(limit: int = 100) -> List[Dict]:
    """
    Reads recent vehicle logs.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of log dictionaries
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(f'''
                SELECT * FROM vehicle_logs 
                ORDER BY entry_time DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return []

def log_system_event(event_type: str, event_data: str = "") -> None:
    """
    Logs system events for debugging and monitoring.
    
    Args:
        event_type: Type of event (e.g., 'GATE_OPEN', 'ERROR', 'STARTUP')
        event_data: Additional event information
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_events (event_type, event_data)
                VALUES (?, ?)
            ''', (event_type, event_data))
            
    except Exception as e:
        logger.error(f"Error logging system event: {e}")

def close_all_connections() -> None:
    """
    Closes all database connections. Call on shutdown.
    """
    if hasattr(_thread_local, 'connection') and _thread_local.connection:
        try:
            _thread_local.connection.close()
            _thread_local.connection = None
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

# --- Quick Test ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    initialize_db()
    
    # Test entry
    entry_id = save_entry("ABC 123", "stored_plates/test.jpg", allocated_spot=1, destination="A")
    print(f"Created entry ID: {entry_id}")
    
    # Check if parked
    print(f"Is parked: {is_vehicle_currently_parked('ABC 123')}")
    
    # Get stats
    print(f"Statistics: {get_parking_statistics()}")
    
    # Test exit
    import time
    time.sleep(2)  # Simulate parking duration
    save_exit("ABC 123")
    
    # Show results
    print("\nRecent logs:")
    for log in fetch_all_logs(5):
        print(log)