"""
Centralized Configuration for Smart Parking System
All configuration variables in one place for easy management.
"""
import os
from typing import Dict

# --- DATABASE CONFIGURATION ---
DB_NAME: str = "parking_system.db"
DB_TIMEOUT: int = 30  # seconds
DB_CHECK_SAME_THREAD: bool = False  # Allow multi-threading

# --- HARDWARE PIN CONFIGURATION ---
# Servo Motors
ENTRY_SERVO_PIN: int = 13    # GPIO 13 (Pin 33)
EXIT_SERVO_PIN: int = 12     # GPIO 12 (Pin 32)

# Sensors
IR_SENSOR_PIN: int = 23      # GPIO 23 (Pin 16) - Exit Sensor
SPOT1_PIN: int = 24          # GPIO 24 (Pin 18)
SPOT2_PIN: int = 27          # GPIO 27 (Pin 13)
SPOT3_PIN: int = 22          # GPIO 22 (Pin 15)

# Servo Configuration
SERVO_FREQUENCY: int = 50    # Hz
SERVO_CLOSED_ANGLE: int = 0  # degrees
SERVO_OPEN_ANGLE: int = 90   # degrees
SERVO_MOVE_DELAY: float = 0.3  # seconds
GATE_AUTO_CLOSE_TIME: int = 5  # seconds

# --- PARKING LAYOUT CONFIGURATION ---
PARKING_SPOTS: Dict[int, Dict] = {
    0: {"id": 1, "location": 1.0},  # Spot 1 - Near Building A
    1: {"id": 2, "location": 2.0},  # Spot 2 - Middle
    2: {"id": 3, "location": 3.0}   # Spot 3 - Near Building B
}

BUILDINGS: Dict[str, float] = {
    "A": 0.0,  # Building A location
    "B": 4.0   # Building B location
}

# --- AI MODEL CONFIGURATION ---
PLATE_MODEL_PATH: str = "plate_detector.pt"
CHAR_MODEL_PATH: str = "character_detector.pt"
TD3_MODEL_PATH: str = "td3_actor.pth"

# Model Parameters
PLATE_CONFIDENCE: float = 0.4
CHAR_CONFIDENCE: float = 0.35
PLATE_MIN_WIDTH: int = 70  # pixels
IMAGE_SIZE: int = 640

# TD3 Network Configuration
TD3_STATE_DIM: int = 4  # 3 sensors + 1 destination
TD3_ACTION_DIM: int = 1  # continuous output
TD3_HIDDEN_DIM: int = 64
TD3_LEARNING_RATE: float = 0.001

# Action Decoding Thresholds
ACTION_THRESHOLD_LOW: float = -0.33  # Below this -> Spot 1
ACTION_THRESHOLD_HIGH: float = 0.33  # Above this -> Spot 3

# --- VISUAL CONFIGURATION ---
FONT_PATH: str = "NotoSansArabic-Regular.ttf"
FONT_SIZE: int = 50
CROP_PADDING: int = 5  # pixels around detected plate

# --- STORAGE CONFIGURATION ---
OUTPUT_FOLDER: str = "stored_plates"
BLOCKCHAIN_FILE: str = "secure_ledger.json"
LOG_FILE: str = "parking_system.log"

# --- SYSTEM BEHAVIOR CONFIGURATION ---
DUPLICATE_COOLDOWN: int = 10  # seconds - prevent same plate re-logging
CAMERA_RETRY_DELAY: int = 2   # seconds - wait before camera reconnect
CAMERA_ID: int = 0  # Default camera device

# --- TRAINING CONFIGURATION ---
TRAINING_EPISODES: int = 10000
MAX_DISTANCE: float = 4.0  # Maximum distance in layout
PENALTY_OCCUPIED_SPOT: int = -100  # Penalty for choosing occupied spot
REWARD_DISTANCE_MULTIPLIER: float = 2.0  # How much to penalize distance

# Ensure output directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- ARABIC CHARACTER SETS ---
ARABIC_NUMBERS = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩']

# --- VISUALIZATION COLORS (BGR format for OpenCV) ---
COLOR_FREE: tuple = (0, 255, 0)      # Green
COLOR_OCCUPIED: tuple = (0, 0, 255)  # Red
COLOR_DETECTED: tuple = (0, 255, 255) # Yellow
COLOR_LOGGED: tuple = (0, 255, 0)    # Green
COLOR_TEXT: tuple = (255, 200, 0)    # Cyan-ish
