"""
Smart Parking System - Main Controller
Improved version with proper error handling, logging, and modular design.
"""
import cv2
from ultralytics import YOLO
import time
import datetime
import os
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image

# Import our modules
import config
import database
import td3_parking
import simple_blockchain
import logger as log_module

# Setup logging
logger = log_module.get_logger(__name__)

# --- SMART GPIO IMPORT ---
try:
    import RPi.GPIO as GPIO
    IS_RASPBERRY_PI = True
    logger.info("✅ Running on Raspberry Pi (Hardware Enabled)")
    print("✅ Running on Raspberry Pi (Hardware Enabled)")
except (ImportError, RuntimeError):
    IS_RASPBERRY_PI = False
    logger.info("⚠️ Running on PC/Laptop (Simulation Mode)")
    print("⚠️ Running on PC/Laptop (Simulation Mode)")
    
    # Simulation Mock Class
    class GPIO_Mock:
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        PUD_UP = "PUD_UP"
        
        def setmode(self, mode): pass
        def setwarnings(self, flag): pass
        def setup(self, pin, mode, pull_up_down=None): pass
        def output(self, pin, state):
            print(f"[SIM] Pin {pin} -> {state}")
        def input(self, pin):
            return 1  # Default to Free spot in simulation
        def cleanup(self): pass
        
        class PWM:
            def __init__(self, pin, freq):
                self.pin = pin
                self.freq = freq
            def start(self, duty): pass
            def ChangeDutyCycle(self, duty):
                print(f"[SIM] Servo Pin {self.pin} -> Duty {duty}")
            def stop(self): pass
    
    GPIO = GPIO_Mock()
    GPIO.PWM = GPIO_Mock.PWM

# --- HARDWARE MANAGER CLASS ---
class HardwareManager:
    """Manages GPIO pins, servos, and sensors with error handling."""
    
    def __init__(self):
        """Initialize hardware components."""
        self.entry_pwm = None
        self.exit_pwm = None
        self.initialized = False
        
        try:
            self._setup_gpio()
            self.initialized = True
            logger.info("✅ Hardware initialized successfully")
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}", exc_info=True)
            raise
    
    def _setup_gpio(self):
        """Setup GPIO pins and servos."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup Sensors (Pull_UP: 1=Free, 0=Occupied when metal detected)
        GPIO.setup(config.IR_SENSOR_PIN, GPIO.IN)
        GPIO.setup(config.SPOT1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(config.SPOT2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(config.SPOT3_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Setup Servos
        GPIO.setup(config.ENTRY_SERVO_PIN, GPIO.OUT)
        GPIO.setup(config.EXIT_SERVO_PIN, GPIO.OUT)
        
        self.entry_pwm = GPIO.PWM(config.ENTRY_SERVO_PIN, config.SERVO_FREQUENCY)
        self.exit_pwm = GPIO.PWM(config.EXIT_SERVO_PIN, config.SERVO_FREQUENCY)
        self.entry_pwm.start(0)
        self.exit_pwm.start(0)
    
    def move_servo(self, pwm_instance, pin: int, angle: int) -> bool:
        """
        Moves a servo to specified angle.
        
        Args:
            pwm_instance: PWM instance
            pin: GPIO pin number
            angle: Target angle (0=Closed, 90=Open)
            
        Returns:
            True if successful
        """
        try:
            duty = 2 + (angle / 18)
            GPIO.output(pin, True)
            pwm_instance.ChangeDutyCycle(duty)
            time.sleep(config.SERVO_MOVE_DELAY)
            GPIO.output(pin, False)
            pwm_instance.ChangeDutyCycle(0)
            return True
        except Exception as e:
            logger.error(f"Servo movement failed on pin {pin}: {e}")
            return False
    
    def read_sensors(self) -> List[int]:
        """
        Read all parking spot sensors.
        
        Returns:
            List of sensor states [spot1, spot2, spot3] where 1=Free, 0=Occupied
        """
        try:
            raw_s1 = GPIO.input(config.SPOT1_PIN)
            raw_s2 = GPIO.input(config.SPOT2_PIN)
            raw_s3 = GPIO.input(config.SPOT3_PIN)
            
            # Convert to logic: 0 = Occupied (metal detected), 1 = Free
            s1 = 0 if raw_s1 == 0 else 1
            s2 = 0 if raw_s2 == 0 else 1
            s3 = 0 if raw_s3 == 0 else 1
            
            return [s1, s2, s3]
        except Exception as e:
            logger.error(f"Sensor read error: {e}")
            return [1, 1, 1]  # Default to all free on error
    
    def read_ir_sensor(self) -> int:
        """Read IR sensor state (exit detection)."""
        try:
            return GPIO.input(config.IR_SENSOR_PIN)
        except Exception as e:
            logger.error(f"IR sensor read error: {e}")
            return 1  # Default to not triggered
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        try:
            if self.entry_pwm:
                self.entry_pwm.stop()
            if self.exit_pwm:
                self.exit_pwm.stop()
            GPIO.cleanup()
            logger.info("Hardware cleanup completed")
        except Exception as e:
            logger.error(f"Hardware cleanup error: {e}")

# --- CAMERA MANAGER CLASS ---
class CameraManager:
    """Manages camera with auto-reconnection."""
    
    def __init__(self, camera_id: int = 0):
        """Initialize camera."""
        self.camera_id = camera_id
        self.cap = None
        self.last_reconnect_time = 0
        self.reconnect_attempts = 0
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to camera."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"✅ Camera {self.camera_id} connected")
            self.reconnect_attempts = 0
            return True
            
        except Exception as e:
            logger.error(f"Camera connection error: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from camera with auto-reconnect on failure.
        
        Returns:
            (success, frame) tuple
        """
        if self.cap is None or not self.cap.isOpened():
            if time.time() - self.last_reconnect_time > config.CAMERA_RETRY_DELAY:
                logger.warning("Attempting camera reconnection...")
                if self._connect():
                    self.last_reconnect_time = time.time()
                else:
                    return False, None
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Camera read failed, scheduling reconnect")
                self.last_reconnect_time = time.time()
                self.reconnect_attempts += 1
                return False, None
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Camera read exception: {e}")
            return False, None
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")

# --- HELPER FUNCTIONS ---
def draw_arabic(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 32,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw Arabic text on image with proper RTL rendering.
    
    Args:
        img: OpenCV image (BGR)
        text: Arabic text to draw
        position: (x, y) position
        font_size: Font size
        color: RGB color tuple
        
    Returns:
        Image with text drawn
    """
    if not text:
        return img
    
    try:
        # Convert to RTL
        bidi_text = get_display(text, base_dir='R')
        
        # Convert to PIL for text rendering
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Load font
        try:
            font = ImageFont.truetype(config.FONT_PATH, font_size)
        except Exception as e:
            logger.warning(f"Failed to load font: {e}, using default")
            font = ImageFont.load_default()
        
        # Draw text with outline
        draw.text(
            position,
            bidi_text,
            font=font,
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0)
        )
        
        # Convert back to OpenCV
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        logger.error(f"Arabic text drawing error: {e}")
        return img

# --- PLATE RECOGNITION CLASS ---
class PlateRecognizer:
    """Handles license plate detection and OCR."""
    
    def __init__(self):
        """Initialize recognition models."""
        logger.info("🧠 Loading AI Models...")
        print("🧠 Loading AI Models...")
        
        try:
            self.plate_model = YOLO(config.PLATE_MODEL_PATH)
            self.char_model = YOLO(config.CHAR_MODEL_PATH)
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            raise
    
    def detect_and_read_plate(
        self,
        img: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect and read license plates in image.
        
        Args:
            img: Input image
            
        Returns:
            List of detected plates with information
        """
        plates = []
        
        try:
            # Detect plates
            results = self.plate_model.predict(
                img,
                imgsz=config.IMAGE_SIZE,
                conf=config.PLATE_CONFIDENCE,
                verbose=False
            )
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Filter out too-small detections
                    if (x2 - x1) < config.PLATE_MIN_WIDTH:
                        continue
                    
                    # Extract plate crop
                    pad = config.CROP_PADDING
                    h, w = img.shape[:2]
                    crop = img[
                        max(0, y1-pad):min(h, y2+pad),
                        max(0, x1-pad):min(w, x2+pad)
                    ]
                    
                    # OCR on crop
                    plate_text = self._read_characters(crop)
                    
                    if plate_text:
                        plates.append({
                            'text': plate_text,
                            'bbox': (x1, y1, x2, y2),
                            'crop': crop
                        })
            
            return plates
            
        except Exception as e:
            logger.error(f"Plate detection error: {e}")
            return []
    
    def _read_characters(self, crop: np.ndarray) -> str:
        """Read characters from plate crop using OCR."""
        try:
            char_results = self.char_model.predict(
                crop,
                conf=config.CHAR_CONFIDENCE,
                verbose=False
            )
            
            detected_items = []
            for r in char_results:
                for b in r.boxes:
                    cls = int(b.cls[0])
                    sym = self.char_model.names[cls]
                    cx = b.xywh[0][0].item()
                    detected_items.append((cx, sym))
            
            if not detected_items:
                return ""
            
            # Sort by x-position
            detected_items.sort(key=lambda x: x[0])
            
            # Separate numbers and letters
            numbers = [x[1] for x in detected_items if x[1] in config.ARABIC_NUMBERS]
            letters = [x[1] for x in detected_items if x[1] not in config.ARABIC_NUMBERS]
            
            # Reverse for RTL
            letters.reverse()
            numbers.reverse()
            
            final_text = " ".join(letters) + "   " + " ".join(numbers)
            return final_text
            
        except Exception as e:
            logger.error(f"Character reading error: {e}")
            return ""

# --- MAIN SYSTEM CLASS ---
class SmartParkingSystem:
    """Main system controller."""
    
    def __init__(self):
        """Initialize parking system."""
        logger.info("⏳ System Starting...")
        print("⏳ System Starting (Smart Parking v4.0)...")
        
        # Initialize components
        self.hardware = HardwareManager()
        self.camera = CameraManager(config.CAMERA_ID)
        self.recognizer = PlateRecognizer()
        self.parking_agent = td3_parking.get_agent()
        
        # Initialize database and blockchain
        database.initialize_db()
        
        # State variables
        self.last_logged_plate = ""
        self.last_logged_time = 0
        self.entry_open = False
        self.exit_open = False
        self.entry_timer = 0
        self.exit_timer = 0
        self.current_building = None
        self.target_spot = -1
        
        logger.info("✅ System initialized successfully")
        print("✅ READY. Press 'A' or 'B' to request parking, 'Q' to quit.")
    
    def handle_exit_gate(self, ir_state: int) -> None:
        """Handle exit gate logic."""
        # IR triggered (vehicle at exit)
        if ir_state == 0 and not self.exit_open:
            logger.info("🚗 IR Triggered: Opening EXIT Gate")
            print("🚗 IR Triggered: Opening EXIT Gate")
            
            self.hardware.move_servo(
                self.hardware.exit_pwm,
                config.EXIT_SERVO_PIN,
                config.SERVO_OPEN_ANGLE
            )
            self.exit_open = True
            self.exit_timer = time.time()
            
            # Log exit to database (last plate logged)
            if self.last_logged_plate:
                database.save_exit(self.last_logged_plate)
            
            database.log_system_event("EXIT_GATE_OPEN", self.last_logged_plate)
        
        # Auto-close logic
        now = time.time()
        if self.exit_open and (now - self.exit_timer > config.GATE_AUTO_CLOSE_TIME):
            logger.info("⬇️ Closing EXIT Gate")
            print("⬇️ Closing EXIT Gate")
            self.hardware.move_servo(
                self.hardware.exit_pwm,
                config.EXIT_SERVO_PIN,
                config.SERVO_CLOSED_ANGLE
            )
            self.exit_open = False
    
    def handle_entry_gate(self) -> None:
        """Handle entry gate auto-close."""
        now = time.time()
        if self.entry_open and (now - self.entry_timer > config.GATE_AUTO_CLOSE_TIME):
            logger.info("⬇️ Closing ENTRY Gate")
            print("⬇️ Closing ENTRY Gate")
            self.hardware.move_servo(
                self.hardware.entry_pwm,
                config.ENTRY_SERVO_PIN,
                config.SERVO_CLOSED_ANGLE
            )
            self.entry_open = False
    
    def handle_user_input(self, key: int, sensor_data: List[int]) -> None:
        """Handle keyboard input for parking request."""
        if key == ord('a') or key == ord('b'):
            self.current_building = "A" if key == ord('a') else "B"
            logger.info(f"👤 User Requested: Building {self.current_building}")
            print(f"\n👤 User Requested: Building {self.current_building}")
            
            # Get AI allocation
            result = self.parking_agent.select_spot(sensor_data, self.current_building)
            self.target_spot = result['spot_id']
            
            if self.target_spot == -1:
                logger.warning("Parking lot full")
                print("❌ LOT FULL! Please wait.")
            else:
                logger.info(f"AI allocated spot {self.target_spot} using {result['method']}")
                print(f"🎯 AI Allocation: Go to Parking Spot {self.target_spot} ({result['method']})")
    
    def process_plate_detection(self, img: np.ndarray) -> np.ndarray:
        """Process plate detection and logging."""
        plates = self.recognizer.detect_and_read_plate(img)
        
        for plate in plates:
            text = plate['text']
            x1, y1, x2, y2 = plate['bbox']
            crop = plate['crop']
            
            # Check if duplicate
            is_duplicate = (
                text == self.last_logged_plate and
                (time.time() - self.last_logged_time) < config.DUPLICATE_COOLDOWN
            )
            
            # Check if already parked
            already_parked = database.is_vehicle_currently_parked(text)
            
            box_color = config.COLOR_DETECTED  # Yellow
            
            if not is_duplicate and not already_parked:
                # Save plate image
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"plate_{timestamp}.jpg"
                save_path = os.path.join(config.OUTPUT_FOLDER, filename)
                cv2.imwrite(save_path, crop)
                
                # Log to database
                entry_id = database.save_entry(
                    text,
                    save_path,
                    allocated_spot=self.target_spot,
                    destination=self.current_building
                )
                
                # Log to blockchain
                if entry_id:
                    block_data = {
                        "plate": text,
                        "image": filename,
                        "access": "GRANTED",
                        "spot_allocated": self.target_spot,
                        "destination": self.current_building,
                        "entry_id": entry_id
                    }
                    new_block = simple_blockchain.ledger.add_block(block_data)
                    if new_block:
                        logger.info(f"🔗 Blockchain: {new_block.hash[:12]}...")
                        print(f"🔗 Secured on Blockchain: Hash {new_block.hash[:8]}...")
                
                # Open entry gate
                if not self.entry_open:
                    logger.info(f"✅ Access Granted: {text}")
                    print(f"✅ Access Granted: {text}")
                    self.hardware.move_servo(
                        self.hardware.entry_pwm,
                        config.ENTRY_SERVO_PIN,
                        config.SERVO_OPEN_ANGLE
                    )
                    self.entry_open = True
                    self.entry_timer = time.time()
                
                self.last_logged_plate = text
                self.last_logged_time = time.time()
                box_color = config.COLOR_LOGGED  # Green
            
            # Draw on image
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
            img = draw_arabic(img, text, (x1, y1-65), font_size=config.FONT_SIZE, color=box_color)
        
        return img
    
    def draw_ui(self, img: np.ndarray, sensor_data: List[int]) -> np.ndarray:
        """Draw UI elements on image."""
        # Show current destination and allocation
        if self.current_building:
            cv2.putText(
                img,
                f"Dest: {self.current_building}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                config.COLOR_TEXT,
                2
            )
            cv2.putText(
                img,
                f"Allocated: Spot {self.target_spot}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        # Show sensor status (circles)
        colors = [config.COLOR_OCCUPIED if s == 0 else config.COLOR_FREE for s in sensor_data]
        cv2.circle(img, (600, 30), 10, colors[0], -1)  # Spot 1
        cv2.circle(img, (600, 60), 10, colors[1], -1)  # Spot 2
        cv2.circle(img, (600, 90), 10, colors[2], -1)  # Spot 3
        
        # Labels
        cv2.putText(img, "Spot 1", (620, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "Spot 2", (620, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "Spot 3", (620, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def run(self) -> None:
        """Main system loop."""
        try:
            while True:
                # Read sensors
                sensor_data = self.hardware.read_sensors()
                ir_state = self.hardware.read_ir_sensor()
                
                # Handle exit gate
                self.handle_exit_gate(ir_state)
                
                # Handle entry gate auto-close
                self.handle_entry_gate()
                
                # Read camera frame
                ret, img = self.camera.read_frame()
                if not ret:
                    # Camera error - show blank frame and continue
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        img,
                        "Camera Error - Reconnecting...",
                        (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow("Smart Parking System", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested shutdown")
                    break
                self.handle_user_input(key, sensor_data)
                
                # Process plate detection
                img = self.process_plate_detection(img)
                
                # Draw UI
                img = self.draw_ui(img, sensor_data)
                
                # Display
                cv2.imshow("Smart Parking System", img)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            print("\n⏸️ Stopping...")
        except Exception as e:
            logger.error(f"System error: {e}", exc_info=True)
            print(f"❌ System error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleanup system resources."""
        logger.info("Starting system cleanup")
        print("🧹 Cleaning up...")
        
        try:
            self.camera.release()
            self.hardware.cleanup()
            cv2.destroyAllWindows()
            database.close_all_connections()
            logger.info("✅ Cleanup completed")
            print("✅ Shutdown complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# --- MAIN ENTRY POINT ---
def main():
    """Main entry point."""
    try:
        system = SmartParkingSystem()
        system.run()
    except Exception as e:
        logger.critical(f"Failed to start system: {e}", exc_info=True)
        print(f"❌ Failed to start: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())