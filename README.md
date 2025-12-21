# 🚗 Smart Parking System v4.0

[![Dataset](https://img.shields.io/badge/Dataset-EALPR-blue)](https://github.com/ahmedramadan96/EALPR)
[![Paper](https://img.shields.io/badge/DOI-10.1109%2FACIRS55390.2022.9845514-orange)](https://doi.org/10.1109/ACIRS55390.2022.9845514)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An intelligent parking management system combining computer vision, reinforcement learning (TD3), and blockchain technology for secure and efficient parking allocation.

---

## ⚠️ **Important Notice**

This repository includes **source code and training scripts only**. You must:

1. 📥 **Download EALPR Dataset** from [here](https://github.com/ahmedramadan96/EALPR)
2. 📂 **Extract to `dataset/` folder** in project root
3. 🏋️ **Train your own models** using provided scripts
4. 🎯 See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for complete instructions

**Pre-trained models are NOT included** but can be requested via z.ahmed2003@gmail.com

## 🌟 Features

### Core Capabilities
- **🧠 AI-Powered Allocation**: TD3 (Twin Delayed DDPG) reinforcement learning for optimal spot selection
- **📸 License Plate Recognition**: YOLOv11-based detection with Arabic character OCR
- **🔗 Blockchain Security**: Immutable logging of all vehicle entries/exits
- **⚡ Real-time Monitoring**: Hardware sensor integration for spot occupancy
- **📊 Database Tracking**: Entry/exit times, parking duration, and statistics
- **🎮 Hardware Control**: Servo motors for entry/exit gates, IR sensors for vehicle detection

### Improvements in v4.0
- ✅ **Proper TD3 Implementation**: Full RL algorithm with replay buffer, target networks
- ✅ **Database Connection Pooling**: Thread-safe operations, exit tracking
- ✅ **Blockchain Validation**: Automatic integrity checking on startup
- ✅ **Centralized Configuration**: All settings in one place
- ✅ **Professional Logging**: File rotation, colored console output
- ✅ **Camera Error Recovery**: Auto-reconnection on failure
- ✅ **Type Hints**: Full type annotations for better code quality
- ✅ **Exception Handling**: Comprehensive error management

---

## 📋 Requirements

### Hardware Requirements
- **Raspberry Pi** (3B+ or higher) or **PC/Laptop** for development
- **USB Camera** (or Pi Camera Module)
- **Servo Motors** (x2) for entry/exit gates
- **IR Sensor** (x1) for exit detection
- **Metal Sensors** (x3) for parking spot detection
- **Power Supply** (5V for Pi, appropriate for servos)

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- Ultralytics YOLOv11
- See `requirements.txt` for complete list

---

## 🚀 Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd SmartParkingSystem
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download EALPR Dataset and Train Models

**⚠️ Important**: This repository does NOT include pre-trained models. You must train them yourself.

```bash
# Download EALPR dataset from: https://github.com/ahmedramadan96/EALPR
# Extract all three folders to dataset/ directory

# Prepare datasets for YOLO format
python prepare_dataset.py
python prepare_character_data.py

# Train models (see TRAINING_GUIDE.md for details)
python train_plates.py          # ~1-2 hours on GPU
python train_characters.py      # ~2-3 hours on GPU
python train_simulation.py      # ~1 minute on CPU
```

**Alternative**: Download `NotoSansArabic-Regular.ttf` font file and contact z.ahmed2003@gmail.com for pre-trained models.

For detailed training instructions, see **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**.

### 5. Initialize Database
```bash
python database.py
```

### 6. Verify Models Work
```bash
# Test the full pipeline
python smart_logger.py test_images
```

If everything works, you're ready to deploy!

---

## ⚙️ Configuration

All configuration is in `config.py`. Key settings:

### Hardware Pins
```python
ENTRY_SERVO_PIN = 13    # GPIO 13
EXIT_SERVO_PIN = 12     # GPIO 12
IR_SENSOR_PIN = 23      # GPIO 23
SPOT1_PIN = 24          # GPIO 24
SPOT2_PIN = 27          # GPIO 27
SPOT3_PIN = 22          # GPIO 22
```

### AI Models
```python
PLATE_MODEL_PATH = "plate_detector.pt"
CHAR_MODEL_PATH = "character_detector.pt"
TD3_MODEL_PATH = "td3_actor.pth"
```

### Parking Layout
```python
PARKING_SPOTS = {
    0: {"id": 1, "location": 1.0},  # Near Building A
    1: {"id": 2, "location": 2.0},  # Middle
    2: {"id": 3, "location": 3.0}   # Near Building B
}

BUILDINGS = {
    "A": 0.0,  # Building A location
    "B": 4.0   # Building B location
}
```

---

## 📖 Usage

### Training Models (Required First Step)
See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for complete training instructions.

```bash
# Prepare datasets
python prepare_dataset.py
python prepare_character_data.py

# Train models
python train_plates.py
python train_characters.py
python train_simulation.py
```

### Testing Your Trained Models
Process images to test detection and recognition:
```bash
python smart_logger.py test_images
```

### View Model Accuracy Metrics
After training, view detailed accuracy percentages:
```bash
python view_metrics.py
```

This displays:
- **Plate Detector**: mAP@0.5, precision, recall, F1 score
- **Character Detector**: mAP@0.5, precision, recall, per-class accuracy
- **TD3 Agent**: Correctness rate, optimality rate, average reward
- **Overall System**: Combined readiness score

### Database Management
```bash
# View all entries
python -c "import database; print(database.fetch_all_logs())"

# Get statistics
python -c "import database; print(database.get_parking_statistics())"

# Reset database
python reset_database.py
```

### Blockchain Verification
```bash
python verify_blockchain.py
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│                  MAIN CONTROLLER                     │
│              (maquette_main.py)                      │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Hardware │  │  Camera  │  │   Plate  │          │
│  │ Manager  │  │ Manager  │  │Recognizer│          │
│  └──────────┘  └──────────┘  └──────────┘          │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │   TD3    │  │ Database │  │Blockchain│          │
│  │  Agent   │  │  Module  │  │  Module  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
```

### Repository Contents

#### ✅ **Included in GitHub Repository:**
```
SmartParkingSystem/
│
├── Core System Files
│   ├── config.py                  # Centralized configuration
│   ├── logger.py                  # Logging framework
│   ├── database.py                # Database operations
│   ├── simple_blockchain.py       # Blockchain implementation
│   ├── td3_parking.py            # TD3 inference agent
│   ├── smart_logger.py           # Batch image processor
│   └── verify_blockchain.py      # Blockchain validator
│
├── Training Pipeline
│   ├── prepare_dataset.py         # Prepare plate dataset for YOLO
│   ├── prepare_character_data.py  # Prepare character dataset for YOLO
│   ├── train_plates.py           # Train plate detection model
│   ├── train_characters.py       # Train character recognition model
│   └── train_simulation.py       # Train TD3 parking agent
│
└── Documentation
    ├── README.md                 # Main documentation
    ├── SETUP_INSTRUCTIONS.md     # Complete setup guide
    ├── TRAINING_GUIDE.md         # Detailed training instructions
    ├── CITATIONS.md              # Dataset and paper citations
    ├── PROJECT_STRUCTURE.md      # File organization guide
    ├── QUICKSTART.md             # Quick reference
    ├── MIGRATION_GUIDE.md        # v3 → v4.0 upgrade
    ├── FIXES_SUMMARY.md          # All improvements
    └── requirements.txt          # Python dependencies
```

#### ⚠️ **NOT Included (You Must Provide):**
```
├── dataset/                      # Download from EALPR repository
│   └── ealpr-master/
│       ├── ealpr vechicles dataset/
│       ├── ealpr- plates dataset/
│       └── ealpr- lp characters dataset/
│
├── plate_detector.pt             # Train with train_plates.py
├── character_detector.pt         # Train with train_characters.py
├── td3_actor.pth                # Train with train_simulation.py
└── NotoSansArabic-Regular.ttf   # Download from Google Fonts
```

**📖 See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for complete step-by-step guide.**

---

## 🧠 How It Works

### 1. Vehicle Entry Flow
1. Driver presses **A** or **B** to indicate destination
2. **TD3 Agent** analyzes sensor data and selects optimal spot
3. Camera detects license plate
4. **YOLOv11 models** perform plate detection and OCR
5. Entry logged to **database** and **blockchain**
6. **Entry gate opens** (servo motor activated)
7. Gate auto-closes after 5 seconds

### 2. TD3 Decision Making
The TD3 agent learns to optimize:
- **Primary Goal**: Select free parking spots (avoid occupied)
- **Secondary Goal**: Minimize walking distance to destination
- **State**: [Spot1_Free, Spot2_Free, Spot3_Free, Destination_Location]
- **Action**: Continuous value [-1, 1] decoded to spot selection

### 3. Vehicle Exit Flow
1. IR sensor detects vehicle at exit
2. **Exit gate opens**
3. Exit time logged to database (calculates parking duration)
4. Gate auto-closes after 5 seconds

### 4. Data Integrity
- All entries secured on **blockchain** (tamper-evident)
- Database validates on startup
- Blockchain integrity checked automatically

---

## 📊 Database Schema

### vehicle_logs Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| plate_number | TEXT | License plate text |
| entry_time | TIMESTAMP | Entry timestamp |
| exit_time | TIMESTAMP | Exit timestamp (nullable) |
| image_path | TEXT | Path to plate image |
| allocated_spot | INTEGER | Spot allocated by AI |
| destination_building | TEXT | User's destination (A/B) |
| parking_duration_seconds | INTEGER | Total parking time |
| is_active | BOOLEAN | Currently parked flag |

### system_events Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| event_type | TEXT | Event category |
| event_data | TEXT | Event details |
| timestamp | TIMESTAMP | Event time |

---

## 🔐 Security Features

1. **Blockchain Integrity**: SHA-256 hashing prevents data tampering
2. **Validation on Startup**: Automatic chain verification
3. **Atomic Operations**: Database transactions prevent corruption
4. **Access Logging**: All entries recorded with timestamps
5. **Image Evidence**: Plate photos stored for each transaction

---

## 🐛 Troubleshooting

### Camera Not Detected
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Update camera ID in config.py
CAMERA_ID = 0  # Change to your camera index
```

### GPIO Errors (Raspberry Pi)
```bash
# Ensure user is in gpio group
sudo usermod -a -G gpio $USER

# Reboot
sudo reboot
```

### Model Loading Errors
```bash
# Verify model files exist
ls -lh *.pt *.pth

# Re-download if corrupted
# (provide download instructions for your specific models)
```

### Database Locked
```bash
# Close all connections and restart
python -c "import database; database.close_all_connections()"
```

---

## 📈 Performance Metrics

After training your models, view detailed accuracy percentages:
```bash
python view_metrics.py
```

### Expected Performance Targets

#### Plate Detector (YOLOv11)
- **mAP@0.5**: >90% (detection accuracy)
- **Precision**: >85% (correct detections)
- **Recall**: >85% (plates found)
- **Inference Speed**: 30+ FPS on Raspberry Pi 4

#### Character Detector (YOLOv11)
- **mAP@0.5**: >85% (character recognition)
- **Precision**: >80% (correct characters)
- **Recall**: >85% (characters found)
- **Per-Class Accuracy**: 80-95% per Arabic character

#### TD3 Agent
- **Correctness Rate**: >95% (avoids occupied spots)
- **Optimality Rate**: >75% (chooses best available spot)
- **Average Reward**: 5-7/10 (smart decisions)

### System Performance
- **Database Writes**: 100+ transactions/second
- **Blockchain Adds**: 50+ blocks/second
- **End-to-End Latency**: <500ms per vehicle

For detailed metrics explanation, see **[METRICS_GUIDE.md](METRICS_GUIDE.md)**.

---

## 🛠️ Development

### Running Tests
```bash
# Test database
python database.py

# Test blockchain
python simple_blockchain.py

# Test TD3 agent
python td3_parking.py

# Test logger
python logger.py
```

### Adding New Features
1. Update `config.py` with new settings
2. Use logging: `logger.info("message")`
3. Add type hints for all functions
4. Handle exceptions gracefully
5. Update this README

---

## 📝 API Reference

### TD3ParkingAgent
```python
from td3_parking import get_agent

agent = get_agent()
result = agent.select_spot(
    sensor_states=[1, 0, 1],  # 1=Free, 0=Occupied
    building_choice="A"        # "A" or "B"
)
# Returns: {"spot_id": 2, "method": "ai", "confidence": 0.8}
```

### Database Functions
```python
import database

# Save entry
entry_id = database.save_entry(
    plate_text="ABC 123",
    image_path="path/to/image.jpg",
    allocated_spot=2,
    destination="A"
)

# Save exit
success = database.save_exit("ABC 123")

# Get statistics
stats = database.get_parking_statistics()
```

### Blockchain Functions
```python
from simple_blockchain import ledger

# Add block
block = ledger.add_block({
    "plate": "ABC 123",
    "spot": 2
})

# Validate chain
is_valid = ledger.is_chain_valid()

# Get statistics
stats = ledger.get_statistics()
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes with type hints and logging
4. Test thoroughly
5. Commit: `git commit -m "Add feature"`
6. Push: `git push origin feature-name`
7. Open Pull Request

---

## 🙏 Acknowledgments

- YOLOv11 by Ultralytics
- TD3 algorithm by Fujimoto et al.
- OpenCV community
- PyTorch team
- **EALPR Dataset** by Ahmed Ramadan Youssef et al. - Egyptian License Plate Recognition dataset

---

## 📚 Citations

This project uses the **EALPR Dataset** for training license plate detection and character recognition models.

**Dataset Citation**:
```bibtex
@INPROCEEDINGS{9845514,
  author={Youssef, Ahmed Ramadan and Sayed, Fawzya Ramadan and Ali, Abdelmgeid Ameen},
  booktitle={2022 7th Asia-Pacific Conference on Intelligent Robot Systems (ACIRS)}, 
  title={A New Benchmark Dataset for Egyptian License Plate Detection and Recognition}, 
  year={2022},
  pages={106-111},
  doi={10.1109/ACIRS55390.2022.9845514}
}
```

**Dataset Repository**: [ahmedramadan96/EALPR](https://github.com/ahmedramadan96/EALPR)

For complete citations and acknowledgments, see [CITATIONS.md](CITATIONS.md).

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Contact: z.ahmed2003@gmail.com


**Made with ❤️ for smart cities**
