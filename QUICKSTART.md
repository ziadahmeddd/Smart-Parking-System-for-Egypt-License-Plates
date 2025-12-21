# 🚀 Quick Start Guide

Get your Smart Parking System running in 5 minutes!

---

## ⚡ Fast Setup

### 1. Install Dependencies (2 min)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install torch torchvision opencv-python ultralytics pillow python-bidi numpy matplotlib
```

### 2. Download and Prepare EALPR Dataset (5 min)
```bash
# Download dataset from: https://github.com/ahmedramadan96/EALPR
# Extract to: dataset/ealpr-master/

# Prepare datasets for YOLO format
python prepare_dataset.py
python prepare_character_data.py
```

### 3. Train Models (2-4 hours on GPU)
```bash
# Train license plate detector
python train_plates.py

# Train character recognizer
python train_characters.py

# Train TD3 agent (fast - 1 minute)
python train_simulation.py
```

### 4. Test System (1 min)
```bash
# Initialize database
python -c "import database; database.initialize_db()"

# Test on sample images
python smart_logger.py test_images
```

**Note**: For the full system with hardware (Raspberry Pi), see the main [README.md](README.md).

---

## 🎮 Test Without Hardware

If you don't have Raspberry Pi or hardware yet:

1. **Simulation Mode**: System automatically detects PC and uses GPIO mock
2. **Test Images**: Process images from folder instead of camera
   ```bash
   python smart_logger.py test_images
   ```

---

## 📁 Required Files

### ✅ **Python Files** (included in repo):
- `config.py` - Centralized settings
- `database.py` - Database operations
- `simple_blockchain.py` - Blockchain security
- `td3_parking.py` - AI parking agent
- `logger.py` - Logging system
- `smart_logger.py` - Batch image processor

### 📦 **Training Scripts** (included in repo):
- `prepare_dataset.py` - Prepare plate dataset
- `prepare_character_data.py` - Prepare character dataset
- `train_plates.py` - Train plate detector
- `train_characters.py` - Train character detector
- `train_simulation.py` - Train TD3 agent

### ⚠️ **Dataset** (you must download):
- **EALPR Dataset** from [here](https://github.com/ahmedramadan96/EALPR)
- Extract to `dataset/` folder
- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### 🎯 **Generated After Training**:
- `plate_detector.pt` - Your trained plate detector
- `character_detector.pt` - Your trained character detector
- `td3_actor.pth` - Your trained TD3 agent

---

## 🐛 Quick Troubleshooting

### Camera not found?
```python
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL')"
```

### Missing models?
```bash
# Check for model files
ls *.pt *.pth

# If missing, you need to train them!
# See TRAINING_GUIDE.md for complete instructions
# Or contact z.ahmed2003@gmail.com
```

### Database errors?
```bash
# Reset database (⚠️ deletes all data)
python reset_database.py
```

### Import errors?
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📊 Quick Test

After setup, test each component:

```bash
# Test 1: Configuration
python -c "import config; print(f'✅ Config: {len(config.PARKING_SPOTS)} parking spots')"

# Test 2: Database
python -c "import database; database.initialize_db(); print('✅ Database OK')"

# Test 3: Blockchain
python -c "from simple_blockchain import ledger; print(f'✅ Blockchain: {len(ledger.chain)} blocks')"

# Test 4: TD3 Agent
python -c "from td3_parking import get_agent; agent = get_agent(); print(f'✅ TD3: Trained={agent.is_trained}')"

# Test 5: Logger
python -c "import logger; logger.info('Test message'); print('✅ Logger OK')"
```

If all tests pass, you're ready to run the system!

---

## 📖 Next Steps

- **Read**: `README.md` for full documentation
- **Configure**: Edit `config.py` for your hardware
- **Train**: Run `train_simulation.py` for better AI
- **Monitor**: Check `parking_system.log` for events

---

## 🆘 Still Stuck?

1. Check log file: `parking_system.log`
2. Run individual component tests above
3. Read `MIGRATION_GUIDE.md` if upgrading
4. Review `FIXES_SUMMARY.md` for what's new
5. Open an issue on GitHub

---

**Happy Parking! 🚗✨**
