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

### 2. Initialize System (1 min)
```bash
# Create database
python -c "import database; database.initialize_db()"

# Verify installation
python -c "import config; import database; import simple_blockchain; print('✅ All modules loaded')"
```

### 3. (Optional) Train AI (1 min)
```bash
# Train TD3 agent - creates td3_actor.pth
python train_simulation.py
```

### 4. Run System (1 min)
```bash
# Start main system
python maquette_main.py
```

**Controls:**
- Press `A` - Request parking near Building A
- Press `B` - Request parking near Building B  
- Press `Q` - Quit

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

Make sure these files exist in your project folder:

✅ **Python Files** (provided):
- `config.py` - Settings
- `database.py` - Database
- `simple_blockchain.py` - Blockchain
- `td3_parking.py` - AI agent
- `train_simulation.py` - Training
- `maquette_main.py` - Main system
- `logger.py` - Logging
- `smart_logger.py` - Batch processor

⚠️ **Model Files** (you must provide):
- `plate_detector.pt` - YOLO plate detection model
- `character_detector.pt` - YOLO character recognition model
- `NotoSansArabic-Regular.ttf` - Arabic font file

💡 **Optional**:
- `td3_actor.pth` - Trained TD3 weights (created by training)

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
ls *.pt
# If missing, download or train your own YOLO models
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
