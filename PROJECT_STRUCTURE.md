# 📁 Project Structure Guide

Complete overview of the Smart Parking System file organization.

---

## 🗂️ **Files Included in GitHub Repository**

### Core System Modules
```
├── config.py                  # Centralized configuration
├── logger.py                  # Professional logging framework
├── database.py                # Database with connection pooling
├── simple_blockchain.py       # Blockchain with validation
├── td3_parking.py            # TD3 parking allocation agent
├── smart_logger.py           # Batch image processing
└── verify_blockchain.py      # Blockchain integrity checker
```

### Training Pipeline Scripts
```
├── prepare_dataset.py         # Convert EALPR plates → YOLO format
├── prepare_character_data.py  # Convert EALPR characters → YOLO format
├── train_plates.py            # Train YOLOv11 plate detector
├── train_characters.py        # Train YOLOv11 character recognizer
└── train_simulation.py        # Train TD3 reinforcement learning agent
```

### Documentation Files
```
├── README.md                  # Main project documentation
├── TRAINING_GUIDE.md         # Complete training instructions
├── CITATIONS.md              # Dataset and research citations
├── QUICKSTART.md             # Fast setup guide
├── MIGRATION_GUIDE.md        # v3 → v4.0 upgrade guide
├── FIXES_SUMMARY.md          # All improvements in v4.0
├── MODELS_README.md          # Model training overview
├── PROJECT_STRUCTURE.md      # This file
└── requirements.txt          # Python dependencies
```

### Git Configuration
```
├── .gitignore                # Excludes models, data, logs
└── .gitattributes            # Git file handling rules
```

---

## 📦 **Files NOT Included (User Must Provide)**

### 1. EALPR Dataset (Download Separately)
**Download from**: [EALPR GitHub Repository](https://github.com/ahmedramadan96/EALPR)

**Extract to**:
```
dataset/
└── ealpr-master/
    ├── ealpr vechicles dataset/
    │   ├── vehicles/
    │   └── vehicles labeling/
    ├── ealpr- plates dataset/
    │   ├── plates images/
    │   └── plates labeling/
    └── ealpr- lp characters dataset/
        ├── train/
        └── val/
```

**Size**: ~2-3 GB total  
**Why not included**: Too large for Git repository  
**Citation required**: See [CITATIONS.md](CITATIONS.md)

---

### 2. Trained Model Files (Generated After Training)
**Created by training scripts:**

```
├── plate_detector.pt          # From train_plates.py (~6MB)
├── character_detector.pt      # From train_characters.py (~6MB)
└── td3_actor.pth             # From train_simulation.py (~50KB)
```

**Why not included**: Users should train their own models for optimal performance

**Alternative**: Contact z.ahmed2003@gmail.com for pre-trained models

---

### 3. Additional Resources (Optional)
```
├── NotoSansArabic-Regular.ttf  # Arabic font (download from Google Fonts)
└── test_images/                # Your own test images
```

---

## 🚫 **Files Excluded from Git**

The `.gitignore` prevents these from being committed:

### Generated During Training
```
SmartParking_Project/          # Training output folders
runs/                          # YOLOv11 training runs
weights/                       # Intermediate model weights
*.pt, *.pth                    # All model files
```

### Runtime Data
```
parking_system.db              # SQLite database
parking_system.log*            # Log files
secure_ledger.json            # Blockchain data
stored_plates/                # Detected plate images
```

### Development Files
```
venv/                         # Virtual environment
__pycache__/                  # Python cache
*.pyc                         # Compiled Python files
.DS_Store, Thumbs.db          # OS files
```

---

## 🔄 **Workflow: From Clone to Running System**

### Stage 1: Setup (10 minutes)
```bash
1. git clone <repo-url>
2. cd SmartParkingSystem
3. python -m venv venv
4. source venv/bin/activate  # or venv\Scripts\activate on Windows
5. pip install -r requirements.txt
```

**Result**: Python environment ready ✅

---

### Stage 2: Get Dataset (15 minutes)
```bash
1. Download EALPR dataset from GitHub
2. Extract three folders to dataset/
3. Verify structure matches TRAINING_GUIDE.md
```

**Result**: Dataset ready for training ✅

---

### Stage 3: Prepare Data (5 minutes)
```bash
1. python prepare_dataset.py
2. python prepare_character_data.py
```

**Result**: Data converted to YOLO format ✅

---

### Stage 4: Train Models (2-4 hours on GPU)
```bash
1. python train_plates.py        # 1-2 hours
2. python train_characters.py    # 2-3 hours
3. python train_simulation.py    # 1 minute
```

**Result**: Three trained models ready ✅

---

### Stage 5: Test & Deploy (5 minutes)
```bash
1. python -c "import database; database.initialize_db()"
2. python smart_logger.py test_images
3. (Deploy to Raspberry Pi - see README.md)
```

**Result**: Working Smart Parking System ✅

---

## 📊 **Disk Space Requirements**

| Component | Size | Location |
|-----------|------|----------|
| Git Repository (code only) | ~500 KB | Cloned repo |
| EALPR Dataset | ~2-3 GB | `dataset/` |
| Prepared YOLO Data | ~1-2 GB | `dataset/*_yolo/` |
| Training Outputs | ~500 MB | `runs/`, `SmartParking_Project/` |
| Trained Models | ~12 MB | `*.pt`, `*.pth` |
| Virtual Environment | ~1 GB | `venv/` |
| **Total** | **~5-8 GB** | Full setup |

---

## 🎯 **What You Get**

### From This Repository:
✅ Complete source code  
✅ Training scripts  
✅ Database system  
✅ Blockchain implementation  
✅ TD3 reinforcement learning  
✅ Comprehensive documentation  

### What You Must Provide:
⚠️ EALPR dataset (download separately)  
⚠️ Train models using provided scripts  
⚠️ Hardware (for Raspberry Pi deployment)  

---

## 🚀 **Repository Purpose**

This repository provides:
1. **Training Pipeline**: Complete scripts to train your own models
2. **System Code**: Production-ready parking management system
3. **Documentation**: Step-by-step guides and references
4. **Best Practices**: Type hints, logging, error handling

**Philosophy**: Train your own models for best results on your specific use case.

---

## 📞 **Questions?**

- **Training Issues**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Setup Problems**: See [QUICKSTART.md](QUICKSTART.md)
- **Dataset Questions**: Check [EALPR Repository](https://github.com/ahmedramadan96/EALPR)
- **Contact**: z.ahmed2003@gmail.com

---

**Clear structure, easy to follow! 📂**
