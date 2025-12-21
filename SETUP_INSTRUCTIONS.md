# 🚀 Complete Setup Instructions

Step-by-step guide to set up the Smart Parking System from scratch.

---

## 📋 **Overview**

This repository provides the **code and training pipeline** only. You must:
1. Download the EALPR dataset
2. Prepare it for YOLOv11 training
3. Train three models
4. Test the system

**Time Required**: ~4-5 hours (mostly training time)

---

## 🎯 **Step-by-Step Setup**

### Step 1: Clone Repository (1 min)
```bash
git clone https://github.com/YOUR_USERNAME/SmartParkingSystem.git
cd SmartParkingSystem
```

---

### Step 2: Setup Python Environment (5 min)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Verify installation**:
```bash
python -c "import torch; import cv2; from ultralytics import YOLO; print('✅ All packages installed')"
```

---

### Step 3: Download EALPR Dataset (10 min)

#### 3.1 Visit EALPR Repository
Go to: **https://github.com/ahmedramadan96/EALPR**

#### 3.2 Download All Three Folders
- `EALPR Vechicles dataset` - Full vehicle images with plates
- `EALPR- Plates dataset` - Cropped license plate images
- `EALPR- LP characters dataset` - Individual character images

#### 3.3 Extract to Project
Create `dataset/` folder and extract the EALPR repository:

```
SmartParkingSystem/
└── dataset/
    └── ealpr-master/
        ├── ealpr vechicles dataset/
        │   ├── vehicles/           # Images
        │   └── vehicles labeling/  # Annotations
        ├── ealpr- plates dataset/
        │   ├── plates images/
        │   └── plates labeling/
        └── ealpr- lp characters dataset/
            ├── train/
            └── val/
```

**Verify**:
```bash
ls dataset/ealpr-master/
# Should show: ealpr vechicles dataset, ealpr- plates dataset, ealpr- lp characters dataset
```

---

### Step 4: Prepare Datasets (5 min)

#### 4.1 Prepare License Plate Detection Dataset
```bash
python prepare_dataset.py
```

**What happens:**
- Reads EALPR vehicle images and annotations
- Converts to YOLO format (class x_center y_center width height)
- Splits into train/val sets (80/20)
- Creates `data.yaml` config file

**Output**: `dataset/egyptian_plates_detection/`

#### 4.2 Prepare Character Recognition Dataset
```bash
python prepare_character_data.py
```

**What happens:**
- Reads EALPR character images and annotations
- Converts Arabic character labels to YOLO format
- Creates class mapping for 36 characters (letters + numbers)
- Creates `data.yaml` config file

**Output**: `dataset/egyptian_chars_detection/`

**Verify both**:
```bash
ls dataset/egyptian_plates_detection/
ls dataset/egyptian_chars_detection/
# Each should have: train/, val/, data.yaml
```

---

### Step 5: Train Models (2-4 hours)

#### 5.1 Train License Plate Detector
```bash
python train_plates.py
```

**Configuration:**
- Model: YOLOv11n (nano - fast inference)
- Epochs: 50
- Image Size: 640x640
- Batch Size: 16

**Training Time**: 1-2 hours on modern GPU, 8-12 hours on CPU

**Output**: `plate_detector.pt` in project root

**Monitor progress**:
```bash
# View training logs
cat runs/detect/train/results.csv

# View training plots
# Check: runs/detect/train/results.png
```

#### 5.2 Train Character Recognition Model
```bash
python train_characters.py
```

**Configuration:**
- Model: YOLOv11n
- Epochs: 100 (characters need more training)
- Image Size: 416x416
- Batch Size: 32

**Training Time**: 2-3 hours on GPU

**Output**: `character_detector.pt` in project root

#### 5.3 Train TD3 Parking Allocation Agent
```bash
python train_simulation.py
```

**Configuration:**
- Algorithm: TD3 (reinforcement learning)
- Episodes: 10,000
- Replay Buffer: 100,000 transitions
- Networks: Actor + Twin Critics

**Training Time**: ~1 minute on CPU (very fast!)

**Output**: `td3_actor.pth` in project root

**Expected Performance**:
```
Episode  1000 → Avg Reward: -9.33
Episode  5000 → Avg Reward:  5.96
Episode 10000 → Avg Reward:  6.04  ✅ Good!
```

---

### Step 6: Download Arabic Font (1 min)
```bash
# Download NotoSansArabic-Regular.ttf from Google Fonts
# Place in project root
```

**Download link**: https://fonts.google.com/noto/specimen/Noto+Sans+Arabic

---

### Step 7: Initialize System (1 min)
```bash
# Create database
python -c "import database; database.initialize_db()"

# Verify models exist
ls *.pt *.pth
# Should show: plate_detector.pt, character_detector.pt, td3_actor.pth
```

---

### Step 8: Test Everything (2 min)
```bash
# Put some test images in test_images/ folder
mkdir test_images
# Copy a few vehicle images with Arabic plates

# Run batch processor
python smart_logger.py test_images
```

**Expected output:**
```
📂 Found X images in 'test_images'
Processing 1/X: image1.jpg
✅ Saved: س ن و   ٦ ٥ ٧ ٦
🔗 Block #1 [Hash: 8e6c88de...]
...
🎉 Processing Complete!
```

If this works, **your system is fully operational!** ✅

---

## 📊 **What You Should Have**

After completing all steps:

### ✅ In Project Root:
```
SmartParkingSystem/
├── plate_detector.pt          ✅ Trained
├── character_detector.pt      ✅ Trained
├── td3_actor.pth             ✅ Trained
├── NotoSansArabic-Regular.ttf ✅ Downloaded
├── parking_system.db         ✅ Created
└── secure_ledger.json        ✅ Created
```

### ✅ In Dataset Folder (ignored by git):
```
dataset/
├── ealpr-master/                      # Downloaded from GitHub
│   ├── ealpr vechicles dataset/
│   ├── ealpr- plates dataset/
│   └── ealpr- lp characters dataset/
├── egyptian_plates_detection/         # YOLO format (prepared)
└── egyptian_chars_detection/          # YOLO format (prepared)
```

### ✅ Training Outputs (ignored by git):
```
runs/
└── detect/
    ├── train/    # Plate training results
    └── train2/   # Character training results
```

---

## 🎯 **Verification Checklist**

Run these tests to ensure everything works:

```bash
# ✅ Test 1: Config loads
python -c "import config; print('✅ Config OK')"

# ✅ Test 2: Database works
python -c "import database; database.initialize_db(); print('✅ Database OK')"

# ✅ Test 3: Blockchain works
python -c "from simple_blockchain import ledger; print(f'✅ Blockchain OK: {len(ledger.chain)} blocks')"

# ✅ Test 4: TD3 agent loads
python -c "from td3_parking import get_agent; agent = get_agent(); print(f'✅ TD3 OK: Trained={agent.is_trained}')"

# ✅ Test 5: Plate detector loads
python -c "from ultralytics import YOLO; m = YOLO('plate_detector.pt'); print('✅ Plate detector OK')"

# ✅ Test 6: Character detector loads
python -c "from ultralytics import YOLO; m = YOLO('character_detector.pt'); print('✅ Character detector OK')"

# ✅ Test 7: Full pipeline
python smart_logger.py test_images
```

If all tests pass: **🎉 System Ready!**

---

## ⏱️ **Time Breakdown**

| Step | Task | Time |
|------|------|------|
| 1-2 | Clone & Setup Python | 5 min |
| 3 | Download EALPR Dataset | 10 min |
| 4 | Prepare Datasets | 5 min |
| 5.1 | Train Plate Detector | 1-2 hours |
| 5.2 | Train Character Detector | 2-3 hours |
| 5.3 | Train TD3 Agent | 1 min |
| 6-8 | Font, Database, Testing | 5 min |
| **Total** | **~4-5 hours** | (Mostly GPU training) |

---

## 🚀 **Next Steps**

After setup:
- 📖 Read [README.md](README.md) for full documentation
- 🎮 Try deploying to Raspberry Pi
- 🔧 Customize `config.py` for your setup
- 📊 Monitor `parking_system.log` for debugging

---

## 🆘 **Stuck? Common Issues**

### "Dataset not found"
- Verify dataset is in `dataset/` folder
- Check folder names match exactly
- Re-run preparation scripts

### "Out of memory during training"
- Reduce batch size in training scripts
- Use CPU training (slower but works)
- Train on Google Colab (free GPU)

### "Models don't load"
- Check file sizes: `ls -lh *.pt *.pth`
- Re-train if corrupted
- Verify training completed successfully

### "Can't find EALPR dataset"
- Visit: https://github.com/ahmedramadan96/EALPR
- Download all three folders
- Extract to `dataset/` directory

---

## 📞 **Get Help**

- 📧 Email: z.ahmed2003@gmail.com
- 🐛 GitHub Issues: [Open an issue](https://github.com/YOUR_USERNAME/SmartParkingSystem/issues)
- 📖 Documentation: See all `.md` files in project root

---

**You've got this! Follow the steps and you'll have a working system.** 💪
