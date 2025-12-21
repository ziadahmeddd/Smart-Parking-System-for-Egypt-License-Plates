# Training Your Own Models

**⚠️ Important**: This repository **does NOT include pre-trained models**. You must train them yourself using the EALPR dataset.

---

## 🎯 **Quick Start: Train All Models**

### 1. Download EALPR Dataset
Visit [EALPR Repository](https://github.com/ahmedramadan96/EALPR) and download all three dataset folders.

### 2. Extract to `dataset/` Folder
```bash
SmartParkingSystem/
├── dataset/
│   ├── EALPR Vechicles dataset/
│   ├── EALPR- Plates dataset/
│   └── EALPR- LP characters dataset/
```

### 3. Prepare Datasets for YOLO Format
```bash
# Prepare license plate dataset
python prepare_dataset.py

# Prepare character recognition dataset
python prepare_character_data.py
```

### 4. Train the Models
```bash
# Train plate detector (~1-2 hours on GPU)
python train_plates.py

# Train character detector (~2-3 hours on GPU)
python train_characters.py

# Train TD3 agent (~1 minute on CPU)
python train_simulation.py
```

### 5. Test Your Models
```bash
# Test on sample images
python smart_logger.py test_images
```

---

## 📦 **What Gets Created**

After training, you'll have:

- ✅ **`plate_detector.pt`** - License plate detection model (~6MB)
- ✅ **`character_detector.pt`** - Character recognition model (~6MB)
- ✅ **`td3_actor.pth`** - Parking allocation agent (~50KB)

These files are needed to run the system but are **NOT included in Git**.

---

## 📖 **Detailed Training Guide**

For complete step-by-step instructions, troubleshooting, and tips, see:

👉 **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**

This guide covers:
- Downloading and preparing the EALPR dataset
- Converting to YOLO format
- Training parameters and optimization
- Expected results and performance metrics
- Troubleshooting common issues

---

## 🎓 **Dataset Attribution**

**IMPORTANT**: This project uses the [EALPR Dataset](https://github.com/ahmedramadan96/EALPR) for training.

**Citation**:
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

You **must** download this dataset separately - it is not included in this repository.

---

## ⚙️ **Using Custom Models**

If you have your own trained YOLOv11 models, update `config.py`:

```python
PLATE_MODEL_PATH = "your_plate_model.pt"
CHAR_MODEL_PATH = "your_char_model.pt"
TD3_MODEL_PATH = "your_td3_model.pth"
```

---

## 🆘 **Need Pre-trained Models?**

Contact: z.ahmed2003@gmail.com

However, **training your own models is recommended** for:
- Better performance on your specific use case
- Understanding the training process
- Customization and fine-tuning

---

## 📁 Project Structure After Setup

After training all models, your directory should look like:

```
SmartParkingSystem/
├── Core Code (from GitHub)
│   ├── config.py
│   ├── database.py
│   ├── td3_parking.py
│   ├── smart_logger.py
│   └── ...
│
├── dataset/
│   └── ealpr-master/              ← Extract EALPR here
│       ├── ealpr vechicles dataset/
│       ├── ealpr- plates dataset/
│       └── ealpr- lp characters dataset/
│
├── Trained Models (after training)
│   ├── plate_detector.pt          ← From train_plates.py
│   ├── character_detector.pt      ← From train_characters.py
│   └── td3_actor.pth             ← From train_simulation.py
│
└── NotoSansArabic-Regular.ttf     ← Download from Google Fonts
```

---

## ⚠️ Important Notes

1. **Never commit large model files to Git** - They bloat the repository
2. **TD3 model can be retrained** - Run `train_simulation.py` (fast!)
3. **YOLO models are project-specific** - You may need to train on your own dataset
4. **Font file is freely available** - Download from Google Fonts

---

## 🚀 Quick Start

After cloning the repository:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add model files (manually)
# - Copy YOLO models to project root
# - Download Arabic font

# 3. Train TD3 model
python train_simulation.py

# 4. Test the system
python smart_logger.py test_images
```

---

For more information, see the main [README.md](README.md) file.
