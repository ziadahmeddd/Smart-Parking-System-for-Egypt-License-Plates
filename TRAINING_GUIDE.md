# 🎓 Training Guide - Building Your Own Models

This guide walks you through training all models from scratch using the EALPR dataset.

---

## 📦 **Step 1: Download EALPR Dataset**

### Get the Dataset
1. Visit the [EALPR Dataset Repository](https://github.com/ahmedramadan96/EALPR)
2. Download all three dataset folders:
   - `EALPR Vechicles dataset` - Full vehicle images
   - `EALPR- Plates dataset` - Cropped license plates
   - `EALPR- LP characters dataset` - Individual characters

### Extract to Project
Create a `dataset/` folder in your project root and extract the EALPR repository:

```bash
SmartParkingSystem/
├── dataset/
│   └── ealpr-master/              # Extract GitHub repo here
│       ├── ealpr vechicles dataset/
│       │   ├── vehicles/
│       │   └── vehicles labeling/
│       ├── ealpr- plates dataset/
│       │   ├── plates images/
│       │   └── plates labeling/
│       └── ealpr- lp characters dataset/
│           ├── train/
│           └── val/
│
├── prepare_dataset.py
├── prepare_character_data.py
└── ...
```

**Important**: The `dataset/` folder is in `.gitignore` and will not be committed to Git.

---

## 🔧 **Step 2: Prepare Datasets for YOLO**

The EALPR dataset needs to be converted to YOLOv11 format.

### 2.1 Prepare License Plate Detection Dataset
```bash
python prepare_dataset.py
```

**What this does:**
- Converts EALPR plate annotations to YOLO format
- Organizes images into `train/` and `val/` folders
- Creates `data.yaml` configuration file
- Validates all annotations

**Output:** `dataset/plates_yolo/` ready for training

### 2.2 Prepare Character Recognition Dataset
```bash
python prepare_character_data.py
```

**What this does:**
- Converts EALPR character annotations to YOLO format
- Organizes Arabic character images
- Creates class mappings for Arabic letters and numbers
- Creates `data.yaml` configuration file

**Output:** `dataset/characters_yolo/` ready for training

---

## 🏋️ **Step 3: Train the Models**

### 3.1 Train License Plate Detector
```bash
python train_plates.py
```

**Training Parameters:**
- Model: YOLOv11n (nano - fast and efficient)
- Epochs: 50 (adjustable in script)
- Image Size: 640x640
- Batch Size: 16 (adjust based on GPU memory)

**Training Time:** ~1-2 hours on GPU, longer on CPU

**Output:**
- `plate_detector.pt` - Trained model
- Training metrics in `runs/detect/train/`
- Validation results and plots

### 3.2 Train Character Recognition Model
```bash
python train_characters.py
```

**Training Parameters:**
- Model: YOLOv11n (optimized for small objects)
- Epochs: 100 (characters need more training)
- Image Size: 416x416
- Batch Size: 32

**Training Time:** ~2-3 hours on GPU

**Output:**
- `character_detector.pt` - Trained model
- Training metrics in `runs/detect/train2/`
- Character recognition accuracy plots

### 3.3 Train TD3 Parking Allocation Agent
```bash
python train_simulation.py
```

**Training Parameters:**
- Episodes: 10,000
- Algorithm: TD3 (Twin Delayed DDPG)
- State Space: 4 dimensions (3 sensors + 1 destination)
- Action Space: 1 continuous value

**Training Time:** ~1 minute on CPU

**Output:**
- `td3_actor.pth` - Trained parking allocation agent
- Training progress and test results

---

## ✅ **Step 4: Verify Trained Models**

After training, verify your models work correctly:

### Test Plate Detection
```bash
python -c "from ultralytics import YOLO; model = YOLO('plate_detector.pt'); results = model.predict('test_images/sample.jpg'); print('✅ Plate detector works!')"
```

### Test Character Recognition
```bash
python -c "from ultralytics import YOLO; model = YOLO('character_detector.pt'); results = model.predict('test_images/sample.jpg'); print('✅ Character detector works!')"
```

### Test TD3 Agent
```bash
python td3_parking.py
```

### Test Full Pipeline
```bash
python smart_logger.py test_images
```

---

## 📊 **Expected Results**

### Plate Detection Model
- **mAP@0.5**: >90% (excellent)
- **mAP@0.5:0.95**: >70% (good)
- **Inference Speed**: 30+ FPS on Raspberry Pi 4

### Character Recognition Model
- **mAP@0.5**: >85% (excellent for Arabic text)
- **Per-Class Accuracy**: 80-95% per character
- **Inference Speed**: 50+ FPS

### TD3 Agent
- **Average Reward**: 5-7 (good)
- **Success Rate**: 95%+ (avoids occupied spots)
- **Optimal Allocation**: Considers both availability and distance

---

## 🔧 **Troubleshooting**

### Issue: Out of Memory During Training
**Solution:**
```python
# In train_plates.py or train_characters.py
# Reduce batch size
model.train(data='data.yaml', epochs=50, batch=8)  # Was 16
```

### Issue: Dataset Not Found
**Solution:**
```bash
# Verify dataset structure
ls dataset/
# Should show: EALPR Vechicles dataset, EALPR- Plates dataset, etc.

# Re-run preparation scripts
python prepare_dataset.py
python prepare_character_data.py
```

### Issue: Training Too Slow on CPU
**Solution:**
```python
# Use smaller model or fewer epochs
model = YOLO('yolo11n.pt')  # Nano is fastest
model.train(data='data.yaml', epochs=25)  # Reduce from 50
```

### Issue: Low Accuracy After Training
**Solution:**
- Train for more epochs
- Check if dataset was prepared correctly
- Verify annotations are correct
- Try data augmentation (already enabled in training scripts)

---

## 📁 **File Structure After Training**

```
SmartParkingSystem/
├── dataset/                       # Downloaded EALPR dataset
│   ├── EALPR Vechicles dataset/
│   ├── EALPR- Plates dataset/
│   └── EALPR- LP characters dataset/
│
├── plate_detector.pt              # ✅ Your trained model
├── character_detector.pt          # ✅ Your trained model
├── td3_actor.pth                  # ✅ Your trained model
│
├── runs/                          # Training outputs
│   └── detect/
│       ├── train/                 # Plate training results
│       └── train2/                # Character training results
│
├── prepare_dataset.py             # Step 2.1 script
├── prepare_character_data.py      # Step 2.2 script
├── train_plates.py                # Step 3.1 script
├── train_characters.py            # Step 3.2 script
└── train_simulation.py            # Step 3.3 script
```

---

## 🎯 **Training on Different Hardware**

### On Laptop/PC (Development)
```bash
# Full training with GPU
python train_plates.py
python train_characters.py
python train_simulation.py
```

### On Raspberry Pi (Not Recommended)
Training on Raspberry Pi is **extremely slow**. Instead:
1. Train models on your PC/laptop
2. Copy trained `.pt` and `.pth` files to Pi
3. Run inference on Pi (works great!)

### On Google Colab (Free GPU)
```python
# Upload scripts to Colab
# Mount Google Drive with dataset
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install ultralytics torch

# Train models
!python train_plates.py
!python train_characters.py
```

---

## 📈 **Improving Model Performance**

### For Better Plate Detection
1. **More Data**: Add your own plate images
2. **Longer Training**: Increase epochs to 100+
3. **Larger Model**: Use `yolo11s.pt` or `yolo11m.pt` instead of `yolo11n.pt`
4. **Data Augmentation**: Already enabled (rotation, scaling, color jitter)

### For Better Character Recognition
1. **Class Balancing**: Ensure all Arabic characters have enough samples
2. **Higher Resolution**: Increase image size to 640x640
3. **More Epochs**: Train for 150-200 epochs
4. **Fine-tuning**: Start with pre-trained model

### For Better TD3 Agent
1. **More Episodes**: Train for 50,000+ episodes
2. **Hyperparameter Tuning**: Adjust learning rate, buffer size
3. **Reward Shaping**: Modify reward function in `train_simulation.py`

---

## 🎓 **Next Steps**

After successfully training all models:

1. ✅ Test models with `smart_logger.py`
2. ✅ Deploy to Raspberry Pi (see main README)
3. ✅ Monitor performance and retrain if needed
4. ✅ Share your trained models (optional)

---

## 📞 **Need Help?**

- **Dataset Issues**: Check [EALPR Repository](https://github.com/ahmedramadan96/EALPR)
- **Training Issues**: Open an issue on GitHub
- **Contact**: z.ahmed2003@gmail.com

---

## 📚 **Additional Resources**

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [EALPR Dataset Paper](https://doi.org/10.1109/ACIRS55390.2022.9845514)

---

**Happy Training! 🚀**
