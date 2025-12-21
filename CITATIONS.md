# Citations and Acknowledgments

This project uses datasets and builds upon the work of other researchers. We are grateful for their contributions to the open-source community.

---

## 📚 Dataset Citations

### EALPR Dataset - Egyptian License Plate Recognition

This project uses the **EALPR (Egyptian Automatic License Plate Recognition) dataset** for training and testing license plate detection and character recognition models.

**Repository**: [ahmedramadan96/EALPR](https://github.com/ahmedramadan96/EALPR)

**Citation**:
```bibtex
@INPROCEEDINGS{9845514,
  author={Youssef, Ahmed Ramadan and Sayed, Fawzya Ramadan and Ali, Abdelmgeid Ameen},
  booktitle={2022 7th Asia-Pacific Conference on Intelligent Robot Systems (ACIRS)}, 
  title={A New Benchmark Dataset for Egyptian License Plate Detection and Recognition}, 
  year={2022},
  volume={},
  number={},
  pages={106-111},
  doi={10.1109/ACIRS55390.2022.9845514}
}
```

**Paper**: [DOI: 10.1109/ACIRS55390.2022.9845514](https://doi.org/10.1109/ACIRS55390.2022.9845514)

**Description**: EALPR is a comprehensive benchmark dataset for Egyptian license plate detection and recognition, containing:
- Vehicle images with Egyptian license plates
- Annotated license plate regions
- Individual character annotations for Arabic license plates

This dataset was instrumental in training our YOLOv11 models for:
- License plate detection (`plate_detector.pt`)
- Arabic character recognition (`character_detector.pt`)

---

## 🔬 Algorithms and Frameworks

### TD3 (Twin Delayed DDPG)
Our parking allocation system uses the TD3 reinforcement learning algorithm:

**Original Paper**:
```bibtex
@inproceedings{fujimoto2018addressing,
  title={Addressing function approximation error in actor-critic methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1587--1596},
  year={2018},
  organization={PMLR}
}
```

### YOLOv11 (Ultralytics)
Object detection framework used for license plate and character detection:

**Repository**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

**Citation**:
```bibtex
@software{yolov11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics}
}
```

---

## 🙏 Acknowledgments

We thank:
- **Ahmed Ramadan Youssef**, **Fawzya Ramadan Sayed**, and **Abdelmgeid Ameen Ali** for creating and sharing the EALPR dataset
- The **Ultralytics** team for the YOLOv11 framework
- The **PyTorch** team for the deep learning framework
- The **OpenCV** community for computer vision tools

---

## 📄 License Compatibility

This project respects the licenses of all datasets and libraries used:
- EALPR Dataset: Please refer to the [original repository](https://github.com/ahmedramadan96/EALPR) for license terms
- Our code: [Specify your license here]

If you use this project, please cite both our work and the EALPR dataset.
