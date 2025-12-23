# Citations and Acknowledgments

This project uses datasets and builds upon the work of other researchers. We are grateful for their contributions to the open-source community.

---

## 🚗 Citing This Project

If you use this Smart Parking System in your research or project, please cite it as:

```bibtex
@software{smartparkingsystem2025,
  author = {Ziad Ahmed},
  title = {Smart Parking System: Egyptian License Plate Recognition with RL-based Allocation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ziadahmeddd/Smart-Parking-System-for-Egypt-License-Plates},
  note = {A comprehensive parking management system using YOLOv11 for Egyptian license plate recognition and TD3 reinforcement learning for optimal parking allocation}
}
```

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

### PyTorch
Deep learning framework used throughout the project:

**Repository**: [pytorch/pytorch](https://github.com/pytorch/pytorch)

**Citation**:
```bibtex
@inproceedings{pytorch2019,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
  booktitle={Advances in Neural Information Processing Systems 32},
  pages={8024--8035},
  year={2019},
  publisher={Curran Associates, Inc.},
  url={http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
}
```

### Stable-Baselines3
Reinforcement learning library used for TD3 implementation:

**Repository**: [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

**Citation**:
```bibtex
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

### OpenCV
Computer vision library used for image processing:

**Repository**: [opencv/opencv](https://github.com/opencv/opencv)

**Citation**:
```bibtex
@article{opencv_library,
  author = {Bradski, G.},
  citeulike-article-id = {2236121},
  journal = {Dr. Dobb's Journal of Software Tools},
  keywords = {bibtex-import},
  title = {{The OpenCV Library}},
  year = {2000}
}
```

---

## 🙏 Acknowledgments

We thank:
- **Ahmed Ramadan Youssef**, **Fawzya Ramadan Sayed**, and **Abdelmgeid Ameen Ali** for creating and sharing the EALPR dataset
- **Scott Fujimoto**, **Herke Hoof**, and **David Meger** for the TD3 algorithm
- The **Ultralytics** team for the YOLOv11 framework
- The **PyTorch** team for the deep learning framework
- The **Stable-Baselines3** team for reliable RL implementations
- The **OpenCV** community for computer vision tools

---

## 📄 License Compatibility

This project respects the licenses of all datasets and libraries used:

### This Project
- **Our code**: MIT License (see [LICENSE](LICENSE) file)

### Datasets
- **EALPR Dataset**: Please refer to the [original repository](https://github.com/ahmedramadan96/EALPR) for license terms

### Dependencies
- **Ultralytics YOLOv11**: AGPL-3.0 License
  - ⚠️ **Important**: YOLOv11 uses AGPL-3.0, which requires that any modifications to the library and any software that incorporates it must also be open-sourced under AGPL-3.0 if distributed. For commercial use, consider Ultralytics' commercial licensing options.
- **PyTorch**: BSD-3-Clause License
- **Stable-Baselines3**: MIT License
- **OpenCV**: Apache 2.0 License

**License Compatibility Notes**:
- Most dependencies (PyTorch, Stable-Baselines3, OpenCV) use permissive licenses compatible with MIT
- The AGPL-3.0 license of Ultralytics YOLOv11 is the most restrictive component and may affect distribution rights
- Users should review the AGPL-3.0 terms if planning commercial deployment

If you use this project, please cite both our work and the EALPR dataset.
