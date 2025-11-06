# 🗑️ Waste Classification System

Deep learning system for automated waste classification with **95% accuracy**, ready for production deployment.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Highlights

- 🎯 **95% Classification Accuracy** using MobileNetV2 Transfer Learning
- 📹 **Real-time Detection** at 30+ FPS with YOLOv8
- 📱 **Edge-Ready** - 74% model size reduction via INT8 quantization
- 🏗️ **Production-Ready** - Modular architecture

## 📦 10 Waste Categories

```
battery • biological • cardboard • clothes • glass • metal • paper • plastic • shoes • trash
```

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/AnHgPham/waste-classification.git
cd waste-classification
pip install -r requirements.txt

# 2. Download dataset to data/raw/ from Kaggle
# https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

# 3. Run pipeline
python scripts/02_preprocessing.py
python scripts/04_transfer_learning.py
python scripts/05_realtime_detection.py
```

## 📈 Results

| Model | Accuracy | Size | Inference (CPU) |
|-------|----------|------|-----------------|
| Baseline CNN | 85% | 4.8 MB | 15 ms |
| MobileNetV2 | **95%** | 9.2 MB | 20 ms |
| MobileNetV2 (INT8) | 94% | **2.4 MB** | **8 ms** |

## 🏗️ Architecture

1. **Baseline CNN** - Custom architecture (85% accuracy)
2. **Transfer Learning** - MobileNetV2 fine-tuning (95% accuracy)
3. **Real-time Detection** - YOLOv8 + MobileNetV2 integration
4. **Model Optimization** - TFLite + INT8 quantization

## 📁 Project Structure

```
├── src/              # Source code (config, models, data, detection)
├── scripts/          # Executable scripts
├── data/             # Dataset (raw & processed)
├── outputs/          # Models, reports, logs
└── main.py           # CLI entry point
```

## 💻 Usage

```bash
# Data preprocessing
python scripts/02_preprocessing.py

# Train baseline model
python scripts/03_baseline_training.py

# Train transfer learning model
python scripts/04_transfer_learning.py

# Real-time detection
python scripts/05_realtime_detection.py

# Evaluate model
python scripts/99_evaluate_model.py --model mobilenetv2
```

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.13+
- 4GB+ RAM
- (Optional) GPU for faster training

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get running in 3 steps
- **[Vietnamese Guide](HUONG_DAN_CHO_NGUOI_MOI.md)** - Hướng dẫn chi tiết

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Pham An** - Waste Classification Capstone Project (2024)

---

## 🇻🇳 Tiếng Việt

Hệ thống phân loại rác thải tự động với độ chính xác **95%** sử dụng Deep Learning.

### Bắt Đầu Nhanh

```bash
pip install -r requirements.txt
python scripts/02_preprocessing.py
python scripts/04_transfer_learning.py
```

### Tài Liệu

📖 **[Hướng Dẫn Chi Tiết](HUONG_DAN_CHO_NGUOI_MOI.md)** - Giải thích từng bước bằng tiếng Việt

---

<div align="center">

**Made with ❤️ for the environment**

[⬆ Back to Top](#-waste-classification-system)

</div>
