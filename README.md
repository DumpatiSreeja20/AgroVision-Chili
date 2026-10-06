# 🌶️ Early Detection of Chili Plant Diseases  
*A Deep-Learning & Thermal-Imaging Approach*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

> **Goal** Detect common chili-plant leaf diseases at an early stage by analysing **RGB + thermal images** and comparing several CNN/transfer-learning backbones (VGG16, ResNet50, Inception-ResNet-v2, MobileNet, custom CNN).  
> This repository contains all experimentation notebooks, trained-model checkpoints, and a minimal Flask demo for inference.

---

## 📂 Project Structure

```
Final Project/
├── notebooks/           # Experiment notebooks
│   ├── Final.ipynb
│   ├── FINALPROJECT.ipynb
│   ├── InceptionResNetV2 (2).ipynb
│   ├── VGG (1).ipynb
│   └── …              # Other phased experiments
├── README.md          # ← you are here

```

| Notebook | Focus | Highlights |
|----------|-------|------------|
| **FINALPROJECT.ipynb** | Complete pipeline | Data import ➜ augmentation ➜ training ➜ evaluation (ResNet-50 baseline) |
| **InceptionResNetV2 (2).ipynb** | Transfer learning | 97 % validation F1 with Inception-ResNet-v2 |
| **VGG (1).ipynb** | Lightweight backbone | Useful for edge deployment (14 MB `.h5`) |
| **secondphase.ipynb** | Thermal-only vs RGB+Thermal ablation | ROC comparison plots |
| **LAST.ipynb** | Model ensembling | Soft-voting ensemble lifts accuracy +1.8 pp |

> **Tip:** Each notebook is standalone; open in **Google Colab** or a local Jupyter environment.

---

## 🗄️ Dataset

* **Classes (5):** `Healthy`, `Anthracnose`, `Cercospora`, `Leaf Curl`, `Early Blight`  
* **Modality:** paired *RGB* and *thermal* leaf images (224 × 224).  
* **Split:** `train/`, `validation/`, `test/` folders expected under `dataset/` (≈ 6 k images total).  
* Original images are not version-controlled here; download or collect your own and keep the above folder layout.

---

## 🏗️ Installation

```bash
# 1. Create & activate a virtual-env (recommended)
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# 2. Install core dependencies
pip install -r requirements.txt
# If using GPU:
#   pip install tensorflow-gpu==2.16.1
```

> Colab users can skip the venv step; each notebook installs its own deps (`!pip install` lines).

---

## 🚀 Quick Start

### 1 · Train a model

```bash
python -m src.train \
  --data_dir /path/to/dataset/train \
  --model resnet50 \
  --epochs 30 \
  --img_size 224
```

*(The CLI helper above is optional; you can also run any notebook end-to-end.)*

### 2 · Inference demo (Flask)

```bash
export MODEL_PATH=models/chilli_disease_model.h5
python src/app.py      # visit http://127.0.0.1:5000
```

> On Colab use the `flask-ngrok` cell inside `Final.ipynb` for a one-click web UI.

---

## 📊 Results

| Backbone | Params | Best Val Acc | Test F1 | Notes |
|----------|--------|--------------|---------|-------|
| ResNet-50 | 23.5 M | **97.6 %** | 0.972 | Strong baseline |
| Inception-ResNet-v2 | 55.9 M | 97.4 % | **0.974** | Best F1 |
| VGG-16 | 14.7 M | 95.1 % | 0.947 | Lightweight |
| Custom CNN | 2.1 M | 91.3 % | 0.908 | From-scratch |

*Confusion matrices, ROC curves, and training-loss plots are embedded in each notebook.*

---

## 🛠️ How It Works

1. **Data Augmentation**  
   `ImageDataGenerator` with random rotation, shift, zoom, flip, plus brightness/contrast jitter to mimic field conditions.

2. **Transfer Learning**  
   Pre-trained ImageNet weights; top layers replaced by GlobalAveragePooling ➜ Dense(128, ReLU) ➜ Dropout(0.5) ➜ Softmax(5).

3. **Regularisation**  
   Early-Stopping (patience = 6) and Model-Checkpoint on validation loss.

4. **Evaluation Metrics**  
   Accuracy, Precision, Recall, F1-Score, ROC AUC (micro & per-class).

---

## 🤝 Contributing

Pull requests are welcome!  
If you add new disease classes or improve thermal preprocessing:

1. Fork → create feature branch  
2. Commit concise, atomic changes  
3. `pre-commit run --all-files`  
4. Open PR (please include before/after metrics)


---

## 🙏 Acknowledgements

* **Dataset:** Images captured in collaboration with AMRITA VISHWA VIDYAPEETHAM and augmented with open-source chilli-leaf datasets.  
* **Frameworks:** TensorFlow/Keras, OpenCV, Scikit-learn.  
* **Inspired by:** *Deep Learning for Plant Disease Detection* – Mohanty et al., 2016.

> Found a bug or have a question? Open an [issue](../../issues) or ping me on **[@varun-bunny](https://github.com/varun-bunny)**.
