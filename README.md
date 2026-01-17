<p align="center">
  <h1 align="center">🎯 Face Recognition Attendance System</h1>
</p>

<p align="center">
  <strong>Custom-trained lightweight face recognition model + Next.js demo web app</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.9-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Model-LightFaceNet-orange" alt="LightFaceNet">
  <img src="https://img.shields.io/badge/Dataset-LFW-green" alt="LFW">
  <img src="https://img.shields.io/badge/Next.js-14-black?logo=next.js&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white" alt="FastAPI">
</p>

---

## ✨ Features

- 🧠 **Custom-Trained Model** - LightFaceNet trained on LFW dataset (not pre-trained)
- ⚡ **Lightweight & Fast** - 1.1M parameters, ~5ms inference on CPU
- 📓 **Jupyter Pipeline** - Complete training workflow in notebooks
- 🌐 **Next.js Web Demo** - Modern React-based UI with webcam
- 🔌 **FastAPI Backend** - REST API for recognition

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  NEXT.JS FRONTEND ◄──────► FASTAPI BACKEND             │
│  • Webcam capture         • /api/recognize              │
│  • Enrollment UI          • /api/enroll                 │
│  • Attendance log         • /api/attendance             │
├─────────────────────────────────────────────────────────┤
│                    LIGHTFACENET                         │
│  Backbone: MobileNetV3-Small | Output: 128-dim embed   │
│  Size: 1.1M params (~4MB) | Speed: ~5ms (CPU)          │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Model Specs

| Spec | Value |
|------|-------|
| **Architecture** | LightFaceNet (MobileNetV3-Small) |
| **Parameters** | 1.1M |
| **Input Size** | 112 × 112 × 3 |
| **Embedding** | 128-dim L2-normalized |
| **Training Loss** | Triplet Loss (margin=0.2) |
| **Dataset** | LFW (Labeled Faces in the Wild) |

---

## 📓 Training Pipeline

| Notebook | Description |
|----------|-------------|
| `01_download_dataset.ipynb` | Download & explore LFW |
| `02_data_preprocessing.ipynb` | Resize, normalize, split |
| `03_model_architecture.ipynb` | Define LightFaceNet |
| `04_model_training.ipynb` | Train with triplet loss |
| `05_model_evaluation.ipynb` | Evaluate on LFW benchmark |

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/yourusername/face-based_attendance_system.git
cd face-based_attendance_system
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r backend/requirements.txt

# 2. Train model (run notebooks in order)
jupyter notebook notebooks/

# 3. Start backend
cd backend
uvicorn app.main:app --reload

# 4. Start frontend
cd frontend
npm run dev
```

---

## 📁 Project Structure

```
face-based_attendance_system/
├── notebooks/           # Jupyter training pipeline
├── backend/
│   └── app/
│       └── ml/
│           └── model.py # LightFaceNet
├── frontend/            # Next.js web app
├── models/              # Trained checkpoints
├── data/lfw/            # LFW dataset
└── scripts/             # Utility scripts
```

---

## 📄 License

MIT License

---

## 👤 Author

**Mishaf Hasan** - [GitHub](https://github.com/mind-flayers)

---

<p align="center">⭐ Star this repo if you find it helpful!</p>
