# Face-Based Attendance System

A production-grade face recognition attendance system built with **MobileFaceNet + ArcFace** architecture. Features a complete end-to-end pipeline from model training on Google Colab to real-time inference via a FastAPI backend with ONNX Runtime.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE (Colab)                    │
│                                                                 │
│  VGGFace2-112x112  ──►  MobileFaceNet  ──►  ArcFace Loss       │
│  (8,631 identities)     (~1M params)       (s=64, m=0.5)       │
│                              │                                  │
│                    ┌─────────┴─────────┐                        │
│                    ▼                   ▼                         │
│              ONNX Export        TorchScript Export               │
│           (mobilefacenet.onnx)  (mobilefacenet.pt)              │
└─────────────────────┬──────────────────┬────────────────────────┘
                      │                  │
                      ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                 INFERENCE BACKEND (FastAPI)                      │
│                                                                 │
│  Image ──► MTCNN Detection ──► ONNX Runtime ──► 512-D Embedding │
│                                                      │          │
│                                              Cosine Similarity  │
│                                                      │          │
│                                    Register / Verify / Identify │
└─────────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Property | Value |
|---|---|
| **Backbone** | MobileFaceNet (depthwise separable convolutions) |
| **Parameters** | ~1M |
| **Input** | 112×112 RGB (normalized to [-1, 1]) |
| **Output** | 512-D L2-normalized embedding |
| **Loss Function** | ArcFace (scale=64.0, margin=0.5 rad / ~28.6°) |
| **Training Data** | VGGFace2-112x112 (8,631 identities, ~3.14M images) |
| **Inference Engine** | ONNX Runtime (CPU/GPU) |
| **Inference Time** | <5ms (CPU) |

---

## Project Structure

```
face-based_attendance_system/
├── notebooks/
│   ├── 01_dataset_download.ipynb               # Section 1 — VGGFace2 download & train/val split
│   ├── 02_model_architecture.ipynb             # Section 2 — MobileFaceNet + ArcFace definition
│   ├── 03_training_optimization.ipynb          # Section 3 — AMP, Partial FC, LR scheduling
│   ├── 04_model_training.ipynb                 # Section 4 — Full training loop (main notebook)
│   ├── 05_model_evaluation.ipynb               # Section 5 — LFW benchmark, ROC, TAR@FAR
│   ├── 06_model_export.ipynb                   # Section 6 — ONNX + TorchScript export
│   └── colab/
│       ├── face_recognition_attendance_complete_pipeline.ipynb  # Combined reference notebook
│       └── data_preprocessing_kaggle_upload.ipynb               # Data preprocessing notebook
├── backend/
│   ├── main.py                     # FastAPI application (ONNX Runtime inference)
│   ├── requirements.txt            # Backend dependencies
│   └── app/
│       ├── config.py               # Pydantic settings (env-based configuration)
│       └── ml/
│           ├── mobilefacenet.py    # MobileFaceNet architecture definition
│           ├── arcface.py          # ArcFace / CosFace / PartialFC loss implementations
│           └── face_service.py     # DeepFace wrapper (Facenet512 — alternative pipeline)
├── models/
│   ├── checkpoints/                # Training checkpoints (.pth)
│   │   ├── best_backbone.pth
│   │   ├── checkpoint_epoch_25.pth
│   │   ├── checkpoint_latest.pth
│   │   └── checkpoint_step_33000.pth
│   └── exported/                   # Production models
│       ├── mobilefacenet.onnx      # ONNX model (~4MB)
│       ├── mobilefacenet.pt        # TorchScript model
│       ├── inference.py            # Standalone inference wrapper
│       ├── requirements.txt        # Inference-only dependencies
│       └── README.md               # Model documentation
├── config/
│   ├── training_config.yaml        # Training hyperparameters (source of truth)
│   └── inference_config.yaml       # Inference settings
├── scripts/
│   ├── model.py                    # Shared MobileFaceNet + ArcFace architecture
│   ├── download_dataset.py         # VGGFace2-112x112 download + train/val split
│   ├── train.py                    # Full training pipeline (AMP, margin warmup, checkpointing)
│   ├── evaluate.py                 # LFW evaluation, ROC, TAR@FAR, benchmarks
│   ├── export_model.py             # ONNX + TorchScript export with verification
│   ├── benchmark.py                # Model performance benchmarking
│   └── monitor_training_performance.py  # GPU/CPU monitoring during training
├── data/
│   ├── enrollments/                # Registered face embeddings (runtime)
│   └── attendance/                 # Attendance logs (runtime)
├── docs/
│   ├── FEASIBILITY_AND_COST_ANALYSIS.md
│   └── GPU_OPTIMIZATION_GUIDE.md
├── config.py                       # Environment detection (Colab vs local)
├── Dockerfile                      # Multi-stage production build
├── docker-compose.yml              # Full stack deployment
├── requirements.txt                # Full stack dependencies
├── verify_setup.py                 # Environment verification utility
```

---

## Quick Start

### 1. Training (Google Colab)

The training pipeline is split into **6 focused notebooks**, designed to be run sequentially in Google Colab:

1. Upload the project to `Google Drive > MyDrive > face_based_attendance_system`
2. Open each notebook in Colab in order
3. Set runtime to **GPU** (Runtime → Change runtime type → T4/V100/A100)
4. Run all cells sequentially

| Notebook | Description | When to run |
|---|---|---|
| `notebooks/01_dataset_download.ipynb` | Download VGGFace2-112×112 from Kaggle, create 90/10 train/val split | Once (cached after first run) |
| `notebooks/02_model_architecture.ipynb` | MobileFaceNet + ArcFace definition, parameter analysis, architecture visualization | Reference / once |
| `notebooks/03_training_optimization.ipynb` | AMP setup, Partial FC, LR scheduling, Colab checkpoint protection | Reference / once |
| `notebooks/04_model_training.ipynb` | **Main training loop** — 25 epochs, auto-checkpoint every 500 steps, auto-resume | Every training run |
| `notebooks/05_model_evaluation.ipynb` | LFW pair verification, ROC curve, TAR@FAR, embedding visualization | After training |
| `notebooks/06_model_export.ipynb` | ONNX (opset 14) + TorchScript export with verification and benchmarks | After evaluation |

> **Reference notebook**: `notebooks/colab/face_recognition_attendance_complete_pipeline.ipynb` retains the original combined notebook for reference.

> **Colab Disconnect Protection**: Checkpoints are saved to Google Drive every 500 steps. Training auto-resumes from the latest checkpoint.

### 2. Backend API (Local)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `/docs`.

### 3. Docker Deployment

```bash
# Full stack
docker-compose up --build

# Backend only
docker build -t face-attendance .
docker run -p 8000:8000 -v ./models:/app/models face-attendance
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/register` | Register a new person with face image |
| `POST` | `/verify` | Verify a face against a registered person (1:1) |
| `POST` | `/identify` | Identify a face from all registered persons (1:N) |
| `POST` | `/attendance/mark` | Mark attendance for identified person |
| `GET` | `/attendance/today` | Get today's attendance records |
| `GET` | `/attendance/report` | Get attendance report for date range |
| `GET` | `/registered` | List all registered persons |
| `DELETE` | `/registered/{person_id}` | Remove a registered person |
| `GET` | `/health` | Health check |

### Example: Register a Person

```bash
curl -X POST http://localhost:8000/register \
  -F "image=@face.jpg" \
  -F "person_id=emp001" \
  -F "name=John Doe" \
  -F "department=Engineering"
```

### Example: Verify Identity

```bash
curl -X POST http://localhost:8000/verify \
  -F "image=@face.jpg" \
  -F "person_id=emp001"
```

---

## Training Configuration

Key hyperparameters are defined in `config/training_config.yaml`:

| Parameter | Value | Notes |
|---|---|---|
| Backbone | MobileFaceNet | ~1M parameters, depthwise separable |
| Embedding Size | 512 | Standard for face recognition |
| ArcFace Scale | 64.0 | Logit scaling factor |
| ArcFace Margin | 0.5 rad | ~28.6° angular margin |
| Batch Size | 256 | 128 for T4, 256+ for V100/A100 |
| Learning Rate | 0.01 | With warmup + cosine decay |
| Margin Warmup | 5 epochs | Gradual 0 → 0.5 margin increase |
| Mixed Precision | Enabled | ~2x speedup with AMP |
| Epochs | 25 | With checkpoint every 500 steps |
| Input Size | 112×112 | Fixed, do not change |
| Normalization | mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] | Maps to [-1, 1] |

---

## Dataset

**VGGFace2-112x112** (`yakhyokhuja/vggface2-112x112` on Kaggle)

- Pre-aligned 112×112 RGB face images
- 8,631 identities with ~3.14 million images
- ~18.69 GB total
- Auto-split 90% train / 10% validation (stratified by identity)

**LFW** (via `sklearn.datasets.fetch_lfw_pairs`)

- Used for evaluation only (pair verification benchmark)
- Standard protocol: 6,000 pairs (3,000 positive + 3,000 negative)

---

## Standalone Inference

The exported model can be used independently:

```python
from models.exported.inference import FaceEmbedder
import numpy as np

# Initialize with ONNX model
embedder = FaceEmbedder("models/exported/mobilefacenet.onnx")

# Get 512-D embedding from a 112x112 aligned face
embedding = embedder.get_embedding(face_image)

# Compare two faces
similarity = embedder.compare(emb1, emb2)
is_match = embedder.is_same_person(emb1, emb2, threshold=0.45)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Training Framework | PyTorch 2.0+ |
| Training Environment | Google Colab (GPU) |
| Inference Engine | ONNX Runtime |
| Face Detection | MTCNN (facenet-pytorch) |
| Web Framework | FastAPI + Uvicorn |
| Containerization | Docker (multi-stage build) |
| Alternative Backend | DeepFace (Facenet512 + RetinaFace) |

---

## Environment Configuration

The system auto-detects the runtime environment via `config.py`:

```python
# Colab: PROJECT_ROOT = "/content/drive/MyDrive/face_based_attendance_system"
# Local: PROJECT_ROOT = <directory containing config.py>
```

Never hardcode paths. Always use `config.py` for `PROJECT_ROOT`.

---

## License

MIT License
