"""
Face-Based Attendance System - FastAPI Backend

Production-ready API for face registration, verification, and attendance tracking.
Uses ONNX Runtime for efficient inference with MobileFaceNet model.
"""

import os
import io
import uuid
import time
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")

# Face detection
try:
    from facenet_pytorch import MTCNN
    import torch
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.warning("facenet-pytorch not available. Face detection will be limited.")


# =============================================================================
# Configuration
# =============================================================================

class Settings:
    """Application settings."""
    APP_NAME: str = "Face Attendance API"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Model paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../models/exported/mobilefacenet.onnx")
    
    # Face detection
    FACE_SIZE: int = 112
    MIN_FACE_SIZE: int = 40
    DETECTION_THRESHOLD: float = 0.9
    
    # Verification
    VERIFICATION_THRESHOLD: float = 0.45  # Cosine similarity threshold
    
    # Storage
    EMBEDDINGS_DIR: str = os.getenv("EMBEDDINGS_DIR", "../data/enrollments")
    ATTENDANCE_LOG_DIR: str = os.getenv("ATTENDANCE_LOG_DIR", "../data/attendance")
    
    # ONNX Runtime
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"


settings = Settings()


# =============================================================================
# API Models
# =============================================================================

class PersonInfo(BaseModel):
    """Person registration information."""
    person_id: str = Field(..., description="Unique identifier for the person")
    name: str = Field(..., description="Full name of the person")
    department: Optional[str] = Field(None, description="Department or group")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RegistrationResponse(BaseModel):
    """Registration response."""
    success: bool
    person_id: str
    message: str
    embedding_id: str


class VerificationRequest(BaseModel):
    """Verification request."""
    person_id: str = Field(..., description="Person ID to verify against")
    threshold: Optional[float] = Field(None, description="Custom threshold (0-1)")


class VerificationResponse(BaseModel):
    """Verification response."""
    success: bool
    verified: bool
    person_id: str
    similarity: float
    threshold: float
    message: str


class AttendanceRecord(BaseModel):
    """Single attendance record."""
    person_id: str
    name: str
    timestamp: datetime
    similarity: float
    status: str  # "check_in" or "check_out"


class AttendanceResponse(BaseModel):
    """Attendance marking response."""
    success: bool
    person_id: str
    name: str
    timestamp: datetime
    similarity: float
    status: str
    message: str


class IdentificationResponse(BaseModel):
    """Identification response (1:N matching)."""
    success: bool
    identified: bool
    person_id: Optional[str]
    name: Optional[str]
    similarity: float
    message: str


# =============================================================================
# Face Recognition Service
# =============================================================================

class FaceRecognitionService:
    """Face recognition service using ONNX Runtime."""
    
    def __init__(self, model_path: str, use_gpu: bool = False):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.session = None
        self.mtcnn = None
        self.embeddings_db: Dict[str, Dict] = {}  # In-memory database
        
        self._initialize()
    
    def _initialize(self):
        """Initialize ONNX session and face detector."""
        # Initialize ONNX Runtime
        if ONNX_AVAILABLE and Path(self.model_path).exists():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"ONNX model loaded from {self.model_path}")
            logger.info(f"Providers: {self.session.get_providers()}")
        else:
            logger.warning(f"ONNX model not found at {self.model_path}")
        
        # Initialize MTCNN
        if MTCNN_AVAILABLE:
            device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
            self.mtcnn = MTCNN(
                image_size=settings.FACE_SIZE,
                margin=14,
                min_face_size=settings.MIN_FACE_SIZE,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=device
            )
            logger.info(f"MTCNN initialized on {device}")
        
        # Load existing embeddings
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings from disk."""
        embeddings_dir = Path(settings.EMBEDDINGS_DIR)
        if not embeddings_dir.exists():
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for person_dir in embeddings_dir.iterdir():
            if person_dir.is_dir():
                person_id = person_dir.name
                metadata_file = person_dir / "metadata.npz"
                
                if metadata_file.exists():
                    data = np.load(metadata_file, allow_pickle=True)
                    self.embeddings_db[person_id] = {
                        'name': str(data.get('name', person_id)),
                        'embedding': data['embedding'],
                        'department': str(data.get('department', '')),
                        'created_at': str(data.get('created_at', datetime.now().isoformat()))
                    }
        
        logger.info(f"Loaded {len(self.embeddings_db)} registered persons")
    
    def detect_face(self, image: np.ndarray) -> tuple:
        """
        Detect and align face in image.
        
        Returns:
            (aligned_face, confidence, landmarks)
        """
        if self.mtcnn is None:
            raise HTTPException(status_code=500, detail="Face detector not initialized")
        
        # Convert to PIL for MTCNN
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect
        boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            return None, 0, None
        
        # Get best face
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]
        
        if confidence < settings.DETECTION_THRESHOLD:
            return None, confidence, None
        
        # Get aligned face
        aligned = self.mtcnn(pil_image)
        
        if aligned is None:
            return None, confidence, None
        
        # Convert to numpy (HWC format)
        aligned = aligned.permute(1, 2, 0).numpy()
        aligned = ((aligned + 1) / 2 * 255).astype(np.uint8)  # Denormalize
        
        return aligned, confidence, landmarks[best_idx]
    
    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        """Get embedding for aligned face."""
        if self.session is None:
            raise HTTPException(status_code=500, detail="ONNX model not loaded")
        
        # Preprocess
        face = cv2.resize(face, (settings.FACE_SIZE, settings.FACE_SIZE))
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5  # Normalize to [-1, 1]
        face = face.transpose(2, 0, 1)  # HWC -> CHW
        face = np.expand_dims(face, 0)  # Add batch dim
        
        # Inference
        embedding = self.session.run([self.output_name], {self.input_name: face})[0]
        return embedding.flatten()
    
    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def register(self, person_id: str, name: str, face: np.ndarray, 
                 department: Optional[str] = None) -> str:
        """Register a new person."""
        # Get embedding
        embedding = self.get_embedding(face)
        
        # Store in memory
        self.embeddings_db[person_id] = {
            'name': name,
            'embedding': embedding,
            'department': department or '',
            'created_at': datetime.now().isoformat()
        }
        
        # Persist to disk
        person_dir = Path(settings.EMBEDDINGS_DIR) / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            person_dir / "metadata.npz",
            name=name,
            embedding=embedding,
            department=department or '',
            created_at=datetime.now().isoformat()
        )
        
        # Save face image
        cv2.imwrite(str(person_dir / "face.jpg"), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        
        embedding_id = f"{person_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Registered person: {name} (ID: {person_id})")
        
        return embedding_id
    
    def verify(self, person_id: str, face: np.ndarray, 
               threshold: Optional[float] = None) -> tuple:
        """Verify a person (1:1 matching)."""
        if person_id not in self.embeddings_db:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        
        threshold = threshold or settings.VERIFICATION_THRESHOLD
        
        # Get embeddings
        probe_embedding = self.get_embedding(face)
        gallery_embedding = self.embeddings_db[person_id]['embedding']
        
        # Compare
        similarity = self.compare(probe_embedding, gallery_embedding)
        verified = similarity >= threshold
        
        return verified, similarity, threshold
    
    def identify(self, face: np.ndarray, threshold: Optional[float] = None) -> tuple:
        """Identify a person (1:N matching)."""
        if not self.embeddings_db:
            return None, None, 0.0
        
        threshold = threshold or settings.VERIFICATION_THRESHOLD
        
        # Get probe embedding
        probe_embedding = self.get_embedding(face)
        
        # Compare against all registered persons
        best_match = None
        best_similarity = 0.0
        
        for person_id, data in self.embeddings_db.items():
            similarity = self.compare(probe_embedding, data['embedding'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id
        
        if best_similarity >= threshold and best_match:
            return best_match, self.embeddings_db[best_match]['name'], best_similarity
        
        return None, None, best_similarity
    
    def get_all_persons(self) -> List[Dict]:
        """Get all registered persons."""
        return [
            {
                'person_id': pid,
                'name': data['name'],
                'department': data.get('department', ''),
                'created_at': data.get('created_at', '')
            }
            for pid, data in self.embeddings_db.items()
        ]


# =============================================================================
# Attendance Service
# =============================================================================

class AttendanceService:
    """Attendance tracking service."""
    
    def __init__(self):
        self.attendance_log: Dict[str, List[AttendanceRecord]] = {}
        self.log_dir = Path(settings.ATTENDANCE_LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def mark_attendance(self, person_id: str, name: str, 
                        similarity: float, status: str = "check_in") -> AttendanceRecord:
        """Mark attendance for a person."""
        record = AttendanceRecord(
            person_id=person_id,
            name=name,
            timestamp=datetime.now(),
            similarity=similarity,
            status=status
        )
        
        # Add to in-memory log
        today = date.today().isoformat()
        if today not in self.attendance_log:
            self.attendance_log[today] = []
        self.attendance_log[today].append(record)
        
        # Persist to file
        log_file = self.log_dir / f"{today}.csv"
        with open(log_file, 'a') as f:
            f.write(f"{record.timestamp.isoformat()},{person_id},{name},{similarity:.4f},{status}\n")
        
        logger.info(f"Attendance marked: {name} ({status}) at {record.timestamp}")
        return record
    
    def get_attendance(self, date_str: Optional[str] = None) -> List[Dict]:
        """Get attendance records for a date."""
        target_date = date_str or date.today().isoformat()
        
        records = self.attendance_log.get(target_date, [])
        
        # Also load from file if not in memory
        if not records:
            log_file = self.log_dir / f"{target_date}.csv"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 5:
                            records.append({
                                'timestamp': parts[0],
                                'person_id': parts[1],
                                'name': parts[2],
                                'similarity': float(parts[3]),
                                'status': parts[4]
                            })
        
        return [r.dict() if hasattr(r, 'dict') else r for r in records]


# =============================================================================
# Initialize Services
# =============================================================================

face_service = FaceRecognitionService(
    model_path=settings.MODEL_PATH,
    use_gpu=settings.USE_GPU
)

attendance_service = AttendanceService()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Face-based attendance system with MobileFaceNet + ArcFace"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

async def process_image(file: UploadFile) -> np.ndarray:
    """Process uploaded image file."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    return image


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "model_loaded": face_service.session is not None,
        "detector_loaded": face_service.mtcnn is not None,
        "registered_persons": len(face_service.embeddings_db)
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model": {
            "loaded": face_service.session is not None,
            "path": settings.MODEL_PATH
        },
        "detector": {
            "loaded": face_service.mtcnn is not None,
            "type": "MTCNN"
        },
        "database": {
            "registered_persons": len(face_service.embeddings_db)
        }
    }


@app.post("/api/register", response_model=RegistrationResponse)
async def register_person(
    person_id: str = Query(..., description="Unique person identifier"),
    name: str = Query(..., description="Person's name"),
    department: Optional[str] = Query(None, description="Department"),
    file: UploadFile = File(..., description="Face image")
):
    """
    Register a new person in the system.
    
    - **person_id**: Unique identifier (e.g., employee ID)
    - **name**: Full name
    - **department**: Optional department/group
    - **file**: Face image (JPEG/PNG)
    """
    # Check if already registered
    if person_id in face_service.embeddings_db:
        raise HTTPException(status_code=400, detail=f"Person {person_id} already registered")
    
    # Process image
    image = await process_image(file)
    
    # Detect face
    aligned_face, confidence, _ = face_service.detect_face(image)
    
    if aligned_face is None:
        raise HTTPException(
            status_code=400,
            detail=f"No face detected (confidence: {confidence:.2f})"
        )
    
    # Register
    embedding_id = face_service.register(person_id, name, aligned_face, department)
    
    return RegistrationResponse(
        success=True,
        person_id=person_id,
        message=f"Successfully registered {name}",
        embedding_id=embedding_id
    )


@app.post("/api/verify", response_model=VerificationResponse)
async def verify_person(
    person_id: str = Query(..., description="Person ID to verify against"),
    threshold: Optional[float] = Query(None, description="Custom threshold"),
    file: UploadFile = File(..., description="Face image to verify")
):
    """
    Verify if the face matches a registered person (1:1 verification).
    
    - **person_id**: ID of the registered person
    - **threshold**: Optional similarity threshold (0-1)
    - **file**: Face image to verify
    """
    # Process image
    image = await process_image(file)
    
    # Detect face
    aligned_face, confidence, _ = face_service.detect_face(image)
    
    if aligned_face is None:
        raise HTTPException(
            status_code=400,
            detail=f"No face detected (confidence: {confidence:.2f})"
        )
    
    # Verify
    try:
        verified, similarity, used_threshold = face_service.verify(
            person_id, aligned_face, threshold
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return VerificationResponse(
        success=True,
        verified=verified,
        person_id=person_id,
        similarity=similarity,
        threshold=used_threshold,
        message="Verified" if verified else "Not verified"
    )


@app.post("/api/identify", response_model=IdentificationResponse)
async def identify_person(
    threshold: Optional[float] = Query(None, description="Minimum similarity threshold"),
    file: UploadFile = File(..., description="Face image")
):
    """
    Identify who the face belongs to (1:N identification).
    
    Searches through all registered persons and returns the best match.
    """
    # Process image
    image = await process_image(file)
    
    # Detect face
    aligned_face, confidence, _ = face_service.detect_face(image)
    
    if aligned_face is None:
        raise HTTPException(
            status_code=400,
            detail=f"No face detected (confidence: {confidence:.2f})"
        )
    
    # Identify
    person_id, name, similarity = face_service.identify(aligned_face, threshold)
    
    identified = person_id is not None
    
    return IdentificationResponse(
        success=True,
        identified=identified,
        person_id=person_id,
        name=name,
        similarity=similarity,
        message=f"Identified as {name}" if identified else "Unknown person"
    )


@app.post("/api/attendance", response_model=AttendanceResponse)
async def mark_attendance(
    status: str = Query("check_in", description="check_in or check_out"),
    threshold: Optional[float] = Query(None, description="Verification threshold"),
    file: UploadFile = File(..., description="Face image")
):
    """
    Mark attendance by identifying the person and logging the event.
    
    - **status**: Either "check_in" or "check_out"
    - **threshold**: Optional similarity threshold
    - **file**: Face image
    """
    if status not in ["check_in", "check_out"]:
        raise HTTPException(status_code=400, detail="Status must be 'check_in' or 'check_out'")
    
    # Process image
    image = await process_image(file)
    
    # Detect face
    aligned_face, confidence, _ = face_service.detect_face(image)
    
    if aligned_face is None:
        raise HTTPException(
            status_code=400,
            detail=f"No face detected (confidence: {confidence:.2f})"
        )
    
    # Identify person
    person_id, name, similarity = face_service.identify(aligned_face, threshold)
    
    if person_id is None:
        raise HTTPException(status_code=404, detail="Person not recognized")
    
    # Mark attendance
    record = attendance_service.mark_attendance(person_id, name, similarity, status)
    
    return AttendanceResponse(
        success=True,
        person_id=person_id,
        name=name,
        timestamp=record.timestamp,
        similarity=similarity,
        status=status,
        message=f"Attendance marked for {name}"
    )


@app.get("/api/attendance")
async def get_attendance(
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD)")
):
    """Get attendance records for a specific date."""
    records = attendance_service.get_attendance(date)
    return {
        "date": date or datetime.now().date().isoformat(),
        "count": len(records),
        "records": records
    }


@app.get("/api/persons")
async def list_persons():
    """Get all registered persons."""
    persons = face_service.get_all_persons()
    return {
        "count": len(persons),
        "persons": persons
    }


@app.delete("/api/persons/{person_id}")
async def delete_person(person_id: str):
    """Delete a registered person."""
    if person_id not in face_service.embeddings_db:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    
    # Remove from memory
    del face_service.embeddings_db[person_id]
    
    # Remove from disk
    person_dir = Path(settings.EMBEDDINGS_DIR) / person_id
    if person_dir.exists():
        import shutil
        shutil.rmtree(person_dir)
    
    return {"success": True, "message": f"Deleted person {person_id}"}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
