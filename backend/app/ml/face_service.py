"""
Face Recognition Service - Core ML pipeline for the Face Attendance System.

This module provides the FaceService class which handles:
- Face detection using RetinaFace
- Embedding generation using Facenet512
- 1:1 Verification (is this person X?)
- 1:N Recognition (who is this person?)
"""
from deepface import DeepFace
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FaceService:
    """
    Main service class for face detection and recognition.
    Uses DeepFace library with Facenet512 model for optimal accuracy.
    """
    
    def __init__(
        self,
        detector_backend: str = "retinaface",
        recognition_model: str = "Facenet512",
        similarity_threshold: float = 0.60
    ):
        """
        Initialize the FaceService.
        
        Args:
            detector_backend: Face detector to use. Options: retinaface, mtcnn, opencv, ssd
            recognition_model: Recognition model. Options: Facenet512, VGG-Face, ArcFace
            similarity_threshold: Minimum cosine similarity for a match (0.55-0.65 typical)
        """
        self.detector = detector_backend
        self.model = recognition_model
        self.threshold = similarity_threshold
        self._warmup_done = False
        logger.info(f"FaceService initialized with {detector_backend} detector and {recognition_model} model")
    
    def warmup(self) -> None:
        """
        Pre-load models to avoid cold start latency.
        Call this during application startup.
        """
        if self._warmup_done:
            return
        
        logger.info("Warming up face recognition models (this may take a moment)...")
        try:
            # Trigger model download/load with a dummy operation
            # DeepFace lazy-loads models on first use
            import os
            test_img = os.path.join(os.path.dirname(__file__), "warmup_test.jpg")
            
            # Create a simple test image if it doesn't exist
            if not os.path.exists(test_img):
                import cv2
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(test_img, dummy)
            
            # This will download/load models if not cached
            DeepFace.build_model(self.model)
            self._warmup_done = True
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (models will load on first use): {e}")
    
    def detect_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect all faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector,
                enforce_detection=False
            )
            # Filter out low-confidence detections
            valid_faces = [f for f in faces if f.get("confidence", 0) > 0.9]
            logger.debug(f"Detected {len(valid_faces)} faces in {image_path}")
            return valid_faces
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate a 512-dimensional embedding vector for a face.
        
        Args:
            image_path: Path to the image file containing a face
            
        Returns:
            512-dim numpy array embedding, or None if no face detected
        """
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self.model,
                detector_backend=self.detector,
                enforce_detection=True
            )
            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"])
                logger.debug(f"Generated {len(embedding)}-dim embedding for {image_path}")
                return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
        return None
    
    def verify(self, img1_path: str, img2_path: str) -> Tuple[bool, float]:
        """
        1:1 Verification - Compare two faces to determine if they're the same person.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Tuple of (is_same_person, similarity_score)
        """
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model,
                detector_backend=self.detector
            )
            verified = result.get("verified", False)
            distance = result.get("distance", 1.0)
            # Convert distance to similarity (1 - distance for cosine distance)
            similarity = 1 - distance if distance <= 1 else 0
            logger.info(f"Verification result: verified={verified}, similarity={similarity:.3f}")
            return verified, similarity
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False, 0.0
    
    def find_match(
        self,
        query_embedding: np.ndarray,
        database_embeddings: List[Tuple[str, np.ndarray]]
    ) -> Optional[Tuple[str, float]]:
        """
        1:N Recognition - Find the best matching person from a database of embeddings.
        
        Args:
            query_embedding: 512-dim embedding of the query face
            database_embeddings: List of (user_id, embedding) tuples
            
        Returns:
            Tuple of (matched_user_id, similarity_score) or None if no match above threshold
        """
        if not database_embeddings:
            logger.warning("Empty database, cannot find match")
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for user_id, db_embedding in database_embeddings:
            similarity = self._cosine_similarity(query_embedding, db_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user_id
        
        if best_similarity >= self.threshold:
            logger.info(f"Match found: {best_match} with similarity {best_similarity:.3f}")
            return (best_match, best_similarity)
        
        logger.debug(f"No match found (best: {best_similarity:.3f} < threshold: {self.threshold})")
        return None
    
    def find_all_matches(
        self,
        query_embedding: np.ndarray,
        database_embeddings: List[Tuple[str, np.ndarray]],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find top-K matches from database, regardless of threshold.
        Useful for debugging and analyzing similarity scores.
        
        Args:
            query_embedding: 512-dim embedding of the query face
            database_embeddings: List of (user_id, embedding) tuples
            top_k: Number of top matches to return
            
        Returns:
            List of (user_id, similarity_score) tuples, sorted by similarity descending
        """
        scores = []
        for user_id, db_embedding in database_embeddings:
            similarity = self._cosine_similarity(query_embedding, db_embedding)
            scores.append((user_id, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def check_image_quality(self, image_path: str) -> Dict[str, Any]:
        """
        Check the quality of a face image for enrollment.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with quality metrics and recommendations
        """
        try:
            faces = self.detect_faces(image_path)
            
            if not faces:
                return {
                    "is_valid": False,
                    "error": "No face detected",
                    "recommendation": "Ensure your face is clearly visible and well-lit"
                }
            
            if len(faces) > 1:
                return {
                    "is_valid": False,
                    "error": "Multiple faces detected",
                    "recommendation": "Only one face should be in the image"
                }
            
            face = faces[0]
            confidence = face.get("confidence", 0)
            facial_area = face.get("facial_area", {})
            
            # Check face size (should be reasonably large)
            width = facial_area.get("w", 0)
            height = facial_area.get("h", 0)
            
            quality_issues = []
            
            if width < 80 or height < 80:
                quality_issues.append("Face is too small - move closer to the camera")
            
            if confidence < 0.95:
                quality_issues.append(f"Low detection confidence ({confidence:.2f}) - improve lighting")
            
            return {
                "is_valid": len(quality_issues) == 0,
                "confidence": confidence,
                "face_size": {"width": width, "height": height},
                "issues": quality_issues if quality_issues else None,
                "recommendation": quality_issues[0] if quality_issues else "Good quality for enrollment"
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "recommendation": "Unable to process image"
            }
