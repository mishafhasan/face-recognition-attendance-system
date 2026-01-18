"""
Real-Time Face Recognition Demo - Webcam-based attendance system demonstration.

This script captures webcam frames, runs face recognition in real-time,
and logs attendance events to a CSV file.

Usage:
    python realtime_demo.py
    python realtime_demo.py --camera 1  # Use different camera
    python realtime_demo.py --threshold 0.55  # Adjust threshold

Controls:
    q - Quit
    s - Save current frame
    r - Reset last result
"""
import cv2
import json
import os
import sys
import argparse
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.ml.face_service import FaceService


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "embeddings.json"
ATTENDANCE_LOG = PROJECT_ROOT / "data" / "attendance.csv"


def load_database() -> list:
    """Load embeddings database."""
    if not DB_PATH.exists():
        return []
    
    with open(DB_PATH, 'r') as f:
        db = json.load(f)
    
    embeddings = []
    for name, data in db.items():
        if isinstance(data, dict) and "embedding" in data:
            embedding = np.array(data["embedding"])
        elif isinstance(data, list):
            embedding = np.array(data)
        else:
            continue
        embeddings.append((name, embedding))
    
    return embeddings


def init_attendance_log():
    """Initialize attendance log with headers if it doesn't exist."""
    if not ATTENDANCE_LOG.exists():
        ATTENDANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ATTENDANCE_LOG, 'w') as f:
            f.write("timestamp,name,confidence,event_type\n")


def log_attendance(name: str, confidence: float, event_type: str = "check_in"):
    """Log an attendance event to CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_LOG, 'a') as f:
        f.write(f"{timestamp},{name},{confidence:.3f},{event_type}\n")
    print(f"📋 Logged: {name} at {timestamp} ({event_type})")


def draw_result(frame, result, processing: bool = False):
    """Draw recognition result on frame."""
    h, w = frame.shape[:2]
    
    # Draw status bar at top
    cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
    
    if processing:
        cv2.putText(frame, "Processing...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    elif result:
        name, confidence = result
        color = (0, 255, 0) if confidence >= 0.65 else (0, 255, 255)
        text = f"{name} ({confidence:.0%})"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    else:
        cv2.putText(frame, "Unknown / No Face", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Draw controls hint
    cv2.putText(frame, "Press 'q' to quit | 's' to screenshot", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time face recognition demo")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", "-t", type=float, default=0.60, help="Recognition threshold")
    parser.add_argument("--skip-frames", "-s", type=int, default=5, help="Process every N frames")
    parser.add_argument("--cooldown", type=int, default=60, help="Seconds between same-person logs")
    
    args = parser.parse_args()
    
    print("\n🎯 Real-Time Face Recognition Demo")
    print("=" * 50)
    
    # Initialize
    print("🔄 Loading face recognition models...")
    service = FaceService(similarity_threshold=args.threshold)
    
    database = load_database()
    print(f"📚 Loaded {len(database)} enrolled person(s)")
    
    if not database:
        print("\n⚠️  No persons enrolled! Recognition will always show 'Unknown'")
        print("   Run: python enroll_person.py <name> <image_path>")
    
    init_attendance_log()
    
    # Open webcam
    print(f"📷 Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {args.camera}")
        print("   Try: python realtime_demo.py --camera 1")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera opened successfully!")
    print("-" * 50)
    print("Controls: q=quit, s=screenshot")
    print("-" * 50)
    
    frame_count = 0
    last_result = None
    last_logged = {}  # Track when each person was last logged
    is_processing = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Process every N frames
        if frame_count % args.skip_frames == 0 and database:
            is_processing = True
            
            # Save temp frame for DeepFace
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                temp_path = tmp.name
            
            try:
                # Generate embedding
                embedding = service.get_embedding(temp_path)
                
                if embedding is not None:
                    # Find match
                    match = service.find_match(embedding, database)
                    last_result = match
                    
                    # Log attendance if match found
                    if match:
                        name, conf = match
                        now = datetime.now()
                        
                        # Only log if cooldown has passed
                        if name not in last_logged or \
                           (now - last_logged[name]).seconds >= args.cooldown:
                            log_attendance(name, conf)
                            last_logged[name] = now
                else:
                    last_result = None
                    
            except Exception as e:
                print(f"⚠️  Recognition error: {e}")
            finally:
                os.unlink(temp_path)
                is_processing = False
        
        # Draw result on frame
        display_frame = draw_result(display_frame, last_result, is_processing)
        
        # Show frame
        cv2.imshow("Face Attendance System - Demo", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"data/test_images/screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved: {filename}")
        elif key == ord('r'):
            last_result = None
            print("🔄 Result reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("📊 Session Summary:")
    print(f"   Frames processed: {frame_count}")
    print(f"   Attendance log: {ATTENDANCE_LOG}")
    print("\n👉 NEXT STEP: Record a demo video for your portfolio!")


if __name__ == "__main__":
    main()
