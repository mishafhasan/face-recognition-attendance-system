"""
Webcam Test Script - Verify webcam access for the Face Attendance System.

Run this script to test if your webcam is working correctly.
Press 'q' to quit.
"""
import cv2
import sys


def test_webcam(camera_index: int = 0):
    """Test webcam access and display live feed."""
    print(f"Attempting to open webcam at index {camera_index}...")
    
    # Try DirectShow backend on Windows for better compatibility
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"❌ Failed to open webcam at index {camera_index}")
        print("   Try running with index 1: python test_webcam.py 1")
        return False
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Webcam opened successfully!")
    print("   Press 'q' to quit")
    print("   Press 's' to save a screenshot")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam")
            break
        
        frame_count += 1
        
        # Add frame counter and instructions
        cv2.putText(
            frame, 
            f"Frame: {frame_count} | Press 'q' to quit, 's' to save", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )
        
        cv2.imshow("Webcam Test - Face Attendance System", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"data/test_images/webcam_capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved screenshot to {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"📹 Webcam test completed. Total frames captured: {frame_count}")
    return True


if __name__ == "__main__":
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_webcam(camera_index)
