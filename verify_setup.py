"""
Verification Script - Test Google Drive and Path Configuration
Run this in Google Colab to verify your setup is correct
"""

import os
import sys

def verify_setup():
    """Verify that the Google Drive setup is correct."""
    
    print("=" * 70)
    print("FACE-BASED ATTENDANCE SYSTEM - SETUP VERIFICATION")
    print("=" * 70)
    print()
    
    # Step 1: Check if running in Colab
    print("Step 1: Checking Environment...")
    try:
        from google.colab import drive
        IN_COLAB = True
        print("✅ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("ℹ️  Running in local environment")
    print()
    
    # Step 2: Mount Google Drive (if Colab)
    if IN_COLAB:
        print("Step 2: Mounting Google Drive...")
        try:
            drive.mount('/content/drive', force_remount=True)
            print("✅ Google Drive mounted successfully")
        except Exception as e:
            print(f"❌ Failed to mount Google Drive: {e}")
            return False
        print()
    
    # Step 3: Set project root
    print("Step 3: Setting Project Root...")
    if IN_COLAB:
        PROJECT_ROOT = "/content/drive/MyDrive/face_based_attendance_system"
    else:
        from pathlib import Path
        PROJECT_ROOT = str(Path(__file__).parent.resolve())
    
    print(f"📂 Project Root: {PROJECT_ROOT}")
    print()
    
    # Step 4: Verify project root exists
    print("Step 4: Verifying Project Root...")
    if os.path.exists(PROJECT_ROOT):
        print(f"✅ Project root exists: {PROJECT_ROOT}")
    else:
        print(f"❌ Project root NOT found: {PROJECT_ROOT}")
        print("\n🔍 Checking available directories in MyDrive:")
        if IN_COLAB:
            try:
                mydrive_path = "/content/drive/MyDrive"
                items = os.listdir(mydrive_path)
                print(f"Found {len(items)} items:")
                for item in sorted(items)[:20]:  # Show first 20 items
                    print(f"   - {item}")
                if len(items) > 20:
                    print(f"   ... and {len(items) - 20} more")
            except Exception as e:
                print(f"❌ Could not list MyDrive: {e}")
        return False
    print()
    
    # Step 5: Check directory structure
    print("Step 5: Verifying Directory Structure...")
    required_dirs = {
        'backend': os.path.join(PROJECT_ROOT, 'backend'),
        'data': os.path.join(PROJECT_ROOT, 'data'),
        'models': os.path.join(PROJECT_ROOT, 'models'),
        'notebooks': os.path.join(PROJECT_ROOT, 'notebooks'),
        'docs': os.path.join(PROJECT_ROOT, 'docs')
    }
    
    all_exists = True
    for name, path in required_dirs.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name:12} -> {path}")
        if not exists:
            all_exists = False
    print()
    
    if not all_exists:
        print("⚠️  Some directories are missing. Please ensure you uploaded the complete project.")
        print()
    
    # Step 6: Check notebooks
    print("Step 6: Checking Notebooks...")
    notebooks_dir = os.path.join(PROJECT_ROOT, 'notebooks')
    if os.path.exists(notebooks_dir):
        notebooks = [
            '01_dataset_preparation.ipynb',
            '02_data_preprocessing.ipynb',
            '03_model_architecture.ipynb',
            '04_training_optimization.ipynb',
            '05_model_training.ipynb',
            '06_model_evaluation.ipynb',
            '07_model_export.ipynb'
        ]
        
        for notebook in notebooks:
            path = os.path.join(notebooks_dir, notebook)
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"{status} {notebook}")
        print()
    else:
        print("❌ Notebooks directory not found")
        print()
    
    # Step 7: Check Python environment
    print("Step 7: Checking Python Environment...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("⚠️  PyTorch not installed")
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError:
        print("⚠️  NumPy not installed")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("⚠️  OpenCV not installed")
    print()
    
    # Step 8: Final summary
    print("=" * 70)
    if all_exists and os.path.exists(PROJECT_ROOT):
        print("🎉 SUCCESS! Your setup is complete and ready to use!")
        print()
        print("Next steps:")
        print("1. Open any notebook from the notebooks/ folder")
        print("2. Run the first cell to mount Google Drive")
        print("3. Continue with the notebook exercises")
    else:
        print("⚠️  SETUP INCOMPLETE - Please fix the issues above")
        print()
        print("Common solutions:")
        print("1. Ensure you uploaded the folder to: /MyDrive/face_based_attendance_system")
        print("2. Check the folder name matches exactly (with underscores)")
        print("3. Verify all project files were uploaded completely")
    print("=" * 70)
    print()
    
    return all_exists and os.path.exists(PROJECT_ROOT)

if __name__ == "__main__":
    # Run verification
    success = verify_setup()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
