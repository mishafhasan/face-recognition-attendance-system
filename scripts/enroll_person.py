"""
Enrollment Script - Add new persons to the face recognition database.

Usage:
    python enroll_person.py "John Doe" "path/to/image.jpg"
    python enroll_person.py "Jane Smith" "path/to/jane.jpg" --multiple
"""
import json
import os
import sys
import argparse
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.ml.face_service import FaceService


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "embeddings.json"
ENROLLMENTS_PATH = PROJECT_ROOT / "data" / "enrollments"


def load_database() -> dict:
    """Load existing embeddings database."""
    if DB_PATH.exists():
        with open(DB_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_database(db: dict) -> None:
    """Save embeddings database to file."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, 'w') as f:
        json.dump(db, f, indent=2)


def enroll_person(name: str, image_path: str, service: FaceService = None) -> bool:
    """
    Enroll a person with their face image.
    
    Args:
        name: Person's name (unique identifier)
        image_path: Path to face image
        service: FaceService instance (creates new one if not provided)
        
    Returns:
        True if enrollment successful, False otherwise
    """
    if service is None:
        service = FaceService()
    
    # Check image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Check image quality
    print(f"📷 Checking image quality for {name}...")
    quality = service.check_image_quality(image_path)
    
    if not quality["is_valid"]:
        print(f"❌ Image quality check failed: {quality.get('error', 'Unknown error')}")
        print(f"   Recommendation: {quality.get('recommendation', 'Try another image')}")
        return False
    
    print(f"✅ Image quality OK (confidence: {quality.get('confidence', 0):.2f})")
    
    # Generate embedding
    print(f"🔄 Generating face embedding...")
    embedding = service.get_embedding(image_path)
    
    if embedding is None:
        print(f"❌ Failed to generate embedding for {name}")
        return False
    
    print(f"✅ Generated {len(embedding)}-dimensional embedding")
    
    # Load database and add entry
    db = load_database()
    
    if name in db:
        print(f"⚠️  {name} already exists in database. Updating...")
    
    db[name] = {
        "embedding": embedding.tolist(),
        "source_image": os.path.basename(image_path),
        "enrolled_at": __import__("datetime").datetime.now().isoformat()
    }
    
    # Save database
    save_database(db)
    print(f"✅ Successfully enrolled {name}")
    print(f"   Database now contains {len(db)} person(s)")
    
    return True


def list_enrolled() -> None:
    """List all enrolled persons."""
    db = load_database()
    
    if not db:
        print("📭 No persons enrolled yet")
        return
    
    print(f"\n📋 Enrolled Persons ({len(db)} total):")
    print("-" * 40)
    for name, data in db.items():
        enrolled_at = data.get("enrolled_at", "Unknown")[:10]
        source = data.get("source_image", "Unknown")
        print(f"  • {name} (enrolled: {enrolled_at}, source: {source})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Enroll a person for face recognition")
    parser.add_argument("name", nargs="?", help="Person's name")
    parser.add_argument("image_path", nargs="?", help="Path to face image")
    parser.add_argument("--list", "-l", action="store_true", help="List enrolled persons")
    
    args = parser.parse_args()
    
    if args.list:
        list_enrolled()
        return
    
    if not args.name or not args.image_path:
        parser.print_help()
        print("\nExamples:")
        print('  python enroll_person.py "John Doe" "data/test_images/john.jpg"')
        print("  python enroll_person.py --list")
        return
    
    print(f"\n🎯 Face Enrollment System")
    print("=" * 40)
    
    service = FaceService()
    success = enroll_person(args.name, args.image_path, service)
    
    if success:
        print("\n👉 NEXT STEP: Run test_recognition.py to verify enrollment")
    else:
        print("\n❌ Enrollment failed. Check the errors above.")


if __name__ == "__main__":
    main()
