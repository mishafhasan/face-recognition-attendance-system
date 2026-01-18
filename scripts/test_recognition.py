"""
Recognition Test Script - Test face recognition against enrolled database.

Usage:
    python test_recognition.py "path/to/test_image.jpg"
    python test_recognition.py "path/to/test_image.jpg" --verbose
"""
import json
import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.ml.face_service import FaceService


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "embeddings.json"


def load_database() -> list:
    """Load embeddings database and return as list of tuples."""
    if not DB_PATH.exists():
        return []
    
    with open(DB_PATH, 'r') as f:
        db = json.load(f)
    
    embeddings = []
    for name, data in db.items():
        if isinstance(data, dict) and "embedding" in data:
            embedding = np.array(data["embedding"])
        elif isinstance(data, list):
            # Legacy format: direct embedding list
            embedding = np.array(data)
        else:
            continue
        embeddings.append((name, embedding))
    
    return embeddings


def test_recognition(image_path: str, verbose: bool = False) -> None:
    """
    Test face recognition on an image.
    
    Args:
        image_path: Path to the test image
        verbose: Show detailed similarity scores
    """
    service = FaceService()
    
    # Check image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load database
    database = load_database()
    
    if not database:
        print("❌ No persons enrolled in database!")
        print("   Run: python enroll_person.py <name> <image_path>")
        return
    
    print(f"\n🔍 Face Recognition Test")
    print("=" * 40)
    print(f"Database contains {len(database)} enrolled person(s)")
    print(f"Test image: {image_path}")
    print("-" * 40)
    
    # Check image quality first
    quality = service.check_image_quality(image_path)
    if not quality["is_valid"]:
        print(f"⚠️  Warning: {quality.get('error', 'Quality issue detected')}")
        print(f"   {quality.get('recommendation', '')}")
    
    # Generate embedding for test image
    print("\n🔄 Generating embedding for test image...")
    embedding = service.get_embedding(image_path)
    
    if embedding is None:
        print("❌ No face detected in test image")
        return
    
    print(f"✅ Generated {len(embedding)}-dimensional embedding")
    
    # Find match
    print("\n🔍 Searching for match...")
    result = service.find_match(embedding, database)
    
    if result:
        name, confidence = result
        print(f"\n✅ RECOGNIZED: {name}")
        print(f"   Confidence: {confidence:.1%}")
        
        if confidence >= 0.7:
            print("   Status: High confidence match")
        elif confidence >= 0.6:
            print("   Status: Good match")
        else:
            print("   Status: Borderline match (consider re-enrolling)")
    else:
        print("\n❌ UNKNOWN FACE")
        print("   This person is not in the database.")
    
    # Show all matches if verbose
    if verbose:
        print("\n📊 All Similarity Scores:")
        print("-" * 40)
        all_matches = service.find_all_matches(embedding, database, top_k=len(database))
        for name, score in all_matches:
            bar = "█" * int(score * 20)
            indicator = "✓" if score >= service.threshold else " "
            print(f"  {indicator} {name}: {score:.3f} {bar}")
        print(f"\n   Threshold: {service.threshold}")
    
    print("\n👉 NEXT STEP: Once recognition works, proceed to realtime_demo.py")


def main():
    parser = argparse.ArgumentParser(description="Test face recognition")
    parser.add_argument("image_path", nargs="?", help="Path to test image")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all similarity scores")
    
    args = parser.parse_args()
    
    if not args.image_path:
        parser.print_help()
        print("\nExamples:")
        print('  python test_recognition.py "data/test_images/test_photo.jpg"')
        print('  python test_recognition.py "data/test_images/test_photo.jpg" --verbose')
        return
    
    test_recognition(args.image_path, args.verbose)


if __name__ == "__main__":
    main()
