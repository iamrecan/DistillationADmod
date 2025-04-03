import os
from pathlib import Path


def validate_mvtec_structure(root_path: str) -> bool:
    """
    Validates MVTec-style dataset structure
    Returns True if valid, False otherwise
    """
    root = Path(root_path)

    # Check required directories
    train_good = root / "train" / "good"
    test_good = root / "test" / "good"

    if not train_good.exists():
        print(f"Error: Missing train/good directory at {train_good}")
        return False

    if not test_good.exists():
        print(f"Error: Missing test/good directory at {test_good}")
        return False

    # Count images
    train_images = list(train_good.glob("*.png")) + list(train_good.glob("*.jpg"))
    test_images = list(test_good.glob("*.png")) + list(test_good.glob("*.jpg"))

    print(f"Found {len(train_images)} training images")
    print(f"Found {len(test_images)} test images")

    return len(train_images) > 0 and len(test_images) > 0


if __name__ == "__main__":
    dataset_path = "H:\\Tools\\DistillationAD\\datasets\\archive\\wood"
    is_valid = validate_mvtec_structure(dataset_path)
    print(f"Dataset structure is {'valid' if is_valid else 'invalid'}")