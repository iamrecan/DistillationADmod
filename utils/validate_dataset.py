import os
from pathlib import Path


def validate_mvtec_dataset(root_path: str) -> bool:
    """Validates MVTec AD dataset structure"""

    root = Path(root_path)
    print(f"\nValidating dataset at: {root}")

    # Required structure
    train_good = root / "train" / "good"
    test_good = root / "test" / "good"

    # Check directories exist
    if not train_good.exists():
        print(f"❌ Missing training directory: {train_good}")
        return False

    # Count images
    train_images = list(train_good.glob("*.png")) + list(train_good.glob("*.jpg"))
    test_images = list(test_good.glob("*.png")) + list(test_good.glob("*.jpg"))

    print("\nDirectory structure:")
    print(f"{'=' * 20}")
    print(f"train/good/: {len(train_images)} images")
    print(f"test/good/: {len(test_images)} images")

    return len(train_images) > 0


if __name__ == "__main__":
    dataset_path = "H:\\Tools\\DistillationAD\\datasets\\archive\\wood"
    is_valid = validate_mvtec_dataset(dataset_path)
    print(f"\nDataset is {'✅ valid' if is_valid else '❌ invalid'}")