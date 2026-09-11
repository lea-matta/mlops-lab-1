from pathlib import Path
from PIL import Image
import shutil


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100

SPLITS = ["training", "evaluation", "validation"]

CLASS_NAMES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def prepare_output_folder(folder: Path) -> None:
    """
    Remove an existing processed folder and recreate it.
    This makes the script safe to run multiple times.
    """
    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Main processing
# --------------------------------------------------

def process_dataset() -> None:
    prepare_output_folder(PROCESSED_DIR)
    prepare_output_folder(MINI_DIR)

    total_processed = 0
    total_mini = 0

    for split in SPLITS:
        raw_split = RAW_DIR / split

        # Keep track of how many images were added to the
        # mini dataset for each category.
        mini_counts = {class_id: 0 for class_id in CLASS_NAMES}

        for image_path in sorted(raw_split.glob("*.jpg")):

            # Example filename:
            # 0_123.jpg
            # ↑
            # class ID = 0
            class_id = int(image_path.stem.split("_")[0])
            class_name = CLASS_NAMES[class_id]

            # Create:
            # data/food11_processed/training/Bread/
            processed_class_dir = (
                PROCESSED_DIR / split / class_name
            )

            # Create:
            # data/food11_processed_mini/training/Bread/
            mini_class_dir = (
                MINI_DIR / split / class_name
            )

            processed_class_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            mini_class_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Open and resize image
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                # Save full processed version
                processed_path = (
                    processed_class_dir / image_path.name
                )
                image.save(processed_path)

                total_processed += 1

                # Save at most 100 images per category
                # for the mini development dataset
                if mini_counts[class_id] < MINI_LIMIT:
                    mini_path = (
                        mini_class_dir / image_path.name
                    )
                    image.save(mini_path)

                    mini_counts[class_id] += 1
                    total_mini += 1

        print(f"Finished {split}")

    print()
    print("Data preparation complete.")
    print(f"Processed images: {total_processed}")
    print(f"Mini images: {total_mini}")


if __name__ == "__main__":
    process_dataset()