"""
Data pre-processing for the Cats vs Dogs dataset.

Responsibilities:
    1. Validate raw images (the Kaggle Cats-vs-Dogs dataset ships with a handful
       of corrupt/truncated JPEGs - these must be filtered out or training will crash).
    2. Resize + convert every valid image to 224x224 RGB.
    3. Split into train/val/test folders (default 80/10/10) preserving class balance.

This module is intentionally split into small, pure, unit-testable functions
(`is_valid_image`, `preprocess_image`, `stratified_split`) plus a `run()` entry
point that wires them together and can be invoked as a script.
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from utils import load_params, set_seed

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def is_valid_image(image_path: str) -> bool:
    """Return True if the file at `image_path` is a readable, non-corrupt image.

    The public Cats-vs-Dogs dataset contains some 0-byte / truncated files
    (e.g. Cat/666.jpg in the original Kaggle set) which raise on load - we
    verify() first (cheap structural check) and then actually re-open + load
    the pixel data, since verify() alone misses some truncation errors.
    """
    path = Path(image_path)
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    if path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() leaves the file handle unusable, so reopen to confirm we
        # can actually decode pixel data (catches truncated-but-parseable files).
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def preprocess_image(image_path: str, output_size: int = 224) -> Image.Image:
    """Load an image, convert to RGB, and resize to (output_size, output_size).

    Pure function: given the same inputs it always returns an image of the
    same size/mode, which makes it straightforward to unit test.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((output_size, output_size), Image.BILINEAR)
        # Return a copy so the file handle can be safely closed by the `with` block.
        return img.copy()


def stratified_split(file_list: list, train_ratio: float, val_ratio: float,
                      test_ratio: float, seed: int = 42) -> dict:
    """Shuffle `file_list` deterministically and split into train/val/test.

    Returns a dict: {"train": [...], "val": [...], "test": [...]}.
    Ratios must sum to ~1.0.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1.0"

    files = list(file_list)
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def run(raw_dir: str, processed_dir: str, image_size: int, train_ratio: float,
        val_ratio: float, test_ratio: float, seed: int = 42) -> dict:
    """Full pre-processing pipeline: validate -> split -> resize -> save.

    Expects raw_dir to contain subfolders "Cat" and "Dog" (Kaggle layout).
    Writes processed_dir/{train,val,test}/{Cat,Dog}/*.jpg and returns a
    summary dict with per-split, per-class counts.
    """
    set_seed(seed)
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)

    classes = ["Cat", "Dog"]
    summary = {}

    for cls in classes:
        cls_dir = raw_path / cls
        if not cls_dir.exists():
            print(f"[WARN] class folder not found, skipping: {cls_dir}")
            continue

        all_files = [p for p in cls_dir.iterdir() if p.is_file()]
        valid_files = [p for p in all_files if is_valid_image(p)]
        skipped = len(all_files) - len(valid_files)
        if skipped:
            print(f"[INFO] {cls}: skipped {skipped} corrupt/invalid file(s) "
                  f"out of {len(all_files)}")

        splits = stratified_split(valid_files, train_ratio, val_ratio, test_ratio, seed)

        for split_name, files in splits.items():
            split_dir = out_path / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for src_file in files:
                img = preprocess_image(src_file, image_size)
                # Always save as .jpg for consistency regardless of source extension.
                dest_file = split_dir / f"{src_file.stem}.jpg"
                img.save(dest_file, format="JPEG", quality=95)
            summary.setdefault(split_name, {})[cls] = len(files)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Pre-process Cats vs Dogs dataset")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    cfg = load_params(args.params)
    data_cfg = cfg["data"]

    # Wipe any previous processed output so re-runs are deterministic and clean.
    processed_dir = Path(data_cfg["processed_dir"])
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    summary = run(
        raw_dir=data_cfg["raw_dir"],
        processed_dir=data_cfg["processed_dir"],
        image_size=data_cfg["image_size"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        seed=cfg["seed"],
    )

    print("\n=== Pre-processing summary ===")
    for split_name, class_counts in summary.items():
        total = sum(class_counts.values())
        print(f"{split_name:5s}: {class_counts} (total={total})")


if __name__ == "__main__":
    main()
