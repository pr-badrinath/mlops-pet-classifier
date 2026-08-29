"""Unit tests for src/data_preprocessing.py"""
import sys
from pathlib import Path

import pytest
from PIL import Image

# Allow `import data_preprocessing` when running pytest from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_preprocessing import is_valid_image, preprocess_image, stratified_split


@pytest.fixture
def sample_image(tmp_path):
    """Create a small valid RGB test image on disk."""
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 150), color=(255, 0, 0)).save(img_path)
    return img_path


@pytest.fixture
def corrupt_image(tmp_path):
    """Create a file with a .jpg extension that is not actually a valid image."""
    img_path = tmp_path / "corrupt.jpg"
    img_path.write_bytes(b"this is not a real jpeg file")
    return img_path


class TestIsValidImage:
    def test_valid_image_returns_true(self, sample_image):
        assert is_valid_image(sample_image) is True

    def test_corrupt_image_returns_false(self, corrupt_image):
        assert is_valid_image(corrupt_image) is False

    def test_empty_file_returns_false(self, tmp_path):
        empty_path = tmp_path / "empty.jpg"
        empty_path.write_bytes(b"")
        assert is_valid_image(empty_path) is False

    def test_wrong_extension_returns_false(self, tmp_path):
        txt_path = tmp_path / "not_an_image.txt"
        txt_path.write_text("hello")
        assert is_valid_image(txt_path) is False


class TestPreprocessImage:
    def test_resizes_to_requested_size(self, sample_image):
        result = preprocess_image(sample_image, output_size=224)
        assert result.size == (224, 224)

    def test_converts_to_rgb(self, sample_image):
        result = preprocess_image(sample_image, output_size=224)
        assert result.mode == "RGB"

    def test_handles_non_square_source(self, tmp_path):
        # Source is 100x150 (non-square) - output must still be square.
        img_path = tmp_path / "tall.jpg"
        Image.new("RGB", (100, 150), color=(0, 255, 0)).save(img_path)
        result = preprocess_image(img_path, output_size=64)
        assert result.size == (64, 64)


class TestStratifiedSplit:
    def test_ratios_sum_correctly(self):
        files = [f"file_{i}.jpg" for i in range(100)]
        splits = stratified_split(files, 0.8, 0.1, 0.1, seed=42)
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_no_overlap_between_splits(self):
        files = [f"file_{i}.jpg" for i in range(50)]
        splits = stratified_split(files, 0.8, 0.1, 0.1, seed=42)
        train_set, val_set, test_set = set(splits["train"]), set(splits["val"]), set(splits["test"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_all_files_are_preserved(self):
        files = [f"file_{i}.jpg" for i in range(37)]  # deliberately not evenly divisible
        splits = stratified_split(files, 0.8, 0.1, 0.1, seed=42)
        combined = splits["train"] + splits["val"] + splits["test"]
        assert sorted(combined) == sorted(files)

    def test_deterministic_with_same_seed(self):
        files = [f"file_{i}.jpg" for i in range(30)]
        splits_a = stratified_split(files, 0.8, 0.1, 0.1, seed=123)
        splits_b = stratified_split(files, 0.8, 0.1, 0.1, seed=123)
        assert splits_a == splits_b

    def test_invalid_ratios_raise(self):
        files = [f"file_{i}.jpg" for i in range(10)]
        with pytest.raises(AssertionError):
            stratified_split(files, 0.8, 0.3, 0.1, seed=42)  # sums to 1.2
