"""Unit tests for src/model.py"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import SimpleCNN, predict


@pytest.fixture
def model():
    m = SimpleCNN(num_classes=2)
    m.eval()
    return m


class TestSimpleCNNForward:
    def test_output_shape_single_image(self, model):
        dummy = torch.randn(1, 3, 224, 224)
        output = model(dummy)
        assert output.shape == (1, 2)

    def test_output_shape_batch(self, model):
        dummy = torch.randn(8, 3, 224, 224)
        output = model(dummy)
        assert output.shape == (8, 2)

    def test_accepts_different_input_size(self, model):
        # AdaptiveAvgPool2d means the model should tolerate other spatial sizes.
        dummy = torch.randn(1, 3, 128, 128)
        output = model(dummy)
        assert output.shape == (1, 2)


class TestPredict:
    def test_returns_expected_keys(self, model):
        dummy = torch.randn(3, 224, 224)
        result = predict(model, dummy, class_names=["Cat", "Dog"])
        assert set(result.keys()) == {"label", "class_index", "probabilities"}

    def test_label_matches_class_names(self, model):
        dummy = torch.randn(3, 224, 224)
        result = predict(model, dummy, class_names=["Cat", "Dog"])
        assert result["label"] in ["Cat", "Dog"]

    def test_class_index_matches_label(self, model):
        dummy = torch.randn(3, 224, 224)
        class_names = ["Cat", "Dog"]
        result = predict(model, dummy, class_names=class_names)
        assert class_names[result["class_index"]] == result["label"]

    def test_probabilities_sum_to_one(self, model):
        dummy = torch.randn(3, 224, 224)
        result = predict(model, dummy, class_names=["Cat", "Dog"])
        assert abs(sum(result["probabilities"]) - 1.0) < 1e-4

    def test_handles_batched_input(self, model):
        # predict() should also accept an already-batched (1, 3, H, W) tensor.
        dummy = torch.randn(1, 3, 224, 224)
        result = predict(model, dummy, class_names=["Cat", "Dog"])
        assert len(result["probabilities"]) == 2
