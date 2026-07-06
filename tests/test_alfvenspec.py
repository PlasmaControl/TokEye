from __future__ import annotations

import csv

import numpy as np
import torch
import torch.nn as nn

from tokeye.alfvenspec import detect, detect_windowed, write_detections_csv


class _StubRCNN(nn.Module):
    """Returns canned torchvision-style detections; records its input."""

    def __init__(self, n: int = 3, height: int = 8, width: int = 6):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.seen = None
        self.canned = {
            "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]] * n),
            "labels": torch.ones(n, dtype=torch.int64),
            "scores": torch.tensor([0.9, 0.6, 0.2][:n]),
            "masks": torch.zeros(n, 1, height, width),
        }

    def forward(self, images):
        self.seen = images
        return [self.canned]


def test_detect_filters_by_score_and_returns_numpy():
    model = _StubRCNN()
    spectrogram = np.random.default_rng(0).normal(size=(8, 6)).astype(np.float32)

    result = detect(spectrogram, model, score_min=0.5)

    assert isinstance(result["boxes"], np.ndarray)
    assert result["boxes"].shape == (2, 4)  # score 0.2 filtered out
    assert result["scores"].shape == (2,)
    assert result["labels"].shape == (2,)
    assert result["masks"].shape == (2, 8, 6)  # channel dim squeezed


def test_detect_feeds_single_channel_standardized_image():
    model = _StubRCNN()
    spectrogram = (np.random.default_rng(1).normal(size=(8, 6)) * 5 + 50).astype(
        np.float32
    )

    detect(spectrogram, model)

    (img,) = model.seen
    assert img.shape == (1, 8, 6)
    assert abs(float(img.mean())) < 1e-4  # standardized per-sample
    # torch .std() is ddof=1 vs numpy's ddof=0 used for standardization
    assert abs(float(img.std()) - 1.0) < 2e-2


def test_detect_honors_explicit_mean_std():
    model = _StubRCNN()
    spectrogram = np.full((8, 6), 10.0, dtype=np.float32)

    detect(spectrogram, model, mean=8.0, std=2.0)

    (img,) = model.seen
    assert torch.allclose(img, torch.ones_like(img))


def test_detect_windowed_offsets_boxes_to_global_columns():
    model = _StubRCNN(n=1)  # one detection (score 0.9) per window call
    spectrogram = np.zeros((8, 112), dtype=np.float32)

    result = detect_windowed(spectrogram, model, window_cols=40)

    # windows: [0:40], [40:80], [80:112] -> 3 detections
    assert result["boxes"].shape == (3, 4)
    np.testing.assert_allclose(result["boxes"][:, 0], [0.0, 40.0, 80.0])
    np.testing.assert_allclose(result["boxes"][:, 2], [2.0, 42.0, 82.0])
    assert result["masks"] is None


def test_detect_windowed_folds_sliver_into_previous_window():
    model = _StubRCNN(n=1)
    spectrogram = np.zeros((8, 100), dtype=np.float32)

    result = detect_windowed(spectrogram, model, window_cols=40)

    # final 20-column sliver folds into [40:100] instead of being dropped
    assert result["boxes"].shape == (2, 4)
    np.testing.assert_allclose(result["boxes"][:, 0], [0.0, 40.0])


def test_detect_windowed_falls_back_to_single_window():
    model = _StubRCNN(n=1)
    spectrogram = np.zeros((8, 30), dtype=np.float32)

    result = detect_windowed(spectrogram, model, window_cols=40)

    assert result["boxes"].shape == (1, 4)
    assert result["masks"] is not None  # single window keeps masks


def test_write_detections_csv(tmp_path):
    out = tmp_path / "ae_detections.csv"
    detections = {
        "boxes": np.array([[1.0, 2.0, 3.0, 4.0]]),
        "labels": np.array([1]),
        "scores": np.array([0.9]),
        "masks": np.zeros((1, 8, 6)),
    }
    write_detections_csv(out, [("shot1.npy", detections)])

    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["input"] == "shot1.npy"
    assert rows[0]["detection"] == "0"
    assert [rows[0][k] for k in ("x1", "y1", "x2", "y2")] == ["1.0", "2.0", "3.0", "4.0"]
    assert rows[0]["score"] == "0.9"
