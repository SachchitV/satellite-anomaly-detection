"""Tests for data preprocessing (normalization, windowing, splitting)."""

import numpy as np
import pandas as pd

from sat_anomaly.data.preprocessor import (
    create_grouped_time_windows,
    normalize_features,
    split_sequences,
    split_sequences_with_labels,
)


def _make_df(n_rows=500):
    """Create a synthetic DataFrame resembling satellite telemetry."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "time_ns": np.arange(n_rows) * 1_000_000_000,
            "time_s": np.arange(n_rows, dtype=float),
            "feature_a": rng.standard_normal(n_rows) * 100,
            "feature_b": rng.standard_normal(n_rows) * 50 + 200,
            "sequence_id": ["seq_0"] * n_rows,
        }
    )
    return df


def test_normalize_features_range():
    df = _make_df()
    normed, scaler = normalize_features(df)
    for col in ["feature_a", "feature_b"]:
        assert normed[col].min() >= -1e-9
        assert normed[col].max() <= 1.0 + 1e-9


def test_create_grouped_time_windows_shape():
    df = _make_df(n_rows=600)
    windows = create_grouped_time_windows(df, group_cols=["sequence_id"], window_size=64, step_size=32)
    assert windows.ndim == 3
    assert windows.shape[1] == 64
    assert windows.shape[2] == 2  # feature_a, feature_b


def test_split_sequences_ratio():
    arr = np.random.randn(100, 64, 2)
    train, val = split_sequences(arr, train_ratio=0.8)
    assert len(train) == 80
    assert len(val) == 20


def test_split_sequences_with_labels():
    arr = np.random.randn(100, 64, 2)
    labels = np.array(["a"] * 50 + ["b"] * 50)
    train_s, val_s, train_l, val_l = split_sequences_with_labels(arr, labels, train_ratio=0.7)
    assert len(train_s) == 70
    assert len(val_l) == 30
