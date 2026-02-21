"""Tests for fault interval labeling logic."""

import numpy as np

from sat_anomaly.data.labeling import (
    FaultInterval,
    build_label_space,
    compute_window_time_bounds,
    label_windows_by_overlap,
)


def test_build_label_space():
    intervals = [
        FaultInterval("rw0", "coulomb_friction", 10.0, 50.0),
        FaultInterval("rw1", "bias_torque", 20.0, 60.0),
        FaultInterval("rw0", "coulomb_friction", 70.0, 80.0),
    ]
    labels = build_label_space(intervals)
    assert labels == ["rw0:coulomb_friction", "rw1:bias_torque"]


def test_compute_window_time_bounds():
    time_s = np.arange(100, dtype=float)
    bounds = compute_window_time_bounds(time_s, window_size=20, step_size=10)
    assert len(bounds) > 0
    first_start, first_end, first_idx = bounds[0]
    assert first_start == 0.0
    assert first_end == 19.0
    assert first_idx == 0


def test_label_windows_by_overlap_positive():
    time_s = np.arange(100, dtype=float)
    bounds = compute_window_time_bounds(time_s, window_size=10, step_size=10)
    intervals = [FaultInterval("rw0", "friction", 15.0, 25.0)]
    matrix, label_list = label_windows_by_overlap(bounds, intervals)
    assert label_list == ["rw0:friction"]
    # Windows starting at 10 and 20 should overlap the interval [15, 25].
    assert matrix[1, 0] == 1.0  # window [10, 19] overlaps [15, 25]
    assert matrix[2, 0] == 1.0  # window [20, 29] overlaps [15, 25]
    assert matrix[0, 0] == 0.0  # window [0, 9] does not overlap


def test_label_windows_by_overlap_no_intervals():
    time_s = np.arange(50, dtype=float)
    bounds = compute_window_time_bounds(time_s, window_size=10, step_size=10)
    matrix, label_list = label_windows_by_overlap(bounds, [])
    assert matrix.shape[1] == 0
    assert label_list == []
