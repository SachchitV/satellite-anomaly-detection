"""Label utilities to derive multi-label window targets from faults.json.

Parses per-run ``faults.json`` and converts fault time intervals into
window-level multi-label targets using simple overlap logic: a window is
positive for a label if any portion of its time span overlaps that fault
interval.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FaultInterval:
    subsystem: str
    fault_type: str
    start_time_s: float
    end_time_s: float


def load_fault_intervals(faults_json_path: str) -> List[FaultInterval]:
    """Load fault intervals from a ``faults.json`` file."""
    if not os.path.exists(faults_json_path):
        return []

    with open(faults_json_path) as f:
        data = json.load(f)

    intervals: List[FaultInterval] = []

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("faults", data.get("fault_annotations", []))
    else:
        entries = []

    for entry in entries:
        try:
            subsystem = entry.get("subsystem") or entry.get("component") or entry.get("system")
            fault_type = entry.get("type") or entry.get("fault_type") or entry.get("label")

            start_val = (
                entry.get("start_time_s")
                or entry.get("start_time")
                or entry.get("start_s")
                or entry.get("start")
                or entry.get("t_start")
            )
            end_val = (
                entry.get("end_time_s")
                or entry.get("end_time")
                or entry.get("end_s")
                or entry.get("end")
                or entry.get("t_end")
                or start_val
            )

            if subsystem is None or fault_type is None or start_val is None:
                continue

            start_time_s = float(start_val)
            end_time_s = float(end_val)

            if end_time_s < start_time_s:
                start_time_s, end_time_s = end_time_s, start_time_s

            intervals.append(
                FaultInterval(
                    subsystem=str(subsystem),
                    fault_type=str(fault_type),
                    start_time_s=start_time_s,
                    end_time_s=end_time_s,
                )
            )
        except Exception:
            continue

    return intervals


def build_label_space(intervals: List[FaultInterval]) -> List[str]:
    """Build a deterministic sorted label list like ``["rw0:coulomb_friction", ...]``."""
    labels = sorted({f"{iv.subsystem}:{iv.fault_type}" for iv in intervals})
    return labels


def compute_window_time_bounds(
    time_s: np.ndarray, window_size: int, step_size: int
) -> List[Tuple[float, float, int]]:
    """Precompute window ``[start_time, end_time]`` pairs and starting row index."""
    bounds: List[Tuple[float, float, int]] = []
    last_start = len(time_s) - window_size
    if last_start < 0:
        return bounds
    for start_idx in range(0, last_start + 1, step_size):
        end_idx = start_idx + window_size - 1
        bounds.append((float(time_s[start_idx]), float(time_s[end_idx]), start_idx))
    return bounds


def label_windows_by_overlap(
    window_bounds: List[Tuple[float, float, int]],
    intervals: List[FaultInterval],
    label_list: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Create a multi-hot label matrix for windows using interval overlap.

    Returns ``(labels_matrix, label_list)`` where ``labels_matrix`` has shape
    ``[num_windows, num_labels]``.
    """
    if label_list is None:
        label_list = build_label_space(intervals)

    num_windows = len(window_bounds)
    num_labels = len(label_list)
    labels_matrix = np.zeros((num_windows, num_labels), dtype=np.float32)

    label_to_index = {label: idx for idx, label in enumerate(label_list)}

    label_to_intervals: Dict[str, List[Tuple[float, float]]] = {}
    for iv in intervals:
        key = f"{iv.subsystem}:{iv.fault_type}"
        label_to_intervals.setdefault(key, []).append((iv.start_time_s, iv.end_time_s))

    for w_idx, (w_start, w_end, _) in enumerate(window_bounds):
        for label, ranges in label_to_intervals.items():
            label_col = label_to_index[label]
            for r_start, r_end in ranges:
                if (w_start <= r_end) and (r_start <= w_end):
                    labels_matrix[w_idx, label_col] = 1.0
                    break

    return labels_matrix, label_list


def generate_window_labels_for_run(
    df: pd.DataFrame,
    faults_json_path: str,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Convenience wrapper: compute window bounds and labels for a single run.

    Returns ``(labels_matrix, label_list, start_indices)``.
    """
    if "time_s" not in df.columns:
        raise ValueError("Dataframe must contain 'time_s' column")

    time_s = df["time_s"].to_numpy()
    bounds = compute_window_time_bounds(time_s, window_size, step_size)
    intervals = load_fault_intervals(faults_json_path)
    labels, label_list = label_windows_by_overlap(bounds, intervals)
    start_indices = [start_idx for (_, _, start_idx) in bounds]
    return labels, label_list, start_indices
