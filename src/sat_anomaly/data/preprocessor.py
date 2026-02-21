"""Data preprocessing utilities for normalization, windowing, and splits."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_features(df):
    """Normalize features using MinMax scaling (0-1)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ["time_ns", "time_s", "label_any_fault"]]

    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df_normalized, scaler


def create_grouped_time_windows(df, group_cols, window_size, step_size):
    """Create sliding windows per group, avoiding cross-boundary windows.

    Returns windows array of shape ``(N, window_size, n_features)``.
    """
    print(f"Creating grouped time windows by {group_cols}: window_size={window_size}, step_size={step_size}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ["time_ns", "time_s", "label_any_fault"]]

    all_windows = []

    if len(group_cols) > 0:
        grouped = df.groupby(group_cols, sort=False)
    else:
        grouped = df.groupby(sort=False)

    print(f"Grouped data by {group_cols}: {len(grouped)} groups")

    for _, group_df in grouped:
        group_values = group_df[feature_cols].values

        starts = list(range(0, len(group_values) - window_size + 1, step_size))

        if len(group_values) >= window_size:
            last_start = len(group_values) - window_size
            if not starts or starts[-1] != last_start:
                starts.append(last_start)

        for i in starts:
            window = group_values[i : i + window_size]
            all_windows.append(window)

    windows_array = np.array(all_windows) if all_windows else np.empty((0, window_size, len(feature_cols)))

    print(f"Created {len(all_windows)} grouped windows of shape {windows_array.shape}")

    return windows_array


def create_grouped_time_windows_with_labels(df, group_cols, window_size, step_size, label_col="fault_label_name"):
    """Create sliding windows per group along with window labels.

    Label for a window is the majority label within the window.
    Returns ``(windows_array, labels_array)``.
    """
    print(f"Creating labeled grouped time windows by {group_cols}: window_size={window_size}, step_size={step_size}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ["time_ns", "time_s", "label_any_fault"]]

    all_windows = []
    all_labels = []

    if len(group_cols) > 0:
        grouped = df.groupby(group_cols, sort=False)
    else:
        grouped = df.groupby(sort=False)

    print(f"Grouped data by {group_cols}: {len(grouped)} groups")

    for _, group_df in grouped:
        feature_values = group_df[feature_cols].values
        if label_col in group_df.columns:
            label_values = group_df[label_col].values
        else:
            label_values = np.array(["none"] * len(group_df))

        starts = list(range(0, len(feature_values) - window_size + 1, step_size))
        if len(feature_values) >= window_size:
            last_start = len(feature_values) - window_size
            if not starts or starts[-1] != last_start:
                starts.append(last_start)

        for i in starts:
            window = feature_values[i : i + window_size]
            window_labels = label_values[i : i + window_size]

            unique, counts = np.unique(window_labels, return_counts=True)
            label = unique[np.argmax(counts)]

            all_windows.append(window)
            all_labels.append(label)

    windows_array = np.array(all_windows) if all_windows else np.empty((0, window_size, len(feature_cols)))
    labels_array = np.array(all_labels, dtype=object) if all_labels else np.empty((0,), dtype=object)

    print(f"Created {len(all_windows)} labeled windows of shape {windows_array.shape}")
    return windows_array, labels_array


def split_sequences(sequences, train_ratio=0.8):
    """Split data into train/validation sets."""
    print(f"Splitting sequences: train_ratio={train_ratio}")

    n_samples = len(sequences)
    n_train = int(n_samples * train_ratio)

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_sequences = sequences[train_indices]
    val_sequences = sequences[val_indices]

    print(f"Train: {len(train_sequences)} samples, Val: {len(val_sequences)} samples")

    return train_sequences, val_sequences


def split_sequences_with_labels(sequences, labels, train_ratio=0.8):
    """Split sequences and corresponding labels into train/validation sets.

    Returns ``(train_sequences, val_sequences, train_labels, val_labels)``.
    """
    print(f"Splitting sequences with labels: train_ratio={train_ratio}")

    n_samples = len(sequences)
    n_train = int(n_samples * train_ratio)

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_sequences = sequences[train_indices]
    val_sequences = sequences[val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    print(f"Train: {len(train_sequences)} samples, Val: {len(val_sequences)} samples")

    return train_sequences, val_sequences, train_labels, val_labels
