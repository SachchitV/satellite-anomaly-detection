"""CSV loading, labeling, and PyTorch dataset utilities."""

import glob
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def load_fault_free_data(base_path):
    """Load all fault-free data from directory structure."""
    print(f"Loading fault-free data from {base_path}\n")

    all_data = []
    day_dirs = sorted(glob.glob(os.path.join(base_path, "day_*")))

    for day_dir in day_dirs:
        none_dir = os.path.join(day_dir, "none")
        if os.path.exists(none_dir):
            modes = ["inertial", "nadir", "sun"]
            for mode in modes:
                csv_path = os.path.join(none_dir, mode, "signals_combined.csv")
                if os.path.exists(csv_path):
                    print(f"Loading {csv_path}")
                    df = pd.read_csv(csv_path)
                    df["sequence_id"] = f"{os.path.basename(day_dir)}_{mode}"
                    all_data.append(df)

    if len(all_data) > 0:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nLoaded {len(combined_df)} rows from {len(all_data)} files \n")
        return combined_df
    else:
        print("No fault-free data found")
        return pd.DataFrame()


def get_data_statistics(df):
    """Generate data statistics."""
    print("Data Statistics:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df.describe()


def _assign_fault_labels_by_intervals(df, intervals, default_label):
    """Assign per-row fault label using fault interval fields."""
    if not intervals:
        return [default_label] * len(df)
    times = df["time_s"].values
    labels = [default_label] * len(df)
    for interval in intervals:
        start = interval["start_time"]
        end = interval["end_time"]
        comp = interval["component"]
        ftype = interval["type"]
        name = f"{comp}:{ftype}"
        mask = (times >= start) & (times <= end)
        for idx, m in enumerate(mask):
            if m:
                labels[idx] = name
    return labels


def load_fault_data_with_annotations(base_path):
    """Load data that contains faults and assign multi-class labels from faults.json.

    Returns combined dataframe with 'fault_label_name' and 'sequence_id' columns,
    and a label name-to-id map.
    """
    print(f"Loading fault data from {base_path}\n")

    all_data = []
    label_name_to_id = {}
    day_dirs = sorted(glob.glob(os.path.join(base_path, "day_*")))

    for day_dir in day_dirs:
        for fault_dir in sorted(os.listdir(day_dir)):
            if fault_dir == "none":
                continue
            candidate = os.path.join(day_dir, fault_dir)
            if not os.path.isdir(candidate):
                continue
            for mode in ["inertial", "nadir", "sun"]:
                mode_dir = os.path.join(candidate, mode)
                csv_path = os.path.join(mode_dir, "signals_combined.csv")
                if not os.path.exists(csv_path):
                    continue
                print(f"Loading {csv_path}")
                df = pd.read_csv(csv_path)
                df["sequence_id"] = f"{os.path.basename(day_dir)}_{fault_dir}_{mode}"

                json_path = os.path.join(mode_dir, "faults.json")
                with open(json_path) as f:
                    intervals = json.load(f)
                default_label = "none"
                labels = _assign_fault_labels_by_intervals(df, intervals, default_label)
                df["fault_label_name"] = labels

                all_data.append(df)

                if "none" not in label_name_to_id:
                    label_name_to_id["none"] = len(label_name_to_id)
                for iv in intervals:
                    name = iv.get("fault_name", iv.get("type", default_label))
                    comp = iv.get("component")
                    ftype = iv.get("type", name)
                    composed = f"{comp}:{ftype}" if comp and ftype else ftype
                    if composed not in label_name_to_id:
                        label_name_to_id[composed] = len(label_name_to_id)

    if len(all_data) == 0:
        print("No fault data found")
        return pd.DataFrame(), {}

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nLoaded {len(combined)} rows from {len(all_data)} files with {len(label_name_to_id)} labels\n")
    return combined, label_name_to_id


class AutoencoderDataset(Dataset):
    """PyTorch dataset for autoencoder training."""

    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_data_loaders(train_data, val_data, batch_size):
    """Create data loaders for training and validation."""
    print(f"Creating data loaders with batch_size={batch_size}")

    train_dataset = AutoencoderDataset(train_data)
    val_dataset = AutoencoderDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    return train_loader, val_loader


class LabeledWindowDataset(Dataset):
    """PyTorch dataset for labeled window classification."""

    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_labeled_data_loaders(train_data, train_labels, val_data, val_labels, batch_size):
    """Create data loaders for labeled classification datasets."""
    print(f"Creating labeled data loaders with batch_size={batch_size}")

    train_dataset = LabeledWindowDataset(train_data, train_labels)
    val_dataset = LabeledWindowDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Labeled Train loader: {len(train_loader)} batches")
    print(f"Labeled Val loader: {len(val_loader)} batches")

    return train_loader, val_loader
