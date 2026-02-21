"""Merge individual channel CSVs into combined datasets with proper columns."""

import glob
import json
import os
from pathlib import Path
from typing import List

import pandas as pd


def get_individual_channel_files(simulation_dir: str) -> List[str]:
    """List channel CSV files in a simulation directory (excluding metadata)."""
    csv_files = []
    for file in os.listdir(simulation_dir):
        if file.endswith(".csv") and file != "signals_combined.csv":
            csv_files.append(os.path.join(simulation_dir, file))
    return sorted(csv_files)


def merge_channel_dataframes(channel_dataframes: List[pd.DataFrame], channel_names: List[str]) -> pd.DataFrame:
    """Merge per-channel DataFrames into one DataFrame keyed by ``time_s``."""
    if not channel_dataframes:
        return pd.DataFrame()

    merged_df = channel_dataframes[0][["time_ns", "time_s"]].copy()

    for df, channel_name in zip(channel_dataframes, channel_names):
        merged_df[channel_name] = df["value"]

    return merged_df


def merge_simulation_channels(simulation_dir: str, output_file: str = "signals_combined.csv") -> pd.DataFrame:
    """Merge all channel CSVs in a simulation directory into one CSV file."""
    channel_map_path = os.path.join(simulation_dir, "channel_map.json")
    if not os.path.exists(channel_map_path):
        raise FileNotFoundError(f"channel_map.json not found in {simulation_dir}")

    with open(channel_map_path) as f:
        channel_map = json.load(f)

    csv_files = get_individual_channel_files(simulation_dir)
    if not csv_files:
        raise ValueError(f"No CSV files found in {simulation_dir}")

    channel_dataframes = []
    channel_names = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        channel_name = os.path.basename(csv_file).replace(".csv", "")
        channel_dataframes.append(df)
        channel_names.append(channel_name)

    merged_df = merge_channel_dataframes(channel_dataframes, channel_names)

    output_path = os.path.join(simulation_dir, output_file)
    merged_df.to_csv(output_path, index=False)

    print(f"Created {output_path} with {len(merged_df)} rows and {len(merged_df.columns)} columns")

    return merged_df


def batch_merge_simulations(base_path: str) -> None:
    """Batch-merge simulations discovered via ``channel_map.json`` files under *base_path*."""
    search_pattern = os.path.join(base_path, "**", "channel_map.json")
    channel_map_files = glob.glob(search_pattern, recursive=True)

    for channel_map_path in channel_map_files:
        sim_dir = os.path.dirname(channel_map_path)
        output_file = "signals_combined.csv"
        try:
            merge_simulation_channels(sim_dir, output_file=output_file)
            print(f"Merged: {sim_dir} -> {os.path.join(sim_dir, output_file)}")
        except Exception as e:
            print(f"Failed to merge {sim_dir}: {e}")
