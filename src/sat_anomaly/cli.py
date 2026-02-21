"""CLI entry point for satellite anomaly detection pipeline.

Usage::

    sat-anomaly train-ae  --config configs/autoencoder.yaml
    sat-anomaly train-cls --config configs/classifier.yaml
    sat-anomaly merge-data --data-path data/raw/main_data/simulations_year
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def _train_ae(args):
    from sat_anomaly.config import load_config
    from sat_anomaly.data.loader import create_data_loaders, load_fault_free_data
    from sat_anomaly.data.preprocessor import create_grouped_time_windows, normalize_features, split_sequences
    from sat_anomaly.models.autoencoder import create_lstm_autoencoder, create_rnn_autoencoder
    from sat_anomaly.models.training import train_autoencoder
    from sat_anomaly.visualization.plots import plot_training_history

    config = load_config(args.config)
    print("Starting autoencoder training...")

    data = load_fault_free_data(config["data_path"])
    normalized_data, _scaler = normalize_features(data)

    _numeric_cols = normalized_data.select_dtypes(include=["number"]).columns
    feature_names = [c for c in _numeric_cols if c not in ["time_ns", "time_s", "label_any_fault"]]
    config["feature_names"] = feature_names

    windows = create_grouped_time_windows(
        normalized_data,
        group_cols=["sequence_id"],
        window_size=config["window_size"],
        step_size=config["step_size"],
    )
    train_data, val_data = split_sequences(windows, train_ratio=config.get("train_ratio", 0.8))
    train_loader, val_loader = create_data_loaders(train_data, val_data, config["batch_size"])

    model_type = config.get("model_type", "lstm_ae")
    if model_type == "lstm_ae":
        model = create_lstm_autoencoder(config)
    elif model_type == "rnn_ae":
        model = create_rnn_autoencoder(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    train_results = train_autoencoder(model, train_loader, val_loader, config)
    print("Training completed!")

    if not args.no_plot:
        plot_training_history(train_results["train_losses"], train_results["val_losses"])


def _train_cls(args):
    from sat_anomaly.config import load_config
    from sat_anomaly.data.loader import create_labeled_data_loaders, load_fault_data_with_annotations
    from sat_anomaly.data.preprocessor import (
        create_grouped_time_windows_with_labels,
        normalize_features,
        split_sequences_with_labels,
    )
    from sat_anomaly.models.classifier import CNNClassifier
    from sat_anomaly.models.training import (
        compute_residual_windows_batched,
        load_pretrained_autoencoder,
        train_classifier,
    )
    from sat_anomaly.visualization.plots import plot_training_history

    config = load_config(args.config)
    print("Starting classifier training...")

    data, label_map = load_fault_data_with_annotations(config["data_path"])

    normalized_data, _scaler = normalize_features(data)

    _numeric_cols = normalized_data.select_dtypes(include=["number"]).columns
    feature_names = [c for c in _numeric_cols if c not in ["time_ns", "time_s", "label_any_fault"]]
    config["feature_names"] = feature_names

    windows, labels = create_grouped_time_windows_with_labels(
        normalized_data,
        group_cols=["sequence_id"],
        window_size=config["window_size"],
        step_size=config["step_size"],
        label_col="fault_label_name",
    )

    train_windows, val_windows, train_labels, val_labels = split_sequences_with_labels(
        windows, labels, train_ratio=config.get("train_ratio", 0.8)
    )

    observed = sorted(set(train_labels.tolist() + val_labels.tolist()))
    label_to_id = {name: idx for idx, name in enumerate(observed)}
    config["n_classes"] = len(label_to_id)

    train_id_labels = np.array([label_to_id[x] for x in train_labels], dtype=np.int64)
    val_id_labels = np.array([label_to_id[x] for x in val_labels], dtype=np.int64)

    ae_model = load_pretrained_autoencoder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = int(config.get("residual_batch_size", max(16, config.get("batch_size", 64))))

    print("Computing residual windows...")
    train_windows = compute_residual_windows_batched(ae_model, train_windows, device=device, batch_size=bs)
    val_windows = compute_residual_windows_batched(ae_model, val_windows, device=device, batch_size=bs)

    train_loader, val_loader = create_labeled_data_loaders(
        train_windows, train_id_labels, val_windows, val_id_labels, config["batch_size"]
    )

    model = CNNClassifier(
        n_features=config["n_features"],
        n_classes=config["n_classes"],
        channels=config.get("cnn_channels", [64, 128, 256]),
        dropout=config.get("dropout", 0.2),
    )
    model = model.to(device)
    print(f"Using device: {device}")

    train_results = train_classifier(model, train_loader, val_loader, config)
    print("Training completed!")

    if not args.no_plot:
        plot_training_history(train_results["train_losses"], train_results["val_losses"])


def _merge_data(args):
    from sat_anomaly.data.merge_channels import batch_merge_simulations

    print(f"Merging channel data under: {args.data_path}")
    batch_merge_simulations(args.data_path)
    print("Done.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sat-anomaly", description="Satellite anomaly detection pipeline")
    sub = parser.add_subparsers(dest="command")

    ae_parser = sub.add_parser("train-ae", help="Train autoencoder")
    ae_parser.add_argument("--config", default="configs/autoencoder.yaml", help="Path to YAML config")
    ae_parser.add_argument("--no-plot", action="store_true", help="Skip plotting")

    cls_parser = sub.add_parser("train-cls", help="Train classifier")
    cls_parser.add_argument("--config", default="configs/classifier.yaml", help="Path to YAML config")
    cls_parser.add_argument("--no-plot", action="store_true", help="Skip plotting")

    merge_parser = sub.add_parser("merge-data", help="Merge channel CSVs")
    merge_parser.add_argument("--data-path", required=True, help="Base path containing simulation directories")

    args = parser.parse_args(argv)

    if args.command == "train-ae":
        _train_ae(args)
    elif args.command == "train-cls":
        _train_cls(args)
    elif args.command == "merge-data":
        _merge_data(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
