"""Tests for YAML configuration loading and path resolution."""

import os
import tempfile

import yaml

from sat_anomaly.config import load_config, validate_config


def test_load_config_from_yaml():
    cfg = {
        "data": {"data_path": "data/test", "n_features": 10, "batch_size": 16},
        "model": {"model_type": "lstm_ae"},
        "training": {"learning_rate": 0.001, "epochs": 5},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_path = f.name

    try:
        flat = load_config(tmp_path)
        assert flat["n_features"] == 10
        assert flat["batch_size"] == 16
        assert flat["model_type"] == "lstm_ae"
        assert flat["epochs"] == 5
    finally:
        os.unlink(tmp_path)


def test_path_resolution():
    cfg = {
        "data": {"data_path": "data/processed"},
        "training": {"model_save_path": "models/test.pth"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_path = f.name

    try:
        flat = load_config(tmp_path)
        assert os.path.isabs(flat["data_path"])
        assert flat["data_path"].endswith("data/processed")
    finally:
        os.unlink(tmp_path)


def test_validate_config_valid():
    config = {"data_path": "/tmp", "model_type": "lstm_ae", "n_features": 23, "batch_size": 32, "epochs": 10}
    assert validate_config(config) is True


def test_validate_config_invalid_model_type():
    config = {"data_path": "/tmp", "model_type": "invalid", "n_features": 23, "batch_size": 32, "epochs": 10}
    assert validate_config(config) is False
