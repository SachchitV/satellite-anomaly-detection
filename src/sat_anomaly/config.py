"""YAML configuration loader with automatic path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def _resolve_paths(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Resolve relative path values against *root*."""
    path_keys = {"data_path", "model_save_path", "ae_checkpoint_path"}
    for key, value in cfg.items():
        if isinstance(value, dict):
            _resolve_paths(value, root)
        elif key in path_keys and isinstance(value, str) and not Path(value).is_absolute():
            cfg[key] = str(root / value)
    return cfg


def load_config(yaml_path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config file and return a flat dictionary.

    Nested sections (data, model, training, autoencoder) are merged into a
    single flat dict so downstream code can do ``config['hidden_size']``
    directly.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # Flatten one level of nesting.
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value

    # Apply CLI overrides.
    if overrides:
        flat.update(overrides)

    # Resolve relative paths against project root.
    _resolve_paths(flat, PROJECT_ROOT)

    return flat


# ---------------------------------------------------------------------------
# Legacy helpers (kept so existing scripts / notebooks can migrate gradually)
# ---------------------------------------------------------------------------

def get_base_config() -> Dict[str, Any]:
    """Return default base configuration with relative paths."""
    return {
        "data_path": str(PROJECT_ROOT / "data/processed/merged_data/simulations_year"),
        "n_features": 23,
        "seq_len": 256,
        "n_classes": 6,
        "window_size": 256,
        "step_size": 128,
        "normalization_method": "standard",
        "train_ratio": 0.8,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
        "model_save_path": str(PROJECT_ROOT / "models/autoencoder.pth"),
    }


def get_lstm_config() -> Dict[str, Any]:
    config = get_base_config()
    config.update(
        model_type="lstm_ae",
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        use_bottleneck=True,
        compression_ratio=0.5,
    )
    return config


def get_rnn_config() -> Dict[str, Any]:
    config = get_base_config()
    config.update(
        model_type="rnn_ae",
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        use_bottleneck=True,
        compression_ratio=0.5,
    )
    return config


def get_classifier_config() -> Dict[str, Any]:
    config = get_base_config()
    config.update(
        model_type="cnn_cls",
        n_classes=2,
        multi_label=False,
        learning_rate=3e-4,
        epochs=10,
        batch_size=64,
        window_size=256,
        step_size=128,
        model_save_path=str(PROJECT_ROOT / "models/classifier.pth"),
        cnn_channels=[64, 128, 256],
        dropout=0.2,
        ae_model_type="lstm_ae",
        ae_checkpoint_path=str(PROJECT_ROOT / "models/autoencoder_best.pth"),
    )
    return config


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    import json

    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def validate_config(config: Dict[str, Any]) -> bool:
    required_keys = ["data_path", "model_type", "n_features", "batch_size", "epochs"]
    for key in required_keys:
        if key not in config:
            print(f"Missing required config key: {key}")
            return False

    valid_model_types = ["lstm_ae", "rnn_ae", "cnn_cls"]
    if config.get("model_type") not in valid_model_types:
        print(f"Invalid model_type: {config.get('model_type')}. Must be one of {valid_model_types}")
        return False

    return True
