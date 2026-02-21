"""Smoke tests for model forward passes using random tensors."""

import torch

from sat_anomaly.models.autoencoder import (
    LSTMAutoencoder,
    RNNAutoencoder,
    create_lstm_autoencoder,
    create_rnn_autoencoder,
)
from sat_anomaly.models.classifier import CNNClassifier


def test_lstm_autoencoder_forward():
    model = LSTMAutoencoder(n_features=23, seq_len=64, hidden_size=32, num_layers=1, dropout=0.0)
    x = torch.randn(4, 64, 23)
    out = model(x)
    assert out.shape == (4, 64, 23)


def test_rnn_autoencoder_forward():
    model = RNNAutoencoder(n_features=23, seq_len=64, hidden_size=32, num_layers=1, dropout=0.0)
    x = torch.randn(4, 64, 23)
    out = model(x)
    assert out.shape == (4, 64, 23)


def test_cnn_classifier_forward():
    model = CNNClassifier(n_features=23, n_classes=5, channels=[32, 64], dropout=0.0)
    x = torch.randn(4, 64, 23)
    logits = model(x)
    assert logits.shape == (4, 5)


def test_lstm_factory():
    config = {"n_features": 10, "seq_len": 32, "hidden_size": 16, "num_layers": 1, "dropout": 0.0}
    model = create_lstm_autoencoder(config)
    x = torch.randn(2, 32, 10)
    out = model(x)
    assert out.shape == (2, 32, 10)


def test_rnn_factory():
    config = {"n_features": 10, "seq_len": 32, "hidden_size": 16, "num_layers": 1, "dropout": 0.0}
    model = create_rnn_autoencoder(config)
    x = torch.randn(2, 32, 10)
    out = model(x)
    assert out.shape == (2, 32, 10)


def test_model_info():
    model = LSTMAutoencoder(n_features=5, seq_len=16, hidden_size=8, num_layers=1)
    info = model.get_model_info()
    assert info["n_features"] == 5
    assert info["hidden_size"] == 8
    assert info["total_parameters"] > 0
