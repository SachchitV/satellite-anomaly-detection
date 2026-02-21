"""LSTM and RNN autoencoders for time-series anomaly detection.

Both architectures share the same encoder-decoder structure with an optional
bottleneck layer.  Use the ``cell_type`` parameter (or the factory helpers) to
switch between LSTM and vanilla RNN cells.
"""

import torch.nn as nn


class _RecurrentAutoencoder(nn.Module):
    """Base recurrent autoencoder (shared by LSTM and RNN variants).

    Args:
        n_features: Number of input features.
        seq_len: Sequence length.
        hidden_size: Hidden size for recurrent layers.
        num_layers: Number of recurrent layers.
        dropout: Dropout rate.
        use_bottleneck: Whether to use a bottleneck layer.
        compression_ratio: Compression ratio for the bottleneck.
        cell_type: ``"lstm"`` or ``"rnn"``.
    """

    def __init__(
        self,
        n_features,
        seq_len,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        use_bottleneck=True,
        compression_ratio=0.5,
        cell_type="lstm",
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bottleneck = use_bottleneck
        self.cell_type = cell_type

        self.bottleneck_size = int(hidden_size * compression_ratio) if use_bottleneck else None

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.RNN
        rnn_kwargs = dict(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True,
        )
        if cell_type == "rnn":
            rnn_kwargs["nonlinearity"] = "tanh"

        self.encoder = rnn_cls(**rnn_kwargs)

        encoder_output_size = hidden_size
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(encoder_output_size, self.bottleneck_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.bottleneck_size, encoder_output_size),
                nn.ReLU(),
            )
        else:
            self.bottleneck = None

        decoder_kwargs = dict(rnn_kwargs)
        decoder_kwargs["input_size"] = encoder_output_size
        self.decoder = rnn_cls(**decoder_kwargs)

        self.output_projection = nn.Linear(hidden_size, n_features)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """Forward pass: ``(batch, seq_len, n_features) -> (batch, seq_len, n_features)``."""
        encoder_output, hidden = self.encoder(x)

        if self.use_bottleneck:
            compressed = self.bottleneck(encoder_output)
        else:
            compressed = encoder_output

        decoder_output, _ = self.decoder(compressed, hidden)

        reconstructed = self.output_projection(decoder_output)
        return reconstructed

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_type": f"{self.cell_type.upper()} Autoencoder",
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bottleneck_size": self.bottleneck_size,
            "use_bottleneck": self.use_bottleneck,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


class LSTMAutoencoder(_RecurrentAutoencoder):
    """LSTM-based autoencoder for time-series anomaly detection."""

    def __init__(self, n_features, seq_len, hidden_size=64, num_layers=2, dropout=0.1, use_bottleneck=True, compression_ratio=0.5):
        super().__init__(
            n_features=n_features,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_bottleneck=use_bottleneck,
            compression_ratio=compression_ratio,
            cell_type="lstm",
        )


class RNNAutoencoder(_RecurrentAutoencoder):
    """Vanilla-RNN-based autoencoder for time-series anomaly detection."""

    def __init__(self, n_features, seq_len, hidden_size=64, num_layers=2, dropout=0.1, use_bottleneck=True, compression_ratio=0.5):
        super().__init__(
            n_features=n_features,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_bottleneck=use_bottleneck,
            compression_ratio=compression_ratio,
            cell_type="rnn",
        )


def create_lstm_autoencoder(config):
    """Factory function to create an LSTM autoencoder from a config dict."""
    return LSTMAutoencoder(
        n_features=config["n_features"],
        seq_len=config.get("seq_len", 256),
        hidden_size=config.get("hidden_size", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
        use_bottleneck=config.get("use_bottleneck", True),
        compression_ratio=config.get("compression_ratio", 0.5),
    )


def create_rnn_autoencoder(config):
    """Factory function to create an RNN autoencoder from a config dict."""
    return RNNAutoencoder(
        n_features=config["n_features"],
        seq_len=config.get("seq_len", 256),
        hidden_size=config.get("hidden_size", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
        use_bottleneck=config.get("use_bottleneck", True),
        compression_ratio=config.get("compression_ratio", 0.5),
    )
