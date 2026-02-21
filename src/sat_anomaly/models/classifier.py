"""1D CNN classifier for windowed time-series classification."""

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=2, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride) if (in_ch != out_ch or stride != 1) else None
        )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.net(x)
        x = x + residual
        return x


class CNNClassifier(nn.Module):
    def __init__(self, n_features, n_classes, channels=None, dropout=0.2):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]
        layers = []
        in_ch = n_features
        for ch in channels:
            layers.append(ConvBlock(in_ch, ch, kernel_size=7, stride=2, dropout=dropout))
            in_ch = ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        # Expect x of shape (batch, seq_len, features); convert to (batch, channels, time).
        if x.dim() == 3:
            x = x.transpose(1, 2)
        feats = self.backbone(x)
        pooled = self.pool(feats).squeeze(-1)
        logits = self.head(pooled)
        return logits

    def get_model_info(self):
        return {"type": "CNNClassifier", "head_out": self.head.out_features}
