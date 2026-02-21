# Performance Metrics

## Autoencoder (LSTM)

| Metric | Value |
|--------|-------|
| Architecture | 2-layer LSTM, hidden_size=64, bottleneck |
| Training Data | 4,860 windows (fault-free) |
| Validation Data | 1,215 windows (fault-free) |
| Best Validation Loss | 0.000130 |
| Epochs Trained | 100 |
| Window Size | 256 timesteps |
| Features | 23 telemetry channels |

## Classifier (1D CNN)

| Metric | Value |
|--------|-------|
| Architecture | 3-block ResNet-style CNN [64, 128, 256] |
| Training Data | 29,160 residual windows |
| Validation Data | 7,290 residual windows |
| Number of Classes | 13 fault labels |
| Best Validation Loss | 0.3961 |
| Validation Accuracy | 88.93% |
| Epochs Trained | 10 |

## Label Distribution

The classifier distinguishes between 13 fault labels of the form `subsystem:fault_type`:

- `none` (no fault)
- `rw0:coulomb_friction`, `rw0:viscous_friction`, `rw0:torque_limit`, `rw0:bias_torque`
- `rw1:coulomb_friction`, `rw1:viscous_friction`, `rw1:torque_limit`, `rw1:bias_torque`
- `rw2:coulomb_friction`, `rw2:viscous_friction`, `rw2:torque_limit`, `rw2:bias_torque`
