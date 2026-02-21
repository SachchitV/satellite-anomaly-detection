# Technical Architecture

## Problem Statement

Detect and classify anomalies in satellite subsystem telemetry across 3 reaction wheels and 1 battery unit using a single AI pipeline. The system must:

- Detect anomalies independently per component
- Classify fault types (friction, torque limit, bias torque, power irregularities)
- Handle both persistent and transient faults

## Data

- **Source**: 24-hour spacecraft simulations across 3 days with 26 telemetry channels
- **Channels**:
  - Time (2): `time_ns`, `time_s`
  - Reaction Wheels (15): `rw0/1/2_omega`, `rw0/1/2_omega_cmd`, `rw0/1/2_torque_cmd`, `rw0/1/2_power`, `rw0/1/2_temp`
  - Power System (5): `bus_power`, `battery_voltage`, `battery_current`, `battery_soc`, `battery_temp`
  - Spacecraft Dynamics (3): `sc_body_rate_x/y/z`
- **Fault Types**:
  - Coulomb friction (low/medium/high severity)
  - Viscous friction (speed-dependent damping)
  - Torque limit (reduced max torque)
  - Bias torque (constant unwanted torque)

## Two-Stage Pipeline

### Stage 1: Anomaly Detection (LSTM Autoencoder)

The autoencoder learns to reconstruct normal satellite telemetry from healthy (fault-free) data.

**Architecture**:
- Encoder LSTM (2 layers, hidden_size=64) encodes input windows into latent representations
- Optional bottleneck layer (compression_ratio=0.5) for regularization
- Decoder LSTM reconstructs the original signal
- Output projection maps back to feature space

**Training**:
- Input: sliding windows of shape `(batch, 256, 23)` from fault-free data
- Loss: weighted MSE (command features `*_cmd` are masked with weight 0)
- Optimizer: Adam with gradient clipping (max_norm=1.0)
- Early stopping on validation loss (patience=10)

**Detection**: Windows with reconstruction error exceeding the 95th percentile threshold (computed on healthy validation data) are flagged as anomalous.

### Stage 2: Fault Classification (1D CNN)

Anomalous windows are classified by feeding their **reconstruction residuals** (input - reconstruction) into a 1D CNN.

**Architecture**:
- 3 residual ConvBlocks with channels [64, 128, 256], stride-2 downsampling, kernel_size=7
- Each block: Conv1d-BN-ReLU-Conv1d-BN-ReLU-Dropout + residual skip connection
- Adaptive average pooling + linear classification head

**Training**:
- Input: residual windows `(batch, 256, 23)` transposed to `(batch, 23, 256)` for Conv1d
- Loss: CrossEntropyLoss (multi-class) or BCEWithLogitsLoss (multi-label)
- 13 fault labels of the form `subsystem:fault_type` (e.g., `rw0:coulomb_friction`)

## Why This Approach

1. **Physics Alignment**: Satellite subsystems follow state-space dynamics. LSTM/RNN cells naturally model state transitions, making them better suited than transformers for this domain.

2. **Interpretable Residuals**: Reconstruction residuals provide physics-based anomaly scores analogous to Kalman filter innovation sequences.

3. **Data Efficiency**: The autoencoder trains on abundant healthy data. The classifier only needs labeled fault data for the residual-based classification stage.

4. **Modular Design**: Detection and classification are independently optimizable and deployable.

## Preprocessing

1. **Channel Merging**: Individual per-channel CSVs are merged into `signals_combined.csv` per simulation
2. **Normalization**: MinMax scaling (0-1) on feature columns, excluding time and label columns
3. **Windowing**: Sliding windows of size 256 with step size 128 (50% overlap), grouped by sequence to prevent cross-boundary contamination
4. **Label Assignment**: Window labels determined by majority vote of per-row fault annotations within the window

## Further Improvements

- Physics-informed loss functions (power conservation, torque relationships)
- State-space RNN/LSTM cells with known system matrices
- Latent-space classification as an alternative to residual-based classification
- Remaining Useful Life (RUL) estimation for degradation tracking
