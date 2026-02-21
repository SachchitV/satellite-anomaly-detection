# Data Format

The pipeline expects satellite simulation data organized under the `data/` directory. This directory is gitignored; you must provide your own data.

## Directory Layout

```
data/
├── raw/
│   └── main_data/
│       └── simulations_year/
│           ├── day_001/
│           │   ├── none/                      # Fault-free simulations
│           │   │   ├── inertial/
│           │   │   │   ├── channel_map.json
│           │   │   │   ├── <channel>.csv      # Per-channel raw data
│           │   │   │   └── signals_combined.csv  # Merged (generated)
│           │   │   ├── nadir/
│           │   │   └── sun/
│           │   ├── single_low/                # Single fault, low severity
│           │   │   ├── inertial/
│           │   │   │   ├── signals_combined.csv
│           │   │   │   └── faults.json        # Fault annotations
│           │   │   └── ...
│           │   ├── single_medium/
│           │   ├── single_high/
│           │   ├── double_low/                # Double fault scenarios
│           │   ├── double_medium/
│           │   └── double_high/
│           ├── day_002/
│           └── day_003/
└── processed/
    └── merged_data/
        └── simulations_year/                  # Same structure, only JSON + merged CSV
```

## File Formats

### `signals_combined.csv`

Merged telemetry with 26 columns:

| Column | Unit | Description |
|--------|------|-------------|
| `time_ns` | ns | Timestamp (nanoseconds) |
| `time_s` | s | Timestamp (seconds) |
| `rw0_omega` | rad/s | Reaction wheel 0 angular velocity |
| `rw0_omega_cmd` | rad/s | RW0 commanded angular velocity |
| `rw0_torque_cmd` | Nm | RW0 commanded torque |
| `rw0_power` | W | RW0 power consumption |
| `rw0_temp` | K | RW0 temperature |
| `rw1_*` | -- | Same as rw0 for reaction wheel 1 |
| `rw2_*` | -- | Same as rw0 for reaction wheel 2 |
| `bus_power` | W | Total bus power |
| `battery_voltage` | V | Battery voltage |
| `battery_current` | A | Battery current |
| `battery_soc` | -- | Battery state of charge |
| `battery_temp` | K | Battery temperature |
| `sc_body_rate_x/y/z` | rad/s | Spacecraft body rates |
| `label_any_fault` | 0/1 | Binary fault indicator |

### `faults.json`

Fault annotations per simulation run:

```json
[
  {
    "component": "rw0",
    "type": "coulomb_friction",
    "start_time": 1000.0,
    "end_time": 5000.0
  }
]
```

### `channel_map.json`

Maps raw channel CSV filenames to readable names (used by the merge utility).

## Building Processed Data

```bash
# Step 1: Merge per-channel CSVs into signals_combined.csv
sat-anomaly merge-data --data-path data/raw/main_data/simulations_year

# Step 2: Copy merged data to processed directory (optional manual step)
# Copy JSON + signals_combined.csv files from raw/ to processed/
```
