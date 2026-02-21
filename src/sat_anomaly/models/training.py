"""Training loops, checkpointing, evaluation, and anomaly detection utilities."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_autoencoder(model, train_loader, val_loader, config):
    """Train an autoencoder model with optional weighted MSE loss."""
    device = next(model.parameters()).device
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    reduction_none = nn.MSELoss(reduction="none")

    feature_names = config.get("feature_names", None)
    feature_weights = None
    if feature_names is not None:
        weights = [0.0 if str(name).endswith("_cmd") else 1.0 for name in feature_names]
        feature_weights = torch.tensor(weights, dtype=torch.float32, device=device).view(1, 1, -1)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = config.get("early_stopping_patience", 10)

    print(f"Starting training for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("-" * 60)

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]

            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_data)

            if feature_weights is not None and feature_weights.size(-1) == batch_data.size(-1):
                per_elem = reduction_none(reconstructed, batch_data)
                loss = (per_elem * feature_weights).mean()
            else:
                loss = reduction_none(reconstructed, batch_data).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]
                batch_data = batch_data.to(device)
                reconstructed = model(batch_data)

                if feature_weights is not None and feature_weights.size(-1) == batch_data.size(-1):
                    per_elem = reduction_none(reconstructed, batch_data)
                    loss = (per_elem * feature_weights).mean()
                else:
                    loss = reduction_none(reconstructed, batch_data).mean()

                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_model(model, config["model_save_path"].replace(".pth", "_best.pth"), epoch, avg_val_loss)
        else:
            patience_counter += 1

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"Best Val Loss: {best_val_loss:.6f}, LR: {current_lr:.2e}")
        print(f"Patience: {patience_counter}/{early_stopping_patience}")
        print("-" * 60)

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    save_model(model, config["model_save_path"], epochs, avg_val_loss)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total epochs: {len(train_losses)}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {config['model_save_path']}")
    print("=" * 60)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
    }


def train_classifier(model, train_loader, val_loader, config):
    """Train a classifier model on labeled windows."""
    device = next(model.parameters()).device
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    multi_label = bool(config.get("multi_label", False))
    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val = float("inf")
    patience_counter = 0
    early_stopping_patience = config.get("early_stopping_patience", 10)

    print(f"Starting classifier training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        batches = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            if multi_label:
                targets = targets.float()
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
            batches += 1
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")

        avg_train = running / max(1, batches)

        model.eval()
        val_running = 0.0
        val_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                if multi_label:
                    targets_f = targets.float()
                    loss = criterion(logits, targets_f)
                    preds = (torch.sigmoid(logits) > 0.5).long()
                    correct += (preds == targets).all(dim=1).sum().item()
                else:
                    loss = criterion(logits, targets)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == targets).sum().item()
                total += targets.size(0)
                val_running += loss.item()
                val_batches += 1

        avg_val = val_running / max(1, val_batches)
        acc = correct / max(1, total)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train:.6f}  Val Loss: {avg_val:.6f}  Val Acc: {acc:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            save_model(model, config["model_save_path"], epoch, avg_val)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    save_model(model, config["model_save_path"], epoch, avg_val)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val,
        "epochs_trained": len(train_losses),
    }


def save_model(model, filepath, epoch, loss):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "model_info": model.get_model_info() if hasattr(model, "get_model_info") else {},
    }

    torch.save(checkpoint, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath, model_class=None, **model_kwargs):
    """Load model checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location="cpu")

    if model_class is not None:
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint
    else:
        return checkpoint


def evaluate_model(model, data_loader, device="cpu"):
    """Evaluate autoencoder on a dataset and return reconstruction error statistics."""
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_samples = 0
    reconstruction_errors = []

    criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]

            batch_data = batch_data.to(device)
            batch_size = batch_data.size(0)

            reconstructed = model(batch_data)

            loss = criterion(reconstructed, batch_data)
            loss_per_sample = loss.mean(dim=(1, 2))

            total_loss += loss.sum().item()
            total_samples += batch_size

            reconstruction_errors.extend(loss_per_sample.cpu().numpy())

    avg_loss = total_loss / total_samples
    reconstruction_errors = np.array(reconstruction_errors)

    mean_error = np.mean(reconstruction_errors)
    std_error = np.std(reconstruction_errors)
    max_error = np.max(reconstruction_errors)
    min_error = np.min(reconstruction_errors)
    anomaly_threshold = np.percentile(reconstruction_errors, 95)

    return {
        "avg_loss": avg_loss,
        "mean_error": mean_error,
        "std_error": std_error,
        "max_error": max_error,
        "min_error": min_error,
        "anomaly_threshold": anomaly_threshold,
        "reconstruction_errors": reconstruction_errors,
    }


def detect_anomalies(model, data_loader, threshold=None, device="cpu"):
    """Detect anomalies using reconstruction error thresholding."""
    model.eval()
    model = model.to(device)

    reconstruction_errors = []
    sample_indices = []

    criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]

            batch_data = batch_data.to(device)
            batch_size = batch_data.size(0)

            reconstructed = model(batch_data)

            loss = criterion(reconstructed, batch_data)
            error_per_sample = loss.mean(dim=(1, 2))

            reconstruction_errors.extend(error_per_sample.cpu().numpy())

            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            sample_indices.extend(range(start_idx, end_idx))

    reconstruction_errors = np.array(reconstruction_errors)

    if threshold is None:
        threshold = np.percentile(reconstruction_errors, 95)

    anomaly_flags = reconstruction_errors > threshold

    num_anomalies = np.sum(anomaly_flags)
    anomaly_rate = num_anomalies / len(reconstruction_errors)

    return {
        "reconstruction_errors": reconstruction_errors,
        "anomaly_flags": anomaly_flags,
        "anomaly_indices": np.where(anomaly_flags)[0],
        "threshold": threshold,
        "num_anomalies": num_anomalies,
        "anomaly_rate": anomaly_rate,
        "sample_indices": sample_indices,
    }


def load_pretrained_autoencoder(config):
    """Create and load a pretrained autoencoder based on config."""
    from sat_anomaly.models.autoencoder import create_lstm_autoencoder, create_rnn_autoencoder

    ae_type = config.get("ae_model_type", "lstm_ae")
    ae_cfg = dict(config)
    ae_cfg["model_type"] = ae_type

    if ae_type == "lstm_ae":
        ae_model = create_lstm_autoencoder(ae_cfg)
    elif ae_type == "rnn_ae":
        ae_model = create_rnn_autoencoder(ae_cfg)
    else:
        raise ValueError(f"Unsupported ae_model_type: {ae_type}")

    ckpt_path = config.get("ae_checkpoint_path")
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        ae_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded AE checkpoint from: {ckpt_path}")
    else:
        print(f"Warning: AE checkpoint not found at {ckpt_path}. Using randomly initialized AE.")

    return ae_model


def compute_residual_windows_batched(ae_model, windows, device="cpu", batch_size=64):
    """Compute residual windows ``(input - reconstruction)`` in mini-batches."""
    ae_model = ae_model.to(device)
    ae_model.eval()

    n = len(windows)
    out = np.empty_like(windows, dtype=np.float32)
    idx = 0
    while idx < n:
        j = min(idx + batch_size, n)
        with torch.no_grad():
            try:
                x = torch.from_numpy(windows[idx:j]).to(device).float()
                recon = ae_model(x)
                resid = x - recon
                out[idx:j] = resid.detach().cpu().numpy()
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and device == torch.device("cuda"):
                    print("CUDA OOM during residuals. Falling back to CPU for this chunk...")
                    x_cpu = torch.from_numpy(windows[idx:j]).cpu().float()
                    ae_cpu = ae_model.to("cpu")
                    recon_cpu = ae_cpu(x_cpu)
                    resid_cpu = x_cpu - recon_cpu
                    out[idx:j] = resid_cpu.detach().numpy()
                    ae_model.to(device)
                else:
                    raise
            finally:
                if isinstance(device, torch.device) and device.type == "cuda":
                    del x
                    if "recon" in locals():
                        del recon
                    if "resid" in locals():
                        del resid
                    torch.cuda.empty_cache()
        idx = j
    return out
