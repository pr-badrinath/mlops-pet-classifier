"""
Train the baseline SimpleCNN on the processed Cats vs Dogs dataset, tracking
everything (params, per-epoch metrics, confusion matrix, loss-curve plot,
and the final model) with MLflow.

Usage:
    python src/train.py                      # uses params.yaml
    python src/train.py --epochs 10          # override a couple of fields
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # then view results at localhost:5000
"""
import argparse
import os
from pathlib import Path

# Quiet down transformers' advisory logging. Some of our dependencies (e.g.
# mlflow) probe for optional ML-framework integrations at import time, which
# can trigger transformers' "Disabling PyTorch because..." notice - it's
# harmless (we don't use transformers) but noisy, especially since it
# reprints on every DataLoader worker restart. Must be set before anything
# imports transformers indirectly, so this sits above all other imports.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import matplotlib
matplotlib.use("Agg")  # headless backend - safe for CI / servers with no display
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

from dataset import get_dataloaders
from model import SimpleCNN
from utils import get_device, load_params, set_seed


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch of training or evaluation; returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def evaluate_with_predictions(model, loader, device):
    """Collect predictions + true labels for the whole loader (for confusion matrix etc.)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def plot_loss_curves(train_losses, val_losses, out_path: Path):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train baseline Cats vs Dogs CNN")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override params.yaml epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_params(args.params)
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("[INFO] Training on CPU. If you have an NVIDIA GPU and expected "
              "CUDA to be used, see the Troubleshooting section in README.md.")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    mlflow_cfg = cfg["mlflow"]

    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    lr = args.lr or train_cfg["learning_rate"]

    # --- Data ---
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        processed_dir=data_cfg["processed_dir"],
        image_size=data_cfg["image_size"],
        batch_size=batch_size,
        num_workers=train_cfg["num_workers"],
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Classes: {class_to_idx}")
    print(f"Train/Val/Test sizes: {len(train_loader.dataset)}/"
          f"{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    # --- Model / optimizer ---
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- MLflow setup ---
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    experiment_name = mlflow_cfg["experiment_name"]
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        # Create explicitly so we can pin a local artifact_location even
        # though the tracking metadata now lives in the sqlite database.
        # NOTE: must be a proper file:// URI (not a raw OS path), or MLflow
        # can't resolve which artifact-repository backend to use - this
        # bites Windows paths especially (e.g. "C:\...\mlruns" has no scheme).
        artifact_dir = Path(mlflow_cfg.get("artifact_location", "mlruns")).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_dir.as_uri(),
        )
    mlflow.set_experiment(experiment_name)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    artifacts_dir = Path("artifacts_tmp")
    artifacts_dir.mkdir(exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            "model": train_cfg["model_name"],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "image_size": data_cfg["image_size"],
            "train_ratio": data_cfg["train_ratio"],
            "val_ratio": data_cfg["val_ratio"],
            "test_ratio": data_cfg["test_ratio"],
            "seed": cfg["seed"],
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
        })

        train_losses, val_losses = [], []
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), models_dir / "model.pt")

        # --- Final test-set evaluation ---
        # Reload best checkpoint (by val accuracy) before evaluating on test set.
        model.load_state_dict(torch.load(models_dir / "model.pt", map_location=device))
        y_true, y_pred = evaluate_with_predictions(model, test_loader, device)
        test_acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names)

        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(report)

        mlflow.log_metric("test_accuracy", test_acc)

        # Save + log classification report as a text artifact.
        report_path = artifacts_dir / "classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        # Loss curves plot.
        loss_curve_path = artifacts_dir / "loss_curves.png"
        plot_loss_curves(train_losses, val_losses, loss_curve_path)
        mlflow.log_artifact(str(loss_curve_path))

        # Confusion matrix plot.
        cm_path = artifacts_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
        mlflow.log_artifact(str(cm_path))

        # Log the trained model itself, both as an MLflow model (for the
        # registry) and as a plain state_dict (for the M2 FastAPI service).
        # NOTE: newer MLflow versions default pytorch model logging to the
        # "pt2" export format, which requires torch>=2.4. We're on 2.2.2,
        # so force the classic pickle format explicitly. This whole block
        # is best-effort - model.pt (logged as a plain artifact below) is
        # already saved and is all M2's FastAPI service actually needs, so
        # a logging-format hiccup here shouldn't fail an otherwise-good run.
        try:
            mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
        except Exception as e:
            print(f"[WARN] mlflow.pytorch.log_model skipped ({e}); "
                  f"model.pt artifact was still saved successfully.")
        mlflow.log_artifact(str(models_dir / "model.pt"))

        print(f"\nBest model saved to: {models_dir / 'model.pt'}")
        print("Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` (from the project root) "
              "and open http://localhost:5000 to inspect this run's params, metrics, and artifacts.")


if __name__ == "__main__":
    main()
