import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter
import json
import time
from tqdm import tqdm
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# === EXPLANATION ===
# This script trains, validates, and tests a ResNet-18 image classification model
# on a mushroom dataset defined by CSV files. It provides a full training pipeline
# with data loading, augmentation, class weighting, mixed-precision training, model
# checkpointing, evaluation metrics, and visualization outputs.
#
# Major features:
#   • Loads images and labels from CSVs using a custom Dataset
#   • Applies data augmentation and preprocessing (center-crop → 224×224 → normalize)
#   • Computes class-balanced weights to reduce label imbalance during training
#   • Supports Apple MPS acceleration (Mac GPU) with automatic mixed precision
#   • Trains a ResNet-18 (pretrained or from scratch) with Adam optimizer + scheduler
#   • Tracks epoch-level metrics (loss/accuracy) for both train and validation sets
#   • Automatically saves the best-performing model based on validation accuracy
#   • Evaluates the final model on a test set
#   • Generates:
#         - classification report (text)
#         - confusion matrix (PNG + raw NumPy array)
#         - per-class accuracy JSON
#         - training history JSON
#         - class→index mapping JSON
#
# This script is intended for large, multi-class mushroom classification but can be
# adapted to any CSV-based image dataset.


# -------------------------
# Dataset
# -------------------------
class MushroomDataset(Dataset):
    def __init__(self, csv_file, image_root="", transform=None):
        df = pd.read_csv(csv_file)
        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'image_path' and 'label' columns")
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

        labels = sorted(self.df["label"].unique())
        self.class_to_idx = {lab: i for i, lab in enumerate(labels)}
        self.idx_to_class = {i: lab for lab, i in self.class_to_idx.items()}

        self.df["label_idx"] = self.df["label"].map(self.class_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        if not os.path.isabs(path):
            path = os.path.join(self.image_root, path)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label_idx"])
        return img, label


# -------------------------
# Utilities
# -------------------------
def compute_class_weights(labels, device=None):
    counts = Counter(labels)
    num_classes = max(counts.keys()) + 1
    freq = np.zeros(num_classes, dtype=np.float64)
    for k, v in counts.items():
        freq[k] = v
    weights = 1.0 / (freq + 1e-12)
    weights = weights / np.mean(weights)
    weights = torch.from_numpy(weights.astype(np.float32))
    if device is not None:
        weights = weights.to(device)
    return weights


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -------------------------
# Training & Evaluation
# -------------------------
def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler, log_interval=100
):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for i, (imgs, labels) in enumerate(
        tqdm(loader, desc="Train batches", leave=False, mininterval=1)
    ):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=True):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total_samples += imgs.size(0)

        if (i + 1) % log_interval == 0:
            avg_loss = running_loss / total_samples
            avg_acc = running_corrects / total_samples
            tqdm.write(
                f"[Batch {i + 1}/{len(loader)}] Running loss: {avg_loss:.4f}, acc: {avg_acc:.4f}"
            )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, log_interval=100):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    preds, trues = [], []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(
            tqdm(loader, desc="Val/Test batches", leave=False, mininterval=1)
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type=device.type, enabled=True):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            running_corrects += (predicted == labels).sum().item()
            total_samples += imgs.size(0)

            preds.append(predicted.cpu().numpy())
            trues.append(labels.cpu().numpy())

            if (i + 1) % log_interval == 0:
                avg_loss = running_loss / total_samples
                avg_acc = running_corrects / total_samples
                tqdm.write(
                    f"[Val Batch {i + 1}/{len(loader)}] Running loss: {avg_loss:.4f}, acc: {avg_acc:.4f}"
                )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return epoch_loss, epoch_acc, trues, preds


# -------------------------
# Main
# -------------------------
def main(args):
    # Use MPS if available on Mac, else CPU
    if torch.backends.mps.is_available() and not args.no_cuda:
        device = torch.device("mps")
        print("Using Apple MPS backend")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # -------------------------
    # Transforms
    # -------------------------
    train_transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: _center_crop_to_square(img)),
            transforms.Resize((224, 224)),
            transforms.RandomApply(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02
                    ),
                    transforms.RandomRotation(15),
                ],
                p=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_test_transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: _center_crop_to_square(img)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # -------------------------
    # Datasets & Dataloaders
    # -------------------------
    train_ds = MushroomDataset(
        args.train_csv, image_root=args.image_root, transform=train_transform
    )
    val_ds = MushroomDataset(
        args.val_csv, image_root=args.image_root, transform=val_test_transform
    )
    test_ds = MushroomDataset(
        args.test_csv, image_root=args.image_root, transform=val_test_transform
    )

    # Ensure consistent class mapping
    train_map = train_ds.class_to_idx

    def remap_dataset(ds, master_map):
        if ds.class_to_idx != master_map:
            ds.df["label_idx"] = ds.df["label"].map(master_map)
            missing = ds.df["label_idx"].isna().sum()
            if missing > 0:
                raise ValueError(f"{missing} labels not in training label set")
            ds.df["label_idx"] = ds.df["label_idx"].astype(int)
            ds.class_to_idx = master_map
            ds.idx_to_class = {i: c for c, i in master_map.items()}
        return ds

    val_ds = remap_dataset(val_ds, train_map)
    test_ds = remap_dataset(test_ds, train_map)
    num_classes = len(train_map)
    print("Number of classes:", num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -------------------------
    # Model
    # -------------------------
    model = models.resnet18(pretrained=not args.no_pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # -------------------------
    # Loss, optimizer, scheduler
    # -------------------------
    train_labels = train_ds.df["label_idx"].values
    class_weights = compute_class_weights(train_labels, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2, factor=0.5, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # AMP works on MPS

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(train_map, os.path.join(args.output_dir, "class_to_idx.json"))

    # -------------------------
    # Training loop
    # -------------------------
    best_val_acc = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pth")
    metrics_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Wrap the epoch loop with tqdm for a progress bar
    for epoch in trange(1, args.epochs + 1, desc="Epochs", unit="epoch"):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, log_interval=200
        )

        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device, log_interval=200
        )

        scheduler.step(val_loss)

        metrics_history["train_loss"].append(train_loss)
        metrics_history["train_acc"].append(train_acc)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_acc"].append(val_acc)

        # Epoch summary
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"Time: {time.time() - t0:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_to_idx": train_map,
                },
                best_path,
            )
            print(f"✅ Saved new best model (val_acc={val_acc:.4f})")

    save_json(metrics_history, os.path.join(args.output_dir, "metrics_history.json"))

    # -------------------------
    # Test evaluation
    # -------------------------
    print("\nLoading best model for test evaluation...")
    chk = torch.load(best_path, map_location=device)
    model.load_state_dict(chk["model_state_dict"])
    test_loss, test_acc, y_true, y_pred = validate(
        model, test_loader, criterion, device
    )
    print(f"\nTest loss: {test_loss:.4f}  acc: {test_acc:.4f}")

    idx_to_class = {v: k for k, v in train_map.items()}
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=[idx_to_class[i] for i in range(num_classes)],
        zero_division=0,
    )
    print("\nClassification report (text):\n")
    print(clf_report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(clf_report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm)
    plt.figure(figsize=(12, 10))
    try:
        sns.heatmap(cm, cmap="Blues")
        plt.title("Confusion Matrix (all classes)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(
            "Failed to render heatmap (likely too large). Saved raw numpy array only.",
            e,
        )

    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
    per_class_acc_dict = {
        idx_to_class[i]: float(per_class_acc[i]) for i in range(num_classes)
    }
    save_json(
        per_class_acc_dict, os.path.join(args.output_dir, "per_class_accuracy.json")
    )

    print("\nDone. Outputs saved to:", args.output_dir)


# -------------------------
# Helper: center-crop to square
# -------------------------
def _center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 on mushroom dataset")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="do not use pretrained imagenet weights",
    )
    parser.add_argument("--no_cuda", action="store_true", help="force CPU")
    args = parser.parse_args()
    main(args)
