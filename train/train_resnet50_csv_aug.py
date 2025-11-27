import os
import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import models
from albumentations import (
    Compose,
    RandomResizedCrop,
    HorizontalFlip,
    VerticalFlip,
    Affine,
    ColorJitter,
    GaussianBlur,
    Normalize,
)
from albumentations.pytorch import ToTensorV2


# -------------------------
# Dataset
# -------------------------
class CSVImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        annotations = pd.read_csv(csv_file)

        if (
            "image_path" not in annotations.columns
            or "label" not in annotations.columns
        ):
            raise ValueError("CSV must contain columns 'image_path' and 'label'")

        annotations["full_path"] = annotations["image_path"].apply(
            lambda p: os.path.join(self.root_dir, str(p).lstrip("/"))
        )

        missing_mask = ~annotations["full_path"].apply(os.path.exists)
        n_missing = missing_mask.sum()
        if n_missing > 0:
            print(f"WARNING: {n_missing} missing images ignored.")
            annotations = annotations[~missing_mask].reset_index(drop=True)

        self.annotations = annotations
        if class_to_idx is None:
            self.classes = sorted(annotations["label"].unique())
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys())

        self.annotations["label_idx"] = self.annotations["label"].map(self.class_to_idx)
        if self.annotations["label_idx"].isnull().any():
            raise ValueError("Some labels not found in class_to_idx mapping.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image = Image.open(row["full_path"]).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, int(row["label_idx"])


# -------------------------
# Training Function
# -------------------------
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    num_epochs=25,
    checkpoint_path="checkpoint.pth",
):
    since = time.time()
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}\n" + "-" * 20)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0
            dataset_size = len(dataloaders[phase].dataset)

            loop = tqdm(dataloaders[phase], desc=f"{phase}", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_mushroom_model.pth")
                print("Saved new best model!")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            },
            checkpoint_path,
        )

    total_time = time.time() - since
    print(f"\nTraining complete in {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"Best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load("best_mushroom_model.pth"))
    return model, history


# -------------------------
# Plot History
# -------------------------
def plot_history(history, out_path="training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import numpy as np

    DATA_ROOT = "."
    TRAIN_CSV, VAL_CSV = "train_reduced_final.csv", "val_reduced_final.csv"
    NUM_EPOCHS, BATCH_SIZE = 30, 1024
    CHECKPOINT_PATH = "checkpoint.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Transforms (Albumentations)
    # -------------------------
    train_transform = Compose(
        [
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.3),
            Affine(
                scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5
            ),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.5),
            GaussianBlur(p=0.2),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = Compose(
        [Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = CSVImageDataset(TRAIN_CSV, DATA_ROOT, transform=train_transform)
    class_to_idx = train_dataset.class_to_idx
    val_dataset = CSVImageDataset(
        VAL_CSV, DATA_ROOT, transform=val_transform, class_to_idx=class_to_idx
    )

    print(
        f"Found {len(class_to_idx)} classes. Train: {len(train_dataset)}, Val: {len(val_dataset)} images."
    )

    # -------------------------
    # DataLoaders (High RAM Setup)
    # -------------------------
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=os.cpu_count() // 2,
            prefetch_factor=4,
            persistent_workers=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
            prefetch_factor=4,
            persistent_workers=True,
        ),
    }

    # -------------------------
    # Model (ResNet50 fine-tuned)
    # -------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(num_features, len(class_to_idx))
    )
    model = model.to(device)

    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"torch.compile not available: {e}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # -------------------------
    # Train Model
    # -------------------------
    trained_model, history = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        device,
        num_epochs=NUM_EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
    )

    plot_history(history)
    print("Training finished. Best model saved to 'best_mushroom_model.pth'")
