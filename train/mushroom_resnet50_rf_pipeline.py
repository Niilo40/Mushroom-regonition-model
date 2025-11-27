import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === EXPLANATION ===
# This script builds a hybrid deep learning + classical ML pipeline for image
# classification using mushrooms. It uses a pretrained or fine-tuned ResNet-50
# CNN to extract feature embeddings from images, reduces feature dimensionality
# via PCA, and trains a Random Forest classifier on the reduced feature set.
# Cached intermediate results are used to speed up repeated runs.
#
# Main features:
#
#   • CSVImageDataset
#         - Loads images and labels from CSV files
#         - Constructs full file paths relative to a root directory
#         - Builds or loads a class-to-index mapping
#         - Handles missing files gracefully
#         - Returns PyTorch tensors suitable for CNN input
#
#   • Data Preprocessing
#         - Resizes all images to 224x224
#         - Converts images to tensors
#         - StandardScaler applied to ResNet feature vectors
#         - PCA reduces feature dimensionality from 2048 → 128
#         - Saves scaler and PCA models for reuse
#
#   • CNN Feature Extraction (ResNet-50)
#         - Loads either a fine-tuned model ("mushroom_model.pth") or
#           ImageNet-pretrained ResNet-50
#         - Removes the final classifier layer (Identity) to output 2048-d features
#         - Extracts features in batches using DataLoader with GPU acceleration
#         - Caches features and labels to ".npy" files
#
#   • Random Forest Classifier
#         - Trains on PCA-reduced ResNet features
#         - Hyperparameters:
#               * n_estimators=300
#               * max_depth=15
#               * min_samples_split=3
#               * class_weight="balanced"
#               * parallelized with n_jobs=-1
#         - Saves the trained model to "rf_model_resnet50.joblib"
#         - Loads model if already trained to avoid retraining
#
#   • Evaluation
#         - Uses the trained Random Forest to predict test set labels
#         - Computes overall accuracy
#         - Prints a full classification report (precision, recall, F1-score per class)
#
# Overall, this script implements a modern hybrid pipeline that leverages CNNs for
# feature representation and Random Forests for interpretable, fast classification.
# It supports caching, GPU acceleration for feature extraction, and automated
# saving/loading of models and preprocessing objects.


# ------------------------------
# CSVImageDataset class
# ------------------------------
class CSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, class_to_idx=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        if class_to_idx is None:
            self.classes = sorted(self.annotations["label"].unique())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        path_from_csv = row["image_path"].lstrip("/")
        img_path = os.path.join(self.root_dir, path_from_csv)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[WARNING] Missing file: {img_path}")
            return torch.zeros((3, 224, 224)), -1

        label = self.class_to_idx[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label


# ------------------------------
# Configuration
# ------------------------------
train_csv = "train_reduced_final.csv"
test_csv = "val_reduced_final.csv"
root_dir = ""
batch_size = 32
num_workers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥 Using device:", device)


# ------------------------------
# Transforms
# ------------------------------
data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# ------------------------------
# Load datasets
# ------------------------------
train_dataset = CSVImageDataset(train_csv, root_dir, transform=data_transforms)
test_dataset = CSVImageDataset(
    test_csv,
    root_dir,
    transform=data_transforms,
    class_to_idx=train_dataset.class_to_idx,
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)


# ------------------------------
# Load pretrained CNN (.pth or ImageNet)
# ------------------------------
cnn_path = "mushroom_model.pth"

if os.path.exists(cnn_path):
    print(f"📦 Loading fine-tuned ResNet50 from '{cnn_path}'...")
    resnet = models.resnet50(weights=None)
    num_features = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_features, len(train_dataset.classes))
    checkpoint = torch.load(cnn_path, map_location=device)
    resnet.load_state_dict(checkpoint, strict=False)
    resnet.fc = torch.nn.Identity()  # remove classifier head after loading
else:
    print("⚙️ No fine-tuned model found — using ImageNet pretrained ResNet50...")
    resnet = models.resnet50(weights="IMAGENET1K_V1")
    resnet.fc = torch.nn.Identity()

resnet = resnet.to(device)
resnet.eval()


# ------------------------------
# Feature extraction function
# ------------------------------
def extract_features(dataloader, save_features_path, save_labels_path):
    features_list, labels_list = [], []
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = resnet(imgs).cpu().numpy()
            features_list.append(feats)
            labels_list.append(labels.numpy())
            total += imgs.size(0)
            if total % 1000 < batch_size:
                print(f"Extracted {total} features...")

    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)
    np.save(save_features_path, features)
    np.save(save_labels_path, labels)
    print(f"✅ Done extracting {total} features — saved to {save_features_path}")
    return features, labels


# ------------------------------
# Extract or load cached features
# ------------------------------
if os.path.exists("train_features.npy") and os.path.exists("train_labels.npy"):
    print("📂 Loading cached features...")
    train_features = np.load("train_features.npy")
    train_labels = np.load("train_labels.npy")
    test_features = np.load("test_features.npy")
    test_labels = np.load("test_labels.npy")
else:
    train_features, train_labels = extract_features(
        train_loader, "train_features.npy", "train_labels.npy"
    )
    test_features, test_labels = extract_features(
        test_loader, "test_features.npy", "test_labels.npy"
    )


# ------------------------------
# Normalize + PCA
# ------------------------------
print("⚙️ Scaling and applying PCA...")

# StandardScaler normalization
scaler_path = "scaler_model.joblib"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Loaded existing StandardScaler.")
else:
    scaler = StandardScaler()
    scaler.fit(train_features)
    joblib.dump(scaler, scaler_path)
    print("Trained and saved StandardScaler.")

train_scaled = scaler.transform(train_features)
test_scaled = scaler.transform(test_features)

# PCA reduction
pca_path = "pca_model.joblib"
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)
    print("Loaded existing PCA model.")
else:
    pca = PCA(n_components=128, random_state=42)
    pca.fit(train_scaled)
    joblib.dump(pca, pca_path)
    print("Trained and saved PCA model.")

train_pca = pca.transform(train_scaled)
test_pca = pca.transform(test_scaled)


# ------------------------------
# Train or load Random Forest
# ------------------------------
rf_model_path = "rf_model_resnet50.joblib"

if os.path.exists(rf_model_path):
    print(f"📦 Loading existing Random Forest from '{rf_model_path}'...")
    rf = joblib.load(rf_model_path)
else:
    print("🌲 Training new Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    rf.fit(train_pca, train_labels)
    joblib.dump(rf, rf_model_path)
    print(f"✅ Saved trained Random Forest to '{rf_model_path}'")


# ------------------------------
# Evaluate
# ------------------------------
print("\n🔍 Evaluating model...")
preds = rf.predict(test_pca)
acc = accuracy_score(test_labels, preds)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print(
    "\nClassification Report:\n",
    classification_report(test_labels, preds, target_names=train_dataset.classes),
)
