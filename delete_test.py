import pandas as pd
import os
import random
import shutil

# === CONFIG ===
train_csv_path = "train_original.csv"
val_csv_path = "val_original.csv"
test_csv_path = "test.csv"  # Test CSV to filter
output_train_csv = "train.csv"
output_val_csv = "val.csv"
output_test_csv = "test_filtered.csv"
delete_count = 100
trash_dir = "deleted_images"

# === PREPARE OUTPUT DIR ===
os.makedirs(trash_dir, exist_ok=True)

# === LOAD CSVs ===
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
test_df = pd.read_csv(test_csv_path)

required_cols = {"label", "image_path"}
for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{df_name}.csv must contain 'label' and 'image_path' columns")

# === CHOOSE SPECIES TO DELETE ===
unique_species = train_df["label"].unique()

if len(unique_species) <= delete_count:
    raise ValueError(
        f"Only {len(unique_species)} unique species found — can't delete {delete_count}"
    )

species_to_delete = random.sample(list(unique_species), delete_count)

print(f"\nDeleting {len(species_to_delete)} species:")
for s in species_to_delete:
    print(f" - {s}")


# === FUNCTION TO MOVE IMAGES ===
def move_images(df, species_list):
    moved = 0
    for _, row in df.iterrows():
        if row["label"] in species_list:
            img_path = row["image_path"]
            if os.path.exists(img_path):
                dest_path = os.path.join(trash_dir, os.path.basename(img_path))
                try:
                    shutil.move(img_path, dest_path)
                    moved += 1
                except Exception as e:
                    print(f"Error moving {img_path}: {e}")
    return moved


# === DELETE FILES (TRAIN + VAL + TEST) ===
deleted_train = move_images(train_df, species_to_delete)
deleted_val = move_images(val_df, species_to_delete)
deleted_test = move_images(test_df, species_to_delete)

print(
    f"\nMoved {deleted_train} train, {deleted_val} val, and {deleted_test} test images to '{trash_dir}'."
)

# === FILTER OUT DELETED SPECIES ===
train_filtered = train_df[~train_df["label"].isin(species_to_delete)]
val_filtered = val_df[~val_df["label"].isin(species_to_delete)]
test_filtered = test_df[~test_df["label"].isin(species_to_delete)]

# === SAVE FILTERED CSVs ===
train_filtered.to_csv(output_train_csv, index=False)
val_filtered.to_csv(output_val_csv, index=False)
test_filtered.to_csv(output_test_csv, index=False)

print(f"\nFiltered train CSV saved to: {output_train_csv}")
print(f"Filtered val CSV saved to:   {output_val_csv}")
print(f"Filtered test CSV saved to:  {output_test_csv}")
print("\nDone!")
