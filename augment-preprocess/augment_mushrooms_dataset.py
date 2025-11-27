import os
from PIL import Image, ImageEnhance
import random


# === EXPLANATION ===
# This script performs basic data augmentation on an existing image dataset.
# It reads images from a source folder, applies random transformations, and
# saves both the original and augmented images into an output folder while
# preserving the folder structure.
# Overall, this script is a lightweight, customizable tool to enrich image
# datasets with simple augmentations, suitable for preliminary CNN training.


input_folder = "./merged_datasets_not_augmented"
output_folder = "./data/mushrooms_augmented"
os.makedirs(output_folder, exist_ok=True)

num_augments = 2


def augment_image(img):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    angle = random.randint(0, 1)
    if angle == 0:
        img = img.rotate(-90)
    elif angle == 0:
        img = img.rotate(-90)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))

    return img


for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_folder)
            output_dir = os.path.join(output_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            img = Image.open(input_path).convert("RGB")

            img.save(os.path.join(output_dir, file))

            if random.random() < 0.2:
                img = Image.open(input_path).convert("RGB")
                aug_img = augment_image(img)
                aug_img.save(os.path.join(output_dir, file))

print("Data augmentation complete! Augmented images saved in:", output_folder)
