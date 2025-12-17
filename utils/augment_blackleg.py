import os
import glob
import random
from PIL import Image, ImageEnhance
import torchvision.transforms as T

# Paths
input_dir = "data/datasets/livestock/blackleg"
output_dir = input_dir  # save augmented images in the same folder

# Augmentation pipeline
augmentations = T.Compose([
    T.RandomRotation(25),        # rotate up to ±25°
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
])

def augment_image(img, num_augments=10):
    """Generate augmented versions of one image"""
    augmented = []
    for i in range(num_augments):
        aug = augmentations(img)
        augmented.append(aug)
    return augmented

def main(target_count=200):
    images = glob.glob(os.path.join(input_dir, "*.*"))
    if not images:
        print("[ERROR] No images found in", input_dir)
        return
    
    # Count current images
    current_count = len(images)
    print(f"[INFO] Found {current_count} original images.")

    # Number of augmentations per image
    aug_per_image = max(1, (target_count // current_count) - 1)

    print(f"[INFO] Generating {aug_per_image} augmentations per image.")

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            augmented_images = augment_image(img, num_augments=aug_per_image)
            for idx, aug in enumerate(augmented_images):
                save_path = os.path.join(output_dir, f"{base_name}_aug{idx}.jpg")
                aug.save(save_path)

        except Exception as e:
            print(f"[ERROR] Failed on {img_path}: {e}")

    total = len(glob.glob(os.path.join(output_dir, '*.*')))
    print(f"[DONE] Augmentation complete. Total images now: {total}")

if __name__ == "__main__":
    main(target_count=200)
