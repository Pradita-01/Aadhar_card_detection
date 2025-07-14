import os
import shutil
import random


original_real_dir = "generated_aadhaar_cards"
original_fake_dir = "tampered_aadhaar_cards"
base_dir = "dataset"


for split in ["train", "validation"]:
    for cls in ["real", "fake"]:
        split_dir = os.path.join(base_dir, split, cls)
        os.makedirs(split_dir, exist_ok=True)

train_ratio = 0.8

def split_and_copy(src_dir, dst_train_dir, dst_val_dir):
    images = os.listdir(src_dir)
    random.shuffle(images) 

    train_count = int(len(images) * train_ratio)

   
    for img in images[:train_count]:
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(dst_train_dir, img)
        shutil.copy2(src_path, dst_path)

   
    for img in images[train_count:]:
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(dst_val_dir, img)
        shutil.copy2(src_path, dst_path)

split_and_copy(
    original_real_dir,
    os.path.join(base_dir, "train", "real"),
    os.path.join(base_dir, "validation", "real")
)


split_and_copy(
    original_fake_dir,
    os.path.join(base_dir, "train", "fake"),
    os.path.join(base_dir, "validation", "fake")
)

print(" Done splitting dataset!")
