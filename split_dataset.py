import os
import shutil
import random

def split_dataset(input_dir, output_dir, split_ratio=0.8):
    """
    input_dir: folder dataset asal (dataset_dapur)
    output_dir: folder hasil split (dataset_dapur_split)
    split_ratio: persentase train (default 0.8)
    """

    # Pastikan output_dir ada
    os.makedirs(output_dir, exist_ok=True)

    # Loop setiap kelas (clean, dirty)
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Buat folder train/val untuk kelas ini
        train_class_dir = os.path.join(output_dir, "train", class_name)
        val_class_dir = os.path.join(output_dir, "val", class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copy file
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_class_dir, img))

        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_class_dir, img))

        print(f"[{class_name}] Total: {len(images)}, Train: {len(train_images)}, Val: {len(val_images)}")

if __name__ == "__main__":
    input_dataset = "./dataset_dapur"       # asal dataset
    output_dataset = "./dataset_dapur_split" # dataset hasil split
    split_dataset(input_dataset, output_dataset, split_ratio=0.8)
    print("✅ Dataset berhasil di-split!")
