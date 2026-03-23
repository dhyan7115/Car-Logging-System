import os
import shutil
import random

# Paths
base_path = "dataset"
images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")

# Output folders
train_img_path = os.path.join(images_path, "train")
val_img_path = os.path.join(images_path, "val")
train_lbl_path = os.path.join(labels_path, "train")
val_lbl_path = os.path.join(labels_path, "val")

# Create folders
for path in [train_img_path, val_img_path, train_lbl_path, val_lbl_path]:
    os.makedirs(path, exist_ok=True)

# Get all images
images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle images
random.shuffle(images)

# Split ratio
split_ratio = 0.8
split_index = int(len(images) * split_ratio)

train_images = images[:split_index]
val_images = images[split_index:]

# Function to move files
def move_files(image_list, img_dest, lbl_dest):
    for img_name in image_list:
        img_src = os.path.join(images_path, img_name)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_src = os.path.join(labels_path, lbl_name)

        # Move image
        if os.path.exists(img_src):
            shutil.move(img_src, os.path.join(img_dest, img_name))

        # Move label
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, os.path.join(lbl_dest, lbl_name))

# Move files
move_files(train_images, train_img_path, train_lbl_path)
move_files(val_images, val_img_path, val_lbl_path)

print("✅ Dataset split completed!")