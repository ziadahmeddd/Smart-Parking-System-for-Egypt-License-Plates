import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
# UPDATE THESE PATHS to match where you extracted your folders
# Example: If your folders are in 'dataset/ealpr vehicles dataset/...'
source_images = "dataset/ealpr-master/ealpr vechicles dataset/vehicles" 
source_labels = "dataset/ealpr-master/ealpr vechicles dataset/vehicles labeling"

# Where we want to store the organized data for YOLO
dest_root = "dataset/egyptian_plates_detection"

# Split ratio (0.8 = 80% training, 20% validation)
split_ratio = 0.8 

def setup_directories():
    # Create the train/val structure
    for split in ['train', 'val']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(dest_root, split, dtype), exist_ok=True)

def move_files():
    # Get all image filenames
    images = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images) # Shuffle to randomize the split
    
    split_index = int(len(images) * split_ratio)
    train_imgs = images[:split_index]
    val_imgs = images[split_index:]
    
    print(f"Total images: {len(images)}")
    print(f"Training: {len(train_imgs)}, Validation: {len(val_imgs)}")

    # Function to move files
    def process_split(file_list, split_name):
        for img_name in file_list:
            # Construct paths
            src_img_path = os.path.join(source_images, img_name)
            dst_img_path = os.path.join(dest_root, split_name, "images", img_name)
            
            # Find matching label (assuming same filename but .txt)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label_path = os.path.join(source_labels, label_name)
            dst_label_path = os.path.join(dest_root, split_name, "labels", label_name)
            
            # Copy Image
            shutil.copy(src_img_path, dst_img_path)
            
            # Copy Label (only if it exists)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)
            else:
                print(f"Warning: No label found for {img_name}")

    print("Moving training files...")
    process_split(train_imgs, 'train')
    
    print("Moving validation files...")
    process_split(val_imgs, 'val')
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    setup_directories()
    move_files()