import os
import shutil
import random

# --- PATH CONFIGURATION (Update these matches your folders) ---
# 1. Where are the images of the PLATES? (Dataset 2)
path_to_images = "dataset/ealpr-master/ealpr- plates dataset" 

# 2. Where are the labels for the CHARACTERS? (Dataset 3)
path_to_labels = "dataset/ealpr-master/ealpr- lp characters dataset/characters labeling"

# 3. Where should we put the ready-to-train data?
dest_root = "dataset/egyptian_characters_detection"

# Split ratio (80% for training, 20% for testing)
split_ratio = 0.8 

def setup_directories():
    # Create YOLO folder structure
    for split in ['train', 'val']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(dest_root, split, dtype), exist_ok=True)

def merge_and_organize():
    # Get all image files (jpg, png, etc.)
    all_images = [f for f in os.listdir(path_to_images) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Shuffle for randomness
    random.shuffle(all_images)
    
    # Split into Train and Val
    split_idx = int(len(all_images) * split_ratio)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    print(f"Found {len(all_images)} plate images.")
    print(f"Training on {len(train_imgs)}, Validating on {len(val_imgs)}")

    def process_batch(file_list, split_type):
        matched_count = 0
        for img_name in file_list:
            # 1. Define source paths
            src_img_path = os.path.join(path_to_images, img_name)
            
            # The label should have the same name but .txt extension
            # Example: "img_123.jpg" -> "img_123.txt"
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label_path = os.path.join(path_to_labels, label_name)
            
            # 2. Check if the label actually exists
            if os.path.exists(src_label_path):
                # Copy Image
                dst_img_path = os.path.join(dest_root, split_type, "images", img_name)
                shutil.copy(src_img_path, dst_img_path)
                
                # Copy Label
                dst_label_path = os.path.join(dest_root, split_type, "labels", label_name)
                shutil.copy(src_label_path, dst_label_path)
                
                matched_count += 1
            else:
                # If no label exists, we skip this image (useless for training)
                pass

        print(f"Finished {split_type}: Successfully paired {matched_count} images with labels.")

    process_batch(train_imgs, 'train')
    process_batch(val_imgs, 'val')
    print("\nData preparation complete!")

if __name__ == "__main__":
    setup_directories()
    merge_and_organize()