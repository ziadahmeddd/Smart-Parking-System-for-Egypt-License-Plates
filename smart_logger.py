"""
Batch Image Processor for License Plate Logging
Processes images from a folder and logs them to database and blockchain.
Updated to use centralized config.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
from bidi.algorithm import get_display
import os
import datetime
import glob
from typing import List, Optional
import logging

import config
import database
import simple_blockchain
import logger as log_module

# Setup logging
logger = log_module.get_logger(__name__)

def draw_arabic(
    img: np.ndarray,
    text: str,
    position: tuple,
    font_size: int = 32,
    color: tuple = (0, 255, 0)
) -> np.ndarray:
    """Draw Arabic text on image with proper RTL rendering."""
    if not text:
        return img
    
    try:
        bidi_text = get_display(text, base_dir='R')
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype(config.FONT_PATH, font_size)
        except Exception:
            font = ImageFont.load_default()
            
        draw.text(
            position,
            bidi_text,
            font=font,
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0)
        )
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error drawing Arabic text: {e}")
        return img

def read_characters(crop: np.ndarray, char_model: YOLO) -> str:
    """Read characters from plate crop using OCR."""
    try:
        char_results = char_model.predict(crop, conf=config.CHAR_CONFIDENCE, verbose=False)
        
        detected_items = []
        for r in char_results:
            for b in r.boxes:
                cls = int(b.cls[0])
                sym = char_model.names[cls]
                cx = b.xywh[0][0].item()
                detected_items.append((cx, sym))
        
        if not detected_items:
            return ""
        
        detected_items.sort(key=lambda x: x[0])
        
        numbers = [x[1] for x in detected_items if x[1] in config.ARABIC_NUMBERS]
        letters = [x[1] for x in detected_items if x[1] not in config.ARABIC_NUMBERS]
        
        letters.reverse()
        numbers.reverse()
        
        final_text = " ".join(letters) + "   " + " ".join(numbers)
        return final_text
        
    except Exception as e:
        logger.error(f"Error reading characters: {e}")
        return ""

def process_image(
    img_path: str,
    plate_model: YOLO,
    char_model: YOLO
) -> int:
    """
    Process a single image file.
    
    Args:
        img_path: Path to image file
        plate_model: Plate detection model
        char_model: Character recognition model
        
    Returns:
        Number of plates detected and logged
    """
    plates_found = 0
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            return 0
        
        results = plate_model.predict(
            img,
            imgsz=config.IMAGE_SIZE,
            conf=config.PLATE_CONFIDENCE,
            verbose=False
        )
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                if (x2 - x1) < config.PLATE_MIN_WIDTH:
                    continue
                
                # Extract crop
                pad = config.CROP_PADDING
                h, w = img.shape[:2]
                crop = img[
                    max(0, y1-pad):min(h, y2+pad),
                    max(0, x1-pad):min(w, x2+pad)
                ]
                
                # Read characters
                final_text = read_characters(crop, char_model)
                
                if final_text:
                    # Save crop
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"plate_{timestamp}_{x1}.jpg"
                    save_path = os.path.join(config.OUTPUT_FOLDER, filename)
                    cv2.imwrite(save_path, crop)
                    
                    # Log to database
                    entry_id = database.save_entry(final_text, save_path)
                    
                    # Log to blockchain
                    if entry_id:
                        block_data = {
                            "plate": final_text,
                            "image_ref": filename,
                            "timestamp": timestamp,
                            "source": "batch_processing",
                            "entry_id": entry_id
                        }
                        new_block = simple_blockchain.ledger.add_block(block_data)
                        
                        if new_block:
                            logger.info(f"Processed: {final_text} -> Block #{new_block.index}")
                            print(f"✅ Saved: {final_text}")
                            print(f"🔗 Block #{new_block.index} [Hash: {new_block.hash[:8]}...]")
                            plates_found += 1
        
        return plates_found
        
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}", exc_info=True)
        return 0

def main(input_folder: str = "test_images") -> None:
    """
    Main batch processing function.
    
    Args:
        input_folder: Folder containing images to process
    """
    logger.info("⏳ Starting Batch Image Logger...")
    print("⏳ Loading AI Logger (Batch Mode)...")
    
    try:
        # Initialize database
        database.initialize_db()
        
        # Load models
        logger.info("Loading AI models...")
        plate_model = YOLO(config.PLATE_MODEL_PATH)
        char_model = YOLO(config.CHAR_MODEL_PATH)
        
        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        
        logger.info(f"Found {len(image_files)} images to process")
        print(f"📂 Found {len(image_files)} images in '{input_folder}'")
        
        if not image_files:
            print(f"❌ No images found in '{input_folder}'")
            return
        
        # Process images
        total_plates = 0
        for i, img_path in enumerate(image_files, 1):
            print(f"\nProcessing {i}/{len(image_files)}: {os.path.basename(img_path)}")
            plates = process_image(img_path, plate_model, char_model)
            total_plates += plates
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🎉 Processing Complete!")
        print(f"   Images processed: {len(image_files)}")
        print(f"   Plates detected: {total_plates}")
        print(f"   Output folder: {config.OUTPUT_FOLDER}")
        print(f"   Database: {config.DB_NAME}")
        print(f"   Blockchain: {config.BLOCKCHAIN_FILE}")
        print(f"{'='*60}")
        
        logger.info(f"Batch processing complete: {total_plates} plates from {len(image_files)} images")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    # Allow specifying input folder as command-line argument
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    main(input_folder)