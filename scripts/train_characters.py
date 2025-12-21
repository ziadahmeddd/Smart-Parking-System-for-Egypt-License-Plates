"""
Character Recognition Model Training Script
Trains YOLOv11 model on Arabic characters with detailed accuracy metrics.
"""
from ultralytics import YOLO
import json
import torch
from pathlib import Path

def main():
    print("=" * 60)
    print("Training Character Recognition Model")
    print("=" * 60)
    
    # Auto-detect device (GPU if available, else CPU)
    # Force CPU if GPU has issues - uncomment line below
    # device = "cpu"  # Force CPU training
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device.upper()}")
    if device == "cpu":
        print("Note: Training on CPU will be slower (5-10x)")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load a fresh YOLOv11 model
    model = YOLO("yolo11n.pt") 

    # Train
    print("\nStarting training...")
    # Use smaller batch if GPU has memory issues
    batch_size = 16 if device != "cpu" else 32  # Smaller batch for GPU stability
    print(f"Batch size: {batch_size}")
    
    results = model.train(
        data="dataset/egyptian_characters_detection/data.yaml", 
        epochs=100,      # Characters need more training
        imgsz=416,       # Smaller images for character detection
        batch=batch_size,
        project="SmartParking_Project",
        name="character_detector",
        device=device,   # Auto-detected device
        verbose=True
    )
    
    # Validate on test set
    print("\n" + "=" * 60)
    print("Training Complete! Running Validation...")
    print("=" * 60)
    
    metrics = model.val()
    
    # Extract accuracy metrics
    print("\nModel Performance Metrics:")
    print("-" * 60)
    
    # mAP metrics
    map50 = metrics.box.map50 * 100
    map75 = metrics.box.map75 * 100
    map50_95 = metrics.box.map * 100
    
    print(f"Accuracy Metrics:")
    print(f"   mAP@0.5      : {map50:.2f}%   (Character detection)")
    print(f"   mAP@0.75     : {map75:.2f}%   (Strict accuracy)")
    print(f"   mAP@0.5:0.95 : {map50_95:.2f}%   (Overall quality)")
    
    # Per-class metrics
    precision = metrics.box.mp * 100
    recall = metrics.box.mr * 100
    
    print(f"\nClassification Metrics:")
    print(f"   Precision    : {precision:.2f}%   (Correct characters)")
    print(f"   Recall       : {recall:.2f}%   (Characters found)")
    
    # F1 Score
    f1_score = 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"   F1 Score     : {f1_score:.2f}%   (Balanced performance)")
    
    # Per-class accuracy (if available)
    num_classes = len(metrics.box.ap_class_index) if hasattr(metrics.box, 'ap_class_index') else 36
    print(f"\nNumber of classes: {num_classes}")
    
    print("-" * 60)
    
    # Interpretation
    print("\nPerformance Grade:")
    if map50 >= 85:
        print("   >> EXCELLENT - Highly accurate!")
    elif map50 >= 75:
        print("   >> GOOD - Works well")
    elif map50 >= 65:
        print("   >> FAIR - Consider more training")
    else:
        print("   >> POOR - Retrain needed")
    
    # Save model
    print(f"\nSaving model to: character_detector.pt")
    best_weights = Path("SmartParking_Project/character_detector/weights/best.pt")
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, "character_detector.pt")
        print("Model saved successfully!")
    
    # Save metrics to JSON
    metrics_dict = {
        "model": "character_detector",
        "mAP@0.5": f"{map50:.2f}%",
        "mAP@0.75": f"{map75:.2f}%",
        "mAP@0.5:0.95": f"{map50_95:.2f}%",
        "precision": f"{precision:.2f}%",
        "recall": f"{recall:.2f}%",
        "f1_score": f"{f1_score:.2f}%",
        "num_classes": num_classes
    }
    
    with open("character_detector_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"Metrics saved to: character_detector_metrics.json")
    print("=" * 60)
    print("Character Recognition Model Training Complete!")
    print("=" * 60)
    
    return metrics_dict

if __name__ == '__main__':
    main()