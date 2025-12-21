"""
License Plate Detection Model Training Script
Trains YOLOv11 model on EALPR dataset with detailed accuracy metrics.
"""
from ultralytics import YOLO
import json
import torch
from pathlib import Path

def main():
    print("=" * 60)
    print("Training License Plate Detection Model")
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
    
    # Load YOLOv11 Nano (fastest for Raspberry Pi)
    model = YOLO("yolo11n.pt") 
    
    # Train
    print("\nStarting training...")
    # Use smaller batch if GPU has memory issues
    batch_size = 8 if device != "cpu" else 16  # Smaller batch for GPU stability
    print(f"Batch size: {batch_size}")
    
    results = model.train(
        data="dataset/egyptian_plates_detection/data.yaml", 
        epochs=50,      # Increased from 30 for better accuracy
        imgsz=640,      
        batch=batch_size,
        project="SmartParking_Project",
        name="plate_detection_model",
        device=device,  # Auto-detected device
        verbose=True
    )
    
    # Validate on test set
    print("\n" + "=" * 60)
    print("Training Complete! Running Validation...")
    print("=" * 60)
    
    metrics = model.val()
    
    # Extract and display accuracy metrics
    print("\nModel Performance Metrics:")
    print("-" * 60)
    
    # mAP metrics (mean Average Precision)
    map50 = metrics.box.map50 * 100  # mAP at IoU=0.5
    map75 = metrics.box.map75 * 100  # mAP at IoU=0.75
    map50_95 = metrics.box.map * 100  # mAP at IoU=0.5:0.95
    
    print(f"Accuracy Metrics:")
    print(f"   mAP@0.5      : {map50:.2f}%   (Detection accuracy)")
    print(f"   mAP@0.75     : {map75:.2f}%   (Strict accuracy)")
    print(f"   mAP@0.5:0.95 : {map50_95:.2f}%   (Overall quality)")
    
    # Precision and Recall
    precision = metrics.box.mp * 100  # Mean precision
    recall = metrics.box.mr * 100     # Mean recall
    
    print(f"\nClassification Metrics:")
    print(f"   Precision    : {precision:.2f}%   (Correct detections)")
    print(f"   Recall       : {recall:.2f}%   (Plates found)")
    
    # F1 Score (harmonic mean of precision and recall)
    f1_score = 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"   F1 Score     : {f1_score:.2f}%   (Overall performance)")
    
    print("-" * 60)
    
    # Interpretation
    print("\nPerformance Grade:")
    if map50 >= 90:
        print("   >> EXCELLENT - Highly accurate!")
    elif map50 >= 80:
        print("   >> GOOD - Works well")
    elif map50 >= 70:
        print("   >> FAIR - Consider more training")
    else:
        print("   >> POOR - Retrain needed")
    
    # Save model with best weights
    print(f"\nSaving model to: plate_detector.pt")
    
    # Copy best weights to root
    best_weights = Path("SmartParking_Project/plate_detection_model/weights/best.pt")
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, "plate_detector.pt")
        print("Model saved successfully!")
    
    # Save metrics to JSON
    metrics_dict = {
        "model": "plate_detector",
        "mAP@0.5": f"{map50:.2f}%",
        "mAP@0.75": f"{map75:.2f}%",
        "mAP@0.5:0.95": f"{map50_95:.2f}%",
        "precision": f"{precision:.2f}%",
        "recall": f"{recall:.2f}%",
        "f1_score": f"{f1_score:.2f}%"
    }
    
    with open("plate_detector_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"Metrics saved to: plate_detector_metrics.json")
    print("=" * 60)
    print("Plate Detection Model Training Complete!")
    print("=" * 60)
    
    return metrics_dict

if __name__ == '__main__':
    main()