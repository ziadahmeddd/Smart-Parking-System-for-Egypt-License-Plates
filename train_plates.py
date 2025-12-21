"""
License Plate Detection Model Training Script
Trains YOLOv11 model on EALPR dataset with detailed accuracy metrics.
"""
from ultralytics import YOLO
import json
from pathlib import Path

def main():
    print("=" * 60)
    print("🚀 Training License Plate Detection Model")
    print("=" * 60)
    
    # Load YOLOv11 Nano (fastest for Raspberry Pi)
    model = YOLO("yolo11n.pt") 
    
    # Train
    print("\n📊 Starting training...")
    results = model.train(
        data="dataset/egyptian_plates_detection/data.yaml", 
        epochs=50,      # Increased from 30 for better accuracy
        imgsz=640,      
        batch=16,
        project="SmartParking_Project",
        name="plate_detection_model",
        device="0",     # Use GPU 0, or "cpu" for CPU training
        verbose=True
    )
    
    # Validate on test set
    print("\n" + "=" * 60)
    print("✅ Training Complete! Running Validation...")
    print("=" * 60)
    
    metrics = model.val()
    
    # Extract and display accuracy metrics
    print("\n📈 Model Performance Metrics:")
    print("-" * 60)
    
    # mAP metrics (mean Average Precision)
    map50 = metrics.box.map50 * 100  # mAP at IoU=0.5
    map75 = metrics.box.map75 * 100  # mAP at IoU=0.75
    map50_95 = metrics.box.map * 100  # mAP at IoU=0.5:0.95
    
    print(f"📊 Accuracy Metrics:")
    print(f"   mAP@0.5      : {map50:.2f}%   (Detection accuracy at 50% overlap)")
    print(f"   mAP@0.75     : {map75:.2f}%   (Stricter detection accuracy)")
    print(f"   mAP@0.5:0.95 : {map50_95:.2f}%   (Overall detection quality)")
    
    # Precision and Recall
    precision = metrics.box.mp * 100  # Mean precision
    recall = metrics.box.mr * 100     # Mean recall
    
    print(f"\n🎯 Classification Metrics:")
    print(f"   Precision    : {precision:.2f}%   (How many detections are correct)")
    print(f"   Recall       : {recall:.2f}%   (How many plates are found)")
    
    # F1 Score (harmonic mean of precision and recall)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"   F1 Score     : {f1_score:.2f}%   (Overall performance)")
    
    print("-" * 60)
    
    # Interpretation
    print("\n💡 Performance Interpretation:")
    if map50 >= 90:
        print("   ✅ EXCELLENT - Model is highly accurate!")
    elif map50 >= 80:
        print("   ✅ GOOD - Model works well for most cases")
    elif map50 >= 70:
        print("   ⚠️  FAIR - Consider training longer or with more data")
    else:
        print("   ❌ POOR - Retrain with more epochs or check dataset")
    
    # Save model with best weights
    print(f"\n💾 Saving model to: plate_detector.pt")
    model.export(format='torchscript', save_dir='.')
    
    # Also save as .pt for easy loading
    Path("SmartParking_Project/plate_detection_model/weights/best.pt").rename("plate_detector.pt")
    
    # Save metrics to JSON
    metrics_dict = {
        "model": "plate_detector",
        "mAP@0.5": f"{map50:.2f}%",
        "mAP@0.75": f"{map75:.2f}%",
        "mAP@0.5:0.95": f"{map50_95:.2f}%",
        "precision": f"{precision:.2f}%",
        "recall": f"{recall:.2f}%",
        "f1_score": f"{f1_score:.2f}%" if precision + recall > 0 else "N/A"
    }
    
    with open("plate_detector_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"\n📊 Metrics saved to: plate_detector_metrics.json")
    print("=" * 60)
    print("🎉 Plate Detection Model Training Complete!")
    print("=" * 60)
    
    return metrics_dict

if __name__ == '__main__':
    main()