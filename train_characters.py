"""
Character Recognition Model Training Script
Trains YOLOv11 model on Arabic characters with detailed accuracy metrics.
"""
from ultralytics import YOLO
import json
from pathlib import Path

def main():
    print("=" * 60)
    print("🚀 Training Character Recognition Model")
    print("=" * 60)
    
    # Load a fresh YOLOv11 model
    model = YOLO("yolo11n.pt") 

    # Train
    print("\n📊 Starting training...")
    results = model.train(
        data="dataset/egyptian_characters_detection/data.yaml", 
        epochs=100,      # Characters need more training
        imgsz=416,       # Smaller images for character detection
        batch=32,        # Larger batch for small objects
        project="SmartParking_Project",
        name="character_detector",
        device="0",      # Use GPU 0, or "cpu" for CPU training
        verbose=True
    )
    
    # Validate on test set
    print("\n" + "=" * 60)
    print("✅ Training Complete! Running Validation...")
    print("=" * 60)
    
    metrics = model.val()
    
    # Extract accuracy metrics
    print("\n📈 Model Performance Metrics:")
    print("-" * 60)
    
    # mAP metrics
    map50 = metrics.box.map50 * 100
    map75 = metrics.box.map75 * 100
    map50_95 = metrics.box.map * 100
    
    print(f"📊 Accuracy Metrics:")
    print(f"   mAP@0.5      : {map50:.2f}%   (Character detection accuracy)")
    print(f"   mAP@0.75     : {map75:.2f}%   (Stricter accuracy)")
    print(f"   mAP@0.5:0.95 : {map50_95:.2f}%   (Overall quality)")
    
    # Per-class metrics
    precision = metrics.box.mp * 100
    recall = metrics.box.mr * 100
    
    print(f"\n🎯 Classification Metrics:")
    print(f"   Precision    : {precision:.2f}%   (Correct character identifications)")
    print(f"   Recall       : {recall:.2f}%   (Characters successfully found)")
    
    # F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"   F1 Score     : {f1_score:.2f}%   (Balanced performance)")
    
    # Per-class accuracy (if available)
    if hasattr(metrics.box, 'ap_class_index'):
        print(f"\n📝 Per-Class Performance:")
        print(f"   Number of classes: {len(metrics.box.ap_class_index)}")
        print(f"   (See runs/detect/character_detector/results.csv for details)")
    
    print("-" * 60)
    
    # Interpretation
    print("\n💡 Performance Interpretation:")
    if map50 >= 85:
        print("   ✅ EXCELLENT - Character recognition is highly accurate!")
    elif map50 >= 75:
        print("   ✅ GOOD - Works well for Arabic text recognition")
    elif map50 >= 65:
        print("   ⚠️  FAIR - Consider training longer or balancing classes")
    else:
        print("   ❌ POOR - Retrain with more epochs or check annotations")
    
    # Save model
    print(f"\n💾 Saving model to: character_detector.pt")
    Path("SmartParking_Project/character_detector/weights/best.pt").rename("character_detector.pt")
    
    # Save metrics to JSON
    metrics_dict = {
        "model": "character_detector",
        "mAP@0.5": f"{map50:.2f}%",
        "mAP@0.75": f"{map75:.2f}%",
        "mAP@0.5:0.95": f"{map50_95:.2f}%",
        "precision": f"{precision:.2f}%",
        "recall": f"{recall:.2f}%",
        "f1_score": f"{f1_score:.2f}%" if precision + recall > 0 else "N/A",
        "num_classes": len(metrics.box.ap_class_index) if hasattr(metrics.box, 'ap_class_index') else 36
    }
    
    with open("character_detector_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"\n📊 Metrics saved to: character_detector_metrics.json")
    print("=" * 60)
    print("🎉 Character Recognition Model Training Complete!")
    print("=" * 60)
    
    return metrics_dict

if __name__ == '__main__':
    main()