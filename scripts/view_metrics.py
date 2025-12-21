"""
View Training Metrics - Display all model accuracy percentages
Shows performance metrics for plate detector, character detector, and TD3 agent.
"""
import json
import os
import sys
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_metrics(filename: str) -> dict:
    """Load metrics from JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def display_metrics():
    """Display all training metrics in a formatted table."""
    print("\n" + "=" * 70)
    print("SMART PARKING SYSTEM - MODEL PERFORMANCE METRICS")
    print("=" * 70)
    
    # Load all metrics
    plate_metrics = load_metrics("plate_detector_metrics.json")
    char_metrics = load_metrics("character_detector_metrics.json")
    td3_metrics = load_metrics("td3_agent_metrics.json")
    
    # Display Plate Detector Metrics
    print("\n[LICENSE PLATE DETECTOR - YOLOv11]")
    print("-" * 70)
    if plate_metrics:
        print(f"   mAP@0.5 (Detection Accuracy)  : {plate_metrics['mAP@0.5']}")
        print(f"   mAP@0.75 (Strict Accuracy)    : {plate_metrics['mAP@0.75']}")
        print(f"   mAP@0.5:0.95 (Overall Quality): {plate_metrics['mAP@0.5:0.95']}")
        print(f"   Precision                     : {plate_metrics['precision']}")
        print(f"   Recall                        : {plate_metrics['recall']}")
        print(f"   F1 Score                      : {plate_metrics['f1_score']}")
        
        # Grade
        map_val = float(plate_metrics['mAP@0.5'].replace('%', ''))
        if map_val >= 90:
            grade = "✅ EXCELLENT"
        elif map_val >= 80:
            grade = "✅ GOOD"
        elif map_val >= 70:
            grade = "⚠️  FAIR"
        else:
            grade = "❌ POOR"
        print(f"\n   Performance Grade: {grade}")
    else:
        print("   ⚠️  Metrics not found. Train model first:")
        print("      python train_plates.py")
    
    # Display Character Detector Metrics
    print("\n\n[CHARACTER RECOGNIZER - YOLOv11]")
    print("-" * 70)
    if char_metrics:
        print(f"   mAP@0.5 (Character Accuracy)  : {char_metrics['mAP@0.5']}")
        print(f"   mAP@0.75 (Strict Accuracy)    : {char_metrics['mAP@0.75']}")
        print(f"   mAP@0.5:0.95 (Overall Quality): {char_metrics['mAP@0.5:0.95']}")
        print(f"   Precision                     : {char_metrics['precision']}")
        print(f"   Recall                        : {char_metrics['recall']}")
        print(f"   F1 Score                      : {char_metrics['f1_score']}")
        print(f"   Number of Classes             : {char_metrics['num_classes']}")
        
        # Grade
        map_val = float(char_metrics['mAP@0.5'].replace('%', ''))
        if map_val >= 85:
            grade = "✅ EXCELLENT"
        elif map_val >= 75:
            grade = "✅ GOOD"
        elif map_val >= 65:
            grade = "⚠️  FAIR"
        else:
            grade = "❌ POOR"
        print(f"\n   Performance Grade: {grade}")
    else:
        print("   ⚠️  Metrics not found. Train model first:")
        print("      python train_characters.py")
    
    # Display TD3 Agent Metrics
    print("\n\n[TD3 PARKING ALLOCATION AGENT]")
    print("-" * 70)
    if td3_metrics:
        print(f"   Average Reward                : {td3_metrics['average_reward']}/10")
        print(f"   Correctness Rate              : {td3_metrics['correctness_rate']}")
        print(f"   Optimality Rate               : {td3_metrics['optimality_rate']}")
        print(f"   Tests Run                     : {td3_metrics['tests_run']}")
        
        # Calculate percentage score
        correctness = float(td3_metrics['correctness_rate'].replace('%', ''))
        optimality = float(td3_metrics['optimality_rate'].replace('%', ''))
        
        # Grade
        if correctness >= 95 and optimality >= 80:
            grade = "✅ EXCELLENT"
        elif correctness >= 90 and optimality >= 70:
            grade = "✅ GOOD"
        elif correctness >= 80:
            grade = "⚠️  FAIR"
        else:
            grade = "❌ POOR"
        print(f"\n   Performance Grade: {grade}")
    else:
        print("   ⚠️  Metrics not found. Train model first:")
        print("      python train_simulation.py")
    
    # Overall System Performance
    print("\n\n" + "=" * 70)
    print("OVERALL SYSTEM READINESS")
    print("=" * 70)
    
    all_trained = plate_metrics and char_metrics and td3_metrics
    
    if all_trained:
        print("   ✅ Plate Detector      : Trained")
        print("   ✅ Character Detector  : Trained")
        print("   ✅ TD3 Agent          : Trained")
        print("\n   >> System is READY for deployment!")
        
        # Calculate overall score
        plate_map = float(plate_metrics['mAP@0.5'].replace('%', ''))
        char_map = float(char_metrics['mAP@0.5'].replace('%', ''))
        td3_correct = float(td3_metrics['correctness_rate'].replace('%', ''))
        
        overall_score = (plate_map + char_map + td3_correct) / 3
        print(f"\n   Overall System Score: {overall_score:.1f}%")
        
        if overall_score >= 85:
            print("   >> EXCELLENT - Production ready!")
        elif overall_score >= 75:
            print("   >> GOOD - Ready for testing")
        elif overall_score >= 65:
            print("   >> FAIR - Consider retraining")
        else:
            print("   >> NEEDS IMPROVEMENT - Retrain models")
    else:
        missing = []
        if not plate_metrics:
            print("   ❌ Plate Detector      : Not trained")
            missing.append("train_plates.py")
        if not char_metrics:
            print("   ❌ Character Detector  : Not trained")
            missing.append("train_characters.py")
        if not td3_metrics:
            print("   ❌ TD3 Agent          : Not trained")
            missing.append("train_simulation.py")
        
        print(f"\n   ⚠️  Train missing models:")
        for script in missing:
            print(f"      python {script}")
    
    print("=" * 70 + "\n")
    
    return {
        "plate_detector": plate_metrics,
        "character_detector": char_metrics,
        "td3_agent": td3_metrics
    }

if __name__ == "__main__":
    display_metrics()
