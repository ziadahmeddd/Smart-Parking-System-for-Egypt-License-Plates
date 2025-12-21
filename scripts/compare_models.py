"""
Model Comparison Tool
Compare performance between different model versions.
"""
import json
import os
from datetime import datetime

def load_metrics(filename):
    """Load metrics from JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def display_comparison():
    """Display side-by-side comparison of model versions."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - Old vs New")
    print("=" * 80)
    
    # Check for comparison files
    old_plate = load_metrics("plate_detector_metrics_old.json")
    new_plate = load_metrics("plate_detector_metrics.json")
    
    old_char = load_metrics("character_detector_metrics_old.json")
    new_char = load_metrics("character_detector_metrics.json")
    
    old_td3 = load_metrics("td3_agent_metrics_old.json")
    new_td3 = load_metrics("td3_agent_metrics.json")
    
    # Plate Detector Comparison
    print("\n[PLATE DETECTOR COMPARISON]")
    print("-" * 80)
    if old_plate and new_plate:
        print(f"{'Metric':<20} {'Old Model':<20} {'New Model':<20} {'Change':<15}")
        print("-" * 80)
        
        # Compare mAP@0.5
        old_map = float(old_plate['mAP@0.5'].replace('%', ''))
        new_map = float(new_plate['mAP@0.5'].replace('%', ''))
        diff = new_map - old_map
        symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{'mAP@0.5':<20} {old_plate['mAP@0.5']:<20} {new_plate['mAP@0.5']:<20} {symbol} {abs(diff):.2f}%")
        
        # Compare Precision
        old_prec = float(old_plate['precision'].replace('%', ''))
        new_prec = float(new_plate['precision'].replace('%', ''))
        diff = new_prec - old_prec
        symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{'Precision':<20} {old_plate['precision']:<20} {new_plate['precision']:<20} {symbol} {abs(diff):.2f}%")
        
        # Compare Recall
        old_rec = float(old_plate['recall'].replace('%', ''))
        new_rec = float(new_plate['recall'].replace('%', ''))
        diff = new_rec - old_rec
        symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{'Recall':<20} {old_plate['recall']:<20} {new_plate['recall']:<20} {symbol} {abs(diff):.2f}%")
        
        # Verdict
        print("\nVerdict:")
        if new_map > old_map:
            print("   >> NEW MODEL IS BETTER!")
        elif new_map < old_map:
            print("   >> OLD MODEL WAS BETTER")
        else:
            print("   >> SAME PERFORMANCE")
    else:
        print("   >> No old model found. Save current metrics as baseline:")
        print("      python save_as_baseline.py")
    
    # Character Detector Comparison
    print("\n\n[CHARACTER DETECTOR COMPARISON]")
    print("-" * 80)
    if old_char and new_char:
        print(f"{'Metric':<20} {'Old Model':<20} {'New Model':<20} {'Change':<15}")
        print("-" * 80)
        
        # Compare mAP@0.5
        old_map = float(old_char['mAP@0.5'].replace('%', ''))
        new_map = float(new_char['mAP@0.5'].replace('%', ''))
        diff = new_map - old_map
        symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{'mAP@0.5':<20} {old_char['mAP@0.5']:<20} {new_char['mAP@0.5']:<20} {symbol} {abs(diff):.2f}%")
        
        # Compare Precision
        old_prec = float(old_char['precision'].replace('%', ''))
        new_prec = float(new_char['precision'].replace('%', ''))
        diff = new_prec - old_prec
        symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{'Precision':<20} {old_char['precision']:<20} {new_char['precision']:<20} {symbol} {abs(diff):.2f}%")
        
        # Compare Recall
        old_rec = float(old_char['recall'].replace('%', ''))
        new_rec = float(new_char['recall'].replace('%', ''))
        diff = new_rec - old_rec
        symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{'Recall':<20} {old_char['recall']:<20} {new_char['recall']:<20} {symbol} {abs(diff):.2f}%")
        
        # Verdict
        print("\nVerdict:")
        if new_map > old_map:
            print("   >> NEW MODEL IS BETTER!")
        elif new_map < old_map:
            print("   >> OLD MODEL WAS BETTER")
        else:
            print("   >> SAME PERFORMANCE")
    else:
        print("   >> No old model found. Save current metrics as baseline:")
        print("      python save_as_baseline.py")
    
    # TD3 Agent Comparison
    print("\n\n[TD3 AGENT COMPARISON]")
    print("-" * 80)
    if old_td3 and new_td3:
        print(f"{'Metric':<20} {'Old Model':<20} {'New Model':<20} {'Change':<15}")
        print("-" * 80)
        
        # Compare Correctness
        old_corr = float(old_td3['correctness_rate'].replace('%', ''))
        new_corr = float(new_td3['correctness_rate'].replace('%', ''))
        diff = new_corr - old_corr
        symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{'Correctness':<20} {old_td3['correctness_rate']:<20} {new_td3['correctness_rate']:<20} {symbol} {abs(diff):.2f}%")
        
        # Compare Optimality
        old_opt = float(old_td3['optimality_rate'].replace('%', ''))
        new_opt = float(new_td3['optimality_rate'].replace('%', ''))
        diff = new_opt - old_opt
        symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"{'Optimality':<20} {old_td3['optimality_rate']:<20} {new_td3['optimality_rate']:<20} {symbol} {abs(diff):.2f}%")
        
        # Verdict
        print("\nVerdict:")
        if new_corr >= old_corr and new_opt >= old_opt:
            print("   >> NEW MODEL IS BETTER!")
        elif new_corr < old_corr or new_opt < old_opt:
            print("   >> OLD MODEL WAS BETTER")
        else:
            print("   >> MIXED RESULTS")
    else:
        print("   >> No old model found. Save current metrics as baseline:")
        print("      python save_as_baseline.py")
    
    print("\n" + "=" * 80 + "\n")

def save_as_baseline():
    """Save current metrics as baseline for future comparisons."""
    print("\nSaving current metrics as baseline...")
    
    files_to_backup = [
        ("plate_detector_metrics.json", "plate_detector_metrics_old.json"),
        ("character_detector_metrics.json", "character_detector_metrics_old.json"),
        ("td3_agent_metrics.json", "td3_agent_metrics_old.json")
    ]
    
    saved_count = 0
    for current, backup in files_to_backup:
        if os.path.exists(current):
            import shutil
            shutil.copy(current, backup)
            print(f"   >> Saved {current} -> {backup}")
            saved_count += 1
        else:
            print(f"   >> {current} not found")
    
    if saved_count > 0:
        print(f"\n✅ Saved {saved_count} baseline metrics!")
        print("   Now you can train new models and compare with: python compare_models.py")
    else:
        print("\n❌ No metrics found. Train models first:")
        print("   python evaluate_models.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save_as_baseline()
    else:
        display_comparison()
