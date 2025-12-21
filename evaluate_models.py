"""
Evaluate Existing Trained Models
Generates accuracy metrics for already trained models.
"""
import json
import os
from pathlib import Path

def evaluate_plate_detector():
    """Evaluate existing plate detector model."""
    print("\n" + "=" * 60)
    print("Evaluating Plate Detector...")
    print("=" * 60)
    
    if not os.path.exists("plate_detector.pt"):
        print("   >> plate_detector.pt not found!")
        return None
    
    try:
        from ultralytics import YOLO
        
        # Load trained model
        model = YOLO("plate_detector.pt")
        
        # Validate on dataset
        print("   Running validation...")
        metrics = model.val(data="dataset/egyptian_plates_detection/data.yaml")
        
        # Extract metrics
        map50 = metrics.box.map50 * 100
        map75 = metrics.box.map75 * 100
        map50_95 = metrics.box.map * 100
        precision = metrics.box.mp * 100
        recall = metrics.box.mr * 100
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_dict = {
            "model": "plate_detector",
            "mAP@0.5": f"{map50:.2f}%",
            "mAP@0.75": f"{map75:.2f}%",
            "mAP@0.5:0.95": f"{map50_95:.2f}%",
            "precision": f"{precision:.2f}%",
            "recall": f"{recall:.2f}%",
            "f1_score": f"{f1_score:.2f}%"
        }
        
        # Save metrics
        with open("plate_detector_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        
        print(f"   >> mAP@0.5: {map50:.2f}%")
        print(f"   >> Metrics saved to: plate_detector_metrics.json")
        return metrics_dict
        
    except Exception as e:
        print(f"   >> Error: {e}")
        print("   Note: Make sure dataset is prepared (python prepare_dataset.py)")
        return None

def evaluate_character_detector():
    """Evaluate existing character detector model."""
    print("\n" + "=" * 60)
    print("Evaluating Character Detector...")
    print("=" * 60)
    
    if not os.path.exists("character_detector.pt"):
        print("   >> character_detector.pt not found!")
        return None
    
    try:
        from ultralytics import YOLO
        
        # Load trained model
        model = YOLO("character_detector.pt")
        
        # Validate on dataset
        print("   Running validation...")
        metrics = model.val(data="dataset/egyptian_characters_detection/data.yaml")
        
        # Extract metrics
        map50 = metrics.box.map50 * 100
        map75 = metrics.box.map75 * 100
        map50_95 = metrics.box.map * 100
        precision = metrics.box.mp * 100
        recall = metrics.box.mr * 100
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_dict = {
            "model": "character_detector",
            "mAP@0.5": f"{map50:.2f}%",
            "mAP@0.75": f"{map75:.2f}%",
            "mAP@0.5:0.95": f"{map50_95:.2f}%",
            "precision": f"{precision:.2f}%",
            "recall": f"{recall:.2f}%",
            "f1_score": f"{f1_score:.2f}%",
            "num_classes": len(metrics.box.ap_class_index) if hasattr(metrics.box, 'ap_class_index') else 36
        }
        
        # Save metrics
        with open("character_detector_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        
        print(f"   >> mAP@0.5: {map50:.2f}%")
        print(f"   >> Metrics saved to: character_detector_metrics.json")
        return metrics_dict
        
    except Exception as e:
        print(f"   >> Error: {e}")
        print("   Note: Make sure dataset is prepared (python prepare_character_data.py)")
        return None

def evaluate_td3_agent():
    """Evaluate existing TD3 agent."""
    print("\n" + "=" * 60)
    print("Evaluating TD3 Agent...")
    print("=" * 60)
    
    if not os.path.exists("td3_actor.pth"):
        print("   >> td3_actor.pth not found!")
        return None
    
    try:
        import torch
        import numpy as np
        from td3_parking import TD3ParkingAgent
        import config
        
        # Load agent
        agent = TD3ParkingAgent()
        
        if not agent.is_trained:
            print("   >> Model loaded but not trained!")
            return None
        
        # Create test environment
        print("   Running 100 test scenarios...")
        from train_simulation import VirtualParkingLot
        env = VirtualParkingLot()
        
        num_tests = 100
        correct_decisions = 0
        optimal_decisions = 0
        test_rewards = []
        
        for i in range(num_tests):
            state = env.reset()
            sensors = state[:3].astype(int).tolist()
            dest_loc = state[3]
            
            # Get agent's decision
            result = agent.select_spot(sensors, "A" if dest_loc < 2.0 else "B")
            chosen_spot = result['spot_id'] - 1  # Convert ID to index
            
            # Calculate reward
            if sensors[chosen_spot] == 1:
                correct_decisions += 1
                spot_loc = [1.0, 2.0, 3.0][chosen_spot]
                distance = abs(spot_loc - dest_loc)
                reward = 10.0 - (distance * 2.0)
                test_rewards.append(reward)
                
                # Check if optimal
                best_dist = float('inf')
                for idx, is_free in enumerate(sensors):
                    if is_free == 1:
                        dist = abs([1.0, 2.0, 3.0][idx] - dest_loc)
                        if dist < best_dist:
                            best_dist = dist
                
                if distance == best_dist:
                    optimal_decisions += 1
            else:
                test_rewards.append(-100)
        
        # Calculate percentages
        avg_reward = np.mean(test_rewards)
        correctness_rate = (correct_decisions / num_tests) * 100
        optimality_rate = (optimal_decisions / num_tests) * 100
        
        metrics_dict = {
            "model": "td3_parking_agent",
            "average_reward": f"{avg_reward:.2f}",
            "correctness_rate": f"{correctness_rate:.1f}%",
            "optimality_rate": f"{optimality_rate:.1f}%",
            "tests_run": num_tests
        }
        
        # Save metrics
        with open("td3_agent_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        
        print(f"   >> Correctness: {correctness_rate:.1f}%")
        print(f"   >> Optimality: {optimality_rate:.1f}%")
        print(f"   >> Metrics saved to: td3_agent_metrics.json")
        return metrics_dict
        
    except Exception as e:
        print(f"   >> Error: {e}")
        return None

def main():
    """Evaluate all existing models."""
    print("\n" + "=" * 60)
    print("EVALUATING EXISTING TRAINED MODELS")
    print("=" * 60)
    
    # Check which models exist
    models_exist = {
        "plate_detector.pt": os.path.exists("plate_detector.pt"),
        "character_detector.pt": os.path.exists("character_detector.pt"),
        "td3_actor.pth": os.path.exists("td3_actor.pth")
    }
    
    print("\nModel Status:")
    for model, exists in models_exist.items():
        status = "[FOUND]" if exists else "[NOT FOUND]"
        print(f"   {model:25s} : {status}")
    
    if not any(models_exist.values()):
        print("\n>> No trained models found!")
        print("   Train models first:")
        print("      python train_plates.py")
        print("      python train_characters.py")
        print("      python train_simulation.py")
        return
    
    # Evaluate each model
    plate_metrics = evaluate_plate_detector() if models_exist["plate_detector.pt"] else None
    char_metrics = evaluate_character_detector() if models_exist["character_detector.pt"] else None
    td3_metrics = evaluate_td3_agent() if models_exist["td3_actor.pth"] else None
    
    # Display summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print("\nView detailed metrics with:")
    print("   python view_metrics.py")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
