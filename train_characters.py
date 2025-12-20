from ultralytics import YOLO

def main():
    # Load a fresh model
    model = YOLO("yolo11n.pt") 

    # Train
    results = model.train(
        data="dataset/egyptian_characters_detection/data.yaml", 
        epochs=50,       # 50 epochs should be enough for high accuracy
        imgsz=640,
        batch=16,
        project="SmartParking_Project",
        name="character_detector",
        device="0"
    )

if __name__ == '__main__':
    main()