from ultralytics import YOLO

def main():
    # Load YOLOv11 Nano (fastest for projects)
    model = YOLO("yolo11n.pt") 

    # Train
    results = model.train(
        data="dataset/egyptian_plates_detection/data.yaml", 
        epochs=30,      # 30 epochs is usually enough for a simple plate detector
        imgsz=640,      
        batch=16,
        project="SmartParking_Project",
        name="plate_detection_model",
        device="0"
    )

if __name__ == '__main__':
    main()