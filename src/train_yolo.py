from ultralytics import YOLO

def train():
    # Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="data.yaml",   # your dataset config
        epochs=50,
        imgsz=640,
        batch=16,
        name="plate_detector"
    )

if __name__ == "__main__":
    train()