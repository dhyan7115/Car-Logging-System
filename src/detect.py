import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")

def detect_image(image_path):
    img = cv2.imread(image_path)

    results = model(img)

    annotated = results[0].plot()

    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detect_image("test.jpg")