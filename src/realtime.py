import cv2
from ultralytics import YOLO

from crop_plate import crop_plates
from ocr import recognize_plate
from logger import log_entry_exit
from filter import is_allowed

# ==============================
# CONFIG
# ==============================
YOLO_MODEL_PATH = "models/best.pt"

# ==============================
# LOAD MODEL
# ==============================
print("[INFO] Loading YOLO model...")
model = YOLO(YOLO_MODEL_PATH)

# ==============================
# MAIN FUNCTION
# ==============================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("[INFO] System started. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)

        # Crop detected plates
        plates = crop_plates(results, frame)

        for plate_img in plates:
            # OCR
            text = recognize_plate(plate_img)

            # Basic validation
            if len(text) >= 8:
                print("[DETECTED]", text)

                # Avoid duplicate logging
                if is_allowed(text):
                    log_entry_exit(text)

                # Display detected text
                cv2.putText(
                    frame,
                    text,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        # Draw YOLO bounding boxes
        annotated = results[0].plot()

        cv2.imshow("Number Plate Detection System", annotated)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()