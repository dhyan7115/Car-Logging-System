import cv2
import sys
from ultralytics import YOLO

from crop_plate import crop_plates
from ocr import extract_text
from logger import log_plate
from filter import is_allowed

# ==============================
# CHECK INPUT ARGUMENT
# ==============================
image_path = "test.jpg"

# ==============================
# LOAD MODEL
# ==============================
model = YOLO("models/best.pt")

# ==============================
# LOAD IMAGE
# ==============================
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found:", image_path)
    exit()

# ==============================
# RUN DETECTION
# ==============================
results = model(img, conf=0.1)

print("🔍 Detections:", results[0].boxes)

# Show YOLO output
cv2.imshow("YOLO Detection", results[0].plot())
cv2.waitKey(0)

# ==============================
# CROP PLATES
# ==============================
plates = crop_plates(results, img)

print("📦 Plates found:", len(plates))

# ==============================
# PROCESS EACH PLATE
# ==============================
for plate_img, (x1, y1, x2, y2) in plates:

    cv2.imshow("Plate", plate_img)
    cv2.waitKey(500)

    text = extract_text(plate_img)
    print("🧠 OCR:", text)

    if len(text) >= 5:
        print("✅ Logging:", text)

        if is_allowed(text):
            log_plate(text)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# ==============================
# FINAL DISPLAY
# ==============================
cv2.imshow("Final Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()