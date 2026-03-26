import cv2
import numpy as np

# ✅ FIX Bug 3: Only process detections above this confidence threshold
CONFIDENCE_THRESHOLD = 0.5


def crop_plates(results, frame, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Crop detected license plates from YOLO results with a small margin.
    Returns: list of (plate_image, (x1, y1, x2, y2), confidence)
    """
    plates = []

    # ✅ FIX Bug 4: Guard against empty results properly
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return plates

    img_height, img_width = frame.shape[:2]

    boxes = results[0].boxes

    for i, box in enumerate(boxes.xyxy):
        # ✅ FIX Bug 3: Skip low-confidence detections
        conf = float(boxes.conf[i])
        if conf < confidence_threshold:
            print(f"⚠️ Skipping low-confidence detection: {conf:.2f}")
            continue

        x1, y1, x2, y2 = map(int, box)

        # Add small margin (5%) for better OCR
        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(img_width, x2 + margin_x)
        y2 = min(img_height, y2 + margin_y)

        plate = frame[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        # ✅ FIX Bug 1 & 2: deskew is now enabled and uses corrected angle logic
        plate = deskew_plate(plate)

        plates.append((plate, (x1, y1, x2, y2), conf))

    return plates


def deskew_plate(plate):
    """
    Auto-rotate skewed license plates using minAreaRect.
    """
    if len(plate.shape) == 3:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # ✅ FIX Bug 1: coords from np.where are (row, col) = (y, x)
    # We need to flip to (x, y) for minAreaRect
    coords = np.column_stack(np.where(thresh > 0))  # shape: (N, 2) as (y, x)

    if len(coords) < 5:
        # Not enough points to reliably compute angle
        return plate

    # Flip to (x, y) for cv2 functions
    coords_xy = coords[:, ::-1]  # now (x, y)

    rect = cv2.minAreaRect(coords_xy)
    angle = rect[-1]

    # ✅ FIX Bug 1: Correct angle interpretation for minAreaRect
    # minAreaRect returns angle in [-90, 0). We normalise to a skew offset:
    # - If angle < -45, the rect is "standing up" → rotate by (90 + angle)
    # - Otherwise rotate by angle directly
    if angle < -45:
        angle = 90 + angle
    # Now angle is in (-45, 45], positive = clockwise skew

    if abs(angle) > 1:  # Only correct if skew is noticeable
        (h, w) = plate.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        plate = cv2.warpAffine(
            plate, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    return plate