import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os

from crop_plate import crop_plates
from ocr import extract_text
from logger import VehicleLogger
from filter import PlateFilter

# ==============================
# CONFIGURATION
# ==============================
class Config:
    MODEL_PATH = "models/best.pt"
    # ✅ FIX Bug 2: One confidence threshold used everywhere.
    # crop_plates() will use this same value, so no detections are silently dropped.
    CONFIDENCE = 0.5
    COOLDOWN_SECONDS = 5
    ENTRY_ZONE = (0, 300)    # y range for entry (top half)
    EXIT_ZONE  = (300, 600)  # y range for exit  (bottom half)

# ==============================
# INITIALIZATION
# ==============================
model = YOLO(Config.MODEL_PATH)
logger = VehicleLogger()
plate_filter = PlateFilter(cooldown_seconds=Config.COOLDOWN_SECONDS)

# ✅ FIX Bug 4: Track previous y-positions per plate index to infer movement direction
_prev_y_positions = {}


def determine_direction(plate_id, y_center):
    """
    Determine entry/exit direction using:
    1. Movement direction (current y vs previous y) — more reliable
    2. Zone-based fallback if no previous position exists
    """
    # ✅ FIX Bug 4: Use prev_y to detect movement direction
    prev_y = _prev_y_positions.get(plate_id)
    _prev_y_positions[plate_id] = y_center

    if prev_y is not None:
        delta = y_center - prev_y
        if delta > 5:       # Moving downward → exiting
            return "exit"
        elif delta < -5:    # Moving upward → entering
            return "entry"
        # If delta is small, fall through to zone-based detection

    # Zone-based fallback
    if Config.ENTRY_ZONE[0] <= y_center <= Config.ENTRY_ZONE[1]:
        return "entry"
    elif Config.EXIT_ZONE[0] <= y_center <= Config.EXIT_ZONE[1]:
        return "exit"

    return None


def _draw_label(img, label, x1, y1, color):
    """Draw a filled rectangle label above a bounding box."""
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def process_image(image_path):
    """
    Process a single image for license plate detection.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Image not found: {image_path}")
        return

    # ✅ FIX Bug 5: Use image basename in all debug output filenames
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'='*50}")
    print(f"Processing: {image_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    results = model(img, conf=Config.CONFIDENCE)
    print(f"🔍 Detections found: {len(results[0].boxes) if results[0].boxes else 0}")

    yolo_output = results[0].plot()
    cv2.imwrite(f"{base_name}_yolo.jpg", yolo_output)

    # ✅ FIX Bug 1: Unpack 3-tuple (plate, bbox, conf) from updated crop_plates()
    plates = crop_plates(results, img, confidence_threshold=Config.CONFIDENCE)
    print(f"📦 Plates cropped: {len(plates)}")

    processed_count = 0

    for i, (plate_img, (x1, y1, x2, y2), conf) in enumerate(plates):
        print(f"\n--- Processing Plate {i+1} (conf: {conf:.2f}) ---")

        # ✅ FIX Bug 5: Unique debug filenames per image + plate index
        cv2.imwrite(f"{base_name}_plate_{i}.jpg", plate_img)

        plate_center_y = (y1 + y2) // 2
        plate_center_x = (x1 + x2) // 2
        print(f"📐 Plate position: x={plate_center_x}, y={plate_center_y}")

        # ✅ FIX Bug 4: Pass a stable plate_id for tracking
        direction = determine_direction(plate_id=i, y_center=plate_center_y)

        if direction is None:
            direction = "entry"
            print(f"⚠️ Could not determine direction — defaulting to: {direction}")
        else:
            print(f"📍 Direction: {direction.upper()}")

        text = extract_text(plate_img)

        if text:
            print(f"🧠 OCR Result: {text}")

            allowed, final_direction = plate_filter.is_allowed(text, direction)

            if allowed:
                if final_direction == "entry":
                    logger.log_entry(text)
                elif final_direction == "exit":
                    logger.log_exit(text)
                processed_count += 1
            else:
                print(f"⏱️ Skipping {text} — cooldown active")

            color = (0, 255, 0) if direction == "entry" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            _draw_label(img, f"{text} ({direction})", x1, y1, color)
        else:
            print(f"⚠️ No valid text extracted from plate {i+1}")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            _draw_label(img, "OCR Failed", x1, y1, (0, 0, 255))

    output_path = f"{base_name}_final.jpg"
    cv2.imwrite(output_path, img)

    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total detections : {len(plates)}")
    print(f"Successfully processed: {processed_count}")

    try:
        active_vehicles = logger.get_status()
        if active_vehicles:
            print(f"Active vehicles inside: {len(active_vehicles)}")
            for v in active_vehicles:
                print(f"   - {v}")
        else:
            print("Active vehicles: None")
    except Exception as e:
        print(f"⚠️ Could not get active vehicles: {e}")

    print(f"\n📁 Output: {output_path}")
    print(f"{'='*50}\n")

    return img


def process_video(video_path, process_every_n_frames=10):
    """
    Process video for license plate detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    print(f"\n{'='*50}")
    print(f"Processing Video: {video_path}")
    print(f"FPS: {fps}, Resolution: {width}x{height}")
    print(f"Processing every {process_every_n_frames} frames")
    print(f"{'='*50}\n")

    frame_count = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            processed_frames += 1
            print(f"\n📹 Frame {frame_count}")

            results = model(frame, conf=Config.CONFIDENCE)

            # ✅ FIX Bug 1: Unpack 3-tuple from updated crop_plates()
            plates = crop_plates(results, frame, confidence_threshold=Config.CONFIDENCE)

            if plates:
                print(f"   Found {len(plates)} plate(s)")

            for i, (plate_img, (x1, y1, x2, y2), conf) in enumerate(plates):
                text = extract_text(plate_img)

                if text:
                    plate_center_y = (y1 + y2) // 2
                    # ✅ FIX Bug 4: Use frame-scoped plate id for tracking
                    direction = determine_direction(plate_id=f"video_{i}", y_center=plate_center_y)
                    if direction is None:
                        direction = "entry"

                    allowed, final_direction = plate_filter.is_allowed(text, direction)

                    if allowed:
                        if final_direction == "entry":
                            logger.log_entry(text)
                        elif final_direction == "exit":
                            logger.log_exit(text)
                        print(f"   ✅ {text} — {final_direction.upper()}")

                    color = (0, 255, 0) if direction == "entry" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{text} ({direction})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ✅ FIX Bug 3: Always write every frame to keep output video at correct fps
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"VIDEO PROCESSING COMPLETE")
    print(f"Frames: {processed_frames} processed / {frame_count} total")
    print(f"Output: output_video.mp4")
    print(f"{'='*50}\n")


def process_multiple_images(image_folder):
    """
    Process all images in a folder.
    """
    if not os.path.exists(image_folder):
        print(f"❌ Folder not found: {image_folder}")
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in os.listdir(image_folder)
              if any(f.lower().endswith(ext) for ext in image_extensions)]

    if not images:
        print(f"❌ No images found in {image_folder}")
        return

    print(f"\n{'='*50}")
    print(f"Processing {len(images)} images from: {image_folder}")
    print(f"{'='*50}\n")

    for idx, image_file in enumerate(images, 1):
        print(f"\n📷 [{idx}/{len(images)}] {image_file}")
        process_image(os.path.join(image_folder, image_file))
        if idx < len(images):
            print("\n" + "-"*50)

    print(f"\n{'='*50}")
    print("ALL IMAGES PROCESSED")
    print(f"{'='*50}\n")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    image_path = "mycar.jpg"

    if os.path.exists(image_path):
        process_image(image_path)
    else:
        print(f"❌ Image not found: {image_path}")
        print(f"Files in current directory: {os.listdir('.')}")

    # Uncomment for video:
    # process_video("test_video.mp4", process_every_n_frames=10)

    # Uncomment for batch:
    # process_multiple_images("test_images")