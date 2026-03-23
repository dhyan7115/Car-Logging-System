import cv2
import pytesseract
import re

def preprocess(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Resize for better OCR
    gray = cv2.resize(gray, None, fx=2, fy=2)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    return thresh

def extract_text(plate_img):
    try:
        processed = preprocess(plate_img)

        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(processed, config=config)

        text = re.sub(r'[^A-Z0-9]', '', text.upper())

        return text

    except Exception as e:
        print("OCR Error:", e)
        return ""