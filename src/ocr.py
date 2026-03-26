import easyocr
import cv2
import numpy as np
import re

# Load once
reader = easyocr.Reader(['en'], gpu=False)

# Indian States and Union Territories codes
INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH',
    'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ',
    'SK', 'TN', 'TS', 'TR', 'UK', 'UP', 'WB',
    'CH', 'DD', 'DL', 'DN', 'LD', 'PY', 'LA'
}

# ✅ FIX: Separate correction maps for letter zones vs digit zones
# In letter zones: digits that look like letters → convert to letters
DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'E',
    '4': 'A', '5': 'S', '6': 'G', '8': 'B'
}

# In digit zones: letters that look like digits → convert to digits
# ⚠️ Do NOT add L→1 or Z→2 here — L and Z are valid series letters (e.g. MW misread as MLZ)
# Only map characters that are NEVER valid in a digit position
LETTER_TO_DIGIT = {
    'O': '0', 'I': '1',
    'S': '5', 'G': '6', 'B': '8', 'T': '7'
}


def preprocess_for_ocr(img):
    """
    Multi-version preprocessor — returns the version that gives EasyOCR
    the best chance of reading Indian plate characters correctly.

    Returns a list of preprocessed images to try in order.
    The caller should run OCR on all versions and merge results.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # --- Step 1: Upscale to fixed width for consistency ---
    target_w = 640
    h, w = gray.shape
    scale = target_w / w
    gray = cv2.resize(gray, (target_w, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # --- Step 2: Sharpen to make strokes crisper ---
    kernel_sharpen = np.array([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)

    # --- Step 3: CLAHE for local contrast ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(sharpened)

    # --- Step 4: Invert if dark background ---
    if np.mean(enhanced) < 127:
        enhanced = cv2.bitwise_not(enhanced)

    # --- Version A: Otsu threshold (best for clean plates) ---
    _, v_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Version B: Adaptive threshold (best for uneven lighting) ---
    v_adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10
    )

    # --- Version C: Raw enhanced grayscale (EasyOCR handles grayscale well) ---
    v_gray = enhanced

    # --- Version D: Dilated — thickens thin strokes, helps with M/W confusion ---
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    v_dilated = cv2.dilate(v_otsu, kernel_dilate, iterations=1)

    return [v_otsu, v_adaptive, v_gray, v_dilated]


def correct_zone(text, expect_letters):
    """
    Correct characters based on the zone they appear in.
    - expect_letters=True  → convert digit lookalikes to letters
    - expect_letters=False → convert letter lookalikes to digits
    """
    result = ''
    for ch in text.upper():
        if expect_letters:
            result += DIGIT_TO_LETTER.get(ch, ch)
        else:
            result += LETTER_TO_DIGIT.get(ch, ch)
    return result


def parse_plate(raw):
    """
    Zone-aware parsing of Indian plate format: XX 00 XX 0000

    Key improvement: slides a window across the raw string and applies
    per-zone correction at each position, so misread characters like
    MLZ (EasyOCR misread of MW) are recovered correctly at the series zone.
    """
    raw = re.sub(r'[^A-Z0-9]', '', raw.upper())
    if len(raw) < 6:
        return None

    def try_parse_from(s):
        if len(s) < 6:
            return None

        # Zone 1 — State (2 letters)
        state = correct_zone(s[:2], expect_letters=True)
        if state not in INDIAN_STATE_CODES:
            return None
        s = s[2:]

        # Zone 2 — RTO (2 digits)
        rto = correct_zone(s[:2], expect_letters=False)
        if not rto.isdigit():
            return None
        s = s[2:]

        # Zone 3 — Series (1-2 letters), apply letter correction char by char
        # This recovers W from LZ, V from U misreads, etc.
        series = ''
        consumed = 0
        for ch in s[:2]:
            corrected = correct_zone(ch, expect_letters=True)
            if corrected.isalpha():
                series += corrected
                consumed += 1
            else:
                break
        if not series:
            return None
        s = s[consumed:]

        # Zone 4 — Number (3-4 digits)
        number = correct_zone(s[:4], expect_letters=False)
        # Accept 4 digits, or fall back to 3
        if len(number) >= 4 and number[:4].isdigit():
            number = number[:4]
        elif len(number) >= 3 and number[:3].isdigit():
            number = number[:3]
        else:
            return None

        return f"{state} {rto} {series} {number}"

    # Slide window — try parsing from every position in the string
    for i in range(len(raw) - 5):
        result = try_parse_from(raw[i:])
        if result:
            print(f"✅ Parsed plate: {result}")
            return result

    return None


def combine_and_extract_plate(all_texts_with_bbox):
    """
    Try to extract plate from OCR results.
    Strategy:
      1. Try each individual text
      2. Try pairing short texts (top line + bottom line) by position
      3. Fall back to full position-sorted combination
    """
    cleaned_texts = []
    for original, conf, bbox in all_texts_with_bbox:
        clean = re.sub(r'[^A-Z0-9]', '', original.upper())
        cleaned_texts.append((clean, conf, original, bbox))
        print(f"   '{original}' -> '{clean}' (conf: {conf:.2f})")

    # 1. Try each text individually
    for text, conf, original, bbox in cleaned_texts:
        result = parse_plate(text)
        if result:
            return result

    # 2. Try pairing: find a ~4-char candidate (top line) + ~6-char candidate (bottom line)
    # and concatenate them — handles two-row plates split by EasyOCR
    short = [(t, c, o, b) for t, c, o, b in cleaned_texts if 2 <= len(t) <= 5]
    long_ = [(t, c, o, b) for t, c, o, b in cleaned_texts if 4 <= len(t) <= 7]
    for s_text, s_conf, _, _ in short:
        for l_text, l_conf, _, _ in long_:
            if s_text == l_text:
                continue
            combined = s_text + l_text
            print(f"\n🔗 Trying pair: '{s_text}' + '{l_text}' = '{combined}'")
            result = parse_plate(combined)
            if result:
                return result
            # Also try reversed order
            combined_r = l_text + s_text
            result = parse_plate(combined_r)
            if result:
                return result

    # 3. Full combination sorted by bbox position (top to bottom)
    def bbox_sort_key(item):
        bbox = item[3]
        if bbox is not None:
            y = min(pt[1] for pt in bbox)
            x = min(pt[0] for pt in bbox)
            return (y, x)
        return (0, 0)

    position_sorted = sorted(cleaned_texts, key=bbox_sort_key)
    combined = ''.join([t[0] for t in position_sorted])
    print(f"\n🔗 Trying full combined (position-sorted): '{combined}'")
    return parse_plate(combined)


def extract_text(img):
    """
    Extract and clean text from license plate image.
    Runs OCR on multiple preprocessed versions and merges all results.
    """
    try:
        print("🔍 Starting OCR...")

        preprocessed_versions = preprocess_for_ocr(img)

        # Save first version for debugging
        cv2.imwrite("debug_processed.jpg", preprocessed_versions[0])

        print("📸 Running OCR on multiple preprocessed versions...")

        all_results = []

        for idx, version in enumerate(preprocessed_versions):
            # Run with two width thresholds per version for better coverage
            r1 = reader.readtext(version, paragraph=False, width_ths=0.7)
            r2 = reader.readtext(version, paragraph=False, width_ths=0.3, contrast_ths=0.1)
            all_results.extend(r1)
            all_results.extend(r2)

        # Also run on original image
        all_results.extend(reader.readtext(img, paragraph=False, width_ths=0.7))

        if not all_results:
            print("❌ No text detected")
            return ""

        # Deduplicate — keep highest confidence per unique text, preserve bbox
        text_dict = {}
        for bbox, text, conf in all_results:
            if conf < 0.1:
                continue
            if text.upper().strip() in ['IND', 'INDIA', 'INDIAN', 'IN']:
                continue
            key = text.strip()
            if key not in text_dict or conf > text_dict[key][1]:
                text_dict[key] = (text, conf, bbox)

        sorted_texts = sorted(text_dict.values(), key=lambda x: x[1], reverse=True)

        print(f"\n📊 OCR Results (top {min(8, len(sorted_texts))}):")
        for text, conf, _ in sorted_texts[:8]:
            print(f"   '{text}' (conf: {conf:.2f})")

        all_texts = [(text, conf, bbox) for text, conf, bbox in sorted_texts]
        plate = combine_and_extract_plate(all_texts)

        if plate:
            print(f"\n✅ Final plate: {plate}")
            return plate

        print("\n⚠️ Could not extract valid plate number")
        return ""

    except Exception as e:
        print(f"❌ OCR Error: {e}")
        import traceback
        traceback.print_exc()
        return ""