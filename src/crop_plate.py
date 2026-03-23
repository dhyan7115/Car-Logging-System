def crop_plates(results, frame):
    plates = []

    if len(results) == 0 or results[0].boxes is None:
        return plates

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(0, x2), max(0, y2)

        plate = frame[y1:y2, x1:x2]

        if plate.size != 0:
            plates.append((plate, (x1, y1, x2, y2)))

    return plates