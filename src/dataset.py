import pandas as pd
import cv2
import numpy as np
from utils import encode_label

IMG_W, IMG_H = 128, 32

def load_data(csv_path, img_dir):
    df = pd.read_csv(csv_path)

    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = f"{img_dir}/{row['filename']}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)

        label = encode_label(row['text'])

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)