import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import string
import os

# ==============================
# CONFIG
# ==============================
IMG_W, IMG_H = 128, 32
MAX_LEN = 10

CHAR_SET = string.ascii_uppercase + string.digits
char_to_num = {c: i+1 for i, c in enumerate(CHAR_SET)}
num_to_char = {i+1: c for i, c in enumerate(CHAR_SET)}

# ==============================
# ENCODE LABEL
# ==============================
def encode_label(text):
    label = [char_to_num.get(c, 0) for c in text]
    label += [0] * (MAX_LEN - len(label))
    return label[:MAX_LEN]

# ==============================
# LOAD DATA
# ==============================
def load_data(csv_path, img_dir):
    df = pd.read_csv(csv_path)

    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row["filename"])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)

        label = encode_label(row["text"])

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# ==============================
# MODEL (CRNN)
# ==============================
def build_model():
    inputs = tf.keras.Input(shape=(IMG_H, IMG_W, 1))

    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Reshape((IMG_H//4, (IMG_W//4)*128))(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(x)

    outputs = tf.keras.layers.Dense(len(CHAR_SET)+1, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

# ==============================
# CTC LOSS
# ==============================
def ctc_loss(y_true, y_pred):
    batch_len = tf.shape(y_pred)[0]
    input_len = tf.shape(y_pred)[1]
    label_len = tf.shape(y_true)[1]

    input_len = input_len * tf.ones((batch_len, 1))
    label_len = label_len * tf.ones((batch_len, 1))

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)

# ==============================
# TRAIN FUNCTION
# ==============================
def train():
    print("[INFO] Loading dataset...")

    X, y = load_data("dataset/labels.csv", "dataset/plates/train")

    print(f"[INFO] Dataset size: {len(X)}")

    model = build_model()
    model.compile(optimizer='adam', loss=ctc_loss)

    model.summary()

    model.fit(
        X, y,
        epochs=20,
        batch_size=32,
        validation_split=0.1
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/crnn.h5")

    print("[INFO] Model saved at models/crnn.h5")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    train()