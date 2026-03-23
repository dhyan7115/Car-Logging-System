import tensorflow as tf
import cv2
import numpy as np
from utils import decode_label

IMG_W, IMG_H = 128, 32

model = tf.keras.models.load_model("models/crnn.h5", compile=False)

def predict(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    pred = model.predict(img)

    decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=[pred.shape[1]])
    result = decoded[0].numpy()[0]

    print("Plate:", decode_label(result))

predict("test_plate.jpg")