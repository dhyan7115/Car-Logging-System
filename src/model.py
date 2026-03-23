import tensorflow as tf
from tensorflow.keras import layers

def build_model():
    input_img = layers.Input(shape=(32, 128, 1))

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # reshape for RNN
    x = layers.Reshape((32//4, (128//4)*128))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    output = layers.Dense(37, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_img, outputs=output)
    return model