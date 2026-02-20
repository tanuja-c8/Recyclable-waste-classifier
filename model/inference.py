import os
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.h5")

IMG_SIZE = (160, 160)

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict_image(model, image, class_names):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]

    confidence = float(np.max(preds) * 100)
    label = class_names[int(np.argmax(preds))]

    predictions = {
        class_names[i]: float(preds[i] * 100)
        for i in range(len(class_names))
    }

    return label, confidence, predictions
