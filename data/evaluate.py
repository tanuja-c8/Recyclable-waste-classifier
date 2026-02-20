import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model

DATA_DIR = "data/dataset"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
MODEL_PATH = "model/trained_model.h5"

print("Looking for model at:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("Model file not found. Please check the path.")
    exit()

model = load_model(MODEL_PATH)
print("Model loaded successfully")

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
