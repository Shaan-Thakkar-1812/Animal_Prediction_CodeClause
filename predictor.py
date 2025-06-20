import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

# Load model and label mapping
model = load_model("model/animal_model.h5")
with open("labels.json", "r") as f:
    class_names = json.load(f)

# Reverse key-value: {0: 'cat', 1: 'dog', ...}
class_names = {v: k for k, v in class_names.items()}

def preprocess_image(img, target_size=(128, 128)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img):
    preprocessed = preprocess_image(img)
    prediction = model.predict(preprocessed)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = round(float(prediction[predicted_index]) * 100, 2)
    return predicted_label, confidence
