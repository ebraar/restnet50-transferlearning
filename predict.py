import numpy as np
import tensorflow as tf
from PIL import Image
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (224, 224)

MODEL_PATH_KERAS = "models/resnet50_flower_model.keras"
MODEL_PATH_H5 = "models/resnet50_flower_model.h5"

# Modeli yükle
if os.path.exists(MODEL_PATH_KERAS):
    model = tf.keras.models.load_model(
        MODEL_PATH_KERAS,
        custom_objects={"preprocess_input": preprocess_input}
    )
elif os.path.exists(MODEL_PATH_H5):
    model = tf.keras.models.load_model(
        MODEL_PATH_H5,
        custom_objects={"preprocess_input": preprocess_input}
    )
else:
    raise FileNotFoundError("Model dosyası bulunamadı.")

# Eğitimde çıkan sınıf isimleriyle aynı olmalı
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    top_idx = int(np.argmax(preds))

    return {
        "class": CLASS_NAMES[top_idx],
        "confidence": float(preds[top_idx]),
        "all_scores": {
            c: float(p) for c, p in zip(CLASS_NAMES, preds)
        }
    }