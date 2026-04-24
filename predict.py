import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (224, 224)
WEIGHTS_PATH = "models/resnet50_flower.weights.h5"

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def build_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = models.Sequential([
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_NAMES), activation="softmax")
    ])

    return model


model = build_model()
model.load_weights(WEIGHTS_PATH)


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