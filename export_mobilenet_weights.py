import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model(
    "models/mobilenetv2_flower_model.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

model.save_weights("models/mobilenetv2_flower.weights.h5")

print("MobileNetV2 weights kaydedildi.")