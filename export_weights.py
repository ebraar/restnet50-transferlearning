import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

model = tf.keras.models.load_model(
    "models/resnet50_flower_model.h5",
    custom_objects={"preprocess_input": preprocess_input}
)

model.save_weights("models/resnet50_flower.weights.h5")

print("Weights kaydedildi.")