from flask import Flask, jsonify, request
import os
import gdown

WEIGHTS_PATH = "models/mobilenetv2_flower.weights.h5"

os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

if not os.path.exists(WEIGHTS_PATH):
    print("Weights indiriliyor...")
    url = "https://drive.google.com/uc?id=1QnkwThRf4RSPiOn39J-n7JwwFOoYy3aO"
    gdown.download(url, WEIGHTS_PATH, quiet=False)

from predict import predict_image

app = Flask(__name__)

UPLOAD_DIR = "uploads"

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API çalışıyor 🚀"})

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "file yok"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "dosya yok"}), 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    result = predict_image(path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)