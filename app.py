from flask import Flask, jsonify, request
import os
import gdown

WEIGHTS_PATH = "models/resnet50_flower.weights.h5"

os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

if not os.path.exists(WEIGHTS_PATH):
    print("Weights indiriliyor...")
    url = "https://drive.google.com/uc?id=1jPqZdYcCd0t2m21FSaozQD5GkM1KfBaV"
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
        return jsonify({"error": "file field yok"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)

    result = predict_image(filepath)
    return jsonify(result)


@app.route("/model/finetune", methods=["POST"])
def finetune():
    return jsonify({
        "message": "Fine-tuning endpoint çalışıyor",
        "status": "training_triggered_demo"
    })


if __name__ == "__main__":
    app.run(debug=True)