from flask import Flask, jsonify, request
import os
import gdown

MODEL_PATH = "models/resnet50_flower_model.h5"

os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Model indiriliyor...")
    url = "https://drive.google.com/uc?id=1Vj3RgQCpMO_67hiENBFLHVqZgFtGhLhS"
    gdown.download(url, MODEL_PATH, quiet=False)

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