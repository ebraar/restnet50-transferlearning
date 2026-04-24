from flask import Flask, jsonify, request
import os
from predict import predict_image

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API çalışıyor 🚀"})


@app.route("/classify", methods=["POST"])
def classify():
    # file field kontrolü
    if "file" not in request.files:
        return jsonify({"error": "file field yok"}), 400

    file = request.files["file"]

    # boş dosya kontrolü
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400

    # dosyayı kaydet
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)

    # tahmin yap
    result = predict_image(filepath)

    # sonucu döndür
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)