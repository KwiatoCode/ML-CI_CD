from flask import Flask, request, jsonify
import os
from model import predict_from_features

app = Flask(__name__)

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "5000")))
APP_ENV_NAME = os.getenv("APP_ENV_NAME", "local")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API modelu ML działa poprawnie",
        "environment_name": APP_ENV_NAME,
        "available_endpoints": ["/", "/health", "/predict"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "environment_name": APP_ENV_NAME
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Brak danych JSON w żądaniu"}), 400

        features = data.get("features")

        if features is None:
            return jsonify({"error": "Brak pola 'features'"}), 400

        if not isinstance(features, list):
            return jsonify({"error": "Pole 'features' musi być listą"}), 400

        if len(features) != 4:
            return jsonify({"error": "Model oczekuje dokładnie 4 cech"}), 400

        result = predict_from_features(features)

        return jsonify({
            "input_features": features,
            "predicted_class_index": result["predicted_class_index"],
            "predicted_class_name": result["predicted_class_name"],
            "environment_name": APP_ENV_NAME
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)