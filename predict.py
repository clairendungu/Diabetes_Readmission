"""Flask web service that serves the 30-day readmission model.

Run locally:
    python predict.py                      # development server on port 9696
    gunicorn --bind 0.0.0.0:9696 predict:app   # production-style server

Endpoints:
    GET  /health    -> {"status": "ok", ...}
    POST /predict   -> JSON describing one patient in, readmission risk out
"""
import os
import pickle

from flask import Flask, jsonify, request

import features

MODEL_PATH = os.getenv("MODEL_PATH", "model.bin")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
FEATURES = bundle["features"]
THRESHOLD = bundle["threshold"]
MODEL_NAME = bundle["model_name"]

app = Flask("readmission")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.post("/predict")
def predict():
    record = request.get_json(silent=True)
    if record is None:
        return jsonify({"error": "Send a JSON body with Content-Type: application/json."}), 400

    try:
        X = features.record_to_frame(record, FEATURES)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    probability = float(pipeline.predict_proba(X)[0, 1])
    return jsonify({
        "readmission_30d_probability": round(probability, 4),
        "high_risk": bool(probability >= THRESHOLD),
        "high_risk_threshold": round(THRESHOLD, 4),
        "model": MODEL_NAME,
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=9696)
