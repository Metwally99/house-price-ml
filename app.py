"""Serve the prediction demo and a JSON inference endpoint."""

import json
import math
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

DEMO = Path(__file__).resolve().parent / "docs"
ARTIFACT = json.loads((DEMO / "models.json").read_text(encoding="utf-8"))
FEATURES = ARTIFACT["features"]
MODELS = ARTIFACT["models"]
DEFAULT_MODEL = ARTIFACT["selected_model"]

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


def infer(features, model_type):
    values = [
        (value - mean) / scale
        for value, mean, scale in zip(features, ARTIFACT["mean"], ARTIFACT["scale"], strict=True)
    ]
    layers = MODELS[model_type]
    for index, layer in enumerate(layers):
        values = [
            sum(weight * value for weight, value in zip(row, values, strict=True)) + bias
            for row, bias in zip(layer["weight"], layer["bias"], strict=True)
        ]
        if index < len(layers) - 1:
            values = [max(0.0, value) for value in values]
    return values[0] * ARTIFACT["target_scale"] + ARTIFACT["target_mean"]


@app.get("/")
def home():
    return send_from_directory(DEMO, "index.html")


@app.get("/<any(models.json,evaluation.json,demo.js):filename>")
def demo_asset(filename):
    return send_from_directory(DEMO, filename)


@app.get("/health")
def health():
    return jsonify(status="ok", model=DEFAULT_MODEL)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(success=False, error="Send a JSON object."), 400
    model_type = data.get("model_type", DEFAULT_MODEL)
    if not isinstance(model_type, str) or model_type not in MODELS:
        return jsonify(success=False, error="Choose model A, B, C, D or E."), 400
    try:
        features = [float(data[name]) for name in FEATURES]
    except (KeyError, TypeError, ValueError, OverflowError):
        return jsonify(success=False, error="Provide all six numeric features."), 400
    if not all(math.isfinite(value) for value in features):
        return jsonify(success=False, error="Features must be finite."), 400
    price = infer(features, model_type)
    if not math.isfinite(price):
        return jsonify(success=False, error="Features are outside numerical limits."), 400
    return jsonify(success=True, price=price, model=model_type, unit="10,000 TWD per ping")


if __name__ == "__main__":
    app.run(port=8000)
