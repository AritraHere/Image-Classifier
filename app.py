"""Flask image classifier — MobileNetV2 (ImageNet), in-memory predict."""

import os
from io import BytesIO

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MAX_BYTES = 5 * 1024 * 1024  # 5 MB

app = Flask(__name__)
model = MobileNetV2(weights="imagenet")


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


def classify_bytes(data: bytes) -> list[dict]:
    """Preprocess image bytes and return top-3 ImageNet predictions."""
    img = Image.open(BytesIO(data)).convert("RGB").resize((224, 224))
    x = preprocess_input(np.expand_dims(np.asarray(img, dtype="float32"), 0))
    preds = decode_predictions(model.predict(x, verbose=0), top=3)[0]
    return [
        {"label": label.replace("_", " "), "confidence": float(score) * 100}
        for _, label, score in preds
    ]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    if "imagefile" not in request.files:
        return render_template("index.html", error="No file part in the request.")

    imagefile = request.files["imagefile"]
    if not imagefile or imagefile.filename == "":
        return render_template("index.html", error="No file selected.")

    if not _allowed(imagefile.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Use PNG, JPG, JPEG, or WEBP.",
        )

    data = imagefile.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return render_template("index.html", error="File too large (max 5 MB).")
    if not data:
        return render_template("index.html", error="Empty file.")

    try:
        predictions = classify_bytes(data)
    except Exception:
        return render_template(
            "index.html",
            error="Could not classify that image. Try another file.",
        )

    return render_template("index.html", predictions=predictions)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
