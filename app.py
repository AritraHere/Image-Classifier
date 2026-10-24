import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from keras.applications.efficientnet_v2 import (
    EfficientNetV2B3,
    decode_predictions,
    preprocess_input,
)
from PIL import Image, ImageOps

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
INPUT_SIZE = 300 
USE_FLIP_AVERAGING = False 
LOW_CONFIDENCE_PCT = 30.0  # below this top-1 score, tell the user the result is uncertain

app = Flask(__name__)
# Reject oversized requests early (extra room for form overhead); the exact check is below.
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 1024 * 1024

model = EfficientNetV2B3(weights="imagenet")


@tf.function(reduce_retracing=True)
def infer(x):
    """Compiled forward pass: much faster than calling the model eagerly."""
    return model(x, training=False)



infer(np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype="float32"))


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


def classify_bytes(data: bytes) -> list[dict]:
    """Preprocess image bytes and return top-3 ImageNet predictions."""
    img = Image.open(BytesIO(data))
    img = ImageOps.exif_transpose(img) 
    img = img.convert("RGB")
    # Keep the aspect ratio (pad to a square) instead of squashing, so objects
    # aren't stretched and nothing at the edges is cut off.
    img = ImageOps.pad(img, (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR, color=(128, 128, 128))
    
    x = preprocess_input(np.expand_dims(np.asarray(img, dtype="float32"), 0))
    probs = infer(x).numpy()
    if USE_FLIP_AVERAGING:
        mirrored = np.ascontiguousarray(x[:, :, ::-1, :])
        probs = (probs + infer(mirrored).numpy()) / 2
    preds = decode_predictions(probs, top=3)[0]
    return [
        {"label": label.replace("_", " "), "confidence": float(score) * 100}
        for _, label, score in preds
    ]


@app.errorhandler(413)
def too_large(_error):
    return render_template("index.html", error="File too large (max 5 MB)."), 413


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

    low_confidence = predictions[0]["confidence"] < LOW_CONFIDENCE_PCT
    return render_template(
        "index.html", predictions=predictions, low_confidence=low_confidence
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)