"""Smoke check: in-memory preprocess + MobileNetV2 top-3 decode."""

from io import BytesIO

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)


def main() -> None:
    model = MobileNetV2(weights="imagenet")
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(40, 120, 200)).save(buf, format="PNG")
    img = Image.open(BytesIO(buf.getvalue())).convert("RGB").resize((224, 224))
    x = preprocess_input(np.expand_dims(np.asarray(img, dtype="float32"), 0))
    preds = decode_predictions(model.predict(x, verbose=0), top=3)[0]
    assert len(preds) == 3, f"expected 3 predictions, got {len(preds)}"
    for row in preds:
        assert len(row) == 3, f"expected (id, label, score), got {row!r}"
    print("ok:", [(label, round(float(score), 4)) for _, label, score in preds])


if __name__ == "__main__":
    main()
