import sys
import time
from pathlib import Path

import numpy as np

from keras.applications import (
    EfficientNetV2B3,
    MobileNetV2,
    efficientnet_v2,
    mobilenet_v2,
)

from PIL import Image, ImageOps
 
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODELS = {
    "MobileNetV2": (MobileNetV2(weights="imagenet"), 224, mobilenet_v2),
    "EffNetV2-B3": (EfficientNetV2B3(weights="imagenet"), 300, efficientnet_v2),
}

def load_batch(path: Path, size: int, mod) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return mod.preprocess_input(np.asarray(img, dtype="float32")[None, ...])


def run(model, x, mod):
    start = time.perf_counter()
    probs = model(x, training=False).numpy()
    ms = (time.perf_counter() - start) * 1000
    top3 = [(label, round(float(score), 3)) for _, label, score in mod.decode_predictions(probs, top=3)[0]]
    return top3, ms


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "test_images")
    paths = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        sys.exit(f"No images found in '{folder}'. Create the folder and add some .jpg/.png files.")
 
   
    for model, size, _ in MODELS.values():
        model(np.zeros((1, size, size, 3), dtype="float32"), training=False)
 
    times = {name: [] for name in MODELS}
    for path in paths:
        print(f"\n{path.name}")
        for name, (model, size, mod) in MODELS.items():
            try:
                top3, ms = run(model, load_batch(path, size, mod), mod)
            except Exception as exc:  # corrupt or unreadable image
                print(f"  {name:<12} skipped ({exc})")
                continue
            times[name].append(ms)
            print(f"  {name:<12} {ms:6.1f} ms  {top3}")
 
    print("\n--- Average latency ---")
    for name, values in times.items():
        if values:
            print(f"{name:<12} {np.mean(values):6.1f} ms/img over {len(values)} images")
 
 
if __name__ == "__main__":
    main()
 