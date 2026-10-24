"""Accuracy + speed comparison of MobileNetV2, EfficientNetV2-B2 and EfficientNetV2-B3.

Folder layout (the folder name is the correct ImageNet label):
    labeled_images/
        golden_retriever/   photo1.jpg photo2.jpg ...
        pizza/              photo1.jpg ...

Usage:
    python compare_accuracy.py                  # evaluate ./labeled_images
    python compare_accuracy.py my_folder        # evaluate another folder
    python compare_accuracy.py --search dog     # list valid ImageNet class names containing "dog"
"""

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.applications import (
    EfficientNetV2B2,
    EfficientNetV2B3,
    MobileNetV2,
    efficientnet_v2,
    mobilenet_v2,
)
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(text: str) -> str:
    return text.lower().replace(" ", "_")


def imagenet_labels() -> list[str]:
    """All 1000 ImageNet class names, in index order."""
    rows = efficientnet_v2.decode_predictions(np.eye(1000, dtype="float32"), top=1)
    return [row[0][1] for row in rows]


def build_models() -> dict:
    # name -> (model, input size, module with preprocess_input / decode_predictions)
    return {
        "MobileNetV2": (MobileNetV2(weights="imagenet"), 224, mobilenet_v2),
        "EffNetV2-B2": (EfficientNetV2B2(weights="imagenet"), 260, efficientnet_v2),
        "EffNetV2-B3": (EfficientNetV2B3(weights="imagenet"), 300, efficientnet_v2),
    }


def make_infer(model):
    @tf.function(reduce_retracing=True)
    def infer(x):
        return model(x, training=False)

    return infer


def load_batch(path: Path, size: int, mod) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
   
    img = ImageOps.pad(img, (size, size), Image.BILINEAR, color=(128, 128, 128))
    return mod.preprocess_input(np.asarray(img, dtype="float32")[None, ...])


def speed_test(models: dict) -> None:
    print("--- Speed test: eager call vs tf.function (average of 10 runs) ---")
    for name, (model, size, _) in models.items():
        x = np.random.rand(1, size, size, 3).astype("float32") * 255
        infer = make_infer(model)
        model(x, training=False)  # warm-up
        infer(x)

        def avg_ms(fn) -> float:
            start = time.perf_counter()
            for _ in range(10):
                fn(x).numpy()
            return (time.perf_counter() - start) / 10 * 1000

        eager = avg_ms(lambda a: model(a, training=False))
        graph = avg_ms(infer)
        print(f"{name:<12} eager {eager:6.1f} ms | tf.function {graph:6.1f} ms")


def collect_samples(folder: Path, valid: set[str]) -> list[tuple[Path, str]]:
    samples = []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        true = norm(sub.name)
        if true not in valid:
            print(f"skipping folder '{sub.name}': not an ImageNet class (try --search)")
            continue
        samples += [
            (p, true) for p in sorted(sub.glob("*")) if p.suffix.lower() in IMAGE_EXTS
        ]
    return samples


def evaluate(samples: list[tuple[Path, str]], models: dict) -> None:
    infers = {name: make_infer(m) for name, (m, _, _) in models.items()}
    for name, (_, size, _) in models.items():  # warm-up
        infers[name](np.zeros((1, size, size, 3), dtype="float32"))

    results = {name: [] for name in models}  # rows of (path, true, top3, ms)
    for path, true in samples:
        for name, (_, size, mod) in models.items():
            try:
                x = load_batch(path, size, mod)
                start = time.perf_counter()
                probs = infers[name](x).numpy()
                ms = (time.perf_counter() - start) * 1000
            except Exception as exc:  # corrupt or unreadable image
                print(f"{name}: skipped {path.name} ({exc})")
                continue
            top3 = [norm(label) for _, label, _ in mod.decode_predictions(probs, top=3)[0]]
            results[name].append((path, true, top3, ms))

    print(f"\n--- Accuracy on {len(samples)} labeled images ---")
    for name, rows in results.items():
        if not rows:
            continue
        n = len(rows)
        top1 = sum(r[2][0] == r[1] for r in rows) / n * 100
        top3 = sum(r[1] in r[2] for r in rows) / n * 100
        ms = np.mean([r[3] for r in rows])
        print(f"{name:<12} top-1 {top1:5.1f}%   top-3 {top3:5.1f}%   {ms:6.1f} ms/img")

    print("\n--- Images each model got wrong (top-1) ---")
    for name, rows in results.items():
        wrong = [r for r in rows if r[2][0] != r[1]]
        print(f"\n{name}: {len(wrong)} wrong")
        for path, true, top3, _ in wrong:
            print(f"  {path.parent.name}/{path.name}: expected {true}, got {top3[0]}")


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--search":
        term = norm(sys.argv[2])
        print("\n".join(label for label in imagenet_labels() if term in norm(label)))
        return

    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "labeled_images")
    if not folder.is_dir():
        sys.exit(f"Folder '{folder}' not found. Create it with one subfolder per class.")

    valid = {norm(label) for label in imagenet_labels()}
    samples = collect_samples(folder, valid)
    if not samples:
        sys.exit("No usable images found. Check folder names and file types.")

    models = build_models()
    speed_test(models)
    evaluate(samples, models)


if __name__ == "__main__":
    main()