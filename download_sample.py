import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/download.tensorflow.org/example_images/"
SAMPLES = {
    "labrador.jpg": "YellowLabradorLooking_new.jpg",
    "cat_on_snow.jpg": "320px-Felis_catus-cat_on_snow.jpg",
    "sunflower.jpg": "592px-Red_sunflower.jpg",
    "grace_hopper.jpg": "grace_hopper.jpg",
}


def main() -> None:
    folder = Path("test_images")
    folder.mkdir(exist_ok=True)
    for local_name, remote_name in SAMPLES.items():
        target = folder / local_name
        if target.exists():
            print(f"already have {local_name}")
            continue
        try:
            urllib.request.urlretrieve(BASE + remote_name, target)
            print(f"downloaded {local_name}")
        except Exception as exc:
            print(f"skipped {local_name}: {exc}")


if __name__ == "__main__":
    main()