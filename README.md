# Image Classifier (Flask)

Upload an image and get the top-3 ImageNet labels from **MobileNetV2**.

Requires **Python 3.10–3.12** (TensorFlow has no wheels for 3.14 yet).

## Setup (Windows)

```text
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python check_predict.py
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The first run downloads ImageNet weights (~14 MB for MobileNetV2).

## Notes

- Allowed types: PNG, JPG, JPEG, WEBP (max 5 MB)
- Inference runs in memory — uploads are not saved to disk
- Optional: set `PORT` to change the listen port
