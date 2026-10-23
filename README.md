# 🖼️ Image Classifier

A lightweight, AI-powered image classification web application built with **Flask and TensorFlow**. Upload an image and let a pretrained MobileNetV2 model identify its three most likely ImageNet categories.

## ✨ Features

- 🤖 **AI-Powered Classification** — Powered by TensorFlow and MobileNetV2.
- 🎯 **Top 3 Predictions** — Displays the most likely categories with confidence scores.
- 🖼️ **Multiple Formats** — Supports PNG, JPG, JPEG, and WEBP.
- ⚡ **Efficient Inference** — Runs predictions directly in memory.
- 🔒 **Upload Validation** — Supports images up to 5 MB.
- 🌐 **Web Interface** — Classify images directly from your browser.
- ☁️ **Cloud Ready** — Supports deployment on platforms such as Render.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10–3.12 | Runtime |
| Flask | Web framework |
| TensorFlow 2.16.1 | Deep learning |
| MobileNetV2 | Image classification |
| Pillow | Image processing |
| NumPy | Numerical operations |

## 📁 Project Structure

```text
Image-Classifier/
├── app.py
├── check_predict.py
├── requirements.txt
├── templates/
│   └── index.html
├── .python-version
└── README.md
````

## 🚀 Installation

### 1\. Clone the Repository

```
git clone https://github.com/AritraHere/Image-Classifier.git
cd Image-Classifier
```

### 2\. Create a Virtual Environment

**Windows**

```
py -3.10 -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```
python3.10 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4\. Verify the Model

```
python check_predict.py
```

This script loads MobileNetV2, performs a sample prediction, and verifies that three predictions are returned.

### 5\. Run the Application

```
python app.py
```

Open your browser and visit:

http://127.0.0.1:5000

Upload an image to see the classification results.

## ☁️ Deploy on Render

### Build Configuration

| Setting | Value |
| --- | --- |
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python check_predict.py && python app.py` |

### Python Version

Create a `.python-version` file in the repository root:

```
3.10.16
```

Alternatively, configure the `PYTHON_VERSION` environment variable in Render:

```
PYTHON_VERSION=3.10.16
```

Follow the official Render Python version guide for the supported configuration method.

### Deployment Notes

- The first startup downloads the pretrained ImageNet weights.
- TensorFlow runs inference on the CPU by default.
- Sufficient memory is required to initialize the model.
- The included smoke test adds startup time because it loads the model before the Flask application loads it again.
- Free-tier services may experience cold starts and resource limitations.

## 🧠 How It Works

The application follows a straightforward image classification pipeline:

1. **Upload** — The user submits an image through the web interface.
2. **Validate** — The application checks the file extension, size, and content.
3. **Preprocess** — The image is converted to RGB and resized to 224 × 224 pixels.
4. **Predict** — MobileNetV2 processes the image.
5. **Decode** — The model's output is converted into readable ImageNet labels.
6. **Display** — The top three predictions and their confidence scores are shown.

### Model Initialization

```
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

model = MobileNetV2(weights="imagenet")
```

MobileNetV2 is a convolutional neural network designed for efficient image recognition.

## 📸 Supported Formats

| Property | Details |
| --- | --- |
| PNG | Supported |
| JPG / JPEG | Supported |
| WEBP | Supported |
| Maximum upload size | 5 MB |
| Input resolution | 224 × 224 pixels |
| Number of predictions | 3 |

## 🔒 Privacy and Security

- Uploaded images are processed in memory and are not intentionally saved to disk.
- Unsupported file extensions are rejected.
- Files exceeding the configured size limit are rejected.
- Production deployments should use HTTPS and appropriate request limits.

**Note:** Hosting providers may retain operational logs or metadata. Avoid uploading sensitive images to public deployments.

## ⚠️ Limitations

- Predictions are restricted to ImageNet categories.
- Confidence scores are model outputs, not guarantees of correctness.
- Classification performance depends on image quality and the model's training categories.
- The application performs image classification, not object detection or segmentation.
- Inference speed depends on available CPU and memory resources.

## 📦 Dependencies

The project uses the following pinned dependencies:

```
Flask==3.0.3
tensorflow==2.16.1
Pillow==10.4.0
```

Install them using:

```
pip install -r requirements.txt
```
