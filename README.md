# Image Classifier

A web-based image classification application powered by **TensorFlow, MobileNetV2, and Flask**. Upload an image and receive the three most likely ImageNet classifications with corresponding confidence scores through a simple web interface.

## Overview

Image Classifier combines a pretrained deep learning model with a lightweight Flask application to make image recognition accessible through a browser. It handles image validation, preprocessing, model inference, and prediction rendering in a streamlined workflow.

## Features

- **Deep Learning Inference** — Powered by TensorFlow and the pretrained MobileNetV2 architecture.
- **Top-3 Predictions** — Returns the three highest-ranked ImageNet classifications with confidence scores.
- **Browser-Based Interface** — Upload images and view results without running commands for each prediction.
- **Multiple Image Formats** — Supports PNG, JPG, JPEG, and WEBP.
- **Input Validation** — Enforces supported file extensions and a maximum upload size of 5 MB.
- **In-Memory Processing** — Processes uploaded images without intentionally saving them to disk.
- **Cloud Deployment** — Configured to run on hosting platforms such as Render.
- **Automated Inference Check** — Includes a smoke test to verify model loading and prediction execution.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10–3.12 | Application runtime |
| Flask 3.0.3 | Web application framework |
| TensorFlow 2.16.1 | Deep learning inference |
| MobileNetV2 | Pretrained image classification model |
| Pillow 10.4.0 | Image processing |
| NumPy | Numerical operations |

## Project Structure

```text
Image-Classifier/
├── app.py
├── check_predict.py
├── requirements.txt
├── templates/
│   └── index.html
├── .python-version
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Flask server, image preprocessing, and prediction endpoints |
| `check_predict.py` | Model initialization and inference smoke test |
| `requirements.txt` | Project dependencies |
| `templates/index.html` | Web interface |
| `.python-version` | Python runtime version configuration |
| `README.md` | Project documentation |

## Getting Started

### Prerequisites

- Python 3.10, 3.11, or 3.12
- pip
- Internet connectivity for the initial download of pretrained model weights

### 1. Clone the Repository

```bash
git clone https://github.com/AritraHere/Image-Classifier.git
cd Image-Classifier
```

### 2. Create a Virtual Environment

**Windows**

```powershell
py -3.10 -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Model Inference

```bash
python check_predict.py
```

The smoke test initializes MobileNetV2, generates a sample image, executes inference, and verifies that the model returns three predictions.

### 5. Launch the Application

```bash
python app.py
```

Open the application in your browser:

**http://127.0.0.1:5000**

Upload an image to receive its classification results.

## Model Architecture

The application uses **MobileNetV2**, a convolutional neural network architecture designed for efficient visual recognition.

The model loads pretrained ImageNet weights:

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

model = MobileNetV2(weights="imagenet")
```

### Inference Pipeline

1. **Image Upload** — Receives an image through the Flask interface.
2. **Validation** — Checks the file extension, upload size, and image data.
3. **Preprocessing** — Converts the image to RGB and resizes it to 224 × 224 pixels.
4. **Normalization** — Applies the preprocessing function required by MobileNetV2.
5. **Inference** — Generates classification scores using the pretrained model.
6. **Prediction Decoding** — Maps ImageNet class identifiers to readable labels.
7. **Result Rendering** — Displays the top three predictions and their confidence scores.

## Supported Image Formats

| Property | Specification |
|---|---|
| PNG | Supported |
| JPG / JPEG | Supported |
| WEBP | Supported |
| Maximum upload size | 5 MB |
| Model input dimensions | 224 × 224 pixels |
| Predictions returned | 3 |
| Model weights | ImageNet pretrained |

## Deployment on Render

The application can be deployed as a Python Web Service on [Render](https://render.com/).

### Build Configuration

| Setting | Value |
|---|---|
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python check_predict.py && python app.py` |

### Python Version

Create a `.python-version` file in the repository root:

```text
3.10.16
```

Alternatively, configure the following environment variable if supported by your Render service:

| Variable | Value |
|---|---|
| `PYTHON_VERSION` | `3.10.16` |

Refer to the official [Render Python version documentation](https://render.com/docs/python-version) for the current configuration requirements.

### Deployment Steps

1. Push the project to GitHub.
2. Create a new Web Service on Render.
3. Connect the repository and select the deployment branch.
4. Configure the build and start commands.
5. Set the Python version.
6. Deploy the application and review the build logs.
7. Open the public URL generated by Render.

During the initial startup, TensorFlow downloads the pretrained MobileNetV2 weights. The application requires sufficient memory to initialize the model and execute inference.

## Dependencies

The project uses the following pinned dependencies:

```text
Flask==3.0.3
tensorflow==2.16.1
Pillow==10.4.0
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Privacy and File Handling

Uploaded images are processed in memory and are not intentionally persisted by the application. The application validates supported file extensions and enforces the configured upload size limit.

For production deployments, use HTTPS and appropriate server-level request limits. Hosting infrastructure may retain operational logs or metadata independently of application-level file handling.

---

<p align="center">
  Built with Flask, TensorFlow, and MobileNetV2.
</p>
