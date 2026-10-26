# Emotion Analysis

A webcam app that detects a face and classifies its expression with a CNN.
Built with Python, Flask, OpenCV and plain HTML, CSS and JavaScript.

The page shows the current expression, confidence, processing time and recent
predictions. It processes the largest face in each frame. Session counts stay
in the browser and disappear on refresh.

## Run locally

Use Python 3.12. The bundled model also works with Python 3.9–3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_model.py
python app.py
```

Open http://127.0.0.1:5000 and allow camera access. Keep the terminal open.
On Windows, use `.venv\Scripts\python.exe` to install packages and run `app.py`.

## Model

The app uses the pretrained FER+ model from the
[ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus).
Its eight outputs are kept intact. The interface records six selected classes:
angry, fearful, happy, sad, surprised and neutral. Disgust and contempt appear
as “Other expression” and are excluded from the six class counts.

The model uses 64×64 grayscale images. OpenCV handles face detection and ONNX
inference. Source, label order and licence notices are in `models/`.
`download_model.py` downloads the weights and verifies their SHA256 checksum.

## Train a six-class model

Obtain the original FER-2013 CSV with `emotion`, `pixels` and `Usage` columns.
The dataset is not included in this repository.

```bash
python -m pip install -r requirements-training.txt
python train.py --csv data/fer2013.csv --epochs 30
python evaluate.py --csv data/fer2013.csv
```

Training uses the official Training and PublicTest splits. Evaluation uses
PrivateTest. The six-class model excludes disgust. Restart the app after
training; it loads `models/emotion_model.keras` and its matching JSON metadata
when both are available. The Colab workflow is in `notebooks/`.

## Deploy on Vercel

Import this repository in Vercel. Keep the project root at the repository root
and leave the build command and output directory at their Flask defaults.
The Python version is pinned to 3.12; `app.py` exports the Flask application.
Only `requirements.txt` is needed for the hosted model. Training stays local.
The build command downloads and verifies the model before it is bundled with
the function; large model weights are kept out of Git history.

Vercel serves the files in `public/static/` through its CDN. The server accepts
the deployment, branch and production hostnames supplied by Vercel. For a
custom domain, set `APP_HOST` to its hostname and redeploy.

## Data handling

Camera access requires HTTPS or localhost. During local use, frames are sent
to the server on the same computer. On the hosted version, frames are sent
to the Vercel server for prediction. The application does not save frames or
session history to disk. Hosting access logs can still contain request metadata.

Expression labels describe visible facial patterns. Model confidence is not
an accuracy measurement or a reliable reading of someone's feelings.

## Checks

```bash
python -m unittest discover -s tests -v
```
