"""Face detection and CNN inference."""
import json
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

CLASSES = ["angry", "fearful", "happy", "sad", "surprised", "neutral"]
FERPLUS_CLASSES = ["neutral", "happy", "surprised", "sad", "angry", "disgust", "fearful", "contempt"]


def softmax(scores):
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Model returned invalid scores.")
    values = np.exp(values - values.max())
    return values / values.sum()


class ExpressionPredictor:
    def __init__(self, model_dir):
        self.lock = Lock()
        self.ready = False
        self.error = None
        self.model_name = "No model loaded"
        self.classes = CLASSES.copy()
        self.kind = None
        self.model_dir = Path(model_dir)
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.detector.empty():
            raise RuntimeError("OpenCV face detector could not be loaded.")
        self.load_model()

    def load_model(self):
        custom_path = self.model_dir / "emotion_model.keras"
        baseline_path = self.model_dir / "emotion-ferplus-8.onnx"
        try:
            if custom_path.exists():
                # Prefer a trained custom model when present.
                import tensorflow as tf
                metadata = json.loads((self.model_dir / "emotion_model.json").read_text())
                if metadata.get("classes") != CLASSES or metadata.get("input_size") != 48:
                    raise ValueError("Custom model metadata does not match the six-class project.")
                if metadata.get("normalization") != "0_1":
                    raise ValueError("Unexpected custom-model normalization.")
                self.model = tf.keras.models.load_model(custom_path, compile=False)
                if tuple(self.model.input_shape[1:]) != (48, 48, 1) or self.model.output_shape[-1] != 6:
                    raise ValueError("Custom model input/output shape is incorrect.")
                self.kind = "custom"
                self.input_size = 48
                self.classes = CLASSES.copy()
                self.model_name = "Six-class CNN (FER-2013)"
            elif baseline_path.exists():
                self.model = cv2.dnn.readNetFromONNX(str(baseline_path))
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.kind = "baseline"
                self.input_size = 64
                self.classes = FERPLUS_CLASSES.copy()
                self.model_name = "Pretrained FER+ CNN baseline (8 outputs)"
            else:
                raise FileNotFoundError("Model missing. Run python download_model.py.")
            self.ready = True
        except Exception as exc:
            self.error = str(exc)

    def status(self):
        return {
            "ready": self.ready, "model_name": self.model_name,
            "model_kind": self.kind, "supported_classes": CLASSES,
            "output_classes": self.classes, "error": self.error,
            "input_size": getattr(self, "input_size", None),
        }

    def face_probabilities(self, grayscale_crop):
        """Apply the preprocessing used by the selected model."""
        resized = cv2.resize(grayscale_crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        with self.lock:
            if self.kind == "baseline":
                # FER+ expects raw grayscale 0..255 pixels, NCHW, and outputs logits.
                blob = resized.astype(np.float32).reshape(1, 1, 64, 64)
                self.model.setInput(blob)
                probabilities = softmax(self.model.forward())
            else:
                # Custom model: normalised grayscale, NHWC, softmax output.
                blob = resized.astype(np.float32).reshape(1, 48, 48, 1) / 255.0
                probabilities = np.asarray(self.model(blob, training=False)).reshape(-1)
                if len(probabilities) != 6 or not np.isfinite(probabilities).all() or (probabilities < 0).any():
                    raise ValueError("Custom CNN returned invalid probabilities.")
                if probabilities.sum() <= 0:
                    raise ValueError("Custom CNN returned an empty probability distribution.")
                probabilities = probabilities / probabilities.sum()
        if len(probabilities) != len(self.classes):
            raise ValueError("Model class count does not match configured labels.")
        return dict(zip(self.classes, (float(p) for p in probabilities)))

    def analyze(self, bgr_frame):
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        # Equalisation assists detection only; inference uses the original grayscale crop.
        with self.lock:
            faces = self.detector.detectMultiScale(
                cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
        if len(faces) == 0:
            return {"status": "no_face", "face_detected": False, "face_count": 0}
        x, y, w, h = map(int, max(faces, key=lambda f: f[2] * f[3]))
        base = {
            "face_detected": True, "face_count": len(faces),
            "box": {"x": x, "y": y, "width": w, "height": h},
            "frame_width": bgr_frame.shape[1], "frame_height": bgr_frame.shape[0],
        }
        if not self.ready:
            return {**base, "status": "model_unavailable", "message": self.error}
        probabilities = self.face_probabilities(gray[y:y+h, x:x+w])
        label = max(probabilities, key=probabilities.get)
        supported = label in CLASSES
        return {
            **base, "status": "ok" if supported else "out_of_scope",
            "expression": label if supported else None,
            "raw_expression": label, "confidence": probabilities[label],
            "probabilities": probabilities,
            "model_name": self.model_name,
        }
