"""Webcam expression classification API."""
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from src.predictor import ExpressionPredictor

BASE_DIR = Path(__file__).resolve().parent


def create_app(predictor=None):
    app = Flask(__name__, static_folder="public/static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    hosted = os.environ.get("VERCEL") == "1" or bool(os.environ.get("APP_HOST"))
    allowed_hosts = {"127.0.0.1", "localhost"}
    for name in ("VERCEL_URL", "VERCEL_BRANCH_URL", "VERCEL_PROJECT_PRODUCTION_URL", "APP_HOST"):
        value = os.environ.get(name, "").strip()
        if value:
            hostname = urlparse(value if "://" in value else "//" + value).hostname
            if hostname:
                allowed_hosts.add(hostname.lower())
    predictor = predictor if predictor is not None else ExpressionPredictor(BASE_DIR / "models")
    app.extensions["expression_predictor"] = predictor

    @app.before_request
    def check_origin():
        if (urlparse("//" + request.host).hostname or "").lower() not in allowed_hosts:
            return jsonify(error="Unrecognised host."), 403
        origin = request.headers.get("Origin")
        if request.method == "POST" and origin:
            parsed = urlparse(origin)
            if parsed.scheme not in ("http", "https") or parsed.netloc != request.host:
                return jsonify(error="Cross-origin frame upload is not allowed."), 403

    @app.after_request
    def privacy_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
        return response

    @app.get("/")
    def index():
        return render_template("index.html", hosted=hosted)

    @app.get("/health")
    def health():
        return jsonify(predictor.status())

    @app.post("/predict")
    def predict():
        if request.mimetype != "image/jpeg":
            return jsonify(error="Send one JPEG camera frame."), 415
        raw = request.get_data()
        if not raw:
            return jsonify(error="Empty camera frame."), 400
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify(error="Could not read the image."), 400
        height, width = frame.shape[:2]
        if min(height, width) < 60 or max(height, width) > 1920:
            return jsonify(error="Frame must be between 60 and 1920 pixels per side."), 400
        started = time.perf_counter()
        try:
            result = predictor.analyze(frame)
            result["processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return jsonify(result)
        except Exception:
            app.logger.exception("Frame inference failed")
            return jsonify(error="Prediction failed. Try again shortly."), 500

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="Camera frame is too large (limit: 1 MB)."), 413

    return app


app = create_app()


if __name__ == "__main__":
    state = app.extensions["expression_predictor"].status()
    print("\nOpen http://127.0.0.1:5000 in your laptop browser.")
    print("Model:", state["model_name"] if state["ready"] else state["error"])
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
