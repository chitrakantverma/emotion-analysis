"""API, dataset and inference checks."""
import os
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app import BASE_DIR, create_app
from evaluate import calculate_metrics
from src.dataset import load_fer_csv
from src.predictor import CLASSES, ExpressionPredictor, softmax


def jpeg(size=100):
    ok, encoded = cv2.imencode(".jpg", np.zeros((size, size, 3), np.uint8))
    assert ok
    return encoded.tobytes()


class FakePredictor:
    def status(self):
        return {"ready": True, "model_name": "test double"}

    def analyze(self, frame):
        return {"status": "no_face", "face_detected": False, "face_count": 0}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(FakePredictor())
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_home_and_health(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Webcam", response.data)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertTrue(self.client.get("/health").json["ready"])

    def test_jpeg_request(self):
        response = self.client.post("/predict", data=jpeg(), content_type="image/jpeg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "no_face")
        self.assertGreaterEqual(response.json["processing_ms"], 0)

    def test_invalid_payloads(self):
        self.assertEqual(self.client.post("/predict", json={}).status_code, 415)
        self.assertEqual(self.client.post("/predict", data=b"", content_type="image/jpeg").status_code, 400)
        self.assertEqual(self.client.post("/predict", data=b"not jpeg", content_type="image/jpeg").status_code, 400)
        self.assertEqual(self.client.post("/predict", data=jpeg(20), content_type="image/jpeg").status_code, 400)
        self.assertEqual(self.client.post("/predict", data=b"x" * (1024 * 1024 + 1), content_type="image/jpeg").status_code, 413)

    def test_local_only(self):
        self.assertEqual(self.client.get("/", base_url="http://example.com").status_code, 403)
        self.assertEqual(self.client.post("/predict", data=jpeg(), content_type="image/jpeg",
                                         headers={"Origin": "https://example.com"}).status_code, 403)
        self.assertEqual(self.client.post("/predict", data=jpeg(), content_type="image/jpeg",
                                         headers={"Origin": "http://localhost"}).status_code, 200)

    def test_model_exception_is_clear(self):
        with patch.object(self.app.extensions["expression_predictor"], "analyze", side_effect=RuntimeError("test error")):
            with self.assertLogs(self.app.logger, level="ERROR"):
                response = self.client.post("/predict", data=jpeg(), content_type="image/jpeg")
        self.assertEqual(response.status_code, 500)
        self.assertIn("Prediction failed", response.json["error"])

    def test_static_assets(self):
        for name in ("style.css", "camera.js"):
            response = self.client.get("/static/" + name)
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_hosted_app_and_same_origin_prediction(self):
        environment = {"VERCEL": "1", "VERCEL_URL": "emotion-preview.vercel.app",
                       "VERCEL_PROJECT_PRODUCTION_URL": "emotion-demo.vercel.app"}
        with patch.dict(os.environ, environment):
            hosted = create_app(FakePredictor()).test_client()
        response = hosted.get("/", base_url="https://emotion-demo.vercel.app")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Frames are sent to this server", response.data)
        self.assertEqual(hosted.get("/health", base_url="https://emotion-preview.vercel.app").status_code, 200)
        self.assertEqual(hosted.get("/", base_url="https://unrelated.vercel.app").status_code, 403)
        response = hosted.post("/predict", base_url="https://emotion-demo.vercel.app",
                               data=jpeg(), content_type="image/jpeg",
                               headers={"Origin": "https://emotion-demo.vercel.app"})
        self.assertEqual(response.status_code, 200)
        response = hosted.post("/predict", base_url="https://emotion-demo.vercel.app",
                               data=jpeg(), content_type="image/jpeg",
                               headers={"Origin": "https://another-site.example"})
        self.assertEqual(response.status_code, 403)

    def test_custom_domain(self):
        with patch.dict(os.environ, {"APP_HOST": "emotion.example.org"}):
            hosted = create_app(FakePredictor()).test_client()
        self.assertEqual(hosted.get("/health", base_url="https://emotion.example.org").status_code, 200)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = ExpressionPredictor(BASE_DIR / "models")

    def test_actual_included_cnn(self):
        self.assertTrue(self.predictor.ready, self.predictor.error)
        probabilities = self.predictor.face_probabilities(np.zeros((48, 48), np.uint8))
        self.assertEqual(len(probabilities), len(self.predictor.classes))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
        self.assertTrue(all(0 <= p <= 1 for p in probabilities.values()))

    def test_blank_frame_has_no_expression(self):
        result = self.predictor.analyze(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(result["status"], "no_face")
        self.assertNotIn("expression", result)

    def test_missing_model_is_not_fake(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = ExpressionPredictor(folder)
            self.assertFalse(missing.ready)
            self.assertIn("Model missing", missing.error)

    def test_largest_face_and_out_of_scope(self):
        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.array([[0, 0, 65, 65], [10, 10, 80, 80]])
        with patch.object(self.predictor, "detector", Detector()):
            with patch.object(self.predictor, "face_probabilities", return_value={"disgust": .8, "happy": .2}):
                result = self.predictor.analyze(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(result["face_count"], 2)
        self.assertEqual(result["box"]["width"], 80)
        self.assertEqual(result["status"], "out_of_scope")
        self.assertIsNone(result["expression"])
        self.assertEqual(result["raw_expression"], "disgust")

    def test_softmax_stable(self):
        self.assertAlmostEqual(softmax([1000, 1001]).sum(), 1)
        with self.assertRaises(ValueError):
            softmax([float("nan")])


class DatasetAndMetricsTests(unittest.TestCase):
    def test_six_class_mapping_and_preserved_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["emotion", "pixels", "Usage"])
                for split in ("Training", "PublicTest", "PrivateTest"):
                    for label in range(7):
                        writer.writerow([label, " ".join([str(label)] * 2304), split])
            data, excluded = load_fer_csv(path)
            self.assertEqual(excluded, 3)
            for images, targets in data.values():
                self.assertEqual(images.shape, (6, 48, 48))
                self.assertEqual(targets.tolist(), list(range(6)))
                self.assertEqual(images[:, 0, 0].tolist(), [0, 2, 3, 4, 5, 6])

    def test_bad_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text("label,pixels\n0,1\n")
            with self.assertRaisesRegex(ValueError, "columns"):
                load_fer_csv(path)

    def test_outside_scope_counts_as_wrong(self):
        result = calculate_metrics([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6])
        self.assertAlmostEqual(result["accuracy"], 5 / 6)
        self.assertEqual(result["out_of_scope_predictions"], 1)
        self.assertEqual(result["per_class"][5]["recall"], 0)
        self.assertEqual(result["confusion_columns"], CLASSES + ["out_of_scope"])


if __name__ == "__main__":
    unittest.main()
