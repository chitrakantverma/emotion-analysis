"""Evaluate the actual loaded model on the untouched FER-2013 private test split."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np

from src.dataset import load_fer_csv
from src.predictor import CLASSES, ExpressionPredictor


def calculate_metrics(targets, predictions):
    # Column 6 records pretrained-baseline outputs outside our six classes.
    matrix = np.zeros((6, 7), dtype=np.int64)
    for truth, pred in zip(targets, predictions):
        matrix[int(truth), int(pred)] += 1
    rows = []
    for index, label in enumerate(CLASSES):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": label, "precision": precision, "recall": recall, "f1": f1, "support": support})
    return {
        "accuracy": float(np.trace(matrix[:, :6]) / matrix.sum()),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "out_of_scope_predictions": int(matrix[:, 6].sum()),
        "test_examples": int(matrix.sum()), "per_class": rows,
        "confusion_matrix": matrix.tolist(),
        "confusion_columns": CLASSES + ["out_of_scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    model = ExpressionPredictor(args.model_dir)
    if not model.ready:
        parser.error(model.error)
    data, excluded = load_fer_csv(args.csv)
    images, targets = data["PrivateTest"]
    predictions = []
    latencies = []
    # Warm up separately so one-time initialization does not dominate timing.
    model.face_probabilities(images[0])
    for index, image in enumerate(images):
        started = time.perf_counter()
        probs = model.face_probabilities(image)
        latencies.append((time.perf_counter() - started) * 1000)
        label = max(probs, key=probs.get)
        predictions.append(CLASSES.index(label) if label in CLASSES else 6)
        if (index + 1) % 500 == 0:
            print(f"Evaluated {index+1}/{len(images)} images")
    report = calculate_metrics(targets, predictions)
    report.update({
        "model_name": model.model_name, "split": "PrivateTest", "excluded_disgust_rows_all_splits": excluded,
        "mean_inference_ms": float(np.mean(latencies)), "p95_inference_ms": float(np.percentile(latencies, 95)),
        "timing_note": "Classification on cropped dataset faces only. Webcam detection/network/UI latency is separate.",
        "benchmark_note": "Filtered six-class FER-2013 evaluation. Not comparable directly to published seven/eight-class scores.",
    })
    output = Path(args.reports)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    with (output / "per_class_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(report["per_class"])
    with (output / "confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual / predicted"] + report["confusion_columns"])
        writer.writerows([[label] + row for label, row in zip(CLASSES, report["confusion_matrix"])])
    print(json.dumps({key: report[key] for key in ("model_name", "test_examples", "accuracy", "macro_f1", "mean_inference_ms")}, indent=2))
    print("Real results saved in", output)


if __name__ == "__main__":
    main()
