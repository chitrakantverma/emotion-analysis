"""Train a six-class CNN on FER-2013."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

from src.dataset import load_fer_csv
from src.predictor import CLASSES


def build_model(tf):
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Input(shape=(48, 48, 1)),
        # These run only during training. Validation/test images are not augmented.
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03, fill_mode="nearest"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(len(CLASSES), activation="softmax"),
    ], name="expression_cnn")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Original fer2013.csv path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="models")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("Epochs and batch size must be positive.")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    data, excluded = load_fer_csv(args.csv)
    train_x, train_y = data["Training"]
    valid_x, valid_y = data["PublicTest"]
    train_x = train_x[..., None].astype(np.float32) / 255.0
    valid_x = valid_x[..., None].astype(np.float32) / 255.0
    class_counts = np.bincount(train_y, minlength=6)
    class_weights = {i: float(len(train_y) / (6 * count)) for i, count in enumerate(class_counts)}
    output = Path(args.output)
    reports = Path(args.reports)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    model = build_model(tf)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    print("Training class counts:", dict(zip(CLASSES, class_counts.tolist())))
    print("Excluded disgust rows:", excluded)
    history = model.fit(
        train_x, train_y, validation_data=(valid_x, valid_y),
        epochs=args.epochs, batch_size=args.batch_size, class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )
    model.save(output / "emotion_model.keras")
    metadata = {
        "classes": CLASSES, "input_size": 48, "normalization": "0_1",
        "dataset": "FER-2013 (disgust excluded)",
        "train_split": "Training", "validation_split": "PublicTest", "test_split": "PrivateTest",
        "seed": 42, "epochs_completed": len(history.history["loss"]),
        "dataset_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((output / "emotion_model.keras").read_bytes()).hexdigest(),
        "training_counts": dict(zip(CLASSES, class_counts.tolist())),
    }
    (output / "emotion_model.json").write_text(json.dumps(metadata, indent=2))
    (reports / "training_history.json").write_text(json.dumps(history.history, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("accuracy", "loss")):
        axis.plot(history.history[metric], label="Training")
        axis.plot(history.history["val_" + metric], label="Validation")
        axis.set_title(metric.title())
        axis.set_xlabel("Epoch (zero-indexed)")
        axis.legend()
    figure.tight_layout()
    figure.savefig(reports / "training_curves.png", dpi=160)
    plt.close(figure)
    print("Model saved. Restart app.py to load it.")
    print("Now run: python evaluate.py --csv", args.csv, "--model-dir", args.output)


if __name__ == "__main__":
    main()
