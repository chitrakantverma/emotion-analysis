"""Read the original FER-2013 CSV without mixing its official splits."""
import csv
from pathlib import Path
import numpy as np
from src.predictor import CLASSES

# Original FER-2013 IDs: angry=0, disgust=1, fear=2, happy=3,
# sad=4, surprise=5, neutral=6. Exclude disgust, never relabel it.
LABEL_MAP = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
SPLITS = ("Training", "PublicTest", "PrivateTest")


def load_fer_csv(path):
    path = Path(path)
    images = {split: [] for split in SPLITS}
    targets = {split: [] for split in SPLITS}
    excluded = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"emotion", "pixels", "Usage"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected FER-2013 CSV columns: emotion, pixels, Usage.")
        for line_number, row in enumerate(reader, start=2):
            try:
                label = int(row["emotion"])
                split = row["Usage"].strip()
                if label not in range(7) or split not in SPLITS:
                    raise ValueError("Unknown expression ID or split.")
                if label == 1:
                    excluded += 1
                    continue
                pixels = np.fromstring(row["pixels"], dtype=np.int32, sep=" ")
                if pixels.size != 48 * 48 or np.any(pixels < 0) or np.any(pixels > 255):
                    raise ValueError("Expected 2304 pixel values between 0 and 255.")
                images[split].append(pixels.astype(np.uint8).reshape(48, 48))
                targets[split].append(LABEL_MAP[label])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid FER CSV at row {line_number}: {exc}") from exc
    result = {}
    for split in SPLITS:
        if not images[split]:
            raise ValueError(f"Missing {split} data. Use the original CSV with all official splits.")
        result[split] = (np.stack(images[split]), np.asarray(targets[split], dtype=np.int64))
        if set(result[split][1].tolist()) != set(range(len(CLASSES))):
            raise ValueError(f"{split} does not contain all six expression classes.")
    return result, excluded
