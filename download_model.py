"""Restore the publicly supplied FER+ ONNX baseline with a SHA256 check."""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
DESTINATION = Path(__file__).resolve().parent / "models" / "emotion-ferplus-8.onnx"


def main():
    manifest = (DESTINATION.parent / "baseline.sha256").read_text().split()[0]
    if DESTINATION.exists() and hashlib.sha256(DESTINATION.read_bytes()).hexdigest() == manifest:
        print("Verified baseline already present.")
        return
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".download")
    try:
        print("Downloading the real pretrained baseline (about 34 MB)…")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != manifest:
            raise RuntimeError("Model checksum mismatch. Download not accepted.")
        temporary.replace(DESTINATION)
        print("Verified model saved to", DESTINATION)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
