from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import shutil

MODEL_URL = (
    "https://huggingface.co/Prakash999/Braille_Dot_Detection/"
    "resolve/main/weights/yolov8_braille.pt"
)

DESTINATION = Path("weights/yolov8_braille.pt")

EXPECTED_SHA256 = (
    "aeb1fe31f75dd9e3860434d4067c7537b1159a407ebc2848156aad8cda4ed842"
)

DESTINATION.parent.mkdir(parents=True, exist_ok=True)

print("Downloading YOLO Braille model...")

with urlopen(MODEL_URL, timeout=300) as response:
    with DESTINATION.open("wb") as output:
        shutil.copyfileobj(response, output)

digest = sha256()

with DESTINATION.open("rb") as model_file:
    for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
        digest.update(chunk)

actual_hash = digest.hexdigest()

if actual_hash != EXPECTED_SHA256:
    raise RuntimeError(
        f"Model checksum mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
    )

print(f"Downloaded valid model: {DESTINATION.stat().st_size} bytes")
