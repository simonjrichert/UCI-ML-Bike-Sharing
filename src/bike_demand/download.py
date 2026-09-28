"""Fetch hour.csv from the UCI Bike Sharing archive."""

import io
import urllib.request
import zipfile
from pathlib import Path

from .data import DATA_PATH

ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def download(path: Path = DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZIP_URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    path.write_bytes(archive.read("hour.csv"))
    return path


if __name__ == "__main__":
    written = download()
    print(f"wrote {written} ({written.stat().st_size / 1024:.0f} KB)")
