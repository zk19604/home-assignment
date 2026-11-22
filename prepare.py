"""Prepare the official YOLO11l checkpoint beside run.py. Internet is allowed here."""

import os
from pathlib import Path
import shutil
import tempfile


def main() -> None:
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    target = Path(__file__).resolve().with_name("yolo11l.pt")
    if target.exists():
        YOLO(str(target), task="detect")
    else:
        # Publish only a checkpoint that Ultralytics successfully deserialized.
        # Supplying an absolute destination keeps downloads independent of cwd.
        with tempfile.TemporaryDirectory(dir=target.parent) as folder:
            downloaded = Path(attempt_download_asset(str(Path(folder) / target.name)))
            YOLO(str(downloaded), task="detect")
            staged = Path(folder) / "verified.pt"
            shutil.copyfile(downloaded, staged)
            os.replace(staged, target)
    print(f"Verified {target}")


if __name__ == "__main__":
    main()
