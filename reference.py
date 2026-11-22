"""Independent, serial reference for small validation samples."""

from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction
import io
from pathlib import Path
from statistics import median
import subprocess
import tempfile
from typing import Any, Callable, Iterator

import av
from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np


def timestamps(payload: bytes) -> Iterator[Fraction]:
    """Read exact presentation timestamps before any display transformation."""
    with av.open(io.BytesIO(payload), options={"protocol_whitelist": "file,pipe"}) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        try:
            fps = Fraction(stream.average_rate or 0)
        except (ValueError, ZeroDivisionError, OverflowError):
            fps = Fraction(0)
        step = Fraction(1, 1) / fps if fps > 0 else Fraction(1, 30)
        next_stamp = Fraction(0)
        for frame in container.decode(stream):
            base = frame.time_base or stream.time_base
            stamp = next_stamp
            if frame.pts is not None and base and base > 0:
                stamp = Fraction(frame.pts) * base
            yield stamp
            next_stamp = stamp + step


def displayed_frames(path: Path) -> Iterator[np.ndarray]:
    """Let FFmpeg apply its native autorotation and emit one PPM per frame."""
    command = [get_ffmpeg_exe(), "-nostdin", "-v", "error", "-xerror",
               "-protocol_whitelist", "file,pipe", "-threads", "1", "-i", str(path),
               "-map", "0:v:0", "-filter_threads", "1", "-vsync", "0",
               "-threads", "1", "-c:v", "ppm", "-pix_fmt", "rgb24", "-f", "image2pipe", "pipe:1"]
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors) as process:
        pipe = process.stdout
        assert pipe is not None
        try:
            while header := pipe.readline():
                if header.strip() != b"P6":
                    raise ValueError("invalid FFmpeg PPM header")
                width, height = map(int, pipe.readline().split())
                if pipe.readline().strip() != b"255" or not 0 < width * height <= 8192**2:
                    raise ValueError("unsupported FFmpeg frame")
                raw = pipe.read(width * height * 3)
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                yield np.ascontiguousarray(rgb[:, :, ::-1])
            if process.wait():
                errors.seek(0)
                raise ValueError(errors.read(180).decode(errors="replace") or "FFmpeg decode failed")
        finally:
            pipe.close()
            if process.poll() is None:
                process.kill()


def reference_video(video: str, payload: bytes | None, model: Any, device: str,
                    emit: Callable[[tuple], None]) -> None:
    """Use native FFmpeg orientation, original PTS, FP32 and per-frame voting."""
    classes = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
    people: dict[int, list[int]] = defaultdict(list)
    votes: dict[int, Counter] = defaultdict(Counter)
    decoded = inferred = 0
    last = -1
    error = ""
    try:
        if not payload:
            raise ValueError("empty or null mp4")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.mp4"
            path.write_bytes(payload)
            origin = None
            with closing(timestamps(payload)) as stamps, closing(displayed_frames(path)) as frames:
                for stamp, bgr in zip(stamps, frames, strict=True):
                    decoded += 1
                    if origin is None:
                        origin = stamp
                    second = (stamp - origin) // 1
                    if second < 0:
                        raise ValueError("timestamp precedes first frame")
                    result = model.predict([bgr], device=device, imgsz=640, conf=0.4, iou=0.7,
                                           classes=list(classes), half=False, verbose=False)[0]
                    counts = Counter(int(c) for c in result.boxes.cls.tolist())
                    people[second].append(counts[0])
                    votes[second].update(counts.keys())
                    inferred += 1
                    last = second
        if not inferred:
            raise ValueError("could not decode video")
    except Exception as exc:
        error = (" ".join(str(exc).split()) or type(exc).__name__)[:200]
        last = max(people, default=0) - 1
    for s in range(last + 1):
        persons = people.get(s, [])
        names = [name for cls, name in classes.items()
                 if persons and 2 * votes[s][cls] >= len(persons)]
        emit(("rows", [dict(video=video, second=s, objects=names,
                            people=int(median(persons)) if persons else 0)]))
    if error:
        emit(("rows", [dict(video=video, error=error)]))
    emit(("video", decoded, inferred))
