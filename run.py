"""Offline YOLO11l video inference over a directory of Parquet shards."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from fractions import Fraction
import io
import json
import math
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
from queue import Full, Queue
import socket
import struct
import sys
import tempfile
from threading import Event
import time
from typing import Any, Callable, Iterator

import av
import numpy as np
import pyarrow.parquet as pq


IDS = [0, 2, 16, 63, 67]
NAMES = ["person", "car", "dog", "laptop", "cell phone"]
WEIGHTS = Path(__file__).resolve().with_name("yolo11l.pt")
MIB = 1024 * 1024
Emit = Callable[[tuple], None]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def short_error(exc: Exception) -> str:
    return (" ".join(str(exc).split()) or type(exc).__name__)[:200]


def offline() -> None:
    """Set inference-process policy before importing torch or Ultralytics."""
    os.environ.update(YOLO_OFFLINE="1", YOLO_AUTOINSTALL="false",
                      YOLO_VERBOSE="false", HF_HUB_OFFLINE="1")
    # Also cover package versions/integrations that probe connectivity on import.
    # Local multiprocessing uses pipes. No inference component needs IP sockets.
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError("network disabled during inference")
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            import errno
            return errno.ENETUNREACH
        return original_connect_ex(sock, address)

    def offline_lookup(*args, **kwargs):
        raise socket.gaierror("network disabled during inference")

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = offline_lookup


def boxes(data: memoryview, lo: int, hi: int) -> Iterator[tuple[bytes, int, int]]:
    """Walk immediate ISO BMFF children without copying compressed video data."""
    while lo + 8 <= hi:
        size, kind = struct.unpack_from(">I4s", data, lo)
        header = 8
        if size == 1:
            if lo + 16 > hi:
                return
            size = struct.unpack_from(">Q", data, lo + 8)[0]
            header = 16
        elif size == 0:
            size = hi - lo
        if size < header or lo + size > hi:
            return
        yield kind, lo + header, lo + size
        lo += size


def track_matrix(payload: bytes, track_id: int) -> tuple[int, ...] | None:
    """Read the MP4 track matrix when PyAV only exposes frame side data."""
    data = memoryview(payload)
    for kind, a, b in boxes(data, 0, len(data)):
        if kind != b"moov":
            continue
        for kind, c, d in boxes(data, a, b):
            if kind != b"trak":
                continue
            for kind, p, end in boxes(data, c, d):
                if kind != b"tkhd" or p >= end or data[p] not in (0, 1):
                    continue
                long_times = data[p] == 1
                matrix_offset = p + (52 if long_times else 40)
                if matrix_offset + 36 > end:
                    continue
                ident = struct.unpack_from(">I", data, p + (20 if long_times else 12))[0]
                if ident == track_id:
                    return struct.unpack_from(">9i", data, matrix_offset)
    return None


def display_matrix(owner: Any) -> tuple[int, ...] | None:
    for side in getattr(owner, "side_data", ()) or ():
        if getattr(getattr(side, "type", None), "name", "") == "DISPLAYMATRIX":
            raw = bytes(side)
            if len(raw) >= 36:
                return struct.unpack_from("=9i", raw)
    return None


def orientation(matrix: tuple[int, ...] | None, rotate: float) -> tuple[int, bool, float]:
    """Return CCW quarter-turns, a subsequent horizontal flip, and residual CCW degrees."""
    if matrix is None:
        # FFmpeg's legacy "rotate" stream tag gives the clockwise display turn.
        ccw = -rotate
        mirror = False
    else:
        a, b, _, c, d, *_ = matrix
        sx, sy = math.hypot(a, c), math.hypot(b, d)
        if not sx or not sy:
            return orientation(None, rotate)
        # FFmpeg's display matrix maps the source x-axis to (a,b) in a
        # y-down image. av_display_rotation_get reports the opposite angle.
        ccw = -math.degrees(math.atan2(b / sy, a / sx))
        mirror = a * d - b * c < 0
        if mirror:
            # Factor the matrix into a rotation followed by an output x flip.
            ccw = 180 - ccw
    quarter = round(ccw / 90)
    if abs(ccw - 90 * quarter) < 1:
        return quarter % 4, mirror, 0.0
    return 0, mirror, ccw


def rotate_frame(frame: np.ndarray, transform: tuple[int, bool, float]) -> np.ndarray:
    """Apply display orientation in BGR space. Positive np.rot90 turns CCW."""
    turns, flip, angle = transform
    if turns:
        frame = np.rot90(frame, turns)
    if angle:
        # Like FFmpeg's rotate filter, keep the original canvas for an oblique angle.
        import cv2
        h, w = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1)
        frame = cv2.warpAffine(frame, matrix, (w, h))
    if flip:
        frame = np.flip(frame, axis=1)
    return np.ascontiguousarray(frame)


@dataclass
class Batch:
    video: str
    seconds: list[int]
    frames: list[np.ndarray]


@dataclass
class End:
    video: str
    decoded: int = 0
    last_second: int = -1
    error: str = ""
    decode_seconds: float = 0.0


def decode_video(video: str, payload: bytes | None, batch_size: int, stride: int,
                 threads: int, send: Callable[[Batch | End], None], cancel: Event,
                 source_seconds: int | None = None) -> None:
    """Decode in presentation order and submit homogeneous batches for one video."""
    end = End(video)
    started = time.perf_counter()
    send_seconds = 0.0

    def submit(batch: Batch) -> None:
        nonlocal send_seconds
        before = time.perf_counter()
        send(batch)
        send_seconds += time.perf_counter() - before

    try:
        if not payload:
            raise ValueError("empty or null mp4")
        with io.BytesIO(payload) as source, av.open(
                source, mode="r", options={"protocol_whitelist": "file,pipe"}) as container:
            if not container.streams.video:
                raise ValueError("no video stream")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = threads
            # Bound individual decoder allocations, including malformed dimensions.
            stream.codec_context.options = {
                **stream.codec_context.options, "max_pixels": str(8192 * 8192)}
            try:
                rate = Fraction(stream.average_rate or 0)
            except (ValueError, OverflowError, ZeroDivisionError):
                rate = Fraction(0)
            step = 1 / rate if rate > 0 else Fraction(1, 30)
            try:
                tag = float(stream.metadata.get("rotate", 0))
                if not math.isfinite(tag):
                    tag = 0.0
            except (TypeError, ValueError):
                tag = 0.0
            matrix = display_matrix(stream)
            if matrix is None:
                matrix = track_matrix(payload, stream.id)
                if matrix == (65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30) and tag:
                    matrix = None
            base_transform = orientation(matrix, tag)
            t0 = None
            anchor_time, anchor_index = Fraction(0), 0
            seen: set[int] = set()
            batch = Batch(video, [], [])
            size = 0
            for index, frame in enumerate(container.decode(stream)):
                if cancel.is_set():
                    break
                end.decoded += 1
                tb = frame.time_base or stream.time_base
                if frame.pts is not None and tb is not None and tb > 0:
                    stamp = frame.pts * Fraction(tb)
                    anchor_time, anchor_index = stamp, index
                else:
                    stamp = anchor_time + (index - anchor_index) * step
                if t0 is None:
                    t0 = stamp
                second = math.floor(stamp - t0)
                if second < 0:
                    raise ValueError("timestamp precedes first frame")
                if source_seconds is not None and second >= source_seconds:
                    end.last_second = source_seconds - 1
                    break
                end.last_second = second
                if stride > 1:
                    if second in seen and index % stride:
                        continue
                    seen.add(second)
                side = display_matrix(frame)
                transform = base_transform if side is None else orientation(side, tag)
                if frame.width * frame.height > 8192 * 8192:
                    raise ValueError("frame exceeds decoder pixel limit")
                pixels = rotate_frame(frame.to_ndarray(format="bgr24"), transform)
                if batch.frames and (batch.frames[0].shape != pixels.shape or
                                     size + pixels.nbytes > 192 * MIB):
                    submit(batch)
                    batch, size = Batch(video, [], []), 0
                batch.frames.append(pixels)
                batch.seconds.append(second)
                size += pixels.nbytes
                if len(batch.frames) >= batch_size:
                    submit(batch)
                    batch, size = Batch(video, [], []), 0
            if batch.frames and not cancel.is_set():
                submit(batch)
            if end.decoded == 0:
                end.error = "could not decode video"
    except Exception as exc:
        end.error = short_error(exc)
    finally:
        end.decode_seconds = time.perf_counter() - started - send_seconds
        send(end)


@dataclass
class Second:
    hits: list[int] = field(default_factory=lambda: [0] * len(IDS))
    people: list[int] = field(default_factory=list)


def aggregate_second(video: str, second: int, data: Second | None) -> dict:
    """Compute the half-of-frames presence rule and floored median."""
    objects, people = [], 0
    if data and data.people:
        counts = sorted(data.people)
        n = len(counts)
        objects = [name for name, hits in zip(NAMES, data.hits) if hits * 2 >= n]
        people = (counts[(n - 1) // 2] + counts[n // 2]) // 2
    return dict(video=video, second=second, objects=objects, people=people)


class Detector:
    def __init__(self, args: argparse.Namespace, device: str):
        offline()
        if device != "cpu":
            # Establish the mask before CUDA initialization. Preserve a scheduler's
            # existing mapping, including masks expressed as GPU/MIG UUIDs.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is not None:
                slots = [slot.strip() for slot in visible.split(",") if slot.strip()]
                if int(device) >= len(slots):
                    raise ValueError(f"device {device} is outside CUDA_VISIBLE_DEVICES")
                device = slots[int(device)]
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        import torch
        from ultralytics import YOLO, settings
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} is unavailable")
        if settings.get("sync"):
            settings.update({"sync": False})
        self.torch = torch
        self.model = YOLO(str(args.weights), task="detect")
        # FP16 can change borderline confidences/NMS decisions relative to FP32.
        half = device != "cpu" and args.half is not False and not args.reference
        # Empty CUDA selector leaves the one-device process mask in place.
        self.options = dict(device="cpu" if device == "cpu" else "", imgsz=640, conf=0.4, iou=0.7,
                            classes=IDS, half=half, verbose=False, agnostic_nms=False)
        self.model.predict([np.zeros((640, 640, 3), dtype=np.uint8)], **self.options)

    def consume(self, batch: Batch, seconds: dict[int, Second]) -> None:
        """Transfer class IDs once per batch. YOLO applies the confidence filter."""
        results = self.model.predict(batch.frames, **self.options)
        if len(results) != len(batch.seconds):
            raise RuntimeError("model returned wrong number of results")
        classes = [result.boxes.cls for result in results]
        lengths = [len(c) for c in classes]
        flat = self.torch.cat(classes).to(dtype=self.torch.int64).cpu().numpy()
        offset = 0
        for s, length in zip(batch.seconds, lengths):
            counts = np.bincount(flat[offset:offset + length], minlength=68)
            offset += length
            if s not in seconds:
                seconds[s] = Second()
            bucket = seconds[s]
            bucket.people.append(int(counts[0]))
            for i, ident in enumerate(IDS):
                bucket.hits[i] += int(counts[ident] > 0)


@dataclass
class VideoState:
    seconds: dict[int, Second] = field(default_factory=dict)
    cancel: Event = field(default_factory=Event)
    error: str = ""
    inference_seconds: float = 0.0


def consume(event: Batch | End, state: VideoState, detector: Detector, emit: Emit) -> None:
    if isinstance(event, Batch):
        if not state.error:
            before = time.perf_counter()
            try:
                detector.consume(event, state.seconds)
            except Exception as exc:
                state.error = short_error(exc)
                state.cancel.set()
            finally:
                state.inference_seconds += time.perf_counter() - before
        return
    error = state.error or event.error
    # For a failed video, the last inferred bucket may only be partly decoded.
    last = max(state.seconds, default=0) - 1 if error else event.last_second
    for start in range(0, last + 1, 128):
        emit(("rows", [aggregate_second(event.video, s, state.seconds.get(s))
                       for s in range(start, min(last + 1, start + 128))]))
    if error:
        emit(("rows", [dict(video=event.video, error=error)]))
    emit(("video", event.decoded, sum(len(s.people) for s in state.seconds.values()),
          event.decode_seconds, state.inference_seconds))


@dataclass
class FileStatus:
    complete: bool = False
    error: str = ""


def _parquet_rows(path: Path, limit: int | None,
                  status: FileStatus, emit: Emit) -> Iterator[tuple[str, bytes | None]]:
    """Read one mp4 cell at a time. Offsets include rows in unreadable groups."""
    with pq.ParquetFile(path) as parquet:
        total = parquet.metadata.num_rows
        wanted = total if limit is None else min(limit, total)

        def report_unread(index: int, reason: str) -> None:
            emit(("rows", [dict(video=f"{path.name}:{index}", error=reason)]))
            emit(("video", 0, 0))

        if "mp4" not in parquet.schema_arrow.names:
            for index in range(wanted):
                report_unread(index, "missing mp4 column")
            status.complete = wanted == total
            return
        offset = 0
        for group in range(parquet.num_row_groups):
            if offset >= wanted:
                break
            try:
                count = parquet.metadata.row_group(group).num_rows
            except Exception as exc:
                # The file footer still gives a row range even if this group's
                # descriptor cannot tell us where the following group begins.
                reason = short_error(exc)
                log(f"{path.name} row group {group}: {reason}")
                for index in range(offset, wanted):
                    report_unread(index, reason)
                break
            read = 0
            try:
                batches = parquet.iter_batches(batch_size=1, columns=["mp4"],
                                               row_groups=[group], use_threads=False)
                for batch in batches:
                    for cell in batch.column(0):
                        index = offset + read
                        read += 1
                        try:
                            payload = cell.as_py()
                        except Exception as exc:
                            report_unread(index, short_error(exc))
                            continue
                        yield f"{path.name}:{index}", payload
                    if offset + read >= wanted:
                        break
                if read != min(count, wanted - offset):
                    raise ValueError("unexpected row count")
            except Exception as exc:
                reason = short_error(exc)
                log(f"{path.name} row group {group}: {reason}")
                for index in range(offset + read, min(offset + count, wanted)):
                    report_unread(index, reason)
            offset += count
        status.complete = wanted == total and not status.error


def iter_videos(path: Path, limit: int | None,
                status: FileStatus, emit: Emit) -> Iterator[tuple[str, bytes | None]]:
    """Isolate file-open, footer and schema failures from subsequent shards."""
    try:
        yield from _parquet_rows(path, limit, status, emit)
    except Exception as exc:
        status.complete = False
        status.error = short_error(exc)


def pipeline(rows: Iterator, detector: Detector, args: argparse.Namespace, emit: Emit) -> None:
    """Bound both active compressed videos and queued batches of decoded frames."""
    workers = args.workers
    queue: Queue = Queue(maxsize=2)
    stopping = Event()
    active: dict[str, VideoState] = {}
    threads = max(1, min(4, (os.cpu_count() or 1) // workers))

    def send(event: Batch | End) -> None:
        while not stopping.is_set():
            try:
                queue.put(event, timeout=0.2)
                return
            except Full:
                continue

    with ThreadPoolExecutor(max_workers=workers) as pool:
        exhausted = False
        try:
            while active or not exhausted:
                while not exhausted and len(active) < workers:
                    item = next(rows, None)
                    if item is None:
                        exhausted = True
                        break
                    video, payload = item
                    state = active[video] = VideoState()
                    pool.submit(decode_video, video, payload, args.batch, args.stride,
                                threads, send, state.cancel)
                    del item, payload
                if active:
                    event = queue.get()
                    consume(event, active[event.video], detector, emit)
                    if isinstance(event, End):
                        del active[event.video]
                    del event
        finally:
            stopping.set()
            for state in active.values():
                state.cancel.set()


def run_files(paths: list[Path], args: argparse.Namespace, device: str, emit: Emit) -> None:
    detector = Detector(args, device)
    remaining = args.source_seconds

    def count_seconds(event: tuple) -> None:
        nonlocal remaining
        if remaining is not None and event[0] == "rows":
            remaining -= sum("second" in row for row in event[1])
        emit(event)

    for path in paths:
        if remaining == 0:
            break
        status = FileStatus()
        with closing(iter_videos(path, args.limit_videos, status, emit)) as rows:
            if args.reference:
                from reference import reference_video
                for video, payload in rows:
                    reference_video(video, payload, detector.model, detector.options["device"], emit)
            elif remaining is not None:
                for video, payload in rows:
                    state = VideoState()

                    def accept(event: Batch | End) -> None:
                        consume(event, state, detector, count_seconds)

                    decode_video(video, payload, args.batch, args.stride, 2, accept,
                                 state.cancel, remaining)
                    if remaining == 0:
                        break
            else:
                pipeline(rows, detector, args, emit)
        if status.error:
            log(f"{path.name}: {status.error}")
            emit(("rows", [dict(video=f"{path.name}:file", error=status.error)]))
        emit(("file", path.name, status.complete))


def child(paths: list[Path], args: argparse.Namespace, device: str, pipe: Connection) -> None:
    try:
        run_files(paths, args, device, pipe.send)
    finally:
        pipe.close()


def multi_gpu(paths: list[Path], args: argparse.Namespace, devices: list[str], emit: Emit) -> None:
    """Spawn before CUDA initialization. The parent exclusively owns output files."""
    ctx = mp.get_context("spawn")
    readers = {}
    processes = []
    try:
        for i, device in enumerate(devices):
            subset = paths[i::len(devices)]
            if not subset:
                break
            reader, sender = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=child, args=(subset, args, device, sender))
            proc.start()
            sender.close()
            processes.append(proc)
            readers[reader] = proc
        while readers:
            for reader in wait(list(readers)):
                try:
                    event = reader.recv()
                except EOFError:
                    readers.pop(reader)
                    reader.close()
                else:
                    emit(event)
        for proc in processes:
            proc.join()
        if any(proc.exitcode != 0 for proc in processes):
            raise RuntimeError("GPU worker exited unsuccessfully; retry with --resume")
    finally:
        for reader in readers:
            reader.close()
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
            proc.join()


def clean_resume(output: Path, selected: set[str], limit: int | None) -> None:
    """Atomically remove stale rows for selected work and a torn final record."""
    with output.open("rb") as old, tempfile.NamedTemporaryFile(
            dir=output.parent, delete=False) as new:
        temporary = Path(new.name)
        try:
            for line in old:
                if not line.endswith(b"\n"):
                    break
                video = json.loads(line)["video"]
                name, index = video.rsplit(":", 1)
                if name in selected and (index == "file" or limit is None or int(index) < limit):
                    continue
                new.write(line)
            new.flush()
            os.fsync(new.fileno())
            new.close()
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)


class Writer:
    """Write complete JSON lines and checkpoint only after output is durable."""
    def __init__(self, output, sidecar, total: int):
        self.output, self.sidecar, self.total = output, sidecar, total
        self.stats: Counter = Counter()
        self.start = self.last = time.monotonic()

    def __call__(self, event: tuple) -> None:
        kind, *args = event
        if kind == "rows":
            self.output.writelines(json.dumps(row) + "\n" for row in args[0])
            self.output.flush()
            return
        if kind == "video":
            self.stats["videos"] += 1
            self.stats["decoded"] += args[0]
            self.stats["inferred"] += args[1]
            if len(args) == 4:
                self.stats["decode_s"] += args[2]
                self.stats["inference_s"] += args[3]
        elif kind == "file":
            name, complete = args
            self.stats["files"] += 1
            if complete:
                self.output.flush()
                os.fsync(self.output.fileno())
                self.sidecar.write(name + "\n")
                self.sidecar.flush()
                os.fsync(self.sidecar.fileno())
        now = time.monotonic()
        if kind == "file" or now - self.last >= 5:
            s = self.stats
            log(f"files {s['files']}/{self.total}, videos {s['videos']}, "
                f"{s['decoded'] / max(now - self.start, 1e-6):.1f} decoded frames/s, "
                f"{s['inferred']} inferred frames")
            self.last = now
            if kind == "file" and "decode_s" in s:
                log(f"stage totals: {s['decode_s']:.2f} decoder-worker seconds, "
                    f"{s['inference_s']:.2f} inference and counting seconds")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="directory of Parquet shards")
    parser.add_argument("--output", required=True, type=Path, help="destination JSONL")
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="local YOLO11l .pt file")
    parser.add_argument("--workers", type=positive, default=min(os.cpu_count() or 1, 16),
                        help="decoder threads per GPU process")
    parser.add_argument("--batch", type=positive, default=32, help="maximum inference batch")
    parser.add_argument("--devices", default="0", help="comma-separated GPU indices or cpu")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=None,
                        help="FP16 inference (default on CUDA)")
    parser.add_argument("--stride", type=positive, default=1, help="sample every Nth frame")
    parser.add_argument("--reference", action="store_true", help="serial FP32 validation mode")
    parser.add_argument("--resume", action="store_true", help="skip checkpointed files")
    parser.add_argument("--limit-videos", type=nonnegative, help="first N rows of each file")
    parser.add_argument("--source-seconds", type=positive,
                        help="stop after N output seconds of source footage, including gaps")
    parser.add_argument("--limit-files", type=nonnegative, help="first N remaining files")
    args = parser.parse_args()
    args.weights = args.weights.resolve()
    if not args.weights.is_file() or args.weights.suffix.lower() != ".pt":
        parser.error("--weights must point to an existing local .pt file")
    if not args.input.is_dir():
        parser.error("--input must be a directory")
    devices = [x.strip() for x in args.devices.split(",")]
    if devices != ["cpu"] and (not all(x.isascii() and x.isdigit() for x in devices)
                               or len(set(map(int, devices))) != len(devices)):
        parser.error("--devices requires distinct numeric GPU indices or cpu")
    if args.reference:
        devices = devices[:1]
    if args.source_seconds and (args.reference or args.resume or len(devices) != 1):
        parser.error("--source-seconds requires one device and a fresh fast-mode output")
    paths = sorted((p for p in args.input.glob("*.parquet") if p.is_file()), key=lambda p: p.name)
    output = args.output
    side = Path(str(output) + ".done")
    protected = {p.resolve() for p in [args.weights, *paths]}
    if output.resolve() in protected or side.resolve() in protected or output.resolve() == side.resolve():
        parser.error("output and checkpoint must be distinct from input files and weights")
    output.parent.mkdir(parents=True, exist_ok=True)
    resume = args.resume and output.exists()
    done = set()
    if resume and side.exists():
        # Ignore and remove a torn checkpoint name before appending another.
        with side.open("rb+") as f:
            while True:
                start = f.tell()
                line = f.readline()
                if not line.endswith(b"\n"):
                    f.truncate(start)
                    break
                done.add(line[:-1].decode("utf-8"))
    paths = [p for p in paths if p.name not in done][:args.limit_files]
    if resume and paths:
        clean_resume(output, {p.name for p in paths}, args.limit_videos)
    mode = "a" if resume else "w"
    with side.open(mode, encoding="utf-8") as checkpoint, output.open(mode, encoding="utf-8") as out:
        writer = Writer(out, checkpoint, len(paths))
        if not paths:
            log("No remaining files selected.")
        elif len(devices) == 1:
            run_files(paths, args, devices[0], writer)
        else:
            multi_gpu(paths, args, devices, writer)


if __name__ == "__main__":
    main()
