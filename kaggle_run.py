"""Kaggle commands for checking FineVideo access, validation, and shard processing."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
OUT = WORK / "takehome-results"
RESULTS = OUT / "results.jsonl"
DONE = OUT / "results.jsonl.done"
DATASET = "HuggingFaceFV/finevideo"
FAST = ["--workers", "2", "--batch", "8", "--half", "--stride", "1"]


def filename(index: int) -> str:
    return f"data/train-{index:05d}-of-01357.parquet"


def authenticate(index: int) -> tuple[str, str]:
    """Check token identity and authorize a HEAD request without fetching video data."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    from kaggle_secrets import UserSecretsClient

    try:
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        raise SystemExit("Enable the HF_TOKEN secret for this notebook in Add-ons → Secrets.") from None
    try:
        username = HfApi(token=token).whoami()["name"]
        print(f"Hugging Face account: {username}", flush=True)
        url = hf_hub_url(DATASET, filename(index), repo_type="dataset")
        meta = get_hf_file_metadata(url, token=token)
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code == 401:
            raise SystemExit("401: Check that HF_TOKEN is a valid read token for the intended account.") from None
        if code == 403:
            raise SystemExit(
                "403: Sign in to https://huggingface.co/datasets/HuggingFaceFV/finevideo "
                "with the token owner's account and accept the terms. If access is already "
                "granted, check that the token allows reading this gated dataset. "
                "After updating a token, replace HF_TOKEN in Kaggle Secrets and retry."
            ) from None
        raise
    if not meta.commit_hash:
        raise SystemExit("Hub did not return a dataset revision.")
    print(f"Read access confirmed for {filename(index)} at {meta.commit_hash}.", flush=True)
    return token, meta.commit_hash


def get_shard(index: int) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    token, revision = authenticate(index)
    path = Path(hf_hub_download(
        repo_id=DATASET, repo_type="dataset", filename=filename(index), revision=revision,
        token=token, local_dir=WORK / "finevideo-chunks" / f"{index:05d}",
    ))
    if set(path.parent.glob("*.parquet")) != {path}:
        raise SystemExit(f"Expected a single shard in {path.parent}.")
    return path, revision


def gpu_name() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()


def setup() -> None:
    print(gpu_name(), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                    "ultralytics==8.3.200", "ultralytics-thop==2.0.17", "av==16.0.1"],
                   check=True)
    subprocess.run([sys.executable, str(PROJECT / "prepare.py")], cwd=PROJECT, check=True)


def detect(shard: Path, output: Path, flags: list[str], minutes: int) -> float:
    start = time.monotonic()
    command = [sys.executable, "-u", str(PROJECT / "run.py"), "--input", str(shard.parent),
               "--output", str(output), "--devices", "0", *flags]
    print("Running", *command, flush=True)
    try:
        subprocess.run(command, cwd=PROJECT, check=True, timeout=minutes * 60)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"Inference reached its {minutes}-minute cap. Output: {output}") from None
    return time.monotonic() - start


def sample(count: int, minutes: int) -> None:
    shard, revision = get_shard(0)
    OUT.mkdir(parents=True, exist_ok=True)
    fast = OUT / f"sample-{count}-fast.jsonl"
    ref = OUT / f"sample-{count}-reference.jsonl"
    limit = ["--limit-videos", str(count)]
    fast_s = detect(shard, fast, FAST + limit, minutes)
    ref_s = detect(shard, ref, ["--reference", *limit], minutes)
    comparison = subprocess.run(
        [sys.executable, str(PROJECT / "evaluate.py"), "--ref", str(ref), "--pred", str(fast)],
        capture_output=True, text=True,
    )
    if comparison.returncode not in (0, 1):
        raise SystemExit(comparison.stderr)
    report = {
        "dataset": DATASET, "revision": revision, "shard": shard.name,
        "first_n_videos_requested": count, "gpu": gpu_name(), "fast_flags": FAST,
        "fast_wall_seconds": fast_s, "reference_wall_seconds": ref_s,
        "wall_clock_speedup": ref_s / fast_s, "evaluation": comparison.stdout.strip(),
    }
    summary = OUT / f"summary-{count}.json"
    summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(comparison.stdout, end="")
    print(f"Fast {fast_s:.1f}s, reference {ref_s:.1f}s, speedup {ref_s / fast_s:.2f}x.")
    print(f"Saved {summary}")
    raise SystemExit(comparison.returncode)


def completed() -> set[str]:
    if DONE.exists() != RESULTS.exists():
        raise SystemExit(f"Restore both {RESULTS} and {DONE} to resume this run.")
    return set(DONE.read_text(encoding="utf-8").splitlines()) if DONE.exists() else set()


def next_shard(minutes: int) -> None:
    done = completed()
    index = next((i for i in range(50) if Path(filename(i)).name not in done), None)
    if index is None:
        print("All 50 shards are checkpointed.")
        return
    shard, _ = get_shard(index)
    OUT.mkdir(parents=True, exist_ok=True)
    elapsed = detect(shard, RESULTS, [*FAST, "--resume"], minutes)
    if shard.name not in completed():
        raise SystemExit(f"{shard.name} was not checkpointed. Inspect {RESULTS} for file errors.")
    print(f"{shard.name} finished in {elapsed / 60:.1f} minutes. {len(completed())}/50 checkpointed.")


def positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("access", help="check account and shard access without downloading data")
    commands.add_parser("setup", help="install Kaggle extras and prepare official weights")
    probe = commands.add_parser("sample", help="compare fast and reference inference")
    probe.add_argument("--videos", type=positive, default=1)
    probe.add_argument("--minutes", type=positive, default=15, help="time cap per inference run")
    chunk = commands.add_parser("next", help="process the next unfinished shard among the first 50")
    chunk.add_argument("--minutes", type=positive, default=60, help="time cap for inference")
    commands.add_parser("status", help="show the number of checkpointed shards")
    args = parser.parse_args()
    if args.command == "access":
        authenticate(0)
    elif args.command == "setup":
        setup()
    elif args.command == "sample":
        sample(args.videos, args.minutes)
    elif args.command == "next":
        next_shard(args.minutes)
    else:
        print(f"{len(completed())}/50 shards checkpointed. Output directory: {OUT}")


if __name__ == "__main__":
    main()
