"""Compare reference and predicted per-second JSONL records, independent of order."""

import argparse
import json
from pathlib import Path


BITS = {name: 1 << i for i, name in enumerate(("person", "car", "dog", "laptop", "cell phone"))}
Key = tuple[str, int]
Value = tuple[int, int]


def read_results(path: Path) -> tuple[dict[Key, Value], int]:
    records: dict[Key, Value] = {}
    errors = 0
    with path.open(encoding="utf-8") as lines:
        for line_no, line in enumerate(lines, 1):
            try:
                row = json.loads(line)
                if not isinstance(row["video"], str):
                    raise ValueError("invalid video identifier")
                if "error" in row:
                    errors += 1
                    continue
                second, people = row["second"], row["people"]
                if type(second) is not int or second < 0 or type(people) is not int or people < 0:
                    raise ValueError("second and people must be nonnegative integers")
                key = (row["video"], second)
                if key in records:
                    raise ValueError(f"duplicate second {key}")
                if not isinstance(row["objects"], list):
                    raise ValueError("objects must be a list")
                mask = 0
                for name in row["objects"]:
                    mask |= BITS[name]
                records[key] = mask, people
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return records, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, type=Path, help="reference JSONL")
    parser.add_argument("--pred", required=True, type=Path, help="candidate JSONL")
    args = parser.parse_args()
    try:
        ref, ref_errors = read_results(args.ref)
        pred, pred_errors = read_results(args.pred)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    tp = fp = fn = exact = within = shared = 0
    total_ref = len(ref)
    for key, (actual, people) in pred.items():
        truth = ref.pop(key, None)
        expected = 0
        if truth is not None:
            shared += 1
            expected, target_people = truth
            exact += people == target_people
            within += abs(people - target_people) <= 1
        tp += (actual & expected).bit_count()
        fp += (actual & ~expected).bit_count()
        fn += (expected & ~actual).bit_count()
    fn += sum(mask.bit_count() for mask, _ in ref.values())
    seconds = len(pred) + len(ref)
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 1.0
    exact_fraction = exact / seconds if seconds else 0.0
    within_fraction = within / seconds if seconds else 0.0
    passed = f1 >= 0.95 and exact_fraction >= 0.85 and within_fraction >= 0.95
    print(f"Seconds: {seconds} union, {shared} shared, {total_ref - shared} reference-only, "
          f"{len(pred) - shared} prediction-only")
    print(f"Object-presence micro F1: {f1:.6f} (TP={tp}, FP={fp}, FN={fn})")
    print(f"People exact: {100 * exact_fraction:.2f}% ({exact}/{seconds})")
    print(f"People within 1: {100 * within_fraction:.2f}% ({within}/{seconds})")
    print(f"Error lines: reference={ref_errors}, prediction={pred_errors}")
    print("PASS" if passed else "FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
