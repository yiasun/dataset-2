import argparse
from pathlib import Path


def checkpoint_sort_key(path):
    stem = path.stem
    number = -1
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            number = int(tail)
    return (path.stat().st_mtime, number, path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--prefer", default=None, help="Use this checkpoint name first if it exists.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if args.prefer:
        preferred = work_dir / args.prefer
        if preferred.exists():
            print(preferred)
            return

    candidates = []
    for pattern in ("epoch_*.pth", "iter_*.pth", "best*.pth", "latest.pth"):
        candidates.extend(work_dir.glob(pattern))

    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in {work_dir}")

    latest = sorted(candidates, key=checkpoint_sort_key)[-1]
    print(latest)


if __name__ == "__main__":
    main()
