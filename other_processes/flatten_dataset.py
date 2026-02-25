#!/usr/bin/env python3
"""
Flatten a dataset where each depth-1 subdirectory has an integer name and contains vocals.wav.

Moves each vocals.wav up one level and renames it to <parent_dir>.wav, then removes the
now-empty subdirectory. Result: dataset root contains only <integer>.wav files.
"""
import argparse
import sys
from pathlib import Path


def is_integer_dir(name: str) -> bool:
    """Return True if name is a non-empty string of digits."""
    return name.isdigit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten dataset: move each subdir/vocals.wav to <subdir>.wav and remove subdir"
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset root (depth-1 subdirs with integer names, each containing vocals.wav)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Only print what would be done",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each move and rmdir",
    )
    args = parser.parse_args()

    root = args.dataset_path.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    moved = 0
    skipped = 0
    errors = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            if args.verbose:
                print(f"Skip (not dir): {subdir.name}")
            skipped += 1
            continue
        if not is_integer_dir(subdir.name):
            if args.verbose:
                print(f"Skip (non-integer dir): {subdir.name}")
            skipped += 1
            continue

        vocals = subdir / "vocals.wav"
        if not vocals.is_file():
            errors.append(f"Missing or not file: {vocals}")
            continue

        dest = root / f"{subdir.name}.wav"
        if dest.exists():
            errors.append(f"Destination already exists: {dest}")
            continue

        if args.dry_run:
            print(f"Would: mv {vocals} -> {dest}; rmdir {subdir}")
            moved += 1
            continue

        try:
            vocals.rename(dest)
            if args.verbose:
                print(f"mv {vocals} -> {dest}")
            subdir.rmdir()
            if args.verbose:
                print(f"rmdir {subdir}")
            moved += 1
        except OSError as e:
            errors.append(f"{vocals}: {e}")

    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        raise SystemExit(1)

    print(f"Moved {moved} files, skipped {skipped} non-matching entries.")


if __name__ == "__main__":
    main()
