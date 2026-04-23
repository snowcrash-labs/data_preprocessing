#!/usr/bin/env python3
"""Download S3 objects whose s3:// URIs are listed in a CSV column named s3_link."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple

import boto3
from tqdm import tqdm


def parse_s3(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri!r}")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket:
        raise ValueError(f"Invalid s3:// URI (missing bucket): {uri!r}")
    return bucket, key


def iter_s3_links(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "s3_link" not in reader.fieldnames:
            raise SystemExit(
                'CSV must have a header row that includes the column "s3_link".'
            )
        seen: dict[str, None] = {}
        for row in reader:
            raw = (row.get("s3_link") or "").strip()
            if not raw or raw in seen:
                continue
            seen[raw] = None
        return list(seen.keys())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to CSV with an s3_link column (s3:// URIs).",
    )
    p.add_argument(
        "--dst-dir",
        type=Path,
        required=True,
        help="Directory to write objects under (mirrors S3 key paths).",
    )
    args = p.parse_args()

    csv_path: Path = args.csv
    dst_dir: Path = args.dst_dir

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    links = iter_s3_links(csv_path)
    if not links:
        print("No s3_link values to download.", file=sys.stderr)
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3")
    errors = 0

    for uri in tqdm(links, desc="Downloading", unit="obj"):
        try:
            bucket, key = parse_s3(uri)
        except ValueError as e:
            print(f"skip (bad URI): {e}", file=sys.stderr)
            errors += 1
            continue
        if not key or key.endswith("/"):
            print(f"skip (not an object key): {uri!r}", file=sys.stderr)
            errors += 1
            continue
        out_path = dst_dir / key
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.download_file(bucket, key, str(out_path))
        except Exception as e:
            print(f"failed {uri}: {e}", file=sys.stderr)
            errors += 1

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
