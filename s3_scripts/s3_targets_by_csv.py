#!/usr/bin/env python3
"""Download or delete S3 objects whose s3:// URIs are listed in a CSV column named s3_link."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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


def parent_s3_prefix(uri: str) -> str:
    """Directory-style prefix for an object URI (for counts / messaging)."""
    bucket, key = parse_s3(uri)
    if not key or key.endswith("/"):
        return f"s3://{bucket}/"
    dirpart, _, _ = key.rpartition("/")
    if not dirpart:
        return f"s3://{bucket}/"
    return f"s3://{bucket}/{dirpart}/"


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


def collect_bucket_keys(links: list[str]) -> tuple[list[tuple[str, str]], int]:
    """Return (bucket, key) pairs ready for API calls and count of skipped invalid rows."""
    pairs: list[tuple[str, str]] = []
    errors = 0
    for uri in links:
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
        pairs.append((bucket, key))
    return pairs, errors


def run_download(
    pairs: list[tuple[str, str]],
    dst_dir: Path,
) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3")
    errors = 0
    for bucket, key in tqdm(pairs, desc="Downloading", unit="obj"):
        out_path = dst_dir / key
        out_path.parent.mkdir(parents=True, exist_ok=True)
        uri = f"s3://{bucket}/{key}"
        try:
            client.download_file(bucket, key, str(out_path))
        except Exception as e:
            print(f"failed {uri}: {e}", file=sys.stderr)
            errors += 1
    return errors


def run_delete(pairs: list[tuple[str, str]]) -> int:
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for bucket, key in pairs:
        by_bucket[bucket].append(key)

    client = boto3.client("s3")
    errors = 0
    batch_size = 1000

    for bucket, keys in by_bucket.items():
        for i in tqdm(
            range(0, len(keys), batch_size),
            desc=f"Deleting ({bucket})",
            unit="batch",
        ):
            chunk = keys[i : i + batch_size]
            try:
                resp = client.delete_objects(
                    Bucket=bucket,
                    Delete={
                        "Objects": [{"Key": k} for k in chunk],
                        "Quiet": True,
                    },
                )
            except Exception as e:
                for k in chunk:
                    print(f"failed s3://{bucket}/{k}: {e}", file=sys.stderr)
                errors += len(chunk)
                continue
            for err in resp.get("Errors") or []:
                print(
                    f"failed s3://{bucket}/{err.get('Key')}: "
                    f"{err.get('Code')} {err.get('Message')}",
                    file=sys.stderr,
                )
                errors += 1

    return errors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to CSV with an s3_link column (s3:// URIs).",
    )
    p.add_argument(
        "--operation",
        choices=("download", "delete"),
        default="download",
        help='What to do with each URI: "download" (default) or "delete".',
    )
    p.add_argument(
        "--dst-dir",
        type=Path,
        default=None,
        help="Directory to write objects under when downloading (mirrors S3 key paths).",
    )
    args = p.parse_args()

    csv_path: Path = args.csv

    if args.operation == "download" and args.dst_dir is None:
        raise SystemExit("--dst-dir is required when --operation is download.")

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    links = iter_s3_links(csv_path)
    if not links:
        print(f"No s3_link values for --operation {args.operation!r}.", file=sys.stderr)
        return

    pairs, parse_errors = collect_bucket_keys(links)
    if not pairs:
        print("No valid object URIs to process.", file=sys.stderr)
        if parse_errors:
            raise SystemExit(1)
        return

    if args.operation == "download":
        assert args.dst_dir is not None
        api_errors = run_download(pairs, args.dst_dir)
        if parse_errors or api_errors:
            raise SystemExit(1)
        return

    # delete
    example = f"s3://{pairs[0][0]}/{pairs[0][1]}"
    num_prefixes = len(
        {parent_s3_prefix(f"s3://{b}/{k}") for b, k in pairs}
    )

    print(
        "Are you sure you want to delete the S3 objects listed in this CSV?\n"
        f"  • {len(pairs)} object(s) will be deleted.\n"
        f"  • They span {num_prefixes} distinct S3 path prefix(es) "
        "(unique parent folders under each bucket).\n"
        f"  • Example object: {example}\n"
        "Type YES to proceed with deletion (anything else aborts): ",
        end="",
        flush=True,
    )
    if input().strip() != "YES":
        print("Aborted.", file=sys.stderr)
        raise SystemExit(1)

    api_errors = run_delete(pairs)
    if parse_errors or api_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
