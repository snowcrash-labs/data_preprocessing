#!/usr/bin/env python3
"""List S3 object keys under a prefix (optionally limited by depth and filename substring) and write the CSV to S3 or locally."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Iterable, Iterator, Tuple

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


def normalize_list_prefix(key_prefix: str) -> str:
    """Use a trailing slash so listing does not match unrelated keys (e.g. data vs database)."""
    if not key_prefix:
        return ""
    return key_prefix if key_prefix.endswith("/") else key_prefix + "/"


def any_segment_starts_with_dot(relative_path: str) -> bool:
    """True if any path component or the basename is a dot-file / dot-directory."""
    if not relative_path:
        return False
    for part in relative_path.split("/"):
        if not part:
            continue
        if part.startswith("."):
            return True
    return False


def iter_object_keys(bucket: str, list_prefix: str) -> Iterable[str]:
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket}
    if list_prefix:
        kwargs["Prefix"] = list_prefix
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents") or []:
            yield obj["Key"]


def relative_depth_slashes(relative: str) -> int:
    """Number of '/' in the path under the list prefix (0 = object directly under prefix)."""
    return relative.count("/")


def passes_depth_filter(relative: str, depth: int | None) -> bool:
    """depth 1: only keys directly under --s3-prefix. depth N: allow up to N-1 subprefix levels."""
    if depth is None or depth < 1:
        return True
    return relative_depth_slashes(relative) <= depth - 1


def key_to_listing_uri(
    key: str,
    *,
    bucket: str,
    list_prefix: str,
    exclude_s3_output: Tuple[str, str] | None,
    depth: int | None,
    target_file: str,
) -> str | None:
    """If this key should appear in the CSV, return its s3:// URI; otherwise None."""
    if key.endswith("/"):
        return None
    if list_prefix:
        if not key.startswith(list_prefix):
            return None
        relative = key[len(list_prefix) :]
    else:
        relative = key
    if any_segment_starts_with_dot(relative):
        return None
    if exclude_s3_output is not None:
        ex_bucket, ex_key = exclude_s3_output
        if bucket == ex_bucket and key == ex_key:
            return None
    if not passes_depth_filter(relative, depth):
        return None
    uri = f"s3://{bucket}/{key}"
    if target_file and target_file not in uri:
        return None
    return uri


def count_object_keys(bucket: str, list_prefix: str) -> int:
    return sum(1 for _ in iter_object_keys(bucket, list_prefix))


def iter_object_keys_with_progress(
    bucket: str,
    list_prefix: str,
    *,
    show_progress: bool,
) -> Iterator[str]:
    if not show_progress:
        yield from iter_object_keys(bucket, list_prefix)
        return
    total = count_object_keys(bucket, list_prefix)
    yield from tqdm(
        iter_object_keys(bucket, list_prefix),
        total=total,
        unit="obj",
        unit_scale=False,
        desc="Listing S3",
    )


def resolve_s3_destination(destination: str) -> Tuple[str, str]:
    """Bucket and object key for the CSV. Trailing slash means a prefix; else destination is the full key."""
    bucket, key = parse_s3(destination)
    key = key.rstrip("/")
    if not key:
        return bucket, "all_file_paths.csv"
    if destination.rstrip().endswith("/"):
        prefix = key + "/"
        return bucket, f"{prefix}all_file_paths.csv"
    return bucket, key


def resolve_local_destination(destination: str) -> Path:
    """Path to the CSV file. Trailing sep, '.', './', existing dir, or non-existent path ending in / → directory."""
    p = Path(destination).expanduser()
    is_dir_hint = (
        destination.endswith(("/", "\\"))
        or destination in (".", "./")
        or p.is_dir()
    )
    if is_dir_hint:
        return (p / "all_file_paths.csv").resolve()
    return p.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write S3 file paths under --s3-prefix to a CSV (--destination: local path or s3:// URI)."
    )
    parser.add_argument(
        "--s3-prefix",
        required=True,
        dest="s3_prefix",
        help="S3 URI prefix to scan (e.g. s3://my-bucket/path/to/folder/)",
    )
    parser.add_argument(
        "--target-file",
        default="",
        help="Substring that must appear in each listed s3:// URI. Empty: no substring filter.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "How many prefix levels to include: 1 = only objects in --s3-prefix; "
            "2 = also one level of subprefixes; etc. Omit for unlimited (fully recursive)."
        ),
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the progress bar (avoids a preliminary full prefix listing to get totals).",
    )
    parser.add_argument(
        "--destination",
        default="./",
        help=(
            "Where to write the CSV. If it starts with s3://, upload there "
            "(trailing / means a prefix; all_file_paths.csv is used under that prefix). "
            "Otherwise a local path (default ./ → ./all_file_paths.csv)."
        ),
    )
    args = parser.parse_args()

    bucket, key_prefix = parse_s3(args.s3_prefix)
    list_prefix = normalize_list_prefix(key_prefix)

    dest = args.destination
    if dest.startswith("s3://"):
        out_bucket, out_key = resolve_s3_destination(dest)
        exclude_s3_output: Tuple[str, str] | None = (out_bucket, out_key)
        out_display = f"s3://{out_bucket}/{out_key}"
    else:
        out_path = resolve_local_destination(dest)
        exclude_s3_output = None
        out_display = str(out_path)

    s3_uris: list[str] = []
    for key in iter_object_keys_with_progress(
        bucket,
        list_prefix,
        show_progress=not args.no_progress_bar,
    ):
        uri = key_to_listing_uri(
            key,
            bucket=bucket,
            list_prefix=list_prefix,
            exclude_s3_output=exclude_s3_output,
            depth=args.depth,
            target_file=args.target_file,
        )
        if uri is not None:
            s3_uris.append(uri)

    s3_uris.sort()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["s3_link"])
    for uri in s3_uris:
        writer.writerow([uri])

    body = buf.getvalue().encode("utf-8")
    if dest.startswith("s3://"):
        boto3.client("s3").put_object(
            Bucket=out_bucket,
            Key=out_key,
            Body=body,
            ContentType="text/csv; charset=utf-8",
        )
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(body)

    print(f"Wrote {len(s3_uris)} paths to {out_display}")


if __name__ == "__main__":
    main()
