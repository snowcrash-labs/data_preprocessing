#!/usr/bin/env python3
"""List all S3 object keys under a prefix (recursive) and write them to a CSV on S3."""

from __future__ import annotations

import argparse
import csv
import io
from typing import Iterable, Tuple

import boto3


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write all S3 file paths under a prefix to all_file_paths.csv in that prefix."
    )
    parser.add_argument(
        "--s3-prefix",
        required=True,
        dest="s3_prefix",
        help="S3 URI prefix to scan (e.g. s3://my-bucket/path/to/folder/)",
    )
    args = parser.parse_args()

    bucket, key_prefix = parse_s3(args.s3_prefix)
    list_prefix = normalize_list_prefix(key_prefix)

    out_base = args.s3_prefix.rstrip("/")
    out_uri = f"{out_base}/all_file_paths.csv"
    out_bucket, out_key = parse_s3(out_uri)

    s3_uris: list[str] = []
    for key in iter_object_keys(bucket, list_prefix):
        if key.endswith("/"):
            continue
        if list_prefix:
            if not key.startswith(list_prefix):
                continue
            relative = key[len(list_prefix) :]
        else:
            relative = key
        if any_segment_starts_with_dot(relative):
            continue
        if key == out_key:
            continue
        s3_uris.append(f"s3://{bucket}/{key}")

    s3_uris.sort()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["s3_link"])
    for uri in s3_uris:
        writer.writerow([uri])

    body = buf.getvalue().encode("utf-8")
    boto3.client("s3").put_object(
        Bucket=out_bucket,
        Key=out_key,
        Body=body,
        ContentType="text/csv; charset=utf-8",
    )

    print(f"Wrote {len(s3_uris)} paths to {out_uri}")


if __name__ == "__main__":
    main()
