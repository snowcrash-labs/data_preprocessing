#!/usr/bin/env python3
"""Download files under an S3 prefix listed in all_file_paths.csv, with flat or tree layout options."""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import boto3
from botocore.exceptions import ClientError

from create_s3_link_csv import normalize_list_prefix, parse_s3

S3_CLIENT = boto3.client("s3")
ALL_FILES_CSV = "all_file_paths.csv"


def all_file_paths_csv_uri(s3_prefix: str) -> str:
    base = s3_prefix.rstrip("/")
    return f"{base}/{ALL_FILES_CSV}"


def s3_object_exists(bucket: str, key: str) -> bool:
    try:
        S3_CLIENT.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def ensure_all_file_paths_csv(s3_prefix: str) -> Tuple[str, str]:
    """Return (bucket, key) for all_file_paths.csv, creating it via create_s3_link_csv.py if missing."""
    uri = all_file_paths_csv_uri(s3_prefix)
    bucket, key = parse_s3(uri)
    if not s3_object_exists(bucket, key):
        script = Path(__file__).resolve().parent / "create_s3_link_csv.py"
        subprocess.run(
            [sys.executable, str(script), "--s3-prefix", s3_prefix],
            check=True,
        )
    if not s3_object_exists(bucket, key):
        raise FileNotFoundError(
            f"Expected {uri} after running create_s3_link_csv.py, but it is still missing."
        )
    return bucket, key


def read_s3_csv_s3_uris(bucket: str, key: str) -> List[str]:
    obj = S3_CLIENT.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "s3_uri" not in reader.fieldnames:
        raise ValueError(f"{ALL_FILES_CSV} must have an 's3_uri' column.")
    out: List[str] = []
    for row in reader:
        u = (row.get("s3_uri") or "").strip()
        if u:
            out.append(u)
    return out


def relative_key_under_prefix(s3_prefix: str, object_uri: str) -> str:
    prefix_bucket, prefix_key = parse_s3(s3_prefix)
    obj_bucket, obj_key = parse_s3(object_uri)
    if obj_bucket != prefix_bucket:
        raise ValueError(
            f"Object bucket {obj_bucket!r} does not match prefix bucket {prefix_bucket!r}: {object_uri}"
        )
    list_prefix = normalize_list_prefix(prefix_key)
    if list_prefix:
        if not obj_key.startswith(list_prefix):
            raise ValueError(f"Object key is not under the given prefix: {object_uri}")
        return obj_key[len(list_prefix) :]
    return obj_key


def output_filename_for_key(
    relative_key: str, filter_name: str, name_as_parent_at_depth: int
) -> str:
    """Flat layout only: relative_key under the listing prefix; depth >= 1 renames using a parent folder."""
    parts = [p for p in relative_key.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Need at least one directory above the file in path {relative_key!r} "
            f"when name_as_parent_at_depth is {name_as_parent_at_depth} (not 0)."
        )
    filename = parts[-1]
    if filename != filter_name:
        raise ValueError(f"Internal error: expected leaf {filename!r} to equal filter {filter_name!r}")
    dir_parts = parts[:-1]
    if len(dir_parts) < name_as_parent_at_depth:
        raise ValueError(
            f"Path {relative_key!r} has only {len(dir_parts)} parent folder(s); "
            f"need at least {name_as_parent_at_depth} for name_as_parent_at_depth."
        )
    name_stem = dir_parts[-name_as_parent_at_depth]
    ext = os.path.splitext(filename)[1]
    return f"{name_stem}{ext}"


def disambiguate(filename: str, used: dict[str, int]) -> str:
    if filename not in used:
        used[filename] = 0
        return filename
    used[filename] += 1
    n = used[filename]
    stem, suff = os.path.splitext(filename)
    return f"{stem}_{n}{suff}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read all_file_paths.csv under an S3 prefix (create it if missing), "
            "select objects whose basename matches the filter, and download them under dest_path "
            "using either original names, parent-folder renaming, or a mirrored subdirectory tree."
        )
    )
    parser.add_argument(
        "--s3-prefix",
        required=True,
        dest="s3_prefix",
        help="S3 URI prefix (same as create_s3_link_csv.py).",
    )
    parser.add_argument(
        "--filter",
        required=True,
        dest="filter",
        help='Leaf filename to match (e.g. "voice.mp3", "acapella.wav").',
    )
    parser.add_argument(
        "--dest-path",
        required=True,
        dest="dest_path",
        help="Local directory to write downloaded files.",
    )
    parser.add_argument(
        "--name-as-parent-at-depth",
        default=0,
        type=int,
        dest="name_as_parent_at_depth",
        help=(
            "0 = keep the object's original filename. "
            "1 = flat layout: name file after immediate parent folder (e.g. .../rty/voice.mp3 -> rty.mp3). "
            "2 = flat layout: grandparent (e.g. .../zxc/rty/voice.mp3 -> zxc.mp3). "
            "Must be 0 when --keep-dir-tree is set."
        ),
    )
    parser.add_argument(
        "--keep-dir-tree",
        action="store_true",
        default=False,
        dest="keep_dir_tree",
        help=(
            "Mirror the key path under s3_prefix below dest_path (requires "
            "--name-as-parent-at-depth 0). Example: .../zxc/rty/fgh/v.wav under prefix .../asd/ "
            "-> DEST/zxc/rty/fgh/v.wav."
        ),
    )
    args = parser.parse_args()

    if args.name_as_parent_at_depth < 0:
        parser.error("--name-as-parent-at-depth must be >= 0")
    if args.keep_dir_tree and args.name_as_parent_at_depth != 0:
        parser.error(
            "Cannot combine --keep-dir-tree with a non-zero --name-as-parent-at-depth. "
            "Directory mirroring already determines full paths; use --name-as-parent-at-depth 0, "
            "or omit --keep-dir-tree and use a positive depth for flat renaming."
        )

    csv_bucket, csv_key = ensure_all_file_paths_csv(args.s3_prefix)
    all_uris = read_s3_csv_s3_uris(csv_bucket, csv_key)

    matches: List[str] = []
    for uri in all_uris:
        base = os.path.basename(parse_s3(uri)[1])
        if base == args.filter:
            matches.append(uri)

    if not matches:
        print(f"No objects with basename {args.filter!r} found under {args.s3_prefix!r}.", file=sys.stderr)
        sys.exit(1)

    dest_dir = Path(args.dest_path).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, int] = {}
    for uri in matches:
        rel = relative_key_under_prefix(args.s3_prefix, uri)
        rel_norm = rel.replace("\\", "/")

        if args.keep_dir_tree:
            parts = [p for p in rel_norm.split("/") if p]
            if not parts or parts[-1] != args.filter:
                raise ValueError(f"Internal error: relative key leaf does not match filter: {rel!r}")
            local_path = dest_dir.joinpath(*parts)
            local_path.parent.mkdir(parents=True, exist_ok=True)
        elif args.name_as_parent_at_depth == 0:
            base_name = os.path.basename(rel_norm)
            if base_name != args.filter:
                raise ValueError(f"Internal error: expected basename {base_name!r} to equal filter")
            local_name = disambiguate(base_name, used_names)
            local_path = dest_dir / local_name
        else:
            out_base = output_filename_for_key(rel_norm, args.filter, args.name_as_parent_at_depth)
            local_name = disambiguate(out_base, used_names)
            local_path = dest_dir / local_name

        obj_bucket, obj_key = parse_s3(uri)
        print(f"Downloading s3://{obj_bucket}/{obj_key} -> {local_path}")
        S3_CLIENT.download_file(obj_bucket, obj_key, str(local_path))

    print(f"Done. Wrote {len(matches)} file(s) to {dest_dir}")


if __name__ == "__main__":
    main()
