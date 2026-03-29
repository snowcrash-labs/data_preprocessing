#!/usr/bin/env python3
"""
Iterate over subprefixes in s3://rvc-data-for-riaa/voice-model-rvc-pipeline-conversions,
download each acapella.wav, pair it with the corresponding instrumental.wav from
s3://rvc-data-for-riaa/voice-model-rvc-pipeline-stems (same subprefix), apply a random
gain adjustment to the acapella (-6 to +3 dB), additively mix them, peak-normalise
to prevent clipping, and upload the result.

Dependencies:
  pip install boto3 soundfile numpy
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import tempfile
from pathlib import Path

import boto3
import numpy as np
from tqdm import tqdm
import soundfile as sf


BUCKET = "rvc-data-for-riaa"
CONVERSIONS_PREFIX = "voice-model-rvc-pipeline-conversions/"
STEMS_PREFIX = "voice-model-rvc-pipeline-stems/"
DEFAULT_OUTPUT_PREFIX = "voice-model-rvc-pipeline-combined/"

SKIP_PREFIXES = {"failed_models/"}

ACAPELLA_DB_LOW = -6.0
ACAPELLA_DB_HIGH = 3.0

MIN_SEGMENT_DURATION_MS = 3000


def list_subprefixes(s3_client, bucket: str, prefix: str) -> list[str]:
    """Return immediate sub-prefix names (without trailing slash) under *prefix*."""
    paginator = s3_client.get_paginator("list_objects_v2")
    names: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            full = cp["Prefix"]
            relative = full[len(prefix):]
            if relative not in SKIP_PREFIXES:
                names.append(relative.rstrip("/"))
    return names


def download_s3_file(s3_client, bucket: str, key: str, local_path: str) -> None:
    s3_client.download_file(bucket, key, local_path)


def upload_s3_file(s3_client, local_path: str, bucket: str, key: str) -> None:
    s3_client.upload_file(local_path, bucket, key)


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float64")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data, sr


def match_channels(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast to the wider channel count (mono -> stereo by duplication)."""
    if a.shape[1] == b.shape[1]:
        return a, b
    if a.shape[1] == 1:
        a = np.repeat(a, b.shape[1], axis=1)
    elif b.shape[1] == 1:
        b = np.repeat(b, a.shape[1], axis=1)
    return a, b


def match_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the shorter array to match the longer one."""
    diff = a.shape[0] - b.shape[0]
    if diff > 0:
        b = np.pad(b, ((0, diff), (0, 0)))
    elif diff < 0:
        a = np.pad(a, ((0, -diff), (0, 0)))
    return a, b


def peak_normalise(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
    return audio


def has_valid_segments(s3_client, bucket: str, segments_key: str) -> bool:
    """Return True if segments.csv exists and has at least one duration_ms > 3000."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=segments_key)
    except s3_client.exceptions.NoSuchKey:
        return False

    body = resp["Body"].read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(body))
    return any(
        float(row["duration_ms"]) > MIN_SEGMENT_DURATION_MS
        for row in reader
        if "duration_ms" in row
    )


def process_subprefix(
    s3_client,
    bucket: str,
    subprefix: str,
    output_prefix: str,
    rng: random.Random,
    output_filename: str,
) -> bool:
    """Returns True if the mix was produced, False if skipped."""
    segments_key = f"{STEMS_PREFIX}{subprefix}/segments.csv"
    if not has_valid_segments(s3_client, bucket, segments_key):
        return False

    acapella_key = f"{CONVERSIONS_PREFIX}{subprefix}/acapella.wav"
    instrumental_key = f"{STEMS_PREFIX}{subprefix}/instrumental.wav"

    with tempfile.TemporaryDirectory() as tmp:
        acapella_local = str(Path(tmp) / "acapella.wav")
        instrumental_local = str(Path(tmp) / "instrumental.wav")
        output_local = str(Path(tmp) / output_filename)

        download_s3_file(s3_client, bucket, acapella_key, acapella_local)
        download_s3_file(s3_client, bucket, instrumental_key, instrumental_local)

        acapella, sr_a = read_wav(acapella_local)
        instrumental, sr_i = read_wav(instrumental_local)

        if sr_a != sr_i:
            raise ValueError(
                f"Sample-rate mismatch for {subprefix}: "
                f"acapella={sr_a}, instrumental={sr_i}. "
                "Resample one of the files before running this script."
            )

        acapella, instrumental = match_channels(acapella, instrumental)
        acapella, instrumental = match_lengths(acapella, instrumental)

        gain_db = rng.uniform(ACAPELLA_DB_LOW, ACAPELLA_DB_HIGH)
        acapella *= db_to_linear(gain_db)

        combined = acapella + instrumental
        combined = peak_normalise(combined)

        output_key = f"{output_prefix}{subprefix}/{output_filename}"
        sf.write(output_local, combined, sr_i, subtype="PCM_16")
        upload_s3_file(s3_client, output_local, bucket, output_key)

    print(f"  [{subprefix}] gain={gain_db:+.2f} dB -> s3://{bucket}/{output_key}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Mix RVC-converted acapellas with their instrumental stems. "
            "Applies a random gain to the acapella and peak-normalises the mix."
        )
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"S3 key prefix for output files (default: {DEFAULT_OUTPUT_PREFIX}).",
    )
    p.add_argument(
        "--output-filename",
        type=str,
        default="combined.wav",
        help="Filename for the mixed output (default: combined.wav).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible gain adjustments.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N subprefixes (useful for testing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List subprefixes without processing.",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    s3 = boto3.client("s3")

    print(f"Listing subprefixes under s3://{BUCKET}/{CONVERSIONS_PREFIX} ...")
    subprefixes = list_subprefixes(s3, BUCKET, CONVERSIONS_PREFIX)
    print(f"Found {len(subprefixes)} subprefixes.")

    if args.limit is not None:
        subprefixes = subprefixes[: args.limit]
        print(f"Limited to first {args.limit}.")

    if args.dry_run:
        for sp in subprefixes:
            print(f"  {sp}")
        return

    output_prefix = args.output_prefix
    if not output_prefix.endswith("/"):
        output_prefix += "/"

    skipped = 0
    for sp in tqdm(subprefixes, desc="Mixing"):
        try:
            produced = process_subprefix(s3, BUCKET, sp, output_prefix, rng, args.output_filename)
            if not produced:
                skipped += 1
        except Exception as e:
            tqdm.write(f"  ERROR ({sp}): {e}")

    print(f"\nDone. Processed {len(subprefixes) - skipped}, skipped {skipped}.")


if __name__ == "__main__":
    main()
