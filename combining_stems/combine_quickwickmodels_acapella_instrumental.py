#!/usr/bin/env python3
"""
Iterate over audio files under CONVERSIONS_PREFIX (RVC / acapella side).

For each file, extract the MD5 hex substring after ``source_`` in the basename, then load
``stem_bass.wav``, ``stem_drums.wav``, and ``stem_other.wav`` from
STEMS_PREFIX/<md5>/ and sum them (additive instrumental). Treat the conversion file as
vocals, apply a random gain to the vocal (-6 to +3 dB), additively mix with the
instrumental, peak-normalise, and upload.

Dependencies:
  pip install boto3 soundfile numpy
"""

from __future__ import annotations

import argparse
import random
import re
import tempfile
from pathlib import Path

import boto3
import librosa
from tqdm import tqdm
import numpy as np
import soundfile as sf


CONVERSIONS_PREFIX = "s3://rvc-data-for-riaa/rvc-acapella/source-human-demucs-stems/QuickWick-models/"
STEMS_PREFIX = "s3://ai-detection-training-data/audio/human/human_demucs_stems/"
DEFAULT_OUTPUT_PREFIX = "s3://rvc-data-for-riaa/quickwickmodels-mixes/"

STEM_FILENAMES = ("stem_bass.wav", "stem_drums.wav", "stem_other.wav")

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".aiff", ".aif")

ACAPELLA_DB_LOW = -6.0
ACAPELLA_DB_HIGH = 3.0

# MD5 = 32 hex digits after "source_"
_SOURCE_MD5_RE = re.compile(r"source_([a-fA-F0-9]{32})")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key_prefix) with trailing slash on prefix when non-empty."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri}")
    rest = uri[5:]
    slash = rest.index("/")
    bucket = rest[:slash]
    prefix = rest[slash + 1 :]
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def list_audio_object_keys(s3_client, bucket: str, prefix: str) -> list[str]:
    """All object keys under *prefix* whose basename looks like an audio file."""
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            base = k.rsplit("/", 1)[-1].lower()
            if base.endswith(AUDIO_SUFFIXES):
                keys.append(k)
    return sorted(keys)


def md5_after_source(filename: str) -> str | None:
    m = _SOURCE_MD5_RE.search(filename)
    return m.group(1).lower() if m else None


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
    if a.shape[1] == b.shape[1]:
        return a, b
    if a.shape[1] == 1:
        a = np.repeat(a, b.shape[1], axis=1)
    elif b.shape[1] == 1:
        b = np.repeat(b, a.shape[1], axis=1)
    return a, b


def match_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = a.shape[0] - b.shape[0]
    if diff > 0:
        b = np.pad(b, ((0, diff), (0, 0)))
    elif diff < 0:
        a = np.pad(a, ((0, -diff), (0, 0)))
    return a, b


def sum_stems(stems: list[np.ndarray]) -> np.ndarray:
    """Align channel count (mono -> max channels) and length; return elementwise sum."""
    if not stems:
        raise ValueError("empty stem list")
    max_ch = max(s.shape[1] for s in stems)
    max_len = max(s.shape[0] for s in stems)
    out = np.zeros((max_len, max_ch), dtype=np.float64)
    for s in stems:
        x = np.asarray(s, dtype=np.float64)
        if x.shape[1] == 1 and max_ch > 1:
            x = np.repeat(x, max_ch, axis=1)
        if x.shape[0] < max_len:
            x = np.pad(x, ((0, max_len - x.shape[0]), (0, 0)))
        elif x.shape[0] > max_len:
            x = x[:max_len, :]
        if x.shape[1] < max_ch:
            x = np.pad(x, ((0, 0), (0, max_ch - x.shape[1])))
        out += x
    return out


def peak_normalise(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
    return audio


def process_conversion_object(
    s3_client,
    conv_bucket: str,
    conv_key: str,
    stems_bucket: str,
    stems_key_prefix: str,
    output_bucket: str,
    output_prefix: str,
    rng: random.Random,
    output_filename: str,
) -> None:
    basename = conv_key.rsplit("/", 1)[-1]
    md5_part = md5_after_source(basename)
    if not md5_part:
        raise ValueError(f"no source_<32hex_md5> in filename: {basename}")

    stem_dir = f"{stems_key_prefix}{md5_part}/"
    stem_keys = [f"{stem_dir}{name}" for name in STEM_FILENAMES]

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        vocal_local = str(t / ("vocal" + Path(basename).suffix))
        stem_locals = [str(t / name) for name in STEM_FILENAMES]
        output_local = str(t / output_filename)

        download_s3_file(s3_client, conv_bucket, conv_key, vocal_local)
        for sk, slocal in zip(stem_keys, stem_locals, strict=True):
            download_s3_file(s3_client, stems_bucket, sk, slocal)

        stem_reads = [read_wav(p) for p in stem_locals]
        stems_audio = [a for a, _ in stem_reads]
        stem_srs = [sr for _, sr in stem_reads]

        if len(set(stem_srs)) != 1:
            raise ValueError(
                f"[{md5_part}] stem sample rates differ: {dict(zip(STEM_FILENAMES, stem_srs, strict=True))}"
            )
        sr_stem = stem_srs[0]

        vocal, sr_v = read_wav(vocal_local)
        if sr_v != sr_stem:
            vocal_channels = [
                librosa.resample(vocal[:, ch], orig_sr=sr_v, target_sr=sr_stem)
                for ch in range(vocal.shape[1])
            ]
            vocal = np.stack(vocal_channels, axis=1)

        instrumental = sum_stems(stems_audio)
        vocal, instrumental = match_channels(vocal, instrumental)
        vocal, instrumental = match_lengths(vocal, instrumental)

        gain_db = rng.uniform(ACAPELLA_DB_LOW, ACAPELLA_DB_HIGH)
        vocal = vocal * db_to_linear(gain_db)

        combined = vocal + instrumental
        combined = peak_normalise(combined)

        output_key = f"{output_prefix}{md5_part}/{output_filename}"
        sf.write(output_local, combined, sr_stem, subtype="PCM_16")
        upload_s3_file(s3_client, output_local, output_bucket, output_key)

    print(
        f"  [{basename}] md5={md5_part} gain={gain_db:+.2f} dB -> s3://{output_bucket}/{output_key}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Mix conversion-side vocals with Demucs bass+drums+other stems (by MD5 under "
            "source_* in the filename). Random vocal gain; peak-normalised mix."
        )
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"S3 URI prefix for output files (default: {DEFAULT_OUTPUT_PREFIX}).",
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
        help="Process only the first N conversion files (after sorting keys).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List conversion keys and resolved stem prefixes without processing.",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    s3 = boto3.client("s3")

    conv_bucket, conv_prefix = parse_s3_uri(CONVERSIONS_PREFIX)
    stems_bucket, stems_prefix = parse_s3_uri(STEMS_PREFIX)
    out_bucket, out_prefix = parse_s3_uri(args.output_prefix)

    print(f"Listing audio objects under s3://{conv_bucket}/{conv_prefix} ...")
    conv_keys = list_audio_object_keys(s3, conv_bucket, conv_prefix)
    print(f"Found {len(conv_keys)} audio object(s).")

    if args.limit is not None:
        conv_keys = conv_keys[: args.limit]
        print(f"Limited to first {args.limit}.")

    if args.dry_run:
        for ck in conv_keys:
            bn = ck.rsplit("/", 1)[-1]
            h = md5_after_source(bn)
            if h:
                print(f"  {ck}")
                print(f"    -> stems: s3://{stems_bucket}/{stems_prefix}{h}/{{stem_bass,drums,other}}.wav")
            else:
                print(f"  {ck}  (skip: no source_<md5> in name)")
        return

    for ck in tqdm(conv_keys, desc="Mixing"):
        bn = ck.rsplit("/", 1)[-1]
        if not md5_after_source(bn):
            tqdm.write(f"  SKIP (no source_<32hex>): {bn}")
            continue
        try:
            process_conversion_object(
                s3,
                conv_bucket,
                ck,
                stems_bucket,
                stems_prefix,
                out_bucket,
                out_prefix,
                rng,
                args.output_filename,
            )
        except Exception as e:
            tqdm.write(f"  ERROR ({bn}): {e}")


if __name__ == "__main__":
    main()
