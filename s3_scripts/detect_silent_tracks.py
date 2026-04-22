#!/usr/bin/env python3
"""
Scan S3 prefixes for audio files and flag tracks that are "mostly silent".

A track is flagged when its silence fraction (proportion of total duration
that is silent) meets or exceeds --silence-fraction (default 0.80), meaning
less than 20% of the track contains audible content.

Input CSV must have an 's3_prefix' column with one S3 URI prefix per row.
Output CSV contains one row per audio file with silence metrics and a flag.

Usage example:
    python detect_silent_tracks.py \
        --csv prefixes.csv \
        --output-csv results.csv \
        --silence-thresh -40 \
        --silence-fraction 0.80 \
        --workers 64
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Allow running from any working directory by ensuring the sibling module is importable.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from create_s3_link_csv import iter_object_keys, normalize_list_prefix, parse_s3  # noqa: E402

FIELDNAMES = [
    "s3_uri",
    "prefix",
    "total_duration_ms",
    "nonsilent_duration_ms",
    "silence_fraction",
    "is_flagged",
    "error",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Optional[str]) -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def list_audio_files_for_prefix(
    s3_prefix: str,
    extensions: Tuple[str, ...],
) -> List[str]:
    """Return sorted S3 URIs under s3_prefix whose extension is in extensions."""
    bucket, key_prefix = parse_s3(s3_prefix)
    list_prefix = normalize_list_prefix(key_prefix)
    uris: List[str] = []
    for key in iter_object_keys(bucket, list_prefix):
        if key.endswith("/"):
            continue
        if Path(key).suffix.lower() in extensions:
            uris.append(f"s3://{bucket}/{key}")
    uris.sort()
    return uris


def collect_all_tasks(
    prefixes: List[str],
    extensions: Tuple[str, ...],
    logger: logging.Logger,
) -> List[Tuple[str, str]]:
    """List audio files under every prefix and return a flat (s3_uri, prefix) list."""
    tasks: List[Tuple[str, str]] = []
    for prefix in prefixes:
        logger.info("Listing %s ...", prefix)
        try:
            uris = list_audio_files_for_prefix(prefix, extensions)
        except Exception as exc:
            logger.error("Failed to list prefix %s: %s", prefix, exc)
            continue
        if not uris:
            logger.warning("No matching audio files found under prefix: %s", prefix)
            continue
        logger.info("  Found %d file(s) under %s", len(uris), prefix)
        for uri in uris:
            tasks.append((uri, prefix))
    return tasks


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_audio(local_path: Path, ext: str) -> AudioSegment:
    if ext == ".wav":
        return AudioSegment.from_wav(str(local_path))
    elif ext == ".mp3":
        return AudioSegment.from_mp3(str(local_path))
    elif ext == ".flac":
        return AudioSegment.from_file(str(local_path), format="flac")
    else:
        return AudioSegment.from_file(str(local_path))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def analyze_audio_file(
    s3_uri: str,
    prefix: str,
    silence_thresh: int,
    min_silence_len: int,
    silence_fraction_threshold: float,
    file_extensions: Tuple[str, ...],
) -> Dict[str, Any]:
    """Download one S3 audio file, measure its silence fraction, return a result dict.

    Always returns — never raises. Exceptions are captured in result["error"].
    A boto3 client is created per call to avoid multiprocessing-unsafe shared state.
    """
    result: Dict[str, Any] = {
        "s3_uri": s3_uri,
        "prefix": prefix,
        "total_duration_ms": None,
        "nonsilent_duration_ms": None,
        "silence_fraction": None,
        "is_flagged": None,
        "error": None,
    }
    try:
        bucket, key = parse_s3(s3_uri)
        ext = Path(key).suffix.lower()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / Path(key).name
            boto3.client("s3").download_file(bucket, key, str(local_path))

            audio = _load_audio(local_path, ext)
            total_duration_ms = len(audio)

            if total_duration_ms == 0:
                result["total_duration_ms"] = 0
                result["nonsilent_duration_ms"] = 0
                result["silence_fraction"] = 1.0
                result["is_flagged"] = True
                result["error"] = "zero_length_audio"
                return result

            nonsilent_intervals = detect_nonsilent(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
            )
            nonsilent_duration_ms = sum(end - start for start, end in nonsilent_intervals)
            silence_fraction = 1.0 - (nonsilent_duration_ms / total_duration_ms)
            is_flagged = silence_fraction >= silence_fraction_threshold

            result["total_duration_ms"] = total_duration_ms
            result["nonsilent_duration_ms"] = nonsilent_duration_ms
            result["silence_fraction"] = round(silence_fraction, 6)
            result["is_flagged"] = is_flagged

    except Exception as exc:
        logging.getLogger(__name__).error("Error processing %s: %s", s3_uri, exc)
        result["error"] = repr(exc)

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Scan S3 prefixes for audio files and flag tracks that are mostly silent. "
            "A track is flagged when its silence fraction >= --silence-fraction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        required=True,
        dest="csv",
        help="Path to input CSV with an 's3_prefix' column.",
    )
    p.add_argument(
        "--silence-thresh",
        type=int,
        default=-40,
        dest="silence_thresh",
        help="dBFS threshold; audio quieter than this is considered silent.",
    )
    p.add_argument(
        "--min-silence-len",
        type=int,
        default=500,
        dest="min_silence_len",
        help="Minimum continuous silence run in ms required by detect_nonsilent().",
    )
    p.add_argument(
        "--silence-fraction",
        type=float,
        default=0.80,
        dest="silence_fraction",
        help=(
            "Flag track if silence_fraction >= this value. "
            "0.80 = flag when 80%% or more of the track is silent (< 20%% audible content)."
        ),
    )
    p.add_argument(
        "--output-csv",
        default="silent_tracks_report.csv",
        dest="output_csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        dest="workers",
        help="Number of parallel worker processes.",
    )
    p.add_argument(
        "--file-extensions",
        nargs="+",
        default=[".wav"],
        dest="file_extensions",
        metavar="EXT",
        help="Audio file extensions to process (e.g. .wav .mp3 .flac).",
    )
    p.add_argument(
        "--log-file",
        default=None,
        dest="log_file",
        help="Optional path for a log file (written in addition to stderr).",
    )
    return p


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not Path(args.csv).is_file():
        parser.error(f"Input CSV not found: {args.csv}")
    if not (0.0 < args.silence_fraction <= 1.0):
        parser.error("--silence-fraction must be in the range (0.0, 1.0]")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.min_silence_len < 1:
        parser.error("--min-silence-len must be >= 1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    logger = setup_logging(args.log_file)

    # Normalise extensions to lowercase with leading dot.
    extensions: Tuple[str, ...] = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.file_extensions
    )

    # Read S3 prefixes from input CSV.
    prefixes: List[str] = []
    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "s3_prefix" not in reader.fieldnames:
            logger.error("Input CSV must have an 's3_prefix' column.")
            sys.exit(1)
        for row in reader:
            prefix = (row.get("s3_prefix") or "").strip()
            if prefix:
                prefixes.append(prefix)

    if not prefixes:
        logger.warning("No prefixes found in %s — nothing to do.", args.csv)
        sys.exit(0)

    logger.info("Loaded %d prefix(es) from %s", len(prefixes), args.csv)

    # Collect all (s3_uri, prefix) tasks via serial S3 listing.
    all_tasks = collect_all_tasks(prefixes, extensions, logger)

    if not all_tasks:
        logger.warning("No audio files found under any prefix — nothing to analyse.")
        sys.exit(0)

    logger.info(
        "Dispatching %d file(s) across %d worker(s) ...",
        len(all_tasks),
        args.workers,
    )

    results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_audio_file,
                s3_uri,
                prefix,
                args.silence_thresh,
                args.min_silence_len,
                args.silence_fraction,
                extensions,
            ): s3_uri
            for s3_uri, prefix in all_tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Analysing"):
            s3_uri = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # Belt-and-suspenders: worker should never raise, but handle anyway.
                logger.error("Unexpected exception for %s: %s", s3_uri, exc)
                result = {
                    "s3_uri": s3_uri,
                    "prefix": "",
                    "total_duration_ms": None,
                    "nonsilent_duration_ms": None,
                    "silence_fraction": None,
                    "is_flagged": None,
                    "error": repr(exc),
                }

            results.append(result)

            if result.get("is_flagged"):
                logger.warning(
                    "FLAGGED  silence=%.1f%%  %s",
                    (result.get("silence_fraction") or 0) * 100,
                    result["s3_uri"],
                )

    write_results_csv(results, args.output_csv)

    flagged = sum(1 for r in results if r.get("is_flagged") is True)
    errors = sum(1 for r in results if r.get("error"))

    logger.info(
        "Done. Total: %d | Flagged: %d | Errors: %d",
        len(results),
        flagged,
        errors,
    )
    logger.info("Results written to %s", args.output_csv)


if __name__ == "__main__":
    main()
