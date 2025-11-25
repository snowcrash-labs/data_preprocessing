#!/usr/bin/env python3
"""
Download audio files from GCS, split on silence, and save segments locally.

Downloads CSV from GCS, extracts audio URIs, downloads each audio file, splits it
into segments based on silence detection, and saves segments to {datasets_dir}/{csv_stem}/desilenced_data/.
"""
import argparse
import csv
import logging
import os
import multiprocessing
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

from google.cloud import storage
from google.oauth2.credentials import Credentials
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("split_upload.log")],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GCS helper
# -----------------------------------------------------------------------------
def make_storage_client():
    # try:
    #     token = subprocess.check_output(
    #         ["gcloud", "auth", "print-access-token"], text=True
    #     ).strip()
    #     creds = Credentials(token)
    #     return storage.Client(credentials=creds)
    # except Exception:
    return storage.Client()


STORAGE = make_storage_client()


def parse_gs(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("Expected gs:// URI")
    bucket, *rest = uri[5:].split("/", 1)
    return bucket, rest[0] if rest else ""


def download_blob_to(uri: str, local: Path):
    b, p = parse_gs(uri)
    STORAGE.bucket(b).blob(p).download_to_filename(str(local))


def upload_blob_from(local: Path, uri: str):
    b, p = parse_gs(uri)
    STORAGE.bucket(b).blob(p).upload_from_filename(str(local))


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def process_and_upload(
    vocals_uri: str,
    ds_audio_dir: str,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    min_segment_len: int,
) -> None:
    """
    - downloads `vocals_uri` into a temp dir
    - splits on silence
    - saves each valid segment locally to ds_audio_dir/<stem_id>/<#####.wav>
    - cleans up temp
    """
    track_filename = (Path(vocals_uri).name)
    track_name = os.path.splitext(str(track_filename))[0]
    out_base = Path(ds_audio_dir) / track_name
    out_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav_path = td / f"{track_name}.wav"

        # 1️⃣ download
        try:
            download_blob_to(vocals_uri, wav_path)
        except Exception as e:
            logger.error(f"[{track_name}] download failed: {e}")
            return

        # 2️⃣ load & split
        audio = AudioSegment.from_wav(str(wav_path))
        segments = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )
        valid = [seg for seg in segments if len(seg) >= min_segment_len]
        logger.info(f"[{track_name}] found {len(valid)} ≥{min_segment_len} ms segments")

        # 3️⃣ export & upload
        target_sample_rate = 16000  # 16kHz
        for idx, seg in enumerate(valid, start=1):
            fname = f"{idx:05d}.wav"
            # Convert to mono if stereo (ensure single channel output)
            if seg.channels > 1:
                seg = seg.set_channels(1)
            # Resample to 16kHz if needed
            if seg.frame_rate != target_sample_rate:
                seg = seg.set_frame_rate(target_sample_rate)
            seg.export(str(out_base / fname), format="wav")
            # local_seg = td / fname
            # gs_target = segments_gs_prefix.rstrip("/") + f"/{track_name}/{fname}"
            # try:
            #     upload_blob_from(local_seg, gs_target)
            # except Exception as e:
            #     logger.error(f"[{track_name}] upload {fname} failed: {e}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    csv_gs_path: str,
    uri_name_header: str,
    ds_gs_prefix: str,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    min_segment_len: int,
    local_datasets_dir: str,
):
    dataset_path = os.path.join(local_datasets_dir, os.path.basename(ds_gs_prefix))
    ds_audio_dir = os.path.join(dataset_path, "audio")
    os.makedirs(ds_audio_dir, exist_ok=True)
    logger.info(f"Output directory: {ds_audio_dir}")
    
    # download CSV from GCS
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        local_csv = td / "input.csv"
        
        logger.info(f"Downloading CSV from {csv_gs_path}")
        try:
            download_blob_to(csv_gs_path, local_csv)
        except Exception as e:
            logger.error(f"Failed to download CSV: {e}")
            return
        
        # load URIs
        uris = []
        with open(local_csv, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            if uri_name_header not in rdr.fieldnames:
                logger.error(f"CSV missing '{uri_name_header}' column")
                return
            for row in rdr:

                track_name = os.path.basename(row[uri_name_header].strip())
                ds_uri = "gs://" +os.path.join(ds_gs_prefix, track_name)
                if ds_uri:
                    uris.append(ds_uri)


        csv_copy_path = os.path.join(dataset_path, "original_gs_input.csv")
        shutil.copy(local_csv, csv_copy_path)
        logger.info(f"Copied CSV to {csv_copy_path}")


    logger.info(f"{len(uris)} URIs to process")

    # use all CPU cores
    num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} parallel processes")

    # process_and_upload(uris[0], str(ds_audio_dir), min_silence_len, silence_thresh, keep_silence, min_segment_len)
    # process in a pool
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {
            exe.submit(
                process_and_upload,
                uri,
                str(ds_audio_dir),
                min_silence_len,
                silence_thresh,
                keep_silence,
                min_segment_len,
            ): uri
            for uri in uris
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Overall"):
            pass

    # Remove empty subdirectories
    logger.info("Checking for and removing empty subdirectories...")
    ds_audio_path = Path(ds_audio_dir)
    if ds_audio_path.exists():
        # Collect all subdirectories with their depth (walking bottom-up)
        # topdown=False gives us deepest directories first
        all_dirs = []
        for root, dirs, files in os.walk(ds_audio_path, topdown=False):
            root_path = Path(root)
            # Skip the root directory itself
            if root_path != ds_audio_path:
                depth = len(root_path.relative_to(ds_audio_path).parts)
                all_dirs.append((depth, root_path))
        
        # Sort by depth descending (deepest first) to handle nested empty dirs
        all_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # Remove empty directories
        removed_count = 0
        for depth, dir_path in all_dirs:
            try:
                # Check if directory still exists and is empty
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    removed_count += 1
                    logger.debug(f"Removed empty directory: {dir_path}")
            except OSError as e:
                logger.warning(f"Could not remove {dir_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} empty subdirectories")
        else:
            logger.info("No empty subdirectories found")

    logger.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download CSV from GCS, download audio stems, split on silence"
    )
    p.add_argument("--csv_gs_path", required=True, help="GCS path to CSV file, e.g. gs://my-bucket/data.csv")
    p.add_argument(
        "--uri_name_header",
        required=True,
        help="Name of the CSV column containing audio URIs",
    )
    p.add_argument(
        "--ds_gs_prefix", required=True, help="GCS prefix for dataset"
    )
    p.add_argument(
        "--min_silence_len", type=int, default=2000, help="ms of silence to split on"
    )
    p.add_argument(
        "--silence_thresh", type=int, default=-40, help="dBFS threshold for silence"
    )
    p.add_argument(
        "--keep_silence", type=int, default=100, help="ms of silence to leave at edges"
    )
    p.add_argument(
        "--min_segment_len", type=int, default=3000, help="ms minimum segment length"
    )
    p.add_argument(
        "--local_datasets_dir", type=str, default="~/gs_imports", help="Directory to store datasets"
    )
    args = p.parse_args()

    main(
        csv_gs_path=args.csv_gs_path,
        uri_name_header=args.uri_name_header,
        ds_gs_prefix=args.ds_gs_prefix,
        min_silence_len=args.min_silence_len,
        silence_thresh=args.silence_thresh,
        keep_silence=args.keep_silence,
        min_segment_len=args.min_segment_len,
        local_datasets_dir=args.local_datasets_dir,
    )
