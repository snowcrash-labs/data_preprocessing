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
    dst_dir: str,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    min_segment_len: int,
) -> None:
    """
    - downloads `vocals_uri` into a temp dir
    - splits on silence
    - saves each valid segment locally to dst_dir/<stem_id>/<#####.wav>
    - cleans up temp
    """
    track_filename = (Path(vocals_uri).name)
    track_name = os.path.splitext(str(track_filename))[0]
    out_base = Path(dst_dir) / track_name
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
        for idx, seg in enumerate(valid, start=1):
            fname = f"{idx:05d}.wav"
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
    workers: int,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    min_segment_len: int,
    local_datasets_dir: str,
):
    # Extract CSV filename stem and create destination directory
    csv_filename = Path(csv_gs_path).name
    csv_stem = csv_filename.rsplit(".", 1)[0] if "." in csv_filename else csv_filename
    
    # Create ~/gs_imports if it doesn't exist
    local_datasets_dir.mkdir(exist_ok=True)
    
    dataset_name = os.path.basename(ds_gs_prefix)
    base_dir = local_datasets_dir / dataset_name
    dst_dir = base_dir / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {dst_dir}")
    
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

        try:
            csv_copy_path = base_dir / "original_gs_input.csv"
            shutil.copy(local_csv, csv_copy_path)
            logger.info(f"Copied CSV to {csv_copy_path}")
        except Exception as e:
            logger.error(f"Failed to copy CSV to data directory: {e}")

    logger.info(f"{len(uris)} URIs to process")

    # use all CPU cores
    num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} parallel processes")

    # process_and_upload(uris[0], str(dst_dir), min_silence_len, silence_thresh, keep_silence, min_segment_len)
    # process in a pool
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {
            exe.submit(
                process_and_upload,
                uri,
                str(dst_dir),
                min_silence_len,
                silence_thresh,
                keep_silence,
                min_segment_len,
            ): uri
            for uri in uris
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Overall"):
            pass

    

    logger.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download CSV from GCS, download audio stems, split on silence, save segments to ~/gs_imports/{csv_stem}/data"
    )
    p.add_argument("--csv_gs_path", required=True, help="GCS path to CSV file, e.g. gs://my-bucket/data.csv")
    p.add_argument(
        "--uri_name_header",
        required=True,
        help="Name of the CSV column containing audio URIs",
    )
    p.add_argument(
        "--ds_gs_prefix", type=str, default="music-dataset-hooktheory-audio/roformer_voice_separated", help="Destination directory"
    )
    p.add_argument(
        "--workers", type=int, default=4, help="Number of parallel processes"
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
        workers=args.workers,
        min_silence_len=args.min_silence_len,
        silence_thresh=args.silence_thresh,
        keep_silence=args.keep_silence,
        min_segment_len=args.min_segment_len,
        local_datasets_dir=args.local_datasets_dir,
    )
