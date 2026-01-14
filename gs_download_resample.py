#!/usr/bin/env python3
"""
Download audio files from GCS and resample to 16kHz mono.

Downloads CSV from GCS, extracts audio URIs, downloads each audio file,
resamples to 16kHz mono, and saves to {datasets_dir}/{ds_name}/audio/.
"""
import argparse
import csv
import logging
import os
import multiprocessing
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

from google.cloud import storage
from pydub import AudioSegment
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("gs_download_resample.log")],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GCS helper
# -----------------------------------------------------------------------------
def make_storage_client():
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


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def download_and_resample(
    vocals_uri: str,
    output_dir: str,
    target_sample_rate: int = 16000,
    gs_file_uri_in_csv: bool = False,
) -> None:
    """
    - Downloads `vocals_uri` into a temp dir
    - Resamples to target sample rate and converts to mono
    - Saves to output_dir/<track_name>.wav
    """
    
    if gs_file_uri_in_csv:
        track_name = os.path.basename(os.path.dirname(vocals_uri))
    else:
        track_filename = Path(vocals_uri).name
        track_name = os.path.splitext(str(track_filename))[0]
    out_path = Path(output_dir) / f"{track_name}.wav"
    
    # Skip if already processed
    if out_path.exists():
        logger.info(f"[{track_name}] already exists, skipping")
        return
    
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav_path = td / f"{track_name}.wav"

        # 1️⃣ download
        try:
            download_blob_to(vocals_uri, wav_path)
        except Exception as e:
            logger.error(f"[{track_name}] download failed: {e}")
            return

        # 2️⃣ load & resample
        try:
            audio = AudioSegment.from_wav(str(wav_path))
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Resample if needed
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
            
            # 3️⃣ export
            audio.export(str(out_path), format="wav")
            logger.info(f"[{track_name}] downloaded resampled to {target_sample_rate}Hz mono, and saved to {out_path}")

        except Exception as e:
            logger.error(f"[{track_name}] processing failed: {e}")
            return


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    csv_gs_path: str,
    uri_name_header: str,
    ds_gs_prefix: str,
    local_datasets_dir: str,
    target_sample_rate: int = 16000,
    gs_file_uri_in_csv: bool = False,
    parallel: bool = True,
):
    # Expand ~ to home directory
    local_datasets_dir = os.path.expanduser(local_datasets_dir)
    dataset_path = os.path.join(local_datasets_dir, os.path.basename(ds_gs_prefix))
    audio_dir = os.path.join(dataset_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    logger.info(f"Output directory: {audio_dir}")
    
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
                if gs_file_uri_in_csv:
                    ds_uri = row[uri_name_header].strip()
                else:
                    track_name = os.path.basename(row[uri_name_header].strip())
                    ds_uri = "gs://" + os.path.join(ds_gs_prefix, track_name)
                if ds_uri:
                    uris.append(ds_uri)

        csv_copy_path = os.path.join(dataset_path, "original_gs_input.csv")
        shutil.copy(local_csv, csv_copy_path)
        logger.info(f"Copied CSV to {csv_copy_path}")

    logger.info(f"{len(uris)} URIs to process")

    if parallel:
        # use all CPU cores
        num_workers = multiprocessing.cpu_count()
        logger.info(f"Using {num_workers} parallel processes")

        # process in a pool
        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            futures = {
                exe.submit(
                    download_and_resample,
                    uri,
                    str(audio_dir),
                    target_sample_rate,
                    gs_file_uri_in_csv,
                ): uri
                for uri in uris
            }
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                pass
    else:
        # sequential processing
        logger.info("Using sequential processing (parallel disabled)")
        for uri in tqdm(uris, desc="Downloading"):
            download_and_resample(uri, str(audio_dir), target_sample_rate, gs_file_uri_in_csv)

    logger.info("Done downloading and resampling.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download audio from GCS and resample to 16kHz mono"
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
        "--local_datasets_dir", type=str, default="~/gs_imports", help="Directory to store datasets"
    )
    p.add_argument(
        "--target_sample_rate", type=int, default=16000, help="Target sample rate in Hz (default: 16000)"
    )
    p.add_argument(
        "--gs_file_uri_in_csv", action="store_true",
        default=False,
        help="Whether the CSV contains the GCS file URI (or otherwise, the relevant folder to the file)"
    )
    p.add_argument(
        "--no-parallel", action="store_true",
        default=False,
        help="Disable parallel processing (process files sequentially)"
    )
    args = p.parse_args()

    main(
        csv_gs_path=args.csv_gs_path,
        uri_name_header=args.uri_name_header,
        ds_gs_prefix=args.ds_gs_prefix,
        local_datasets_dir=args.local_datasets_dir,
        target_sample_rate=args.target_sample_rate,
        gs_file_uri_in_csv=args.gs_file_uri_in_csv,
        parallel=not args.no_parallel,
    )

