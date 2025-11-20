#!/usr/bin/env python3
"""
Flatten nested directory structure in Google Cloud Storage.

Reorganizes audio files in GCS by moving files from subdirectories to parent directories,
flattening the structure. Skips files already at the prefix level. Processes files in parallel.
"""
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Flatten song-level directory datasets in Google Cloud Storage"
)
parser.add_argument(
    "--project_id",
    default="sc_music_research",
    help="Google Cloud project ID (default: sc_music_research)",
)
parser.add_argument(
    "--bucket_name",
    default="music-dataset-hooktheory-audio",
    help="Google Cloud Storage bucket name (default: music-dataset-hooktheory-audio)",
)
parser.add_argument(
    "--prefix",
    default="roformer_voice_sep_custom_sample/",
    help="Common prefix where data lives under (default: roformer_voice_sep_custom_sample/)",
)
args = parser.parse_args()

PROJECT_ID = args.project_id
BUCKET_NAME = args.bucket_name
PREFIX = args.prefix


def process_blob(blob_info: tuple) -> tuple:
    """
    Process a single blob: flatten its path if needed.
    
    Args:
        blob_info: Tuple of (blob_name, bucket, prefix) where bucket is a GCS bucket object
    
    Returns:
        Tuple of (blob_name, new_name, status) where status is 'skipped', 'flattened', or 'error'
    """
    blob_name, bucket, prefix = blob_info
    
    try:
        blob = bucket.blob(blob_name)
        
        # Remove prefix to get relative path
        if blob_name.startswith(prefix):
            relative_path = blob_name[len(prefix):]
        else:
            return (blob_name, None, "error")
        
        # Split into parts after prefix
        parts = relative_path.rstrip("/").split("/")
        
        # If file is already directly after prefix (only filename, no subdirectory), skip it
        if len(parts) == 1:
            return (blob_name, None, "skipped")
        
        # If less than 2 parts, can't flatten
        if len(parts) < 2:
            return (blob_name, None, "skipped")
        
        # Extract directory name and parent parts
        dir_name = parts[-2]  # the directory containing the wav
        parent_parts = parts[:-2]  # everything above that directory
        
        # Construct new name
        if parent_parts:
            parent_prefix = "/".join(parent_parts)
            new_name = f"{prefix}{parent_prefix}/{dir_name}.wav"
        else:
            new_name = f"{prefix}{dir_name}.wav"
        
        # If new name is same as old name, skip
        if new_name == blob_name:
            return (blob_name, None, "skipped")
        
        # COPY to new name
        new_blob = bucket.copy_blob(blob, bucket, new_name)
        
        # DELETE old object
        blob.delete()
        
        return (blob_name, new_name, "flattened")
        
    except Exception as e:
        return (blob_name, None, f"error: {str(e)}")


def main():
    print(f"Connecting to GCS bucket: {BUCKET_NAME}")
    print(f"Prefix: {PREFIX}")
    
    # Create client and list all blobs
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    print("Listing blobs...")
    blobs = list(client.list_blobs(BUCKET_NAME, prefix=PREFIX))
    
    # Filter to only .wav files
    wav_blobs = [blob.name for blob in blobs if blob.name.endswith(".wav")]
    
    print(f"Found {len(wav_blobs)} .wav files")
    
    if not wav_blobs:
        print("No .wav files found. Exiting.")
        return
    
    # Prepare blob info tuples for workers (pass bucket object for thread safety)
    blob_infos = [
        (blob_name, bucket, PREFIX)
        for blob_name in wav_blobs
    ]
    
    # Use all CPU cores for I/O-bound operations
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} parallel threads")
    
    # Process blobs in parallel
    skipped_count = 0
    flattened_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_blob, blob_info): blob_info[0]
            for blob_info in blob_infos
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            blob_name = futures[future]
            try:
                original_name, new_name, status = future.result()
                
                if status == "skipped":
                    skipped_count += 1
                elif status == "flattened":
                    flattened_count += 1
                    print(f"\n{original_name}  ->  {new_name}")
                else:
                    error_count += 1
                    print(f"\n❌ Error processing {blob_name}: {status}")
            except Exception as e:
                error_count += 1
                print(f"\n❌ Exception processing {blob_name}: {str(e)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total files: {len(wav_blobs)}")
    print(f"  Flattened: {flattened_count}")
    print(f"  Skipped (already flat): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
