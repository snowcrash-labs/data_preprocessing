#!/usr/bin/env python3
"""
Flatten nested directory structure in Google Cloud Storage.

First copies all files from the original prefix to a new prefix with "_FLATTENED" appended,
then reorganizes the copied files by moving them from subdirectories to parent directories,
flattening the structure. The original prefix remains unchanged. Skips files already at the
prefix level. Processes files in parallel.
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


def copy_blob_to_flattened(blob_info: tuple) -> tuple:
    """
    Copy a blob from original prefix to _FLATTENED prefix.
    
    Args:
        blob_info: Tuple of (original_blob_name, bucket, original_prefix, flattened_prefix)
    
    Returns:
        Tuple of (original_blob_name, copied_blob_name, status) where status is 'copied' or 'error'
    """
    original_blob_name, bucket, original_prefix, flattened_prefix = blob_info
    
    try:
        blob = bucket.blob(original_blob_name)
        
        # Construct new name in flattened prefix
        if original_blob_name.startswith(original_prefix):
            relative_path = original_blob_name[len(original_prefix):].lstrip("/")
            new_blob_name = flattened_prefix + relative_path
        else:
            return (original_blob_name, None, "error")
        
        # Copy blob to new location
        new_blob = bucket.copy_blob(blob, bucket, new_blob_name)
        
        return (original_blob_name, new_blob_name, "copied")
        
    except Exception as e:
        return (original_blob_name, None, f"error: {str(e)}")


def process_blob(blob_info: tuple) -> tuple:
    """
    Process a single blob: flatten its path if needed.
    
    Args:
        blob_info: Tuple of (blob_name, bucket, prefix) where bucket is a GCS bucket object
                   and prefix is the _FLATTENED prefix
    
    Returns:
        Tuple of (blob_name, new_name, status) where status is 'skipped', 'flattened', or 'error'
    """
    blob_name, bucket, prefix = blob_info
    
    try:
        blob = bucket.blob(blob_name)
        
        # Remove prefix to get relative path
        if blob_name.startswith(prefix):
            relative_path = blob_name[len(prefix):].lstrip("/")
        else:
            return (blob_name, None, "error")
        
        # Split into parts after prefix and filter out empty strings
        parts = [p for p in relative_path.rstrip("/").split("/") if p]
        
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
    print(f"Original prefix: {PREFIX}")
    
    # Create flattened prefix by appending "_FLATTENED"
    # Handle both cases: prefix ending with "/" and without
    if PREFIX.endswith("/"):
        FLATTENED_PREFIX = PREFIX[:-1] + "_FLATTENED/"
    else:
        FLATTENED_PREFIX = PREFIX + "_FLATTENED/"
    
    print(f"Flattened prefix: {FLATTENED_PREFIX}")
    
    # Create client and list all blobs
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    print("\nStep 1: Listing blobs in original prefix...")
    blobs = list(client.list_blobs(BUCKET_NAME, prefix=PREFIX))
    
    # Filter to only .wav files
    wav_blobs = [blob.name for blob in blobs if blob.name.endswith(".wav")]
    
    print(f"Found {len(wav_blobs)} .wav files")
    
    if not wav_blobs:
        print("No .wav files found. Exiting.")
        return
    
    # Check if there are any files that need flattening (i.e., files in subdirectories)
    # A file needs flattening if it has more than 1 part after removing the prefix
    files_need_flattening = []
    for blob_name in wav_blobs:
        if blob_name.startswith(PREFIX):
            relative_path = blob_name[len(PREFIX):]
            parts = relative_path.rstrip("/").split("/")
            # If more than 1 part, it's in a subdirectory and needs flattening
            if len(parts) > 1:
                files_need_flattening.append(blob_name)
    
    # If no files need flattening, all files are already flat - exit without doing anything
    if not files_need_flattening:
        print(f"\n✓ All {len(wav_blobs)} files are already flat (no subdirectories found).")
        print("No copy or flattening operations will be performed. Exiting.")
        return
    
    # Use all CPU cores for I/O-bound operations
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} parallel threads")
    
    # Step 1: Copy all files to _FLATTENED prefix
    print(f"\nStep 2: Copying all files to {FLATTENED_PREFIX}...")
    copy_blob_infos = [
        (blob_name, bucket, PREFIX, FLATTENED_PREFIX)
        for blob_name in wav_blobs
    ]
    
    copied_count = 0
    copy_error_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(copy_blob_to_flattened, blob_info): blob_info[0]
            for blob_info in copy_blob_infos
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying"):
            blob_name = futures[future]
            try:
                original_name, copied_name, status = future.result()
                
                if status == "copied":
                    copied_count += 1
                else:
                    copy_error_count += 1
                    print(f"\n❌ Error copying {blob_name}: {status}")
            except Exception as e:
                copy_error_count += 1
                print(f"\n❌ Exception copying {blob_name}: {str(e)}")
    
    print(f"Copied {copied_count} files to {FLATTENED_PREFIX}")
    if copy_error_count > 0:
        print(f"Copy errors: {copy_error_count}")
    
    # Step 2: Flatten files in the _FLATTENED prefix
    print(f"\nStep 3: Flattening files in {FLATTENED_PREFIX}...")
    
    # List blobs in flattened prefix
    flattened_blobs = list(client.list_blobs(BUCKET_NAME, prefix=FLATTENED_PREFIX))
    flattened_wav_blobs = [blob.name for blob in flattened_blobs if blob.name.endswith(".wav")]
    
    # Prepare blob info tuples for flattening workers
    flatten_blob_infos = [
        (blob_name, bucket, FLATTENED_PREFIX)
        for blob_name in flattened_wav_blobs
    ]
    
    skipped_count = 0
    flattened_count = 0
    flatten_error_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_blob, blob_info): blob_info[0]
            for blob_info in flatten_blob_infos
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Flattening"):
            blob_name = futures[future]
            try:
                original_name, new_name, status = future.result()
                
                if status == "skipped":
                    skipped_count += 1
                elif status == "flattened":
                    flattened_count += 1
                    print(f"\n{original_name}  ->  {new_name}")
                else:
                    flatten_error_count += 1
                    print(f"\n❌ Error flattening {blob_name}: {status}")
            except Exception as e:
                flatten_error_count += 1
                print(f"\n❌ Exception flattening {blob_name}: {str(e)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total files: {len(wav_blobs)}")
    print(f"  Copied to {FLATTENED_PREFIX}: {copied_count}")
    print(f"  Flattened: {flattened_count}")
    print(f"  Skipped (already flat): {skipped_count}")
    print(f"  Copy errors: {copy_error_count}")
    print(f"  Flatten errors: {flatten_error_count}")
    print(f"\n✓ Flattened files are available at: {FLATTENED_PREFIX}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
