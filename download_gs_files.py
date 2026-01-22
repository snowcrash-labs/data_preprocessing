#!/usr/bin/env python3
"""
Script to download files from Google Cloud Storage based on URLs in a CSV file.
"""

import csv
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
CSV_FILE = "/home/brendanoconnor/studio_results_20251209_2008.csv"
OUTPUT_DIR = "/home/brendanoconnor/gs_imports/wildSVDD_partial_download"  # Note: using path as specified
MAX_WORKERS = 4  # Number of parallel downloads


def download_file(gs_url: str, output_dir: str) -> tuple[str, bool, str]:
    """
    Download a single file from GCS.
    
    Returns:
        Tuple of (url, success, message)
    """
    try:
        # Run gsutil cp command
        result = subprocess.run(
            ["gsutil", "cp", gs_url, output_dir],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per file
        )
        
        if result.returncode == 0:
            return (gs_url, True, "Success")
        else:
            return (gs_url, False, result.stderr.strip())
    except subprocess.TimeoutExpired:
        return (gs_url, False, "Timeout")
    except Exception as e:
        return (gs_url, False, str(e))


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Read URLs from CSV
    urls = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('url', '').strip()
            if url and url.startswith('gs://'):
                urls.append(url)
    
    print(f"Found {len(urls)} GCS URLs to download")
    
    if not urls:
        print("No URLs found. Exiting.")
        return
    
    # Download files
    successful = 0
    failed = 0
    failed_urls = []
    
    print(f"\nDownloading with {MAX_WORKERS} parallel workers...")
    print("-" * 60)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, url, OUTPUT_DIR): url for url in urls}
        
        for i, future in enumerate(as_completed(futures), 1):
            url, success, message = future.result()
            filename = os.path.basename(url)
            
            if success:
                successful += 1
                print(f"[{i}/{len(urls)}] ✓ {filename[:60]}...")
            else:
                failed += 1
                failed_urls.append((url, message))
                print(f"[{i}/{len(urls)}] ✗ {filename[:60]}... - {message}")
    
    # Summary
    print("-" * 60)
    print(f"\nDownload complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed_urls:
        print(f"\nFailed downloads:")
        for url, msg in failed_urls[:10]:  # Show first 10 failures
            print(f"  - {os.path.basename(url)}: {msg}")
        if len(failed_urls) > 10:
            print(f"  ... and {len(failed_urls) - 10} more")


if __name__ == "__main__":
    main()

