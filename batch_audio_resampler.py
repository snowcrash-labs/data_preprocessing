#!/usr/bin/env python3
"""
Resample all audio files in a directory tree to a target sample rate in place.
Uses ffmpeg for high-quality resampling.
"""

import argparse
import logging
import os
import subprocess
import tempfile
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("resample_audio.log")],
)
logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 32000


def get_sample_rate(audio_path: str) -> int | None:
    """
    Quickly get the sample rate of an audio file using ffprobe.
    Returns None if unable to determine.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def resample_file(audio_path: str, target_sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[str, bool, str]:
    """
    Resample a single audio file to target sample rate in place.
    
    Returns:
        (path, success, message)
        message will be "SKIPPED" if already at target sample rate
    """
    audio_path = Path(audio_path)
    
    try:
        # Check if already at target sample rate
        current_rate = get_sample_rate(str(audio_path))
        if current_rate == target_sample_rate:
            return (str(audio_path), True, "SKIPPED")
        
        # Create temp file in same directory to ensure same filesystem (atomic rename)
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".wav",
            dir=audio_path.parent
        )
        os.close(temp_fd)
        
        # Use ffmpeg to resample
        # -y: overwrite output
        # -i: input file
        # -ar: target sample rate
        # -acodec pcm_s16le: 16-bit PCM output
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-ar", str(target_sample_rate),
            "-acodec", "pcm_s16le",
            temp_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return (str(audio_path), False, f"ffmpeg error: {result.stderr[:200]}")
        
        # Replace original with resampled version
        os.replace(temp_path, audio_path)
        
        return (str(audio_path), True, "OK")
        
    except Exception as e:
        # Clean up temp file on exception
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return (str(audio_path), False, str(e))


def main(audio_dir: str, target_sample_rate: int = DEFAULT_SAMPLE_RATE, num_workers: int = None):
    audio_dir = Path(audio_dir)
    
    if not audio_dir.exists():
        logger.error(f"Directory not found: {audio_dir}")
        return
    
    # Find all wav files recursively
    logger.info(f"Scanning for audio files in {audio_dir}...")
    audio_files = list(audio_dir.rglob("*.wav"))
    logger.info(f"Found {len(audio_files)} .wav files to resample to {target_sample_rate}Hz")
    
    if not audio_files:
        logger.warning("No audio files found")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} parallel workers")
    
    # Process files in parallel
    resampled_count = 0
    skipped_count = 0
    fail_count = 0
    total_files = len(audio_files)
    log_interval = max(1, total_files // 100)  # Log every 1%
    start_time = time.time()
    
    print(f"\nStarting resampling of {total_files:,} files to {target_sample_rate}Hz...")
    print(f"Workers: {num_workers}")
    print(f"(Files already at {target_sample_rate}Hz will be skipped)\n")
    
    # Create a partial function with the target sample rate bound
    resample_func = partial(resample_file, target_sample_rate=target_sample_rate)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(resample_func, str(f)): f
            for f in audio_files
        }
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing"), 1):
            path, success, message = future.result()
            if success:
                if message == "SKIPPED":
                    skipped_count += 1
                else:
                    resampled_count += 1
            else:
                fail_count += 1
                logger.error(f"Failed: {path} - {message}")
            
            # Print progress at intervals
            if i % log_interval == 0 or i == total_files:
                pct = (i / total_files) * 100
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total_files - i) / rate if rate > 0 else 0
                
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                
                print(f"Progress: {i:,}/{total_files:,} ({pct:.1f}%) | "
                      f"Resampled: {resampled_count:,} | Skipped: {skipped_count:,} | Failed: {fail_count} | "
                      f"Elapsed: {elapsed_str} | ETA: {remaining_str} | "
                      f"Rate: {rate:.1f} files/sec")
    
    # Final summary
    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    
    print(f"\n{'='*60}")
    print(f"COMPLETED")
    print(f"{'='*60}")
    print(f"Total files:    {total_files:,}")
    print(f"Resampled:      {resampled_count:,}")
    print(f"Skipped:        {skipped_count:,} (already at {target_sample_rate}Hz)")
    print(f"Failed:         {fail_count}")
    print(f"Total time:     {total_time_str}")
    print(f"Average rate:   {total_files / total_time:.1f} files/sec")
    print(f"{'='*60}")
    
    logger.info(f"Done. Resampled: {resampled_count}, Skipped: {skipped_count}, Failed: {fail_count}, Time: {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample all audio files to a target sample rate in place"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files to resample"
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Target sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all CPUs)"
    )
    args = parser.parse_args()
    
    main(
        audio_dir=args.audio_dir,
        target_sample_rate=args.target_sample_rate,
        num_workers=args.num_workers
    )
