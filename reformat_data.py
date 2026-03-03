#!/usr/bin/env python3
"""
Resample all audio files in a directory tree to a target sample rate and channel layout.
Uses ffmpeg for high-quality resampling.

Channel output:
  - Default: output is mono (stereo inputs downmixed to mono, mono left as mono).
  - With --stereo_out: output is stereo (2 channels); mono inputs are duplicated to stereo.

Modes:
  - In-place (default): overwrite each file with its resampled version.
  - Copy: write resampled files to a new directory, preserving structure.
"""

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

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

# Extensions treated as audio (lowercase); ffmpeg will decode/encode by extension
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus", ".webm", ".wma", ".mka"}


def collect_audio_files(root: Path) -> List[Path]:
    """Recursively find all files under root whose suffix is in AUDIO_EXTENSIONS."""
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p)
    return sorted(files)


def get_sample_rate(audio_path: str) -> Optional[int]:
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


def get_channel_count(audio_path: str) -> Optional[int]:
    """
    Get the number of audio channels using ffprobe.
    Returns None if unable to determine.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def resample_file(
    src_path: str,
    target_sample_rate: int,
    dst_path: Optional[str] = None,
    stereo_out: bool = False,
) -> tuple[str, bool, str]:
    """
    Resample a single audio file to target sample rate and set channel layout.

    If dst_path is None: resample in place (temp file then replace original).
    If dst_path is set: write resampled output to dst_path (parent dirs created).

    stereo_out: if True, output is stereo (2 channels); if False, output is mono (1 channel).
    Stereo inputs are downmixed to mono when stereo_out is False; mono inputs are
    duplicated to stereo when stereo_out is True.

    Returns:
        (path, success, message)
        message will be "SKIPPED" if already at target sample rate and desired channels
    """
    src_path = Path(src_path)
    in_place = dst_path is None
    dst_path = Path(dst_path) if dst_path else src_path
    want_channels = 2 if stereo_out else 1

    try:
        current_rate = get_sample_rate(str(src_path))
        current_channels = get_channel_count(str(src_path))
        if current_rate == target_sample_rate and current_channels is not None and current_channels == want_channels:
            if not in_place:
                os.makedirs(dst_path.parent, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            return (str(src_path), True, "SKIPPED")

        if in_place:
            temp_fd, out_path = tempfile.mkstemp(
                suffix=src_path.suffix,
                dir=src_path.parent,
            )
            os.close(temp_fd)
        else:
            os.makedirs(dst_path.parent, exist_ok=True)
            out_path = dst_path

        # ffmpeg: -ar sample rate, -ac channels (1=mono, 2=stereo)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(src_path),
            "-ar", str(target_sample_rate),
            "-ac", str(want_channels),
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            if in_place and os.path.exists(out_path):
                os.remove(out_path)
            return (str(src_path), False, f"ffmpeg error: {result.stderr[:200]}")

        if in_place:
            os.replace(out_path, src_path)

        return (str(src_path), True, "OK")

    except Exception as e:
        if in_place and "out_path" in locals() and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return (str(src_path), False, str(e))


def _resample_task(args: Tuple[str, Optional[str], int, bool]) -> Tuple[str, bool, str]:
    """Module-level wrapper for ProcessPoolExecutor (picklable)."""
    src, dst, target_sr, stereo_out = args
    return resample_file(src, target_sr, dst, stereo_out)


def main(
    audio_dir: str,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_workers: Optional[int] = None,
    output_dir: Optional[str] = None,
    stereo_out: bool = False,
):
    audio_dir = Path(audio_dir).resolve()
    if not audio_dir.exists():
        logger.error(f"Directory not found: {audio_dir}")
        return

    in_place = output_dir is None
    out_dir = Path(output_dir).resolve() if output_dir else None
    if not in_place and out_dir.exists() and not out_dir.is_dir():
        logger.error(f"Output path exists and is not a directory: {out_dir}")
        return

    logger.info(f"Scanning for audio files in {audio_dir}...")
    audio_files = collect_audio_files(audio_dir)
    channel_mode = "stereo" if stereo_out else "mono"
    logger.info(
        f"Found {len(audio_files)} audio files to resample to {target_sample_rate}Hz, output {channel_mode} "
        f"({'in place' if in_place else f'copy to {out_dir}'})"
    )

    if not audio_files:
        logger.warning("No audio files found")
        return

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} parallel workers")

    # (src_path, dst_path|None) for each file
    if in_place:
        tasks = [(str(f), None) for f in audio_files]
    else:
        tasks = [(str(f), str(out_dir / f.relative_to(audio_dir))) for f in audio_files]

    resampled_count = 0
    skipped_count = 0
    fail_count = 0
    total_files = len(tasks)
    log_interval = max(1, total_files // 100)
    start_time = time.time()

    print(f"\nStarting resampling of {total_files:,} files to {target_sample_rate}Hz ({channel_mode})...")
    print(f"Mode: {'in place' if in_place else 'copy to new dataset'}")
    print(f"Workers: {num_workers}")
    print(f"(Files already at {target_sample_rate}Hz and {channel_mode} will be skipped)\n")

    task_args = [(src, dst, target_sample_rate, stereo_out) for src, dst in tasks]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_resample_task, t): t[0]
            for t in task_args
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
    print(f"Skipped:        {skipped_count:,} (already at {target_sample_rate}Hz and {channel_mode})")
    print(f"Failed:         {fail_count}")
    print(f"Total time:     {total_time_str}")
    print(f"Average rate:   {total_files / total_time:.1f} files/sec")
    print(f"{'='*60}")
    
    logger.info(f"Done. Resampled: {resampled_count}, Skipped: {skipped_count}, Failed: {fail_count}, Time: {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample all audio files to a target sample rate (in place or copy to new dir)"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files to resample",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Target sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If set, copy resampled files here (preserving structure) instead of changing in place",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all CPUs)",
    )
    parser.add_argument(
        "--stereo_out",
        action="store_true",
        help="Ensure output is stereo (2 channels). Default: output is mono (stereo inputs downmixed, mono left as mono)",
    )
    args = parser.parse_args()

    main(
        audio_dir=args.audio_dir,
        target_sample_rate=args.target_sample_rate,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        stereo_out=args.stereo_out,
    )
