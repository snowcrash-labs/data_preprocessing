#!/usr/bin/env python3
"""
Resample an entire dataset to a target sample rate, preserving directory structure.

Expected input structure:
    <dataset_path>/
        <singer_id>/
            <song_id>/
                *.wav (or other audio files)

Usage:
    python3 resample_dataset.py \
        --input-dir /path/to/source_dataset \
        --output-dir /path/to/output_dataset \
        --target-sr 16000 \
        [--num-workers 4]
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile as sf
import librosa
from tqdm import tqdm


def resample_file(src_path: str, dst_path: str, target_sr: int) -> str:
    """Resample a single audio file and save to dst_path."""
    try:
        audio, orig_sr = sf.read(src_path)

        if orig_sr != target_sr:
            if audio.ndim == 2:
                audio = audio.T
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
                audio = audio.T
            else:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        sf.write(dst_path, audio, target_sr)
        return f"OK: {src_path} ({orig_sr} -> {target_sr})"
    except Exception as e:
        return f"FAIL: {src_path} — {e}"


def collect_audio_files(input_dir: str) -> list[tuple[str, str]]:
    """Walk the dataset and collect (full_path, relative_path) pairs for audio files."""
    audio_extensions = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in audio_extensions:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, input_dir)
                files.append((full_path, rel_path))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Resample a dataset to a target sample rate, preserving directory structure."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Path to the source dataset directory"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Path to the output dataset directory"
    )
    parser.add_argument(
        "--target-sr", type=int, required=True,
        help="Target sample rate in Hz (e.g. 16000)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    target_sr = args.target_sr
    num_workers = args.num_workers

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Target SR:  {target_sr} Hz")
    print(f"Workers:    {num_workers}")
    print()

    # Collect all audio files
    print("Scanning for audio files...")
    audio_files = collect_audio_files(input_dir)
    total = len(audio_files)
    print(f"Found {total} audio files.")
    print()

    if total == 0:
        print("Nothing to do.")
        sys.exit(0)

    # Process files in parallel with tqdm progress bar
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for src_path, rel_path in audio_files:
            dst_path = os.path.join(output_dir, rel_path)
            future = executor.submit(resample_file, src_path, dst_path, target_sr)
            futures[future] = rel_path

        with tqdm(total=total, desc="Resampling", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result.startswith("FAIL"):
                    failed += 1
                    tqdm.write(result)
                pbar.update(1)

    print()
    print("========================================")
    print(f"  Done! {total - failed}/{total} files resampled successfully.")
    if failed > 0:
        print(f"  {failed} files failed.")
    print(f"  Output: {output_dir}")
    print("========================================")


if __name__ == "__main__":
    main()
