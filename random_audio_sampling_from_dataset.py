#!/usr/bin/env python3
"""
Randomly sample N WAV files from a dataset directory (recursively) and copy
them into a flat output directory.

Structure assumed: dataset_dir/voice_id/song_id/*.wav
"""
import argparse
import os
import random
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Randomly sample WAV files from a dataset and copy to a flat output directory'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Root directory to search for WAV files (e.g. voice_id/song_id/*.wav)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory where sampled WAV files will be copied (flat list of files)'
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=20,
        help='Number of WAV files to sample (default: 20)'
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.output_dir)
    n = args.num_samples

    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all .wav paths (relative to dataset_dir)
    wavs = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, dataset_dir)
                wavs.append(rel)

    if not wavs:
        print("No WAV files found.", file=sys.stderr)
        sys.exit(1)

    if len(wavs) < n:
        print(f"Only {len(wavs)} WAV files found, sampling all", file=sys.stderr)
        chosen = wavs
    else:
        chosen = random.sample(wavs, n)

    os.makedirs(out_dir, exist_ok=True)
    for rel in chosen:
        src = os.path.join(dataset_dir, rel)
        # Flat filename: path with os.sep replaced so names are unique
        flat_name = rel.replace(os.sep, '__')
        dst = os.path.join(out_dir, flat_name)
        shutil.copy2(src, dst)
        print(dst)

    print(f"\nCopied {len(chosen)} files to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
