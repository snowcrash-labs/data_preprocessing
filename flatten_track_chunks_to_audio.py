#!/usr/bin/env python3
"""
Flatten a dataset of track subdirectories (each containing audio chunks) into a
single directory of one audio file per track. Each output file is the concatenation
of all chunks from that track, optionally with 0.5s silence between chunks.

Input structure:
  input_dir/
    track_a/
      chunk1.wav
      chunk2.wav
      ...
    track_b/
      ...

Output structure:
  output_dir/
    track_a.wav
    track_b.wav
    ...

Requires: pydub (and ffmpeg on PATH for non-WAV input).
  pip install pydub
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pydub import AudioSegment

# Extensions we consider as audio chunks (pydub can load these with ffmpeg)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _iter_immediate_subdirs(parent: Path) -> list[Path]:
    return sorted(
        [p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )


def _collect_audio_chunks(track_dir: Path) -> list[Path]:
    chunks = [
        p
        for p in track_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(chunks, key=lambda p: p.name)


def _ensure_audio_format(
    seg: AudioSegment,
    target_frame_rate: int,
    target_channels: int,
    target_sample_width: int,
) -> AudioSegment:
    if seg.channels != target_channels:
        seg = seg.set_channels(target_channels)
    if seg.frame_rate != target_frame_rate:
        seg = seg.set_frame_rate(target_frame_rate)
    if seg.sample_width != target_sample_width:
        seg = seg.set_sample_width(target_sample_width)
    return seg


def flatten_dataset(
    input_dir: Path,
    output_dir: Path,
    silence_between: bool = True,
    silence_s: float = 0.5,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    track_dirs = _iter_immediate_subdirs(input_dir)
    if not track_dirs:
        raise RuntimeError(f"No subdirectories found under: {input_dir}")

    silence_ms = int(round(silence_s * 1000)) if silence_between else 0
    silence_seg = AudioSegment.silent(duration=silence_ms) if silence_between else None

    for track_dir in track_dirs:
        chunks = _collect_audio_chunks(track_dir)
        if not chunks:
            continue

        first = AudioSegment.from_file(str(chunks[0]))
        target_frame_rate = first.frame_rate
        target_channels = first.channels
        target_sample_width = first.sample_width

        out = AudioSegment.empty()
        for i, chunk_path in enumerate(chunks):
            seg = AudioSegment.from_file(str(chunk_path))
            seg = _ensure_audio_format(
                seg, target_frame_rate, target_channels, target_sample_width
            )
            out += seg
            if silence_between and i != len(chunks) - 1:
                out += silence_seg

        # Output filename = track name, always as .wav
        stem = track_dir.name
        if stem.endswith(".wav"):
            out_name = stem
        else:
            out_name = f"{stem}.wav"
        out_path = output_dir / out_name
        out.export(str(out_path), format="wav")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Flatten track subdirectories of audio chunks into one file per track."
    )
    p.add_argument(
        "input_dir",
        type=Path,
        help="Root directory containing one subdirectory per track, each with audio chunks.",
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write one concatenated WAV per track (no subdirs).",
    )
    p.add_argument(
        "--silence_between",
        action="store_true",
        default=True,
        help="Insert 0.5s silence between chunks (default: True).",
    )
    p.add_argument(
        "--no_silence_between",
        action="store_false",
        dest="silence_between",
        help="Do not insert silence between chunks.",
    )
    p.add_argument(
        "--silence_s",
        type=float,
        default=0.5,
        help="Silence duration in seconds between chunks when --silence_between (default: 0.5).",
    )
    args = p.parse_args()

    flatten_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        silence_between=args.silence_between,
        silence_s=args.silence_s,
    )


if __name__ == "__main__":
    main()
