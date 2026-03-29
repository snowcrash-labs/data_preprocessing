#!/usr/bin/env python3
"""
1) Create anonymized mapping for singer subdirectories (no renames):
   - Build a randomized mapping of immediate subdirectory name -> id### format
   - Save a JSON mapping of original_name -> anonymized_id

2) Synthesize diarisation mixtures:
   - Randomly select N speakers (configurable; default 2) from a singers directory
   - Randomly concatenate their chunk wav files with a fixed silence padding between chunks
   - Write a CSV per track with start_time/end_time/speaker labels
   - Output filenames use anonymized ids (e.g. id001-id014__00001.wav/.csv)

Dependencies:
  - pydub (and ffmpeg installed/available on PATH)
      pip install pydub
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydub import AudioSegment


@dataclass(frozen=True)
class SegmentLabel:
    start_time: float
    end_time: float
    speaker: int


def _derived_seed(seed: int | None, purpose: str) -> int | None:
    """
    Derive independent deterministic seeds for different purposes so that using the same
    seed value doesn't correlate mapping creation with synthesis sampling.
    """
    if seed is None:
        return None
    h = hashlib.sha256(f"{seed}:{purpose}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _iter_immediate_subdirs(parent: Path) -> List[Path]:
    return sorted(
        [p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )


def create_anonymized_mapping(
    directory: Path,
    mapping_json_path: Path | None = None,
    start_index: int = 1,
    seed: int | None = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    """
    Creates a randomized mapping of immediate subdirectory names to id### and writes mapping JSON.
    Does NOT rename any directories.

    Returns:
      mapping: original_name -> anonymized_id
    """
    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    subdirs = _iter_immediate_subdirs(directory)
    if not subdirs:
        raise RuntimeError(f"No subdirectories found under: {directory}")

    n = len(subdirs)
    width = max(3, len(str(start_index + n - 1)))

    if mapping_json_path is None:
        mapping_json_path = directory / "singer_id_mapping.json"
    mapping_json_path = mapping_json_path.resolve()
    if mapping_json_path.exists() and not overwrite:
        raise RuntimeError(
            f"Mapping file already exists: {mapping_json_path}. "
            "Pass --overwrite_mapping to overwrite."
        )

    rng = random.Random(_derived_seed(seed, "mapping"))
    names = [p.name for p in subdirs]
    rng.shuffle(names)

    # orig_name -> anon_id (used internally for synthesis)
    orig_to_anon: Dict[str, str] = {}
    for i, name in enumerate(names, start=start_index):
        new_name = f"id{i:0{width}d}"
        orig_to_anon[name] = new_name

    # Write JSON as anon_id -> orig_name (requested orientation)
    anon_to_orig = {anon: orig for orig, anon in orig_to_anon.items()}
    mapping_json_path.write_text(json.dumps(anon_to_orig, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return orig_to_anon


def _list_singer_dirs(singers_dir: Path) -> List[Path]:
    singers_dir = singers_dir.resolve()
    singer_dirs = _iter_immediate_subdirs(singers_dir)
    if not singer_dirs:
        raise RuntimeError(f"No singer subdirectories found under: {singers_dir}")
    return singer_dirs


def _collect_wavs_recursive(singer_dir: Path) -> List[Path]:
    wavs = sorted([p for p in singer_dir.rglob("*.wav") if p.is_file()], key=lambda p: str(p))
    return wavs


def _ensure_audio_format(seg: AudioSegment, target_frame_rate: int, target_channels: int, target_sample_width: int) -> AudioSegment:
    if seg.channels != target_channels:
        seg = seg.set_channels(target_channels)
    if seg.frame_rate != target_frame_rate:
        seg = seg.set_frame_rate(target_frame_rate)
    if seg.sample_width != target_sample_width:
        seg = seg.set_sample_width(target_sample_width)
    return seg


def _unique_stem(dest_audio_dir: Path, base_stem: str, track_index: int) -> str:
    """
    Keeps the filename stem starting with base_stem, and disambiguates with __{track_index}.
    """
    stem = f"{base_stem}__{track_index:05d}"
    wav_path = dest_audio_dir / f"{stem}.wav"
    if not wav_path.exists():
        return stem

    # Extremely unlikely unless re-running; fall back to random suffix
    for k in range(1, 10000):
        alt = f"{stem}__r{k:04d}"
        if not (dest_audio_dir / f"{alt}.wav").exists():
            return alt
    raise RuntimeError("Could not find a unique filename stem after many attempts.")


def synthesize_tracks(
    singers_dir: Path,
    dest_dir: Path,
    num_tracks: int,
    mapping: Dict[str, str],
    num_speakers: int = 2,
    chunks_per_track: Optional[int] = None,
    silence_s: float = 0.5,
    seed: int | None = None,
    max_chunk_s: float | None = None,
) -> None:
    singers_dir = singers_dir.resolve()
    dest_dir = dest_dir.resolve()
    rng = random.Random(_derived_seed(seed, "synthesize"))

    singer_dirs = _list_singer_dirs(singers_dir)

    # Pre-collect wavs for each singer (skip empty).
    singer_wavs: Dict[str, List[Path]] = {}
    singer_dirs_with_wavs: List[Path] = []
    for sd in singer_dirs:
        wavs = _collect_wavs_recursive(sd)
        if wavs:
            singer_wavs[sd.name] = wavs
            singer_dirs_with_wavs.append(sd)

    if len(singer_dirs_with_wavs) < num_speakers:
        raise RuntimeError(
            f"Need at least {num_speakers} singers with wav chunks; found {len(singer_dirs_with_wavs)} under {singers_dir}"
        )

    missing_mapping = [sd.name for sd in singer_dirs_with_wavs if sd.name not in mapping]
    if missing_mapping:
        raise RuntimeError(
            "Mapping JSON is missing entries for some singer directories (first few): "
            + ", ".join(missing_mapping[:10])
        )

    dest_audio_dir = dest_dir / "audio"
    dest_csv_dir = dest_dir / "speaker_timestamps"
    dest_audio_dir.mkdir(parents=True, exist_ok=True)
    dest_csv_dir.mkdir(parents=True, exist_ok=True)

    silence_ms = int(round(silence_s * 1000))
    silence_seg = AudioSegment.silent(duration=silence_ms)

    # Cycle through shuffled singer list without replacement per track; reshuffle when needed.
    pool = singer_dirs_with_wavs[:]
    rng.shuffle(pool)
    pool_idx = 0

    for track_i in range(1, num_tracks + 1):
        if pool_idx + num_speakers > len(pool):
            rng.shuffle(pool)
            pool_idx = 0

        chosen_dirs = list(pool[pool_idx : pool_idx + num_speakers])
        pool_idx += num_speakers
        rng.shuffle(chosen_dirs)

        chosen_names = [p.name for p in chosen_dirs]
        chosen_anon = [mapping[n] for n in chosen_names]
        base_stem = "-".join(chosen_anon)
        stem = _unique_stem(dest_audio_dir, base_stem, track_i)

        # Assign speaker labels by chosen order: 0..num_speakers-1
        speakers = list(range(num_speakers))
        speaker_to_name = {spk: chosen_names[spk] for spk in speakers}

        # Load first chunk to set target audio format.
        first_path = rng.choice(singer_wavs[speaker_to_name[0]])
        first_seg = AudioSegment.from_wav(str(first_path))
        target_frame_rate = first_seg.frame_rate
        target_channels = first_seg.channels
        target_sample_width = first_seg.sample_width

        out = AudioSegment.empty()
        labels: List[SegmentLabel] = []
        cursor_ms = 0

        # Build the concatenation plan.
        # - If chunks_per_track is None: use ALL chunks from each singer exactly once.
        # - Else: draw that many chunks total, cycling through each singer's chunks without
        #   repeating until exhausted (then reshuffle and continue).
        plan: List[Tuple[int, Path]] = []
        if chunks_per_track is None:
            for spk in speakers:
                name = speaker_to_name[spk]
                wavs = singer_wavs[name][:]
                rng.shuffle(wavs)
                plan.extend((spk, w) for w in wavs)
            rng.shuffle(plan)
        else:
            if chunks_per_track < num_speakers:
                raise RuntimeError("--chunks_per_track must be >= --num_speakers (or omit it to use all chunks once).")

            per_speaker_queues: Dict[int, List[Path]] = {}
            per_speaker_idx: Dict[int, int] = {}
            for spk in speakers:
                name = speaker_to_name[spk]
                wavs = singer_wavs[name][:]
                rng.shuffle(wavs)
                per_speaker_queues[spk] = wavs
                per_speaker_idx[spk] = 0

            def next_wav(spk: int) -> Path:
                q = per_speaker_queues[spk]
                i = per_speaker_idx[spk]
                if i >= len(q):
                    q = q[:]
                    rng.shuffle(q)
                    per_speaker_queues[spk] = q
                    i = 0
                per_speaker_idx[spk] = i + 1
                return q[i]

            # Ensure each speaker appears at least once
            speaker_choices: List[int] = list(speakers)
            while len(speaker_choices) < chunks_per_track:
                speaker_choices.append(rng.choice(speakers))
            rng.shuffle(speaker_choices)
            speaker_choices = speaker_choices[:chunks_per_track]

            for spk in speaker_choices:
                plan.append((spk, next_wav(spk)))

        for j, (spk, wav_path) in enumerate(plan):
            seg = AudioSegment.from_wav(str(wav_path))
            seg = _ensure_audio_format(seg, target_frame_rate, target_channels, target_sample_width)

            if max_chunk_s is not None:
                max_ms = int(round(max_chunk_s * 1000))
                if len(seg) > max_ms:
                    seg = seg[:max_ms]

            start_s = cursor_ms / 1000.0
            out += seg
            cursor_ms += len(seg)
            end_s = cursor_ms / 1000.0
            labels.append(SegmentLabel(start_time=start_s, end_time=end_s, speaker=spk))

            # Add silence between chunks (not after last)
            if j != len(plan) - 1:
                out += silence_seg
                cursor_ms += silence_ms

        wav_out = dest_audio_dir / f"{stem}.wav"
        csv_out = dest_csv_dir / f"{stem}.csv"

        out.export(str(wav_out), format="wav")

        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_time", "end_time", "speaker"])
            for row in labels:
                w.writerow([f"{row.start_time:.6f}", f"{row.end_time:.6f}", str(row.speaker)])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Anonymize singer dirs and synthesize diarisation mixtures.")

    p.add_argument("--dir", required=True, type=Path, help="Directory containing singer subdirectories to map.")
    p.add_argument(
        "--mapping_json",
        type=Path,
        default=None,
        help="Mapping JSON path (default: <dest_dir>/singer_id_mapping.json).",
    )
    p.add_argument("--start_index", type=int, default=1, help="Starting index for id### numbering (default: 1).")
    p.add_argument(
        "--overwrite_mapping",
        action="store_true",
        help="Overwrite mapping JSON if it already exists.",
    )
    p.add_argument("--dest_dir", required=True, type=Path, help="Destination directory to create (audio/ and speaker_timestamps/).")
    p.add_argument("--num_tracks", required=True, type=int, help="Number of synthesized tracks to generate.")
    p.add_argument("--num_speakers", type=int, default=2, help="Number of singers per track (default: 2).")
    p.add_argument(
        "--chunks_per_track",
        type=int,
        default=None,
        help="Total chunks per track. If omitted, uses ALL chunks from each chosen singer exactly once.",
    )
    p.add_argument("--silence_s", type=float, default=0.5, help="Silence padding between chunks in seconds (default: 0.5).")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    p.add_argument("--max_chunk_s", type=float, default=None, help="Optional max duration per chunk in seconds (truncate if longer).")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    dest_dir = Path(args.dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    mapping_json = args.mapping_json
    if mapping_json is None:
        mapping_json = dest_dir / "singer_id_mapping.json"
    mapping_json = Path(mapping_json).resolve()

    create_anonymized_mapping(
        directory=args.dir,
        mapping_json_path=mapping_json,
        start_index=args.start_index,
        seed=args.seed,
        overwrite=args.overwrite_mapping,
    )

    singers_dir = Path(args.dir).resolve()

    if not mapping_json.exists():
        raise FileNotFoundError(
            f"Mapping JSON not found: {mapping_json}. "
            "Run the anonymize command first (it does not rename folders)."
        )
    # Mapping JSON is anon_id -> orig_name; invert to orig_name -> anon_id for synthesis.
    anon_to_orig = json.loads(mapping_json.read_text(encoding="utf-8"))
    mapping = {orig: anon for anon, orig in anon_to_orig.items()}

    synthesize_tracks(
        singers_dir=singers_dir,
        dest_dir=args.dest_dir,
        num_tracks=args.num_tracks,
        mapping=mapping,
        num_speakers=args.num_speakers,
        chunks_per_track=args.chunks_per_track,
        silence_s=args.silence_s,
        seed=args.seed,
        max_chunk_s=args.max_chunk_s,
    )




if __name__ == "__main__":
    main()

