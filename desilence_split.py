#!/usr/bin/env python3
"""
Split audio files on silence and save segments.

Depth 1 (--wav_depth): Reads .wav files directly under {dataset_path}/audio/,
splits each on silence, saves segments to {dataset_path}/audio/<track_name>/<#####.wav>.

Depth 2 (--wav_depth): Reads {dataset_path}/audio/<voice_id>/<song_id>.wav for each
voice_id subdir. Splits each file and saves to {dataset_path}/audio/<voice_id>/<song_id>/<#####.wav>.
Result: dataset/audio/voice_id/track_id/chunks.
"""
import argparse
import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("analyse_split_audio.log")],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def split_audio_file(
    audio_path: str,
    output_base_dir: str,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    min_segment_len: int,
) -> None:
    """
    - Loads audio file
    - Splits on silence
    - Saves each valid segment to output_base_dir/<track_name>/<#####.wav>
    """
    audio_path = Path(audio_path)
    track_name = audio_path.stem
    out_base = Path(output_base_dir) / track_name
    
    # Skip if directory already exists (already processed)
    if out_base.exists() and out_base.is_dir():
        logger.info(f"[{track_name}] already processed, skipping")
        return
    
    out_base.mkdir(parents=True, exist_ok=True)

    try:
        # Load audio
        audio = AudioSegment.from_wav(str(audio_path))
        
        # Split on silence
        segments = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )
        valid = [seg for seg in segments if len(seg) >= min_segment_len]
        logger.info(f"[{track_name}] found {len(valid)} ≥{min_segment_len} ms segments")

        # Export segments
        for idx, seg in enumerate(valid, start=1):
            fname = f"{idx:05d}.wav"
            seg.export(str(out_base / fname), format="wav")
            
    except Exception as e:
        logger.error(f"[{track_name}] processing failed: {e}")
        return


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    dataset_path: str,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    min_segment_len: int,
    wav_depth: int,
):
    dataset_path = Path(dataset_path)
    dataset_audio_dir = dataset_path / "audio"

    if not dataset_audio_dir.exists():
        logger.error(f"Audio directory not found: {dataset_audio_dir}")
        return

    os.makedirs(dataset_audio_dir, exist_ok=True)
    logger.info(f"Input directory: {dataset_audio_dir} (wav_depth={wav_depth})")

    # Collect (audio_path, output_base_dir) for each file to process
    if wav_depth == 2:
        # Depth 2: dataset/audio/voice_id/<song_id>.wav -> output under dataset/audio/voice_id/<song_id>/
        tasks = []
        for voice_dir in sorted(dataset_audio_dir.iterdir()):
            if not voice_dir.is_dir():
                continue
            for wav in voice_dir.glob("*.wav"):
                tasks.append((wav, voice_dir))
        logger.info(f"{len(tasks)} audio files to process (depth 2: voice_id/song_id)")
    else:
        # Depth 1: dataset/audio/*.wav -> output under dataset/audio/<track_name>/
        audio_files = list(dataset_audio_dir.glob("*.wav"))
        tasks = [(wav, dataset_audio_dir) for wav in audio_files]
        logger.info(f"{len(tasks)} audio files to process (depth 1)")

    if not tasks:
        logger.warning("No audio files found to process")
        return

    num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} parallel processes")

    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {
            exe.submit(
                split_audio_file,
                str(audio_path),
                str(output_base_dir),
                min_silence_len,
                silence_thresh,
                keep_silence,
                min_segment_len,
            ): audio_path
            for audio_path, output_base_dir in tasks
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Splitting"):
            pass

    # Remove empty subdirectories (bottom-up)
    logger.info("Checking for and removing empty subdirectories...")
    if dataset_audio_dir.exists():
        all_dirs = []
        for root, dirs, files in os.walk(dataset_audio_dir, topdown=False):
            root_path = Path(root)
            if root_path != dataset_audio_dir:
                depth = len(root_path.relative_to(dataset_audio_dir).parts)
                all_dirs.append((depth, root_path))
        all_dirs.sort(key=lambda x: x[0], reverse=True)
        removed_count = 0
        for depth, dir_path in all_dirs:
            try:
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    removed_count += 1
                    logger.debug(f"Removed empty directory: {dir_path}")
            except OSError as e:
                logger.warning(f"Could not remove {dir_path}: {e}")
        if removed_count > 0:
            logger.info(f"Removed {removed_count} empty subdirectories")
        else:
            logger.info("No empty subdirectories found")

    # Remove original wav files (depth 1: at audio/ root; depth 2: inside each voice_id dir)
    logger.info("Removing original wav files...")
    if wav_depth == 1:
        original_wavs = [
            w for v in dataset_audio_dir.iterdir()
            if v.is_dir() for w in v.glob("*.wav")
        ]
    else:
        original_wavs = list(dataset_audio_dir.glob("*.wav"))
    removed_wav_count = 0
    for wav_file in original_wavs:
        try:
            wav_file.unlink()
            removed_wav_count += 1
            logger.debug(f"Removed original wav: {wav_file.name}")
        except OSError as e:
            logger.warning(f"Could not remove {wav_file}: {e}")
    if removed_wav_count > 0:
        logger.info(f"Removed {removed_wav_count} original wav files")
    else:
        logger.info("No original wav files to remove")

    logger.info("Done splitting audio.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Split audio files on silence and save segments"
    )
    p.add_argument(
        "--dataset_path", required=True, help="Path to dataset directory containing audio/"
    )
    p.add_argument(
        "--wav_depth",
        type=int,
        default=1,
        help="Depth of the wav directory. Default is 1: audio/*.wav -> audio/<track_name>/chunks. 2: dataset/audio/<voice_id>/<song_id>.wav -> dataset/audio/<voice_id>/<song_id>/chunks.",
    )
    p.add_argument(
        "--min_silence_len", type=int, default=2000, help="ms of silence to split on"
    )
    p.add_argument(
        "--silence_thresh", type=int, default=-40, help="dBFS threshold for silence"
    )
    p.add_argument(
        "--keep_silence", type=int, default=100, help="ms of silence to leave at edges"
    )
    p.add_argument(
        "--min_segment_len", type=int, default=3000, help="ms minimum segment length"
    )
    args = p.parse_args()

    main(
        dataset_path=args.dataset_path,
        min_silence_len=args.min_silence_len,
        silence_thresh=args.silence_thresh,
        keep_silence=args.keep_silence,
        min_segment_len=args.min_segment_len,
        wav_depth=args.wav_depth,
    )

