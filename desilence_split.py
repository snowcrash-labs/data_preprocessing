#!/usr/bin/env python3
"""
Split audio files on silence and save segments.

Reads audio files from {dataset_path}/resampled_audio/, splits each on silence,
and saves segments to {dataset_path}/audio/<track_name>/<#####.wav>.
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
):
    dataset_path = Path(dataset_path)
    resampled_audio_dir = dataset_path / "audio"
    output_audio_dir = dataset_path / "audio"
    
    if not resampled_audio_dir.exists():
        logger.error(f"Resampled audio directory not found: {resampled_audio_dir}")
        return
    
    os.makedirs(output_audio_dir, exist_ok=True)
    logger.info(f"Input directory: {resampled_audio_dir}")
    logger.info(f"Output directory: {output_audio_dir}")
    
    # Collect all audio files
    audio_files = list(resampled_audio_dir.glob("*.wav"))
    logger.info(f"{len(audio_files)} audio files to process")
    
    if not audio_files:
        logger.warning("No audio files found to process")
        return

    # use all CPU cores
    num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} parallel processes")

    # process in a pool
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {
            exe.submit(
                split_audio_file,
                str(audio_file),
                str(output_audio_dir),
                min_silence_len,
                silence_thresh,
                keep_silence,
                min_segment_len,
            ): audio_file
            for audio_file in audio_files
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Splitting"):
            pass

    # Remove empty subdirectories
    logger.info("Checking for and removing empty subdirectories...")
    if output_audio_dir.exists():
        # Collect all subdirectories with their depth (walking bottom-up)
        all_dirs = []
        for root, dirs, files in os.walk(output_audio_dir, topdown=False):
            root_path = Path(root)
            # Skip the root directory itself
            if root_path != output_audio_dir:
                depth = len(root_path.relative_to(output_audio_dir).parts)
                all_dirs.append((depth, root_path))
        
        # Sort by depth descending (deepest first) to handle nested empty dirs
        all_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # Remove empty directories
        removed_count = 0
        for depth, dir_path in all_dirs:
            try:
                # Check if directory still exists and is empty
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

    logger.info("Done splitting audio.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Split audio files on silence and save segments"
    )
    p.add_argument(
        "--dataset_path", required=True, help="Path to dataset directory containing resampled_audio/"
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
    )

