#!/usr/bin/env python3
"""
Calculate the combined duration of all audio files in a directory (recursively).

Requirements:
    pip install soundfile librosa
    
For mp3 support, you also need ffmpeg installed:
    sudo apt install ffmpeg  # Linux
    brew install ffmpeg      # macOS
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

import soundfile as sf
import librosa


# Supported audio extensions
AUDIO_EXTENSIONS = {
    '.wav', '.flac', '.ogg', '.aiff', '.aif', '.mp3', '.m4a', '.aac', '.opus'
}


def get_audio_duration(filepath: str) -> Optional[float]:
    """Get the duration of an audio file in seconds."""
    ext = os.path.splitext(filepath)[1].lower()
    
    # Use soundfile for formats it handles well (faster)
    if ext in {'.wav', '.flac', '.ogg', '.aiff', '.aif'}:
        try:
            info = sf.info(filepath)
            return info.duration
        except Exception:
            pass
    
    # Use librosa for mp3 and other formats (uses ffmpeg/audioread backend)
    try:
        duration = librosa.get_duration(path=filepath)
        return duration
    except Exception:
        return None


def find_audio_files(directory: str) -> List[str]:
    """Recursively find all audio files in a directory."""
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(root, filename))
    
    return audio_files


def get_voice_id_from_path(filepath: str, base_dir: str) -> Optional[str]:
    """
    Extract the voice_id from a file path.
    Assumes structure: base_dir/voice_id/[subdirs]/file.wav
    """
    rel_path = os.path.relpath(filepath, base_dir)
    parts = rel_path.split(os.sep)
    if len(parts) >= 1:
        return parts[0]
    return None


def count_tracks_per_voice_id(directory: str) -> Dict[str, int]:
    """
    Count the number of track subdirectories per voice_id.
    Assumes structure: directory/voice_id/track_subdir/...
    """
    voice_id_track_counts = {}
    
    if not os.path.exists(directory):
        return voice_id_track_counts
    
    for voice_id in os.listdir(directory):
        voice_id_path = os.path.join(directory, voice_id)
        if os.path.isdir(voice_id_path):
            # Count immediate subdirectories (tracks)
            track_count = len([
                d for d in os.listdir(voice_id_path)
                if os.path.isdir(os.path.join(voice_id_path, d))
            ])
            voice_id_track_counts[voice_id] = track_count
    
    return voice_id_track_counts


def get_range_bin(track_count: int) -> str:
    """Categorize track count into a range bin."""
    ranges = [
        (1, 2, "1 track"),
        (2, 5, "2-5 tracks"),
        (5, 10, "5-10 tracks"),
        (10, 30, "10-30 tracks"),
        (30, 100, "30-100 tracks"),
        (100, float('inf'), "100+ tracks")
    ]
    
    for min_tracks, max_tracks, desc in ranges:
        if max_tracks == float('inf'):
            if track_count >= min_tracks:
                return desc
        else:
            if min_tracks <= track_count < max_tracks:
                return desc
    
    return "unknown"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.2f}s"


def main():
    parser = argparse.ArgumentParser(
        description='Calculate combined duration of all audio files in a directory'
    )
    parser.add_argument(
        '--directory', '-d',
        type=str,
        help='Directory to search for audio files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show duration of each file'
    )
    parser.add_argument(
        '--by-extension',
        action='store_true',
        help='Show breakdown by file extension'
    )
    parser.add_argument(
        '--by-voice-range',
        action='store_true',
        help='Show breakdown by voice_id track count ranges (assumes directory/voice_id/track/... structure)'
    )
    args = parser.parse_args()
    
    directory = args.directory
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory")
        sys.exit(1)
    
    print(f"Searching for audio files in: {directory}")
    audio_files = find_audio_files(directory)
    print(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("No audio files found.")
        return
    
    total_duration = 0.0
    successful = 0
    failed = 0
    duration_by_ext = {}  # type: Dict[str, float]
    count_by_ext = {}  # type: Dict[str, int]
    duration_by_voice_id = {}  # type: Dict[str, float]
    count_by_voice_id = {}  # type: Dict[str, int]
    
    # Pre-compute voice_id track counts if needed
    voice_id_track_counts = {}  # type: Dict[str, int]
    if args.by_voice_range:
        voice_id_track_counts = count_tracks_per_voice_id(directory)
        print(f"Found {len(voice_id_track_counts)} voice_ids")
    
    for i, filepath in enumerate(audio_files, 1):
        duration = get_audio_duration(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        
        if duration is not None:
            total_duration += duration
            successful += 1
            
            # Track by extension
            duration_by_ext[ext] = duration_by_ext.get(ext, 0) + duration
            count_by_ext[ext] = count_by_ext.get(ext, 0) + 1
            
            # Track by voice_id
            if args.by_voice_range:
                voice_id = get_voice_id_from_path(filepath, directory)
                if voice_id:
                    duration_by_voice_id[voice_id] = duration_by_voice_id.get(voice_id, 0) + duration
                    count_by_voice_id[voice_id] = count_by_voice_id.get(voice_id, 0) + 1
            
            if args.verbose:
                print(f"  [{i}/{len(audio_files)}] {format_duration(duration):>12} - {filepath}")
        else:
            failed += 1
            if args.verbose:
                print(f"  [{i}/{len(audio_files)}] {'FAILED':>12} - {filepath}")
        
        # Progress indicator (every 100 files or at end)
        if not args.verbose and (i % 100 == 0 or i == len(audio_files)):
            sys.stdout.write(f"\r  Processing: {i}/{len(audio_files)} files...")
            sys.stdout.flush()
    
    if not args.verbose:
        print()  # Newline after progress
    
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {successful}")
    if failed > 0:
        print(f"Failed to read: {failed}")
    print(f"Total duration: {format_duration(total_duration)}")
    print(f"               ({total_duration:.2f} seconds)")
    print(f"               ({total_duration / 3600:.2f} hours)")
    
    if args.by_extension and duration_by_ext:
        print()
        print("Breakdown by extension:")
        for ext in sorted(duration_by_ext.keys()):
            dur = duration_by_ext[ext]
            count = count_by_ext[ext]
            print(f"  {ext:8} : {count:6} files, {format_duration(dur):>15} ({dur/3600:.2f} hours)")
    
    if args.by_voice_range and duration_by_voice_id:
        # Aggregate durations by range bin
        duration_by_range = {}  # type: Dict[str, float]
        count_by_range = {}  # type: Dict[str, int]
        voice_count_by_range = {}  # type: Dict[str, int]
        
        for voice_id, dur in duration_by_voice_id.items():
            track_count = voice_id_track_counts.get(voice_id, 0)
            range_bin = get_range_bin(track_count)
            
            duration_by_range[range_bin] = duration_by_range.get(range_bin, 0) + dur
            count_by_range[range_bin] = count_by_range.get(range_bin, 0) + count_by_voice_id.get(voice_id, 0)
            voice_count_by_range[range_bin] = voice_count_by_range.get(range_bin, 0) + 1
        
        # Define order for range bins
        range_order = [
            "1 track",
            "2-5 tracks",
            "5-10 tracks",
            "10-30 tracks",
            "30-100 tracks",
            "100+ tracks",
            "unknown"
        ]
        
        print()
        print("Breakdown by voice track count range:")
        print(f"  {'Range':<15} {'Voices':>8} {'Files':>10} {'Duration':>18} {'Hours':>10}")
        print("  " + "-" * 65)
        
        for range_bin in range_order:
            if range_bin in duration_by_range:
                dur = duration_by_range[range_bin]
                file_count = count_by_range[range_bin]
                voice_count = voice_count_by_range[range_bin]
                print(f"  {range_bin:<15} {voice_count:>8} {file_count:>10} {format_duration(dur):>18} {dur/3600:>10.2f}")


if __name__ == '__main__':
    main()
