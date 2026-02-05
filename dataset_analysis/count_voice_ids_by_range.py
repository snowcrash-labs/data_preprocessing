#!/usr/bin/env python3
"""
Count voice_ids by track count range bins.
"""
import argparse
import os
from typing import Dict, List


def count_tracks_per_voice_id(audio_dir: str) -> Dict[str, int]:
    """
    Count the number of track subdirectories per voice_id.
    Assumes structure: audio_dir/subset/voice_id/track_subdir/...
    """
    voice_id_track_counts = {}
    
    if not os.path.exists(audio_dir):
        print(f"Warning: Directory does not exist: {audio_dir}")
        return voice_id_track_counts
    
    for subset in os.listdir(audio_dir):
        subset_path = os.path.join(audio_dir, subset)
        if os.path.isdir(subset_path):
            for voice_id in os.listdir(subset_path):
                voice_id_path = os.path.join(subset_path, voice_id)
                if os.path.isdir(voice_id_path):
                    # Count immediate subdirectories (tracks)
                    track_count = len([
                        d for d in os.listdir(voice_id_path)
                        if os.path.isdir(os.path.join(voice_id_path, d))
                    ])
                    voice_id_track_counts[voice_id] = track_count
    
    return voice_id_track_counts


def categorize_by_range(voice_id_track_counts: Dict[str, int]) -> Dict[str, List[str]]:
    """Categorize voice_ids into range bins."""
    ranges = [
        (2, 5, "2-5 tracks"),
        (5, 10, "5-10 tracks"),
        (10, 30, "10-30 tracks"),
    ]
    
    voice_ids_by_range = {desc: [] for _, _, desc in ranges}
    
    for voice_id, track_count in voice_id_track_counts.items():
        for min_tracks, max_tracks, desc in ranges:
            if max_tracks == float('inf'):
                if track_count >= min_tracks:
                    voice_ids_by_range[desc].append(voice_id)
                    break
            else:
                if min_tracks <= track_count < max_tracks:
                    voice_ids_by_range[desc].append(voice_id)
                    break
    
    return voice_ids_by_range


def main():
    parser = argparse.ArgumentParser(
        description='Count voice_ids by track count range bins'
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to dataset directory (should contain audio/ subdirectory)'
    )
    parser.add_argument(
        '--audio-subdir',
        type=str,
        default='audio',
        help='Name of the audio subdirectory (default: audio)'
    )
    args = parser.parse_args()
    
    audio_dir = os.path.join(args.dataset_path, args.audio_subdir)
    
    print(f"Scanning: {audio_dir}")
    print()
    
    voice_id_track_counts = count_tracks_per_voice_id(audio_dir)
    print(f"Found {len(voice_id_track_counts)} total voice_ids")
    print()
    
    voice_ids_by_range = categorize_by_range(voice_id_track_counts)
    
    # Define order for display
    range_order = [
        "2-5 tracks",
        "5-10 tracks",
        "10-30 tracks",
    ]
    
    print("Voice ID distribution by track count:")
    print("-" * 40)
    print(f"{'Range':<20} {'Count':>10}")
    print("-" * 40)
    
    total = 0
    for range_bin in range_order:
        count = len(voice_ids_by_range.get(range_bin, []))
        total += count
        print(f"{range_bin:<20} {count:>10}")
    
    print("-" * 40)
    print(f"{'Total':<20} {total:>10}")


if __name__ == '__main__':
    main()
