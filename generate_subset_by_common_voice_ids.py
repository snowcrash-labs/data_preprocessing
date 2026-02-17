#!/usr/bin/env python3
"""
Create voice ID mapping between source and comparative datasets.

This script maps voice_ids between two datasets that share the same structure
(both containing audio/ subdirectory and singer_id_mapping_filtered.json).

Workflow:
1. Collects voice_ids from the source dataset's audio/ subdirectory
2. Matches them to the comparative dataset using 'lowercase' artist name lookups
3. Finds the INTERSECTION of song_ids that exist in both datasets per voice_id
4. Filters out songs where audio duration differs too much between datasets
5. Counts common tracks per voice_id and categorizes into range bins (2-5, 5-10, 10-30 songs)
6. Samples voice_ids proportionally across ranges (default: 70%, 20%, 10%)
7. Copies only the common song subdirectories from both datasets to a common output location

This ensures the two output datasets have the exact same singers, same songs per
singer, and similar audio duration per song.

Outputs (in between_datasets_voiceid_mapping/):
- org_hooktheory_to_new_hooktheory_voice_mapping.json: Full mapping of matched voice_ids
- unmatched_voice_ids.txt: Voice_ids that couldn't be matched
- sampled_voice_ids_report.json: Details of the sampled voice_ids with song counts
- org_dataset_common_data/: Copied source voice_id directories (common songs only)
- new_dataset_common_data/: Copied comparative voice_id directories (common songs only)
"""
import argparse
import json
import os
import random
import shutil
import sys
from typing import Dict, List, Set, Tuple, Union

import soundfile as sf


# =============================================================================
# Data Collection Functions
# =============================================================================

def collect_voice_ids_from_source(source_audio_dirpath: str) -> Dict[str, str]:
    """
    Collect voice_ids from subdirectories under source_audio_dirpath.
    
    Returns:
        Dict mapping voice_id -> subset name (e.g., "train", "test")
    """
    voice_id_to_subset = {}
    
    if not os.path.exists(source_audio_dirpath):
        print(f"Warning: Source path does not exist: {source_audio_dirpath}")
        return voice_id_to_subset
    
    for subset in os.listdir(source_audio_dirpath):
        subset_path = os.path.join(source_audio_dirpath, subset)
        if os.path.isdir(subset_path):
            for voice_id in os.listdir(subset_path):
                voice_id_path = os.path.join(subset_path, voice_id)
                if os.path.isdir(voice_id_path):
                    voice_id_to_subset[voice_id] = subset
    
    return voice_id_to_subset


def build_lowercase_to_id_lookup(singer_mapping: dict) -> Dict[str, str]:
    """
    Build a reverse lookup from lowercase artist name to new voice_id.
    
    Returns:
        Dict mapping lowercase artist name -> new_voice_id
    """
    lowercase_to_new_id = {}
    for new_id, data in singer_mapping.items():
        lowercase = data.get('lowercase', '')
        if lowercase:
            lowercase_to_new_id[lowercase] = new_id
    return lowercase_to_new_id


def build_new_id_path_lookup(new_audio_dir: str) -> Dict[str, str]:
    """
    Build a map of new_id -> full path by searching recursively under new_audio_dir.
    
    Returns:
        Dict mapping new_voice_id -> full directory path
    """
    new_id_to_path = {}
    if os.path.exists(new_audio_dir):
        for root, dirs, _ in os.walk(new_audio_dir):
            for d in dirs:
                if d.startswith('id'):
                    new_id_to_path[d] = os.path.join(root, d)
    return new_id_to_path


# =============================================================================
# Matching Functions
# =============================================================================

def match_voice_ids(
    voice_ids: Set[str],
    voice_id_to_subset: Dict[str, str],
    split_by_singer: dict,
    lowercase_to_new_id: Dict[str, str]
) -> Tuple[dict, List[str]]:
    """
    Match voice_ids from source dataset to comparative dataset using 'lowercase' artist names.
    
    Args:
        voice_ids: Set of voice_ids found in the source dataset
        voice_id_to_subset: Dict mapping voice_id -> subset name (e.g., "train", "test")
        split_by_singer: Source dataset's singer_id_mapping JSON (voice_id -> {lowercase: name})
        lowercase_to_new_id: Reverse lookup from lowercase name -> comparative dataset voice_id
    
    Returns:
        Tuple of:
        - matched: Dict mapping source voice_id -> {new_id, org_subset}
        - unmatched: List of tab-separated strings describing unmatched voice_ids
    """
    matched = {}
    unmatched = []
    
    for voice_id in sorted(voice_ids):
        subset = voice_id_to_subset.get(voice_id, 'unknown')
        
        if voice_id in split_by_singer:
            lowercase_name = split_by_singer[voice_id].get('lowercase', '')
            if lowercase_name in lowercase_to_new_id:
                matched[voice_id] = {
                    "new_id": lowercase_to_new_id[lowercase_name],
                    "org_subset": subset
                }
            else:
                unmatched.append(
                    f"{voice_id}\t{subset}\t{lowercase_name}\t(lowercase not found in singer_mapping)"
                )
        else:
            unmatched.append(f"{voice_id}\t{subset}\t(not found in split_by_singer.json)")
    
    return matched, unmatched


# =============================================================================
# Song Count & Sampling Functions
# =============================================================================

def count_songs_per_voice_id(
    matched: dict,
    source_audio_dirpath: str
) -> Dict[str, int]:
    """
    Count the number of track subdirectories for each matched voice_id.
    
    Assumes structure: source_audio_dirpath/subset/voice_id/track_subdir/...
    
    Args:
        matched: Dict mapping voice_id -> {new_id, org_subset}
        source_audio_dirpath: Path to source dataset's audio directory
    
    Returns:
        Dict mapping voice_id -> number of track subdirectories
    """
    voice_id_song_counts = {}
    
    for voice_id, info in matched.items():
        org_subset = info['org_subset']
        voice_id_path = os.path.join(source_audio_dirpath, org_subset, voice_id)
        
        if os.path.exists(voice_id_path):
            song_count = len([
                d for d in os.listdir(voice_id_path)
                if os.path.isdir(os.path.join(voice_id_path, d))
            ])
            voice_id_song_counts[voice_id] = song_count
    
    return voice_id_song_counts


def get_common_songs_per_voice_id(
    matched: dict,
    source_audio_dirpath: str,
    new_id_to_path: Dict[str, str],
) -> Dict[str, List[str]]:
    """
    For each matched voice_id, find song_ids present in BOTH datasets.

    Args:
        matched: Dict mapping source voice_id -> {new_id, org_subset}
        source_audio_dirpath: Path to source dataset's audio directory
        new_id_to_path: Dict mapping comparative voice_id -> full path

    Returns:
        Dict mapping source voice_id -> list of common song_ids
    """
    common_songs = {}
    for voice_id, info in matched.items():
        org_path = os.path.join(source_audio_dirpath, info['org_subset'], voice_id)
        new_path = new_id_to_path.get(info['new_id'])
        if not new_path:
            continue
        if not os.path.isdir(org_path) or not os.path.isdir(new_path):
            continue

        org_songs = {d for d in os.listdir(org_path)
                     if os.path.isdir(os.path.join(org_path, d))}
        new_songs = {d for d in os.listdir(new_path)
                     if os.path.isdir(os.path.join(new_path, d))}
        intersection = sorted(org_songs & new_songs)
        if intersection:
            common_songs[voice_id] = intersection
    return common_songs


def get_song_duration(song_dir: str) -> float:
    """Sum durations of all audio files in a song directory."""
    audio_exts = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
    total = 0.0
    for fname in os.listdir(song_dir):
        if os.path.splitext(fname)[1].lower() in audio_exts:
            try:
                info = sf.info(os.path.join(song_dir, fname))
                total += info.duration
            except Exception:
                pass
    return total


def filter_songs_by_duration(
    common_songs: Dict[str, List[str]],
    matched: dict,
    source_audio_dirpath: str,
    new_id_to_path: Dict[str, str],
    max_duration_ratio: float = 1.5,
) -> Dict[str, List[str]]:
    """
    Keep only songs where audio duration is within tolerance between datasets.

    A song is kept if the ratio of the longer side's duration to the shorter
    side's duration is <= max_duration_ratio. Songs with zero duration on
    either side are dropped.

    Args:
        common_songs: Dict mapping voice_id -> list of common song_ids
        matched: Dict mapping voice_id -> {new_id, org_subset}
        source_audio_dirpath: Path to source dataset's audio directory
        new_id_to_path: Dict mapping comparative voice_id -> full path
        max_duration_ratio: Maximum allowed ratio between the two durations

    Returns:
        Dict mapping voice_id -> filtered list of song_ids
    """
    filtered = {}
    total_checked = 0
    total_dropped = 0

    for voice_id, song_ids in common_songs.items():
        info = matched[voice_id]
        org_base = os.path.join(source_audio_dirpath, info['org_subset'], voice_id)
        new_base = new_id_to_path[info['new_id']]

        kept = []
        for song_id in song_ids:
            total_checked += 1
            org_dur = get_song_duration(os.path.join(org_base, song_id))
            new_dur = get_song_duration(os.path.join(new_base, song_id))
            if org_dur == 0 or new_dur == 0:
                total_dropped += 1
                continue
            ratio = max(org_dur, new_dur) / min(org_dur, new_dur)
            if ratio <= max_duration_ratio:
                kept.append(song_id)
            else:
                total_dropped += 1
        if kept:
            filtered[voice_id] = kept

    print(f"  Duration filter: checked {total_checked} songs, "
          f"dropped {total_dropped}, kept {total_checked - total_dropped}")
    return filtered


def categorize_by_song_count(
    voice_id_song_counts: Dict[str, int],
    ranges: List[tuple]
) -> Dict[str, List[str]]:
    """
    Categorize voice_ids into range bins based on song/track count.
    
    Args:
        voice_id_song_counts: Dict mapping voice_id -> number of tracks
        ranges: List of (min_count, max_count, description) tuples
    
    Returns:
        Dict mapping range description -> list of voice_ids in that range
    """
    voice_ids_by_range = {desc: [] for _, _, desc in ranges}
    
    for voice_id, song_count in voice_id_song_counts.items():
        for min_songs, max_songs, desc in ranges:
            if max_songs == float('inf'):
                if song_count >= min_songs:
                    voice_ids_by_range[desc].append(voice_id)
                    break
            else:
                if min_songs <= song_count < max_songs:
                    voice_ids_by_range[desc].append(voice_id)
                    break
    
    return voice_ids_by_range


def sample_proportionally_across_ranges(
    voice_ids_by_range: Dict[str, List[str]],
    voice_id_song_counts: Dict[str, int],
    ranges: List[tuple],
    proportions: List[float],
    total_sample: int = 1000
) -> Tuple[List[str], List[dict]]:
    """
    Sample voice_ids proportionally across ranges.
    
    Args:
        voice_ids_by_range: Dict mapping range description to list of voice_ids
        voice_id_song_counts: Dict mapping voice_id to song count
        ranges: List of (min, max, description) tuples
        proportions: List of proportions for each range (should sum to 1.0)
        total_sample: Total number of samples to collect. 1000 is a good size as we can do reduced analysis on this if necessary after.
    
    Returns:
        Tuple of (sampled voice_ids list, sampled info list with metadata)
    """
    sampled_voice_ids = []
    sampled_voice_id_info = []
    
    for i, (_, _, desc) in enumerate(ranges):
        available = voice_ids_by_range[desc]
        target_count = int(total_sample * proportions[i])
        
        if len(available) >= target_count:
            sampled = random.sample(available, target_count)
        else:
            sampled = available
            print(f"  Warning: Only {len(sampled)} voice_ids available for {desc}, taking all (target: {target_count})")
        
        sampled_voice_ids.extend(sampled)
        
        for vid in sampled:
            sampled_voice_id_info.append({
                "voice_id": vid,
                "song_count": voice_id_song_counts[vid],
                "range_bin": desc
            })
        
        print(f"  Sampled {len(sampled)} voice_ids from {desc} (target: {target_count})")
    
    return sampled_voice_ids, sampled_voice_id_info


# =============================================================================
# File Copy Functions
# =============================================================================

def copy_common_songs(
    src_voice_dir: str,
    dst_voice_dir: str,
    song_ids: List[str],
) -> int:
    """
    Copy only the specified song subdirectories from src to dst.

    Args:
        src_voice_dir: Source voice_id directory path
        dst_voice_dir: Destination voice_id directory path
        song_ids: List of song_id subdirectories to copy

    Returns:
        Number of song directories actually copied
    """
    os.makedirs(dst_voice_dir, exist_ok=True)
    copied = 0
    for song_id in song_ids:
        src = os.path.join(src_voice_dir, song_id)
        dst = os.path.join(dst_voice_dir, song_id)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
            copied += 1
    return copied


def copy_with_progress(
    sampled_voice_ids: List[str],
    common_songs: Dict[str, List[str]],
    matched: dict,
    source_audio_dirpath: str,
    new_id_to_path: Dict[str, str],
    org_common_dir: str,
    new_common_dir: str
) -> Tuple[int, int]:
    """
    Selectively copy only common song directories from both datasets.

    For each sampled voice_id, copies only the song subdirectories that
    exist in both datasets (from common_songs) to ensure identical content.

    Args:
        sampled_voice_ids: List of source voice_ids to copy
        common_songs: Dict mapping voice_id -> list of common song_ids
        matched: Dict mapping source voice_id -> {new_id, org_subset}
        source_audio_dirpath: Path to source dataset's audio directory
        new_id_to_path: Dict mapping comparative voice_id -> full path
        org_common_dir: Output directory for source voice_id copies
        new_common_dir: Output directory for comparative voice_id copies

    Returns:
        Tuple of (source songs copied, comparative songs copied)
    """
    total = len(sampled_voice_ids)
    copied_org_songs = 0
    copied_new_songs = 0

    print(f"\nCopying {total} voice_id pairs (common songs only)...")

    for i, voice_id in enumerate(sampled_voice_ids, 1):
        info = matched[voice_id]
        new_id = info['new_id']
        org_subset = info['org_subset']
        song_ids = common_songs.get(voice_id, [])

        # Copy common songs from source dataset
        org_src = os.path.join(source_audio_dirpath, org_subset, voice_id)
        org_dst = os.path.join(org_common_dir, voice_id)
        n_org = copy_common_songs(org_src, org_dst, song_ids)
        copied_org_songs += n_org

        # Copy common songs from comparative dataset
        n_new = 0
        if new_id in new_id_to_path:
            new_src = new_id_to_path[new_id]
            new_dst = os.path.join(new_common_dir, new_id)
            n_new = copy_common_songs(new_src, new_dst, song_ids)
            copied_new_songs += n_new

        # Progress indicator
        progress = (i / total) * 100
        sys.stdout.write(
            f"\r  [{i}/{total}] {progress:.1f}% - {voice_id} -> {new_id} "
            f"({len(song_ids)} songs, copied org:{n_org} new:{n_new})"
        )
        sys.stdout.flush()

    print()  # Newline after progress
    return copied_org_songs, copied_new_songs


# =============================================================================
# I/O Functions
# =============================================================================

def save_json(data: Union[dict, list], path: str, description: str):
    """Save data to JSON file with logging."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {description} to {path}")


def save_text(lines: List[str], path: str, description: str):
    """Save lines to text file with logging."""
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {description} to {path}")


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Create voice ID mapping between datasets'
    )
    parser.add_argument(
        '--mapping-dir',
        type=str,
        required=True,
        help='Directory where between_datasets_voiceid_mapping is set up (outputs and common data subdirs)'
    )
    parser.add_argument(
        '--source-dataset-path', type=str,
        default='/mnt/data/gs_imports/hooktheory_demucs_16khz_brens_generated',
        help='Path to source dataset directory (contains audio/ and singer_id_mapping_filtered.json)'
    )
    parser.add_argument(
        '--comparative-dataset-path', type=str,
        default='/mnt/data/gs_imports/hooktheory_roformered_32khz',
        help='Path to comparative dataset directory (contains audio/ and singer_id_mapping_filtered.json)'
    )
    parser.add_argument(
        '--max-duration-ratio', type=float, default=1.5,
        help='Max allowed duration ratio between datasets for a song to be kept (default: 1.5)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
    source_dataset_path = args.source_dataset_path
    comparative_dataset_path = args.comparative_dataset_path
    
    # Both datasets share the same singer_id_mapping filename
    singer_id_map = "singer_id_mapping_filtered.json"
    
    # JSON paths
    split_by_singer_path = os.path.join(source_dataset_path, singer_id_map)
    singer_mapping_path = os.path.join(comparative_dataset_path, singer_id_map)
    
    # Audio directories
    source_audio_dir = os.path.join(source_dataset_path, 'audio')
    comparative_audio_dir = os.path.join(comparative_dataset_path, 'audio')
    
    output_dir = args.mapping_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_mapping_path = os.path.join(output_dir, 'org_hooktheory_to_new_hooktheory_voice_mapping.json')
    unmatched_path = os.path.join(output_dir, 'unmatched_voice_ids.txt')
    sampled_report_path = os.path.join(output_dir, 'sampled_voice_ids_report.json')
    org_common_dir = os.path.join(output_dir, 'org_dataset_common_data')
    new_common_dir = os.path.join(output_dir, 'new_dataset_common_data')
    
    # Step 1: Collect voice_ids from source
    print("Step 1: Collecting voice_ids from source dataset...")
    voice_id_to_subset = collect_voice_ids_from_source(source_audio_dir)
    voice_ids = set(voice_id_to_subset.keys())
    print(f"  Found {len(voice_ids)} voice_ids across {len(set(voice_id_to_subset.values()))} subsets")
    
    # Step 2: Load JSON files
    print("Step 2: Loading JSON files...")
    split_by_singer = load_json(split_by_singer_path)
    singer_mapping = load_json(singer_mapping_path)
    
    # Step 3: Build lookups
    print("Step 3: Building lookups...")
    lowercase_to_new_id = build_lowercase_to_id_lookup(singer_mapping)
    print(f"  Built lowercase->id lookup with {len(lowercase_to_new_id)} entries")
    
    new_id_to_path = build_new_id_path_lookup(comparative_audio_dir)
    print(f"  Found {len(new_id_to_path)} new voice_id directories under {comparative_audio_dir}")
    
    # Step 4: Match voice_ids
    print("Step 4: Matching voice_ids...")
    matched, unmatched = match_voice_ids(
        voice_ids, voice_id_to_subset, split_by_singer, lowercase_to_new_id
    )
    print(f"  Matched: {len(matched)}, Unmatched: {len(unmatched)}")
    
    # Step 5: Save matching results
    print("Step 5: Saving matching results...")
    save_json(matched, output_mapping_path, "voice ID mapping")
    if unmatched:
        save_text(unmatched, unmatched_path, f"{len(unmatched)} unmatched voice_ids")
    
    # Step 6: Find common songs between both datasets
    print("Step 6: Finding common songs between datasets...")
    common_songs = get_common_songs_per_voice_id(matched, source_audio_dir, new_id_to_path)
    total_common = sum(len(s) for s in common_songs.values())
    print(f"  Found {total_common} common songs across {len(common_songs)} voice_ids")

    # Step 6b: Filter by audio duration similarity
    print(f"Step 6b: Filtering songs by duration ratio (max {args.max_duration_ratio}x)...")
    common_songs = filter_songs_by_duration(
        common_songs, matched, source_audio_dir, new_id_to_path,
        max_duration_ratio=args.max_duration_ratio,
    )
    total_after_filter = sum(len(s) for s in common_songs.values())
    print(f"  After duration filter: {total_after_filter} songs across {len(common_songs)} voice_ids")

    # Build song counts from common songs (replaces source-only counting)
    voice_id_song_counts = {vid: len(songs) for vid, songs in common_songs.items()}

    # Define song count ranges (ignoring "1 song")
    ranges = [
        (2, 5, "2-5 songs"),
        (5, 10, "5-10 songs"),
        (10, 30, "10-30 songs"),
    ]

    voice_ids_by_range = categorize_by_song_count(voice_id_song_counts, ranges)

    print("  Distribution by common song count:")
    for desc, ids in voice_ids_by_range.items():
        print(f"    {desc}: {len(ids)} voice_ids")

    # Step 7: Sample proportionally across ranges (70%, 20%, 10%)
    print("Step 7: Sampling voice_ids proportionally across ranges...")
    proportions = [0.70, 0.20, 0.10]  # 70% for 2-5 songs, 20% for 5-10 songs, 10% for 10-30 songs
    sampled_voice_ids, sampled_voice_id_info = sample_proportionally_across_ranges(
        voice_ids_by_range, voice_id_song_counts, ranges, proportions, total_sample=1000
    )
    print(f"  Total sampled: {len(sampled_voice_ids)} voice_ids")

    save_json(sampled_voice_id_info, sampled_report_path, "sampled voice_ids report")

    # Step 8: Copy only common song directories
    print("Step 8: Copying common song directories...")
    os.makedirs(org_common_dir, exist_ok=True)
    os.makedirs(new_common_dir, exist_ok=True)

    copied_org, copied_new = copy_with_progress(
        sampled_voice_ids, common_songs, matched, source_audio_dir,
        new_id_to_path, org_common_dir, new_common_dir
    )

    # Summary
    total_songs_copied = sum(
        len(common_songs.get(vid, [])) for vid in sampled_voice_ids
    )
    print(f"\nDone!")
    print(f"  Copied {copied_org} org song directories to {org_common_dir}")
    print(f"  Copied {copied_new} new song directories to {new_common_dir}")
    print(f"  Total common songs across sampled voice_ids: {total_songs_copied}")
    print(f"  Both datasets now contain the exact same singers and songs")


if __name__ == '__main__':
    main()
