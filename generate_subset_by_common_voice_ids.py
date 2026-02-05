#!/usr/bin/env python3
"""
Create voice ID mapping between source and comparative datasets.

This script maps voice_ids between two datasets that share the same structure
(both containing audio/ subdirectory and singer_id_mapping_filtered.json).

Workflow:
1. Collects voice_ids from the source dataset's audio/ subdirectory
2. Matches them to the comparative dataset using 'lowercase' artist name lookups
3. Counts tracks per voice_id and categorizes into range bins (2-5, 5-10, 10-30 songs)
4. Samples voice_ids proportionally across ranges (default: 70%, 20%, 10%) these were empiracally choosen to match the expected distributions
5. Copies matched voice_id directories from both datasets to a common output location

Outputs (in between_datasets_voiceid_mapping/):
- org_hooktheory_to_new_hooktheory_voice_mapping.json: Full mapping of matched voice_ids
- unmatched_voice_ids.txt: Voice_ids that couldn't be matched
- sampled_voice_ids_report.json: Details of the sampled voice_ids with song counts
- org_dataset_common_data/: Copied source voice_id directories
- new_dataset_common_data/: Copied comparative voice_id directories
"""
import argparse
import json
import os
import random
import shutil
import sys
from typing import Dict, List, Set, Tuple, Union


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

def copy_with_progress(
    sampled_voice_ids: List[str],
    matched: dict,
    source_audio_dirpath: str,
    new_id_to_path: Dict[str, str],
    org_common_dir: str,
    new_common_dir: str
) -> Tuple[int, int]:
    """
    Copy voice_id directories from both datasets with live progress reporting.
    
    For each sampled voice_id, copies:
    - The source dataset's voice_id directory to org_common_dir
    - The corresponding comparative dataset's voice_id directory to new_common_dir
    
    Args:
        sampled_voice_ids: List of source voice_ids to copy
        matched: Dict mapping source voice_id -> {new_id, org_subset}
        source_audio_dirpath: Path to source dataset's audio directory
        new_id_to_path: Dict mapping comparative voice_id -> full path
        org_common_dir: Output directory for source voice_id copies
        new_common_dir: Output directory for comparative voice_id copies
    
    Returns:
        Tuple of (number of source dirs copied, number of comparative dirs copied)
    """
    total = len(sampled_voice_ids)
    copied_org = 0
    copied_new = 0
    
    print(f"\nCopying {total} voice_id pairs...")
    
    for i, voice_id in enumerate(sampled_voice_ids, 1):
        info = matched[voice_id]
        new_id = info['new_id']
        org_subset = info['org_subset']
        
        # Copy from source dataset (org)
        org_src = os.path.join(source_audio_dirpath, org_subset, voice_id)
        org_dst = os.path.join(org_common_dir, voice_id)
        org_copied_this = False
        
        if os.path.exists(org_src) and not os.path.exists(org_dst):
            shutil.copytree(org_src, org_dst)
            copied_org += 1
            org_copied_this = True
        
        # Copy from new dataset
        new_copied_this = False
        if new_id in new_id_to_path:
            new_src = new_id_to_path[new_id]
            new_dst = os.path.join(new_common_dir, new_id)
            if not os.path.exists(new_dst):
                shutil.copytree(new_src, new_dst)
                copied_new += 1
                new_copied_this = True
        
        # Progress indicator
        progress = (i / total) * 100
        status = []
        if org_copied_this:
            status.append("org")
        if new_copied_this:
            status.append("new")
        status_str = f"copied: {'+'.join(status)}" if status else "skipped (exists)"
        
        sys.stdout.write(f"\r  [{i}/{total}] {progress:.1f}% - {voice_id} -> {new_id} ({status_str})")
        sys.stdout.flush()
    
    print()  # Newline after progress
    return copied_org, copied_new


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
    
    # Step 6: Count songs and categorize
    print("Step 6: Counting songs per voice_id...")
    voice_id_song_counts = count_songs_per_voice_id(matched, source_audio_dir)
    print(f"  Counted songs for {len(voice_id_song_counts)} voice_ids")
    
    # Define song count ranges (ignoring "1 song")
    ranges = [
        (2, 5, "2-5 songs"),
        (5, 10, "5-10 songs"),
        (10, 30, "10-30 songs"),
    ]
    
    voice_ids_by_range = categorize_by_song_count(voice_id_song_counts, ranges)
    
    print("  Distribution by song count:")
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
    
    # Step 8: Copy directories
    print("Step 8: Copying voice_id directories...")
    os.makedirs(org_common_dir, exist_ok=True)
    os.makedirs(new_common_dir, exist_ok=True)
    
    copied_org, copied_new = copy_with_progress(
        sampled_voice_ids, matched, source_audio_dir,
        new_id_to_path, org_common_dir, new_common_dir
    )
    
    print(f"\nDone!")
    print(f"  Copied {copied_org} org voice_id directories to {org_common_dir}")
    print(f"  Copied {copied_new} new voice_id directories to {new_common_dir}")


if __name__ == '__main__':
    main()
