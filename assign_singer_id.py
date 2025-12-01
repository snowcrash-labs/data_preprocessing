"""
Assign unique singer IDs to artists and filter invalid entries.

Reads deduplicated_data.csv, filters out invalid artist names (orchestras, DJs, etc.),
assigns unique IDs (id00001, id00002, ...) to each artist, and adds singer_id column
to the CSV. Creates singer_id_mapping_filtered.json mapping file.
"""
import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
import shutil
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Assign singer IDs to artists in the deduplicated dataset"
)
parser.add_argument(
    "--dataset_path",
    required=True,
    help="Path to the dataset directory (e.g. ~/gs_imports/roformer_voice_sep_custom_sample)",
)
parser.add_argument(
    "--artist_name_header",
    required=True,
    help="Name of the CSV column header containing artist names",
)
parser.add_argument(
    "--singer_id_mapping_json",
    type=str,
    default=None,
    help="Optional path to JSON file with pre-existing singer ID mappings. JSON should have singer_id keys with 'lowercase' and 'variations' nested dicts. If provided, uses this mapping instead of generating new IDs.",
)
args = parser.parse_args()

print("Starting singer ID assignment process...")

# Determine file paths
dataset_path = Path(args.dataset_path)
filtered_df_path = dataset_path / "data.csv"
output_file = dataset_path / "singer_id_mapping_filtered.json"

# Load the filtered dataset with local file mappings
print(f"Loading filtered dataset from {filtered_df_path}...")
df = pd.read_csv(filtered_df_path)
print(f"Loaded filtered dataset with {len(df)} rows")

if len(df) == 0:
    print("Warning: Dataset is empty. Nothing to process.")
    sys.exit(0)

print(f"Artist header name: {args.artist_name_header}")

def filter_and_assign_singer_ids(df):
    # All filter patterns in one list - patterns we want to exclude
    filters = [
        'feat\\.', 
        'Orchestra|Philharmonic|Symphonica|Chamber|Ensemble|Piano|string|strings|Symphony|Symphonie|symphon|Concerto|choir|chorus|Philharmoniker',
        '交响|乐团|协奏曲|合唱',
        'DJ |D\\.J\\.',  # Matching both "DJ " and "D.J."
        'vs\\.',
        '\' \'',
        '&',
        'unknown|anonymous|unknown artist',
        'with', 'collection', 'collective', ','
    ]
    
    # Combine all patterns with OR operator (only if filters exist)
    if filters:
        combined_pattern = '|'.join(filters)
        # Create masks for both pattern matches and NaN values
        pattern_mask = df[args.artist_name_header].str.contains(combined_pattern, case=False, na=False)
    else:
        # If no filters, create empty mask
        pattern_mask = pd.Series([False] * len(df), index=df.index)
    
    nan_mask = df[args.artist_name_header].isna()
    
    # Count NaN values
    nan_count = nan_mask.sum()
    total_songs = len(df)
    nan_percentage = (nan_count / total_songs * 100) if total_songs > 0 else 0.0
    
    # Combine both masks - anything matching filters or NaN will be removed
    filter_mask = pattern_mask | nan_mask
    
    # Count unique artist names matching any filter (excluding NaN)
    filtered_artists = df[pattern_mask][args.artist_name_header].unique()
    filtered_artist_count = len(filtered_artists)
    
    # Count songs (rows) with filtered artists or NaN values
    filtered_song_count = filter_mask.sum()
    
    # Calculate percentages
    total_artists = df[args.artist_name_header].nunique()
    artist_percentage = (filtered_artist_count / total_artists * 100) if total_artists > 0 else 0.0
    song_percentage = (filtered_song_count / total_songs * 100) if total_songs > 0 else 0.0
    
    # Print results
    print(f"Total unique artist names in filtered dataset: {total_artists}")
    print(f"NaN artist names: {nan_count} ({nan_percentage:.2f}% of total songs)")
    print(f"Artist names matching at least one filter: {filtered_artist_count} ({artist_percentage:.2f}%)")
    print(f"\nTotal songs in filtered dataset: {total_songs}")
    print(f"Songs with filtered artists or NaN: {filtered_song_count} ({song_percentage:.2f}%)")
    
    # Get all unique artists
    all_artists = df[args.artist_name_header].unique()
    
    # Convert filtered_artists to a set for faster lookup
    filtered_artists_set = set(filtered_artists)
    
    # Get artists that are NOT in filtered_artists and are not NaN
    unfiltered_artists = [artist for artist in all_artists 
                         if artist not in filtered_artists_set and not pd.isna(artist)]
    
    print(f"Number of unique unfiltered artists (case sensitive): {len(set(unfiltered_artists))}")
    
    # Process artists case-insensitively
    # Create a dictionary mapping lowercase artist names to their original forms
    case_mapping = {}
    for artist in unfiltered_artists:
        if isinstance(artist, str):
            lowercase = artist.lower()
            if lowercase in case_mapping:
                case_mapping[lowercase].append(artist)
            else:
                case_mapping[lowercase] = [artist]
    
    print(f"Number of unique unfiltered artists (case insensitive): {len(case_mapping)}")
    
    # Assign singer IDs to each unique lowercase artist name
    singer_id_mapping = {}
    id_counter = 1
    
    for lowercase_name in sorted(case_mapping.keys()):
        # Create 5-digit ID with padding
        singer_id = f"id{id_counter:05d}"
        # Store the mapping with original variations
        singer_id_mapping[singer_id] = {
            "lowercase": lowercase_name,
            "variations": case_mapping[lowercase_name]
        }
        id_counter += 1
    
    print(f"Created mapping for {len(singer_id_mapping)} unique singers")
    
    return singer_id_mapping

# Check if using pre-existing mapping or generating new one
if args.singer_id_mapping_json:
    print(f"Loading singer ID mapping from {args.singer_id_mapping_json}...")
    mapping_path = Path(args.singer_id_mapping_json)
    
    if not mapping_path.exists():
        print(f"Error: Mapping file not found at {mapping_path}")
        sys.exit(1)
    
    with open(mapping_path, 'r') as f:
        singer_id_mapping = json.load(f)
    
    print(f"Loaded mapping with {len(singer_id_mapping)} singer IDs")
    
    # Validate mapping structure
    for singer_id, info in singer_id_mapping.items():
        if not isinstance(info, dict):
            print(f"Error: Invalid mapping structure for {singer_id}. Expected dict, got {type(info)}")
            sys.exit(1)
        if "lowercase" not in info or "variations" not in info:
            print(f"Error: Invalid mapping structure for {singer_id}. Missing 'lowercase' or 'variations' keys")
            sys.exit(1)
        if not isinstance(info["variations"], list):
            print(f"Error: Invalid mapping structure for {singer_id}. 'variations' should be a list")
            sys.exit(1)
    
    print("Mapping structure validated successfully")
    
    # Save a copy to the output file for reference
    print(f"Saving mapping copy to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(singer_id_mapping, f, indent=2)
else:
    # Run the filtering and ID assignment (original behavior)
    print("Filtering artists and assigning IDs...")
    singer_id_mapping = filter_and_assign_singer_ids(df)
    
    # Save the singer ID mapping to a file
    print(f"Saving singer ID mapping to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(singer_id_mapping, f, indent=2)

# Create a reverse mapping (from artist name to ID) for easier lookup
reverse_mapping = {}
for singer_id, info in singer_id_mapping.items():
    lowercase = info["lowercase"]
    reverse_mapping[lowercase] = singer_id
    for variation in info["variations"]:
        reverse_mapping[variation] = singer_id

# Add singer_id column to the dataframe
print("Adding singer_id column to the dataframe...")
def get_singer_id(artist_name):
    if pd.isna(artist_name):
        return None
    
    # Check direct mapping first (case-sensitive)
    if artist_name in reverse_mapping:
        return reverse_mapping[artist_name]
    
    # Try lowercase version
    lowercase = artist_name.lower()
    if lowercase in reverse_mapping:
        return reverse_mapping[lowercase]
    
    # If not found, it's a filtered artist
    return None

df['singer_id'] = df[args.artist_name_header].apply(get_singer_id)

# Count rows before filtering
rows_before = len(df)
filtered_out_count = df['singer_id'].isna().sum()

# Filter out rows that don't have a singer_id assigned (these are filtered artists)
print(f"\nFiltering out {filtered_out_count} rows with no singer_id assigned...")
df_filtered = df[df['singer_id'].notna()].copy()
rows_after = len(df_filtered)

print(f"Removed {rows_before - rows_after} rows ({filtered_out_count} rows)")
kept_percentage = (rows_after / rows_before * 100) if rows_before > 0 else 0.0
print(f"Kept {rows_after} rows ({kept_percentage:.2f}% of original dataset)")

# Remove corresponding audio files/directories from the filesystem
# Check if 'local_file_name' column exists (from previous preprocessing step)
audio_dir = dataset_path / "audio"
if 'local_file_name' in df.columns and audio_dir.exists():
    rows_to_remove = df[df['singer_id'].isna()]
    removed_dirs = 0
    failed_removals = 0
    
    if len(rows_to_remove) > 0:
        print(f"\nRemoving {len(rows_to_remove)} audio directories from filesystem...")
        for idx, row in rows_to_remove.iterrows():
            local_file_name = row.get('local_file_name')
            if pd.notna(local_file_name) and local_file_name:
                # Try to find the directory - could be at root or nested under singer_id
                dir_to_remove = None
                
                # Check if it's directly under audio/
                direct_path = audio_dir / str(local_file_name)
                if direct_path.exists() and direct_path.is_dir():
                    dir_to_remove = direct_path
                else:
                    # Check if it's nested under a singer_id directory
                    for singer_dir in audio_dir.iterdir():
                        if singer_dir.is_dir():
                            nested_path = singer_dir / str(local_file_name)
                            if nested_path.exists() and nested_path.is_dir():
                                dir_to_remove = nested_path
                                break
                
                if dir_to_remove and dir_to_remove.exists():
                    try:
                        shutil.rmtree(dir_to_remove)
                        removed_dirs += 1
                        if removed_dirs % 100 == 0:
                            print(f"  Progress: {removed_dirs}/{len(rows_to_remove)} directories removed...")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not remove {dir_to_remove}: {e}")
                        failed_removals += 1
        
        print(f"\nFilesystem cleanup complete:")
        print(f"  ✅ Successfully removed: {removed_dirs} directories")
        if failed_removals > 0:
            print(f"  ❌ Failed to remove: {failed_removals} directories")

# Save the filtered dataframe back to the same file
print(f"Saving filtered dataframe to {filtered_df_path}")
df_filtered.to_csv(filtered_df_path, index=False)

# Print some statistics
assigned_count = len(df_filtered)
assigned_percentage = (assigned_count / rows_before * 100) if rows_before > 0 else 0.0
print(f"\nAssigned singer IDs to {assigned_count} rows ({assigned_percentage:.2f}% of original dataset)")
print(f"Saved reverse mapping for {len(reverse_mapping)} artist name variations")

print("\nProcess completed successfully!")

