'''
This script creates pairs of audio files for testing singer verification.
For each singer in the test set:
1. Creates pairs of files from the same singer (positive pairs, label 1)
2. Creates pairs with files from different singers (negative pairs, label 0)
'''

# imports
import os
import json
import random
import itertools
import pandas as pd
import argparse
from pathlib import Path
import glob

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Create test pairs for singer verification from CSV and directory structure"
)
parser.add_argument(
    "--csv_path",
    required=True,
    help="Path to CSV file containing dataset information with columns: singer_id, split, local_file_name, etc."
)
parser.add_argument(
    "--test_dir",
    required=True,
    help="Path to test directory containing structure: {singer_id}/{song_name}/{audio_chunks}"
)
parser.add_argument(
    "--output_path",
    required=True,
    help="Path to output test_pairs.txt file"
)
parser.add_argument(
    "--singer_id_header",
    default="singer_id",
    help="Name of CSV column containing singer IDs (default: singer_id)"
)
parser.add_argument(
    "--split_header",
    default="split",
    help="Name of CSV column containing split assignment (default: split)"
)
parser.add_argument(
    "--file_name_header",
    default="local_file_name",
    help="Name of CSV column containing file/song names (default: local_file_name)"
)
args = parser.parse_args()

# Load CSV file
print(f"Loading CSV from {args.csv_path}")
try:
    df = pd.read_csv(args.csv_path)
    print(f"Loaded CSV with {len(df)} rows")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Filter for test set
# Handle both string values and numeric values
# Numeric mapping: 0=train, 1=test (validation), 2=exp (test set)
# String mapping: 'train'=train, 'test'=validation, 'exp'=test set
split_col = df[args.split_header]

# Convert to string for comparison, handling both numeric and string values
split_values_str = split_col.astype(str).str.lower().str.strip()
split_values_num = pd.to_numeric(split_col, errors='coerce')

# Check for test set: numeric 1 (validation/test) or 2 (exp/test), or string 'test' or 'exp'
test_mask = (
    split_values_num.isin([1, 2]) |  # Numeric: 1=test/validation, 2=exp/test
    split_values_str.isin(['test', 'exp', '1', '2'])  # String values
)
test_df = df[test_mask].copy()

print(f"Found {len(test_df)} rows in test set")
if len(test_df) > 0:
    print(f"Unique split values in test set: {test_df[args.split_header].unique()}")
else:
    print("Warning: No test rows found. Check the split column values.")
    print(f"Available split values in dataset: {df[args.split_header].unique()}")

# Get test directory path
test_wav_dir = Path(args.test_dir)
if not test_wav_dir.exists():
    print(f"Error: Test directory does not exist: {test_wav_dir}")
    exit(1)

# Build mapping from (singer_id, song_name) to CSV row for reference
# The song_name in directory might match local_file_name or be derived from it
csv_singer_song_map = {}
for _, row in test_df.iterrows():
    singer_id = str(row[args.singer_id_header])
    file_name = str(row[args.file_name_header])
    # Store mapping - we'll use this to match directory structure
    if singer_id not in csv_singer_song_map:
        csv_singer_song_map[singer_id] = set()
    csv_singer_song_map[singer_id].add(file_name)

# Walk the test directory to find all audio files
print(f"Scanning test directory: {test_wav_dir}")
all_wav_files = []
test_singers = {}  # Dictionary to store test singers and their files

# Expected structure: test_dir/{singer_id}/{song_name}/{audio_chunks.wav}
for singer_dir in test_wav_dir.iterdir():
    if not singer_dir.is_dir():
        continue
    
    singer_id = singer_dir.name
    
    # Only process singers that are in the test CSV
    if singer_id not in csv_singer_song_map:
        continue
    
    # Walk through song directories
    for song_dir in singer_dir.iterdir():
        if not song_dir.is_dir():
            continue
        
        song_name = song_dir.name
        
        # Find all WAV files in this song directory
        wav_files = list(song_dir.glob("*.wav"))
        
        for wav_file in wav_files:
            # Create relative path: {singer_id}/{song_name}/{filename.wav}
            rel_path = os.path.relpath(wav_file, start=test_wav_dir)
            all_wav_files.append(rel_path)
            
            # Add to the test_singers dictionary
            if singer_id not in test_singers:
                test_singers[singer_id] = []
            test_singers[singer_id].append(rel_path)

print(f"Total test wav files found: {len(all_wav_files)}")
print(f"Test singers found: {len(test_singers)}")


# Create pairs
all_pairs = []
used_pairs = set()  # Track all pairs to ensure no duplicates

# For each singer in the test set
for singer_id, singer_files in test_singers.items():
    print(f"Processing test singer {singer_id} with {len(singer_files)} files")
    
    # Make all combinations of 2 files from this singer
    same_singer_pairs = list(itertools.combinations(singer_files, 2))
    print(f"  Created {len(same_singer_pairs)} positive pairs")
    
    # Get files not from this singer - do this once per singer
    # Extract singer_id from path (first part of the path)
    other_singers_files = [f for f in all_wav_files if not f.startswith(f"{singer_id}/")]
    random.shuffle(other_singers_files)  # Shuffle to ensure random selection
    
    # For each positive pair, create a corresponding negative pair
    for i, (file1, file2) in enumerate(same_singer_pairs):
        # Create a unique key for this pair to check for duplicates
        pair_key = tuple(sorted([file1, file2]))
        if pair_key in used_pairs:
            print(f"  Skipping duplicate positive pair: {file1}, {file2}")
            continue
            
        # Add the positive pair (label 1)
        all_pairs.append((1, file1, file2))
        used_pairs.add(pair_key)
        
        # Find a file from a different singer for a negative pair
        if other_singers_files:
            # Take the next file from the shuffled list
            negative_idx = i % len(other_singers_files)
            random_file = other_singers_files[negative_idx]
            
            # Create a unique key for this negative pair
            neg_pair_key1 = tuple(sorted([file1, random_file]))
            neg_pair_key2 = tuple(sorted([file2, random_file]))
            
            # Check if this negative pair already exists
            if neg_pair_key1 not in used_pairs and neg_pair_key2 not in used_pairs:
                # Add the negative pair (label 0)
                all_pairs.append((0, file1, random_file))
                used_pairs.add(neg_pair_key1)
            else:
                # Try to find another file that hasn't been used
                found_unused = False
                for alt_file in other_singers_files:
                    alt_pair_key = tuple(sorted([file1, alt_file]))
                    if alt_pair_key not in used_pairs:
                        all_pairs.append((0, file1, alt_file))
                        used_pairs.add(alt_pair_key)
                        found_unused = True
                        break
                
                if not found_unused:
                    print(f"  Warning: Could not find unused pair for {file1}")
        else:
            print(f"  Warning: No files found from other singers")

# Final check for duplicate pairs
unique_pairs = set()
final_pairs = []
duplicates_found = 0

for label, file1, file2 in all_pairs:
    # Normalize the pair by sorting
    if label == 1:
        # For positive pairs, order doesn't matter
        pair_key = (label, tuple(sorted([file1, file2])))
    else:
        # For negative pairs, first file is from test set, second is from other singers
        pair_key = (label, file1, file2)
    
    if pair_key not in unique_pairs:
        unique_pairs.add(pair_key)
        final_pairs.append((label, file1, file2))
    else:
        duplicates_found += 1

if duplicates_found > 0:
    print(f"Found and removed {duplicates_found} duplicate pairs")

print(f"Total pairs created: {len(final_pairs)}")
print(f"Positive pairs: {sum(1 for label, _, _ in final_pairs if label == 1)}")
print(f"Negative pairs: {sum(1 for label, _, _ in final_pairs if label == 0)}")

# Write pairs to file
try:
    with open(args.output_path, 'w') as f:
        for label, file1, file2 in final_pairs:
            f.write(f"{label} {file1} {file2}\n")
    print(f"Successfully saved pairs to {args.output_path}")
except Exception as e:
    print(f"Error saving pairs to {args.output_path}: {e}")

print("Done!")

