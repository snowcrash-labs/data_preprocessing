"""
Reorganize track directories to be nested under singer_id directories.

Moves track directories from desilenced_data/{track_name}/ to desilenced_data/{singer_id}/{track_name}/,
grouping all tracks by the same singer under their ID directory.
"""
import argparse
import pandas as pd
import os
import shutil
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Reorganize track directories to be nested under singer_id directories"
)
parser.add_argument(
    "--dataset_path",
    required=True,
    help="Path to the dataset directory (e.g. ~/gs_imports/roformer_voice_sep_custom_sample)",
)
parser.add_argument(
    "--file_name_header",
    required=True,
    help="Name of the CSV column header containing track/folder names",
)
parser.add_argument(
    "--singer_id_header",
    required=True,
    help="Name of the CSV column header containing singer IDs",
)
args = parser.parse_args()

# Determine file paths
dataset_directory = Path(args.dataset_path)
csv_path = dataset_directory / "data.csv"
source_base = dataset_directory / "audio"

# Load the dataset
print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Loaded dataset with {len(df)} rows")

# Count for tracking progress
total_rows = len(df)
processed = 0
skipped = 0
errors = 0
moved = 0
already_organized = 0

print(f"Processing {total_rows} tracks...")
print(f"Source base directory: {source_base}")

# Process each row in the DataFrame
for index, row in df.iterrows():
    # Update progress counter
    processed += 1
    
    # Show progress every 1000 tracks
    if processed % 1000 == 0:
        print(f"Progress: {processed}/{total_rows} ({processed/total_rows*100:.1f}%) | Moved: {moved} | Skipped: {skipped} | Already organized: {already_organized}")
    
    # Skip if singer_id is missing
    if pd.isna(row[args.singer_id_header]):
        skipped += 1
        continue
        
    # Skip if file_name is missing
    if pd.isna(row[args.file_name_header]):
        skipped += 1
        continue
    
    # Get singer_id and folder_name
    singer_id = str(row[args.singer_id_header])
    folder_name = str(row[args.file_name_header])
    
    # Define source and destination paths (both within source_base)
    src_path = source_base / folder_name
    dest_path = source_base / singer_id / folder_name
    
    # Check if destination already exists (already organized)
    if dest_path.exists():
        already_organized += 1
        continue
    
    # Check if source exists
    if not src_path.exists():
        print(f"Warning: Source folder not found: {src_path}")
        skipped += 1
        continue
    
    try:
        # Create singer_id directory if it doesn't exist
        singer_dir = source_base / singer_id
        singer_dir.mkdir(parents=True, exist_ok=True)
        
        # Move the folder to the new location
        shutil.move(str(src_path), str(dest_path))
        moved += 1
        
    except Exception as e:
        print(f"Error moving {src_path} to {dest_path}: {str(e)}")
        errors += 1

# Print summary
print("\nReorganization complete!")
print(f"Total tracks processed: {processed}")
print(f"Successfully moved: {moved}")
print(f"Already organized: {already_organized}")
print(f"Skipped (missing singer_id, folder, or already exists): {skipped}")
print(f"Errors during move: {errors}")