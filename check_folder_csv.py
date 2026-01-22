"""
Match local audio files with CSV entries and create deduplicated dataset.

Purpose: Creates a clean CSV that only references audio data you actually have locally, dropping any entries where the download/processing failed.
"""
import argparse
import csv
import numpy as np
import pandas as pd
import os
import re
import random
from pathlib import Path
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Match local files with CSV entries and create filtered CSV"
)
parser.add_argument(
    "--dataset_path",
    required=True,
    help="Path to the dataset directory (e.g. ~/gs_imports/roformer_voice_sep_custom_sample)",
)

parser.add_argument(
    "--uri_name_header",
    required=True,
    help="Name of the CSV column header containing track names/URIs",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
parser.add_argument(
    "--gs_file_uri_in_csv",
    action="store_true",
    default=False,
    help="If set, the CSV contains full GCS file URIs (extract track name from parent directory). "
         "Otherwise, extract track name from the filename itself.",
)
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(args.seed)

# Determine CSV file path: {dataset_path}/original_gs_input.csv
dataset_path = Path(args.dataset_path)
csvfile_path = dataset_path / "original_gs_input.csv"

# Output CSV path: {dataset_path}/deduplicated_data.csv
output_csv_path = dataset_path / "data.csv"
data_dir_path = dataset_path / "audio"
# load CSV file
# Note: The CSV uses proper quoting for fields containing commas, so we use csv.reader
print(f"Loading CSV file from {csvfile_path}...")

rows = []
skipped_lines = 0
with open(csvfile_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # Read header row
    print(f"CSV headers: {header}")
    
    for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
        # Handle rows with expected number of columns
        if len(row) == 5:
            rows.append({
                'index': row[0],
                'title': row[1],
                'artist_name': row[2],
                'youtube_link': row[3],
                'gcs_link': row[4]
            })
        elif len(row) > 5:
            # More columns than expected - likely unquoted commas in a field
            # Try to reconstruct: gcs_link is last (starts with gs://), youtube_link is second-to-last
            gcs_link = row[-1]
            youtube_link = row[-2]
            # index is first, title is second, artist_name is everything in between
            index_val = row[0]
            title = row[1]
            artist_name = ','.join(row[2:-2])  # Join middle parts back together
            rows.append({
                'index': index_val,
                'title': title,
                'artist_name': artist_name,
                'youtube_link': youtube_link,
                'gcs_link': gcs_link
            })
        else:
            # Too few columns - skip
            skipped_lines += 1
            if skipped_lines <= 5:
                print(f"Warning: Row {row_num} has {len(row)} columns (expected 5): {str(row)[:100]}...")

if skipped_lines > 5:
    print(f"Warning: {skipped_lines} total rows could not be parsed")

df = pd.DataFrame(rows)

# print headers of each column
print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")


print("Extracting song names from CSV...")
# Create only Set 1: Basenames with .wav removed
csv_names_no_wav = {}  # Set 1: Basenames with .wav removed -> original gcs_link

# Track the index of each filename to map back to the dataframe
for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting song names"):
    try:
        link = row[args.uri_name_header]
        
        # Extract track name based on URI structure
        if args.gs_file_uri_in_csv:
            # URI contains full file path like gs://bucket/path/track_name/vocals.wav
            # Extract track name from parent directory
            songname = os.path.basename(os.path.dirname(link))
        else:
            # URI contains the track name as the filename
            songname = os.path.splitext(os.path.basename(link))[0]
        
        # Set 1: Remove .wav extension if still present
        if not songname.endswith('.wav'):
            csv_names_no_wav[songname] = index
    except:
        print(f"Error extracting song name from {row}")
        continue

# Get all items in data_dir_path directory (only depth 1, no subdirectories)
print(f"Reading items from {data_dir_path} directory...")
sad_items = [item for item in os.listdir(str(data_dir_path)) if item != '.DS_Store']
# Create a dictionary to map df indices to their local_file_name in data_dir_path
local_file_mapping = {}
# Track duplicate mappings (when multiple folder names match to the same track)
duplicate_mappings = []
# Store folders that need to find alternative matches
folders_needing_rematch = []

# Helper function to handle duplicates
def handle_duplicate(index, existing_name, new_name):
    if len(new_name) > len(existing_name):
        # New name is longer, prefer it
        print(f"Duplicate mapping: Preferring longer name '{new_name}' over '{existing_name}' for index {index}")
        folders_needing_rematch.append(existing_name)
        return new_name
    else:
        # Existing name is longer or equal, keep it
        print(f"Duplicate mapping: Keeping longer name '{existing_name}' over '{new_name}' for index {index}")
        folders_needing_rematch.append(new_name)
        return existing_name

# First pass: check exact matches only with Set 1
files_in_csv = []
potentially_not_found = []

for item in tqdm(sad_items, desc="Matching local files to CSV"):
    found = False

    # Check Set 1 (no_wav)
    if item in csv_names_no_wav:
        files_in_csv.append(item)
        index = csv_names_no_wav[item]
        if index in local_file_mapping:
            local_file_mapping[index] = handle_duplicate(index, local_file_mapping[index], item)
        else:
            local_file_mapping[index] = item
        found = True
    
    if not found:
        potentially_not_found.append(item)

print(f"After first pass checks (no_wav only):")
print(f"  - Found in CSV: {len(files_in_csv)}")
print(f"  - Potentially not found: {len(potentially_not_found)}")
print(f"  - Folders needing rematch: {len(folders_needing_rematch)}")

# Print some examples of potentially not found items
if potentially_not_found:
    print("\nSample of potentially not found items:")
    sample_size = min(10, len(potentially_not_found))
    for item in random.sample(potentially_not_found, sample_size):
        print(f"  - {item}")

# Print some examples of found items
if files_in_csv:
    print("\nSample of found items:")
    sample_size = min(10, len(files_in_csv))
    for item in random.sample(files_in_csv, sample_size):
        print(f"  - {item}")

# Print total number of matches
print(f"\nTotal unique mappings: {len(local_file_mapping)}")

# Also note that, some files in the original dataframe may not be found in the data_dir_path directory. If they are not found, do not save them to the new dataframe. 

# Add local_file_name column to the dataframe
print("Adding local_file_name column to dataframe...")
df['local_file_name'] = None

# Populate the local_file_name column using the mapping
for index, folder_name in tqdm(local_file_mapping.items(), desc="Populating local_file_name"):
    df.at[index, 'local_file_name'] = folder_name

# Filter the dataframe to only include rows with local_file_name
filtered_df = df[df['local_file_name'].notna()].copy()
print(f"Original dataframe size: {len(df)} rows")
print(f"Filtered dataframe size: {len(filtered_df)} rows")
print(f"Removed {len(df) - len(filtered_df)} rows without local file matches")

# Save the filtered dataframe to a new CSV file
print(f"Saving filtered dataframe to {output_csv_path}...")
filtered_df.to_csv(output_csv_path, index=False)
print("Done!")

# Print summary of the updated dataframe
print(f"Total records in saved dataframe: {len(filtered_df)}")
