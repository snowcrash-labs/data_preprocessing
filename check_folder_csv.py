"""
Match local audio files with CSV entries and create deduplicated dataset.

Reads CSV from {dataset_path}/original_gs_input.csv, matches entries with local files
in {dataset_path}/desilenced_data/, and creates deduplicated_data.csv with only
matched entries and local_file_name column.
"""
import argparse
import numpy as np
import pandas as pd
import os
import re
import random
from pathlib import Path

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
args = parser.parse_args()

# Determine CSV file path: {dataset_path}/original_gs_input.csv
dataset_path = Path(args.dataset_path)
csvfile_path = dataset_path / "original_gs_input.csv"

# Output CSV path: {dataset_path}/deduplicated_data.csv
output_csv_path = dataset_path / "data.csv"
data_dir_path = dataset_path / "data"
# load CSV file
df = pd.read_csv(csvfile_path)

# print headers of each column
print(df.columns.tolist())


print("Extracting song names from CSV...")
# Create only Set 1: Basenames with .wav removed
csv_names_no_wav = {}  # Set 1: Basenames with .wav removed -> original gcs_link

# Track the index of each filename to map back to the dataframe
for index, row in df.iterrows():
    try:
        link = row[args.uri_name_header]
        songname = os.path.splitext(os.path.basename(link))[0]
        
        # Set 1: Remove .wav extension
        if not songname.endswith('.wav'):
            csv_names_no_wav[songname] = index
    except:
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

for item in sad_items:

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
for index, folder_name in local_file_mapping.items():
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
