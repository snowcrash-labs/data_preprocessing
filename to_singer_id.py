"""
Reorganize track directories to be nested under singer_id directories.

Moves track directories from desilenced_data/{track_name}/ to desilenced_data/{singer_id}/{track_name}/,
grouping all tracks by the same singer under their ID directory.
"""
import argparse
import pandas as pd
import os
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

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
parser.add_argument(
    "--no-parallel",
    action="store_true",
    default=False,
    help="Disable parallel processing (process files sequentially)",
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


def move_track(row_data: tuple) -> tuple:
    """
    Move a single track directory to its singer_id subdirectory.
    Returns: (status, message) where status is 'moved', 'skipped', 'already', or 'error'
    """
    singer_id, folder_name, source_base_str = row_data
    source_base_path = Path(source_base_str)
    
    if pd.isna(singer_id) or pd.isna(folder_name):
        return ('skipped', 'missing data')
    
    singer_id = str(singer_id)
    folder_name = str(folder_name)
    
    src_path = source_base_path / folder_name
    dest_path = source_base_path / singer_id / folder_name
    
    # Check if destination already exists
    if dest_path.exists():
        return ('already', folder_name)
    
    # Check if source exists
    if not src_path.exists():
        return ('skipped', f'source not found: {folder_name}')
    
    try:
        # Create singer_id directory if it doesn't exist
        singer_dir = source_base_path / singer_id
        singer_dir.mkdir(parents=True, exist_ok=True)
        
        # Move the folder
        shutil.move(str(src_path), str(dest_path))
        return ('moved', folder_name)
    except Exception as e:
        return ('error', f'{folder_name}: {str(e)}')


# Prepare work items
print(f"Preparing {len(df)} tracks for processing...")
work_items = [
    (row[args.singer_id_header], row[args.file_name_header], str(source_base))
    for _, row in df.iterrows()
]

# Counters
moved = 0
skipped = 0
already_organized = 0
errors = 0

if getattr(args, 'no_parallel', False):
    # Sequential processing
    print("Processing sequentially...")
    for item in tqdm(work_items, desc="Moving tracks"):
        status, msg = move_track(item)
        if status == 'moved':
            moved += 1
        elif status == 'skipped':
            skipped += 1
        elif status == 'already':
            already_organized += 1
        elif status == 'error':
            errors += 1
            print(f"Error: {msg}")
else:
    # Parallel processing with ThreadPoolExecutor (I/O bound)
    num_workers = min(32, multiprocessing.cpu_count() * 2)
    print(f"Processing with {num_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(move_track, item): item for item in work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Moving tracks"):
            status, msg = future.result()
            if status == 'moved':
                moved += 1
            elif status == 'skipped':
                skipped += 1
            elif status == 'already':
                already_organized += 1
            elif status == 'error':
                errors += 1
                print(f"Error: {msg}")

# Print summary
print("\nReorganization complete!")
print(f"Total tracks processed: {len(work_items)}")
print(f"Successfully moved: {moved}")
print(f"Already organized: {already_organized}")
print(f"Skipped (missing singer_id, folder, or source not found): {skipped}")
print(f"Errors during move: {errors}")
