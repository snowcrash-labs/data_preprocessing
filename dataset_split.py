"""
Randomly split dataset into train, validation, and test sets (80:10:10).

Splits singers randomly into train (80%), validation (10%), and test (10%) sets,
moves singer directories to respective subdirectories, and creates CSV with split assignments.
"""
import argparse
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Randomly split dataset into train, val, and test sets based on singer IDs (80:10:10)"
)
parser.add_argument(
    "--dataset_path",
    required=True,
    help="Path to the dataset directory (e.g. ~/gs_imports/roformer_voice_sep_custom_sample)",
)
parser.add_argument(
    "--input_csv_name",
    required=True,
    help="Name of the input CSV file (appended to dataset_directory)",
)
parser.add_argument(
    "--output_csv_name",
    required=True,
    help="Name of the output CSV file (appended to dataset_directory)",
)
parser.add_argument(
    "--artist_name_header",
    required=True,
    help="Name of the CSV column header containing artist names",
)
parser.add_argument(
    "--singer_id_header",
    required=True,
    help="Name of the CSV column header containing singer IDs",
)
args = parser.parse_args()

# Determine file paths
dataset_directory = Path(args.dataset_path)
input_csv_path = dataset_directory / args.input_csv_name
output_file = dataset_directory / args.output_csv_name

print("Loading dataframe...")
# load dataframe
df = pd.read_csv(input_csv_path)

# Get unique singer IDs
unique_singer_ids = df[args.singer_id_header].unique()
total_singers = len(unique_singer_ids)
print(f"Total unique singer IDs: {total_singers}")

# Calculate split sizes (80:10:10)
num_train = int(total_singers * 0.8)
num_val = int(total_singers * 0.1)
num_test = total_singers - num_train - num_val  # Remaining goes to test

print(f"Split sizes: Train={num_train}, Val={num_val}, Test={num_test}")

# Randomly shuffle singer IDs
np.random.seed(42)
shuffled_singer_ids = np.random.permutation(unique_singer_ids)

# Split into train, val, test
train_singers = shuffled_singer_ids[:num_train].tolist()
val_singers = shuffled_singer_ids[num_train:num_train + num_val].tolist()
test_singers = shuffled_singer_ids[num_train + num_val:].tolist()

print(f"Selected {len(train_singers)} singers for train set")
print(f"Selected {len(val_singers)} singers for val set")
print(f"Selected {len(test_singers)} singers for test set")

# Create a mapping from singer_id to split for quick lookup
singer_split_map = {}
for singer_id in train_singers:
    singer_split_map[singer_id] = 'train'
for singer_id in val_singers:
    singer_split_map[singer_id] = 'val'
for singer_id in test_singers:
    singer_split_map[singer_id] = 'test'

# Add split column to dataframe
def assign_split(singer_id):
    return singer_split_map.get(singer_id, 'train')

df['split'] = df[args.singer_id_header].apply(assign_split)

# Convert to numeric for consistency
df['split'] = df['split'].map({'train': 0, 'val': 1, 'test': 2})

# Print statistics
train_songs = (df['split'] == 0).sum()
val_songs = (df['split'] == 1).sum()
test_songs = (df['split'] == 2).sum()

print(f"\nDataset split statistics:")
print(f"Train songs: {train_songs} ({train_songs/len(df)*100:.1f}%)")
print(f"Val songs: {val_songs} ({val_songs/len(df)*100:.1f}%)")
print(f"Test songs: {test_songs} ({test_songs/len(df)*100:.1f}%)")
print(f"Total songs: {len(df)}")

# save the dataframe to a new csv file
df.to_csv(output_file, index=False)
print(f"Saved dataframe with split info to: {output_file}")

# now inside the data folder, move each folder to the train, val, or test folder
sad_id_path = dataset_directory / "data"
train_path = sad_id_path / 'train'
val_path = sad_id_path / 'val'
test_path = sad_id_path / 'test'

# Create directories if they don't exist
train_path.mkdir(parents=True, exist_ok=True)
val_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)

print(f"\nMoving folders from {sad_id_path}...")

# Get all folders in data directory (these should be singer_ids)
if sad_id_path.exists():
    folders = [f for f in sad_id_path.iterdir() if f.is_dir() and f.name not in ['train', 'val', 'test']]
    total_folders = len(folders)
    print(f"Found {total_folders} folders to move")
    
    moved_counts = {'train': 0, 'val': 0, 'test': 0}
    progress_count = 0
    
    for i, folder in enumerate(folders, 1):
        singer_id = folder.name  # folder name should be the singer_id (e.g., id09707)
        
        # Look up split for this singer_id
        split_name = singer_split_map.get(singer_id, 'train')  # default to train
        
        # Determine destination
        if split_name == 'train':
            destination = train_path / singer_id
            moved_counts['train'] += 1
        elif split_name == 'val':
            destination = val_path / singer_id
            moved_counts['val'] += 1
        elif split_name == 'test':
            destination = test_path / singer_id
            moved_counts['test'] += 1
        
        # Move the folder
        try:
            shutil.move(str(folder), str(destination))
            progress_count += 1
        except Exception as e:
            print(f"Error moving {singer_id}: {e}")
            continue
        
        # Print progress every 1000 folders
        if progress_count % 1000 == 0:
            print(f"Progress: {progress_count}/{total_folders} folders moved "
                  f"({progress_count/total_folders*100:.1f}%) | "
                  f"Train: {moved_counts['train']}, Val: {moved_counts['val']}, Test: {moved_counts['test']}")
    
    print(f"\nFolder movement completed:")
    print(f"Train folders: {moved_counts['train']}")
    print(f"Val folders: {moved_counts['val']}")
    print(f"Test folders: {moved_counts['test']}")
    print(f"Total moved: {sum(moved_counts.values())}")
    
else:
    print(f"Error: data directory not found at {sad_id_path}")

print("\nSplit completed successfully!")