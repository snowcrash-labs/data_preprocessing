'''
This script splits the HookTheory dataset into training and testing sets by singer.
It ensures that all songs from the same singer are in either train or test set.
Test set consists of approximately 10% of singers, prioritizing those with 2-4 songs.
'''

import os
import json
import random
import shutil
from pathlib import Path
import time
import glob

# Base paths
base_dir = '/home/aik2/sc-rawnet3/datasets/hooktheory/audio_16k'
wav_dir = os.path.join(base_dir, 'wav')
train_dir = os.path.join(base_dir, 'train', 'wav')
test_dir = os.path.join(base_dir, 'test', 'wav')

# Create output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Path to singer data JSON
singer_data_path = '/home/aik2/sc-rawnet3/datasets/hooktheory/singer_data_complete.json'

# Load singer data
try:
    with open(singer_data_path, 'r') as f:
        singer_data = json.load(f)
    print(f"Successfully loaded {singer_data_path}")
except Exception as e:
    print(f"Error loading singer data: {e}")
    exit(1)

# Start counting statistics
start_time = time.time()

# Count number of singers in the dataset
singer_dirs = [d for d in os.listdir(wav_dir) if os.path.isdir(os.path.join(wav_dir, d))]
total_singers = len(singer_dirs)
print(f"Total number of singers: {total_singers}")

# Calculate the number of singers for test set (10% of total)
ntest = int(total_singers * 0.1)
print(f"Target number of singers for test set: {ntest}")

# Find singers with 2-4 songs
singers_with_2to4_songs = []
singer_song_counts = {}

for singer_id in singer_dirs:
    singer_dir = os.path.join(wav_dir, singer_id)
    song_dirs = [d for d in os.listdir(singer_dir) if os.path.isdir(os.path.join(singer_dir, d))]
    song_count = len(song_dirs)
    singer_song_counts[singer_id] = song_count
    
    if 2 <= song_count <= 4:
        singers_with_2to4_songs.append(singer_id)

print(f"Singers with 2-4 songs: {len(singers_with_2to4_songs)}")

# Randomly select test singers from those with 2-4 songs
# If not enough, randomly select from remaining singers
random.seed(42)  # For reproducibility
random.shuffle(singers_with_2to4_songs)

test_singers = singers_with_2to4_songs[:ntest]
if len(test_singers) < ntest:
    remaining_singers = [s for s in singer_dirs if s not in test_singers and s not in singers_with_2to4_songs]
    random.shuffle(remaining_singers)
    test_singers += remaining_singers[:ntest - len(test_singers)]

# Train singers are all singers not in test set
train_singers = [s for s in singer_dirs if s not in test_singers]

print(f"Final test set size: {len(test_singers)} singers")
print(f"Final train set size: {len(train_singers)} singers")

# Create dictionary to keep track of singers and their split
singer_split = {singer_id: "train" if singer_id in train_singers else "test" for singer_id in singer_dirs}

# Counters for statistics
train_files_moved = 0
test_files_moved = 0
errors = 0

# Move files to their respective directories
print("\nMoving files to train/test directories...")

for singer_id in singer_dirs:
    # Source directory
    src_singer_dir = os.path.join(wav_dir, singer_id)
    
    # Destination directory
    if singer_id in train_singers:
        dest_singer_dir = os.path.join(train_dir, singer_id)
    else:
        dest_singer_dir = os.path.join(test_dir, singer_id)
    
    # Create destination directory
    os.makedirs(dest_singer_dir, exist_ok=True)
    
    # Iterate through song directories
    song_dirs = [d for d in os.listdir(src_singer_dir) if os.path.isdir(os.path.join(src_singer_dir, d))]
    
    for song_id in song_dirs:
        src_song_dir = os.path.join(src_singer_dir, song_id)
        dest_song_dir = os.path.join(dest_singer_dir, song_id)
        
        # Create destination song directory
        os.makedirs(dest_song_dir, exist_ok=True)
        
        # Find all wav files in the song directory
        wav_files = glob.glob(os.path.join(src_song_dir, "*.wav"))
        
        for wav_file in wav_files:
            filename = os.path.basename(wav_file)
            dest_file = os.path.join(dest_song_dir, filename)
            
            try:
                shutil.copy2(wav_file, dest_file)  # Using copy2 to preserve metadata
                if singer_id in train_singers:
                    train_files_moved += 1
                else:
                    test_files_moved += 1
                
                # Print progress every 100 files
                if (train_files_moved + test_files_moved) % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = (train_files_moved + test_files_moved) / elapsed if elapsed > 0 else 0
                    print(f"Progress: {train_files_moved + test_files_moved} files moved")
                    print(f"Train: {train_files_moved}, Test: {test_files_moved}")
                    print(f"Speed: {speed:.2f} files/second")
            except Exception as e:
                print(f"Error copying file {wav_file}: {str(e)}")
                errors += 1

# Create new split_by_singer.json
print("\nCreating split_by_singer.json...")

split_by_singer = {}

for singer_id, info in singer_data.items():
    if singer_id not in singer_dirs:
        continue  # Skip singers not in the wav directory
    
    split_type = "train" if singer_id in train_singers else "test"
    artist_name = info["artist_name"]
    audio_paths = []
    
    # Get song directories for this singer
    singer_dir = os.path.join(wav_dir, singer_id)
    if os.path.exists(singer_dir):
        song_dirs = [d for d in os.listdir(singer_dir) if os.path.isdir(os.path.join(singer_dir, d))]
        
        for song_id in song_dirs:
            # Find wav files
            wav_files = glob.glob(os.path.join(singer_dir, song_id, "*.wav"))
            
            for wav_file in wav_files:
                # Get original path from singer_data_complete.json
                original_path = None
                for audio_item in info.get("audio_paths", []):
                    if song_id in audio_item:
                        original_path = audio_item
                        break
                
                # Create path to 16k wav file in train/test directory
                filename = os.path.basename(wav_file)
                path_16k = os.path.join(base_dir, split_type, "wav", singer_id, song_id, filename)
                
                # Add to audio_paths
                audio_paths.append({
                    "path": original_path,
                    "path_16k": path_16k,
                    "split": split_type
                })
    
    # Add singer to split_by_singer
    split_by_singer[singer_id] = {
        "artist_name": artist_name,
        "split": split_type,
        "audio_paths": audio_paths
    }

# Save the split_by_singer.json file
split_by_singer_path = os.path.join(base_dir, "split_by_singer.json")
try:
    with open(split_by_singer_path, "w") as f:
        json.dump(split_by_singer, f, indent=2)
    print(f"Successfully saved {split_by_singer_path}")
except Exception as e:
    print(f"Error saving split_by_singer.json: {e}")

# Print final statistics
elapsed_time = time.time() - start_time
print("\nSplit by singer complete!")
print(f"Train singers: {len(train_singers)}")
print(f"Test singers: {len(test_singers)}")
print(f"Train files: {train_files_moved}")
print(f"Test files: {test_files_moved}")
print(f"Errors encountered: {errors}")
print(f"Total processing time: {elapsed_time:.2f} seconds")

