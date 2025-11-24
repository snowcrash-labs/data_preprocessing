import pandas as pd
import os
import json
import random  # Added for random assignment
from pathlib import Path

# 1. Load the dataframe
# Assuming the data is stored in a CSV or similar format
# If it's in a different format, you'll need to adjust this part
def load_dataframe(file_path):
    try:
        # Check file extension to determine the format
        if file_path.endswith('.jsonl'):
            # For JSONL files, use lines=True
            df = pd.read_json(file_path, lines=True)
        elif file_path.endswith('.csv'):
            # Try loading as CSV
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            # Try loading as parquet
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            # Try loading as regular JSON
            df = pd.read_json(file_path)
        else:
            # If extension doesn't match, try each format
            try:
                # Try JSONL first (most common for this dataset)
                df = pd.read_json(file_path, lines=True)
            except:
                try:
                    # Try CSV
                    df = pd.read_csv(file_path)
                except:
                    try:
                        # Try parquet
                        df = pd.read_parquet(file_path)
                    except:
                        try:
                            # Try regular JSON
                            df = pd.read_json(file_path)
                        except Exception as e:
                            print(f"Error loading dataframe: {e}")
                            return None
    except Exception as e:
        print(f"Error loading dataframe: {e}")
        return None
    return df

# Define output directory - use the same directory as the script
output_dir = '/home/aik2/sc-rawnet3/datasets/hooktheory/'
os.makedirs(output_dir, exist_ok=True)

# Base path for the 16kHz files
base_16k_path = '/home/aik2/sc-rawnet3/datasets/hooktheory/audio_16k/wav'

# Path to your data file
data_file = '/home/aik2/sc-rawnet3/datasets/hooktheory/cartesia-dataset-dec_10th-hooktheory_18k_melody_cartesia_44k_outputs_v1_with_full_metadata.jsonl'  # Update this with your actual file path
df = load_dataframe(data_file)

if df is not None:
    # 2. Create a set of all artists in the dataset
    artists = set(df['artist'].unique())
    print(f"Found {len(artists)} unique artists")
    
    # 3. Create a dictionary that assigns a 5-digit unique singer ID to each artist
    singer_id_dict = {}
    for i, artist in enumerate(sorted(artists)):
        # Create ID in the format "idXXXXX" where XXXXX is a zero-padded 5-digit number
        singer_id = f"id{i+1:05d}"
        singer_id_dict[artist] = singer_id
    
    # 4. Create a dictionary mapping singer IDs to lists of audio file paths
    singer_audio_paths = {singer_id: [] for singer_id in singer_id_dict.values()}
    
    # Populate the dictionary with audio file paths
    for _, row in df.iterrows():
        artist = row['artist']
        audio_id = row['audio_id']
        singer_id = singer_id_dict[artist]
        
        # Create the audio file path
        audio_path = f"/home/aik2/gcs-mount/cartesia-dataset/dec_10th/hooktheory_18k_melody_cartesia_44k_outputs/{audio_id}/vocals.wav"
        
        # Add the path to the list for this singer ID
        singer_audio_paths[singer_id].append(audio_path)
    
    # Print some statistics
    print(f"Created mappings for {len(singer_id_dict)} artists")
    total_paths = sum(len(paths) for paths in singer_audio_paths.values())
    print(f"Total audio paths: {total_paths}")
    
    # Print the first 10 entries of each dictionary
    print("\nFirst 10 entries of singer ID dictionary:")
    for i, (artist, singer_id) in enumerate(list(singer_id_dict.items())[:10]):
        print(f"{artist}: {singer_id}")
    
    print("\nFirst 10 entries of audio paths dictionary:")
    for i, (singer_id, paths) in enumerate(list(singer_audio_paths.items())[:10]):
        print(f"{singer_id}: {len(paths)} paths")
        if paths and i < 3:  # Show example paths for first 3 entries only
            print(f"  Example path: {paths[0]}")
    
    # Create a reverse mapping from singer ID to artist name
    id_to_artist_dict = {singer_id: artist for artist, singer_id in singer_id_dict.items()}
    
    print("\nFirst 10 entries of ID to artist mapping:")
    for i, (singer_id, artist) in enumerate(list(id_to_artist_dict.items())[:10]):
        print(f"{singer_id}: {artist}")
    
    # Create a comprehensive data structure with train/test split
    singer_data = {}
    test_count = 0
    train_count = 0
    
    for artist, singer_id in singer_id_dict.items():
        # Get all audio paths for this singer
        paths = singer_audio_paths[singer_id]
        
        # Convert paths to the new format with train/test split
        processed_paths = []
        for path in paths:
            # Randomly assign 10% to test set (is_train=0) and 90% to train set (is_train=1)
            is_train = 0 if random.random() < 0.1 else 1
            
            # Count for statistics
            if is_train == 0:
                test_count += 1
            else:
                train_count += 1
            
            # Extract the song_id (folder name) from the original path
            # e.g., from "/path/to/bWgMwEPPolX/vocals.wav" extract "bWgMwEPPolX"
            song_id = Path(path).parent.name
            
            # Construct the path to the 16kHz downsampled file
            path_16k = f"{base_16k_path}/{singer_id}/{song_id}/00001.wav"
                
            # Add as dictionary with path, path_16k and is_train flag
            processed_paths.append({
                "path": path,
                "path_16k": path_16k,
                "is_train": is_train
            })
        
        # Store in singer_data
        singer_data[singer_id] = {
            "artist_name": artist,
            "audio_paths": processed_paths
        }
    
    print("\nFirst 10 entries of comprehensive singer data with train/test split and 16kHz paths:")
    for i, (singer_id, data) in enumerate(list(singer_data.items())[:10]):
        print(f"{singer_id}: {data['artist_name']}")
        print(f"  Audio paths ({len(data['audio_paths'])} total):")
        # Print up to 5 paths per artist to avoid overwhelming output
        for path_data in data['audio_paths'][:5]:
            print(f"    - Original: {path_data['path']}")
            print(f"      16kHz: {path_data['path_16k']}")
            print(f"      is_train: {path_data['is_train']}")
        if len(data['audio_paths']) > 5:
            print(f"    ... and {len(data['audio_paths']) - 5} more paths")
        print()  # Add a blank line between entries for readability
    
    # Print train/test statistics
    print(f"\nTrain/Test Split Statistics:")
    print(f"Train set: {train_count} files ({train_count/total_paths*100:.1f}%)")
    print(f"Test set: {test_count} files ({test_count/total_paths*100:.1f}%)")
    
    # Save the new data structure with absolute paths
    split_file_path = os.path.join(output_dir, 'singer_data_split.json')
    complete_file_path = os.path.join(output_dir, 'singer_data_complete.json')
    
    try:
        with open(split_file_path, 'w') as f:
            json.dump(singer_data, f, indent=2)
        print(f"Successfully saved to {split_file_path}")
    except Exception as e:
        print(f"Error saving split file: {e}")
    
    # Also save the original comprehensive data for reference
    try:
        with open(complete_file_path, 'w') as f:
            json.dump({
                singer_id: {
                    "artist_name": data["artist_name"],
                    "audio_paths": [path_data["path"] for path_data in data["audio_paths"]]
                } for singer_id, data in singer_data.items()
            }, f, indent=2)
        print(f"Successfully saved to {complete_file_path}")
    except Exception as e:
        print(f"Error saving complete file: {e}")
    
    print("\nSaved mappings to:")
    print(f"- {complete_file_path} (original format with paths as strings)")
    print(f"- {split_file_path} (new format with train/test split and 16kHz paths)")
else:
    print("Failed to load dataframe")