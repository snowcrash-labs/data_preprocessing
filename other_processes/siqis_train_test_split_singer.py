"""
Split dataset into training and test sets by singer (90:10).

This script splits singers into train (90%) and test (10%) sets,
prioritizing singers with 2-4 songs for the test set.
All songs from the same singer are kept together in the same split.

Can be used as an alternative to dataset_split.py when --siqi_split is provided.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
import time
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train and test sets by singer (90:10), "
                    "prioritizing singers with 2-4 songs for test set."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the dataset directory (e.g. ~/gs_imports/roformer_voice_sep_custom_sample)",
    )
    parser.add_argument(
        "--input_csv_name",
        default="data.csv",
        help="Name of the input CSV file (appended to dataset_path). This file will be edited in place.",
    )
    parser.add_argument(
        "--artist_name_header",
        default="Artist",
        help="Name of the CSV column header containing artist names",
    )
    parser.add_argument(
        "--singer_id_header",
        default="singer_id",
        help="Name of the CSV column header containing singer IDs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of singers to use for test set (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--min_songs",
        type=int,
        default=2,
        help="Minimum songs per singer for test set priority selection (default: 2)",
    )
    parser.add_argument(
        "--max_songs",
        type=int,
        default=4,
        help="Maximum songs per singer for test set priority selection (default: 4)",
    )
    parser.add_argument(
        "--singer_data_json",
        type=str,
        default=None,
        help="Optional path to singer_data_complete.json for additional metadata. "
             "If not provided, metadata will be derived from the CSV and directory structure.",
    )
    parser.add_argument(
        "--copy_files",
        action="store_true",
        help="Copy files instead of moving them (default: move files)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Start timing
    start_time = time.time()
    
    # Determine paths
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    input_csv_path = dataset_path / args.input_csv_name
    audio_dir = dataset_path / "audio"
    train_dir = audio_dir / "train"
    test_dir = audio_dir / "test"
    
    print(f"Dataset path: {dataset_path}")
    print(f"Input CSV: {input_csv_path}")
    print(f"Audio directory: {audio_dir}")
    
    # Create output directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV (required, same as dataset_split.py)
    print("Loading dataframe...")
    df = pd.read_csv(input_csv_path)
    print(f"CSV loaded with {len(df)} rows")
    
    # Get unique singer IDs from CSV (same as dataset_split.py)
    unique_singer_ids = df[args.singer_id_header].unique()
    total_singers = len(unique_singer_ids)
    print(f"Total unique singer IDs: {total_singers}")
    
    # Load singer data JSON if provided
    singer_data = {}
    if args.singer_data_json:
        singer_data_path = Path(args.singer_data_json).expanduser().resolve()
        if singer_data_path.exists():
            try:
                with open(singer_data_path, 'r') as f:
                    singer_data = json.load(f)
                print(f"Successfully loaded singer data from {singer_data_path}")
            except Exception as e:
                print(f"Warning: Error loading singer data: {e}")
        else:
            print(f"Warning: Singer data JSON not found at {singer_data_path}")
    
    # Find singer directories in audio folder (for song counting and file moving)
    # Exclude train/test/exp directories
    exclude_dirs = {'train', 'test', 'exp'}
    singer_dirs_on_disk = set()
    if audio_dir.exists():
        singer_dirs_on_disk = {
            d.name for d in audio_dir.iterdir() 
            if d.is_dir() and d.name not in exclude_dirs
        }
    print(f"Singer directories found on disk: {len(singer_dirs_on_disk)}")
    
    if total_singers == 0:
        print("Error: No singer IDs found in CSV")
        return 1
    
    # Count songs per singer (from directory structure for priority logic)
    singer_song_counts = {}
    for singer_id in unique_singer_ids:
        singer_path = audio_dir / singer_id
        if singer_path.exists():
            # Count subdirectories (songs) or wav files directly
            song_dirs = [d for d in singer_path.iterdir() if d.is_dir()]
            if song_dirs:
                singer_song_counts[singer_id] = len(song_dirs)
            else:
                # Count wav files if no subdirectories
                wav_files = list(singer_path.glob("*.wav"))
                singer_song_counts[singer_id] = len(wav_files) if wav_files else 1
        else:
            # Singer in CSV but not on disk - count from CSV
            singer_song_counts[singer_id] = len(df[df[args.singer_id_header] == singer_id])
    
    # Calculate target number of test singers
    ntest = int(total_singers * args.test_ratio)
    ntest = max(1, ntest)  # At least 1 singer in test
    print(f"Target number of singers for test set: {ntest}")
    
    # Find singers with preferred song count range (2-4 by default)
    singers_preferred = [
        singer_id for singer_id, count in singer_song_counts.items()
        if args.min_songs <= count <= args.max_songs
    ]
    print(f"Singers with {args.min_songs}-{args.max_songs} songs: {len(singers_preferred)}")
    
    # Randomly select test singers, prioritizing those with preferred song counts
    random.shuffle(singers_preferred)
    test_singers = singers_preferred[:ntest]
    
    # If not enough preferred singers, select from remaining
    if len(test_singers) < ntest:
        remaining_singers = [
            s for s in unique_singer_ids 
            if s not in test_singers and s not in singers_preferred
        ]
        random.shuffle(remaining_singers)
        test_singers += remaining_singers[:ntest - len(test_singers)]
    
    # Train singers are all singers not in test set
    train_singers = [s for s in unique_singer_ids if s not in test_singers]
    
    print(f"Final test set size: {len(test_singers)} singers")
    print(f"Final train set size: {len(train_singers)} singers")
    
    # Create split mapping
    singer_split_map = {}
    for singer_id in train_singers:
        singer_split_map[singer_id] = 'train'
    for singer_id in test_singers:
        singer_split_map[singer_id] = 'test'
    
    # Counters for statistics
    train_files_moved = 0
    test_files_moved = 0
    errors = 0
    
    # Move/copy files to their respective directories
    operation_name = "Copying" if args.copy_files else "Moving"
    print(f"\n{operation_name} files to train/test directories...")
    
    processed = 0
    for singer_id in unique_singer_ids:
        src_singer_dir = audio_dir / singer_id
        
        # Determine destination
        if singer_id in train_singers:
            dest_singer_dir = train_dir / singer_id
        else:
            dest_singer_dir = test_dir / singer_id
        
        # Check if source directory exists and isn't already in train/test
        if not src_singer_dir.exists():
            continue
            
        try:
            if args.copy_files:
                # Copy entire directory tree
                if dest_singer_dir.exists():
                    shutil.rmtree(dest_singer_dir)
                shutil.copytree(src_singer_dir, dest_singer_dir)
                
                # Count files copied
                wav_count = len(list(dest_singer_dir.rglob("*.wav")))
                if singer_id in train_singers:
                    train_files_moved += wav_count
                else:
                    test_files_moved += wav_count
            else:
                # Move entire directory
                shutil.move(str(src_singer_dir), str(dest_singer_dir))
                
                # Count files moved
                wav_count = len(list(dest_singer_dir.rglob("*.wav")))
                if singer_id in train_singers:
                    train_files_moved += wav_count
                else:
                    test_files_moved += wav_count
            
            processed += 1
            # Print progress every 100 singers
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {processed}/{total_singers} singers processed")
                
        except Exception as e:
            print(f"Error processing {singer_id}: {str(e)}")
            errors += 1
    
    # Update CSV with split information (same approach as dataset_split.py)
    print("\nUpdating CSV with split information...")
    
    def assign_split(singer_id):
        return singer_split_map.get(singer_id, 'train')
    
    df['split'] = df[args.singer_id_header].apply(assign_split)
    
    # Convert to numeric (0=train, 1=test) - note: no exp split in this version
    df['split'] = df['split'].map({'train': 0, 'test': 1})
    
    # Print split statistics from CSV
    train_songs = (df['split'] == 0).sum()
    test_songs = (df['split'] == 1).sum()
    print(f"\nDataset split statistics:")
    print(f"Train songs: {train_songs} ({train_songs/len(df)*100:.1f}%)")
    print(f"Test songs: {test_songs} ({test_songs/len(df)*100:.1f}%)")
    print(f"Total songs: {len(df)}")
    
    # Save the dataframe back to the input csv file (overwrites in place)
    df.to_csv(input_csv_path, index=False)
    print(f"Updated input CSV file with split info: {input_csv_path}")
    
    # Create split_by_singer.json if singer_data was provided
    if singer_data:
        print("\nCreating split_by_singer.json...")
        split_by_singer = {}
        
        for singer_id, info in singer_data.items():
            if singer_id not in unique_singer_ids:
                continue
            
            split_type = singer_split_map.get(singer_id, "train")
            artist_name = info.get("artist_name", "Unknown")
            
            split_by_singer[singer_id] = {
                "artist_name": artist_name,
                "split": split_type,
                "audio_paths": info.get("audio_paths", [])
            }
        
        split_by_singer_path = dataset_path / "split_by_singer.json"
        try:
            with open(split_by_singer_path, "w") as f:
                json.dump(split_by_singer, f, indent=2)
            print(f"Successfully saved {split_by_singer_path}")
        except Exception as e:
            print(f"Error saving split_by_singer.json: {e}")
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Split by singer complete!")
    print("=" * 60)
    print(f"Train singers: {len(train_singers)}")
    print(f"Test singers: {len(test_singers)}")
    print(f"Train files: {train_files_moved}")
    print(f"Test files: {test_files_moved}")
    print(f"Errors encountered: {errors}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
