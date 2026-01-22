"""
Split dataset into train, test, and exp sets by singer.

This script splits singers into train, test (10%), and exp sets.
Test set: Samples from singers with 2-5 songs.
Exp set: Samples 10 singers from each song count range (1, 2-5, 5-10, 10-30, 30-100, 100+).
Train set: All remaining singers.

All songs from the same singer are kept together in the same split.

Can be used as an alternative to dataset_split.py or siqis_train_test_split_singer.py.
"""

import argparse
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train, test, and exp sets by singer. "
                    "Test set from singers with 2-5 songs, exp set sampled from various ranges."
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
        "--exp_samples_per_range",
        type=int,
        default=10,
        help="Number of singers to sample per song count range for exp set (default: 10)",
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
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (run sequentially)",
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
    exp_dir = audio_dir / "exp"
    
    print(f"Dataset path: {dataset_path}")
    print(f"Input CSV: {input_csv_path}")
    print(f"Audio directory: {audio_dir}")
    
    # Create output directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    print("Loading dataframe...")
    df = pd.read_csv(input_csv_path)
    print(f"CSV loaded with {len(df)} rows")
    
    # Get singer counts
    singer_id_counts = df.groupby(args.singer_id_header).agg({
        args.artist_name_header: ['count', lambda x: x.str.lower().unique()]
    }).reset_index()
    
    # Flatten column names
    singer_id_counts.columns = [args.singer_id_header, 'song_count', 'artist_names']
    
    total_singers = len(singer_id_counts)
    print(f"Total unique singer IDs: {total_singers}")
    
    if total_singers == 0:
        print("Error: No singer IDs found in CSV")
        return 1
    
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
    
    # Calculate the number needed for test set
    num_test_singers = int(total_singers * args.test_ratio)
    num_test_singers = max(1, num_test_singers)  # At least 1 singer in test
    print(f"Target test singers: {num_test_singers}")
    
    # Randomly sample test singers from singers with 2-5 songs
    singers_2_5 = singer_id_counts[singer_id_counts['song_count'].between(2, 5)]
    print(f"Singers with 2-5 songs: {len(singers_2_5)}")
    
    if len(singers_2_5) < num_test_singers:
        print(f"Warning: Not enough singers with 2-5 songs ({len(singers_2_5)}) for test set ({num_test_singers})")
        test_singers = singers_2_5[args.singer_id_header].tolist()
    else:
        test_singers = singers_2_5[args.singer_id_header].sample(
            n=num_test_singers, random_state=args.seed
        ).tolist()
    
    print(f"Selected {len(test_singers)} singers for test set")
    
    # From the remaining singers, sample for exp set
    remaining_singers = singer_id_counts[~singer_id_counts[args.singer_id_header].isin(test_singers)]
    
    # Define the ranges and sample from each
    exp_singers = []
    ranges = [
        (1, 1, "1 song"),
        (2, 5, "2-5 songs"),
        (5, 10, "5-10 songs"),
        (10, 30, "10-30 songs"),
        (30, 100, "30-100 songs"),
        (100, float('inf'), "100+ songs")
    ]
    
    for min_songs, max_songs, description in ranges:
        if max_songs == float('inf'):
            range_singers = remaining_singers[remaining_singers['song_count'] >= min_songs]
        else:
            range_singers = remaining_singers[remaining_singers['song_count'].between(min_songs, max_songs)]
        
        print(f"Singers with {description}: {len(range_singers)}")
        
        if len(range_singers) >= args.exp_samples_per_range:
            sampled = range_singers[args.singer_id_header].sample(
                n=args.exp_samples_per_range, random_state=args.seed
            ).tolist()
        else:
            sampled = range_singers[args.singer_id_header].tolist()
            print(f"  Warning: Only {len(sampled)} singers available, taking all")
        
        exp_singers.extend(sampled)
        print(f"  Selected {len(sampled)} singers for exp set")
    
    print(f"Total exp singers: {len(exp_singers)}")
    
    # Get unique singer IDs for iteration
    unique_singer_ids = df[args.singer_id_header].unique()
    
    # Train singers are all singers not in test or exp sets
    train_singers = [s for s in unique_singer_ids if s not in test_singers and s not in exp_singers]
    
    print(f"Final train set size: {len(train_singers)} singers")
    print(f"Final test set size: {len(test_singers)} singers")
    print(f"Final exp set size: {len(exp_singers)} singers")
    
    # Create split mapping
    singer_split_map = {}
    for singer_id in train_singers:
        singer_split_map[singer_id] = 'train'
    for singer_id in test_singers:
        singer_split_map[singer_id] = 'test'
    for singer_id in exp_singers:
        singer_split_map[singer_id] = 'exp'
    
    # Find singer directories in audio folder
    exclude_dirs = {'train', 'test', 'exp'}
    singer_dirs_on_disk = set()
    if audio_dir.exists():
        singer_dirs_on_disk = {
            d.name for d in audio_dir.iterdir()
            if d.is_dir() and d.name not in exclude_dirs
        }
    print(f"Singer directories found on disk: {len(singer_dirs_on_disk)}")
    
    # Counters for statistics
    train_files_moved = 0
    test_files_moved = 0
    exp_files_moved = 0
    skipped = 0
    errors = 0
    
    # Move/copy files to their respective directories
    operation_name = "Copying" if args.copy_files else "Moving"
    print(f"\n{operation_name} files to train/test/exp directories...")
    
    # Helper function to process a single singer
    def process_singer(singer_id):
        """Process a single singer directory. Returns (split_type, wav_count, status)."""
        src_singer_dir = audio_dir / singer_id
        
        # Determine destination
        split_type = singer_split_map.get(singer_id, 'train')
        if split_type == 'train':
            dest_singer_dir = train_dir / singer_id
        elif split_type == 'test':
            dest_singer_dir = test_dir / singer_id
        else:  # exp
            dest_singer_dir = exp_dir / singer_id
        
        # Check if source directory exists
        if not src_singer_dir.exists():
            return (split_type, 0, 'no_source')
        
        # Check if destination already exists (skip if so)
        if dest_singer_dir.exists():
            # Count existing files for stats
            wav_count = len(list(dest_singer_dir.rglob("*.wav")))
            return (split_type, wav_count, 'skipped')
        
        try:
            if args.copy_files:
                # Copy entire directory tree
                shutil.copytree(src_singer_dir, dest_singer_dir)
            else:
                # Move entire directory
                shutil.move(str(src_singer_dir), str(dest_singer_dir))
            
            # Count files moved/copied
            wav_count = len(list(dest_singer_dir.rglob("*.wav")))
            return (split_type, wav_count, 'success')
            
        except Exception as e:
            return (split_type, 0, f'error: {str(e)}')
    
    # Filter to only singers that need processing
    singers_to_process = list(unique_singer_ids)
    
    if getattr(args, 'no_parallel', False):
        # Sequential processing
        print("Running in sequential mode...")
        for singer_id in tqdm(singers_to_process, desc="Processing singers"):
            split_type, wav_count, status = process_singer(singer_id)
            
            if status == 'success':
                if split_type == 'train':
                    train_files_moved += wav_count
                elif split_type == 'test':
                    test_files_moved += wav_count
                else:
                    exp_files_moved += wav_count
            elif status == 'skipped':
                skipped += 1
                # Still count the files for statistics
                if split_type == 'train':
                    train_files_moved += wav_count
                elif split_type == 'test':
                    test_files_moved += wav_count
                else:
                    exp_files_moved += wav_count
            elif status.startswith('error'):
                print(f"Error processing {singer_id}: {status}")
                errors += 1
    else:
        # Parallel processing
        print("Running in parallel mode...")
        max_workers = min(32, len(singers_to_process))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_singer = {
                executor.submit(process_singer, singer_id): singer_id
                for singer_id in singers_to_process
            }
            
            # Process results with progress bar
            with tqdm(total=len(singers_to_process), desc="Processing singers") as pbar:
                for future in as_completed(future_to_singer):
                    singer_id = future_to_singer[future]
                    try:
                        split_type, wav_count, status = future.result()
                        
                        if status == 'success':
                            if split_type == 'train':
                                train_files_moved += wav_count
                            elif split_type == 'test':
                                test_files_moved += wav_count
                            else:
                                exp_files_moved += wav_count
                        elif status == 'skipped':
                            skipped += 1
                            # Still count the files for statistics
                            if split_type == 'train':
                                train_files_moved += wav_count
                            elif split_type == 'test':
                                test_files_moved += wav_count
                            else:
                                exp_files_moved += wav_count
                        elif status.startswith('error'):
                            tqdm.write(f"Error processing {singer_id}: {status}")
                            errors += 1
                    except Exception as e:
                        tqdm.write(f"Exception processing {singer_id}: {str(e)}")
                        errors += 1
                    
                    pbar.update(1)
    
    if skipped > 0:
        print(f"Skipped {skipped} singers (destination already exists)")
    
    # Update CSV with split information
    print("\nUpdating CSV with split information...")
    
    def assign_split(singer_id):
        return singer_split_map.get(singer_id, 'train')
    
    df['split'] = df[args.singer_id_header].apply(assign_split)
    
    # Convert to numeric (0=train, 1=test, 2=exp)
    df['split'] = df['split'].map({'train': 0, 'test': 1, 'exp': 2})
    
    # Print split statistics from CSV
    train_songs = (df['split'] == 0).sum()
    test_songs = (df['split'] == 1).sum()
    exp_songs = (df['split'] == 2).sum()
    
    print(f"\nDataset split statistics:")
    print(f"Train songs: {train_songs} ({train_songs/len(df)*100:.1f}%)")
    print(f"Test songs: {test_songs} ({test_songs/len(df)*100:.1f}%)")
    print(f"Exp songs: {exp_songs} ({exp_songs/len(df)*100:.1f}%)")
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
    print(f"Exp singers: {len(exp_singers)}")
    print(f"Train files: {train_files_moved}")
    print(f"Test files: {test_files_moved}")
    print(f"Exp files: {exp_files_moved}")
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
