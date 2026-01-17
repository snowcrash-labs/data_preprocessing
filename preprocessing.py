#!/usr/bin/env python3
"""
Preprocessing pipeline script that executes all preprocessing steps in order.
Takes 5 common arguments and maps them appropriately to each script.
"""

import argparse
import subprocess
import sys
import os
import random
import numpy as np
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

def run_command(cmd: list, step_name: str):
    """Run a command and handle errors. Exits immediately on any failure."""
    print(f"\n{'='*60}")
    print(f"Step: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, check=False)
        
        if result.returncode != 0:
            print(f"\n❌ Error in {step_name}")
            print(f"Command failed with exit code {result.returncode}")
            print(f"Pipeline stopped. Fix the error before continuing.")
            sys.exit(1)
        else:
            print(f"\n✅ {step_name} completed successfully")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error in {step_name}")
        print(f"Script not found: {e}")
        print(f"Pipeline stopped.")
        sys.exit(1)
    except subprocess.SubprocessError as e:
        print(f"\n❌ Error in {step_name}")
        print(f"Subprocess error: {e}")
        print(f"Pipeline stopped.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error in {step_name}")
        print(f"Error: {e}")
        print(f"Pipeline stopped.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete preprocessing pipeline for voice dataset"
    )
    parser.add_argument(
        "--csv_gs_path",
        required=True,
        # default="gs://music-dataset-hooktheory-audio/roformer_voice_sep_custom_sample/roformer_voice_sep_custom_sample.csv",
        help="GCS path to CSV file (e.g. gs://bucket/path/file.csv)",
    )
    parser.add_argument(
        "--ds_gs_prefix",
        required=True,
        # default="music-dataset-hooktheory-audio/roformer_voice_sep_custom_sample",
        help="GCS prefix for dataset",
    )
    parser.add_argument(
        "--uri_name_header",
        default="GCloud Url",
        help="Name of CSV column header containing audio URIs",
    )
    parser.add_argument(
        "--file_name_header",
        default="local_file_name",
        help="Name of CSV column header containing track/folder names",
    )
    parser.add_argument(
        "--artist_name_header",
        default="Artist",
        help="Name of CSV column header containing artist names",
    )

    parser.add_argument(
        "--datasets_dir",
        required=True,
        # default="/home/brendanoconnor/gs_imports",
        help="Directory to store datasets)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Starting step number (1-8). Steps before this will be skipped. Default: 1",
    )
    parser.add_argument(
        "--stop_step",
        type=int,
        default=8,
        help="Stopping step number (1-8). Steps after this will be skipped. Default: 8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--reference_dataset_path",
        type=str,
        default=None,
        help="Optional path to reference dataset to mimic split structure. If provided, singer IDs will be assigned to the same splits as in the reference dataset.",
    )
    parser.add_argument(
        "--singer_id_mapping_json",
        type=str,
        default=None,
        help="Optional path to JSON file with pre-existing singer ID mappings. JSON should have singer_id keys with 'lowercase' and 'variations' nested dicts. If provided, uses this mapping instead of generating new IDs.",
    )
    parser.add_argument(
        "--siqi_exp_split",
        action="store_true",
        help="Use Siqi's train/test/exp split method instead of standard dataset_split.py. "
             "Test set from singers with 2-5 songs, exp set sampled from various song count ranges.",
    )
    parser.add_argument(
        "--siqi_test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio when using --siqi_split or --siqi_exp_split (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--siqi_exp_samples_per_range",
        type=int,
        default=10,
        help="Number of singers to sample per song count range for exp set when using --siqi_exp_split (default: 10)",
    )
    parser.add_argument(
        "--siqi_singer_data_json",
        type=str,
        default=None,
        help="Optional path to singer_data_complete.json for Siqi split. Adds extra metadata to output.",
    )
    parser.add_argument(
        "--gs_file_uri_in_csv",
        action="store_true",
        default=False,
        help="Whether the CSV contains the GCS file URI (or otherwise, the relevant folder to the file)",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        required=True,
        help="Target sample rate in Hz",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help="Disable parallel processing in all steps (process files sequentially)",
    )
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Random seed set to {args.seed} for reproducibility")
    
    # Validate step arguments
    if args.step < 1 or args.step > 8:
        print(f"❌ Error: --step must be between 1 and 8, got {args.step}")
        sys.exit(1)
    if args.stop_step < 1 or args.stop_step > 8:
        print(f"❌ Error: --stop_step must be between 1 and 8, got {args.stop_step}")
        sys.exit(1)
    if args.step > args.stop_step:
        print(f"❌ Error: --step ({args.step}) cannot be greater than --stop_step ({args.stop_step})")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Running preprocessing steps {args.step} to {args.stop_step} (inclusive)")
    print(f"{'='*60}\n")
    
    # Derive dataset_path from csv_gs_path
    dataset_path = Path(args.datasets_dir) / os.path.basename(args.ds_gs_prefix)
    dataset_path_str = str(dataset_path)
    
    # Helper to add --no-parallel flag if set
    no_parallel_flag = ['--no-parallel'] if getattr(args, 'no_parallel', False) else []
    
    # Step 1: Download and resample from GCS
    if args.step <= 1 <= args.stop_step:
        cmd = [
            sys.executable,
            "gs_download_resample.py",
            "--csv_gs_path", args.csv_gs_path,
            "--uri_name_header", args.uri_name_header,
            "--ds_gs_prefix", args.ds_gs_prefix,
            "--local_datasets_dir", args.datasets_dir,
            "--target_sample_rate", str(args.target_sample_rate),
            *(['--gs_file_uri_in_csv'] if args.gs_file_uri_in_csv else []),
            *no_parallel_flag,
        ]
        run_command(
            cmd,
            "1. Download and resample from GCS"
        )
    
    # Step 2: Split audio on silence
    if args.step <= 2 <= args.stop_step:
        run_command(
            [
                sys.executable,
                "desilence_split.py",
                "--dataset_path", dataset_path_str,
            ],
            "2. Split audio on silence"
        )
    
    # Step 3: Check folder CSV and create deduplicated_data.csv
    if args.step <= 3 <= args.stop_step:
        cmd = [
            sys.executable,
            "check_folder_csv.py",
            "--dataset_path", dataset_path_str,
            "--uri_name_header", args.uri_name_header,
            "--seed", str(args.seed),
            *(['--gs_file_uri_in_csv'] if args.gs_file_uri_in_csv else []),
        ]
        run_command(
            cmd,
            "3. Check folder CSV and deduplicate"
        )
    
    # Step 4: Assign singer IDs
    if args.step <= 4 <= args.stop_step:
        cmd = [
            sys.executable,
            "assign_singer_id.py",
            "--dataset_path", dataset_path_str,
            "--artist_name_header", args.artist_name_header,
            *no_parallel_flag,
        ]
        # Add singer ID mapping JSON if provided
        if args.singer_id_mapping_json:
            cmd.extend(["--singer_id_mapping_json", args.singer_id_mapping_json])
        
        run_command(
            cmd,
            "4. Assign singer IDs"
        )
    
    # Step 5: Reorganize to singer_id directories
    # Assuming file_name_header is same as file_name_header, and singer_id_header is "singer_id"
    if args.step <= 5 <= args.stop_step:
        run_command(
            [
                sys.executable,
                "to_singer_id.py",
                "--dataset_path", dataset_path_str,
                "--file_name_header", args.file_name_header,
                "--singer_id_header", "singer_id",
                *no_parallel_flag,
            ],
            "5. Reorganize to singer_id directories"
        )
    
    # Step 6: Hash song names
    if args.step <= 6 <= args.stop_step:
        output_csv_path = dataset_path / "trackname_to_md5name_mapping.csv"
        run_command(
            [
                sys.executable,
                "hash_songnames.py",
                "--dataset_path", str(dataset_path),
                "--output_csv_path", str(output_csv_path),
                *no_parallel_flag,
            ],
            "6. Hash song names"
        )
    
    # Step 7: Dataset split (standard 80:10:10, Siqi's 90:10, Siqi's exp split, or matching reference dataset)
    if args.step <= 7 <= args.stop_step:
        if args.siqi_exp_split:
            # Use Siqi's train/test/exp split (test from 2-5 songs, exp sampled from ranges)
            cmd = [
                sys.executable,
                "siqi_train_test_exp_split_singer.py",
                "--dataset_path", dataset_path_str,
                "--input_csv_name", "data.csv",
                "--artist_name_header", args.artist_name_header,
                "--singer_id_header", "singer_id",
                "--seed", str(args.seed),
                "--test_ratio", str(args.siqi_test_ratio),
                "--exp_samples_per_range", str(args.siqi_exp_samples_per_range),
                *no_parallel_flag,
            ]
            # Add singer data JSON if provided
            if args.siqi_singer_data_json:
                cmd.extend(["--singer_data_json", args.siqi_singer_data_json])
            
            run_command(
                cmd,
                "7. Dataset split - Siqi exp method (train/test/exp)"
            )
        else:
            # Use standard dataset_split.py (80:10:10)
            cmd = [
                sys.executable,
                "dataset_split.py",
                "--dataset_path", dataset_path_str,
                "--input_csv_name", "data.csv",
                "--artist_name_header", args.artist_name_header,
                "--singer_id_header", "singer_id",
                "--seed", str(args.seed),
                *no_parallel_flag,
            ]
            # Add reference dataset path if provided
            if args.reference_dataset_path:
                cmd.extend(["--reference_dataset_path", args.reference_dataset_path])
            
            run_command(
                cmd,
                "7. Dataset split (train/val/test 80:10:10)"
            )
    
    # Step 8: Create test pairs
    if args.step <= 8 <= args.stop_step:
        test_dir = dataset_path / "audio" / "test"
        output_pairs_path = dataset_path / "test_pairs.txt"
        subset_split_csv = dataset_path / "data.csv"
        
        run_command(
            [
                sys.executable,
                "make_test_pairs.py",
                "--csv_path", str(subset_split_csv),
                "--test_dir", str(test_dir),
                "--output_path", str(output_pairs_path),
                "--singer_id_header", "singer_id",
                "--split_header", "split",
                "--file_name_header", args.file_name_header,
                "--seed", str(args.seed),
            ],
            "8. Create test pairs"
        )
    
    print(f"\n{'='*60}")
    print(f"🎉 Preprocessing steps {args.step} to {args.stop_step} completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error in preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

