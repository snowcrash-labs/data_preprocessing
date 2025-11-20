#!/usr/bin/env python3
"""
Preprocessing pipeline script that executes all preprocessing steps in order.
Takes 5 common arguments and maps them appropriately to each script.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent


def extract_dataset_path_from_csv_gs_path(csv_gs_path: str) -> Path:
    """Extract dataset path from CSV GCS path."""
    # Extract CSV filename from GCS path
    csv_filename = Path(csv_gs_path).name
    csv_stem = csv_filename.rsplit(".", 1)[0] if "." in csv_filename else csv_filename
    
    # Create dataset path: ~/gs_imports/{csv_stem}
    home_dir = Path.home()
    dataset_path = home_dir / "gs_imports" / csv_stem
    
    return dataset_path


def run_command(cmd: list, step_name: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    
    if result.returncode != 0:
        print(f"\n❌ Error in {step_name}")
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)
    else:
        print(f"\n✅ {step_name} completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete preprocessing pipeline for voice dataset"
    )
    parser.add_argument(
        "--csv_gs_path",
        required=True,
        help="GCS path to CSV file (e.g. gs://bucket/path/file.csv)",
    )
    parser.add_argument(
        "--ds_gs_prefix",
        default="music-dataset-hooktheory-audio/roformer_voice_sep_custom_sample",
        help="GCS prefix for dataset (default: music-dataset-hooktheory-audio/roformer_voice_sep_custom_sample)",
    )
    parser.add_argument(
        "--uri_name_header",
        required=True,
        help="Name of CSV column header containing audio URIs",
    )
    parser.add_argument(
        "--file_name_header",
        required=True,
        help="Name of CSV column header containing track/folder names",
    )
    parser.add_argument(
        "--artist_name_header",
        required=True,
        help="Name of CSV column header containing artist names",
    )

    parser.add_argument(
        "--datasets_dir",
        default="~/gs_imports",
        help="Directory to store datasets (default: ~/gs_imports)",
    )
    parser.add_argument(
        "--skip_gs_flatten",
        action="store_true",
        help="If set, skip flattening GCS directory structure before processing",
    )
    args = parser.parse_args()
    
    # Derive dataset_path from csv_gs_path
    dataset_path = os.path.join(args.datasets_dir, os.path.basename(args.ds_gs_prefix))
    dataset_path_str = str(dataset_path)
    
    # Step 0: Flatten GCS directory structure (optional)
    if not args.skip_gs_flatten:
        # Extract bucket name and prefix from ds_gs_prefix
        # ds_gs_prefix format: "bucket-name/path/to/prefix"
        prefix_parts = args.ds_gs_prefix.split("/", 1)
        if len(prefix_parts) > 1:
            gcs_bucket = prefix_parts[0]
            gcs_prefix = prefix_parts[1] + "/" if not prefix_parts[1].endswith("/") else prefix_parts[1]
        else:
            # Fallback to defaults if format is unexpected
            gcs_bucket = args.ds_gs_prefix.split("/")[0]
            gcs_prefix = args.ds_gs_prefix.split("/")[1]
            
        run_command(
            [
                sys.executable,
                "flatten_song_level_dir_datasets.py",
                "--bucket_name", gcs_bucket,
                "--prefix", gcs_prefix,
            ],
            "0. Flatten GCS directory structure"
        )
    
    # Step 1: Split audio on silence
    run_command(
        [
            sys.executable,
            "12m_split_on_silence.py",
            "--csv_gs_path", args.csv_gs_path,
            "--uri_name_header", args.uri_name_header,
            "--ds_gs_prefix", args.ds_gs_prefix,
            "--datasets_dir", args.datasets_dir,
        ],
        "1. Split audio on silence"
    )
    
    # Step 2: Check folder CSV and create deduplicated_data.csv
    run_command(
        [
            sys.executable,
            "check_folder_csv.py",
            "--dataset_path", dataset_path_str,
            "--uri_name_header", args.uri_name_header,
        ],
        "2. Check folder CSV and deduplicate"
    )
    
    # Step 3: Assign singer IDs
    run_command(
        [
            sys.executable,
            "assign_singer_id.py",
            "--dataset_path", dataset_path_str,
            "--artist_name_header", args.artist_name_header,
        ],
        "3. Assign singer IDs"
    )
    
    # Step 4: Reorganize to singer_id directories
    # Assuming file_name_header is same as file_name_header, and singer_id_header is "singer_id"
    run_command(
        [
            sys.executable,
            "to_singer_id.py",
            "--dataset_path", dataset_path_str,
            "--file_name_header", args.file_name_header,
            "--singer_id_header", "singer_id",
        ],
        "4. Reorganize to singer_id directories"
    )
    
    # Step 5: Hash song names
    output_csv_path = dataset_path / "trackname_to_md5name_mapping.csv"
    run_command(
        [
            sys.executable,
            "hash_songnames.py",
            "--dataset_path", str(dataset_path / "desilenced_data"),
            "--output_csv_path", str(output_csv_path),
        ],
        "5. Hash song names"
    )
    
    # Step 6: Dataset split (standard 80:10:10)
    run_command(
        [
            sys.executable,
            "dataset_split.py",
            "--dataset_path", dataset_path_str,
            "--input_csv_name", "data.csv",
            "--output_csv_name", "subset_split.csv",
            "--artist_name_header", args.artist_name_header,
            "--singer_id_header", "singer_id",
        ],
        "6. Dataset split (train/val/test)"
    )
    
    # Step 7: Dataset split (custom)
    # run_command(
    #     [
    #         sys.executable,
    #         "dataset_split_custom.py",
    #         "--dataset_path", dataset_path_str,
    #         "--input_csv_name", "deduplicated_data.csv",
    #         "--output_csv_name", "train_test_split_custom.csv",
    #         "--artist_name_header", args.artist_name_header,
    #         "--singer_id_header", "singer_id",
    #     ],
    #     "7. Dataset split (custom)"
    # )
    
    print(f"\n{'='*60}")
    print("🎉 All preprocessing steps completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

