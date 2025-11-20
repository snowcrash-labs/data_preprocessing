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
        required=True,
        # default="~/gs_imports",
        help="Directory to store datasets)",
    )
    args = parser.parse_args()
    
    # Derive dataset_path from csv_gs_path
    dataset_path = Path(args.datasets_dir) / os.path.basename(args.ds_gs_prefix)
    dataset_path_str = str(dataset_path)
    
    # Step 1: Split audio on silence
    run_command(
        [
            sys.executable,
            "12m_split_on_silence.py",
            "--csv_gs_path", args.csv_gs_path,
            "--uri_name_header", args.uri_name_header,
            "--ds_gs_prefix", args.ds_gs_prefix,
            "--local_datasets_dir", args.datasets_dir,
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

