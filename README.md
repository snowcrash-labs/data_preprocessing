# Voice Dataset Preprocessing Pipeline

Automates the complete preprocessing pipeline for voice/speaker recognition datasets, from downloading audio files from Google Cloud Storage to creating train/validation/test splits.

## Prerequisites

- Python 3.x with `google-cloud-storage`, `pydub`, `pandas`, `tqdm`
- Google Cloud Storage authentication configured
- All preprocessing scripts in the same directory

## Important: GCS Flattening Step

⚠️ **The `flatten_song_level_dir_datasets.py` script modifies your Google Cloud Storage bucket by moving files and deleting parent prefixes.** It should be run separately before the main preprocessing pipeline if your GCS structure needs flattening.

**Run it separately:**
```bash
python flatten_song_level_dir_datasets.py \
  --bucket_name "your-bucket-name" \
  --prefix "path/to/prefix/"
```

**Warning:** This script will:
- Copy files to flattened paths
- **Delete the original files** from their nested locations
- Permanently alter your GCS bucket structure

Only run this if you need to flatten nested directory structures. If your files are already at the desired level, skip this step.

## Arguments

- `--csv_gs_path`: GCS path to input CSV file (e.g., `gs://bucket/path/file.csv`)
- `--uri_name_header`: CSV column name containing audio URIs
- `--file_name_header`: CSV column name containing track/folder names
- `--artist_name_header`: CSV column name containing artist names
- `--ds_gs_prefix`: GCS prefix (default: `music-dataset-hooktheory-audio/roformer_voice_sep_custom_sample`)
- `--datasets_dir`: Local directory for datasets (default: `~/gs_imports`)

## Usage

```bash
python preprocessing.py \
  --csv_gs_path "gs://bucket/path/file.csv" \
  --uri_name_header "GCloud Url" \
  --file_name_header "local_file_name" \
  --artist_name_header "Artist"
```

## Pipeline Steps

1. **Split on silence**: Downloads and splits audio into segments
2. **Deduplicate**: Matches local files with CSV, creates `deduplicated_data.csv`
3. **Assign singer IDs**: Assigns unique IDs to artists, creates `singer_id_mapping_filtered.json`
4. **Reorganize by singer**: Moves tracks under `{singer_id}/` directories
5. **Hash song names**: Renames directories to MD5 hashes
6. **Create splits**: Splits into train/val/test (80:10:10)

## Output Structure

```
{datasets_dir}/{dataset_name}/
├── original_gs_input.csv          # Downloaded CSV from GCS
├── deduplicated_data.csv           # Filtered CSV with local file mappings
├── singer_id_mapping_filtered.json # Mapping of singer IDs to artist names
├── trackname_to_md5name_mapping.csv # Mapping of original names to hashes
├── subset_split.csv                # Dataset split assignments
└── desilenced_data/
    ├── train/
    │   ├── id00001/
    │   │   ├── {hash1}/
    │   │   │   ├── 00001.wav
    │   │   │   └── 00002.wav
    │   │   └── {hash2}/
    │   └── id00002/
    ├── val/
    │   └── {singer_ids}/
    └── test/
        └── {singer_ids}/
```

