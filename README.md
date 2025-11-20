# Voice Dataset Preprocessing Pipeline

Automates the complete preprocessing pipeline for voice/speaker recognition datasets, from downloading audio files from Google Cloud Storage to creating train/validation/test splits.

## Prerequisites

- Python 3.x with `google-cloud-storage`, `pydub`, `pandas`, `tqdm`
- Google Cloud Storage authentication configured
- All preprocessing scripts in the same directory

## Arguments

- `--csv_gs_path`: GCS path to input CSV file (e.g., `gs://bucket/path/file.csv`)
- `--uri_name_header`: CSV column name containing audio URIs
- `--file_name_header`: CSV column name containing track/folder names
- `--artist_name_header`: CSV column name containing artist names
- `--ds_gs_prefix`: GCS prefix (default: `music-dataset-hooktheory-audio/roformer_voice_sep_custom_sample`)
- `--datasets_dir`: Local directory for datasets (default: `~/gs_imports`)
- `--skip_gs_flatten`: Skip GCS flattening step

## Usage

```bash
python preprocessing.py \
  --csv_gs_path "gs://bucket/path/file.csv" \
  --uri_name_header "GCloud Url" \
  --file_name_header "local_file_name" \
  --artist_name_header "Artist"
```

## Pipeline Steps

1. **Flatten GCS** (optional): Reorganizes audio files in GCS
2. **Split on silence**: Downloads and splits audio into segments
3. **Deduplicate**: Matches local files with CSV, creates `deduplicated_data.csv`
4. **Assign singer IDs**: Assigns unique IDs to artists, creates `singer_id_mapping_filtered.json`
5. **Reorganize by singer**: Moves tracks under `{singer_id}/` directories
6. **Hash song names**: Renames directories to MD5 hashes
7. **Create splits**: Splits into train/val/test (80:10:10)

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

