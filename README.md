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

### Required Arguments

- `--csv_gs_path`: GCS path to input CSV file (e.g., `gs://bucket/path/file.csv`)
- `--ds_gs_prefix`: GCS prefix for the dataset
- `--datasets_dir`: Local directory for datasets (e.g., `~/gs_imports`)

### Optional Arguments

- `--uri_name_header`: CSV column name containing audio URIs (default: `GCloud Url`)
- `--file_name_header`: CSV column name containing track/folder names (default: `local_file_name`)
- `--artist_name_header`: CSV column name containing artist names (default: `Artist`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--reference_dataset_path`: Optional path to reference dataset to mimic split structure. If provided, singer IDs will be assigned to the same splits (train/test/exp) as in the reference dataset.
- `--singer_id_mapping_json`: Optional path to JSON file with pre-existing singer ID mappings. If provided, uses this mapping instead of generating new IDs. JSON should have `singer_id` keys with `lowercase` and `variations` nested dicts.
- `--step`: Starting step number (1-7). Steps before this will be skipped (default: `1`)
- `--stop_step`: Stopping step number (1-7). Steps after this will be skipped (default: `7`)

## Usage

### Basic Usage

```bash
python preprocessing.py \
  --csv_gs_path "gs://bucket/path/file.csv" \
  --ds_gs_prefix "music-dataset-hooktheory-audio/roformer_voice_sep_custom_sample" \
  --datasets_dir "~/gs_imports"
```

### Using Reference Dataset (Matching Split Structure)

To match the split structure of an existing dataset:

```bash
python preprocessing.py \
  --csv_gs_path "gs://bucket/path/file.csv" \
  --ds_gs_prefix "music-dataset-hooktheory-audio/new_dataset" \
  --datasets_dir "~/gs_imports" \
  --reference_dataset_path "~/gs_imports/existing_dataset"
```

This will assign singer IDs to the same train/test/exp splits as in the reference dataset. Unmatched singer IDs will be randomly assigned using the same seed.

### Using Pre-existing Singer ID Mapping

To use a pre-existing singer ID mapping:

```bash
python preprocessing.py \
  --csv_gs_path "gs://bucket/path/file.csv" \
  --ds_gs_prefix "music-dataset-hooktheory-audio/new_dataset" \
  --datasets_dir "~/gs_imports" \
  --singer_id_mapping_json "~/gs_imports/existing_dataset/singer_id_mapping_filtered.json"
```

### Custom Seed for Reproducibility

```bash
python preprocessing.py \
  --csv_gs_path "gs://bucket/path/file.csv" \
  --ds_gs_prefix "music-dataset-hooktheory-audio/new_dataset" \
  --datasets_dir "~/gs_imports" \
  --seed 123
```

### Running Specific Steps

To run only steps 3-6:

```bash
python preprocessing.py \
  --csv_gs_path "gs://bucket/path/file.csv" \
  --ds_gs_prefix "music-dataset-hooktheory-audio/new_dataset" \
  --datasets_dir "~/gs_imports" \
  --step 3 \
  --stop_step 6
```

## Pipeline Steps

1. **Split on silence**: Downloads audio files from GCS, splits on silence, and saves segments locally. Optionally filters using a reference dataset's trackname mapping.
2. **Deduplicate**: Matches local files with CSV entries, creates `data.csv` with `local_file_name` column
3. **Assign singer IDs**: 
   - Filters out invalid artist names (orchestras, DJs, collaborations, etc.) - **these are permanently removed from both CSV and filesystem**
   - Assigns unique IDs (id00001, id00002, ...) to each artist
   - Creates `singer_id_mapping_filtered.json`
   - Optionally uses pre-existing mapping if provided
4. **Reorganize by singer**: Moves track directories under `{singer_id}/` directories
5. **Hash song names**: Renames song directories to MD5 hashes, creates `trackname_to_md5name_mapping.csv`
6. **Create splits**: Splits into train/val/test (80:10:10) or matches reference dataset structure
7. **Create test pairs**: Creates `test_pairs.txt` for singer verification testing

## Output Structure

```
{datasets_dir}/{dataset_name}/
├── original_gs_input.csv              # Downloaded CSV from GCS
├── original_gs_input_ref_reduced.csv  # Reduced CSV if reference dataset used (step 1)
├── data.csv                            # Filtered CSV with local file mappings and singer IDs
├── singer_id_mapping_filtered.json     # Mapping of singer IDs to artist names
├── trackname_to_md5name_mapping.csv    # Mapping of original names to MD5 hashes
├── test_pairs.txt                      # Test pairs for singer verification
└── audio/
    ├── train/
    │   ├── id00001/
    │   │   ├── {hash1}/
    │   │   │   ├── 00001.wav
    │   │   │   └── 00002.wav
    │   │   └── {hash2}/
    │   └── id00002/
    ├── test/  # Validation set
    │   └── {singer_ids}/
    └── exp/   # Test set
        └── {singer_ids}/
```

## Important Notes

### Artist Filtering

⚠️ **The `assign_singer_id.py` script (Step 3) permanently removes filtered artists from your dataset.**

Artists matching these patterns are **completely removed**:
- Collaborations: `feat.`, `vs.`, `&`, `with`
- Orchestras/Ensembles: `Orchestra`, `Philharmonic`, `Symphony`, `Choir`, etc.
- DJs: `DJ`, `D.J.`
- Unknown/Anonymous artists
- Collections/Collectives

**This removal includes:**
- Removing rows from the CSV
- **Deleting audio directories from the filesystem**

Make sure you want to filter these out before running the pipeline. You can modify the filters in `assign_singer_id.py` if needed.

### Reproducibility

All randomness in the pipeline is seeded for exact reproduction. Use `--seed` to set a custom seed (default: 42). The seed affects:
- Dataset splitting (step 6)
- Test pair generation (step 7)
- Sample selection in deduplication (step 2)

### Reference Dataset Matching

When using `--reference_dataset_path`, the script will:
1. Match singer IDs that exist in both datasets to the same splits
2. Randomly assign unmatched singer IDs using the same 80:10:10 ratio
3. Ensure reproducible splits when using the same seed

This is useful for maintaining consistent train/test splits across related datasets.

