## Roformer-separated hooktheory sample dataset
`python flatten_song_level_dir_datasets.py --project_id sc-music-research --bucket_name music-dataset-hooktheory-audio --prefix roformer_voice_sep_custom_sample`
`python preprocessing.py --csv_gs_path gs://music-dataset-hooktheory-audio/roformer_voice_sep_custom_sample/roformer_voice_sep_custom_sample.csv --ds_gs_prefix music-dataset-hooktheory-audio/roformer_voice_sep_custom_sample --uri_name_header "GCloud Url" --file_name_header local_file_name --artist_name_header Artist --datasets_dir "/home/brendanoconnor/gs_imports"`

## Roformer-separated hooktheory full dataset
