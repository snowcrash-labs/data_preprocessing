import os
import hashlib
import csv
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def hash_name(name: str) -> str:
    """
    Generate an MD5 hash of the given name.
    """
    return hashlib.md5(name.encode("utf-8")).hexdigest()


def rename_folder(args_tuple: tuple) -> dict:
    """
    Rename a single folder to its hash.
    Returns a dict with 'original' and 'new' paths, or None if failed.
    """
    first_path, second = args_tuple
    second_path = os.path.join(first_path, second)
    
    if not os.path.isdir(second_path):
        return None
    
    # Compute hash
    new_name = hash_name(second)
    new_path = os.path.join(first_path, new_name)
    
    # Handle potential name collisions
    if os.path.exists(new_path):
        suffix = 1
        while os.path.exists(new_path + f"_{suffix}"):
            suffix += 1
        new_path = new_path + f"_{suffix}"
        new_name = f"{new_name}_{suffix}"
    
    try:
        os.rename(second_path, new_path)
        return {"original": second_path, "new": new_path}
    except Exception as e:
        print(f"Error renaming {second_path}: {e}")
        return None


def process_second_level_folders(base_dir: str, csv_path: str, parallel: bool = True) -> None:
    """
    For each second-level subfolder under base_dir (i.e., base_dir/*/*),
    compute a hash of its folder name, rename the folder to the hash,
    and record the mapping (original path -> new path) in a CSV.
    """
    # Collect all work items first
    work_items = []
    
    first_level_dirs = [
        os.path.join(base_dir, first)
        for first in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, first))
    ]
    print(f"Found {len(first_level_dirs)} first-level directories")
    
    # Collect all second-level directories
    print("Collecting second-level directories...")
    for first_path in tqdm(first_level_dirs, desc="Scanning directories"):
        try:
            for second in os.listdir(first_path):
                second_path = os.path.join(first_path, second)
                if os.path.isdir(second_path):
                    work_items.append((first_path, second))
        except Exception as e:
            print(f"Error scanning {first_path}: {e}")
    
    print(f"Found {len(work_items)} song directories to rename")
    
    mappings = []
    
    if parallel:
        # Parallel processing with ThreadPoolExecutor (I/O bound)
        num_workers = min(32, multiprocessing.cpu_count() * 2)
        print(f"Processing with {num_workers} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(rename_folder, item): item for item in work_items}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Renaming directories"):
                result = future.result()
                if result:
                    mappings.append(result)
    else:
        # Sequential processing
        print("Processing sequentially...")
        for item in tqdm(work_items, desc="Renaming directories"):
            result = rename_folder(item)
            if result:
                mappings.append(result)
    
    print(f"Renamed {len(mappings)} song directories")

    # Write mapping CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["original", "new"])
        writer.writeheader()
        for row in mappings:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Hash and rename second-level subfolders, output mapping CSV."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the dataset directory containing first-level folders",
    )
    parser.add_argument(
        "--output_csv_path",
        required=True,
        help="Path to output CSV mapping file",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help="Disable parallel processing (process files sequentially)",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dataset_path) + "/audio"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Error: {base_dir} is not a valid directory.")
    else:
        print(f"Successfully found {base_dir}")

    parallel = not getattr(args, 'no_parallel', False)
    process_second_level_folders(base_dir, args.output_csv_path, parallel=parallel)
    print(f"Renaming complete. Mapping written to {args.output_csv_path}.")


if __name__ == "__main__":
    main()
