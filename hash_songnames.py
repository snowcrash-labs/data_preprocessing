import os
import hashlib
import csv
import argparse


def hash_name(name: str) -> str:
    """
    Generate an MD5 hash of the given name.
    """
    return hashlib.md5(name.encode("utf-8")).hexdigest()


def process_second_level_folders(base_dir: str, csv_path: str) -> None:
    """
    For each second-level subfolder under base_dir (i.e., base_dir/*/*),
    compute a hash of its folder name, rename the folder to the hash,
    and record the mapping (original path -> new path) in a CSV.
    """
    mappings = []

    # First-level directories
    for first in os.listdir(base_dir):
        first_path = os.path.join(base_dir, first)
        if not os.path.isdir(first_path):
            continue
        # Second-level directories
        for second in os.listdir(first_path):
            second_path = os.path.join(first_path, second)
            if not os.path.isdir(second_path):
                continue

            # Compute hash
            new_name = hash_name(second)
            new_path = os.path.join(first_path, new_name)

            # Handle potential name collisions
            if os.path.exists(new_path):
                # Append a short suffix to avoid collision
                suffix = 1
                while os.path.exists(new_path + f"_{suffix}"):
                    suffix += 1
                new_path = new_path + f"_{suffix}"
                new_name = f"{new_name}_{suffix}"

            # Rename folder
            os.rename(second_path, new_path)

            # Record mapping
            mappings.append({"original": second_path, "new": new_path})

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
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dataset_path) + "/audio"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Error: {base_dir} is not a valid directory.")
    else:
        print(f"Successfully found {base_dir}")

    process_second_level_folders(base_dir, args.output_csv_path)
    print(f"Renaming complete. Mapping written to {args.output_csv_path}.")


if __name__ == "__main__":
    main()
