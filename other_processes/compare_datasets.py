#!/usr/bin/env python3
"""
Compare subdirectories between two datasets and identify differences.

Produces two lists:
1. Subdirectories in dataset A that are not in dataset B
2. Subdirectories in dataset B that are not in dataset A
"""

import argparse
import shutil
from pathlib import Path
from typing import Set, List


def get_subdirectories(dataset_path: Path, level: int = 1) -> Set[str]:
    """
    Get subdirectories at a specific level from a dataset path.
    
    Args:
        dataset_path: Path to the dataset directory
        level: Depth level to check (1 = immediate subdirectories, 2 = second level, etc.)
    
    Returns:
        Set of subdirectory names at the specified level
    """
    subdirs = set()
    
    if not dataset_path.exists():
        print(f"Warning: Dataset path does not exist: {dataset_path}")
        return subdirs
    
    if level == 1:
        # Get immediate subdirectories
        for item in dataset_path.iterdir():
            if item.is_dir():
                subdirs.add(item.name)
    else:
        # For deeper levels, recursively collect subdirectories
        for item in dataset_path.iterdir():
            if item.is_dir():
                if level == 2:
                    # Get second-level subdirectories
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            subdirs.add(subitem.name)
                else:
                    # Recursive for deeper levels
                    subdirs.update(get_subdirectories(item, level - 1))
    
    return subdirs


def compare_datasets(
    dataset_a_path: Path,
    dataset_b_path: Path,
    level: int = 1,
    base_subdir: str = None
) -> tuple[List[str], List[str]]:
    """
    Compare subdirectories between two datasets.
    
    Args:
        dataset_a_path: Path to first dataset
        dataset_b_path: Path to second dataset
        level: Depth level to compare (1 = immediate subdirectories, 2 = second level, etc.)
        base_subdir: Optional subdirectory to compare within (e.g., 'audio', 'train', 'test', 'exp')
    
    Returns:
        Tuple of (subdirs_in_a_not_b, subdirs_in_b_not_a)
    """
    # If base_subdir is specified, append it to both paths
    if base_subdir:
        path_a = dataset_a_path / base_subdir
        path_b = dataset_b_path / base_subdir
    else:
        path_a = dataset_a_path
        path_b = dataset_b_path
    
    # Get subdirectories from both datasets
    subdirs_a = get_subdirectories(path_a, level)
    subdirs_b = get_subdirectories(path_b, level)
    
    # Find differences
    only_in_a = sorted(list(subdirs_a - subdirs_b))
    only_in_b = sorted(list(subdirs_b - subdirs_a))
    
    return only_in_a, only_in_b


def copy_exclusive_directories(
    source_dataset_path: Path,
    exclusive_subdirs: List[str],
    base_subdir: str = None,
    level: int = 1
) -> Path:
    """
    Move exclusive directories from source dataset to a new location.
    
    Args:
        source_dataset_path: Path to source dataset
        exclusive_subdirs: List of subdirectory names to move
        base_subdir: Optional subdirectory to move from (e.g., 'audio', 'train', 'test', 'exp')
        level: Depth level of the subdirectories (1 = immediate, 2 = second level, etc.)
    
    Returns:
        Path to the destination directory
    """
    if not exclusive_subdirs:
        return None
    
    # Create destination path: {source_dataset_name}_EXCLUSIVE
    dest_dataset_path = source_dataset_path.parent / f"{source_dataset_path.name}_EXCLUSIVE"
    
    # Determine source path
    if base_subdir:
        source_path = source_dataset_path / base_subdir
        dest_path = dest_dataset_path / base_subdir
    else:
        source_path = source_dataset_path
        dest_path = dest_dataset_path
    
    # Create destination directory structure
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nMoving {len(exclusive_subdirs)} exclusive directories from:")
    print(f"  Source: {source_path}")
    print(f"  Destination: {dest_path}")
    
    copied_count = 0
    failed_count = 0
    
    for subdir_name in exclusive_subdirs:
        source_dir = None
        dest_dir = None
        
        if level == 1:
            # Direct subdirectory at the specified level
            source_dir = source_path / subdir_name
            dest_dir = dest_path / subdir_name
        elif level == 2:
            # Second level - copy from all parent directories that contain this subdirectory
            found_any = False
            for parent_dir in source_path.iterdir():
                if parent_dir.is_dir():
                    subdir_path = parent_dir / subdir_name
                    if subdir_path.exists() and subdir_path.is_dir():
                        dest_parent = dest_path / parent_dir.name
                        dest_parent.mkdir(parents=True, exist_ok=True)
                        dest_dir = dest_parent / subdir_name
                        source_dir = subdir_path
                        
                        try:
                            shutil.move(str(source_dir), str(dest_dir))
                            found_any = True
                            # Continue to check other parent directories (don't break)
                        except Exception as e:
                            print(f"  ❌ Error moving {subdir_name} from {parent_dir.name}: {e}")
                            failed_count += 1
            if not found_any:
                print(f"  ⚠️  Warning: Could not find {subdir_name} at level 2")
                failed_count += 1
            else:
                copied_count += 1
                if copied_count % 100 == 0:
                    print(f"  Progress: {copied_count}/{len(exclusive_subdirs)} directories moved...")
            continue
        else:
            # For deeper levels, recursively search
            found = False
            for root, dirs, files in source_path.rglob(subdir_name):
                if root.is_dir() and root.name == subdir_name:
                    # Calculate relative path from source_path
                    rel_path = root.relative_to(source_path)
                    source_dir = root
                    dest_dir = dest_path / rel_path
                    dest_dir.parent.mkdir(parents=True, exist_ok=True)
                    found = True
                    break
            if not found:
                print(f"  ⚠️  Warning: Could not find {subdir_name} at level {level}")
                failed_count += 1
                continue
            
            # Move the directory found at deeper level
            if source_dir and source_dir.exists() and source_dir.is_dir():
                try:
                    shutil.move(str(source_dir), str(dest_dir))
                    copied_count += 1
                    if copied_count % 100 == 0:
                        print(f"  Progress: {copied_count}/{len(exclusive_subdirs)} directories moved...")
                except Exception as e:
                    print(f"  ❌ Error moving {subdir_name}: {e}")
                    failed_count += 1
            continue
        
        # Only process if not already handled by level 2 logic above
        if level != 2 and source_dir and source_dir.exists() and source_dir.is_dir():
            try:
                shutil.move(str(source_dir), str(dest_dir))
                copied_count += 1
                if copied_count % 100 == 0:
                    print(f"  Progress: {copied_count}/{len(exclusive_subdirs)} directories moved...")
            except Exception as e:
                print(f"  ❌ Error moving {subdir_name}: {e}")
                failed_count += 1
        elif level != 2:
            print(f"  ⚠️  Warning: Source directory not found: {source_dir}")
            failed_count += 1
    
    print(f"\nMoving complete:")
    print(f"  ✅ Successfully moved: {copied_count}")
    if failed_count > 0:
        print(f"  ❌ Failed: {failed_count}")
    
    return dest_dataset_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare subdirectories between two datasets"
    )
    parser.add_argument(
        "--dataset_a",
        required=True,
        help="Path to first dataset directory",
    )
    parser.add_argument(
        "--dataset_b",
        required=True,
        help="Path to second dataset directory",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Depth level to compare (1 = immediate subdirectories, 2 = second level, etc.) (default: 1)",
    )
    parser.add_argument(
        "--base_subdir",
        type=str,
        default=None,
        help="Optional subdirectory to compare within (e.g., 'audio', 'audio/train', 'audio/test', 'audio/exp'). "
             "For single-tier structure (dataset/audio/singer_id/), use 'audio'. "
             "For two-tier structure (dataset/audio/train/singer_id/), use 'audio/train' or 'audio' with --level 2.",
    )
    parser.add_argument(
        "--exclusively_a_path",
        type=str,
        default="exclusively_a.txt",
        help="Path to save list of subdirectories exclusively in dataset A",
    )
    parser.add_argument(
        "--exclusively_b_path",
        type=str,
        default="exclusively_b.txt",
        help="Path to save list of subdirectories exclusively in dataset B",
    )
    parser.add_argument(
        "--copy_exclusive",
        action="store_true",
        help="Move exclusive directories to new locations. Creates {dataset_name}_EXCLUSIVE directories.",
    )
    args = parser.parse_args()
    
    dataset_a_path = Path(args.dataset_a).expanduser().resolve()
    dataset_b_path = Path(args.dataset_b).expanduser().resolve()
    
    print(f"Comparing datasets:")
    print(f"  Dataset A: {dataset_a_path}")
    print(f"  Dataset B: {dataset_b_path}")
    if args.base_subdir:
        print(f"  Base subdirectory: {args.base_subdir}")
    print(f"  Level: {args.level}")
    print()
    
    # Compare datasets
    only_in_a, only_in_b = compare_datasets(
        dataset_a_path,
        dataset_b_path,
        level=args.level,
        base_subdir=args.base_subdir
    )
    
    # Print results
    print("=" * 60)
    print(f"Subdirectories only in Dataset A: {len(only_in_a)}")
    print("=" * 60)
    if only_in_a:
        for subdir in only_in_a:
            print(f"  {subdir}")
    else:
        print("  (none)")
    
    print()
    print("=" * 60)
    print(f"Subdirectories only in Dataset B: {len(only_in_b)}")
    print("=" * 60)
    if only_in_b:
        for subdir in only_in_b:
            print(f"  {subdir}")
    else:
        print("  (none)")
    
    # Save to files
    for output_path, subdirs in [(args.exclusively_a_path, only_in_a), (args.exclusively_b_path, only_in_b)]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:    
            for subdir in subdirs:
                f.write(f"{subdir}\n")
        print(f"\nSaved exclusively in Dataset {output_path.stem} list to: {output_path.resolve()}")
    
    # Move exclusive directories if requested
    if args.copy_exclusive:
        print()
        print("=" * 60)
        print("Moving exclusive directories...")
        print("=" * 60)
        
        if only_in_a:
            dest_a = copy_exclusive_directories(
                dataset_a_path,
                only_in_a,
                base_subdir=args.base_subdir,
                level=args.level
            )
            if dest_a:
                print(f"\n✅ Dataset A exclusives moved to: {dest_a}")
        
        if only_in_b:
            dest_b = copy_exclusive_directories(
                dataset_b_path,
                only_in_b,
                base_subdir=args.base_subdir,
                level=args.level
            )
            if dest_b:
                print(f"\n✅ Dataset B exclusives moved to: {dest_b}")
    
    # Summary
    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Only in Dataset A: {len(only_in_a)}")
    print(f"  Only in Dataset B: {len(only_in_b)}")
    if only_in_a or only_in_b:
        print("  ⚠️  Datasets have differences")
        if args.copy_exclusive:
            print("  📁 Exclusive directories have been moved to _EXCLUSIVE directories")
    else:
        print("  ✅ Datasets have identical subdirectories")


if __name__ == "__main__":
    main()

