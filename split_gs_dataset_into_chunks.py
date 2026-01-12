#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from google.cloud import storage
from tqdm import tqdm


@dataclass
class Obj:
    name: str
    size: int


def human_bytes(n: int) -> str:
    # Simple human-readable formatter (base 1024)
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    for s in suffixes:
        if f < 1024.0 or s == suffixes[-1]:
            return f"{f:.2f} {s}"
        f /= 1024.0
    return f"{f:.2f} B"


def greedy_binpack(objs: List[Obj], k: int) -> List[List[Obj]]:
    """
    Greedy bin packing by size (descending), assign each object to the chunk with current minimum total.
    Produces fairly balanced chunks by bytes.
    """
    objs_sorted = sorted(objs, key=lambda o: o.size, reverse=True)
    bins: List[List[Obj]] = [[] for _ in range(k)]
    totals = [0] * k

    for o in objs_sorted:
        idx = min(range(k), key=lambda i: totals[i])
        bins[idx].append(o)
        totals[idx] += o.size

    return bins


def load_done_set(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(line)
    return done


def append_done(path: str, dst_name: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(dst_name + "\n")


def copy_blob_with_retries(
    bucket: storage.Bucket,
    src_name: str,
    dst_name: str,
    max_attempts: int = 8,
    base_sleep: float = 1.0,
) -> None:
    """
    Server-side copy within the same bucket (Rewrite API under the hood).
    Retries transient errors.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            src_blob = bucket.blob(src_name)
            # copy_blob returns a Blob; for large objects, google-cloud-storage handles rewrite loops internally.
            bucket.copy_blob(src_blob, bucket, new_name=dst_name)
            return
        except Exception as e:
            last_err = e
            # Exponential backoff with jitter
            sleep_s = base_sleep * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 60.0)
            time.sleep(sleep_s + (0.1 * attempt))
    raise RuntimeError(f"Failed copying {src_name} -> {dst_name}: {last_err}") from last_err


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split a GCS prefix into K chunks by total bytes and copy into chunk_X prefixes."
    )
    ap.add_argument("--bucket", default="12m-youtube", help="GCS bucket name (default: 12m-youtube)")
    ap.add_argument(
        "--src-prefix",
        default="roformer_vocal_stems/",
        help="Source prefix within bucket (ensure trailing slash).",
    )
    ap.add_argument(
        "--dst-prefix",
        default="roformer_vocal_stem_chunks/",
        help="Destination prefix within bucket (ensure trailing slash).",
    )
    ap.add_argument("--chunks", type=int, default=10, help="Number of chunks (default: 10)")
    ap.add_argument(
        "--manifest",
        default="chunk_manifest.jsonl",
        help="Where to write assignments (JSONL).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not copy, only compute and write manifest.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume copying using per-chunk done logs (chunk_X.done).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="For testing: limit number of objects listed (0 = no limit).",
    )
    args = ap.parse_args()

    if not args.src_prefix.endswith("/"):
        args.src_prefix += "/"
    if not args.dst_prefix.endswith("/"):
        args.dst_prefix += "/"

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    # 1) List objects under src prefix
    print(f"Listing gs://{args.bucket}/{args.src_prefix} ...", file=sys.stderr)
    objs: List[Obj] = []
    it = client.list_blobs(args.bucket, prefix=args.src_prefix)
    for i, b in enumerate(it, start=1):
        # Skip "directory placeholder" objects if any (rare in GCS, but possible)
        if b.name.endswith("/") and (b.size == 0):
            continue
        objs.append(Obj(name=b.name, size=int(b.size or 0)))
        if args.limit and i >= args.limit:
            break

    if not objs:
        print("No objects found under source prefix. Exiting.", file=sys.stderr)
        return 2

    total_bytes = sum(o.size for o in objs)
    print(
        f"Found {len(objs):,} objects, total {human_bytes(total_bytes)}",
        file=sys.stderr,
    )

    # 2) Assign to chunks by bytes
    bins = greedy_binpack(objs, args.chunks)
    totals = [sum(o.size for o in b) for b in bins]

    print("\nPlanned chunk sizes:", file=sys.stderr)
    for i, t in enumerate(totals, start=1):
        print(f"  chunk_{i}: {human_bytes(t)} ({t/total_bytes*100:.2f}%)", file=sys.stderr)

    # 3) Write manifest (JSONL): one line per object, includes src and dst
    #    Also useful for audit / later transfer.
    with open(args.manifest, "w", encoding="utf-8") as f:
        for i, b in enumerate(bins, start=1):
            chunk_name = f"chunk_{i}"
            for o in b:
                rel = o.name[len(args.src_prefix):]  # strip source prefix
                dst_name = f"{args.dst_prefix}{chunk_name}/{rel}"
                f.write(
                    json.dumps(
                        {
                            "chunk": chunk_name,
                            "src": f"gs://{args.bucket}/{o.name}",
                            "dst": f"gs://{args.bucket}/{dst_name}",
                            "size": o.size,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"\nWrote manifest: {args.manifest}", file=sys.stderr)

    if args.dry_run:
        print("Dry-run enabled; not copying.", file=sys.stderr)
        return 0

    # 4) Copy per chunk
    #    Resume support uses chunk_X.done containing destination object names (within bucket).
    for i, b in enumerate(bins, start=1):
        chunk_name = f"chunk_{i}"
        done_path = f"{chunk_name}.done"
        done = load_done_set(done_path) if args.resume else set()

        print(f"\n==> Copying {chunk_name}: {len(b):,} objects", file=sys.stderr)
        pbar = tqdm(total=len(b), unit="obj", desc=chunk_name)

        # If resuming, advance bar for already done objects in this chunk:
        if done:
            already = 0
            for o in b:
                rel = o.name[len(args.src_prefix):]
                dst_name = f"{args.dst_prefix}{chunk_name}/{rel}"
                if dst_name in done:
                    already += 1
            if already:
                pbar.update(already)

        for o in b:
            rel = o.name[len(args.src_prefix):]
            dst_name = f"{args.dst_prefix}{chunk_name}/{rel}"

            if args.resume and dst_name in done:
                continue

            copy_blob_with_retries(bucket, o.name, dst_name)
            append_done(done_path, dst_name)
            pbar.update(1)

        pbar.close()
        print(f"Completed {chunk_name}. Done log: {done_path}", file=sys.stderr)

    print("\nAll chunks copied successfully.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
