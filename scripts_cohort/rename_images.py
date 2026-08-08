#!/usr/bin/env python3
"""Rename and re-save images in-place (optionally remove originals).

Usage examples:
  # Single file -> new name, then remove original
  python scripts_cohort/rename_images.py --path /path/to/img.jpg --new-name newname.jpg --remove-old

  # Batch rename in a directory with prefix and zero-padded index
  python scripts_cohort/rename_images.py --dir /path/to/dir --pattern '*.png' --prefix img_ --start-index 1 --padding 3 --remove-old

This script reads each image using Pillow, writes it back under a new filename
in the same folder (preserving format when possible), and optionally deletes
the original file after a successful save. By default it performs a dry run
unless `--remove-old` is provided for deletion; use `--dry-run` to inspect.
"""

from pathlib import Path
import argparse
from PIL import Image
import sys


def save_image(src_path: Path, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        fmt = im.format
        try:
            im.save(dest_path, format=fmt)
        except Exception:
            im.save(dest_path)


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    base = path.stem
    suf = path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{base}_{i}{suf}")
        if not candidate.exists():
            return candidate
        i += 1


def process_single(path: Path, new_name: str, remove_old: bool, dry_run: bool):
    if not path.exists():
        print(f"Source not found: {path}")
        return
    dest = path.with_name(new_name)
    dest = unique_target(dest)
    print(f"Will save: {path} -> {dest}")
    if dry_run:
        return
    save_image(path, dest)
    if remove_old:
        path.unlink()


def process_dir(folder: Path, pattern: str, prefix: str, start_index: int, padding: int, remove_old: bool, dry_run: bool):
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"No files matched in {folder} for pattern {pattern}")
        return
    idx = start_index
    for p in files:
        if p.is_dir():
            continue
        ext = p.suffix
        new_name = f"{prefix}{str(idx).zfill(padding)}{ext}"
        dest = p.with_name(new_name)
        dest = unique_target(dest)
        print(f"Will save: {p} -> {dest}")
        if not dry_run:
            save_image(p, dest)
            if remove_old:
                p.unlink()
        idx += 1


def main():
    p = argparse.ArgumentParser(description="Rename and re-save images in-place")
    p.add_argument("--path", help="Single image file path")
    p.add_argument("--new-name", help="New filename (for single file) e.g. new.jpg")
    p.add_argument("--dir", help="Directory for batch rename")
    p.add_argument("--pattern", default="*.png", help="Glob pattern for batch (default: '*.png')")
    p.add_argument("--prefix", default="", help="Prefix for batch new names")
    p.add_argument("--start-index", type=int, default=1, help="Starting index for batch numbering")
    p.add_argument("--padding", type=int, default=3, help="Zero-padding width for index (default 3)")
    p.add_argument("--remove-old", action="store_true", help="Remove original files after successful save")
    p.add_argument("--dry-run", action="store_true", help="Print actions without writing or deleting files")

    args = p.parse_args()

    if not args.path and not args.dir:
        p.print_help()
        sys.exit(1)

    if args.path:
        if not args.new_name:
            print("--new-name is required when using --path")
            sys.exit(1)
        process_single(Path(args.path), args.new_name, args.remove_old, args.dry_run)

    if args.dir:
        process_dir(Path(args.dir), args.pattern, args.prefix, args.start_index, args.padding, args.remove_old, args.dry_run)


if __name__ == "__main__":
    main()
