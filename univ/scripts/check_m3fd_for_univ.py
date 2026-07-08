#!/usr/bin/env python3
"""Lightweight M3FD directory/split checker for UNIV experiments."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check M3FD files required by UNIV experiments.")
    parser.add_argument("--root", "--data", dest="root", default="/home/jinlei/database/M3FD_Detection",
                        help="M3FD_Detection root directory.")
    return parser.parse_args()


def count_files(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.is_file()) if path.is_dir() else 0


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser()
    print(f"Checking M3FD root: {root}", flush=True)
    ok = True
    for sub in ["vi", "ir", "labels", "Annotation", "meta"]:
        path = root / sub
        exists = path.exists()
        ok &= exists
        extra = f", files={count_files(path)}" if path.is_dir() else ""
        print(f"{sub}: {'OK' if exists else 'MISSING'}{extra}", flush=True)
    for split in ["train.txt", "val.txt", "test.txt"]:
        path = root / "meta" / split
        exists = path.is_file()
        ok &= exists
        lines = sum(1 for _ in path.open()) if exists else 0
        print(f"meta/{split}: {'OK' if exists else 'MISSING'}, lines={lines}", flush=True)
    print("Summary: " + ("PASS" if ok else "FAILED"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
