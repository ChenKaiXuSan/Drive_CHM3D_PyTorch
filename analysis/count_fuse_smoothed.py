#!/usr/bin/env python3
"""Count fused/smoothed file counts per person and environment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CountRow:
    person: str
    env: str
    fused_count: int
    smoothed_count: int

    @property
    def match(self) -> bool:
        return self.fused_count == self.smoothed_count


def count_files(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file())


def iter_rows(root: Path) -> Iterable[CountRow]:
    for person_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        for env_dir in sorted((p for p in person_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            fused_dir = env_dir / "fused_npz"
            smoothed_dir = env_dir / "smoothed_fused_npz"
            yield CountRow(
                person=person_dir.name,
                env=env_dir.name,
                fused_count=count_files(fused_dir),
                smoothed_count=count_files(smoothed_dir),
            )


def print_report(rows: list[CountRow]) -> None:
    header = ["person", "env", "fused_count", "smoothed_count", "match"]
    print(",".join(header))

    total_fused = 0
    total_smoothed = 0

    for row in rows:
        total_fused += row.fused_count
        total_smoothed += row.smoothed_count
        print(
            f"{row.person},{row.env},{row.fused_count},{row.smoothed_count},"
            f"{'YES' if row.match else 'NO'}"
        )

    print(
        f"TOTAL,-,{total_fused},{total_smoothed},"
        f"{'YES' if total_fused == total_smoothed else 'NO'}"
    )


def write_csv(rows: list[CountRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total_fused = sum(r.fused_count for r in rows)
    total_smoothed = sum(r.smoothed_count for r in rows)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["person", "env", "fused_count", "smoothed_count", "match"])

        for row in rows:
            writer.writerow(
                [
                    row.person,
                    row.env,
                    row.fused_count,
                    row.smoothed_count,
                    "YES" if row.match else "NO",
                ]
            )

        writer.writerow(
            [
                "TOTAL",
                "-",
                total_fused,
                total_smoothed,
                "YES" if total_fused == total_smoothed else "NO",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count files under person/env/fused_npz and person/env/smoothed_fused_npz, "
            "then compare counts."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="/work/SSR/share/data/drive/head3d_fuse_results",
        help="Root directory that contains person folders.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")

    rows = list(iter_rows(root))
    print_report(rows)

    if args.csv:
        write_csv(rows, Path(args.csv))
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
