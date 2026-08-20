#!/usr/bin/env python3
"""Generate the committed E001 repeatability result from local run manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nvfp4_doctor.report import summarize_e001_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_e001_manifests(tuple(args.manifests))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"summary={args.output}")
    print(f"repetitions={result['repetitions']}")
    print(f"gate0_repeatability={result['gate0_repeatability']}")
    print(f"decision={result['decision']}")


if __name__ == "__main__":
    main()
