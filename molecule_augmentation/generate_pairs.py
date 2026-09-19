#!/usr/bin/env python
"""Generate virtual intermediate pairs for a molecular property CSV."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from .virtual_intermediates import augment_molecular_dataset
except ImportError:  # direct ``python molecule_augmentation/generate_pairs.py``
    from virtual_intermediates import augment_molecular_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with smiles and one property column")
    parser.add_argument("--output", required=True, help="Output CSV for virtual intermediate pairs")
    parser.add_argument("--property", required=True, help="Property column used to choose neighbors")
    parser.add_argument("--percentage", "--perc", dest="percentage", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs, skipped = augment_molecular_dataset(
        pd.read_csv(args.input),
        property_name=args.property,
        percentage=args.percentage,
        gamma=args.gamma,
        alpha=args.alpha,
        seed=args.seed,
        return_skipped=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output, index=False)
    print(f"Wrote {len(pairs)} virtual intermediate pairs to {output}")
    if skipped:
        summary = ", ".join(
            f"{reason}={count}" for reason, count in Counter(skipped).most_common()
        )
        print(f"Skipped {len(skipped)} failed interpolations: {summary}")


if __name__ == "__main__":
    main()
