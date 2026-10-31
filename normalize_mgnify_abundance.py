"""Normalize an MGnify wide taxonomy matrix into sample-level long abundances."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RANK_PREFIXES = {"p": "phylum", "c": "class", "o": "order", "f": "family", "g": "genus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-mapped-runs", type=float, default=0.95)
    return parser.parse_args()


def parse_lineage(lineage: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {rank: None for rank in RANK_PREFIXES.values()}
    for token in str(lineage).split(";"):
        prefix, _, name = token.partition("__")
        rank = RANK_PREFIXES.get(prefix[-1:])
        if rank and name:
            result[rank] = name.replace("_", " ").strip()
    return result


def main() -> None:
    args = parse_args()
    matrix = pd.read_csv(args.matrix, sep="\t", index_col=0)
    matrix.index.name = "taxonomy_original"
    run_map = pd.read_csv(args.run_map)
    matrix_runs = pd.Index(matrix.columns)
    mapped_runs = matrix_runs.intersection(run_map.run_accession.astype(str))
    fraction = len(mapped_runs) / len(matrix_runs) if len(matrix_runs) else 0
    if fraction < args.minimum_mapped_runs:
        raise ValueError(f"Only {fraction:.1%} of matrix runs map to sample metadata")
    matrix = matrix.loc[:, mapped_runs]
    long = matrix.reset_index().melt(id_vars="taxonomy_original", var_name="run_accession", value_name="abundance")
    long["abundance"] = pd.to_numeric(long.abundance, errors="coerce").fillna(0).astype("int64")
    long = long[long.abundance > 0]
    long = long.merge(run_map, on="run_accession", how="inner", validate="many_to_one")
    parsed = pd.DataFrame([parse_lineage(value) for value in long.taxonomy_original], index=long.index)
    long = pd.concat([long, parsed], axis=1)
    long["source"] = "mgnify"
    long["taxonomy_format"] = "MGnify SSU lineage"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(args.output, index=False)
    print(f"Wrote {len(long):,} non-zero sample-taxon abundances to {args.output}")
    print(f"Mapped {len(mapped_runs):,}/{len(matrix_runs):,} matrix run columns ({fraction:.1%}).")


if __name__ == "__main__":
    main()
