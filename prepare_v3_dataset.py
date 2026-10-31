"""Create a source-aware, quality-controlled microbiome occurrence table for v3.

The output is deliberately a small canonical schema. Downloading datasets is kept
outside this script so every release records the exact source export and license.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TAXONOMIC_LEVELS = ["phylum", "class", "order", "family", "genus"]
REQUIRED_COLUMNS = ["decimalLatitude", "decimalLongitude", *TAXONOMIC_LEVELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="TSV/CSV source export")
    parser.add_argument("--output", type=Path, required=True, help="Canonical parquet or CSV output")
    parser.add_argument("--source", required=True, choices=["gbif", "mgnify", "emp", "normalized"])
    parser.add_argument("--country", default=None, help="Optional ISO country filter, e.g. IN")
    parser.add_argument("--max-coordinate-uncertainty-m", type=float, default=10_000)
    parser.add_argument("--minimum-year", type=int, default=2000)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    # GBIF commonly labels Darwin Core exports as .csv even when they are tab-delimited.
    with path.open(encoding="utf-8", errors="replace") as handle:
        delimiter = "\t" if "\t" in handle.readline() else ","
    return pd.read_csv(path, sep=delimiter, low_memory=False)


def find_column(frame: pd.DataFrame, *names: str) -> str | None:
    lookup = {column.lower(): column for column in frame.columns}
    return next((lookup[name.lower()] for name in names if name.lower() in lookup), None)


def canonicalize(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    aliases = {
        "decimalLatitude": ("decimalLatitude", "latitude", "lat"),
        "decimalLongitude": ("decimalLongitude", "longitude", "lon"),
        "coordinateUncertaintyInMeters": ("coordinateUncertaintyInMeters", "coordinate_uncertainty_m"),
        "countryCode": ("countryCode", "country_code"),
        "year": ("year", "collection_year"),
        "basisOfRecord": ("basisOfRecord", "sample_type"),
        "license": ("license", "license_url"),
        "study_id": ("study_id", "study_accession", "study", "bioproject"),
    }
    result = pd.DataFrame(index=frame.index)
    for target, options in aliases.items():
        column = find_column(frame, *options)
        if column:
            result[target] = frame[column]
    for level in TAXONOMIC_LEVELS:
        column = find_column(frame, level, f"{level}_name")
        if column:
            result[level] = frame[column]
    result["source"] = source
    return result


def clean(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input cannot be normalized; missing columns: {', '.join(missing)}")
    for column in ("decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters", "year"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["decimalLatitude", "decimalLongitude", "phylum", "genus"]).copy()
    frame = frame[frame.decimalLatitude.between(-90, 90) & frame.decimalLongitude.between(-180, 180)]
    if "coordinateUncertaintyInMeters" in frame:
        frame = frame[frame.coordinateUncertaintyInMeters.isna() |
                      (frame.coordinateUncertaintyInMeters <= args.max_coordinate_uncertainty_m)]
    if "year" in frame:
        frame = frame[frame.year.isna() | (frame.year >= args.minimum_year)]
    if args.country and "countryCode" in frame:
        frame = frame[frame.countryCode.fillna("").str.upper() == args.country.upper()]
    for level in TAXONOMIC_LEVELS:
        frame[level] = frame[level].astype(str).str.strip()
        frame = frame[~frame[level].isin(["", "Unknown", "nan", "None"])]
    return frame.drop_duplicates(subset=["source", "decimalLatitude", "decimalLongitude", *TAXONOMIC_LEVELS])


def main() -> None:
    args = parse_args()
    data = clean(canonicalize(read_table(args.input), args.source), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        data.to_parquet(args.output, index=False)
    else:
        data.to_csv(args.output, index=False)
    print(f"Wrote {len(data):,} validated {args.source} records to {args.output}")
    print("Keep the raw export, DOI/download date, query, and license in data/provenance/.")


if __name__ == "__main__":
    main()
