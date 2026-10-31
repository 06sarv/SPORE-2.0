"""Fetch study-level metadata for the highest-priority MGnify review candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("data/provenance/mgnify_global_review_queue.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/provenance/mgnify_pilot_study_details.csv"))
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args()


def get_json(url: str) -> dict:
    response = requests.get(url, timeout=30, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    queue = pd.read_csv(args.queue)
    candidates = queue[queue.recommendation == "priority_review"].nlargest(args.limit, "priority_score")
    rows = []
    for _, candidate in candidates.iterrows():
        accession = candidate.study_accession
        payload = get_json(f"https://www.ebi.ac.uk/metagenomics/api/v1/studies/{accession}?format=json")
        item = payload.get("data", {})
        attributes = item.get("attributes", {})
        relationships = item.get("relationships", {})
        rows.append({
            "study_accession": accession,
            "study_name": attributes.get("study-name"),
            "sample_count": attributes.get("samples-count"),
            "bioproject": attributes.get("bioproject"),
            "last_update": attributes.get("last-update"),
            "available_relationships": ";".join(sorted(relationships)),
            "has_samples_link": "samples" in relationships,
            "has_analyses_link": "analyses" in relationships,
            "has_coordinates_link": "geocoordinates" in relationships,
            "has_downloads_link": "downloads" in relationships,
            "study_api_url": f"https://www.ebi.ac.uk/metagenomics/api/v1/studies/{accession}?format=json",
            "license_review": "required",
            "sample_coordinate_review": "required",
            "taxonomy_table_review": "required",
            "decision": "pending",
            "raw_relationships_json": json.dumps(relationships, sort_keys=True),
        })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote metadata for {len(result)} pilot studies to {args.output}")


if __name__ == "__main__":
    main()
