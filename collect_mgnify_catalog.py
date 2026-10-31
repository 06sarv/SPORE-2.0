"""Inventory public MGnify soil studies without downloading sequence or sample data.

The resulting CSV is a review queue, not a training dataset. A study must pass
licence, coordinate, habitat, and taxonomy-method review before its data is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

DEFAULT_URL = (
    "https://www.ebi.ac.uk/metagenomics/api/v1/biomes/"
    "root:Environmental:Terrestrial:Soil/studies?format=json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/provenance/mgnify_soil_catalog.csv"))
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--url", default=DEFAULT_URL, help="MGnify study-list endpoint")
    return parser.parse_args()


def request_page(url: str) -> dict:
    response = requests.get(url, timeout=30, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    url = args.url
    rows = []
    for _ in range(args.max_pages):
        payload = request_page(url)
        for item in payload.get("data", []):
            attributes = item.get("attributes", {})
            accession = attributes.get("accession", item.get("id"))
            rows.append({
                "source": "mgnify",
                "study_accession": accession,
                "bioproject": attributes.get("bioproject"),
                "study_name": attributes.get("study-name"),
                "sample_count": attributes.get("samples-count"),
                "last_update": attributes.get("last-update"),
                "study_abstract": attributes.get("study-abstract"),
                "api_url": f"https://www.ebi.ac.uk/metagenomics/api/v1/studies/{accession}?format=json",
                "review_status": "pending",
                "license_status": "verify_at_source",
                "coordinates_status": "verify_per_sample",
                "method_status": "verify_per_analysis",
                "inclusion_reason": "",
                "exclusion_reason": "",
            })
        url = payload.get("links", {}).get("next")
        if not url:
            break

    result = pd.DataFrame(rows).drop_duplicates(subset=["study_accession"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} MGnify soil-study candidates to {args.output}")
    print("Review this queue before downloading any samples, analyses, or taxonomy tables.")


if __name__ == "__main__":
    main()
