"""Audit sample metadata and download manifests for shortlisted MGnify studies.

Only JSON metadata is fetched. This creates an evidence record for deciding whether
a study can be downloaded and normalized, rather than treating an API listing as
scientific approval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

API_BASE = "https://www.ebi.ac.uk/metagenomics/api/v1/studies"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/provenance/mgnify_pilot_study_details.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/provenance/mgnify_pilot_sample_audit.csv"))
    parser.add_argument("--max-studies", type=int, default=3)
    parser.add_argument("--sample-preview", type=int, default=25, help="Metadata records inspected per study")
    return parser.parse_args()


def get_json(url: str) -> dict:
    response = requests.get(url, timeout=30, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def coordinate_count(samples: list[dict]) -> int:
    count = 0
    for sample in samples:
        attrs = sample.get("attributes", {})
        text = json.dumps(attrs).lower()
        if "latitude" in text and "longitude" in text:
            count += 1
    return count


def main() -> None:
    args = parse_args()
    studies = pd.read_csv(args.input).head(args.max_studies)
    rows = []
    for _, study in studies.iterrows():
        accession = study.study_accession
        samples = get_json(f"{API_BASE}/{accession}/samples?format=json")
        analyses = get_json(f"{API_BASE}/{accession}/analyses?format=json")
        downloads = get_json(f"{API_BASE}/{accession}/downloads?format=json")
        sample_items = samples.get("data", [])[:args.sample_preview]
        analysis_items = analyses.get("data", [])[:args.sample_preview]
        download_items = downloads.get("data", [])
        attribute_keys = sorted({key for item in sample_items for key in item.get("attributes", {})})
        download_names = []
        for item in download_items:
            attributes = item.get("attributes", {})
            download_names.append(attributes.get("name") or attributes.get("description") or item.get("id"))
        rows.append({
            "study_accession": accession,
            "sample_records_previewed": len(sample_items),
            "sample_records_reported": samples.get("meta", {}).get("pagination", {}).get("count"),
            "preview_records_with_coordinate_fields": coordinate_count(sample_items),
            "sample_attribute_keys": ";".join(attribute_keys),
            "analysis_records_reported": analyses.get("meta", {}).get("pagination", {}).get("count"),
            "analysis_attribute_keys": ";".join(sorted({key for item in analysis_items for key in item.get("attributes", {})})),
            "analysis_relationship_keys": ";".join(sorted({key for item in analysis_items for key in item.get("relationships", {})})),
            "analysis_preview_raw_json": json.dumps(analysis_items[:1], sort_keys=True),
            "download_manifest_entries": len(download_items),
            "download_manifest_names": ";".join(str(name) for name in download_names),
            "download_manifest_raw_json": json.dumps(download_items, sort_keys=True),
            "coordinate_review": "pending_manual_value_check",
            "license_review": "pending_source_license_check",
            "taxonomy_table_review": "pending_download_manifest_check",
            "decision": "pending",
        })
    audit = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    print(f"Wrote sample/download audit for {len(audit)} studies to {args.output}")


if __name__ == "__main__":
    main()
