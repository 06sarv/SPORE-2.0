"""Create a run-to-geolocated-sample map from public MGnify metadata.

This is a prerequisite for safely joining a study taxonomy matrix to sample
coordinates. It deliberately fetches metadata only, not the taxonomy matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

API_BASE = "https://www.ebi.ac.uk/metagenomics/api/v1/studies"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--pipeline", default="5.0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def with_page_size(url: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.setdefault("format", "json")
    query.setdefault("page_size", "1000")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def get_all(url: str) -> list[dict]:
    items: list[dict] = []
    next_url = with_page_size(url)
    while next_url:
        response = requests.get(next_url, timeout=60, headers={"Accept": "application/json"})
        response.raise_for_status()
        payload = response.json()
        items.extend(payload.get("data", []))
        next_url = payload.get("links", {}).get("next")
        if next_url:
            next_url = with_page_size(next_url)
    return items


def main() -> None:
    args = parse_args()
    samples = get_all(f"{API_BASE}/{args.study}/samples")
    analyses = get_all(f"{API_BASE}/{args.study}/analyses")
    sample_rows = []
    for item in samples:
        attrs = item.get("attributes", {})
        sample_rows.append({
            "sample_accession": item.get("id"),
            "latitude": attrs.get("latitude"),
            "longitude": attrs.get("longitude"),
            "environment_biome": attrs.get("environment-biome"),
            "environment_feature": attrs.get("environment-feature"),
            "environment_material": attrs.get("environment-material"),
            "collection_date": attrs.get("collection-date"),
        })
    mapping_rows = []
    for item in analyses:
        attrs = item.get("attributes", {})
        relations = item.get("relationships", {})
        run = relations.get("run", {}).get("data") or {}
        sample = relations.get("sample", {}).get("data") or {}
        if attrs.get("pipeline-version") == args.pipeline and run.get("id") and sample.get("id"):
            mapping_rows.append({"run_accession": run["id"], "sample_accession": sample["id"],
                                 "analysis_accession": item.get("id"), "pipeline_version": args.pipeline})
    mapping = pd.DataFrame(mapping_rows).drop_duplicates(subset=["run_accession", "sample_accession"])
    output = mapping.merge(pd.DataFrame(sample_rows), on="sample_accession", how="left", validate="many_to_one")
    output.insert(0, "study_accession", args.study)
    output["latitude"] = pd.to_numeric(output["latitude"], errors="coerce")
    output["longitude"] = pd.to_numeric(output["longitude"], errors="coerce")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    located = output.latitude.notna().sum() if not output.empty else 0
    print(f"Wrote {len(output):,} run mappings; {located:,} have latitude/longitude.")


if __name__ == "__main__":
    main()
