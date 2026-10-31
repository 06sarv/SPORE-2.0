"""Inspect metadata and a bounded header from an MGnify taxonomy TSV.

The script uses a byte-range request to avoid downloading the full study file.
It does not approve a source licence or create model training rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("data/provenance/mgnify_pilot_sample_audit.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/provenance/mgnify_taxonomy_schema_audit.json"))
    parser.add_argument("--preview-bytes", type=int, default=20_000)
    parser.add_argument("--study", help="Study accession from the pilot sample audit")
    return parser.parse_args()


def bounded_preview(url: str, preview_bytes: int) -> tuple[list[str], dict[str, str]]:
    # Streaming caps local transfer even if an upstream service ignores Range.
    with requests.get(url, timeout=30, stream=True, headers={"Range": f"bytes=0-{preview_bytes - 1}"}) as response:
        response.raise_for_status()
        chunk = next(response.iter_content(chunk_size=preview_bytes), b"")
        headers = {key: value for key, value in response.headers.items()
                   if key.lower() in {"content-type", "content-range", "content-length", "accept-ranges"}}
    return chunk.decode("utf-8", errors="replace").splitlines()[:3], headers


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.audit)
    row = audit[audit.study_accession == args.study].iloc[0] if args.study else audit.iloc[0]
    items = json.loads(row.download_manifest_raw_json)
    selected = next(item for item in items if "Taxonomic assignments SSU" in str(item.get("attributes", {})))
    metadata_url = selected["links"]["self"]
    header_preview, response_headers = bounded_preview(metadata_url, args.preview_bytes)
    result = {
        "study_accession": row.study_accession,
        "file_id": selected["id"],
        "metadata_url": metadata_url,
        "file_url": metadata_url,
        "response_headers": response_headers,
        "header_preview": header_preview,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote taxonomy-schema audit to {args.output}")


if __name__ == "__main__":
    main()
