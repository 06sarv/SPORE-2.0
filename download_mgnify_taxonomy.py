"""Download one reviewed public MGnify taxonomy matrix with provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--license-reference", required=True,
                        help="Reviewed public-data policy or record-level licence URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.license_reference.startswith("https://"):
        raise ValueError("--license-reference must be an HTTPS URL recorded in provenance")
    audit = json.loads(args.schema_audit.read_text())
    url = audit["file_url"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    digest = hashlib.sha256()
    with requests.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
    temporary.replace(args.output)
    provenance = {
        "study_accession": audit["study_accession"],
        "source_url": url,
        "license_reference": args.license_reference,
        "sha256": digest.hexdigest(),
        "bytes": args.output.stat().st_size,
    }
    provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2))
    print(f"Downloaded {args.output.stat().st_size:,} bytes to {args.output}")
    print(f"SHA-256: {provenance['sha256']}")


if __name__ == "__main__":
    main()
