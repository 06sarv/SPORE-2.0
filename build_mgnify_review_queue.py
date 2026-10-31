"""Prioritize global MGnify soil-study candidates for human scientific review.

This is intentionally conservative. It ranks metadata evidence but never approves
or downloads a study: coordinate, licence, sample, and taxonomy checks remain
mandatory before records become training input.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

INCLUDE_TERMS = ("soil", "edaphic", "pedosphere", "grassland", "forest soil", "agricultural")
BACTERIAL_TERMS = ("bacter", "archaea", "16s")
EXCLUDE_TERMS = ("fungi", "fungal", "rhizosphere", "root", "river", "marine", "sediment", "wastewater")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/provenance/mgnify_soil_catalog.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/provenance/mgnify_global_review_queue.csv"))
    parser.add_argument("--minimum-samples", type=int, default=20)
    return parser.parse_args()


def classify(row: pd.Series, minimum_samples: int) -> pd.Series:
    text = " ".join(str(row.get(column, "")) for column in ("study_name", "study_abstract")).lower()
    sample_count = pd.to_numeric(row.get("sample_count"), errors="coerce")
    reasons = []
    score = 0.0
    if any(term in text for term in EXCLUDE_TERMS):
        return pd.Series({"recommendation": "exclude_or_split", "priority_score": 0.0,
                          "review_reason": "Metadata suggests non-bulk-soil or incompatible habitat."})
    if not any(term in text for term in INCLUDE_TERMS):
        return pd.Series({"recommendation": "needs_habitat_review", "priority_score": 0.0,
                          "review_reason": "Soil habitat is not explicit in study metadata."})
    score += 3
    reasons.append("soil-focused metadata")
    if any(term in text for term in BACTERIAL_TERMS):
        score += 3
        reasons.append("bacterial/archaeal signal")
    else:
        reasons.append("taxonomy method still needs confirmation")
    if pd.notna(sample_count) and sample_count >= minimum_samples:
        score += min(3.0, math.log10(float(sample_count) / minimum_samples + 1) * 2)
        reasons.append(f"{int(sample_count)} samples")
    else:
        return pd.Series({"recommendation": "too_small_or_review", "priority_score": score,
                          "review_reason": "; ".join(reasons + ["insufficient sample count"])})
    return pd.Series({"recommendation": "priority_review", "priority_score": round(score, 2),
                      "review_reason": "; ".join(reasons)})


def main() -> None:
    args = parse_args()
    studies = pd.read_csv(args.input)
    ranked = studies.join(studies.apply(classify, axis=1, minimum_samples=args.minimum_samples))
    ranked = ranked.sort_values(["recommendation", "priority_score"], ascending=[True, False])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    print(ranked.recommendation.value_counts().to_string())
    print(f"Wrote global review queue to {args.output}")


if __name__ == "__main__":
    main()
