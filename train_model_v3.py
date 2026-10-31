"""Train SPORE v3 with H3 spatial validation and source-aware metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from h3 import latlng_to_cell
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

TAXONOMIC_LEVELS = ["phylum", "class", "order", "family", "genus"]
SOIL_FEATURES = ["COARSE", "SAND", "SILT", "CLAY", "ORG_CARBON", "PH_WATER", "CEC_CLAY", "BULK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True, help="Validated CSV/parquet source files")
    parser.add_argument("--soil-layers", type=Path, default=Path("data/HWSD2_LAYERS.csv"))
    parser.add_argument("--soil-raster", type=Path, default=Path("data/HWSD2.bil"))
    parser.add_argument("--environment", type=Path, help="Optional numeric H3 feature table, keyed by h3_cell")
    parser.add_argument("--output-dir", type=Path, default=Path("models_v3"))
    parser.add_argument("--h3-resolution", type=int, default=6)
    parser.add_argument("--split-h3-resolution", type=int, default=4)
    parser.add_argument("--validation-group", choices=["h3", "study"], default="h3",
                        help="Use H3 cells for geographic transfer or whole studies for study transfer")
    parser.add_argument("--study-column", default="study_id")
    parser.add_argument("--min-samples-per-class", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=800, help="Boosting rounds; use a small value only for smoke tests")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)


def raster_smu_ids(data: pd.DataFrame, raster_path: Path) -> pd.Series:
    """Sample only requested pixels instead of loading the global raster into RAM."""
    values: list[int | None] = []
    with rasterio.open(raster_path) as raster:
        coordinates = list(zip(data.decimalLongitude, data.decimalLatitude))
        for coordinate, sample in zip(coordinates, raster.sample(coordinates)):
            value = sample[0]
            if raster.nodata is not None and value == raster.nodata:
                values.append(None)
            elif np.isfinite(value) and value > 0:
                values.append(int(value))
            else:
                values.append(None)
    return pd.Series(values, index=data.index, dtype="Int64")


def aggregate_topsoil_layers(layers: pd.DataFrame) -> pd.DataFrame:
    """Avoid multiplying occurrences by every HWSD component/depth layer."""
    layers = layers.copy()
    layers["TOPDEP"] = pd.to_numeric(layers["TOPDEP"], errors="coerce")
    layers["SHARE"] = pd.to_numeric(layers["SHARE"], errors="coerce").fillna(1.0)
    minimum_depth = layers.groupby("HWSD2_SMU_ID")["TOPDEP"].transform("min")
    topsoil = layers[layers["TOPDEP"] == minimum_depth]

    rows = []
    for smu_id, group in topsoil.groupby("HWSD2_SMU_ID"):
        row = {"HWSD2_SMU_ID": smu_id}
        for feature in SOIL_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce")
            valid = values.notna() & group["SHARE"].gt(0)
            row[feature] = np.average(values[valid], weights=group.loc[valid, "SHARE"]) if valid.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def engineer_features(data: pd.DataFrame) -> list[str]:
    data["sand_clay_ratio"] = data.SAND / (data.CLAY + 1)
    data["silt_clay_ratio"] = data.SILT / (data.CLAY + 1)
    data["texture_index"] = (data.SAND * .5 + data.SILT * .3 + data.CLAY * .2) / 100
    data["carbon_ph_interaction"] = data.ORG_CARBON * data.PH_WATER
    data["ph_squared"] = data.PH_WATER ** 2
    return SOIL_FEATURES + ["sand_clay_ratio", "silt_clay_ratio", "texture_index", "carbon_ph_interaction", "ph_squared"]


def filter_taxonomy(data: pd.DataFrame, minimum: int) -> pd.DataFrame:
    result = data.copy()
    for level in TAXONOMIC_LEVELS:
        supported = result[level].value_counts()
        result = result[result[level].isin(supported[supported >= minimum].index)]
    return result


def tree_mapping(data: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    tree: dict[str, dict[str, list[str]]] = {}
    for parent, child in zip(TAXONOMIC_LEVELS, TAXONOMIC_LEVELS[1:]):
        tree[child] = data.groupby(parent)[child].agg(lambda values: sorted(set(values))).to_dict()
    return tree


def main() -> None:
    args = parse_args()
    if args.split_h3_resolution >= args.h3_resolution:
        raise ValueError("--split-h3-resolution must be coarser than --h3-resolution")
    records = pd.concat([read_table(path) for path in args.records], ignore_index=True)
    required = {"decimalLatitude", "decimalLongitude", *TAXONOMIC_LEVELS}
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Canonical records missing columns: {', '.join(missing)}")
    records["h3_cell"] = [latlng_to_cell(lat, lon, args.h3_resolution)
                          for lat, lon in zip(records.decimalLatitude, records.decimalLongitude)]
    records["split_h3_cell"] = [latlng_to_cell(lat, lon, args.split_h3_resolution)
                                for lat, lon in zip(records.decimalLatitude, records.decimalLongitude)]
    records["HWSD2_SMU_ID"] = raster_smu_ids(records, args.soil_raster)
    records = records.dropna(subset=["HWSD2_SMU_ID"]).copy()
    records.HWSD2_SMU_ID = records.HWSD2_SMU_ID.astype(int)

    layers = pd.read_csv(args.soil_layers, low_memory=False)
    soil = aggregate_topsoil_layers(layers)
    data = records.merge(soil, on="HWSD2_SMU_ID", how="left", validate="many_to_one")
    for feature in SOIL_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna(subset=SOIL_FEATURES)
    features = engineer_features(data)

    if args.environment:
        environment = read_table(args.environment)
        if "h3_cell" not in environment:
            raise ValueError("Environment table must include h3_cell")
        environment_features = [column for column in environment.columns if column != "h3_cell"]
        if not environment_features or not all(pd.api.types.is_numeric_dtype(environment[column]) for column in environment_features):
            raise ValueError("Environment table must contain numeric features besides h3_cell")
        data = data.merge(environment, on="h3_cell", how="left").dropna(subset=environment_features)
        features.extend(environment_features)

    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + TAXONOMIC_LEVELS)
    if data.empty:
        raise ValueError("No training rows remain after quality filtering")

    if args.validation_group == "study":
        if args.study_column not in data or data[args.study_column].isna().any():
            raise ValueError(f"Study validation requires a complete {args.study_column!r} column")
        validation_groups = data[args.study_column].astype(str)
        validation_description = f"GroupShuffleSplit held-out studies ({args.study_column})"
    else:
        validation_groups = data.split_h3_cell.astype(str)
        validation_description = "GroupShuffleSplit held-out H3 cells"

    # Hold out whole groups before learning class support, scaling, or labels.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(data, groups=validation_groups))
    raw_train, raw_test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
    if set(validation_groups.iloc[train_idx]) & set(validation_groups.iloc[test_idx]):
        raise RuntimeError("Spatial/study validation groups overlap")
    train = filter_taxonomy(raw_train, args.min_samples_per_class)
    if train.empty:
        raise ValueError("No training rows remain after train-only taxonomy support filtering; lower --min-samples-per-class")
    test = raw_test.copy()
    for level in TAXONOMIC_LEVELS:
        test = test[test[level].isin(train[level].unique())]
    if test.empty:
        raise ValueError("The held-out groups contain no rows with taxonomic labels represented in training")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict] = {}
    eligible = data.copy()
    for level in TAXONOMIC_LEVELS:
        eligible = eligible[eligible[level].isin(train[level].unique())]
    if eligible.empty:
        raise ValueError("No full-data rows match the taxonomy supported by the training partition")

    def make_model(class_count: int) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=args.n_estimators, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
            objective="multi:softprob", eval_metric="mlogloss", random_state=args.seed,
            tree_method="hist", device=args.device,
        )

    validation_scaler = StandardScaler().fit(train[features])
    X_train = pd.DataFrame(validation_scaler.transform(train[features]), columns=features, index=train.index)
    X_test = pd.DataFrame(validation_scaler.transform(test[features]), columns=features, index=test.index)

    for level in TAXONOMIC_LEVELS:
        validation_encoder = LabelEncoder().fit(train[level])
        known_test = test[level].isin(validation_encoder.classes_)
        if not known_test.any():
            raise ValueError(f"No held-out {level} labels are represented in training")
        y_train = validation_encoder.transform(train[level])
        y_test = validation_encoder.transform(test.loc[known_test, level])
        validation_model = make_model(len(validation_encoder.classes_))
        validation_model.fit(X_train, y_train, verbose=False)
        predicted = validation_model.predict(X_test.loc[known_test]).astype(int)
        metrics[level] = {
            "weighted_f1": float(f1_score(y_test, predicted, average="weighted")),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "n_classes": int(len(validation_encoder.classes_)),
            "validation_rows": int(len(y_test)),
            "validation": validation_description,
        }

        # Deployment artifacts are then refit on all eligible records; reported
        # metrics remain strictly from the untouched spatial/study holdout.
        deployment_encoder = LabelEncoder().fit(eligible[level])
        y_all = deployment_encoder.transform(eligible[level])
        final_model = make_model(len(deployment_encoder.classes_))
        final_scaler = StandardScaler().fit(eligible[features])
        X_all = pd.DataFrame(final_scaler.transform(eligible[features]), columns=features, index=eligible.index)
        final_model.fit(X_all, y_all, verbose=False)
        joblib.dump(final_model, output / f"{level}_model.pkl")
        joblib.dump(deployment_encoder, output / f"{level}_encoder.pkl")
        print(f"{level:>6}: held-out weighted F1={metrics[level]['weighted_f1']:.3f}, macro F1={metrics[level]['macro_f1']:.3f}")

    joblib.dump(final_scaler, output / "scaler.pkl")
    joblib.dump(features, output / "features.pkl")
    (output / "taxonomy_tree.json").write_text(json.dumps(tree_mapping(eligible), indent=2))
    sources = Counter(eligible.source) if "source" in eligible else {}
    metadata = {
        "version": "3.0", "model_type": "XGBoost", "device": args.device,
        "taxonomic_levels": TAXONOMIC_LEVELS, "feature_names": features,
        "training_samples": int(len(eligible)), "metrics": metrics,
        "h3": {"join_resolution": args.h3_resolution, "validation_resolution": args.split_h3_resolution},
        "source_records": dict(sources),
        "spatial_validation": validation_description,
        "validation_method": "single group holdout; preprocessing and support filtering fit on training partition; deployment artifacts refit on all eligible records",
        "validation_group_count": int(validation_groups.iloc[test_idx].nunique()),
        "validation_rows_before_label_coverage": int(len(raw_test)),
        "deployment_rows": int(len(eligible)),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved reproducible v3 artifacts to {output}")


if __name__ == "__main__":
    main()
