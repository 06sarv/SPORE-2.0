"""Train and evaluate a leakage-controlled v2-compatible XGBoost bundle.

The random holdout is split by GBIF occurrence before soil joins. Taxon support,
imputation, and scaling are learned from training data only. Soil layers are
aggregated exactly as the Flask app does. The saved deployment models are then
refit on all occurrences whose labels are supported by the training partition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

LEVELS = ["phylum", "class", "order", "family", "genus"]
SOIL_FEATURES = ["COARSE", "SAND", "SILT", "CLAY", "ORG_CARBON", "PH_WATER", "CEC_CLAY", "BULK"]
FEATURES = SOIL_FEATURES + [
    "sand_clay_ratio", "silt_clay_ratio", "texture_index",
    "carbon_ph_interaction", "ph_squared", "lat_bin", "lon_bin",
]
PARAMS = {
    "n_estimators": 250, "max_depth": 10, "learning_rate": 0.222,
    "subsample": 0.839, "colsample_bytree": 0.662, "min_child_weight": 2,
    "gamma": 0.029, "reg_alpha": 0.866, "reg_lambda": 0.601,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("data/raw_gbif_data.csv"))
    parser.add_argument("--soil-layers", type=Path, default=Path("data/HWSD2_LAYERS.csv"))
    parser.add_argument("--soil-raster", type=Path, default=Path("data/HWSD2.bil"))
    parser.add_argument("--output-dir", type=Path, default=Path("models_v2_fixed_candidate"))
    parser.add_argument("--minimum-class-records", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def aggregate_topsoil(layers: pd.DataFrame) -> pd.DataFrame:
    layers = layers.copy()
    layers["HWSD2_SMU_ID"] = pd.to_numeric(layers["HWSD2_SMU_ID"], errors="coerce")
    layers["TOPDEP"] = pd.to_numeric(layers["TOPDEP"], errors="coerce")
    layers["SHARE"] = pd.to_numeric(layers["SHARE"], errors="coerce").fillna(1.0)
    layers = layers.dropna(subset=["HWSD2_SMU_ID"])
    min_depth = layers.groupby("HWSD2_SMU_ID")["TOPDEP"].transform("min")
    topsoil = layers[layers.TOPDEP.eq(min_depth)]
    rows = []
    for smu_id, group in topsoil.groupby("HWSD2_SMU_ID"):
        row = {"HWSD2_SMU_ID": int(smu_id)}
        for feature in SOIL_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce")
            valid = values.notna() & group.SHARE.gt(0)
            row[feature] = float(np.average(values[valid], weights=group.loc[valid, "SHARE"])) if valid.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def engineer(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["sand_clay_ratio"] = result.SAND / (result.CLAY + 1)
    result["silt_clay_ratio"] = result.SILT / (result.CLAY + 1)
    result["texture_index"] = (result.SAND * 0.5 + result.SILT * 0.3 + result.CLAY * 0.2) / 100
    result["carbon_ph_interaction"] = result.ORG_CARBON * result.PH_WATER
    result["ph_squared"] = result.PH_WATER ** 2
    result["lat_bin"] = (result.decimalLatitude * 2).round() / 2
    result["lon_bin"] = (result.decimalLongitude * 2).round() / 2
    return result


def filter_by_support(data: pd.DataFrame, support: dict[str, set[str]]) -> pd.DataFrame:
    selected = data
    for level in LEVELS:
        selected = selected[selected[level].isin(support[level])]
    return selected.copy()


def make_model(n_classes: int, seed: int, n_estimators: int) -> XGBClassifier:
    params = dict(PARAMS)
    params["n_estimators"] = n_estimators
    binary = n_classes == 2
    return XGBClassifier(
        **params, objective="binary:logistic" if binary else "multi:softprob",
        eval_metric="logloss" if binary else "mlogloss", tree_method="hist",
        n_jobs=-1, random_state=seed,
    )


def main():
    args = parse_args()
    print("Reading GBIF occurrences and mapping coordinates to soil units...", flush=True)
    records = pd.read_csv(args.records, sep="\t", low_memory=False)
    if "countryCode" in records:
        records = records[records.countryCode.eq("IN")].copy()
    else:
        raise ValueError("GBIF records need a countryCode column")
    for level in LEVELS:
        if level not in records:
            records[level] = "Unknown"
        records[level] = records[level].fillna("Unknown").astype(str)
    records = records[(records.phylum != "Unknown") & (records.genus != "Unknown")].copy()
    records["decimalLatitude"] = pd.to_numeric(records.decimalLatitude, errors="coerce")
    records["decimalLongitude"] = pd.to_numeric(records.decimalLongitude, errors="coerce")
    records = records.dropna(subset=["decimalLatitude", "decimalLongitude"])
    valid_coords = records.decimalLatitude.between(-90, 90) & records.decimalLongitude.between(-180, 180)
    records = records[valid_coords].copy()
    with rasterio.open(args.soil_raster) as raster:
        pixels = raster.sample(list(zip(records.decimalLongitude, records.decimalLatitude)))
        records["HWSD2_SMU_ID"] = [
            int(value[0]) if np.isfinite(value[0]) and value[0] > 0 and (raster.nodata is None or value[0] != raster.nodata) else np.nan
            for value in pixels
        ]
    records = records.dropna(subset=["HWSD2_SMU_ID"]).copy()
    records["HWSD2_SMU_ID"] = records.HWSD2_SMU_ID.astype(int)
    if "gbifID" in records:
        records = records.drop_duplicates("gbifID")

    train_occ, test_occ = train_test_split(records, test_size=0.2, random_state=args.seed, shuffle=True)
    # Determine supported labels only from the training occurrence partition.
    support: dict[str, set[str]] = {}
    filtered_train = train_occ.copy()
    filtered_test = test_occ.copy()
    for level in LEVELS:
        counts = filtered_train[level].value_counts()
        support[level] = set(counts[counts >= args.minimum_class_records].index)
        filtered_train = filtered_train[filtered_train[level].isin(support[level])].copy()
        filtered_test = filtered_test[filtered_test[level].isin(support[level])].copy()
        print(f"{level}: {filtered_train[level].nunique()} supported labels", flush=True)
    if filtered_train.empty or filtered_test.empty:
        raise ValueError("No usable rows remain after train-only taxonomy support filtering")

    print("Aggregating app-matched topsoil features...", flush=True)
    layers = pd.read_csv(args.soil_layers, low_memory=False)
    soil = aggregate_topsoil(layers)
    train = engineer(filtered_train.merge(soil, on="HWSD2_SMU_ID", how="left", validate="many_to_one"))
    test = engineer(filtered_test.merge(soil, on="HWSD2_SMU_ID", how="left", validate="many_to_one"))
    for feature in SOIL_FEATURES:
        train[feature] = pd.to_numeric(train[feature], errors="coerce")
        test[feature] = pd.to_numeric(test[feature], errors="coerce")
    for data in (train, test):
        data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    train = train.dropna(subset=LEVELS).copy()
    test = test.dropna(subset=LEVELS).copy()
    if train.empty or test.empty:
        raise ValueError("No rows remain after soil feature preparation")

    print("Evaluating untouched random holdout...", flush=True)
    validation_imputer = SimpleImputer(strategy="median").fit(train[FEATURES])
    x_train = validation_imputer.transform(train[FEATURES])
    x_test = validation_imputer.transform(test[FEATURES])
    validation_scaler = StandardScaler().fit(x_train)
    x_train = validation_scaler.transform(x_train)
    x_test = validation_scaler.transform(x_test)
    metrics = {}
    for level in LEVELS:
        encoder = LabelEncoder().fit(train[level])
        y_train = encoder.transform(train[level])
        y_test = encoder.transform(test[level])
        model = make_model(len(encoder.classes_), args.seed, args.n_estimators)
        model.fit(x_train, y_train, verbose=False)
        prediction = model.predict(x_test).astype(int)
        metrics[level] = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "weighted_f1": float(f1_score(y_test, prediction, average="weighted")),
            "f1_score": float(f1_score(y_test, prediction, average="weighted")),
            "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
            "n_classes": int(len(encoder.classes_)),
            "classes": encoder.classes_.tolist(),
            "validation_rows": int(len(test)),
            "validation": "random 80/20 GBIF occurrence holdout; split before soil join; train-only support/imputation/scaling; app-matched topsoil aggregation",
        }
        print(f"{level:>6}: accuracy={metrics[level]['accuracy']:.4f}, weighted F1={metrics[level]['weighted_f1']:.4f}, macro F1={metrics[level]['macro_f1']:.4f}", flush=True)

    # Refit deployment artifacts on all records whose labels are supported by
    # the training partition. Test labels are used only after evaluation metrics
    # have been computed, and only if their class was already supported.
    all_occ = filter_by_support(records, support)
    deployment = engineer(all_occ.merge(soil, on="HWSD2_SMU_ID", how="left", validate="many_to_one"))
    for feature in SOIL_FEATURES:
        deployment[feature] = pd.to_numeric(deployment[feature], errors="coerce")
    deployment[FEATURES] = deployment[FEATURES].replace([np.inf, -np.inf], np.nan)
    deployment = deployment.dropna(subset=LEVELS).copy()
    if deployment.empty:
        raise ValueError("No deployment rows remain after app-matched topsoil aggregation")
    imputer = SimpleImputer(strategy="median").fit(deployment[FEATURES])
    x_all = imputer.transform(deployment[FEATURES])
    scaler = StandardScaler().fit(x_all)
    x_all = scaler.transform(x_all)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    for level in LEVELS:
        encoder = LabelEncoder().fit(deployment[level])
        y_all = encoder.transform(deployment[level])
        model = make_model(len(encoder.classes_), args.seed, args.n_estimators)
        model.fit(x_all, y_all, verbose=False)
        joblib.dump(model, output / f"{level}_model.pkl")
        joblib.dump(encoder, output / f"{level}_encoder.pkl")
        metrics[level]["deployment_n_classes"] = int(len(encoder.classes_))
    joblib.dump(imputer, output / "imputer.pkl")
    joblib.dump(scaler, output / "scaler.pkl")
    joblib.dump(FEATURES, output / "features.pkl")
    metadata = {
        "version": "2.1-fixed", "model_type": "XGBoost", "taxonomic_levels": LEVELS,
        "feature_names": FEATURES, "metrics": metrics,
        "training_samples": int(len(deployment)),
        "training_occurrences": int(deployment.gbifID.nunique()) if "gbifID" in deployment else int(len(deployment)),
        "soil_aggregation": "SHARE-weighted shallowest TOPDEP per HWSD2_SMU_ID; matches app",
        "validation_method": "random 80/20 occurrence split, seed {}; deployment artifacts refit on supported full-data labels".format(args.seed),
        "training_parameters": {**PARAMS, "n_estimators": args.n_estimators, "minimum_class_records": args.minimum_class_records},
        "data_source": "GBIF India occurrences and HWSD2 soil map",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    report = {
        "evaluation": "XGBoost v2 configuration; random 80/20 GBIF occurrence holdout; split before soil join; train-only label support, imputation and scaling; app-matched SHARE-weighted shallowest topsoil aggregation; fixed tree count; no held-out early stopping",
        "seed": args.seed, "params": metadata["training_parameters"],
        "occurrences_train_after_support": int(len(filtered_train)),
        "occurrences_test_after_support": int(len(filtered_test)),
        "deployment_occurrences": metadata["training_occurrences"],
        "deployment_rows": int(len(deployment)), "results": metrics,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"Saved candidate bundle and evaluation to {output}", flush=True)


if __name__ == "__main__":
    main()
