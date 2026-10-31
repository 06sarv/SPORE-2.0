"""Controlled synthetic benchmark for SPORE's tree models.

This creates simulated soil measurements and nested taxonomic labels in memory.
It is a pipeline/model-behavior demo only, not evidence about real microbes.
Run: python3 synthetic_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

SEED = 2026
N_REGIONS = 12
SAMPLES_PER_REGION = 500
N_GENERA = 32
TAXONOMIC_LEVELS = ["phylum", "class", "order", "family", "genus"]
SOIL_FEATURES = ["COARSE", "SAND", "SILT", "CLAY", "ORG_CARBON", "PH_WATER", "CEC_CLAY", "BULK"]
FEATURES = SOIL_FEATURES + [
    "sand_clay_ratio", "silt_clay_ratio", "texture_index",
    "carbon_ph_interaction", "ph_squared", "lat_bin", "lon_bin",
]


def make_synthetic_data(seed: int = SEED) -> pd.DataFrame:
    """Generate balanced taxa with known soil-profile associations and site shifts."""
    rng = np.random.default_rng(seed)
    n = N_REGIONS * SAMPLES_PER_REGION
    regions = np.repeat(np.arange(N_REGIONS), SAMPLES_PER_REGION)

    # Each region contains each genus approximately equally often, so the spatial
    # holdout still evaluates all taxa while testing transfer across site shifts.
    genus_id = np.concatenate([
        np.resize(rng.permutation(N_GENERA), SAMPLES_PER_REGION)
        for _ in range(N_REGIONS)
    ])
    prototypes_texture = rng.dirichlet([2.0, 2.0, 2.0], size=N_GENERA) * 100
    prototype_coarse = rng.uniform(0, 22, N_GENERA)
    prototype_carbon = rng.uniform(3, 42, N_GENERA)
    prototype_ph = rng.uniform(4.4, 8.8, N_GENERA)
    prototype_cec = rng.uniform(8, 95, N_GENERA)
    prototype_bulk = rng.uniform(0.9, 1.75, N_GENERA)

    # Regional offsets mimic changes in measurement and environmental context.
    region_offsets = rng.normal(0, [2.0, 3.0, 3.0, 3.0, 2.5, 0.18, 5.0, 0.04], (N_REGIONS, 8))
    coarse = prototype_coarse[genus_id] + region_offsets[regions, 0] + rng.normal(0, 3, n)
    texture = prototypes_texture[genus_id] + region_offsets[regions, 1:4] + rng.normal(0, 6, (n, 3))
    texture = np.clip(texture, 0.2, None)
    texture = texture / texture.sum(axis=1, keepdims=True) * 100
    carbon = prototype_carbon[genus_id] + region_offsets[regions, 4] + rng.normal(0, 5, n)
    ph = prototype_ph[genus_id] + region_offsets[regions, 5] + rng.normal(0, 0.28, n)
    cec = prototype_cec[genus_id] + region_offsets[regions, 6] + rng.normal(0, 9, n)
    bulk = prototype_bulk[genus_id] + region_offsets[regions, 7] + rng.normal(0, 0.06, n)

    # Introduce realistic bounds and derive labels with controlled label noise.
    data = pd.DataFrame({
        "COARSE": np.clip(coarse, 0, 100),
        "SAND": texture[:, 0], "SILT": texture[:, 1], "CLAY": texture[:, 2],
        "ORG_CARBON": np.clip(carbon, 0, 100), "PH_WATER": np.clip(ph, 4, 9),
        "CEC_CLAY": np.clip(cec, 0, 200), "BULK": np.clip(bulk, 0.5, 2.2),
        "region": regions,
    })
    noisy_genus = genus_id.copy()
    flip = rng.random(n) < 0.06
    noisy_genus[flip] = rng.integers(0, N_GENERA, flip.sum())
    data["genus"] = [f"Genus_{i:02d}" for i in noisy_genus]
    data["family"] = [f"Family_{i // 2:02d}" for i in noisy_genus]
    data["order"] = [f"Order_{i // 4:02d}" for i in noisy_genus]
    data["class"] = [f"Class_{i // 8:02d}" for i in noisy_genus]
    data["phylum"] = [f"Phylum_{i // 16:02d}" for i in noisy_genus]

    # Geographic features are included as in the v2 feature set, while region
    # remains solely a grouping variable and is never given to the estimator.
    region_row, region_col = regions // 4, regions % 4
    data["latitude"] = 8 + region_row * 8 + rng.normal(0, 0.35, n)
    data["longitude"] = 67 + region_col * 7 + rng.normal(0, 0.35, n)
    data["sand_clay_ratio"] = data.SAND / (data.CLAY + 1)
    data["silt_clay_ratio"] = data.SILT / (data.CLAY + 1)
    data["texture_index"] = (data.SAND * 0.5 + data.SILT * 0.3 + data.CLAY * 0.2) / 100
    data["carbon_ph_interaction"] = data.ORG_CARBON * data.PH_WATER
    data["ph_squared"] = data.PH_WATER ** 2
    data["lat_bin"] = (data.latitude * 2).round() / 2
    data["lon_bin"] = (data.longitude * 2).round() / 2

    # Random missingness is imputed using training-only medians in evaluation.
    for feature in SOIL_FEATURES:
        data.loc[rng.random(n) < 0.01, feature] = np.nan
    return data


def make_splits(data: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(data))
    random_train, random_test = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=data["genus"]
    )
    spatial_test = data.index[data["region"].isin([9, 10, 11])].to_numpy()
    spatial_train = data.index[~data["region"].isin([9, 10, 11])].to_numpy()
    return {
        "random holdout": (random_train, random_test),
        "held-out regions": (spatial_train, spatial_test),
    }


def make_model(name: str, class_count: int):
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        )
    binary = class_count == 2
    return XGBClassifier(
        n_estimators=120, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic" if binary else "multi:softprob",
        eval_metric="logloss" if binary else "mlogloss",
        tree_method="hist", n_jobs=4, random_state=SEED,
    )


def evaluate(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, (train_idx, test_idx) in make_splits(data).items():
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        imputer = SimpleImputer(strategy="median").fit(train[FEATURES])
        x_train, x_test = imputer.transform(train[FEATURES]), imputer.transform(test[FEATURES])
        for level in TAXONOMIC_LEVELS:
            encoder = LabelEncoder().fit(train[level])
            y_train, y_test = encoder.transform(train[level]), encoder.transform(test[level])
            majority = train[level].value_counts().idxmax()
            baseline = np.full(len(test), majority, dtype=object)
            rows.append({
                "split": split_name, "model": "Majority baseline", "level": level,
                "accuracy": accuracy_score(test[level], baseline),
                "macro_f1": f1_score(test[level], baseline, average="macro", zero_division=0),
                "weighted_f1": f1_score(test[level], baseline, average="weighted", zero_division=0),
                "classes": len(encoder.classes_),
            })
            for model_name in ("Random Forest", "XGBoost"):
                model = make_model(model_name, len(encoder.classes_))
                model.fit(x_train, y_train)
                predicted = encoder.inverse_transform(model.predict(x_test).astype(int))
                rows.append({
                    "split": split_name, "model": model_name, "level": level,
                    "accuracy": accuracy_score(test[level], predicted),
                    "macro_f1": f1_score(test[level], predicted, average="macro", zero_division=0),
                    "weighted_f1": f1_score(test[level], predicted, average="weighted", zero_division=0),
                    "classes": len(encoder.classes_),
                })
    return pd.DataFrame(rows)


def main() -> None:
    data = make_synthetic_data()
    results = evaluate(data)
    print("SYNTHETIC CONTROLLED BENCHMARK — NOT REAL MICROBIAL PERFORMANCE")
    print(f"seed={SEED}; rows={len(data)}; regions={N_REGIONS}; genera={N_GENERA}; label_noise=6%; missingness=1% per soil feature")
    print(results.to_string(index=False, formatters={
        "accuracy": "{:.3f}".format, "macro_f1": "{:.3f}".format,
        "weighted_f1": "{:.3f}".format,
    }))
    print("\nSynthetic labels are generated from planted soil-profile classes. Scores validate code behavior only.")


if __name__ == "__main__":
    main()
