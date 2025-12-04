"""
SPORE v2.0 - Fast Training with Optimized Parameters
Uses best parameters from Trial 0 (F1: 0.306)
"""

import pandas as pd
import numpy as np
import rasterio
import joblib
import json
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from collections import Counter

warnings.filterwarnings('ignore')

print("=" * 80)
print("SPORE v2.0 - FAST TRAINING (Optimized Parameters)")
print("=" * 80)

# Best parameters from Trial 0
BEST_PARAMS = {
    'n_estimators': 250,
    'max_depth': 10,
    'learning_rate': 0.222,
    'subsample': 0.839,
    'colsample_bytree': 0.662,
    'min_child_weight': 2,
    'gamma': 0.029,
    'reg_alpha': 0.866,
    'reg_lambda': 0.601,
}

MIN_SAMPLES_PER_CLASS = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = -1

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/8] Loading data...")

soil_smu = pd.read_csv('data/HWSD2_SMU.csv')
soil_layers = pd.read_csv('data/HWSD2_LAYERS.csv', low_memory=False)
soil_data = pd.merge(soil_smu, soil_layers, on='HWSD2_SMU_ID', how='left')

bacteria_data = pd.read_csv('data/raw_gbif_data.csv', delimiter='\t', low_memory=False)
bacteria_data = bacteria_data[bacteria_data['countryCode'] == 'IN']

hwsd_raster = rasterio.open('data/HWSD2.bil')

print(f"  Soil records: {len(soil_data):,}")
print(f"  Bacteria records: {len(bacteria_data):,}")

# ============================================================================
# STEP 2: Extract Taxonomy
# ============================================================================
print("\n[2/8] Extracting taxonomy...")

taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus']

for level in taxonomic_levels:
    if level in bacteria_data.columns:
        bacteria_data[level] = bacteria_data[level].fillna('Unknown')
    else:
        bacteria_data[level] = 'Unknown'

bacteria_data = bacteria_data[
    (bacteria_data['phylum'] != 'Unknown') &
    (bacteria_data['genus'] != 'Unknown')
]

print(f"  Valid records: {len(bacteria_data):,}")

# ============================================================================
# STEP 3: Filter Rare Taxa
# ============================================================================
print(f"\n[3/8] Filtering rare taxa (min {MIN_SAMPLES_PER_CLASS} samples)...")

for level in taxonomic_levels:
    counts = bacteria_data[level].value_counts()
    valid_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    bacteria_data = bacteria_data[bacteria_data[level].isin(valid_classes)]
    print(f"  {level.capitalize()}: {len(valid_classes)} classes")

print(f"  Remaining samples: {len(bacteria_data):,}")

# ============================================================================
# STEP 4: Map to Soil Units
# ============================================================================
print("\n[4/8] Mapping bacteria to soil units...")

def get_smu_id(lat, lon, raster):
    try:
        row, col = raster.index(lon, lat)
        if 0 <= row < raster.height and 0 <= col < raster.width:
            smu_id = raster.read(1, window=((row, row+1), (col, col+1)))[0, 0]
            return int(smu_id) if smu_id > 0 else None
    except:
        pass
    return None

bacteria_data['decimalLatitude'] = pd.to_numeric(bacteria_data['decimalLatitude'], errors='coerce')
bacteria_data['decimalLongitude'] = pd.to_numeric(bacteria_data['decimalLongitude'], errors='coerce')
bacteria_data = bacteria_data.dropna(subset=['decimalLatitude', 'decimalLongitude'])

bacteria_data['HWSD2_SMU_ID'] = bacteria_data.apply(
    lambda row: get_smu_id(row['decimalLatitude'], row['decimalLongitude'], hwsd_raster), 
    axis=1
)
bacteria_data = bacteria_data.dropna(subset=['HWSD2_SMU_ID'])
bacteria_data['HWSD2_SMU_ID'] = bacteria_data['HWSD2_SMU_ID'].astype(int)

merged_data = pd.merge(bacteria_data, soil_data, on='HWSD2_SMU_ID', how='left')
print(f"  Merged records: {len(merged_data):,}")

# ============================================================================
# STEP 5: Engineer Features
# ============================================================================
print("\n[5/8] Engineering features...")

SOIL_FEATURES = ['COARSE', 'SAND', 'SILT', 'CLAY', 'ORG_CARBON', 'PH_WATER', 'CEC_CLAY', 'BULK']

for col in SOIL_FEATURES:
    if col in merged_data.columns:
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

merged_data = merged_data.dropna(subset=SOIL_FEATURES)

# Derived features
merged_data['sand_clay_ratio'] = merged_data['SAND'] / (merged_data['CLAY'] + 1)
merged_data['silt_clay_ratio'] = merged_data['SILT'] / (merged_data['CLAY'] + 1)
merged_data['texture_index'] = (merged_data['SAND'] * 0.5 + merged_data['SILT'] * 0.3 + merged_data['CLAY'] * 0.2) / 100
merged_data['carbon_ph_interaction'] = merged_data['ORG_CARBON'] * merged_data['PH_WATER']
merged_data['ph_squared'] = merged_data['PH_WATER'] ** 2
merged_data['lat_bin'] = (merged_data['decimalLatitude'] * 2).round() / 2
merged_data['lon_bin'] = (merged_data['decimalLongitude'] * 2).round() / 2

ALL_FEATURES = SOIL_FEATURES + ['sand_clay_ratio', 'silt_clay_ratio', 'texture_index', 
                                 'carbon_ph_interaction', 'ph_squared', 'lat_bin', 'lon_bin']

X = merged_data[ALL_FEATURES].fillna(merged_data[ALL_FEATURES].median())
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

print(f"  Features: {len(ALL_FEATURES)}")
print(f"  Samples: {len(X):,}")

# ============================================================================
# STEP 6: Train Models
# ============================================================================
print("\n[6/8] Training models with optimized parameters...")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=ALL_FEATURES, index=X.index)

models = {}
encoders = {}
metrics = {}

for level in taxonomic_levels:
    print(f"\n  Training {level}...")
    
    y = merged_data[level]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    model = XGBClassifier(
        **BEST_PARAMS,
        objective='multi:softprob',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        early_stopping_rounds=20
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    
    print(f"    ✓ F1: {f1:.4f} | Acc: {acc:.4f} | Classes: {len(encoder.classes_)}")
    
    models[level] = model
    encoders[level] = encoder
    metrics[level] = {
        'f1_score': float(f1),
        'accuracy': float(acc),
        'n_classes': len(encoder.classes_),
        'classes': encoder.classes_.tolist()
    }

# ============================================================================
# STEP 7: Save Models
# ============================================================================
print("\n[7/8] Saving models...")

Path('models').mkdir(exist_ok=True)

for level in taxonomic_levels:
    joblib.dump(models[level], f'models/{level}_model.pkl')
    joblib.dump(encoders[level], f'models/{level}_encoder.pkl')

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(ALL_FEATURES, 'models/features.pkl')

# ============================================================================
# STEP 8: Save Metadata
# ============================================================================
print("\n[8/8] Saving metadata...")

feature_importance = models['phylum'].feature_importances_
importance_list = [{'feature': f, 'importance': float(i)} for f, i in zip(ALL_FEATURES, feature_importance)]

metadata = {
    'version': '2.0',
    'taxonomic_levels': taxonomic_levels,
    'feature_names': ALL_FEATURES,
    'metrics': metrics,
    'training_samples': len(merged_data),
    'feature_importance': sorted(importance_list, key=lambda x: x['importance'], reverse=True),
    'model_type': 'XGBoost',
    'best_params': BEST_PARAMS
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

for level in taxonomic_levels:
    m = metrics[level]
    print(f"  {level.capitalize():10} | F1: {m['f1_score']:.4f} | Classes: {m['n_classes']}")

print("\n✓ Models saved to /models")
print("✓ Run: python app.py")