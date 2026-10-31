"""
Hierarchical SPORE Application
- Supports BOTH lat/lon AND individual soil characteristics
- Uses hierarchical taxonomy for better predictions
- Multiple prediction levels (phylum → genus)
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from pathlib import Path
import os
import joblib
import rasterio
from report_generator import generate_hierarchical_report

try:
    import h3
except ImportError:
    h3 = None

app = Flask(__name__)

MODEL_DIR = Path(os.environ.get('SPORE_MODEL_DIR', 'models'))
DATA_DIR = Path(os.environ.get('SPORE_DATA_DIR', 'data'))
H3_RESOLUTION = int(os.environ.get('SPORE_H3_RESOLUTION', '6'))
ENVIRONMENT_FEATURES_PATH = os.environ.get('SPORE_ENVIRONMENT_FEATURES')

# Global variables
hwsd_raster = None
soil_data = None
hierarchical_models = {}
hierarchical_encoders = {}
scaler = None
imputer = None
feature_names = None
model_metadata = None
taxonomy_tree = {}
environment_data = None

SOIL_FEATURE_DEFAULTS = {
    'COARSE': 0.0,
    'SAND': 33.0,
    'SILT': 33.0,
    'CLAY': 33.0,
    'ORG_CARBON': 10.0,
    'PH_WATER': 7.0,
    'CEC_CLAY': 20.0,
    'BULK': 1.3,
}


def _coerce_float(payload, *keys, default=None):
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if value in (None, ''):
                continue
            try:
                result = float(value)
                return result if np.isfinite(result) else default
            except (TypeError, ValueError):
                return default
    return default


def _build_soil_feature_values(payload):
    values = {}
    for key in SOIL_FEATURE_DEFAULTS:
        raw = next((payload[name] for name in (key, key.lower(), key.replace('_', '').lower()) if name in payload), None)
        if raw in (None, ''):
            raise ValueError(f'{key} is required for manual soil input')
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f'{key} must be a number') from None
        if not np.isfinite(value):
            raise ValueError(f'{key} must be finite')
        low, high = SOIL_FEATURES_INFO[key]['range']
        if not low <= value <= high:
            raise ValueError(f'{key} must be between {low} and {high}')
        values[key] = value
    return values


def _aggregate_topsoil_layers(layers):
    """Collapse HWSD component layers to one share-weighted topsoil row per SMU."""
    layers = layers.copy()
    layers['TOPDEP'] = pd.to_numeric(layers.get('TOPDEP'), errors='coerce')
    layers['SHARE'] = pd.to_numeric(layers.get('SHARE'), errors='coerce').fillna(1.0)
    minimum_depth = layers.groupby('HWSD2_SMU_ID')['TOPDEP'].transform('min')
    topsoil = layers[layers['TOPDEP'] == minimum_depth]

    def weighted_feature(group, feature):
        values = pd.to_numeric(group[feature], errors='coerce')
        valid = values.notna() & group['SHARE'].gt(0)
        if not valid.any():
            return np.nan
        return np.average(values[valid], weights=group.loc[valid, 'SHARE'])

    rows = []
    for smu_id, group in topsoil.groupby('HWSD2_SMU_ID'):
        row = {'HWSD2_SMU_ID': smu_id}
        for feature in SOIL_FEATURE_DEFAULTS:
            row[feature] = weighted_feature(group, feature)
        rows.append(row)
    return pd.DataFrame(rows)


def _lookup_smu_id(lat, lon):
    if hwsd_raster is None:
        raise ValueError("Soil raster is not loaded")

    bounds = hwsd_raster.bounds
    if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
        raise ValueError("Coordinates are outside the raster coverage")

    row, col = hwsd_raster.index(lon, lat)
    if row < 0 or col < 0 or row >= hwsd_raster.height or col >= hwsd_raster.width:
        raise ValueError("Coordinates are outside the raster coverage")

    smu_value = hwsd_raster.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
    if hwsd_raster.nodata is not None and smu_value == hwsd_raster.nodata:
        raise ValueError("No soil mapping unit found at the provided coordinates")

    smu_id = int(smu_value)
    if smu_id <= 0:
        raise ValueError("No soil mapping unit found at the provided coordinates")
    return smu_id


def _compute_h3_index(lat, lon):
    if h3 is None:
        return None

    if hasattr(h3, 'latlng_to_cell'):
        return h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
    if hasattr(h3, 'geo_to_h3'):
        return h3.geo_to_h3(lat, lon, H3_RESOLUTION)
    return None


def _metric_f1(metric):
    return metric.get('weighted_f1', metric.get('f1_score', 0.0))


def _environment_features(h3_cell):
    if environment_data is None or h3_cell is None:
        return {}
    match = environment_data[environment_data['h3_cell'] == h3_cell]
    if match.empty:
        return {}
    row = match.iloc[0]
    return {
        column: float(row[column])
        for column in environment_data.columns
        if column != 'h3_cell' and pd.notna(row[column])
    }

print("\n" + "=" * 80)
print("Loading Hierarchical ML Models...")
print("=" * 80)

try:
    # Load hierarchical models
    import json
    with open(MODEL_DIR / 'metadata.json', 'r') as f:
        model_metadata = json.load(f)
    # A promoted v3 bundle is authoritative for its H3 join resolution unless
    # an operator explicitly overrides it for a controlled compatibility test.
    if 'SPORE_H3_RESOLUTION' not in os.environ:
        H3_RESOLUTION = model_metadata.get('h3', {}).get('join_resolution', H3_RESOLUTION)
    
    for level in model_metadata['taxonomic_levels']:
        hierarchical_models[level] = joblib.load(MODEL_DIR / f'{level}_model.pkl')
        hierarchical_encoders[level] = joblib.load(MODEL_DIR / f'{level}_encoder.pkl')
        metric = model_metadata['metrics'][level]
        print(f"✓ Loaded {level} model ({metric['n_classes']} classes, F1: {_metric_f1(metric):.3f})")
    
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    imputer_path = MODEL_DIR / 'imputer.pkl'
    imputer = joblib.load(imputer_path) if imputer_path.exists() else None
    feature_names = joblib.load(MODEL_DIR / 'features.pkl')
    tree_path = MODEL_DIR / 'taxonomy_tree.json'
    if tree_path.exists():
        taxonomy_tree = json.loads(tree_path.read_text())
    
    print("=" * 80 + "\n")
    models_loaded = True
except Exception as e:
    print(f"✗ Error loading models: {str(e)}")
    print("  Run train_model.py first")
    models_loaded = False

def load_data():
    global hwsd_raster, soil_data
    try:
        print("Loading soil database...")
        layers_data = pd.read_csv(DATA_DIR / 'HWSD2_LAYERS.csv', low_memory=False)
        layers_data['HWSD2_SMU_ID'] = pd.to_numeric(layers_data['HWSD2_SMU_ID'], errors='coerce')
        soil_data = _aggregate_topsoil_layers(layers_data)
        
        hwsd_raster = rasterio.open(DATA_DIR / 'HWSD2.bil')
        print("✓ Loaded HWSD2 raster and soil database\n")
        return True
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

load_data()

if ENVIRONMENT_FEATURES_PATH:
    environment_path = Path(ENVIRONMENT_FEATURES_PATH)
    environment_data = (
        pd.read_parquet(environment_path)
        if environment_path.suffix.lower() == '.parquet'
        else pd.read_csv(environment_path)
    )
    if 'h3_cell' not in environment_data.columns:
        raise ValueError('SPORE_ENVIRONMENT_FEATURES must contain an h3_cell column')

# Soil features info for UI
SOIL_FEATURES_INFO = {
    'COARSE': {'name': 'Gravel Content (%)', 'range': (0, 100), 'unit': '%'},
    'SAND': {'name': 'Sand Content (%)', 'range': (0, 100), 'unit': '%'},
    'SILT': {'name': 'Silt Content (%)', 'range': (0, 100), 'unit': '%'},
    'CLAY': {'name': 'Clay Content (%)', 'range': (0, 100), 'unit': '%'},
    'ORG_CARBON': {'name': 'Organic Carbon (g/kg)', 'range': (0, 100), 'unit': 'g/kg'},
    'PH_WATER': {'name': 'Soil pH', 'range': (4, 9), 'unit': ''},
    'CEC_CLAY': {'name': 'CEC of Clay (cmol(+)/kg)', 'range': (0, 200), 'unit': 'cmol(+)/kg'},
    'BULK': {'name': 'Bulk Density (kg/dm³)', 'range': (0.5, 2.0), 'unit': 'kg/dm³'},
}

@app.route('/')
def home():
    return render_template('index.html', 
                         features=SOIL_FEATURES_INFO,
                         model_info=model_metadata if models_loaded else None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'A JSON object is required'}), 400
        mode = data.get('mode', 'latlon')
        if mode not in {'latlon', 'manual'}:
            return jsonify({'success': False, 'error': "mode must be 'latlon' or 'manual'"}), 400
        print(f"Mode: {mode}")
        
        if mode == 'latlon':
            lat = _coerce_float(data, 'latitude', 'lat')
            lon = _coerce_float(data, 'longitude', 'lon')
            if lat is None or lon is None:
                return jsonify({'success': False, 'error': 'Latitude and longitude must be finite numbers'}), 400
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                return jsonify({'success': False, 'error': 'Latitude must be between -90 and 90 and longitude between -180 and 180'}), 400
            if soil_data is None:
                return jsonify({'success': False, 'error': 'Soil lookup data is unavailable'}), 503
            try:
                smu_id = _lookup_smu_id(lat, lon)
            except ValueError as exc:
                status = 404 if 'No soil mapping unit' in str(exc) else 400
                return jsonify({'success': False, 'error': str(exc)}), status

            # Get soil properties
            soil_match = soil_data[soil_data['HWSD2_SMU_ID'] == smu_id]
            if soil_match.empty:
                return jsonify({'success': False, 'error': f'No soil data found for SMU {smu_id}'}), 404
            soil_row = soil_match.iloc[0]
            
            feature_values = {
                'COARSE': float(soil_row.get('COARSE', 0)),
                'SAND': float(soil_row.get('SAND', 33)),
                'SILT': float(soil_row.get('SILT', 33)),
                'CLAY': float(soil_row.get('CLAY', 33)),
                'ORG_CARBON': float(soil_row.get('ORG_CARBON', 10)),
                'PH_WATER': float(soil_row.get('PH_WATER', 7)),
                'CEC_CLAY': float(soil_row.get('CEC_CLAY', 20)),
                'BULK': float(soil_row.get('BULK', 1.3)),
            }
            
            location_info = {
                'latitude': lat,
                'longitude': lon,
                'smu_id': smu_id,
                'h3_resolution': H3_RESOLUTION,
                'h3_cell': _compute_h3_index(lat, lon),
            }
            
        else:  # manual soil input
            feature_values = _build_soil_feature_values(data)

            lat = _coerce_float(data, 'latitude', 'lat', default=20.0)
            lon = _coerce_float(data, 'longitude', 'lon', default=78.0)
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                return jsonify({'success': False, 'error': 'Latitude must be between -90 and 90 and longitude between -180 and 180'}), 400
            location_info = {
                'latitude': lat,
                'longitude': lon,
                'smu_id': 'Manual Input',
                'h3_resolution': H3_RESOLUTION,
                'h3_cell': _compute_h3_index(lat, lon),
            }
            smu_id = 'Manual'

        # Optional climate/land-cover features must come from the exact H3 table
        # used at training time. Missing rows are rejected below instead of filled.
        feature_values.update(_environment_features(location_info['h3_cell']))
        
        # Compute derived features
        feature_values['sand_clay_ratio'] = feature_values['SAND'] / (feature_values['CLAY'] + 1)
        feature_values['silt_clay_ratio'] = feature_values['SILT'] / (feature_values['CLAY'] + 1)
        feature_values['texture_index'] = (feature_values['SAND'] * 0.5 + feature_values['SILT'] * 0.3 + feature_values['CLAY'] * 0.2) / 100
        feature_values['carbon_ph_interaction'] = feature_values['ORG_CARBON'] * feature_values['PH_WATER']
        feature_values['ph_squared'] = feature_values['PH_WATER'] ** 2
        feature_values['lat_bin'] = round(lat * 2) / 2
        feature_values['lon_bin'] = round(lon * 2) / 2
        
        # Create feature array in correct order
        missing_features = [feat for feat in feature_names if feat not in feature_values]
        if missing_features:
            return jsonify({
                'success': False,
                'error': f'Missing engineered features: {", ".join(missing_features)}'
            }), 400

        X_input_df = pd.DataFrame([[feature_values[feat] for feat in feature_names]], columns=feature_names)
        if imputer is not None:
            X_input_df = pd.DataFrame(imputer.transform(X_input_df), columns=feature_names)
        X_scaled = scaler.transform(X_input_df)
        
        # Get predictions for each taxonomic level
        predictions_by_level = {}
        selected_parent = None
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            model = hierarchical_models[level]
            encoder = hierarchical_encoders[level]
            
            probs = model.predict_proba(X_scaled)[0]
            ranked_indices = np.argsort(probs)[::-1]
            allowed_children = taxonomy_tree.get(level, {}).get(selected_parent)
            if allowed_children:
                allowed_set = set(allowed_children)
                ranked_indices = [index for index in ranked_indices if encoder.classes_[index] in allowed_set]
            top_indices = ranked_indices[:5]
            
            predictions_by_level[level] = [
                {
                    'name': encoder.classes_[i],
                    'probability': float(probs[i]),
                    'confidence': 'high' if probs[i] > 0.5 else 'medium' if probs[i] > 0.2 else 'low'
                }
                for i in top_indices
            ]
            if predictions_by_level[level]:
                selected_parent = predictions_by_level[level][0]['name']
        
        # DON'T generate report here - only on download request
        return jsonify({
            'success': True,
            'predictions': predictions_by_level,
            'soil_properties': feature_values,
            'location': location_info,
            'smu_id': smu_id,
            'enable_ai': bool(data.get('enable_ai', True))
        })
        
    except Exception as e:
        print(f"Error: {e}")
        if isinstance(e, ValueError):
            return jsonify({'success': False, 'error': str(e)}), 400
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Prediction failed due to an internal error'}), 500


@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate report only when user requests it"""
    try:
        data = request.get_json(silent=True) or {}
        
        report_filename = generate_hierarchical_report(
            smu_id=data.get('smu_id', 'Unknown'),
            soil_chars=data.get('soil_properties', {}),
            predictions_by_level=data.get('predictions', {}),
            location_info=data.get('location', {}),
            model_metrics=model_metadata.get('metrics', {}),
            model_info=model_metadata,
            enable_ai=bool(data.get('enable_ai', True))
        )
        
        return jsonify({
            'success': True,
            'report_file': report_filename
        })
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Report generation now handled by report_generator.py

@app.route('/download_report/<filename>')
def download_report(filename):
    safe_name = Path(filename).name
    report_path = Path.cwd() / safe_name
    if not report_path.exists():
        return jsonify({'success': False, 'error': 'Report file not found'}), 404
    return send_from_directory(os.getcwd(), safe_name, as_attachment=True)

@app.route('/model_info')
def model_info():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    return jsonify(model_metadata)

if __name__ == '__main__':
    if models_loaded and soil_data is not None:
        print("🚀 Starting SPORE Hierarchical Application...")
        print(f"   Taxonomic Levels: {', '.join(model_metadata['taxonomic_levels'])}")
        print(f"   Input Modes: Lat/Lon OR Individual Soil Characteristics")
        print("\n")
        app.run(
            debug=os.environ.get('FLASK_DEBUG', '0') == '1',
            port=int(os.environ.get('PORT', '5002'))
        )
    else:
        print("\n✗ Cannot start. Ensure:")
        print("  1. Run train_hierarchical_model.py")
        print("  2. Data files present")
