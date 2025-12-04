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

app = Flask(__name__)

# Global variables
hwsd_raster = None
soil_data = None
hierarchical_models = {}
hierarchical_encoders = {}
scaler = None
feature_names = None
model_metadata = None

print("\n" + "=" * 80)
print("Loading Hierarchical ML Models...")
print("=" * 80)

try:
    # Load hierarchical models
    import json
    with open('models/metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    for level in model_metadata['taxonomic_levels']:
        hierarchical_models[level] = joblib.load(f'models/{level}_model.pkl')
        hierarchical_encoders[level] = joblib.load(f'models/{level}_encoder.pkl')
        print(f"✓ Loaded {level} model ({model_metadata['metrics'][level]['n_classes']} classes, F1: {model_metadata['metrics'][level]['f1_score']:.3f})")
    
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/features.pkl')
    
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
        smu_data = pd.read_csv(Path('data/HWSD2_SMU.csv'))
        layers_data = pd.read_csv(Path('data/HWSD2_LAYERS.csv'), low_memory=False)
        soil_data = pd.merge(smu_data, layers_data, on='HWSD2_SMU_ID', how='left')
        
        hwsd_raster = rasterio.open(Path('data/HWSD2.bil'))
        print("✓ Loaded HWSD2 raster and soil database\n")
        return True
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

load_data()

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
        data = request.json
        mode = data.get('mode', 'latlon')
        print(f"Mode: {mode}")
        
        if mode == 'latlon':
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            
            # Get SMU ID from raster
            row, col = hwsd_raster.index(lon, lat)
            smu_id = int(hwsd_raster.read(1, window=((row, row+1), (col, col+1)))[0, 0])
            
            # Get soil properties
            soil_row = soil_data[soil_data['HWSD2_SMU_ID'] == smu_id].iloc[0]
            
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
            
            location_info = {'latitude': lat, 'longitude': lon, 'smu_id': smu_id}
            
        else:  # manual soil input
            feature_values = {
                'COARSE': float(data.get('coarse', 0)),
                'SAND': float(data.get('sand', 33)),
                'SILT': float(data.get('silt', 33)),
                'CLAY': float(data.get('clay', 33)),
                'ORG_CARBON': float(data.get('org_carbon', 10)),
                'PH_WATER': float(data.get('ph_water', 7)),
                'CEC_CLAY': float(data.get('cec_clay', 20)),
                'BULK': float(data.get('bulk', 1.3)),
            }
            
            lat = float(data.get('latitude', 20.0))
            lon = float(data.get('longitude', 78.0))
            location_info = {'latitude': lat, 'longitude': lon, 'smu_id': 'Manual Input'}
            smu_id = 'Manual'
        
        # Compute derived features
        feature_values['sand_clay_ratio'] = feature_values['SAND'] / (feature_values['CLAY'] + 1)
        feature_values['silt_clay_ratio'] = feature_values['SILT'] / (feature_values['CLAY'] + 1)
        feature_values['texture_index'] = (feature_values['SAND'] * 0.5 + feature_values['SILT'] * 0.3 + feature_values['CLAY'] * 0.2) / 100
        feature_values['carbon_ph_interaction'] = feature_values['ORG_CARBON'] * feature_values['PH_WATER']
        feature_values['ph_squared'] = feature_values['PH_WATER'] ** 2
        feature_values['lat_bin'] = round(lat * 2) / 2
        feature_values['lon_bin'] = round(lon * 2) / 2
        
        # Create feature array in correct order
        X_input = np.array([[feature_values[feat] for feat in feature_names]])
        X_input_df = pd.DataFrame(X_input, columns=feature_names)  # Add this line
        X_scaled = scaler.transform(X_input_df)  # Change this line
        
        # Get predictions for each taxonomic level
        predictions_by_level = {}
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            model = hierarchical_models[level]
            encoder = hierarchical_encoders[level]
            
            probs = model.predict_proba(X_scaled)[0]
            top_indices = np.argsort(probs)[::-1][:5]
            
            predictions_by_level[level] = [
                {
                    'name': encoder.classes_[i],
                    'probability': float(probs[i]),
                    'confidence': 'high' if probs[i] > 0.5 else 'medium' if probs[i] > 0.2 else 'low'
                }
                for i in top_indices
            ]
        
        # DON'T generate report here - only on download request
        return jsonify({
            'success': True,
            'predictions': predictions_by_level,
            'soil_properties': feature_values,
            'location': location_info,
            'smu_id': smu_id
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate report only when user requests it"""
    try:
        data = request.json
        
        report_filename = generate_hierarchical_report(
            smu_id=data.get('smu_id', 'Unknown'),
            soil_chars=data.get('soil_properties', {}),
            predictions_by_level=data.get('predictions', {}),
            location_info=data.get('location', {}),
            model_metrics=model_metadata.get('metrics', {}),
            enable_ai=True
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
    return send_from_directory(os.getcwd(), filename, as_attachment=True)

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
        app.run(debug=True, port=5002)
    else:
        print("\n✗ Cannot start. Ensure:")
        print("  1. Run train_hierarchical_model.py")
        print("  2. Data files present")