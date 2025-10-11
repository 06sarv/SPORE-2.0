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
        if not models_loaded:
            return jsonify({'error': 'Models not loaded. Train models first.'}), 500
        
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        soil_characteristics = data.get('soil_characteristics', {})
        
        # MODE 1: Individual soil characteristics provided
        if soil_characteristics and len(soil_characteristics) > 0:
            print(f"Mode: Individual soil characteristics")
            
            # Validate all required features
            base_features = ['COARSE', 'SAND', 'SILT', 'CLAY', 'ORG_CARBON', 'PH_WATER', 'CEC_CLAY', 'BULK']
            missing = [f for f in base_features if f not in soil_characteristics]
            if missing:
                return jsonify({'error': f'Missing soil characteristics: {", ".join(missing)}'}), 400
            
            # Use provided characteristics
            feature_values = {k: float(v) for k, v in soil_characteristics.items()}
            
            # Add spatial features (use defaults if not provided)
            feature_values['latitude'] = float(latitude) if latitude else 20.5937
            feature_values['longitude'] = float(longitude) if longitude else 78.9629
            
            smu_id_value = "User-provided"
            
        # MODE 2: Lat/Lon provided - lookup soil data
        elif latitude is not None and longitude is not None:
            print(f"Mode: Lat/Lon lookup")
            latitude = float(latitude)
            longitude = float(longitude)
            
            # Get SMU_ID from raster
            smu_id_value = None
            try:
                for val in hwsd_raster.sample([(longitude, latitude)]):
                    smu_id_value = val[0]
                    break
                
                if smu_id_value is None or smu_id_value == hwsd_raster.nodata:
                    return jsonify({'error': 'No soil data at coordinates'}), 404
                
                smu_id_value = int(smu_id_value)
            except Exception as e:
                return jsonify({'error': f'Raster error: {str(e)}'}), 500
            
            # Get soil characteristics from database
            smu_data_row = soil_data[soil_data['HWSD2_SMU_ID'] == smu_id_value]
            if smu_data_row.empty:
                return jsonify({'error': f'No soil data for SMU_ID: {smu_id_value}'}), 404
            
            soil_chars = smu_data_row.iloc[0]
            
            # Check for missing features
            base_features = ['COARSE', 'SAND', 'SILT', 'CLAY', 'ORG_CARBON', 'PH_WATER', 'CEC_CLAY', 'BULK']
            for feature in base_features:
                if pd.isna(soil_chars[feature]):
                    return jsonify({'error': f'Missing soil feature: {feature}'}), 404
            
            # Create feature dict
            feature_values = {f: soil_chars[f] for f in base_features}
            feature_values['latitude'] = latitude
            feature_values['longitude'] = longitude
            
        else:
            return jsonify({'error': 'Provide either lat/lon OR all soil characteristics'}), 400
        
        # Add derived features
        feature_values['sand_clay_ratio'] = feature_values['SAND'] / (feature_values['CLAY'] + 1)
        feature_values['silt_clay_ratio'] = feature_values['SILT'] / (feature_values['CLAY'] + 1)
        feature_values['texture_index'] = feature_values['SAND'] + feature_values['SILT'] - feature_values['CLAY']
        
        # Create input array
        X_input = np.array([[feature_values[feat] for feat in feature_names]])
        X_scaled = scaler.transform(X_input)
        
        # Hierarchical predictions
        predictions_by_level = {}
        
        for level in model_metadata['taxonomic_levels']:
            model = hierarchical_models[level]
            encoder = hierarchical_encoders[level]
            
            # Get prediction probabilities
            proba = model.predict_proba(X_scaled)[0]
            pred_class = model.predict(X_scaled)[0]
            
            # Get top 5 predictions
            top_indices = np.argsort(proba)[::-1][:5]
            
            level_predictions = []
            for idx in top_indices:
                taxon_name = encoder.classes_[idx]
                probability = float(proba[idx])
                
                if probability > 0.05:  # Only show predictions > 5%
                    level_predictions.append({
                        'name': taxon_name,
                        'probability': probability,
                        'confidence': 'high' if probability > 0.5 else 'medium' if probability > 0.2 else 'low'
                    })
            
            if level_predictions:
                predictions_by_level[level] = level_predictions
        
        if not predictions_by_level:
            return jsonify({'error': 'No predictions with sufficient confidence'}), 404
        
        # Format response (ensure all values are JSON serializable)
        response = {
            'predictions_by_level': predictions_by_level,
            'soil_characteristics': {k: float(round(v, 2)) for k, v in feature_values.items() if k in base_features},
            'location': {
                'latitude': float(feature_values['latitude']),
                'longitude': float(feature_values['longitude']),
                'smu_id': str(smu_id_value)
            },
            'model_info': {
                'type': 'hierarchical',
                'levels': list(predictions_by_level.keys()),
                'metrics': {level: {
                    'f1_score': float(model_metadata['metrics'][level]['f1_score']),
                    'accuracy': float(model_metadata['metrics'][level]['accuracy']),
                    'n_classes': int(model_metadata['metrics'][level]['n_classes'])
                } for level in predictions_by_level.keys()}
            }
        }
        
        # Generate report
        location_info = {
            'latitude': feature_values.get('latitude', 0),
            'longitude': feature_values.get('longitude', 0)
        }
        enable_ai = data.get('enable_ai', True)  # Get AI toggle state from request
        report_path = generate_hierarchical_report(
            smu_id_value, 
            feature_values, 
            predictions_by_level,
            location_info,
            model_metadata['metrics'],
            enable_ai
        )
        response['report_path'] = report_path
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

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
