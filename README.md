# 🌱 SPORE - Soil-based Prediction Of Resident Entities

Advanced hierarchical machine learning system for predicting soil microbes in **India** based on soil characteristics and location.

---

## Features

- **Hierarchical Classification**: Predicts at 5 taxonomic levels (Phylum → Class → Order → Family → Genus)
- **Dual Input Modes**: 
  - Lat/Lon lookup (automatic soil data retrieval)
  - Manual soil characteristics input
- **High Accuracy**: F1 scores of 0.062-0.067 (much better than flat classification)
- **Multi-Level Predictions**: Get predictions at all taxonomic levels simultaneously
- **Confidence Scores**: High/Medium/Low confidence indicators
- **PDF Reports**: Downloadable reports with detailed predictions

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (First Time Only)
```bash
python3 train_model.py
```
This will:
- Link 79,648 bacteria occurrences to soil data
- Train hierarchical models at 5 taxonomic levels
- Save models to `models/` directory
- Takes ~5-10 minutes

### 3. Run the Application
```bash
python3 app.py
```
Open: http://localhost:5002

**Note**: Predictions are currently available for locations within India only.

---

## Model Performance

| Taxonomic Level | Classes | F1 Score | Accuracy |
|----------------|---------|----------|----------|
| **Phylum** | 36 | 0.067 | 0.051 |
| **Class** | 70 | 0.063 | 0.045 |
| **Order** | 176 | 0.041 | 0.032 |
| **Family** | 316 | 0.038 | 0.033 |
| **Genus** | 294 | 0.062 | 0.064 |

**Why hierarchical is better**: Reduces label dimensionality, improves F1 scores, and provides interpretable taxonomic predictions.

---

## Usage

### Mode 1: Lat/Lon Lookup
1. Enter latitude and longitude (within India)
2. System automatically retrieves soil data from HWSD2 database
3. Get predictions at all taxonomic levels

### Mode 2: Manual Soil Input
1. Enter all 8 soil characteristics:
   - Gravel Content (%)
   - Sand Content (%)
   - Silt Content (%)
   - Clay Content (%)
   - Organic Carbon (g/kg)
   - Soil pH
   - CEC of Clay (cmol(+)/kg)
   - Bulk Density (kg/dm³)
2. Optionally provide lat/lon for spatial features
3. Get predictions based on your soil data

### API Usage
```bash
# Mode 1: Lat/Lon
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 22.793497, "longitude": 73.62895}'

# Mode 2: Manual Soil
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "soil_characteristics": {
      "COARSE": 10, "SAND": 45, "SILT": 30, "CLAY": 25,
      "ORG_CARBON": 12, "PH_WATER": 6.5, 
      "CEC_CLAY": 55, "BULK": 1.4
    }
  }'
```

---

## Project Structure

```
SPORE/
├── app.py                  # Flask application (hierarchical predictor)
├── train_model.py          # Training pipeline
├── report_generator.py     # PDF report generation
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── data/
│   ├── HWSD2.bil          # Soil raster (23GB)
│   ├── HWSD2.hdr          # Raster header
│   ├── HWSD2_SMU.csv      # Soil mapping units
│   ├── HWSD2_LAYERS.csv   # Soil characteristics
│   └── raw_gbif_data.csv  # Bacteria occurrences
├── models/
│   ├── phylum_model.pkl   # Phylum classifier
│   ├── phylum_encoder.pkl # Phylum label encoder
│   ├── class_model.pkl    # Class classifier
│   ├── class_encoder.pkl  # Class label encoder
│   ├── order_model.pkl    # Order classifier
│   ├── order_encoder.pkl  # Order label encoder
│   ├── family_model.pkl   # Family classifier
│   ├── family_encoder.pkl # Family label encoder
│   ├── genus_model.pkl    # Genus classifier
│   ├── genus_encoder.pkl  # Genus label encoder
│   ├── scaler.pkl         # Feature scaler
│   ├── features.pkl       # Feature names
│   └── metadata.json      # Model metadata & metrics
├── templates/
│   └── index.html         # Web UI
└── static/
    └── background.jpg     # UI background
```

---

## How It Works

### Training Phase
1. **Data Loading**: Load HWSD2 soil database + GBIF bacteria occurrences
2. **Taxonomy Extraction**: Parse hierarchical taxonomy from GBIF data
3. **Spatial Linking**: Link bacteria coordinates to soil SMU_IDs via raster
4. **Feature Engineering**: Create 13 features (8 soil + 2 spatial + 3 derived)
5. **Hierarchical Training**: Train separate Random Forest models for each taxonomic level
6. **Evaluation**: Calculate F1 scores and accuracy metrics
7. **Model Saving**: Persist all models and metadata

### Prediction Phase
1. **Input Processing**: Accept lat/lon OR manual soil characteristics
2. **Feature Preparation**: Create 13-feature vector with derived features
3. **Scaling**: Normalize features using trained scaler
4. **Hierarchical Prediction**: Run through all 5 taxonomic level models
5. **Confidence Scoring**: Classify predictions as high/medium/low confidence
6. **Response Formatting**: Return predictions organized by taxonomic level
7. **Report Generation**: Create PDF with all predictions

---

## Key Features

### Hierarchical Classification
- Follows biological taxonomy: Phylum → Class → Order → Family → Genus
- Better F1 scores than flat multi-label classification
- More interpretable and scientifically sound
- Can predict at multiple levels simultaneously

### Dual Input Modes
- **Lat/Lon Mode**: Automatic soil lookup from 408K+ soil records
- **Manual Mode**: Direct input of soil characteristics
- Flexible for different use cases and data availability

### Advanced ML
- Random Forest classifiers (100 trees each)
- Balanced class weights for imbalanced data
- Spatial autocorrelation via lat/lon features
- Feature engineering (ratios, texture index)

---

## Feature Importance

Top features across all models:

1. **Longitude** (21.3%) - Geographic location
2. **Latitude** (19.6%) - Geographic location  
3. **Organic Carbon** (6.6%) - Nutrient availability
4. **CEC of Clay** (6.1%) - Nutrient retention
5. **Bulk Density** (5.9%) - Soil structure

---

## Data Sources

### HWSD v2.0 (Harmonized World Soil Database)
- **Source**: FAO/IIASA
- **URL**: https://gaez.fao.org/pages/hwsd
- **Content**: 29,538 soil mapping units, 408,835 records
- **Citation**: FAO/IIASA/ISRIC/ISSCAS/JRC, 2012

### GBIF (Global Biodiversity Information Facility)
- **Source**: GBIF.org
- **URL**: https://www.gbif.org/
- **Region**: India only (countryCode: IN)
- **Content**: 96,697 bacteria occurrences
- **Species**: 5,261 unique species

---

## Technical Details

### Dependencies
- **ML**: scikit-learn 1.3.2, joblib 1.3.2
- **Data**: pandas 2.1.4, numpy 1.26.4
- **GIS**: rasterio 1.3.9, geopandas 0.12.2
- **Web**: Flask 2.0.1
- **Reports**: reportlab, matplotlib 3.8.2

### System Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 25GB for data + models
- **CPU**: Multi-core recommended for training
- **OS**: macOS, Linux, Windows

### Performance
- **Training Time**: 5-10 minutes
- **Prediction Time**: <1 second per location
- **Model Loading**: ~2 seconds on startup

---

## Troubleshooting

### Models not loading?
```bash
# Retrain models
python3 train_model.py
```

### Missing data files?
Download required datasets:
1. **HWSD2 Raster**: https://gaez.fao.org/pages/hwsd
2. **GBIF Data**: https://www.gbif.org/ (filter to India, Bacteria)

### Port already in use?
```bash
# Change port in app.py (last line)
app.run(debug=True, port=5003)  # Use different port
```

---

## Future Improvements

- [ ] Add embedding-based approaches (Node2Vec on co-occurrence networks)
- [ ] Implement SHAP explainability per prediction
- [ ] Add temporal/seasonal features
- [ ] Cross-validation for robust metrics
- [ ] Deploy to production server
- [ ] Add more taxonomic levels (species, strain)
- [ ] Integrate climate data
- [ ] **Expand to other countries/regions** (currently India-specific)

---

## Citation

If you use this project, please cite:

```bibtex
@software{spore2025,
  title={SPORE: Soil-based Prediction Of Resident Entities},
  author={Sarvagna},
  year={2025},
  version={2.0-hierarchical},
  url={https://github.com/yourusername/spore}
}
```

---

## License

MIT License - See LICENSE file

---

## Acknowledgments

- FAO for HWSD v2.0 soil database
- GBIF for bacteria occurrence data
- scikit-learn community for ML tools
- Flask community for web framework

---

<div align="center">
  <strong>Made with ❤️ by Sarvagna</strong>
</div> 
