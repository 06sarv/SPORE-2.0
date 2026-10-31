# 🌱 SPORE - Soil-based Prediction Of Resident Entities

Advanced hierarchical machine learning system for predicting soil microbes in **India** based on soil characteristics and location.

---

## Features

- **Hierarchical Classification**: Predicts at 5 taxonomic levels (Phylum → Class → Order → Family → Genus)
- **Dual Input Modes**: 
  - Lat/Lon lookup (automatic soil data retrieval)
  - Manual soil characteristics input
- **Model evaluation**: Includes a leakage-controlled random occurrence split; scores are preliminary and do not establish performance in unseen regions
- **Multi-Level Predictions**: Get predictions at all taxonomic levels simultaneously
- **Probability bands**: High/Medium/Low labels are raw model-probability bands, not calibrated confidence
- **Professional PDF Reports**: Comprehensive scientific reports with:
  - Dynamic visualizations
  - Optional local Ollama-generated interpretation, explicitly labeled as a hypothesis
  - Executive summary, soil analysis, methodology sections
  - Professional scientific styling and formatting

---

## Model Performance

### Leakage-controlled v2.1 fixed bundle

| Taxonomic Level | Classes | Accuracy | Weighted F1 | Macro F1 |
|----------------|--------:|---------:|------------:|---------:|
| Phylum | 15 | 58.0% | 0.514 | 0.110 |
| Class | 23 | 47.8% | 0.413 | 0.082 |
| Order | 47 | 32.6% | 0.278 | 0.090 |
| Family | 72 | 29.6% | 0.231 | 0.109 |
| Genus | 97 | 27.7% | 0.217 | 0.113 |

These metrics are from a random 80/20 split of GBIF occurrences (seed 42), made before joining soil records. Taxon support, median imputation, and scaling are fit on training data only. Soil features use the app's SHARE-weighted shallowest soil layer. XGBoost uses a fixed 250 trees; the held-out set does not control early stopping. The evaluation record is [`evaluation/v2_random_split_train_only_preprocessing.json`](evaluation/v2_random_split_train_only_preprocessing.json), and `models/metadata.json` records the same metrics for the model now loaded by the app.

The split contains 10,672 training and 2,621 validation occurrences after train-only taxon filtering. Macro F1 is much lower than accuracy, showing that performance is uneven across labels. This remains a random split, so it does not measure transfer to unseen regions or studies. The prior v2 scores (52.6%, 42.3%, 26.8%, 23.2%, and 20.8% accuracy) came from a leaky evaluation and are retained only as historical context.

### Synthetic controlled benchmark

Run `python3 synthetic_demo.py` to generate 6,000 simulated records and compare Random Forest, XGBoost, and a majority-class baseline on a random split and three held-out regions. The generator plants soil-profile/taxon relationships, adds 6% label noise and 1% missingness per soil feature, and groups genera into a consistent taxonomy. For genus prediction on held-out regions, this run produced 91.9% accuracy for Random Forest, 91.1% for XGBoost, and 3.0% for the majority baseline.

These high scores are expected because the synthetic labels were generated from soil profiles. They demonstrate controlled pipeline behavior only; **they are not real microbial prediction scores and must not be compared directly with the GBIF results**.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Model training scripts

The checked-in v2 model metadata and saved model objects identify the bundle as **XGBoost**. `train_model.py` is a separate legacy **Random Forest** pipeline; `train_model_fast.py` is the v2 XGBoost trainer, whose historical random-split evaluation has the limitations described above. For a controlled code demonstration, run `python3 synthetic_demo.py`. For new real-data model claims, use the v3 source-aware pipeline and validate on held-out studies or regions before promoting its artifacts.

### 3. Setup Ollama (Optional - for AI explanations)
```bash
# Install Ollama
brew install ollama

# Pull Mistral model
ollama pull mistral:7b

# Start Ollama service (runs in background)
ollama serve
```

### 4. Run the Application
```bash
python3 app.py
```
Open: http://localhost:5002

**Note**: Predictions are currently available for locations within India only.

**Local smoke test**: try latitude `22.793497`, longitude `73.62895`, or manual soil values `COARSE=10`, `SAND=35`, `SILT=35`, `CLAY=30`, `ORG_CARBON=12`, `PH_WATER=6.8`, `CEC_CLAY=25`, `BULK=1.3`.

Ollama report text is generated locally, but it may still make unsupported ecological claims despite the cautious prompt. Treat it as unverified text, not scientific evidence. Reports remain available without Ollama when the AI option is disabled.

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
# Mode 1: Lat/Lon with AI explanations
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 22.793497, "longitude": 73.62895, "enable_ai": true}'

# Mode 2: Manual Soil without AI explanations
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "soil_characteristics": {
      "COARSE": 10, "SAND": 45, "SILT": 30, "CLAY": 25,
      "ORG_CARBON": 12, "PH_WATER": 6.5, 
      "CEC_CLAY": 55, "BULK": 1.4
    },
    "enable_ai": false
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
5. **Hierarchical Training**: Train separate XGBoost models for each taxonomic level
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
- XGBoost classifiers with optimized hyperparameters
- Feature engineering (15 features: 8 soil + derived ratios + spatial bins)
- Early stopping to prevent overfitting
- Stratified train/test split
---

## Feature Importance

Top features across all models:

1. **pH Squared** - Non-linear pH effects
2. **Carbon-pH Interaction** - Nutrient-pH relationship
3. **Organic Carbon** - Nutrient availability
4. **Sand/Clay Ratio** - Soil texture
5. **Latitude Bin** - Geographic patterns

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
- **Citation**: GBIF.org (2 October 2025) GBIF Occurrence Download https://doi.org/10.15468/dl.z3ysfj

---

## Technical Details

### Dependencies
- **ML**: XGBoost 2.0+, scikit-learn 1.7+, joblib
- **Data**: pandas, numpy, rasterio
- **Web**: Flask 2.0+
- **Reports**: reportlab, matplotlib, seaborn
- **AI**: Ollama (Mistral 7B for offline explanations)

### System Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 25GB for data + models
- **CPU**: Multi-core recommended for training
- **OS**: macOS, Linux, Windows

### Performance
- **Training Time**: 5-10 minutes
- **Prediction Time**: <1 second per location
- **Model Loading**: ~2 seconds on startup
- **Report Generation**: 
  - Without AI: ~10-15 seconds
  - With AI explanations: ~3-4 minutes (Mistral 7B)

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

### AI explanations not working?
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Or disable AI in the UI (uncheck the toggle)
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
