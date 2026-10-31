# SPORE v3: Data and Training Protocol

SPORE predicts likely taxa from environmental context. It is **not** a laboratory identification tool and must not claim organism presence, abundance, pathogenicity, or clinical/agricultural safety. A high predicted probability ranks a taxon within the training distribution; it is not proof of detection.

## Data Standard

Keep raw downloads outside Git in `data/raw/`. For every release, add a provenance record in `data/provenance/` containing the source URL or DOI, download date, query, license, geographic scope, sequencing/identification method, and preprocessing version.

Convert each reviewed source to a canonical CSV or Parquet file with:

```text
decimalLatitude,decimalLongitude,phylum,class,order,family,genus,source
```

Recommended optional fields are `coordinateUncertaintyInMeters`, `countryCode`, `year`, `license`, `sample_type`, `study_id`, and `taxonomy_database`. Use `prepare_v3_dataset.py` on GBIF exports. For MGnify and EMP, export only soil-associated samples with coordinates and resolved taxonomic tables, then produce this canonical schema. Do not mix amplicon and shotgun labels blindly; retain `method` and evaluate them separately if both are used.

## Source Policy

- **MGnify** should be the primary source for observed microbiome samples and sample metadata.
- **Earth Microbiome Project** is strong supplemental soil data where metadata and taxonomic processing are compatible.
- **GBIF** is occurrence evidence only. It improves coverage but contains observation and collection bias and is not a soil sequencing dataset.
- **GTDB** is the preferred bacterial/archaeal taxonomy backbone. Normalize to one frozen GTDB release and retain the original label plus release ID. Use NCBI Taxonomy cross-references when GTDB does not cover a record.

The training release must contain only sources whose license permits the intended use. “Publicly downloadable” is not a license.

## Environmental Data

`train_model_v3.py` optionally joins an `environment.csv` table by `h3_cell`. It needs one `h3_cell` column and numeric covariates, for example:

```text
h3_cell,annual_precip_mm,annual_temp_c,land_cover_code,elevation_m
```

Generate H3 cells at the same `--h3-resolution` used in training. Candidate additions are climate normals, elevation/slope, land cover, and soil chemistry. Check resolution, temporal reference period, license, missingness, and live-serving availability before including any source. Never train a feature that production cannot reproduce.

## Colab Runbook

1. Upload this repository and reviewed input tables to Drive. Do not upload credentials or unreviewed public dumps into Git.
2. In a GPU runtime, install focused dependencies:

   ```bash
   pip install "xgboost>=2" "h3>=4.1,<5" rasterio pandas scikit-learn pyarrow joblib
   ```

3. Build canonical inputs and retain provenance:

   ```bash
   python prepare_v3_dataset.py --input data/raw/gbif.tsv --source gbif --country IN --output data/processed/gbif.parquet
   ```

4. Train with a spatial holdout, not a random split:

   ```bash
   python train_model_v3.py --records data/processed/gbif.parquet data/processed/mgnify.parquet data/processed/emp.parquet --output-dir models_v3 --device cuda
   ```

5. For a global model, run both validation modes and keep the reports with the model release:

   ```bash
   python train_model_v3.py --records data/processed/*.parquet --output-dir models_v3_h3 --device cuda --validation-group h3
   python train_model_v3.py --records data/processed/*.parquet --output-dir models_v3_study --device cuda --validation-group study
   ```

   Promote a model only if genus-level macro F1, per-class recall, calibration, and continental/geographic error maps improve on completely held-out studies and regions. One weighted F1 score is not a release criterion.
6. Serve a promoted bundle without copying files:

   ```bash
   SPORE_MODEL_DIR=models_v3 python app.py
   ```

   If environmental features were used, production must load the exact reviewed H3 feature table before this is enabled.

## Model Selection

Begin with GPU XGBoost: this is tabular data and it is a strong fast baseline. Benchmark CatBoost only on identical H3-grouped folds; do not pick a model from random-split results. Neural networks are not automatically better here; boosted trees will be more reliable until sample count, labels, and spatial coverage are much stronger.

## Release Gate

- Validate coordinates, uncertainty, habitat, date, taxonomy release, and license.
- Deduplicate repeated records and prevent the same study/H3 cell from crossing train/test.
- Publish weighted F1, macro F1, per-class recall, calibration, confusion matrices, and geographic error maps.
- Report sources, sample count, H3 resolutions, feature version, and known geographic gaps in every generated report.
- Test the Flask API on manual input, in-bounds coordinates, unmapped coordinates, and the exact promoted model bundle.
