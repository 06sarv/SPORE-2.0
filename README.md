# SPORE

SPORE is a prototype that estimates which bacterial taxa are associated with soil and location inputs in India. It joins GBIF occurrence records to HWSD2 soil-map values and returns rankings at five taxonomic levels.

**A prediction is not confirmation that a microbe is present.** The current training labels are presence-only occurrence records, not a consistent set of sequenced soil samples with verified absences. The model can support exploration and sampling decisions; it is not a laboratory identification tool.

## What the app does

- Looks up topsoil characteristics from latitude and longitude, or accepts manual soil values.
- Uses five separate XGBoost classifiers to rank phylum, class, order, family, and genus labels.
- Generates a PDF with input values, ranked labels, model metrics, and an optional local Ollama text section.
- Displays raw probability bands. The High/Medium/Low labels are not calibrated confidence estimates.

The five classifiers predict each taxonomic level separately. Their outputs are not guaranteed to form one consistent parent-to-child taxonomy path.

## Current model and evaluation

The local app bundle is version `2.1-fixed`. Its random-split evaluation is:

| Taxonomic level | Retained classes | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|
| Phylum | 15 | 58.0% | 0.514 | 0.110 |
| Class | 23 | 47.8% | 0.413 | 0.082 |
| Order | 47 | 32.6% | 0.278 | 0.090 |
| Family | 72 | 29.6% | 0.231 | 0.109 |
| Genus | 97 | 27.7% | 0.217 | 0.113 |

The evaluation uses a random 80/20 split of GBIF occurrences (seed 42), before the soil join. Taxon-support filtering, median imputation, and scaling are fit on training data only. Soil features use the same SHARE-weighted shallowest-layer aggregation as the app. The fixed 250-tree XGBoost models do not use the holdout for early stopping. After evaluation, deployment models are refit on all records with labels supported by the training partition. The evaluation has 10,672 training and 2,621 validation occurrences after filtering. Full metrics and settings are in [`evaluation/v2_random_split_train_only_preprocessing.json`](evaluation/v2_random_split_train_only_preprocessing.json); the local app bundle metadata is in [`models/metadata.json`](models/metadata.json).

These scores are preliminary. The random split does not hold out whole regions or studies, so it does not measure transfer to new places. Macro F1 is much lower than accuracy, indicating uneven performance across labels. GBIF occurrence records are spatially and taxonomically biased, and linking them to mapped soil values does not establish a sequenced soil sample or an absence label. Earlier v2 scores used preprocessing and early stopping that involved the full dataset or holdout; they are historical development metrics and are superseded here.

## Synthetic controlled benchmark

Run `python3 synthetic_demo.py` to generate 6,000 simulated records and compare Random Forest, XGBoost, and a majority-class baseline on a random split and three held-out regions. The generator plants soil-profile/taxon relationships, adds 6% label noise and 1% missingness per soil feature, and creates a nested taxonomy. In the recorded run, genus accuracy on held-out regions was 91.9% for Random Forest, 91.1% for XGBoost, and 3.0% for the majority baseline.

The synthetic labels were generated from soil profiles, so these scores demonstrate code behavior under controlled assumptions only. They are not real microbial prediction scores and must not be compared with the GBIF evaluation.

## Run the app locally

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Start the app

The app needs local HWSD2/GBIF data files and model `.pkl` files under `data/` and `models/`. These large/local artifacts are ignored by Git and are not included in a fresh clone. Provide licensed data files and a compatible trained model bundle before starting the app.

```bash
python3 app.py
```

Open [http://127.0.0.1:5002](http://127.0.0.1:5002).

A tested smoke input is latitude `22.793497`, longitude `73.62895`. For manual mode, try `COARSE=10`, `SAND=35`, `SILT=35`, `CLAY=30`, `ORG_CARBON=12`, `PH_WATER=6.8`, `CEC_CLAY=25`, and `BULK=1.3`.

### Optional local AI report text

Install Ollama from its official source, make sure its service is running, and make `mistral:7b` available:

```bash
ollama pull mistral:7b
ollama serve
```

Enable **AI Explanations** in the app and use **Download Full Report** after a prediction. The Mistral output is generated locally, but it may still make unsupported ecological claims despite the cautious prompt. Treat it as unverified text, not scientific evidence. Reports can be generated without Ollama by disabling the AI option.

## Training scripts

- `train_model_v2_fixed.py` evaluates a random occurrence holdout and creates an app-compatible candidate bundle using app-matched topsoil aggregation. It expects the local GBIF and HWSD2 files and writes to `models_v2_fixed_candidate/` by default.
- `train_model_v3.py` is an experimental source-aware XGBoost pipeline with H3-cell or study-group validation. A successful smoke run is not evidence of improved predictive quality; promote a v3 bundle only after full training and review on held-out regions or studies.
- `train_model_fast.py` is the historical v2 XGBoost trainer. Its evaluation has leakage risks and should not be used to claim unbiased performance.
- `train_model.py` is a separate legacy Random Forest pipeline; it does not reproduce the XGBoost bundle used by the app.
- `synthetic_demo.py` compares tree models on generated data only.

## API examples

Coordinate prediction:

```bash
curl -X POST http://127.0.0.1:5002/predict \
  -H 'Content-Type: application/json' \
  -d '{"mode":"latlon","latitude":22.793497,"longitude":73.62895}'
```

Manual soil prediction (all eight values are required):

```bash
curl -X POST http://127.0.0.1:5002/predict \
  -H 'Content-Type: application/json' \
  -d '{"mode":"manual","COARSE":10,"SAND":35,"SILT":35,"CLAY":30,"ORG_CARBON":12,"PH_WATER":6.8,"CEC_CLAY":25,"BULK":1.3,"enable_ai":false}'
```

## Data sources

- **HWSD v2.0:** FAO/IIASA/ISRIC/ISSCAS/JRC Harmonized World Soil Database. The app uses mapped soil properties, aggregated to the shallowest layer by soil mapping unit.
- **GBIF:** India-filtered bacterial occurrence export, cited as GBIF.org (2 October 2025), [https://doi.org/10.15468/dl.z3ysfj](https://doi.org/10.15468/dl.z3ysfj). Occurrences are not equivalent to paired soil microbiome sequencing samples.
- Source review and acquisition notes are in [`SOURCES.md`](SOURCES.md). Data licenses and redistribution rights must be checked for each release.

## Repository and local artifacts

The Git repository contains source code, documentation, evaluation metadata, and synthetic-demo code. Local `data/`, `models/*.pkl`, generated reports, `.env`, and the Ollama model are excluded from Git. The tracked `models/metadata.json` describes the local bundle but does not contain its model weights. A fresh clone therefore needs local data and compatible model artifacts before `app.py` can serve predictions.

## License

MIT License. See [LICENSE](LICENSE).
