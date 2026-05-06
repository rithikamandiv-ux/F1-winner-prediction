# F1 Race Winner Prediction

Python-based machine learning pipeline that predicts Formula 1 race winners with leakage-safe features.

## Pipeline

1. `python fetch_data.py --years 2020-2026`
2. `python combine.py`
3. `python preprocess.py`
4. `python feature_engineering.py`
5. `python train_model.py`

## Notes

- Training seasons: `2020-2025`
- Test season: `2026`
- Leakage-safe features exclude post-race outcomes such as `Position`, `Points`, and `Laps` from model inputs.
- Raw season files are stored in `data_raw/`.
