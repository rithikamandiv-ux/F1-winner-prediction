# F1 Winner Prediction

A machine learning-powered Formula 1 race winner prediction system built using Python, FastF1, Scikit-learn, and Streamlit.

This project collects historical Formula 1 race data, preprocesses and engineers features, trains a machine learning model using previous seasons, and predicts future race winners through an interactive dashboard interface.

---

# Overview

The system uses Formula 1 race data from multiple seasons to identify patterns related to race-winning performance. The machine learning pipeline processes qualifying data, driver performance metrics, constructor performance, race conditions, and other engineered features to predict likely race winners.

The project is structured as a complete end-to-end machine learning workflow including:

- Data collection
- Data preprocessing
- Feature engineering
- Model training
- Prediction generation
- Interactive UI visualization

---

# Features

- Historical Formula 1 data collection using FastF1
- Multi-season dataset generation
- Automated preprocessing pipeline
- Feature engineering system
- Machine learning-based race winner prediction
- Probability-based ranking predictions
- 2026 season testing workflow
- Interactive Streamlit dashboard
- Race-by-race prediction visualization
- Clean modular pipeline structure

---

# Tech Stack

## Programming Language

- Python

## Libraries and Frameworks

- Pandas
- NumPy
- Scikit-learn
- FastF1
- Streamlit
- Matplotlib

---

# Machine Learning Features

The current model uses the following features:

- Qualifying position (`q_pos`)
- Grid position
- Driver average finishing position
- Team average finishing position
- Driver recent form
- DNF rate
- Wet race indicator
- Race status
- Driver points
- Number of laps completed

Additional features can be added later to improve prediction quality.

---

# Project Structure

```text
F1-winner-prediction/
│
├── app.py
├── fetch_data.py
├── combine.py
├── preprocess.py
├── feature_engineering.py
├── train_model.py
│
├── predictions_2026.csv
├── f1_combined.csv
├── f1_processed.csv
├── f1_features.csv
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│
├── data_raw/
│   ├── f1_2020.csv
│   ├── f1_2021.csv
│   ├── f1_2022.csv
│   ├── f1_2023.csv
│   ├── f1_2024.csv
│   ├── f1_2025.csv
│   └── f1_2026.csv
│
└── cache/
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/rithikamandiv-ux/F1-winner-prediction.git
cd F1-winner-prediction
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Step 1 — Fetch Formula 1 Data

```bash
python fetch_data.py --years 2020-2026
```

This downloads historical race datasets using FastF1.

---

## Step 2 — Combine Datasets

```bash
python combine.py
```

This merges all yearly datasets into a single dataset.

---

## Step 3 — Preprocess Data

```bash
python preprocess.py
```

This step:
- cleans the dataset
- removes missing critical values
- formats columns
- prepares the dataset for ML

---

## Step 4 — Generate Features

```bash
python feature_engineering.py
```

This creates additional machine learning features including:
- average finishing positions
- recent driver form
- team performance metrics
- DNF statistics
- weather indicators

---

## Step 5 — Train the Machine Learning Model

```bash
python train_model.py
```

The model:
- trains using 2020–2025 seasons
- tests using 2026 races
- prints predicted vs actual winners
- generates prediction probabilities

---

# Running the Dashboard

Launch the Streamlit UI:

```bash
streamlit run app.py
```

The dashboard includes:
- race winner predictions
- probability rankings
- top 5 predicted drivers
- interactive visualizations
- race filtering system

---

# Current Workflow

| Dataset | Purpose |
|---|---|
| 2020–2025 | Training |
| 2026 | Testing / future race prediction |

---

# Example Prediction Output

```text
2026-Miami Grand Prix
Predicted winner: ANT
Actual winner: ANT
```

---

# Model Performance

The project currently achieves high prediction accuracy on available 2026 race data using engineered performance features and historical race patterns.

The system is intended primarily as:
- a machine learning project
- a data engineering project
- a sports analytics project

rather than a betting or real-world forecasting system.

---

# Future Improvements

Potential future additions include:

- Tyre strategy modeling
- Live qualifying integration
- Real-time race prediction updates
- Weather API integration
- Track-specific performance analysis
- Driver championship prediction
- Constructor championship prediction
- Ensemble learning models
- Deep learning experimentation
- Cloud deployment

---

# Requirements

Example dependencies used in this project:

```text
pandas
numpy
scikit-learn
streamlit
matplotlib
fastf1
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# License

This project is licensed under the MIT License.

---

# Author

Rithika Mandiv

GitHub:
https://github.com/rithikamandiv-ux
