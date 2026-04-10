# Weather Prediction ML Project (Flask Dashboard)

This project trains a machine learning model to predict **rainfall (mm)** using weather features and serves predictions in a simple **Flask dashboard**.

## Features
- Included dataset: `data/weather_data.csv`
- Model training script: `train.py`
- Saved trained model: `models/weather_model.joblib`
- Flask dashboard: `app.py` + `templates/index.html`

## Project Structure

```text
.
├── app.py
├── train.py
├── requirements.txt
├── data/
│   └── weather_data.csv
├── models/
│   └── weather_model.joblib
└── templates/
    └── index.html
```

## 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Train the model

```bash
python train.py
```

This will:
- Load the dataset from `data/weather_data.csv`
- Train a Random Forest Regressor
- Print MAE and R²
- Save model to `models/weather_model.joblib`

## 3) Run dashboard

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## Inputs expected in dashboard
- Temperature (°C)
- Humidity (%)
- Wind Speed (km/h)
- Pressure (hPa)

Output:
- Predicted rainfall (mm)

## Notes
- Dataset is synthetic but realistic enough for end-to-end practice.
- You can replace `data/weather_data.csv` with a real weather dataset later.
