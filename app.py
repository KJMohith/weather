from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = Path("models/weather_model.joblib")
DATA_PATH = Path("data/weather_data.csv")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Please run `python train.py` first."
        )
    return joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    defaults = {
        "temperature_c": 25,
        "humidity_pct": 70,
        "wind_kmh": 12,
        "pressure_hpa": 1010,
    }

    if request.method == "POST":
        try:
            features = {
                "temperature_c": float(request.form.get("temperature_c", defaults["temperature_c"])),
                "humidity_pct": float(request.form.get("humidity_pct", defaults["humidity_pct"])),
                "wind_kmh": float(request.form.get("wind_kmh", defaults["wind_kmh"])),
                "pressure_hpa": float(request.form.get("pressure_hpa", defaults["pressure_hpa"])),
            }

            model = load_model()
            df = pd.DataFrame([features])
            prediction = max(0.0, float(model.predict(df)[0]))
            defaults = features
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

    sample_data = []
    if DATA_PATH.exists():
        data_df = pd.read_csv(DATA_PATH).head(40)
        sample_data = data_df.to_dict(orient="records")

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=defaults,
        sample_data=sample_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
