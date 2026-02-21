from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("temperature_model.pkl")

templates = Jinja2Templates(directory="templates")


class WeatherInput(BaseModel):
    Summary: str
    Precip_Type: str
    Humidity: float
    Wind_Speed_km_h: float
    Wind_Bearing_degrees: float
    Visibility_km: float
    year: int
    month: int
    day: int
    hour: int


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict")
def predict_temperature(data: WeatherInput):

    X = pd.DataFrame([{
        "Summary": data.Summary,
        "Precip Type": data.Precip_Type,
        "Humidity": data.Humidity,
        "Wind Speed (km/h)": data.Wind_Speed_km_h,
        "Wind Bearing (degrees)": data.Wind_Bearing_degrees,
        "Visibility (km)": data.Visibility_km,
        "year": data.year,
        "month": data.month,
        "day": data.day,
        "hour": data.hour
    }])

    prediction = model.predict(X)[0]

    return {
        "predicted_temperature_c": round(float(prediction), 2),
        "target": "Temperature (C)"
    }
