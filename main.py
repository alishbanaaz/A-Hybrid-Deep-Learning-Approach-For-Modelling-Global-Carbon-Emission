"""
FastAPI service for the "Hybrid SARIMAX+LSTM CO2 Emissions Forecasting" FYP.

Run with:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000 in a browser.

Endpoints
---------
GET  /api/data-summary        -> current dataset info (rows, years, countries)
GET  /api/countries           -> top emitting countries in the current dataset
POST /api/upload-data         -> upload a CSV to replace the working dataset
POST /api/reset-data          -> revert to the original bundled FYP dataset
POST /api/train/{country}     -> train the hybrid model for one country and
                                  return accuracy metrics + a future forecast
GET  /api/forecast/{country}  -> retrieve the last trained result for a country
"""
import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import model_pipeline as mp

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "global-data-on-sustainable-energy.csv"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="CO2 Hybrid Forecast API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# In-memory application state.
# For a student portfolio / demo app this is fine; for real multi-user
# deployment you'd move this into a database or per-session store.
# --------------------------------------------------------------------------
STATE: dict = {
    "df": None,          # currently active dataset
    "source": "default",  # "default" or "uploaded"
    "results": {},        # country -> last train_and_forecast() result
}


def _load_default() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA_PATH)


def _get_df() -> pd.DataFrame:
    if STATE["df"] is None:
        STATE["df"] = _load_default()
    return STATE["df"]


@app.on_event("startup")
def _startup():
    STATE["df"] = _load_default()
    STATE["source"] = "default"


# ---------------------------- data endpoints ------------------------------

@app.get("/api/data-summary")
def data_summary():
    df = _get_df()
    return {
        "source": STATE["source"],
        "rows": int(len(df)),
        "countries": int(df["Entity"].nunique()),
        "year_min": int(df["Year"].min()),
        "year_max": int(df["Year"].max()),
        "columns": list(df.columns),
    }


@app.get("/api/countries")
def countries(n: int = Query(10, ge=1, le=50)):
    df = _get_df()
    return {"countries": mp.top_countries(df, n=n)}


@app.post("/api/upload-data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    raw = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    missing = mp.validate_schema(new_df)
    if missing:
        raise HTTPException(
            400,
            "Uploaded CSV is missing required columns: " + ", ".join(missing) +
            ". Required columns are: " + ", ".join(mp.REQUIRED_COLS),
        )

    STATE["df"] = new_df
    STATE["source"] = f"uploaded:{file.filename}"
    STATE["results"] = {}  # invalidate cached models trained on the old data
    return data_summary()


@app.post("/api/reset-data")
def reset_data():
    STATE["df"] = _load_default()
    STATE["source"] = "default"
    STATE["results"] = {}
    return data_summary()


# ---------------------------- model endpoints ------------------------------

@app.post("/api/train/{country}")
def train_country(
    country: str,
    test_years: int = Query(5, ge=1, le=15, description="Held-out years used to report accuracy"),
    forecast_years: int = Query(5, ge=1, le=20, description="Years beyond the data to forecast"),
):
    df = _get_df()
    if country not in df["Entity"].unique():
        raise HTTPException(404, f"'{country}' not found in the current dataset.")

    try:
        result = mp.train_and_forecast(
            df, country, test_years=test_years, forecast_years=forecast_years
        )
    except mp.InsufficientDataError as e:
        raise HTTPException(400, str(e))

    STATE["results"][country] = result
    return result


@app.get("/api/forecast/{country}")
def get_forecast(country: str):
    if country not in STATE["results"]:
        raise HTTPException(
            404, f"No trained model for '{country}' yet — POST /api/train/{country} first."
        )
    return STATE["results"][country]


# ---------------------------- frontend ------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
