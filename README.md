---
title: CO2 Hybrid Forecast
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# CO₂ Hybrid Forecast — FastAPI App

A web app around your FYP's **Hybrid SARIMAX + LSTM** model, so anyone can:

- See the original dataset (top 10 CO₂ emitters, 2000–2020)
- **Upload their own CSV** with the same columns to use different / updated data
- Pick any country and train the hybrid model live, with adjustable
  test-years and forecast-years
- View the forecast on a chart, alongside SARIMAX-only accuracy for comparison

It reuses the exact modelling logic from your notebooks
(`02_sarimax_model.ipynb`, `03_lstm_hybrid_model.ipynb`) — same hyperparameters,
same SARIMAX+LSTM-residual approach — just wrapped as callable functions
instead of notebook cells.

## Project layout

```
fyp_app/
└── backend/
    ├── main.py             FastAPI app (routes)
    ├── model_pipeline.py   The hybrid SARIMAX+LSTM logic
    ├── requirements.txt
    ├── data/                Default bundled dataset (your original CSV)
    └── static/index.html    Frontend (vanilla HTML/JS + Chart.js)
```

## Run it locally

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

> First install will take a few minutes because of `tensorflow` and `pmdarima`.
> Training a country typically takes 5–30 seconds (LSTM has early stopping).

## How "someone else can use the model" works

1. They open the app (locally, or wherever you deploy it).
2. They either use your bundled dataset, or upload their own CSV with
   these columns:
   `Entity, Year, Value_co2_emissions_kt_by_country, gdp_per_capita,
   gdp_growth, Primary energy consumption per capita (kWh/person),
   Renewable energy share in the total final energy consumption (%)`
3. They pick a country, set how many years to hold out for testing and how
   many years to forecast ahead, and click **Train Model & Forecast**.
4. The API retrains SARIMAX + the LSTM residual model on the current dataset
   for that country, and returns accuracy metrics + a future forecast, shown
   as a chart.

Each user's uploaded data lives only in that server's memory for the current
session (`POST /api/reset-data` reverts to your original dataset).

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/data-summary` | Current dataset info |
| GET | `/api/countries?n=10` | Top-N countries by mean emissions |
| POST | `/api/upload-data` | Upload a CSV (multipart `file`) to replace the working dataset |
| POST | `/api/reset-data` | Revert to the original bundled dataset |
| POST | `/api/train/{country}?test_years=5&forecast_years=5` | Train + forecast for one country |
| GET | `/api/forecast/{country}` | Fetch the last trained result for a country |

Interactive Swagger docs are auto-generated at **http://127.0.0.1:8000/docs**.

## Deploy to Hugging Face Spaces (Docker)

This repo already includes a `Dockerfile` and the YAML block at the very top
of this README that Spaces reads as config (`sdk: docker`, `app_port: 7860`).

**Steps:**

1. Go to https://huggingface.co/new-space
2. Pick a name (e.g. `co2-hybrid-forecast`), set **SDK = Docker**, visibility
   Public or Private, then click **Create Space**.
3. Push this whole `fyp_app/` folder's contents to the Space's repo — either:
   - **Web UI**: on the Space page, use "Files" → "Add file" → "Upload files"
     and drag in everything (keep the folder structure: `Dockerfile`,
     `README.md`, `backend/...`).
   - **Git** (recommended once you have more changes to push):
     ```bash
     git clone https://huggingface.co/spaces/<your-username>/co2-hybrid-forecast
     cd co2-hybrid-forecast
     # copy this fyp_app folder's contents in here (Dockerfile, README.md, backend/)
     git add .
     git commit -m "Deploy CO2 hybrid forecast app"
     git push
     ```
     You'll need a Hugging Face access token (Settings → Access Tokens) as
     your git password when it prompts for auth.
4. The Space will build automatically (watch the "Logs" tab — first build
   takes several minutes because of `tensorflow`/`pmdarima`). Once it says
   "Running", your app is live at:
   `https://huggingface.co/spaces/<your-username>/co2-hybrid-forecast`
5. That's a real, shareable link — put it in your FYP report, LinkedIn post,
   or CV.

**A few Spaces-specific notes:**
- Free-tier Spaces sleep after a period of inactivity and take ~30-60s to
  wake up on the next visit — normal, not a bug.
- Uploaded CSVs and trained results live only in that container's memory, so
  they reset whenever the Space restarts/sleeps. Good enough for a portfolio
  demo; say so if you want persistence added later (e.g. via a small database
  or Spaces' persistent storage add-on).
- If the free CPU tier feels slow for LSTM training, Spaces lets you upgrade
  to a paid CPU/GPU tier from the Space's **Settings** tab.

## Other deployment options

Render, Railway, and Fly.io all work too if you'd rather not use Spaces —
same `Dockerfile` works there as-is (they auto-detect it), or you can point
them at `uvicorn main:app --host 0.0.0.0 --port $PORT` directly.

## Notes / things worth knowing before you demo it

- Training is done **in-memory per request** — there's no database, no
  authentication, and no multi-user isolation. Fine for a portfolio/demo app,
  not production-ready as-is.
- A country needs at least 8 years of usable rows (target + all 4 exogenous
  columns present) to train; the API returns a clear 400 error otherwise.
- The LSTM residual model is retrained from scratch on every `/api/train`
  call (matches your notebook's approach — no persisted `.h5` weights).
