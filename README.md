# HydraSense — Groundwater Level Monitoring and Prediction

HydraSense is a final-year B.E. IT project that presents DWLR (Dynamic Ground
Water Level Recorder) groundwater observations for a monitoring station and,
where a trained model is available, shows model-estimated forecasts. It follows
the five-page SRS: **Home, Dashboard, Historical Data, Predictions, About**.

Predictions are model estimates. They are not official groundwater
measurements and they are not accuracy guarantees.

## Stack

| Layer | Technology |
| --- | --- |
| Frontend | React 18 + Vite 6, React Router, Tailwind CSS |
| Charts | Chart.js (line chart with observed and predicted series) |
| Backend | Python + FastAPI (HTTP JSON API only) |
| Data processing | Pandas + NumPy |
| Model | Scikit-learn (per-horizon RandomForest / Ridge on the change from the origin level, with a persistence baseline) |
| Model artifact | `ml/model.pkl` (joblib) |
| Local storage | SQLite (`sqlite3`) for the forecast cache |
| Shared storage | PostgreSQL (optional, `POSTGRES_DSN`) for station metadata and prediction history |

## Project layout

```
backend/        FastAPI app, model loading/inference, SQLite + PostgreSQL helpers
frontend/       React SPA (five pages on one shared layout/header, shared chart and map)
ml/             data_preprocessing.py, train.py, evaluate.py, model.pkl
data/           DWLR_Dataset_2023.csv (single daily series, 365 rows for 2023)
stitch_pixel_perfect_ui_clone/   Design references the pages were ported from
```

## Data provenance and limitations (read this first)

- The dataset is **one daily series** with no state, district or station
  columns. The API therefore exposes exactly one station (`dwlr-001`) whose
  state/district/coordinates are explicitly marked `illustrative` (SRS 1.5, 11).
- The official DWLR source, licensing terms, measurement unit and depth datum
  are **not confirmed**. The unit is taken from the column name
  (`Water_Level_m`) and the API reports `unit_confirmed: false`.
- The bundled series looks synthetic (for example a non-zero rainfall value on
  every day of the year, implausible temperatures for the labelled location).
  Treat every value as sample data until an official extract replaces it.
- Only horizons 1–30 days are supported, because that is what the held-out
  evaluation covers.

## Run locally

### Backend (port 8000)

```bash
# Windows
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
cd backend && ./.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# macOS / Linux
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
cd backend && ../backend/.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>.

Configuration is optional. Copy `.env.example` to `.env` to set
`POSTGRES_DSN`; `.env` is loaded at startup and is now covered by `.gitignore`.
This checkout still tracks the original (empty) `.env`, so run
`git rm --cached .env` once to stop tracking it before adding any credential.
On first start the backend creates `.hydrasense.db` (SQLite) for the forecast
cache.

### Frontend (port 5173)

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies /api to 127.0.0.1:8000
npm run build    # production build in frontend/dist
npm run preview  # serve the built app
```

The browser only talks HTTP to the API; it never opens SQLite or PostgreSQL
(SRS 2.4).

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Service, dataset and model status (`ok`, `degraded`, `no_model`, `no_data`) |
| GET | `/api/stations` | Station catalogue with coverage and coordinates |
| GET | `/api/locations` | State → district → station tree for the filters |
| GET | `/api/groundwater` | Historical readings (`station_id`, `start`, `end`, `limit`, `aggregate=none|monthly`) |
| GET | `/api/predictions` | Forecast (`station_id`, `horizon_days` 1–90; clamped to the served horizon) |
| POST | `/api/predictions` | Same forecast with a JSON body |
| GET | `/api/metrics` | Held-out and backtest metrics recorded in the artifact |

Errors are JSON `{"detail": ...}`: `404` for an unknown station, `422` for an
invalid date range or horizon, `503` when the dataset or a usable model is
missing.

## Model

Training and selection (`ml/train.py`):

1. Clean a working copy of the CSV (no source file is modified).
2. Build origin-known features (lags, rolling means, rainfall/temperature,
   day-of-year sin/cos).
3. Candidates: persistence baseline, RandomForest and Ridge, both modelling the
   **change from the origin level**.
4. Select per horizon on a recency-weighted rolling-origin backtest with a 25%
   margin in favour of persistence.
5. Score the untouched final 60 days once (reporting only) and refit the
   selected candidate on all available origins.

Results recorded in the current `ml/model.pkl`:

| Metric | Served models | Persistence |
| --- | --- | --- |
| Backtest MAE (mean, selection) | **0.150 m** | 0.179 m |
| Held-out MAE (mean, final 60 days) | 0.121 m | **0.112 m** |

Learned models are served only for horizons 23–30 days; horizons 1–22 fall back
to persistence. On the held-out window the served mix does **not** beat
persistence overall, so the UI labels forecasts exploratory rather than
validated. The Predictions page reads these figures from the artifact; nothing
is hardcoded.

Reproduce the recorded numbers:

```bash
backend/.venv/Scripts/python.exe ml/evaluate.py   # exit 0 = every stored metric reproduced
backend/.venv/Scripts/python.exe ml/train.py      # optional: retrain ml/model.pkl
```

## Verification performed

- All endpoints and edge cases exercised: unknown station (404), inverted date
  range (422), empty range (404), out-of-range horizon/limit (422), monthly
  aggregation, forecast horizons 1/30/90, cache hit path.
- `ml/evaluate.py` re-derives every stored metric and selection decision and
  exits 0.
- `npm run build` completes; every page rendered in a browser against the live
  API with no console errors.
- Forecasts are cached in SQLite for one hour; the cache key includes the
  dataset and model artifact timestamps, so retraining invalidates it.

## Known gaps

- PostgreSQL paths are implemented but were not exercised against a live
  database in this workspace (`asyncpg` is not installed here).
- Mobile navigation is a shared bottom bar; page headers themselves stay
  desktop-only, matching the reference designs.
- `ml/notebook.ipynb` is an empty placeholder; exploratory work currently lives
  in `ml/train.py` / `ml/evaluate.py`.
- The initial commit also tracks `backend/__pycache__/*.pyc` and
  `backend/.venv-install.log`. The new `.gitignore` prevents new ones from being
  added, but those files need `git rm --cached` to leave the index.
