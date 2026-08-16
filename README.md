# Dublin Bikes Availability Predictor

## What the project does
A full-stack machine learning application that predicts the availability of bikes at Dublin Bikes stations using historical time-series data. 

## Problem statement
Urban bike-sharing systems suffer from severe rebalancing issues, where stations become completely empty or entirely full during peak hours. Predicting these states allows for proactive fleet management and improves the user experience by guaranteeing bike availability.

## Architecture
An end-to-end ML application featuring modular data processing, an XGBoost model, a REST API inference server, and a reactive frontend UI.

## Dataset
Uses the official Dublin Bikes historical dataset. The current project scope utilizes a one-month slice (November 1st - November 30th, 2021) containing station status, timestamps, and capacity metrics.

## Feature engineering
Engineered strict chronological time-series features to capture trends:
- **Lags**: Past available bikes at t-1, t-2, t-4, t-8.
- **Rolling Windows**: 4-step and 8-step rolling means and standard deviations.
- **Cyclical Encoding**: Sine and cosine transformations for Hour and Day of Week.
- **Leakage Prevention**: All rolling calculations utilize `.shift(1)` to strictly prevent future data from leaking into the training set.

## ML approach
A modular `scikit-learn` Pipeline utilizing an `XGBRegressor`. Categorical variables (Station ID, Status) are handled via `OneHotEncoder(handle_unknown="ignore")`, while numeric features are median-imputed. Data is split chronologically to mirror production realities.

## Model results
The model was evaluated on a strict chronological test set, yielding strong predictive performance. Predictions are mathematically clipped between 0 and the station's maximum bike capacity.

**Test Set Metrics (XGBoost):**
- **MAE:** 1.55 bikes
- **RMSE:** 2.53 bikes
- **R²:** 0.91

## Inference architecture
The system employs a historical time-travel inference pattern. Instead of deploying a complex live feature store, the backend loads the pre-engineered dataset into an in-memory Pandas dataframe. It dynamically slices historical data up to the requested target time to simulate real-time feature retrieval and predict the next immediate time step.

## FastAPI
The backend is a lightweight `FastAPI` service with strict `Pydantic` schema validation. It is configured to serve requests from the frontend with explicit CORS configuration.

## React frontend
A decoupled Single Page Application (SPA) built with React and TypeScript. It features a modern slate-themed UI with `datetime-local` calendar constraints strictly bound to the dataset's valid dates.

## Running locally

**1. Train the Model**
```bash
python main.py
```

**2. Start Backend API**
```bash
uvicorn api.main:app --reload
```
API runs at `http://localhost:8000` (Swagger docs at `/docs`).

**3. Start Frontend UI**
```bash
cd frontend
npm install
npm run dev
```
UI runs at `http://localhost:5173`.

## Docker
The entire stack is containerized for reproducible deployments.

To build and run the services simultaneously:
```bash
docker-compose up --build
```
This spins up:
- The backend API on `http://localhost:8000`
- The optimized Nginx frontend on `http://localhost:5173`

## Project structure
```text
├── api/             # FastAPI backend service
├── frontend/        # React + TypeScript UI
├── notebooks/       # EDA and feature prototyping
├── outputs/         # Serialized .joblib models
├── src/             # Core ML module (features, models)
├── main.py          # Training orchestration script
└── README.md
```

## Limitations
- **t+1 Forecasting**: The model is currently trained to predict only the immediate next 5-minute interval. It does not perform recursive forecasting.

## Future improvements
- Direct multi-step forecasting (t+15m, t+30m).
- Live API data ingestion via Apache Airflow.
- MLflow integration for experiment tracking.
