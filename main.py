import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "random_forest_model.joblib"))
COLUMNS_PATH = Path(os.getenv("COLUMNS_PATH", BASE_DIR / "model_columns.joblib"))
DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "dataset.csv"))

TARGET = "Current_Drought_Stress_Category"
DROP_COLUMNS = [
    "Record_ID", "Plot_ID", "Farm_ID",
    "Observation_Date", "Observation_Time", "Observation_Datetime",
    "Time_of_Day",
    "Forecasted_Soil_Moisture_24h_Pct",
    "Forecasted_Soil_Moisture_48h_Pct",
    "Forecasted_Soil_Moisture_72h_Pct",
]
ID_COLUMNS = {"Record_ID", "Farm_ID", "Plot_ID", "Observation_Date",
              "Observation_Time", "Observation_Datetime", "obs_dt_time"}

app = FastAPI(
    title="CropWise API",
    description="ML-based crop stress prediction and irrigation priority decision support.",
    version="1.0.0",
)

# For local development. Restrict allow_origins to your frontend URL when deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_columns: List[str] = []
training_df: Optional[pd.DataFrame] = None
feature_defaults: Dict[str, Any] = {}
raw_feature_columns: List[str] = []
class_order = ["Low", "Moderate", "High", "Severe", "Waterlogging Risk"]

def load_artifacts():
    global model, model_columns, training_df, feature_defaults, raw_feature_columns
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file missing: {MODEL_PATH}. Copy random_forest_model.joblib beside main.py.")
    if not COLUMNS_PATH.exists():
        raise RuntimeError(f"Column file missing: {COLUMNS_PATH}. Copy model_columns.joblib beside main.py.")
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset missing: {DATA_PATH}. Copy the same training CSV as dataset.csv beside main.py.")
    model = joblib.load(MODEL_PATH)
    model_columns = list(joblib.load(COLUMNS_PATH))
    raw = pd.read_csv(DATA_PATH)
    raw = raw.drop(columns=["obs_dt_time"], errors="ignore")
    train = raw.drop(columns=DROP_COLUMNS, errors="ignore")
    if TARGET in train.columns:
        train = train.drop(columns=[TARGET])
    raw_feature_columns = list(train.columns)
    # Match the notebook's preprocessing: numeric median, categorical mode, then pd.get_dummies.
    for col in train.columns:
        if train[col].isna().any():
            if pd.api.types.is_numeric_dtype(train[col]):
                val = train[col].median()
                feature_defaults[col] = float(val) if pd.notna(val) else 0.0
            else:
                modes = train[col].mode(dropna=True)
                feature_defaults[col] = modes.iloc[0] if len(modes) else "Unknown"
        else:
            if pd.api.types.is_numeric_dtype(train[col]):
                feature_defaults[col] = float(train[col].median())
            else:
                modes = train[col].mode(dropna=True)
                feature_defaults[col] = modes.iloc[0] if len(modes) else "Unknown"
    training_df = raw

def preprocess_one(payload: Dict[str, Any]) -> pd.DataFrame:
    # Only use the exact predictor columns retained by the notebook after DROP_COLUMNS.
    row = {}
    for col in raw_feature_columns:
        value = payload.get(col, feature_defaults.get(col))
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = feature_defaults.get(col)
        row[col] = value
    frame = pd.DataFrame([row], columns=raw_feature_columns)
    # Coerce numeric columns using the original training CSV schema.
    for col in raw_feature_columns:
        if pd.api.types.is_numeric_dtype(training_df[col]):
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(feature_defaults[col])
        else:
            frame[col] = frame[col].fillna(feature_defaults[col]).astype(str)
    encoded = pd.get_dummies(frame)
    # Critical: preserve the exact training feature order and columns.
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    return encoded

def priority_details(risk: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Transparent heuristic for ranking; it is NOT an ML-predicted irrigation quantity.
    severity = {
        "Low": 10, "Moderate": 35, "High": 65, "Severe": 90,
        "Waterlogging Risk": 0
    }.get(risk, 25)
    moisture = payload.get("Soil_Moisture_Current_10cm_Pct")
    fc = payload.get("Field_Capacity_Pct")
    pwp = payload.get("Permanent_Wilting_Point_Pct")
    moisture_component = 50.0
    if moisture is not None and fc is not None and pwp is not None:
        try:
            moisture, fc, pwp = float(moisture), float(fc), float(pwp)
            available = max(fc - pwp, 0.001)
            depletion = np.clip((fc - moisture) / available, 0, 1)
            moisture_component = float(depletion * 100)
        except (TypeError, ValueError):
            pass
    stage = str(payload.get("Crop_Growth_Stage", "")).strip().lower()
    stage_weights = {
        "germination": 70, "seedling": 65, "vegetative": 55,
        "flowering": 100, "fruiting": 90, "reproductive": 95,
        "maturity": 40
    }
    stage_component = stage_weights.get(stage, 50)
    if risk == "Waterlogging Risk":
        return {
            "priority_score": None,
            "priority_explanation": "Drainage review required. Do not automatically add irrigation.",
            "action_type": "Inspect drainage",
            "criteria": "Waterlogging is handled separately from irrigation urgency."
        }
    score = round(0.60 * severity + 0.25 * moisture_component + 0.15 * stage_component, 1)
    if score >= 75:
        action = "Urgent field review"
    elif score >= 50:
        action = "Prioritize for irrigation assessment"
    elif score >= 30:
        action = "Monitor and plan"
    else:
        action = "Routine monitoring"
    return {
        "priority_score": score,
        "priority_explanation": f"{action}. Score combines predicted risk (60%), soil-moisture depletion (25%) and growth-stage urgency (15%).",
        "action_type": action,
        "criteria": "Heuristic decision-support score; proposed weights must be validated with agronomic guidance."
    }

def explain_factors(payload: Dict[str, Any], importance_count: int = 5):
    # Global RF feature importance is not a local causal explanation. Return top model features
    # with their submitted values and label them as global-importance context.
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    pairs = sorted(zip(model_columns, importances), key=lambda x: float(x[1]), reverse=True)
    output = []
    for encoded_name, imp in pairs:
        if "_" in encoded_name:
            raw_name = next((c for c in raw_feature_columns if encoded_name.startswith(c + "_")), encoded_name)
            value = payload.get(raw_name, feature_defaults.get(raw_name, ""))
        else:
            raw_name = encoded_name
            value = payload.get(raw_name, feature_defaults.get(raw_name, ""))
        # Avoid duplicate raw feature names.
        if any(x["feature"] == raw_name for x in output):
            continue
        output.append({
            "feature": raw_name,
            "value": value,
            "importance": round(float(imp), 5),
            "impact": "global model importance; not a causal effect"
        })
        if len(output) >= importance_count:
            break
    return output

@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.get("/")
def root():
    return {"app": "CropWise API", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "model_features": len(model_columns)}

@app.get("/api/model-info")
def model_info():
    return {
        "model": type(model).__name__,
        "target": TARGET,
        "classes": [str(x) for x in getattr(model, "classes_", class_order)],
        "input_feature_count": len(model_columns),
        "preprocessing": "Notebook-compatible median/mode imputation + pandas get_dummies + saved-column alignment"
    }

@app.post("/api/predict")
def predict(payload: Dict[str, Any]):
    try:
        X = preprocess_one(payload)
        prediction = str(model.predict(X)[0])
        probabilities = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            probabilities = {str(c): round(float(p), 5) for c, p in zip(model.classes_, probs)}
            confidence = float(np.max(probs))
        else:
            confidence = None
        priority = priority_details(prediction, payload)
        return {
            "risk_category": prediction,
            "confidence": confidence,
            "class_probabilities": probabilities,
            "priority_score": priority["priority_score"],
            "priority_explanation": priority["priority_explanation"],
            "action_type": priority["action_type"],
            "priority_criteria": priority["criteria"],
            "contributing_factors": explain_factors(payload)
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

@app.get("/api/fields")
def fields():
    # Create a current/latest-observation-per-plot demo ranking from the supplied dataset.
    # These are model predictions on historical dataset rows, not live sensor readings.
    try:
        df = training_df.copy()
        if "Observation_Datetime" in df.columns:
            df["Observation_Datetime"] = pd.to_datetime(df["Observation_Datetime"], errors="coerce")
            df = df.sort_values("Observation_Datetime")
        latest = df.drop_duplicates(subset=["Farm_ID", "Plot_ID"], keep="last")
        records = []
        for _, row in latest.iterrows():
            raw_payload = row.to_dict()
            X = preprocess_one(raw_payload)
            risk = str(model.predict(X)[0])
            priority = priority_details(risk, raw_payload)
            factors = explain_factors(raw_payload, 1)
            records.append({
                "farm_id": str(row.get("Farm_ID", "")),
                "plot_id": str(row.get("Plot_ID", "")),
                "crop_type": str(row.get("Crop_Type", "")),
                "growth_stage": str(row.get("Crop_Growth_Stage", "")),
                "risk_category": risk,
                "priority_score": priority["priority_score"],
                "priority_explanation": priority["priority_explanation"],
                "main_contributor": factors[0]["feature"] if factors else "Not available",
                "observation_date": str(row.get("Observation_Date", "")),
                "data_source": "historical dataset observation; model-generated prediction"
            })
        # Waterlogging gets a separate review queue; other fields sorted by priority score.
        records.sort(key=lambda x: (x["risk_category"] == "Waterlogging Risk",
                                    -(x["priority_score"] if x["priority_score"] is not None else -1)))
        for i, item in enumerate(records, 1):
            item["rank"] = i
        return {"count": len(records), "data_note": "Historical dataset-based predictions; not live field telemetry.", "fields": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not create field ranking: {exc}")

@app.get("/api/feature-importance")
def feature_importance(limit: int = 15):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {"features": [], "note": "This estimator does not expose feature_importances_."}
    pairs = sorted(zip(model_columns, importances), key=lambda x: float(x[1]), reverse=True)[:max(1, min(limit, 50))]
    return {"features": [{"feature": str(name), "importance": float(value)}
                         for name, value in pairs],
            "note": "Global Random Forest impurity-based importance; not a causal explanation."}

@app.get("/api/summary")
def summary():
    df = training_df
    counts = df[TARGET].value_counts().to_dict() if TARGET in df.columns else {}
    return {
        "observations": int(len(df)),
        "farms": int(df["Farm_ID"].nunique()) if "Farm_ID" in df else None,
        "plots": int(df["Plot_ID"].nunique()) if "Plot_ID" in df else None,
        "crop_types": int(df["Crop_Type"].nunique()) if "Crop_Type" in df else None,
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
        "data_note": "Counts describe the supplied dataset, not live field conditions."
    }

@app.get("/api/forecast")
def forecast(farm_id: str, plot_id: str):
    # These columns are excluded from the current RF model, but are already present in the CSV.
    # This endpoint displays dataset-provided forecast columns, not a forecast trained by this RF.
    df = training_df
    if not {"Farm_ID", "Plot_ID"}.issubset(df.columns):
        raise HTTPException(400, "Farm_ID and Plot_ID are unavailable in dataset.")
    subset = df[(df["Farm_ID"].astype(str) == farm_id) & (df["Plot_ID"].astype(str) == plot_id)]
    if subset.empty:
        raise HTTPException(404, "Field not found.")
    row = subset.iloc[-1]
    keys = ["Forecasted_Soil_Moisture_24h_Pct", "Forecasted_Soil_Moisture_48h_Pct",
            "Forecasted_Soil_Moisture_72h_Pct", "Forecasted_Moisture_Regime_Category"]
    return {"farm_id": farm_id, "plot_id": plot_id,
            "values": {k: (None if pd.isna(row.get(k)) else row.get(k)) for k in keys},
            "note": "Values are taken from the supplied dataset. The current Random Forest classifier does not predict these forecast values."}
