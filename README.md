# CropWise FastAPI Backend

## 1. Put the model artifacts and dataset in this folder

Copy these three files from the machine where the notebook was trained:

- `random_forest_model.joblib`
- `model_columns.joblib`
- Rename the supplied training CSV to `dataset.csv`

The notebook must have saved the model and columns from the same final fitted estimator. In your notebook, cell 16 saves `best_model`; ensure `model_columns.joblib` was created from the same `X.columns` used to train that estimator.

## 2. Install and run (Windows)

Open Command Prompt/VS Code terminal in this folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open API documentation: `http://127.0.0.1:8000/docs`

## 3. Frontend connection

In the frontend `app.js`, set:

```js
const API_BASE = "http://127.0.0.1:8000";
```

If the frontend runs on a different laptop, use the backend laptop's LAN IP, e.g. `http://192.168.1.10:8000`. Both devices must be on the same network and the backend port must be reachable. Do not use `127.0.0.1` from the other laptop: it points to that laptop itself.

## 4. Endpoints

- `GET /api/health` - server/model status
- `GET /api/model-info` - estimator, target and input feature count
- `GET /api/summary` - dataset summary and class distribution
- `POST /api/predict` - model prediction from a JSON observation
- `GET /api/fields` - latest dataset observation per farm/plot, predicted and ranked
- `GET /api/feature-importance` - Random Forest global feature importance
- `GET /api/forecast?farm_id=FARM_001&plot_id=PLOT_001_01` - returns the dataset's forecast columns for a field

## 5. Important: this follows your notebook's pipeline

Your notebook:
1. Uses `Current_Drought_Stress_Category` as target.
2. Drops `Record_ID`, `Plot_ID`, `Farm_ID`, observation date/time columns, `Time_of_Day`, and the 24h/48h/72h forecast moisture columns.
3. Imputes missing numeric features with their median and categorical features with their mode.
4. Applies `pd.get_dummies()` to predictors.
5. Trains RandomForestClassifier.
6. Saves `random_forest_model.joblib` and `model_columns.joblib`.

This backend applies median/mode defaults calculated from the supplied training CSV, applies `pd.get_dummies()` and reindexes to the saved training columns in their exact order. Farm_ID and Plot_ID are metadata only, not model inputs.

The current notebook does NOT train the classifier to predict the 24h/48h/72h forecast columns; it explicitly drops them before model training. `/api/forecast` only reads those supplied dataset values and labels them as dataset-provided.

## 6. Priority score

The API returns a transparent proposed score:
- 60% predicted stress severity
- 25% soil-moisture depletion relative to field capacity and wilting point
- 15% crop growth-stage urgency

This is a heuristic decision-support ranking, not a learned model output and not an irrigation quantity. Validate the weights with your faculty/domain expert. Waterlogging Risk is handled separately and returns a drainage-inspection action rather than an irrigation score.

## 7. Data integrity notes

`/api/fields` generates predictions for the latest observation in each Farm_ID + Plot_ID from the supplied CSV. These are historical-data predictions, not live sensor measurements. Do not describe them as real-time data.

Feature importance is global Random Forest importance. It describes model reliance, not causality. For per-field explanations, consider adding SHAP or another local explanation method after validating it.
