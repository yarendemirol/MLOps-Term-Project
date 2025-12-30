import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.features.prepare_dataset import build_features

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(current_dir))
MODEL_PATH = os.path.join(BASE_DIR, "final_deployment_model.pkl")

app = FastAPI(title="Ad Click Prediction API")


class AdClickData(BaseModel):
    Daily_Time_Spent: float
    Age: int
    Area_Income: float
    Daily_Internet_Usage: float
    Male: int
    Hour: int
    Day_of_Week: int
    Is_Weekend: int


if not os.path.exists(MODEL_PATH):
    model = None
else:
    model = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(data: AdClickData):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        raw_data = {
            "Daily Time Spent on Site": [data.Daily_Time_Spent],
            "Age": [data.Age],
            "Area Income": [data.Area_Income],
            "Daily Internet Usage": [data.Daily_Internet_Usage],
            "Male": [data.Male],
            "hour": [data.Hour],
            "day_of_week": [data.Day_of_Week],
            "is_weekend": [data.Is_Weekend],
            # Hashing fonksiyonunun hata vermemesi için boş bile olsa bu kolonlar lazım:
            "Ad Topic Line": ["unknown"],
            "City": ["unknown"],
            "Country": ["unknown"],
            "Timestamp": ["2016-01-01 00:00:00"]
        }

        df = pd.DataFrame(raw_data)

        processed_df = build_features(df)

        if "Clicked on Ad" in processed_df.columns:
            processed_df = processed_df.drop(columns=["Clicked on Ad"])

        prediction = model.predict(processed_df)

        return {"prediction": int(prediction[0]), "status": "Success"}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)