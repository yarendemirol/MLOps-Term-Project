import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.features.prepare_dataset import build_features
# --- SADECE BU IMPORTU EKLEDİK ---
from src.monitoring.drift_detector import validate_statistical_skew

# Dosya yollarını ayarla
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(current_dir))
MODEL_PATH = os.path.join(BASE_DIR, "final_deployment_model.pkl")

app = FastAPI(title="Ad Click Prediction API")

# Gelen veri şablonu
class AdClickData(BaseModel):
    Daily_Time_Spent: float
    Age: int
    Area_Income: float
    Daily_Internet_Usage: float
    Male: int
    Hour: int
    Day_of_Week: int
    Is_Weekend: int

# Modeli yükle
if not os.path.exists(MODEL_PATH):
    model = None
else:
    model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(data: AdClickData):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        # 1. Ham veriyi oluştur
        raw_data = {
            "Daily Time Spent on Site": [data.Daily_Time_Spent],
            "Age": [data.Age],
            "Area Income": [data.Area_Income],
            "Daily Internet Usage": [data.Daily_Internet_Usage],
            "Male": [data.Male],
            "hour": [data.Hour],
            "day_of_week": [data.Day_of_Week],
            "is_weekend": [data.Is_Weekend],
            "Ad Topic Line": ["unknown"],
            "City": ["unknown"],
            "Country": ["unknown"],
            "Timestamp": ["2016-01-01 00:00:00"] 
        }

        df = pd.DataFrame(raw_data)
        
        # 2. Özellikleri oluştur
        processed_df = build_features(df)

        # --- BURAYA SADECE 2 SATIR EKLEDİK (MONITORING) ---
        stats_path = os.path.join(BASE_DIR, "data", "feature_baseline_stats.csv")
        validate_statistical_skew(processed_df, stats_path)
        # --------------------------------------------------

        # 3. FIX: Scaler hatasını düzeltmek için değerleri manuel geri yükle
        processed_df["Daily Time Spent on Site"] = data.Daily_Time_Spent
        processed_df["Age"] = data.Age
        processed_df["Area Income"] = data.Area_Income
        processed_df["Daily Internet Usage"] = data.Daily_Internet_Usage
        processed_df["Male"] = data.Male
        processed_df["hour"] = data.Hour
        processed_df["day_of_week"] = data.Day_of_Week
        processed_df["is_weekend"] = data.Is_Weekend

        # Gereksiz kolonları temizle
        if "Clicked on Ad" in processed_df.columns:
            processed_df = processed_df.drop(columns=["Clicked on Ad"])

        # --- YENİ EKLENEN KISIM (OLASILIK HESABI) ---
        
        # 1. Sınıf Tahmini (0 veya 1)
        prediction = model.predict(processed_df)
        
        # 2. Olasılık Tahmini
        probability = model.predict_proba(processed_df)[0][1]

        return {
            "prediction": int(prediction[0]),
            "click_probability": float(probability),
            "status": "Success"
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Docker ve K8s dışarıdan erişebilsin diye 0.0.0.0 yaptık
    uvicorn.run(app, host="0.0.0.0", port=8000)
