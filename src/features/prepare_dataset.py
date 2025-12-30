import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(current_dir))

INPUT_FILE = os.path.join(BASE_DIR, "data", "advertising.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "processed_adv_data.csv")
# STATS_FILE yolunu data/ klasörünün içine alacak şekilde güncelledim
STATS_FILE = os.path.join(BASE_DIR, "data", "feature_baseline_stats.csv")

HASH_FEATURES = {
    "Ad Topic Line": 128,
    "City": 32,
    "Country": 16,
    "Ad_Country_Cross": 64,
}

NUMERIC_COLS = ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Male"]
LABEL_COL = "Clicked on Ad"

def hash_column(series, n_features, prefix):
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    tokens = series.astype(str).apply(lambda x: [x])
    hashed = hasher.transform(tokens)
    return pd.DataFrame(hashed.toarray(), columns=[f"{prefix}_hash_{i}" for i in range(n_features)])

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns: df[col] = df[col].fillna(df[col].mean())
    text_cols = ["Ad Topic Line", "City", "Country"]
    for col in text_cols:
        if col in df.columns: df[col] = df[col].fillna("Unknown")
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["hour"] = df["Timestamp"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["Timestamp"].dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df.drop(columns=["Timestamp"], inplace=True)
    if "Ad Topic Line" in df.columns and "Country" in df.columns:
        df["Ad_Country_Cross"] = df["Ad Topic Line"].astype(str) + "_" + df["Country"].astype(str)
    hashed_dfs = [hash_column(df[col], n, col.replace(" ", "_")) for col, n in HASH_FEATURES.items() if col in df.columns]
    num_df_scaled = pd.DataFrame(StandardScaler().fit_transform(df[[c for c in NUMERIC_COLS if c in df.columns]]), columns=[c for c in NUMERIC_COLS if c in df.columns])
    time_cols = ["hour", "day_of_week", "is_weekend"]
    dfs_to_concat = [num_df_scaled] + hashed_dfs + [df[[c for c in time_cols if c in df.columns]]]
    if LABEL_COL in df.columns: dfs_to_concat.append(df[[LABEL_COL]])
    return pd.concat(dfs_to_concat, axis=1)

def save_statistics(df, output_path):
    feature_stats = {col: {"mean": float(df[col].mean()), "std": float(df[col].std()), "min": float(df[col].min()), "max": float(df[col].max())} for col in df.columns if col != LABEL_COL}
    pd.DataFrame(feature_stats).T.to_csv(output_path)
    print(f"Stats saved: {output_path}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found")
    else:
        raw_df = pd.read_csv(INPUT_FILE)
        processed_df = build_features(raw_df)
        save_statistics(processed_df, STATS_FILE)
        processed_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Success! Files: {OUTPUT_FILE}, {STATS_FILE}")