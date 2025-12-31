import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score


def check_model_degradation(y_true, y_pred, threshold=0.85):
    """
    Döküman III.3: Continued Model Evaluation & Fallback
    Model performansını kontrol eder. Eğer doğruluk threshold'un altına
    düşerse Algorithmic Fallback tetiklenir.
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"Current Model Accuracy: {acc:.2f}")

    if acc < threshold:
        print("!!! WARNING: Performance Degradation Detected !!!")
        print("Action: Triggering Algorithmic Fallback to older version.")
        return True  
    return False  


def validate_statistical_skew(current_df, baseline_stats_path):
    """
    Döküman III.3: Statistical Checks & Feature Validation
    API'ye gelen veriyi (current_df), eğitim verisinin istatistikleri (baseline) ile
    karşılaştırarak veri kaymasını (Data Skew) tespit eder.
    """
    try:
        if not os.path.exists(baseline_stats_path):
            print(f"[ERROR] Baseline stats file not found at: {baseline_stats_path}")
            return "STATS_NOT_FOUND"

        baseline_df = pd.read_csv(baseline_stats_path, index_col=0)

        for col in current_df.columns:
            if col in baseline_df.columns:
                current_val = current_df[col].iloc[0]
                b_min = baseline_df.loc['min', col]
                b_max = baseline_df.loc['max', col]

                if current_val > b_max * 1.5 or current_val < b_min * 0.5:
                    print(f"[STATISTICAL ALERT] Skew detected in {col}: {current_val} is an outlier.")
                    return "SKEW_DETECTED"

        print("[MONITORING] Statistical feature validation: SUCCESS (No skew detected)")
        return "HEALTHY"
    except Exception as e:
        print(f"Statistical monitoring error: {e}")
        return "MONITORING_ERROR"


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("RUNNING DRIFT DETECTOR LOCAL TEST")
    print("=" * 50)

    y_true_test = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    y_pred_test = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]  

    check_model_degradation(y_true_test, y_pred_test, threshold=0.85)

    test_data = pd.DataFrame({
        "Age": [35],
        "Area Income": [60000]
    })

    current_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(current_dir))
    stats_path = os.path.join(BASE_DIR, "data", "feature_baseline_stats.csv")

    print(f"\nChecking stats at: {stats_path}")
    validate_statistical_skew(test_data, stats_path)
    print("=" * 50 + "\n")
