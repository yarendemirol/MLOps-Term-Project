import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from prefect import flow, task
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.features.prepare_dataset import build_features
from src.training.train_model import train_full_pipeline
from src.monitoring.quality_check import run_quality_check


@task(name="1_Data_Ingestion")
def ingestion_step():
    input_path = os.path.join(BASE_DIR, "data", "advertising.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found at: {input_path}")
    return pd.read_csv(input_path)


@task(name="2_Advanced_Feature_Engineering")
def preparation_step(raw_df):
    processed_df = build_features(raw_df)
    print("Feature Engineering completed with Hashing and Scaling.")
    return processed_df


@task(name="3_Model_Training_and_Comparison")
def training_step(df):
    X = df.drop('Clicked on Ad', axis=1)
    y = df['Clicked on Ad']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    rf, xgb, ensemble = train_full_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    models_dict = {"Bagging_RF": rf, "Boosting_XGB": xgb, "Ensemble_Voting": ensemble}
    results = []

    for name, model in models_dict.items():
        p = model.predict(X_test)
        pr = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, p)
        f1 = f1_score(y_test, p)
        auc = roc_auc_score(y_test, pr)
        pre = precision_score(y_test, p)
        rec = recall_score(y_test, p)

        mlflow.log_metric(f"{name}_accuracy", acc)
        mlflow.log_metric(f"{name}_f1", f1)
        mlflow.log_metric(f"{name}_auc", auc)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 2),
            "Precision": round(pre, 4),
            "Recall": round(rec, 2),
            "F1_Score": round(f1, 4),
            "AUC_ROC": round(auc, 4)
        })

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Model'], results_df['Accuracy'], color=['blue', 'green', 'red'])
    plt.title('Model Accuracy Comparison')
    plt.ylim(0.8, 1.0)
    plot_path = os.path.join(BASE_DIR, "accuracy_comparison.png")
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

    cm = confusion_matrix(y_test, ensemble.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    cm_path = os.path.join(BASE_DIR, "ensemble_confusion_matrix.png")
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    return ensemble, X_test, y_test


@task(name="4_Model_Registry_and_Quality_Check")
def registry_step(ensemble, X_test, y_test):
    stats_path = os.path.join(BASE_DIR, "data", "feature_baseline_stats.csv")
    if os.path.exists(stats_path):
        mlflow.log_artifact(stats_path)

    mlflow.sklearn.log_model(
        ensemble,
        "final_model",
        registered_model_name="AdClickPredictionModel"
    )

    y_pred = ensemble.predict(X_test)
    status = run_quality_check(y_test, y_pred)

    mlflow.log_param("final_status", status)
    mlflow.log_param("feature_engineering", "Hashing_Trick_and_Scaling")
    mlflow.set_tag("owner", "yarendemirol")
    mlflow.set_tag("mlops_level", "2")

    return status


@flow(name="MLOps_Level2_Detailed_Execution")
def main_training_flow():
    mlruns_path = os.path.join(BASE_DIR, "mlruns")
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment("Ad_Click_Production_Project")

    with mlflow.start_run(run_name="Full_Detailed_EndToEnd_Execution"):
        raw_df = ingestion_step()
        processed_df = preparation_step(raw_df)
        ensemble_model, X_test, y_test = training_step(processed_df)
        final_status = registry_step(ensemble_model, X_test, y_test)

        print(f"Pipeline finished successfully. Final status: {final_status}")


if __name__ == "__main__":
    main_training_flow()