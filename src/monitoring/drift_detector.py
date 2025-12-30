import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def check_model_degradation(y_true, y_pred, threshold=0.90):
    acc = accuracy_score(y_true, y_pred)
    print(f"Current Model Accuracy: {acc}")
    if acc < threshold:
        print("!!! WARNING: Performance Degradation Detected !!!")
        print("Action: Triggering Algorithmic Fallback to older version.")
        return True
    return False
