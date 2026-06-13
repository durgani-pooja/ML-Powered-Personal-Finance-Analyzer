# anomaly.py
# Detects unusually high spending using Isolation Forest
# Think of it as: "this transaction looks weird compared to your normal pattern"

from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'Is_Anomaly' column — True means unusual spending.
    Only runs on expense rows (positive amounts).
    """
    expenses = df[df["Amount"] > 0].copy()

    if len(expenses) < 5:
        df["Is_Anomaly"] = False
        return df

    # Train Isolation Forest on the Amount column
    model = IsolationForest(
        contamination=0.1,  # expect ~10% anomalies
        random_state=42
    )
    amounts = expenses[["Amount"]]
    predictions = model.fit_predict(amounts)

    # -1 means anomaly, 1 means normal
    expenses["Is_Anomaly"] = predictions == -1

    # Merge back into original dataframe
    df = df.merge(
        expenses[["Is_Anomaly"]],
        left_index=True,
        right_index=True,
        how="left"
    )
    df["Is_Anomaly"] = df["Is_Anomaly"].fillna(False)

    return df