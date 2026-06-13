# forecaster.py
# Predicts your next 30 days of spending using Facebook Prophet
# Prophet is great for time-series with weekly/monthly patterns

from prophet import Prophet
import pandas as pd

def forecast_spending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes transaction history, returns 30-day forecast.
    Returns a dataframe with columns: ds (date), yhat (predicted spend),
    yhat_lower, yhat_upper (confidence interval).
    """
    # Prophet needs columns named 'ds' and 'y'
    daily = (
        df[df["Amount"] > 0]           # only expenses
        .groupby("Date")["Amount"]
        .sum()
        .reset_index()
        .rename(columns={"Date": "ds", "Amount": "y"})
    )

    daily["ds"] = pd.to_datetime(daily["ds"])

    if len(daily) < 10:
        return pd.DataFrame()  # not enough data

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.80
    )
    model.fit(daily)

    # Predict next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Only return future predictions
    last_date = daily["ds"].max()
    future_forecast = forecast[forecast["ds"] > last_date][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ]

    # Make sure predicted values aren't negative
    future_forecast["yhat"] = future_forecast["yhat"].clip(lower=0)
    future_forecast["yhat_lower"] = future_forecast["yhat_lower"].clip(lower=0)

    return future_forecast