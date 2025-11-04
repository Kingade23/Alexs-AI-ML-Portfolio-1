# coding my project
# usd_forecast_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ------------------------------
# DATA FETCH FUNCTION for project
# ------------------------------
def fetch_exchange_data(base="USD", target="NGN", start="2020-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    url = f"https://api.exchangerate.host/timeseries?start_date={start}&end_date={end}&base={base}&symbols={target}"
    r = requests.get(url)
    data = r.json()
    if not data.get("success"):
        st.error("Failed to fetch data.")
        return pd.DataFrame()
    rates = data["rates"]
    df = pd.DataFrame([(date, rates[date][target]) for date in rates], columns=["date", "rate"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df = df.asfreq("D")
    df["rate"] = df["rate"].interpolate()
    return df

# ------------------------------
# FEATURE CREATION
# ------------------------------
def make_features(df):
    df["lag1"] = df["rate"].shift(1)
    df["lag2"] = df["rate"].shift(2)
    df["roll_mean7"] = df["rate"].rolling(7).mean()
    df = df.dropna()
    X = df[["lag1", "lag2", "roll_mean7"]]
    y = df["rate"]
    return X, y

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def train_and_predict(df, horizon=7):
    X, y = make_features(df)
    model = LinearRegression()
    model.fit(X, y)
    last_known = df.iloc[-1]["rate"]
    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    preds = []
    temp_df = df.copy()
    for _ in range(horizon):
        temp_df = temp_df.copy()
        X_new, _ = make_features(temp_df)
        new_pred = model.predict(X_new.iloc[[-1]])[0]
        preds.append(new_pred)
        temp_df.loc[temp_df.index[-1] + timedelta(days=1), "rate"] = new_pred
    pred_df = pd.DataFrame({"date": future_dates, "predicted": preds}).set_index("date")
    return model, pred_df

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="USD Exchange Rate Predictor", layout="wide")
st.title("💵 USD Exchange Rate Forecast App")

base = st.text_input("Base Currency (default USD)", "USD")
target = st.text_input("Target Currency (e.g. NGN, EUR, GBP)", "NGN")
start_date = st.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.date_input("End Date", datetime.today())
horizon = st.slider("Prediction Horizon (days ahead)", 3, 30, 7)

if st.button("Fetch and Predict"):
    with st.spinner("Fetching and training model..."):
        df = fetch_exchange_data(base, target, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df.empty:
            st.stop()
        model, pred_df = train_and_predict(df, horizon=horizon)
        full_df = pd.concat([df, pred_df.rename(columns={"predicted": "rate"})])

        pct_change = (pred_df.iloc[-1]["predicted"] - df.iloc[-1]["rate"]) / df.iloc[-1]["rate"] * 100
        direction = "📈 RISE" if pct_change > 0 else "📉 FALL"
        st.metric(label="Predicted Change", value=f"{pct_change:.2f}% ({direction})")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["rate"], mode="lines", name="Historical"))
        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["predicted"], mode="lines+markers",
                                 name="Predicted", line=dict(dash="dash")))
        fig.update_layout(title=f"{base}/{target} Exchange Rate Forecast",
                          xaxis_title="Date", yaxis_title=f"Exchange Rate ({target})",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download Predictions as CSV",
                           full_df.to_csv().encode("utf-8"),
                           file_name="usd_forecast.csv")

st.caption("This is an educational AI forecast demo using Linear Regression on historical exchange rate data.")
