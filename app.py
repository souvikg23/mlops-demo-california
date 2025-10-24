import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="🏠 California House Price Predictor", page_icon="🏡")

st.title("🏠 California House Price Predictor")
st.markdown("This demo app loads the **latest trained model** from MLflow and predicts median house prices using the California Housing dataset features.")

# --- MLflow client setup ---
client = MlflowClient()
experiment = client.get_experiment_by_name("california_house_price_experiment")

if experiment:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs:
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        st.success(f"✅ Loaded latest model (Run ID: {run_id})")
    else:
        st.error("No MLflow runs found! Please run `python train.py` first.")
        st.stop()
else:
    st.error("Experiment not found! Train a model first using `python train.py`.")
    st.stop()

# --- Input fields for prediction ---
st.subheader("Enter House Features:")

MedInc = st.number_input("Median Income (10k USD)", 0.0, 20.0, 5.0)
HouseAge = st.number_input("House Age (years)", 0.0, 100.0, 20.0)
AveRooms = st.number_input("Average Rooms per Household", 0.0, 20.0, 6.0)
AveBedrms = st.number_input("Average Bedrooms per Household", 0.0, 5.0, 1.0)
Population = st.number_input("Population", 0.0, 50000.0, 3000.0)
AveOccup = st.number_input("Average Occupants per Household", 0.0, 10.0, 3.0)
Latitude = st.number_input("Latitude", 30.0, 45.0, 35.0)
Longitude = st.number_input("Longitude", -125.0, -110.0, -120.0)

if st.button("Predict House Price"):
    X = pd.DataFrame(
        [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
        columns=["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    )
    prediction = model.predict(X)[0]
    st.success(f"🏡 Estimated Median House Value: **${prediction * 100000:,.2f}**")
