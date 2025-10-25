import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

st.set_page_config(page_title="🏠 California House Price Predictor", page_icon="🏡")

st.title("🏠 California House Price Predictor")
st.markdown(
    "This demo app loads the **latest trained model** from a public URL and predicts "
    "median house prices using the California Housing dataset features."
)

# --- Load model from public GitHub URL ---
MODEL_URL = "https://raw.githubusercontent.com/souvikg23/mlops-demo-california/main/models/linear_model.pkl"


try:
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model = joblib.load(BytesIO(response.content))
    st.success("✅ Loaded latest model successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
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
