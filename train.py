import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# --- Make cross-platform path for MLflow tracking ---
mlruns_path = os.path.abspath("mlruns")

# Convert Windows-style path to URI-safe format
mlruns_uri = mlruns_path.replace("\\", "/")
mlflow.set_tracking_uri(f"file:///{mlruns_uri}")

mlflow.set_experiment("california_house_price_experiment")

# --- Load dataset ---
data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train and log model ---
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, name="model", input_example=X_test.iloc[:5])

print(f"✅ Model trained successfully. MSE: {mse:.4f}")
print("Run `mlflow ui` to visualize experiments at http://localhost:5000")
