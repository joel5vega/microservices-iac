# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder

# Load the synthetic time series data
data_path = "synthetic_time_series_data.csv"
df = pd.read_csv(data_path)

# Extract features (excluding time)
X = df["Value"].values.reshape(-1, 1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize anomaly detection models
iforest_model = IForest(contamination=0.05, random_state=42)
ocsvm_model = OCSVM(contamination=0.05)
autoencoder_model = AutoEncoder(hidden_neurons=[8, 4, 4, 8], epochs=50, contamination=0.05)

# Fit the models
iforest_model.fit(X_scaled)
ocsvm_model.fit(X_scaled)
autoencoder_model.fit(X_scaled)

# Predict anomalies
df["IForest Anomaly"] = iforest_model.predict(X_scaled)
df["OCSVM Anomaly"] = ocsvm_model.predict(X_scaled)
df["AutoEncoder Anomaly"] = autoencoder_model.predict(X_scaled)

# Visualize anomalies
plt.figure(figsize=(10, 6))
plt.plot(df["Time"], df["Value"], label="Time Series Data")
plt.scatter(df.loc[df["IForest Anomaly"] == 1, "Time"], df.loc[df["IForest Anomaly"] == 1, "Value"], color="red", label="Isolation Forest Anomalies")
plt.scatter(df.loc[df["OCSVM Anomaly"] == 1, "Time"], df.loc[df["OCSVM Anomaly"] == 1, "Value"], color="orange", label="One-Class SVM Anomalies")
plt.scatter(df.loc[df["AutoEncoder Anomaly"] == 1, "Time"], df.loc[df["AutoEncoder Anomaly"] == 1, "Value"], color="purple", label="AutoEncoder Anomalies")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Anomaly Detection Results")
plt.legend()
plt.show()

# Save the updated DataFrame
df.to_csv("synthetic_time_series_data_with_anomalies.csv", index=False)
