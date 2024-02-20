# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from pyod.models.abod import ABOD

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic time series data
n_samples = 1000
time_points = np.linspace(0, 10, n_samples)
sin_wave = np.sin(time_points) + np.random.normal(0, 0.1, n_samples)

# Introduce anomalies (outliers)
anomaly_indices = [200, 400, 600]
sin_wave[anomaly_indices] += 2.5  # Add outliers to the sine wave

# Create a DataFrame
df = pd.DataFrame({"Time": time_points, "Value": sin_wave})

# Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(df["Time"], df["Value"], label="Time Series Data")
plt.scatter(df.loc[anomaly_indices, "Time"], df.loc[anomaly_indices, "Value"], color="red", label="Anomalies")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Synthetic Time Series Data with Anomalies")
plt.legend()
plt.show()

# Save the data to a CSV file
df.to_csv("synthetic_time_series_data.csv", index=False)

# Detect anomalies using ABOD (Angle-Based Outlier Detection)
X = df[["Time", "Value"]].values
clf = ABOD()
clf.fit(X)

# Predict anomaly scores
anomaly_scores = clf.decision_function(X)

# Add anomaly scores to the DataFrame
df["Anomaly Score"] = anomaly_scores

# Identify anomalies based on a threshold (you can adjust this threshold)
threshold = 0.5
df["Is Anomaly"] = df["Anomaly Score"] > threshold

# Print the DataFrame with anomaly information
print(df.head())

# Save the DataFrame with anomaly information to a CSV file
df.to_csv("synthetic_time_series_data_with_anomalies.csv", index=False)

