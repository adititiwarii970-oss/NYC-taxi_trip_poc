import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from haversine import haversine
import joblib

# Generate synthetic data similar to NYC Taxi Dataset
np.random.seed(42)
n_samples = 10000

# Random lat/lng (approx NYC area)
pickup_latitudes = np.random.uniform(40.70, 40.85, n_samples)
pickup_longitudes = np.random.uniform(-74.02, -73.90, n_samples)
dropoff_latitudes = pickup_latitudes + np.random.uniform(-0.01, 0.01, n_samples)
dropoff_longitudes = pickup_longitudes + np.random.uniform(-0.01, 0.01, n_samples)

# Time and passenger features
hours = np.random.randint(0, 24, n_samples)
day_of_week = np.random.randint(0, 7, n_samples)
passenger_counts = np.random.randint(1, 5, n_samples)

# Haversine distance
distances = [
    haversine((pickup_latitudes[i], pickup_longitudes[i]), 
              (dropoff_latitudes[i], dropoff_longitudes[i])) 
    for i in range(n_samples)
]

# Synthetic trip durations (in seconds)
trip_durations = [max(60, int(d * 200 + np.random.normal(0, 60))) for d in distances]

# Create DataFrame
df = pd.DataFrame({
    'pickup_latitude': pickup_latitudes,
    'pickup_longitude': pickup_longitudes,
    'dropoff_latitude': dropoff_latitudes,
    'dropoff_longitude': dropoff_longitudes,
    'hour': hours,
    'dayofweek': day_of_week,
    'passenger_count': passenger_counts,
    'distance_km': distances,
    'trip_duration': trip_durations
})

# Model features
features = ['hour', 'dayofweek', 'passenger_count', 'distance_km']
target = 'trip_duration'

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} seconds")
print(f"R² Score: {r2:
