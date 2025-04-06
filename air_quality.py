# -*- coding: utf-8 -*-

# Step 1: Install Required Libraries
!pip install pandas numpy scikit-learn matplotlib seaborn xgboost

# Step 2: Load and Explore the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("city_day.csv")  # Replace with the correct path to your dataset

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Step 3: Clean the Dataset
# Convert all numeric columns to numeric, coercing errors to NaN
numeric_columns = ["PM2.5", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Drop rows where PM2.5 is NaN (target variable cannot be NaN)
data = data.dropna(subset=["PM2.5"])

# Fill missing values in other numeric columns with the mean
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Drop non-numeric columns that are not useful for prediction
data = data.drop(columns=["City", "AQI", "AQI_Bucket"])

# Display basic statistics
print("\nDataset statistics:")
print(data.describe())

# Step 4: Preprocess the Data
# Convert 'Date' column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Extract year, month, and day from the 'Date' column
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day

# Drop the original 'Date' column
data = data.drop(columns=["Date"])

# Select features and target variable
X = data.drop(columns=["PM2.5"])
y = data["PM2.5"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 5: Train a Regression Model
# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Step 6: Visualize the Results
# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs. Predicted PM2.5 Levels")
plt.show()

# Step 7: Feature Importance
# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(10))
plt.title("Top 10 Important Features for PM2.5 Prediction")
plt.show()

# Step 8: Save the Model
import joblib

# Save the model
joblib.dump(model, "air_quality_model.pkl")
print("\nModel saved as 'air_quality_model.pkl'")

# Step 9: Make Predictions on New Data
# Example: Predict PM2.5 for a new sample
new_sample = pd.DataFrame({
    "Year": [2023],
    "Month": [10],
    "Day": [15],
    "NO2": [20],
    "NOx": [30],
    "NH3": [10],
    "CO": [1.5],
    "SO2": [15],
    "O3": [40],
    "Benzene": [0.5],
    "Toluene": [0.3],
    "Xylene": [0.2]
})

# Ensure the new sample has the same columns as the training data
new_sample = new_sample.reindex(columns=X.columns, fill_value=0)

# Predict PM2.5
predicted_pm25 = model.predict(new_sample)
print("\nPredicted PM2.5 for the new sample:", predicted_pm25[0])
