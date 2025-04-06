---

# 🌫️ PM2.5 Air Quality Prediction

This project predicts PM2.5 (fine particulate matter) concentrations using air quality data and machine learning. A Random Forest Regressor is trained on historical air pollution metrics to forecast future PM2.5 levels based on related pollutant readings.

---

## 📁 Dataset

The dataset used is `city_day.csv`, which contains daily air quality statistics for various Indian cities. It includes pollutant levels like NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, and the target variable `PM2.5`.

---

## 🛠️ Installation

To get started, install the required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
```

---

## 🚀 Steps

### 1. **Data Loading & Exploration**
- Read the CSV dataset using `pandas`.
- Display the first few rows and check for missing values.

### 2. **Data Cleaning**
- Convert numeric columns to proper formats.
- Handle missing values using mean imputation.
- Drop unnecessary columns like `City`, `AQI`, and `AQI_Bucket`.

### 3. **Feature Engineering**
- Extract `Year`, `Month`, and `Day` from the `Date` column.
- Drop the original `Date` column.
- Define features `X` and target variable `y`.

### 4. **Model Training**
- Split the dataset into training and testing sets.
- Train a `RandomForestRegressor` model.

### 5. **Model Evaluation**
- Evaluate the model using Mean Squared Error (MSE) and R² Score.
- Visualize Actual vs. Predicted PM2.5 values.

### 6. **Feature Importance**
- Plot the top 10 most important features affecting PM2.5 levels.

### 7. **Model Saving**
- Save the trained model as a `.pkl` file using `joblib`.

### 8. **Prediction on New Data**
- Predict PM2.5 values for a new sample with pollutant data for a specific date.

---

## 📊 Results

- **Model Metrics**: Displays the MSE and R² score to evaluate performance.
- **Visualization**: 
  - Actual vs Predicted PM2.5 levels.
  - Feature importance ranking (Top 10).

---

## 📦 Output Files

- `air_quality_model.pkl`: Saved trained model for later use.

---

## 🧪 Sample Prediction

```python
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

predicted_pm25 = model.predict(new_sample)
print("Predicted PM2.5:", predicted_pm25[0])
```

---

## 📌 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- joblib

---
