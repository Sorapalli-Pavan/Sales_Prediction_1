import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('D:\Sales Prediction1\data\sales-data-sample.csv')

# Load trained model
with open('sales_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Encode categorical features
categorical_cols = ['Category', 'City', 'Country', 'Customer Name', 'Product Name', 'Region', 'Segment', 'Ship Mode']
for col in categorical_cols:
    if col in label_encoders:
        data[col] = label_encoders[col].transform(data[col].astype(str))

# Selecting relevant features
features = ['Category', 'City', 'Country', 'Discount', 'Quantity', 'Product Name', 'Unit Price', 'Profit', 'Region', 'Segment', 'Ship Mode', 'Latitude', 'Longitude']
X = data[features]
y_actual = data['Sales']

# Predict sales
y_pred = model.predict(X)

# Model evaluation
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_actual, y=y_pred, alpha=0.7)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Residual Plot
residuals = y_actual - y_pred
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Prediction Error")
plt.title("Error Distribution")
plt.show()

# Line plot for actual vs predicted sales
plt.figure(figsize=(12, 6))
plt.plot(y_actual.values, label='Actual Sales', marker='o')
plt.plot(y_pred, label='Predicted Sales', linestyle='dashed', marker='x')
plt.xlabel("Index")
plt.ylabel("Sales")
plt.legend()
plt.title("Comparison of Actual vs Predicted Sales")
plt.show()

# Feature Importance Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x=model.feature_importances_, y=features)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Sales Prediction Model")
plt.show()
