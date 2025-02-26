import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load dataset
file_path = 'D:\\Sales Prediction1\\data\\sales-data-sample.csv'
data = pd.read_csv(file_path)

# Drop columns that contain only NaN values
data.dropna(axis=1, how='all', inplace=True)

# Separate features (X) and target variable (y)
target_column = 'SalesForecast'  # Adjust this based on your dataset
X = data.drop(columns=[target_column], errors='ignore')  # Drop target column from features
y = data[target_column] if target_column in data else None

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

# Handle missing values
categorical_imputer = SimpleImputer(strategy='most_frequent')
numerical_imputer = SimpleImputer(strategy='mean')

X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])

# Verify no missing values remain
assert not X.isna().sum().any(), "There are still missing values in the dataset after imputation."

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Standardize numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'sales_prediction_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model training complete. Saved as 'sales_prediction_model.pkl'.")
