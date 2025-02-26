import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = 'D:\\Sales Prediction1\\data\\sales-data-sample.csv'
data = pd.read_csv(file_path)

# Print column names for debugging
print("Dataset Columns:", data.columns)

# Ensure column names are consistent
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces

# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Convert to string before encoding
    label_encoders[col] = le

print("✅ Categorical encoding complete.")
