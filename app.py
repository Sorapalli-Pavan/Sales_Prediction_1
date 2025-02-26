from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

# Load trained model
with open('sales_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        data = request.form
        
        # Prepare input data
        input_data = {
            'Category': data['Category'],
            'City': data['City'],
            'Country': data['Country'],
            'Discount': float(data['Discount']),
            'Quantity': int(data['Quantity']),
            'Product Name': data['Product Name'],
            'Unit Price': float(data['Unit Price']),
            'Profit': float(data['Profit']),
            'Region': data['Region'],
            'Segment': data['Segment'],
            'Ship Mode': data['Ship Mode'],
            'Latitude': float(data['Latitude']),
            'Longitude': float(data['Longitude'])
        }
        
        # Encode categorical values
        for col in label_encoders:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]
        
        # Convert to dataframe
        input_df = pd.DataFrame([input_data])
        
        # Predict sales
        prediction = model.predict(input_df)[0]
        
        return jsonify({'predicted_sales': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
