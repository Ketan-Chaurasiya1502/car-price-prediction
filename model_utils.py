import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def load_and_preprocess_data(csv_path):
    """Load and preprocess car data"""
    df = pd.read_csv(csv_path)
    
    # Encode categorical variables
    le_type = LabelEncoder()
    le_fuel = LabelEncoder()
    le_transmission = LabelEncoder()
    le_condition = LabelEncoder()
    
    df['type_encoded'] = le_type.fit_transform(df['type'])
    df['fuel_type_encoded'] = le_fuel.fit_transform(df['fuel_type'])
    df['transmission_encoded'] = le_transmission.fit_transform(df['transmission'])
    df['condition_encoded'] = le_condition.fit_transform(df['condition'])
    
    return df, {
        'type': le_type,
        'fuel_type': le_fuel,
        'transmission': le_transmission,
        'condition': le_condition
    }

def train_model(X, y):
    """Train Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def predict_price(features_dict, model, encoders):
    """
    Predict car price from features
    
    Parameters:
    features_dict: dict with keys: year, mileage, engine_size, type, fuel_type, transmission, condition
    model: trained RandomForestRegressor
    encoders: dictionary of LabelEncoders
    
    Returns:
    predicted_price: float
    """
    # Encode categorical features
    features_dict['type_encoded'] = encoders['type'].transform([features_dict['type']])[0]
    features_dict['fuel_type_encoded'] = encoders['fuel_type'].transform([features_dict['fuel_type']])[0]
    features_dict['transmission_encoded'] = encoders['transmission'].transform([features_dict['transmission']])[0]
    features_dict['condition_encoded'] = encoders['condition'].transform([features_dict['condition']])[0]
    
    features = np.array([[
        features_dict['year'],
        features_dict['mileage'],
        features_dict['engine_size'],
        features_dict['type_encoded'],
        features_dict['fuel_type_encoded'],
        features_dict['transmission_encoded'],
        features_dict['condition_encoded']
    ]])
    
    return model.predict(features)[0]

if __name__ == "__main__":
    # Example usage
    features = {
        'year': 2022,
        'mileage': 15000,
        'engine_size': 2.5,
        'type': 'Sedan',
        'fuel_type': 'Petrol',
        'transmission': 'Automatic',
        'condition': 'Good'
    }
    
    # Load data
    df, encoders = load_and_preprocess_data('data/cars.csv')
    
    # Prepare features
    feature_columns = ['year', 'mileage', 'engine_size', 'type_encoded', 'fuel_type_encoded', 
                       'transmission_encoded', 'condition_encoded']
    X = df[feature_columns]
    y = df['price']
    
    # Train model
    model = train_model(X, y)
    
    # Make prediction
    predicted_price = predict_price(features, model, encoders)
    print(f"Predicted Price: ${predicted_price:,.2f}")
