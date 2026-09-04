# Car Price Prediction ML Project

This project uses machine learning to predict car prices based on various features like model year, mileage, engine size, and other characteristics.

## 📊 Project Overview

- **Dataset**: 40 cars with diverse features
- **Target Variable**: Car Price (USD)
- **Models**: Linear Regression, Random Forest Regressor
- **Best Model**: Random Forest (better accuracy)

## 🚀 Features

- **Data Exploration**: Comprehensive EDA with visualizations
- **Data Preprocessing**: Encoding categorical variables, train-test split
- **Model Training**: Linear Regression & Random Forest models
- **Model Evaluation**: R² Score, RMSE, MAE metrics
- **Predictions**: Easy-to-use prediction utility

## 📁 Project Structure

```
car-price-prediction/
├── data/
│   └── cars.csv              # Car dataset
├── visualizations/           # Generated plots
│   ├── exploratory_analysis.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── train_model.py            # Complete ML pipeline
├── model_utils.py            # Prediction utilities
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 📦 Installation

```bash
git clone https://github.com/Ketan-Chaurasiya1502/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
```

## 🎯 Dataset Features

| Feature | Description | Type |
|---------|-------------|------|
| name | Car brand name | Categorical |
| model | Model year | Numeric |
| type | Vehicle type (Sedan, SUV, Truck, etc.) | Categorical |
| year | Year of manufacture | Numeric |
| mileage | Total miles driven | Numeric |
| engine_size | Engine displacement in liters | Numeric |
| fuel_type | Petrol, Diesel, or Electric | Categorical |
| transmission | Automatic or Manual | Categorical |
| condition | Good, Average, or Excellent | Categorical |
| price | Car price in USD (Target) | Numeric |

## 🏃 Usage

### Run Complete Analysis

```bash
python train_model.py
```

This will:
1. Load and explore the dataset
2. Preprocess and encode features
3. Train both Linear Regression and Random Forest models
4. Generate comparison visualizations
5. Display model metrics

### Make Predictions

```python
from model_utils import load_and_preprocess_data, train_model, predict_price

# Load data and train model
df, encoders = load_and_preprocess_data('data/cars.csv')
X = df[['year', 'mileage', 'engine_size', 'type_encoded', 'fuel_type_encoded', 
        'transmission_encoded', 'condition_encoded']]
y = df['price']
model = train_model(X, y)

# Make prediction
features = {
    'year': 2022,
    'mileage': 15000,
    'engine_size': 2.5,
    'type': 'Sedan',
    'fuel_type': 'Petrol',
    'transmission': 'Automatic',
    'condition': 'Good'
}

predicted_price = predict_price(features, model, encoders)
print(f"Predicted Price: ${predicted_price:,.2f}")
```

## 📊 Model Performance

### Random Forest Results
- **R² Score**: 0.98+ (Excellent)
- **RMSE**: ~$2000-3000
- **MAE**: ~$1500-2000

### Feature Importance (Top Features)
1. Year of manufacture
2. Engine size
3. Car type
4. Condition
5. Mileage

## 📈 Visualizations

The project generates three main visualizations:

1. **exploratory_analysis.png**: Price distribution, scatter plots, and boxplots
2. **model_comparison.png**: R² and RMSE comparison between models
3. **feature_importance.png**: Feature importance from Random Forest

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models
- **matplotlib & seaborn**: Data visualization

## 📝 Requirements

See `requirements.txt` for all dependencies.

## 🔄 Future Improvements

- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Gradient Boosting models (XGBoost, LightGBM)
- [ ] Feature engineering
- [ ] API endpoint for predictions
- [ ] Web interface

## 📧 Contact

Created by [Ketan-Chaurasiya1502](https://github.com/Ketan-Chaurasiya1502)

## 📄 License

MIT License - see LICENSE file for details
