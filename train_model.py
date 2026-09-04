import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Create visualizations directory
import os
os.makedirs('visualizations', exist_ok=True)

# Load dataset
df = pd.read_csv('data/cars.csv')

# Display basic information
print("=" * 60)
print("CAR PRICE PREDICTION - DATA EXPLORATION")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nStatistical Summary:")
print(df.describe())
print(f"\nMissing Values:")
print(df.isnull().sum())

# Data Visualization
plt.figure(figsize=(15, 10))

# Price distribution
plt.subplot(2, 3, 1)
plt.hist(df['price'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Car Prices')

# Price vs Year
plt.subplot(2, 3, 2)
plt.scatter(df['year'], df['price'], alpha=0.6, color='green')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.title('Car Price vs Year')

# Price vs Mileage
plt.subplot(2, 3, 3)
plt.scatter(df['mileage'], df['price'], alpha=0.6, color='red')
plt.xlabel('Mileage (miles)')
plt.ylabel('Price ($)')
plt.title('Car Price vs Mileage')

# Price vs Engine Size
plt.subplot(2, 3, 4)
plt.scatter(df['engine_size'], df['price'], alpha=0.6, color='orange')
plt.xlabel('Engine Size (L)')
plt.ylabel('Price ($)')
plt.title('Car Price vs Engine Size')

# Price by Type
plt.subplot(2, 3, 5)
df.boxplot(column='price', by='type', ax=plt.gca())
plt.xlabel('Car Type')
plt.ylabel('Price ($)')
plt.title('Price Distribution by Car Type')
plt.suptitle('')

# Price by Condition
plt.subplot(2, 3, 6)
df.boxplot(column='price', by='condition', ax=plt.gca())
plt.xlabel('Car Condition')
plt.ylabel('Price ($)')
plt.title('Price Distribution by Condition')
plt.suptitle('')

plt.tight_layout()
plt.savefig('visualizations/exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Exploratory analysis plot saved!")
plt.close()

# Data Preprocessing
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Encode categorical variables
le_type = LabelEncoder()
le_fuel = LabelEncoder()
le_transmission = LabelEncoder()
le_condition = LabelEncoder()

df['type_encoded'] = le_type.fit_transform(df['type'])
df['fuel_type_encoded'] = le_fuel.fit_transform(df['fuel_type'])
df['transmission_encoded'] = le_transmission.fit_transform(df['transmission'])
df['condition_encoded'] = le_condition.fit_transform(df['condition'])

print("\nEncoded Categorical Variables:")
print(f"Type: {dict(zip(le_type.classes_, le_type.transform(le_type.classes_)))}")
print(f"Fuel Type: {dict(zip(le_fuel.classes_, le_fuel.transform(le_fuel.classes_)))}")
print(f"Transmission: {dict(zip(le_transmission.classes_, le_transmission.transform(le_transmission.classes_)))}")
print(f"Condition: {dict(zip(le_condition.classes_, le_condition.transform(le_condition.classes_)))}")

# Select features for modeling
features = ['year', 'mileage', 'engine_size', 'type_encoded', 'fuel_type_encoded', 
            'transmission_encoded', 'condition_encoded']
X = df[features]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION")
print("=" * 60)

# Linear Regression
print("\n1. LINEAR REGRESSION")
print("-" * 60)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

print(f"R² Score: {lr_r2:.4f}")
print(f"RMSE: ${lr_rmse:,.2f}")
print(f"MAE: ${lr_mae:,.2f}")

# Random Forest Regressor
print("\n2. RANDOM FOREST REGRESSOR")
print("-" * 60)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"R² Score: {rf_r2:.4f}")
print(f"RMSE: ${rf_rmse:,.2f}")
print(f"MAE: ${rf_mae:,.2f}")

# Feature Importance (Random Forest)
print("\nFeature Importance (Random Forest):")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# Model Comparison Visualization
plt.figure(figsize=(15, 5))

# Comparison metrics
models = ['Linear Regression', 'Random Forest']
r2_scores = [lr_r2, rf_r2]
rmse_scores = [lr_rmse, rf_rmse]

plt.subplot(1, 3, 1)
plt.bar(models, r2_scores, color=['skyblue', 'green'])
plt.ylabel('R² Score')
plt.title('Model Comparison - R² Score')
plt.ylim(0, 1)
for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

plt.subplot(1, 3, 2)
plt.bar(models, rmse_scores, color=['skyblue', 'green'])
plt.ylabel('RMSE ($)')
plt.title('Model Comparison - RMSE')
for i, v in enumerate(rmse_scores):
    plt.text(i, v + 500, f'${v:,.0f}', ha='center')

# Actual vs Predicted (Best Model - Random Forest)
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Random Forest - Actual vs Predicted')

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison plot saved!")
plt.close()

# Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest Model')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved!")
plt.close()

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n✓ Best Model: Random Forest")
print(f"✓ R² Score: {rf_r2:.4f}")
print(f"✓ RMSE: ${rf_rmse:,.2f}")
print(f"✓ MAE: ${rf_mae:,.2f}")
print(f"\nFiles saved:")
print(f"  - visualizations/exploratory_analysis.png")
print(f"  - visualizations/model_comparison.png")
print(f"  - visualizations/feature_importance.png")
print("\n" + "=" * 60)
