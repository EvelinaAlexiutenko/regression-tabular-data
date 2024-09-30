import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv(r"D:\Projects\DS Engineer_Test Task\train.csv")

X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)
rfr_rmse = np.sqrt(mean_squared_error(y_test, rfr_pred))
print(f'Random Forest RMSE: {rfr_rmse:.4f}')

# Train XGBoost Regressor
xgbr = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgbr.fit(X_train, y_train)
xgbr_pred = xgbr.predict(X_test)
xgbr_rmse = np.sqrt(mean_squared_error(y_test, xgbr_pred))
print(f'XGBoost RMSE: {xgbr_rmse:.4f}')


# Save the models and scaler
joblib.dump(rfr, 'random_forest_model.pkl')
joblib.dump(xgbr, 'xgboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
