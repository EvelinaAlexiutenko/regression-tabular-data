import pandas as pd
import joblib
import numpy as np

# Load the trained models and scaler
rfr = joblib.load('random_forest_model.pkl')
xgbr = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

test_file_path = r'D:\Projects\DS Engineer_Test Task\hidden_test.csv'
test_data = pd.read_csv(test_file_path)

# Scale the features for SVR
X_test_scaled = scaler.transform(test_data)

# Make predictions
y_pred_rfr = rfr.predict(test_data)
y_pred_xgbr = xgbr.predict(test_data)

# Store predictions in the test data
test_data['y_pred_rfr'] = y_pred_rfr
test_data['y_pred_xgbr'] = y_pred_xgbr

# Save predictions to a CSV file
test_data.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")