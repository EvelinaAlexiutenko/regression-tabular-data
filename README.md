
## Table of Contents
- [Installation](#installation)
- [Model Training](#model-training)
- [Inference](#inference)
- [Performance Metrics](#performance-metrics)


## Installation
To set up the required environment, install the necessary packages listed in `requirements.txt` using pip:

```
pip install -r requirements.txt
```
# Model Training
This project involves training two regression models: Random Forest Regressor and XGBoost Regressor.
To train the models, run the following command on the train.csv file:

```
python train.py
```

# Inference
To perform inference on a hidden test dataset:
1. Load the test dataset from a specified path.
2. Use the trained Random Forest model to predict the target variable.

```
python predict.py
```

## Performance Metrics
- **Random Forest RMSE**: 0.0038
- **XGBoost RMSE**: 0.4158
