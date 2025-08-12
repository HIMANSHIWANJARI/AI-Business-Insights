import os
import pandas as pd
import joblib

# Get the directory where forecast.py is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct absolute path to the model file (assumes 'models' folder is one level up from 'app/')
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'prophet_sales_model.joblib')

def forecast_sales(period=7):
    """Load saved Prophet model and forecast future sales."""
    model = joblib.load(MODEL_PATH)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
