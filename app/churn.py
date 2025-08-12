import os
import joblib

# Get directory of this script (churn.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the model file
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'xgb_churn_model.joblib')

def predict_churn(df):
    # Load the trained model using absolute path
    model = joblib.load(MODEL_PATH)

    features = ['Quantity', 'Price', 'TotalAmount', 'Age']
    df['churn_score'] = model.predict(df[features])
    return df
