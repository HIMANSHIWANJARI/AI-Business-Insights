import pandas as pd
import numpy as np
import os
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# ==== Load and clean sales data ====
df = pd.read_csv("data/sample_sales.csv")
df.columns = df.columns.str.strip().str.lower()  # Standardize column names

# Validate essential columns
required_cols = {'date', 'order_id', 'product_name', 'price', 'cost_price', 'quantity'}
if not required_cols.issubset(df.columns):
    raise Exception(f"Missing columns in data. Required: {required_cols}")

# Convert date column
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'price', 'cost_price', 'quantity'], inplace=True)

# ==== 1. Prophet Model: Forecasting ====
df['revenue'] = df['price'] * df['quantity']
prophet_df = df.groupby('date')['revenue'].sum().reset_index()
prophet_df = prophet_df.rename(columns={'date': 'ds', 'revenue': 'y'})

model_prophet = Prophet()
model_prophet.fit(prophet_df)
joblib.dump(model_prophet, "models/prophet_sales_model.joblib")
print("✅ Saved Prophet model.")

# ==== 2. KMeans: Customer Segmentation (RFM) ====
rfm = df.groupby('order_id').agg({
    'date': lambda x: (df['date'].max() - x.max()).days,
    'order_id': 'count',
    'revenue': 'sum'
}).rename(columns={
    'date': 'recency',
    'order_id': 'frequency',
    'revenue': 'monetary'
})

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(rfm_scaled)
joblib.dump(kmeans, "models/kmeans_rfm_model.joblib")
print("✅ Saved KMeans segmentation model.")

# ==== 3. XGBoost/GradientBoosting: Dummy Churn Model ====
# Simulated churn labels (in real case, load or label manually)
df['customer_id'] = df['order_id'] % 1000  # fake customer_id
customer_df = df.groupby('customer_id').agg({
    'revenue': 'sum',
    'quantity': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'order_count'}).reset_index()

# Add fake churn label
np.random.seed(42)
customer_df['churn'] = np.random.choice([0, 1], size=len(customer_df))

X = customer_df[['revenue', 'quantity', 'order_count']]
y = customer_df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# Accuracy check (optional)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"✅ Trained churn model with accuracy: {acc:.2f}")

joblib.dump(clf, "models/xgb_churn_model.joblib")
print("✅ Saved churn prediction model.")
