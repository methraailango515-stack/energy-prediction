import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("Loading dataset...")

df = pd.read_csv(
    'household_power_consumption.txt',
    sep=';',
    low_memory=False,
    na_values=['?']
)

print(f"Loaded {len(df):,} rows")

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df = df.dropna(subset=['Global_active_power'])
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df = df.dropna(subset=['Global_active_power'])

df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df['datetime'].dt.month

df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
df['Global_intensity'] = pd.to_numeric(df['Global_intensity'], errors='coerce')
df = df.dropna(subset=['Voltage', 'Global_intensity'])

df['consumption'] = df['Global_active_power'] * 100

features = ['hour', 'day_of_week', 'is_weekend', 'month', 'Voltage', 'Global_intensity']

X = df[features]
y = df['consumption']

if len(df) > 200_000:
    idx = np.random.RandomState(42).choice(len(df), 200_000, replace=False)
    X = X.iloc[idx]
    y = y.iloc[idx]
    print(f"Subsampled to {len(X):,} rows for speed")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest (this may take 1-3 minutes)...")
model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"\nModel R2 score: {r2_score(y_test, preds):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, preds):.3f} kWh")

joblib.dump(model, 'model.pkl')
joblib.dump(features, 'features.pkl')
print("\nModel saved as model.pkl")
print("Feature list saved as features.pkl")