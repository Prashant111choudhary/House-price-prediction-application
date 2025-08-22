



import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Load dataset
df = pd.read_csv(r"D:\Downloads\Housing.csv")

# Encode yes/no columns
yes_no_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in yes_no_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encode furnishingstatus
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Save model and scaler
with open("ridge_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save column names for consistent prediction
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model and scaler saved successfully!")
