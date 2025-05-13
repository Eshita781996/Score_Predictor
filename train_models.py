import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os

# Load dataset
df = pd.read_csv("data/students.csv")
X = df[['hours_studied', 'hours_slept', 'attendance']]
y = df['score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

# Evaluate and find the best model
best_model = None
lowest_mse = float('inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"{name} MSE: {mse:.2f}")
    if mse < lowest_mse:
        best_model = model
        lowest_mse = mse

# Save the best model
os.makedirs("app/model", exist_ok=True)
with open("app/model/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"✅ Best model saved as best_model.pkl with MSE: {lowest_mse:.2f}")
