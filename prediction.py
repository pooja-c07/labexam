
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. LOAD OR GENERATE DATASET
# ==========================================

# To use your real dataset, uncomment:
# df = pd.read_csv("house_data.csv")

np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    "SquareFeet": np.random.randint(800, 4500, n_samples),
    "Bedrooms": np.random.randint(1, 6, n_samples),
    "Bathrooms": np.random.randint(1, 4, n_samples),
    "Neighborhood": np.random.choice(
        ["Downtown", "Suburbs", "Rural"], n_samples
    ),
    "YearBuilt": np.random.randint(1950, 2025, n_samples),
    "Price": np.random.randint(150000, 800000, n_samples)
})

# Add missing values
df.loc[df.sample(frac=0.05, random_state=42).index, "SquareFeet"] = np.nan

# Remove rows where the target price is missing
df = df.dropna(subset=["Price"])

# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Price"])
y = df["Price"]

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 3. PREPROCESSING
# ==========================================

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ==========================================
# 4. BUILD AND TRAIN MODEL
# ==========================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ))
])

print("Training house price prediction model...")
model.fit(X_train, y_train)

# ==========================================
# 5. MODEL EVALUATION
# ==========================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# ==========================================
# 6. PREDICT PRICE FOR A NEW HOUSE
# ==========================================

new_house = pd.DataFrame([{
    "SquareFeet": 2500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Neighborhood": "Suburbs",
    "YearBuilt": 2015
}])

predicted_price = model.predict(new_house)[0]

print("\n--- New House Prediction ---")
print(f"Predicted house price: ${predicted_price:,.2f}")

