import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. GENERATE MOCK DATASET
# (Replace this section by loading your real CSV)
# ==========================================
# Example: df = pd.read_csv('your_housing_data.csv')
np.random.seed(42)
n_samples = 1000

mock_data = {
    "SquareFeet": np.random.randint(800, 4500, size=n_samples),
    "Bedrooms": np.random.randint(1, 6, size=n_samples),
    "Bathrooms": np.random.randint(1, 4, size=n_samples),
    "Neighborhood": np.random.choice(
        ["Downtown", "Suburbs", "Rural"], size=n_samples
    ),
    "YearBuilt": np.random.randint(1950, 2025, size=n_samples),
    # Missing values simulated in 'SquareFeet'
    "Price": np.random.randint(150000, 800000, size=n_samples),
}
df = pd.DataFrame(mock_data)
# Introduce random missing values for demo purposes
df.loc[df.sample(frac=0.05).index, "SquareFeet"] = np.nan


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================
X = df.drop(columns=["Price"])
y = df["Price"]

# Identify column types automatically
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Split into Training and Testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# 3. PREPROCESSING & PIPELINE SETUP
# ==========================================
# Preprocessing for numerical features: impute missing values with median, then scale
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Preprocessing for categorical features: impute with most frequent, then one-hot encode
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

# Combine transformers into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Create the full modeling pipeline containing preprocessing and the regressor
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(n_estimators=100, random_state=42),
        ),
    ]
)


# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================
# Train the model
print("Training the Random Forest model...")
model_pipeline.fit(X_train, y_train)

# Make predictions on test data
predictions = model_pipeline.predict(X_test)

# Calculate Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")


# ==========================================
# 5. PREDICTING ON NEW DATA
# ==========================================
new_house = pd.DataFrame(
    [
        {
            "SquareFeet": 2500,
            "Bedrooms": 3,
            "Bathrooms": 2,
            "Neighborhood": "Suburbs",
            "YearBuilt": 2015,
        }
    ]
)

predicted_price = model_pipeline.predict(new_house)[0]
print(f"\nPredicted price for the new house layout: ${predicted_price:,.2f}")
