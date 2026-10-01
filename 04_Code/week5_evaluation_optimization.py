# Week 5 — Model Evaluation, Optimization & Reporting
# This script demonstrates chronological holdout evaluation, time-series CV,
# Random Forest hyperparameter tuning and Ridge regularization.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("../02_Input_Data/agriculture_yield_ml_clean_features.csv")
features = [
    "year_index","state","crop","annual_rainfall_mm","avg_temp_c","soil_n_mgkg",
    "soil_ph","irrigation_index","area_ha","rainfall_deviation_mm",
    "temp_stress_index","rainfall_irrigation_interaction","soil_n_ph_interaction",
    "previous_yield_t_ha"
]
target = "yield_t_ha"

train = df["year"] <= 2022
test = df["year"] >= 2023
X_train, X_test = df.loc[train, features], df.loc[test, features]
y_train, y_test = df.loc[train, target], df.loc[test, target]

num = [c for c in features if c not in ["state","crop"]]
cat = ["state","crop"]

def prep(scale=False):
    nums = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        nums.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(nums), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat)
    ])

tscv = TimeSeriesSplit(n_splits=5)

rf = Pipeline([
    ("prep", prep(False)),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

grid = GridSearchCV(
    rf,
    {
        "model__n_estimators": [100, 200],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [1, 2, 4]
    },
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1
)
grid.fit(X_train, y_train)

pred = grid.predict(X_test)
print("Best parameters:", grid.best_params_)
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R2:", r2_score(y_test, pred))

ridge = Pipeline([
    ("prep", prep(True)),
    ("model", Ridge())
])
ridge_grid = GridSearchCV(
    ridge, {"model__alpha": [0.001,0.01,0.1,1,10,100]},
    scoring="neg_mean_absolute_error", cv=tscv, n_jobs=-1
)
ridge_grid.fit(X_train, y_train)
ridge_pred = ridge_grid.predict(X_test)
print("Best Ridge:", ridge_grid.best_params_)
print("Ridge MAE:", mean_absolute_error(y_test, ridge_pred))
