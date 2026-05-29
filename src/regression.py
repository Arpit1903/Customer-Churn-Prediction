
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np

def evaluate_regression(model, X_test, y_test):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    results = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 Score": r2_score(y_test, predictions),
    }

    return results

def run_regression_models(X_train, X_test, y_train, y_test):

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
    }

    regression_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        regression_results[name] = evaluate_regression(
            model,
            X_test,
            y_test
        )

    return regression_results
