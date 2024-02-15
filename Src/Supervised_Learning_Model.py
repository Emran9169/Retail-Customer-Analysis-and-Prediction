import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

def train_evaluate_linear_regression(X, y):
    """
    Trains and evaluates a linear regression model.
    Args:
    X (DataFrame): Features for training and testing.
    y (Series): Target variable.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, 'Linear Regression')

def train_evaluate_random_forest(X, y):
    """
    Trains and evaluates a random forest regressor with randomized search.
    Args:
    X (DataFrame): Features for training and testing.
    y (Series): Target variable.
    params (dict): Hyperparameters for RandomizedSearchCV.
    """
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    params = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    
}
    random_search = RandomizedSearchCV(rf, param_distributions=params, n_iter=10, cv=5,
                                       scoring='neg_mean_squared_error', random_state=42)
    random_search.fit(X_train, y_train)
    y_pred = random_search.best_estimator_.predict(X_test)
    evaluate_model(y_test, y_pred, 'Random Forest')

def train_evaluate_xgboost(X, y):
    """
    Trains and evaluates an XGBoost model with randomized search for hyperparameter tuning.
    
    Args:
    X (DataFrame): Features for training and testing.
    y (Series): Target variable.
    params (dict): Hyperparameters for RandomizedSearchCV.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    params = {
    'n_estimators': [100, 300, 500, 700],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'colsample_bytree': [0.3, 0.5, 0.7, 0.9],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2, 0.5]
    }
    xgb_search = RandomizedSearchCV(
        estimator=xg_reg,
        param_distributions=params,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=2
    )
    
    xgb_search.fit(X_train, y_train)
    y_pred = xgb_search.best_estimator_.predict(X_test)
    
    evaluate_model(y_test, y_pred, 'XGBoost')

    

def train_evaluate_lightgbm(X, y):
    """
    Trains and evaluates a LightGBM model.
    
    Args:
    X (DataFrame): Features for training and testing.
    y (Series): Target variable.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgbm_model = lgb.LGBMRegressor(random_state=42)
    lgbm_model.fit(X_train, y_train)
    
    y_pred = lgbm_model.predict(X_test)
    
    evaluate_model(y_test, y_pred, 'LightGBM')


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates the performance of a model using R2 score, RMSE, and MAE.
    Args:
    y_true (Series): True target values.
    y_pred (Series): Predicted target values.
    model_name (str): Name of the model being evaluated.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
