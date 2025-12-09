import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import os

def _rmsp_error_xgb(y_pred, y_true):
    """Custom RMSPE metric for XGBoost evaluation."""
    y_true = y_true.get_label()
    y_true[y_true == 0] = 1e-6 # Avoid division by zero
    percentage_error = (y_true - y_pred) / y_true
    rmspe = np.sqrt(np.mean(np.square(percentage_error)))
    return 'rmspe', rmspe

def train_model(X_train, y_train, X_val, y_val, params):
    """
    Trains an XGBoost model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        params (dict): XGBoost parameters.

    Returns:
        xgb.Booster: The trained XGBoost model.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=watchlist,
        custom_metric=_rmsp_error_xgb,
        maximize=False,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    return model

def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model on the validation set using RMSPE.

    Args:
        model (xgb.Booster): The trained model.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
    """
    dval = xgb.DMatrix(X_val)
    y_pred = model.predict(dval)
    
    # Use a separate RMSPE calculation for the final validation score
    y_true_np = y_val.to_numpy()
    y_true_np[y_true_np == 0] = 1e-6

    rmspe = np.sqrt(np.mean(np.square((y_true_np - y_pred) / y_true_np)))
    
    print(f'Final Validation RMSPE: {rmspe:.4f}')
    return rmspe

def save_model(model, model_path, model_name):
    """
    Saves the trained model to a file.

    Args:
        model (xgb.Booster): The trained model.
        model_path (str): The directory to save the model in.
        model_name (str): The name of the model file.
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    file_path = os.path.join(model_path, model_name)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


