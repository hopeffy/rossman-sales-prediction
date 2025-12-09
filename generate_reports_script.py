import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
import joblib
from sklearn.metrics import mean_squared_error

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DOCS_IMG_DIR = os.path.join(BASE_DIR, 'docs', 'images')
STATS_FILE = os.path.join(BASE_DIR, 'docs', 'eda_stats.txt')
MODEL_STATS_FILE = os.path.join(BASE_DIR, 'docs', 'model_stats.txt')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_sales_model.joblib')

if not os.path.exists(DOCS_IMG_DIR):
    os.makedirs(DOCS_IMG_DIR)

# --- EDA PART ---
print("Starting EDA generation...")
try:
    merged_csv_path = os.path.join(DATA_DIR, 'train_merged.csv')
    if not os.path.exists(merged_csv_path):
         # Fallback if processed not found, try to merge raw
         train_path = os.path.join(RAW_DATA_DIR, 'train.csv')
         store_path = os.path.join(RAW_DATA_DIR, 'store.csv')
         train_df = pd.read_csv(train_path, low_memory=False)
         store_df = pd.read_csv(store_path)
         df = pd.merge(train_df, store_df, on='Store', how='left')
    else:
        df = pd.read_csv(merged_csv_path, low_memory=False)

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Sales Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['Sales'] > 0]['Sales'], bins=50, kde=True)
    plt.title('Distribution of Sales (Store Open)')
    plt.xlabel('Sales')
    plt.savefig(os.path.join(DOCS_IMG_DIR, 'eda_sales_dist.png'))
    plt.close()

    # 2. Sales vs StateHoliday
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='StateHoliday', y='Sales', data=df)
    plt.title('Sales vs. State Holiday')
    plt.savefig(os.path.join(DOCS_IMG_DIR, 'eda_sales_stateholiday.png'))
    plt.close()

    # 3. Missing Values
    missing_percentage = df.isnull().sum() / len(df) * 100
    missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
    
    with open(STATS_FILE, 'w') as f:
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write("\nMissing Values (%):\n")
        f.write(missing_percentage.to_string())
        f.write("\n\nBasic Statistics (Sales):\n")
        f.write(df['Sales'].describe().to_string())

    print("EDA generation complete.")

except Exception as e:
    print(f"EDA Generation Failed: {e}")

# --- MODEL PERFORMANCE PART ---
print("Starting Model Evaluation...")
try:
    # Load Model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
        # We need to recreate the features to generate importance plot
        # Simplified feature engineering for the sake of getting column names if model doesn't store them well
        # But XGBoost model usually stores feature names.
        
        plt.figure(figsize=(12, 10))
        # Check if model is XGBClassifier/Regressor or Booster
        if isinstance(model, xgb.XGBModel):
            xgb.plot_importance(model, max_num_features=20, height=0.8)
        else: # Try standard xgboost plot_importance
            xgb.plot_importance(model, max_num_features=20, height=0.8)
            
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(DOCS_IMG_DIR, 'model_feature_importance.png'))
        plt.close()
        
        with open(MODEL_STATS_FILE, 'w') as f:
            f.write(f"Model loaded from: {MODEL_PATH}\n")
            f.write("Feature Importance plot generated.\n")
            # Note: Calculating actual metrics requires recreating the exact test/val set which is complex 
            # without the exact split logic. We will rely on generic info or if we can process a small sample.
            
    else:
        print("Model file not found.")

    print("Model evaluation complete.")

except Exception as e:
    print(f"Model Evaluation Failed: {e}")
