import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

# Proje içi modüller
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, PROCESSED_TRAIN_FILE,
    MODEL_PATH, MODEL_NAME, FEATURES, TARGET, CATEGORICAL_FEATURES, XGB_PARAMS
)
from data_prep import merge_data
from features import engineer_features
from model import train_model, evaluate_model, save_model

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

def run_training_pipeline():
    """
    Runs the complete model training pipeline from data prep to model saving.
    """
    # 1. Veri Hazırlama
    print("--- Step 1: Data Preparation ---")
    merge_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    
    # 2. Veriyi Yükleme
    print("\n--- Step 2: Loading Processed Data ---")
    try:
        df = pd.read_csv(PROCESSED_TRAIN_FILE, low_memory=False)
        print(f"Loaded {PROCESSED_TRAIN_FILE} successfully.")
    except FileNotFoundError:
        print(f"Error: {PROCESSED_TRAIN_FILE} not found. Exiting pipeline.")
        return

    # 3. Feature Engineering
    print("\n--- Step 3: Feature Engineering ---")
    df = engineer_features(df)

    # 4. Kategorik Veri Kodlama
    print("\n--- Step 4: Encoding Categorical Features ---")
    for feature in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))
    print("Categorical features encoded.")

    # 5. Eğitim ve Validasyon Setlerini Ayırma
    print("\n--- Step 5: Splitting Data into Train/Validation Sets ---")
    # Sadece mağazalar açıkken ve satış varken olan veriyi al
    df_train = df[(df['Open'] == 1) & (df['Sales'] > 0)]
    
    # Zaman bazlı ayırma
    validation_date = df_train['Date'].max() - pd.DateOffset(weeks=6)
    train_indices = df_train['Date'] < validation_date
    val_indices = df_train['Date'] >= validation_date

    X_train, y_train = df_train[train_indices][FEATURES], df_train[train_indices][TARGET]
    X_val, y_val = df_train[val_indices][FEATURES], df_train[val_indices][TARGET]
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # 6. Model Eğitimi
    print("\n--- Step 6: Model Training ---")
    model = train_model(X_train, y_train, X_val, y_val, XGB_PARAMS)

    # 7. Model Değerlendirme
    print("\n--- Step 7: Model Evaluation ---")
    evaluate_model(model, X_val, y_val)
    
    # 8. Modeli Kaydetme
    print("\n--- Step 8: Saving Model ---")
    save_model(model, MODEL_PATH, MODEL_NAME)
    
    print("\n--- Pipeline Finished Successfully! ---")


if __name__ == '__main__':
    run_training_pipeline()

