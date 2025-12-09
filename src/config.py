import os

# --- Proje Kök Dizini ---
# Bu dosyanın (config.py) bulunduğu dizinin bir üst dizini (src/.. -> proje kökü)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Veri Yolları ---
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')
PROCESSED_TRAIN_FILE = os.path.join(PROCESSED_DATA_PATH, 'train_merged.csv')


# --- Model Kayıt Yolu ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')
MODEL_NAME = 'xgb_sales_model.joblib'
MODEL_FILE_PATH = os.path.join(MODEL_PATH, MODEL_NAME)


# --- Model Özellikleri ve Parametreleri ---

# Modelde kullanılacak özelliklerin listesi
FEATURES = [
    'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
    'Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpen',
    'Promo', 'Promo2', 'IsPromo2',
    'StateHoliday', 'SchoolHoliday'
]

# Hedef değişken
TARGET = 'Sales'

# Kategorik olarak ele alınacak özellikler
CATEGORICAL_FEATURES = ['StoreType', 'Assortment', 'StateHoliday']


# --- XGBoost Parametreleri ---
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'eval_metric': 'rmse',
    'seed': 42
}

