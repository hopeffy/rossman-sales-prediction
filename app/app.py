import streamlit as st
import pandas as pd
import joblib
import os
import sys
import xgboost as xgb
from datetime import datetime

# --- Sistemin 'src' klasörünü görmesini sağlama ---
# Bu scriptin bulunduğu dizinin bir üst dizinini (proje kökü) al
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# `src` içerisindeki modülleri import et
from src.features import engineer_features
from src.config import MODEL_FILE_PATH, FEATURES, CATEGORICAL_FEATURES

# --- Model ve Veri Yükleme ---
@st.cache_resource
def load_model_and_data():
    """Modeli ve mağaza verisini yükler."""
    try:
        model = joblib.load(MODEL_FILE_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_FILE_PATH}. Please run the training pipeline first.")
        return None, None

    try:
        store_df_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'store.csv')
        store_df = pd.read_csv(store_df_path)
    except FileNotFoundError:
        st.error(f"Store data not found at {store_df_path}. Please ensure 'store.csv' is in 'data/raw/'.")
        return model, None
        
    return model, store_df

model, store_df = load_model_and_data()

# --- Streamlit Arayüzü ---
st.title('Rossmann Mağaza Satış Tahmini  예측')

if model is None or store_df is None:
    st.warning("Model veya mağaza verisi yüklenemediği için uygulama çalıştırılamıyor.")
else:
    # --- Kullanıcı Girdileri ---
    st.header('Tahmin İçin Bilgileri Girin')

    # Mağaza ID'si
    store_id = st.number_input('Mağaza ID', min_value=1, max_value=store_df['Store'].max(), value=1, step=1)

    # Tarih
    date = st.date_input('Tarih', value=datetime.now())

    # Promosyon durumu
    promo = st.selectbox('Promosyon Var Mı?', (1, 0), format_func=lambda x: 'Evet' if x == 1 else 'Hayır')
    
    # Diğer gerekli bilgiler (kullanıcıdan direkt alınmayacak, store_df'ten çekilecek)
    store_info = store_df[store_df['Store'] == store_id]
    if store_info.empty:
        st.warning(f"'{store_id}' ID'li mağaza bulunamadı.")
    else:
        # Buton
        if st.button('Satışları Tahmin Et'):
            with st.spinner('Tahmin yapılıyor...'):
                # --- Tahmin için DataFrame Oluşturma ---
                input_data = {
                    'Store': store_id,
                    'Date': pd.to_datetime(date),
                    'Promo': promo,
                    'StateHoliday': '0', # Varsayılan olarak tatil değil
                    'SchoolHoliday': 0, # Varsayılan olarak okul tatili değil
                    'Open': 1 # Tahmin yapılıyorsa mağaza açık varsayılır
                }
                
                # Mağaza bilgilerini birleştir
                input_df = pd.DataFrame([input_data])
                # Store bilgisini kullanarak store_df'ten geri kalan bilgileri al ve merge et
                # Reset index to avoid issues on merge
                input_df = pd.merge(input_df.reset_index(drop=True), store_info, on='Store', how='left')


                # --- Feature Engineering ---
                # Pipeline'da eğitilen modelle aynı feature'ları oluştur
                prediction_df = engineer_features(input_df.copy())
                
                # Kategorik özellikleri encode et (LabelEncoder eğitimde kullanıldığı için burada da kullanmak riskli olabilir)
                # Basit bir maplama daha güvenli olur veya pipeline'da kaydedilmiş encoder'lar gerekir.
                # Şimdilik en basit haliyle bırakalım, pipeline'daki gibi encode edelim.
                from sklearn.preprocessing import LabelEncoder
                for feature in CATEGORICAL_FEATURES:
                    # Not: Bu yaklaşım, eğer train setinde olmayan bir değer gelirse hata verir.
                    # Daha robust bir çözüm için, eğitim sırasında encoder'ları kaydetmek gerekir.
                    le = LabelEncoder()
                    if feature == 'StateHoliday':
                        # StateHoliday, mağaza verisinde (store.csv) bulunmaz.
                        # Bu nedenle encoder'ı olası tüm değerlerle (train.csv'den bilinen) eğitiyoruz.
                        le.fit(['0', 'a', 'b', 'c'])
                    else:
                        # Diğer kategorik özellikler için store_df'teki tüm olası değerlere uyum sağla
                        le.fit(store_df[feature].astype(str))
                    
                    # Girişi dönüştür
                    prediction_df[feature] = le.transform(prediction_df[feature].astype(str))


                # --- Tahmin ---
                # Modelin beklediği sırada özellikleri düzenle
                final_features = prediction_df[FEATURES]
                dpredict = xgb.DMatrix(final_features)
                
                prediction = model.predict(dpredict)
                
                # --- Sonucu Göster ---
                st.success(f"**Tahmin Edilen Satış: {prediction[0]:,.2f} €**")

st.sidebar.info(
    "Bu uygulama, bir Rossmann mağazasının belirli bir tarihteki "
    "satışını tahmin etmek için bir XGBoost modeli kullanır."
)

