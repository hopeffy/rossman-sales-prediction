import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Rossmann Satış Simülasyonu",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sabitler ve Yollar ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'xgb_sales_model.joblib')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'store.csv')

# --- Yardımcı Fonksiyonlar ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_store_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Veri dosyası bulunamadı: {DATA_PATH}")
        return None
    return pd.read_csv(DATA_PATH)

def get_sample_store_id(store_df, store_type, assortment):
    """Seçilen özelliklere uygun bir örnek mağaza ID'si döndürür."""
    subset = store_df[
        (store_df['StoreType'] == store_type) & 
        (store_df['Assortment'] == assortment)
    ]
    if subset.empty:
        # Tam eşleşme yoksa sadece tipe göre döndür
        subset = store_df[store_df['StoreType'] == store_type]
        if subset.empty:
            return 1 # Varsayılan
    return subset.iloc[0]['Store']

def prepare_input_features(date, store_id, promo, state_holiday, school_holiday, store_data, competition_distance_override=None):
    """Modelin beklediği formatta DataFrame oluşturur."""
    
    # Tarih özellikleri
    input_data = {
        'Store': [store_id],
        'DayOfWeek': [date.weekday()],
        'Date': [pd.to_datetime(date)], # Tarih formatı
        'Sales': [0], # Dummy
        'Customers': [0], # Dummy
        'Open': [1],
        'Promo': [promo],
        'StateHoliday': [state_holiday],
        'SchoolHoliday': [school_holiday],
        'Year': [date.year],
        'Month': [date.month],
        'Day': [date.day],
        'WeekOfYear': [date.isocalendar()[1]]
    }
    
    df = pd.DataFrame(input_data)
    
    # Store verilerini merge et
    store_info = store_data[store_data['Store'] == store_id].iloc[0]
    
    # Statik özellikleri ekle
    # Not: Model eğitiminde kullanılan sütun isimleri ve feature engineering adımları buraya eklenmeli
    # 4_Model_Opt.ipynb ve features.py'deki mantığı taklit ediyoruz
    
    # Feature Engineering (Özet)
    df['StoreType'] = store_info['StoreType']
    df['Assortment'] = store_info['Assortment']
    
    # CompetitionDistance (Kullanıcı değiştirebilsin diye override seçeneği)
    if competition_distance_override is not None:
        df['CompetitionDistance'] = competition_distance_override
    else:
        df['CompetitionDistance'] = store_info['CompetitionDistance']
        
    df['CompetitionOpenSinceMonth'] = store_info['CompetitionOpenSinceMonth']
    df['CompetitionOpenSinceYear'] = store_info['CompetitionOpenSinceYear']
    df['Promo2'] = store_info['Promo2']
    df['Promo2SinceWeek'] = store_info['Promo2SinceWeek']
    df['Promo2SinceYear'] = store_info['Promo2SinceYear']
    df['PromoInterval'] = store_info['PromoInterval']

    # Eksik verileri doldur (Model eğitimiyle tutarlı olmalı)
    df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].mean(), inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('', inplace=True)

    # CompetitionOpen Hesaplama
    df['CompetitionOpen'] = (df['Year'] - df['CompetitionOpenSinceYear']) * 12 + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: max(x, 0))

    # IsPromo2 Hesaplama
    def is_promo2_active(row):
        if row['Promo2'] == 0: return 0
        if row['Year'] < row['Promo2SinceYear']: return 0
        month_str = row['Date'].strftime('%b')
        return 1 if month_str in row['PromoInterval'] else 0
    
    df['IsPromo2'] = df.apply(is_promo2_active, axis=1)

    # Label Encoding (Manuel Mapping - Model eğitimindeki LabelEncoder'ı kaydetmediysek bu risklidir ama basit map ile çözelim)
    # Rossmann verisinde genelde: a:0, b:1, c:2, d:3
    type_map = {'a':0, 'b':1, 'c':2, 'd':3}
    assort_map = {'a':0, 'b':1, 'c':2}
    holiday_map = {0:0, '0':0, 'a':1, 'b':2, 'c':3} # StateHoliday genellikle object gelir
    
    df['StoreType'] = df['StoreType'].map(type_map)
    df['Assortment'] = df['Assortment'].map(assort_map)
    # StateHoliday UI'dan string veya int gelebilir
    df['StateHoliday'] = df['StateHoliday'].map(lambda x: holiday_map.get(str(x), 0))

    # Modelin beklediği özellik sırası
    features = [
        'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
        'Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpen',
        'Promo', 'Promo2', 'IsPromo2',
        'StateHoliday', 'SchoolHoliday'
    ]
    
    return df[features]

# --- UI Tasarımı ---

st.title("🛍️ Mağaza Satış Tahmin Simülatörü")
st.markdown("""
Bu araç, Rossmann mağazaları için **yapay zeka tabanlı** satış tahmini yapar. 
Sol panelden mağaza özelliklerini ve tarihi değiştirerek farklı senaryoları test edebilirsiniz.
""")

# --- Sidebar (Girdiler) ---
model = load_model()
store_data = load_store_data()

with st.sidebar:
    st.header("⚙️ Simülasyon Parametreleri")
    
    # 1. Mağaza Konfigürasyonu
    st.subheader("Mağaza Özellikleri")
    store_type = st.selectbox("Mağaza Tipi", ['a', 'b', 'c', 'd'], format_func=lambda x: f"Tip {x.upper()} (Standart/AVM vb.)")
    assortment = st.selectbox("Ürün Çeşitliliği", ['a', 'c'], format_func=lambda x: "Temel" if x=='a' else "Geniş Kapsamlı")
    
    # Otomatik ID seçimi
    selected_store_id = get_sample_store_id(store_data, store_type, assortment)
    st.info(f"Seçilen özelliklere uygun referans mağaza: **Store {selected_store_id}**")

    competition_dist = st.slider("En Yakın Rakip Mesafesi (m)", 0, 20000, 1000, step=100, help="Mağazaya en yakın rakibin metre cinsinden uzaklığı.")

    st.divider()

    # 2. Tarih ve Kampanya
    st.subheader("Zaman ve Kampanya")
    prediction_date = st.date_input("Tahmin Tarihi", value=pd.to_datetime("2015-08-01"))
    promo = st.toggle("Promosyon Var mı?", value=True)
    school_holiday = st.toggle("Okul Tatili mi?", value=False)
    state_holiday = st.selectbox("Resmi Tatil Durumu", ['0', 'a', 'b', 'c'], format_func=lambda x: "Yok" if x=='0' else f"Tatil Tipi {x}")

# --- Ana Ekran (Hesaplama ve Sonuçlar) ---

if st.button("🚀 Satışları Simüle Et", type="primary", use_container_width=True):
    if model and store_data is not None:
        with st.spinner('Yapay zeka hesaplama yapıyor...'):
            # 1. Ana Senaryo Tahmini
            input_df = prepare_input_features(
                prediction_date, selected_store_id, 1 if promo else 0, 
                state_holiday, 1 if school_holiday else 0, store_data,
                competition_distance_override=competition_dist
            )
            # XGBoost Booster modeli DMatrix bekler
            dmatrix = xgb.DMatrix(input_df)
            prediction = model.predict(dmatrix)[0]
            
            # 2. Karşılaştırma Senaryosu (Promosyonun tersi durumu)
            input_df_alt = input_df.copy()
            alt_promo = 0 if promo else 1
            input_df_alt['Promo'] = alt_promo
            
            dmatrix_alt = xgb.DMatrix(input_df_alt)
            prediction_alt = model.predict(dmatrix_alt)[0]
            
            # Fark Hesaplama
            diff = prediction - prediction_alt
            diff_pct = (diff / prediction_alt) * 100 if prediction_alt != 0 else 0
            
            # --- Sonuç Gösterimi ---
            st.markdown("### 📊 Tahmin Sonuçları")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Tahmini Günlük Satış", 
                    value=f"€{prediction:,.0f}",
                    delta=f"{diff:,.0f} € ({'Promosyon Etkisi' if promo else 'Promosyon Yok'})",
                    delta_color="normal" if promo else "off"
                )
            
            with col2:
                st.metric(
                    label="Alternatif Senaryo",
                    value=f"€{prediction_alt:,.0f}",
                    help="Eğer promosyon durumu tam tersi olsaydı beklenen satış."
                )
                
            with col3:
                impact_color = "green" if diff > 0 else "red"
                st.markdown(f"""
                **Promosyonun Etkisi:**  
                :<span style='color:{impact_color}'>{impact_color}</span>: **%{abs(diff_pct):.1f}** değişim yaratıyor.
                """, unsafe_allow_html=True)
            
            # --- Grafiksel Analiz ---
            st.markdown("---")
            st.subheader("Senaryo Karşılaştırması")
            
            chart_data = pd.DataFrame({
                'Senaryo': ['Seçilen Durum', 'Alternatif (Promo Ters)'],
                'Tahmini Satış (€)': [prediction, prediction_alt],
                'Durum': ['Mevcut', 'Alternatif']
            })
            
            # Basit Bar Chart
            st.bar_chart(chart_data, x='Senaryo', y='Tahmini Satış (€)', color='Durum')
            
            # Ek Analizler
            st.info(f"""
            💡 **Analiz Notu:**  
            Seçtiğiniz **Tip {store_type.upper()}** mağazası ve **{competition_dist}m** rakip mesafesi ile yapılan simülasyona göre;
            Promosyon yapılması satışları **€{abs(diff):,.0f}** kadar {'artırıyor' if diff > 0 else 'azaltıyor'}.
            """)
            
    else:
        st.warning("Model veya veri yüklenemediği için tahmin yapılamıyor.")
else:
    st.info("Simülasyonu başlatmak için sol panelden parametreleri seçip butona basın.")