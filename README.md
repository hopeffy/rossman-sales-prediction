# Rossmann Mağaza Satış Tahmin Projesi

Bu proje, Rossmann mağaza zincirinin satışlarını tahmin etmek için geliştirilmiş bir makine öğrenmesi modelini ve bu modeli sunan web uygulamasını içerir. Projenin amacı, geçmiş verileri kullanarak gelecekteki satışları yüksek doğrulukla öngörmektir.

## Problem Tanımı
Rossmann, Avrupa'nın en büyük eczane ve perakende zincirlerinden biridir. Mağaza yöneticileri, envanter yönetimi, personel planlaması ve promosyon stratejilerini optimize etmek için günlük satış tahminlerine ihtiyaç duymaktadır. Bu proje, zaman serisi analizi ve makine öğrenmesi tekniklerini kullanarak bu ihtiyacı karşılamayı hedefler.

## Veri Seti
Projede kullanılan veri seti, Kaggle'daki [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) yarışmasından alınmıştır. Veri seti, 1115 Rossmann mağazasının 2013-2015 yılları arasındaki satış verilerini, mağaza özelliklerini ve promosyon bilgilerini içerir.

- `train.csv`: Mağaza ID, tarih, satış miktarı ve müşteri sayısı gibi günlük veriler.
- `store.csv`: Mağaza türü, rekabet bilgileri ve promosyon detayları gibi her mağazaya özel statik bilgiler.
- `test.csv`: Tahmin yapılacak gelecek periyot için bilgiler.

## Kullanılan Teknolojiler
- **Programlama Dili:** Python 3.x
- **Veri Analizi ve İşleme:** Pandas, NumPy
- **Makine Öğrenmesi Modeli:** XGBoost
- **Web Uygulaması:** Streamlit
- **Kod ve Ortam Yönetimi:** Jupyter Notebooks, Scikit-learn, Joblib

## Proje Yapısı
```
rossmann-sales-prediction/
├── app/
│   └── app.py              # Streamlit web uygulaması
├── data/
│   ├── raw/                # Ham veri setleri (train.csv, store.csv)
│   └── processed/          # İşlenmiş ve birleştirilmiş veri
├── docs/                   # Proje raporları ve sunumlar
├── models/                 # Eğitilmiş ve kaydedilmiş modeller
├── notebooks/              # Veri analizi ve model geliştirme adımları
├── src/                    # Üretim (production) kodları
│   ├── config.py           # Konfigürasyon ve parametreler
│   ├── data_prep.py        # Veri hazırlama script'i
│   ├── features.py         # Özellik mühendisliği script'i
│   ├── model.py            # Model eğitimi ve değerlendirme script'i
│   └── pipeline.py         # Uçtan uca eğitim pipeline'ı
├── .gitignore
├── README.md
└── requirements.txt        # Gerekli Python kütüphaneleri
```

## Kurulum ve Çalıştırma

**1. Depoyu Klonlama:**
```bash
git clone <repository-url>
cd rossmann-sales-prediction
```

**2. Sanal Ortam Oluşturma ve Aktive Etme:**
```bash
python -m venv venv
# Windows için
.\venv\Scripts\activate
# macOS/Linux için
source venv/bin/activate
```

**3. Gerekli Kütüphaneleri Yükleme:**
```bash
pip install -r requirements.txt
```

**3. Veri Setini İndirme:**
Kaggle'dan `train.csv`, `store.csv` ve `test.csv` dosyalarını indirin ve `data/raw/` klasörüne yerleştirin.

**4. Eğitim Pipeline'ını Çalıştırma:**
Modeli eğitmek ve `models/` klasörüne kaydetmek için aşağıdaki komutu çalıştırın:
```bash
python src/pipeline.py
```
Bu script, veri hazırlama, özellik mühendisliği ve model eğitimini otomatik olarak gerçekleştirir.

**5. Web Uygulamasını Başlatma:**
Tahmin uygulamasını başlatmak için:
```bash
streamlit run app/app.py
```
Uygulama, varsayılan web tarayıcınızda açılacaktır.

## Model Sonuçları

Modelin performansı, yarışmanın resmi metriği olan **Kök Ortalama Kare Yüzde Hatası (RMSPE)** ile ölçülmüştür.

- **Baseline Model (Basit Ortalamaya Dayalı):**
  - Yaklaşık RMSPE: ~0.35 - 0.40
  - Bu model, her mağazanın geçmiş satış ortalamasını tahmin olarak kullanır ve basit bir referans noktası sağlar.

- **Final Model (XGBoost):**
  - **Validasyon RMSPE: ~0.12 - 0.15**
  - Bu model, detaylı özellik mühendisliği ve gradient boosting algoritmasının gücü sayesinde baseline modele göre **%60'tan fazla iyileşme** sağlamıştır. Model, özellikle promosyonlar, tatiller ve rekabet gibi karmaşık ilişkileri başarıyla öğrenmiştir.

## İş Etkisi
Bu projenin çıktıları, Rossmann için aşağıdaki alanlarda doğrudan iş değeri yaratabilir:

- **Stok Optimizasyonu:** Doğru satış tahminleri, mağazaların doğru ürünleri doğru zamanda stoklamasını sağlayarak fazla stok maliyetini veya stok tükenmesi nedeniyle oluşacak satış kayıplarını azaltır.
- **Personel Planlaması:** Yoğun geçmesi beklenen günler için ek personel planlaması, sakin günler için ise personel optimizasyonu yapılarak operasyonel verimlilik artırılır.
- **Pazarlama ve Promosyon Stratejileri:** Promosyonların satışlar üzerindeki etkisinin modellenmesi, gelecekteki pazarlama kampanyalarının daha etkili bir şekilde planlanmasına yardımcı olur.
