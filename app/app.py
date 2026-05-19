"""
app.py - Streamlit Ana Giriş Noktası
=======================================
Intel Image Classifier Dashboard'un ana sayfası.
Multi-page yapısı Streamlit'in pages/ klasörü ile otomatik çalışır.

Çalıştırma: streamlit run app/app.py
"""

import streamlit as st

# ===== SAYFA AYARLARI =====
st.set_page_config(
    page_title="Intel Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    /* Ana tema renkleri */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #06b6d4;
        --bg-dark: #0f172a;
        --card-bg: #1e293b;
    }

    /* Sidebar stili */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }

    /* Metric kartları */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 16px;
    }

    /* Başlık stili */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Kart stili */
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: transform 0.2s;
    }

    .feature-card:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ===== ANA SAYFA =====
st.markdown('<p class="main-title">🖼️ Intel Image Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">'
    'PyTorch CNN ile Görüntü Sınıflandırma & Hiperparametre Optimizasyonu Dashboard'
    '</p>',
    unsafe_allow_html=True
)

st.divider()

# Proje açıklaması
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Proje Hakkında")
    st.markdown("""
    Bu proje, **Intel Image Classification** veri seti üzerinde
    iki farklı CNN modelini eğitip karşılaştırır:

    - **Custom CNN:** Sıfırdan tasarlanmış evrişimli sinir ağı
    - **ResNet18:** Transfer Learning ile fine-tune edilmiş model

    Her iki model de **Optuna** ile hiperparametre optimizasyonu
    yapılarak en iyi performansa ulaştırılmıştır.
    """)

with col2:
    st.markdown("### 📊 Veri Seti")
    st.markdown("""
    **Intel Image Classification** — 6 Sınıf:

    | Sınıf | Açıklama |
    |-------|----------|
    | 🏢 Buildings | Binalar |
    | 🌲 Forest | Orman |
    | 🏔️ Glacier | Buzul |
    | ⛰️ Mountain | Dağ |
    | 🌊 Sea | Deniz |
    | 🛣️ Street | Sokak |
    """)

st.divider()

# Sayfa navigasyonu
st.markdown("### 📑 Sayfalar")

pages_info = {
    "1️⃣ EDA": "Veri setini keşfet: sınıf dağılımları, örnek görüntüler, augmentation örnekleri",
    "2️⃣ Training": "Model eğitimi: hiperparametre seçimi, eğitim süreci, loss/accuracy grafikleri",
    "3️⃣ Model Comparison": "Custom CNN vs ResNet18: metrik karşılaştırma, eğitim eğrileri",
    "4️⃣ Prediction": "Resim yükle ve model tahmini + sınıf olasılıkları görselleştirme",
}

cols = st.columns(len(pages_info))
for col, (page, desc) in zip(cols, pages_info.items()):
    with col:
        st.markdown(f"**{page}**")
        st.caption(desc)

st.divider()

# Teknoloji yığını
st.markdown("### 🛠️ Kullanılan Teknolojiler")
tech_cols = st.columns(4)
techs = [
    ("🔥 PyTorch", "Model & Eğitim"),
    ("🔍 Optuna", "HPO"),
    ("📊 Streamlit", "Arayüz"),
    ("📦 ONNX", "Model Export"),
]
for col, (tech, desc) in zip(tech_cols, techs):
    with col:
        st.metric(label=tech, value=desc)
