"""
1_EDA.py - Keşifsel Veri Analizi Sayfası
==========================================
Veri setinin görselleştirilmesi ve istatistiksel analizi.
"""

import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Proje kök dizinini Python path'ine ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.transforms import get_train_transforms, get_test_transforms, denormalize
from src.data.dataset import load_dataset, get_class_distribution, get_sample_images, CLASS_NAMES

st.set_page_config(page_title="EDA - Intel Image Classifier", page_icon="📊", layout="wide")

st.title("📊 Keşifsel Veri Analizi (EDA)")
st.markdown("Veri setini tanıyalım: sınıf dağılımları, örnek görüntüler ve augmentation etkileri.")

st.divider()

# Veri seti yolu
DATA_DIR = st.sidebar.text_input(
    "Veri seti dizini",
    value="data/seg_train/seg_train",
    help="Intel Image Classification eğitim veri setinin yolu"
)

if not os.path.exists(DATA_DIR):
    st.warning(
        f"⚠️ Veri seti dizini bulunamadı: `{DATA_DIR}`\n\n"
        "Lütfen veri setini [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) "
        "üzerinden indirip `data/` klasörüne çıkarın."
    )
    st.stop()

# ===== VERİ SETİNİ YÜKLE =====
@st.cache_data
def load_data_info():
    """Veri seti bilgilerini önbelleğe al."""
    test_transform = get_test_transforms()
    dataset = load_dataset(DATA_DIR, transform=test_transform)
    distribution = get_class_distribution(dataset)
    return distribution, len(dataset)

try:
    distribution, total_images = load_data_info()

    # ===== GENEL İSTATİSTİKLER =====
    st.subheader("📈 Genel İstatistikler")
    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("Toplam Görüntü", f"{total_images:,}")
    with metric_cols[1]:
        st.metric("Sınıf Sayısı", len(distribution))
    with metric_cols[2]:
        st.metric("Görüntü Boyutu", "150 × 150 px")
    with metric_cols[3]:
        avg_per_class = total_images // len(distribution)
        st.metric("Sınıf Başına Ort.", f"{avg_per_class:,}")

    st.divider()

    # ===== SINIF DAĞILIMI =====
    st.subheader("📊 Sınıf Dağılımı")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(distribution))
    bars = ax.bar(distribution.keys(), distribution.values(), color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Sınıf", fontsize=12)
    ax.set_ylabel("Görüntü Sayısı", fontsize=12)
    ax.set_title("Sınıf Başına Görüntü Dağılımı", fontsize=14, fontweight="bold")

    # Bar'ların üzerine sayıları yaz
    for bar, count in zip(bars, distribution.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ===== ÖRNEK GÖRÜNTÜLER =====
    st.subheader("🖼️ Sınıf Başına Örnek Görüntüler")

    test_transform = get_test_transforms()
    dataset = load_dataset(DATA_DIR, transform=test_transform)
    samples = get_sample_images(dataset, num_per_class=3)

    for class_name in CLASS_NAMES:
        if class_name in samples:
            st.markdown(f"**{class_name.capitalize()}**")
            cols = st.columns(3)
            for i, (img_tensor, _) in enumerate(samples[class_name]):
                with cols[i]:
                    img = denormalize(img_tensor).permute(1, 2, 0).numpy()
                    st.image(img, use_container_width=True)

    st.divider()

    # ===== DATA AUGMENTATION ÖRNEKLERİ =====
    st.subheader("🔄 Data Augmentation Örnekleri")
    st.markdown(
        "Aynı görüntüye uygulanan farklı augmentation'lar. "
        "Her çalıştırmada rastgele farklı dönüşümler uygulanır."
    )

    if st.button("🔄 Yeniden Augmentation Uygula"):
        st.rerun()

    # İlk sınıftan bir örnek al ve birden fazla augmentation uygula
    from PIL import Image
    raw_dataset = load_dataset(DATA_DIR, transform=None)
    sample_img, sample_label = raw_dataset[0]  # PIL Image

    train_transform = get_train_transforms()
    aug_cols = st.columns(5)

    with aug_cols[0]:
        st.markdown("**Orijinal**")
        st.image(sample_img, use_container_width=True)

    for i in range(1, 5):
        with aug_cols[i]:
            st.markdown(f"**Augmented #{i}**")
            aug_img = train_transform(sample_img)
            aug_img = denormalize(aug_img).permute(1, 2, 0).numpy()
            st.image(aug_img, use_container_width=True)

except Exception as e:
    st.error(f"Hata: {e}")
