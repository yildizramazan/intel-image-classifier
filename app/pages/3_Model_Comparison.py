"""
3_Model_Comparison.py - Model Karşılaştırma Sayfası
Eğitim sonuçlarını JSON dosyasından okuyarak karşılaştırır.
"""

import streamlit as st
import sys, os
import json
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

st.set_page_config(page_title="Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Model Karşılaştırma")
st.markdown("Custom CNN vs ResNet18 (Transfer Learning) sonuçlarını karşılaştırın.")
st.divider()

RESULTS_PATH = "models/results.json"

if not os.path.exists(RESULTS_PATH):
    st.warning(
        "⚠️ Henüz sonuç dosyası bulunamadı.\n\n"
        "Colab'da eğitim tamamlandıktan sonra `models/results.json` dosyasını indirip "
        "projenin `models/` klasörüne koyun."
    )

    st.markdown("### 📝 Beklenen JSON Formatı")
    st.code('''{
    "custom_cnn": {
        "val_accuracy": 0.85,
        "val_f1_score": 0.84,
        "train_loss": 0.42,
        "val_loss": 0.48,
        "training_time_sec": 120.5,
        "total_params": 1500000,
        "trainable_params": 1500000,
        "history": {
            "train_loss": [1.2, 0.8, ...],
            "val_loss": [1.1, 0.7, ...],
            "train_acc": [0.4, 0.6, ...],
            "val_acc": [0.5, 0.65, ...]
        }
    },
    "resnet18": { ... }
}''', language="json")
    st.stop()

try:
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    st.subheader("🏆 En İyi Sonuçlar")
    col1, col2 = st.columns(2)

    for col, model_name in [(col1, "custom_cnn"), (col2, "resnet18")]:
        with col:
            title = "Custom CNN" if model_name == "custom_cnn" else "ResNet18"
            st.markdown(f"### {title}")
            if model_name in results:
                r = results[model_name]
                st.metric("Val Accuracy", f"{r.get('val_accuracy', 0):.4f}")
                st.metric("Val F1-Score", f"{r.get('val_f1_score', 0):.4f}")
                st.metric("Eğitim Süresi", f"{r.get('training_time_sec', 0):.1f}s")
                st.metric("Eğitilebilir Param.", f"{r.get('trainable_params', 0):,}")
            else:
                st.info(f"Henüz {title} eğitimi yok.")

    st.divider()

    # Loss eğrileri karşılaştırma
    st.subheader("📊 Eğitim Eğrileri Karşılaştırması")

    has_history = any("history" in results.get(m, {}) for m in ["custom_cnn", "resnet18"])

    if has_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        colors = {"custom_cnn": "#6366f1", "resnet18": "#10b981"}
        labels = {"custom_cnn": "Custom CNN", "resnet18": "ResNet18"}

        for model_name in ["custom_cnn", "resnet18"]:
            if model_name in results and "history" in results[model_name]:
                h = results[model_name]["history"]
                ax1.plot(h["val_loss"], label=labels[model_name], color=colors[model_name], lw=2)
                ax2.plot(h["val_acc"], label=labels[model_name], color=colors[model_name], lw=2)

        ax1.set_title("Val Loss Karşılaştırması", fontweight="bold")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.set_title("Val Accuracy Karşılaştırması", fontweight="bold")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Eğitim eğrileri (history) verisi bulunamadı.")

except Exception as e:
    st.error(f"Hata: {e}")
