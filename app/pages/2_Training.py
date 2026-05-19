"""
2_Training.py - Model Eğitim Sayfası
"""
import streamlit as st
import sys, os, time
import torch, torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.transforms import get_train_transforms, get_test_transforms
from src.data.dataset import load_dataset
from src.data.dataloader import create_dataloaders
from src.models.custom_cnn import CustomCNN
from src.models.transfer_model import TransferResNet18
from src.training.optimizer_factory import create_optimizer, create_scheduler
from src.training.trainer import Trainer
from src.evaluation.metrics import calculate_f1


st.set_page_config(page_title="Training", page_icon="🏋️", layout="wide")
st.title("🏋️ Model Eğitimi")
st.markdown("Hiperparametreler seçin ve modeli eğitin.")
st.divider()

# Sidebar: Hiperparametreler
st.sidebar.header("⚙️ Hiperparametreler")
model_type = st.sidebar.selectbox("Model Tipi", ["custom_cnn", "resnet18"])
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    value=1e-3, format_func=lambda x: f"{x:.0e}"
)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
optimizer_name = st.sidebar.selectbox("Optimizer", ["Adam", "SGD"])
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.3, 0.05)
num_epochs = st.sidebar.slider("Epoch Sayısı", 5, 30, 15)

num_blocks, base_filters = 4, 32
if model_type == "custom_cnn":
    num_blocks = st.sidebar.slider("Conv Blok Sayısı", 3, 5, 4)
    base_filters = st.sidebar.selectbox("Başlangıç Filtre", [16, 32, 64], index=1)

train_dir = st.sidebar.text_input("Eğitim Dizini", "data/seg_train/seg_train")
test_dir = st.sidebar.text_input("Test Dizini", "data/seg_test/seg_test")

# Parametre özeti
st.subheader("📋 Seçilen Parametreler")
cols = st.columns(4)
cols[0].metric("Model", model_type)
cols[1].metric("LR", f"{learning_rate:.0e}")
cols[2].metric("Batch", batch_size)
cols[3].metric("Optimizer", optimizer_name)
st.divider()

if st.button("🚀 Eğitimi Başlat", type="primary", use_container_width=True):
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        st.error("⚠️ Veri seti dizinleri bulunamadı!")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    st.info(f"🖥️ Cihaz: **{device}**")

    progress = st.progress(0)
    status = st.empty()

    # Veri yükleme
    status.text("📂 Veri yükleniyor...")
    train_ds = load_dataset(train_dir, get_train_transforms())
    test_ds = load_dataset(test_dir, get_test_transforms())
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, test_ds, batch_size)
    progress.progress(15)

    # Model
    status.text("🏗️ Model oluşturuluyor...")
    if model_type == "custom_cnn":
        model = CustomCNN(6, num_blocks, base_filters, dropout_rate)
    else:
        model = TransferResNet18(6, dropout_rate)
    param_info = model.count_parameters()
    st.write(f"Parametreler — Toplam: **{param_info['total']:,}** | Eğitilebilir: **{param_info['trainable']:,}**")
    progress.progress(25)

    # Eğitim
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, optimizer_name, learning_rate)
    scheduler = create_scheduler(optimizer)
    trainer = Trainer(model, criterion, optimizer, device, scheduler)

    status.text("🏋️ Eğitim devam ediyor...")
    t0 = time.time()
    history = trainer.fit(train_loader, val_loader, num_epochs, 5, model_type, verbose=True)
    training_time = time.time() - t0
    progress.progress(85)

    # Sonuçlar
    trainer.load_best_model(model_type)
    f1 = calculate_f1(model, val_loader, device)

    st.divider()
    st.subheader("📊 Sonuçlar")
    rc = st.columns(4)
    rc[0].metric("Best Val Acc", f"{max(history['val_acc']):.4f}")
    rc[1].metric("Best Val Loss", f"{min(history['val_loss']):.4f}")
    rc[2].metric("F1-Score", f"{f1:.4f}")
    rc[3].metric("Süre", f"{training_time:.1f}s")

    # Grafikler
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train", color="#6366f1", lw=2)
    ax1.plot(history["val_loss"], label="Val", color="#f43f5e", lw=2)
    ax1.set_title("Loss Eğrisi", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history["train_acc"], label="Train", color="#6366f1", lw=2)
    ax2.plot(history["val_acc"], label="Val", color="#10b981", lw=2)
    ax2.set_title("Accuracy Eğrisi", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    progress.progress(100)
    status.text("✅ Tamamlandı!")
    st.success("✅ Model eğitildi ve kaydedildi!")
