"""
4_Prediction.py - Tahmin Sayfası
"""

import streamlit as st
import sys, os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.transforms import get_test_transforms, denormalize
from src.data.dataset import CLASS_NAMES
from src.models.custom_cnn import CustomCNN
from src.models.transfer_model import TransferResNet18

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Görüntü Tahmini")
st.markdown("Bir görüntü yükleyin ve model tahminini görün.")
st.divider()

# Model seçimi
model_type = st.sidebar.selectbox("Model Seçin", ["custom_cnn", "resnet18"])
model_path = st.sidebar.text_input("Model Dosyası", f"models/{model_type}_best.pt")

# Resim yükleme
uploaded_file = st.file_uploader(
    "📷 Bir görüntü yükleyin",
    type=["jpg", "jpeg", "png"],
    help="Buildings, Forest, Glacier, Mountain, Sea veya Street görüntüsü"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Yüklenen Görüntü", use_container_width=True)

    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()

    # Model yükle
    device = torch.device("cpu")

    if model_type == "custom_cnn":
        model = CustomCNN(num_classes=6)
    else:
        model = TransferResNet18(num_classes=6)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Görüntüyü dönüştür
    transform = get_test_transforms()
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    # Tahmin
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx].item()

    with col2:
        st.markdown("### 📊 Tahmin Sonucu")
        st.metric("Tahmin", predicted_class.capitalize(), delta=f"{confidence:.1%} güven")

        # Tüm sınıf olasılıkları
        st.markdown("**Sınıf Olasılıkları:**")
        for i, class_name in enumerate(CLASS_NAMES):
            prob = probabilities[i].item()
            st.progress(prob, text=f"{class_name}: {prob:.1%}")
