"""
onnx_export.py - ONNX Model Dönüştürme
=========================================
Eğitilmiş PyTorch modelini ONNX formatına dönüştürür.

ONNX (Open Neural Network Exchange) Nedir?
    - Farklı framework'ler arası model taşınabilirliği sağlar
    - PyTorch'ta eğitilen model, TensorFlow/ONNX Runtime'da çalıştırılabilir
    - Edge cihazlar ve üretim ortamları için optimize edilebilir

Kullanılan PyTorch Kavramları:
    - torch.onnx.export(): Modeli ONNX formatına çevirir
    - Tracing: Örnek bir girişle modelin hesaplama grafiğini kaydeder
    - dynamic_axes: Değişken batch boyutu desteği
"""

import torch
import torch.nn as nn
import os
from typing import Optional, Tuple


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device("cpu"),
    dynamic_batch: bool = True
) -> str:
    """
    PyTorch modelini ONNX formatına dönüştürür.

    Args:
        model: Eğitilmiş PyTorch modeli
        save_path: ONNX dosyasının kaydedileceği yol
        input_shape: Örnek giriş boyutu (batch, channels, height, width)
        device: Cihaz
        dynamic_batch: Değişken batch boyutu desteği

    Returns:
        str: Kaydedilen dosya yolu
    """
    model = model.to(device)
    model.eval()

    # Örnek giriş tensörü (tracing için gerekli)
    dummy_input = torch.randn(*input_shape).to(device)

    # Dinamik eksenler: batch boyutu değişebilir
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }

    # Dizini oluştur
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # ONNX'e dönüştür
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,         # Ağırlıkları dahil et
        opset_version=11,           # ONNX operatör seti versiyonu
        do_constant_folding=True,   # Sabit ifadeleri optimize et
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )

    # Dosya boyutunu kontrol et
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"ONNX modeli kaydedildi: {save_path} ({file_size_mb:.2f} MB)")

    return save_path


def verify_onnx_model(onnx_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> bool:
    """
    ONNX modelinin geçerliliğini kontrol eder ve test inference yapar.

    Args:
        onnx_path: ONNX dosya yolu
        input_shape: Test giriş boyutu

    Returns:
        bool: Model geçerli mi?
    """
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        # 1. ONNX model yapısını doğrula
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model yapısı geçerli")

        # 2. ONNX Runtime ile test inference
        session = ort.InferenceSession(onnx_path)
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        outputs = session.run(None, {"input": dummy_input})

        print(f"✓ ONNX Runtime inference başarılı")
        print(f"  Çıkış boyutu: {outputs[0].shape}")

        return True

    except Exception as e:
        print(f"✗ ONNX doğrulama hatası: {e}")
        return False
