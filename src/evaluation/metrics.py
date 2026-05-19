"""
metrics.py - Değerlendirme Metrikleri
=======================================
Model performansını ölçmek için accuracy, F1-score ve Confusion Matrix hesaplar.

Kullanılan Kavramlar:
    - sklearn.metrics.confusion_matrix: Sınıflandırma hatalarını matris olarak gösterir
    - sklearn.metrics.classification_report: Precision, Recall, F1 detaylı raporu
    - torch.max(): Tahmin edilen sınıfı belirler (en yüksek logit)
"""

import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score
)
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Tuple, List


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Batch bazında doğruluk (accuracy) hesaplar.

    Args:
        outputs: Model çıkışı (batch_size, num_classes)
        labels: Gerçek etiketler (batch_size,)

    Returns:
        float: Doğruluk oranı (0.0 - 1.0)
    """
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


@torch.no_grad()
def get_all_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tüm veri seti üzerinde tahmin yapar ve gerçek/tahmin etiketlerini toplar.
    Confusion Matrix ve Classification Report için gereklidir.

    Args:
        model: Eğitilmiş model
        data_loader: Test DataLoader'ı
        device: Cihaz (cpu/cuda/mps)

    Returns:
        tuple: (all_labels, all_predictions) numpy dizileri
    """
    model.eval()
    all_labels = []
    all_preds = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def compute_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str] = None
) -> np.ndarray:
    """
    Confusion Matrix hesaplar.

    Confusion Matrix Nedir?
        - Satırlar: Gerçek sınıflar
        - Sütunlar: Tahmin edilen sınıflar
        - Köşegen: Doğru tahminler
        - Köşegen dışı: Yanlış tahminler (hangi sınıflar karıştırılıyor?)

    Args:
        model: Eğitilmiş model
        data_loader: Test DataLoader'ı
        device: Cihaz
        class_names: Sınıf isimleri (opsiyonel, görselleştirme için)

    Returns:
        np.ndarray: Confusion matrix
    """
    all_labels, all_preds = get_all_predictions(model, data_loader, device)
    cm = confusion_matrix(all_labels, all_preds)
    return cm


def calculate_f1(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    average: str = "weighted"
) -> float:
    """
    F1-Score hesaplar.

    F1-Score Nedir?
        Precision ve Recall'un harmonik ortalaması.
        Dengesiz veri setlerinde accuracy'den daha güvenilir bir metriktir.

    Args:
        model: Eğitilmiş model
        data_loader: Test DataLoader'ı
        device: Cihaz
        average: "weighted" (sınıf dengesine göre ağırlıklı), "macro", "micro"

    Returns:
        float: F1-Score
    """
    all_labels, all_preds = get_all_predictions(model, data_loader, device)
    return f1_score(all_labels, all_preds, average=average)


def classification_summary(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str] = None
) -> str:
    """
    Detaylı sınıflandırma raporu üretir (Precision, Recall, F1 her sınıf için).

    Args:
        model: Eğitilmiş model
        data_loader: Test DataLoader'ı
        device: Cihaz
        class_names: Sınıf isimleri

    Returns:
        str: Formatlı sınıflandırma raporu
    """
    all_labels, all_preds = get_all_predictions(model, data_loader, device)
    return classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )
