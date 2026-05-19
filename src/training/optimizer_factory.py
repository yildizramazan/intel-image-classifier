"""
optimizer_factory.py - Optimizer Oluşturma Fonksiyonu
=======================================================
Hiperparametre optimizasyonunda farklı optimizer'ları kolayca
değiştirmek için factory pattern kullanıyoruz.

Kullanılan PyTorch Kavramları:
    - torch.optim.Adam: Adaptif öğrenme oranı kullanan optimizer.
      Momentum ve RMSProp'un avantajlarını birleştirir.
      Çoğu durumda iyi bir varsayılan seçimdir.

    - torch.optim.SGD: Stokastik Gradyan İnişi. Daha basit ama
      momentum ile birlikte kullanıldığında bazı durumlarda Adam'dan
      daha iyi genelleme yapabilir.

    - torch.optim.lr_scheduler.StepLR: Belirli epoch aralıklarında
      learning rate'i gamma oranında azaltır. Eğitimin ilerleyen
      aşamalarında daha ince ayar yapılmasını sağlar.
"""

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from typing import Tuple, Optional


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "Adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9
) -> optim.Optimizer:
    """
    Belirtilen tipte optimizer oluşturur.

    Optuna ile optimize edilecek parametreler:
        - optimizer_name: "Adam" veya "SGD"
        - learning_rate: 1e-5 ile 1e-2 arası (log ölçek)
        - weight_decay: L2 regularization (overfitting önleme)

    Args:
        model: Eğitilecek model
        optimizer_name: Optimizer tipi ("Adam" veya "SGD")
        learning_rate: Öğrenme oranı
        weight_decay: L2 regularization katsayısı
        momentum: SGD momentum değeri (sadece SGD için)

    Returns:
        Optimizer nesnesi
    """
    # Sadece eğitilebilir (requires_grad=True) parametreleri optimize et
    # Transfer Learning'de dondurulmuş katmanlar bu sayede atlanır
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == "Adam":
        return optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        return optim.SGD(
            trainable_params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Bilinmeyen optimizer: {optimizer_name}. 'Adam' veya 'SGD' kullanın.")


def create_scheduler(
    optimizer: optim.Optimizer,
    step_size: int = 5,
    gamma: float = 0.5
) -> StepLR:
    """
    Learning rate scheduler oluşturur.

    StepLR: Her 'step_size' epoch'ta learning rate'i gamma ile çarpar.
    Örnek: lr=0.001, step_size=5, gamma=0.5
        Epoch 1-5:  lr = 0.001
        Epoch 6-10: lr = 0.0005
        Epoch 11-15: lr = 0.00025

    Args:
        optimizer: Scheduler'ın bağlı olacağı optimizer
        step_size: Kaç epoch'ta bir lr düşürülecek
        gamma: lr çarpanı (0 < gamma < 1)

    Returns:
        StepLR scheduler nesnesi
    """
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
