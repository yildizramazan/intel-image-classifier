"""
hpo.py - Optuna ile Hiperparametre Optimizasyonu
===================================================
Optuna kullanarak modelin en iyi hiperparametre kombinasyonunu otomatik buluyoruz.

Kullanılan Optuna Kavramları:
    - optuna.create_study(): Optimizasyon çalışması oluşturma
    - study.optimize(): Denemeleri çalıştırma
    - trial.suggest_float(): Sürekli değer arama alanı (learning rate vb.)
    - trial.suggest_int(): Tam sayı arama alanı (katman sayısı vb.)
    - trial.suggest_categorical(): Kategorik seçenekler (optimizer tipi vb.)
    - MedianPruner: Medyan bazlı erken durdurma (kötü denemeleri buda)
    - trial.report(): Ara sonuçları Optuna'ya raporlama
    - trial.should_prune(): Budama kararı kontrolü
"""

import optuna
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from src.models.custom_cnn import CustomCNN
from src.models.transfer_model import TransferResNet18
from src.training.optimizer_factory import create_optimizer, create_scheduler
from src.training.trainer import Trainer


def objective(
    trial: optuna.Trial,
    train_loader,
    val_loader,
    model_type: str,
    device: torch.device,
    num_epochs: int = 15,
    num_classes: int = 6
) -> float:
    """
    Optuna'nın her denemede (trial) çağırdığı amaç fonksiyonu.

    Bu fonksiyon:
        1. trial.suggest_* ile hiperparametre değerlerini seçer
        2. Seçilen parametrelerle model oluşturur
        3. Modeli eğitir
        4. Doğrulama doğruluğunu (val_accuracy) döndürür

    Optuna bu fonksiyonun döndürdüğü değeri maximize etmeye çalışır.

    Args:
        trial: Optuna trial nesnesi (hiperparametre önerileri için)
        train_loader: Eğitim DataLoader'ı
        val_loader: Doğrulama DataLoader'ı
        model_type: "custom_cnn" veya "resnet18"
        device: Eğitim cihazı
        num_epochs: Deneme başına epoch sayısı
        num_classes: Sınıf sayısı

    Returns:
        float: Doğrulama doğruluğu (maximize edilecek)
    """

    # ===== 1. HİPERPARAMETRE ÖNERİLERİ =====

    # Öğrenme oranı: 1e-5 ile 1e-2 arası, logaritmik ölçekte
    # log=True: Küçük değerlerin de yeterince örneklenmesini sağlar
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Optimizer tipi: Adam veya SGD
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # Dropout oranı: 0.1 ile 0.5 arası
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)

    # ===== 2. MODEL OLUŞTURMA =====
    if model_type == "custom_cnn":
        # Custom CNN'e özel hiperparametreler
        num_blocks = trial.suggest_int("num_blocks", 3, 5)
        base_filters = trial.suggest_categorical("base_filters", [16, 32, 64])

        model = CustomCNN(
            num_classes=num_classes,
            num_blocks=num_blocks,
            base_filters=base_filters,
            dropout_rate=dropout_rate
        )
    elif model_type == "resnet18":
        # ResNet18 Transfer Learning
        unfreeze_last_n = trial.suggest_int("unfreeze_last_n", 0, 2)

        model = TransferResNet18(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=True,
            unfreeze_last_n=unfreeze_last_n
        )
    else:
        raise ValueError(f"Bilinmeyen model tipi: {model_type}")

    # ===== 3. EĞİTİM =====
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, optimizer_name, learning_rate)
    scheduler = create_scheduler(optimizer)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir="models"
    )

    # trial parametresini Trainer'a geçir -> pruning desteği
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=5,
        model_name=f"trial_{trial.number}",
        trial=trial,
        verbose=False  # HPO sırasında sessiz mod
    )

    # En iyi validation accuracy'yi döndür
    best_val_acc = max(history["val_acc"])
    return best_val_acc


def run_hpo_study(
    train_loader,
    val_loader,
    model_type: str = "custom_cnn",
    device: torch.device = torch.device("cpu"),
    n_trials: int = 20,
    num_epochs: int = 15,
    study_name: str = "intel_image_hpo"
) -> Dict[str, Any]:
    """
    Tam bir Optuna hiperparametre optimizasyonu çalışması yürütür.

    Args:
        train_loader: Eğitim DataLoader'ı
        val_loader: Doğrulama DataLoader'ı
        model_type: "custom_cnn" veya "resnet18"
        device: Eğitim cihazı
        n_trials: Toplam deneme sayısı
        num_epochs: Deneme başına epoch sayısı
        study_name: Çalışma adı

    Returns:
        dict: {
            "best_params": En iyi parametreler,
            "best_value": En iyi accuracy,
            "study": Optuna study nesnesi (görselleştirme için)
        }
    """
    # MedianPruner: Ara sonuçları diğer denemelerin medyanı ile karşılaştırır
    # Medyanın altında kalan denemeler erken durdurulur (budanır)
    pruner = MedianPruner(
        n_startup_trials=5,    # İlk 5 deneme budanmaz (referans oluşturmak için)
        n_warmup_steps=3       # İlk 3 epoch budanmaz (modelin oturması için)
    )

    # Study oluştur: direction="maximize" -> val_accuracy'yi en yüksek yapmaya çalış
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=pruner
    )

    print(f"\n{'='*60}")
    print(f"Optuna HPO Başlatılıyor")
    print(f"Model: {model_type} | Deneme Sayısı: {n_trials} | Epoch/Deneme: {num_epochs}")
    print(f"{'='*60}\n")

    # Optimizasyonu çalıştır
    study.optimize(
        lambda trial: objective(
            trial, train_loader, val_loader,
            model_type, device, num_epochs
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Sonuçları yazdır
    print(f"\n{'='*60}")
    print(f"HPO Tamamlandı!")
    print(f"En iyi Accuracy: {study.best_value:.4f}")
    print(f"En iyi Parametreler:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Tamamlanan denemeler: {len(study.trials)}")
    print(f"Budanan denemeler: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"{'='*60}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study
    }
