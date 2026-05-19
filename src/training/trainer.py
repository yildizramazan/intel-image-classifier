"""
trainer.py - Eğitim Döngüsü
==============================
PyTorch'un manuel eğitim döngüsünü (training loop) profesyonel bir
Trainer sınıfı içinde kapsüllüyoruz.

Kullanılan PyTorch Kavramları:
    - model.train(): Modeli eğitim moduna alır (Dropout ve BatchNorm aktif)
    - model.eval(): Modeli değerlendirme moduna alır (Dropout kapalı, BatchNorm sabit)
    - loss.backward(): Geri yayılım - gradyanları hesapla
    - optimizer.step(): Ağırlıkları gradyanlara göre güncelle
    - optimizer.zero_grad(): Gradyanları sıfırla (bir sonraki iterasyon için)
    - torch.no_grad(): Gradyan hesabını kapat (bellek ve hız tasarrufu)
    - torch.save(): Model ağırlıklarını diske kaydet (checkpoint)
    - torch.load(): Kaydedilmiş ağırlıkları yükle
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import time
import os


class Trainer:
    """
    PyTorch model eğitim ve doğrulama döngüsünü yöneten sınıf.

    Özellikler:
        - Train ve validation döngüsü
        - En iyi modeli otomatik kaydetme (best checkpoint)
        - Early stopping desteği
        - Eğitim geçmişi (history) takibi
        - Cihaz yönetimi (CPU/GPU/MPS)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[object] = None,
        checkpoint_dir: str = "models"
    ):
        """
        Args:
            model: Eğitilecek PyTorch modeli (nn.Module)
            criterion: Kayıp fonksiyonu (nn.CrossEntropyLoss)
            optimizer: Optimizer (Adam veya SGD)
            device: Eğitim cihazı (cpu, cuda, mps)
            scheduler: Learning rate scheduler (opsiyonel)
            checkpoint_dir: Model checkpoint'larının kaydedileceği dizin
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir

        # Checkpoint dizinini oluştur
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Eğitim geçmişi: Her epoch'taki loss ve accuracy değerleri
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_one_epoch(self, train_loader: DataLoader) -> tuple:
        """
        Bir epoch'luk eğitim döngüsü.

        Adımlar (her batch için):
            1. model.train()         -> Eğitim modunu aktifleştir
            2. inputs, labels        -> Batch'i cihaza gönder
            3. optimizer.zero_grad() -> Önceki gradyanları temizle
            4. outputs = model(inputs) -> İleri yayılım (forward pass)
            5. loss = criterion(outputs, labels) -> Kayıp hesapla
            6. loss.backward()       -> Geri yayılım (gradyanları hesapla)
            7. optimizer.step()      -> Ağırlıkları güncelle

        Args:
            train_loader: Eğitim DataLoader'ı

        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()  # Dropout ve BatchNorm eğitim modunda

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Veriyi eğitim cihazına gönder (CPU -> GPU)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # 1. Gradyanları sıfırla
            # Her batch'te gradyanlar birikir, bu yüzden önce temizlenmeli
            self.optimizer.zero_grad()

            # 2. İleri yayılım (Forward Pass)
            outputs = self.model(inputs)

            # 3. Kayıp hesapla
            # CrossEntropyLoss: softmax + negative log likelihood
            loss = self.criterion(outputs, labels)

            # 4. Geri yayılım (Backward Pass)
            # Kaybın her ağırlığa göre kısmi türevini (gradyan) hesapla
            loss.backward()

            # 5. Ağırlıkları güncelle
            # optimizer.step() gradyanlara göre parametreleri günceller
            self.optimizer.step()

            # İstatistikleri topla
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)  # En yüksek olasılıklı sınıf
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()  # Gradyan hesabını kapat: bellek ve hız tasarrufu
    def validate(self, val_loader: DataLoader) -> tuple:
        """
        Doğrulama (Validation) döngüsü.

        model.eval() ile:
            - Dropout katmanları devre dışı bırakılır (tüm nöronlar aktif)
            - BatchNorm eğitim istatistikleri yerine toplanan istatistikleri kullanır

        torch.no_grad() ile:
            - Gradyan hesabı yapılmaz (sadece ileri yayılım)
            - Bellek kullanımı azalır, hız artar

        Args:
            val_loader: Doğrulama DataLoader'ı

        Returns:
            tuple: (val_loss, val_accuracy)
        """
        self.model.eval()  # Değerlendirme modu

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total

        return val_loss, val_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        early_stopping_patience: int = 5,
        model_name: str = "model",
        trial=None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Tam eğitim döngüsü: Birden fazla epoch boyunca eğit ve doğrula.

        Özellikler:
            - Her epoch sonunda train/val loss ve accuracy kaydeder
            - En iyi val_accuracy'ye sahip modeli otomatik kaydeder (best checkpoint)
            - Early stopping: N epoch boyunca iyileşme olmazsa durur
            - Optuna entegrasyonu: trial.report() ve trial.should_prune()

        Args:
            train_loader: Eğitim DataLoader'ı
            val_loader: Doğrulama DataLoader'ı
            num_epochs: Toplam epoch sayısı
            early_stopping_patience: Kaç epoch iyileşme olmazsa dur
            model_name: Checkpoint dosya adı
            trial: Optuna trial nesnesi (HPO sırasında)
            verbose: Epoch bilgilerini yazdır

        Returns:
            dict: Eğitim geçmişi {"train_loss": [...], "val_loss": [...], ...}
        """
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()

        # Geçmişi sıfırla
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Eğitim
            train_loss, train_acc = self.train_one_epoch(train_loader)

            # Doğrulama
            val_loss, val_acc = self.validate(val_loader)

            # Learning rate scheduler adımı
            if self.scheduler is not None:
                self.scheduler.step()

            # Geçmişe kaydet
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            epoch_time = time.time() - epoch_start

            if verbose:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.6f} | Süre: {epoch_time:.1f}s"
                )

            # ===== EN İYİ MODELİ KAYDET =====
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"{model_name}_best.pt"
                )
                # state_dict: Modelin tüm ağırlıklarını ve bias'larını içeren sözlük
                torch.save(self.model.state_dict(), checkpoint_path)
                if verbose:
                    print(f"  ✓ En iyi model kaydedildi: {checkpoint_path} (Acc: {val_acc:.4f})")
            else:
                patience_counter += 1

            # ===== EARLY STOPPING =====
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\n⚠ Early Stopping: {early_stopping_patience} epoch boyunca iyileşme olmadı.")
                break

            # ===== OPTUNA ENTEGRASYONU =====
            if trial is not None:
                import optuna
                # Ara sonucu Optuna'ya raporla (pruning için)
                trial.report(val_acc, epoch)
                # Optuna bu denemeyi budamalı mı?
                if trial.should_prune():
                    raise optuna.TrialPruned()

        total_time = time.time() - start_time

        if verbose:
            print(f"\nEğitim tamamlandı! Toplam süre: {total_time:.1f}s")
            print(f"En iyi Validation Accuracy: {best_val_acc:.4f}")

        return self.history

    def load_best_model(self, model_name: str = "model"):
        """
        Kaydedilmiş en iyi modeli yükler.

        torch.load() kullanarak state_dict'i diskten okur,
        model.load_state_dict() ile ağırlıkları modele yükler.

        Args:
            model_name: Checkpoint dosya adı
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{model_name}_best.pt"
        )
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.eval()
        print(f"En iyi model yüklendi: {checkpoint_path}")
