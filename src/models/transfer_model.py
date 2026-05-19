"""
transfer_model.py - ResNet18 Transfer Learning
=================================================
Önceden ImageNet üzerinde eğitilmiş ResNet18 modelini alıp,
Intel Image Classification veri setine göre ince ayar (fine-tune) yapıyoruz.

Transfer Learning Neden Kullanılır?
    - ImageNet'te 1.2 milyon görüntü ile eğitilmiş ağırlıklar zaten
      kenar, doku, şekil gibi genel özellikleri öğrenmiştir
    - Bu bilgiyi sıfırdan öğrenmek yerine aktararak (transfer) daha az
      veri ve daha az eğitim süresiyle daha iyi sonuç elde edebiliriz
    - Özellikle küçük-orta boyutlu veri setlerinde çok etkilidir

Kullanılan PyTorch Kavramları:
    - torchvision.models.resnet18(weights=...): Pretrained model yükleme
    - param.requires_grad = False: Katmanları dondurma (freeze)
    - model.fc = nn.Linear(...): Son sınıflandırma katmanını değiştirme
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class TransferResNet18(nn.Module):
    """
    ResNet18 tabanlı Transfer Learning modeli.

    Strateji:
        1. ImageNet ağırlıklarıyla ResNet18'i yükle
        2. Tüm evrişim (feature extractor) katmanlarını dondur
        3. Son tam bağlı (fc) katmanı 6 sınıfa göre yeniden tanımla
        4. Sadece yeni fc katmanını eğit (çok daha hızlı)

    Fine-Tuning Opsiyonu:
        - unfreeze_last_n parametresi ile son N katmanı açarak
          daha derin bir ince ayar yapılabilir
    """

    def __init__(
        self,
        num_classes: int = 6,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0
    ):
        """
        Args:
            num_classes: Sınıf sayısı (6)
            dropout_rate: Dropout oranı
            freeze_backbone: Evrişim katmanlarını dondur (True = sadece fc eğitilir)
            unfreeze_last_n: Son N ResNet bloğunu aç (0 = hepsi dondurulmuş)
        """
        super(TransferResNet18, self).__init__()

        # ===== PRETRAINED MODEL YÜKLEME =====
        # weights="IMAGENET1K_V1": ImageNet üzerinde eğitilmiş ağırlıklar
        # Bu ağırlıklar kenar, doku, şekil gibi genel özelllikleri zaten öğrenmiştir
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # ===== KATMANLARI DONDURMA (FREEZE) =====
        # requires_grad = False: Bu parametreler eğitim sırasında güncellenmez
        # Böylece önceden öğrenilmiş özellikler korunur
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ===== SON N BLOĞU AÇMA (UNFREEZE) =====
        # İsteğe bağlı: Son birkaç bloğu da eğitilebilir yap
        # Daha derin fine-tuning için (ama daha fazla hesaplama gerektirir)
        if unfreeze_last_n > 0:
            # ResNet18'in katman grupları: layer1, layer2, layer3, layer4
            layers = [self.backbone.layer1, self.backbone.layer2,
                      self.backbone.layer3, self.backbone.layer4]
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # ===== SINIFLANDIRMA KATMANINI DEĞİŞTİRME =====
        # ResNet18'in orijinal fc katmanı: Linear(512, 1000) (ImageNet 1000 sınıf)
        # Biz bunu 6 sınıfa göre yeniden tanımlıyoruz
        in_features = self.backbone.fc.in_features  # 512

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri yayılım. ResNet18 backbone'u + yeni fc katmanı.

        Args:
            x: Giriş tensörü (batch_size, 3, 224, 224)

        Returns:
            Çıkış logit'leri (batch_size, num_classes)
        """
        return self.backbone(x)

    def count_parameters(self) -> dict:
        """
        Toplam ve eğitilebilir parametre sayısını hesaplar.
        Transfer Learning'in avantajını görmek için önemli:
        Custom CNN'deki eğitilebilir parametre sayısı ile karşılaştırılacak.

        Returns:
            dict: {"total": toplam, "trainable": eğitilebilir, "non_trainable": donmuş}
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable
        }

    def get_last_conv_layer(self) -> nn.Module:
        """
        Son evrişim katmanını döndürür.
        Model görselleştirme ve analiz için kullanılabilir.
        ResNet18'de bu layer4'ün son Conv2d katmanıdır.

        Returns:
            nn.Conv2d: Son evrişim katmanı
        """
        return self.backbone.layer4[-1].conv2

    def unfreeze_all(self):
        """
        Tüm katmanları eğitilebilir yapar.
        İlk birkaç epoch dondurulmuş olarak eğittikten sonra,
        tüm katmanları açarak tam fine-tuning yapılabilir.
        """
        for param in self.parameters():
            param.requires_grad = True
        print("Tüm katmanlar eğitim için açıldı (unfreeze).")
