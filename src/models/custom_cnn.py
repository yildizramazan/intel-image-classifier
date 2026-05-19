"""
custom_cnn.py - Sıfırdan CNN Mimarisi
========================================
PyTorch nn.Module kullanarak kendi evrişimli sinir ağımızı (CNN) tasarlıyoruz.

Kullanılan PyTorch Kavramları:
    - nn.Module: Tüm PyTorch modellerinin temel sınıfı.
      İki metot override edilir:
        * __init__: Katmanları tanımla
        * forward: Verinin katmanlardan nasıl geçeceğini tanımla (ileri yayılım)

    - nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding):
      2D evrişim katmanı. Görüntüdeki kenar, doku, şekil gibi özellikleri tespit eder.
        * in_channels: Giriş kanal sayısı (RGB=3, önceki katmanın çıkışı)
        * out_channels: Üretilecek filtre/özellik haritası sayısı
        * kernel_size: Filtre boyutu (3x3 en yaygın)
        * padding: Kenar dolgusu ("same" boyutu korur)

    - nn.BatchNorm2d(num_features):
      Batch normalization. Her mini-batch için aktivasyonları normalize eder.
      Faydaları: Eğitimi stabilize eder, daha yüksek learning rate kullanılabilir.

    - nn.MaxPool2d(kernel_size):
      Havuzlama katmanı. Özellik haritasının boyutunu yarıya indirir.
      En büyük değeri seçerek en belirgin özellikleri korur.

    - nn.Dropout(p):
      Eğitim sırasında rastgele p oranında nöronu kapatır.
      Modelin tek bir nörona aşırı bağımlı olmasını engeller (overfitting önleme).

    - nn.Sequential:
      Katmanları sıralı bir şekilde gruplar. forward() içinde daha temiz kod sağlar.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CustomCNN(nn.Module):
    """
    Intel Image Classification için sıfırdan tasarlanmış CNN mimarisi.

    Mimari Yapı (varsayılan 4 blok):
        Blok 1: Conv2d(3, 32)   -> BatchNorm -> ReLU -> MaxPool  -> 112x112
        Blok 2: Conv2d(32, 64)  -> BatchNorm -> ReLU -> MaxPool  -> 56x56
        Blok 3: Conv2d(64, 128) -> BatchNorm -> ReLU -> MaxPool  -> 28x28
        Blok 4: Conv2d(128, 256)-> BatchNorm -> ReLU -> MaxPool  -> 14x14

        Flatten -> Linear(256*14*14, 512) -> ReLU -> Dropout -> Linear(512, 6)

    Hiperparametre Optimizasyonu (Optuna) ile ayarlanabilecek parametreler:
        - num_blocks: Evrişim bloğu sayısı (3-5)
        - base_filters: İlk bloktaki filtre sayısı (16, 32, 64)
        - dropout_rate: Dropout oranı (0.1 - 0.5)
    """

    def __init__(
        self,
        num_classes: int = 6,
        num_blocks: int = 4,
        base_filters: int = 32,
        dropout_rate: float = 0.3,
        input_size: int = 224
    ):
        """
        Args:
            num_classes: Sınıf sayısı (Intel Image = 6)
            num_blocks: Evrişim bloğu sayısı (Conv+BN+ReLU+Pool)
            base_filters: İlk katmandaki filtre sayısı, her blokta 2x artar
            dropout_rate: Dropout oranı (0.0 - 1.0 arası)
            input_size: Giriş görüntü boyutu (224x224)
        """
        super(CustomCNN, self).__init__()

        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate

        # ===== FEATURE EXTRACTOR (Özellik Çıkarıcı) =====
        # Evrişim blokları: Görüntüdeki desenleri öğrenir
        layers = []
        in_channels = 3  # RGB görüntü: 3 kanal

        for i in range(num_blocks):
            out_channels = base_filters * (2 ** i)  # 32, 64, 128, 256, ...

            layers.extend([
                # Conv2d: 3x3 filtre ile özellik çıkarma
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1  # padding=1 -> boyut korunur (same padding)
                ),
                # BatchNorm: Aktivasyonları normalize et
                nn.BatchNorm2d(out_channels),
                # ReLU: Negatif değerleri sıfırla, doğrusal olmayan ilişkileri öğren
                nn.ReLU(inplace=True),
                # MaxPool: Boyutu yarıya indir, en belirgin özellikleri koru
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])

            in_channels = out_channels  # Sonraki katmanın girişi, bu katmanın çıkışı

        self.features = nn.Sequential(*layers)

        # ===== CLASSIFIER (Sınıflandırıcı) =====
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B, C, 1, 1) -> (B, C)
            nn.Linear(in_channels, 512),           # Tam bağlı katman
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),            # Overfitting önleme
            nn.Linear(512, 128),                    # İkinci tam bağlı katman
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),            # Çıkış katmanı (6 sınıf)
        )

        # Ağırlıkları başlat (weight initialization)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri yayılım (Forward Pass): Verinin ağın içinden nasıl geçeceğini tanımlar.

        Bu metot eğitim sırasında otomatik çağrılır (model(x) yapıldığında).
        Geri yayılım (backward pass) PyTorch'un autograd sistemi tarafından
        otomatik olarak hesaplanır.

        Args:
            x: Giriş tensörü, boyut: (batch_size, 3, 224, 224)

        Returns:
            Çıkış tensörü, boyut: (batch_size, num_classes)
            NOT: CrossEntropyLoss kendi içinde softmax uygular,
            bu yüzden çıkışa softmax UYGULAMIYORUZ (ham logit'ler).
        """
        x = self.features(x)    # Evrişim blokları: özellik çıkarma
        x = self.avgpool(x)     # Özellik haritalarının ortalamasını al
        x = self.classifier(x)  # Sınıflandırma katmanları
        return x

    def _initialize_weights(self):
        """
        Kaiming (He) ağırlık başlatma.
        ReLU aktivasyon fonksiyonu ile kullanıldığında en iyi sonucu verir.
        Conv ve Linear katmanlardaki ağırlıkları akıllıca başlatır.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> dict:
        """
        Modeldeki toplam ve eğitilebilir parametre sayısını hesaplar.
        Model karşılaştırma sayfasında kullanılacak.

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
        Son evrişim katmanını döndürür.
        Model görselleştirme ve analiz için kullanılabilir.

        Returns:
            nn.Conv2d: Son evrişim katmanı
        """
        for layer in reversed(list(self.features.modules())):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise ValueError("Model içinde Conv2d katmanı bulunamadı!")
