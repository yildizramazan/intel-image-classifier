# Staj Haftalık Rapor, 14. Hafta

Stajyer: Ramazan YILDIZ  
Tarih Aralığı: [Başlangıç Tarihi] – [Bitiş Tarihi]  
Proje: CNN Görüntü Sınıflandırma, Intel Image Classification Veri Seti, Proje Altyapısı, Veri Pipeline, Model Tasarımı, Eğitim Döngüsü ve Metrikler

---

## 60. Gün: Proje Kurulumu, Ortam Hazırlığı ve Veri Seti

Haftaya uçtan uca bir CNN görüntü sınıflandırma (image classification) sistemi için proje altyapısını kurarak başladım. Veri yükleme ve dönüşümler için `src/data/`, CNN ve aktarım öğrenimi (transfer learning) tanımları için `src/models/`, eğitim döngüsü, optimizer ve HPO için `src/training/`, performans metrikleri için `src/evaluation/`, ONNX dönüştürme için `src/export/`, Streamlit çok sayfalı web uygulaması için `app/` ve Colab eğitim defteri için `notebooks/` içeren modüler bir klasör yapısı oluşturdum. torch, torchvision, optuna, streamlit, onnx, scikit-learn ve diğer bağımlılıkları içeren `requirements.txt` hazırladım.

Ardından Kaggle'dan 6 sınıf (buildings, forest, glacier, mountain, sea, street) içeren ~14K eğitim ve ~3K test görüntüsünden oluşan Intel Image Classification veri setini indirdim. Verileri `data/seg_train/seg_train/` ve `data/seg_test/seg_test/` dizinlerine çıkardım, klasör yapısını ve sınıf dağılımını doğruladım:

```
intel-image-classifier/
├── src/
│   ├── data/          # transforms.py, dataset.py, dataloader.py
│   ├── models/        # custom_cnn.py, transfer_model.py
│   ├── training/      # trainer.py, optimizer_factory.py, hpo.py
│   ├── evaluation/    # metrics.py
│   └── export/        # onnx_export.py
├── app/               # Streamlit multi-page application
├── notebooks/         # Colab training notebook
├── models/            # Trained model files (.pt, .onnx)
└── requirements.txt
```

---

## 61. Gün: Veri Pipeline, Dönüşümler, Dataset ve DataLoader

`src/data/transforms.py` ile başlayarak eksiksiz veri pipeline'ını oluşturdum. `get_train_transforms()` fonksiyonu kırpma payı için `Resize(256)`, her epoch'ta rastgele bölge seçimi için `RandomCrop(224)`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(15)`, renk artırımı (augmentation) için `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)`, PIL'den tensöre dönüşüm ve pikselleri [0,255]'ten [0.0,1.0]'a ölçekleme için `ToTensor()` ve ImageNet istatistikleri ile `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` zincirler. Test verisi artırım almaması gerektiğinden yalnızca Resize, ToTensor ve Normalize içeren `get_test_transforms()` ile Streamlit görselleştirmesi için `denormalize()` yardımcı fonksiyonunu da yazdım.

`src/data/dataset.py` dosyasında her alt dizini sınıf etiketi olarak ele alan `torchvision.datasets.ImageFolder` kullanan `load_dataset()`, EDA için `Counter(dataset.targets)` ile `get_class_distribution()` ve sınıf başına N örnek alan `get_sample_images()` fonksiyonlarını uyguladım. `src/data/dataloader.py` dosyasında tekrarlanabilirlik için tohumlanmış üreteç (generator) ile `torch.utils.data.random_split()` aracılığıyla %80/20 eğitim/doğrulama bölümlemesi yapan, eğitim için `shuffle=True`, paralel yükleme için `num_workers=2`, hızlı CPU-GPU aktarımı için `pin_memory=True` ve BatchNorm kararlılığı için `drop_last=True` ile `batch_size=32` DataLoader'lar oluşturan `create_dataloaders()` fonksiyonunu yazdım:

```python
def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
```

---

## 62. Gün: Sıfırdan Custom CNN Mimarisi Tasarımı

`src/models/custom_cnn.py` dosyasında `nn.Module`'u genişleten, yapılandırılabilir bloklardan oluşan özellik çıkarıcı (feature extractor) ile bir `CustomCNN` sınıfı tasarladım. Her blok aynı dolgu (same padding) evrişim için `Conv2d(in, out, kernel_size=3, padding=1)`, aktivasyon normalizasyonu için `BatchNorm2d`, doğrusal olmayanlık (non-linearity) için `ReLU(inplace=True)` ve her aşamada uzamsal boyutları yarılayan `MaxPool2d(2,2)` içerir (224->112->56->28->14). Filtre sayısı blok başına iki katına çıkar (32->64->128->256) ve tümü `self.features = nn.Sequential(*layers)` içinde gruplanır.

Sınıflandırıcı (classifier) başlığı (B, 256, 14, 14)'ten (B, 256*14*14)'e yeniden şekillendirmek için `Flatten()`, ardından `Linear(flatten_size, 512)`, `ReLU`, `Dropout(p=dropout_rate)`, `Linear(512, 128)` ve `CrossEntropyLoss` dahili olarak softmax uyguladığı için softmax olmadan 6 sınıflık çıktı için `Linear(128, 6)` kullanır. Conv2d ve Linear katmanları için `nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')` ile Kaiming He başlatma uygulayan `_initialize_weights()` ve toplam ile eğitilebilir parametre sayısını raporlayan `count_parameters()` fonksiyonlarını uyguladım:

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes=6, num_blocks=4, base_filters=32, dropout_rate=0.3):
        super(CustomCNN, self).__init__()
        layers = []
        in_channels = 3
        for i in range(num_blocks):
            out_channels = base_filters * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 512), nn.ReLU(inplace=True), nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )
```

---

## 63. Gün: ResNet18 ile Transfer Learning

`src/models/transfer_model.py` dosyasında kenarları, dokuları ve şekilleri zaten tanıyan ImageNet ağırlıklarını yükleyen `models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` ile `TransferResNet18` sınıfını uyguladım. Önceden öğrenilmiş özellikleri korumak için `param.requires_grad = False` ile tüm omurga (backbone) parametrelerini dondurdum, ardından daha derin ince ayar (fine-tuning) için isteğe bağlı olarak `layers[-unfreeze_last_n:]` ile son N ResNet bloğunu açtım. Orijinal `Linear(512, 1000)` sınıflandırma başlığını 6 sınıfımız için `Dropout(p=dropout_rate)`, `Linear(512, 256)`, `ReLU`, `Dropout(p=dropout_rate/2)` ve `Linear(256, 6)` ile değiştirdim.

Custom CNN (tümü eğitilebilir) ile dondurulmuş ResNet18 (yalnızca yeni fc katmanı eğitilebilir) arasında eğitilebilir parametre karşılaştırması için `count_parameters()`, ilk dondurulmuş eğitimden sonra tam ince ayar için `unfreeze_all()` ve model analizi için `layer4[-1].conv2` döndüren `get_last_conv_layer()` fonksiyonlarını da uyguladım:

```python
class TransferResNet18(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3, freeze_backbone=True, unfreeze_last_n=0):
        super(TransferResNet18, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if unfreeze_last_n > 0:
            layers = [self.backbone.layer1, self.backbone.layer2,
                      self.backbone.layer3, self.backbone.layer4]
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_features, 256),
            nn.ReLU(inplace=True), nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )
```

---

## 64. Gün: Eğitim Döngüsü, Optimizer ve Değerlendirme Metrikleri

`src/training/trainer.py` dosyasında tam eğitim yaşam döngüsünü kapsayan bir `Trainer` sınıfı kurdum. `train_one_epoch()` metodu standart PyTorch döngüsünü izler: Dropout ve BatchNorm'u etkinleştirmek için `model.train()`, ardından her batch için gradyanları temizlemek için `optimizer.zero_grad()`, ileri yayılım (forward pass) için `outputs = model(inputs)`, geri yayılım (backpropagation) için `loss.backward()` ve ağırlık güncelleme için `optimizer.step()`. `validate()` metodu gradyan hesabı olmadan çıkarım için `model.eval()` ile `@torch.no_grad()` kullanır. `fit()` metodu geçmiş takibi, doğrulama doğruluğu iyileştiğinde `torch.save(model.state_dict(), path)` ile en iyi model kaydetme, N epoch iyileşme olmazsa erken durdurma (early stopping) ve `trial.report()` ile `trial.should_prune()` aracılığıyla Optuna entegrasyonu ile çoklu epoch eğitimini yönetir.

`src/training/optimizer_factory.py` dosyasında yalnızca eğitilebilir parametreleri optimize etmek için `filter(lambda p: p.requires_grad, model.parameters())` ile Adam ve SGD destekleyen `create_optimizer()` ve her 5 epoch'ta öğrenme oranını yarıya indiren `StepLR(step_size=5, gamma=0.5)` döndüren `create_scheduler()` fonksiyonlarını yazdım. `src/evaluation/metrics.py` dosyasında `torch.max()` kullanan `calculate_accuracy()`, test tahminlerini toplayan `get_all_predictions()`, sklearn ile `compute_confusion_matrix()`, ağırlıklı ortalama ile `calculate_f1()` ve sınıf bazında Precision, Recall ve F1 için `classification_report` kullanan `classification_summary()` fonksiyonlarını uyguladım:

```python
class Trainer:
    def train_one_epoch(self, train_loader):
        self.model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def fit(self, train_loader, val_loader, num_epochs=20, early_stopping_patience=5):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            if val_acc > best_val_acc:
                torch.save(self.model.state_dict(), checkpoint_path)
```

---

## Haftalık Değerlendirme

Bu hafta Intel Image Classification veri seti (6 sınıf, ~17K görüntü) ile uçtan uca bir CNN görüntü sınıflandırma projesinin ilk yarısını oluşturdum. Modüler proje yapısını kurdum, artırım dönüşümleri ve DataLoader oluşturma ile eksiksiz veri pipeline'ını uyguladım, yapılandırılabilir Conv+BN+ReLU+Pool blokları ve Kaiming başlatma ile sıfırdan CNN mimarisi tasarladım, omurga dondurma ve özel sınıflandırma başlığı ile ResNet18 aktarım öğrenimi modeli uyguladım ve eğitim/doğrulama döngüleri, erken durdurma, kontrol noktası kaydetme, optimizer/zamanlayıcı fabrikası ve scikit-learn tabanlı değerlendirme metrikleri ile Trainer sınıfını kurdum.
