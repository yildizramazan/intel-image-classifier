# CNN Image Classification Project — Detaylı 10 Günlük Plan

---

## HAFTA 1 (Gün 1-5): Proje Altyapısı, Veri İşleme ve Model Tasarımı

---

### 📅 Gün 1: Proje Kurulumu, Ortam Hazırlığı ve Veri Seti

**Sabah — Proje İskeleti:**
- [ ] Modüler klasör yapısını oluştur:
  - `src/data/` → veri yükleme ve dönüştürme modülleri
  - `src/models/` → CNN ve Transfer Learning model tanımları
  - `src/training/` → eğitim döngüsü, optimizer ve HPO
  - `src/evaluation/` → performans metrikleri
  - `src/export/` → ONNX model dönüştürme
  - `app/` → Streamlit çok sayfalı web uygulaması
  - `notebooks/` → Colab eğitim notebook'u
  - `models/` → eğitilmiş model dosyaları (.pt, .onnx)
- [ ] `requirements.txt` oluştur: `torch`, `torchvision`, `optuna`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `streamlit`, `onnx`, `onnxruntime`, `opencv-python`, `Pillow`
- [ ] `.gitignore` oluştur: `data/`, `models/*.pt`, `models/*.onnx`, `__pycache__/`, `.venv/`
- [ ] Her modül için `__init__.py` dosyaları oluştur (public API tanımları)

**Öğleden Sonra — Veri Seti ve Git:**
- [ ] Kaggle'dan "Intel Image Classification" veri setini indir (https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- [ ] Zip dosyasını `data/` klasörüne çıkar
- [ ] Klasör yapısını doğrula:
  ```
  data/seg_train/seg_train/  → buildings/, forest/, glacier/, mountain/, sea/, street/
  data/seg_test/seg_test/    → aynı 6 sınıf
  ```
- [ ] Toplam görüntü sayısını ve sınıf başına dağılımı kontrol et (~14K train, ~3K test)
- [ ] `git init` → `git add .` → `git commit -m "Initial project structure"` → GitHub repo oluştur → `git push`

---

### 📅 Gün 2: Veri Pipeline — Transforms, Dataset ve DataLoader

**Sabah — `src/data/transforms.py`:**
- [ ] `get_train_transforms()` fonksiyonunu yaz/incele:
  - `transforms.Resize(256)` → görüntüyü 256×256'ya büyüt (kırpma payı bırakmak için)
  - `transforms.RandomCrop(224)` → rastgele 224×224 bölge kes (her epoch farklı bölge)
  - `transforms.RandomHorizontalFlip(p=0.5)` → %50 olasılıkla yatay çevirme
  - `transforms.RandomRotation(degrees=15)` → ±15 derece rastgele döndürme
  - `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)` → renk/parlaklık/kontrast değişimi
  - `transforms.ToTensor()` → PIL Image'ı PyTorch tensörüne çevir, piksel değerlerini [0,255] → [0.0,1.0] aralığına normalize et
  - `transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` → ImageNet ortalaması ve standart sapması ile kanal bazında standartlaştırma
  - `transforms.Compose([...])` → tüm dönüşümleri sıralı zincire bağla
- [ ] `get_test_transforms()` fonksiyonunu yaz/incele:
  - Sadece `Resize(224,224)` → `ToTensor()` → `Normalize()` (augmentation YOK, çünkü test verisinin orijinal halini değerlendirmek istiyoruz)
- [ ] `denormalize()` yardımcı fonksiyonunu yaz: normalize edilmiş tensörü tekrar [0,1] aralığına döndür (Streamlit'te görselleştirme için)

**Öğleden Sonra — `src/data/dataset.py` ve `src/data/dataloader.py`:**
- [ ] `load_dataset()` fonksiyonu — `torchvision.datasets.ImageFolder` kullanımı:
  - `ImageFolder(root=data_dir, transform=transform)` → her alt klasörü bir sınıf olarak algılar
  - `dataset.classes` → sınıf isimleri listesi (alfabetik sıralı)
  - `dataset.class_to_idx` → sınıf adı → indeks eşlemesi
  - `dataset[i]` → `(image_tensor, label)` çifti döndürür
- [ ] `get_class_distribution()` → `Counter(dataset.targets)` ile her sınıftaki örnek sayısını hesapla
- [ ] `get_sample_images()` → her sınıftan N adet örnek görüntü çek (EDA sayfası için)
- [ ] `create_dataloaders()` fonksiyonu:
  - `torch.utils.data.random_split(train_dataset, [train_size, val_size])` → eğitim setini %80 train / %20 validation olarak böl
  - `DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)` → eğitim loader
  - `shuffle=True` → her epoch'ta veri sırasını karıştır (ezberlemeyi önler)
  - `num_workers=2` → paralel veri yükleme (CPU çekirdekleri kullanarak)
  - `pin_memory=True` → CPU→GPU veri transferini hızlandır
  - `drop_last=True` → son eksik batch'i at (BatchNorm stabilitesi için)
- [ ] Terminal/Jupyter'da test: bir batch çek, `images.shape` → `(32, 3, 224, 224)`, `labels.shape` → `(32,)` doğrula

---

### 📅 Gün 3: Custom CNN Mimarisi (Sıfırdan Model Tasarımı)

**Sabah — `src/models/custom_cnn.py` Feature Extractor kısmı:**
- [ ] `class CustomCNN(nn.Module)` → `nn.Module` alt sınıfı oluştur
- [ ] `__init__()` içinde feature extractor bloklarını tanımla (her blok):
  - `nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)`:
    - 3×3 filtre görüntü üzerinde kayarak özellik haritaları (feature maps) üretir
    - `padding=1` → giriş ve çıkış boyutunu aynı tutar (same padding)
    - Filtre sayıları her blokta 2 katına çıkar: 32 → 64 → 128 → 256
  - `nn.BatchNorm2d(out_channels)`:
    - Her mini-batch'te aktivasyonları normalize eder
    - Eğitimi stabilize eder, daha yüksek learning rate kullanılabilir
    - `model.train()` modunda çalışma ortalamasını günceller, `model.eval()` modunda sabit ortalamaları kullanır
  - `nn.ReLU(inplace=True)`:
    - Aktivasyon fonksiyonu: negatif değerleri sıfırlar → `max(0, x)`
    - Doğrusal olmayan (non-linear) ilişkileri öğrenmeyi sağlar
  - `nn.MaxPool2d(kernel_size=2, stride=2)`:
    - 2×2 pencerede en büyük değeri seçer → boyutu yarılar (224→112→56→28→14)
    - En belirgin özellikleri korur, hesaplama yükünü azaltır
- [ ] `nn.Sequential(*layers)` ile tüm blokları sıralı gruba topla → `self.features`

**Öğleden Sonra — Classifier kısmı ve yardımcı metotlar:**
- [ ] `__init__()` içinde classifier (sınıflandırıcı) katmanlarını tanımla:
  - `nn.Flatten()` → (B, 256, 14, 14) → (B, 256×14×14) şeklinde düzleştir
  - `nn.Linear(flatten_size, 512)` → tam bağlı katman
  - `nn.ReLU()` → aktivasyon
  - `nn.Dropout(p=dropout_rate)` → eğitim sırasında rastgele nöronları kapat (overfitting önleme)
  - `nn.Linear(512, 128)` → ikinci tam bağlı katman
  - `nn.Linear(128, 6)` → çıkış katmanı (6 sınıf, softmax UYGULANMAZ çünkü CrossEntropyLoss kendi içinde uygular)
- [ ] `forward(self, x)` metodunu yaz: `x = self.features(x)` → `x = self.classifier(x)` → `return x`
- [ ] `_initialize_weights()` → Kaiming (He) başlatma:
  - `nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')` → Conv ve Linear katmanlar için
  - ReLU ile kullanıldığında en iyi sonucu veren başlatma stratejisi
- [ ] `count_parameters()` → toplam, eğitilebilir ve donmuş parametre sayısını hesapla
- [ ] Modeli oluştur, `print(model)` ile yapısını gör, parametre sayısını doğrula

---

### 📅 Gün 4: Transfer Learning — ResNet18 Modeli

**Sabah — `src/models/transfer_model.py`:**
- [ ] Transfer Learning kavramını öğren:
  - ImageNet'te 1.2M görüntü ile eğitilmiş ağırlıklar zaten kenar/doku/şekil tanıyor
  - Bu bilgiyi sıfırdan öğrenmek yerine aktararak (transfer) az veriyle daha iyi sonuç alınır
- [ ] `torchvision.models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)` → pretrained model yükle
- [ ] Katmanları dondurma (freeze):
  - `for param in self.backbone.parameters(): param.requires_grad = False`
  - Dondurulmuş katmanlar eğitim sırasında güncellenmez → önceden öğrenilmiş özellikler korunur
- [ ] Son N bloğu açma (unfreeze):
  - `layers[-unfreeze_last_n:]` → son 1-2 ResNet bloğunu eğitilebilir yap
  - Daha derin fine-tuning sağlar ama daha fazla hesaplama gerektirir
- [ ] Son `fc` katmanını değiştir:
  - Orijinal: `Linear(512, 1000)` (ImageNet 1000 sınıf)
  - Yeni: `Dropout → Linear(512, 256) → ReLU → Dropout → Linear(256, 6)` (bizim 6 sınıf)

**Öğleden Sonra — Karşılaştırma ve doğrulama:**
- [ ] `count_parameters()` → Custom CNN vs ResNet18 parametre karşılaştırması:
  - Custom CNN: tüm parametreler eğitilebilir
  - ResNet18 (frozen): sadece yeni fc katmanı eğitilebilir (~çok daha az parametre)
- [ ] `unfreeze_all()` metodu → tüm katmanları eğitilebilir yapma (tam fine-tuning)
- [ ] `get_last_conv_layer()` → model analizi için son conv katmanını döndür
- [ ] Her iki modeli de oluştur, parametre sayılarını ve yapılarını yan yana karşılaştır

---

### 📅 Gün 5: Eğitim Döngüsü, Optimizer ve Metrikler

**Sabah — `src/training/trainer.py`:**
- [ ] `Trainer` sınıfı → `__init__`: model, criterion, optimizer, device, scheduler, checkpoint_dir
- [ ] `train_one_epoch()` fonksiyonu — bir epoch'luk eğitim:
  1. `self.model.train()` → Dropout aktif, BatchNorm eğitim modunda
  2. Her batch için döngü: `for inputs, labels in train_loader:`
  3. `inputs = inputs.to(self.device)` → veriyi GPU'ya gönder
  4. `self.optimizer.zero_grad()` → önceki gradyanları temizle (aksi halde birikir!)
  5. `outputs = self.model(inputs)` → ileri yayılım (forward pass)
  6. `loss = self.criterion(outputs, labels)` → CrossEntropyLoss hesapla
  7. `loss.backward()` → geri yayılım (her ağırlığın gradyanını hesapla)
  8. `self.optimizer.step()` → gradyanlara göre ağırlıkları güncelle
  9. `_, predicted = torch.max(outputs, 1)` → en yüksek olasılıklı sınıf
- [ ] `validate()` fonksiyonu:
  - `self.model.eval()` → Dropout kapalı, BatchNorm sabit istatistikler kullanır
  - `@torch.no_grad()` → gradyan hesabı kapalı (bellek ve hız tasarrufu)
  - Sadece forward pass + metrik hesaplama (backward yok!)
- [ ] `fit()` fonksiyonu — tam eğitim döngüsü:
  - Epoch döngüsü → her epoch'ta train + validate
  - `history` dict'ine train_loss, val_loss, train_acc, val_acc kaydet
  - En iyi val_acc gördüğünde `torch.save(model.state_dict(), path)` ile checkpoint kaydet
  - Early stopping: N epoch boyunca val_acc iyileşmezse eğitimi durdur
  - `scheduler.step()` → her epoch sonunda learning rate'i güncelle

**Öğleden Sonra — Optimizer, Scheduler ve Metrikler:**
- [ ] `src/training/optimizer_factory.py`:
  - `create_optimizer()` → `optim.Adam(params, lr, weight_decay)` veya `optim.SGD(params, lr, momentum, weight_decay)`
  - `filter(lambda p: p.requires_grad, model.parameters())` → sadece eğitilebilir parametreleri optimize et
  - `create_scheduler()` → `StepLR(optimizer, step_size=5, gamma=0.5)`:
    - Her 5 epoch'ta learning rate'i yarıya indir
    - Örnek: 0.001 → 0.0005 → 0.00025 → ...
- [ ] `src/evaluation/metrics.py`:
  - `calculate_accuracy()` → `torch.max(outputs, dim=1)` ile doğru tahmin sayısı / toplam
  - `get_all_predictions()` → tüm test setinde tahmin yap, gerçek ve tahmin etiketlerini numpy dizisi olarak döndür
  - `compute_confusion_matrix()` → `sklearn.metrics.confusion_matrix` kullanarak hangi sınıfların karıştırıldığını göster
  - `calculate_f1()` → `sklearn.metrics.f1_score(average='weighted')` ile F1-score hesapla
  - `classification_summary()` → `sklearn.metrics.classification_report` ile sınıf bazında Precision/Recall/F1

---

## HAFTA 2 (Gün 6-10): Eğitim, Optimizasyon, Streamlit ve Dokümantasyon

---

### 📅 Gün 6: Google Colab'da İlk Model Eğitimi

**Sabah — Colab ortam hazırlığı:**
- [ ] GitHub'a son değişiklikleri push et
- [ ] `notebooks/training.ipynb`'yi Google Colab'a yükle
- [ ] Runtime > Change runtime type > **GPU (T4)** seç
- [ ] `torch.cuda.is_available()` ve `torch.cuda.get_device_name(0)` ile GPU doğrula
- [ ] `!git clone` ile repo'yu clone et, `!pip install -q optuna` ile Optuna kur
- [ ] Kaggle API ile veri setini indir: `!kaggle datasets download ...` → `!unzip`
- [ ] Veri pipeline'ını çalıştır: `load_dataset()` → `create_dataloaders(batch_size=32)`

**Öğleden Sonra — Model eğitimi:**
- [ ] **Custom CNN eğitimi** (~20 epoch):
  - `CustomCNN(num_classes=6, num_blocks=4, base_filters=32, dropout_rate=0.3)`
  - `Adam` optimizer, `lr=1e-3`, `StepLR(step_size=5, gamma=0.5)`
  - Loss ve accuracy grafiklerini çiz (`matplotlib`)
  - Train vs Val farkını gözlemle → overfitting var mı?
- [ ] **ResNet18 eğitimi** (~15 epoch):
  - `TransferResNet18(num_classes=6, dropout_rate=0.3, freeze_backbone=True, unfreeze_last_n=1)`
  - `Adam` optimizer, `lr=5e-4`
  - Transfer Learning'in ne kadar hızlı yakınsadığını gözlemle
- [ ] İki modelin confusion matrix'lerini yan yana çiz (`seaborn.heatmap`)
- [ ] `classification_report` ile sınıf bazında Precision/Recall/F1 karşılaştır

---

### 📅 Gün 7: Optuna Hiperparametre Optimizasyonu

**Sabah — `src/training/hpo.py` inceleme:**
- [ ] `objective()` fonksiyonu — Optuna'nın her denemede çağırdığı fonksiyon:
  - `trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)` → logaritmik ölçekte lr arama
  - `trial.suggest_categorical("optimizer", ["Adam", "SGD"])` → optimizer seçimi
  - `trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)` → dropout arama
  - `trial.suggest_int("num_blocks", 3, 5)` → CNN blok sayısı (Custom CNN için)
  - `trial.suggest_categorical("base_filters", [16, 32, 64])` → başlangıç filtre sayısı
- [ ] `MedianPruner(n_startup_trials=5, n_warmup_steps=3)`:
  - İlk 5 deneme budanmaz (referans oluşturmak için)
  - Her denemede ilk 3 epoch budanmaz (modelin oturması için)
  - Sonrasında medyan altında kalan denemeler erken durdurulur
- [ ] `trial.report(val_accuracy, epoch)` → ara sonucu Optuna'ya raporla
- [ ] `trial.should_prune()` → bu deneme budanmalı mı?
- [ ] `optuna.create_study(direction="maximize")` → val_accuracy'yi maximize et

**Öğleden Sonra — Colab'da HPO çalıştırma:**
- [ ] Custom CNN için HPO: `run_hpo_study(n_trials=10, num_epochs=10)`
- [ ] Tamamlanan ve budanan deneme sayılarını gözlemle
- [ ] `study.best_params` → en iyi hiperparametre kombinasyonunu al
- [ ] En iyi parametrelerle final modeli eğit (~25 epoch, `early_stopping_patience=7`)
- [ ] Final modeli `models/custom_cnn_best.pt` olarak kaydet
- [ ] `models/results.json` dosyasını oluştur (her iki modelin sonuçları + history)

---

### 📅 Gün 8: ONNX Export ve Streamlit Uygulamasını Test Etme

**Sabah — `src/export/onnx_export.py`:**
- [ ] ONNX kavramını öğren:
  - Open Neural Network Exchange → framework-bağımsız model formatı
  - PyTorch'ta eğitilen model, TensorFlow veya ONNX Runtime'da çalıştırılabilir
- [ ] `export_to_onnx()` fonksiyonu:
  - `dummy_input = torch.randn(1, 3, 224, 224)` → örnek giriş (tracing için)
  - `torch.onnx.export(model, dummy_input, path, ...)` → modeli ONNX'e dönüştür
  - `dynamic_axes={"input": {0: "batch_size"}}` → değişken batch boyutu desteği
  - `opset_version=11` → ONNX operatör seti versiyonu
- [ ] `verify_onnx_model()` fonksiyonu:
  - `onnx.checker.check_model()` → ONNX model yapısını doğrula
  - `onnxruntime.InferenceSession()` → ONNX Runtime ile test inference
- [ ] Colab'da her iki modeli `.onnx` olarak export et ve doğrula

**Öğleden Sonra — Streamlit test:**
- [ ] Colab'dan indir: `custom_cnn_best.pt`, `resnet18_best.pt`, `*.onnx`, `results.json`
- [ ] İndirilen dosyaları lokal projenin `models/` klasörüne kopyala
- [ ] `streamlit run app/app.py` çalıştır
- [ ] Ana sayfayı kontrol et: proje açıklaması, sayfa navigasyonu, teknoloji kartları
- [ ] Her sayfayı sırayla aç ve temel çalışıp çalışmadığını doğrula

---

### 📅 Gün 9: Streamlit Sayfalarını Detaylı Test ve Debugging

**Sabah — EDA ve Training sayfaları:**
- [ ] `app/pages/1_EDA.py` test:
  - Sınıf dağılımı bar chart'ı düzgün çiziliyor mu?
  - Her sınıftan 3 örnek görüntü grid halinde gösteriliyor mu?
  - "Yeniden Augmentation Uygula" butonu çalışıyor mu?
  - Augmentation öncesi (orijinal) ve sonrası (4 farklı) yan yana mı?
- [ ] `app/pages/2_Training.py` test:
  - Sidebar'daki hiperparametre slider/selectbox'ları çalışıyor mu?
  - Model tipi (custom_cnn/resnet18) seçimi doğru mu?
  - Eğitim başlatma butonu çalışıyor mu? (lokalde kısa epoch ile test et)
  - Loss ve accuracy grafikleri çiziliyor mu?
  - Sonuç metrikleri (Best Val Acc, F1, Süre) gösteriliyor mu?

**Öğleden Sonra — Comparison ve Prediction sayfaları:**
- [ ] `app/pages/3_Model_Comparison.py` test:
  - `models/results.json` dosyasından sonuçlar okunuyor mu?
  - Custom CNN ve ResNet18 metrikleri yan yana gösteriliyor mu?
  - Val Loss ve Val Accuracy karşılaştırma grafikleri çiziliyor mu?
  - JSON dosyası yokken uyarı mesajı gösteriliyor mu?
- [ ] `app/pages/4_Prediction.py` test:
  - `st.file_uploader` ile resim yüklenebiliyor mu?
  - Model seçimi (custom_cnn/resnet18) çalışıyor mu?
  - Tahmin sonucu ve güven skoru gösteriliyor mu?
  - 6 sınıf için olasılık çubukları (`st.progress`) düzgün mü?
  - Model dosyası bulunamadığında hata mesajı gösteriliyor mu?
- [ ] Tüm hataları not al ve düzelt

---

### 📅 Gün 10: README, Son Test ve GitHub Push

**Sabah — `README.md`:**
- [ ] Proje başlığı ve açıklaması
- [ ] Veri seti bilgisi (Intel Image Classification, 6 sınıf, ~25K görüntü)
- [ ] Kullanılan teknolojiler listesi (PyTorch, Optuna, Streamlit, ONNX, torchvision)
- [ ] Kurulum adımları:
  ```
  git clone ...
  pip install -r requirements.txt
  # Veri setini indir + Colab'da eğitim yap + dosyaları indir
  streamlit run app/app.py
  ```
- [ ] Proje yapısı (klasör ağacı)
- [ ] Sonuçlar tablosu: Custom CNN vs ResNet18 (accuracy, F1, parametre sayısı, süre)
- [ ] Streamlit ekran görüntüleri (her sayfa için)

**Öğleden Sonra — Son kontroller:**
- [ ] Uçtan uca test: Colab notebook → dosya indirme → lokal Streamlit → her sayfa çalışıyor
- [ ] Kod yorumlarını kontrol et, eksik yorum varsa ekle
- [ ] `.gitignore` kontrol: `data/`, `models/*.pt`, `models/*.onnx`, `__pycache__/` dahil mi?
- [ ] `git add . && git commit -m "Project complete" && git push`
- [ ] GitHub'da README'nin düzgün render edildiğini kontrol et
- [ ] 🎉 **Proje tamamlandı!**
