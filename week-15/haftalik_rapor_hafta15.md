# Staj Haftalık Rapor, 15. Hafta

Stajyer: Ramazan YILDIZ  
Tarih Aralığı: [Başlangıç Tarihi] – [Bitiş Tarihi]  
Proje: CNN Görüntü Sınıflandırma, Intel Image Classification Veri Seti, Model Eğitimi, Optuna HPO, ONNX Export ve Streamlit Dashboard

---

## 65. Gün: Google Colab'da Model Eğitimi

Haftaya T4 GPU kullanarak Google Colab'da her iki modeli eğiterek başladım. `pip install -q optuna` ile Optuna'yı kurarak ve Kaggle API ile `kaggle datasets download` aracılığıyla veri setini indirerek Colab ortamını hazırladım. `torch.cuda.is_available()` ve `torch.cuda.get_device_name(0)` ile GPU doğrulamasının ardından `load_dataset()` ve `create_dataloaders(batch_size=32)` ile veri pipeline'ını yükledim.

Custom CNN'i `CustomCNN(num_classes=6, num_blocks=4, base_filters=32, dropout_rate=0.3)` ile 20 epoch boyunca lr=1e-3'te Adam optimizer ve `StepLR(step_size=5, gamma=0.5)` ile eğittim. Ardından ResNet18 modelini `TransferResNet18(num_classes=6, dropout_rate=0.3, freeze_backbone=True, unfreeze_last_n=1)` ile 15 epoch boyunca lr=5e-4'te eğittim ve aktarım öğreniminin (transfer learning) önemli ölçüde daha hızlı yakınsadığını gözlemledim. Her iki model için matplotlib ile loss ve accuracy eğrilerini çizdim, karışıklık matrislerini (confusion matrix) `seaborn.heatmap` ile yan yana karşılaştırdım ve `classification_summary()` ile sınıf bazında Precision, Recall ve F1 raporları oluşturdum. Tüm sonuçları eğitim geçmişi dahil Streamlit karşılaştırma sayfası için `models/results.json` dosyasına kaydettim:

```python
custom_model = CustomCNN(num_classes=6, num_blocks=4, base_filters=32, dropout_rate=0.3)
criterion = nn.CrossEntropyLoss()
optimizer = create_optimizer(custom_model, 'Adam', learning_rate=1e-3)
scheduler = create_scheduler(optimizer, step_size=5, gamma=0.5)
trainer = Trainer(custom_model, criterion, optimizer, device, scheduler, 'models')
custom_history = trainer.fit(train_loader, val_loader, num_epochs=20,
                             early_stopping_patience=5, model_name='custom_cnn')

resnet_model = TransferResNet18(num_classes=6, dropout_rate=0.3,
                                freeze_backbone=True, unfreeze_last_n=1)
optimizer = create_optimizer(resnet_model, 'Adam', learning_rate=5e-4)
resnet_history = trainer_resnet.fit(train_loader, val_loader, num_epochs=15,
                                    early_stopping_patience=5, model_name='resnet18')
```

---

## 66. Gün: Optuna ile Hiperparametre Optimizasyonu

İkinci gün `src/training/hpo.py` dosyasındaki Optuna hiperparametre optimizasyonu üzerinde çalıştım. `objective()` fonksiyonu logaritmik öğrenme oranı araması için `trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)`, optimizer seçimi için `trial.suggest_categorical("optimizer", ["Adam", "SGD"])`, dropout ayarlaması için `trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)` ve Custom CNN'e özel olarak `trial.suggest_int("num_blocks", 3, 5)` ile `trial.suggest_categorical("base_filters", [16, 32, 64])` kullanır. Her deneme (trial) bir model oluşturur, erken durdurma (early stopping) ile eğitir ve en iyi doğrulama doğruluğunu (validation accuracy) döndürür.

`run_hpo_study()` fonksiyonu `optuna.create_study(direction="maximize")` ve ilk 5 denemeyi ve her denemede ilk 3 epoch'u budamayı atlayan `MedianPruner(n_startup_trials=5, n_warmup_steps=3)` ile bir çalışma (study) oluşturur; ardından medyan altında kalan denemeleri erken durdurur. Colab'da `run_hpo_study(n_trials=10, num_epochs=10)` çalıştırdım, tamamlanan ve budanan deneme sayılarını inceledim, `study.best_params` ile en iyi parametreleri çıkardım ve en iyi hiperparametrelerle final modeli 25 epoch boyunca `early_stopping_patience=7` ile eğitip `models/custom_cnn_best.pt` olarak kaydettim:

```python
def objective(trial, train_loader, val_loader, model_type, device, num_epochs=15):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
    num_blocks = trial.suggest_int("num_blocks", 3, 5)
    base_filters = trial.suggest_categorical("base_filters", [16, 32, 64])
    model = CustomCNN(num_classes=6, num_blocks=num_blocks,
                      base_filters=base_filters, dropout_rate=dropout_rate)
    trainer = Trainer(model, criterion, optimizer, device, scheduler)
    history = trainer.fit(train_loader, val_loader, num_epochs, trial=trial, verbose=False)
    return max(history["val_acc"])

pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(lambda trial: objective(trial, ...), n_trials=10)
```

---

## 67. Gün: ONNX Export ve Streamlit Uygulamasını Test Etme

Üçüncü gün her iki modeli ONNX formatına aktardım ve Streamlit uygulamasını test etmeye başladım. `src/export/onnx_export.py` dosyasında `export_to_onnx()` fonksiyonu modeli `eval()` olarak ayarlar, izleme (tracing) için `torch.randn(1, 3, 224, 224)` sahte girdi oluşturur ve `export_params=True`, `opset_version=11`, `do_constant_folding=True` ile değişken toplu boyut (batch size) için `dynamic_axes` parametreleriyle `torch.onnx.export()` çağırır. `verify_onnx_model()` fonksiyonu model yapısını `onnx.checker.check_model()` ile doğrular ve `onnxruntime.InferenceSession` ile test çıkarımı (inference) çalıştırır. Colab'da hem `custom_cnn.onnx` hem de `resnet18.onnx` dosyalarını dışa aktarıp doğruladım.

Ardından eğitilmiş model dosyalarını (`custom_cnn_best.pt`, `resnet18_best.pt`, `.onnx` dosyaları ve `results.json`) Colab'dan yerel projenin `models/` dizinine indirdim. `streamlit run app/app.py` ile Streamlit uygulamasını başlattım ve proje açıklamasını, 6 sınıflı veri seti bilgisini tablo halinde, sayfa navigasyon kartlarını ve PyTorch, Optuna, Streamlit, ONNX metriklerini içeren teknoloji yığını bölümünü gösteren ana sayfayı test ettim:

```python
def export_to_onnx(model, save_path, input_shape=(1, 3, 224, 224), dynamic_batch=True):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    torch.onnx.export(model, dummy_input, save_path,
                      export_params=True, opset_version=11,
                      do_constant_folding=True,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes=dynamic_axes)

def verify_onnx_model(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(onnx_path)
    outputs = session.run(None, {"input": dummy_input})
```

---

## 68. Gün: Streamlit Sayfalarını Detaylı Test ve Debugging

Dördüncü gün dört Streamlit sayfasının tümünü detaylı olarak test ettim. EDA sayfası (`1_EDA.py`) performans için `@st.cache_data` kullanır, `st.metric` ile genel istatistikleri (toplam görüntü, sınıf sayısı, görüntü boyutları, sınıf başına ortalama) gösterir, `seaborn` renk paletiyle sınıf dağılımı çubuk grafiği çizer, `denormalize()` ve `st.image()` kullanarak sınıf başına 3 örnek görüntüyü ızgara halinde gösterir ve orijinal ile 4 rastgele artırılmış versiyonu yan yana gösteren `st.rerun()` çağıran bir "Yeniden Augmentation Uygula" butonuyla artırım karşılaştırma bölümü sunar.

Training sayfası (`2_Training.py`) model tipi, öğrenme oranı, batch boyutu, optimizer, dropout ve epoch için `st.selectbox`, `st.select_slider` ve `st.slider` ile kenar çubuğu hiperparametre kontrollerini sunar ve tam pipeline'ı çalıştırıp loss/accuracy eğrilerini gösteren bir "Eğitimi Başlat" butonu içerir. Model Karşılaştırma sayfası (`3_Model_Comparison.py`) `models/results.json` dosyasını okuyarak Custom CNN ve ResNet18 metriklerini doğrulama kaybı ve doğruluk karşılaştırma grafikleriyle yan yana gösterir; dosya yoksa uyarı mesajı verir. Tahmin sayfası (`4_Prediction.py`) `st.file_uploader` ile görüntü yüklemelerini kabul eder, seçilen modeli yükler, sınıf olasılıklarını almak için `F.softmax()` ile çıkarım yapar ve tahmin edilen sınıfı güven skoru ve `st.progress` ile sınıf bazında olasılık çubuklarıyla gösterir:

```python
# Prediction page core logic
image = Image.open(uploaded_file).convert("RGB")
transform = get_test_transforms()
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)[0]
    predicted_idx = probabilities.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item()

for i, class_name in enumerate(CLASS_NAMES):
    st.progress(probabilities[i].item(), text=f"{class_name}: {probabilities[i]:.1%}")
```

---

## 69. Gün: Son Test ve Proje Tamamlama

Son gün projenin eksiksiz uçtan uca doğrulamasını gerçekleştirdim. Colab notebook'unu sıfırdan çalıştırdım, eğitilmiş model dosyalarını yerel projenin `models/` dizinine indirdim, Streamlit dashboard'unu başlattım ve eğitilmiş ağırlıklar ve sonuçlar JSON'u ile her sayfanın (EDA, Training, Comparison, Prediction) doğru çalıştığını doğruladım.

Kod yorumlarını eksiksizlik açısından gözden geçirdim ve proje yapısının son düzenlemesini gerçekleştirdim:

```python
# Results saved to models/results.json for Streamlit
results = {
    'custom_cnn': {
        'val_accuracy': max(custom_history['val_acc']),
        'val_f1_score': f1_custom,
        'training_time_sec': custom_time,
        'total_params': custom_model.count_parameters()['total'],
        'trainable_params': custom_model.count_parameters()['trainable'],
        'history': custom_history
    },
    'resnet18': {
        'val_accuracy': max(resnet_history['val_acc']),
        'val_f1_score': f1_resnet,
        'training_time_sec': resnet_time,
        'total_params': resnet_model.count_parameters()['total'],
        'trainable_params': resnet_model.count_parameters()['trainable'],
        'history': resnet_history
    }
}
```

---

## Haftalık Değerlendirme

Bu hafta CNN görüntü sınıflandırma projesinin ikinci yarısını tamamladım. 65. gün GPU ile Google Colab'da hem Custom CNN hem de ResNet18 modellerini eğittim, karışıklık matrisleri ve sınıflandırma raporları ile performanslarını karşılaştırdım ve sonuçları JSON'a kaydettim. 66. gün deneme budama (trial pruning) için MedianPruner ile Optuna hiperparametre optimizasyonu çalıştırdım, en iyi hiperparametre kombinasyonunu buldum ve final modeli eğittim. 67. gün her iki modeli değişken batch boyutu desteğiyle ONNX formatına aktardım, ONNX Runtime ile doğruladım ve Streamlit uygulamasını test etmeye başladım. 68. gün dört Streamlit sayfasının tümünü detaylı olarak test ettim: sınıf dağılımı ve artırım görselleştirmesi ile EDA, etkileşimli hiperparametre kontrolleri ile Training, JSON'dan sonuçları okuyan Model Comparison ve görüntü yükleme ile olasılık gösterimi içeren Prediction. 69. gün uçtan uca doğrulama gerçekleştirerek projeyi tamamladım.

