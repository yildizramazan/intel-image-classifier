# Internship Weekly Report, Week 15

Intern: Ramazan YILDIZ  
Date Range: [Start Date] – [End Date]  
Project: CNN Image Classification, Intel Image Classification Dataset, Model Training, Optuna HPO, ONNX Export and Streamlit Dashboard

---

## Day 65: Model Training on Google Colab

I started the week by training both models on Google Colab using a T4 GPU. I set up the Colab environment by installing Optuna via `pip install -q optuna` and downloading the Intel Image Classification dataset through the Kaggle API with `kaggle datasets download`. After verifying GPU availability with `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`, I loaded the data pipeline via `load_dataset()` and `create_dataloaders(batch_size=32)`.

I trained the Custom CNN with `CustomCNN(num_classes=6, num_blocks=4, base_filters=32, dropout_rate=0.3)` for 20 epochs using Adam optimizer at lr=1e-3 with `StepLR(step_size=5, gamma=0.5)`. I then trained the ResNet18 model with `TransferResNet18(num_classes=6, dropout_rate=0.3, freeze_backbone=True, unfreeze_last_n=1)` for 15 epochs at lr=5e-4, observing how transfer learning converges significantly faster. I plotted loss and accuracy curves for both models using matplotlib, compared their confusion matrices side by side with `seaborn.heatmap`, and generated per-class Precision, Recall, and F1 reports via `classification_summary()`. I saved all results including training history to `models/results.json` for the Streamlit comparison page:

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

## Day 66: Optuna Hyperparameter Optimization

On the second day, I worked on hyperparameter optimization using Optuna in `src/training/hpo.py`. The `objective()` function uses `trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)` for logarithmic learning rate search, `trial.suggest_categorical("optimizer", ["Adam", "SGD"])` for optimizer selection, `trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)` for dropout tuning, and for Custom CNN specifically `trial.suggest_int("num_blocks", 3, 5)` and `trial.suggest_categorical("base_filters", [16, 32, 64])`. Each trial builds a model, trains it with early stopping, and returns the best validation accuracy.

The `run_hpo_study()` function creates a study with `optuna.create_study(direction="maximize")` and `MedianPruner(n_startup_trials=5, n_warmup_steps=3)` which skips pruning for the first 5 trials and first 3 epochs per trial, then prunes below-median trials early. I ran `run_hpo_study(n_trials=10, num_epochs=10)` on Colab, examined completed and pruned trial counts, extracted `study.best_params`, and trained the final model with the best hyperparameters for 25 epochs with `early_stopping_patience=7`, saving it as `models/custom_cnn_best.pt`:

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

## Day 67: ONNX Export and Streamlit Application Testing

On the third day, I exported both models to ONNX format and began testing the Streamlit application. In `src/export/onnx_export.py`, the `export_to_onnx()` function sets the model to `eval()`, creates a dummy input `torch.randn(1, 3, 224, 224)` for tracing, and calls `torch.onnx.export()` with `export_params=True`, `opset_version=11`, `do_constant_folding=True`, and `dynamic_axes` for variable batch size. The `verify_onnx_model()` function validates the model structure with `onnx.checker.check_model()` and runs test inference via `onnxruntime.InferenceSession`. I exported and verified both `custom_cnn.onnx` and `resnet18.onnx` on Colab.

I then downloaded the trained model files (`custom_cnn_best.pt`, `resnet18_best.pt`, `.onnx` files, and `results.json`) from Colab to the local project's `models/` directory. I launched the Streamlit application with `streamlit run app/app.py` and tested the main page which displays the project description, 6-class dataset info in a table, page navigation cards, and a technology stack section with PyTorch, Optuna, Streamlit, and ONNX metrics:

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

## Day 68: Streamlit Pages Detailed Testing and Debugging

On the fourth day, I tested all four Streamlit pages in detail. The EDA page (`1_EDA.py`) uses `@st.cache_data` for performance, displays general statistics with `st.metric` (total images, class count, image dimensions, average per class), renders a class distribution bar chart with `seaborn` color palette, shows 3 sample images per class in a grid using `denormalize()` and `st.image()`, and provides an augmentation comparison section with a "Reapply Augmentation" button calling `st.rerun()` to show the original alongside 4 randomly augmented versions.

The Training page (`2_Training.py`) presents sidebar hyperparameter controls via `st.selectbox`, `st.select_slider`, and `st.slider` for model type, learning rate, batch size, optimizer, dropout, and epochs, with a "Start Training" button that runs the full pipeline and displays loss/accuracy curves. The Model Comparison page (`3_Model_Comparison.py`) reads `models/results.json` and shows Custom CNN vs ResNet18 metrics side by side with validation loss and accuracy comparison graphs, or a warning if the results file is missing. The Prediction page (`4_Prediction.py`) accepts image uploads via `st.file_uploader`, loads the selected model, runs inference with `F.softmax()` to get class probabilities, and displays the predicted class with confidence and per-class probability bars via `st.progress`:

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

## Day 69: Final Testing and Project Completion

On the final day, I performed a complete end-to-end verification of the project. I ran the Colab notebook from scratch, downloaded the trained model files to the local `models/` directory, launched the Streamlit dashboard, and verified that every page (EDA, Training, Comparison, Prediction) works correctly with the trained weights and results JSON.

I reviewed code comments for completeness and performed final cleanup of the project structure:

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

## Weekly Summary

This week I completed the second half of the CNN image classification project. On Day 65, I trained both Custom CNN and ResNet18 models on Google Colab with GPU, compared their performance via confusion matrices and classification reports, and saved results to JSON. On Day 66, I ran Optuna hyperparameter optimization with MedianPruner for trial pruning, found the best hyperparameter combination, and trained the final model. On Day 67, I exported both models to ONNX format with dynamic batch size support, verified them with ONNX Runtime, and began testing the Streamlit application. On Day 68, I tested all four Streamlit pages in detail: EDA with class distribution and augmentation visualization, Training with interactive hyperparameter controls, Model Comparison reading results from JSON, and Prediction with image upload and probability display. On Day 69, I performed end-to-end verification, completing the project.
