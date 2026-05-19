# Internship Weekly Report, Week 14

Intern: Ramazan YILDIZ  
Date Range: [Start Date] – [End Date]  
Project: CNN Image Classification, Intel Image Classification Dataset, Project Infrastructure, Data Pipeline, Model Design, Training Loop and Metrics

---

## Day 60: Project Setup, Environment and Dataset

I started the week by setting up the project infrastructure for an end-to-end CNN image classification system. I created a modular folder structure with `src/data/` for data loading and transforms, `src/models/` for CNN and transfer learning definitions, `src/training/` for the training loop, optimizer and HPO, `src/evaluation/` for performance metrics, `src/export/` for ONNX conversion, `app/` for the Streamlit multi-page web application, and `notebooks/` for the Colab training notebook. I prepared `requirements.txt` with torch, torchvision, optuna, streamlit, onnx, scikit-learn and other dependencies.

I then downloaded the Intel Image Classification dataset from Kaggle containing ~14K training and ~3K test images across 6 classes (buildings, forest, glacier, mountain, sea, street). I extracted the data into `data/seg_train/seg_train/` and `data/seg_test/seg_test/` directories and verified the folder structure and class distribution:

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

## Day 61: Data Pipeline, Transforms, Dataset and DataLoader

I built the complete data pipeline starting with `src/data/transforms.py`. The `get_train_transforms()` function chains `Resize(256)` for crop margin, `RandomCrop(224)` for random region selection each epoch, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)` for color augmentation, `ToTensor()` for PIL to tensor conversion scaling pixels from [0,255] to [0.0,1.0], and `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` using ImageNet statistics. I also wrote `get_test_transforms()` with only Resize, ToTensor, and Normalize since test data should not receive augmentation, plus a `denormalize()` helper for Streamlit visualization.

In `src/data/dataset.py`, I implemented `load_dataset()` using `torchvision.datasets.ImageFolder` which treats each subdirectory as a class label, `get_class_distribution()` via `Counter(dataset.targets)` for EDA, and `get_sample_images()` to retrieve N samples per class. In `src/data/dataloader.py`, `create_dataloaders()` splits the training set 80/20 via `torch.utils.data.random_split()` with a seeded generator for reproducibility, and creates DataLoaders with `batch_size=32`, `shuffle=True` for training, `num_workers=2` for parallel loading, `pin_memory=True` for faster CPU-to-GPU transfer, and `drop_last=True` for BatchNorm stability:

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

## Day 62: Custom CNN Architecture from Scratch

I designed a `CustomCNN` class in `src/models/custom_cnn.py` extending `nn.Module` with a feature extractor built from configurable blocks. Each block contains `Conv2d(in, out, kernel_size=3, padding=1)` for same-padding convolution, `BatchNorm2d` for activation normalization, `ReLU(inplace=True)` for non-linearity, and `MaxPool2d(2,2)` halving spatial dimensions at each stage (224->112->56->28->14). The filter count doubles per block (32->64->128->256), all grouped into `self.features = nn.Sequential(*layers)`.

The classifier head uses `Flatten()` to reshape from (B, 256, 14, 14) to (B, 256*14*14), followed by `Linear(flatten_size, 512)`, `ReLU`, `Dropout(p=dropout_rate)`, `Linear(512, 128)`, and `Linear(128, 6)` for the output without softmax since `CrossEntropyLoss` applies it internally. I implemented `_initialize_weights()` using Kaiming He initialization with `nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')` for Conv2d and Linear layers, and `count_parameters()` for reporting total and trainable parameter counts:

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

## Day 63: Transfer Learning with ResNet18

I implemented the `TransferResNet18` class in `src/models/transfer_model.py`, loading pre-trained ImageNet weights via `models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` which already recognizes edges, textures, and shapes. I froze all backbone parameters with `param.requires_grad = False` to preserve pre-trained features, then optionally unfroze the last N ResNet blocks via `layers[-unfreeze_last_n:]` for deeper fine-tuning. I replaced the original `Linear(512, 1000)` classification head with `Dropout(p=dropout_rate)`, `Linear(512, 256)`, `ReLU`, `Dropout(p=dropout_rate/2)`, and `Linear(256, 6)` for our 6 classes.

I also implemented `count_parameters()` for comparing trainable parameter counts between Custom CNN (all trainable) and frozen ResNet18 (only the new fc layer trainable), `unfreeze_all()` for full fine-tuning after initial frozen training, and `get_last_conv_layer()` returning `layer4[-1].conv2` for model analysis:

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

## Day 64: Training Loop, Optimizer and Evaluation Metrics

I built a `Trainer` class in `src/training/trainer.py` encapsulating the full training lifecycle. The `train_one_epoch()` method follows the standard PyTorch loop: `model.train()` to activate Dropout and BatchNorm, then per batch `optimizer.zero_grad()` to clear gradients, `outputs = model(inputs)` for forward pass, `loss.backward()` for backpropagation, and `optimizer.step()` to update weights. The `validate()` method runs under `model.eval()` with `@torch.no_grad()` for inference without gradient computation. The `fit()` method handles multi-epoch training with history tracking, best-model checkpointing via `torch.save(model.state_dict(), path)` when validation accuracy improves, early stopping after N epochs without improvement, and Optuna integration via `trial.report()` and `trial.should_prune()`.

In `src/training/optimizer_factory.py`, I implemented `create_optimizer()` supporting Adam and SGD with `filter(lambda p: p.requires_grad, model.parameters())` to optimize only trainable parameters, and `create_scheduler()` returning `StepLR(step_size=5, gamma=0.5)` which halves the learning rate every 5 epochs. In `src/evaluation/metrics.py`, I implemented `calculate_accuracy()` using `torch.max()`, `get_all_predictions()` for collecting test predictions, `compute_confusion_matrix()` via sklearn, `calculate_f1()` with weighted averaging, and `classification_summary()` using `classification_report` for per-class Precision, Recall, and F1:

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

## Weekly Summary

This week I built the first half of an end-to-end CNN image classification project using the Intel Image Classification dataset (6 classes, ~17K images). I set up the modular project structure, implemented the full data pipeline with augmentation transforms and DataLoader creation, designed a custom CNN with configurable Conv+BN+ReLU+Pool blocks and Kaiming initialization, implemented a ResNet18 transfer learning model with backbone freezing and a custom classification head, and built the Trainer class with train/validate loops, early stopping, checkpoint saving, optimizer/scheduler factory, and scikit-learn based evaluation metrics.
