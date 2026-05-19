"""
transforms.py - Veri Dönüşüm Pipeline'ları
============================================
Eğitim (train) ve test/doğrulama (test) için ayrı transforms.Compose zinciri tanımlar.
Eğitim seti Data Augmentation içerir, test seti sadece resize + normalize yapar.

Kullanılan PyTorch Kavramları:
    - torchvision.transforms.Compose: Birden fazla dönüşümü sıralı zincire bağlar
    - torchvision.transforms.Resize: Görüntüyü hedef boyuta ölçekler
    - torchvision.transforms.RandomHorizontalFlip: %50 olasılıkla yatay çevirir
    - torchvision.transforms.RandomRotation: Belirtilen derece aralığında döndürür
    - torchvision.transforms.ColorJitter: Parlaklık, kontrast, doygunluk değiştirir
    - torchvision.transforms.RandomCrop: Rastgele kırpma yapar
    - torchvision.transforms.ToTensor: PIL Image -> Tensor [0, 1] dönüşümü
    - torchvision.transforms.Normalize: Kanal bazında ortalama ve std ile normalize eder
"""

from torchvision import transforms

# ImageNet veri setinin ortalama ve standart sapma değerleri.
# Pretrained ResNet18 bu değerlerle eğitildiği için aynısını kullanıyoruz.
# Custom CNN için de tutarlılık sağlamak adına aynı normalizasyonu uyguluyoruz.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Hedef görüntü boyutu. ResNet18 girişi 224x224 bekler.
IMAGE_SIZE = 224


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Eğitim verisi için Data Augmentation pipeline'ı oluşturur.

    Data Augmentation Neden Gerekli?
        - Modelin eğitim verisini ezberlemesini (overfitting) önler
        - Veri setini yapay olarak çoğaltarak modelin genelleme yeteneğini artırır
        - Her epoch'ta aynı görüntüyü farklı şekillerde görerek daha sağlam (robust) öğrenir

    Dönüşüm Sırası (pipeline):
        1. Resize(256)         -> Görüntüyü 256x256'ya büyüt (kırpma için pay bırak)
        2. RandomCrop(224)     -> Rastgele 224x224'lük bölge kes
        3. RandomHorizontalFlip-> %50 olasılıkla yatay çevir (doğa fotoğrafları için uygun)
        4. RandomRotation(15)  -> ±15 derece rastgele döndür
        5. ColorJitter         -> Parlaklık ve kontrast rastgele değiştir
        6. ToTensor()          -> PIL Image -> FloatTensor, [0, 255] -> [0.0, 1.0]
        7. Normalize()         -> ImageNet ortalaması ile standartlaştır

    Args:
        image_size: Hedef görüntü boyutu (varsayılan: 224)

    Returns:
        transforms.Compose: Eğitim dönüşüm zinciri
    """
    return transforms.Compose([
        transforms.Resize(image_size + 32),          # 256x256 - kırpma payı
        transforms.RandomCrop(image_size),            # Rastgele 224x224 kırpma
        transforms.RandomHorizontalFlip(p=0.5),       # %50 olasılıkla yatay çevirme
        transforms.RandomRotation(degrees=15),        # ±15 derece döndürme
        transforms.ColorJitter(
            brightness=0.2,                           # Parlaklık ±%20
            contrast=0.2,                             # Kontrast ±%20
            saturation=0.1,                           # Doygunluk ±%10
            hue=0.05                                  # Ton ±%5
        ),
        transforms.ToTensor(),                        # Tensor'a çevir [0, 1]
        transforms.Normalize(
            mean=IMAGENET_MEAN,                       # Kanal ortalamaları
            std=IMAGENET_STD                          # Kanal standart sapmaları
        ),
    ])


def get_test_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Test/doğrulama verisi için dönüşüm pipeline'ı oluşturur.

    Test verisine augmentation UYGULANMAZ çünkü:
        - Modelin gerçek performansını ölçmek istiyoruz
        - Augmentation sadece eğitimde veri çeşitliliği sağlamak içindir

    Dönüşüm Sırası:
        1. Resize(224)   -> Görüntüyü doğrudan hedef boyuta getir
        2. ToTensor()     -> Tensor'a çevir
        3. Normalize()    -> Aynı ImageNet normalizasyonu uygula

    Args:
        image_size: Hedef görüntü boyutu (varsayılan: 224)

    Returns:
        transforms.Compose: Test dönüşüm zinciri
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Tam boyuta resize
        transforms.ToTensor(),                         # Tensor'a çevir [0, 1]
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Normalize edilmiş tensörü tekrar [0, 1] aralığına döndürür.
    Streamlit ve matplotlib ile görselleştirme için gereklidir.

    Formül: pixel = (normalized_pixel * std) + mean

    Args:
        tensor: Normalize edilmiş görüntü tensörü (C, H, W)
        mean: Normalizasyonda kullanılan ortalama
        std: Normalizasyonda kullanılan standart sapma

    Returns:
        Denormalize edilmiş tensör
    """
    import torch
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
