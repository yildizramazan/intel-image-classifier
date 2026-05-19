"""
dataset.py - Veri Seti Yükleme
================================
Intel Image Classification veri setini torchvision.datasets.ImageFolder ile yükler.

Kullanılan PyTorch Kavramları:
    - torchvision.datasets.ImageFolder: Klasör yapısından (her sınıf bir alt klasör)
      otomatik olarak etiketli veri seti oluşturur.
      Klasör yapısı:
          data/seg_train/
              buildings/   -> etiket 0
              forest/      -> etiket 1
              glacier/     -> etiket 2
              mountain/    -> etiket 3
              sea/         -> etiket 4
              street/      -> etiket 5
"""

import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from typing import Optional, Tuple

# Intel Image Classification veri setinin 6 sınıfı
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)


def load_dataset(
    data_dir: str,
    transform: Optional[transforms.Compose] = None
) -> ImageFolder:
    """
    Belirtilen dizindeki görüntüleri ImageFolder ile yükler.

    ImageFolder Nasıl Çalışır?
        - Her alt klasör bir sınıf olarak kabul edilir
        - Alt klasör isimleri alfabetik sıralanıp 0'dan başlayarak numaralandırılır
        - Her görüntü (image, label) çifti olarak döndürülür
        - transform parametresi ile görüntüler otomatik dönüştürülür

    Args:
        data_dir: Veri setinin kök dizini (örn: "data/seg_train")
        transform: Uygulanacak dönüşüm pipeline'ı (transforms.Compose)

    Returns:
        ImageFolder: Etiketli görüntü veri seti

    Raises:
        FileNotFoundError: Dizin bulunamazsa
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Veri seti dizini bulunamadı: {data_dir}\n"
            f"Lütfen veri setini Kaggle'dan indirip '{data_dir}' konumuna çıkarın."
        )

    dataset = ImageFolder(root=data_dir, transform=transform)

    print(f"Veri seti yüklendi: {data_dir}")
    print(f"  Toplam görüntü sayısı: {len(dataset)}")
    print(f"  Sınıflar: {dataset.classes}")
    print(f"  Sınıf -> İndeks: {dataset.class_to_idx}")

    return dataset


def get_class_distribution(dataset: ImageFolder) -> dict:
    """
    Veri setindeki sınıf dağılımını hesaplar.
    EDA (Keşifsel Veri Analizi) sayfasında kullanılacak.

    Args:
        dataset: ImageFolder veri seti

    Returns:
        dict: {sınıf_adı: görüntü_sayısı} şeklinde dağılım
    """
    from collections import Counter
    label_counts = Counter(dataset.targets)
    distribution = {
        dataset.classes[label]: count
        for label, count in sorted(label_counts.items())
    }
    return distribution


def get_sample_images(dataset: ImageFolder, num_per_class: int = 3) -> dict:
    """
    Her sınıftan belirtilen sayıda örnek görüntü döndürür.
    EDA sayfasında grid halinde gösterilecek.

    Args:
        dataset: ImageFolder veri seti
        num_per_class: Her sınıftan alınacak örnek sayısı

    Returns:
        dict: {sınıf_adı: [(image_tensor, label), ...]}
    """
    from collections import defaultdict
    samples = defaultdict(list)

    for idx in range(len(dataset)):
        image, label = dataset[idx]
        class_name = dataset.classes[label]
        if len(samples[class_name]) < num_per_class:
            samples[class_name].append((image, label))

        # Tüm sınıflar için yeterli örnek toplandıysa dur
        if all(len(v) >= num_per_class for v in samples.values()):
            break

    return dict(samples)
