"""
dataloader.py - DataLoader Factory
====================================
Eğitim, doğrulama ve test için DataLoader nesneleri oluşturur.

Kullanılan PyTorch Kavramları:
    - torch.utils.data.DataLoader: Veri setini batch'ler halinde modele besler.
      Temel parametreleri:
        * batch_size: Her iterasyonda modele gönderilen örnek sayısı
        * shuffle: Epoch başında verilerin sırasını karıştırır (eğitimde True)
        * num_workers: Paralel veri yükleme için kullanılan CPU çekirdeği sayısı
        * pin_memory: GPU eğitiminde veri transferini hızlandırır
        * drop_last: Son eksik batch'i atar (BatchNorm için önemli)

    - torch.utils.data.random_split: Veri setini rastgele parçalara böler
      (eğitim ve doğrulama ayrımı için)
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from typing import Tuple, Optional


def create_dataloaders(
    train_dataset: ImageFolder,
    test_dataset: ImageFolder,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Eğitim, doğrulama ve test DataLoader'larını oluşturur.

    DataLoader Neden Gerekli?
        - Verileri tek tek değil, batch (yığın) halinde işler -> GPU verimli kullanılır
        - shuffle=True ile her epoch'ta farklı sırada veri gösterilir -> overfitting azalır
        - num_workers ile paralel veri yükleme yapılır -> eğitim hızlanır
        - pin_memory=True ile CPU->GPU veri transferi hızlanır

    random_split ile Validation Ayrımı:
        - Eğitim setinin %20'si doğrulama (validation) için ayrılır
        - Bu ayırma rastgeledir ama seed ile tekrarlanabilir
        - Doğrulama seti, modelin eğitim sırasında görmediği verilerle test edilmesini sağlar

    Args:
        train_dataset: Eğitim veri seti (ImageFolder)
        test_dataset: Test veri seti (ImageFolder)
        batch_size: Her batch'teki örnek sayısı (varsayılan: 32)
        val_split: Eğitim setinden doğrulama için ayrılacak oran (varsayılan: 0.2)
        num_workers: Paralel veri yükleme işçi sayısı (varsayılan: 2)
        pin_memory: GPU transferini hızlandırma (varsayılan: True)
        seed: Rastgelelik tohumu, tekrarlanabilirlik için (varsayılan: 42)

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # Tekrarlanabilirlik için rastgelelik üretecini (generator) seed'le
    generator = torch.Generator().manual_seed(seed)

    # Eğitim setini train ve validation olarak böl
    total_size = len(train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"Veri seti bölümleme:")
    print(f"  Eğitim (Train):     {train_size} görüntü")
    print(f"  Doğrulama (Val):    {val_size} görüntü")
    print(f"  Test:               {len(test_dataset)} görüntü")
    print(f"  Batch boyutu:       {batch_size}")

    # Eğitim DataLoader'ı: shuffle=True, veri sırasını karıştırır
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,               # Her epoch'ta karıştır
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,             # Son eksik batch'i at (BatchNorm stabilitesi)
        generator=generator
    )

    # Doğrulama DataLoader'ı: shuffle=False, sıra önemli değil ama tutarlılık için sabit
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    # Test DataLoader'ı: shuffle=False
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
