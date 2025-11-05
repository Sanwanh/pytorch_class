import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import datasets


# ===== 路徑設定 =====
data_root = './datasets2/cifar10_folders'
train_dir = os.path.join(data_root, 'train')
test_dir = os.path.join(data_root, 'test')

# CIFAR-10 類別名稱
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']
if not os.path.exists(data_root):
    os.makedirs(data_root, exist_ok=True)
    transform_to_pil = transforms.ToPILImage()

    # 載入訓練與測試資料
    cifar_train = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True
    )

    cifar_test = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True
    )

    # 建立類別資料夾
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # ====== 儲存部分測試圖片 ======
    print("saving train data...")
    class_counts = {i: 0 for i in range(10) } 
    # 每類最多存500張
    for idx, (img, label) in enumerate(cifar_train):
        if class_counts[label] < 500:
            img.save(os.path.join(train_dir, classes[label], f'{class_counts[label]}.png'))
            class_counts[label] += 1

        # 若每類已達500張就停止
        if all(count >= 500 for count in class_counts.values()):
            break

    # ====== 儲存部分測試圖片 ======
    print("saving test data...")
    class_counts = {i: 0 for i in range(10) } 
    # 每類最多存100張
    for idx, (img, label) in enumerate(cifar_test):
        if class_counts[label] < 100:
            img.save(os.path.join(test_dir, classes[label], f'{class_counts[label]}.png'))
            class_counts[label] += 1

        # 若每類已達100張就停止
        if all(count >= 100 for count in class_counts.values()):
            break

    print("done.")
else:
    print("skip")

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("train: Resize + RandomFlip + RandomRotation + ToTensor + Normalize")
print("test: Resize + ToTensor + Normalize")

full_train_dataset = ImageFolder(root=train_dir, transform=train_transform)

test_dataset = ImageFolder(root=test_dir, transform=test_transform)

train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)