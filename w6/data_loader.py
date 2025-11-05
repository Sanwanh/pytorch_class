import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./datasets/cifar10',
    train=True,
    download=False,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./datasets/cifar10',
    train=False,
    download=False,
    transform=transform
)

print(f"train data size: {len(trainset)}")
print(f"test data size: {len(testset)}")
print("class =", len(trainset.classes))
print("class names =", trainset.classes)
print("class_to_idx =", trainset.class_to_idx)

# 取一筆看看
img, lbl = trainset[0]
print("single image shape =", img.shape, "label =", lbl, trainset.classes[lbl])

train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size

train_dataset, val_dataset = random_split(
    trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"train dataset size: {len(train_dataset)}")
print(f"validation dataset size: {len(val_dataset)}")
print(f"test dataset size: {len(testset)}")
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
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"train batches: {len(train_loader)}")
print(f"validation batches: {len(val_loader)}")
print(f"test batches: {len(test_loader)}")

def imshow(img):
    img = img / 2 + 0.5 # 反正規化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# 取得一批訓練資料
dataiter = iter(train_loader)
images, labels = next(dataiter)

classes = ['plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
# 建立圖形
fig = plt.figure(figsize=(12, 6))
fig.suptitle('CIFAR-10', fontsize=16, fontweight='bold')

# 顯示 8 張圖片
for idx in range(8):
    ax = plt.subplot(2, 4, idx + 1)
    img = images[idx]
    img = img / 2 + 0.5 # 反正規化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'class: {classes[labels[idx]]}')
    plt.axis('off')

plt.tight_layout()
os.makedirs('./outputs', exist_ok=True)
plt.savefig('./outputs/cifar10_samples.png', dpi=150, bbox_inches='tight')
print("saved: ./outputs/cifar10_samples.png")