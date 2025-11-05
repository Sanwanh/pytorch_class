import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset 
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import datasets

# ===== 路徑設定 =====
data_root = './datasets3/cifar10_folders'
images_dir = os.path.join(data_root,'images')
os.makedirs(images_dir, exist_ok=True)

csv_path = os.path.join(data_root,'labels.csv')

# CIFAR-10 類別名稱
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']

# 數據創建與 CSV 儲存邏輯
if not os.path.exists(csv_path):
    cifar_dataset = torchvision.datasets.CIFAR10(
        root='./datasets3/cifar10',
        train=True,
        download=True
    )

    data_list = []
    # classes 已在前面定義

    class_counts = {i: 0 for i in range(10)}

    for idx, (img, label) in enumerate(cifar_dataset):
        if class_counts[label] < 10:
            img_name = f'{classes[label]}_{class_counts[label]}.png'
            img_path = os.path.join(images_dir, img_name)
            img.save(img_path)

            data_list.append({
            'image_name': img_name,
            'label': label,
            'class_name': classes[label]
            })

        class_counts[label] += 1

        if all(count >= 10 for count in class_counts.values()):
            break

    df = pd.DataFrame(data_list)
    df.to_csv(csv_path, index=False)
    print(f"Created {len(df)} data entries")
    print(f"CSV file: {csv_path}")
    print(f"Image directory: {images_dir}")

else:
    print("Data already exists, skipping creation step")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries")

# 自訂資料集類別
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        print(f"Loaded: {len(self.data_frame)} samples")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx]['image_name']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = int(self.data_frame.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx):
        return self.data_frame.iloc[idx]['class_name']

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

# 1. 載入完整的自訂資料集 
full_dataset = CustomImageDataset(
    csv_file=csv_path,
    img_dir=images_dir,
    transform=train_transform 
)
print(f"Total custom dataset samples: {len(full_dataset)}")

# 2. 定義分割比例 (假設 80% 訓練, 10% 驗證, 10% 測試)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size 
    
# 3. 分割資料集
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

batch_size = 32

# 4. 創建 DataLoader
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

# 5. 圖片顯示
if len(train_dataset) > 0:
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('CustomImageDataset Example (from CSV)', fontsize=16, fontweight='bold')

    num_samples_to_show = min(8, len(images))

    for idx in range(num_samples_to_show):
        ax = plt.subplot(2, 4, idx + 1)
        img = images[idx]
        img = img / 2 + 0.5 # 反正規化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(f'class: {classes[labels[idx]]}') 
        plt.axis('off')

    plt.tight_layout()
    os.makedirs('./w6/outputs3', exist_ok=True)
    plt.savefig('./w6/outputs3/custom_dataset_samples.png', dpi=150, bbox_inches='tight')
    print("saving: ./w6/outputs3/custom_dataset_samples.png")
else:
    print("Training dataset is empty, skipping image display.")