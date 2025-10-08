import os, numpy as np, pandas as pd
from typing import List
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = "iris_course"
ARTIFACTS = os.path.join(ROOT, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

def describe_stats(X: np.ndarray, names: List[str], title: str):
    m, s = X.mean(axis=0), X.std(axis=0)
    print(f"\n[{title}]")
    for n, mi, sd in zip(names, m, s):
        print(f"{n:<14s} mean={mi:8.4f} std={sd:8.4f}")

print("=== STEP 1 | 載入資料探索 ===")
iris = load_iris()
X, y = iris.data, iris.target
feature_names = ("sepal length", "sepal width", "petal length", "petal width")
target_names = iris.target_names.tolist()

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
print("\n前5筆資料:")
print(df.head())
print("\n類別分布:")
for i, name in enumerate(target_names):
    print(f"{i}={name:<10s}: {(y == i).sum()}")
describe_stats(X, feature_names, "原始資料(為標準化)")

out_csv = os.path.join(ARTIFACTS, "inis_preview.csv")
df.head(20).to_csv(out_csv, index=False)
print(f"\n已存前20筆預覽: {out_csv}")
print("STEP 1 完成")

print("\n=== STEP 2 | 切分 Train/Val/Test ===")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)
print(f"切分形狀: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
print("STEP 2 完成")


print("\n=== STEP 3 | 標準化(只用訓練集fit)並存檔 ===")
scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

describe_stats(X_train, feature_names, "訓練集（標準化前）")
describe_stats(X_train_sc, feature_names, "訓練集（標準化後）")

npz_path = os.path.join(ARTIFACTS, "train_val_test_scaled.npz")
np.savez(
    npz_path,
    X_train_sc=X_train_sc,
    y_train=y_train,
    X_val_sc=X_val_sc,
    y_val=y_val,
    X_test_sc=X_test_sc,
    y_test=y_test,
    feature_names=np.array(feature_names, dtype=object),
    target_names=np.array(target_names, dtype=object),
)
scaler_path = os.path.join(ARTIFACTS, "scaler.pkl")
joblib.dump(scaler, scaler_path)

print(f"已存標準化資料：{npz_path}")
print(f"已存標準化器:{scaler_path}")
print("STEP 3 完成")

print("\n=== STEP 4 Tensor 與 DataLoader ===")
X_train_t = torch.tensor(X_train_sc, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val_sc, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t), batch_size=16, shuffle=False
)
xb, yb = next(iter(train_loader))
print(f"第一個 batch: xb.shape={xb.shape}, yb.shape={yb.shape}")
print(f"xb[0](標準化後)= {xb[0].tolist()}")
print(f"yb[0](類別) = {yb[0].item()}")
batch_preview = os.path.join(ARTIFACTS, "batch_preview.csv")
pd.DataFrame(xb.numpy(), columns=feature_names).assign(
    label=yb.numpy()
).to_csv(batch_preview, index=False)
print(f"→已存 batch 預覽:{batch_preview}")
print("STEP 4 完成")