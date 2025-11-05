import torch, torch.nn as nn

L = 21 # 序列長度
x = torch.zeros(1, 1, L)
x[0, 0, L // 2] = 1.0 # 中心位置放一個脈衝 (one-hot)

layers = []
num_layers = 3 # 試試 1 層、2 層、3 層看看

for _ in range(num_layers):
    conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.ones_(conv.weight) # 權重全 1，好觀察影響範圍
    layers.append(conv)

net = nn.Sequential(*layers)

with torch.no_grad():
    y = net(x)
    nz = (y[0][0] != 0).nonzero().flatten()
    left, right = nz[0].item(), nz[-1].item()
    print(f"非零區間寬度 = {right - left + 1}；理論感受野 R_{num_layers}")