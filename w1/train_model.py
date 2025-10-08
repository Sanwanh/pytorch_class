import torch
from torch import nn, optim
from model import Model
import matplotlib.pyplot as plt
import os
import numpy as np

torch.manual_seed(0)
n = 100
x = torch.randn(n, 1)
w_turn = torch.tensor([10.0])
b_turn = torch.tensor([3.0])
y = w_turn * x + b_turn + torch.randn(n,1) * 2.5

net = Model()
opt = optim.SGD(net.parameters(), lr = 0.1) 
loss_f = nn.MSELoss()
loss_hist = []

for epoch in range(11):
    y_hat = net(x)
    loss = loss_f(y, y_hat)
    loss.backward()
    opt.step()
    opt.zero_grad()
    loss_hist.append(float(loss.item()))
    print(f'epoch:{epoch}, loss:{loss.item()}')
    print(loss)

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

plt.figure()
plt.plot(range(len(loss_hist)),loss_hist,marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, linestyle= '--', alpha=0.5)
plt.tight_layout()
loss_path = os.path.join(OUT_DIR, 'loss.png')
plt.savefig('loss_path.png')
plt.close()

net.eval()
with torch.no_grad():
    y_hat = net(x)

arr = np.hstack([
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    y_hat.detach().cpu().numpy()
])

pred_csv_path = os.path.join(OUT_DIR, 'predictions.csv')
np.savetxt(pred_csv_path, arr ,delimiter = ',', header='x,y,y_hat', comments='')

idx = x.squeeze(1).argsort()
x_sorted = x[idx]
y_hat_sorted = y_hat[idx]

plt.figure()
plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), 'o', alpha = 0.6 ,label='DATA(x,y)')
plt.plot(x.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), 'x', alpha = 0.8 ,label='Model output on training')
plt.plot(x_sorted.detach().cpu().numpy(), y_hat_sorted.detach().cpu().numpy(), '-', linewidth = 2 ,label='connected model optput')


plt.xlabel('x')
plt.ylabel('value')
plt.title('Model Output on Training Points')
plt.legend()
plt.tight_layout()
scatter_outputs_path = os.path.join(OUT_DIR , 'data_with_true_output.png')
plt.savefig(scatter_outputs_path, dpi =150)
plt.show()

