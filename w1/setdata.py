import torch
import matplotlib.pyplot as plt 

torch.manual_seed(0)
n = 100
x = torch.randn(n, 1)
w_turn = torch.tensor([10.0])
b_turn = torch.tensor([3.0])
y = w_turn * x + b_turn + torch.randn(n,1) *2.5

plt.figure()
plt.plot(x.numpy(), y.numpy(), 'o', color='r')
plt.title('DATA (x,y)')
plt.xlabel('x');  plt.ylabel('y')
plt.tight_layout()
plt.savefig('step1_scatter.png')
plt.show()
