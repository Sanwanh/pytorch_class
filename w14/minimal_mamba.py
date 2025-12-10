import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * self.weight
        return output


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
        in_channels=self.d_inner,
        out_channels=self.d_inner,
        kernel_size=d_conv,
        bias=True,
        groups=self.d_inner, # depthwise conv
        padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_model + 2, bias=False)
        self.dt_proj = nn.Linear(self.d_state, self.d_state, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.B = nn.Parameter(torch.ones(self.d_inner, self.d_inner))

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2) # (B, E, L)
        x = self.conv1d(x)[:, :, :seq_len] # 保持長度不變
        x = x.transpose(1, 2) # (B, L, E)
        x = F.silu(x)

        y = self.ssm(x)
        y = F.silu(z) * y
        y = self.norm(y)
        return self.out_project

    def ssm(self, x):
        A = torch.exp(self.A_log.float()) # (E, N)
        D = self.d_state

        x_dbl = self.x_proj(x) # (B, L, dt_rank + 2*N)
        delta, B, C = x_dbl.split([self.d_state, self.d_model, self.d_model], dim=-1)

        delta = F.softplus(self.dt_proj(delta)) # (B, L, E)
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        delta_A = torch.exp(delta.unsqueeze(-1) * A) # (B, L, E, N)
        delta_B_U = (delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)) # (B, L, E, N)

        h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
        ys = []

        for i in range(seq_len):
            h = delta_A[:, i] * h + delta_B_U[:, i]
            y = (h @ C[:, i, :].unsqueeze(-1)).squeeze(-1) # (B, E)
            ys.append(y)

        y = torch.stack(ys, dim=1) # (B, L, E)

        return y


if __name__ == "__main__":

    # 參數配置
    d_model = 64
    batch_size = 2
    seq_len = 128

    # 建構模型
    model = MambaBlock(d_model)
    print("MambaBlock created successfully.")

    # 建構輸入張量
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # 前向傳播
    output = model(x)
    print(f"Output shape: {output.shape}")

    # 驗證形狀是否正確
    assert output.shape == x.shape
    print("Output shape is correct.")