try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float = 0.0):
    if TORCH_AVAILABLE:
        # PyTorch 版本
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        if dropout_p and dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

            output = attn_weights @ v

        return output, attn_weights

    else:
    # NumPy 版本（當 torch 不可用時）
        import numpy as np
        import math
        d_k = q.shape[-1]
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        scores_max = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - scores_max)
        attn_weights = exp / exp.sum(axis=-1, keepdims=True)
        output = np.matmul(attn_weights, v)
        return output, attn_weights